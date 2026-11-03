"""Regression tests for isolated, complete tracking publication without ML."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import behave_tracking
import tools.tracking_results as results_module
from behave_tracking import TrackingMixin
from tools.tracking_results import MANIFEST_NAME, TrackingError, load_tracking_result


TRACKS = {"penguin_1": [{"second": 0, "x": 1, "y": 2, "area": 3}]}


class TrackingPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"original source video")
        self.output = self.root / "output"
        self.agent = self.make_agent()

    def make_agent(self):
        agent = TrackingMixin()
        agent.videos_dir = str(self.root)
        agent.model = "test-model"
        return agent

    @staticmethod
    def extractor(video, json_path, annotated, **kwargs):
        Path(json_path).write_text(json.dumps(TRACKS), encoding="utf-8")
        Path(annotated).write_bytes(b"validated fresh encoded video")
        return json_path, annotated

    def successful_run(self, agent=None, extractor=None):
        agent = agent or self.agent
        with mock.patch("behave_tracking.extract_tracks", side_effect=extractor or self.extractor):
            agent.run_tracking(self.video.name, self.output)
        return agent.require_tracking_result(self.video.name, self.output)

    def assert_no_current_result(self):
        self.assertIsNone(self.agent.tracks)
        self.assertIsNone(self.agent.tracking_result)
        with self.assertRaises(TrackingError):
            self.agent.require_tracking_result(self.video.name, self.output)
        with self.assertRaises(TrackingError):
            self.make_agent().require_tracking_result(self.video.name, self.output)

    def test_success_creates_unique_complete_runs_and_retains_history(self):
        first = self.successful_run()
        first_contents = (first.json_path.read_bytes(), first.annotated_video_path.read_bytes())
        second = self.successful_run()
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first_contents, (first.json_path.read_bytes(), first.annotated_video_path.read_bytes()))
        # Compare canonical paths even when Windows TEMP uses a short-name alias.
        self.assertEqual(second.output_dir, (self.output / "runs" / second.run_id).resolve())
        self.assertEqual(second.tracks, TRACKS)
        self.assertFalse((self.output / "tracks.json").exists())
        self.assertFalse((self.output / ".tracking.lock").exists())

    def test_results_from_the_previous_identity_algorithm_require_reprocessing(self):
        result = self.successful_run()
        path = self.output / MANIFEST_NAME
        old_manifest = json.loads(path.read_text(encoding="utf-8"))
        old_manifest["schema_version"] = 1
        path.write_text(json.dumps(old_manifest), encoding="utf-8")
        with self.assertRaisesRegex(TrackingError, "Unsupported tracking manifest"):
            self.agent.require_tracking_result(self.video.name, self.output)
        self.assertTrue(result.json_path.exists())
        self.assertTrue(result.annotated_video_path.exists())

    def test_run_records_processing_settings_and_passes_explicit_sam_config(self):
        self.agent.config_path = str(self.root / "custom.env")
        with mock.patch("behave_tracking.extract_tracks", side_effect=self.extractor) as extract:
            self.agent.run_tracking(self.video.name, self.output)
        self.assertEqual(extract.call_args.kwargs["sam3_config_path"], self.agent.config_path)
        receipt = json.loads((self.output / MANIFEST_NAME).read_text(encoding="utf-8"))
        from tools import tracking_config
        self.assertEqual(receipt["parameters"]["target_fps"], tracking_config.TARGET_FPS)
        self.assertEqual(receipt["parameters"]["chunk_duration_seconds"], tracking_config.CHUNK_DURATION_SEC)

    def test_worker_failure_blocks_cached_and_disk_results_after_restart(self):
        old = self.successful_run()
        original = old.json_path.read_bytes()
        def fail(video, json_path, annotated, **kwargs):
            Path(json_path).write_text('{"partial":', encoding="utf-8")
            raise TrackingError("worker crashed")
        with mock.patch("behave_tracking.extract_tracks", side_effect=fail):
            with self.assertRaisesRegex(TrackingError, "worker crashed"):
                self.agent.run_tracking(self.video.name, self.output)
        self.assertEqual(old.json_path.read_bytes(), original)
        self.assertEqual(json.loads((self.output / MANIFEST_NAME).read_text())["status"], "failed")
        self.assertEqual(list(self.output.glob(".tracking-stage-*")), [])
        self.assert_no_current_result()

    def test_legacy_failure_tuple_cannot_reload_old_success(self):
        old = self.successful_run()
        with mock.patch("behave_tracking.extract_tracks", return_value=(None, None)):
            with self.assertRaisesRegex(TrackingError, "fresh artifacts"):
                self.agent.run_tracking(self.video.name, self.output)
        self.assertTrue(old.annotated_video_path.exists())
        self.assert_no_current_result()

    def test_extractor_cannot_return_files_from_a_previous_run(self):
        old = self.successful_run()
        with mock.patch("behave_tracking.extract_tracks", return_value=(old.json_path, old.annotated_video_path)):
            with self.assertRaisesRegex(TrackingError, "fresh artifacts"):
                self.agent.run_tracking(self.video.name, self.output)
        self.assert_no_current_result()

    def test_partial_empty_malformed_or_invalid_output_is_never_published(self):
        cases = ["missing_video", "empty_video", "missing_json", "malformed_json", "wrong_schema", "duplicate_time", "nonfinite"]
        old = self.successful_run()
        for case in cases:
            with self.subTest(case=case):
                def partial(video, json_path, annotated, **kwargs):
                    if case != "missing_json":
                        tracks = copy.deepcopy(TRACKS)
                        if case == "wrong_schema":
                            tracks = []
                        elif case == "duplicate_time":
                            tracks["penguin_1"] *= 2
                        elif case == "nonfinite":
                            tracks["penguin_1"][0]["x"] = float("nan")
                        Path(json_path).write_text("{" if case == "malformed_json" else json.dumps(tracks), encoding="utf-8")
                    if case != "missing_video":
                        Path(annotated).write_bytes(b"" if case == "empty_video" else b"video")
                    return json_path, annotated
                with mock.patch("behave_tracking.extract_tracks", side_effect=partial):
                    with self.assertRaises(TrackingError):
                        self.agent.run_tracking(self.video.name, self.output)
                self.assertTrue(old.json_path.exists())
                self.assert_no_current_result()

    def test_valid_empty_scene_is_distinct_from_a_failure(self):
        def empty_scene(video, json_path, annotated, **kwargs):
            Path(json_path).write_text("{}", encoding="utf-8")
            Path(annotated).write_bytes(b"encoded frames without detections")
            return json_path, annotated
        result = self.successful_run(extractor=empty_scene)
        self.assertEqual(result.tracks, {})
        self.assertEqual(json.loads((self.output / MANIFEST_NAME).read_text())["status"], "success")

    def test_input_change_during_extraction_prevents_publication(self):
        def mutate_source(*args, **kwargs):
            self.video.write_bytes(b"changed source")
            return self.extractor(*args, **kwargs)
        with mock.patch("behave_tracking.extract_tracks", side_effect=mutate_source):
            with self.assertRaisesRegex(TrackingError, "changed during tracking"):
                self.agent.run_tracking(self.video.name, self.output)
        self.assert_no_current_result()

    def test_success_pointer_publication_failure_cannot_reactivate_old_run(self):
        old = self.successful_run()
        real_write = behave_tracking.atomic_write_json
        def fail_success_pointer(path, value):
            if Path(path).name == MANIFEST_NAME and value["status"] == "success":
                raise OSError("simulated disk failure")
            return real_write(path, value)
        with mock.patch("behave_tracking.atomic_write_json", side_effect=fail_success_pointer):
            with self.assertRaisesRegex(TrackingError, "disk failure"):
                self.successful_run()
        self.assertTrue(old.annotated_video_path.exists())
        self.assert_no_current_result()

    def test_unwritable_status_retains_lock_instead_of_reusing_previous_success(self):
        self.successful_run()
        with mock.patch("behave_tracking.atomic_write_json", side_effect=OSError("read-only status")):
            with self.assertRaisesRegex(TrackingError, "lock file was retained"):
                self.agent.run_tracking(self.video.name, self.output)
        self.assertTrue((self.output / ".tracking.lock").exists())
        self.assert_no_current_result()

    def test_current_result_is_unavailable_while_a_rerun_is_in_progress(self):
        old = self.successful_run()
        def during_run(*args, **kwargs):
            with self.assertRaisesRegex(TrackingError, "in progress"):
                load_tracking_result(self.video, self.output)
            return self.extractor(*args, **kwargs)
        new = self.successful_run(extractor=during_run)
        self.assertNotEqual(old.run_id, new.run_id)

    def test_lock_contender_can_use_a_later_success_from_the_owner(self):
        old = self.successful_run()
        contender = self.make_agent()
        def owner_extractor(*args, **kwargs):
            with self.assertRaisesRegex(TrackingError, "Another tracking run"):
                contender.run_tracking(self.video.name, self.output)
            with self.assertRaises(TrackingError):
                contender.require_tracking_result(self.video.name, self.output)
            return self.extractor(*args, **kwargs)
        new = self.successful_run(extractor=owner_extractor)
        self.assertNotEqual(old.run_id, new.run_id)
        self.assertEqual(contender.require_tracking_result(self.video.name, self.output).run_id, new.run_id)

    def test_another_agent_success_can_recover_after_this_agent_failed(self):
        old = self.successful_run()
        with mock.patch("behave_tracking.extract_tracks", side_effect=TrackingError("worker failed")):
            with self.assertRaises(TrackingError):
                self.agent.run_tracking(self.video.name, self.output)
        new = self.successful_run(agent=self.make_agent())
        self.assertNotEqual(old.run_id, new.run_id)
        self.assertEqual(self.agent.require_tracking_result(self.video.name, self.output).run_id, new.run_id)

    def test_artifact_tampering_and_stale_mapping_are_rejected(self):
        first = self.successful_run()
        first.annotated_video_path.write_bytes(b"replaced video")
        with self.assertRaisesRegex(TrackingError, "changed after publication"):
            self.agent.require_tracking_result(self.video.name, self.output)
        (self.output / "tracks_species_mapping.txt").write_text("1: wrong_previous_species\n", encoding="utf-8")
        second = self.successful_run()
        self.assertEqual(second.tracks, TRACKS)

    def test_context_change_and_missing_input_clear_agent_derived_state(self):
        self.successful_run()
        self.agent.current_rag_refs = {1: {"source": "old report"}}
        self.assertTrue(self.agent.set_video_context(self.video.name, self.root / "different-output"))
        self.assertIsNone(self.agent.tracks)
        self.assertEqual(self.agent.current_rag_refs, {})
        self.agent.tracks = TRACKS
        with self.assertRaises(TrackingError):
            self.agent.set_video_context("missing.mp4", self.output)
        self.assertIsNone(self.agent.tracks)

    def test_video_replaced_while_artifacts_are_loaded_is_rejected(self):
        self.successful_run()
        original_reader = results_module.read_tracks
        def replace_source(*args, **kwargs):
            self.video.write_bytes(b"a different video")
            return original_reader(*args, **kwargs)
        with mock.patch.object(results_module, "read_tracks", side_effect=replace_source):
            with self.assertRaisesRegex(TrackingError, "source video changed during result validation"):
                self.agent.require_tracking_result(self.video.name, self.output)

    def test_json_replaced_between_hash_check_and_parse_is_rejected(self):
        result = self.successful_run()
        original_reader = results_module.read_tracks
        def replace_json(*args, **kwargs):
            result.json_path.write_text(json.dumps({"forged": TRACKS["penguin_1"]}), encoding="utf-8")
            return original_reader(*args, **kwargs)
        with mock.patch.object(results_module, "read_tracks", side_effect=replace_json):
            with self.assertRaisesRegex(TrackingError, "artifact changed after publication"):
                self.agent.require_tracking_result(self.video.name, self.output)

    def test_encoded_video_replaced_during_json_parsing_is_rejected(self):
        result = self.successful_run()
        original_reader = results_module.read_tracks
        def replace_video(*args, **kwargs):
            result.annotated_video_path.write_bytes(b"wrong annotated video")
            return original_reader(*args, **kwargs)
        with mock.patch.object(results_module, "read_tracks", side_effect=replace_video):
            with self.assertRaisesRegex(TrackingError, "artifact changed after publication"):
                self.agent.require_tracking_result(self.video.name, self.output)


if __name__ == "__main__":
    unittest.main()
