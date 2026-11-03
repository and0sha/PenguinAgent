"""Synthetic chunk outputs through the real extractor, identity matching and renderer."""

import copy
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest import mock

import test_tracking_pipeline as pipeline_fixtures


def frame(*observations):
    """Each observation is (chunk-local raw ID, animal marker, x, y)."""
    return {raw_id: (animal, x, y) for raw_id, animal, x, y in observations}


class ChunkIdentityIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Composition reuses video/ML adapters without collecting their old tests twice.
        self.fixture = pipeline_fixtures.PipelineTests("test_zero_detections_are_a_valid_complete_result")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fps = 1.0
        self.chunks = []
        self.samples = None
        self.worker_crop_dirs = []
        self.classification_inputs = []
        self.rendered_annotations = []
        self.semantic_names = {}
        original_capture = sys.modules["cv2"].VideoCapture
        state = self

        class Capture(original_capture):
            def get(self, key):
                if key == sys.modules["cv2"].CAP_PROP_FPS:
                    return state.fps
                return super().get(key)

        self.capture_patch = mock.patch.object(sys.modules["cv2"], "VideoCapture", Capture)
        self.capture_patch.start()
        self.addCleanup(self.capture_patch.stop)

        class Process:
            def __init__(self, target, args):
                self.target, self.args = target, args
                self.exitcode = 0

            def start(self):
                if self.target.__name__ == "_run_sam3_subprocess":
                    state.write_worker_result(self.args)
                elif self.target.__name__ == "_run_siglip_subprocess":
                    batch, destination, _ = self.args
                    results = {}
                    for path, global_id in batch:
                        marker = Path(path).read_text(encoding="utf-8")
                        state.classification_inputs.append((int(global_id), marker, str(path)))
                        results[str(global_id)] = marker
                    Path(destination).write_text(json.dumps({"status": "ok", "results": results}), encoding="utf-8")
                else:
                    raise AssertionError(f"Unexpected child process target: {self.target.__name__}")

            def join(self, *args, **kwargs):
                pass

            def is_alive(self):
                return False

        self.process_patch = mock.patch.object(self.fixture.pipeline.multiprocessing, "get_context",
                                              return_value=types.SimpleNamespace(Process=Process))
        self.process_patch.start()
        self.addCleanup(self.process_patch.stop)
        self.ffmpeg_patch = mock.patch.object(self.fixture.utils.subprocess, "run", side_effect=self.ffmpeg)
        self.ffmpeg_patch.start()
        self.addCleanup(self.ffmpeg_patch.stop)

        original_render = self.fixture.pipeline.render_final_video

        def capture_render(videos, annotations, names, *args):
            self.rendered_annotations = copy.deepcopy(annotations)
            self.semantic_names = dict(names)
            return original_render(videos, annotations, names, *args)

        self.render_patch = mock.patch.object(self.fixture.pipeline, "render_final_video", side_effect=capture_render)
        self.render_patch.start()
        self.addCleanup(self.render_patch.stop)

    def ffmpeg(self, command, **kwargs):
        if "-f" in command:
            for index, observations in enumerate(self.chunks):
                destination = str(command[-1] % index)
                Path(destination).write_bytes(b"synthetic chunk")
                self.fixture.videos[destination] = [pipeline_fixtures.Frame() for _ in observations]
        else:
            source = str(command[command.index("-i") + 1])
            destination = str(command[-1])
            Path(destination).write_bytes(b"synthetic encoded video")
            self.fixture.videos[destination] = list(self.fixture.videos[source])
        return types.SimpleNamespace(returncode=0)

    def write_worker_result(self, args):
        source, video, data = args[:3]
        crops = Path(args[-1])
        crops.mkdir(parents=True, exist_ok=True)
        self.worker_crop_dirs.append(crops)
        index = int(re.search(r"_chunk_(\d+)\.mp4$", str(source)).group(1))
        observations = self.chunks[index]
        sample_indices = set(range(len(observations))) if self.samples is None else set(self.samples[index])
        tracks, annotations, crop_ids = {}, [], []
        for frame_index, points in enumerate(observations):
            annotations.append([])
            for raw_id, (animal, x, y) in points.items():
                if raw_id not in crop_ids:
                    crop_ids.append(raw_id)
                    (crops / f"penguin_crop_{raw_id}.jpg").write_text(animal, encoding="utf-8")
                annotations[-1].append({"obj_id": raw_id, "cx": x, "cy": y})
                if frame_index in sample_indices:
                    tracks.setdefault(f"obj_{raw_id}", []).append({
                        "second": frame_index / self.fps, "x": x, "y": y, "area": 10.0,
                        "vx": 999.0, "vy": -999.0,
                    })
        Path(video).write_bytes(b"synthetic worker video")
        self.fixture.videos[str(video)] = [pipeline_fixtures.Frame() for _ in observations]
        Path(data).write_text(json.dumps({
            "status": "ok", "tracks": tracks, "text_annotations": annotations,
            "crops_saved": crop_ids, "crop_dir": str(crops),
        }), encoding="utf-8")

    def extract(self, chunks, *, fps=1.0, samples=None):
        self.chunks = chunks
        self.fps = fps
        self.samples = samples
        self.fixture.videos[str(self.fixture.source)] = [pipeline_fixtures.Frame() for _ in range(sum(map(len, chunks)))]
        result = self.fixture.extract()
        self.assertEqual(result, (str(self.fixture.output_json), str(self.fixture.output_video)))
        return json.loads(self.fixture.output_json.read_text(encoding="utf-8"))

    def assert_render_crop_consistency(self):
        expected_frames = [points for chunk in self.chunks for points in chunk]
        self.assertEqual(len(self.rendered_annotations), len(expected_frames))
        markers_by_global_id = {global_id: marker for global_id, marker, _ in self.classification_inputs}
        self.assertEqual(len(markers_by_global_id), len(self.classification_inputs))
        for expected, actual in zip(expected_frames, self.rendered_annotations):
            by_position = {(x, y): marker for marker, x, y in expected.values()}
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(len({ann["obj_id"] for ann in actual}), len(actual))
            for ann in actual:
                marker = by_position[(ann["cx"], ann["cy"])]
                self.assertEqual(markers_by_global_id[ann["obj_id"]], marker)
                self.assertEqual(self.semantic_names[ann["obj_id"]].rsplit("_", 1)[0], marker)
        self.assertEqual(len(set(self.worker_crop_dirs)), len(self.chunks))
        self.assertTrue(all(not path.exists() for path in self.worker_crop_dirs))

    def test_continuing_object_and_late_arrival_at_old_position_remain_distinct(self):
        tracks = self.extract([
            [frame((0, "A", 0, 20))],
            [frame((0, "A", 15, 20)), frame((0, "A", 20, 20), (2, "B", 0, 20))],
        ])
        self.assertEqual([point["x"] for point in tracks["A_1"]], [0, 15, 20])
        self.assertEqual([point["second"] for point in tracks["A_1"]], [0, 1, 2])
        self.assertEqual([(point["second"], point["x"]) for point in tracks["B_1"]], [(2, 0)])
        self.assert_render_crop_consistency()

    def test_motion_prediction_preserves_two_crossing_trajectories(self):
        tracks = self.extract([
            [frame((7, "A", 20, 20), (8, "B", 30, 20)), frame((7, "A", 24, 20), (8, "B", 26, 20))],
            [frame((33, "A", 28, 20), (2, "B", 22, 20)), frame((33, "A", 32, 20), (2, "B", 18, 20))],
        ])
        self.assertEqual([point["x"] for point in tracks["A_1"]], [20, 24, 28, 32])
        self.assertEqual([point["x"] for point in tracks["B_1"]], [30, 26, 22, 18])
        self.assert_render_crop_consistency()

    def test_raw_ids_reshuffled_between_chunks_do_not_override_identity(self):
        tracks = self.extract([
            [frame((7, "A", 0, 20), (5, "B", 60, 20)), frame((7, "A", 5, 20), (5, "B", 55, 20))],
            [frame((5, "A", 10, 20), (7, "B", 50, 20)), frame((5, "A", 15, 20), (7, "B", 45, 20))],
        ])
        self.assertEqual(set(tracks), {"A_1", "B_1"})
        self.assertEqual([point["x"] for point in tracks["A_1"]], [0, 5, 10, 15])
        self.assertEqual([point["x"] for point in tracks["B_1"]], [60, 55, 50, 45])
        self.assert_render_crop_consistency()

    def test_ambiguous_boundary_with_repeated_raw_ids_allocates_fresh_global_ids(self):
        tracks = self.extract([
            [frame((0, "A", 20, 20), (1, "B", 24, 20))],
            [frame((0, "C", 22, 20), (1, "D", 26, 20))],
        ])
        self.assertEqual(set(tracks), {"A_1", "B_1", "C_1", "D_1"})
        previous_ids = {ann["obj_id"] for ann in self.rendered_annotations[0]}
        current_ids = {ann["obj_id"] for ann in self.rendered_annotations[1]}
        self.assertTrue(previous_ids.isdisjoint(current_ids))
        self.assert_render_crop_consistency()

    def test_disappeared_before_last_frame_object_is_not_reused_at_next_boundary(self):
        tracks = self.extract([
            [frame((0, "A", 0, 20), (1, "B", 60, 20)), frame((0, "A", 5, 20))],
            [frame((0, "A", 10, 20), (1, "C", 60, 20))],
        ])
        self.assertEqual(set(tracks), {"A_1", "B_1", "C_1"})
        old_id = next(ann["obj_id"] for ann in self.rendered_annotations[0] if ann["cx"] == 60)
        new_id = next(ann["obj_id"] for ann in self.rendered_annotations[2] if ann["cx"] == 60)
        self.assertNotEqual(old_id, new_id)
        self.assert_render_crop_consistency()

    def test_empty_chunk_gap_cannot_resurrect_an_old_id(self):
        tracks = self.extract([[frame((0, "A", 10, 20))], [frame()], [frame((0, "A", 11, 20))]])
        self.assertEqual(set(tracks), {"A_1", "A_2"})
        self.assertNotEqual(self.rendered_annotations[0][0]["obj_id"], self.rendered_annotations[2][0]["obj_id"])
        self.assertEqual(sorted(point["second"] for points in tracks.values() for point in points), [0, 2])
        self.assert_render_crop_consistency()

    def test_unsampled_transient_object_reserves_a_global_id_and_its_crop(self):
        tracks = self.extract([
            [frame((10, "A", 0, 20)), frame((10, "A", 1, 20), (99, "Transient", 80, 20)), frame((10, "A", 2, 20))],
            [frame((0, "A", 3, 20), (99, "B", 140, 20))],
        ], samples=[(0, 2), (0,)])
        self.assertEqual(set(tracks), {"A_1", "B_1"})
        transient_id = next(ann["obj_id"] for ann in self.rendered_annotations[1] if ann["cx"] == 80)
        new_id = next(ann["obj_id"] for ann in self.rendered_annotations[3] if ann["cx"] == 140)
        self.assertGreater(new_id, transient_id)
        self.assertEqual(len(self.classification_inputs), 3)
        self.assert_render_crop_consistency()

    def test_many_chunks_keep_one_observation_per_identity_and_timestamp(self):
        chunks = []
        for chunk_index in range(12):
            a_raw, b_raw = chunk_index % 4, 10 + ((chunk_index + 1) % 4)
            chunks.append([frame((a_raw, "A", chunk_index * 3 + index, 20), (b_raw, "B", 100, 40)) for index in range(3)])
        tracks = self.extract(chunks)
        self.assertEqual(set(tracks), {"A_1", "B_1"})
        for points in tracks.values():
            self.assertEqual(len(points), 36)
            timestamps = [point["second"] for point in points]
            self.assertEqual(timestamps, list(range(36)))
            self.assertEqual(len(timestamps), len(set(timestamps)))
        self.assert_render_crop_consistency()

    def test_global_times_and_boundary_velocities_use_actual_frame_rate(self):
        for fps in (0.5, 2.0, 2.5):
            with self.subTest(fps=fps):
                tracks = self.extract([
                    [frame((7, "A", 0, 20)), frame((7, "A", 2, 20)), frame((7, "A", 4, 20))],
                    [frame((2, "A", 6, 20)), frame((2, "A", 8, 20))],
                ], fps=fps)
                points = tracks["A_1"]
                self.assertEqual(len(points), 5)
                for index, point in enumerate(points):
                    self.assertAlmostEqual(point["second"], index / fps)
                    self.assertAlmostEqual(point["vx"], 0.0 if index == 0 else 2.0 * fps)
                    self.assertAlmostEqual(point["vy"], 0.0)
                self.worker_crop_dirs.clear()
                self.classification_inputs.clear()

    def test_velocity_uses_elapsed_time_between_sparse_samples_across_boundary(self):
        tracks = self.extract([
            [frame((0, "A", 0, 20)), frame((0, "A", 2, 20)), frame((0, "A", 4, 20)), frame((0, "A", 6, 20))],
            [frame((4, "A", 8, 20)), frame((4, "A", 10, 20))],
        ], fps=2.5, samples=[(0, 3), (0,)])
        points = tracks["A_1"]
        self.assertEqual([point["x"] for point in points], [0, 6, 8])
        for point, expected_time, expected_speed in zip(points, (0, 1.2, 1.6), (0, 5, 5)):
            self.assertAlmostEqual(point["second"], expected_time)
            self.assertAlmostEqual(point["vx"], expected_speed)
        self.assert_render_crop_consistency()


    def test_malformed_telemetry_cannot_contradict_annotated_frames(self):
        cases = ('canonical_duplicate', 'wrong_timestamp', 'fractional_frame',
                 'wrong_location', 'missing_object', 'duplicate_timestamp',
                 'near_duplicate_frame', 'negative_time', 'reversed_time')
        original_worker = self.write_worker_result
        for case in cases:
            with self.subTest(case=case):
                def malformed_result(args):
                    original_worker(args)
                    path = Path(args[2])
                    data = json.loads(path.read_text(encoding='utf-8'))
                    points = data['tracks']['obj_0']
                    if case == 'canonical_duplicate':
                        data['tracks']['obj_00'] = copy.deepcopy(points)
                    elif case == 'wrong_timestamp':
                        for point in points:
                            point['second'] += 100
                    elif case == 'fractional_frame':
                        points[0]['second'] = 0.25
                    elif case == 'wrong_location':
                        points[0]['x'] += 999
                    elif case == 'missing_object':
                        data['text_annotations'][0] = []
                    elif case == 'duplicate_timestamp':
                        points[1]['second'] = points[0]['second']
                    elif case == 'near_duplicate_frame':
                        points[1] = dict(points[0], second=1e-8)
                    elif case == 'negative_time':
                        points[0]['second'] = -1
                    else:
                        points.reverse()
                    path.write_text(json.dumps(data), encoding='utf-8')
                with mock.patch.object(self, 'write_worker_result', side_effect=malformed_result):
                    with self.assertRaises(pipeline_fixtures.TrackingError):
                        self.extract([[frame((0, 'A', 0, 20)), frame((0, 'A', 1, 20))]])
                self.assertFalse(self.fixture.output_json.exists())
                self.assertFalse(self.fixture.output_video.exists())

    def test_compensating_per_chunk_annotation_counts_cannot_hide_frame_mismatch(self):
        original_worker = self.write_worker_result
        calls = []
        def malformed_counts(args):
            original_worker(args)
            path = Path(args[2])
            data = json.loads(path.read_text(encoding='utf-8'))
            if not calls:
                # Keep the one sparse sample valid; only the video/IPC frame count differs.
                data['text_annotations'].pop()
            else:
                data['text_annotations'].append([])
            calls.append(True)
            path.write_text(json.dumps(data), encoding='utf-8')
        with mock.patch.object(self, 'write_worker_result', side_effect=malformed_counts):
            with self.assertRaisesRegex(pipeline_fixtures.TrackingError, 'frame count differs'):
                self.extract([
                    [frame((0, 'A', 0, 20)), frame((0, 'A', 1, 20))],
                    [frame((0, 'A', 2, 20)), frame((0, 'A', 3, 20))],
                ], samples=[(0,), (0, 1)])
        self.assertFalse(self.fixture.output_json.exists())
        self.assertFalse(self.fixture.output_video.exists())

    def test_chunk_and_worker_fps_must_match_the_proxy(self):
        original_capture = sys.modules['cv2'].VideoCapture
        for mismatch in ('input', 'worker'):
            with self.subTest(mismatch=mismatch):
                class MismatchedCapture(original_capture):
                    def get(capture, key):
                        value = super().get(key)
                        if key == sys.modules['cv2'].CAP_PROP_FPS:
                            is_chunk = re.search(r'_chunk_\d+\.mp4$', capture.path)
                            is_worker = capture.path.endswith('.temp.mp4')
                            if (mismatch == 'input' and is_chunk) or (mismatch == 'worker' and is_worker):
                                return value + 1
                        return value
                with mock.patch.object(sys.modules['cv2'], 'VideoCapture', MismatchedCapture):
                    with self.assertRaisesRegex(pipeline_fixtures.TrackingError, 'frame rate differs'):
                        self.extract([[frame((0, 'A', 0, 20))]])
                self.assertFalse(self.fixture.output_json.exists())
                self.assertFalse(self.fixture.output_video.exists())


if __name__ == "__main__":
    unittest.main()
