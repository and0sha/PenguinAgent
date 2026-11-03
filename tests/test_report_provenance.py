"""Report provenance checks using real tracking runs and lightweight I/O stubs."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.tracking_results import TrackingError


REPOSITORY = Path(__file__).resolve().parents[1]


def load_local_module(name):
    spec = importlib.util.spec_from_file_location('_provenance_test_' + name, REPOSITORY / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReportProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.videos = self.directory / 'videos'
        self.videos.mkdir()
        self.video_a = self.videos / 'a.mp4'
        self.video_b = self.videos / 'b.mp4'
        self.video_a.write_bytes(b'video-content-A')
        self.video_b.write_bytes(b'video-content-B')
        self.output_root = self.directory / 'output'
        self.fail_extractor = False
        self.before_extract = None
        self.captured_videos = []
        self.expected_tracks = {'A_1': [{'second': 0, 'x': 0, 'y': 0, 'area': 1}, {'second': 1, 'x': 1, 'y': 2, 'area': 2}]}
        fake_extractor_module = types.ModuleType('tools.feature_tracking')
        fake_extractor_module.extract_tracks = mock.Mock(side_effect=self.extract_tracks)
        self.ollama = types.ModuleType('ollama')
        self.ollama.chat = mock.Mock(return_value={'message': {'role': 'assistant', 'content': 'Validated report β'}})
        fake_torch = types.ModuleType('torch')
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_cv2 = types.ModuleType('cv2')
        fake_cv2.CAP_PROP_FPS = 1
        fake_cv2.CAP_PROP_FRAME_COUNT = 2
        captured_videos = self.captured_videos
        class EmptyCapture:
            def __init__(self, path):
                captured_videos.append(Path(path))
            def get(self, property_id):
                return 30 if property_id == fake_cv2.CAP_PROP_FPS else 0
            def read(self):
                return False, None
            def release(self):
                pass
        fake_cv2.VideoCapture = EmptyCapture
        patcher = mock.patch.dict(sys.modules, {
            'tools.feature_tracking': fake_extractor_module,
            'ollama': self.ollama, 'torch': fake_torch, 'cv2': fake_cv2,
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tracking_module = load_local_module('behave_tracking')
        report_module = load_local_module('behave_report')
        huddle_module = load_local_module('behave_huddle')
        class Agent(self.tracking_module.TrackingMixin, report_module.ReportMixin, huddle_module.HuddleMixin):
            pass
        self.agent = Agent()
        self.agent.videos_dir = str(self.videos)
        self.agent.model = 'stub-model'
        self.agent.tracks = None

    def extract_tracks(self, video_path, output_json, output_video, **kwargs):
        if self.before_extract:
            self.before_extract()
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(self.expected_tracks), encoding='utf-8')
        Path(output_video).write_bytes(b'nonempty annotated video fixture')
        if self.fail_extractor:
            raise RuntimeError('intentional extractor failure')
        return str(output_json), str(output_video)

    def run_success(self, video_name='a.mp4'):
        self.agent.run_tracking(video_name, output_dir=str(self.output_root))
        result = self.agent.require_tracking_result(video_name, str(self.output_root))
        self.assertEqual(result.tracks, self.expected_tracks)
        self.assertTrue(Path(result.json_path).is_file())
        self.assertTrue(Path(result.annotated_video_path).is_file())
        return result

    def reports(self):
        return [self.agent.generate_anomaly_report, self.agent.generate_huddle_paper]

    def assert_reports_rejected(self, video_name='a.mp4', output_root=None):
        output_root = self.output_root if output_root is None else output_root
        self.ollama.chat.reset_mock()
        for report in self.reports():
            with self.subTest(report=report.__name__):
                with self.assertRaises(TrackingError):
                    report(video_name, output_dir=str(output_root))
        self.ollama.chat.assert_not_called()

    def test_missing_manifest_rejects_cached_tracks_before_any_report_output(self):
        self.agent.tracks = copy.deepcopy(self.expected_tracks)
        self.assert_reports_rejected()
        self.assertFalse(self.output_root.exists())
        self.assertEqual(self.captured_videos, [])

    def test_tracks_from_a_cannot_generate_report_for_video_b(self):
        result = self.run_success('a.mp4')
        self.agent.tracks = copy.deepcopy(self.expected_tracks)
        self.assert_reports_rejected('b.mp4')
        self.assertEqual(list(Path(result.output_dir).glob('final*report.md')), [])
        self.assertEqual(self.captured_videos, [])

    def test_same_basename_replaced_bytes_rejects_even_if_size_and_mtime_match(self):
        self.run_success()
        stat = self.video_a.stat()
        self.video_a.write_bytes(b'video-content-X')
        os.utime(self.video_a, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(self.video_a.stat().st_size, stat.st_size)
        self.assert_reports_rejected()

    def test_legacy_loose_json_is_not_accepted_by_either_report(self):
        self.output_root.mkdir()
        (self.output_root / 'tracks.json').write_text(json.dumps(self.expected_tracks), encoding='utf-8')
        (self.output_root / 'annotated.mp4').write_bytes(b'legacy annotated video')
        self.agent.tracks = copy.deepcopy(self.expected_tracks)
        self.assert_reports_rejected()
        self.assertEqual(list(self.output_root.glob('final*report.md')), [])

    def test_failed_latest_attempt_blocks_older_success_and_cached_tracks(self):
        previous = self.run_success()
        self.fail_extractor = True
        with self.assertRaisesRegex(TrackingError, 'intentional extractor failure'):
            self.agent.run_tracking('a.mp4', output_dir=str(self.output_root))
        self.assertTrue(Path(previous.json_path).is_file())
        self.agent.tracks = copy.deepcopy(self.expected_tracks)
        self.assert_reports_rejected()
        self.assertEqual(list(Path(previous.output_dir).glob('final*report.md')), [])

    def test_running_latest_attempt_blocks_older_success(self):
        self.run_success()
        def verify_during_extraction():
            self.agent.tracks = copy.deepcopy(self.expected_tracks)
            self.assert_reports_rejected()
        self.before_extract = verify_during_extraction
        self.agent.run_tracking('a.mp4', output_dir=str(self.output_root))

    def test_valid_reports_reload_requested_run_and_write_only_to_run_directory(self):
        previous = self.run_success()
        self.agent.tracks = {'poisoned_cached_penguin': []}
        for report in self.reports():
            with self.subTest(report=report.__name__):
                report_text = report('a.mp4', output_dir=str(self.output_root))
                self.assertTrue(report_text.startswith('Validated report β'))
                self.assertEqual(self.agent.tracks, self.expected_tracks)
        run_dir = Path(previous.output_dir)
        self.assertNotEqual(run_dir, self.output_root)
        self.assertTrue((run_dir / 'final_scientific_report.md').is_file())
        self.assertTrue((run_dir / 'final_huddle_report.md').is_file())
        self.assertTrue((run_dir / 'huddle_simulation.html').is_file())
        self.assertFalse((self.output_root / 'final_scientific_report.md').exists())
        self.assertFalse((self.output_root / 'final_huddle_report.md').exists())
        self.assertEqual(self.captured_videos, [self.video_a])
        prompts = '\n'.join(call.kwargs['messages'][-1]['content'] for call in self.ollama.chat.call_args_list)
        self.assertNotIn('poisoned_cached_penguin', prompts)
        self.assertIn('A_1', prompts)

    def test_source_replaced_during_model_generation_prevents_report_publication(self):
        for report in self.reports():
            with self.subTest(report=report.__name__):
                self.video_a.write_bytes(b'video-content-A')
                result = self.run_success()
                def change_source(**kwargs):
                    self.video_a.write_bytes(b'video-content-X')
                    return {'message': {'role': 'assistant', 'content': 'Must not be published'}}
                self.ollama.chat.side_effect = change_source
                with self.assertRaises(TrackingError):
                    report('a.mp4', output_dir=str(self.output_root))
                self.ollama.chat.side_effect = None
                self.assertEqual(list(Path(result.output_dir).glob('final*report.md')), [])
                self.assertFalse((Path(result.output_dir) / 'huddle_simulation.html').exists())

    def test_failed_rerun_during_model_generation_prevents_report_publication(self):
        for report in self.reports():
            with self.subTest(report=report.__name__):
                result = self.run_success()
                triggered = []
                def fail_new_tracking(**kwargs):
                    if not triggered:
                        triggered.append(True)
                        self.fail_extractor = True
                        try:
                            with self.assertRaises(TrackingError):
                                self.agent.run_tracking('a.mp4', output_dir=str(self.output_root))
                        finally:
                            self.fail_extractor = False
                    return {'message': {'role': 'assistant', 'content': 'Must not be published'}}
                self.ollama.chat.side_effect = fail_new_tracking
                with self.assertRaises(TrackingError):
                    report('a.mp4', output_dir=str(self.output_root))
                self.ollama.chat.side_effect = None
                self.assertEqual(list(Path(result.output_dir).glob('final*report.md')), [])
                self.assertFalse((Path(result.output_dir) / 'huddle_simulation.html').exists())

    def test_new_successful_run_during_model_generation_cannot_publish_old_report(self):
        for report in self.reports():
            with self.subTest(report=report.__name__):
                result = self.run_success()
                triggered = []
                def publish_new_tracking(**kwargs):
                    if not triggered:
                        triggered.append(True)
                        new_result = self.run_success()
                        self.assertNotEqual(new_result.run_id, result.run_id)
                    return {'message': {'role': 'assistant', 'content': 'Must not be published'}}
                self.ollama.chat.side_effect = publish_new_tracking
                with self.assertRaisesRegex(TrackingError, 'run changed'):
                    report('a.mp4', output_dir=str(self.output_root))
                self.ollama.chat.side_effect = None
                self.assertEqual(list(Path(result.output_dir).glob('final*report.md')), [])
                self.assertFalse((Path(result.output_dir) / 'huddle_simulation.html').exists())

    def test_tampered_tracking_artifact_is_rejected_before_model(self):
        result = self.run_success()
        Path(result.json_path).write_text(json.dumps({'tampered': self.expected_tracks['A_1']}), encoding='utf-8')
        self.assert_reports_rejected()


if __name__ == '__main__':
    unittest.main()
