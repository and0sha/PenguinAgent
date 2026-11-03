"""Security and integration checks for bounded tracking charts (no model server)."""

import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools import tracking_charts as charts


class SafeChartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name)
        self.tracks = {
            'A_1': [
                {'second': 0, 'x': 0, 'y': 0, 'area': 10, 'vx': 999, 'vy': 999},
                {'second': 2, 'x': 3, 'y': 4, 'area': 20, 'vx': 999, 'vy': 999},
            ],
            'B_1': [
                {'second': 0, 'x': 8, 'y': 9, 'area': 12},
                {'second': 4, 'x': 8, 'y': 9, 'area': 12},
            ],
        }

    def test_all_chart_types_create_bounded_real_pngs_without_mutation(self):
        original = copy.deepcopy(self.tracks)
        cwd = Path.cwd()
        for chart_type in sorted(charts.CHART_TYPES):
            with self.subTest(chart_type=chart_type):
                result = charts.render_tracking_chart(self.tracks, self.output_dir, chart_type)
                image_path = Path(result['path'])
                self.assertTrue(result['success'])
                self.assertEqual(image_path.parent, self.output_dir.resolve())
                self.assertRegex(image_path.name, r'^tracking_chart_[0-9a-f]{32}\.png$')
                contents = image_path.read_bytes()
                self.assertEqual(contents[:8], b'\x89PNG\r\n\x1a\n')
                self.assertEqual(struct.unpack('>II', contents[16:24]), (960, 600))
                self.assertEqual(result['summary']['observation_count'], 4)
        self.assertEqual(self.tracks, original)
        self.assertEqual(Path.cwd(), cwd)

    def test_speed_uses_elapsed_time_and_positions_not_stored_velocity(self):
        result = charts.render_tracking_chart(self.tracks, self.output_dir, 'speed', 'A_1')
        self.assertEqual(result['summary']['unit'], 'pixels/second')
        self.assertEqual(result['summary']['per_track']['A_1']['maximum_speed_px_per_second'], 2.5)
        self.assertEqual(result['summary']['selected_track_count'], 1)

    def test_population_counts_only_observed_exact_timestamps(self):
        result = charts.render_tracking_chart(self.tracks, self.output_dir, 'population')
        self.assertEqual(result['summary']['maximum_count'], 2)
        self.assertEqual(result['summary']['minimum_count'], 1)
        self.assertIn('not interpolated', result['summary']['population_definition'])

    def test_area_statistics_and_sorted_copy(self):
        self.tracks['A_1'].reverse()
        result = charts.render_tracking_chart(self.tracks, self.output_dir, 'area', 'A_1')
        self.assertEqual(result['summary']['per_track']['A_1']['mean_area_px_squared'], 15)
        self.assertEqual(self.tracks['A_1'][0]['second'], 2)

    def test_titles_and_ids_are_literal_text_not_math_or_code(self):
        marker = self.output_dir / 'must_not_exist'
        literal = f"open({str(marker)!r}, 'w').write('marker') $\\invalidmathcommand$"
        tracks = {r'$\unknowncommand$': self.tracks['A_1']}
        result = charts.render_tracking_chart(tracks, self.output_dir, 'trajectory', title=literal)
        self.assertTrue(Path(result['path']).exists())
        self.assertFalse(marker.exists())

    def test_invalid_tool_payloads_are_rejected_without_calling_renderer(self):
        callback = mock.Mock()
        invalid = [
            None, [], 'trajectory', {},
            {'chart_type': 'histogram'},
            {'chart_type': '__import__("os")'},
            {'chart_type': 'trajectory', 'python_code': 'pass'},
            {'chart_type': 'trajectory', 'output_path': '../escape.png'},
            {'chart_type': 'trajectory', 'formula': 'x**2'},
            {'chart_type': 'trajectory', 'track_id': 1},
            {'chart_type': 'trajectory', 'title': []},
            {'chart_type': 'trajectory', 'title': 'a' * (charts.MAX_TITLE_LENGTH + 1)},
            {'chart_type': 'trajectory', 'title': 'multiline\ntitle'},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                result = json.loads(charts.dispatch_tracking_chart(payload, callback))
                self.assertFalse(result['success'])
        callback.assert_not_called()

    def test_valid_payload_uses_only_documented_arguments(self):
        callback = mock.Mock(return_value='done')
        self.assertEqual(charts.dispatch_tracking_chart({'chart_type': 'area'}, callback), 'done')
        callback.assert_called_once_with(chart_type='area', track_id='', title='')

    def test_nonfinite_and_invalid_observations_fail_before_writing(self):
        invalid = [float('nan'), float('inf'), -float('inf'), True, '5', None, 1e100]
        for value in invalid:
            with self.subTest(value=value):
                tracks = copy.deepcopy(self.tracks)
                tracks['A_1'][0]['x'] = value
                with self.assertRaises(ValueError):
                    charts.render_tracking_chart(tracks, self.output_dir, 'trajectory')
        for field, value in [('second', -1), ('area', -1), ('vx', float('nan'))]:
            with self.subTest(field=field):
                tracks = copy.deepcopy(self.tracks)
                tracks['A_1'][0][field] = value
                with self.assertRaises(ValueError):
                    charts.render_tracking_chart(tracks, self.output_dir, 'area')
        self.assertEqual(list(self.output_dir.iterdir()), [])

    def test_missing_fields_unknown_track_duplicates_and_empty_data(self):
        cases = [({}, 'trajectory', ''), (self.tracks, 'trajectory', '../outside'),
                 ({'A': []}, 'area', ''), ({'A': [{'second': 0}]}, 'trajectory', ''),
                 ({'A': [{'second': 0, 'x': 1, 'y': 2}]}, 'speed', ''),
                 ({'A': [self.tracks['A_1'][0], self.tracks['A_1'][0]]}, 'trajectory', '')]
        for tracks, chart_type, track_id in cases:
            with self.subTest(tracks=tracks, chart_type=chart_type):
                with self.assertRaises(ValueError):
                    charts.render_tracking_chart(tracks, self.output_dir, chart_type, track_id)

    def test_track_and_point_limits(self):
        with mock.patch.object(charts, 'MAX_TRACKS', 1):
            with self.assertRaisesRegex(ValueError, 'select a track_id'):
                charts.render_tracking_chart(self.tracks, self.output_dir, 'area')
            self.assertTrue(charts.render_tracking_chart(self.tracks, self.output_dir, 'area', 'A_1')['success'])
        with mock.patch.object(charts, 'MAX_POINTS_PER_TRACK', 1):
            with self.assertRaisesRegex(ValueError, 'per track'):
                charts.render_tracking_chart(self.tracks, self.output_dir, 'area')
        with mock.patch.object(charts, 'MAX_TOTAL_POINTS', 3):
            with self.assertRaisesRegex(ValueError, 'total observations'):
                charts.render_tracking_chart(self.tracks, self.output_dir, 'area')

    def test_tiny_time_interval_cannot_return_infinite_speed(self):
        tracks = {'A': [{'second': 0, 'x': 0, 'y': 0}, {'second': 1e-320, 'x': 1, 'y': 1}]}
        with self.assertRaisesRegex(ValueError, 'finite numeric range'):
            charts.render_tracking_chart(tracks, self.output_dir, 'speed')


class ReportChartIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # patch.dict restores newly imported modules too. Warm real native plotting
        # dependencies before the isolated ML stubs to avoid reimporting C extensions.
        with tempfile.TemporaryDirectory() as output:
            charts.render_tracking_chart({'warmup': [{'second': 0, 'x': 0, 'y': 0}]}, output, 'trajectory')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name)
        self.tracks = {'penguin_1': [{'second': 0, 'x': 0, 'y': 0, 'area': 1}, {'second': 2, 'x': 3, 'y': 4, 'area': 2}]}

    def run_report(self, module_name, tool_calls):
        fake_ollama = types.ModuleType('ollama')
        fake_torch = types.ModuleType('torch')
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_cv2 = types.ModuleType('cv2')
        turns = []
        def chat(**kwargs):
            if not kwargs.get('tools'):
                return {'message': {'content': 'unloaded'}}
            turns.append(copy.deepcopy(kwargs['messages']))
            offered_names = [callback.__name__ for callback in kwargs['tools']]
            self.assertIn('generate_tracking_chart', offered_names)
            self.assertNotIn('generate_custom_chart', offered_names)
            if len(turns) == 1 and tool_calls:
                return {'message': {'role': 'assistant', 'content': '', 'tool_calls': tool_calls}}
            return {'message': {'role': 'assistant', 'content': 'Отчёт: β, пингвины.'}}
        fake_ollama.chat = chat
        from tools.tracking_results import TrackingResult
        source_path = Path(__file__).resolve().parents[1] / (module_name + '.py')
        with mock.patch.dict(sys.modules, {'ollama': fake_ollama, 'torch': fake_torch, 'cv2': fake_cv2}):
            spec = importlib.util.spec_from_file_location('_safe_test_' + module_name, source_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            mixin = module.ReportMixin if module_name == 'behave_report' else module.HuddleMixin
            agent = mixin()
            agent.videos_dir = str(self.output_dir)
            agent.model = 'stub'
            validated_tracks = copy.deepcopy(self.tracks)
            agent.tracks = {'stale_cache': []}
            def require_tracking_result(video_name, output_dir=None):
                run_dir = Path(output_dir) / 'runs' / 'chart-only'
                run_dir.mkdir(parents=True, exist_ok=True)
                agent.tracks = validated_tracks
                return TrackingResult(
                    run_id='chart-only', output_dir=run_dir,
                    json_path=run_dir / 'tracks.json',
                    annotated_video_path=run_dir / 'annotated.mp4',
                    tracks=validated_tracks,
                )
            agent.require_tracking_result = require_tracking_result
            method = agent.generate_anomaly_report if module_name == 'behave_report' else agent.generate_huddle_paper
            result = method('nonexistent.mp4', output_dir=str(self.output_dir / module_name))
            self.assertEqual(agent.tracks, self.tracks)
        tool_results = [json.loads(message['content']) for message in turns[-1] if message['role'] == 'tool']
        return result, tool_results, turns

    def test_both_reports_dispatch_valid_charts_and_reject_legacy_or_unknown_args(self):
        calls = [
            {'function': {'name': 'generate_tracking_chart', 'arguments': {'chart_type': 'speed', 'title': 'Пингвины β'}}},
            {'function': {'name': 'generate_tracking_chart', 'arguments': {'chart_type': 'trajectory', 'python_code': 'raise RuntimeError()'}}},
            {'function': {'name': 'generate_tracking_chart', 'arguments': ['trajectory']}},
            {'function': {'name': 'generate_custom_chart', 'arguments': {'python_code': 'raise RuntimeError()'}}},
        ]
        for module_name, report_filename in [('behave_report', 'final_scientific_report.md'), ('behave_huddle', 'final_huddle_report.md')]:
            with self.subTest(module_name=module_name):
                result, tool_results, turns = self.run_report(module_name, calls)
                self.assertTrue(result.startswith('Отчёт: β, пингвины.'))
                self.assertIn('data:image/png;base64,', result)
                self.assertTrue(tool_results[0]['success'])
                self.assertEqual(tool_results[0]['summary']['per_track']['penguin_1']['maximum_speed_px_per_second'], 2.5)
                self.assertTrue(all(not entry['success'] for entry in tool_results[1:]))
                output = self.output_dir / module_name / 'runs' / 'chart-only'
                self.assertEqual((output / report_filename).read_text(encoding='utf-8'), result)
                self.assertEqual(len(list(output.glob('*.png'))), 1)
                prompts = '\n'.join(message['content'] for message in turns[0])
                self.assertIn('generate_tracking_chart', prompts)
                self.assertNotIn('generate_custom_chart', prompts)
                if module_name == 'behave_huddle':
                    html = (output / 'huddle_simulation.html').read_text(encoding='utf-8')
                    self.assertIn('β', html)

    def test_huddle_embedded_track_id_cannot_close_script(self):
        self.tracks = {'</script><script>alert(1)</script>': self.tracks['penguin_1']}
        result, _, _ = self.run_report('behave_huddle', [])
        self.assertTrue(result.startswith('Отчёт'))
        html = (self.output_dir / 'behave_huddle' / 'runs' / 'chart-only' / 'huddle_simulation.html').read_text(encoding='utf-8')
        self.assertNotIn('</script><script>alert(1)', html)
        self.assertIn('\\u003c/script>', html)


if __name__ == '__main__':
    unittest.main()
