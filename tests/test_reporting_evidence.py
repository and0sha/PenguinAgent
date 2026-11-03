"""Bounded numeric evidence, behavioral context and honest report/viewer regressions."""

import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.tracking_charts import (build_tracking_evidence, EVIDENCE_MAX_TRACKS, EVIDENCE_MAX_POINTS_PER_TRACK,
                                   ToolResponseBudget, atomic_write_text)
from tools.tracking_results import TrackingError, TrackingResult

ROOT = Path(__file__).resolve().parents[1]


def load_report_module(name, ollama):
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    with mock.patch.dict(sys.modules, {'ollama': ollama, 'torch': torch}):
        spec = importlib.util.spec_from_file_location('_evidence_test_' + name, ROOT / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def points(moving=True):
    return [{'second': 0, 'x': 0, 'y': 0, 'area': 10},
            {'second': 2, 'x': 3 if moving else 0, 'y': 4 if moving else 0, 'area': 20}]


class TrackingEvidenceTests(unittest.TestCase):
    def test_motion_and_elapsed_time_are_sent_as_numeric_evidence(self):
        moving = build_tracking_evidence({'A_1': points()})['tracks'][0]
        still = build_tracking_evidence({'A_1': points(False)})['tracks'][0]
        self.assertEqual(moving['observation_count'], 2)
        self.assertEqual(moving['observed_span_seconds'], 2)
        self.assertEqual(moving['endpoint_displacement_px'], 5)
        self.assertEqual(moving['maximum_sampled_speed_px_per_second'], 2.5)
        self.assertEqual(still['maximum_sampled_speed_px_per_second'], 0)
        self.assertNotEqual(moving, still)

    def test_preview_velocity_uses_full_neighbor_not_previous_preview(self):
        observations = [{'second': i, 'x': i*i, 'y': 0} for i in range(100)]
        original = copy.deepcopy(observations)
        evidence = build_tracking_evidence({'A_1': observations})['tracks'][0]
        self.assertEqual(evidence['observations'][-1]['vx'], 197)
        self.assertEqual(evidence['preview_observation_count'], EVIDENCE_MAX_POINTS_PER_TRACK)
        self.assertEqual(evidence['omitted_observation_count'], 100-EVIDENCE_MAX_POINTS_PER_TRACK)
        self.assertEqual(observations, original)

    def test_work_and_prompt_size_stay_bounded_for_very_large_tracks(self):
        class VirtualTrack(list):
            reads = 0
            def __len__(self):
                return 10**9
            def __getitem__(self, index):
                self.reads += 1
                return {'second': index / 5, 'x': index, 'y': 0, 'area': 5}
        tracks = {f'penguin_{i}': VirtualTrack() for i in range(100)}
        evidence = build_tracking_evidence(tracks)
        self.assertEqual(evidence['included_track_count'], EVIDENCE_MAX_TRACKS)
        self.assertEqual(evidence['omitted_track_count'], 100-EVIDENCE_MAX_TRACKS)
        self.assertLessEqual(sum(track.reads for track in tracks.values()), EVIDENCE_MAX_TRACKS * EVIDENCE_MAX_POINTS_PER_TRACK * 2)
        self.assertLess(len(json.dumps(evidence)), 50000)
        self.assertEqual(evidence['tracks'][0]['observed_span_seconds'], (10**9-1)/5)

    def test_empty_scene_has_no_fabricated_observations(self):
        evidence = build_tracking_evidence({})
        self.assertEqual(evidence['total_track_count'], 0)
        self.assertEqual(evidence['tracks'], [])

    def test_invalid_data_or_unknown_id_is_not_silently_interpreted(self):
        with self.assertRaisesRegex(ValueError, 'Unknown track_id'):
            build_tracking_evidence({'A_1': points()}, 'missing')
        invalid = points()
        invalid[1]['second'] = 0
        with self.assertRaises(ValueError):
            build_tracking_evidence({'A_1': invalid})
        invalid = points()
        invalid[1]['x'] = float('nan')
        with self.assertRaises(ValueError):
            build_tracking_evidence({'A_1': invalid})


    def test_tool_response_budget_is_cumulative_and_stops_with_a_notice(self):
        budget = ToolResponseBudget(1000)
        responses = [budget.take('x'*400) for _ in range(20)]
        self.assertLessEqual(sum(map(len, responses)), 1000)
        self.assertTrue(budget.exhausted)
        self.assertTrue(any('budget exhausted' in response for response in responses))

    def test_atomic_utf8_write_preserves_existing_text_if_publication_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.md'
            atomic_write_text(path, 'Previous β report')
            with mock.patch('os.replace', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    atomic_write_text(path, 'Replacement report')
            self.assertEqual(path.read_text(encoding='utf-8'), 'Previous β report')
            self.assertEqual(list(path.parent.glob('.*.tmp')), [])


class BehaviorEvidenceFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'output' / 'runs' / ('a'*32)
        self.run.mkdir(parents=True)
        self.ollama = types.ModuleType('ollama')
        self.ollama.chat = mock.Mock(return_value={'message': {'content': 'Evidence-based answer'}})
        module = load_report_module('behave_analysis', self.ollama)
        self.agent = module.AnalysisMixin()
        self.agent.model = 'stub'
        self.agent.tracks = {'POISONED_CACHE': points(False)}
        self.agent._tracking_context = (str(self.root/'video.mp4'), 'hash', str(self.root/'output'))
        self.result = TrackingResult('a'*32, self.run, self.run/'tracks.json', self.run/'annotated.mp4', {'A_1': points()})
        self.agent.tracking_result = self.result
        self.agent.require_tracking_result = mock.Mock(side_effect=lambda *args: self.result)

    def prompt(self):
        return self.ollama.chat.call_args.kwargs['messages'][-1]['content']

    def test_global_question_and_real_telemetry_reach_model(self):
        self.agent.analyze_behavior(query='Compare the observed motion', output_dir=self.run)
        prompt = self.prompt()
        self.assertIn('Compare the observed motion', prompt)
        self.assertIn('A_1', prompt)
        self.assertIn('2.5', prompt)
        self.assertNotIn('POISONED_CACHE', prompt)
        self.assertIn('insufficient evidence', prompt)
        self.assertEqual(self.agent.require_tracking_result.call_count, 2)

    def test_stationary_and_moving_tracks_produce_different_prompts(self):
        self.agent.analyze_behavior(output_dir=self.run)
        moving = self.prompt()
        self.result = TrackingResult('a'*32, self.run, self.run/'tracks.json', self.run/'annotated.mp4', {'A_1': points(False)})
        self.agent.analyze_behavior(output_dir=self.run)
        self.assertNotEqual(moving, self.prompt())

    def test_only_current_exact_report_files_are_read_with_a_budget(self):
        (self.root/'other_video_report.md').write_text('OTHER_VIDEO_SECRET A_1', encoding='utf-8')
        (self.run/'arbitrary.json').write_text('ARBITRARY_SECRET A_1', encoding='utf-8')
        (self.run/'final_scientific_report.md').write_text('CURRENT_RUN_PREFIX\n'+'x'*50000, encoding='utf-8')
        self.agent.analyze_behavior(specific_id='A_1', query='Describe this track', output_dir=self.run)
        prompt = self.prompt()
        self.assertIn('CURRENT_RUN_PREFIX', prompt)
        self.assertNotIn('OTHER_VIDEO_SECRET', prompt)
        self.assertNotIn('ARBITRARY_SECRET', prompt)
        self.assertLess(len(prompt), 13000)

    def test_missing_or_wrong_run_rejects_before_model(self):
        with self.assertRaises(TrackingError):
            self.agent.analyze_behavior(output_dir=None)
        with self.assertRaises(TrackingError):
            self.agent.analyze_behavior(output_dir=self.root/'old_run')
        self.agent._tracking_context = None
        with self.assertRaises(TrackingError):
            self.agent.analyze_behavior(output_dir=self.run)
        self.ollama.chat.assert_not_called()

    def test_unknown_id_rejects_before_model(self):
        with self.assertRaises(ValueError):
            self.agent.analyze_behavior(specific_id='missing', output_dir=self.run)
        self.ollama.chat.assert_not_called()

    def test_transport_failure_is_an_exception_not_an_answer_string(self):
        self.ollama.chat.side_effect = RuntimeError('server unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Behavior analysis failed'):
            self.agent.analyze_behavior(output_dir=self.run)

    def test_run_change_while_model_answers_does_not_return_stale_analysis(self):
        def change_run(**kwargs):
            self.result = TrackingResult('b'*32, self.run, self.run/'tracks.json', self.run/'annotated.mp4', {'A_1': points()})
            return {'message': {'content': 'Stale answer'}}
        self.ollama.chat.side_effect = change_run
        with self.assertRaisesRegex(TrackingError, 'run changed'):
            self.agent.analyze_behavior(output_dir=self.run)


class ReportInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ollama = types.ModuleType('ollama')
        self.ollama.chat = mock.Mock(return_value={'message': {'content': 'Observation report'}})
        self.report = load_report_module('behave_report', self.ollama)
        self.huddle = load_report_module('behave_huddle', self.ollama)

    def agent(self, mixin):
        agent = mixin()
        agent.model = 'actual-configured-model'
        agent.videos_dir = str(self.root)
        run = self.root/'run'
        run.mkdir(exist_ok=True)
        result = TrackingResult('a'*32, run, run/'tracks.json', run/'annotated.mp4', {'A_1': points()})
        agent.require_tracking_result = mock.Mock(return_value=result)
        return agent, run

    def test_question_files_are_module_relative_bounded_and_confined(self):
        module_path = self.root/'package'/'behave_report.py'
        questions = module_path.parent/'questions'
        questions.mkdir(parents=True)
        (questions/'100.txt').write_text('BUNDLED_QUESTION\n'+'q'*10000, encoding='utf-8')
        (self.root/'secret.txt').write_text('PRIVATE_SECRET', encoding='utf-8')
        with mock.patch.object(self.report, '__file__', str(module_path)):
            text = self.report._question_context(['../../secret_1'])
        self.assertTrue(text.startswith('BUNDLED_QUESTION'))
        self.assertNotIn('PRIVATE_SECRET', text)
        self.assertLessEqual(len(text), self.report.MAX_QUESTION_CHARS)

    def test_long_video_sampling_has_a_read_budget_and_explicit_timestamps(self):
        source = self.root/'source.mp4'
        source.write_bytes(b'video')
        frame = types.SimpleNamespace(shape=(1080,1920,3))
        capture = mock.Mock()
        capture.get.side_effect = lambda key: 30 if key == 1 else 3600
        capture.set.return_value = True
        capture.read.return_value = (True, frame)
        cv2 = types.ModuleType('cv2')
        cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_POS_FRAMES = 1,2,3
        cv2.VideoCapture = mock.Mock(return_value=capture)
        cv2.imwrite = mock.Mock(return_value=True)
        with mock.patch.dict(sys.modules, {'cv2': cv2}):
            images, metadata = self.report._extract_report_frames(source, self.root)
        self.assertEqual(len(images), self.report.MAX_REPORT_FRAMES)
        self.assertEqual(capture.read.call_count, self.report.MAX_REPORT_FRAMES)
        self.assertEqual(metadata[1]['second'], 120/self.report.MAX_REPORT_FRAMES)
        self.assertEqual(metadata[0]['source_width'], 1920)
        self.assertIn('nominal', metadata[0]['timestamp_basis'])
        capture.release.assert_called_once()

    def test_failed_image_write_releases_capture_and_fails_explicitly(self):
        source = self.root/'source.mp4'
        source.write_bytes(b'video')
        cv2 = types.ModuleType('cv2')
        cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_POS_FRAMES = 1,2,3
        capture = mock.Mock()
        capture.get.side_effect = lambda key: 1 if key == 1 else 2
        capture.set.return_value = True
        capture.read.return_value = (True, types.SimpleNamespace(shape=(10,10,3)))
        cv2.VideoCapture = mock.Mock(return_value=capture)
        cv2.imwrite = mock.Mock(return_value=False)
        with mock.patch.dict(sys.modules, {'cv2': cv2}):
            with self.assertRaisesRegex(TrackingError, 'save a report frame'):
                self.report._extract_report_frames(source, self.root)
        capture.release.assert_called_once()

    def test_invalid_anomaly_weights_do_not_reach_model(self):
        agent, _ = self.agent(self.report.ReportMixin)
        for weights in [(-1,1,1), (float('nan'),.3,.2), (.8,.3,.2), (True,0,0)]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                agent.generate_anomaly_report('missing.mp4', self.root, *weights)
        self.ollama.chat.assert_not_called()

    def test_huddle_partial_assumptions_are_labeled_without_inventing_physics(self):
        agent, run = self.agent(self.huddle.HuddleMixin)
        text = agent.generate_huddle_paper('missing.mp4', self.root, pe=0)
        prompt = self.ollama.chat.call_args_list[0].kwargs['messages'][-1]['content']
        self.assertIn('Pe (dimensionless transport ratio): 0', prompt)
        self.assertIn('not measured or fitted', prompt)
        self.assertIn('A_1', prompt)
        self.assertIn('maximum_sampled_speed', prompt)
        self.assertNotIn('running on Gemma', prompt)
        self.assertIn('observation viewer', text)
        self.assertTrue((run/'huddle_simulation.html').is_file())
        with self.assertRaises(ValueError):
            agent.generate_huddle_paper('missing.mp4', self.root, beta=float('inf'))

    def test_retrieval_and_tool_context_have_cumulative_report_budgets(self):
        agent, _ = self.agent(self.report.ReportMixin)
        document = types.SimpleNamespace(metadata={'source': 'paper.pdf'}, page_content='e'*100000)
        search = mock.Mock(return_value=[document]*4)
        agent.vector_store = types.SimpleNamespace(max_marginal_relevance_search=search)
        calls = [{'function': {'name': 'search_scientific_literature', 'arguments': {'query': 'behavior'}}} for _ in range(8)]
        rounds = []
        def chat(**kwargs):
            rounds.append(copy.deepcopy(kwargs['messages']))
            if len(rounds) == 1:
                return {'message': {'role': 'assistant', 'content': '', 'tool_calls': calls}}
            return {'message': {'content': 'Supported report'}}
        self.ollama.chat.side_effect = chat
        text = agent.generate_anomaly_report('missing.mp4', self.root)
        self.assertTrue(text.startswith('Supported report'))
        self.assertLessEqual(search.call_count, 4)
        self.assertLessEqual(sum(len(ref['text']) for ref in agent.current_rag_refs.values()), 12000)
        tool_contents = [message['content'] for message in rounds[-1] if message['role'] == 'tool']
        self.assertLessEqual(sum(map(len, tool_contents)), 12000)
        self.assertTrue(any('budget exhausted' in content for content in tool_contents))

    def test_model_html_is_literal_and_frame_cleanup_never_deletes_user_temp_frames(self):
        agent, run = self.agent(self.report.ReportMixin)
        user_frames = run/'temp_frames'
        user_frames.mkdir()
        (user_frames/'keep.txt').write_text('user data', encoding='utf-8')
        self.ollama.chat.return_value = {'message': {'content': '<script>alert(1)</script> report'}}
        text = agent.generate_anomaly_report('missing.mp4', self.root)
        self.assertNotIn('<script>', text)
        self.assertIn('&lt;script&gt;', text)
        self.assertTrue((user_frames/'keep.txt').is_file())
        self.assertEqual(list(run.glob('.report-frames-*')), [])
        self.ollama.chat.side_effect = RuntimeError('model unavailable')
        self.assertIn('Error generating report', agent.generate_anomaly_report('missing.mp4', self.root))
        self.assertEqual(list(run.glob('.report-frames-*')), [])
        self.assertTrue((user_frames/'keep.txt').is_file())


class ObservationViewerTests(unittest.TestCase):
    def test_viewer_is_offline_and_has_no_generated_physics(self):
        template = (ROOT/'huddle_simulation_template.html').read_text(encoding='utf-8')
        self.assertNotRegex(template, r'<(?:script|link)[^>]+(?:src|href)="https?://')
        self.assertNotIn('Math.random', template)
        self.assertNotIn('Math.pow(aspectRatio', template)
        self.assertNotIn('innerHTML', template)
        self.assertIn('No interpolation', template)

    @unittest.skipUnless(shutil.which('node'), 'Node is required to execute the observation viewer regression')
    def test_viewer_uses_later_observations_and_does_not_fabricate_empty_tracks(self):
        template = (ROOT/'huddle_simulation_template.html').read_text(encoding='utf-8')
        script = re.search(r'<script>([\s\S]*?)</script>', template).group(1)
        runner = r'''
const fs=require('fs'), vm=require('vm');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const elements={};
const drawing=new Proxy({}, {get:(target,key)=>target[key] || (()=>{}),set:(target,key,value)=>(target[key]=value,true)});
function element(){return {textContent:'',value:0,checked:true,children:[],parentElement:{clientWidth:800,clientHeight:600},getContext:()=>drawing,addEventListener:()=>{},replaceChildren(){this.children=[]},append(...children){this.children.push(...children)}}}
const document={getElementById:id=>elements[id] ||= element(),createElement:()=>element()};
const context=vm.createContext({document,window:{addEventListener:()=>{}},Math,Set,Number,String});
vm.runInContext(input.script.replace('{tracks_json}',JSON.stringify(input.evidence)),context);
const initialVisible=elements['visible-count'].textContent;
const initialNames=elements.observations.children.map(row=>row.children[0].textContent);
vm.runInContext('selectTime(1)',context);
process.stdout.write(JSON.stringify({initialVisible,initialNames,time:elements.timestamp.textContent,visible:elements['visible-count'].textContent,speed:elements.speed.textContent,rows:elements.observations.children.length,points:vm.runInContext('points.length',context)}));
'''
        arbitrary_id = '</script><svg onload=alert(1)>'
        for tracks in ({'A_1': points(), arbitrary_id: [points()[0]]}, {}):
            evidence = build_tracking_evidence(tracks)
            process = subprocess.run([shutil.which('node'), '-e', runner], input=json.dumps({'script':script,'evidence':evidence}),
                                     capture_output=True, text=True, check=True, timeout=10)
            state = json.loads(process.stdout)
            if tracks:
                self.assertEqual(state['initialVisible'], '2')
                self.assertIn(arbitrary_id, state['initialNames'])
                self.assertEqual(state['time'], '2.000 s')
                self.assertEqual(state['visible'], '1')
                self.assertEqual(state['speed'], '2.50 px/s')
                self.assertEqual(state['rows'], 1)
            else:
                self.assertEqual(state['visible'], '0')
                self.assertEqual(state['points'], 0)
                self.assertEqual(state['rows'], 0)


if __name__ == '__main__':
    unittest.main()
