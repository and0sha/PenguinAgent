"""GUI provenance transitions, upload preservation and error handling without Streamlit."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from tools.gui_tracking_state import (
    begin_tracking_attempt, current_tracking_result, report_for_result,
    reset_session_analysis_state, save_uploaded_video, store_report_for_result,
    sync_tracking_context,
)
from tools.tracking_results import TrackingError
from behave_tracking import TrackingMixin


def derived_session(run_id="first"):
    return {
        "report": "Report for the first video", "report_run_id": run_id,
        "analysis_run_id": run_id, "messages": [{"role": "user", "content": "Old question"}],
        "pending_tag": "@first_1", "last_copied_tag": "@first_1", "annotated_video": "old.mp4",
        "current_video": "video.mp4", "output_dir": "chosen-output", "model": "chosen-model",
    }


class FakeAgent:
    """Expose the public tracking contract; test real file identity transitions."""
    def __init__(self, videos_dir):
        self.videos_dir = Path(videos_dir)
        self.context = None
        self.result = None
        self.tracks = None
        self.current_rag_refs = {1: "old reference"}
        self.invalidations = 0

    def invalidate_tracking_state(self):
        self.invalidations += 1
        self.result = None
        self.tracks = None
        self.current_rag_refs = {}

    def set_video_context(self, video_name, output_dir=None):
        source = self.videos_dir / video_name
        context = (source.resolve(), hashlib.sha256(source.read_bytes()).digest(), Path(output_dir).resolve())
        changed = context != self.context
        if changed:
            self.invalidate_tracking_state()
            self.context = context
        return changed

    def require_tracking_result(self, video_name, output_dir=None):
        if self.result is None:
            raise TrackingError("Latest attempt has no validated result")
        return self.result


class GuiTrackingStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.video = self.directory / "video.mp4"
        self.video.write_bytes(b"first video contents")
        self.output = self.directory / "analysis"
        self.agent = FakeAgent(self.directory)
        self.agent.set_video_context(self.video.name, self.output)
        self.result = SimpleNamespace(run_id="first", output_dir=self.output / "runs" / "first", tracks={"first_1": []})
        self.agent.result = self.result
        self.agent.tracks = self.result.tracks

    def test_reset_removes_all_analysis_values_and_preserves_user_settings(self):
        session = derived_session()
        reset_session_analysis_state(session)
        self.assertEqual(session, {"current_video": "video.mp4", "output_dir": "chosen-output", "model": "chosen-model"})

    def test_same_context_keeps_report_chat_and_current_run(self):
        session = derived_session()
        before = dict(session)
        self.assertFalse(sync_tracking_context(session, self.agent, self.video.name, self.output))
        self.assertIs(current_tracking_result(session, self.agent, self.video.name, self.output), self.result)
        self.assertEqual(session, before)

    def test_replacing_contents_under_same_filename_invalidates_analysis(self):
        session = derived_session()
        self.video.write_bytes(b"other video contents")
        self.assertTrue(sync_tracking_context(session, self.agent, self.video.name, self.output))
        self.assertIsNone(self.agent.tracks)
        self.assertNotIn("report", session)
        self.assertNotIn("messages", session)
        self.assertNotIn("pending_tag", session)

    def test_switching_video_or_output_directory_clears_results(self):
        other = self.directory / "another.mp4"
        other.write_bytes(self.video.read_bytes())
        for name, output in ((other.name, self.output), (self.video.name, self.directory / "other-output")):
            with self.subTest(name=name, output=output):
                session = derived_session()
                self.agent.result = self.result
                self.assertTrue(sync_tracking_context(session, self.agent, name, output))
                self.assertNotIn("annotated_video", session)
                self.assertNotIn("report_run_id", session)
                self.assertIsNone(self.agent.result)

    def test_failed_rerun_keeps_legacy_files_off_the_active_ui(self):
        self.output.mkdir()
        (self.output / "annotated.mp4").write_bytes(b"legacy video")
        (self.output / "tracks.json").write_text('{"legacy_1": []}', encoding="utf-8")
        session = derived_session()
        begin_tracking_attempt(session, self.agent)
        self.assertIsNone(current_tracking_result(session, self.agent, self.video.name, self.output))
        self.assertIsNone(report_for_result(session, None))
        self.assertNotIn("annotated_video", session)
        self.assertNotIn("messages", session)
        self.assertEqual(self.agent.current_rag_refs, {})
        self.assertTrue((self.output / "annotated.mp4").exists())

    def test_removing_uploaded_video_disables_current_analysis(self):
        session = derived_session()
        self.assertIsNone(current_tracking_result(session, self.agent, None, self.output))
        self.assertIsNone(self.agent.tracks)
        self.assertNotIn("report", session)

    def test_new_run_id_discards_report_and_chat_from_previous_success(self):
        session = derived_session()
        self.agent.result = SimpleNamespace(run_id="second", output_dir=self.output / "runs" / "second")
        current_tracking_result(session, self.agent, self.video.name, self.output)
        self.assertEqual(session["analysis_run_id"], "second")
        self.assertNotIn("report", session)
        self.assertNotIn("messages", session)

    def test_literature_only_chat_history_survives_repeated_validation_without_analysis(self):
        messages = [{"role": "user", "content": "What does the literature say about huddling?"}]
        session = {"messages": messages}
        self.agent.invalidate_tracking_state()
        for _ in range(2):
            self.assertIsNone(current_tracking_result(session, self.agent, self.video.name, self.output))
            self.assertIsNone(current_tracking_result(session, self.agent, None, self.output))
        self.assertIs(session["messages"], messages)
        self.assertIsNone(self.agent.tracks)

    def test_report_is_visible_only_for_its_validated_run(self):
        session = {}
        store_report_for_result(session, "A successful report", self.result)
        self.assertEqual(report_for_result(session, self.result), "A successful report")
        self.assertIsNone(report_for_result(session, SimpleNamespace(run_id="second")))
        self.assertIsNone(report_for_result(session, None))

    def test_error_text_and_empty_report_cannot_be_stored_as_success(self):
        for paper in (None, "", "  ", "Error generating report via Ollama: down", "Error: missing", "No tracks available. Please run tracking first."):
            with self.subTest(paper=paper):
                session = derived_session()
                with self.assertRaises(ValueError):
                    store_report_for_result(session, paper, self.result)
                self.assertNotIn("report", session)
                self.assertNotIn("report_run_id", session)

    def test_report_citations_are_a_snapshot_of_its_own_sources(self):
        references = {1: {"source": "first.pdf", "text": "original evidence"}}
        session = {}
        store_report_for_result(session, "Finding [1]", self.result, references)
        references[1]["text"] = "a different paper from a later operation"
        self.assertEqual(session["report_rag_refs"][1]["text"], "original evidence")
        reset_session_analysis_state(session)
        self.assertNotIn("report_rag_refs", session)


class UploadedVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "videos"

    def test_unchanged_upload_does_not_rewrite_file_on_rerun(self):
        first = save_uploaded_video("video.mp4", memoryview(b"same contents"), self.directory)
        os.utime(first.path, ns=(1_500_000_000_000_000_000, 1_500_000_000_000_000_000))
        original = first.path.stat()
        with mock.patch("tools.gui_tracking_state.os.replace") as replace:
            second = save_uploaded_video("video.mp4", memoryview(b"same contents"), self.directory)
        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(second.path.stat().st_mtime_ns, original.st_mtime_ns)
        self.assertEqual(second.path.stat().st_ino, original.st_ino)
        replace.assert_not_called()

    def test_same_filename_and_size_with_different_contents_are_replaced(self):
        save_uploaded_video("video.mp4", b"first", self.directory)
        result = save_uploaded_video("video.mp4", b"other", self.directory)
        self.assertTrue(result.changed)
        self.assertEqual(result.path.read_bytes(), b"other")
        # Windows CI may expose TEMP through an 8.3 alias such as RUNNER~1.
        self.assertEqual([path.resolve() for path in self.directory.iterdir()], [result.path])

    def test_filename_traversal_is_rejected(self):
        for name in ("../video.mp4", "..\\video.mp4", "C:\\video.mp4", "/tmp/video.mp4", ".", "", "video.mp4\0"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                save_uploaded_video(name, b"contents", self.directory)
        self.assertFalse(self.directory.exists())

    def test_publish_failure_preserves_previous_source_and_removes_temporary_file(self):
        original = save_uploaded_video("video.mp4", b"first", self.directory)
        with mock.patch("tools.gui_tracking_state.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                save_uploaded_video("video.mp4", b"other", self.directory)
        self.assertEqual(original.path.read_bytes(), b"first")
        self.assertEqual([path.resolve() for path in self.directory.iterdir()], [original.path])


class Session(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeStreamlit(ModuleType):
    def __init__(self, session, upload, output_dir, buttons=(), chat_prompt=None, report_type=None, assumptions=False):
        super().__init__("streamlit")
        self.session_state = session
        self.__path__ = []
        self.sidebar = Context()
        self.upload = upload
        self.output_dir = str(output_dir)
        self.buttons = set(buttons)
        self.chat_prompt = chat_prompt
        self.report_type = report_type
        self.assumptions = assumptions
        self.events = []

    def __getattr__(self, method):
        def record(*args, **kwargs):
            self.events.append((method, args, kwargs))
        return record

    def cache_data(self, **kwargs):
        return lambda callback: callback

    def columns(self, specification):
        return [Context() for _ in range(specification if isinstance(specification, int) else len(specification))]

    def spinner(self, *args, **kwargs):
        return Context()

    chat_message = spinner
    popover = spinner

    def selectbox(self, label, options, **kwargs):
        return options[0]

    def text_input(self, label, value="", **kwargs):
        return self.output_dir if label == "Output Directory" else value

    def file_uploader(self, *args, **kwargs):
        return self.upload

    def button(self, label, **kwargs):
        return label in self.buttons and not kwargs.get("disabled", False)

    def checkbox(self, *args, **kwargs):
        if args and args[0] == "Include optional thermodynamic assumptions":
            return self.assumptions
        return kwargs.get("value", False)

    def radio(self, label, options, **kwargs):
        return self.report_type or options[0]

    def number_input(self, *args, **kwargs):
        return kwargs.get("value")

    def chat_input(self, *args, **kwargs):
        self.events.append(("chat_input", args, kwargs))
        return self.chat_prompt


class GuiExecutionTests(unittest.TestCase):
    """Execute the real main_gui module with UI/model/network adapters replaced."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.video_bytes = b"fake uploaded video"
        (self.directory / "video.mp4").write_bytes(self.video_bytes)
        self.output = self.directory / "analysis"
        self.agent = FakeAgent(self.directory)
        self.agent.rag_dir = str(self.directory)
        self.agent.model = "stub"
        self.agent.set_video_context("video.mp4", self.output)
        self.result = SimpleNamespace(
            run_id="first", output_dir=self.output / "runs" / "first",
            annotated_video_path=self.output / "runs" / "first" / "annotated.mp4",
            json_path=self.output / "runs" / "first" / "tracks.json", tracks={"current_1": [{"second": 0}]},
        )
        self.agent.result = self.result
        self.agent.tracks = self.result.tracks
        self.agent.generate_anomaly_report = mock.Mock(return_value="A report for this run")
        self.agent.generate_huddle_paper = mock.Mock(return_value="Huddle report for this run")
        self.agent.run_tracking = mock.Mock()

    def execute_gui(self, buttons=(), *, session=None, upload_bytes=None, chat_prompt=None, with_video=True,
                    chat_error=None, chat_response="Current analysis answer", report_type=None, assumptions=False):
        upload_bytes = self.video_bytes if upload_bytes is None else upload_bytes
        upload = SimpleNamespace(name="video.mp4", getbuffer=lambda: memoryview(upload_bytes))
        if not with_video:
            upload = None
        session = Session(derived_session() if session is None else session)
        session.agent = self.agent
        fake_streamlit = FakeStreamlit(session, upload, self.output, buttons, chat_prompt, report_type, assumptions)
        fake_behavior = ModuleType("behave_ai")
        fake_behavior.BehaviorAgent = lambda: self.agent
        fake_interactive = ModuleType("interactive_mode")
        fake_interactive.handle_interactive_query = mock.Mock(return_value=chat_response)
        fake_interactive.handle_interactive_query.side_effect = chat_error
        fake_transformers = ModuleType("transformers")
        fake_transformers.utils = SimpleNamespace(logging=SimpleNamespace(set_verbosity_error=lambda: None))
        fake_components = ModuleType("streamlit.components")
        fake_components.__path__ = []
        fake_components_v1 = ModuleType("streamlit.components.v1")
        fake_components_v1.html = mock.Mock()
        fake_components.v1 = fake_components_v1
        fake_streamlit.components = fake_components
        source = Path(__file__).resolve().parents[1] / "main_gui.py"
        with mock.patch.dict(sys.modules, {"streamlit": fake_streamlit, "behave_ai": fake_behavior,
                                          "interactive_mode": fake_interactive, "transformers": fake_transformers,
                                          "streamlit.components": fake_components,
                                          "streamlit.components.v1": fake_components_v1}), \
                mock.patch("urllib.request.urlopen", side_effect=OSError("offline test")):
            specification = importlib.util.spec_from_file_location("_gui_execution_test", source)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
        return session, fake_streamlit, fake_interactive.handle_interactive_query

    def test_failed_tracking_attempt_does_not_display_previous_success(self):
        self.agent.run_tracking.side_effect = TrackingError("SAM3 failed")
        session, ui, _ = self.execute_gui(["Start processing and annotating video"])
        videos = [args[0] for method, args, _ in ui.events if method == "video"]
        self.assertEqual(videos, [str(self.directory / "video.mp4")])
        self.assertFalse(any(method == "success" for method, _, _ in ui.events))

        self.assertTrue(any(method == "error" and "SAM3 failed" in args[0] for method, args, _ in ui.events))
        self.assertNotIn("report", session)
        self.assertEqual(session["messages"], [])

    def test_legacy_files_do_not_enable_report_but_allow_literature_only_chat(self):
        self.output.mkdir()
        (self.output / "annotated.mp4").write_bytes(b"legacy")
        (self.output / "tracks.json").write_text("{}", encoding="utf-8")
        self.agent.invalidate_tracking_state()
        session, ui, chat = self.execute_gui(["Generate Scientific Report"], chat_prompt="Use the old report")
        self.agent.generate_anomaly_report.assert_not_called()
        chat.assert_called_once()
        self.assertIsNone(chat.call_args.kwargs["output_dir"])
        self.assertIsNone(self.agent.tracks)
        self.assertFalse(next(kwargs for method, _, kwargs in ui.events if method == "chat_input").get("disabled", False))
        self.assertNotIn("report", session)

    def test_report_error_string_is_shown_as_failure_and_never_as_report(self):
        self.agent.generate_anomaly_report.return_value = "Error generating report via Ollama: offline"
        session, ui, _ = self.execute_gui(["Generate Scientific Report"])
        self.assertNotIn("report", session)
        self.assertFalse(any(method == "success" for method, _, _ in ui.events))
        self.assertTrue(any(method == "error" and "offline" in args[0] for method, args, _ in ui.events))

    def test_success_uses_verified_run_paths_and_report_base_directory(self):
        session, ui, chat = self.execute_gui(["Generate Scientific Report"], chat_prompt="Explain current tracks")
        self.agent.generate_anomaly_report.assert_called_once()
        self.assertEqual(self.agent.generate_anomaly_report.call_args.kwargs["output_dir"], str(self.output))
        self.assertEqual(session["report_run_id"], self.result.run_id)
        self.assertIn(str(self.result.annotated_video_path), [args[0] for method, args, _ in ui.events if method == "video"])
        self.assertEqual(chat.call_args.kwargs["output_dir"], self.result.output_dir)
        self.assertFalse(next(kwargs for method, _, kwargs in ui.events if method == "chat_input").get("disabled", False))

    def test_success_message_requires_new_validated_run(self):
        def finish(*args, **kwargs):
            self.agent.result = SimpleNamespace(**{**vars(self.result), "run_id": "second"})
            self.agent.tracks = self.result.tracks
            return self.result.tracks, str(self.result.annotated_video_path)
        self.agent.run_tracking.side_effect = finish
        session, ui, _ = self.execute_gui(["Start processing and annotating video"])
        self.assertTrue(any(method == "success" and "Tracks and annotated" in args[0] for method, args, _ in ui.events))
        self.assertEqual(session["analysis_run_id"], "second")
        self.assertNotIn("report", session)

    def test_same_filename_new_upload_invalidates_video_context_but_keeps_literature_chat_available(self):
        session, ui, chat = self.execute_gui(chat_prompt="Explain old tracks", upload_bytes=b"different uploaded video")
        self.assertIsNone(self.agent.result)
        self.assertNotIn("report", session)
        chat.assert_called_once()
        self.assertIsNone(chat.call_args.kwargs["output_dir"])
        self.assertIsNone(self.agent.tracks)
        self.assertNotIn({"role": "user", "content": "Old question"}, session["messages"])
        self.assertFalse(any(method == "success" for method, _, _ in ui.events))

    def test_chat_without_uploaded_video_keeps_history_across_reruns(self):
        first_session, ui, chat = self.execute_gui(session={}, with_video=False, chat_prompt="Explain huddling literature")
        chat.assert_called_once()
        self.assertIsNone(chat.call_args.kwargs["output_dir"])
        self.assertIsNone(self.agent.tracks)
        self.assertEqual(len(first_session["messages"]), 2)
        rerun_session, _, _ = self.execute_gui(session=first_session, with_video=False)
        self.assertEqual(rerun_session["messages"], first_session["messages"])
        self.assertTrue(any(method == "caption" and "scientific literature only" in args[0] for method, args, _ in ui.events))

    def test_rag_failure_is_not_reported_as_success(self):
        for failure in (RuntimeError("bad PDF"), None):
            with self.subTest(failure=failure):
                self.agent.initialize_rag = mock.Mock(return_value=None,
                                                       side_effect=failure if failure is not None else None)
                _, ui, _ = self.execute_gui(["Initialize Local RAG"])
                self.assertFalse(any(method == "success" for method, _, _ in ui.events))
                self.assertTrue(any(method == "error" and "scientific RAG" in args[0] for method, args, _ in ui.events))

    def test_rag_success_requires_explicit_successful_backend_result(self):
        self.agent.initialize_rag = mock.Mock(return_value=True)
        _, ui, _ = self.execute_gui(["Initialize Local RAG"])
        self.assertTrue(any(method == "success" and "RAG Initialized" in args[0] for method, args, _ in ui.events))

    def test_chat_exception_remains_visible_after_streamlit_rerun(self):
        session, _, _ = self.execute_gui(session={}, with_video=False, chat_prompt="Explain literature",
                                         chat_error=RuntimeError("retriever unavailable"))
        self.assertTrue(session["messages"][-1]["error"])
        self.assertIn("retriever unavailable", session["messages"][-1]["content"])
        _, ui, _ = self.execute_gui(session=session, with_video=False)
        self.assertTrue(any(method == "error" and "retriever unavailable" in args[0] for method, args, _ in ui.events))

    def test_valid_answer_about_error_rates_is_not_reported_as_a_failure(self):
        answer = "Error rates depend on occlusion and camera position."
        session, ui, _ = self.execute_gui(session={}, with_video=False, chat_prompt="What affects error rates?",
                                         chat_response=answer)
        self.assertEqual(session["messages"][-1]["content"], answer)
        self.assertFalse(session["messages"][-1].get("error", False))
        self.assertFalse(any(method == "error" for method, _, _ in ui.events))

    def test_huddle_thermodynamic_assumptions_are_absent_by_default(self):
        self.execute_gui(["Generate Scientific Report"], report_type="Huddling Analysis")
        self.agent.generate_huddle_paper.assert_called_once_with("video.mp4", output_dir=str(self.output))

    def test_huddle_assumptions_are_only_sent_after_opt_in(self):
        self.execute_gui(["Generate Scientific Report"], report_type="Huddling Analysis", assumptions=True)
        self.agent.generate_huddle_paper.assert_called_once_with("video.mp4", output_dir=str(self.output),
                                                                pe=100.0, beta=0.25, tp=-10.0, t_inf=-40.0)


class RealTrackingGuiTests(unittest.TestCase):
    """Exercise the real manifest validator and GUI together with only extraction stubbed."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.video_bytes = b"current source video"
        (self.directory / "video.mp4").write_bytes(self.video_bytes)
        self.output = self.directory / "analysis"
        self.agent = TrackingMixin()
        self.agent.videos_dir = str(self.directory)
        self.agent.rag_dir = str(self.directory)
        self.agent.model = "stub"
        self.agent.generate_anomaly_report = mock.Mock(return_value="Current successful report")
        with mock.patch("behave_tracking.extract_tracks", side_effect=self.extract_fixture):
            self.agent.run_tracking("video.mp4", self.output)
        self.previous = self.agent.require_tracking_result("video.mp4", self.output)

    @staticmethod
    def extract_fixture(video, json_path, annotated_path, **kwargs):
        Path(json_path).write_text(json.dumps({"current_1": [{"second": 0, "x": 1, "y": 2}]}), encoding="utf-8")
        Path(annotated_path).write_bytes(b"fresh annotated artifact")
        return json_path, annotated_path

    def execute(self, *args, **kwargs):
        return GuiExecutionTests.execute_gui(self, *args, **kwargs)

    def test_real_failed_run_preserves_history_but_clears_active_display(self):
        with mock.patch("behave_tracking.extract_tracks", side_effect=RuntimeError("worker failed")):
            session, ui, _ = self.execute(["Start processing and annotating video"])
        self.assertTrue(self.previous.annotated_video_path.exists())
        self.assertEqual(json.loads((self.output / "tracking_manifest.json").read_text())["status"], "failed")
        self.assertIsNone(self.agent.tracking_result)
        self.assertNotIn("report", session)
        self.assertEqual([args[0] for method, args, _ in ui.events if method == "video"], [str(self.directory / "video.mp4")])
        self.assertFalse(any(method == "success" for method, _, _ in ui.events))

    def test_real_success_displays_only_new_published_run(self):
        with mock.patch("behave_tracking.extract_tracks", side_effect=self.extract_fixture):
            session, ui, _ = self.execute(["Start processing and annotating video"])
        current = self.agent.require_tracking_result("video.mp4", self.output)
        self.assertNotEqual(current.run_id, self.previous.run_id)
        self.assertEqual(session["analysis_run_id"], current.run_id)
        self.assertEqual([args[0] for method, args, _ in ui.events if method == "video"], [str(current.annotated_video_path)])
        self.assertTrue(any(method == "success" for method, _, _ in ui.events))

    def test_corrupted_artifact_disables_report_and_removes_video_context_from_chat(self):
        self.previous.json_path.write_text('{"tampered_1": []}', encoding="utf-8")
        session, ui, chat = self.execute(["Generate Scientific Report"], chat_prompt="Use tampered data")
        self.agent.generate_anomaly_report.assert_not_called()
        chat.assert_called_once()
        self.assertIsNone(chat.call_args.kwargs["output_dir"])
        self.assertIsNone(self.agent.tracks)
        self.assertNotIn("report", session)
        self.assertFalse(any(method == "success" for method, _, _ in ui.events))


if __name__ == "__main__":
    unittest.main()
