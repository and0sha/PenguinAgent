"""Scientific questions without a run must not inherit reports from other videos."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


class RagVideoContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.messages = []
        def chat(**kwargs):
            self.messages.append(kwargs["messages"])
            return {"message": {"content": "answer"}}
        ollama = types.ModuleType("ollama")
        ollama.chat = chat
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        path = Path(__file__).resolve().parents[1] / "behave_rag.py"
        with mock.patch.dict(sys.modules, {"ollama": ollama, "torch": torch}):
            spec = importlib.util.spec_from_file_location("rag_context_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.module_patch = mock.patch.dict(sys.modules, {"torch": torch})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.agent = module.RAGMixin()
        self.ollama = ollama
        self.agent.videos_dir = str(self.root)
        self.agent.vector_store = None
        self.agent.model = "stub"
        self.old = self.root / "video_A_output"
        self.old.mkdir()
        (self.old / "final_scientific_report.md").write_text("OLD_VIDEO_A_REPORT", encoding="utf-8")

    def context(self):
        return self.messages[-1][0]["content"]

    def test_literature_only_chat_never_collects_all_video_reports(self):
        self.assertEqual(self.agent.ask("What is huddling?"), "answer")
        self.assertNotIn("OLD_VIDEO_A_REPORT", self.context())
        self.assertNotIn("Context from Local Video Reports", self.context())

    def test_explicit_run_reads_both_report_types_and_no_other_video(self):
        current = self.root / "video_B_output" / "runs" / "current"
        current.mkdir(parents=True)
        (current / "final_scientific_report.md").write_text("CURRENT_ANOMALY", encoding="utf-8")
        (current / "final_huddle_report.md").write_text("CURRENT_HUDDLE", encoding="utf-8")
        self.agent.ask("Describe this run", output_dir=current)
        self.assertIn("CURRENT_ANOMALY", self.context())
        self.assertIn("CURRENT_HUDDLE", self.context())
        self.assertNotIn("OLD_VIDEO_A_REPORT", self.context())

    def test_missing_run_does_not_fall_back_to_all_videos(self):
        self.agent.ask("Describe this run", output_dir=self.root / "missing")
        self.assertNotIn("OLD_VIDEO_A_REPORT", self.context())

    def test_scientific_retrieval_is_available_without_any_video_result(self):
        document = types.SimpleNamespace(page_content="SCIENTIFIC_LITERATURE")
        self.agent.vector_store = types.SimpleNamespace(
            max_marginal_relevance_search=lambda *args, **kwargs: [document],
        )
        self.agent.ask("What is huddling?")
        self.assertIn("SCIENTIFIC_LITERATURE", self.context())
        self.assertNotIn("OLD_VIDEO_A_REPORT", self.context())

    def test_both_large_reports_have_bounded_context_excerpts(self):
        current = self.root / "current-run"
        current.mkdir()
        (current / "final_scientific_report.md").write_text("ANOMALY " + "a" * 20000, encoding="utf-8")
        (current / "final_huddle_report.md").write_text("HUDDLE " + "h" * 20000, encoding="utf-8")
        self.agent.ask("Compare these reports", output_dir=current)
        self.assertIn("ANOMALY", self.context())
        self.assertIn("HUDDLE", self.context())
        self.assertLess(len(self.context()), 11000)
        self.assertEqual(self.context().count("[Report excerpt truncated]"), 2)

    def test_unreadable_authoritative_report_does_not_silently_answer_without_it(self):
        with mock.patch("builtins.open", side_effect=PermissionError("permission denied")):
            with self.assertRaisesRegex(RuntimeError, "Cannot read current video report"):
                self.agent.ask("Describe this run", output_dir=self.old)
        self.assertEqual(self.messages, [])

    def test_retrieval_failure_does_not_silently_answer_from_ungrounded_memory(self):
        self.agent.vector_store = types.SimpleNamespace(max_marginal_relevance_search=mock.Mock(side_effect=ValueError("bad vector")))
        with self.assertRaisesRegex(RuntimeError, "Cannot search the scientific index"):
            self.agent.ask("What is huddling?")
        self.assertEqual(self.messages, [])

    def test_ollama_transport_error_is_an_exception_not_answer_text(self):
        self.ollama.chat = mock.Mock(side_effect=ConnectionError("offline"))
        with self.assertRaisesRegex(RuntimeError, "Cannot communicate with Ollama"):
            self.agent.ask("What is huddling?")

    def test_empty_model_answer_is_a_failure(self):
        self.ollama.chat = mock.Mock(return_value={"message": {"content": "  "}})
        with self.assertRaisesRegex(RuntimeError, "empty answer"):
            self.agent.ask("What is huddling?")


if __name__ == "__main__":
    unittest.main()
