"""Interaction routing must preserve questions and avoid stale video analysis."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from interactive_mode import handle_interactive_query, parse_tags


class InteractiveModeTests(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleNamespace(tracks={"Adelie_1": [{"second": 0}]},
                                     ask=mock.Mock(return_value="literature"),
                                     analyze_behavior=mock.Mock(return_value="video evidence"))

    def test_email_address_does_not_route_as_a_penguin(self):
        query = "Does researcher@example.org describe @Adelie_1?"
        clean, tags = parse_tags(query)
        self.assertEqual(tags, ["Adelie_1"])
        self.assertIn("researcher@example.org", clean)

    def test_repeated_tag_runs_inference_only_once(self):
        handle_interactive_query(self.agent, "@Adelie_1 compare @Adelie_1", output_dir=Path("current-run"))
        self.agent.analyze_behavior.assert_called_once()
        self.assertEqual(self.agent.analyze_behavior.call_args.kwargs["query"], "compare")

    def test_anomaly_receives_user_question_and_run_directory(self):
        handle_interactive_query(self.agent, "@anomaly Explain acceleration", output_dir=Path("current-run"))
        self.agent.analyze_behavior.assert_called_once_with(query="Explain acceleration", output_dir=Path("current-run"))

    def test_no_context_cannot_use_cached_tracks(self):
        text = handle_interactive_query(self.agent, "@anomaly Explain", output_dir=None)
        self.agent.analyze_behavior.assert_not_called()
        self.assertIn("No validated video analysis", text)
        handle_interactive_query(self.agent, "@Adelie_1 Explain", output_dir=None)
        self.agent.analyze_behavior.assert_not_called()
        self.agent.ask.assert_called_once_with("Adelie_1 Explain", output_dir=None)

    def test_literature_query_remains_available_without_tracking(self):
        self.agent.tracks = None
        self.assertIn("literature", handle_interactive_query(self.agent, "@rag Why huddle?"))
        self.agent.ask.assert_called_once_with("Why huddle?", output_dir=None)

    def test_rag_aliases_share_one_retrieval_and_inference(self):
        handle_interactive_query(self.agent, "@rag @RAG @knowledge Why huddle?")
        self.agent.ask.assert_called_once_with("Why huddle?", output_dir=None)

    def test_bare_rag_tag_requires_a_question(self):
        with self.assertRaisesRegex(ValueError, "scientific question"):
            handle_interactive_query(self.agent, "@rag")
        self.agent.ask.assert_not_called()

    def test_empty_question_is_rejected(self):
        for query in (None, "", "   "):
            with self.subTest(query=query), self.assertRaises(ValueError):
                handle_interactive_query(self.agent, query)


if __name__ == "__main__":
    unittest.main()
