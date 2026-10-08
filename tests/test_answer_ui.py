from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from llm.generator import GenerationError
from test_app import uploaded_pdf


class AnswerUITests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        self.context = dict(chunk_id="id", filename="test.pdf", page=2, text="Short second page", similarity=0.9)
        self.store = Mock(index=SimpleNamespace(ntotal=4))
        self.store.search.return_value = [self.context, dict(self.context)]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("streamlit.file_uploader", return_value=[uploaded_pdf()]))
        self.stack.enter_context(patch("research_assistant.search_ui.create_vector_store", return_value=self.store))
        self.classify = self.stack.enter_context(patch("research_assistant.answer_ui.predict_intent",
            return_value={"intent": "RESULTS", "confidence": 0.4}))
        self.key = self.stack.enter_context(patch("research_assistant.answer_ui.get_api_key", return_value="test-credential"))
        self.generate = self.stack.enter_context(patch("research_assistant.answer_ui.generate_answer", return_value="Short second page [test.pdf, Page 2]."))

    def ask(self, question="What does page two say?"):
        self.app.text_input(key="rag_question").set_value(question)
        self.app.button(key="rag_ask").click().run(timeout=20)
        self.assertFalse(self.app.exception)

    def test_answer_sources_and_persistent_budget(self):
        self.app.run(timeout=20)
        self.ask()
        self.store.search.assert_called_once_with("What does page two say?", top_k=5)
        self.assertEqual(self.app.session_state["generated_answers"], 1)
        self.assertEqual(self.app.session_state["rag_sources"], [dict(filename="test.pdf", page=2)])
        self.assertTrue(any("[test.pdf, Page 2]" in m.value for m in self.app.markdown))
        self.app.run(timeout=20)
        self.generate.assert_called_once()
        with patch("streamlit.file_uploader", return_value=[]):
            self.app.run(timeout=20)
            self.assertIsNone(self.app.session_state["rag_answer"])
            self.assertEqual(self.app.session_state["generated_answers"], 1)

    def test_empty_retrieval_failure_and_empty_question(self):
        self.app.run(timeout=20)
        self.ask("  ")
        self.generate.assert_not_called()
        self.store.search.return_value = []
        self.ask()
        self.generate.assert_not_called()
        self.store.search.return_value = [self.context]
        self.generate.side_effect = GenerationError("Gemini request failed safely.")
        self.ask()
        self.assertEqual(self.app.session_state["generated_answers"], 0)

    def test_tenth_answer_disables_further_requests(self):
        self.app.session_state["generated_answers"] = 9
        self.app.run(timeout=20)
        self.ask()
        self.assertEqual(self.app.session_state["generated_answers"], 10)
        self.assertTrue(self.app.button(key="rag_ask").disabled)
        self.assertTrue(any("0 / 10" in c.value for c in self.app.caption))
        self.generate.assert_called_once()

    def test_intent_expansion_debug_and_five_chunk_limit(self):
        self.classify.return_value = {"intent": "LIMITATIONS", "confidence": 0.91}
        self.store.search.side_effect = [
            [dict(self.context, chunk_id=f"original{i}") for i in range(10)],
            [dict(self.context, chunk_id=f"expanded{i}") for i in range(10)],
        ]
        self.app.run(timeout=20)
        self.ask("Where does the approach fail?")
        self.assertEqual(self.store.search.call_args_list[0].args[0], "Where does the approach fail?")
        self.assertIn("weaknesses", self.store.search.call_args_list[1].args[0])
        context = self.generate.call_args.args[1]
        self.assertEqual(len(context), 5)
        self.assertEqual(context[0]["chunk_id"], "original0")
        self.assertTrue(any("Detected Intent: LIMITATIONS" in c.value and "Confidence: 91%" in c.value for c in self.app.caption))
        self.assertTrue(any(t.value == "Confidence: 91.0%" for t in self.app.text))
        self.assertTrue(any(e.label == "Technical details" for e in self.app.expander))
        self.assertTrue(any("Origin" in d.value.columns for d in self.app.dataframe))

    def test_classifier_failure_keeps_basic_rag_available(self):
        self.classify.side_effect = FileNotFoundError("missing model")
        self.app.run(timeout=20)
        self.ask()
        self.store.search.assert_called_once_with("What does page two say?", top_k=5)
        self.generate.assert_called_once()

    def test_missing_key_no_papers_and_missing_index(self):
        self.key.return_value = None
        self.app.run(timeout=20)
        self.assertTrue(self.app.button(key="rag_ask").disabled)
        self.assertTrue(any("Gemini API key" in i.value for i in self.app.sidebar.info))
        self.key.return_value = "test-credential"
        with patch("streamlit.file_uploader", return_value=[]):
            self.app.run(timeout=20)
            self.assertTrue(self.app.button(key="rag_ask").disabled)
            self.generate.assert_not_called()
        with patch("research_assistant.search_ui.create_vector_store", return_value=None):
            self.app.run(timeout=20)
            self.assertTrue(self.app.button(key="rag_ask").disabled)
            self.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
