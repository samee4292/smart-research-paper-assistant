from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from research_assistant.answer_ui import emphasize_citations
from test_app import uploaded_pdf

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class PortfolioUITests(unittest.TestCase):
    def test_empty_workspace_sidebar_and_hidden_tools(self):
        app = AppTest.from_file(APP)
        with patch("streamlit.file_uploader", return_value=[]), patch("research_assistant.answer_ui.get_api_key", return_value=None):
            app.run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, "Smart Research Paper Assistant")
            self.assertEqual(app.sidebar.header[0].value, "Research Papers")
            self.assertEqual([m.value for m in app.sidebar.metric], ["0", "0", "0"])
            self.assertTrue(app.button(key="rag_ask").disabled)
            self.assertTrue(any(i.value == "Upload at least one research paper to begin." for i in app.info))
            self.assertFalse(any("Test " in h.value or "Inspect Chunks" in h.value for h in app.subheader))
            self.assertEqual({e.label for e in app.expander}, {"Technical details", "How does intent detection help?"})

    def test_examples_never_submit_and_sources_are_unique_and_optional(self):
        app = AppTest.from_file(APP)
        chunk = dict(chunk_id="c1", filename="test.pdf", page=2, text="Short second page", similarity=.9)
        store = Mock(index=SimpleNamespace(ntotal=4))
        store.search.return_value = [chunk, dict(chunk, chunk_id="c2")]
        with patch("streamlit.file_uploader", return_value=[uploaded_pdf()]), patch("research_assistant.search_ui.create_vector_store", return_value=store), patch("research_assistant.answer_ui.get_api_key", return_value="fixture-key"), patch("research_assistant.answer_ui.predict_intent", return_value={"intent": "RESULTS", "confidence": .4}), patch("research_assistant.answer_ui.generate_answer", return_value="A finding [test.pdf, Page 2].") as generate:
            app.run(timeout=20)
            generate.assert_not_called()
            self.assertTrue(any("Summarize the main findings." in m.value for m in app.markdown))
            self.assertTrue(any(s.value == "Documents indexed and ready" for s in app.sidebar.success))
            app.text_input(key="rag_question").set_value("What was found?")
            app.button(key="rag_ask").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["rag_sources"], [{"filename": "test.pdf", "page": 2}])
            self.assertTrue(any(e.label == "Sources used" for e in app.expander))
            self.assertTrue(any("**[test.pdf, Page 2]**" in m.value for m in app.markdown))
            self.assertTrue(any("Standard semantic retrieval used due to low intent confidence." == c.value for c in app.caption))
            self.assertTrue(any("9 / 10" in c.value for c in app.sidebar.caption))
            self.assertFalse(any("Test " in h.value for h in app.subheader))

    def test_api_failure_is_generic_and_does_not_consume_quota(self):
        app = AppTest.from_file(APP)
        store = Mock(index=SimpleNamespace(ntotal=4))
        store.search.return_value = [dict(chunk_id="c", filename="test.pdf", page=2, text="text", similarity=.9)]
        with patch("streamlit.file_uploader", return_value=[uploaded_pdf()]), patch("research_assistant.search_ui.create_vector_store", return_value=store), patch("research_assistant.answer_ui.get_api_key", return_value="fixture-key"), patch("research_assistant.answer_ui.predict_intent", return_value={"intent": "COMPARISON", "confidence": .95}), patch("research_assistant.answer_ui.generate_answer", side_effect=RuntimeError("private SDK diagnostic fixture-key")):
            app.run(timeout=20)
            app.text_input(key="rag_question").set_value("Compare methods within this paper.")
            app.button(key="rag_ask").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["generated_answers"], 0)
            self.assertEqual([e.value for e in app.error], ["Unable to generate an answer right now. Please try again."])
            self.assertFalse(any("Only one paper" in i.value for i in app.info))

    def test_citation_emphasis_preserves_source_label_text(self):
        answer = "Evidence [study.v2.pdf, Page 4]. Unrelated [other.pdf, Page 7]."
        sources = [{"filename": "study.v2.pdf", "page": 4}]
        rendered = emphasize_citations(answer, sources)
        self.assertEqual(rendered, "Evidence **[study.v2.pdf, Page 4]**. Unrelated [other.pdf, Page 7].")


if __name__ == "__main__":
    unittest.main()
