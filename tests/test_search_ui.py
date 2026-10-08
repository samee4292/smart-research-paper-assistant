from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from test_app import uploaded_pdf


class SearchUITests(unittest.TestCase):
    def test_search_reuse_and_upload_changes(self):
        result = dict(chunk_id="test.pdf_p2_c1", filename="test.pdf", page=2,
                      text="Short second page", similarity=0.81)
        store = Mock(index=SimpleNamespace(ntotal=3))
        store.search.return_value = [result]
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        with patch("streamlit.file_uploader", return_value=[uploaded_pdf()]), patch(
            "research_assistant.search_ui.create_vector_store", return_value=store
        ) as build:
            app.run(timeout=20)
            self.assertFalse(app.exception)
            app.checkbox(key="show_development_tools").check().run(timeout=20)
            app.button(key="semantic_search").click().run(timeout=20)
            self.assertTrue(any("Enter a question" in w.value for w in app.warning))
            store.search.assert_not_called()
            app.text_input(key="semantic_query").set_value("What are the methods?")
            app.button(key="semantic_search").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any("Similarity: 0.810" in t.value for t in app.text))
            build.assert_called_once()
            app.run(timeout=20)
            build.assert_called_once()
            replacement = uploaded_pdf()
            replacement.name = "replacement.pdf"
            with patch("streamlit.file_uploader", return_value=[replacement]):
                app.run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual(build.call_count, 2)
                self.assertEqual(app.session_state["search_results"], [])
        with patch("streamlit.file_uploader", return_value=[]):
            app.run(timeout=20)
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["vector_store"])
            self.assertEqual(app.session_state["search_results"], [])

    def test_index_failure_keeps_extraction_working(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        with patch("streamlit.file_uploader", return_value=[uploaded_pdf()]), patch(
            "research_assistant.search_ui.create_vector_store", side_effect=RuntimeError("offline")
        ):
            app.run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.sidebar.metric[2].value, "0")
            self.assertEqual(len(app.session_state["chunks"]), 4)
            self.assertTrue(any("couldn't prepare" in e.value for e in app.sidebar.error))
            self.assertTrue(any(b.label == "Retry indexing" for b in app.button))


if __name__ == "__main__":
    unittest.main()
