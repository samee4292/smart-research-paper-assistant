"""Phase 1/2 UI regression checks without embedding-model downloads."""

import io
from pathlib import Path
import unittest
from unittest.mock import patch

import pymupdf
from streamlit.testing.v1 import AppTest


def uploaded_pdf():
    with pymupdf.open() as doc:
        first = doc.new_page()
        # Multiple insertions keep the generated fixture text inside the page.
        for line in range(60):
            first.insert_text((40, 30 + line * 12), " ".join(f"w{line * 10 + n}" for n in range(10)), fontsize=8)
        doc.new_page().insert_text((72, 72), "Short second page")
        upload = io.BytesIO(doc.tobytes())
    upload.name = "test.pdf"
    return upload


class AppTests(unittest.TestCase):
    def test_upload_inspection_and_removal(self):
        bad = io.BytesIO(b"")
        bad.name = "empty.pdf"
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        # Phase 1/2 regression checks do not need model downloads.
        with patch("streamlit.file_uploader", return_value=[uploaded_pdf(), bad]), patch(
            "research_assistant.search_ui.create_vector_store", return_value=None
        ):
            app.run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual([m.value for m in app.sidebar.metric], ["2", "2", "0"])
            app.checkbox(key="show_development_tools").check().run(timeout=20)
            self.assertTrue(any("empty" in e.value for e in app.error))
            self.assertEqual(len(app.session_state["chunks"]), 4)
            app.selectbox(key="inspect_chunk").select(1).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any(t.value == "Chunk ID: test.pdf_p1_c2" for t in app.text))
            app.selectbox(key="inspect_chunk").select(3).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any(t.value == "Page number: 2" for t in app.text))
            self.assertTrue(any(t.value == "Short second page" for t in app.text_area))
        with patch("streamlit.file_uploader", return_value=[bad]):
            app.run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual([m.value for m in app.sidebar.metric], ["1", "0", "0"])
            self.assertEqual(app.session_state["chunks"], [])
        with patch("streamlit.file_uploader", return_value=[]):
            app.run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual([m.value for m in app.sidebar.metric], ["0", "0", "0"])
            self.assertEqual(app.session_state["extractions"], {})


if __name__ == "__main__":
    unittest.main()
