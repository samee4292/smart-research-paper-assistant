"""Opt-in live local journey using artificial PDFs; two real Gemini requests."""

import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import pymupdf
from streamlit.testing.v1 import AppTest

from evaluation.compare_retrieval import FIXTURE
from llm.generator import INSUFFICIENT_CONTEXT, unique_sources, validate_answer


def fixture_uploads():
    uploads = []
    for name, texts in FIXTURE.items():
        with pymupdf.open() as document:
            for text in texts:
                assert document.new_page().insert_textbox((40, 40, 555, 780), text, fontsize=11) >= 0
            upload = io.BytesIO(document.tobytes())
            upload.name = name
            uploads.append(upload)
    return uploads


@unittest.skipUnless(os.environ.get("RUN_LIVE_GEMINI_TESTS") == "1", "Opt-in: requires a Gemini key, network, and API quota")
class LivePortfolioJourneyTests(unittest.TestCase):
    def test_real_upload_to_grounded_answer_and_confidence_fallback(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        # Only the browser upload widget is replaced. All AI/PDF/API work is real.
        with patch("streamlit.file_uploader", return_value=fixture_uploads()):
            app.run(timeout=120)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["chunks"]), 11)
            self.assertEqual(app.session_state["vector_store"].index.ntotal, 11)
            self.assertFalse(app.button(key="rag_ask").disabled, "Configure the Gemini key before this opt-in test.")
            for question, expected_mode in [
                ("How were students assigned and followed in the adaptive planning experiment?", "INTENT_EXPANSION"),
                ("What performance did the adaptive planner achieve?", "BASIC_FALLBACK"),
            ]:
                app.text_input(key="rag_question").set_value(question)
                app.button(key="rag_ask").click().run(timeout=120)
                self.assertFalse(app.exception)
                self.assertFalse(app.error, "Live answer generation failed; inspect the safe UI error.")
                answer = app.session_state["rag_answer"]
                context = app.session_state["retrieval_debug"]
                self.assertTrue(answer, "The live pipeline did not produce an answer.")
                self.assertNotEqual(answer, INSUFFICIENT_CONTEXT)
                self.assertEqual(context[0]["retrieval_mode"], expected_mode)
                self.assertEqual(validate_answer(answer, context), answer)
                self.assertEqual(app.session_state["rag_sources"], unique_sources(context))
                self.assertTrue(any(e.label == "Sources used" for e in app.expander))
                self.assertTrue(any(e.label == "Technical details" for e in app.expander))
                self.assertFalse(any("Test " in h.value for h in app.subheader))
                print("Live journey passed:", expected_mode, "| intent", app.session_state["detected_intent"],
                      "| unique source pages", len(app.session_state["rag_sources"]))
            self.assertEqual(app.session_state["generated_answers"], 2)
            self.assertTrue(any("8 / 10" in c.value for c in app.sidebar.caption))


if __name__ == "__main__":
    unittest.main()
