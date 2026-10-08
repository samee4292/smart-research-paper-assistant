"""Opt-in real model check; the regular suite stays offline and fast."""

import os
import unittest

import pymupdf

from rag.chunker import chunk_pages
from rag.embeddings import load_embedding_model
from rag.vector_store import create_vector_store
from research_assistant.pdf_extraction import extract_pdf


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "Set RUN_MODEL_TESTS=1 to download/use MiniLM")
class SemanticIntegrationTests(unittest.TestCase):
    def test_pdf_to_semantic_results(self):
        with pymupdf.open() as pdf:
            for text in [
                "The transformer model requires large amounts of memory. Quadratic attention cost limits long sequences.",
                "This study measures plant growth in soil with different amounts of nitrogen fertilizer.",
                "The experiment investigates the orbital motion of planets around a star.",
            ]:
                pdf.new_page().insert_textbox((40, 40, 550, 300), text)
            content = pdf.tobytes()
        pages = extract_pdf("research.pdf", content).pages
        chunks = chunk_pages(pages)
        store = create_vector_store(chunks)
        results = store.search("What computational limitations affect transformers?")
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["filename"], "research.pdf")
        self.assertEqual(results[0]["page"], 1)
        self.assertEqual(results[0]["text"], chunks[0]["text"])
        self.assertTrue(all(-1.00001 <= r["similarity"] <= 1.00001 for r in results))
        self.assertIs(load_embedding_model(), load_embedding_model())
        print("Real-model ranking:", [(r["page"], round(r["similarity"], 3)) for r in results])


if __name__ == "__main__":
    unittest.main()
