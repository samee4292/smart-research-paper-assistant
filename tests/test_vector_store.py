import unittest
from unittest.mock import patch

import numpy as np

from rag.vector_store import create_vector_store, normalized_vectors
from rag.embeddings import embed_chunks, embed_texts


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [dict(chunk_id=f"x{i}", filename="paper.pdf", page=i + 1, text=f"text {i}")
                       for i in range(3)]

    def test_real_faiss_ranking_and_mapping(self):
        vectors = np.array([[10, 0], [1, 1], [0, 5]], dtype=np.float64)
        with patch("rag.vector_store.embed_chunks", return_value=vectors):
            store = create_vector_store(self.chunks)
        self.chunks[0]["text"] = "changed after indexing"
        with patch("rag.vector_store.embed_texts", return_value=np.array([[2, 0]])):
            results = store.search("methods")
        self.assertEqual([r["chunk_id"] for r in results], ["x0", "x1", "x2"])
        self.assertEqual(results[0]["text"], "text 0")
        self.assertEqual(results[0]["page"], 1)
        np.testing.assert_allclose([r["similarity"] for r in results], [1, 2**-0.5, 0], atol=1e-6)
        self.assertEqual(len(results), 3)  # Fewer than top five is safe.
        with self.assertRaises(ValueError):
            store.search("   ")

    def test_empty_index_and_bad_vectors(self):
        self.assertIsNone(create_vector_store([]))
        for vectors in [np.zeros((1, 2)), np.array([[np.nan, 1]]), np.array([1, 2])]:
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                normalized_vectors(vectors)

    def test_embedding_order_and_configuration(self):
        from unittest.mock import Mock
        model = Mock()
        model.encode.return_value = np.ones((3, 2))
        original = [dict(c) for c in self.chunks]
        with patch("rag.embeddings.load_embedding_model", return_value=model):
            result = embed_chunks(self.chunks)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(self.chunks, original)
        self.assertEqual(model.encode.call_args.args[0], [c["text"] for c in self.chunks])
        self.assertTrue(model.encode.call_args.kwargs["normalize_embeddings"])
        self.assertEqual(embed_texts([]).shape, (0, 384))


if __name__ == "__main__":
    unittest.main()
