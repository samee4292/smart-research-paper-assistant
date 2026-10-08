from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rag.retriever import EXPANSIONS, basic_retrieve, intent_aware_retrieve, retrieve_with_intent


def chunk(identifier, paper="a.pdf", score=0.8):
    return dict(chunk_id=identifier, filename=paper, page=1, text=f"Text for {identifier}", similarity=score)


class RetrieverTests(unittest.TestCase):
    def test_threshold_and_unknown_intent_fallback(self):
        store = Mock()
        store.search.return_value = [chunk("a"), chunk("a"), chunk("b")]
        for intent, confidence in [("RESULTS", 0.69), ("RESULTS", float("nan")), ("UNKNOWN", 0.99)]:
            store.reset_mock()
            results = retrieve_with_intent("question", intent, confidence, store)
            store.search.assert_called_once_with("question", top_k=5)
            self.assertEqual([r["chunk_id"] for r in results], ["a", "b"])
            self.assertTrue(all(r["retrieval_origin"] == "ORIGINAL" for r in results))

    def test_all_expansions_preserve_original_and_metadata(self):
        for intent, terms in EXPANSIONS.items():
            with self.subTest(intent=intent):
                original = [chunk("anchor"), chunk("both"), chunk("original-only")]
                expanded = [chunk("intent-only", "b.pdf"), chunk("both")]
                store = Mock()
                store.search.side_effect = [original, expanded]
                results = retrieve_with_intent("original question", intent, 0.70, store)
                self.assertEqual(store.search.call_args_list[0].args[0], "original question")
                self.assertEqual(store.search.call_args_list[1].args[0], f"original question {terms}")
                self.assertEqual(results[0]["chunk_id"], "anchor")
                by_id = {r["chunk_id"]: r for r in results}
                self.assertEqual(by_id["both"]["retrieval_origin"], "BOTH")
                self.assertEqual(by_id["intent-only"]["retrieval_origin"], "INTENT")
                self.assertEqual(by_id["original-only"]["retrieval_origin"], "ORIGINAL")
                self.assertEqual(len(results), len(by_id))
                self.assertTrue(all({"filename", "page", "chunk_id", "text", "similarity", "relevance_score"} <= r.keys() for r in results))
                self.assertNotIn("retrieval_origin", original[0])  # No metadata mutation.

    def test_context_cap_and_strongest_original_survives(self):
        store = Mock()
        store.search.side_effect = [[chunk(f"o{i}") for i in range(10)], [chunk(f"e{i}") for i in range(10)]]
        results = retrieve_with_intent("question", "RESULTS", 0.95, store, top_k=5)
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["chunk_id"], "o0")
        self.assertTrue(any(r["retrieval_origin"] == "INTENT" for r in results))

    def test_comparison_finds_second_paper_beyond_initial_hits(self):
        store = Mock(index=SimpleNamespace(ntotal=15))
        store.search.return_value = [chunk(f"a{i}", score=0.9 - i * 0.01) for i in range(12)] + [chunk("b", "b.pdf", 0.6), chunk("c", "c.pdf", 0.1)]
        results = retrieve_with_intent("compare approaches", "COMPARISON", 0.99, store)
        store.search.assert_called_once_with("compare approaches", top_k=15)
        self.assertEqual(results[0]["chunk_id"], "a0")
        self.assertEqual(results[1]["filename"], "b.pdf")
        self.assertEqual(len(results), 5)
        self.assertFalse(any(r["filename"] == "c.pdf" for r in results))

    def test_comparison_does_not_force_weak_second_paper(self):
        store = Mock(index=SimpleNamespace(ntotal=3))
        store.search.return_value = [chunk("a1"), chunk("a2"), chunk("b", "b.pdf", 0.1)]
        results = retrieve_with_intent("compare approaches", "COMPARISON", 0.9, store)
        self.assertEqual({r["filename"] for r in results}, {"a.pdf"})

    def test_empty_failure_and_baseline_wrapper(self):
        store = Mock()
        store.search.return_value = []
        self.assertEqual(retrieve_with_intent("question", "RESULTS", 0.9, store), [])
        for question, index, count in [(" ", store, 5), ("question", None, 5), ("question", store, 0)]:
            with self.assertRaises(ValueError):
                basic_retrieve(question, index, count)
        store.search.side_effect = [[chunk("a")], RuntimeError("expansion unavailable")]
        self.assertEqual(retrieve_with_intent("question", "RESULTS", 0.9, store)[0]["retrieval_mode"], "EXPANSION_FAILED_FALLBACK")
        store.search.side_effect = None
        store.search.return_value = [chunk("a")]
        with patch("classifier.predict.predict_intent", side_effect=FileNotFoundError):
            self.assertEqual(intent_aware_retrieve("question", store)[0]["chunk_id"], "a")


if __name__ == "__main__":
    unittest.main()
