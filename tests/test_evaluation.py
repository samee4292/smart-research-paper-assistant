from collections import Counter
import json
from pathlib import Path
import unittest

from classifier.model import INTENTS
from evaluation.compare_retrieval import FIXTURE
from evaluation.evaluate_classifier import evaluate_questions, load_questions
from evaluation.evaluate_retrieval import evaluate_questions as evaluate_retrieval, page_hit

DIRECTORY = Path(__file__).resolve().parents[1] / "evaluation"


class EvaluationTests(unittest.TestCase):
    def test_locked_evaluation_is_balanced_and_disjoint(self):
        rows = load_questions(DIRECTORY / "eval_questions.csv")
        self.assertEqual(len(rows), 60)
        self.assertEqual(Counter(r["expected_intent"] for r in rows), {i: 10 for i in INTENTS})

    def test_classifier_metrics_include_errors_confidence_and_matrix(self):
        rows = [{"question": str(i), "expected_intent": intent} for i, intent in enumerate(INTENTS)]
        def predict(question):
            index = int(question) if question.isdigit() else 3
            return {"intent": INTENTS[3] if index == 2 else INTENTS[index], "confidence": 0.8}
        result = evaluate_questions(rows, predict)
        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        self.assertEqual(result["per_intent"]["METHODOLOGY"]["accuracy"], 0)
        self.assertEqual(result["confusion_matrix"][2][3], 1)
        self.assertEqual(len(result["misclassified"]), 1)
        self.assertEqual(result["misclassified"][0]["confidence"], 0.8)
        self.assertEqual(result["known_failure"]["predicted_intent"], "RESULTS")

    def test_page_hits_use_filename_and_rank_and_distinguish_complete_coverage(self):
        expected = [{"filename": "a.pdf", "page": 3}, {"filename": "b.pdf", "page": 3}]
        results = [{"filename": "wrong.pdf", "page": 3}, {"filename": "a.pdf", "page": 3}]
        self.assertFalse(page_hit(results, expected, 1))
        self.assertTrue(page_hit(results, expected, 3))
        self.assertFalse(page_hit(results, expected, 5, require_all=True))
        with self.assertRaises(ValueError):
            page_hit(results, [], 5)

    def test_retrieval_scores_losses_without_hiding_them(self):
        rows = [{"question": "q", "expected_intent": "RESULTS",
                 "expected_sources": [{"filename": "a.pdf", "page": 3}]}]
        good = [{"filename": "a.pdf", "page": 3}]
        bad = [{"filename": "b.pdf", "page": 3}]
        result = evaluate_retrieval(rows, object(), basic=lambda *a, **kw: good,
            aware=lambda *a, **kw: bad, predictor=lambda q: {"intent": "RESULTS", "confidence": 0.9})
        self.assertEqual(result["metrics"]["basic"]["top_5_hit_rate"], 1)
        self.assertEqual(result["metrics"]["intent_aware"]["top_5_hit_rate"], 0)
        self.assertEqual(result["changes"]["top_5"]["worsened"], ["q"])

    def test_retrieval_labels_cover_all_intents_and_existing_pages(self):
        rows = json.loads((DIRECTORY / "retrieval_questions.json").read_text())
        self.assertEqual(Counter(r["expected_intent"] for r in rows), {i: 2 for i in INTENTS})
        for row in rows:
            for source in row["expected_sources"]:
                self.assertIn(source["filename"], FIXTURE)
                self.assertTrue(1 <= source["page"] <= len(FIXTURE[source["filename"]]))


if __name__ == "__main__":
    unittest.main()
