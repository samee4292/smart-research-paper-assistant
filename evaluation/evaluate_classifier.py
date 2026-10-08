"""Evaluate the saved FFNN: python -m evaluation.evaluate_classifier."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from sklearn.metrics import classification_report, confusion_matrix

from classifier.model import INTENTS
from classifier.predict import MODEL_PATH, predict_intent
from classifier.train import load_dataset

DIRECTORY = Path(__file__).resolve().parent
KNOWN_FAILURE = "What performance did the adaptive planner achieve?"


def normalize(text):
    return " ".join(text.casefold().split()).rstrip(".?!")


def load_questions(path, training_path=DIRECTORY.parent / "classifier/dataset.csv"):
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["question", "expected_intent"]:
            raise ValueError("Evaluation columns must be question,expected_intent.")
        rows = list(reader)
    training, _ = load_dataset(Path(training_path))
    training_texts, seen = {normalize(q) for q in training}, set()
    for row in rows:
        question, intent = row["question"].strip(), row["expected_intent"].strip()
        key = normalize(question)
        if not question or intent not in INTENTS or key in seen or key in training_texts:
            raise ValueError("Evaluation questions must be valid, unique, and absent from training data.")
        seen.add(key)
        row.update(question=question, expected_intent=intent)
    if not rows or set(Counter(r["expected_intent"] for r in rows)) != set(INTENTS):
        raise ValueError("Evaluation must cover all six intents.")
    if normalize(KNOWN_FAILURE) in training_texts:
        raise ValueError("The known failure must not be a training example.")
    return rows


def evaluate_questions(rows, predictor=predict_intent):
    predictions = []
    for row in rows:
        prediction = predictor(row["question"])
        predictions.append({**row, "predicted_intent": prediction["intent"],
                            "confidence": prediction["confidence"],
                            "correct": row["expected_intent"] == prediction["intent"]})
    expected = [r["expected_intent"] for r in predictions]
    predicted = [r["predicted_intent"] for r in predictions]
    per_intent = {}
    for intent in INTENTS:
        subset = [r for r in predictions if r["expected_intent"] == intent]
        correct = sum(r["correct"] for r in subset)
        per_intent[intent] = {"correct": correct, "total": len(subset), "accuracy": correct / len(subset)}
    known = predictor(KNOWN_FAILURE)
    return {"question_count": len(rows), "accuracy": sum(r["correct"] for r in predictions) / len(rows),
            "per_intent": per_intent,
            "classification_report": classification_report(expected, predicted, labels=INTENTS,
                target_names=INTENTS, output_dict=True, zero_division=0),
            "confusion_matrix_order": INTENTS,
            "confusion_matrix": confusion_matrix(expected, predicted, labels=INTENTS).tolist(),
            "predictions": predictions, "misclassified": [r for r in predictions if not r["correct"]],
            "known_failure": {"question": KNOWN_FAILURE, "expected_intent": "RESULTS",
                              "predicted_intent": known["intent"], "confidence": known["confidence"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DIRECTORY / "eval_questions.csv")
    parser.add_argument("--output", type=Path, default=DIRECTORY / "classifier_after.json")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="Saved FFNN checkpoint; supports reproducing the preserved baseline.")
    parser.add_argument("--training-data", type=Path, default=DIRECTORY.parent / "classifier/dataset.csv")
    args = parser.parse_args()
    rows = load_questions(args.questions, args.training_data)
    # Only this standalone evaluation process redirects the existing predictor;
    # no application files or checkpoints are changed.
    with patch("classifier.predict.MODEL_PATH", args.model):
        result = evaluate_questions(rows)
    result.update(evaluation_sha256=sha256(args.questions.read_bytes()).hexdigest(),
                  model_sha256=sha256(args.model.read_bytes()).hexdigest(),
                  training_dataset_sha256=sha256(args.training_data.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f'Overall accuracy: {result["accuracy"]:.2%} ({len(rows)} questions)', flush=True)
    for intent, metrics in result["per_intent"].items():
        print(f'{intent}: {metrics["accuracy"]:.2%} ({metrics["correct"]}/{metrics["total"]})', flush=True)
    print("Confusion matrix order:", INTENTS)
    for row in result["confusion_matrix"]:
        print(row)
    print("Classification report:", json.dumps(result["classification_report"], indent=2))
    print("\nMisclassified questions:")
    for row in result["misclassified"]:
        print(f'Question: {row["question"]}\nExpected: {row["expected_intent"]}; Predicted: {row["predicted_intent"]}; Confidence: {row["confidence"]:.1%}\n')
    known = result["known_failure"]
    print(f'Known failure: {known["question"]}\nExpected: RESULTS; Predicted: {known["predicted_intent"]}; Confidence: {known["confidence"]:.1%}')


if __name__ == "__main__":
    main()
