"""Combine recorded measurements into evaluation/results.json and results.md."""

import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent


def main():
    before = json.loads((DIRECTORY / "classifier_before.json").read_text(encoding="utf-8"))
    after = json.loads((DIRECTORY / "classifier_after.json").read_text(encoding="utf-8"))
    retrieval = json.loads((DIRECTORY / "retrieval_results.json").read_text(encoding="utf-8"))
    if before["evaluation_sha256"] != after["evaluation_sha256"]:
        raise ValueError("Before and after must use exactly the same evaluation CSV.")
    if after["model_sha256"] != retrieval["model_sha256"]:
        raise ValueError("Retrieval must use the evaluated retrained classifier.")
    baseline_config = json.loads((DIRECTORY / "baseline/intent_config.json").read_text())
    current_config = json.loads((DIRECTORY.parent / "classifier/intent_config.json").read_text())
    if baseline_config["architecture"] != current_config["architecture"]:
        raise ValueError("The FFNN architecture must remain unchanged.")
    limitations = [
        "The same 60 evaluation questions were frozen before training-data changes; no exact normalized evaluation question or known failure was added to training.",
        "Baseline errors informed the added examples. Therefore the before/after evaluation is a development comparison, not an untouched final test set or proof of broad generalization.",
        "Training stayed balanced: 300 to 390 authored examples (50 to 65 per intent). One retraining run used unchanged architecture and default hyperparameters, selecting a checkpoint by training validation loss, not by this evaluation accuracy.",
        "The larger training dataset changes its internal 70/15/15 split. Its internal test accuracy is not a directly paired before/after measurement; the fixed 60-question CSV is the paired comparison.",
        "Retrieval annotations were authored before executing the benchmark and the Phase 6 fixture, retrieval algorithm, threshold, and embedding model were not tuned to its outcomes.",
        "A page hit is weak evidence of retrieval success: it does not measure precision, completeness, valid cross-paper comparisons, or answer quality. The fixture has only 11 chunks and 12 questions.",
        "No Gemini request or Gemini-based scoring was used.",
    ]
    report = {"classifier": {"before": before, "after": after,
              "accuracy_change_percentage_points": 100 * (after["accuracy"] - before["accuracy"])},
              "retrieval": retrieval, "limitations": limitations}
    (DIRECTORY / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Phase 7 measured evaluation", "", "## Classifier: fixed 60-question comparison", "",
             f'Before: **{before["accuracy"]:.2%}** ({round(before["accuracy"] * before["question_count"])}/60).', "",
             f'After: **{after["accuracy"]:.2%}** ({round(after["accuracy"] * after["question_count"])}/60).', "",
             "| Intent | Before | After |", "|---|---:|---:|"]
    for intent in before["per_intent"]:
        lines.append(f'| {intent} | {before["per_intent"][intent]["accuracy"]:.0%} | {after["per_intent"][intent]["accuracy"]:.0%} |')
    lines += ["", "## Known failure (separate from the 60 questions)", "", before["known_failure"]["question"], ""]
    for label, result in [("Before", before), ("After", after)]:
        known = result["known_failure"]
        lines.append(f'{label}: expected RESULTS, predicted **{known["predicted_intent"]}**, confidence **{known["confidence"]:.1%}**.')
    for label, result in [("Before", before), ("After", after)]:
        lines += ["", f"## {label} confusion matrix", "", "Rows are expected; columns are predicted.", "",
                  "| Expected | " + " | ".join(result["confusion_matrix_order"]) + " |",
                  "|---|" + "---:|" * 6]
        for intent, row in zip(result["confusion_matrix_order"], result["confusion_matrix"]):
            lines.append("| " + intent + " | " + " | ".join(map(str, row)) + " |")
        lines += ["", f"### {label} incorrect predictions", ""]
        for row in result["misclassified"]:
            lines += [f'- {row["question"]} Expected **{row["expected_intent"]}**, predicted **{row["predicted_intent"]}**, confidence {row["confidence"]:.1%}.']
        if not result["misclassified"]:
            lines.append("None.")
    lines += ["", "## Retrieval: fixed 12-question fixture", "", retrieval["hit_definition"], "",
              "| Metric | Basic | Intent-aware |", "|---|---:|---:|"]
    for k in (1, 3, 5):
        metric = f"top_{k}_hit_rate"
        lines.append(f'| Top-{k} hit rate | {retrieval["metrics"]["basic"][metric]:.2%} | {retrieval["metrics"]["intent_aware"][metric]:.2%} |')
    lines += [f'| All labeled pages in Top 5 | {retrieval["metrics"]["basic"]["all_expected_pages_top_5_rate"]:.2%} | {retrieval["metrics"]["intent_aware"]["all_expected_pages_top_5_rate"]:.2%} |', ""]
    for k in (1, 3, 5):
        change = retrieval["changes"][f"top_{k}"]
        lines += [f'### Top-{k} changes', "", "Improved: " + ("; ".join(change["improved"]) or "none") + ".", "",
                  "Worsened: " + ("; ".join(change["worsened"]) or "none") + ".", ""]
    lines += ["## Question-level retrieval", ""]
    for row in retrieval["questions"]:
        expected = ", ".join(f'{s["filename"]} page {s["page"]}' for s in row["expected_sources"])
        lines += [f'### {row["question"]}', "", f'Expected: {expected}. Classifier: {row["predicted_intent"]} ({row["confidence"]:.1%}).', ""]
        for name, label in [("basic", "Basic"), ("intent_aware", "Intent-aware")]:
            chunks = ", ".join(f'{r["filename"]} p{r["page"]}' for r in row[name]["chunks"])
            lines += [f'{label}: {chunks}. Top-3 hit: {row[name]["hits"]["top_3"]}; Top-5 hit: {row[name]["hits"]["top_5"]}.', ""]
    lines += ["## Scope and limitations", ""] + ["- " + note for note in limitations]
    lines += ["", "Full classification reports, all prediction confidences, retrieved chunk text/scores, and dataset/model hashes are in `results.json`.", ""]
    (DIRECTORY / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f'Classifier: {before["accuracy"]:.2%} -> {after["accuracy"]:.2%}')
    print("Retrieval:", json.dumps(retrieval["metrics"], indent=2))
    print("Saved evaluation/results.json and evaluation/results.md")


if __name__ == "__main__":
    main()
