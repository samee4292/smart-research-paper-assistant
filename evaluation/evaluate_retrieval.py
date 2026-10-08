"""Measure page hits on the unchanged Phase 6 fixture, without Gemini."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pymupdf

from classifier.predict import MODEL_PATH, predict_intent
from evaluation.compare_retrieval import FIXTURE
from rag.chunker import chunk_pages
from rag.retriever import basic_retrieve, intent_aware_retrieve
from rag.vector_store import create_vector_store
from research_assistant.pdf_extraction import extract_pdf

DIRECTORY = Path(__file__).resolve().parent


def fixture_pages():
    """Render the existing artificial page text into in-memory PDFs and extract it."""
    pages = []
    for filename, texts in FIXTURE.items():
        with pymupdf.open() as document:
            for text in texts:
                page = document.new_page()
                if page.insert_textbox((40, 40, 555, 780), text, fontsize=11) < 0:
                    raise ValueError("Fixture text did not fit on its PDF page.")
            extraction = extract_pdf(filename, document.tobytes())
        if extraction.error:
            raise ValueError(extraction.error)
        pages.extend(extraction.pages)
    return pages


def page_hit(results, expected_sources, k, require_all=False):
    retrieved = {(r["filename"], r["page"]) for r in results[:k]}
    expected = {(s["filename"], s["page"]) for s in expected_sources}
    if not expected:
        raise ValueError("Every retrieval question needs a labeled source page.")
    return expected <= retrieved if require_all else bool(expected & retrieved)


def evaluate_questions(rows, store, basic=basic_retrieve, aware=intent_aware_retrieve,
                       predictor=predict_intent):
    results = []
    for row in rows:
        prediction = predictor(row["question"])
        entry = {**row, "predicted_intent": prediction["intent"], "confidence": prediction["confidence"]}
        for name, retrieve in [("basic", basic), ("intent_aware", aware)]:
            chunks = retrieve(row["question"], store, top_k=5)
            entry[name] = {"hits": {f"top_{k}": page_hit(chunks, row["expected_sources"], k) for k in (1, 3, 5)},
                           "all_expected_pages_top_5": page_hit(chunks, row["expected_sources"], 5, require_all=True),
                           "chunks": chunks}
        results.append(entry)
    metrics = {}
    for name in ("basic", "intent_aware"):
        metrics[name] = {f"top_{k}_hit_rate": sum(r[name]["hits"][f"top_{k}"] for r in results) / len(results)
                         for k in (1, 3, 5)}
        metrics[name]["all_expected_pages_top_5_rate"] = sum(r[name]["all_expected_pages_top_5"] for r in results) / len(results)
    changes = {}
    for k in (1, 3, 5):
        key = f"top_{k}"
        changes[key] = {
            "improved": [r["question"] for r in results if not r["basic"]["hits"][key] and r["intent_aware"]["hits"][key]],
            "worsened": [r["question"] for r in results if r["basic"]["hits"][key] and not r["intent_aware"]["hits"][key]],
        }
    return {"question_count": len(rows), "metrics": metrics, "changes": changes, "questions": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "retrieval_results.json")
    args = parser.parse_args()
    question_path = DIRECTORY / "retrieval_questions.json"
    rows = json.loads(question_path.read_text(encoding="utf-8"))
    if len(rows) != 12 or Counter(r["expected_intent"] for r in rows) != {label: 2 for label in ["SUMMARY", "COMPARISON", "METHODOLOGY", "RESULTS", "LIMITATIONS", "DEFINITION"]}:
        raise ValueError("Expected twelve labeled questions, two per intent.")
    pages = fixture_pages()
    valid_sources = {(p.filename, p.page_number) for p in pages}
    if any(not r["expected_sources"] or any((s["filename"], s["page"]) not in valid_sources for s in r["expected_sources"]) for r in rows):
        raise ValueError("A labeled source page is missing from the fixture.")
    store = create_vector_store(chunk_pages(pages))
    result = evaluate_questions(rows, store)
    result.update(dataset="Unchanged Phase 6 artificial planning papers plus unrelated ocean page; not real research.",
                  fixture_sha256=sha256(json.dumps(FIXTURE, sort_keys=True).encode()).hexdigest(),
                  questions_sha256=sha256(question_path.read_bytes()).hexdigest(),
                  model_sha256=sha256(MODEL_PATH.read_bytes()).hexdigest(),
                  page_count=len(pages), chunk_count=len(store.chunks),
                  hit_definition="At least one manually labeled (filename, page) occurs within the first k of the same five selected chunks.",
                  limitations="Small synthetic fixture. Page hits do not measure passage completeness, precision, factual answers, or broad generalization. Labels were authored before running retrieval and were not adjusted to scores.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print("Changes:", json.dumps(result["changes"], indent=2))
    for row in result["questions"]:
        print(row["question"], "|", row["predicted_intent"], f'{row["confidence"]:.1%}',
              "| basic", row["basic"]["hits"], "| aware", row["intent_aware"]["hits"])


if __name__ == "__main__":
    main()
