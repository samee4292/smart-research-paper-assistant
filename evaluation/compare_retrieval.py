"""Compare real classifier/FAISS results on fixture pages or supplied PDFs."""

import argparse
import json
from pathlib import Path

from classifier.predict import predict_intent
from rag.chunker import chunk_pages
from rag.retriever import basic_retrieve, retrieve_with_intent
from rag.vector_store import create_vector_store
from research_assistant.models import ExtractedPage
from research_assistant.pdf_extraction import extract_pdf

QUESTIONS = [
    "Give me an overview of this research.",
    "How was the experiment carried out?",
    "What performance did the system achieve?",
    "Where does the proposed approach fall short?",
    "How does the adaptive approach differ from the fixed approach?",
    "What does adaptive study planning mean?",
]

# Entirely invented development material, NOT findings from real research.
FIXTURE = {
    "adaptive_planning.pdf": [
        "Overview and purpose. This research explores adaptive study planning for university students. The main idea is to adjust study tasks according to weekly quiz results rather than follow an unchanged timetable. The proposed system targets topics with low mastery. Key findings suggest better retention with adaptive planning, while the conclusion calls for longer evaluations across institutions.",
        "Methods and experimental procedure. The experiment randomly assigned sixty students to adaptive or fixed study planning for six weeks. Both groups studied identical biology material. Researchers collected weekly quizzes and a delayed retention test. The adaptive implementation updated topic priorities using quiz mastery scores. The fixed group used a predefined schedule. Evaluation used group mean retention and study time.",
        "Results and performance evaluation. The adaptive study planning system achieved a mean delayed retention accuracy of eighty percent, compared with seventy-two percent for the fixed schedule. Weekly quiz scores improved by nine percentage points. Average weekly study time was similar in both groups. These fictional outcomes describe this development fixture only and are not actual scientific findings.",
        "Limitations and future work. The proposed adaptive approach falls short because the evaluation lasted only six weeks and used students at one institution. Weaknesses include a small sample and dependence on accurate quiz scores. Device availability constrains participation. The study cannot establish long-term learning benefits. Future work should evaluate other subjects and diverse students.",
        "Definition and terminology. Adaptive study planning means dynamically changing a learner's study schedule based on measured topic mastery. The concept refers to choosing what to study next using recent assessment data. In this system, low quiz scores increase the priority of a topic. Adaptive does not mean that students necessarily spend more time studying.",
    ],
    "fixed_planning.pdf": [
        "Overview. This research studies fixed study planning as a simple approach to organizing university revision. Its main idea is to give every learner a stable predefined timetable. The purpose is predictable coverage of all course topics. The conclusion is that a fixed schedule is easy to implement but does not respond to changes in individual topic mastery.",
        "Methodology and experiment setup. Sixty students were randomly allocated to two conditions for six weeks. The fixed study planning group followed the same timetable each week, regardless of quiz results. Researchers used identical course material and collected weekly assessments and delayed retention scores. The experiment controlled total scheduled study time to compare planning approaches.",
        "Results. Fixed study planning achieved seventy-two percent mean delayed retention accuracy. The adaptive comparison group achieved eighty percent. The fixed system's performance was lower on weak topics, although its schedule had less administrative overhead. These numbers are invented for testing retrieval and must not be interpreted as published research findings.",
        "Limitations and constraints. Fixed study planning cannot adapt its topic allocation when a learner struggles. The approach may waste time on already mastered concepts and neglect weaker topics. The evaluation used one institution and a short duration, limiting generalization. Future work could explore hybrid schedules that retain predictable timing while allowing topic adjustments.",
        "Definition and comparison. Fixed study planning means following a predefined timetable that does not change with quiz performance. The adaptive approach differs from the fixed approach by updating topic priorities based on learner mastery. Fixed planning prioritizes predictability, while adaptive planning responds to assessment data. Both approaches can allocate the same total study time.",
    ],
    "ocean_monitoring.pdf": [
        "Ocean sensor calibration. The ocean monitoring system measures salinity and water temperature using submerged instruments. Technicians calibrate electrodes against reference solutions and record tidal conditions. This unrelated fixture paper discusses marine measurements rather than learning schedules or educational experiments.",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", nargs="+", type=Path, help="Use your real PDFs instead of invented fixture pages.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/intent_retrieval_comparison.json"))
    args = parser.parse_args()
    if args.pdfs:
        pages = []
        for path in args.pdfs:
            extraction = extract_pdf(path.name, path.read_bytes())
            if extraction.error:
                raise ValueError(f"{path.name}: {extraction.error}")
            pages.extend(extraction.pages)
        dataset = "User-supplied PDFs: " + ", ".join(p.name for p in args.pdfs)
    else:
        pages = [ExtractedPage(name, number, text) for name, texts in FIXTURE.items()
                 for number, text in enumerate(texts, 1)]
        dataset = "Invented development fixture: 11 page records, not real research findings."
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No extractable chunks available for evaluation.")
    store = create_vector_store(chunks)
    examples = []
    for question in QUESTIONS:
        prediction = predict_intent(question)
        basic = basic_retrieve(question, store)
        aware = retrieve_with_intent(question, prediction["intent"], prediction["confidence"], store)
        examples.append({"question": question, **prediction, "basic": basic, "intent_aware": aware})
        print(f'{prediction["intent"]} {prediction["confidence"]:.4f}: {question}')
        print("  basic: " + ", ".join(r["chunk_id"] for r in basic))
        print("  aware: " + ", ".join(r["chunk_id"] for r in aware))
    report = {"dataset": dataset, "chunk_count": len(chunks),
              "note": "Real saved FFNN and MiniLM/FAISS outputs. Ranking changes are not evidence of improved retrieval quality.",
              "examples": examples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Basic versus intent-aware retrieval", "", dataset, "", report["note"], ""]
    for example in examples:
        lines += [f'## {example["question"]}', "", f'Prediction: **{example["intent"]}**, confidence **{example["confidence"]:.4f}**.', ""]
        for key, label in [("basic", "Basic"), ("intent_aware", "Intent-aware")]:
            lines += [f"### {label}", ""]
            for rank, result in enumerate(example[key], 1):
                lines += [f'{rank}. **{result["filename"]}, Page {result["page"]}** — `{result["chunk_id"]}`; origin {result["retrieval_origin"]}; score {result["relevance_score"]:.5f} ({result["ranking_metric"]}); original cosine {result["original_similarity"]}; expansion cosine {result["expansion_similarity"]}.', "", f'   {result["text"]}', ""]
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
