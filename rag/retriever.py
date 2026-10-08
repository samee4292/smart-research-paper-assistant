"""Enhance existing FAISS search with deterministic, classifier-guided queries."""

import math

CONFIDENCE_THRESHOLD = 0.70
COMPARISON_SOURCE_MIN_SIMILARITY = 0.25  # Diversity heuristic, not proof of relevance.
RRF_OFFSET = 60
EXPANSIONS = {
    "SUMMARY": "summary main idea purpose key findings conclusion overview",
    "METHODOLOGY": "method methodology methods experiment procedure dataset data collection training setup implementation",
    "RESULTS": "results findings performance evaluation outcome accuracy scores improvement",
    "LIMITATIONS": "limitations weaknesses drawbacks constraints shortcomings challenges future work",
    "DEFINITION": "definition meaning concept terminology refers to defined as",
}


def _unique(results):
    seen, unique = set(), []
    for result in results:
        if result["chunk_id"] not in seen:
            seen.add(result["chunk_id"])
            unique.append(dict(result))
    return unique


def _check(question, vector_store, top_k):
    if not question.strip():
        raise ValueError("Enter a non-empty retrieval question.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if vector_store is None:
        raise ValueError("The FAISS index has not been created.")


def _original_results(results, mode):
    return [{**r, "retrieval_origin": "ORIGINAL", "retrieval_mode": mode,
             "original_similarity": r["similarity"], "expansion_similarity": None,
             "relevance_score": r["similarity"], "ranking_metric": "COSINE"}
            for r in _unique(results)]


def basic_retrieve(question, vector_store, top_k=5):
    """The original baseline: one unexpanded question search, with debug fields."""
    _check(question, vector_store, top_k)
    return _original_results(vector_store.search(question.strip(), top_k=top_k), "BASIC")[:top_k]


def _merge(original, expanded):
    """Reciprocal-rank fusion: each query contributes equally by result rank.

    Cosines from different queries are not directly used as comparable ranking
    scores. A candidate appearing in both lists earns both rank contributions.
    """
    candidates = {}
    for origin, results in [("ORIGINAL", original), ("INTENT", expanded)]:
        for rank, result in enumerate(_unique(results), start=1):
            identifier = result["chunk_id"]
            if identifier not in candidates:
                candidates[identifier] = {**result, "retrieval_origin": origin,
                    "retrieval_mode": "INTENT_EXPANSION", "original_similarity": None,
                    "expansion_similarity": None, "relevance_score": 0.0, "ranking_metric": "RRF"}
            candidate = candidates[identifier]
            candidate["relevance_score"] += 1.0 / (RRF_OFFSET + rank)
            field = "original_similarity" if origin == "ORIGINAL" else "expansion_similarity"
            candidate[field] = result["similarity"]
            if candidate["original_similarity"] is not None and candidate["expansion_similarity"] is not None:
                candidate["retrieval_origin"] = "BOTH"
            # Preserve the original-query cosine when available; retain each
            # query's individual cosine too, so debug output is unambiguous.
            candidate["similarity"] = (candidate["original_similarity"]
                if candidate["original_similarity"] is not None else candidate["expansion_similarity"])
    return sorted(candidates.values(), key=lambda r: (-r["relevance_score"], r["chunk_id"]))


def retrieve_with_intent(question, predicted_intent, confidence, vector_store, top_k=5):
    """Always search the original question; return at most top_k unique chunks."""
    _check(question, vector_store, top_k)
    question = question.strip()
    confident = (isinstance(confidence, (int, float)) and math.isfinite(confidence)
                 and CONFIDENCE_THRESHOLD <= confidence <= 1.0)
    if not confident or predicted_intent not in {*EXPANSIONS, "COMPARISON"}:
        results = vector_store.search(question, top_k=top_k)
        return _original_results(results, "BASIC_FALLBACK")[:top_k]

    # Use wider candidate pools locally, but send only the final top_k to Gemini.
    candidate_count = top_k * 2
    if predicted_intent == "COMPARISON":
        # Exact FAISS already scans all vectors. Request all ranked hits so a
        # useful second paper cannot be hidden below ten hits from the first.
        index_count = getattr(getattr(vector_store, "index", None), "ntotal", 0)
        if isinstance(index_count, int):
            candidate_count = max(candidate_count, index_count)
    original = _unique(vector_store.search(question, top_k=candidate_count))
    if not original:
        return []

    if predicted_intent == "COMPARISON":
        ranked = _original_results(original, "COMPARISON_DIVERSITY")
        first_paper = ranked[0]["filename"]
        # Keep the best original hit. Add the best eligible other paper before
        # filling remaining slots by original cosine rank. Never force a weak
        # secondary source solely because another PDF was uploaded.
        eligible = [r for r in ranked if r["filename"] == first_paper
                    or r["similarity"] >= COMPARISON_SOURCE_MIN_SIMILARITY]
        selected = [eligible[0]]
        second = next((r for r in eligible if r["filename"] != first_paper), None)
        if second is not None and top_k > 1:
            selected.append(second)
        return _unique(selected + eligible)[:top_k]

    enhanced_query = f"{question} {EXPANSIONS[predicted_intent]}"
    try:
        expanded = vector_store.search(enhanced_query, top_k=candidate_count)
    except Exception:
        # Optional enhancement failure must not discard a successful original search.
        return _original_results(original, "EXPANSION_FAILED_FALLBACK")[:top_k]
    ranked = _merge(original, expanded)
    # Protect the single strongest original hit even if the classifier is wrong
    # with high confidence. All other original candidates participate in fusion.
    anchor = next(r for r in ranked if r["chunk_id"] == original[0]["chunk_id"])
    return _unique([anchor] + ranked)[:top_k]


def intent_aware_retrieve(question, vector_store, top_k=5):
    """Convenience counterpart to basic_retrieve for standalone A/B evaluation."""
    from classifier.predict import predict_intent

    try:
        prediction = predict_intent(question)
    except Exception:
        return basic_retrieve(question, vector_store, top_k)
    return retrieve_with_intent(question, prediction["intent"], prediction["confidence"], vector_store, top_k)
