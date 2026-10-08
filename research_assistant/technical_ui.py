"""Optional evidence diagnostics and preserved developer tools."""

from hashlib import sha256

import streamlit as st

from rag.retriever import CONFIDENCE_THRESHOLD
from research_assistant.classifier_ui import render_intent_classifier
from research_assistant.search_ui import render_semantic_search


def render_technical_details(results, chunks):
    with st.expander("Technical details"):
        intent = st.session_state.get("detected_intent")
        confidence = st.session_state.get("intent_confidence")
        context = st.session_state.get("retrieval_debug", [])
        st.text(f"Detected Intent: {intent or 'Not available yet'}")
        if confidence is not None:
            st.text(f"Confidence: {confidence:.1%}")
        if context:
            strategy = context[0]["retrieval_mode"]
            mode = "Intent-aware" if strategy in {"INTENT_EXPANSION", "COMPARISON_DIVERSITY"} else "Standard semantic"
            st.text(f"Retrieval mode: {mode}")
            if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
                st.caption("Standard semantic retrieval used due to low intent confidence.")
            elif not intent:
                st.caption("Intent detection was unavailable; standard semantic retrieval was used.")
            elif strategy == "EXPANSION_FAILED_FALLBACK":
                st.caption("Intent expansion was unavailable; standard semantic retrieval was used.")
            st.caption(f"{len(context)} retrieved chunks · Strategy: {strategy}. Cosine scores and rank-fusion scores are not probabilities.")
            st.dataframe([{
                "Rank": rank, "Filename": c["filename"], "Page": c["page"],
                "Chunk ID": c["chunk_id"], "Origin": c["retrieval_origin"],
                "Original cosine": c["original_similarity"], "Expansion cosine": c["expansion_similarity"],
                "Relevance score": c["relevance_score"], "Ranking metric": c["ranking_metric"],
            } for rank, c in enumerate(context, 1)], hide_index=True, width="stretch")
            selected = st.selectbox("Retrieved passage", range(len(context)),
                format_func=lambda i: f"{i + 1}. {context[i]['filename']} · Page {context[i]['page']}",
                key="retrieved_passage")
            st.text(context[selected]["text"])
        else:
            st.caption("Ask a question to inspect its retrieval evidence.")

        if st.checkbox("Show development tools", key="show_development_tools"):
            st.divider()
            render_document_inspection(results, chunks)
            st.divider()
            render_semantic_search(chunks, embedded=True)
            st.divider()
            render_intent_classifier(embedded=True)


def render_document_inspection(results, chunks):
    st.subheader("Document inspection")
    valid = [r for r in results if not r.error and r.pages]
    if valid:
        index = st.selectbox("Paper", range(len(valid)),
                             format_func=lambda i: valid[i].filename, key="inspect_paper")
        paper = valid[index]
        if paper.warnings:
            st.warning("\n\n".join(paper.warnings))
        selected = st.selectbox("Page", range(1, paper.page_count + 1),
                                key=f"paper_page_{index}")
        page = paper.pages[selected - 1]
        st.caption(f"Source: {page.filename} · Page {page.page_number}")
        page_key = sha256(repr(page).encode()).hexdigest()
        st.text_area("Extracted text", page.text, height=220, disabled=True, key=f"page_{page_key}")
    st.subheader("Inspect Chunks")
    st.caption("Up to 250 words · 40-word overlap · Page-local chunks")
    if not chunks:
        st.info("No chunks available. Choose a PDF with readable text.")
        return
    selected = st.selectbox("Chunk", range(len(chunks)),
                           format_func=lambda i: chunks[i]["chunk_id"], key="inspect_chunk")
    chunk = chunks[selected]
    st.text(f"Chunk ID: {chunk['chunk_id']}")
    st.text(f"Paper filename: {chunk['filename']}")
    st.text(f"Page number: {chunk['page']}")
    preview_id = sha256(repr(chunk).encode()).hexdigest()
    st.text_area("Chunk text", chunk["text"], height=220, disabled=True, key=f"chunk_text_{preview_id}")

