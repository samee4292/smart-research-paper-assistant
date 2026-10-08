"""Index lifecycle and development search interface."""

from hashlib import sha256
from contextlib import nullcontext
import logging

import streamlit as st

from rag.chunker import TextChunk
from rag.vector_store import create_vector_store

logger = logging.getLogger(__name__)


def ensure_session_index(chunks: list[TextChunk]) -> None:
    """Rebuild only when chunks change; never reuse results from old uploads."""
    signature = sha256(repr(chunks).encode("utf-8")).hexdigest()
    if signature != st.session_state.get("index_signature"):
        st.session_state.update(index_signature=signature, vector_store=None,
                                index_error=None, search_results=[], searched_query=None)
        if chunks:
            try:
                with st.spinner("Preparing your documents…"):
                    st.session_state["vector_store"] = create_vector_store(chunks)
            except Exception:
                logger.exception("Could not build semantic search index")
                st.session_state["index_error"] = (
                    "Could not create the index. Check that dependencies are installed "
                    "and the model can download on first use."
                )


def render_semantic_search(chunks: list[TextChunk], embedded: bool = False) -> None:
    """Keep the development search interface on the same session index."""
    with nullcontext() if embedded else st.expander("Test Semantic Search", expanded=True):
        if embedded:
            st.subheader("Test Semantic Search")
        if not chunks:
            st.info("Upload PDFs with extractable text to enable semantic search.")
            return
        store = st.session_state.get("vector_store")
        if store is None:
            st.error(st.session_state.get("index_error") or "The FAISS index has not been created.")
            if st.button("Retry indexing"):
                st.session_state.pop("index_signature", None)
                st.rerun()
            return
        st.caption(f"Index ready · {store.index.ntotal} chunks · all-MiniLM-L6-v2")
        st.caption("Long chunks are truncated to the model's input limit. Scores indicate similarity, not confidence.")
        query = st.text_input("Question or query", key="semantic_query")
        if st.button("Search", key="semantic_search"):
            st.session_state["search_results"] = []
            st.session_state["searched_query"] = None
            if not query.strip():
                st.warning("Enter a question or query before searching.")
            else:
                try:
                    with st.spinner("Searching uploaded papers…"):
                        st.session_state["search_results"] = store.search(query)
                    st.session_state["searched_query"] = query.strip()
                except Exception:
                    logger.exception("Semantic search failed")
                    st.error("Search failed. Check the local model and dependencies, then try again.")
        searched_query = st.session_state.get("searched_query")
        if searched_query:
            st.text(f"Results for: {searched_query}")
            if not st.session_state["search_results"]:
                st.info("No results available.")
            for rank, result in enumerate(st.session_state["search_results"], start=1):
                with st.container(border=True):
                    st.text(f"{rank}. {result['filename']} — Page {result['page']}")
                    st.text(f"Similarity: {result['similarity']:.3f} · Chunk ID: {result['chunk_id']}")
                    st.text(result["text"])
