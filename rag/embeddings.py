"""Shared CPU embedding model and normalized float32 text vectors."""

from pathlib import Path

import numpy as np
import streamlit as st

from rag.chunker import TextChunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load once per Streamlit process; reuse across indexing and queries.

    Only the model is shared. Paper text and indexes stay in session state.
    The first call downloads model files; subsequent calls use the local cache.
    """
    from sentence_transformers import SentenceTransformer

    cache_folder = Path(__file__).resolve().parents[1] / ".model-cache"
    return SentenceTransformer(MODEL_NAME, device="cpu", cache_folder=str(cache_folder))


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode in input order, producing one vector for each text.

    MiniLM truncates beyond its 256-word-piece input limit. We preserve the
    complete chunk text separately, even when the model truncates its input.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    if any(not text.strip() for text in texts):
        raise ValueError("Embedding inputs must contain text.")
    vectors = load_embedding_model().encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def embed_chunks(chunks: list[TextChunk]) -> np.ndarray:
    """Row i corresponds to chunks[i]; do not modify the metadata."""
    return embed_texts([chunk["text"] for chunk in chunks])
