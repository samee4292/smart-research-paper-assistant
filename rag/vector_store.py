"""Session-local FAISS cosine search with an ordered chunk mapping."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rag.chunker import TextChunk
from rag.embeddings import embed_chunks, embed_texts


def normalized_vectors(vectors: np.ndarray) -> np.ndarray:
    """FAISS expects contiguous float32 rows; normalize without mutating input."""
    vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Embeddings must be a two-dimensional vector matrix.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("Embeddings must be finite, nonzero vectors.")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


@dataclass
class VectorStore:
    index: Any
    chunks: list[TextChunk]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return chunk metadata plus cosine similarity, highest score first."""
        if not query.strip():
            raise ValueError("Enter a non-empty search query.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if self.index is None or self.index.ntotal == 0:
            return []
        vector = normalized_vectors(embed_texts([query.strip()]))
        if vector.shape != (1, self.index.d):
            raise ValueError("Query and index embedding dimensions do not match.")
        scores, positions = self.index.search(vector, min(top_k, len(self.chunks)))
        return [
            {**self.chunks[int(position)], "similarity": float(score)}
            for score, position in zip(scores[0], positions[0]) if position >= 0
        ]


def create_vector_store(chunks: list[TextChunk]) -> VectorStore | None:
    """Embed chunks in order and map each FAISS row back to its source."""
    if not chunks:
        return None
    import faiss

    vectors = normalized_vectors(embed_chunks(chunks))
    if vectors.shape[0] != len(chunks):
        raise ValueError("Each chunk must have exactly one embedding.")
    # Inner product equals cosine similarity when both sides have unit length.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return VectorStore(index, [dict(chunk) for chunk in chunks])
