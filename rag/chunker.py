"""Word-based chunking that keeps every chunk on its original PDF page."""

from collections.abc import Iterable
from typing import TypedDict
from urllib.parse import quote

from research_assistant.models import ExtractedPage


class TextChunk(TypedDict):
    chunk_id: str
    filename: str
    page: int
    text: str


def chunk_pages(
    pages: Iterable[ExtractedPage],
    chunk_size: int = 250,
    overlap: int = 40,
) -> list[TextChunk]:
    """Split Phase 1 page records into overlapping word windows.

    Chunk numbers are one-based per filename/page. Full filenames are URL-
    escaped in IDs so punctuation and paths cannot collide with ID separators.
    No window crosses a page boundary. Whitespace is normalized by split/join;
    this is word chunking, not sentence or model-token chunking.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be an integer from 0 to chunk_size - 1")

    chunks: list[TextChunk] = []
    counts: dict[tuple[str, int], int] = {}
    step = chunk_size - overlap

    for page in pages:
        words = page.text.split()
        source = (page.filename, page.page_number)
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            counts[source] = counts.get(source, 0) + 1
            chunks.append({
                "chunk_id": f"{quote(page.filename, safe='')}_p{page.page_number}_c{counts[source]}",
                "filename": page.filename,
                "page": page.page_number,
                "text": " ".join(words[start:end]),
            })
            # Stop as soon as the page ends, avoiding overlap-only tails.
            if end == len(words):
                break
            start += step

    return chunks
