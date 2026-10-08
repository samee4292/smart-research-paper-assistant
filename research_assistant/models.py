"""Simple data structures shared by extraction and the user interface."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractedPage:
    filename: str
    page_number: int  # One-based, matching the PDF viewer.
    text: str


@dataclass
class PaperExtraction:
    filename: str
    page_count: int = 0
    pages: list[ExtractedPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "Failed"
        if not any(page.text.strip() for page in self.pages):
            return "No extractable text"
        return "Extracted with warnings" if self.warnings else "Extracted"
