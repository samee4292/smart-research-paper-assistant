"""Extract PDF text in memory, keeping source metadata for every page."""

import logging

import pymupdf

from research_assistant.models import ExtractedPage, PaperExtraction

logger = logging.getLogger(__name__)


def extract_pdf(filename: str, content: bytes) -> PaperExtraction:
    """Return an extraction result rather than crashing on a bad upload.

    Blank pages are retained. This phase extracts embedded text only; it does
    not perform OCR on scanned pages or interpret figures and tables.
    """
    result = PaperExtraction(filename=filename)
    if not content:
        result.error = "The uploaded file is empty. Choose a valid PDF."
        return result

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if not document.is_pdf:
                result.error = "The uploaded file is not a PDF."
                return result
            if document.needs_pass:
                result.error = "This PDF is password-protected. Upload an unlocked copy."
                return result

            result.page_count = document.page_count
            if not result.page_count:
                result.error = "This PDF contains no pages."
                return result

            for index in range(result.page_count):
                page_number = index + 1
                try:
                    text = document.load_page(index).get_text("text", sort=True)
                    if not text.strip():
                        result.warnings.append(
                            f"Page {page_number} has no embedded text; it may be blank or scanned."
                        )
                except Exception:
                    logger.exception("Failed to extract %s page %s", filename, page_number)
                    text = ""
                    result.warnings.append(f"Text extraction failed on page {page_number}.")
                result.pages.append(ExtractedPage(filename, page_number, text))
    except Exception:
        logger.exception("Failed to open PDF %s", filename)
        result.error = "Could not read this PDF. It may be damaged or not a valid PDF."

    return result
