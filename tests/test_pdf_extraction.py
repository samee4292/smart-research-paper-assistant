"""Small generated PDFs verify extraction without committing research papers."""

import unittest

import pymupdf

from research_assistant.pdf_extraction import extract_pdf


class ExtractionTests(unittest.TestCase):
    def test_pages_and_metadata(self):
        with pymupdf.open() as doc:
            for text in ["First page methods", "Second page results"]:
                doc.new_page().insert_text((72, 72), text)
            content = doc.tobytes()
        result = extract_pdf("paper.pdf", content)
        self.assertEqual(result.status, "Extracted")
        self.assertEqual(result.page_count, 2)
        self.assertEqual([p.page_number for p in result.pages], [1, 2])
        self.assertTrue(all(p.filename == "paper.pdf" for p in result.pages))
        self.assertIn("Second page results", result.pages[1].text)

    def test_blank_page_retained(self):
        with pymupdf.open() as doc:
            doc.new_page()
            content = doc.tobytes()
        result = extract_pdf("blank.pdf", content)
        self.assertEqual(result.status, "No extractable text")
        self.assertEqual(len(result.pages), 1)
        self.assertTrue(result.warnings)

    def test_invalid_and_empty_uploads(self):
        for content in [b"", b"not a PDF"]:
            with self.subTest(content=content):
                self.assertEqual(extract_pdf("bad.pdf", content).status, "Failed")

    def test_password_protected_pdf(self):
        with pymupdf.open() as doc:
            doc.new_page()
            content = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                  owner_pw="owner", user_pw="reader")
        result = extract_pdf("locked.pdf", content)
        self.assertEqual(result.status, "Failed")
        self.assertIn("password-protected", result.error)


if __name__ == "__main__":
    unittest.main()
