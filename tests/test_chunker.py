import unittest

from rag.chunker import chunk_pages
from research_assistant.models import ExtractedPage


def page(word_count, filename="paper.pdf", number=1):
    return ExtractedPage(filename, number, " ".join(f"w{i}" for i in range(word_count)))


class ChunkerTests(unittest.TestCase):
    def test_long_page_overlap_and_complete_coverage(self):
        chunks = chunk_pages([page(1200)])
        words = [c["text"].split() for c in chunks]
        self.assertEqual([len(w) for w in words], [250, 250, 250, 250, 250, 150])
        for previous, current in zip(words, words[1:]):
            self.assertEqual(previous[-40:], current[:40])
        reconstructed = words[0] + [word for window in words[1:] for word in window[40:]]
        self.assertEqual(reconstructed, page(1200).text.split())

    def test_boundary_sizes_without_overlap_only_tails(self):
        for size, lengths in [(1, [1]), (249, [249]), (250, [250]),
                              (251, [250, 41]), (460, [250, 250])]:
            with self.subTest(size=size):
                self.assertEqual([len(c["text"].split()) for c in chunk_pages([page(size)])], lengths)

    def test_empty_input_and_whitespace(self):
        self.assertEqual(chunk_pages([]), [])
        self.assertEqual(chunk_pages([ExtractedPage("blank.pdf", 1, " \n\t ")]), [])

    def test_metadata_and_page_isolation(self):
        chunks = chunk_pages([page(251, "a.pdf", 3), page(5, "a.pdf", 4), page(2, "a.PDF", 3)])
        self.assertEqual([c["page"] for c in chunks], [3, 3, 4, 3])
        self.assertEqual([c["filename"] for c in chunks], ["a.pdf", "a.pdf", "a.pdf", "a.PDF"])
        self.assertEqual(chunks[1]["chunk_id"], "a.pdf_p3_c2")
        self.assertEqual(len({c["chunk_id"] for c in chunks}), 4)
        self.assertEqual(chunks[2]["text"], page(5).text)

    def test_custom_settings_and_validation(self):
        self.assertEqual([len(c["text"].split()) for c in chunk_pages([page(7)], 3, 0)], [3, 3, 1])
        for size, overlap in [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6), (1.5, 0), (5, True)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_pages([], size, overlap)


if __name__ == "__main__":
    unittest.main()
