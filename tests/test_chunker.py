"""
Unit tests for document chunking module (src.chunker).
"""

import unittest
from src.chunker import create_chunks


class TestChunker(unittest.TestCase):
    def setUp(self):
        # Sample document with 3 pages
        self.sample_pages = [
            {
                "page_number": 1,
                "total_pages": 3,
                "file_name": "sample_notes.pdf",
                "file_path": "/data/sample_notes.pdf",
                "text": "A" * 1500,  # Longer than chunk_size=1000
                "has_text": True,
                "char_count": 1500,
            },
            {
                "page_number": 2,
                "total_pages": 3,
                "file_name": "sample_notes.pdf",
                "file_path": "/data/sample_notes.pdf",
                "text": "B" * 400,   # Shorter than chunk_size=1000
                "has_text": True,
                "char_count": 400,
            },
            {
                "page_number": 3,
                "total_pages": 3,
                "file_name": "sample_notes.pdf",
                "file_path": "/data/sample_notes.pdf",
                "text": "",          # Blank page
                "has_text": False,
                "char_count": 0,
            },
        ]

    def test_parameter_validation(self):
        # chunk_size <= 0
        with self.assertRaises(ValueError):
            create_chunks(self.sample_pages, chunk_size=0, overlap=0)
        with self.assertRaises(ValueError):
            create_chunks(self.sample_pages, chunk_size=-100, overlap=50)

        # overlap < 0
        with self.assertRaises(ValueError):
            create_chunks(self.sample_pages, chunk_size=1000, overlap=-10)

        # overlap >= chunk_size
        with self.assertRaises(ValueError):
            create_chunks(self.sample_pages, chunk_size=500, overlap=500)
        with self.assertRaises(ValueError):
            create_chunks(self.sample_pages, chunk_size=500, overlap=600)

    def test_short_page_creates_single_chunk(self):
        short_page = [self.sample_pages[1]]  # Page 2: 400 chars
        chunks = create_chunks(short_page, chunk_size=1000, overlap=200)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "sample_notes.pdf_p2_c0")
        self.assertEqual(chunks[0]["char_count"], 400)
        self.assertEqual(chunks[0]["chunk_index"], 0)
        self.assertEqual(chunks[0]["page_number"], 2)

    def test_empty_page_is_skipped(self):
        empty_page = [self.sample_pages[2]]  # Page 3: ""
        chunks = create_chunks(empty_page, chunk_size=1000, overlap=200)
        self.assertEqual(len(chunks), 0)

    def test_normal_chunking_and_overlap(self):
        # Test exact overlap slicing
        text = "0123456789" * 150  # 1500 chars
        page = [{
            "page_number": 1,
            "total_pages": 1,
            "file_name": "doc.pdf",
            "file_path": "doc.pdf",
            "text": text,
            "has_text": True,
            "char_count": 1500,
        }]

        chunk_size = 1000
        overlap = 200
        chunks = create_chunks(page, chunk_size=chunk_size, overlap=overlap)

        # Chunk 0: chars 0 to 1000
        # Chunk 1: chars 800 to 1500 (length 700)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["char_count"], 1000)
        self.assertEqual(chunks[1]["char_count"], 700)

        # Verify that the overlap matches exactly
        chunk0_suffix = chunks[0]["text"][-overlap:]
        chunk1_prefix = chunks[1]["text"][:overlap]
        self.assertEqual(chunk0_suffix, chunk1_prefix)
        self.assertEqual(len(chunk0_suffix), 200)

    def test_deterministic_chunk_ids(self):
        chunks_run1 = create_chunks(self.sample_pages, chunk_size=1000, overlap=200)
        chunks_run2 = create_chunks(self.sample_pages, chunk_size=1000, overlap=200)

        self.assertEqual(len(chunks_run1), len(chunks_run2))
        for c1, c2 in zip(chunks_run1, chunks_run2):
            self.assertEqual(c1["chunk_id"], c2["chunk_id"])

        # Check expected IDs
        expected_ids = [
            "sample_notes.pdf_p1_c0",
            "sample_notes.pdf_p1_c1",
            "sample_notes.pdf_p2_c0",
        ]
        self.assertEqual([c["chunk_id"] for c in chunks_run1], expected_ids)

    def test_metadata_preservation(self):
        chunks = create_chunks(self.sample_pages, chunk_size=1000, overlap=200)
        for c in chunks:
            self.assertIn("chunk_id", c)
            self.assertIn("text", c)
            self.assertIn("page_number", c)
            self.assertIn("total_pages", c)
            self.assertIn("file_name", c)
            self.assertIn("file_path", c)
            self.assertIn("chunk_index", c)
            self.assertIn("char_count", c)
            self.assertEqual(c["char_count"], len(c["text"]))

    def test_page_boundaries_preserved(self):
        chunks = create_chunks(self.sample_pages, chunk_size=1000, overlap=200)

        # Page 1 has 'A', Page 2 has 'B'
        for c in chunks:
            if c["page_number"] == 1:
                self.assertTrue(all(ch == "A" for ch in c["text"]))
            elif c["page_number"] == 2:
                self.assertTrue(all(ch == "B" for ch in c["text"]))

    def test_unicode_handling(self):
        unicode_text = "Database Notes: ? ACID Properties ? Atomicity ? Consistency"
        page = [{
            "page_number": 1,
            "total_pages": 1,
            "file_name": "unicode_doc.pdf",
            "file_path": "unicode_doc.pdf",
            "text": unicode_text,
            "has_text": True,
            "char_count": len(unicode_text),
        }]
        chunks = create_chunks(page, chunk_size=1000, overlap=200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], unicode_text)

    def test_original_page_dicts_not_mutated(self):
        original_copy = [dict(p) for p in self.sample_pages]
        _ = create_chunks(self.sample_pages, chunk_size=1000, overlap=200)
        self.assertEqual(self.sample_pages, original_copy)


if __name__ == "__main__":
    unittest.main()
