"""
Unit tests for context builder module (src.context_builder).
"""

import unittest
from src.context_builder import build_context


class TestContextBuilder(unittest.TestCase):
    def test_empty_results_returns_empty_string(self):
        self.assertEqual(build_context([]), "")

    def test_single_chunk_formatting_and_metadata(self):
        results = [
            {
                "file_name": "notes.pdf",
                "page_number": 2,
                "chunk_id": "notes.pdf_p2_c0",
                "text": "This is page 2 content.",
            }
        ]
        context = build_context(results)
        self.assertIn("[Document: notes.pdf | Page: 2 | Chunk: notes.pdf_p2_c0]", context)
        self.assertIn("This is page 2 content.", context)

    def test_multiple_chunks_formatting(self):
        results = [
            {
                "file_name": "doc1.pdf",
                "page_number": 1,
                "chunk_id": "doc1_p1_c0",
                "text": "First passage.",
            },
            {
                "file_name": "doc1.pdf",
                "page_number": 3,
                "chunk_id": "doc1_p3_c0",
                "text": "Second passage.",
            },
        ]
        context = build_context(results)
        self.assertIn("[Document: doc1.pdf | Page: 1 | Chunk: doc1_p1_c0]", context)
        self.assertIn("[Document: doc1.pdf | Page: 3 | Chunk: doc1_p3_c0]", context)
        self.assertIn("---", context)
        self.assertIn("First passage.", context)
        self.assertIn("Second passage.", context)


if __name__ == "__main__":
    unittest.main()
