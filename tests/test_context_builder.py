import unittest
from src.context_builder import build_context, extract_page_citations


class TestContextBuilder(unittest.TestCase):

    def test_empty_results(self):
        self.assertEqual(build_context([]), "")

    def test_deduplication_by_id_and_text(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "TCP is a connection-oriented protocol.",
                "file_name": "net.pdf",
                "page_number": 1,
            },
            {
                "chunk_id": "c1",  # Duplicate ID
                "text": "TCP is a connection-oriented protocol.",
                "file_name": "net.pdf",
                "page_number": 1,
            },
            {
                "chunk_id": "c2",  # Different ID, same text
                "text": "TCP is a connection-oriented protocol.",
                "file_name": "net.pdf",
                "page_number": 1,
            },
            {
                "chunk_id": "c3",
                "text": "UDP is connectionless and faster.",
                "file_name": "net.pdf",
                "page_number": 2,
            }
        ]

        context = build_context(chunks, deduplicate=True)
        self.assertEqual(context.count("connection-oriented protocol"), 1)
        self.assertEqual(context.count("UDP is connectionless"), 1)

    def test_context_character_budget(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "Passage 1 content here.",
                "file_name": "doc1.pdf",
                "page_number": 1,
            },
            {
                "chunk_id": "c2",
                "text": "Passage 2 content that will exceed the budget.",
                "file_name": "doc1.pdf",
                "page_number": 2,
            }
        ]

        # First chunk length with header is ~60 chars
        ctx_limited = build_context(chunks, max_chars=80)
        self.assertIn("Passage 1", ctx_limited)
        self.assertNotIn("Passage 2", ctx_limited)

    def test_extract_page_citations(self):
        text = "Key concepts [Page 1], [Page 2, 3], and [Page 10]."
        cites = extract_page_citations(text)
        self.assertEqual(cites, [1, 2, 3, 10])


if __name__ == "__main__":
    unittest.main()
