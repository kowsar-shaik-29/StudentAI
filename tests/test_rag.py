"""
Unit tests for end-to-end RAG pipeline module (src.rag).
Uses mocking to test end-to-end pipeline orchestration and error handling.
"""

import unittest
from unittest.mock import patch, MagicMock
from src.rag import answer_question


class TestRAG(unittest.TestCase):
    def test_empty_question_raises_value_error(self):
        with self.assertRaises(ValueError):
            answer_question("")
        with self.assertRaises(ValueError):
            answer_question("   ")

    def test_invalid_top_k_raises_value_error(self):
        with self.assertRaises(ValueError):
            answer_question("Valid question?", top_k=0)
        with self.assertRaises(ValueError):
            answer_question("Valid question?", top_k=-2)

    @patch("src.rag.search")
    @patch("src.rag.generate_answer")
    def test_complete_rag_pipeline(self, mock_gen, mock_search):
        # Mock retrieved chunks
        mock_chunks = [
            {
                "score": 0.85,
                "file_name": "notes.pdf",
                "page_number": 1,
                "chunk_id": "notes.pdf_p1_c0",
                "text": "Star topology uses a central hub.",
                "total_pages": 5,
                "chunk_index": 0,
                "char_count": 32,
            }
        ]
        mock_search.return_value = mock_chunks
        mock_gen.return_value = "Star topology uses a central hub. [Page 1]"

        mock_qdrant = MagicMock()
        mock_genai = MagicMock()

        result = answer_question(
            question="What is star topology?",
            top_k=3,
            client=mock_qdrant,
            genai_client=mock_genai,
        )

        # 1. Search called with correct query and top_k
        mock_search.assert_called_once_with(
            query="What is star topology?",
            top_k=3,
            client=mock_qdrant,
            collection_name=None,
        )

        # 2. Generator called with question and formatted context
        mock_gen.assert_called_once()
        gen_args = mock_gen.call_args.kwargs
        self.assertEqual(gen_args["question"], "What is star topology?")
        self.assertIn("Star topology uses a central hub.", gen_args["context"])
        self.assertIn("[Document: notes.pdf | Page: 1 | Chunk: notes.pdf_p1_c0]", gen_args["context"])

        # 3. Structured result contains answer and sources
        self.assertEqual(result["question"], "What is star topology?")
        self.assertEqual(result["answer"], "Star topology uses a central hub. [Page 1]")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["score"], 0.85)
        self.assertEqual(result["sources"][0]["page_number"], 1)
        self.assertEqual(result["sources"][0]["chunk_id"], "notes.pdf_p1_c0")

    @patch("src.rag.search")
    def test_retrieval_failure_propagates(self, mock_search):
        mock_search.side_effect = RuntimeError("Qdrant connection failure")
        mock_qdrant = MagicMock()

        with self.assertRaises(RuntimeError) as ctx:
            answer_question("Question?", client=mock_qdrant)
        self.assertIn("Qdrant connection failure", str(ctx.exception))

    @patch("src.rag.search")
    @patch("src.rag.generate_answer")
    def test_generation_failure_propagates(self, mock_gen, mock_search):
        mock_search.return_value = [{"score": 0.5, "file_name": "f.pdf", "page_number": 1, "chunk_id": "c1", "text": "t"}]
        mock_gen.side_effect = RuntimeError("Gemini rate limit exceeded")
        mock_qdrant = MagicMock()

        with self.assertRaises(RuntimeError) as ctx:
            answer_question("Question?", client=mock_qdrant)
        self.assertIn("Gemini rate limit exceeded", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
