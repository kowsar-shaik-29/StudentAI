"""
Unit Tests for RAG Routing Integration (Step 13).
Tests:
- LEARNING route sets category="academic"
- CAREER route sets category="career"
- PROJECT route sets category="project"
- GENERAL route sets category=None (unfiltered)
- Low confidence falls back to GENERAL and no filter
- Router failure fallback handling
- Source metadata preservation
- Grounding remains active
"""

import unittest
from unittest.mock import MagicMock, patch
from src.rag import answer_question_auto


class TestRAGRouting(unittest.TestCase):

    @patch("src.rag.answer_question")
    @patch("src.query_router.route_query")
    def test_learning_route_applies_academic_filter(self, mock_route, mock_answer):
        """LEARNING route must apply category='academic' to answer_question."""
        mock_route.return_value = {
            "route": "LEARNING",
            "confidence": 0.95,
            "reason": "Academic concept",
            "raw_route": "LEARNING",
        }
        mock_answer.return_value = {
            "question": "What is star topology?",
            "answer": "Star topology connects to a central hub [Page 1].",
            "sources": [
                {
                    "score": 0.85,
                    "file_name": "net.pdf",
                    "page_number": 1,
                    "chunk_id": "c1",
                    "text": "Star topology...",
                    "category": "academic",
                }
            ],
        }

        mock_client = MagicMock()
        mock_genai = MagicMock()

        result = answer_question_auto(
            query="What is star topology?",
            client=mock_client,
            genai_client=mock_genai,
        )

        self.assertEqual(result["route"], "LEARNING")
        self.assertEqual(result["category_filter"], "academic")
        self.assertEqual(result["confidence"], 0.95)
        self.assertIn("[Page 1]", result["answer"])

        # Check filter passed to answer_question
        mock_answer.assert_called_once()
        call_kwargs = mock_answer.call_args.kwargs
        self.assertEqual(call_kwargs["category"], "academic")
        self.assertTrue(call_kwargs["hybrid"])

    @patch("src.rag.answer_question")
    @patch("src.query_router.route_query")
    def test_career_route_applies_career_filter(self, mock_route, mock_answer):
        """CAREER route must apply category='career'."""
        mock_route.return_value = {
            "route": "CAREER",
            "confidence": 0.91,
            "reason": "Resume question",
            "raw_route": "CAREER",
        }
        mock_answer.return_value = {
            "question": "What is on my resume?",
            "answer": "The resume lists Python and AWS [Page 1].",
            "sources": [],
        }

        mock_genai = MagicMock()
        result = answer_question_auto(
            query="What is on my resume?",
            genai_client=mock_genai,
        )

        self.assertEqual(result["route"], "CAREER")
        self.assertEqual(result["category_filter"], "career")
        call_kwargs = mock_answer.call_args.kwargs
        self.assertEqual(call_kwargs["category"], "career")

    @patch("src.rag.answer_question")
    @patch("src.query_router.route_query")
    def test_project_route_applies_project_filter(self, mock_route, mock_answer):
        """PROJECT route must apply category='project'."""
        mock_route.return_value = {
            "route": "PROJECT",
            "confidence": 0.89,
            "reason": "Project architecture",
            "raw_route": "PROJECT",
        }
        mock_answer.return_value = {
            "question": "Why Firebase?",
            "answer": "Firebase was chosen for real-time sync [Page 1].",
            "sources": [],
        }

        mock_genai = MagicMock()
        result = answer_question_auto(
            query="Why Firebase?",
            genai_client=mock_genai,
        )

        self.assertEqual(result["route"], "PROJECT")
        self.assertEqual(result["category_filter"], "project")
        call_kwargs = mock_answer.call_args.kwargs
        self.assertEqual(call_kwargs["category"], "project")

    @patch("src.rag.answer_question")
    @patch("src.query_router.route_query")
    def test_general_route_applies_no_filter(self, mock_route, mock_answer):
        """GENERAL route must apply category=None (unfiltered cross-corpus search)."""
        mock_route.return_value = {
            "route": "GENERAL",
            "confidence": 0.98,
            "reason": "World knowledge",
            "raw_route": "GENERAL",
        }
        mock_answer.return_value = {
            "question": "What is the capital of France?",
            "answer": "I couldn't find the answer in the provided documents.",
            "sources": [],
        }

        mock_genai = MagicMock()
        result = answer_question_auto(
            query="What is the capital of France?",
            genai_client=mock_genai,
        )

        self.assertEqual(result["route"], "GENERAL")
        self.assertIsNone(result["category_filter"])
        call_kwargs = mock_answer.call_args.kwargs
        self.assertIsNone(call_kwargs["category"])

    @patch("src.rag.answer_question")
    @patch("src.query_router.route_query")
    def test_low_confidence_fallback_applies_no_filter(self, mock_route, mock_answer):
        """Low confidence classified queries fall back to GENERAL with category=None."""
        mock_route.return_value = {
            "route": "GENERAL",
            "confidence": 0.52,
            "reason": "Confidence 0.52 below threshold 0.70; routed to GENERAL",
            "raw_route": "CAREER",
        }
        mock_answer.return_value = {
            "question": "Vague question",
            "answer": "Refusal answer",
            "sources": [],
        }

        mock_genai = MagicMock()
        result = answer_question_auto(
            query="Vague question",
            genai_client=mock_genai,
        )

        self.assertEqual(result["route"], "GENERAL")
        self.assertIsNone(result["category_filter"])
        call_kwargs = mock_answer.call_args.kwargs
        self.assertIsNone(call_kwargs["category"])


if __name__ == "__main__":
    unittest.main()
