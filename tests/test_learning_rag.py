"""
Unit Tests for Learning RAG Pipeline (Step 14).
Tests:
- Academic filter applied (category="academic")
- Intent routing automatic and override
- Context assembly and intent-tailored generation
- Source metadata preservation
- Quiz structured output format (questions list with difficulties)
- Strict factual grounding and refusal on missing concepts
- Error handling
"""

import unittest
from unittest.mock import MagicMock, patch
from src.learning_rag import (
    answer_learning_question,
    STANDARD_REFUSAL,
    _generate_learning_content,
)


class TestLearningRAG(unittest.TestCase):

    @patch("src.learning_rag.hybrid_search")
    @patch("src.learning_rag.route_learning_query")
    def test_explain_intent_and_academic_filter(self, mock_route, mock_hybrid):
        """EXPLAIN intent applies academic category filter and returns grounded text."""
        mock_route.return_value = {
            "intent": "EXPLAIN",
            "confidence": 0.95,
            "reason": "Concept explanation",
            "raw_intent": "EXPLAIN",
        }
        mock_hybrid.return_value = [
            {
                "chunk_id": "net_c1",
                "text": "Star topology connects to a central hub.",
                "page_number": 1,
                "total_pages": 5,
                "file_name": "net.pdf",
                "score": 0.88,
                "category": "academic",
                "subject": "computer_networks",
                "document_type": "notes",
            }
        ]

        mock_genai = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Star topology uses a central hub [Page 1]."
        mock_genai.models.generate_content.return_value = mock_resp

        mock_qdrant = MagicMock()

        result = answer_learning_question(
            query="Explain star topology",
            client=mock_qdrant,
            genai_client=mock_genai,
        )

        self.assertEqual(result["intent"], "EXPLAIN")
        self.assertEqual(result["confidence"], 0.95)
        self.assertIn("[Page 1]", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["category"], "academic")

        # Verify category='academic' passed to hybrid_search
        mock_hybrid.assert_called_once()
        call_kwargs = mock_hybrid.call_args.kwargs
        self.assertEqual(call_kwargs["category"], "academic")

    @patch("src.learning_rag.hybrid_search")
    def test_intent_override(self, mock_hybrid):
        """Explicit intent argument bypasses router."""
        mock_hybrid.return_value = [
            {
                "chunk_id": "c1",
                "text": "TCP is reliable, UDP is connectionless.",
                "page_number": 2,
                "file_name": "net.pdf",
                "score": 0.9,
                "category": "academic",
            }
        ]
        mock_genai = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "| Protocol | Type |\n|---|---|\n| TCP | Reliable |\n| UDP | Unreliable | [Page 2]"
        mock_genai.models.generate_content.return_value = mock_resp

        result = answer_learning_question(
            query="TCP vs UDP",
            intent="COMPARE",
            genai_client=mock_genai,
        )

        self.assertEqual(result["intent"], "COMPARE")
        self.assertEqual(result["confidence"], 1.0)
        self.assertIn("Protocol", result["answer"])
        self.assertIn("[Page 2]", result["answer"])

    @patch("src.learning_rag.hybrid_search")
    def test_quiz_structured_output(self, mock_hybrid):
        """QUIZ intent returns structured questions list without leaking answers."""
        mock_hybrid.return_value = [
            {
                "chunk_id": "c1",
                "text": "ACID properties: Atomicity, Consistency, Isolation, Durability.",
                "page_number": 3,
                "file_name": "db.pdf",
                "score": 0.85,
                "category": "academic",
            }
        ]
        mock_genai = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"questions": [{"question": "What does Atomicity ensure?", "difficulty": "easy"}, {"question": "Define Durability in ACID.", "difficulty": "medium"}]}'
        mock_genai.models.generate_content.return_value = mock_resp

        result = answer_learning_question(
            query="Quiz me on ACID",
            intent="QUIZ",
            genai_client=mock_genai,
        )

        self.assertEqual(result["intent"], "QUIZ")
        self.assertIn("questions", result)
        self.assertEqual(len(result["questions"]), 2)
        self.assertEqual(result["questions"][0]["difficulty"], "easy")
        self.assertEqual(result["questions"][0]["question"], "What does Atomicity ensure?")
        self.assertNotIn("answer", result)  # Quiz does not give away answers

    @patch("src.learning_rag.hybrid_search")
    def test_empty_context_refusal_and_grounding(self, mock_hybrid):
        """When hybrid retrieval finds zero relevant chunks, system returns standard refusal."""
        mock_hybrid.return_value = []
        mock_genai = MagicMock()

        result = answer_learning_question(
            query="Explain quantum computing",
            intent="EXPLAIN",
            genai_client=mock_genai,
        )

        self.assertEqual(result["answer"], STANDARD_REFUSAL)
        # LLM not called when context is empty
        mock_genai.models.generate_content.assert_not_called()

    @patch("src.learning_rag.hybrid_search")
    def test_empty_context_quiz_returns_empty_questions(self, mock_hybrid):
        """When context is empty for quiz, returns empty questions list."""
        mock_hybrid.return_value = []
        mock_genai = MagicMock()

        result = answer_learning_question(
            query="Quiz me on nonexistent topic",
            intent="QUIZ",
            genai_client=mock_genai,
        )

        self.assertEqual(result["intent"], "QUIZ")
        self.assertEqual(result["questions"], [])
        mock_genai.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
