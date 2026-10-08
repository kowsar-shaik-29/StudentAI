"""
Unit Tests for Reranker Module (Step 9).
Validates parameter checks, candidate ordering, metadata preservation, top-k truncation,
and graceful fallback handling without burning API credits.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.reranker import rerank


class TestReranker(unittest.TestCase):
    def setUp(self):
        self.sample_candidates = [
            {
                "score": 0.85,
                "chunk_id": "doc1_p1_c0",
                "text": "General introduction to networking.",
                "page_number": 1,
                "total_pages": 5,
                "file_name": "doc1.pdf",
                "chunk_index": 0,
                "char_count": 35,
            },
            {
                "score": 0.75,
                "chunk_id": "doc1_p1_c1",
                "text": "If the central device in a star topology fails, the whole network fails.",
                "page_number": 1,
                "total_pages": 5,
                "file_name": "doc1.pdf",
                "chunk_index": 1,
                "char_count": 72,
            },
            {
                "score": 0.70,
                "chunk_id": "doc2_p1_c0",
                "text": "A primary key uniquely identifies records in a database table.",
                "page_number": 1,
                "total_pages": 2,
                "file_name": "doc2.pdf",
                "chunk_index": 0,
                "char_count": 62,
            },
        ]

    def test_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            rerank("", self.sample_candidates)

        with self.assertRaises(ValueError):
            rerank("   ", self.sample_candidates)

    def test_invalid_top_k_raises_value_error(self):
        with self.assertRaises(ValueError):
            rerank("Valid query", self.sample_candidates, top_k=0)

        with self.assertRaises(ValueError):
            rerank("Valid query", self.sample_candidates, top_k=-1)

    def test_empty_candidate_list_returns_empty(self):
        result = rerank("What is star topology?", [], top_k=5)
        self.assertEqual(result, [])

    def test_candidate_reordering_and_score_preservation(self):
        """Mock LLM response to promote the second candidate to top rank."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Candidate 1 (index 1) gets 0.95, Candidate 0 (index 0) gets 0.40, Candidate 2 (index 2) gets 0.05
        mock_response.text = (
            '[{"index": 0, "relevance_score": 0.40}, '
            '{"index": 1, "relevance_score": 0.95}, '
            '{"index": 2, "relevance_score": 0.05}]'
        )
        mock_client.models.generate_content.return_value = mock_response

        reranked = rerank(
            query="What happens if the central device fails?",
            results=self.sample_candidates,
            top_k=3,
            client=mock_client,
        )

        self.assertEqual(len(reranked), 3)

        # Candidate 1 should now be rank 1
        top_result = reranked[0]
        self.assertEqual(top_result["chunk_id"], "doc1_p1_c1")
        self.assertAlmostEqual(top_result["rerank_score"], 0.95)
        self.assertAlmostEqual(top_result["score"], 0.75)  # Original score preserved!

        # Candidate 0 should be rank 2
        second_result = reranked[1]
        self.assertEqual(second_result["chunk_id"], "doc1_p1_c0")
        self.assertAlmostEqual(second_result["rerank_score"], 0.40)
        self.assertAlmostEqual(second_result["score"], 0.85)

        # Candidate 2 should be rank 3
        third_result = reranked[2]
        self.assertEqual(third_result["chunk_id"], "doc2_p1_c0")
        self.assertAlmostEqual(third_result["rerank_score"], 0.05)

    def test_top_k_truncation(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            '[{"index": 0, "relevance_score": 0.5}, '
            '{"index": 1, "relevance_score": 0.9}, '
            '{"index": 2, "relevance_score": 0.1}]'
        )
        mock_client.models.generate_content.return_value = mock_response

        reranked = rerank(
            query="test query",
            results=self.sample_candidates,
            top_k=2,
            client=mock_client,
        )

        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["chunk_id"], "doc1_p1_c1")
        self.assertEqual(reranked[1]["chunk_id"], "doc1_p1_c0")

    def test_metadata_preservation_and_vector_exclusion(self):
        candidates_with_vectors = [
            dict(c, vector=[0.1] * 768, embedding=[0.2] * 768)
            for c in self.sample_candidates
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '[{"index": 0, "relevance_score": 0.8}]'
        mock_client.models.generate_content.return_value = mock_response

        reranked = rerank(
            query="test metadata",
            results=candidates_with_vectors,
            top_k=1,
            client=mock_client,
        )

        item = reranked[0]
        # Metadata checks
        self.assertEqual(item["chunk_id"], "doc1_p1_c0")
        self.assertEqual(item["page_number"], 1)
        self.assertEqual(item["total_pages"], 5)
        self.assertEqual(item["file_name"], "doc1.pdf")
        self.assertEqual(item["chunk_index"], 0)
        self.assertEqual(item["char_count"], 35)
        # Vector exclusion checks
        self.assertNotIn("vector", item)
        self.assertNotIn("embedding", item)

    def test_api_failure_graceful_fallback(self):
        """When LLM API fails, rerank should fall back to original Qdrant scores without crashing."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API connection timeout")

        reranked = rerank(
            query="test fallback",
            results=self.sample_candidates,
            top_k=2,
            client=mock_client,
        )

        self.assertEqual(len(reranked), 2)
        # Fallback preserves original order
        self.assertEqual(reranked[0]["chunk_id"], "doc1_p1_c0")
        self.assertEqual(reranked[1]["chunk_id"], "doc1_p1_c1")
        # Sets rerank_score to original score
        self.assertEqual(reranked[0]["rerank_score"], 0.85)


if __name__ == "__main__":
    unittest.main()
