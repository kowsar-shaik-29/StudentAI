"""
Unit Tests for HybridRetriever (Step 10).
Tests Reciprocal Rank Fusion (RRF), deterministic RRF scoring formula,
handling disjoint lists, duplicate candidates, top-k truncation, and empty queries.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.hybrid_retriever import reciprocal_rank_fusion, hybrid_search


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        self.chunk_a = {
            "chunk_id": "chunk_A",
            "score": 0.90,
            "text": "Text for Chunk A",
            "page_number": 1,
            "total_pages": 3,
            "file_name": "doc1.pdf",
            "chunk_index": 0,
            "char_count": 16,
        }
        self.chunk_b = {
            "chunk_id": "chunk_B",
            "score": 0.80,
            "text": "Text for Chunk B",
            "page_number": 1,
            "total_pages": 3,
            "file_name": "doc1.pdf",
            "chunk_index": 1,
            "char_count": 16,
        }
        self.chunk_c = {
            "chunk_id": "chunk_C",
            "score": 0.70,
            "sparse_score": 5.0,
            "text": "Text for Chunk C",
            "page_number": 2,
            "total_pages": 3,
            "file_name": "doc1.pdf",
            "chunk_index": 2,
            "char_count": 16,
        }
        self.chunk_d = {
            "chunk_id": "chunk_D",
            "sparse_score": 3.0,
            "text": "Text for Chunk D",
            "page_number": 3,
            "total_pages": 3,
            "file_name": "doc2.pdf",
            "chunk_index": 0,
            "char_count": 16,
        }

    def test_deterministic_rrf_scoring(self):
        """
        Specific Requirement from Step 10:
        Dense ranking:
          A -> rank 1
          B -> rank 2
          C -> rank 3
        Sparse ranking:
          C -> rank 1
          A -> rank 2
          D -> rank 3

        Verify fused ranking is calculated correctly according to RRF with k=60.
        Calculations:
          RRF(A) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.0163934 + 0.0161290 = 0.0325224
          RRF(B) = 1/(60+2)            = 1/62        ≈ 0.0161290
          RRF(C) = 1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.0158730 + 0.0163934 = 0.0322664
          RRF(D) = 1/(60+3)            = 1/63        ≈ 0.0158730

        Expected Ranking:
          1. A (0.032522)
          2. C (0.032266)
          3. B (0.016129)
          4. D (0.015873)
        """
        dense_list = [self.chunk_a, self.chunk_b, self.chunk_c]
        sparse_list = [
            dict(self.chunk_c, sparse_score=8.0),
            dict(self.chunk_a, sparse_score=6.0),
            dict(self.chunk_d, sparse_score=4.0),
        ]

        fused = reciprocal_rank_fusion(dense_list, sparse_list, top_k=4, rrf_k=60)

        self.assertEqual(len(fused), 4)
        self.assertEqual(fused[0]["chunk_id"], "chunk_A")
        self.assertEqual(fused[1]["chunk_id"], "chunk_C")
        self.assertEqual(fused[2]["chunk_id"], "chunk_B")
        self.assertEqual(fused[3]["chunk_id"], "chunk_D")

        # Verify exact math
        expected_score_a = (1.0 / 61) + (1.0 / 62)
        expected_score_c = (1.0 / 63) + (1.0 / 61)
        self.assertAlmostEqual(fused[0]["fused_score"], expected_score_a, places=6)
        self.assertAlmostEqual(fused[1]["fused_score"], expected_score_c, places=6)

    def test_dense_only_candidates(self):
        dense_list = [self.chunk_a, self.chunk_b]
        sparse_list = []
        fused = reciprocal_rank_fusion(dense_list, sparse_list, top_k=2, rrf_k=60)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["chunk_id"], "chunk_A")
        self.assertEqual(fused[1]["chunk_id"], "chunk_B")

    def test_sparse_only_candidates(self):
        dense_list = []
        sparse_list = [self.chunk_c, self.chunk_d]
        fused = reciprocal_rank_fusion(dense_list, sparse_list, top_k=2, rrf_k=60)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["chunk_id"], "chunk_C")
        self.assertEqual(fused[1]["chunk_id"], "chunk_D")

    def test_metadata_preservation(self):
        dense_list = [self.chunk_a]
        sparse_list = [dict(self.chunk_a, sparse_score=2.5)]
        fused = reciprocal_rank_fusion(dense_list, sparse_list, top_k=1, rrf_k=60)
        item = fused[0]
        self.assertEqual(item["chunk_id"], "chunk_A")
        self.assertEqual(item["file_name"], "doc1.pdf")
        self.assertEqual(item["page_number"], 1)
        self.assertEqual(item["total_pages"], 3)
        self.assertEqual(item["char_count"], 16)
        self.assertAlmostEqual(item["dense_score"], 0.90)
        self.assertAlmostEqual(item["sparse_score"], 2.5)
        self.assertIn("fused_score", item)
        self.assertNotIn("vector", item)

    def test_top_k_truncation(self):
        dense_list = [self.chunk_a, self.chunk_b, self.chunk_c]
        sparse_list = [self.chunk_d]
        fused = reciprocal_rank_fusion(dense_list, sparse_list, top_k=2, rrf_k=60)
        self.assertEqual(len(fused), 2)

    def test_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            hybrid_search("")
        with self.assertRaises(ValueError):
            hybrid_search("   ")

    def test_invalid_parameters_raise_value_error(self):
        with self.assertRaises(ValueError):
            hybrid_search("valid", top_k=0)
        with self.assertRaises(ValueError):
            hybrid_search("valid", dense_candidates=0)
        with self.assertRaises(ValueError):
            hybrid_search("valid", sparse_candidates=0)


if __name__ == "__main__":
    unittest.main()
