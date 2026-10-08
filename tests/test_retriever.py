"""
Unit tests for semantic retriever module (src.retriever).
Uses isolated temporary Qdrant storage and mock query embeddings to avoid external API calls.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.vector_store import initialize_vector_store, ensure_collection, upsert_chunks
from src.retriever import search


class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary directory for local Qdrant persistence
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)
        self.client = initialize_vector_store(storage_path=self.storage_path)
        self.collection_name = "test_retrieval_collection"
        self.vector_size = 768

        ensure_collection(self.client, collection_name=self.collection_name, vector_size=self.vector_size)

        # Seed 3 points with distinct vectors:
        # Point A: aligned with [1.0, 0, 0, ...]
        # Point B: aligned with [0.5, 0.5, 0, ...]
        # Point C: orthogonal [0, 1.0, 0, ...]
        vec_a = [1.0] + [0.0] * (self.vector_size - 1)
        vec_b = [0.5, 0.5] + [0.0] * (self.vector_size - 2)
        vec_c = [0.0, 1.0] + [0.0] * (self.vector_size - 2)

        chunks = [
            {
                "chunk_id": "doc_p1_c0",
                "text": "Star topology connects all devices to a central hub.",
                "page_number": 1,
                "total_pages": 3,
                "file_name": "network.pdf",
                "file_path": "/network.pdf",
                "chunk_index": 0,
                "char_count": 52,
                "embedding": vec_a,
            },
            {
                "chunk_id": "doc_p2_c0",
                "text": "Bus topology connects devices along a single backbone cable.",
                "page_number": 2,
                "total_pages": 3,
                "file_name": "network.pdf",
                "file_path": "/network.pdf",
                "chunk_index": 0,
                "char_count": 60,
                "embedding": vec_b,
            },
            {
                "chunk_id": "doc_p3_c0",
                "text": "Ring topology connects devices in a circular loop.",
                "page_number": 3,
                "total_pages": 3,
                "file_name": "network.pdf",
                "file_path": "/network.pdf",
                "chunk_index": 0,
                "char_count": 50,
                "embedding": vec_c,
            },
        ]
        upsert_chunks(self.client, chunks, collection_name=self.collection_name, vector_size=self.vector_size)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_search_ordering_and_metadata(self):
        # Query vector perfectly aligned with vec_a
        query_vec = [1.0] + [0.0] * (self.vector_size - 1)

        results = search(
            query="What is star topology?",
            top_k=3,
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_vec,
        )

        self.assertEqual(len(results), 3)

        # Result 1 should have highest score and correspond to doc_p1_c0
        self.assertEqual(results[0]["chunk_id"], "doc_p1_c0")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=4)
        self.assertEqual(results[0]["page_number"], 1)
        self.assertEqual(results[0]["file_name"], "network.pdf")
        self.assertIn("Star topology", results[0]["text"])

        # Result 2 score should be less than Result 1 and greater than Result 3
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertGreater(results[1]["score"], results[2]["score"])

    def test_top_k_limiting(self):
        query_vec = [1.0] + [0.0] * (self.vector_size - 1)
        results = search(
            query="test",
            top_k=2,
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_vec,
        )
        self.assertEqual(len(results), 2)

    def test_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            search("", top_k=3, client=self.client, collection_name=self.collection_name)
        with self.assertRaises(ValueError):
            search("   ", top_k=3, client=self.client, collection_name=self.collection_name)

    def test_invalid_top_k_raises_value_error(self):
        with self.assertRaises(ValueError):
            search("query", top_k=0, client=self.client, collection_name=self.collection_name)
        with self.assertRaises(ValueError):
            search("query", top_k=-5, client=self.client, collection_name=self.collection_name)

    def test_missing_collection_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            search("query", top_k=3, client=self.client, collection_name="nonexistent_collection")
        self.assertIn("does not exist in Qdrant", str(ctx.exception))

    def test_dimension_mismatch_raises_value_error(self):
        # Pass vector with 512 dims when 768 is expected
        bad_query_vec = [0.1] * 512
        with self.assertRaises(ValueError) as ctx:
            search(
                "query",
                top_k=3,
                client=self.client,
                collection_name=self.collection_name,
                query_vector=bad_query_vec,
            )
        self.assertIn("dimension mismatch", str(ctx.exception))

    @patch("src.retriever.embed_query")
    def test_embed_query_invoked_with_retrieval_query(self, mock_embed_query):
        mock_embed_query.return_value = [0.1] * self.vector_size

        _ = search(
            "What is a network?",
            top_k=2,
            client=self.client,
            collection_name=self.collection_name,
        )
        mock_embed_query.assert_called_once_with("What is a network?")


if __name__ == "__main__":
    unittest.main()
