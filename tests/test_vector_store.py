"""
Unit tests for Qdrant vector store module (src.vector_store).
Operates strictly in an isolated temporary directory with zero external API calls.
"""

import unittest
import tempfile
import math
from pathlib import Path
from src.vector_store import (
    initialize_vector_store,
    ensure_collection,
    count_vectors,
    get_collection_info,
    upsert_chunks,
    generate_point_id,
)


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary directory for local Qdrant persistence
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)
        self.client = initialize_vector_store(storage_path=self.storage_path)
        self.collection_name = "test_collection"
        self.vector_size = 768

        # Create valid synthetic embedded chunk
        self.sample_chunks = [
            {
                "chunk_id": "test_doc.pdf_p1_c0",
                "text": "First test snippet text",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "test_doc.pdf",
                "file_path": "/path/test_doc.pdf",
                "chunk_index": 0,
                "char_count": 23,
                "embedding": [0.01] * self.vector_size,
            },
            {
                "chunk_id": "test_doc.pdf_p1_c1",
                "text": "Second test snippet text",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "test_doc.pdf",
                "file_path": "/path/test_doc.pdf",
                "chunk_index": 1,
                "char_count": 24,
                "embedding": [0.02] * self.vector_size,
            },
        ]

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_collection_creation_and_info(self):
        ensure_collection(
            self.client,
            collection_name=self.collection_name,
            vector_size=self.vector_size,
        )
        info = get_collection_info(self.client, collection_name=self.collection_name)
        self.assertTrue(info["exists"])
        self.assertEqual(info["points_count"], 0)
        self.assertEqual(info["vector_size"], self.vector_size)
        self.assertEqual(info["distance"].upper(), "COSINE")

    def test_deterministic_point_ids(self):
        id1 = generate_point_id("chunk_abc_p1_c0")
        id2 = generate_point_id("chunk_abc_p1_c0")
        id3 = generate_point_id("chunk_abc_p1_c1")

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_upsert_chunks_and_payload_preservation(self):
        count = upsert_chunks(
            self.client,
            self.sample_chunks,
            collection_name=self.collection_name,
            vector_size=self.vector_size,
        )
        self.assertEqual(count, 2)
        self.assertEqual(count_vectors(self.client, collection_name=self.collection_name), 2)

        # Retrieve and verify payload
        first_point_id = generate_point_id(self.sample_chunks[0]["chunk_id"])
        retrieved = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[first_point_id],
            with_payload=True,
            with_vectors=True,
        )
        self.assertEqual(len(retrieved), 1)
        point = retrieved[0]
        self.assertEqual(point.payload["chunk_id"], "test_doc.pdf_p1_c0")
        self.assertEqual(point.payload["text"], "First test snippet text")
        self.assertEqual(point.payload["page_number"], 1)
        self.assertEqual(len(point.vector), self.vector_size)

    def test_idempotent_ingestion_does_not_duplicate(self):
        # Ingest once
        upsert_chunks(
            self.client,
            self.sample_chunks,
            collection_name=self.collection_name,
            vector_size=self.vector_size,
        )
        self.assertEqual(count_vectors(self.client, collection_name=self.collection_name), 2)

        # Ingest the exact same chunks a second time
        upsert_chunks(
            self.client,
            self.sample_chunks,
            collection_name=self.collection_name,
            vector_size=self.vector_size,
        )
        # Vector count must still be 2, not 4
        self.assertEqual(count_vectors(self.client, collection_name=self.collection_name), 2)

    def test_invalid_vector_dimension_rejected(self):
        invalid_chunk = [
            dict(self.sample_chunks[0], embedding=[0.1] * 512)
        ]
        with self.assertRaises(ValueError) as ctx:
            upsert_chunks(
                self.client,
                invalid_chunk,
                collection_name=self.collection_name,
                vector_size=self.vector_size,
            )
        self.assertIn("Vector dimension mismatch", str(ctx.exception))

    def test_non_finite_vector_rejected(self):
        bad_emb = [0.1] * (self.vector_size - 1) + [float("nan")]
        invalid_chunk = [
            dict(self.sample_chunks[0], embedding=bad_emb)
        ]
        with self.assertRaises(ValueError) as ctx:
            upsert_chunks(
                self.client,
                invalid_chunk,
                collection_name=self.collection_name,
                vector_size=self.vector_size,
            )
        self.assertIn("contains non-finite values", str(ctx.exception))

    def test_missing_metadata_rejected(self):
        invalid_chunk = [
            {
                "chunk_id": "test_id",
                "embedding": [0.1] * self.vector_size,
                # Missing text, page_number, etc.
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            upsert_chunks(
                self.client,
                invalid_chunk,
                collection_name=self.collection_name,
                vector_size=self.vector_size,
            )
        self.assertIn("missing required metadata field", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
