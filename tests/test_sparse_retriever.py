"""
Unit Tests for SparseRetriever (Step 10).
Tests indexing, BM25 exact keyword matching, multi-document search,
empty query, invalid top_k, metadata preservation, and JSON persistence.
"""

import unittest
import shutil
from pathlib import Path
from src.sparse_retriever import SparseRetriever, tokenize


class TestSparseRetriever(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_sparse_index")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.retriever = SparseRetriever(index_dir=self.test_dir)

        self.sample_chunks = [
            {
                "chunk_id": "doc1_p1_c0",
                "text": "Star topology connects all network nodes to a single central hub.",
                "page_number": 1,
                "total_pages": 5,
                "file_name": "networking.pdf",
                "chunk_index": 0,
                "char_count": 67,
            },
            {
                "chunk_id": "doc1_p1_c1",
                "text": "If the central device in a star topology fails, the whole network fails.",
                "page_number": 1,
                "total_pages": 5,
                "file_name": "networking.pdf",
                "chunk_index": 1,
                "char_count": 72,
            },
            {
                "chunk_id": "doc2_p1_c0",
                "text": "A primary key uniquely identifies rows in a relational database table.",
                "page_number": 1,
                "total_pages": 2,
                "file_name": "database.pdf",
                "chunk_index": 0,
                "char_count": 70,
            },
            {
                "chunk_id": "doc2_p2_c0",
                "text": "ACID properties guarantee Atomicity, Consistency, Isolation, and Durability.",
                "page_number": 2,
                "total_pages": 2,
                "file_name": "database.pdf",
                "chunk_index": 0,
                "char_count": 76,
            },
        ]

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_tokenize(self):
        tokens = tokenize("What is ACID? ACID guarantees atomicity!")
        self.assertEqual(tokens, ["what", "is", "acid", "acid", "guarantees", "atomicity"])

    def test_indexing_and_statistics(self):
        indexed_count = self.retriever.index_chunks(self.sample_chunks)
        self.assertEqual(indexed_count, 4)
        self.assertEqual(self.retriever.total_docs, 4)
        self.assertIn("acid", self.retriever.doc_frequencies)
        self.assertEqual(self.retriever.doc_frequencies["acid"], 1)

    def test_exact_keyword_matching_and_ranking(self):
        self.retriever.index_chunks(self.sample_chunks)

        # Query targeting ACID chunk
        results = self.retriever.search("ACID", top_k=2)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["chunk_id"], "doc2_p2_c0")
        self.assertGreater(results[0]["sparse_score"], 0.0)

        # Query targeting primary key
        pk_results = self.retriever.search("primary key", top_k=2)
        self.assertEqual(pk_results[0]["chunk_id"], "doc2_p1_c0")

    def test_multi_document_search(self):
        self.retriever.index_chunks(self.sample_chunks)

        # Both documents should be represented
        net_res = self.retriever.search("star topology", top_k=2)
        self.assertEqual(net_res[0]["file_name"], "networking.pdf")

        db_res = self.retriever.search("relational database", top_k=2)
        self.assertEqual(db_res[0]["file_name"], "database.pdf")

    def test_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.retriever.search("")
        with self.assertRaises(ValueError):
            self.retriever.search("   ")

    def test_invalid_top_k_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.retriever.search("topology", top_k=0)
        with self.assertRaises(ValueError):
            self.retriever.search("topology", top_k=-5)

    def test_metadata_preservation(self):
        self.retriever.index_chunks(self.sample_chunks)
        results = self.retriever.search("atomicity", top_k=1)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["chunk_id"], "doc2_p2_c0")
        self.assertEqual(r["page_number"], 2)
        self.assertEqual(r["total_pages"], 2)
        self.assertEqual(r["file_name"], "database.pdf")
        self.assertEqual(r["chunk_index"], 0)
        self.assertEqual(r["char_count"], 76)
        self.assertNotIn("vector", r)

    def test_persistence_save_and_reload(self):
        self.retriever.index_chunks(self.sample_chunks, auto_save=True)

        # Create new retriever pointing to same test_dir
        new_retriever = SparseRetriever(index_dir=self.test_dir)
        self.assertEqual(new_retriever.total_docs, 4)

        results = new_retriever.search("ACID", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "doc2_p2_c0")


if __name__ == "__main__":
    unittest.main()
