"""
Unit Tests for Step 12: Metadata-Aware Knowledge.
Tests the Document Metadata Model, deterministic document_id generation,
metadata propagation across chunking and vector storage, and metadata filtering
across dense, sparse, and hybrid retrievers.
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from src.metadata import (
    DocumentMetadata,
    create_document_metadata,
    generate_document_id,
    VALID_CATEGORIES,
    VALID_DOCUMENT_TYPES,
)
from src.chunker import create_chunks
from src.vector_store import (
    initialize_vector_store,
    ensure_collection,
    upsert_chunks,
    count_vectors,
)
from src.sparse_retriever import SparseRetriever
from src.retriever import search as dense_search
from src.hybrid_retriever import hybrid_search


class TestMetadata(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.qdrant_path = Path(self.temp_dir) / "qdrant"
        self.client = initialize_vector_store(storage_path=self.qdrant_path)
        self.collection_name = "test_metadata_collection"
        ensure_collection(self.client, collection_name=self.collection_name, vector_size=768)

    def tearDown(self):
        self.client.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deterministic_document_id(self):
        """Verify that document IDs are deterministic and case-insensitive."""
        id1 = generate_document_id("Computer_Networking_Notes.pdf")
        id2 = generate_document_id("computer_networking_notes.pdf")
        id3 = generate_document_id("/some/path/computer_networking_notes.pdf")
        id_diff = generate_document_id("database_notes.pdf")

        self.assertEqual(id1, id2)
        self.assertEqual(id1, id3)
        self.assertNotEqual(id1, id_diff)
        self.assertTrue(len(id1) > 0)

    def test_document_metadata_model_defaults_and_normalization(self):
        """Verify default values and case normalization in DocumentMetadata."""
        meta = create_document_metadata(
            file_name="sample.pdf",
            category="ACADEMIC",
            subject="Computer Networks",
            document_type="NOTES",
        )

        self.assertEqual(meta.category, "academic")
        self.assertEqual(meta.subject, "computer_networks")
        self.assertEqual(meta.document_type, "notes")
        self.assertEqual(meta.file_name, "sample.pdf")
        self.assertTrue(meta.document_id)

        # Defaults test
        meta_default = create_document_metadata(file_name="unspecified.pdf")
        self.assertEqual(meta_default.category, "other")
        self.assertEqual(meta_default.subject, "general")
        self.assertEqual(meta_default.document_type, "other")

    def test_metadata_propagation_in_chunker(self):
        """Verify that pages with metadata transfer attributes into text chunks."""
        pages = [
            {
                "page_number": 1,
                "text": "Academic computer networks page 1 with substantial content text for chunking.",
                "char_count": 70,
                "has_text": True,
                "file_name": "net.pdf",
                "file_path": "data/net.pdf",
                "total_pages": 1,
                "document_id": "doc-net-123",
                "category": "academic",
                "subject": "computer_networks",
                "document_type": "notes",
            }
        ]

        chunks = create_chunks(pages, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 0)
        chunk = chunks[0]
        self.assertEqual(chunk["document_id"], "doc-net-123")
        self.assertEqual(chunk["category"], "academic")
        self.assertEqual(chunk["subject"], "computer_networks")
        self.assertEqual(chunk["document_type"], "notes")

    def test_qdrant_payload_and_dense_metadata_filtering(self):
        """Verify that Qdrant stores metadata and filters correctly by category and subject."""
        chunks = [
            {
                "chunk_id": "net_c0",
                "text": "TCP IP network protocols and subnetting.",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "net.pdf",
                "file_path": "data/net.pdf",
                "chunk_index": 0,
                "char_count": 40,
                "embedding": [0.1] * 768,
                "document_id": "net-id",
                "category": "academic",
                "subject": "computer_networks",
                "document_type": "notes",
            },
            {
                "chunk_id": "db_c0",
                "text": "Relational database indexing and SQL queries.",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "db.pdf",
                "file_path": "data/db.pdf",
                "chunk_index": 0,
                "char_count": 45,
                "embedding": [0.1] * 768,
                "document_id": "db-id",
                "category": "academic",
                "subject": "database",
                "document_type": "notes",
            },
            {
                "chunk_id": "resume_c0",
                "text": "Software engineer resume with Python and AWS experience.",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "resume.pdf",
                "file_path": "data/resume.pdf",
                "chunk_index": 0,
                "char_count": 55,
                "embedding": [0.1] * 768,
                "document_id": "resume-id",
                "category": "career",
                "subject": "career",
                "document_type": "resume",
            },
        ]

        upsert_chunks(self.client, chunks, collection_name=self.collection_name, vector_size=768)
        self.assertEqual(count_vectors(self.client, self.collection_name), 3)

        # 1. Unfiltered search: returns all 3
        res_all = dense_search(
            query="test",
            top_k=5,
            client=self.client,
            collection_name=self.collection_name,
            query_vector=[0.1] * 768,
        )
        self.assertEqual(len(res_all), 3)

        # 2. Filter by category='career': returns only resume
        res_career = dense_search(
            query="test",
            top_k=5,
            client=self.client,
            collection_name=self.collection_name,
            query_vector=[0.1] * 768,
            category="career",
        )
        self.assertEqual(len(res_career), 1)
        self.assertEqual(res_career[0]["chunk_id"], "resume_c0")
        self.assertEqual(res_career[0]["category"], "career")

        # 3. Filter by subject='database': returns only db chunk
        res_db = dense_search(
            query="test",
            top_k=5,
            client=self.client,
            collection_name=self.collection_name,
            query_vector=[0.1] * 768,
            subject="database",
        )
        self.assertEqual(len(res_db), 1)
        self.assertEqual(res_db[0]["chunk_id"], "db_c0")
        self.assertEqual(res_db[0]["subject"], "database")

        # 4. Filter by combined category='academic' and subject='computer_networks'
        res_net = dense_search(
            query="test",
            top_k=5,
            client=self.client,
            collection_name=self.collection_name,
            query_vector=[0.1] * 768,
            category="academic",
            subject="computer_networks",
        )
        self.assertEqual(len(res_net), 1)
        self.assertEqual(res_net[0]["chunk_id"], "net_c0")

    def test_sparse_retriever_metadata_filtering(self):
        """Verify that Sparse BM25 retriever preserves and filters by metadata."""
        sr = SparseRetriever(index_dir=Path(self.temp_dir) / "sparse_idx")
        chunks = [
            {
                "chunk_id": "c1",
                "text": "Python programming syntax functions classes.",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "python.pdf",
                "file_path": "data/python.pdf",
                "chunk_index": 0,
                "char_count": 44,
                "document_id": "py-id",
                "category": "programming",
                "subject": "python",
                "document_type": "notes",
            },
            {
                "chunk_id": "c2",
                "text": "Python interview questions and resume project experience.",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "resume.pdf",
                "file_path": "data/resume.pdf",
                "chunk_index": 0,
                "char_count": 56,
                "document_id": "res-id",
                "category": "career",
                "subject": "career",
                "document_type": "resume",
            },
        ]
        sr.index_chunks(chunks, auto_save=False)

        # Unfiltered search matches both on 'python'
        all_matches = sr.search("Python", top_k=5)
        self.assertEqual(len(all_matches), 2)

        # Filtered to career only matches c2
        career_matches = sr.search("Python", top_k=5, category="career")
        self.assertEqual(len(career_matches), 1)
        self.assertEqual(career_matches[0]["chunk_id"], "c2")

        # Filtered to programming only matches c1
        prog_matches = sr.search("Python", top_k=5, category="programming")
        self.assertEqual(len(prog_matches), 1)
        self.assertEqual(prog_matches[0]["chunk_id"], "c1")

    def test_hybrid_search_metadata_filtering(self):
        """Verify that hybrid search correctly filters candidates down to matched metadata."""
        sr = SparseRetriever(index_dir=Path(self.temp_dir) / "sparse_hybrid")
        chunks = [
            {
                "chunk_id": "h_net",
                "text": "Network subnetting protocols TCP and UDP.",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "net.pdf",
                "file_path": "data/net.pdf",
                "chunk_index": 0,
                "char_count": 40,
                "embedding": [0.2] * 768,
                "document_id": "net-id",
                "category": "academic",
                "subject": "computer_networks",
                "document_type": "notes",
            },
            {
                "chunk_id": "h_career",
                "text": "Network engineer resume and Cisco certifications.",
                "page_number": 1,
                "total_pages": 1,
                "file_name": "resume.pdf",
                "file_path": "data/resume.pdf",
                "chunk_index": 0,
                "char_count": 49,
                "embedding": [0.2] * 768,
                "document_id": "res-id",
                "category": "career",
                "subject": "career",
                "document_type": "resume",
            },
        ]

        upsert_chunks(self.client, chunks, collection_name=self.collection_name, vector_size=768)
        sr.index_chunks(chunks, auto_save=False)

        # Unfiltered hybrid search returns both
        unfiltered = hybrid_search(
            query="Network",
            top_k=5,
            client=self.client,
            sparse_retriever=sr,
            collection_name=self.collection_name,
            query_vector=[0.2] * 768,
        )
        self.assertEqual(len(unfiltered), 2)

        # Hybrid search filtered to category='career' returns only resume
        filtered = hybrid_search(
            query="Network",
            top_k=5,
            client=self.client,
            sparse_retriever=sr,
            collection_name=self.collection_name,
            query_vector=[0.2] * 768,
            category="career",
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["chunk_id"], "h_career")
        self.assertEqual(filtered[0]["category"], "career")


if __name__ == "__main__":
    unittest.main()
