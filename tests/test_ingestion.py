"""
Unit tests for multi-document ingestion module (src.ingestion).
Uses mocked embeddings and temporary Qdrant storage to test ingestion workflows offline.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pymupdf

from src.vector_store import initialize_vector_store, count_vectors
from src.retriever import search
from src.ingestion import ingest_document, ingest_directory


def _create_sample_pdf(file_path: Path, text: str):
    """Helper to create a simple single-page PDF with text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    doc.save(str(file_path))
    doc.close()


class TestIngestion(unittest.TestCase):
    def setUp(self):
        # Isolated temporary directories
        self.temp_qdrant_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_qdrant_dir.name)
        self.client = initialize_vector_store(storage_path=self.storage_path)
        self.collection_name = "test_ingestion_coll"

        self.temp_docs_dir = tempfile.TemporaryDirectory()
        self.docs_path = Path(self.temp_docs_dir.name)

        # Create two sample PDFs
        self.doc1_path = self.docs_path / "networking_guide.pdf"
        _create_sample_pdf(self.doc1_path, "Star topology connects all nodes to a central hub.")

        self.doc2_path = self.docs_path / "database_guide.pdf"
        _create_sample_pdf(self.doc2_path, "SQL relational databases use primary keys to identify rows.")

    def tearDown(self):
        self.client.close()
        self.temp_qdrant_dir.cleanup()
        self.temp_docs_dir.cleanup()

    @patch("src.ingestion.embed_chunks")
    def test_single_document_ingestion(self, mock_embed_chunks):
        # Mock embeddings: return copy of chunks with 768-dim vector attached
        def fake_embed(chunks, **kwargs):
            return [dict(c, embedding=[0.05] * 768) for c in chunks]

        mock_embed_chunks.side_effect = fake_embed

        stats = ingest_document(
            file_path=self.doc1_path,
            client=self.client,
            collection_name=self.collection_name,
        )

        self.assertEqual(stats["file_name"], "networking_guide.pdf")
        self.assertEqual(stats["pages"], 1)
        self.assertEqual(stats["chunks"], 1)
        self.assertEqual(stats["vectors_stored"], 1)
        self.assertEqual(count_vectors(self.client, collection_name=self.collection_name), 1)

    @patch("src.ingestion.embed_chunks")
    def test_idempotent_ingestion_no_duplicates(self, mock_embed_chunks):
        mock_embed_chunks.side_effect = lambda chunks, **kwargs: [
            dict(c, embedding=[0.05] * 768) for c in chunks
        ]

        # Ingest doc 1 twice
        ingest_document(self.doc1_path, client=self.client, collection_name=self.collection_name)
        self.assertEqual(count_vectors(self.client, collection_name=self.collection_name), 1)

        ingest_document(self.doc1_path, client=self.client, collection_name=self.collection_name)
        # Vector count must still be 1
        self.assertEqual(count_vectors(self.client, collection_name=self.collection_name), 1)

    @patch("src.ingestion.embed_chunks")
    def test_multi_document_isolation_and_retrieval(self, mock_embed_chunks):
        # Distinct vectors for networking vs database
        def fake_embed(chunks, **kwargs):
            results = []
            for c in chunks:
                if "networking" in c["file_name"]:
                    vec = [1.0] + [0.0] * 767
                else:
                    vec = [0.0, 1.0] + [0.0] * 766
                results.append(dict(c, embedding=vec))
            return results

        mock_embed_chunks.side_effect = fake_embed

        # Ingest both documents into the same collection
        stats1 = ingest_document(self.doc1_path, client=self.client, collection_name=self.collection_name)
        stats2 = ingest_document(self.doc2_path, client=self.client, collection_name=self.collection_name)

        self.assertEqual(count_vectors(self.client, collection_name=self.collection_name), 2)

        # Retrieval using networking-aligned vector
        res_net = search(
            query="networking question",
            top_k=1,
            client=self.client,
            collection_name=self.collection_name,
            query_vector=[1.0] + [0.0] * 767,
        )
        self.assertEqual(res_net[0]["file_name"], "networking_guide.pdf")

        # Retrieval using database-aligned vector
        res_db = search(
            query="database question",
            top_k=1,
            client=self.client,
            collection_name=self.collection_name,
            query_vector=[0.0, 1.0] + [0.0] * 766,
        )
        self.assertEqual(res_db[0]["file_name"], "database_guide.pdf")

    @patch("src.ingestion.embed_chunks")
    def test_directory_ingestion_and_non_pdf_handling(self, mock_embed_chunks):
        mock_embed_chunks.side_effect = lambda chunks, **kwargs: [
            dict(c, embedding=[0.05] * 768) for c in chunks
        ]

        # Add a non-PDF file to docs_path
        (self.docs_path / "readme.txt").write_text("Not a PDF file")

        summary = ingest_directory(
            directory_path=self.docs_path,
            client=self.client,
            collection_name=self.collection_name,
        )

        # Only the 2 PDFs should be ingested
        self.assertEqual(summary["total_files_found"], 2)
        self.assertEqual(summary["successful_files"], 2)
        self.assertEqual(summary["total_vectors_stored"], 2)

    def test_invalid_directory_raises_not_found(self):
        with self.assertRaises(FileNotFoundError):
            ingest_directory(self.docs_path / "nonexistent_dir", client=self.client)

    def test_empty_directory_raises_value_error(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(ValueError) as ctx:
                ingest_directory(empty_dir, client=self.client)
            self.assertIn("No PDF files found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
