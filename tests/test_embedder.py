"""
Unit tests for embedding module (src.embedder).
Uses mocking to test parsing, dimension validation, batching, and chunk enrichment.
"""

import unittest
from unittest.mock import MagicMock
from google.genai import types
from src.embedder import embed_documents, embed_query, embed_chunks, _sanitize_error


class TestEmbedder(unittest.TestCase):
    def test_sanitize_error_replaces_api_key(self):
        msg = "Error 403: Invalid API key"
        sanitized = _sanitize_error(msg)
        self.assertNotIn("AIzaSy", sanitized)

    def test_embed_documents_parsing(self):
        mock_client = MagicMock()
        mock_embedding1 = types.ContentEmbedding(values=[0.1] * 768)
        mock_embedding2 = types.ContentEmbedding(values=[0.2] * 768)

        mock_response = types.EmbedContentResponse(
            embeddings=[mock_embedding1, mock_embedding2]
        )
        mock_client.models.embed_content.return_value = mock_response

        texts = ["Text chunk 1", "Text chunk 2"]
        vectors = embed_documents(texts, client=mock_client, output_dim=768)

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 768)
        self.assertEqual(len(vectors[1]), 768)
        self.assertEqual(vectors[0][0], 0.1)
        self.assertEqual(vectors[1][0], 0.2)

    def test_dimension_mismatch_raises_value_error(self):
        mock_client = MagicMock()
        mock_embedding = types.ContentEmbedding(values=[0.1] * 512)
        mock_response = types.EmbedContentResponse(embeddings=[mock_embedding])
        mock_client.models.embed_content.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            embed_documents(["Test"], client=mock_client, output_dim=768)
        self.assertIn("Expected embedding dimension 768, but received 512", str(ctx.exception))

    def test_non_finite_values_raise_value_error(self):
        mock_client = MagicMock()
        values_with_nan = [0.1] * 767 + [float("nan")]
        mock_embedding = types.ContentEmbedding(values=values_with_nan)
        mock_response = types.EmbedContentResponse(embeddings=[mock_embedding])
        mock_client.models.embed_content.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            embed_documents(["Test"], client=mock_client, output_dim=768)
        self.assertIn("Embedding vector contains non-finite values", str(ctx.exception))

    def test_embed_chunks_metadata_preservation_and_mapping(self):
        # Sample chunks
        chunks = [
            {
                "chunk_id": "test.pdf_p1_c0",
                "text": "First chunk text",
                "page_number": 1,
                "total_pages": 2,
                "file_name": "test.pdf",
                "file_path": "/test.pdf",
                "chunk_index": 0,
                "char_count": 16,
            },
            {
                "chunk_id": "test.pdf_p2_c0",
                "text": "Second chunk text",
                "page_number": 2,
                "total_pages": 2,
                "file_name": "test.pdf",
                "file_path": "/test.pdf",
                "chunk_index": 0,
                "char_count": 17,
            },
        ]

        mock_client = MagicMock()
        mock_embedding1 = types.ContentEmbedding(values=[0.05] * 768)
        mock_embedding2 = types.ContentEmbedding(values=[0.08] * 768)
        mock_response = types.EmbedContentResponse(
            embeddings=[mock_embedding1, mock_embedding2]
        )
        mock_client.models.embed_content.return_value = mock_response

        enriched = embed_chunks(chunks, client=mock_client, output_dim=768)

        # 1. All chunks receive embeddings
        self.assertEqual(len(enriched), len(chunks))

        # 2. One embedding per chunk with correct dimension
        for i, item in enumerate(enriched):
            self.assertIn("embedding", item)
            self.assertEqual(len(item["embedding"]), 768)
            # Metadata preserved
            self.assertEqual(item["chunk_id"], chunks[i]["chunk_id"])
            self.assertEqual(item["text"], chunks[i]["text"])
            self.assertEqual(item["page_number"], chunks[i]["page_number"])
            self.assertEqual(item["total_pages"], chunks[i]["total_pages"])
            self.assertEqual(item["file_name"], chunks[i]["file_name"])

        # 3. Correct 1-to-1 mapping
        self.assertEqual(enriched[0]["embedding"][0], 0.05)
        self.assertEqual(enriched[1]["embedding"][0], 0.08)

    def test_embed_documents_batching(self):
        # 5 items with batch_size=2 should produce 3 API calls: 2 + 2 + 1
        mock_client = MagicMock()

        def side_effect(model, contents, config):
            count = len(contents)
            return types.EmbedContentResponse(
                embeddings=[types.ContentEmbedding(values=[0.1] * 768) for _ in range(count)]
            )

        mock_client.models.embed_content.side_effect = side_effect

        texts = [f"Text {i}" for i in range(5)]
        vectors = embed_documents(texts, client=mock_client, output_dim=768, batch_size=2)

        self.assertEqual(len(vectors), 5)
        self.assertEqual(mock_client.models.embed_content.call_count, 3)


if __name__ == "__main__":
    unittest.main()
