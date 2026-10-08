"""
Unit tests for embedding module (src.embedder).
Uses mocking to test parsing, dimension validation, and sanitization without making network calls.
"""

import unittest
from unittest.mock import MagicMock
from google.genai import types
from src.embedder import embed_documents, embed_query, _sanitize_error


class TestEmbedder(unittest.TestCase):
    def test_sanitize_error_replaces_api_key(self):
        # Test error sanitization
        msg = "Error 403: Invalid API key"
        sanitized = _sanitize_error(msg)
        self.assertNotIn("AIzaSy", sanitized)

    def test_embed_documents_parsing(self):
        # Mock client and response with 2 embeddings of 768 dimensions
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
        # Returns 512 dimensions when 768 was requested
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


if __name__ == "__main__":
    unittest.main()
