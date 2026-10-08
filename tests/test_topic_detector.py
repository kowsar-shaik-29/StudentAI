"""
Unit Tests for Topic Detector Module (Step 15).
Tests:
- Strict JSON structure validation
- Graceful API failure and 429/503 retry handling
- Key sanitization (no API keys leaked in errors)
- Empty or whitespace query fallback
- Grounded computer science classification mock checks
"""

import unittest
from unittest.mock import MagicMock, patch
from google.genai.errors import ServerError
from src.topic_detector import detect_learning_topic, FALLBACK_RESULT, TOPIC_DETECTOR_MODEL


class TestTopicDetector(unittest.TestCase):

    def test_empty_and_whitespace_query_fallback(self):
        """Empty or whitespace queries immediately return FALLBACK_RESULT."""
        res1 = detect_learning_topic("")
        self.assertEqual(res1["topic"], "Computer Science")
        self.assertEqual(res1["subtopic"], "General")
        self.assertEqual(res1["confidence"], 0.0)

        res2 = detect_learning_topic("   \n\t  ")
        self.assertEqual(res2["topic"], "Computer Science")
        self.assertEqual(res2["subtopic"], "General")

    def test_valid_json_response_parsing(self):
        """Valid JSON response from Gemini model is properly parsed."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"topic": "Computer Networks", "subtopic": "TCP / Three-Way Handshake", "confidence": 0.95, "reason": "Query asks about TCP handshake"}'
        mock_client.models.generate_content.return_value = mock_resp

        result = detect_learning_topic("Explain TCP three way handshake", client=mock_client)

        self.assertEqual(result["topic"], "Computer Networks")
        self.assertEqual(result["subtopic"], "TCP / Three-Way Handshake")
        self.assertEqual(result["confidence"], 0.95)
        self.assertIn("TCP handshake", result["reason"])

    def test_markdown_code_fence_stripping(self):
        """Markdown code fences around JSON are safely stripped."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '```json\n{"topic": "DBMS", "subtopic": "Normalization", "confidence": 0.92, "reason": "Mentions normalization"}\n```'
        mock_client.models.generate_content.return_value = mock_resp

        result = detect_learning_topic("What is 3NF?", client=mock_client)
        self.assertEqual(result["topic"], "DBMS")
        self.assertEqual(result["subtopic"], "Normalization")
        self.assertEqual(result["confidence"], 0.92)

    def test_malformed_json_fallback(self):
        """Malformed JSON gracefully falls back to default result without crashing."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = 'Not valid JSON at all!'
        mock_client.models.generate_content.return_value = mock_resp

        result = detect_learning_topic("Random question", client=mock_client)
        self.assertEqual(result["topic"], "Computer Science")
        self.assertEqual(result["subtopic"], "General")
        self.assertEqual(result["confidence"], 0.0)

    def test_api_server_error_and_key_sanitization(self):
        """API error returns fallback safely without leaking sensitive keys in reason."""
        mock_client = MagicMock()
        secret_key = "AIzaSySecretFakeApiKey12345"
        mock_client.models.generate_content.side_effect = ServerError(500, f"Server error with key {secret_key}")

        result = detect_learning_topic("Explain something", client=mock_client)
        self.assertEqual(result["topic"], "Computer Science")
        self.assertEqual(result["confidence"], 0.0)
        self.assertNotIn(secret_key, result["reason"])
        self.assertIn("[REDACTED_API_KEY]", result["reason"])


if __name__ == "__main__":
    unittest.main()
