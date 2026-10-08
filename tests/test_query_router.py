"""
Unit Tests for Query Router (Step 13).
Tests:
- Valid JSON parsing for all routes (LEARNING, CAREER, PROJECT, GENERAL)
- Confidence parsing and threshold filtering
- Low confidence fallback to GENERAL
- Malformed JSON handling
- Missing fields handling
- Invalid route handling
- Empty query validation
- API failure graceful fallback
- Route to metadata category mapping
"""

import unittest
from unittest.mock import MagicMock
from google.genai.errors import APIError

from src.query_router import (
    route_query,
    get_category_filter_for_route,
    VALID_ROUTES,
    ROUTE_CATEGORY_MAP,
)


class TestQueryRouter(unittest.TestCase):

    def test_empty_query_raises_value_error(self):
        """Empty or whitespace-only queries must raise ValueError."""
        with self.assertRaises(ValueError):
            route_query("")
        with self.assertRaises(ValueError):
            route_query("   \n\t  ")

    def test_category_filter_mapping(self):
        """Verify route to metadata filter mapping."""
        self.assertEqual(get_category_filter_for_route("LEARNING"), "academic")
        self.assertEqual(get_category_filter_for_route("CAREER"), "career")
        self.assertEqual(get_category_filter_for_route("PROJECT"), "project")
        self.assertIsNone(get_category_filter_for_route("GENERAL"))
        self.assertIsNone(get_category_filter_for_route("UNKNOWN"))

    def test_valid_json_routes(self):
        """Test proper parsing and routing for all standard routes."""
        test_cases = [
            ("Explain B+ trees", '{"route": "LEARNING", "confidence": 0.95, "reason": "Academic concept"}', "LEARNING", 0.95),
            ("Review my resume", '{"route": "CAREER", "confidence": 0.90, "reason": "Resume question"}', "CAREER", 0.90),
            ("Project architecture", '{"route": "PROJECT", "confidence": 0.88, "reason": "Project question"}', "PROJECT", 0.88),
            ("What is the capital of France?", '{"route": "GENERAL", "confidence": 0.99, "reason": "World trivia"}', "GENERAL", 0.99),
        ]

        for query, mock_json, expected_route, expected_conf in test_cases:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = mock_json
            mock_client.models.generate_content.return_value = mock_resp

            result = route_query(query, client=mock_client)
            self.assertEqual(result["route"], expected_route)
            self.assertAlmostEqual(result["confidence"], expected_conf, places=2)
            self.assertTrue(result["reason"])

    def test_confidence_threshold_behavior(self):
        """When model confidence is below threshold, fallback to GENERAL."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"route": "CAREER", "confidence": 0.55, "reason": "Ambiguous career reference"}'
        mock_client.models.generate_content.return_value = mock_resp

        # With default threshold 0.70, 0.55 falls back to GENERAL
        result = route_query("Maybe career related", client=mock_client, confidence_threshold=0.70)
        self.assertEqual(result["route"], "GENERAL")
        self.assertAlmostEqual(result["confidence"], 0.55, places=2)
        self.assertEqual(result["raw_route"], "CAREER")
        self.assertIn("below threshold", result["reason"])

        # With lower threshold 0.50, 0.55 is accepted as CAREER
        result2 = route_query("Maybe career related", client=mock_client, confidence_threshold=0.50)
        self.assertEqual(result2["route"], "CAREER")
        self.assertAlmostEqual(result2["confidence"], 0.55, places=2)

    def test_markdown_wrapped_json_handling(self):
        """Verify that markdown code fences around JSON are safely stripped."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '```json\n{"route": "LEARNING", "confidence": 0.92, "reason": "Algorithm"}\n```'
        mock_client.models.generate_content.return_value = mock_resp

        result = route_query("How does Dijkstra work?", client=mock_client)
        self.assertEqual(result["route"], "LEARNING")
        self.assertAlmostEqual(result["confidence"], 0.92, places=2)

    def test_malformed_json_fallback(self):
        """Malformed JSON from model gracefully falls back to GENERAL without crashing."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = 'NOT_JSON {"route": "LEARNING"'
        mock_client.models.generate_content.return_value = mock_resp

        result = route_query("Some question", client=mock_client)
        self.assertEqual(result["route"], "GENERAL")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("Failed to parse", result["reason"])

    def test_missing_fields_and_invalid_route(self):
        """Missing confidence or unknown route falls back safely."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        # Invalid route name
        mock_resp.text = '{"route": "UNKNOWN_DOMAIN", "confidence": 0.95}'
        mock_client.models.generate_content.return_value = mock_resp

        result = route_query("Test query", client=mock_client)
        self.assertEqual(result["route"], "GENERAL")

        # Missing confidence field defaults to 0.0 -> below threshold -> GENERAL
        mock_resp.text = '{"route": "LEARNING"}'
        result2 = route_query("Test query", client=mock_client)
        self.assertEqual(result2["route"], "GENERAL")
        self.assertEqual(result2["confidence"], 0.0)

    def test_api_failure_fallback_and_no_key_leak(self):
        """API errors fall back to GENERAL and never leak API key in error reason."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = APIError(
            code=503,
            response_json={"error": {"message": "Service Unavailable key=AIzaSyD-SecretKey12345"}},
        )

        result = route_query("What is DNS?", client=mock_client)
        self.assertEqual(result["route"], "GENERAL")
        self.assertEqual(result["confidence"], 0.0)
        self.assertNotIn("AIzaSyD-SecretKey12345", result["reason"])
        self.assertIn("[REDACTED_API_KEY]", result["reason"])


if __name__ == "__main__":
    unittest.main()
