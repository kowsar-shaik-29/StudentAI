import unittest
from unittest.mock import MagicMock, patch
import json
from src.query_rewriter import rewrite_query


class TestQueryRewriter(unittest.TestCase):

    def test_empty_and_whitespace_query(self):
        res = rewrite_query("")
        self.assertEqual(res["rewritten_query"], "")
        self.assertFalse(res["is_rewritten"])

        res_spaces = rewrite_query("   ")
        self.assertEqual(res_spaces["rewritten_query"], "   ")
        self.assertFalse(res_spaces["is_rewritten"])

    def test_successful_acronym_expansion(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "rewritten_query": "OSI model physical layer layer 1 function and data transmission",
            "is_rewritten": True,
            "reason": "Expanded abbreviation OSI and clarified lowest layer 1",
        })
        mock_client.models.generate_content.return_value = mock_resp

        res = rewrite_query("OSI lowest layer", client=mock_client)
        self.assertTrue(res["is_rewritten"])
        self.assertIn("physical layer", res["rewritten_query"])
        self.assertEqual(res["original_query"], "OSI lowest layer")

    def test_already_clear_query_not_rewritten(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "rewritten_query": "What is star topology in computer networking?",
            "is_rewritten": False,
            "reason": "Query is already specific and clear",
        })
        mock_client.models.generate_content.return_value = mock_resp

        res = rewrite_query("What is star topology in computer networking?", client=mock_client)
        self.assertFalse(res["is_rewritten"])

    def test_malformed_json_fallback_to_original(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "This is not json."
        mock_client.models.generate_content.return_value = mock_resp

        res = rewrite_query("Explain B+ tree", client=mock_client, max_retries=0)
        self.assertEqual(res["rewritten_query"], "Explain B+ tree")
        self.assertFalse(res["is_rewritten"])

    def test_api_error_fallback_sanitization(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API key AIzaSyD9ExampleFakeKey123456 error")

        res = rewrite_query("What is primary key?", client=mock_client, max_retries=0)
        self.assertEqual(res["rewritten_query"], "What is primary key?")
        self.assertFalse(res["is_rewritten"])
        self.assertNotIn("AIzaSyD9ExampleFakeKey123456", res["reason"])
        self.assertIn("[REDACTED_API_KEY]", res["reason"])


if __name__ == "__main__":
    unittest.main()
