import unittest
from unittest.mock import MagicMock, patch
import json
from google.genai.errors import APIError

from src.skill_extractor import (
    normalize_skill_name,
    extract_skills,
    _clean_json_response,
)


class TestSkillExtractor(unittest.TestCase):

    def test_normalize_skill_name(self):
        self.assertEqual(normalize_skill_name("  Python 3  "), "python 3")
        self.assertEqual(normalize_skill_name("C++"), "c++")
        self.assertEqual(normalize_skill_name("C#"), "c#")
        self.assertEqual(normalize_skill_name("Node.js"), "node.js")
        self.assertEqual(normalize_skill_name("PostgreSQL!"), "postgresql")
        self.assertEqual(normalize_skill_name(""), "")

    def test_clean_json_response(self):
        markdown_wrapped = "```json\n{\"skills\": []}\n```"
        self.assertEqual(_clean_json_response(markdown_wrapped), "{\"skills\": []}")
        plain = "{\"skills\": []}"
        self.assertEqual(_clean_json_response(plain), "{\"skills\": []}")

    def test_extract_skills_empty_input(self):
        res = extract_skills("", document_type="resume")
        self.assertEqual(res["skills"], [])
        self.assertEqual(res["raw_count"], 0)

        res_spaces = extract_skills("   \n\t  ", document_type="job_description")
        self.assertEqual(res_spaces["skills"], [])
        self.assertEqual(res_spaces["raw_count"], 0)

    def test_extract_skills_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "document_type": "resume",
            "skills": [
                {
                    "name": "Python",
                    "normalized_name": "python",
                    "skill_type": "language",
                    "evidence_quote": "5 years experience in Python",
                    "confidence": 0.98,
                },
                {
                    "name": "Docker",
                    "normalized_name": "docker",
                    "skill_type": "tool",
                    "evidence_quote": "built Docker images",
                    "confidence": 0.95,
                }
            ]
        })
        mock_client.models.generate_content.return_value = mock_response

        text = "Candidate with 5 years experience in Python. Built Docker images."
        res = extract_skills(text, document_type="resume", client=mock_client, page_number=2)

        self.assertEqual(res["document_type"], "resume")
        self.assertEqual(len(res["skills"]), 2)
        self.assertEqual(res["skills"][0]["name"], "Python")
        self.assertEqual(res["skills"][0]["page_number"], 2)
        self.assertEqual(res["skills"][1]["name"], "Docker")
        self.assertEqual(res["skills"][1]["normalized_name"], "docker")

    def test_extract_skills_invalid_json_fallback(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not json."
        mock_client.models.generate_content.return_value = mock_response

        res = extract_skills("Some resume text", client=mock_client, max_retries=0)
        self.assertEqual(res["skills"], [])
        self.assertEqual(res["raw_count"], 0)


if __name__ == "__main__":
    unittest.main()
