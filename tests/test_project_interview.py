import unittest
from unittest.mock import MagicMock, patch
import json
from src.project_interview import generate_project_interview_questions


class TestProjectInterview(unittest.TestCase):

    def test_invalid_count_raises_error(self):
        with self.assertRaises(ValueError):
            generate_project_interview_questions("my_proj", count=0)
        with self.assertRaises(ValueError):
            generate_project_interview_questions("my_proj", count=-3)

    def test_insufficient_evidence_when_project_not_found(self):
        mock_sr = MagicMock()
        mock_sr.documents = {}

        res = generate_project_interview_questions("nonexistent_proj", sparse_retriever=mock_sr)
        self.assertTrue(res["insufficient_evidence"])
        self.assertEqual(res["generated_count"], 0)
        self.assertEqual(res["questions"], [])

    def test_generate_interview_questions_success(self):
        mock_sr = MagicMock()
        mock_sr.documents = {
            "c1": {
                "chunk_id": "c1",
                "text": "STUDENT PROJECT: Expense Tracker Application\nArchitecture: React frontend, FastAPI backend, PostgreSQL database, Docker containerization.",
                "file_name": "expense_tracker_readme.md",
                "document_id": "doc1",
                "category": "project",
                "subject": "expense_tracker",
                "page_number": 1,
            }
        }

        mock_genai_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "project_name": "Expense Tracker",
            "questions": [
                {
                    "question": "Why did you choose FastAPI over Flask or Django for this backend?",
                    "difficulty": "MEDIUM",
                    "topic": "technology_selection",
                    "evidence_summary": "Uses FastAPI backend for high performance and asynchronous request handling",
                    "source_filename": "expense_tracker_readme.md",
                    "page_number": 1,
                    "answer_outline": "Explain automatic Swagger documentation, Pydantic data validation, and async I/O.",
                },
                {
                    "question": "How is PostgreSQL structured for storing user expenses?",
                    "difficulty": "EASY",
                    "topic": "data_flow",
                    "evidence_summary": "PostgreSQL database used for storage",
                    "source_filename": "expense_tracker_readme.md",
                    "page_number": 1,
                    "answer_outline": "Relational schema ensuring ACID compliance for monetary balances.",
                },
                # Duplicate question that should be filtered out
                {
                    "question": "Why did you choose FastAPI over Flask or Django for this backend?",
                    "difficulty": "MEDIUM",
                    "topic": "technology_selection",
                    "evidence_summary": "Duplicate",
                    "source_filename": "expense_tracker_readme.md",
                    "page_number": 1,
                    "answer_outline": "",
                }
            ]
        })
        mock_genai_client.models.generate_content.return_value = mock_resp

        res = generate_project_interview_questions(
            "expense_tracker",
            count=5,
            difficulty="MIXED",
            genai_client=mock_genai_client,
            sparse_retriever=mock_sr,
        )

        self.assertFalse(res["insufficient_evidence"])
        self.assertEqual(res["generated_count"], 2)  # Duplicate stripped
        self.assertEqual(len(res["questions"]), 2)
        self.assertEqual(res["questions"][0]["difficulty"], "MEDIUM")
        self.assertEqual(res["questions"][0]["topic"], "technology_selection")
        self.assertEqual(res["questions"][0]["source_filename"], "expense_tracker_readme.md")
        self.assertEqual(res["questions"][0]["page_number"], 1)

    def test_json_parse_failure_fallback(self):
        mock_sr = MagicMock()
        mock_sr.documents = {
            "c1": {
                "chunk_id": "c1",
                "text": "STUDENT PROJECT: Sample Project",
                "file_name": "sample.md",
                "document_id": "doc1",
                "category": "project",
                "subject": "sample_project",
                "page_number": 1,
            }
        }

        mock_genai_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Not JSON output"
        mock_genai_client.models.generate_content.return_value = mock_resp

        res = generate_project_interview_questions(
            "sample_project",
            count=3,
            genai_client=mock_genai_client,
            sparse_retriever=mock_sr,
            max_retries=0,
        )

        self.assertTrue(res["insufficient_evidence"])
        self.assertEqual(res["generated_count"], 0)
        self.assertEqual(res["questions"], [])


if __name__ == "__main__":
    unittest.main()
