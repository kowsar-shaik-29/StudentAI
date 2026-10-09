import unittest
from unittest.mock import MagicMock, patch
from src.project_rag import (
    answer_project_question,
    list_projects,
    get_project_all_text,
    summarize_project,
    compare_projects,
    STANDARD_PROJECT_REFUSAL,
)


class TestProjectRAG(unittest.TestCase):

    def test_empty_query_raises_error(self):
        with self.assertRaises(ValueError):
            answer_project_question("")
        with self.assertRaises(ValueError):
            answer_project_question("   ")

    def test_list_projects(self):
        mock_sr = MagicMock()
        mock_sr.documents = {
            "c1": {
                "chunk_id": "c1",
                "text": "STUDENT PROJECT: Expense Tracker Application\nFastAPI backend and React frontend.",
                "file_name": "expense_tracker_readme.md",
                "document_id": "p1_doc",
                "document_type": "project_documentation",
                "category": "project",
                "subject": "expense_tracker",
            },
            "c2": {
                "chunk_id": "c2",
                "text": "STUDENT PROJECT: Student Guidance Platform\nNext.js and Django.",
                "file_name": "guidance_readme.md",
                "document_id": "p2_doc",
                "document_type": "project_documentation",
                "category": "project",
                "subject": "student_guidance",
            },
            "c3": {
                "chunk_id": "c3",
                "text": "Academic notes on networks.",
                "file_name": "networking.pdf",
                "document_id": "acad_doc",
                "document_type": "notes",
                "category": "academic",
                "subject": "computer_networks",
            }
        }

        projects = list_projects(sparse_retriever=mock_sr)
        self.assertEqual(len(projects), 2)
        proj_ids = {p["project_id"] for p in projects}
        self.assertIn("expense_tracker", proj_ids)
        self.assertIn("student_guidance", proj_ids)

    def test_ambiguity_when_multiple_projects_and_no_project_id(self):
        mock_sr = MagicMock()
        mock_sr.documents = {
            "c1": {
                "chunk_id": "c1",
                "text": "STUDENT PROJECT: Expense Tracker Application",
                "file_name": "expense_tracker.pdf",
                "document_id": "p1",
                "category": "project",
                "subject": "expense_tracker",
            },
            "c2": {
                "chunk_id": "c2",
                "text": "STUDENT PROJECT: Guidance Platform",
                "file_name": "guidance.pdf",
                "document_id": "p2",
                "category": "project",
                "subject": "guidance",
            }
        }

        res = answer_project_question(
            "Explain the architecture",
            project_id=None,
            sparse_retriever=mock_sr,
            genai_client=MagicMock(),
        )
        self.assertTrue(res["ambiguous"])
        self.assertIn("Multiple projects found", res["answer"])
        self.assertEqual(len(res["available_projects"]), 2)

    @patch("src.project_rag.hybrid_search")
    def test_no_relevant_documents_returns_refusal(self, mock_hybrid_search):
        mock_hybrid_search.return_value = []
        mock_sr = MagicMock()
        mock_sr.documents = {}

        res = answer_project_question(
            "What database is used?",
            project_id="test_proj",
            sparse_retriever=mock_sr,
            genai_client=MagicMock(),
        )
        self.assertEqual(res["answer"], STANDARD_PROJECT_REFUSAL)
        self.assertEqual(res["sources"], [])

    @patch("src.project_rag.hybrid_search")
    def test_grounded_answer_with_citations(self, mock_hybrid_search):
        mock_hybrid_search.return_value = [
            {
                "chunk_id": "c1",
                "text": "The Expense Tracker uses PostgreSQL for transactional financial logs. [Page 1]",
                "file_name": "expense_tracker.md",
                "page_number": 1,
                "score": 0.92,
                "category": "project",
                "subject": "expense_tracker",
                "document_id": "p1",
            }
        ]

        mock_sr = MagicMock()
        mock_sr.documents = {
            "c1": {
                "chunk_id": "c1",
                "text": "The Expense Tracker uses PostgreSQL for transactional financial logs.",
                "file_name": "expense_tracker.md",
                "document_id": "p1",
                "category": "project",
                "subject": "expense_tracker",
            }
        }

        mock_genai_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "The Expense Tracker uses PostgreSQL for storing transactional financial logs [Page 1]."
        mock_genai_client.models.generate_content.return_value = mock_resp

        res = answer_project_question(
            "What database is used?",
            project_id="expense_tracker",
            sparse_retriever=mock_sr,
            genai_client=mock_genai_client,
        )

        self.assertFalse(res["ambiguous"])
        self.assertIn("PostgreSQL", res["answer"])
        self.assertIn("[Page 1]", res["answer"])
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["file_name"], "expense_tracker.md")

    def test_get_project_all_text_not_found(self):
        mock_sr = MagicMock()
        mock_sr.documents = {}
        with self.assertRaises(ValueError):
            get_project_all_text("nonexistent_project", sparse_retriever=mock_sr)


if __name__ == "__main__":
    unittest.main()
