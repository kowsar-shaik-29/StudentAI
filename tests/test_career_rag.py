import unittest
from unittest.mock import MagicMock, patch
from src.career_rag import (
    answer_career_question,
    STANDARD_CAREER_REFUSAL,
    get_career_document_text,
    analyze_job_fit,
)


class TestCareerRAG(unittest.TestCase):

    def test_empty_query_raises_error(self):
        with self.assertRaises(ValueError):
            answer_career_question("")
        with self.assertRaises(ValueError):
            answer_career_question("   ")

    @patch("src.career_rag.hybrid_search")
    def test_no_retrieved_chunks_returns_refusal(self, mock_hybrid_search):
        mock_hybrid_search.return_value = []
        mock_qdrant = MagicMock()

        res = answer_career_question("What is the candidate experience?", client=mock_qdrant)
        self.assertEqual(res["answer"], STANDARD_CAREER_REFUSAL)
        self.assertEqual(res["sources"], [])

    @patch("src.career_rag.hybrid_search")
    def test_career_qa_success(self, mock_hybrid_search):
        mock_hybrid_search.return_value = [
            {
                "chunk_id": "chunk_c1",
                "text": "The candidate has 4 years of experience building scalable backend services with Java and Spring Boot. [Page 1]",
                "file_name": "candidate_resume.pdf",
                "page_number": 1,
                "score": 0.85,
                "category": "career",
                "document_type": "resume",
                "document_id": "res_123",
            }
        ]

        mock_genai_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "The candidate has 4 years of backend experience using Java and Spring Boot [Page 1]."
        mock_genai_client.models.generate_content.return_value = mock_resp
        mock_qdrant = MagicMock()

        res = answer_career_question(
            "What backend experience does the candidate have?",
            client=mock_qdrant,
            genai_client=mock_genai_client,
        )

        self.assertIn("Spring Boot", res["answer"])
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["file_name"], "candidate_resume.pdf")

    def test_get_career_document_text(self):
        mock_sr = MagicMock()
        mock_sr.documents = {
            "c1": {
                "chunk_id": "c1",
                "chunk_index": 0,
                "document_id": "doc_resume_1",
                "file_name": "resume.pdf",
                "document_type": "resume",
                "category": "career",
                "text": "Skills: Python, SQL",
            },
            "c2": {
                "chunk_id": "c2",
                "chunk_index": 1,
                "document_id": "doc_resume_1",
                "file_name": "resume.pdf",
                "document_type": "resume",
                "category": "career",
                "text": "Experience: 3 years as backend developer",
            }
        }

        doc = get_career_document_text("doc_resume_1", sparse_retriever=mock_sr)
        self.assertEqual(doc["file_name"], "resume.pdf")
        self.assertIn("Python, SQL", doc["full_text"])
        self.assertIn("backend developer", doc["full_text"])

    @patch("src.career_rag.extract_skills")
    def test_analyze_job_fit(self, mock_extract):
        mock_sr = MagicMock()
        mock_sr.documents = {
            "c1": {
                "chunk_id": "c1",
                "chunk_index": 0,
                "document_id": "res_1",
                "file_name": "resume.pdf",
                "document_type": "resume",
                "category": "career",
                "text": "Resume text",
            },
            "c2": {
                "chunk_id": "c2",
                "chunk_index": 0,
                "document_id": "job_1",
                "file_name": "job.pdf",
                "document_type": "job_description",
                "category": "career",
                "text": "Job text",
            }
        }

        # Mock resume extraction then job extraction
        mock_extract.side_effect = [
            {
                "skills": [
                    {"name": "Java", "normalized_name": "java", "evidence_quote": "Java dev"},
                    {"name": "SQL", "normalized_name": "sql", "evidence_quote": "SQL db"},
                    {"name": "React", "normalized_name": "react", "evidence_quote": "React web"},
                ]
            },
            {
                "skills": [
                    {"name": "Java", "normalized_name": "java", "evidence_quote": "Must know Java"},
                    {"name": "SQL", "normalized_name": "sql", "evidence_quote": "Must know SQL"},
                    {"name": "React", "normalized_name": "react", "evidence_quote": "Must know React"},
                    {"name": "AWS", "normalized_name": "aws", "evidence_quote": "Must know AWS"},
                    {"name": "Docker", "normalized_name": "docker", "evidence_quote": "Must know Docker"},
                ]
            }
        ]

        fit = analyze_job_fit("res_1", "job_1", sparse_retriever=mock_sr, genai_client=MagicMock())
        self.assertEqual(fit["match"]["coverage_percentage"], 60.0)
        self.assertEqual(fit["match"]["matched_count"], 3)
        self.assertEqual(fit["match"]["missing_count"], 2)


if __name__ == "__main__":
    unittest.main()
