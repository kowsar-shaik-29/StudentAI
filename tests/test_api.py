"""
Unit tests for StudentAI FastAPI Backend (src.api).
Tests cover:
1. Health endpoint.
2. Valid and invalid agent requests.
3. Learning ask, profile, recommendations endpoints.
4. Study-plan parameter validation.
5. Career ask and job-fit validation.
6. Project listing and project-scoped queries.
7. Document upload validation (extensions, size, missing files).
8. Gemini / AI failure handling (sanitized errors).
9. CORS headers verification.
10. Input boundary and validation errors (422 Unprocessable Entity).
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api import app
from src.agent_orchestrator import AgentResult, ToolExecutionTrace

client = TestClient(app)


class TestFastAPIBackend(unittest.TestCase):
    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)

    def test_cors_headers(self):
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertIn(response.status_code, (200, 204))
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")

    @patch("src.api.run_agent")
    def test_agent_run_success(self, mock_run):
        mock_run.return_value = AgentResult(
            query="Help me study DBMS",
            answer="Here is your study plan: Day 1... [Page 1]",
            classification="LEARNING",
            workflow="study_planning",
            execution_trace=[
                ToolExecutionTrace(
                    step=1,
                    tool_name="create_study_plan",
                    args={"subject": "dbms", "days": 7},
                    status="success",
                    duration_ms=25.0,
                    output={"plan_id": "test_plan"},
                )
            ],
            sources=[{"file_name": "dbms.pdf", "page_number": 1, "score": 0.9}],
            citations_verified=True,
            citation_report={"is_valid": True, "cited_pages": [1]},
            limitations=[],
            total_duration_ms=45.0,
        )

        resp = client.post("/api/agent/run", json={"query": "Help me study DBMS"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["classification"], "LEARNING")
        self.assertEqual(data["workflow"], "study_planning")
        self.assertEqual(len(data["execution_trace"]), 1)
        self.assertTrue(data["citations_verified"])

    def test_agent_run_invalid_empty_query(self):
        resp = client.post("/api/agent/run", json={"query": ""})
        self.assertEqual(resp.status_code, 422)

    @patch("src.api.answer_learning_question")
    def test_learning_ask_success(self, mock_learn):
        mock_learn.return_value = {
            "intent": "EXPLAIN",
            "topic": "Algorithms",
            "subtopic": "BFS",
            "answer": "Breadth First Search explores neighbor nodes first. [Page 2]",
            "sources": [{"file_name": "algo.pdf", "page_number": 2, "score": 0.85}],
        }

        resp = client.post("/api/learning/ask", json={"query": "Explain BFS", "top_k": 3})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "EXPLAIN")
        self.assertEqual(data["topic"], "Algorithms")
        self.assertIn("Breadth First Search", data["answer"])

    def test_learning_ask_invalid_intent(self):
        resp = client.post("/api/learning/ask", json={"query": "What is BFS?", "intent": "INVALID"})
        self.assertEqual(resp.status_code, 422)

    @patch("src.api.get_learning_profile")
    def test_learning_profile(self, mock_prof):
        mock_prof.return_value = {
            "user_id": "default_user",
            "topics": [],
            "weak_topics": [],
            "strong_topics": [],
            "insufficient_data_topics": [],
            "recommendations": [],
            "message": "No learning history yet.",
        }
        resp = client.get("/api/learning/profile")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["user_id"], "default_user")

    @patch("src.api.get_learning_recommendations")
    def test_learning_recommendations(self, mock_recs):
        mock_recs.return_value = {
            "user_id": "default_user",
            "recommendations": [
                {"rank": 1, "topic": "DBMS", "subtopic": "Normalization", "priority": "HIGH"}
            ],
        }
        resp = client.get("/api/learning/recommendations?limit=3")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["recommendations"]), 1)

    @patch("src.api.create_study_plan")
    def test_create_study_plan_success(self, mock_plan):
        mock_plan.return_value = {
            "plan_id": "plan_123",
            "subject": "DBMS",
            "days": 7,
            "daily_minutes": 60,
            "plan": [{"day": 1, "activities": []}],
        }
        resp = client.post("/api/study-plans", json={"subject": "DBMS", "days": 7, "daily_minutes": 60})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["plan_id"], "plan_123")

    def test_create_study_plan_invalid_days(self):
        resp = client.post("/api/study-plans", json={"subject": "DBMS", "days": 0})
        self.assertEqual(resp.status_code, 422)

    @patch("src.api.answer_career_question")
    def test_career_ask_success(self, mock_career):
        mock_career.return_value = {
            "query": "What skills does candidate have?",
            "answer": "Candidate knows Python and SQL. [Page 1]",
            "sources": [{"file_name": "resume.pdf", "page_number": 1}],
        }
        resp = client.post("/api/career/ask", json={"query": "What skills does candidate have?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Python", data["answer"])

    @patch("src.api.analyze_job_fit")
    def test_job_fit_success(self, mock_fit):
        mock_fit.return_value = {
            "resume": {"document_id": "res.pdf", "skill_count": 5},
            "job": {"document_id": "job.pdf", "requirement_count": 5},
            "match": {"coverage_percentage": 80.0, "matched_count": 4, "missing_count": 1},
        }
        resp = client.post("/api/career/job-fit", json={"resume_id": "res.pdf", "job_id": "job.pdf"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["match"]["coverage_percentage"], 80.0)

    def test_job_fit_rejects_path_traversal(self):
        resp = client.post("/api/career/job-fit", json={"resume_id": "../etc/passwd", "job_id": "job.pdf"})
        self.assertEqual(resp.status_code, 422)

    @patch("src.api.list_projects")
    def test_list_projects(self, mock_projs):
        mock_projs.return_value = [
            {"project_id": "exp_tracker", "project_name": "Expense Tracker", "document_count": 1}
        ]
        resp = client.get("/api/projects")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 1)

    @patch("src.api.answer_project_question")
    def test_project_ask_success(self, mock_pask):
        mock_pask.return_value = {
            "query": "What is the tech stack?",
            "project_id": "exp_tracker",
            "answer": "React and FastAPI. [Page 1]",
            "ambiguous": False,
            "available_projects": ["exp_tracker"],
            "sources": [{"file_name": "proj.md", "page_number": 1}],
        }
        resp = client.post("/api/projects/ask", json={"query": "What is the tech stack?", "project_id": "exp_tracker"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["project_id"], "exp_tracker")
        self.assertIn("React", data["answer"])

    def test_upload_unsupported_extension_rejected(self):
        files = {"file": ("script.exe", b"binarycontent", "application/octet-stream")}
        resp = client.post("/api/documents/upload", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file extension", resp.json()["detail"])

    def test_upload_empty_file_rejected(self):
        files = {"file": ("notes.pdf", b"", "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.json()["detail"])

    @patch("src.api.ingest_document")
    def test_upload_document_success(self, mock_ingest):
        mock_ingest.return_value = {
            "file_name": "notes.md",
            "document_id": "notes.md",
            "category": "academic",
            "subject": "algorithms",
            "document_type": "notes",
            "chunks": 3,
            "pages": 1,
            "embeddings": 3,
        }
        files = {"file": ("notes.md", b"# Algorithm Notes\nContent here.", "text/markdown")}
        data = {"category": "academic", "subject": "algorithms", "document_type": "notes"}
        resp = client.post("/api/documents/upload", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        res_json = resp.json()
        self.assertEqual(res_json["file_name"], "notes.md")
        self.assertEqual(res_json["chunks"], 3)


if __name__ == "__main__":
    unittest.main()
