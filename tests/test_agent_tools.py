import unittest
from unittest.mock import patch, MagicMock
from src.agent_tools import (
    TOOL_REGISTRY,
    get_available_tools,
    execute_tool,
    ToolExecutionError,
)

class TestAgentTools(unittest.TestCase):
    def test_registry_contains_required_tools(self):
        expected_tools = {
            "learning_ask",
            "learning_profile",
            "learning_recommendations",
            "revision_topics",
            "create_study_plan",
            "career_ask",
            "analyze_job_fit",
            "project_list",
            "project_ask",
            "project_summary",
            "project_interview",
        }
        self.assertTrue(expected_tools.issubset(set(TOOL_REGISTRY.keys())))

    def test_reject_unregistered_tool(self):
        with self.assertRaises(ToolExecutionError):
            execute_tool("unknown_tool", {})

    def test_parameter_validation_rejects_empty_query(self):
        res = execute_tool("learning_ask", {"query": ""})
        self.assertFalse(res["success"])
        self.assertIn("non-empty string", res["error"])

    def test_parameter_validation_rejects_invalid_intent(self):
        res = execute_tool("learning_ask", {"query": "What is BFS?", "intent": "INVALID_INTENT"})
        self.assertFalse(res["success"])
        self.assertIn("Invalid intent", res["error"])

    @patch("src.agent_tools.answer_learning_question")
    def test_parameter_validation_normalizes_intent_aliases(self, mock_learn):
        mock_learn.return_value = {
            "intent": "EXPLAIN",
            "topic": "Algorithms",
            "subtopic": "BFS",
            "answer": "BFS traverses level by level. [Page 1]",
            "sources": [{"file_name": "algo.pdf", "page_number": 1}],
        }
        res = execute_tool("learning_ask", {"query": "What is BFS?", "intent": "concept_explanation"})
        self.assertTrue(res["success"])
        mock_learn.assert_called_once()
        self.assertEqual(mock_learn.call_args.kwargs["intent"], "EXPLAIN")

    def test_parameter_validation_rejects_invalid_days(self):
        res = execute_tool("create_study_plan", {"subject": "DBMS", "days": -1})
        self.assertFalse(res["success"])
        self.assertIn("between 1 and 365", res["error"])

    @patch("src.agent_tools.answer_learning_question")
    def test_learning_ask_execution(self, mock_learn):
        mock_learn.return_value = {
            "intent": "EXPLAIN",
            "topic": "Algorithms",
            "subtopic": "BFS",
            "answer": "BFS traverses level by level. [Page 1]",
            "sources": [{"file_name": "algo.pdf", "page_number": 1}],
        }
        res = execute_tool("learning_ask", {"query": "Explain BFS"})
        self.assertTrue(res["success"])
        self.assertEqual(res["output"]["topic"], "Algorithms")
        self.assertIn("BFS", res["output"]["answer"])

    @patch("src.agent_tools.list_projects")
    def test_project_list_execution(self, mock_list):
        mock_list.return_value = [
            {"project_id": "p1", "project_name": "Expense Tracker", "document_count": 2, "filenames": ["doc.md"]}
        ]
        res = execute_tool("project_list", {})
        self.assertTrue(res["success"])
        self.assertEqual(res["output"]["total_projects"], 1)

if __name__ == "__main__":
    unittest.main()
