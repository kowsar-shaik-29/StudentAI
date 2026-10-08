"""
Unit Tests for Recommendation Engine Module (Step 16).
Tests:
A. No learning history -> empty recommendations, helpful message
B. HIGH weakness -> HIGH priority recommendation
C. MEDIUM weakness -> MEDIUM recommendation
D. LOW weakness -> suppressed when higher priority exists, recommended when alone
E. INSUFFICIENT_DATA -> diagnostic assessment recommendation
F. Multiple topics -> sorted strictly by priority rank and weakness score
G. Deduplication -> duplicate (topic, subtopic) pairs collapsed
H. Source filtering -> category="academic" allowed; career/project/resume excluded
I. Missing source -> empty source_documents, no fabricated documents
J. Limit parameter -> max_recommendations respected
K. Topic filter -> filters down to specific topic
L. User isolation -> user A events do not affect user B
M. Learning profile -> structured overview of weak, strong, and insufficient data
"""

import unittest
from pathlib import Path
import tempfile
import shutil
from src.learning_tracker import (
    record_learning_event,
    record_quiz_attempt,
)
from src.sparse_retriever import SparseRetriever
from src.recommendation_engine import (
    get_learning_recommendations,
    get_learning_profile,
    find_academic_sources,
)


class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.test_dir) / "test_events.json"
        self.sparse_dir = Path(self.test_dir) / "sparse"
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        # Setup isolated sparse retriever with sample academic and non-academic documents
        self.mock_sparse = SparseRetriever(index_dir=self.sparse_dir)
        sample_chunks = [
            {
                "chunk_id": "c_academic_db",
                "text": "Database Management Systems. Normalization, 1NF, 2NF, 3NF, BCNF, and ACID properties.",
                "page_number": 1,
                "file_name": "dbms_notes.pdf",
                "category": "academic",
                "subject": "database",
            },
            {
                "chunk_id": "c_academic_net",
                "text": "Computer Networks. TCP three-way handshake, SYN, ACK, and reliable transport layer.",
                "page_number": 2,
                "file_name": "networking_notes.pdf",
                "category": "academic",
                "subject": "computer_networks",
            },
            {
                "chunk_id": "c_career_resume",
                "text": "Database software engineer resume with expertise in SQL, normalization, and DBMS.",
                "page_number": 1,
                "file_name": "student_resume.pdf",
                "category": "career",
                "subject": "career",
            },
            {
                "chunk_id": "c_project_repo",
                "text": "Networking project building a custom TCP handshake client in Python.",
                "page_number": 1,
                "file_name": "project_report.pdf",
                "category": "project",
                "subject": "project",
            },
        ]
        self.mock_sparse.index_chunks(sample_chunks)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_a_no_learning_history(self):
        """No learning events returns empty recommendations with guidance message."""
        res = get_learning_recommendations(
            user_id="new_user",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        self.assertEqual(res["recommendations"], [])
        self.assertIn("Not enough learning history", res["message"])

    def test_b_high_weakness_recommendation(self):
        """Low quiz accuracy produces HIGH priority recommendation with focused action."""
        # 10 attempted, 2 correct, 8 incorrect -> failure rate 0.80 -> HIGH weakness
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Normalization",
            questions_attempted=10,
            questions_correct=2,
            questions_incorrect=8,
            user_id="user_1",
            storage_path=self.storage_path,
        )

        res = get_learning_recommendations(
            user_id="user_1",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        recs = res["recommendations"]
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["priority"], "HIGH")
        self.assertEqual(r["topic"], "DBMS")
        self.assertEqual(r["subtopic"], "Normalization")
        self.assertIn("low", r["reason"].lower())
        self.assertIn("focused quiz", r["recommended_action"].lower())
        self.assertIn("dbms_notes.pdf", r["source_documents"])

    def test_c_medium_weakness_recommendation(self):
        """Moderate quiz accuracy produces MEDIUM priority recommendation."""
        # 10 attempted, 6 correct, 4 incorrect -> failure rate 0.40 -> MEDIUM weakness
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Normalization",
            questions_attempted=10,
            questions_correct=6,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.storage_path,
        )

        res = get_learning_recommendations(
            user_id="user_1",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        recs = res["recommendations"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["priority"], "MEDIUM")
        self.assertIn("practice", recs[0]["recommended_action"].lower())

    def test_d_low_weakness_handling(self):
        """LOW weakness is suppressed when higher priority exists, but shown when alone."""
        # User has HIGH in DBMS and LOW in Networks
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Normalization",
            questions_attempted=10,
            questions_correct=2,
            questions_incorrect=8,
            user_id="user_1",
            storage_path=self.storage_path,
        )
        record_quiz_attempt(
            topic="Computer Networks",
            subtopic="TCP Handshake",
            questions_attempted=10,
            questions_correct=9,
            questions_incorrect=1,
            user_id="user_1",
            storage_path=self.storage_path,
        )

        res = get_learning_recommendations(
            user_id="user_1",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        # Low should be filtered out because HIGH is present
        recs = res["recommendations"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["priority"], "HIGH")

        # When alone, LOW can be shown
        clear_res = get_learning_recommendations(
            user_id="user_strong_only",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        record_quiz_attempt(
            topic="Computer Networks",
            subtopic="TCP Handshake",
            questions_attempted=10,
            questions_correct=10,
            questions_incorrect=0,
            user_id="user_strong_only",
            storage_path=self.storage_path,
        )
        strong_res = get_learning_recommendations(
            user_id="user_strong_only",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        self.assertEqual(len(strong_res["recommendations"]), 1)
        self.assertEqual(strong_res["recommendations"][0]["priority"], "LOW")

    def test_e_insufficient_data_recommendation(self):
        """Interactions exist but zero quiz attempts -> diagnostic assessment recommendation."""
        record_learning_event(
            query="Explain BFS algorithm",
            topic="Data Structures & Algorithms",
            subtopic="BFS / Graph Traversal",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.storage_path,
        )

        res = get_learning_recommendations(
            user_id="user_1",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        recs = res["recommendations"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["priority"], "INSUFFICIENT_DATA")
        self.assertIn("diagnostic quiz", recs[0]["recommended_action"].lower())

    def test_f_multiple_topics_sorting(self):
        """Recommendations sorted strictly by priority: HIGH -> MEDIUM -> INSUFFICIENT_DATA."""
        # 1. Medium
        record_quiz_attempt(
            topic="OS",
            subtopic="Deadlock",
            questions_attempted=10,
            questions_correct=6,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.storage_path,
        )
        # 2. High
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Transactions",
            questions_attempted=10,
            questions_correct=1,
            questions_incorrect=9,
            user_id="user_1",
            storage_path=self.storage_path,
        )
        # 3. Insufficient data
        record_learning_event(
            query="What is AVL tree?",
            topic="DSA",
            subtopic="AVL Trees",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.storage_path,
        )

        res = get_learning_recommendations(
            user_id="user_1",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        recs = res["recommendations"]
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0]["priority"], "HIGH")
        self.assertEqual(recs[0]["topic"], "DBMS")
        self.assertEqual(recs[1]["priority"], "MEDIUM")
        self.assertEqual(recs[1]["topic"], "OS")
        self.assertEqual(recs[2]["priority"], "INSUFFICIENT_DATA")
        self.assertEqual(recs[2]["topic"], "DSA")

    def test_g_deduplication(self):
        """Repeated events for the exact same (topic, subtopic) generate only 1 recommendation."""
        record_learning_event(
            query="Explain ACID 1",
            topic="DBMS",
            subtopic="ACID Properties",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.storage_path,
        )
        record_learning_event(
            query="Explain ACID 2",
            topic="DBMS",
            subtopic="ACID Properties",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.storage_path,
        )

        res = get_learning_recommendations(
            user_id="user_1",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        recs = res["recommendations"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["subtopic"], "ACID Properties")

    def test_h_source_filtering_academic_only(self):
        """Only category='academic' documents are matched; career/project never appear."""
        sources = find_academic_sources(
            topic="DBMS",
            subtopic="Normalization",
            sparse_retriever=self.mock_sparse,
        )
        self.assertIn("dbms_notes.pdf", sources)
        self.assertNotIn("student_resume.pdf", sources)
        self.assertNotIn("project_report.pdf", sources)

    def test_i_missing_source_no_fabrication(self):
        """If concept is not in academic corpus, source_documents is empty without fabricating names."""
        sources = find_academic_sources(
            topic="Physics",
            subtopic="Quantum Computing",
            sparse_retriever=self.mock_sparse,
        )
        self.assertEqual(sources, [])

    def test_j_max_recommendations_limit(self):
        """max_recommendations limits returned recommendations."""
        for i in range(5):
            record_learning_event(
                query=f"topic {i}",
                topic="TopicGroup",
                subtopic=f"Subtopic {i}",
                intent="EXPLAIN",
                status="ANSWERED",
                user_id="user_1",
                storage_path=self.storage_path,
            )

        res = get_learning_recommendations(
            user_id="user_1",
            max_recommendations=2,
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        self.assertEqual(len(res["recommendations"]), 2)

    def test_k_topic_filtering(self):
        """Filtering by topic restricts recommendations to that topic."""
        record_learning_event(
            query="DBMS",
            topic="DBMS",
            subtopic="SQL",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.storage_path,
        )
        record_learning_event(
            query="Net",
            topic="Networks",
            subtopic="Routing",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.storage_path,
        )

        res = get_learning_recommendations(
            user_id="user_1",
            topic="DBMS",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        recs = res["recommendations"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["topic"], "DBMS")

    def test_l_user_isolation(self):
        """User A events must not influence User B recommendations."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Transactions",
            questions_attempted=10,
            questions_correct=1,
            questions_incorrect=9,
            user_id="user_a",
            storage_path=self.storage_path,
        )

        res_b = get_learning_recommendations(
            user_id="user_b",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        self.assertEqual(res_b["recommendations"], [])

    def test_m_get_learning_profile_structure(self):
        """get_learning_profile returns structured weak, strong, and insufficient data sections."""
        # Strong
        record_quiz_attempt(
            topic="Networks",
            subtopic="TCP",
            questions_attempted=5,
            questions_correct=5,
            questions_incorrect=0,
            user_id="user_profile",
            storage_path=self.storage_path,
        )
        # Weak
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Normalization",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_profile",
            storage_path=self.storage_path,
        )
        # Insufficient
        record_learning_event(
            query="Explain BFS",
            topic="DSA",
            subtopic="BFS",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_profile",
            storage_path=self.storage_path,
        )

        prof = get_learning_profile(
            user_id="user_profile",
            storage_path=self.storage_path,
            sparse_retriever=self.mock_sparse,
        )
        self.assertEqual(len(prof["weak_topics"]), 1)
        self.assertEqual(prof["weak_topics"][0]["topic"], "DBMS")
        self.assertEqual(len(prof["strong_topics"]), 1)
        self.assertEqual(prof["strong_topics"][0]["topic"], "Networks")
        self.assertEqual(len(prof["insufficient_data_topics"]), 1)
        self.assertEqual(prof["insufficient_data_topics"][0]["topic"], "DSA")
        self.assertEqual(len(prof["recommendations"]), 2)  # HIGH and INSUFFICIENT_DATA (LOW suppressed)


if __name__ == "__main__":
    unittest.main()
