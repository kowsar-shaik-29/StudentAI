"""
Unit Tests for Revision Engine Module (Step 17).
Tests:
P. High weakness prioritized
Q. Medium weakness
R. Low weakness
S. Insufficient data diagnostic activity
T. Subject filtering
U. Limit respected
V. User isolation
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from src.learning_tracker import record_learning_event, record_quiz_attempt
from src.sparse_retriever import SparseRetriever
from src.revision_engine import get_revision_recommendations


class TestRevisionEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.events_path = Path(self.test_dir) / "test_events.json"
        self.sparse_dir = Path(self.test_dir) / "sparse"
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        self.mock_sparse = SparseRetriever(index_dir=self.sparse_dir)
        self.mock_sparse.index_chunks([
            {
                "chunk_id": "c_dbms",
                "text": "Database Management Systems. Normalization, transactions, and ACID properties.",
                "page_number": 1,
                "file_name": "dbms_notes.pdf",
                "category": "academic",
                "subject": "database",
            },
        ])

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_p_high_weakness_prioritized(self):
        """High weakness topic is prioritized at Rank 1 with review + quiz activity."""
        # Medium
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Normalization",
            questions_attempted=10,
            questions_correct=6,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        # High
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID Properties",
            questions_attempted=10,
            questions_correct=1,
            questions_incorrect=9,
            user_id="user_1",
            storage_path=self.events_path,
        )

        res = get_revision_recommendations(
            user_id="user_1",
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
        )
        items = res["revision_items"]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["subtopic"], "ACID Properties")
        self.assertEqual(items[0]["priority"], "HIGH")
        self.assertIn("targeted quiz", items[0]["recommended_activity"].lower())

    def test_q_medium_weakness(self):
        """Medium weakness specifies review and practice exercises."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Normalization",
            questions_attempted=10,
            questions_correct=6,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )

        res = get_revision_recommendations(
            user_id="user_1",
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
        )
        items = res["revision_items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["priority"], "MEDIUM")
        self.assertIn("practice", items[0]["recommended_activity"].lower())

    def test_r_low_weakness(self):
        """Low weakness specifies periodic maintenance revision."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="SQL Basics",
            questions_attempted=10,
            questions_correct=10,
            questions_incorrect=0,
            user_id="user_1",
            storage_path=self.events_path,
        )

        res = get_revision_recommendations(
            user_id="user_1",
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
        )
        items = res["revision_items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["priority"], "LOW")
        self.assertIn("active recall", items[0]["recommended_activity"].lower())

    def test_s_insufficient_data(self):
        """Insufficient data recommends diagnostic assessment."""
        record_learning_event(
            query="What is 3NF?",
            topic="DBMS",
            subtopic="3NF Form",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.events_path,
        )

        res = get_revision_recommendations(
            user_id="user_1",
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
        )
        items = res["revision_items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["priority"], "INSUFFICIENT_DATA")
        self.assertIn("diagnostic", items[0]["recommended_activity"].lower())

    def test_t_subject_filtering(self):
        """Subject filtering isolates revision items to that subject."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        record_quiz_attempt(
            topic="Computer Networks",
            subtopic="TCP",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )

        res = get_revision_recommendations(
            user_id="user_1",
            subject="DBMS",
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
        )
        items = res["revision_items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["topic"], "DBMS")

    def test_u_limit_respected(self):
        """Limit parameter bounds the returned list."""
        for i in range(5):
            record_quiz_attempt(
                topic="DBMS",
                subtopic=f"Topic {i}",
                questions_attempted=5,
                questions_correct=1,
                questions_incorrect=4,
                user_id="user_1",
                storage_path=self.events_path,
            )

        res = get_revision_recommendations(
            user_id="user_1",
            limit=2,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
        )
        self.assertEqual(len(res["revision_items"]), 2)

    def test_v_user_isolation(self):
        """User A events do not leak into User B revision recommendations."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="alice",
            storage_path=self.events_path,
        )
        res_bob = get_revision_recommendations(
            user_id="bob",
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
        )
        self.assertEqual(res_bob["revision_items"], [])


if __name__ == "__main__":
    unittest.main()
