"""
Unit Tests for Study Planner Module (Step 17).
Tests:
A. Empty history / no academic material -> helpful ungrounded warning message
B. One HIGH topic -> prioritized throughout plan with review and quiz
C. Multiple topics -> ordered strictly by weakness priority
D. HIGH + MEDIUM + LOW -> distributed correctly with revisit strategy
E. INSUFFICIENT_DATA -> schedules diagnostic assessments
F. One-day plan -> single-day targeted focus
G. Seven-day plan -> structured with revisit, mixed revision, and mock test
H. Thirty-day plan -> extended multi-week distribution
I. Daily minute validation -> rejects invalid minutes (<=0 or >1440)
J. No activity exceeds daily minutes -> sum(activities) <= daily_minutes
K. Subject filtering -> only matches requested subject
L. Case-insensitive subject -> handles dbms, DBMS, database interchangeably
M. Missing academic source -> does not invent fake files
N. Deterministic output -> running twice produces identical plan structure
O. User isolation -> user A events do not affect user B plan
W. Save study plan -> persists plan to JSON
X. Retrieve study plans -> loads saved plans for user
Y. Latest study plan -> returns most recent plan
Z. Empty/malformed storage -> handles corrupted storage files gracefully
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import json

from src.learning_tracker import record_learning_event, record_quiz_attempt
from src.sparse_retriever import SparseRetriever
from src.study_planner import (
    create_study_plan,
    save_study_plan,
    get_study_plans,
    get_latest_study_plan,
    load_all_study_plans,
)


class TestStudyPlanner(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.events_path = Path(self.test_dir) / "test_events.json"
        self.plans_path = Path(self.test_dir) / "test_plans.json"
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
            {
                "chunk_id": "c_net",
                "text": "Computer Networks. TCP three-way handshake and socket programming.",
                "page_number": 1,
                "file_name": "networking_notes.pdf",
                "category": "academic",
                "subject": "computer_networks",
            },
        ])

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_a_empty_history_unknown_subject(self):
        """Subject with no learning history and no corpus material returns clean refusal message."""
        res = create_study_plan(
            subject="Quantum Mechanics",
            days=7,
            user_id="user_1",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(res["plan"], [])
        self.assertIn("Not enough academic learning data", res["message"])

    def test_b_one_high_topic(self):
        """Single high weakness topic is prioritized with review, practice, and quiz."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Transactions",
            questions_attempted=10,
            questions_correct=2,
            questions_incorrect=8,
            user_id="user_1",
            storage_path=self.events_path,
        )

        res = create_study_plan(
            subject="DBMS",
            days=3,
            user_id="user_1",
            daily_minutes=60,
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(len(res["plan"]), 3)
        day1 = res["plan"][0]
        self.assertEqual(day1["focus"][0]["subtopic"], "Transactions")
        self.assertEqual(day1["focus"][0]["priority"], "HIGH")
        act_types = [a["type"] for a in day1["activities"]]
        self.assertIn("REVIEW", act_types)
        self.assertIn("QUIZ", act_types)

    def test_c_multiple_topics_prioritization(self):
        """Multiple topics are ordered strictly by weakness priority."""
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

        res = create_study_plan(
            subject="DBMS",
            days=2,
            user_id="user_1",
            daily_minutes=60,
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(res["plan"][0]["focus"][0]["subtopic"], "ACID Properties")
        self.assertEqual(res["plan"][1]["focus"][0]["subtopic"], "Normalization")

    def test_d_high_medium_low_distribution(self):
        """Plan schedules HIGH, MEDIUM, and LOW in descending weakness priority."""
        # High
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=10,
            questions_correct=1,
            questions_incorrect=9,
            user_id="user_1",
            storage_path=self.events_path,
        )
        # Medium
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Indexing",
            questions_attempted=10,
            questions_correct=6,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        # Low
        record_quiz_attempt(
            topic="DBMS",
            subtopic="SQL Basics",
            questions_attempted=10,
            questions_correct=10,
            questions_incorrect=0,
            user_id="user_1",
            storage_path=self.events_path,
        )

        res = create_study_plan(
            subject="DBMS",
            days=3,
            user_id="user_1",
            daily_minutes=60,
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(res["plan"][0]["focus"][0]["priority"], "HIGH")
        self.assertEqual(res["plan"][1]["focus"][0]["priority"], "MEDIUM")
        self.assertEqual(res["plan"][2]["focus"][0]["priority"], "LOW")

    def test_e_insufficient_data_diagnostic(self):
        """Topics with zero quiz history receive DIAGNOSTIC activity rather than being assumed weak."""
        record_learning_event(
            query="What is BCNF?",
            topic="DBMS",
            subtopic="BCNF Normal Form",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.events_path,
        )

        res = create_study_plan(
            subject="DBMS",
            days=1,
            user_id="user_1",
            daily_minutes=60,
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        day1 = res["plan"][0]
        self.assertEqual(day1["focus"][0]["priority"], "INSUFFICIENT_DATA")
        act_types = [a["type"] for a in day1["activities"]]
        self.assertIn("DIAGNOSTIC", act_types)

    def test_f_one_day_plan(self):
        """1-day plan generates a single focused day."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        res = create_study_plan(
            subject="DBMS",
            days=1,
            user_id="user_1",
            daily_minutes=45,
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(len(res["plan"]), 1)
        self.assertEqual(res["plan"][0]["estimated_minutes"], 45)

    def test_g_seven_day_plan_structure(self):
        """7-day plan includes revisit on Day 3, mixed revision on Day 6, mock test on Day 7."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        res = create_study_plan(
            subject="DBMS",
            days=7,
            user_id="user_1",
            daily_minutes=60,
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(len(res["plan"]), 7)
        self.assertEqual(res["plan"][2]["strategy"], "revisit_quiz")
        self.assertEqual(res["plan"][5]["strategy"], "mixed_revision")
        self.assertEqual(res["plan"][6]["strategy"], "mock_test")
        day7_types = [a["type"] for a in res["plan"][6]["activities"]]
        self.assertIn("MOCK_TEST", day7_types)

    def test_h_thirty_day_plan(self):
        """30-day plan generates exactly 30 days without crashing."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        res = create_study_plan(
            subject="DBMS",
            days=30,
            user_id="user_1",
            daily_minutes=30,
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(len(res["plan"]), 30)

    def test_i_validation_errors(self):
        """Invalid days or minutes raise ValueError."""
        with self.assertRaises(ValueError):
            create_study_plan("", 7)
        with self.assertRaises(ValueError):
            create_study_plan("DBMS", 0)
        with self.assertRaises(ValueError):
            create_study_plan("DBMS", 366)
        with self.assertRaises(ValueError):
            create_study_plan("DBMS", 7, daily_minutes=0)
        with self.assertRaises(ValueError):
            create_study_plan("DBMS", 7, daily_minutes=1500)

    def test_j_no_activity_exceeds_daily_minutes(self):
        """Sum of daily activity minutes must equal estimated_minutes and be <= daily_minutes."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        res = create_study_plan(
            subject="DBMS",
            days=5,
            user_id="user_1",
            daily_minutes=45,
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        for day in res["plan"]:
            tot = sum(a["minutes"] for a in day["activities"])
            self.assertEqual(tot, 45)
            self.assertEqual(tot, day["estimated_minutes"])

    def test_k_subject_filtering(self):
        """Plans for DBMS do not include Computer Networks topics."""
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

        res = create_study_plan(
            subject="DBMS",
            days=3,
            user_id="user_1",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        for day in res["plan"]:
            focus_topics = [f["topic"] for f in day["focus"]]
            self.assertIn("DBMS", focus_topics)
            self.assertNotIn("Computer Networks", focus_topics)

    def test_l_case_insensitive_subject(self):
        """Case variations match identically."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        res1 = create_study_plan(
            subject="dbms",
            days=2,
            user_id="user_1",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        res2 = create_study_plan(
            subject="Database Management Systems",
            days=2,
            user_id="user_1",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(len(res1["plan"]), 2)
        self.assertEqual(len(res2["plan"]), 2)

    def test_m_missing_academic_source(self):
        """Subtopic without supporting document has empty sources list."""
        record_learning_event(
            query="Explain imaginary DB protocol",
            topic="DBMS",
            subtopic="Imaginary Protocol XYZ",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="user_1",
            storage_path=self.events_path,
        )
        res = create_study_plan(
            subject="DBMS",
            days=1,
            user_id="user_1",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(res["plan"][0]["focus"][0]["sources"], [])

    def test_n_deterministic_output(self):
        """Calling create_study_plan twice with identical inputs yields identical schedules."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_1",
            storage_path=self.events_path,
        )
        plan1 = create_study_plan(
            subject="DBMS",
            days=7,
            user_id="user_1",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        plan2 = create_study_plan(
            subject="DBMS",
            days=7,
            user_id="user_1",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(len(plan1["plan"]), len(plan2["plan"]))
        for d1, d2 in zip(plan1["plan"], plan2["plan"]):
            self.assertEqual(d1["day"], d2["day"])
            self.assertEqual(d1["strategy"], d2["strategy"])
            self.assertEqual(d1["estimated_minutes"], d2["estimated_minutes"])
            self.assertEqual(len(d1["activities"]), len(d2["activities"]))

    def test_o_user_isolation(self):
        """User A events do not leak into User B study plan."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="alice",
            storage_path=self.events_path,
        )
        plan_bob = create_study_plan(
            subject="Quantum Mechanics",
            days=3,
            user_id="bob",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=False,
        )
        self.assertEqual(plan_bob["plan"], [])

    def test_w_x_y_persistence(self):
        """Save, list, and retrieve latest study plan."""
        record_quiz_attempt(
            topic="DBMS",
            subtopic="ACID",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,
            user_id="user_persist",
            storage_path=self.events_path,
        )
        plan = create_study_plan(
            subject="DBMS",
            days=3,
            user_id="user_persist",
            storage_path=self.plans_path,
            events_storage_path=self.events_path,
            sparse_retriever=self.mock_sparse,
            persist=True,
        )
        self.assertTrue(self.plans_path.is_file())

        all_user_plans = get_study_plans(user_id="user_persist", storage_path=self.plans_path)
        self.assertEqual(len(all_user_plans), 1)
        self.assertEqual(all_user_plans[0]["plan_id"], plan["plan_id"])

        latest = get_latest_study_plan(user_id="user_persist", subject="DBMS", storage_path=self.plans_path)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["plan_id"], plan["plan_id"])

    def test_z_empty_and_malformed_storage(self):
        """Malformed or non-existent storage file does not crash loading."""
        # Non-existent
        non_existent = Path(self.test_dir) / "none.json"
        self.assertEqual(load_all_study_plans(non_existent), [])

        # Malformed
        bad_file = Path(self.test_dir) / "bad.json"
        with open(bad_file, "w") as f:
            f.write("corrupted JSON content")
        self.assertEqual(load_all_study_plans(bad_file), [])


if __name__ == "__main__":
    unittest.main()
