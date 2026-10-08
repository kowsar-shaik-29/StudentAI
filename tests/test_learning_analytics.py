"""
Unit Tests for Learning Analytics Module (Step 15).
Tests:
- Deterministic weakness score calculation
- Insufficient data handling (0 quiz attempts -> 0.0 score, INSUFFICIENT_DATA level)
- Weakness levels: LOW (<0.30), MEDIUM (<0.60), HIGH (>=0.60)
- Topic statistics aggregation
- Subtopic statistics aggregation
- Sorting orders (weakest first)
"""

import unittest
from pathlib import Path
import tempfile
import shutil
from src.learning_tracker import (
    record_learning_event,
    record_quiz_attempt,
)
from src.learning_analytics import (
    _compute_weakness,
    get_topic_statistics,
    get_subtopic_statistics,
)


class TestLearningAnalytics(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.test_dir) / "analytics_events.json"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_compute_weakness_insufficient_data(self):
        """0 questions attempted returns 0.0 and INSUFFICIENT_DATA."""
        score, level = _compute_weakness(
            questions_attempted=0,
            questions_incorrect=0,
            interaction_count=5,
            refused_count=2,
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(level, "INSUFFICIENT_DATA")

    def test_compute_weakness_low_level(self):
        """High quiz accuracy yields LOW weakness score (<0.30)."""
        # 10 attempted, 1 incorrect (failure_rate = 0.10), refused = 0
        # weakness = 0.85 * 0.10 + 0 = 0.085 -> LOW
        score, level = _compute_weakness(
            questions_attempted=10,
            questions_incorrect=1,
            interaction_count=5,
            refused_count=0,
        )
        self.assertAlmostEqual(score, 0.085, places=3)
        self.assertEqual(level, "LOW")

    def test_compute_weakness_medium_level(self):
        """Moderate failure yields MEDIUM weakness score (0.30 - 0.59)."""
        # 10 attempted, 4 incorrect (failure_rate = 0.40), refused = 1 of 5 (0.20)
        # weakness = 0.85 * 0.40 + 0.15 * 0.20 = 0.34 + 0.03 = 0.37 -> MEDIUM
        score, level = _compute_weakness(
            questions_attempted=10,
            questions_incorrect=4,
            interaction_count=5,
            refused_count=1,
        )
        self.assertAlmostEqual(score, 0.37, places=2)
        self.assertEqual(level, "MEDIUM")

    def test_compute_weakness_high_level(self):
        """High failure rate yields HIGH weakness score (>=0.60)."""
        # 10 attempted, 8 incorrect (failure_rate = 0.80), refused = 2 of 4 (0.50)
        # weakness = 0.85 * 0.80 + 0.15 * 0.50 = 0.68 + 0.075 = 0.755 -> HIGH
        score, level = _compute_weakness(
            questions_attempted=10,
            questions_incorrect=8,
            interaction_count=4,
            refused_count=2,
        )
        self.assertAlmostEqual(score, 0.755, places=3)
        self.assertEqual(level, "HIGH")

    def test_topic_statistics_aggregation(self):
        """Topic statistics accurately aggregate and sort weak topics first."""
        # Topic 1: DBMS (Weak)
        record_learning_event(
            query="What is 2NF?",
            topic="DBMS",
            subtopic="Normalization",
            intent="EXPLAIN",
            status="ANSWERED",
            storage_path=self.storage_path,
        )
        record_quiz_attempt(
            topic="DBMS",
            subtopic="Normalization",
            questions_attempted=5,
            questions_correct=1,
            questions_incorrect=4,  # failure_rate = 0.80
            storage_path=self.storage_path,
        )

        # Topic 2: Computer Networks (Strong)
        record_learning_event(
            query="Explain TCP",
            topic="Computer Networks",
            subtopic="TCP",
            intent="EXPLAIN",
            status="ANSWERED",
            storage_path=self.storage_path,
        )
        record_quiz_attempt(
            topic="Computer Networks",
            subtopic="TCP",
            questions_attempted=5,
            questions_correct=5,
            questions_incorrect=0,  # failure_rate = 0.0
            storage_path=self.storage_path,
        )

        # Topic 3: Operating Systems (Unquizzed)
        record_learning_event(
            query="What is a process?",
            topic="Operating Systems",
            subtopic="Processes",
            intent="EXPLAIN",
            status="ANSWERED",
            storage_path=self.storage_path,
        )

        stats = get_topic_statistics(storage_path=self.storage_path)
        self.assertEqual(len(stats), 3)

        # DBMS should be ranked #1 weakest
        self.assertEqual(stats[0]["topic"], "DBMS")
        self.assertEqual(stats[0]["weakness_level"], "HIGH")
        self.assertEqual(stats[0]["questions_attempted"], 5)
        self.assertEqual(stats[0]["questions_correct"], 1)
        self.assertEqual(stats[0]["quiz_accuracy"], 0.20)

        # Operating Systems should have INSUFFICIENT_DATA
        os_stat = next(s for s in stats if s["topic"] == "Operating Systems")
        self.assertEqual(os_stat["weakness_level"], "INSUFFICIENT_DATA")
        self.assertIsNone(os_stat["quiz_accuracy"])

    def test_subtopic_statistics_filtering(self):
        """Subtopic statistics can be filtered by topic."""
        record_learning_event(
            query="What is 2NF?",
            topic="DBMS",
            subtopic="Normalization",
            intent="EXPLAIN",
            status="ANSWERED",
            storage_path=self.storage_path,
        )
        record_learning_event(
            query="What is ACID?",
            topic="DBMS",
            subtopic="Transactions",
            intent="EXPLAIN",
            status="ANSWERED",
            storage_path=self.storage_path,
        )
        record_learning_event(
            query="Explain TCP",
            topic="Computer Networks",
            subtopic="Transport Layer",
            intent="EXPLAIN",
            status="ANSWERED",
            storage_path=self.storage_path,
        )

        dbms_sub = get_subtopic_statistics(topic="DBMS", storage_path=self.storage_path)
        self.assertEqual(len(dbms_sub), 2)
        subtopics = {s["subtopic"] for s in dbms_sub}
        self.assertEqual(subtopics, {"Normalization", "Transactions"})


if __name__ == "__main__":
    unittest.main()
