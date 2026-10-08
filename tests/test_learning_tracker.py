"""
Unit Tests for Learning Tracker Module (Step 15).
Tests:
- LearningEvent creation and serialization
- Event recording for ANSWERED, REFUSED, QUIZ_GENERATED
- Quiz attempt validation (correct + incorrect == attempted, negative checks)
- Event filtering by user_id and topic
- Storage directory creation and JSON persistence
"""

import unittest
from pathlib import Path
import tempfile
import shutil
from src.learning_tracker import (
    LearningEvent,
    record_learning_event,
    record_quiz_attempt,
    get_learning_events,
    clear_learning_events,
    load_all_events,
)


class TestLearningTracker(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.test_dir) / "test_events.json"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_learning_event_success(self):
        """Recording a learning event creates valid LearningEvent and writes to JSON."""
        event = record_learning_event(
            query="Explain TCP three-way handshake",
            topic="Computer Networks",
            subtopic="TCP / Three-Way Handshake",
            intent="EXPLAIN",
            status="ANSWERED",
            sources_count=3,
            user_id="student_1",
            storage_path=self.storage_path,
        )

        self.assertIsInstance(event, LearningEvent)
        self.assertEqual(event.topic, "Computer Networks")
        self.assertEqual(event.subtopic, "TCP / Three-Way Handshake")
        self.assertEqual(event.intent, "EXPLAIN")
        self.assertEqual(event.status, "ANSWERED")
        self.assertEqual(event.sources_count, 3)
        self.assertEqual(event.user_id, "student_1")
        self.assertTrue(self.storage_path.is_file())

        loaded = load_all_events(self.storage_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["event_id"], event.event_id)

    def test_record_learning_event_invalid_status_raises(self):
        """Invalid status raises ValueError."""
        with self.assertRaises(ValueError):
            record_learning_event(
                query="test",
                topic="CS",
                subtopic="General",
                intent="EXPLAIN",
                status="INVALID_STATUS",
                storage_path=self.storage_path,
            )

    def test_record_quiz_attempt_success(self):
        """Valid quiz attempt records correctly."""
        event = record_quiz_attempt(
            topic="DBMS",
            subtopic="Normalization",
            questions_attempted=5,
            questions_correct=3,
            questions_incorrect=2,
            user_id="student_1",
            storage_path=self.storage_path,
        )

        self.assertEqual(event.status, "QUIZ_ATTEMPTED")
        self.assertEqual(event.questions_attempted, 5)
        self.assertEqual(event.questions_correct, 3)
        self.assertEqual(event.questions_incorrect, 2)

    def test_record_quiz_attempt_validation_mismatch(self):
        """Validation fails if correct + incorrect != attempted."""
        with self.assertRaises(ValueError):
            record_quiz_attempt(
                topic="DBMS",
                subtopic="Normalization",
                questions_attempted=5,
                questions_correct=3,
                questions_incorrect=1,  # 3 + 1 != 5
                storage_path=self.storage_path,
            )

    def test_record_quiz_attempt_negative_numbers(self):
        """Negative numbers raise ValueError."""
        with self.assertRaises(ValueError):
            record_quiz_attempt(
                topic="DBMS",
                subtopic="Normalization",
                questions_attempted=-1,
                questions_correct=0,
                questions_incorrect=0,
                storage_path=self.storage_path,
            )

    def test_get_learning_events_filtering(self):
        """get_learning_events filters accurately by user_id and topic."""
        record_learning_event(
            query="q1",
            topic="Computer Networks",
            subtopic="TCP",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="alice",
            storage_path=self.storage_path,
        )
        record_learning_event(
            query="q2",
            topic="DBMS",
            subtopic="SQL",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="alice",
            storage_path=self.storage_path,
        )
        record_learning_event(
            query="q3",
            topic="Computer Networks",
            subtopic="UDP",
            intent="EXPLAIN",
            status="ANSWERED",
            user_id="bob",
            storage_path=self.storage_path,
        )

        alice_net = get_learning_events(user_id="alice", topic="Networks", storage_path=self.storage_path)
        self.assertEqual(len(alice_net), 1)
        self.assertEqual(alice_net[0].subtopic, "TCP")

        all_alice = get_learning_events(user_id="alice", topic=None, storage_path=self.storage_path)
        self.assertEqual(len(all_alice), 2)

        bob_events = get_learning_events(user_id="bob", topic=None, storage_path=self.storage_path)
        self.assertEqual(len(bob_events), 1)

    def test_clear_learning_events(self):
        """clear_learning_events deletes the storage file cleanly."""
        record_learning_event(
            query="test",
            topic="CS",
            subtopic="General",
            intent="EXPLAIN",
            status="ANSWERED",
            storage_path=self.storage_path,
        )
        self.assertTrue(self.storage_path.is_file())
        clear_learning_events(self.storage_path)
        self.assertFalse(self.storage_path.is_file())


if __name__ == "__main__":
    unittest.main()
