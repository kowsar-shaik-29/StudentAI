"""
Unit Tests for Evaluator and Generation Evaluator Modules (Step 11).
Tests deterministic metrics calculation: Recall@1, Recall@3, Recall@5, MRR,
Document Recall@5, unanswerable question handling, and empty/malformed dataset handling.
"""

import unittest
from src.evaluator import calculate_retrieval_metrics
from src.generation_evaluator import check_generation_response


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.sample_eval_dataset = [
            {
                "id": "q1",
                "question": "What is star topology?",
                "expected_documents": ["networking.pdf"],
                "expected_pages": [1],
                "expected_chunk_ids": ["net_p1_c0", "net_p1_c1"],
                "answerable": True,
            },
            {
                "id": "q2",
                "question": "What is a primary key?",
                "expected_documents": ["db.pdf"],
                "expected_pages": [1],
                "expected_chunk_ids": ["db_p1_c0"],
                "answerable": True,
            },
            {
                "id": "q3",
                "question": "What is quantum computing?",
                "expected_documents": [],
                "expected_pages": [],
                "expected_chunk_ids": [],
                "answerable": False,
            },
        ]

    def test_perfect_retrieval_metrics(self):
        # q1 hit at rank 1, q2 hit at rank 1, q3 unanswerable
        predictions = [
            [{"chunk_id": "net_p1_c0", "file_name": "networking.pdf"}],
            [{"chunk_id": "db_p1_c0", "file_name": "db.pdf"}],
            [{"chunk_id": "net_p1_c0", "file_name": "networking.pdf"}],
        ]

        metrics = calculate_retrieval_metrics(self.sample_eval_dataset, predictions)

        self.assertEqual(metrics["total_answerable"], 2)
        self.assertAlmostEqual(metrics["recall_at_1"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_3"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_5"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(metrics["doc_recall_at_5"], 1.0)

    def test_partial_and_ranked_mrr(self):
        # q1 hit at rank 2 (RR = 0.5), q2 hit at rank 4 (RR = 0.25)
        # MRR = (0.5 + 0.25) / 2 = 0.375
        predictions = [
            [
                {"chunk_id": "other_c1", "file_name": "other.pdf"},
                {"chunk_id": "net_p1_c0", "file_name": "networking.pdf"},
            ],
            [
                {"chunk_id": "other_c1", "file_name": "other.pdf"},
                {"chunk_id": "other_c2", "file_name": "other.pdf"},
                {"chunk_id": "other_c3", "file_name": "other.pdf"},
                {"chunk_id": "db_p1_c0", "file_name": "db.pdf"},
            ],
            [{"chunk_id": "other_c1", "file_name": "other.pdf"}],
        ]

        metrics = calculate_retrieval_metrics(self.sample_eval_dataset, predictions)

        self.assertEqual(metrics["total_answerable"], 2)
        self.assertAlmostEqual(metrics["recall_at_1"], 0.0)
        self.assertAlmostEqual(metrics["recall_at_3"], 0.5)  # only q1 is in top 3
        self.assertAlmostEqual(metrics["recall_at_5"], 1.0)  # both are in top 5
        self.assertAlmostEqual(metrics["mrr"], 0.375)
        self.assertAlmostEqual(metrics["doc_recall_at_5"], 1.0)

    def test_no_hits_metrics(self):
        predictions = [
            [{"chunk_id": "irrelevant_1", "file_name": "wrong.pdf"}],
            [{"chunk_id": "irrelevant_2", "file_name": "wrong.pdf"}],
            [{"chunk_id": "irrelevant_3", "file_name": "wrong.pdf"}],
        ]

        metrics = calculate_retrieval_metrics(self.sample_eval_dataset, predictions)

        self.assertAlmostEqual(metrics["recall_at_1"], 0.0)
        self.assertAlmostEqual(metrics["recall_at_3"], 0.0)
        self.assertAlmostEqual(metrics["recall_at_5"], 0.0)
        self.assertAlmostEqual(metrics["mrr"], 0.0)
        self.assertAlmostEqual(metrics["doc_recall_at_5"], 0.0)

    def test_empty_dataset(self):
        metrics = calculate_retrieval_metrics([], [])
        self.assertEqual(metrics["total_answerable"], 0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_generation_answerable_pass(self):
        item = self.sample_eval_dataset[0]
        rag_response = {
            "answer": "Star topology connects devices to a central hub [Page 1].",
            "sources": [{"file_name": "networking.pdf", "page_number": 1}],
        }
        check = check_generation_response(item, rag_response)
        self.assertTrue(check["passed"])
        self.assertTrue(check["has_citation"])
        self.assertTrue(check["citation_matches_expected_page"])
        self.assertTrue(check["expected_doc_in_sources"])

    def test_generation_citation_mismatch_fails(self):
        item = self.sample_eval_dataset[0]  # expected page 1
        rag_response = {
            "answer": "Star topology connects devices to a central hub [Page 9].",
            "sources": [{"file_name": "networking.pdf", "page_number": 9}],
        }
        check = check_generation_response(item, rag_response)
        self.assertFalse(check["passed"])
        self.assertFalse(check["citation_matches_expected_page"])

    def test_generation_unanswerable_refusal_pass(self):
        unanswerable_item = self.sample_eval_dataset[2]
        rag_response = {
            "answer": "I couldn't find the answer in the provided documents.",
            "sources": [],
        }
        check = check_generation_response(unanswerable_item, rag_response)
        self.assertTrue(check["passed"])
        self.assertTrue(check["refusal_correct"])

    def test_generation_unanswerable_hallucination_fails(self):
        unanswerable_item = self.sample_eval_dataset[2]
        rag_response = {
            "answer": "Quantum networking uses quantum entanglement to transmit qubits.",
            "sources": [],
        }
        check = check_generation_response(unanswerable_item, rag_response)
        self.assertFalse(check["passed"])
        self.assertFalse(check["refusal_correct"])


if __name__ == "__main__":
    unittest.main()
