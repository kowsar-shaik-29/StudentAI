import unittest
from src.citation_verifier import verify_citations


class TestCitationVerifier(unittest.TestCase):

    def test_empty_answer(self):
        res = verify_citations("", sources=[])
        self.assertTrue(res["is_valid"])
        self.assertFalse(res["has_citations"])

    def test_valid_citations(self):
        answer = "Star topology connects all nodes to a central hub [Page 1]. In bus topology, nodes connect to a single backbone [Page 2]."
        sources = [
            {"page_number": 1, "file_name": "networking.pdf"},
            {"page_number": 2, "file_name": "networking.pdf"},
        ]
        res = verify_citations(answer, sources=sources)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["cited_pages"], [1, 2])
        self.assertEqual(res["valid_pages"], [1, 2])
        self.assertEqual(res["invalid_pages"], [])
        self.assertEqual(len(res["errors"]), 0)

    def test_invalid_nonexistent_page_citation(self):
        answer = "The relational model was defined by Codd [Page 99]."
        sources = [
            {"page_number": 1, "file_name": "database.pdf"},
            {"page_number": 2, "file_name": "database.pdf"},
        ]
        res = verify_citations(answer, sources=sources)
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["cited_pages"], [99])
        self.assertEqual(res["invalid_pages"], [99])
        self.assertIn("Answer cites pages not present", res["errors"][0])

    def test_malformed_citation_syntax(self):
        answer = "Normalization reduces redundancy [page ] and [Page abc]."
        sources = [
            {"page_number": 1, "file_name": "database.pdf"},
        ]
        res = verify_citations(answer, sources=sources)
        self.assertFalse(res["is_valid"])
        self.assertIn("Malformed citation syntax", res["errors"][0])

    def test_standard_refusal_without_citations_is_valid(self):
        answer = "I couldn't find the answer in the provided documents."
        sources = [
            {"page_number": 1, "file_name": "database.pdf"},
        ]
        res = verify_citations(answer, sources=sources)
        self.assertTrue(res["is_valid"])
        self.assertEqual(len(res["warnings"]), 0)


if __name__ == "__main__":
    unittest.main()
