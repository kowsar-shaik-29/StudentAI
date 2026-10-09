import unittest
from src.job_matcher import match_resume_to_job


class TestJobMatcher(unittest.TestCase):

    def test_zero_requirements(self):
        res = match_resume_to_job(
            resume_skills=[{"name": "Python", "normalized_name": "python"}],
            job_requirements=[]
        )
        self.assertEqual(res["requirement_count"], 0)
        self.assertEqual(res["matched_count"], 0)
        self.assertEqual(res["coverage_percentage"], 0.0)
        self.assertEqual(res["matched_skills"], [])
        self.assertEqual(res["missing_skills"], [])

    def test_exact_matches(self):
        resume_skills = [
            {"name": "Java", "normalized_name": "java", "evidence_quote": "Proficient in Java"},
            {"name": "SQL", "normalized_name": "sql", "evidence_quote": "Wrote SQL queries"},
            {"name": "React", "normalized_name": "react", "evidence_quote": "Developed React UI"},
        ]
        job_requirements = [
            {"name": "Java", "normalized_name": "java", "evidence_quote": "Requires Java 17+"},
            {"name": "SQL", "normalized_name": "sql", "evidence_quote": "Requires SQL proficiency"},
            {"name": "React", "normalized_name": "react", "evidence_quote": "Frontend in React"},
            {"name": "AWS", "normalized_name": "aws", "evidence_quote": "Experience with AWS"},
            {"name": "Docker", "normalized_name": "docker", "evidence_quote": "Containerization with Docker"},
        ]

        res = match_resume_to_job(resume_skills, job_requirements)
        self.assertEqual(res["requirement_count"], 5)
        self.assertEqual(res["matched_count"], 3)
        self.assertEqual(res["missing_count"], 2)
        self.assertEqual(res["coverage_percentage"], 60.0)

        matched_reqs = {m["requirement"] for m in res["matched_skills"]}
        self.assertEqual(matched_reqs, {"Java", "SQL", "React"})

        missing_reqs = {m["requirement"] for m in res["missing_skills"]}
        self.assertEqual(missing_reqs, {"AWS", "Docker"})

    def test_aliases(self):
        resume_skills = [
            {"name": "JS", "normalized_name": "js", "evidence_quote": "JS developer"},
            {"name": "Postgres", "normalized_name": "postgres", "evidence_quote": "Postgres DB admin"},
            {"name": "K8s", "normalized_name": "k8s", "evidence_quote": "K8s cluster setup"},
        ]
        job_requirements = [
            {"name": "JavaScript", "normalized_name": "javascript", "evidence_quote": "Expert in JavaScript"},
            {"name": "PostgreSQL", "normalized_name": "postgresql", "evidence_quote": "PostgreSQL relational database"},
            {"name": "Kubernetes", "normalized_name": "kubernetes", "evidence_quote": "Kubernetes deployment"},
        ]

        res = match_resume_to_job(resume_skills, job_requirements)
        self.assertEqual(res["requirement_count"], 3)
        self.assertEqual(res["matched_count"], 3)
        self.assertEqual(res["coverage_percentage"], 100.0)

    def test_no_false_positive_java_javascript(self):
        # Java and JavaScript must NEVER match
        resume_skills = [
            {"name": "Java", "normalized_name": "java", "evidence_quote": "Core Java developer"},
        ]
        job_requirements = [
            {"name": "JavaScript", "normalized_name": "javascript", "evidence_quote": "JavaScript frontend"},
        ]

        res = match_resume_to_job(resume_skills, job_requirements)
        self.assertEqual(res["matched_count"], 0)
        self.assertEqual(res["coverage_percentage"], 0.0)
        self.assertEqual(res["missing_count"], 1)

    def test_partial_matches_not_counted_in_coverage(self):
        resume_skills = [
            {"name": "AWS Lambda", "normalized_name": "aws lambda", "evidence_quote": "Built AWS Lambda functions"},
        ]
        job_requirements = [
            {"name": "AWS", "normalized_name": "aws", "evidence_quote": "AWS cloud architect"},
        ]

        res = match_resume_to_job(resume_skills, job_requirements)
        self.assertEqual(res["matched_count"], 0)
        self.assertEqual(res["partial_count"], 1)
        self.assertEqual(res["coverage_percentage"], 0.0)
        self.assertEqual(len(res["partial_matches"]), 1)
        self.assertEqual(res["partial_matches"][0]["requirement"], "AWS")


if __name__ == "__main__":
    unittest.main()
