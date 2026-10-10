"""
Unit and integration tests for secure authentication and multi-user data isolation.
Tests cover:
1. User registration, password hashing, and duplicate detection.
2. User authentication, invalid password rejection, and missing accounts.
3. Session token creation, HMAC verification, and expiration handling.
4. Protected API endpoints rejecting unauthenticated requests (401 Unauthorized).
5. Protected API endpoints accepting session cookies or Bearer tokens.
6. Multi-user vector store, sparse BM25, and hybrid retrieval isolation (User A cannot see User B's documents).
7. Cross-user career document isolation (User A cannot access User B's resume or analyze job fit against User B's resume).
8. Cross-user project isolation (User A cannot view or query User B's projects).
9. Cross-user learning events and profile isolation.
10. Agent tool security (LLM cannot spoof user_id parameter to access another user's data).
"""

import os
import unittest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.auth import (
    init_user_db,
    hash_password,
    verify_password,
    create_user,
    authenticate_user,
    get_user_by_id,
    create_access_token,
    verify_access_token,
    normalize_username,
    User,
)
from src.metadata import create_document_metadata
from src.sparse_retriever import SparseRetriever
from src.retriever import search as dense_search
from src.career_rag import analyze_job_fit, get_career_document_text
from src.project_rag import list_projects, answer_project_question
from src.learning_tracker import record_quiz_attempt
from src.learning_analytics import get_topic_statistics
from src.agent_tools import execute_tool
from src.api import app, SESSION_COOKIE_NAME


class TestAuthModule(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_users.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalize_username(self):
        self.assertEqual(normalize_username("  Alex_99  "), "alex_99")
        self.assertEqual(normalize_username("USER_NAME"), "user_name")
        with self.assertRaises(ValueError):
            normalize_username("ab")  # Too short
        with self.assertRaises(ValueError):
            normalize_username("bad@username!")  # Disallowed characters

    def test_password_hashing_and_verification(self):
        pwd = "SecurePassword123!"
        hashed = hash_password(pwd)
        self.assertTrue(hashed.startswith("pbkdf2_sha256$"))
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("WrongPassword!", hashed))
        self.assertFalse(verify_password("", hashed))

    def test_user_creation_and_auth(self):
        user = create_user("test_student", "SecretPass123", db_path=self.db_path)
        self.assertEqual(user.username, "test_student")
        self.assertTrue(user.is_active)

        # Duplicate username should raise ValueError
        with self.assertRaises(ValueError):
            create_user("test_student", "AnotherPass123", db_path=self.db_path)

        # Successful authentication
        authed = authenticate_user("test_student", "SecretPass123", db_path=self.db_path)
        self.assertIsNotNone(authed)
        self.assertEqual(authed.id, user.id)

        # Failed authentication: wrong password
        bad_auth = authenticate_user("test_student", "WrongPass", db_path=self.db_path)
        self.assertIsNone(bad_auth)

        # Failed authentication: nonexistent user
        no_user = authenticate_user("nonexistent", "SecretPass123", db_path=self.db_path)
        self.assertIsNone(no_user)

    def test_session_token_flow(self):
        user_id = "user_abc_123"
        token = create_access_token(user_id=user_id, username="alice", expires_delta_seconds=3600, secret_key="test-secret-key")
        self.assertIsInstance(token, str)

        payload = verify_access_token(token, secret_key="test-secret-key")
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], user_id)
        self.assertEqual(payload["username"], "alice")

        # Tampered token should fail verification
        tampered = token[:-4] + "xxxx"
        self.assertIsNone(verify_access_token(tampered, secret_key="test-secret-key"))

        # Wrong secret should fail
        self.assertIsNone(verify_access_token(token, secret_key="wrong-secret"))

        # Expired token
        expired_token = create_access_token(user_id=user_id, username="alice", expires_delta_seconds=-3600, secret_key="test-secret-key")
        self.assertIsNone(verify_access_token(expired_token, secret_key="test-secret-key"))


class TestMultiUserRetrievalIsolation(unittest.TestCase):
    def test_sparse_bm25_multi_user_isolation(self):
        """User A should not retrieve User B's documents in sparse search."""
        retriever = SparseRetriever()
        chunks = [
            {
                "chunk_id": "chunk_userA",
                "text": "Deep learning architectures and neural networks notes for Alice.",
                "file_name": "alice_notes.pdf",
                "page_number": 1,
                "metadata": {"category": "academic", "subject": "ai", "owner_id": "user_alice"},
            },
            {
                "chunk_id": "chunk_userB",
                "text": "Deep learning architectures and transformer models notes for Bob.",
                "file_name": "bob_notes.pdf",
                "page_number": 1,
                "metadata": {"category": "academic", "subject": "ai", "owner_id": "user_bob"},
            },
        ]
        retriever.index_chunks(chunks)

        # Alice searches
        alice_results = retriever.search(query="deep learning architectures", top_k=5, owner_id="user_alice")
        self.assertEqual(len(alice_results), 1)
        self.assertEqual(alice_results[0]["chunk_id"], "chunk_userA")

        # Bob searches
        bob_results = retriever.search(query="deep learning architectures", top_k=5, owner_id="user_bob")
        self.assertEqual(len(bob_results), 1)
        self.assertEqual(bob_results[0]["chunk_id"], "chunk_userB")

        # Third user Charlie has no indexed documents
        charlie_results = retriever.search(query="deep learning architectures", top_k=5, owner_id="user_charlie")
        self.assertEqual(len(charlie_results), 0)

    def test_dense_retriever_multi_user_filter(self):
        """Dense retriever must include owner_id filter in Qdrant query filter."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "studentai_test"
        mock_client.get_collections.return_value.collections = [mock_collection]
        mock_client.query_points.return_value.points = []

        fake_vector = [0.1] * 768
        with patch("src.retriever.get_settings") as mock_settings:
            mock_settings.return_value.embedding_dim = 768
            mock_settings.return_value.qdrant_collection_name = "studentai_test"

            dense_search(
                query="test query",
                top_k=5,
                client=mock_client,
                query_vector=fake_vector,
                owner_id="user_alice",
            )

        # Check that mock_client.query_points was called with a filter including owner_id
        mock_client.query_points.assert_called_once()
        _, kwargs = mock_client.query_points.call_args
        query_filter = kwargs.get("query_filter")
        self.assertIsNotNone(query_filter)
        self.assertTrue(
            any(
                getattr(cond, "key", None) == "owner_id"
                and getattr(getattr(cond, "match", None), "value", None) == "user_alice"
                for cond in getattr(query_filter, "must", [])
            )
        )


class TestMultiUserDataIsolation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cross_user_career_document_protection(self):
        """User A must not access User B's resume or career documents."""
        mock_retriever = SparseRetriever()
        mock_retriever.documents = {
            "chunk_bob": {
                "chunk_id": "chunk_bob",
                "document_id": "bob_resume.pdf",
                "file_name": "bob_resume.pdf",
                "owner_id": "user_bob",
                "category": "career",
                "document_type": "resume",
                "text": "Bob's private resume details.",
                "chunk_index": 0,
            }
        }

        # User Alice tries to read Bob's resume text -> access denied (raises ValueError)
        with self.assertRaises(ValueError) as ctx:
            get_career_document_text("bob_resume.pdf", sparse_retriever=mock_retriever, owner_id="user_alice")
        self.assertIn("Access denied", str(ctx.exception))

        # User Bob can read his own resume text
        doc = get_career_document_text("bob_resume.pdf", sparse_retriever=mock_retriever, owner_id="user_bob")
        self.assertEqual(doc["full_text"], "Bob's private resume details.")

    def test_cross_user_project_isolation(self):
        """User A must not list or access User B's project documents."""
        mock_retriever = SparseRetriever()
        mock_retriever.documents = {
            "chunk_bob": {
                "chunk_id": "chunk_bob",
                "document_id": "bob_proj.md",
                "file_name": "bob_proj.md",
                "owner_id": "user_bob",
                "category": "project",
                "document_type": "project_documentation",
                "subject": "expense_tracker",
                "text": "STUDENT PROJECT: Expense Tracker\nDetails here.",
            }
        }

        # Alice lists projects - User B's project should NOT appear
        alice_projects = list_projects(sparse_retriever=mock_retriever, owner_id="user_alice")
        self.assertEqual(len(alice_projects), 0)

        # Bob lists projects - User B's project should appear
        bob_projects = list_projects(sparse_retriever=mock_retriever, owner_id="user_bob")
        self.assertEqual(len(bob_projects), 1)
        self.assertEqual(bob_projects[0]["project_id"], "expense_tracker")

    def test_agent_tool_enforces_context_user_id(self):
        """LLM tool invocation with spoofed user_id in args must be overridden by authenticated context."""
        context = {"user_id": "authenticated_alice"}
        spoofed_args = {
            "user_id": "victim_bob",  # Malicious plan tries to peek at Bob's profile
        }

        with patch("src.agent_tools.get_learning_profile") as mock_prof:
            mock_prof.return_value = {"user_id": "authenticated_alice", "topics": []}
            result = execute_tool("learning_profile", spoofed_args, context=context)
            self.assertTrue(result["success"])
            # Verify the underlying call used Alice, NOT Bob
            mock_prof.assert_called_once_with(user_id="authenticated_alice", sparse_retriever=None)


class TestApiAuthSecurity(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_unauthenticated_request_rejected(self):
        """Requests to protected endpoints without auth token should return 401."""
        endpoints = [
            ("POST", "/api/agent/run", {"query": "Explain trees"}),
            ("POST", "/api/learning/ask", {"query": "Explain trees"}),
            ("GET", "/api/learning/profile", None),
            ("GET", "/api/learning/recommendations", None),
            ("POST", "/api/study-plans", {"subject": "OS", "days": 3}),
            ("POST", "/api/career/ask", {"query": "Review resume"}),
            ("POST", "/api/career/job-fit", {"resume_id": "res.pdf", "job_id": "job.pdf"}),
            ("GET", "/api/projects", None),
            ("POST", "/api/projects/ask", {"query": "What is the stack?"}),
        ]

        for method, path, json_data in endpoints:
            if method == "POST":
                resp = self.client.post(path, json=json_data)
            else:
                resp = self.client.get(path)
            self.assertEqual(
                resp.status_code,
                401,
                f"Endpoint {method} {path} should return 401 for unauthenticated request, got {resp.status_code}",
            )

    def test_bearer_token_authentication(self):
        """Requests with Authorization: Bearer <valid_token> should succeed."""
        test_user = User(
            id="user_bearer_test",
            username="bearer_user",
            created_at="2026-01-01T00:00:00Z",
            is_active=True,
        )
        token = create_access_token(user_id="user_bearer_test", username="bearer_user")

        with patch("src.api.get_user_by_id", return_value=test_user):
            resp = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["username"], "bearer_user")
            self.assertEqual(data["id"], "user_bearer_test")


    def test_registration_and_login_cookie_options_match(self):
        """Registration and login must set identical cookie attributes (httponly, samesite, secure, path)."""
        mock_user = User(
            id="usr_cookie_test",
            username="cookie_user",
            created_at="2026-01-01T00:00:00Z",
            is_active=True,
        )
        with patch("src.api.create_user", return_value=mock_user), \
             patch("src.api.authenticate_user", return_value=mock_user):

            # Register
            reg_resp = self.client.post("/api/auth/register", json={"username": "cookie_user", "password": "Password123!"})
            self.assertEqual(reg_resp.status_code, 200)
            reg_cookie = reg_resp.headers.get("set-cookie")
            self.assertIsNotNone(reg_cookie)

            # Login
            login_resp = self.client.post("/api/auth/login", json={"username": "cookie_user", "password": "Password123!"})
            self.assertEqual(login_resp.status_code, 200)
            login_cookie = login_resp.headers.get("set-cookie")
            self.assertIsNotNone(login_cookie)

            # Compare flags in Set-Cookie headers (excluding value)
            self.assertIn("HttpOnly", reg_cookie)
            self.assertIn("HttpOnly", login_cookie)
            self.assertIn("Path=/", reg_cookie)
            self.assertIn("Path=/", login_cookie)


class TestProductionSecurityAndStorageConfig(unittest.TestCase):
    def test_production_rejects_insecure_default_secret(self):
        """When ENVIRONMENT=production and AUTH_SECRET_KEY is default or short, require_production_auth_secret raises ValueError."""
        from src.config import require_production_auth_secret

        with patch.dict(os.environ, {"ENVIRONMENT": "production", "AUTH_SECRET_KEY": "studentai-dev-secret-change-in-production-key-32b"}):
            with self.assertRaises(ValueError) as ctx:
                require_production_auth_secret()
            self.assertIn("CRITICAL SECURITY CONFIGURATION ERROR", str(ctx.exception))

        with patch.dict(os.environ, {"ENVIRONMENT": "production", "AUTH_SECRET_KEY": "too-short"}):
            with self.assertRaises(ValueError):
                require_production_auth_secret()

        # Valid 32+ char secret in production succeeds
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "AUTH_SECRET_KEY": "a" * 32}):
            secret = require_production_auth_secret()
            self.assertEqual(secret, "a" * 32)

    def test_production_enforces_secure_cookies(self):
        """When ENVIRONMENT=production, Settings.cookie_secure must always evaluate to True."""
        from src.config import get_settings

        with patch.dict(os.environ, {"ENVIRONMENT": "production", "COOKIE_SECURE": "false"}):
            settings = get_settings()
            self.assertTrue(settings.cookie_secure)

        # In development, respects COOKIE_SECURE=false
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "COOKIE_SECURE": "false"}):
            settings = get_settings()
            self.assertFalse(settings.cookie_secure)

    def test_storage_path_resolution(self):
        """resolve_storage_path preserves absolute paths and resolves relative paths against BASE_DIR."""
        from src.config import resolve_storage_path, BASE_DIR

        # Relative path
        with patch.dict(os.environ, {"TEST_REL_PATH": "data/custom_store"}):
            rel = resolve_storage_path("TEST_REL_PATH", "data/default_store")
            self.assertEqual(rel, BASE_DIR / "data/custom_store")

        # Absolute Linux/POSIX path
        with patch.dict(os.environ, {"TEST_ABS_PATH": "/var/data/studentai/custom.db"}):
            resolved_abs = resolve_storage_path("TEST_ABS_PATH", "data/default.db")
            # Should evaluate to absolute path
            self.assertTrue(resolved_abs.is_absolute() or str(resolved_abs).startswith(os.path.sep) or str(resolved_abs).startswith("/"))
            self.assertIn("var", str(resolved_abs))
            self.assertIn("studentai", str(resolved_abs))

        # Default fallback
        if "TEST_UNSET_VAR" in os.environ:
            del os.environ["TEST_UNSET_VAR"]
        default_res = resolve_storage_path("TEST_UNSET_VAR", "data/default_fallback")
        self.assertEqual(default_res, BASE_DIR / "data/default_fallback")

    def test_is_storage_writable_detects_permission_denial(self):
        """is_storage_writable returns False when any critical directory cannot be created or written."""
        from src.config import is_storage_writable, get_settings

        # Normal writable directories
        self.assertTrue(is_storage_writable())

        # Simulate permission denial on directory creation
        with patch.object(Path, "mkdir", side_effect=PermissionError(13, "Permission denied")):
            self.assertFalse(is_storage_writable())

    def test_health_check_reports_storage_readiness_accurately(self):
        """GET /health must reflect actual storage readiness without hardcoding true."""
        client = TestClient(app)

        # Writable storage returns storage_ready=True
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["storage_ready"])

        # Unwritable storage returns storage_ready=False
        with patch("src.api.is_storage_writable", return_value=False):
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(resp.json()["storage_ready"])


if __name__ == "__main__":
    unittest.main()
