"""
Authentication and User Persistence Module for StudentAI.
Provides:
- SQLite persistent user store with schema initialization
- Modern password hashing using PBKDF2-HMAC-SHA256 (600,000 iterations + random salt)
- Signed session tokens using HMAC-SHA256 with timestamped expiration
- Input normalization, validation, and length checks
"""

import os
import sqlite3
import hashlib
import hmac
import json
import base64
import time
import secrets
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from datetime import datetime, timezone

from src.config import get_settings

PBKDF2_ITERATIONS = 600000
SALT_SIZE = 16


@dataclass
class User:
    id: str
    username: str
    created_at: str
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "created_at": self.created_at,
            "is_active": self.is_active,
        }


def hash_password(password: str) -> str:
    """Hashes password using PBKDF2-HMAC-SHA256 with random salt."""
    if not password or not isinstance(password, str):
        raise ValueError("Password must be a non-empty string")
    salt = secrets.token_bytes(SALT_SIZE)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    salt_hex = salt.hex()
    hash_hex = dk.hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_hex}${hash_hex}"


def verify_password(password: str, hashed: str) -> bool:
    """Verifies password against stored PBKDF2 hash using constant-time comparison."""
    if not password or not hashed or not isinstance(password, str) or not isinstance(hashed, str):
        return False
    parts = hashed.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_hash = bytes.fromhex(parts[3])
    except Exception:
        return False

    candidate_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate_hash, expected_hash)


def normalize_username(username: str) -> str:
    """Normalizes and validates username."""
    if not username or not isinstance(username, str):
        raise ValueError("Username must be a non-empty string")
    clean = username.strip().lower()
    if len(clean) < 3 or len(clean) > 50:
        raise ValueError("Username must be between 3 and 50 characters")
    if not re.match(r"^[a-z0-9_.-]+$", clean):
        raise ValueError("Username can only contain alphanumeric characters, underscores, hyphens, and periods")
    return clean


def init_user_db(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Initializes the SQLite user database table if not present."""
    settings = get_settings()
    path = Path(db_path) if db_path else settings.user_db_path
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        conn.commit()
    finally:
        conn.close()
    return path


def create_user(username: str, password: str, db_path: Optional[Union[str, Path]] = None) -> User:
    """Creates a new user record in the SQLite database."""
    clean_username = normalize_username(username)
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long")
    if len(password) > 128:
        raise ValueError("Password must not exceed 128 characters")

    path = init_user_db(db_path)
    pwd_hash = hash_password(password)
    user_id = f"usr_{secrets.token_hex(12)}"
    created_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (id, username, password_hash, created_at, is_active) VALUES (?, ?, ?, ?, 1)",
            (user_id, clean_username, pwd_hash, created_at),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{clean_username}' is already registered")
    finally:
        conn.close()

    return User(id=user_id, username=clean_username, created_at=created_at, is_active=True)


def authenticate_user(username: str, password: str, db_path: Optional[Union[str, Path]] = None) -> Optional[User]:
    """Authenticates user credentials against the SQLite database."""
    try:
        clean_username = normalize_username(username)
    except ValueError:
        return None

    path = init_user_db(db_path)
    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password_hash, created_at, is_active FROM users WHERE username = ?",
            (clean_username,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    user_id, uname, pwd_hash, created_at, is_active = row
    if not bool(is_active):
        return None

    if not verify_password(password, pwd_hash):
        return None

    return User(id=user_id, username=uname, created_at=created_at, is_active=bool(is_active))


def get_user_by_id(user_id: str, db_path: Optional[Union[str, Path]] = None) -> Optional[User]:
    """Retrieves user by stable user ID."""
    if not user_id:
        return None
    path = init_user_db(db_path)
    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, created_at, is_active FROM users WHERE id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    uid, uname, created_at, is_active = row
    return User(id=uid, username=uname, created_at=created_at, is_active=bool(is_active))


def create_access_token(
    user_id: str,
    username: str,
    expires_delta_seconds: Optional[int] = None,
    secret_key: Optional[str] = None,
) -> str:
    """Creates a signed HMAC-SHA256 session token."""
    settings = get_settings()
    secret = secret_key or settings.auth_secret_key
    exp_seconds = expires_delta_seconds or (settings.auth_token_expire_hours * 3600)

    now = int(time.time())
    exp = now + exp_seconds

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": user_id, "username": username, "iat": now, "exp": exp}

    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    msg = f"{h_b64}.{p_b64}".encode()
    sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    s_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")

    return f"{h_b64}.{p_b64}.{s_b64}"


def verify_access_token(
    token: str,
    secret_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Verifies HMAC-SHA256 signature and expiration of session token."""
    if not token or not isinstance(token, str):
        return None

    parts = token.split(".")
    if len(parts) != 3:
        return None

    h_b64, p_b64, s_b64 = parts
    settings = get_settings()
    secret = secret_key or settings.auth_secret_key

    # Validate signature
    msg = f"{h_b64}.{p_b64}".encode()
    expected_sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    expected_s_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")

    if not hmac.compare_digest(s_b64, expected_s_b64):
        return None

    # Decode payload
    try:
        p_padded = p_b64 + ("=" * ((4 - len(p_b64) % 4) % 4))
        payload_bytes = base64.urlsafe_b64decode(p_padded.encode())
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None

    # Check expiration
    exp = payload.get("exp")
    if exp is None or int(time.time()) > int(exp):
        return None

    return payload
