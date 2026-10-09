"""
Configuration module for StudentAI.
Loads environment variables from .env and provides sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


DEFAULT_DEV_AUTH_SECRET = "studentai-dev-secret-change-in-production-key-32b"


def resolve_storage_path(env_var_name: str, default_relative_path: str, ensure_parent_dir: bool = False) -> Path:
    """
    Resolves storage path configuration:
    - Absolute paths (e.g. /var/data/studentai/users.db) are preserved directly.
    - Relative paths are resolved against project BASE_DIR.
    - Optionally creates parent directories safely.
    """
    raw_val = os.getenv(env_var_name, default_relative_path)
    if not raw_val:
        raw_val = default_relative_path

    # Check if absolute POSIX or Windows path
    p = Path(raw_val)
    if not (p.is_absolute() or (len(raw_val) > 1 and raw_val.startswith("/"))):
        p = BASE_DIR / raw_val

    if ensure_parent_dir:
        try:
            target_dir = p if p.suffix == "" else p.parent
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    return p


@dataclass(frozen=True)
class Settings:
    """Immutable configuration settings for StudentAI."""

    gemini_api_key: str
    embedding_model: str
    embedding_dim: int
    generation_model: str
    qdrant_storage_path: Path
    qdrant_collection_name: str
    chunk_size: int
    chunk_overlap: int
    sparse_index_dir: Path
    learning_events_path: Path
    study_plans_path: Path
    uploads_dir: Path
    environment: str
    app_version: str
    cors_origins: str
    user_db_path: Path
    auth_secret_key: str
    auth_token_expire_hours: int
    cookie_secure: bool
    cookie_samesite: str


def get_settings() -> Settings:
    """Returns the application settings reflecting current environment variables."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    cookie_sec = True if env == "production" else os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1", "yes")

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        generation_model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite"),
        qdrant_storage_path=resolve_storage_path("QDRANT_STORAGE_PATH", "data/qdrant_storage", ensure_parent_dir=True),
        qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "student_knowledge_base"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        sparse_index_dir=resolve_storage_path("SPARSE_INDEX_DIR", "data/sparse_index", ensure_parent_dir=True),
        learning_events_path=resolve_storage_path("LEARNING_EVENTS_PATH", "data/learning/learning_events.json", ensure_parent_dir=True),
        study_plans_path=resolve_storage_path("STUDY_PLANS_PATH", "data/learning/study_plans.json", ensure_parent_dir=True),
        uploads_dir=resolve_storage_path("UPLOADS_DIR", "data/uploads", ensure_parent_dir=True),
        environment=env,
        app_version="1.0.0",
        cors_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"),
        user_db_path=resolve_storage_path("USER_DB_PATH", "data/users.db", ensure_parent_dir=True),
        auth_secret_key=os.getenv("AUTH_SECRET_KEY", DEFAULT_DEV_AUTH_SECRET),
        auth_token_expire_hours=int(os.getenv("AUTH_TOKEN_EXPIRE_HOURS", "24")),
        cookie_secure=cookie_sec,
        cookie_samesite=os.getenv("COOKIE_SAMESITE", "lax").lower(),
    )


def require_api_key() -> str:
    """
    Validates that GEMINI_API_KEY is present in the environment or .env file.
    Raises ValueError with a clear instructional message if missing.
    """
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or not configured. "
            "Please create a .env file based on .env.example and set your valid Gemini API key."
        )
    return settings.gemini_api_key


def require_production_auth_secret() -> str:
    """
    Validates that production environment does not silently use the development auth secret.
    Raises ValueError if ENVIRONMENT is production and AUTH_SECRET_KEY is unchanged or insecure.
    """
    settings = get_settings()
    if settings.environment == "production":
        if (
            not settings.auth_secret_key
            or settings.auth_secret_key == DEFAULT_DEV_AUTH_SECRET
            or "change-in-production" in settings.auth_secret_key
            or len(settings.auth_secret_key) < 32
        ):
            raise ValueError(
                "CRITICAL SECURITY CONFIGURATION ERROR: In production mode (ENVIRONMENT=production), "
                "AUTH_SECRET_KEY must be set to a cryptographically secure random string of at least 32 characters. "
                "The default development key is not permitted."
            )
    return settings.auth_secret_key
