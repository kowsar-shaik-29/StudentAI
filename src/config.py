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


@dataclass(frozen=True)
class Settings:
    """Immutable configuration settings for StudentAI."""

    # API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # Model Configuration
    # Official Google GenAI SDK recommendation (GA successor to deprecated text-embedding-004)
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))

    # Generation Model (Gemini 3.5 Flash Lite offers high throughput and generous quota)
    generation_model: str = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite")

    # Qdrant Storage Configuration
    # Local persistent folder path (does not require Docker)
    qdrant_storage_path: Path = BASE_DIR / os.getenv("QDRANT_STORAGE_PATH", "data/qdrant_storage")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "student_knowledge_base")

    # Text Chunking Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Sparse Index Configuration
    sparse_index_dir: Path = BASE_DIR / os.getenv("SPARSE_INDEX_DIR", "data/sparse_index")

    # API Configuration
    app_version: str = "1.0.0"
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

    # Authentication & User Database
    user_db_path: Path = BASE_DIR / os.getenv("USER_DB_PATH", "data/users.db")
    auth_secret_key: str = os.getenv("AUTH_SECRET_KEY", "studentai-dev-secret-change-in-production-key-32b")
    auth_token_expire_hours: int = int(os.getenv("AUTH_TOKEN_EXPIRE_HOURS", "24"))



def get_settings() -> Settings:
    """Returns the application settings."""
    return Settings()


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
