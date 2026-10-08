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

    # Generation Model (Gemini 2.5 Flash offers strong reasoning with low latency)
    generation_model: str = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")

    # Qdrant Storage Configuration
    # Local persistent folder path (does not require Docker)
    qdrant_storage_path: Path = BASE_DIR / os.getenv("QDRANT_STORAGE_PATH", "data/qdrant_storage")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "student_knowledge_base")

    # Text Chunking Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))


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
