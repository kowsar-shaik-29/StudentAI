"""
Document Metadata Model Module for StudentAI.
Provides structured metadata representation, deterministic document IDs,
validation, and default attributes for ingested documents and text chunks.

Model Structure:
- document_id: Deterministic SHA-256 / UUID5 derived from content or normalized path/filename.
- file_name: Base file name (e.g. computer_networking_notes.pdf).
- file_path: Absolute or relative file path string.
- category: Knowledge category ('academic', 'dsa', 'programming', 'career', 'project', 'other').
- subject: Subject or domain tag (e.g. 'computer_networks', 'database', 'system_design', 'general').
- document_type: Content type ('notes', 'textbook', 'resume', 'job_description', 'project_documentation', 'question_bank', 'other').
- created_at: ISO-8601 creation/ingestion timestamp.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union
import uuid
import hashlib

VALID_CATEGORIES = {
    "academic",
    "dsa",
    "programming",
    "career",
    "project",
    "other",
}

VALID_DOCUMENT_TYPES = {
    "notes",
    "textbook",
    "resume",
    "job_description",
    "project_documentation",
    "question_bank",
    "other",
}


def generate_document_id(file_name: str, file_path: Optional[str] = None) -> str:
    """
    Generates a deterministic document_id based on normalized file_name.
    Ensures that re-ingesting the exact same document produces the exact same document_id.
    """
    clean_name = Path(file_name).name.strip().lower()
    if not clean_name:
        raise ValueError("file_name must not be empty to generate document_id")
    # UUID5 in the DNS namespace using clean filename
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, clean_name))


@dataclass(frozen=True)
class DocumentMetadata:
    """
    Structured metadata associated with an ingested document.
    """
    file_name: str
    file_path: str = ""
    category: str = "other"
    subject: str = "general"
    document_type: str = "other"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    document_id: str = ""
    owner_id: str = "system"

    def __post_init__(self):
        # Normalize and validate
        cat = (self.category or "other").strip().lower()
        doc_type = (self.document_type or "other").strip().lower()
        subj = (self.subject or "general").strip().lower().replace(" ", "_")
        owner = (self.owner_id or "system").strip()

        object.__setattr__(self, "category", cat)
        object.__setattr__(self, "document_type", doc_type)
        object.__setattr__(self, "subject", subj)
        object.__setattr__(self, "owner_id", owner)

        if not self.document_id:
            doc_id = generate_document_id(self.file_name, self.file_path)
            object.__setattr__(self, "document_id", doc_id)

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata dataclass to dictionary."""
        return asdict(self)


def create_document_metadata(
    file_path: Optional[Union[str, Path]] = None,
    file_name: Optional[str] = None,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    document_type: Optional[str] = None,
    created_at: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> DocumentMetadata:
    """
    Factory helper to construct a validated DocumentMetadata instance.
    Accepts either file_path, file_name, or both.
    """
    if file_path is not None:
        path = Path(file_path)
        resolved_name = file_name or path.name
        try:
            resolved_path = str(path.resolve())
        except Exception:
            resolved_path = str(file_path)
    elif file_name is not None:
        resolved_name = file_name
        resolved_path = file_name
    else:
        raise ValueError("Either file_path or file_name must be provided to create_document_metadata")

    return DocumentMetadata(
        file_name=resolved_name,
        file_path=resolved_path,
        category=category or "other",
        subject=subject or "general",
        document_type=document_type or "other",
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        owner_id=owner_id or "system",
    )


