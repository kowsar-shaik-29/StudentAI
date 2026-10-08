"""
Vector Store Module.
Manages Qdrant vector database in local persistent mode (embedded file storage).

Why Qdrant Local Mode:
`QdrantClient(path='...')` runs Qdrant directly inside the Python process using local files.
It requires NO external Docker container, NO server process, and NO cloud account,
yet uses the exact same SDK and indexing capabilities as production Qdrant clusters.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import get_settings


def get_qdrant_client(storage_path: Optional[Path | str] = None) -> qdrant_client.QdrantClient:
    """
    Initializes and returns a QdrantClient configured for local persistent storage.
    """
    settings = get_settings()
    path = Path(storage_path) if storage_path else settings.qdrant_storage_path
    path.mkdir(parents=True, exist_ok=True)
    return qdrant_client.QdrantClient(path=str(path))


def init_collection(
    client: qdrant_client.QdrantClient,
    collection_name: Optional[str] = None,
    vector_size: Optional[int] = None,
) -> None:
    """
    Ensures that the target collection exists in Qdrant with Cosine distance metric.
    Creates it if it does not already exist.
    """
    settings = get_settings()
    target_collection = collection_name or settings.qdrant_collection_name
    target_dim = vector_size or settings.embedding_dim

    # Check if collection already exists
    existing_collections = [c.name for c in client.get_collections().collections]
    if target_collection not in existing_collections:
        client.create_collection(
            collection_name=target_collection,
            vectors_config=VectorParams(
                size=target_dim,
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(
    client: qdrant_client.QdrantClient,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: Optional[str] = None,
) -> int:
    """
    Stores text chunks and their embeddings into the Qdrant collection.

    Args:
        client: Qdrant client instance
        chunks: List of chunk dictionaries containing text and metadata
        embeddings: List of embedding vectors corresponding to chunks
        collection_name: Optional collection name override

    Returns:
        Number of points upserted
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks provided but {len(embeddings)} embeddings."
        )

    settings = get_settings()
    target_collection = collection_name or settings.qdrant_collection_name

    # Build Qdrant PointStruct items
    points = []
    for chunk, emb in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        payload = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "total_pages": chunk["total_pages"],
            "file_name": chunk["file_name"],
            "file_path": chunk["file_path"],
            "char_start": chunk["char_start"],
            "char_end": chunk["char_end"],
        }
        points.append(PointStruct(id=point_id, vector=emb, payload=payload))

    if points:
        client.upsert(collection_name=target_collection, points=points)

    return len(points)
