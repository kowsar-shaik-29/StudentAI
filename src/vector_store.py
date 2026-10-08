"""
Vector Store Module.
Manages Qdrant vector database in local persistent mode (embedded file storage).

Core Concepts:
1. Why Local Persistent Mode:
   - `QdrantClient(path="data/qdrant_storage")` stores vectors and payloads directly on disk.
   - Operates embedded inside Python without Docker, servers, or cloud dependencies,
     while using the exact same API and indexing structures as production Qdrant clusters.
2. Deterministic Point IDs:
   - Points are indexed using deterministic RFC 4122 UUID5 identifiers derived from `chunk_id`.
   - This ensures idempotent indexing: re-indexing the same document overwrites existing points
     rather than generating duplicate vector entries.
3. Cosine Similarity:
   - Normalized dot product measuring orientation rather than magnitude, optimal for
     high-dimensional text embeddings.
"""

import uuid
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import get_settings

REQUIRED_METADATA_FIELDS = (
    "chunk_id",
    "text",
    "page_number",
    "total_pages",
    "file_name",
    "file_path",
    "chunk_index",
    "char_count",
)


def generate_point_id(chunk_id: str) -> str:
    """
    Generates a deterministic RFC 4122 UUID5 identifier from a chunk ID string.
    Ensures idempotent storage and zero vector duplication across ingestion runs.
    """
    if not chunk_id or not chunk_id.strip():
        raise ValueError("chunk_id must be a non-empty string to generate a point ID")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id.strip()))


def initialize_vector_store(storage_path: Optional[Union[Path, str]] = None) -> qdrant_client.QdrantClient:
    """
    Initializes and returns a QdrantClient configured for local persistent storage.
    """
    settings = get_settings()
    path = Path(storage_path) if storage_path else settings.qdrant_storage_path
    path.mkdir(parents=True, exist_ok=True)
    return qdrant_client.QdrantClient(path=str(path))


# Alias for backward compatibility
get_qdrant_client = initialize_vector_store


def ensure_collection(
    client: qdrant_client.QdrantClient,
    collection_name: Optional[str] = None,
    vector_size: Optional[int] = None,
) -> None:
    """
    Ensures that the target collection exists in Qdrant with Cosine distance metric.
    Creates it if it does not already exist; does not recreate or modify if it exists.
    """
    settings = get_settings()
    target_collection = collection_name or settings.qdrant_collection_name
    target_dim = vector_size or settings.embedding_dim

    existing_collections = [c.name for c in client.get_collections().collections]
    if target_collection not in existing_collections:
        client.create_collection(
            collection_name=target_collection,
            vectors_config=VectorParams(
                size=target_dim,
                distance=Distance.COSINE,
            ),
        )


# Alias for backward compatibility
init_collection = ensure_collection


def count_vectors(
    client: qdrant_client.QdrantClient,
    collection_name: Optional[str] = None,
) -> int:
    """
    Returns the exact number of vectors stored in the specified collection.
    Returns 0 if the collection does not exist.
    """
    settings = get_settings()
    target_collection = collection_name or settings.qdrant_collection_name

    existing_collections = [c.name for c in client.get_collections().collections]
    if target_collection not in existing_collections:
        return 0

    return client.count(collection_name=target_collection, exact=True).count


def get_collection_info(
    client: qdrant_client.QdrantClient,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves collection status, vector count, and vector dimension.
    """
    settings = get_settings()
    target_collection = collection_name or settings.qdrant_collection_name

    existing_collections = [c.name for c in client.get_collections().collections]
    if target_collection not in existing_collections:
        return {
            "name": target_collection,
            "exists": False,
            "points_count": 0,
            "vector_size": None,
            "distance": None,
        }

    info = client.get_collection(collection_name=target_collection)
    vector_params = info.config.params.vectors

    # Handle single vector vs multivector configuration
    if isinstance(vector_params, VectorParams):
        v_size = vector_params.size
        v_dist = vector_params.distance.name if hasattr(vector_params.distance, "name") else str(vector_params.distance)
    else:
        v_size = None
        v_dist = None

    return {
        "name": target_collection,
        "exists": True,
        "points_count": info.points_count,
        "vector_size": v_size,
        "distance": v_dist,
        "status": info.status.name if hasattr(info.status, "name") else str(info.status),
    }


def upsert_chunks(
    client: qdrant_client.QdrantClient,
    embedded_chunks: List[Dict[str, Any]],
    collection_name: Optional[str] = None,
    vector_size: Optional[int] = None,
) -> int:
    """
    Stores embedded text chunks into the Qdrant collection using deterministic point IDs.

    Args:
        client: QdrantClient instance
        embedded_chunks: List of chunk dictionaries with 'embedding' key attached
        collection_name: Optional collection name override
        vector_size: Expected vector dimension for validation (defaults to 768)

    Returns:
        Number of points upserted

    Raises:
        ValueError: If vector dimension does not match, contains non-finite values,
                    or required metadata is missing.
    """
    if not embedded_chunks:
        return 0

    settings = get_settings()
    target_collection = collection_name or settings.qdrant_collection_name
    expected_dim = vector_size or settings.embedding_dim

    # Ensure collection exists
    ensure_collection(client, collection_name=target_collection, vector_size=expected_dim)

    points: List[PointStruct] = []
    for chunk in embedded_chunks:
        # 1. Validate embedding presence
        if "embedding" not in chunk:
            raise ValueError(f"Chunk '{chunk.get('chunk_id')}' is missing required 'embedding' vector")

        emb = chunk["embedding"]

        # 2. Validate vector dimension
        if len(emb) != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch for chunk '{chunk.get('chunk_id')}': "
                f"expected {expected_dim}, got {len(emb)}"
            )

        # 3. Validate finite values
        if not all(math.isfinite(v) for v in emb):
            raise ValueError(
                f"Vector for chunk '{chunk.get('chunk_id')}' contains non-finite values (NaN or Inf)"
            )

        # 4. Validate metadata fields
        for field in REQUIRED_METADATA_FIELDS:
            if field not in chunk:
                raise ValueError(
                    f"Chunk is missing required metadata field '{field}': {chunk}"
                )

        point_id = generate_point_id(chunk["chunk_id"])
        payload = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "total_pages": chunk["total_pages"],
            "file_name": chunk["file_name"],
            "file_path": chunk["file_path"],
            "chunk_index": chunk["chunk_index"],
            "char_count": chunk["char_count"],
        }

        points.append(PointStruct(id=point_id, vector=emb, payload=payload))

    client.upsert(collection_name=target_collection, points=points)
    return len(points)
