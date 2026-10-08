"""
Semantic Retriever Module.
Performs semantic similarity search against local Qdrant vector database using query embeddings.

Core Concept:
1. The student question is converted into the same high-dimensional embedding space using
   gemini-embedding-2 with task_type="RETRIEVAL_QUERY".
2. Qdrant computes the Cosine Similarity between the query vector and stored chunk vectors.
3. The top-K nearest neighbor chunks (ranked by highest cosine score) are returned with their full
   citation metadata, without raw vector payloads.
"""

from typing import List, Dict, Any, Optional
import qdrant_client
from src.config import get_settings
from src.embedder import embed_query
from src.vector_store import initialize_vector_store


def search(
    query: str,
    top_k: int = 5,
    client: Optional[qdrant_client.QdrantClient] = None,
    collection_name: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Executes semantic vector search for a natural language user query.

    Args:
        query: User question or natural language query (must be non-empty)
        top_k: Number of most similar chunks to retrieve (must be > 0)
        client: Optional pre-initialized QdrantClient instance
        collection_name: Optional collection name override
        query_vector: Optional pre-computed query vector (useful for testing/mocking)

    Returns:
        List of dictionaries ranked by descending similarity score:
        [
            {
                "score": float,
                "chunk_id": str,
                "text": str,
                "page_number": int,
                "total_pages": int,
                "file_name": str,
                "chunk_index": int,
                "char_count": int,
            }, ...
        ]

    Raises:
        ValueError: If query is empty, top_k <= 0, or collection does not exist.
        RuntimeError: If query embedding or vector search fails.
    """
    # 1. Parameter Validation
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace only")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

    settings = get_settings()
    target_collection = collection_name or settings.qdrant_collection_name
    active_client = client or initialize_vector_store()

    # 2. Verify Collection Existence
    existing_collections = [c.name for c in active_client.get_collections().collections]
    if target_collection not in existing_collections:
        raise ValueError(
            f"Collection '{target_collection}' does not exist in Qdrant. "
            "Please index documents first using 'python -m src.main index <pdf>'."
        )

    # 3. Generate Query Embedding
    if query_vector is not None:
        q_vec = query_vector
    else:
        q_vec = embed_query(query.strip())

    # 4. Validate Query Vector
    if len(q_vec) != settings.embedding_dim:
        raise ValueError(
            f"Query embedding dimension mismatch: expected {settings.embedding_dim}, got {len(q_vec)}"
        )

    # 5. Query Qdrant for Nearest Neighbors
    try:
        search_result = active_client.query_points(
            collection_name=target_collection,
            query=q_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,  # Exclude heavy raw vectors from results
        )
    except Exception as e:
        raise RuntimeError(f"Qdrant query execution failed: {e}") from e

    # 6. Format and Return Results with Metadata
    results: List[Dict[str, Any]] = []
    for point in search_result.points:
        payload = point.payload or {}
        text = payload.get("text", "")
        results.append({
            "score": float(point.score),
            "chunk_id": payload.get("chunk_id", str(point.id)),
            "text": text,
            "page_number": payload.get("page_number", 0),
            "total_pages": payload.get("total_pages", 0),
            "file_name": payload.get("file_name", "unknown"),
            "chunk_index": payload.get("chunk_index", 0),
            "char_count": payload.get("char_count", len(text)),
        })

    return results


# Backward compatibility alias
retrieve_context = search
