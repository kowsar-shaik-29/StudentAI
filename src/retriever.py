"""
Retriever Module.
Performs semantic similarity search in Qdrant given a user query.

Core Concept:
1. The user question is converted into the same embedding space using the embedding model.
2. Qdrant computes the Cosine Similarity between the query vector and stored chunk vectors.
3. The top-K closest chunks are returned as relevant context.
"""

from typing import List, Dict, Any, Optional
import qdrant_client
from src.config import get_settings
from src.embedder import embed_text
from src.vector_store import get_qdrant_client


def retrieve_context(
    query: str,
    top_k: int = 4,
    client: Optional[qdrant_client.QdrantClient] = None,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Searches for the most relevant text chunks matching the user query.

    Args:
        query: User question or natural language query
        top_k: Number of most similar chunks to return
        client: Optional pre-initialized Qdrant client
        collection_name: Optional collection name override

    Returns:
        List of dictionaries with retrieved chunk text, score, and source metadata.
    """
    settings = get_settings()
    active_client = client or get_qdrant_client()
    target_collection = collection_name or settings.qdrant_collection_name

    # Step 1: Embed the search query into vector space
    query_vector = embed_text(query)

    # Step 2: Query Qdrant for nearest neighbors using query_points
    search_result = active_client.query_points(
        collection_name=target_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for scored_point in search_result.points:
        payload = scored_point.payload or {}
        results.append({
            "score": scored_point.score,
            "text": payload.get("text", ""),
            "page_number": payload.get("page_number", None),
            "file_name": payload.get("file_name", ""),
            "chunk_id": payload.get("chunk_id", ""),
        })

    return results
