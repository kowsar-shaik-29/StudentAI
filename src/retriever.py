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
from src.reranker import rerank as rerank_fn
from google import genai
from qdrant_client.models import Filter, FieldCondition, MatchValue


def search(
    query: str,
    top_k: int = 5,
    client: Optional[qdrant_client.QdrantClient] = None,
    collection_name: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    rerank: bool = False,
    initial_candidates: Optional[int] = None,
    genai_client: Optional[genai.Client] = None,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    document_type: Optional[str] = None,
    document_id: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Executes semantic vector search for a natural language user query.
    Optionally filters by document metadata (category, subject, document_type, document_id, owner_id)
    and optionally reranks candidates using cross-evaluation relevance scoring.

    Args:
        query: User question or natural language query (must be non-empty)
        top_k: Number of most similar chunks to retrieve (must be > 0)
        client: Optional pre-initialized QdrantClient instance
        collection_name: Optional collection name override
        query_vector: Optional pre-computed query vector (useful for testing/mocking)
        rerank: If True, retrieves top candidates from Qdrant and applies second-stage reranking
        initial_candidates: Number of candidate chunks to fetch before reranking (defaults to max(10, top_k * 2))
        genai_client: Optional pre-authenticated GenAI client for reranking
        category: Optional category filter (e.g. 'academic', 'career')
        subject: Optional subject filter (e.g. 'database', 'computer_networks')
        document_type: Optional document type filter (e.g. 'notes', 'resume')
        document_id: Optional exact document ID filter


    Returns:
        List of dictionaries ranked by descending similarity (or rerank_score if rerank=True):
        [
            {
                "score": float,
                "rerank_score": float,  # Present if rerank=True
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

    if initial_candidates is not None and (not isinstance(initial_candidates, int) or initial_candidates <= 0):
        raise ValueError(f"initial_candidates must be a positive integer greater than 0, got {initial_candidates}")

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

    # 5. Determine Number of Candidates to Fetch
    # If rerank is enabled, retrieve Top-N initial candidates (e.g., initial_candidates or max(10, top_k * 2))
    if rerank:
        candidate_count = initial_candidates if initial_candidates is not None else max(10, top_k * 2)
        fetch_limit = max(candidate_count, top_k)
    else:
        fetch_limit = top_k

    # 6. Build Structured Qdrant Filter for Metadata
    conditions = []
    if category:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category.strip().lower())))
    if subject:
        conditions.append(FieldCondition(key="subject", match=MatchValue(value=subject.strip().lower().replace(" ", "_"))))
    if document_type:
        conditions.append(FieldCondition(key="document_type", match=MatchValue(value=document_type.strip().lower())))
    if document_id:
        conditions.append(FieldCondition(key="document_id", match=MatchValue(value=document_id.strip())))
    if owner_id:
        conditions.append(FieldCondition(key="owner_id", match=MatchValue(value=owner_id.strip())))

    query_filter = Filter(must=conditions) if conditions else None

    # 7. Query Qdrant for Nearest Neighbors
    try:
        search_result = active_client.query_points(
            collection_name=target_collection,
            query=q_vec,
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,  # Exclude heavy raw vectors from results
        )
    except Exception as e:
        raise RuntimeError(f"Qdrant query execution failed: {e}") from e

    # 8. Format Candidate Results with Metadata
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
            "file_path": payload.get("file_path", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "char_count": payload.get("char_count", len(text)),
            "document_id": payload.get("document_id", ""),
            "category": payload.get("category", "other"),
            "subject": payload.get("subject", "general"),
            "document_type": payload.get("document_type", "other"),
            "owner_id": payload.get("owner_id", "system"),
        })

    # 9. Apply Reranking if enabled
    if rerank:
        return rerank_fn(
            query=query.strip(),
            results=results,
            top_k=top_k,
            client=genai_client,
        )

    return results


# Backward compatibility alias
retrieve_context = search

