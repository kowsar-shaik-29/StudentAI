"""
Hybrid Retriever Module for StudentAI.
Combines Dense Semantic Retrieval (Qdrant) and Sparse Lexical Retrieval (BM25)
using Reciprocal Rank Fusion (RRF).

RRF Concept:
For each document d across retrieval lists:
    RRF_score(d) = Σ [ 1.0 / (k + rank(d)) ]
where:
    k is a smoothing constant (default: 60)
    rank(d) is 1-based rank position in each candidate list.

Pipeline:
Query
  ├──> Dense Vector Retrieval (Qdrant)   ──> Dense Candidates (with dense_score)
  └──> Sparse BM25 Retrieval             ──> Sparse Candidates (with sparse_score)
            │
            └───> Reciprocal Rank Fusion (RRF) ───> Final Top-K Chunks (fused_score)

Key Design Requirements:
1. Preserves dense_score, sparse_score, and fused_score.
2. Preserves all chunk metadata without raw vector payloads.
3. Configurable top_k, dense_candidates, sparse_candidates, and rrf_k.
4. Truncates final output to top_k.
"""

from typing import List, Dict, Any, Optional
import qdrant_client
from src.retriever import search as dense_search
from src.sparse_retriever import get_sparse_retriever, SparseRetriever


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fuses two ranked result lists (dense and sparse) using Reciprocal Rank Fusion.

    Args:
        dense_results: Candidate list from dense retrieval (ordered rank 1 to N).
        sparse_results: Candidate list from sparse retrieval (ordered rank 1 to M).
        top_k: Number of fused results to return.
        rrf_k: RRF smoothing constant (standard is 60).

    Returns:
        Combined list sorted by descending fused_score.
    """
    fused_scores: Dict[str, float] = {}
    chunk_metadata: Dict[str, Dict[str, Any]] = {}
    dense_scores: Dict[str, float] = {}
    sparse_scores: Dict[str, float] = {}

    # Process Dense results (1-based rank)
    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        dense_scores[cid] = item.get("score", 0.0)
        chunk_metadata[cid] = item
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    # Process Sparse results (1-based rank)
    for rank, item in enumerate(sparse_results, start=1):
        cid = item["chunk_id"]
        sparse_scores[cid] = item.get("sparse_score", 0.0)
        if cid not in chunk_metadata:
            chunk_metadata[cid] = item
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    # Sort chunks by descending fused_score, breaking ties by chunk_id
    sorted_cids = sorted(fused_scores.keys(), key=lambda cid: (fused_scores[cid], cid), reverse=True)

    fused_results: List[Dict[str, Any]] = []
    for cid in sorted_cids[:top_k]:
        meta = chunk_metadata[cid]
        fused_entry = {
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
            "page_number": meta["page_number"],
            "total_pages": meta["total_pages"],
            "file_name": meta["file_name"],
            "chunk_index": meta["chunk_index"],
            "char_count": meta["char_count"],
            "document_id": meta.get("document_id"),
            "category": meta.get("category", "other"),
            "subject": meta.get("subject", "general"),
            "document_type": meta.get("document_type", "other"),
            "dense_score": float(dense_scores.get(cid, 0.0)),
            "sparse_score": float(sparse_scores.get(cid, 0.0)),
            "fused_score": float(fused_scores[cid]),
            # Maintain backward-compatible 'score' pointing to fused_score
            "score": float(fused_scores[cid]),
        }
        fused_results.append(fused_entry)

    return fused_results


def hybrid_search(
    query: str,
    top_k: int = 5,
    dense_candidates: int = 10,
    sparse_candidates: int = 10,
    rrf_k: int = 60,
    client: Optional[qdrant_client.QdrantClient] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
    collection_name: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    document_type: Optional[str] = None,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Executes hybrid search combining dense semantic retrieval and sparse BM25 retrieval.

    Args:
        query: User question or natural language query (must be non-empty).
        top_k: Final number of ranked chunks to return (must be > 0).
        dense_candidates: Number of candidate chunks to fetch from Qdrant.
        sparse_candidates: Number of candidate chunks to fetch from BM25.
        rrf_k: Reciprocal Rank Fusion smoothing parameter.
        client: Optional pre-initialized QdrantClient instance.
        sparse_retriever: Optional SparseRetriever instance.
        collection_name: Optional Qdrant collection override.
        query_vector: Optional pre-computed query vector (useful for testing/mocking).
        category: Optional metadata category filter.
        subject: Optional metadata subject filter.
        document_type: Optional metadata document_type filter.
        document_id: Optional metadata document_id filter.

    Returns:
        List of dictionaries ranked by descending fused_score:
        [
            {
                "chunk_id": str,
                "text": str,
                "page_number": int,
                "total_pages": int,
                "file_name": str,
                "chunk_index": int,
                "char_count": int,
                "document_id": Optional[str],
                "category": str,
                "subject": str,
                "document_type": str,
                "dense_score": float,
                "sparse_score": float,
                "fused_score": float,
                "score": float,
            }, ...
        ]
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace only")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

    if not isinstance(dense_candidates, int) or dense_candidates <= 0:
        raise ValueError(f"dense_candidates must be a positive integer greater than 0, got {dense_candidates}")

    if not isinstance(sparse_candidates, int) or sparse_candidates <= 0:
        raise ValueError(f"sparse_candidates must be a positive integer greater than 0, got {sparse_candidates}")

    clean_query = query.strip()

    # 1. Dense Semantic Retrieval
    dense_results = dense_search(
        query=clean_query,
        top_k=dense_candidates,
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        rerank=False,
        category=category,
        subject=subject,
        document_type=document_type,
        document_id=document_id,
    )

    # 2. Sparse Lexical Retrieval
    sr = sparse_retriever or get_sparse_retriever()
    sparse_results = sr.search(
        query=clean_query,
        top_k=sparse_candidates,
        category=category,
        subject=subject,
        document_type=document_type,
        document_id=document_id,
    )

    # 3. Reciprocal Rank Fusion
    return reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=top_k,
        rrf_k=rrf_k,
    )

