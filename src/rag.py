"""
End-to-End RAG Pipeline Module.
Coordinates semantic retrieval, context formulation, and grounded answer generation.

Core Pipeline:
Question
   ?
Semantic Retrieval (src.retriever.search)
   ?
Context Assembly (src.context_builder.build_context)
   ?
Grounded LLM Generation (src.generator.generate_answer)
   ?
Structured Answer + Cited Sources
"""

from typing import Dict, Any, Optional
import qdrant_client
from google import genai
from src.retriever import search
from src.hybrid_retriever import hybrid_search
from src.context_builder import build_context
from src.generator import generate_answer, get_genai_client
from src.vector_store import initialize_vector_store
from src.sparse_retriever import SparseRetriever


def answer_question(
    question: str,
    top_k: int = 5,
    client: Optional[qdrant_client.QdrantClient] = None,
    genai_client: Optional[genai.Client] = None,
    collection_name: Optional[str] = None,
    rerank: bool = False,
    initial_candidates: Optional[int] = None,
    hybrid: bool = False,
    sparse_retriever: Optional[SparseRetriever] = None,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    document_type: Optional[str] = None,
    document_id: Optional[str] = None,
    rewrite: bool = False,
) -> Dict[str, Any]:
    """
    Executes complete end-to-end RAG pipeline for a given user question.

    Args:
        question: User query string (must be non-empty)
        top_k: Number of relevant context chunks to retrieve
        client: Optional pre-initialized QdrantClient
        genai_client: Optional pre-authenticated GenAI client
        collection_name: Optional Qdrant collection override
        rerank: If True, applies second-stage reranking to retrieved candidates
        initial_candidates: Number of candidates to fetch if rerank=True
        hybrid: If True, uses Hybrid (Dense + Sparse RRF) retrieval
        sparse_retriever: Optional SparseRetriever instance
        category: Optional metadata category filter
        subject: Optional metadata subject filter
        document_type: Optional metadata document_type filter
        document_id: Optional metadata document_id filter

    Returns:
        Structured dictionary containing:
        {
            "question": str,
            "answer": str,
            "sources": [
                {
                    "score": float,
                    "file_name": str,
                    "page_number": int,
                    "chunk_id": str,
                    "text": str,
                    "category": str,
                    "subject": str,
                    "document_type": str,
                    "document_id": Optional[str],
                }, ...
            ]
        }

    Raises:
        ValueError: If question is empty or top_k is invalid.
        RuntimeError: If retrieval or generation fails.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty or whitespace only")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

    clean_question = question.strip()
    search_query = clean_question
    rewrite_info = None

    if rewrite:
        from src.query_rewriter import rewrite_query
        rewrite_info = rewrite_query(clean_question, client=genai_client)
        if rewrite_info.get("is_rewritten") and rewrite_info.get("rewritten_query"):
            search_query = rewrite_info["rewritten_query"]

    q_client = client or initialize_vector_store()

    try:
        filter_kwargs = {}
        if category is not None:
            filter_kwargs["category"] = category
        if subject is not None:
            filter_kwargs["subject"] = subject
        if document_type is not None:
            filter_kwargs["document_type"] = document_type
        if document_id is not None:
            filter_kwargs["document_id"] = document_id

        # Step 1: Retrieval (Dense, Hybrid, with optional Reranking)
        if hybrid:
            retrieved_chunks = hybrid_search(
                query=search_query,
                top_k=top_k,
                client=q_client,
                sparse_retriever=sparse_retriever,
                collection_name=collection_name,
                **filter_kwargs,
            )
        elif rerank:
            retrieved_chunks = search(
                query=search_query,
                top_k=top_k,
                client=q_client,
                collection_name=collection_name,
                rerank=True,
                initial_candidates=initial_candidates,
                genai_client=genai_client,
                **filter_kwargs,
            )
        else:
            retrieved_chunks = search(
                query=search_query,
                top_k=top_k,
                client=q_client,
                collection_name=collection_name,
                **filter_kwargs,
            )
    finally:
        # Close client only if created locally inside this function
        if client is None:
            q_client.close()

    # Step 2: Assemble Context
    context_str = build_context(retrieved_chunks)

    # Step 3: Generate Grounded Answer
    answer_text = generate_answer(
        question=clean_question,
        context=context_str,
        client=genai_client,
    )

    # Step 4: Package clean sources (excluding raw vectors)
    sources = []
    for item in retrieved_chunks:
        src_entry = {
            "score": item["score"],
            "file_name": item["file_name"],
            "page_number": item["page_number"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "category": item.get("category", "other"),
            "subject": item.get("subject", "general"),
            "document_type": item.get("document_type", "other"),
            "document_id": item.get("document_id"),
        }
        if "rerank_score" in item:
            src_entry["rerank_score"] = item["rerank_score"]
        sources.append(src_entry)

    # Step 5: Citation Verification
    from src.citation_verifier import verify_citations
    citation_report = verify_citations(answer=answer_text, sources=sources)

    result_payload = {
        "question": clean_question,
        "answer": answer_text,
        "sources": sources,
        "citations": citation_report,
    }
    if rewrite_info:
        result_payload["query_rewrite"] = rewrite_info

    return result_payload


def answer_question_auto(
    query: str,
    top_k: int = 5,
    client: Optional[qdrant_client.QdrantClient] = None,
    genai_client: Optional[genai.Client] = None,
    collection_name: Optional[str] = None,
    confidence_threshold: float = 0.70,
    sparse_retriever: Optional[SparseRetriever] = None,
) -> Dict[str, Any]:
    """
    Executes automated query routing and grounded RAG answer generation.
    Pipeline:
      Query -> Query Router -> Domain Route -> Category Filter -> Hybrid Retrieval -> Context -> Grounded Answer.

    Args:
        query: User question or query string.
        top_k: Number of relevant context chunks to retrieve.
        client: Optional pre-initialized QdrantClient.
        genai_client: Optional pre-authenticated GenAI client.
        collection_name: Optional Qdrant collection override.
        confidence_threshold: Minimum confidence for domain route before falling back to GENERAL.
        sparse_retriever: Optional SparseRetriever instance.

    Returns:
        Structured response dictionary:
        {
            "question": str,
            "route": str,
            "confidence": float,
            "reason": str,
            "category_filter": Optional[str],
            "answer": str,
            "sources": List[Dict[str, Any]],
        }
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace only")

    from src.query_router import route_query, get_category_filter_for_route

    clean_query = query.strip()
    g_client = genai_client or get_genai_client()

    # 1. Route Query
    routing_result = route_query(
        clean_query,
        client=g_client,
        confidence_threshold=confidence_threshold,
    )
    route = routing_result["route"]
    confidence = routing_result["confidence"]
    reason = routing_result["reason"]

    # 2. Derive Category Filter from Route
    category_filter = get_category_filter_for_route(route)

    # 3. Execute Grounded RAG with Hybrid Retrieval and Route Category Filter
    rag_result = answer_question(
        question=clean_query,
        top_k=top_k,
        client=client,
        genai_client=g_client,
        collection_name=collection_name,
        hybrid=True,
        sparse_retriever=sparse_retriever,
        category=category_filter,
    )

    return {
        "question": clean_query,
        "route": route,
        "confidence": confidence,
        "reason": reason,
        "category_filter": category_filter,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
    }



