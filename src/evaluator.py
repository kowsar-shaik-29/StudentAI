"""
RAG Evaluator Module (Step 11).
Computes retrieval evaluation metrics:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- Document Recall@5
Supports evaluating dense, sparse, hybrid, and hybrid+reranker modes.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import qdrant_client
from google import genai
from src.retriever import search as dense_search
from src.sparse_retriever import get_sparse_retriever, SparseRetriever
from src.hybrid_retriever import hybrid_search
from src.reranker import rerank as rerank_fn
from src.query_rewriter import rewrite_query


def calculate_retrieval_metrics(
    eval_dataset: List[Dict[str, Any]],
    predictions: List[List[Dict[str, Any]]],
) -> Dict[str, float]:
    """
    Computes Recall@1, Recall@3, Recall@5, MRR, and Document Recall@5 over answerable evaluation items.

    Args:
        eval_dataset: List of eval item dictionaries with expected_chunk_ids, expected_documents, answerable.
        predictions: Parallel list of retrieved chunk candidate lists for each query in eval_dataset.

    Returns:
        Dictionary of computed metric values:
        {
            "recall_at_1": float,
            "recall_at_3": float,
            "recall_at_5": float,
            "mrr": float,
            "doc_recall_at_5": float,
            "total_answerable": int,
        }
    """
    if not eval_dataset:
        return {
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "doc_recall_at_5": 0.0,
            "total_answerable": 0,
        }

    answerable_count = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    doc_hits_at_5 = 0

    for item, retrieved in zip(eval_dataset, predictions):
        if not item.get("answerable", True):
            continue

        answerable_count += 1
        expected_cids: Set[str] = set(item.get("expected_chunk_ids", []))
        expected_docs: Set[str] = set(item.get("expected_documents", []))

        # 1. Chunk-level Evaluation
        retrieved_cids = [r.get("chunk_id") for r in retrieved]

        # Check Recall@1
        if len(retrieved_cids) >= 1 and retrieved_cids[0] in expected_cids:
            hits_at_1 += 1

        # Check Recall@3
        if any(cid in expected_cids for cid in retrieved_cids[:3]):
            hits_at_3 += 1

        # Check Recall@5
        if any(cid in expected_cids for cid in retrieved_cids[:5]):
            hits_at_5 += 1

        # Calculate Reciprocal Rank for MRR
        rr = 0.0
        for rank, cid in enumerate(retrieved_cids, start=1):
            if cid in expected_cids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # 2. Document-level Evaluation (Recall@5)
        retrieved_docs_5 = set(r.get("file_name") for r in retrieved[:5])
        if any(doc in retrieved_docs_5 for doc in expected_docs):
            doc_hits_at_5 += 1

    if answerable_count == 0:
        return {
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "doc_recall_at_5": 0.0,
            "total_answerable": 0,
        }

    return {
        "recall_at_1": hits_at_1 / answerable_count,
        "recall_at_3": hits_at_3 / answerable_count,
        "recall_at_5": hits_at_5 / answerable_count,
        "mrr": sum(reciprocal_ranks) / answerable_count,
        "doc_recall_at_5": doc_hits_at_5 / answerable_count,
        "total_answerable": answerable_count,
    }


def evaluate_retrieval_mode(
    eval_dataset: List[Dict[str, Any]],
    mode: str,
    top_k: int = 5,
    client: Optional[qdrant_client.QdrantClient] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
    genai_client: Optional[genai.Client] = None,
) -> Dict[str, Any]:
    """
    Runs retrieval across all dataset queries for a given mode and computes metrics.

    Supported modes: 'dense', 'sparse', 'hybrid', 'hybrid_rerank'.
    """
    sr = sparse_retriever or get_sparse_retriever()
    predictions: List[List[Dict[str, Any]]] = []
    per_question_details: List[Dict[str, Any]] = []

    for item in eval_dataset:
        query = item["question"]
        expected_cids = set(item.get("expected_chunk_ids", []))
        expected_docs = set(item.get("expected_documents", []))
        is_answerable = item.get("answerable", True)

        if mode == "dense":
            results = dense_search(query=query, top_k=top_k, client=client, rerank=False)
        elif mode == "sparse":
            results = sr.search(query=query, top_k=top_k)
        elif mode == "hybrid":
            results = hybrid_search(
                query=query,
                top_k=top_k,
                dense_candidates=10,
                sparse_candidates=10,
                client=client,
                sparse_retriever=sr,
            )
        elif mode == "hybrid_rerank":
            # Hybrid candidates followed by LLM reranking
            hybrid_candidates = hybrid_search(
                query=query,
                top_k=10,
                dense_candidates=10,
                sparse_candidates=10,
                client=client,
                sparse_retriever=sr,
            )
            results = rerank_fn(
                query=query,
                results=hybrid_candidates,
                top_k=top_k,
                client=genai_client,
            )
        elif mode == "hybrid_rewrite":
            rw_res = rewrite_query(query=query, client=genai_client)
            active_q = rw_res.get("rewritten_query") or query
            results = hybrid_search(
                query=active_q,
                top_k=top_k,
                dense_candidates=10,
                sparse_candidates=10,
                client=client,
                sparse_retriever=sr,
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        predictions.append(results)

        # Per question analysis
        first_relevant_rank = None
        for rank, r in enumerate(results, start=1):
            if r.get("chunk_id") in expected_cids:
                first_relevant_rank = rank
                break

        top_r = results[0] if results else None
        per_question_details.append({
            "id": item["id"],
            "question": query,
            "answerable": is_answerable,
            "top_result": top_r.get("chunk_id") if top_r else None,
            "top_doc": top_r.get("file_name") if top_r else None,
            "top_page": top_r.get("page_number") if top_r else None,
            "first_relevant_rank": first_relevant_rank,
            "is_hit_top1": (first_relevant_rank == 1),
            "is_hit_top5": (first_relevant_rank is not None and first_relevant_rank <= 5),
            "preview": top_r.get("text", "")[:120].strip().replace("\n", " ") if top_r else "",
        })

    metrics = calculate_retrieval_metrics(eval_dataset, predictions)
    return {
        "mode": mode,
        "metrics": metrics,
        "per_question": per_question_details,
    }
