"""
Reranker Module for StudentAI.
Performs second-stage relevance evaluation of retrieved candidate document chunks against the user query.

Architecture:
User query
    ↓
Gemini query embedding
    ↓
Qdrant
    ↓
Top-N candidate chunks (high-recall initial candidates)
    ↓
Reranker (cross-evaluates query directly against candidate chunk text)
    ↓
Top-K final chunks (high-precision results)

Key Design Requirements:
1. Evaluates QUERY against DOCUMENT CHUNK TEXT directly.
2. Preserves both the original retrieval score ('score') and the reranker score ('rerank_score').
3. Preserves all metadata (chunk_id, page_number, total_pages, file_name, chunk_index, char_count, text).
4. Does not include vectors.
5. Re-ranks candidates in descending order of 'rerank_score'.
6. Truncates to top_k.
"""

from typing import List, Dict, Any, Optional
import json
from google import genai
from google.genai import types
from src.config import require_api_key, get_settings


def rerank(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Reranks candidate document chunks based on direct query-to-text relevance evaluation.

    Args:
        query: Original user query string (must be non-empty)
        results: List of candidate chunk dictionaries retrieved from the vector store
        top_k: Number of final top results to return (must be > 0)
        client: Optional pre-authenticated GenAI client (useful for dependency injection / mocking)
        model: Optional model override (defaults to configured generation model)

    Returns:
        List of dictionaries reordered by descending reranker relevance:
        [
            {
                "score": float,         # Original Qdrant cosine similarity score
                "rerank_score": float,  # Second-stage cross-evaluation score (0.0 to 1.0)
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
        ValueError: If query is empty or top_k <= 0.
    """
    # 1. Parameter Validation
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace only")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

    # Empty candidate list edge case: return empty list immediately without making API calls
    if not results:
        return []

    clean_query = query.strip()
    settings = get_settings()
    target_model = model or settings.generation_model

    # 2. Prepare Structured Prompt for Cross-Evaluation
    prompt_lines = [
        "You are an expert search reranker for a Retrieval-Augmented Generation (RAG) system.",
        "Your task is to evaluate how directly, accurately, and thoroughly each candidate document chunk answers the User Query.",
        "",
        f"User Query:\n{clean_query}",
        "",
        "Candidate Chunks:",
    ]

    for idx, item in enumerate(results):
        chunk_text = item.get("text", "")
        prompt_lines.append(f"--- Chunk Index {idx} ---")
        prompt_lines.append(chunk_text)
        prompt_lines.append("")

    prompt_lines.extend([
        "Scoring Criteria:",
        "- Assign a relevance_score between 0.00 (completely irrelevant or off-topic) and 1.00 (directly answers the query with specific facts).",
        "- If a chunk partially mentions relevant concepts but doesn't answer the question, assign a score between 0.20 and 0.60.",
        "- If a chunk explicitly provides the answer to the question, assign a score between 0.85 and 1.00.",
        "",
        "Output Format:",
        "Return a strictly valid JSON array of objects, one for each candidate chunk index:",
        '[{"index": 0, "relevance_score": 0.95}, {"index": 1, "relevance_score": 0.20}]',
    ])

    scoring_prompt = "\n".join(prompt_lines)

    # 3. Call LLM for Structured Scoring
    ai_client = client or genai.Client(api_key=require_api_key())

    try:
        response = ai_client.models.generate_content(
            model=target_model,
            contents=scoring_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        raw_text = response.text if response.text else "[]"
        scores_data = json.loads(raw_text)
    except Exception:
        # Fallback gracefully if reranking API call fails:
        # Retain original candidate ranking and set rerank_score equal to normalized cosine score
        scored_results: List[Dict[str, Any]] = []
        for item in results:
            item_copy = dict(item)
            item_copy["rerank_score"] = float(item_copy.get("score", 0.0))
            scored_results.append(item_copy)
        return scored_results[:top_k]

    # Map candidate indices to scores
    index_to_score: Dict[int, float] = {}
    if isinstance(scores_data, list):
        for entry in scores_data:
            if isinstance(entry, dict) and "index" in entry and "relevance_score" in entry:
                try:
                    idx = int(entry["index"])
                    score_val = float(entry["relevance_score"])
                    # Clamp between 0.0 and 1.0
                    index_to_score[idx] = max(0.0, min(1.0, score_val))
                except (ValueError, TypeError):
                    continue

    # 4. Construct Output Structure Preserving All Metadata and Scores
    reranked_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(results):
        item_copy = dict(item)
        # Never include raw vector payloads
        item_copy.pop("vector", None)
        item_copy.pop("embedding", None)

        # Assign rerank_score (default to 0.0 if omitted by LLM output)
        item_copy["rerank_score"] = index_to_score.get(idx, 0.0)
        reranked_items.append(item_copy)

    # 5. Sort by Reranker Relevance (Descending), breaking ties by original score
    reranked_items.sort(key=lambda x: (x["rerank_score"], x.get("score", 0.0)), reverse=True)

    # 6. Truncate to top_k
    return reranked_items[:top_k]
