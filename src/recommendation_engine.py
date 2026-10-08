"""
Recommendation Engine Module for StudentAI.
Generates personalized, actionable study recommendations based on:
1. Learning analytics (interaction count, quiz performance, failure rate, refusal rate)
2. Deterministic weakness priorities (HIGH, MEDIUM, INSUFFICIENT_DATA, LOW)
3. Academic document discovery grounded strictly in existing academic corpus
4. Deduplication and deterministic multi-key sorting

Pure deterministic Python — NO LLM/Gemini calls, NO machine learning.
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path

from src.learning_analytics import (
    get_topic_statistics,
    get_subtopic_statistics,
)
from src.sparse_retriever import get_sparse_retriever, SparseRetriever

PRIORITY_RANK = {
    "HIGH": 1,
    "MEDIUM": 2,
    "INSUFFICIENT_DATA": 3,
    "LOW": 4,
}

REASON_TEMPLATES = {
    "HIGH": "Your recent quiz accuracy for this topic is low.",
    "MEDIUM": "Your performance indicates this topic may need additional revision.",
    "INSUFFICIENT_DATA": "There is not enough quiz data to measure your mastery of this topic.",
    "LOW": "Your recent quiz performance is strong.",
}

ACTION_TEMPLATES = {
    "HIGH": "Review the topic, then take a focused quiz.",
    "MEDIUM": "Review the topic and complete additional practice questions.",
    "INSUFFICIENT_DATA": "Take a diagnostic quiz to measure your understanding.",
    "LOW": "Continue with the next related topic.",
}


def find_academic_sources(
    topic: str,
    subtopic: str,
    sparse_retriever: Optional[SparseRetriever] = None,
    min_score: float = 0.01,
) -> List[str]:
    """
    Discovers academic source documents relevant to the topic and subtopic.
    Only considers documents with category="academic" present in the knowledge base.
    Never fabricates document names.

    Args:
        topic: Primary academic discipline.
        subtopic: Specific concept or subtopic.
        sparse_retriever: Optional pre-loaded SparseRetriever instance.
        min_score: Minimum BM25 score to consider document genuinely relevant.

    Returns:
        List of distinct academic file names (e.g. ['database_notes.pdf']).
    """
    sr = sparse_retriever or get_sparse_retriever()
    if sr.total_docs == 0:
        return []

    # Prioritize subtopic first for specific matching; fall back to subtopic + topic if subtopic is trivial
    sub_clean = subtopic.strip()
    top_clean = topic.strip()
    if sub_clean and sub_clean.lower() not in ("general", "none", ""):
        query_str = sub_clean
    elif top_clean:
        query_str = top_clean
    else:
        return []

    try:
        hits = sr.search(
            query=query_str,
            top_k=5,
            category="academic",
        )
    except Exception:
        return []

    matched_files: List[str] = []
    seen: Set[str] = set()

    for h in hits:
        score = h.get("sparse_score", 0.0)
        file_name = h.get("file_name", "")
        if score >= min_score and file_name and file_name not in seen:
            if h.get("category") == "academic":
                seen.add(file_name)
                matched_files.append(file_name)

    return matched_files


def get_learning_recommendations(
    user_id: str = "default_user",
    max_recommendations: int = 5,
    topic: Optional[str] = None,
    storage_path: Optional[Path] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
) -> Dict[str, Any]:
    """
    Generates personalized learning recommendations for a user.

    Args:
        user_id: Identifier of the student.
        max_recommendations: Maximum number of recommendations to return (default 5).
        topic: Optional topic filter.
        storage_path: Optional events storage path.
        sparse_retriever: Optional SparseRetriever instance.

    Returns:
        Structured recommendation dictionary:
        {
            "user_id": str,
            "recommendations": [
                {
                    "rank": int,
                    "topic": str,
                    "subtopic": str,
                    "priority": "HIGH" | "MEDIUM" | "INSUFFICIENT_DATA" | "LOW",
                    "weakness_score": float,
                    "reason": str,
                    "recommended_action": str,
                    "source_documents": List[str]
                }
            ],
            "message": Optional[str]
        }
    """
    subtopics_stats = get_subtopic_statistics(
        user_id=user_id,
        topic=topic,
        storage_path=storage_path,
    )

    if not subtopics_stats:
        return {
            "user_id": user_id,
            "recommendations": [],
            "message": "Not enough learning history yet. Ask questions and complete quizzes to build your learning profile.",
        }

    # Deduplicate strictly by (topic.lower(), subtopic.lower())
    unique_items: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in subtopics_stats:
        t_key = item["topic"].strip()
        s_key = item["subtopic"].strip()
        pair_key = (t_key.lower(), s_key.lower())
        if pair_key not in unique_items:
            unique_items[pair_key] = item

    # Filter out LOW priority if higher priority items (HIGH, MEDIUM, INSUFFICIENT_DATA) exist
    has_higher_priority = any(
        item["weakness_level"] in ("HIGH", "MEDIUM", "INSUFFICIENT_DATA")
        for item in unique_items.values()
    )

    candidate_items = []
    for item in unique_items.values():
        lvl = item["weakness_level"]
        if lvl == "LOW" and has_higher_priority:
            continue
        candidate_items.append(item)

    # Sort candidate items:
    # 1. Priority rank (HIGH=1, MEDIUM=2, INSUFFICIENT_DATA=3, LOW=4)
    # 2. Weakness score descending
    # 3. Topic name ascending, subtopic name ascending (for reproducible deterministic order)
    def sort_key(x: Dict[str, Any]):
        p_rank = PRIORITY_RANK.get(x["weakness_level"], 99)
        w_score = x.get("weakness_score", 0.0)
        return (p_rank, -w_score, x["topic"].lower(), x["subtopic"].lower())

    candidate_items.sort(key=sort_key)

    # Slice to max_recommendations
    top_candidates = candidate_items[:max_recommendations]

    # Pre-load sparse retriever once for efficiency
    sr = sparse_retriever or get_sparse_retriever()

    recommendations = []
    for idx, item in enumerate(top_candidates, start=1):
        top_name = item["topic"]
        sub_name = item["subtopic"]
        level = item["weakness_level"]
        score = item["weakness_score"]

        reason = REASON_TEMPLATES.get(level, REASON_TEMPLATES["MEDIUM"])
        action = ACTION_TEMPLATES.get(level, ACTION_TEMPLATES["MEDIUM"])

        sources = find_academic_sources(
            topic=top_name,
            subtopic=sub_name,
            sparse_retriever=sr,
        )

        recommendations.append({
            "rank": idx,
            "topic": top_name,
            "subtopic": sub_name,
            "priority": level,
            "weakness_score": score,
            "reason": reason,
            "recommended_action": action,
            "source_documents": sources,
        })

    return {
        "user_id": user_id,
        "recommendations": recommendations,
    }


def get_learning_profile(
    user_id: str = "default_user",
    storage_path: Optional[Path] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
) -> Dict[str, Any]:
    """
    Computes a personalized learning profile for a student based on existing analytics.

    Returns:
        Structured learning profile:
        {
            "user_id": str,
            "topics": [...],
            "weak_topics": [...],
            "strong_topics": [...],
            "insufficient_data_topics": [...],
            "recommendations": [...]
        }
    """
    topic_stats = get_topic_statistics(user_id=user_id, storage_path=storage_path)
    subtopic_stats = get_subtopic_statistics(user_id=user_id, storage_path=storage_path)

    weak_topics = []
    strong_topics = []
    insufficient_data_topics = []

    for s in subtopic_stats:
        level = s["weakness_level"]
        summary = {
            "topic": s["topic"],
            "subtopic": s["subtopic"],
            "accuracy": s["quiz_accuracy"],
            "weakness_score": s["weakness_score"],
            "weakness_level": s["weakness_level"],
            "questions_attempted": s["questions_attempted"],
            "interaction_count": s["interaction_count"],
        }
        if level in ("HIGH", "MEDIUM"):
            weak_topics.append(summary)
        elif level == "LOW":
            strong_topics.append(summary)
        elif level == "INSUFFICIENT_DATA":
            insufficient_data_topics.append(summary)

    recs_res = get_learning_recommendations(
        user_id=user_id,
        storage_path=storage_path,
        sparse_retriever=sparse_retriever,
    )

    return {
        "user_id": user_id,
        "topics": topic_stats,
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
        "insufficient_data_topics": insufficient_data_topics,
        "recommendations": recs_res.get("recommendations", []),
        "message": recs_res.get("message"),
    }
