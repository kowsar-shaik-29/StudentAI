"""
Revision Engine Module for StudentAI.
Generates prioritized revision recommendations based on:
1. Topic/subtopic weakness levels (HIGH, MEDIUM, INSUFFICIENT_DATA, LOW)
2. Recent quiz outcomes and performance degradation
3. Subject filtering and limit respect

Pure deterministic Python — NO LLM/Gemini calls, NO machine learning.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from src.learning_analytics import get_subtopic_statistics
from src.study_planner import is_subject_match
from src.recommendation_engine import find_academic_sources
from src.sparse_retriever import get_sparse_retriever, SparseRetriever

REVISION_PRIORITY_MAP = {
    "HIGH": 1,
    "MEDIUM": 2,
    "INSUFFICIENT_DATA": 3,
    "LOW": 4,
}

REVISION_ACTIVITIES = {
    "HIGH": "Review core concepts and complete a targeted quiz",
    "MEDIUM": "Review notes and solve practice exercises",
    "INSUFFICIENT_DATA": "Take a diagnostic assessment quiz",
    "LOW": "Periodic flashcard review and light active recall",
}

REVISION_REASONS = {
    "HIGH": "Quiz performance indicates significant weakness in this area.",
    "MEDIUM": "Performance shows moderate gaps requiring reinforced practice.",
    "INSUFFICIENT_DATA": "No quiz data available; diagnostic assessment needed.",
    "LOW": "Strong performance; maintain proficiency with light revision.",
}


def get_revision_recommendations(
    user_id: str = "default_user",
    subject: Optional[str] = None,
    limit: int = 5,
    events_storage_path: Optional[Path] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
) -> Dict[str, Any]:
    """
    Generates prioritized revision items for a student.

    Args:
        user_id: Student user ID.
        subject: Optional subject filter.
        limit: Maximum number of revision items to return (default: 5).
        events_storage_path: Optional storage path for learning events.
        sparse_retriever: Optional SparseRetriever instance.

    Returns:
        Structured revision recommendations:
        {
            "user_id": str,
            "subject": Optional[str],
            "revision_items": [
                {
                    "topic": str,
                    "subtopic": str,
                    "priority": str,
                    "weakness_score": float,
                    "accuracy": Optional[float],
                    "reason": str,
                    "recommended_activity": str,
                    "source_documents": List[str]
                }
            ]
        }
    """
    if limit <= 0:
        limit = 5

    all_subtopics = get_subtopic_statistics(
        user_id=user_id,
        storage_path=events_storage_path,
    )

    if subject and subject.strip():
        filtered_subtopics = [
            s for s in all_subtopics
            if is_subject_match(s.get("topic", ""), subject.strip())
        ]
    else:
        filtered_subtopics = all_subtopics

    if not filtered_subtopics:
        return {
            "user_id": user_id,
            "subject": subject,
            "revision_items": [],
            "message": "No revision items found for the specified criteria.",
        }

    sr = sparse_retriever or get_sparse_retriever()

    # Sort items:
    # 1. Priority rank (HIGH -> MEDIUM -> INSUFFICIENT_DATA -> LOW)
    # 2. Weakness score descending
    # 3. Topic name ascending, subtopic name ascending
    def revision_sort_key(x: Dict[str, Any]):
        p_num = REVISION_PRIORITY_MAP.get(x.get("weakness_level", "LOW"), 99)
        w_score = x.get("weakness_score", 0.0)
        return (p_num, -w_score, x.get("topic", "").lower(), x.get("subtopic", "").lower())

    sorted_subtopics = sorted(filtered_subtopics, key=revision_sort_key)
    top_items = sorted_subtopics[:limit]

    revision_items: List[Dict[str, Any]] = []
    for item in top_items:
        t_name = item["topic"]
        s_name = item["subtopic"]
        level = item["weakness_level"]
        score = item["weakness_score"]
        acc = item["quiz_accuracy"]

        reason = REVISION_REASONS.get(level, REVISION_REASONS["LOW"])
        activity = REVISION_ACTIVITIES.get(level, REVISION_ACTIVITIES["LOW"])

        sources = find_academic_sources(
            topic=t_name,
            subtopic=s_name,
            sparse_retriever=sr,
        )

        revision_items.append({
            "topic": t_name,
            "subtopic": s_name,
            "priority": level,
            "weakness_score": score,
            "accuracy": acc,
            "reason": reason,
            "recommended_activity": activity,
            "source_documents": sources,
        })

    return {
        "user_id": user_id,
        "subject": subject,
        "revision_items": revision_items,
    }
