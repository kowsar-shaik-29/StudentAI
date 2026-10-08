"""
Learning Analytics Module for StudentAI.
Computes topic-level and subtopic-level performance metrics, interaction statistics,
and deterministic weakness scores from recorded learning events.
Pure deterministic Python — NO machine learning or external LLM calls.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from src.learning_tracker import get_learning_events, LearningEvent


def _compute_weakness(
    questions_attempted: int,
    questions_incorrect: int,
    interaction_count: int,
    refused_count: int,
) -> tuple[float, str]:
    """
    Computes deterministic weakness score and qualitative weakness level.

    Formula:
        if questions_attempted == 0:
            weakness_score = 0.0
            weakness_level = "INSUFFICIENT_DATA"
        else:
            quiz_failure_rate = questions_incorrect / questions_attempted
            refusal_rate = refused_count / max(interaction_count, 1)
            weakness_score = min(1.0, max(0.0, 0.85 * quiz_failure_rate + 0.15 * refusal_rate))
            Level:
                [0.00, 0.30) -> "LOW"
                [0.30, 0.60) -> "MEDIUM"
                [0.60, 1.00] -> "HIGH"
    """
    if questions_attempted == 0:
        return 0.0, "INSUFFICIENT_DATA"

    quiz_failure_rate = questions_incorrect / questions_attempted
    refusal_rate = refused_count / max(interaction_count, 1)
    raw_score = 0.85 * quiz_failure_rate + 0.15 * refusal_rate
    weakness_score = max(0.0, min(1.0, round(raw_score, 4)))

    if weakness_score < 0.30:
        weakness_level = "LOW"
    elif weakness_score < 0.60:
        weakness_level = "MEDIUM"
    else:
        weakness_level = "HIGH"

    return weakness_score, weakness_level


def get_topic_statistics(
    user_id: Optional[str] = "default_user",
    storage_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregates learning events by primary academic topic.

    Returns:
        List of topic summary dicts sorted by weakness_score descending.
    """
    events = get_learning_events(user_id=user_id, storage_path=storage_path)
    topic_groups: Dict[str, List[LearningEvent]] = {}
    for ev in events:
        t = ev.topic or "Computer Science"
        topic_groups.setdefault(t, []).append(ev)

    results: List[Dict[str, Any]] = []

    for topic_name, ev_list in topic_groups.items():
        interaction_count = len(ev_list)
        answered_count = sum(1 for e in ev_list if e.status == "ANSWERED")
        refused_count = sum(1 for e in ev_list if e.status == "REFUSED")
        quiz_attempts = sum(1 for e in ev_list if e.status == "QUIZ_ATTEMPTED")
        questions_attempted = sum(e.questions_attempted for e in ev_list)
        questions_correct = sum(e.questions_correct for e in ev_list)
        questions_incorrect = sum(e.questions_incorrect for e in ev_list)

        quiz_accuracy = (
            round(questions_correct / questions_attempted, 4)
            if questions_attempted > 0
            else None
        )

        weakness_score, weakness_level = _compute_weakness(
            questions_attempted=questions_attempted,
            questions_incorrect=questions_incorrect,
            interaction_count=interaction_count,
            refused_count=refused_count,
        )

        results.append({
            "topic": topic_name,
            "interaction_count": interaction_count,
            "answered_count": answered_count,
            "refused_count": refused_count,
            "quiz_attempts": quiz_attempts,
            "questions_attempted": questions_attempted,
            "questions_correct": questions_correct,
            "questions_incorrect": questions_incorrect,
            "quiz_accuracy": quiz_accuracy,
            "weakness_score": weakness_score,
            "weakness_level": weakness_level,
        })

    # Sort primarily by weakness_score descending, secondary by interaction_count descending
    results.sort(key=lambda x: (x["weakness_score"], x["interaction_count"]), reverse=True)
    return results


def get_subtopic_statistics(
    user_id: Optional[str] = "default_user",
    topic: Optional[str] = None,
    storage_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregates learning events by (topic, subtopic) pairs.

    Returns:
        List of subtopic summary dicts sorted by weakness_score descending.
    """
    events = get_learning_events(user_id=user_id, topic=topic, storage_path=storage_path)
    subtopic_groups: Dict[tuple[str, str], List[LearningEvent]] = {}
    for ev in events:
        key = (ev.topic or "Computer Science", ev.subtopic or "General")
        subtopic_groups.setdefault(key, []).append(ev)

    results: List[Dict[str, Any]] = []

    for (top_name, sub_name), ev_list in subtopic_groups.items():
        interaction_count = len(ev_list)
        answered_count = sum(1 for e in ev_list if e.status == "ANSWERED")
        refused_count = sum(1 for e in ev_list if e.status == "REFUSED")
        quiz_attempts = sum(1 for e in ev_list if e.status == "QUIZ_ATTEMPTED")
        questions_attempted = sum(e.questions_attempted for e in ev_list)
        questions_correct = sum(e.questions_correct for e in ev_list)
        questions_incorrect = sum(e.questions_incorrect for e in ev_list)

        quiz_accuracy = (
            round(questions_correct / questions_attempted, 4)
            if questions_attempted > 0
            else None
        )

        weakness_score, weakness_level = _compute_weakness(
            questions_attempted=questions_attempted,
            questions_incorrect=questions_incorrect,
            interaction_count=interaction_count,
            refused_count=refused_count,
        )

        results.append({
            "topic": top_name,
            "subtopic": sub_name,
            "interaction_count": interaction_count,
            "answered_count": answered_count,
            "refused_count": refused_count,
            "quiz_attempts": quiz_attempts,
            "questions_attempted": questions_attempted,
            "questions_correct": questions_correct,
            "questions_incorrect": questions_incorrect,
            "quiz_accuracy": quiz_accuracy,
            "weakness_score": weakness_score,
            "weakness_level": weakness_level,
        })

    results.sort(key=lambda x: (x["weakness_score"], x["interaction_count"]), reverse=True)
    return results
