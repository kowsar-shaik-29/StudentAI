"""
Study Planner Module for StudentAI.
Generates structured, deterministic, multi-day academic study plans based on:
1. Learning analytics & subtopic weakness levels (HIGH, MEDIUM, INSUFFICIENT_DATA, LOW)
2. Academic document discovery (strictly category="academic")
3. Multi-day distribution with deterministic time allocation
4. Diagnostic quiz scheduling for unquizzed topics
5. Local persistence to data/learning/study_plans.json

Pure deterministic Python — NO LLM/Gemini calls, NO machine learning.
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime, timezone
import json
import uuid

from src.learning_analytics import get_subtopic_statistics
from src.recommendation_engine import find_academic_sources
from src.sparse_retriever import get_sparse_retriever, SparseRetriever

DEFAULT_PLANS_PATH = Path("data/learning/study_plans.json")

SUBJECT_ALIASES: Dict[str, List[str]] = {
    "dbms": ["dbms", "database", "databases", "database management", "database management systems", "sql"],
    "computer_networks": ["computer networks", "networking", "networks", "computer network", "cn", "tcp", "network"],
    "operating_systems": ["operating systems", "os", "operating system"],
    "dsa": ["dsa", "data structures", "algorithms", "data structures & algorithms", "graphs", "graph"],
}


def normalize_subject_name(subject_str: str) -> str:
    """Normalizes subject names and aliases to a canonical key."""
    if not subject_str:
        return ""
    clean = subject_str.strip().lower()
    for canon, aliases in SUBJECT_ALIASES.items():
        if clean == canon or clean in aliases:
            return canon
        for a in aliases:
            if clean in a or a in clean:
                return canon
    return clean.replace(" ", "_")


def is_subject_match(item_subject_or_topic: str, query_subject: str) -> bool:
    """Checks whether an item topic or subject matches the query subject."""
    if not item_subject_or_topic or not query_subject:
        return False
    norm_query = normalize_subject_name(query_subject)
    norm_item = normalize_subject_name(item_subject_or_topic)
    if norm_query == norm_item:
        return True
    # Substring check
    q_low = query_subject.strip().lower()
    i_low = item_subject_or_topic.strip().lower()
    return q_low in i_low or i_low in q_low


def _ensure_plans_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_all_study_plans(storage_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = storage_path or DEFAULT_PLANS_PATH
    if not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_study_plan(plan_data: Dict[str, Any], storage_path: Optional[Path] = None) -> None:
    path = storage_path or DEFAULT_PLANS_PATH
    _ensure_plans_dir(path)
    plans = load_all_study_plans(path)
    plans.append(plan_data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2, ensure_ascii=False)


def get_study_plans(
    user_id: str = "default_user",
    storage_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    plans = load_all_study_plans(storage_path)
    return [p for p in plans if p.get("user_id") == user_id]


def get_latest_study_plan(
    user_id: str = "default_user",
    subject: Optional[str] = None,
    storage_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    user_plans = get_study_plans(user_id=user_id, storage_path=storage_path)
    if subject and subject.strip():
        user_plans = [p for p in user_plans if is_subject_match(p.get("subject", ""), subject)]
    if not user_plans:
        return None
    return user_plans[-1]


def _build_daily_activities(
    focus_topic: Dict[str, Any],
    daily_minutes: int,
    strategy: str,
) -> List[Dict[str, Any]]:
    """
    Builds a list of deterministic activities for a single day based on topic priority.
    Total estimated minutes will never exceed daily_minutes.
    """
    subtopic = focus_topic["subtopic"]
    priority = focus_topic["priority"]
    activities: List[Dict[str, Any]] = []

    if strategy == "mock_test":
        # Mock Test day: Mock Test (60%), Revision (40%)
        t_mock = max(5, int(daily_minutes * 0.60))
        t_rev = daily_minutes - t_mock
        return [
            {"type": "MOCK_TEST", "description": f"Comprehensive mock test covering {subtopic}", "minutes": t_mock},
            {"type": "REVISION", "description": "Post-mock answer review and error analysis", "minutes": t_rev},
        ]

    if strategy == "mixed_revision":
        # Mixed Revision: Revision (60%), Practice (40%)
        t_rev = max(5, int(daily_minutes * 0.60))
        t_prac = daily_minutes - t_rev
        return [
            {"type": "REVISION", "description": f"Active recall summary and formula review of {subtopic}", "minutes": t_rev},
            {"type": "PRACTICE", "description": f"Mixed problem set on {subtopic}", "minutes": t_prac},
        ]

    if priority == "HIGH":
        if strategy == "revisit_quiz":
            # Revisit day: Review (40%), Practice (25%), Quiz (35%)
            t_rev = max(5, int(daily_minutes * 0.40))
            t_prac = max(5, int(daily_minutes * 0.25))
            t_quiz = daily_minutes - (t_rev + t_prac)
            activities = [
                {"type": "REVIEW", "description": f"Review complex concepts in {subtopic}", "minutes": t_rev},
                {"type": "PRACTICE", "description": f"Solve challenging practice questions on {subtopic}", "minutes": t_prac},
                {"type": "QUIZ", "description": f"Take timed targeted quiz on {subtopic}", "minutes": t_quiz},
            ]
        else:
            # Standard High day: Review (45%), Practice (25%), Quiz (30%)
            t_rev = max(5, int(daily_minutes * 0.45))
            t_prac = max(5, int(daily_minutes * 0.25))
            t_quiz = daily_minutes - (t_rev + t_prac)
            activities = [
                {"type": "REVIEW", "description": f"Detailed review of {subtopic}", "minutes": t_rev},
                {"type": "PRACTICE", "description": f"Guided practice problems on {subtopic}", "minutes": t_prac},
                {"type": "QUIZ", "description": f"Focused assessment quiz on {subtopic}", "minutes": t_quiz},
            ]

    elif priority == "MEDIUM":
        # Review (50%), Practice (50%)
        t_rev = max(5, int(daily_minutes * 0.50))
        t_prac = daily_minutes - t_rev
        activities = [
            {"type": "REVIEW", "description": f"Conceptual revision of {subtopic}", "minutes": t_rev},
            {"type": "PRACTICE", "description": f"Exercise sets and practice questions on {subtopic}", "minutes": t_prac},
        ]

    elif priority == "INSUFFICIENT_DATA":
        # Diagnostic strategy: Review primer (40%), Diagnostic Quiz (60%)
        t_diag = max(5, int(daily_minutes * 0.60))
        t_rev = daily_minutes - t_diag
        activities = [
            {"type": "REVIEW", "description": f"Quick overview of {subtopic}", "minutes": t_rev},
            {"type": "DIAGNOSTIC", "description": f"Diagnostic quiz to measure baseline understanding of {subtopic}", "minutes": t_diag},
        ]

    elif strategy == "mock_test":
        # Mock Test day: Mock Test (60%), Revision (40%)
        t_mock = max(5, int(daily_minutes * 0.60))
        t_rev = daily_minutes - t_mock
        activities = [
            {"type": "MOCK_TEST", "description": f"Comprehensive mock test covering {subtopic}", "minutes": t_mock},
            {"type": "REVISION", "description": "Post-mock answer review and error analysis", "minutes": t_rev},
        ]

    elif strategy == "mixed_revision":
        # Mixed Revision: Revision (60%), Practice (40%)
        t_rev = max(5, int(daily_minutes * 0.60))
        t_prac = daily_minutes - t_rev
        activities = [
            {"type": "REVISION", "description": f"Active recall summary of {subtopic}", "minutes": t_rev},
            {"type": "PRACTICE", "description": f"Mixed problem set on {subtopic}", "minutes": t_prac},
        ]

    else:
        # LOW weakness: Periodic Revision (60%), Mock Practice (40%)
        t_rev = max(5, int(daily_minutes * 0.60))
        t_prac = daily_minutes - t_rev
        activities = [
            {"type": "REVISION", "description": f"Quick review of key definitions in {subtopic}", "minutes": t_rev},
            {"type": "PRACTICE", "description": f"Reinforcement exercises on {subtopic}", "minutes": t_prac},
        ]

    return activities


def create_study_plan(
    subject: str,
    days: int,
    user_id: str = "default_user",
    daily_minutes: int = 60,
    storage_path: Optional[Path] = None,
    events_storage_path: Optional[Path] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Creates a deterministic, grounded study plan for a subject.

    Validation rules:
        - subject: non-empty string
        - days: 1 <= days <= 365
        - daily_minutes: 1 <= daily_minutes <= 1440
    """
    if not subject or not subject.strip():
        raise ValueError("Subject cannot be empty or whitespace only")
    if not isinstance(days, int) or days <= 0:
        raise ValueError(f"Days must be a positive integer, got {days}")
    if days > 365:
        raise ValueError(f"Days cannot exceed 365, got {days}")
    if not isinstance(daily_minutes, int) or daily_minutes <= 0:
        raise ValueError(f"daily_minutes must be a positive integer, got {daily_minutes}")
    if daily_minutes > 1440:
        raise ValueError(f"daily_minutes cannot exceed 1440 (24 hours), got {daily_minutes}")

    clean_subject = subject.strip()
    sr = sparse_retriever or get_sparse_retriever()

    # 1. Fetch user analytics filtered by subject
    all_subtopics = get_subtopic_statistics(
        user_id=user_id,
        storage_path=events_storage_path,
    )
    matching_subtopics = [
        s for s in all_subtopics
        if is_subject_match(s.get("topic", ""), clean_subject)
    ]

    # 2. Check academic sources in corpus for this subject
    academic_corpus_docs = set()
    for d in sr.documents.values():
        if d.get("category") == "academic":
            doc_subj = d.get("subject", "")
            doc_file = d.get("file_name", "")
            if is_subject_match(doc_subj, clean_subject):
                academic_corpus_docs.add(doc_file)

    # If no learning history for this subject and no academic documents in corpus
    if not matching_subtopics and not academic_corpus_docs:
        return {
            "subject": clean_subject,
            "days": days,
            "daily_minutes": daily_minutes,
            "plan": [],
            "message": f"Not enough academic learning data is available to create a grounded study plan for '{clean_subject}'.",
        }

    # 3. Build prioritized list of topics
    # Priority order: HIGH (1), MEDIUM (2), INSUFFICIENT_DATA (3), LOW (4)
    # Higher weakness score first, then alphabetical subtopic
    priority_map = {"HIGH": 1, "MEDIUM": 2, "INSUFFICIENT_DATA": 3, "LOW": 4}

    prioritized_topics: List[Dict[str, Any]] = []

    for item in matching_subtopics:
        t_name = item["topic"]
        s_name = item["subtopic"]
        lvl = item["weakness_level"]
        score = item["weakness_score"]

        # Validate academic source
        sources = find_academic_sources(
            topic=t_name,
            subtopic=s_name,
            sparse_retriever=sr,
        )

        prioritized_topics.append({
            "topic": t_name,
            "subtopic": s_name,
            "priority": lvl,
            "weakness_score": score,
            "accuracy": item["quiz_accuracy"],
            "sources": sources,
        })

    def topic_sort_key(x: Dict[str, Any]):
        p_num = priority_map.get(x["priority"], 99)
        return (p_num, -x["weakness_score"], x["subtopic"].lower())

    prioritized_topics.sort(key=topic_sort_key)

    # If matching_subtopics is empty but academic docs exist, create a diagnostic topic
    if not prioritized_topics and academic_corpus_docs:
        prioritized_topics.append({
            "topic": clean_subject,
            "subtopic": f"General {clean_subject} Concepts",
            "priority": "INSUFFICIENT_DATA",
            "weakness_score": 0.0,
            "accuracy": None,
            "sources": sorted(list(academic_corpus_docs)),
        })

    # 4. Multi-day schedule generation
    high_topics = [t for t in prioritized_topics if t["priority"] == "HIGH"]
    med_topics = [t for t in prioritized_topics if t["priority"] == "MEDIUM"]
    insuf_topics = [t for t in prioritized_topics if t["priority"] == "INSUFFICIENT_DATA"]
    low_topics = [t for t in prioritized_topics if t["priority"] == "LOW"]

    daily_plan: List[Dict[str, Any]] = []

    for d in range(1, days + 1):
        strategy = "standard"

        # Determine focus topic for day d
        if days == 1:
            focus = prioritized_topics[0]
            strategy = "focused"
        elif days >= 7 and d == days:
            # Last day of week/plan: Final mock test & review
            focus = prioritized_topics[0]
            strategy = "mock_test"
        elif days >= 7 and d == days - 1:
            # Penultimate day: Mixed active recall revision
            focus = high_topics[0] if high_topics else prioritized_topics[0]
            strategy = "mixed_revision"
        elif high_topics and days >= 7 and d == 3 and len(prioritized_topics) <= 2:
            # Day 3: Revisit highest weak topic with focused quiz in 7+ day plans
            focus = high_topics[0]
            strategy = "revisit_quiz"
        else:
            # Distribute topics across available days
            # Cycle through prioritized topics
            topic_idx = (d - 1) % len(prioritized_topics)
            focus = prioritized_topics[topic_idx]
            strategy = "standard"

        activities = _build_daily_activities(
            focus_topic=focus,
            daily_minutes=daily_minutes,
            strategy=strategy,
        )

        total_est = sum(a["minutes"] for a in activities)

        daily_plan.append({
            "day": d,
            "strategy": strategy,
            "focus": [
                {
                    "topic": focus["topic"],
                    "subtopic": focus["subtopic"],
                    "priority": focus["priority"],
                    "sources": focus["sources"],
                }
            ],
            "estimated_minutes": total_est,
            "activities": activities,
        })

    plan_result = {
        "plan_id": str(uuid.uuid4()),
        "user_id": user_id,
        "subject": clean_subject,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "daily_minutes": daily_minutes,
        "plan": daily_plan,
    }

    if persist:
        save_study_plan(plan_result, storage_path)

    return plan_result
