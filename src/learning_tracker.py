"""
Learning Tracker Module for StudentAI.
Tracks learning interactions, quiz outcomes, and query topics.
Persists learning events to a local JSON file: data/learning/learning_events.json.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
from src.config import get_settings

DEFAULT_EVENTS_PATH = Path("data/learning/learning_events.json")

def get_default_events_path() -> Path:
    try:
        return get_settings().learning_events_path
    except Exception:
        return DEFAULT_EVENTS_PATH

VALID_STATUSES = {"ANSWERED", "REFUSED", "QUIZ_GENERATED", "QUIZ_ATTEMPTED"}


@dataclass
class LearningEvent:
    event_id: str
    user_id: str
    timestamp: str
    query: str
    topic: str
    subtopic: str
    intent: str
    status: str
    sources_count: int
    questions_attempted: int = 0
    questions_correct: int = 0
    questions_incorrect: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_storage_dir(path: Path) -> None:
    """Ensures parent directory for learning events exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_all_events(storage_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads all learning event dictionaries from storage file."""
    path = storage_path or get_default_events_path()
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


def save_all_events(events: List[Dict[str, Any]], storage_path: Optional[Path] = None) -> None:
    """Saves learning event dictionaries to storage file atomically."""
    path = storage_path or get_default_events_path()
    _ensure_storage_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)


def record_learning_event(
    query: str,
    topic: str,
    subtopic: str,
    intent: str,
    status: str,
    sources_count: int = 0,
    user_id: str = "default_user",
    storage_path: Optional[Path] = None,
) -> LearningEvent:
    """
    Records a learning query interaction event.

    Args:
        query: User query or prompt.
        topic: Primary academic topic.
        subtopic: Subtopic or concept.
        intent: Pedagogical intent.
        status: One of ANSWERED, REFUSED, QUIZ_GENERATED, QUIZ_ATTEMPTED.
        sources_count: Number of context sources retrieved.
        user_id: Identifier for student user (default: 'default_user').
        storage_path: Optional custom path for test isolation.

    Returns:
        The newly recorded LearningEvent instance.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {VALID_STATUSES}")

    event = LearningEvent(
        event_id=str(uuid.uuid4()),
        user_id=user_id or "default_user",
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=query.strip() if query else "",
        topic=topic.strip() if topic else "Computer Science",
        subtopic=subtopic.strip() if subtopic else "General",
        intent=intent.strip() if intent else "GENERAL_LEARNING",
        status=status,
        sources_count=max(0, sources_count),
        questions_attempted=0,
        questions_correct=0,
        questions_incorrect=0,
    )

    events = load_all_events(storage_path)
    events.append(event.to_dict())
    save_all_events(events, storage_path)
    return event


def record_quiz_attempt(
    topic: str,
    subtopic: str,
    questions_attempted: int,
    questions_correct: int,
    questions_incorrect: int,
    user_id: str = "default_user",
    storage_path: Optional[Path] = None,
) -> LearningEvent:
    """
    Records student performance outcomes for a practice quiz attempt.

    Validation rules:
        - questions_attempted >= 0
        - questions_correct >= 0
        - questions_incorrect >= 0
        - questions_correct + questions_incorrect == questions_attempted

    Args:
        topic: Academic topic.
        subtopic: Academic subtopic.
        questions_attempted: Total questions answered.
        questions_correct: Questions answered correctly.
        questions_incorrect: Questions answered incorrectly.
        user_id: Identifier for student user.
        storage_path: Optional custom path for test isolation.

    Returns:
        The newly created LearningEvent instance.
    """
    if questions_attempted < 0:
        raise ValueError("questions_attempted cannot be negative")
    if questions_correct < 0:
        raise ValueError("questions_correct cannot be negative")
    if questions_incorrect < 0:
        raise ValueError("questions_incorrect cannot be negative")
    if questions_correct + questions_incorrect != questions_attempted:
        raise ValueError(
            f"questions_correct ({questions_correct}) + questions_incorrect ({questions_incorrect}) "
            f"must equal questions_attempted ({questions_attempted})"
        )

    event = LearningEvent(
        event_id=str(uuid.uuid4()),
        user_id=user_id or "default_user",
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=f"Quiz attempt: {subtopic} ({questions_correct}/{questions_attempted})",
        topic=topic.strip() if topic else "Computer Science",
        subtopic=subtopic.strip() if subtopic else "General",
        intent="QUIZ",
        status="QUIZ_ATTEMPTED",
        sources_count=0,
        questions_attempted=questions_attempted,
        questions_correct=questions_correct,
        questions_incorrect=questions_incorrect,
    )

    events = load_all_events(storage_path)
    events.append(event.to_dict())
    save_all_events(events, storage_path)
    return event


def get_learning_events(
    user_id: Optional[str] = "default_user",
    topic: Optional[str] = None,
    storage_path: Optional[Path] = None,
) -> List[LearningEvent]:
    """
    Retrieves filtered learning events.

    Args:
        user_id: User identifier to filter by (if None, all users).
        topic: Academic topic to filter by (case-insensitive substring/match).
        storage_path: Optional custom storage path.

    Returns:
        List of matching LearningEvent objects.
    """
    raw_events = load_all_events(storage_path)
    results: List[LearningEvent] = []

    for item in raw_events:
        if user_id and item.get("user_id") != user_id:
            continue
        if topic and topic.strip():
            evt_topic = item.get("topic", "").strip().lower()
            filt_topic = topic.strip().lower()
            if filt_topic not in evt_topic:
                continue
        results.append(LearningEvent(**item))

    return results


def clear_learning_events(storage_path: Optional[Path] = None) -> None:
    """Clears all stored learning events (primarily for testing and resets)."""
    path = storage_path or DEFAULT_EVENTS_PATH
    if path.is_file():
        path.unlink()
