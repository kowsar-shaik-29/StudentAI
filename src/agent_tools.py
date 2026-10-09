"""
Agent Tools Registry for StudentAI Orchestrator.
Exposes a bounded, strict registry of existing StudentAI capabilities:
- learning_ask
- learning_profile
- learning_recommendations
- revision_topics
- create_study_plan
- career_ask
- analyze_job_fit
- project_list
- project_ask
- project_summary
- project_interview
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import time

# Existing system imports
from src.learning_rag import answer_learning_question
from src.recommendation_engine import get_learning_profile, get_learning_recommendations
from src.revision_engine import get_revision_recommendations
from src.study_planner import create_study_plan
from src.career_rag import answer_career_question, analyze_job_fit
from src.project_rag import list_projects, answer_project_question, summarize_project
from src.project_interview import generate_project_interview_questions


@dataclass
class ToolDefinition:
    name: str
    description: str
    is_write: bool
    parameter_schema: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]


class ToolExecutionError(Exception):
    """Raised when tool validation or execution fails."""
    pass


# ---------------------------------------------------------------------------
# Individual Tool Implementations
# ---------------------------------------------------------------------------

def _tool_learning_ask(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    query = args.get("query")
    if not query or not isinstance(query, str) or not query.strip():
        raise ToolExecutionError("Argument 'query' must be a non-empty string.")
    if len(query) > 2000:
        raise ToolExecutionError("Argument 'query' exceeds maximum length of 2000 characters.")

    top_k = args.get("top_k", 5)
    if not isinstance(top_k, int) or top_k <= 0 or top_k > 20:
        raise ToolExecutionError("Argument 'top_k' must be an integer between 1 and 20.")

    raw_intent = args.get("intent")
    intent = None
    if raw_intent is not None and isinstance(raw_intent, str):
        normalized = raw_intent.strip().upper()
        # Common intent alias mapping
        alias_map = {
            "CONCEPT_EXPLANATION": "EXPLAIN",
            "EXPLANATION": "EXPLAIN",
            "CONCEPT": "EXPLAIN",
            "SUMMARY": "SUMMARIZE",
            "RECAP": "SUMMARIZE",
            "COMPARISON": "COMPARE",
            "CONTRAST": "COMPARE",
            "PRACTICE": "QUIZ",
            "ASSESSMENT": "QUIZ",
            "EXAM": "EXAM_ANSWER",
            "EXAM_PREP": "EXAM_ANSWER",
            "GENERAL": "GENERAL_LEARNING",
            "LEARNING": "GENERAL_LEARNING",
        }
        if normalized in ("EXPLAIN", "SUMMARIZE", "COMPARE", "QUIZ", "EXAM_ANSWER", "GENERAL_LEARNING"):
            intent = normalized
        elif normalized in alias_map:
            intent = alias_map[normalized]
        else:
            raise ToolExecutionError(f"Invalid intent '{raw_intent}'. Valid options: EXPLAIN, SUMMARIZE, COMPARE, QUIZ, EXAM_ANSWER, GENERAL_LEARNING.")

    # Enforce authenticated user_id from execution context
    auth_user_id = context.get("user_id") if context else None
    user_id = auth_user_id or args.get("user_id", "default_user")

    res = answer_learning_question(
        query=query.strip(),
        top_k=top_k,
        intent=intent,
        user_id=user_id,
        sparse_retriever=context.get("sparse_retriever") if context else None,
        client=context.get("qdrant_client") if context else None,
        genai_client=context.get("genai_client") if context else None,
    )
    return {
        "intent": res.get("intent"),
        "topic": res.get("topic"),
        "subtopic": res.get("subtopic"),
        "answer": res.get("answer", ""),
        "questions": res.get("questions", []),
        "sources": res.get("sources", []),
    }


def _tool_learning_profile(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    auth_user_id = context.get("user_id") if context else None
    user_id = auth_user_id or args.get("user_id", "default_user")
    if not isinstance(user_id, str) or not user_id.strip():
        user_id = "default_user"

    prof = get_learning_profile(
        user_id=user_id.strip(),
        sparse_retriever=context.get("sparse_retriever") if context else None,
    )
    return {
        "user_id": prof.get("user_id"),
        "topics_count": len(prof.get("topics", [])),
        "weak_topics": prof.get("weak_topics", []),
        "strong_topics": prof.get("strong_topics", []),
        "insufficient_data_topics": prof.get("insufficient_data_topics", []),
        "recommendations_count": len(prof.get("recommendations", [])),
        "message": prof.get("message"),
    }


def _tool_learning_recommendations(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    auth_user_id = context.get("user_id") if context else None
    user_id = auth_user_id or args.get("user_id", "default_user")
    limit = args.get("limit", 5)
    if not isinstance(limit, int) or limit <= 0 or limit > 50:
        raise ToolExecutionError("Argument 'limit' must be an integer between 1 and 50.")

    topic = args.get("topic")
    if topic is not None and not isinstance(topic, str):
        raise ToolExecutionError("Argument 'topic' must be a string if provided.")

    recs_res = get_learning_recommendations(
        user_id=user_id,
        topic=topic.strip() if topic else None,
        max_recommendations=limit,
        sparse_retriever=context.get("sparse_retriever") if context else None,
    )
    return {
        "user_id": recs_res.get("user_id"),
        "recommendations": recs_res.get("recommendations", []),
        "message": recs_res.get("message"),
    }


def _tool_revision_topics(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    auth_user_id = context.get("user_id") if context else None
    user_id = auth_user_id or args.get("user_id", "default_user")
    subject = args.get("subject")
    limit = args.get("limit", 5)
    if not isinstance(limit, int) or limit <= 0 or limit > 50:
        raise ToolExecutionError("Argument 'limit' must be an integer between 1 and 50.")

    rev_res = get_revision_recommendations(
        user_id=user_id,
        subject=subject.strip() if subject else None,
        limit=limit,
        sparse_retriever=context.get("sparse_retriever") if context else None,
    )
    return {
        "user_id": rev_res.get("user_id"),
        "subject_filter": rev_res.get("subject_filter"),
        "revision_items": rev_res.get("revision_items", []),
        "total_items": rev_res.get("total_items", 0),
        "message": rev_res.get("message"),
    }


def _tool_create_study_plan(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    subject = args.get("subject")
    if not subject or not isinstance(subject, str) or not subject.strip():
        raise ToolExecutionError("Argument 'subject' must be a non-empty string.")

    days = args.get("days", 7)
    if not isinstance(days, int) or days <= 0 or days > 365:
        raise ToolExecutionError("Argument 'days' must be an integer between 1 and 365.")

    daily_minutes = args.get("daily_minutes", 60)
    if not isinstance(daily_minutes, int) or daily_minutes <= 0 or daily_minutes > 1440:
        raise ToolExecutionError("Argument 'daily_minutes' must be an integer between 1 and 1440.")

    auth_user_id = context.get("user_id") if context else None
    user_id = auth_user_id or args.get("user_id", "default_user")
    persist = args.get("persist", True)

    plan_res = create_study_plan(
        subject=subject.strip(),
        days=days,
        user_id=user_id,
        daily_minutes=daily_minutes,
        sparse_retriever=context.get("sparse_retriever") if context else None,
        persist=persist,
    )
    return {
        "plan_id": plan_res.get("plan_id"),
        "subject": plan_res.get("subject"),
        "days": plan_res.get("days"),
        "daily_minutes": plan_res.get("daily_minutes"),
        "plan": plan_res.get("plan", []),
        "message": plan_res.get("message"),
    }


def _tool_career_ask(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    query = args.get("query")
    if not query or not isinstance(query, str) or not query.strip():
        raise ToolExecutionError("Argument 'query' must be a non-empty string.")

    top_k = args.get("top_k", 5)
    if not isinstance(top_k, int) or top_k <= 0 or top_k > 20:
        raise ToolExecutionError("Argument 'top_k' must be an integer between 1 and 20.")

    doc_type = args.get("document_type")
    if doc_type is not None and doc_type not in ("resume", "job_description", "other"):
        raise ToolExecutionError(f"Invalid document_type '{doc_type}'.")

    doc_id = args.get("document_id")
    auth_user_id = context.get("user_id") if context else None

    res = answer_career_question(
        query=query.strip(),
        top_k=top_k,
        document_type=doc_type,
        document_id=doc_id.strip() if doc_id else None,
        sparse_retriever=context.get("sparse_retriever") if context else None,
        client=context.get("qdrant_client") if context else None,
        genai_client=context.get("genai_client") if context else None,
        owner_id=auth_user_id,
    )
    return {
        "query": res.get("query"),
        "answer": res.get("answer"),
        "sources": res.get("sources", []),
    }


def _tool_analyze_job_fit(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resume_id = args.get("resume_id")
    if not resume_id or not isinstance(resume_id, str) or not resume_id.strip():
        raise ToolExecutionError("Argument 'resume_id' must be a non-empty string.")

    job_id = args.get("job_id")
    if not job_id or not isinstance(job_id, str) or not job_id.strip():
        raise ToolExecutionError("Argument 'job_id' must be a non-empty string.")

    auth_user_id = context.get("user_id") if context else None

    try:
        res = analyze_job_fit(
            resume_document_id=resume_id.strip(),
            job_document_id=job_id.strip(),
            sparse_retriever=context.get("sparse_retriever") if context else None,
            genai_client=context.get("genai_client") if context else None,
            owner_id=auth_user_id,
        )
    except ValueError as e:
        raise ToolExecutionError(str(e))

    return {
        "resume_id": res.get("resume", {}).get("document_id"),
        "job_id": res.get("job", {}).get("document_id"),
        "coverage_percentage": res.get("match", {}).get("coverage_percentage", 0.0),
        "matched_count": res.get("match", {}).get("matched_count", 0),
        "missing_count": res.get("match", {}).get("missing_count", 0),
        "partial_count": res.get("match", {}).get("partial_count", 0),
        "matched_skills": res.get("match", {}).get("matched_skills", []),
        "partial_matches": res.get("match", {}).get("partial_matches", []),
        "missing_skills": res.get("match", {}).get("missing_skills", []),
        "limitations": res.get("match", {}).get("limitations", []),
    }


def _tool_project_list(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    sr = context.get("sparse_retriever") if context else None
    auth_user_id = context.get("user_id") if context else None
    projs = list_projects(sparse_retriever=sr, owner_id=auth_user_id)
    return {
        "total_projects": len(projs),
        "projects": [
            {
                "project_id": p.get("project_id"),
                "project_name": p.get("project_name"),
                "document_count": p.get("document_count", 0),
                "filenames": p.get("filenames", []),
            }
            for p in projs
        ],
    }


def _tool_project_ask(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    query = args.get("query")
    if not query or not isinstance(query, str) or not query.strip():
        raise ToolExecutionError("Argument 'query' must be a non-empty string.")

    project_id = args.get("project_id")
    top_k = args.get("top_k", 5)
    if not isinstance(top_k, int) or top_k <= 0 or top_k > 20:
        raise ToolExecutionError("Argument 'top_k' must be an integer between 1 and 20.")

    auth_user_id = context.get("user_id") if context else None

    res = answer_project_question(
        query=query.strip(),
        project_id=project_id.strip() if project_id else None,
        top_k=top_k,
        sparse_retriever=context.get("sparse_retriever") if context else None,
        client=context.get("qdrant_client") if context else None,
        genai_client=context.get("genai_client") if context else None,
        owner_id=auth_user_id,
    )
    return {
        "query": res.get("query"),
        "project_id": res.get("project_id"),
        "answer": res.get("answer"),
        "ambiguous": res.get("ambiguous", False),
        "available_projects": res.get("available_projects", []),
        "sources": res.get("sources", []),
    }


def _tool_project_summary(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    project_id = args.get("project_id")
    if not project_id or not isinstance(project_id, str) or not project_id.strip():
        raise ToolExecutionError("Argument 'project_id' must be a non-empty string.")

    auth_user_id = context.get("user_id") if context else None

    try:
        res = summarize_project(
            project_id=project_id.strip(),
            sparse_retriever=context.get("sparse_retriever") if context else None,
            genai_client=context.get("genai_client") if context else None,
            owner_id=auth_user_id,
        )
    except ValueError as e:
        raise ToolExecutionError(str(e))
    return res


def _tool_project_interview(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    project_id = args.get("project_id")
    if not project_id or not isinstance(project_id, str) or not project_id.strip():
        raise ToolExecutionError("Argument 'project_id' must be a non-empty string.")

    count = args.get("count", 5)
    if not isinstance(count, int) or count <= 0 or count > 20:
        raise ToolExecutionError("Argument 'count' must be an integer between 1 and 20.")

    difficulty = args.get("difficulty", "MIXED")
    if difficulty not in ("EASY", "MEDIUM", "HARD", "MIXED"):
        raise ToolExecutionError(f"Invalid difficulty '{difficulty}'.")

    auth_user_id = context.get("user_id") if context else None

    res = generate_project_interview_questions(
        project_id=project_id.strip(),
        count=count,
        difficulty=difficulty,
        sparse_retriever=context.get("sparse_retriever") if context else None,
        genai_client=context.get("genai_client") if context else None,
        owner_id=auth_user_id,
    )
    return {
        "project_id": res.get("project_id"),
        "project_name": res.get("project_name"),
        "questions": res.get("questions", []),
        "insufficient_evidence": res.get("insufficient_evidence", False),
        "sources": res.get("sources", []),
    }


# ---------------------------------------------------------------------------
# Strict Tool Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, ToolDefinition] = {
    "learning_ask": ToolDefinition(
        name="learning_ask",
        description="Ask an academic concept question with intent classification and page citations.",
        is_write=False,
        parameter_schema={
            "query": {"type": "string", "required": True},
            "intent": {"type": "string", "required": False, "choices": ["EXPLAIN", "SUMMARIZE", "COMPARE", "QUIZ", "EXAM_ANSWER", "GENERAL_LEARNING"]},
            "top_k": {"type": "integer", "required": False, "default": 5},
            "user_id": {"type": "string", "required": False, "default": "default_user"},
        },
        handler=_tool_learning_ask,
    ),
    "learning_profile": ToolDefinition(
        name="learning_profile",
        description="Fetch student learning profile showing weak, strong, and unassessed topics.",
        is_write=False,
        parameter_schema={
            "user_id": {"type": "string", "required": False, "default": "default_user"},
        },
        handler=_tool_learning_profile,
    ),
    "learning_recommendations": ToolDefinition(
        name="learning_recommendations",
        description="Fetch prioritized actionable study recommendations and academic sources.",
        is_write=False,
        parameter_schema={
            "user_id": {"type": "string", "required": False, "default": "default_user"},
            "topic": {"type": "string", "required": False},
            "limit": {"type": "integer", "required": False, "default": 5},
        },
        handler=_tool_learning_recommendations,
    ),
    "revision_topics": ToolDefinition(
        name="revision_topics",
        description="Retrieve prioritized revision topics based on weakness and quiz accuracy.",
        is_write=False,
        parameter_schema={
            "user_id": {"type": "string", "required": False, "default": "default_user"},
            "subject": {"type": "string", "required": False},
            "limit": {"type": "integer", "required": False, "default": 5},
        },
        handler=_tool_revision_topics,
    ),
    "create_study_plan": ToolDefinition(
        name="create_study_plan",
        description="Generate a deterministic multi-day study schedule partitioned by daily minutes.",
        is_write=True,
        parameter_schema={
            "subject": {"type": "string", "required": True},
            "days": {"type": "integer", "required": False, "default": 7},
            "daily_minutes": {"type": "integer", "required": False, "default": 60},
            "user_id": {"type": "string", "required": False, "default": "default_user"},
            "persist": {"type": "boolean", "required": False, "default": True},
        },
        handler=_tool_create_study_plan,
    ),
    "career_ask": ToolDefinition(
        name="career_ask",
        description="Answer career, resume, and job description queries grounded in career documents.",
        is_write=False,
        parameter_schema={
            "query": {"type": "string", "required": True},
            "document_type": {"type": "string", "required": False, "choices": ["resume", "job_description", "other"]},
            "document_id": {"type": "string", "required": False},
            "top_k": {"type": "integer", "required": False, "default": 5},
        },
        handler=_tool_career_ask,
    ),
    "analyze_job_fit": ToolDefinition(
        name="analyze_job_fit",
        description="Perform deterministic skill matching and gap analysis between resume and job description.",
        is_write=False,
        parameter_schema={
            "resume_id": {"type": "string", "required": True},
            "job_id": {"type": "string", "required": True},
        },
        handler=_tool_analyze_job_fit,
    ),
    "project_list": ToolDefinition(
        name="project_list",
        description="List all indexed student software projects and their documentation files.",
        is_write=False,
        parameter_schema={},
        handler=_tool_project_list,
    ),
    "project_ask": ToolDefinition(
        name="project_ask",
        description="Answer technical questions about architecture or stack grounded in project documentation.",
        is_write=False,
        parameter_schema={
            "query": {"type": "string", "required": True},
            "project_id": {"type": "string", "required": False},
            "top_k": {"type": "integer", "required": False, "default": 5},
        },
        handler=_tool_project_ask,
    ),
    "project_summary": ToolDefinition(
        name="project_summary",
        description="Generate a structured technical architecture summary of a software project.",
        is_write=False,
        parameter_schema={
            "project_id": {"type": "string", "required": True},
        },
        handler=_tool_project_summary,
    ),
    "project_interview": ToolDefinition(
        name="project_interview",
        description="Generate grounded technical interview questions from software project documentation.",
        is_write=False,
        parameter_schema={
            "project_id": {"type": "string", "required": True},
            "count": {"type": "integer", "required": False, "default": 5},
            "difficulty": {"type": "string", "required": False, "default": "MIXED", "choices": ["EASY", "MEDIUM", "HARD", "MIXED"]},
        },
        handler=_tool_project_interview,
    ),
}


def get_available_tools() -> List[Dict[str, Any]]:
    """Returns a list of tool metadata for planner inspection."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "is_write": t.is_write,
            "parameters": t.parameter_schema,
        }
        for t in TOOL_REGISTRY.values()
    ]


def execute_tool(tool_name: str, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes a tool from the registry with strict validation.

    Args:
        tool_name: Name of registered tool.
        args: Input arguments dictionary.
        context: Shared runtime context (clients, caches).

    Returns:
        Structured result dictionary:
        {
            "success": bool,
            "tool_name": str,
            "output": Any,
            "error": Optional[str],
            "duration_ms": float,
        }
    """
    if tool_name not in TOOL_REGISTRY:
        raise ToolExecutionError(f"Tool '{tool_name}' is not registered. Available tools: {list(TOOL_REGISTRY.keys())}")

    tool_def = TOOL_REGISTRY[tool_name]
    t0 = time.perf_counter()

    try:
        res = tool_def.handler(args, context=context)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": True,
            "tool_name": tool_name,
            "output": res,
            "error": None,
            "duration_ms": round(t_ms, 2),
        }
    except ToolExecutionError as e:
        t_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "tool_name": tool_name,
            "output": None,
            "error": str(e),
            "duration_ms": round(t_ms, 2),
        }
    except Exception as e:
        t_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "tool_name": tool_name,
            "output": None,
            "error": f"Internal execution failure in tool '{tool_name}': {str(e)}",
            "duration_ms": round(t_ms, 2),
        }
