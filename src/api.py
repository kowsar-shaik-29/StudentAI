"""
FastAPI Backend for StudentAI.
Exposes modular REST endpoints for frontend integration:
- Health check: GET /health
- Agent Orchestration: POST /api/agent/run
- Learning RAG: POST /api/learning/ask
- Learning Profile: GET /api/learning/profile
- Recommendations: GET /api/learning/recommendations
- Study Planning: POST /api/study-plans
- Career RAG: POST /api/career/ask
- Job Fit Analysis: POST /api/career/job-fit
- Projects Listing: GET /api/projects
- Project QA: POST /api/projects/ask
- Document Upload & Ingestion: POST /api/documents/upload

Design & Safety:
- Input validation with strict Pydantic models.
- Sanitized error handling (no secrets or leaked paths in responses).
- Configurable CORS middleware.
- Threadpool offloading (async/sync safety) for AI tasks.
- Upload security: size limits (max 10MB), extension whitelisting (.pdf, .md, .txt), path traversal prevention.
"""

from typing import Dict, Any, List, Optional
import os
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.config import get_settings
from src.agent_orchestrator import run_agent, AgentResult
from src.learning_rag import answer_learning_question
from src.recommendation_engine import get_learning_profile, get_learning_recommendations
from src.study_planner import create_study_plan
from src.career_rag import answer_career_question, analyze_job_fit
from src.project_rag import list_projects, answer_project_question
from src.ingestion import ingest_document

settings = get_settings()

app = FastAPI(
    title="StudentAI API",
    description="Personal Knowledge, Learning & Career RAG Backend",
    version=settings.app_version,
)

# CORS Setup
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str


class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User goal or query")


class ExecutionTraceItem(BaseModel):
    step: int
    tool_name: str
    args: Dict[str, Any]
    status: str
    duration_ms: float
    error: Optional[str] = None


class AgentRunResponse(BaseModel):
    query: str
    answer: str
    classification: str
    workflow: str
    execution_trace: List[ExecutionTraceItem]
    sources: List[Dict[str, Any]]
    citations_verified: bool
    citation_report: Optional[Dict[str, Any]] = None
    limitations: List[str]
    total_duration_ms: float


class LearningAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    intent: Optional[str] = Field(None, description="Optional intent override (EXPLAIN, SUMMARIZE, COMPARE, QUIZ, EXAM_ANSWER)")
    user_id: str = Field("default_user", max_length=100)

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v):
        if v is not None and v not in ("EXPLAIN", "SUMMARIZE", "COMPARE", "QUIZ", "EXAM_ANSWER", "GENERAL_LEARNING"):
            raise ValueError(f"Invalid intent '{v}'.")
        return v


class LearningAskResponse(BaseModel):
    intent: str
    topic: str
    subtopic: str
    answer: str
    questions: Optional[List[Dict[str, Any]]] = None
    sources: List[Dict[str, Any]]


class StudyPlanRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=100)
    days: int = Field(7, ge=1, le=365)
    daily_minutes: int = Field(60, ge=1, le=1440)
    user_id: str = Field("default_user", max_length=100)


class StudyPlanResponse(BaseModel):
    plan_id: Optional[str]
    subject: str
    days: int
    daily_minutes: int
    plan: List[Dict[str, Any]]
    message: Optional[str] = None


class CareerAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    document_type: Optional[str] = Field(None, description="resume or job_description")
    document_id: Optional[str] = None

    @field_validator("document_type")
    @classmethod
    def validate_doc_type(cls, v):
        if v is not None and v not in ("resume", "job_description", "other"):
            raise ValueError(f"Invalid document_type '{v}'.")
        return v


class CareerAskResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]


class JobFitRequest(BaseModel):
    resume_id: str = Field(..., min_length=1, max_length=200)
    job_id: str = Field(..., min_length=1, max_length=200)

    @field_validator("resume_id", "job_id")
    @classmethod
    def validate_id_not_path(cls, v):
        if "/" in v or "\\" in v or ".." in v:
            raise ValueError("Document ID cannot contain path characters.")
        return v


class ProjectAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    project_id: Optional[str] = Field(None, max_length=100)
    top_k: int = Field(5, ge=1, le=20)


class ProjectAskResponse(BaseModel):
    query: str
    project_id: Optional[str]
    answer: str
    ambiguous: bool
    available_projects: List[str]
    sources: List[Dict[str, Any]]


class UploadDocumentResponse(BaseModel):
    file_name: str
    document_id: str
    category: str
    subject: str
    document_type: str
    chunks: int
    pages: int
    embeddings: int


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns application status and version without leaking secrets."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
    )


@app.post("/api/agent/run", response_model=AgentRunResponse, tags=["Agent"])
async def run_agent_endpoint(req: AgentRunRequest):
    """Executes multi-step or single-step agentic orchestration for a user goal."""
    try:
        res: AgentResult = await run_in_threadpool(run_agent, user_query=req.query)
        traces = [
            ExecutionTraceItem(
                step=t.step,
                tool_name=t.tool_name,
                args=t.args,
                status=t.status,
                duration_ms=t.duration_ms,
                error=t.error,
            )
            for t in res.execution_trace
        ]
        return AgentRunResponse(
            query=res.query,
            answer=res.answer,
            classification=res.classification,
            workflow=res.workflow,
            execution_trace=traces,
            sources=res.sources,
            citations_verified=res.citations_verified,
            citation_report=res.citation_report,
            limitations=res.limitations,
            total_duration_ms=res.total_duration_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(e)}",
        )


@app.post("/api/learning/ask", response_model=LearningAskResponse, tags=["Learning"])
async def learning_ask_endpoint(req: LearningAskRequest):
    """Answers an academic question with pedagogical intent classification."""
    try:
        res = await run_in_threadpool(
            answer_learning_question,
            query=req.query,
            top_k=req.top_k,
            intent=req.intent,
            user_id=req.user_id,
        )
        return LearningAskResponse(
            intent=res.get("intent", "GENERAL_LEARNING"),
            topic=res.get("topic", "Computer Science"),
            subtopic=res.get("subtopic", "General"),
            answer=res.get("answer", ""),
            questions=res.get("questions"),
            sources=res.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Learning query error: {str(e)}",
        )


@app.get("/api/learning/profile", tags=["Learning"])
async def learning_profile_endpoint(user_id: str = Query("default_user", max_length=100)):
    """Fetches student learning profile showing weak, strong, and unassessed topics."""
    try:
        prof = await run_in_threadpool(get_learning_profile, user_id=user_id)
        return prof
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving learning profile: {str(e)}",
        )


@app.get("/api/learning/recommendations", tags=["Learning"])
async def learning_recommendations_endpoint(
    user_id: str = Query("default_user", max_length=100),
    topic: Optional[str] = Query(None, max_length=100),
    limit: int = Query(5, ge=1, le=50),
):
    """Fetches actionable personalized study recommendations and academic sources."""
    try:
        recs = await run_in_threadpool(
            get_learning_recommendations,
            user_id=user_id,
            topic=topic,
            max_recommendations=limit,
        )
        return recs
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving recommendations: {str(e)}",
        )


@app.post("/api/study-plans", response_model=StudyPlanResponse, tags=["Study Plan"])
async def study_plan_endpoint(req: StudyPlanRequest):
    """Generates a structured, multi-day academic study schedule."""
    try:
        plan_res = await run_in_threadpool(
            create_study_plan,
            subject=req.subject,
            days=req.days,
            user_id=req.user_id,
            daily_minutes=req.daily_minutes,
        )
        return StudyPlanResponse(
            plan_id=plan_res.get("plan_id"),
            subject=plan_res.get("subject", req.subject),
            days=plan_res.get("days", req.days),
            daily_minutes=plan_res.get("daily_minutes", req.daily_minutes),
            plan=plan_res.get("plan", []),
            message=plan_res.get("message"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Study plan creation error: {str(e)}",
        )


@app.post("/api/career/ask", response_model=CareerAskResponse, tags=["Career"])
async def career_ask_endpoint(req: CareerAskRequest):
    """Answers a career question grounded strictly in career documents."""
    try:
        res = await run_in_threadpool(
            answer_career_question,
            query=req.query,
            top_k=req.top_k,
            document_type=req.document_type,
            document_id=req.document_id,
        )
        return CareerAskResponse(
            query=res.get("query", req.query),
            answer=res.get("answer", ""),
            sources=res.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Career query error: {str(e)}",
        )


@app.post("/api/career/job-fit", tags=["Career"])
async def job_fit_endpoint(req: JobFitRequest):
    """Conducts deterministic skill matching and gap analysis between resume and job."""
    try:
        res = await run_in_threadpool(
            analyze_job_fit,
            resume_document_id=req.resume_id,
            job_document_id=req.job_id,
        )
        return res
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job fit analysis error: {str(e)}",
        )


@app.get("/api/projects", tags=["Projects"])
async def list_projects_endpoint():
    """Lists all indexed student software projects and their documentation files."""
    try:
        projs = await run_in_threadpool(list_projects)
        return {"projects": projs, "total": len(projs)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing projects: {str(e)}",
        )


@app.post("/api/projects/ask", response_model=ProjectAskResponse, tags=["Projects"])
async def project_ask_endpoint(req: ProjectAskRequest):
    """Answers a question grounded in software project documentation."""
    try:
        res = await run_in_threadpool(
            answer_project_question,
            query=req.query,
            project_id=req.project_id,
            top_k=req.top_k,
        )
        return ProjectAskResponse(
            query=res.get("query", req.query),
            project_id=res.get("project_id"),
            answer=res.get("answer", ""),
            ambiguous=res.get("ambiguous", False),
            available_projects=res.get("available_projects", []),
            sources=res.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project query error: {str(e)}",
        )


@app.post("/api/documents/upload", response_model=UploadDocumentResponse, tags=["Documents"])
async def upload_document_endpoint(
    file: UploadFile = File(...),
    category: str = Form("academic"),
    subject: Optional[str] = Form("general"),
    document_type: Optional[str] = Form("notes"),
):
    """
    Safely ingests an uploaded PDF, Markdown, or text document into the knowledge base.
    Validates file extension, bounds file size (max 10MB), and isolates execution.
    """
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}",
        )

    # Read content with size check
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB.",
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Save to a temporary file for ingestion pipeline
    temp_dir = Path(tempfile.gettempdir()) / "studentai_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / filename

    try:
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        stats = await run_in_threadpool(
            ingest_document,
            file_path=temp_file_path,
            category=category,
            subject=subject,
            document_type=document_type,
        )

        return UploadDocumentResponse(
            file_name=stats.get("file_name", filename),
            document_id=stats.get("document_id", filename),
            category=stats.get("category", category),
            subject=stats.get("subject", subject or "general"),
            document_type=stats.get("document_type", document_type or "notes"),
            chunks=stats.get("chunks", 0),
            pages=stats.get("pages", 0),
            embeddings=stats.get("embeddings", 0),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )
    finally:
        # Clean up temporary upload file
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass
