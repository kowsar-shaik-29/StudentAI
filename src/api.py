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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Query, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.config import get_settings, is_storage_writable
from src.auth import (
    User,
    create_user,
    authenticate_user,
    get_user_by_id,
    create_access_token,
    verify_access_token,
)
from src.agent_orchestrator import run_agent, AgentResult
from src.learning_rag import answer_learning_question
from src.recommendation_engine import get_learning_profile, get_learning_recommendations
from src.study_planner import create_study_plan
from src.career_rag import answer_career_question, analyze_job_fit
from src.project_rag import list_projects, answer_project_question
from src.ingestion import ingest_document
from src.generator import _sanitize_error

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
    environment: str = "development"
    storage_ready: bool = True


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class UserLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: str
    username: str
    created_at: str
    is_active: bool


# ---------------------------------------------------------------------------
# Authentication Dependency
# ---------------------------------------------------------------------------

SESSION_COOKIE_NAME = "studentai_session"


async def get_current_user(request: Request) -> User:
    """
    Extracts and verifies authentication token from HTTP-only session cookie
    or 'Authorization: Bearer <token>' header.
    Raises HTTPException(401) on missing or invalid authentication.
    """
    token: Optional[str] = request.cookies.get(SESSION_COOKIE_NAME)

    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
        )

    payload = verify_access_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired or is invalid. Please log in again.",
        )

    user = await run_in_threadpool(get_user_by_id, user_id=payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or deactivated.",
        )

    return user


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

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns application status, version, and readiness without leaking secrets."""
    current_settings = get_settings()
    storage_ready = await run_in_threadpool(is_storage_writable, settings=current_settings)

    return HealthResponse(
        status="ok",
        version=current_settings.app_version,
        environment=current_settings.environment,
        storage_ready=storage_ready,
    )


# --- Authentication Endpoints ---

@app.post("/api/auth/register", response_model=UserResponse, tags=["Auth"])
async def register_endpoint(req: UserRegisterRequest, response: Response):
    """Registers a new user and issues a signed session cookie."""
    try:
        user = await run_in_threadpool(create_user, username=req.username, password=req.password)
        token = create_access_token(user_id=user.id, username=user.username)

        # Set secure HTTP-only cookie
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite=settings.cookie_samesite,
            secure=settings.cookie_secure,
            max_age=settings.auth_token_expire_hours * 3600,
            path="/",
        )
        return UserResponse(
            id=user.id,
            username=user.username,
            created_at=user.created_at,
            is_active=user.is_active,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


@app.post("/api/auth/login", response_model=UserResponse, tags=["Auth"])
async def login_endpoint(req: UserLoginRequest, response: Response):
    """Authenticates credentials and establishes session."""
    user = await run_in_threadpool(authenticate_user, username=req.username, password=req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    token = create_access_token(user_id=user.id, username=user.username)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite=settings.cookie_samesite,
        secure=settings.cookie_secure,
        max_age=settings.auth_token_expire_hours * 3600,
        path="/",
    )
    return UserResponse(
        id=user.id,
        username=user.username,
        created_at=user.created_at,
        is_active=user.is_active,
    )


@app.post("/api/auth/logout", tags=["Auth"])
async def logout_endpoint(response: Response):
    """Clears session cookie."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite=settings.cookie_samesite,
        secure=settings.cookie_secure,
    )
    return {"message": "Logged out successfully."}


@app.get("/api/auth/me", response_model=UserResponse, tags=["Auth"])
async def me_endpoint(current_user: User = Depends(get_current_user)):
    """Returns profile of currently authenticated user."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        created_at=current_user.created_at,
        is_active=current_user.is_active,
    )


# --- Protected Domain Workspaces ---

@app.post("/api/agent/run", response_model=AgentRunResponse, tags=["Agent"])
async def run_agent_endpoint(
    req: AgentRunRequest,
    current_user: User = Depends(get_current_user),
):
    """Executes multi-step or single-step agentic orchestration for authenticated user."""
    try:
        context = {"user_id": current_user.id}
        res: AgentResult = await run_in_threadpool(run_agent, user_query=req.query, context=context)
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
async def learning_ask_endpoint(
    req: LearningAskRequest,
    current_user: User = Depends(get_current_user),
):
    """Answers an academic question with user-scoped isolation."""
    try:
        res = await run_in_threadpool(
            answer_learning_question,
            query=req.query,
            top_k=req.top_k,
            intent=req.intent,
            user_id=current_user.id,
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
async def learning_profile_endpoint(
    current_user: User = Depends(get_current_user),
):
    """Fetches authenticated student's learning profile."""
    try:
        prof = await run_in_threadpool(get_learning_profile, user_id=current_user.id)
        return prof
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving learning profile: {str(e)}",
        )


@app.get("/api/learning/recommendations", tags=["Learning"])
async def learning_recommendations_endpoint(
    topic: Optional[str] = Query(None, max_length=100),
    limit: int = Query(5, ge=1, le=50),
    current_user: User = Depends(get_current_user),
):
    """Fetches personalized study recommendations for authenticated user."""
    try:
        recs = await run_in_threadpool(
            get_learning_recommendations,
            user_id=current_user.id,
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
async def study_plan_endpoint(
    req: StudyPlanRequest,
    current_user: User = Depends(get_current_user),
):
    """Generates a structured, multi-day academic study schedule for authenticated user."""
    try:
        plan_res = await run_in_threadpool(
            create_study_plan,
            subject=req.subject,
            days=req.days,
            user_id=current_user.id,
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
async def career_ask_endpoint(
    req: CareerAskRequest,
    current_user: User = Depends(get_current_user),
):
    """Answers a career question grounded strictly in authenticated user's career documents."""
    try:
        res = await run_in_threadpool(
            answer_career_question,
            query=req.query,
            top_k=req.top_k,
            document_type=req.document_type,
            document_id=req.document_id,
            owner_id=current_user.id,
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
async def job_fit_endpoint(
    req: JobFitRequest,
    current_user: User = Depends(get_current_user),
):
    """Conducts deterministic skill matching for documents owned by authenticated user."""
    try:
        res = await run_in_threadpool(
            analyze_job_fit,
            resume_document_id=req.resume_id,
            job_document_id=req.job_id,
            owner_id=current_user.id,
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
async def list_projects_endpoint(
    current_user: User = Depends(get_current_user),
):
    """Lists indexed student software projects belonging to authenticated user."""
    try:
        projs = await run_in_threadpool(list_projects, owner_id=current_user.id)
        return {"projects": projs, "total": len(projs)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing projects: {str(e)}",
        )


@app.post("/api/projects/ask", response_model=ProjectAskResponse, tags=["Projects"])
async def project_ask_endpoint(
    req: ProjectAskRequest,
    current_user: User = Depends(get_current_user),
):
    """Answers a question grounded in authenticated user's project documentation."""
    try:
        res = await run_in_threadpool(
            answer_project_question,
            query=req.query,
            project_id=req.project_id,
            top_k=req.top_k,
            owner_id=current_user.id,
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
    current_user: User = Depends(get_current_user),
):
    """
    Safely ingests an uploaded document into the knowledge base, owned by current_user.
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

    upload_dir = settings.uploads_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = upload_dir / filename

    try:
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        stats = await run_in_threadpool(
            ingest_document,
            file_path=temp_file_path,
            category=category,
            subject=subject,
            document_type=document_type,
            owner_id=current_user.id,
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
        safe_msg = _sanitize_error(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {safe_msg}",
        )
    finally:
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass

