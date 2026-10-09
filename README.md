# StudentAI — Personal Knowledge, Learning & Career RAG

StudentAI is an academic and career AI companion designed for students to ingest study materials (lecture notes, textbooks, syllabi, resumes, and technical documentation) and perform grounded question-answering with verifiable page citations.

This repository is built incrementally from scratch to master the core principles of Retrieval-Augmented Generation (RAG) before introducing high-level orchestration frameworks.

---

## Phase 1 Architecture: Foundational RAG

Phase 1 establishes a modular, transparent RAG foundation without abstractions:

```
[ PDF Document ]
       ¦
       ? (PyMuPDF: extract page-by-page text & metadata)
[ Extracted Pages ]
       ¦
       ? (Chunker: sliding window with overlap & metadata preservation)
[ Text Chunks ]
       ¦
       ? (Embedder: Google GenAI SDK `text-embedding-004`)
[ Dense Vectors (768 dimensions) ]
       ¦
       ? (Vector Store: Qdrant embedded local disk storage)
[ Qdrant Collection ]
       ¦
       ? (Retriever: Cosine similarity search against user query)
[ Top-K Context Passages ]
       ¦
       ? (Generator: Gemini 2.5 Flash augmented with cited passages)
[ Grounded Academic Answer with Citations ]
```

---

## Directory Structure

```text
StudentAI/
+-- .env.example          # Environment template (API keys and configuration)
+-- .gitignore            # Protects credentials, venv, and local database
+-- requirements.txt      # Pinned dependency versions for Python 3.12
+-- README.md             # Project documentation and architectural overview
+-- data/
¦   +-- sample_docs/      # Directory for study PDFs to index
¦   +-- qdrant_storage/   # Local persistent Qdrant vector database (gitignored)
+-- src/
    +-- __init__.py       # Package marker
    +-- config.py         # Loads and validates environment settings (.env)
    +-- extractor.py      # Page-level PDF extraction using PyMuPDF (pymupdf)
    +-- chunker.py        # Sliding-window chunker with character overlap
    +-- embedder.py       # Dense embeddings via official Google GenAI SDK
    +-- vector_store.py   # Embedded Qdrant collection management & point upsert
    +-- retriever.py      # Semantic similarity search using Cosine distance
    +-- generator.py      # RAG prompt synthesis and Gemini generation
    +-- main.py           # CLI entry point (status, ingest, query)
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.12+
- Git

### 2. Virtual Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Secrets
1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` and set your `GEMINI_API_KEY`:
   ```ini
   GEMINI_API_KEY=AIzaSy...your_actual_key...
   ```

### 5. Check Environment Status
```powershell
python -m src.main status
```

---

## Step 15: Learning Analytics Layer

Step 15 introduces the first Learning Analytics layer for StudentAI:
- **Interaction Tracking (`src/learning_tracker.py`)**: Persists learning queries and practice quiz sessions to `data/learning/learning_events.json` with event statuses (`ANSWERED`, `REFUSED`, `QUIZ_GENERATED`, `QUIZ_ATTEMPTED`).
- **Topic & Subtopic Detection (`src/topic_detector.py`)**: Uses `gemini-3.5-flash-lite` (temperature 0.0) with prompt injection defense to extract structured disciplines and subtopics (e.g. `Computer Networks -> TCP / Three-Way Handshake`, `DBMS -> Normalization`).
- **Deterministic Weakness Scoring (`src/learning_analytics.py`)**: Computes weakness metrics without ML or LLM hallucinations:
  - `quiz_failure_rate = questions_incorrect / questions_attempted`
  - `refusal_rate = refused_count / max(interaction_count, 1)`
  - `weakness_score = 0.85 * quiz_failure_rate + 0.15 * refusal_rate` (clamped between `0.0` and `1.0`)
  - **Weakness Levels**:
    - `0.00 - 0.29`: `LOW`
    - `0.30 - 0.59`: `MEDIUM`
    - `0.60 - 1.00`: `HIGH`
    - Unquizzed topics (`questions_attempted == 0`): `INSUFFICIENT_DATA` (score `0.00`)
- **CLI Commands**:
  - `python -m src.main learning-stats [--topic <TOPIC>]`: View aggregated topic or subtopic statistics and weakness rankings.
  - `python -m src.main learning-events [--topic <TOPIC>]`: Inspect recent learning interaction events.
  - `python -m src.main record-quiz --topic "..." --subtopic "..." --attempted N --correct N --incorrect N`: Record quiz outcomes.

---

## Step 16: Personalized Learning Recommendations

Step 16 turns learning analytics into deterministic, actionable study recommendations:
- **Personalized Recommendations (`src/recommendation_engine.py`)**:
  - `get_learning_recommendations(user_id="default_user", max_recommendations=5, topic=None)`
  - `get_learning_profile(user_id="default_user")`
- **Priority Rules**:
  1. `HIGH`: Prioritized immediately with review + focused quiz actions.
  2. `MEDIUM`: Recommended after HIGH topics for review + practice questions.
  3. `INSUFFICIENT_DATA`: Recommends diagnostic quizzes to measure baseline mastery without assuming weakness.
  4. `LOW`: Suppressed when higher-priority topics exist; shown only if alone.
- **Academic Source Matching (`find_academic_sources`)**:
  - Automatically identifies supporting documents using sparse BM25 lexical search strictly filtered to `category="academic"`.
  - Non-academic categories (`career`, `project`, `resume`) are excluded.
  - Topics absent from the academic corpus yield empty source lists without fabricating documents or page numbers.
- **Deterministic & High-Performance**:
  - Pure deterministic Python (no LLM latency or non-determinism).
  - Multi-key sorting: Priority rank $\to$ Weakness score (descending) $\to$ Topic/Subtopic name (alphabetical).
  - Typical local recommendation execution time: **~10â€“20 ms**.
- **CLI Commands**:
  - `python -m src.main recommendations [--limit 5] [--topic <TOPIC>]`: View prioritized study recommendations and actions.
  - `python -m src.main learning-profile`: Full student profile detailing weak topics, strong topics, diagnostic needs, and actionable next steps.

---

## Step 17: Study Planner + Revision Engine

Step 17 turns learning analytics and personalized recommendations into deterministic, actionable multi-day study schedules and prioritized revision queues:
- **Study Planner (`src/study_planner.py`)**:
  - `create_study_plan(subject, days, user_id="default_user", daily_minutes=60)`: Deterministically partitions study time across days.
  - Multi-day strategy: Distributes prioritized weak topics, revisits high-weakness topics with focused quizzes on Day 3, schedules mixed active recall revision on the penultimate day, and comprehensive mock tests on the final day.
  - Strict time allocation: Guarantees the sum of daily activities equals `daily_minutes` and never exceeds it.
  - Validation: Strictly enforces `1 <= days <= 365` and `1 <= daily_minutes <= 1440`.
  - Persistence: Automatically persists plans to `data/learning/study_plans.json` with `plan_id`, `created_at`, `days`, `daily_minutes`, and `plan`.
- **Revision Engine (`src/revision_engine.py`)**:
  - `get_revision_recommendations(user_id="default_user", subject=None, limit=5)`: Computes focused revision queues sorted by weakness priority and accuracy.
- **Activity Types**:
  - `REVIEW`: In-depth concept review.
  - `PRACTICE`: Problem solving and exercise sets.
  - `QUIZ`: Timed targeted assessment.
  - `DIAGNOSTIC`: Baseline measurement for topics with insufficient quiz data.
  - `REVISION`: Active recall and formula review.
  - `MOCK_TEST`: Comprehensive exam simulation.
- **CLI Commands**:
  - `python -m src.main study-plan --subject <SUBJECT> [--days 7] [--minutes 60]`: Generate a structured study schedule.
  - `python -m src.main revision [--subject <SUBJECT>] [--limit 5]`: View prioritized revision recommendations.
  - `python -m src.main study-plans`: Inspect saved study plans.

---

## Step 18: Career RAG and Resume/Job Matching

Step 18 introduces a modular Career RAG subsystem for candidate resumes and job postings:
- **Career Domain Isolation (`src/career_rag.py`)**:
  - `answer_career_question(query, top_k=5, document_type=None, document_id=None)`: Scopes question answering strictly to documents with `category="career"` (e.g. `document_type="resume"` or `"job_description"`).
  - Grounded answering with citations (`[Page X]`) and prompt injection defense (untrusted candidate/job document text).
  - Explicit refusal when queries fall outside the indexed career corpus: `"I couldn't find the answer in the career documents."`
- **Structured Skill Extractor (`src/skill_extractor.py`)**:
  - `extract_skills(text, document_type="other", client=None, model=None, page_number=None)`: Extracts canonical and normalized skill names, skill types (`language`, `framework`, `database`, `cloud`, `tool`, `concept`, etc.), and verbatim textual evidence quotes from document text.
  - Temperature 0.0 with strict JSON schema and sanitized error handling.
- **Deterministic Job Matcher (`src/job_matcher.py`)**:
  - `match_resume_to_job(resume_skills, job_requirements)`: Deterministic matching between candidate resume skills and job requirements.
  - **No LLM in matching**: Skill comparisons and coverage scores are 100% deterministic (no LLM hallucination or arbitrary scoring).
  - **Exact and Canonical Alias Matching**: Handles equivalent technologies (e.g., `js` -> `javascript`, `postgres` -> `postgresql`, `k8s` -> `kubernetes`, `aws` -> `amazon web services`).
  - **Distinction of Disjoint Technologies**: Never conflates distinct technologies (e.g., `Java` and `JavaScript` never match).
  - **Partial Matches**: Identified and reported separately for transparency; strictly **not counted** in the coverage percentage.
  - **Deterministic Coverage Score**: `coverage_percentage = (matched_count / requirement_count) * 100.0` (or `0.0` when requirements count is 0).
  - **Transparent Disclaimer**: Explicitly identifies output as deterministic keyword matching, not an ATS score, hiring probability, or certified qualification metric.
- **CLI Commands**:
  - `python -m src.main career-docs`: List all indexed career documents with their document IDs and types.
  - `python -m src.main career-ask "<query>" [--type resume|job_description] [--document-id <ID>]`: Ask questions about candidate resumes or job postings.
  - `python -m src.main career-skills --document-id <ID>`: Extract explicit skills, categories, and evidence quotes from an indexed document.
  - `python -m src.main match-job --resume-id <RESUME_ID> --job-id <JOB_ID>`: Perform deterministic skill match and gap analysis between candidate and job posting.

---

## Step 19: Project RAG and Project-Aware Interview Preparation

Step 19 introduces a modular Project RAG subsystem designed for student software projects, architecture explanations, technical documentation, and grounded interview preparation:
- **Project Domain Isolation & Listing (`src/project_rag.py`)**:
  - `list_projects()`: Discovers all indexed software projects, associated documents, document IDs, and types.
  - `answer_project_question(query, project_id=None, top_k=5)`: Scopes question answering strictly to documents with `category="project"`. When a project ID is specified, retrieval is isolated to that specific project.
  - **Ambiguity Guard**: If multiple projects exist and no project ID is specified, prompts the user to select one with `--project-id` rather than contaminating context across unrelated codebases.
  - **Prompt Injection Defense**: Treats all documentation text as untrusted data (`--- UNTRUSTED PROJECT DOCUMENTATION ---`) with instructions preventing execution of embedded directives.
  - **Grounded Refusal**: Returns `"I couldn't find the answer in the provided project documentation."` when requested information or unsupported features cannot be verified in the retrieved documents.
- **Project Technical Summary & Comparison (`src/project_rag.py`)**:
  - `summarize_project(project_id)`: Generates a structured JSON technical summary covering documented purpose, architecture, technologies, features, and limitations. Missing fields are explicitly marked `"Unknown"` rather than hallucinated.
  - `compare_projects(project_id_a, project_id_b)`: Compares two documented projects across purpose, architecture, stack, storage, and limitations.
- **Project-Aware Interview Preparation (`src/project_interview.py`)**:
  - `generate_project_interview_questions(project_id, count=5, difficulty="MIXED")`: Generates technically rigorous interview questions based strictly on the project documentation.
  - **Supported Difficulties**: `EASY`, `MEDIUM`, `HARD`, and `MIXED`.
  - **Supported Categories**: `architecture`, `design_decisions`, `data_flow`, `technology_selection`, `error_handling`, `security`, `scalability`, `testing`, `tradeoffs`, `limitations`.
  - **Deduplication & Grounding**: Deduplicates questions, provides expected answer outlines, evidence summaries, and preserves filename and page references for every generated question.
- **Multi-Format Ingestion (`src/extractor.py`, `src/ingestion.py`)**:
  - Unified `extract_document()` supporting `.pdf`, `.md`, `.txt`, and `.markdown`.
  - Automatically indexes project documentation with `category="project"` and links them to designated project identifiers via `--subject <project_id>`.
- **CLI Commands**:
  - `python -m src.main projects`: Lists all indexed student projects and their documented files.
  - `python -m src.main project-ask "<query>" [--project-id <ID>] [--top-k 5]`: Ask questions about software architecture, tech stack, or implementation.
  - `python -m src.main project-summary --project-id <ID>`: Generate a structured technical summary of a project.
  - `python -m src.main project-interview --project-id <ID> [--count 5] [--difficulty MIXED]`: Generate grounded technical interview questions.
  - `python -m src.main project-compare --project-a <ID> --project-b <ID>`: Compare architectures and tech stacks of two documented projects.

---

## Step 20: Advanced RAG Quality and Reliability

Step 20 enhances retrieval quality, context selection, citation reliability, and benchmark evaluation across all domains:
- **Conservative Query Rewriting (src/query_rewriter.py)**:
  - ewrite_query(query, client=None, model=None): Conservatively expands technical acronyms, abbreviations, and informal jargon (e.g., NIC -> Network Interface Card MAC address, ACID -> ACID properties atomicity consistency isolation durability database transactions).
  - **Intent Preservation & Guardrails**: Treats user queries as untrusted data to defend against prompt injection. Preserves user intent strictly without hallucinating technologies.
  - **Fallback Safety**: Gracefully falls back to the original raw query upon JSON decode errors, API errors, or low confidence.
  - **Strictly Opt-in**: Kept optional via --rewrite to prevent extra LLM latency and API costs on standard queries.
- **Context Selection & Budget Enforcement (src/context_builder.py)**:
  - Context deduplication by chunk ID and normalized text content.
  - Character budget enforcement (max_chars parameter, default 12,000 chars) ensuring context stays within token limits while preserving multi-document evidence.
  - Preserves document header metadata (Document: X | Page: Y | Chunk: Z) and extracts page citations (extract_page_citations()).
- **Citation Verification (src/citation_verifier.py)**:
  - erify_citations(answer, sources, expected_documents=None): Analyzes LLM response citations against retrieved context passages.
  - Validates [Page X] citations against retrieved page numbers, reporting valid and invalid page citations.
  - Detects malformed citation syntaxes (e.g., [page], [Page abc]).
  - Distinguishes syntax validity from factual refusal (I couldn't find the answer in the provided documents.).
- **Unified Multi-Mode Benchmark Evaluation (src/evaluator.py, src/generation_evaluator.py)**:
  - Expanded benchmark dataset (data/evaluation/rag_eval.json) with cross-domain questions (academic, career, project, out-of-domain refusals).
  - Benchmarks 5 distinct retrieval strategies on identical evaluation queries:
    1. **Dense Retrieval** (Gemini embeddings + Qdrant cosine similarity)
    2. **Sparse Retrieval** (BM25 term frequency)
    3. **Hybrid Retrieval** (Dense + Sparse Reciprocal Rank Fusion)
    4. **Hybrid + Reranker** (Dense + Sparse RRF + Gemini two-stage reranking)
    5. **Hybrid + Rewriter** (Gemini query rewriting + Dense + Sparse RRF)
  - End-to-end generation evaluation checking citation accuracy, expected document recall, and refusal correctness.
- **CLI Commands**:
  - python -m src.main ask <question> [--rewrite] [--hybrid] [--rerank]: Grounded QA with citation verification and optional query rewriting.
  - python -m src.main evaluate [--file data/evaluation/rag_eval.json] [--top-k 5] [--skip-generation]: Run full multi-mode retrieval and generation benchmark suite.

---

## Step 21: Agentic Orchestration Layer

Step 21 introduces a controlled, deterministic orchestration layer that understands user goals, selects minimal or multi-step capabilities, executes bounded tools, and synthesizes grounded answers:
- **Strict Tool Registry (src/agent_tools.py)**:
  - Exposes 11 registered StudentAI capabilities:
    - learning_ask: Query academic concepts with notes and citations.
    - learning_profile: Fetch student learning profile showing weak, strong, and unassessed topics.
    - learning_recommendations: Fetch actionable next topics to study with grounded academic sources.
    - evision_topics: View prioritized revision items based on weakness and quiz performance.
    - create_study_plan: Generate deterministic multi-day study schedule.
    - career_ask: Query resumes and job descriptions with grounded citations.
    - nalyze_job_fit: Deterministic skill match and gap analysis between resume and job description.
    - project_list: Discover all indexed student software projects and documents.
    - project_ask: Answer questions grounded in software project documentation.
    - project_summary: Generate structured technical summaries of software projects.
    - project_interview: Generate grounded technical interview questions from project documentation.
  - Strict parameter validation: Enforces type constraints, string bounds, and allowed choices.
  - Rejects unknown tools, arbitrary code execution, dynamic imports, eval, or shell commands.
- **Agent Orchestrator (src/agent_orchestrator.py)**:
  - plan_task(user_query, context=None): Creates structured execution plans with classification (LEARNING, CAREER, PROJECT, CROSS_DOMAIN, GENERAL) and workflow selection.
  - execute_plan(plan, context=None): Executes tool calls in dependency order with strict execution bounds ($\le 5$ tool calls) and error containment.
  - un_agent(user_query, context=None): End-to-end execution returning grounded answers, execution traces, sources, citation verification, and explicit limitations.
- **Multi-Step Workflows Supported**:
  - **Workflow A (Personalized Study Planning)**: Assesses learning profile, checks revision topics, and constructs structured multi-day study plans.
  - **Workflow B (Career & Job Analysis)**: Inspects resume evidence, executes deterministic job fit match, and outlines missing skills.
  - **Workflow C (Project Interview Preparation)**: Verifies project docs, summarizes technical architecture, and generates grounded interview questions.
  - **Workflow D (Cross-Domain Assistance)**: Seamlessly combines concept explanations, related projects, and career preparation.
- **Safety & Observability Controls**:
  - Maximum 5 tool calls per user query; no unbounded loops.
  - Prompt injection defense: Treats user input and tool outputs as untrusted data (--- UNTRUSTED DATA ---).
  - Sensitive argument redaction ([REDACTED]).
  - Integrated citation verification via erify_citations().
- **CLI Command**:
  - python -m src.main agent --query <goal>: Execute multi-step agentic orchestration from the command line.

---

## Step 22: FastAPI Backend Service

Step 22 exposes StudentAI capabilities as clean, reliable, and testable REST APIs for frontend integration (e.g. Next.js):
- **Service Endpoints (src/api.py)**:
  - GET /health: Health status and application version without exposing secrets or paths.
  - POST /api/agent/run: Multi-step and single-step agentic orchestration (un_agent()).
  - POST /api/learning/ask: Academic question answering with intent routing and page citations.
  - GET /api/learning/profile: Student learning analytics profile (weak/strong topics).
  - GET /api/learning/recommendations: Actionable personalized study recommendations.
  - POST /api/study-plans: Structured multi-day academic schedule generator (create_study_plan()).
  - POST /api/career/ask: Career and resume question answering with grounded citations.
  - POST /api/career/job-fit: Deterministic skill extraction and job requirement gap matching.
  - GET /api/projects: List of all indexed software projects.
  - POST /api/projects/ask: Technical QA grounded in software project documentation.
  - POST /api/documents/upload: Safe ingestion for .pdf, .md, and .txt files (bounds size to 10MB, verifies extensions, prevents path traversal).
- **Architecture & Reliability**:
  - Pure synchronous AI modules are offloaded to a threadpool via un_in_threadpool, preventing event loop blocking.
  - Structured request and response validation using Pydantic schemas.
  - Configurable CORS middleware via CORS_ORIGINS environment variable (defaults to http://localhost:3000,http://127.0.0.1:3000).
  - Never leaks API keys, passwords, or raw filesystem paths to API clients.
- **Starting the API Server**:
  `ash
  # Local development server with auto-reload
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
  `
- **Interactive Documentation**:
  - OpenAPI UI: http://localhost:8000/docs
  - ReDoc: http://localhost:8000/redoc
- **Authentication & Multi-User Disclaimer**:
  - This API is designed for local single-user prototyping. In multi-user production deployments, an authentication and authorization layer (e.g. JWT / OAuth2) must be integrated before exposing private documents or shared persistent records.
