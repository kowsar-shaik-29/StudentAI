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
