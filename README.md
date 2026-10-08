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
