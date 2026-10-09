"""
CLI Entry Point for StudentAI.
Provides commands to inspect configuration, extract PDF text, chunk documents, index vectors,
ingest directories, search, and ask questions.

Usage:
  python -m src.main status
  python -m src.main extract <path-to-pdf>
  python -m src.main chunk <path-to-pdf>
  python -m src.main ingest <path-to-pdf>
  python -m src.main ingest-dir <directory-path>
  python -m src.main search "<question>" [--top-k 5]
  python -m src.main ask "<question>" [--top-k 5]
"""

import sys
import argparse
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from collections import Counter

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import get_settings, require_api_key
from src.extractor import extract_pdf
from src.chunker import create_chunks
from src.vector_store import (
    initialize_vector_store,
    count_vectors,
    get_collection_info,
)
from src.retriever import search
from src.sparse_retriever import get_sparse_retriever
from src.hybrid_retriever import hybrid_search
from src.rag import answer_question, answer_question_auto
from src.learning_rag import answer_learning_question
from src.ingestion import ingest_document, ingest_directory
from src.evaluator import evaluate_retrieval_mode
from src.generation_evaluator import evaluate_generation_pipeline
from src.learning_tracker import (
    record_quiz_attempt,
    get_learning_events,
)
from src.learning_analytics import (
    get_topic_statistics,
    get_subtopic_statistics,
)
from src.recommendation_engine import (
    get_learning_recommendations,
    get_learning_profile,
)
from src.study_planner import (
    create_study_plan,
    get_study_plans,
    get_latest_study_plan,
)
from src.revision_engine import (
    get_revision_recommendations,
)
from src.career_rag import (
    answer_career_question,
    get_career_document_text,
    analyze_job_fit,
)
from src.skill_extractor import extract_skills
from src.job_matcher import match_resume_to_job
from src.project_rag import (
    answer_project_question,
    list_projects,
    summarize_project,
    compare_projects,
)
from src.project_interview import generate_project_interview_questions



def cmd_status() -> None:
    """Checks configuration and local Qdrant database status."""
    settings = get_settings()
    print("=" * 60)
    print("StudentAI - System & Vector Store Status")
    print("=" * 60)
    print(f"Embedding Model:    {settings.embedding_model}")
    print(f"Embedding Dim:      {settings.embedding_dim}")
    print(f"Generation Model:   {settings.generation_model}")
    print(f"Qdrant Storage:     {settings.qdrant_storage_path}")
    print(f"Collection Name:    {settings.qdrant_collection_name}")
    print(f"Chunk Size/Overlap: {settings.chunk_size} / {settings.chunk_overlap} chars")

    # API Key status
    if settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here":
        masked_key = settings.gemini_api_key[:4] + "..." + settings.gemini_api_key[-4:]
        print(f"Gemini API Key:     Configured ({masked_key})")
    else:
        print("Gemini API Key:     [NOT CONFIGURED] (Set GEMINI_API_KEY in .env)")

    # Qdrant collection status
    try:
        q_client = initialize_vector_store()
        info = get_collection_info(q_client)
        if info["exists"]:
            print(f"Collection Status:  Active (Status: {info.get('status', 'Ready')})")
            print(f"Stored Vectors:     {info['points_count']}")
            print(f"Vector Dimension:   {info['vector_size']} (Distance: {info['distance']})")
        else:
            print("Collection Status:  Not yet created (0 vectors)")
        q_client.close()
    except Exception as e:
        print(f"Qdrant Status:      Error accessing store: {e}")
    print("=" * 60)


def cmd_extract(pdf_path: str, preview_chars: int = 250, max_pages_preview: int = 5) -> None:
    """Extracts text page-by-page from a PDF and displays a concise preview."""
    path = Path(pdf_path)
    print("=" * 60)
    print(f"Extracting PDF: {path.name}")
    print("=" * 60)

    try:
        pages = extract_pdf(path, include_empty=True)
    except Exception as err:
        print(f"Extraction Error: {err}")
        sys.exit(1)

    total_pages = len(pages)
    pages_with_text = sum(1 for p in pages if p["has_text"])
    empty_pages = total_pages - pages_with_text

    print(f"File:        {path.name}")
    print(f"Total pages: {total_pages}")
    print(f"With text:   {pages_with_text}")
    if empty_pages > 0:
        print(f"Blank/Image: {empty_pages} page(s)")
    print("-" * 60)

    preview_count = min(total_pages, max_pages_preview)
    for p in pages[:preview_count]:
        page_num = p["page_number"]
        char_count = p["char_count"]
        print(f"\nPage {page_num} ({char_count} chars):")
        if p["has_text"]:
            text_snippet = p["text"][:preview_chars].replace("\r", "")
            if len(p["text"]) > preview_chars:
                text_snippet += "..."
            print(text_snippet)
        else:
            print("[Empty page / No extractable text]")

    if total_pages > preview_count:
        print(f"\n... and {total_pages - preview_count} more pages.")
    print("=" * 60)
    print("Extraction completed successfully.")


def cmd_chunk(pdf_path: str, chunk_size: int = 1000, overlap: int = 200, preview_count: int = 5) -> None:
    """Chunks extracted PDF pages and displays concise metrics and previews."""
    path = Path(pdf_path)
    print("=" * 60)
    print(f"Chunking Document: {path.name}")
    print("=" * 60)

    try:
        pages = extract_pdf(path, include_empty=True)
        chunks = create_chunks(pages, chunk_size=chunk_size, overlap=overlap)
    except Exception as err:
        print(f"Chunking Error: {err}")
        sys.exit(1)

    total_pages = len(pages)
    total_chunks = len(chunks)

    if total_chunks == 0:
        print("No chunks generated (document may be empty).")
        return

    char_counts = [c["char_count"] for c in chunks]
    avg_chunk_size = sum(char_counts) / total_chunks
    min_chunk_size = min(char_counts)
    max_chunk_size = max(char_counts)
    chunks_per_page = Counter(c["page_number"] for c in chunks)

    print(f"File:               {path.name}")
    print(f"Total Pages:        {total_pages}")
    print(f"Total Chunks:       {total_chunks}")
    print(f"Chunk Size Config:  {chunk_size} chars (Overlap: {overlap} chars)")
    print(f"Average Chunk Size: {avg_chunk_size:.1f} characters")
    print(f"Min Chunk Size:     {min_chunk_size} characters")
    print(f"Max Chunk Size:     {max_chunk_size} characters")
    print("-" * 60)
    print("Chunks per Page:")
    for page_num in sorted(chunks_per_page.keys()):
        print(f"  Page {page_num:2d}: {chunks_per_page[page_num]} chunk(s)")
    print("-" * 60)

    show_count = min(total_chunks, preview_count)
    print(f"Previewing first {show_count} chunk(s):\n")
    for i in range(show_count):
        c = chunks[i]
        preview_text = c["text"][:200].replace("\r", "").replace("\n", " ")
        if len(c["text"]) > 200:
            preview_text += "..."
        print(f"Chunk {i + 1}:")
        print(f"  ID:           {c['chunk_id']}")
        print(f"  Page:         {c['page_number']}")
        print(f"  Characters:   {c['char_count']}")
        print(f"  Text Preview: {preview_text}\n")

    if total_chunks > show_count:
        print(f"... and {total_chunks - show_count} more chunk(s).")
    print("=" * 60)
    print("Chunking completed successfully.")


def cmd_ingest_single(
    pdf_path: str,
    category: str = None,
    subject: str = None,
    document_type: str = None,
) -> None:
    """Ingests a single PDF document into Qdrant with optional metadata."""
    require_api_key()
    try:
        stats = ingest_document(
            pdf_path,
            category=category,
            subject=subject,
            document_type=document_type,
        )
    except Exception as err:
        print(f"Ingestion Error: {err}")
        sys.exit(1)

    print("=" * 60)
    print("DOCUMENT INGESTION")
    print("=" * 60)
    print(f"\nDocument:\n{stats['file_name']}\n")
    print(f"Document ID:\n{stats.get('document_id', 'N/A')}\n")
    print(f"Category:\n{stats.get('category', 'other')}\n")
    print(f"Subject:\n{stats.get('subject', 'general')}\n")
    print(f"Type:\n{stats.get('document_type', 'other')}\n")
    print(f"Pages:\n{stats['pages']} ({stats['pages_with_text']} with text)\n")
    print(f"Chunks:\n{stats['chunks']}\n")
    print(f"Embeddings:\n{stats['embeddings']}\n")
    print(f"Vectors stored:\n{stats['vectors_stored']}\n")
    print("=" * 60)



def cmd_ingest_dir(directory_path: str) -> None:
    """Ingests all PDF files from a directory into Qdrant."""
    require_api_key()
    try:
        summary = ingest_directory(directory_path)
    except Exception as err:
        print(f"Directory Ingestion Error: {err}")
        sys.exit(1)

    print("=" * 60)
    print("DIRECTORY INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total PDFs Found:       {summary['total_files_found']}")
    print(f"Successfully Ingested:  {summary['successful_files']}")
    print(f"Failed Files:           {len(summary['failed_files'])}")
    print(f"Total Pages Processed:  {summary['total_pages']}")
    print(f"Total Chunks Created:   {summary['total_chunks']}")
    print(f"Total Vectors Stored:   {summary['total_vectors_stored']}")
    print("-" * 60)

    if summary["details"]:
        print("Indexed Documents:")
        for doc in summary["details"]:
            print(f"  - {doc['file_name']}: {doc['pages']} pages, {doc['chunks']} chunks, {doc['vectors_stored']} vectors")

    if summary["failed_files"]:
        print("\nFailed Documents:")
        for failed in summary["failed_files"]:
            print(f"  - {failed['file_name']}: {failed['error']}")

    print("=" * 60)


def cmd_search(
    query: str,
    top_k: int = 5,
    preview_chars: int = 300,
    rerank: bool = False,
    initial_candidates: int = 10,
    hybrid: bool = False,
    sparse_only: bool = False,
    category: str = None,
    subject: str = None,
    document_type: str = None,
) -> None:
    """Executes search (dense, sparse, or hybrid) with optional reranking and metadata filters."""
    clean_query = query.strip()

    print("=" * 60)
    if hybrid:
        print("HYBRID SEARCH (Dense + Sparse BM25 RRF)")
    elif sparse_only:
        print("SPARSE SEARCH (BM25 Lexical)")
    else:
        print("SEMANTIC SEARCH (Dense Vector)")
    print("=" * 60)
    print(f"\nQuery:\n{clean_query}\n")
    if category or subject or document_type:
        print(f"Filter -> Category: {category or 'ANY'} | Subject: {subject or 'ANY'} | Type: {document_type or 'ANY'}\n")

    if hybrid:
        require_api_key()
        q_client = initialize_vector_store()
        try:
            results = hybrid_search(
                query=clean_query,
                top_k=top_k,
                client=q_client,
                dense_candidates=initial_candidates,
                sparse_candidates=initial_candidates,
                category=category,
                subject=subject,
                document_type=document_type,
            )
        finally:
            q_client.close()
    elif sparse_only:
        sr = get_sparse_retriever()
        results = sr.search(
            query=clean_query,
            top_k=top_k,
            category=category,
            subject=subject,
            document_type=document_type,
        )
    else:
        require_api_key()
        if rerank:
            print(f"Initial candidates:\n{initial_candidates}\n")
            print(f"Final results:\n{top_k}\n")

        q_client = initialize_vector_store()
        try:
            results = search(
                query=clean_query,
                top_k=top_k,
                client=q_client,
                rerank=rerank,
                initial_candidates=initial_candidates,
                category=category,
                subject=subject,
                document_type=document_type,
            )
        finally:
            q_client.close()

    if not rerank:
        print(f"Results: {len(results)}\n")

    for i, res in enumerate(results, start=1):
        doc = res["file_name"]
        page = res["page_number"]
        chunk_id = res["chunk_id"]
        res_cat = res.get("category", "other")
        res_subj = res.get("subject", "general")
        res_type = res.get("document_type", "other")
        raw_text = res["text"].strip().replace("\r", "")

        preview_text = raw_text[:preview_chars]
        if len(raw_text) > preview_chars:
            preview_text += "..."

        print("-" * 60)
        print(f"Result {i}")
        if hybrid:
            print(f"Fused score (RRF): {res.get('fused_score', 0.0):.6f}")
            print(f"Dense score:       {res.get('dense_score', 0.0):.4f}")
            print(f"Sparse score:      {res.get('sparse_score', 0.0):.4f}")
            print(f"Document:          {doc} [{res_cat}/{res_subj}/{res_type}]")
            print(f"Page:              {page}")
            print(f"Chunk:             {chunk_id}")
            print("Text:")
            print(preview_text)
        elif sparse_only:
            print(f"Sparse score (BM25): {res.get('sparse_score', 0.0):.4f}")
            print(f"Document:            {doc} [{res_cat}/{res_subj}/{res_type}]")
            print(f"Page:                {page}")
            print(f"Chunk:               {chunk_id}")
            print("Text:")
            print(preview_text)
        elif rerank:
            print(f"Qdrant score:\n{res.get('score', 0.0):.4f}\n")
            rerank_score = res.get("rerank_score", res.get("score", 0.0))
            print(f"Reranker score:\n{rerank_score:.4f}\n")
            print(f"Document:\n{doc} [{res_cat}/{res_subj}/{res_type}]\n")
            print(f"Page:\n{page}\n")
            print(f"Chunk:\n{chunk_id}\n")
            print("Text:")
            print(preview_text)
        else:
            print(f"Score:    {res.get('score', 0.0):.4f}")
            print(f"Document: {doc} [{res_cat}/{res_subj}/{res_type}]")
            print(f"Page:     {page}")
            print(f"Chunk ID: {chunk_id}")
            print("Text:")
            print(preview_text)
        print()

    print("=" * 60)


def cmd_ask(
    question: str,
    top_k: int = 5,
    rerank: bool = False,
    initial_candidates: int = 10,
    hybrid: bool = False,
    category: str = None,
    subject: str = None,
    document_type: str = None,
    rewrite: bool = False,
) -> None:
    """Executes end-to-end RAG question answering with optional reranking or hybrid retrieval, query rewriting, and metadata filters."""
    require_api_key()

    print("=" * 60)
    print("STUDENTAI RAG")
    print("=" * 60)
    print(f"\nQuestion:\n{question.strip()}\n")
    if category or subject or document_type:
        print(f"Filter -> Category: {category or 'ANY'} | Subject: {subject or 'ANY'} | Type: {document_type or 'ANY'}\n")
    if rewrite:
        print("Query Rewriting: Enabled\n")
    if hybrid:
        print("Retrieval Mode: Hybrid (Dense + Sparse BM25 RRF)\n")
    elif rerank:
        print(f"Reranking: Enabled (Candidates: {initial_candidates} -> Top: {top_k})\n")

    result = answer_question(
        question=question.strip(),
        top_k=top_k,
        rerank=rerank,
        initial_candidates=initial_candidates,
        hybrid=hybrid,
        category=category,
        subject=subject,
        document_type=document_type,
        rewrite=rewrite,
    )

    if result.get("query_rewrite") and result["query_rewrite"].get("is_rewritten"):
        print(f"Rewritten Query:\n{result['query_rewrite']['rewritten_query']}\n")
        print(f"Rewrite Reason:\n{result['query_rewrite']['reason']}\n")

    print("-" * 60)
    print("ANSWER")
    print("-" * 60)
    print(result["answer"])
    print()

    # Citation Verification Summary
    citations_info = result.get("citations")
    if citations_info:
        print("-" * 60)
        print("CITATION VERIFICATION")
        print("-" * 60)
        valid_status = "VALID" if citations_info["is_valid"] else "INVALID"
        print(f"Status:             {valid_status}")
        if citations_info["cited_pages"]:
            print(f"Cited Pages:        {citations_info['cited_pages']}")
            print(f"Valid Pages:        {citations_info['valid_pages']}")
        else:
            print("Cited Pages:        None")
        if citations_info["invalid_pages"]:
            print(f"Invalid Pages:      {citations_info['invalid_pages']}")
        if citations_info["malformed_citations"]:
            print(f"Malformed Citations:{citations_info['malformed_citations']}")
        if citations_info["errors"]:
            print(f"Errors:             {citations_info['errors']}")
        print()

    print("-" * 60)
    print("SOURCES")
    print("-" * 60)
    for i, src in enumerate(result["sources"], start=1):
        print(f"{i}. {src['file_name']}")
        print(f"   Page:  {src['page_number']}")
        print(f"   Score: {src['score']:.4f}")
        if "rerank_score" in src:
            print(f"   Rerank Score: {src['rerank_score']:.4f}")
        print(f"   Chunk: {src['chunk_id']}")
        print()

    print("=" * 60)


def cmd_ask_auto(
    question: str,
    top_k: int = 5,
    threshold: float = 0.70,
) -> None:
    """Executes query routing followed by grounded RAG question answering."""
    require_api_key()

    clean_question = question.strip()
    result = answer_question_auto(
        query=clean_question,
        top_k=top_k,
        confidence_threshold=threshold,
    )

    print("=" * 60)
    print("STUDENTAI AUTO ROUTER")
    print("=" * 60)
    print(f"\nQuestion:\n{clean_question}\n")
    print(f"Route:\n{result['route']}\n")
    print(f"Confidence:\n{result['confidence']:.2f}\n")
    if result.get("reason"):
        print(f"Reason:\n{result['reason']}\n")
    if result.get("category_filter"):
        print(f"Applied Metadata Filter: category='{result['category_filter']}'\n")
    else:
        print("Applied Metadata Filter: None (Unfiltered)\n")

    print("-" * 60)
    print("ANSWER")
    print("-" * 60)
    print(result["answer"])
    print()

    print("-" * 60)
    print("SOURCES")
    print("-" * 60)
    for i, src in enumerate(result["sources"], start=1):
        print(f"{i}. {src['file_name']}")
        print(f"   Page:     {src['page_number']}")
        print(f"   Category: {src.get('category', 'other')}")
        print(f"   Score:    {src['score']:.4f}")
        print(f"   Chunk:    {src['chunk_id']}")
        print()

    print("=" * 60)


def cmd_learn(
    question: str,
    top_k: int = 5,
    intent: Optional[str] = None,
    threshold: float = 0.70,
) -> None:
    """Executes Learning RAG with intent routing and academic domain isolation."""
    require_api_key()

    clean_question = question.strip()
    result = answer_learning_question(
        query=clean_question,
        top_k=top_k,
        intent=intent,
        confidence_threshold=threshold,
    )

    print("=" * 60)
    print("STUDENTAI LEARNING")
    print("=" * 60)
    print(f"\nQuestion:\n{clean_question}\n")
    print(f"Topic:       {result.get('topic', 'Computer Science')}")
    print(f"Subtopic:    {result.get('subtopic', 'General')}")
    print(f"Intent:      {result['intent']}")
    print(f"Confidence:  {result['confidence']:.2f}\n")
    if result.get("reason"):
        print(f"Reason:\n{result['reason']}\n")

    if result["intent"] == "QUIZ":
        print("-" * 60)
        print("PRACTICE QUIZ QUESTIONS")
        print("-" * 60)
        questions = result.get("questions", [])
        if not questions:
            print("No practice questions could be generated from the available academic notes.")
        else:
            for idx, q_item in enumerate(questions, start=1):
                diff = q_item.get("difficulty", "medium").upper()
                print(f"Q{idx} [{diff}]: {q_item.get('question')}")
        print()
    else:
        print("-" * 60)
        print("ANSWER")
        print("-" * 60)
        print(result["answer"])
        print()

    print("-" * 60)
    print("SOURCES")
    print("-" * 60)
    for i, src in enumerate(result["sources"], start=1):
        print(f"{i}. {src['file_name']}")
        print(f"   Page:     {src['page_number']}")
        print(f"   Category: {src.get('category', 'academic')}")
        print(f"   Subject:  {src.get('subject', 'general')}")
        print(f"   Score:    {src['score']:.4f}")
        print(f"   Chunk:    {src['chunk_id']}")
        print()

    print("=" * 60)


def cmd_learning_stats(topic: Optional[str] = None) -> None:
    """Displays aggregated learning statistics and weak topics."""
    print("=" * 70)
    print("STUDENTAI LEARNING ANALYTICS")
    print("=" * 70)

    if topic:
        sub_stats = get_subtopic_statistics(topic=topic)
        print(f"\nSubtopic Statistics for: {topic}")
        print("-" * 70)
        if not sub_stats:
            print("No learning events found for this topic.")
        else:
            header = f"{'Subtopic':<25}{'Interactions':<14}{'Quizzed':<10}{'Accuracy':<10}{'Weakness':<10}{'Level':<10}"
            print(header)
            print("-" * 70)
            for s in sub_stats:
                acc_str = f"{s['quiz_accuracy'] * 100:.0f}%" if s["quiz_accuracy"] is not None else "N/A"
                print(
                    f"{s['subtopic'][:23]:<25}"
                    f"{s['interaction_count']:<14}"
                    f"{s['questions_attempted']:<10}"
                    f"{acc_str:<10}"
                    f"{s['weakness_score']:<10.2f}"
                    f"{s['weakness_level']:<10}"
                )
    else:
        topic_stats = get_topic_statistics()
        print("\nTopic Overview & Weak-Topic Analysis")
        print("-" * 70)
        if not topic_stats:
            print("No learning events recorded yet.")
        else:
            header = f"{'Topic':<25}{'Interactions':<14}{'Quizzed':<10}{'Accuracy':<10}{'Weakness':<10}{'Level':<10}"
            print(header)
            print("-" * 70)
            for t in topic_stats:
                acc_str = f"{t['quiz_accuracy'] * 100:.0f}%" if t["quiz_accuracy"] is not None else "N/A"
                print(
                    f"{t['topic'][:23]:<25}"
                    f"{t['interaction_count']:<14}"
                    f"{t['questions_attempted']:<10}"
                    f"{acc_str:<10}"
                    f"{t['weakness_score']:<10.2f}"
                    f"{t['weakness_level']:<10}"
                )

    print("=" * 70)


def cmd_learning_events(topic: Optional[str] = None) -> None:
    """Lists recent learning interaction events."""
    events = get_learning_events(topic=topic)
    print("=" * 70)
    print("STUDENTAI LEARNING EVENTS")
    print("=" * 70)
    if not events:
        print("No learning events recorded.")
    else:
        for idx, ev in enumerate(events[-20:], start=1):
            ts = ev.timestamp[:19].replace("T", " ")
            print(f"{idx}. [{ts}] [{ev.status}] [{ev.intent}] {ev.topic} -> {ev.subtopic}")
            print(f"   Query: {ev.query}")
            if ev.questions_attempted > 0:
                print(f"   Score: {ev.questions_correct}/{ev.questions_attempted} correct")
            print()
    print("=" * 70)


def cmd_record_quiz(
    topic: str,
    subtopic: str,
    attempted: int,
    correct: int,
    incorrect: int,
) -> None:
    """Records a practice quiz attempt."""
    try:
        ev = record_quiz_attempt(
            topic=topic,
            subtopic=subtopic,
            questions_attempted=attempted,
            questions_correct=correct,
            questions_incorrect=incorrect,
        )
        print("=" * 60)
        print("QUIZ ATTEMPT RECORDED")
        print("=" * 60)
        print(f"Topic:       {ev.topic}")
        print(f"Subtopic:    {ev.subtopic}")
        print(f"Attempted:   {ev.questions_attempted}")
        print(f"Correct:     {ev.questions_correct}")
        print(f"Incorrect:   {ev.questions_incorrect}")
        acc = (ev.questions_correct / ev.questions_attempted * 100) if ev.questions_attempted > 0 else 0
        print(f"Accuracy:    {acc:.1f}%")
        print("=" * 60)
    except Exception as err:
        print(f"Error recording quiz attempt: {err}")
        sys.exit(1)


def cmd_recommendations(limit: int = 5, topic: Optional[str] = None) -> None:
    """Displays personalized learning recommendations."""
    recs_res = get_learning_recommendations(max_recommendations=limit, topic=topic)
    print("=" * 70)
    print("STUDENTAI PERSONALIZED RECOMMENDATIONS")
    print("=" * 70)

    if recs_res.get("message") and not recs_res.get("recommendations"):
        print(f"\n{recs_res['message']}\n")
        print("=" * 70)
        return

    recs = recs_res.get("recommendations", [])
    if not recs:
        print("\nNo recommendations available.\n")
        print("=" * 70)
        return

    print()
    for r in recs:
        print(f"{r['rank']}. {r['topic']} / {r['subtopic']}")
        print(f"   Priority: {r['priority']}")
        print(f"   Reason:   {r['reason']}")
        print(f"   Action:   {r['recommended_action']}")
        if r.get("source_documents"):
            print(f"   Sources:  {', '.join(r['source_documents'])}")
        else:
            print("   Sources:  None (No supporting academic document found)")
        print()
    print("=" * 70)


def cmd_learning_profile() -> None:
    """Displays structured personalized learning profile."""
    profile = get_learning_profile()
    print("=" * 70)
    print("STUDENTAI PERSONALIZED LEARNING PROFILE")
    print("=" * 70)

    # 1. Weak Topics
    weak_list = profile.get("weak_topics", [])
    if weak_list:
        print("\nWeak Topics")
        print("-" * 70)
        for idx, w in enumerate(weak_list, start=1):
            acc_str = f"{w['accuracy'] * 100:.0f}%" if w["accuracy"] is not None else "N/A"
            print(f"{idx}. {w['topic']} / {w['subtopic']}")
            print(f"   Accuracy: {acc_str}")
            print(f"   Weakness: {w['weakness_level']} (Score: {w['weakness_score']:.2f})")
            print()

    # 2. Strong Topics
    strong_list = profile.get("strong_topics", [])
    if strong_list:
        print("\nStrong Topics")
        print("-" * 70)
        for idx, s in enumerate(strong_list, start=1):
            acc_str = f"{s['accuracy'] * 100:.0f}%" if s["accuracy"] is not None else "N/A"
            print(f"{idx}. {s['topic']} / {s['subtopic']}")
            print(f"   Accuracy: {acc_str}")
            print(f"   Weakness: {s['weakness_level']} (Score: {s['weakness_score']:.2f})")
            print()

    # 3. Insufficient Data Topics
    insufficient_list = profile.get("insufficient_data_topics", [])
    if insufficient_list:
        print("\nInsufficient Data (Needs Assessment)")
        print("-" * 70)
        for idx, item in enumerate(insufficient_list, start=1):
            print(f"{idx}. {item['topic']} / {item['subtopic']}")
            print("   Status:   No quiz history (Interactions recorded)")
            print()

    # 4. Recommendations
    recs = profile.get("recommendations", [])
    print("\nActionable Recommendations")
    print("-" * 70)
    if not recs:
        if profile.get("message"):
            print(f"{profile['message']}")
        else:
            print("No recommendations available.")
    else:
        for r in recs:
            print(f"{r['rank']}. {r['topic']} / {r['subtopic']}")
            print(f"   Priority: {r['priority']}")
            print(f"   Action:   {r['recommended_action']}")
            if r.get("source_documents"):
                print(f"   Sources:  {', '.join(r['source_documents'])}")
            print()

    print("=" * 70)


def cmd_study_plan(subject: str, days: int = 7, minutes: int = 60) -> None:
    """Generates and displays an academic study plan."""
    print("=" * 70)
    print("STUDENTAI STUDY PLAN")
    print("=" * 70)
    print(f"Subject:    {subject}")
    print(f"Duration:   {days} days")
    print(f"Daily time: {minutes} minutes")
    print("=" * 70)

    try:
        plan_res = create_study_plan(
            subject=subject,
            days=days,
            daily_minutes=minutes,
        )
    except Exception as err:
        print(f"\nError creating study plan: {err}\n")
        print("=" * 70)
        sys.exit(1)

    if plan_res.get("message") and not plan_res.get("plan"):
        print(f"\n{plan_res['message']}\n")
        print("=" * 70)
        return

    days_plan = plan_res.get("plan", [])
    for d in days_plan:
        focus_topics = ", ".join(f"{f['topic']} / {f['subtopic']}" for f in d.get("focus", []))
        print(f"\nDay {d['day']}")
        print("-" * 70)
        print(f"Focus: {focus_topics}")
        for idx, a in enumerate(d.get("activities", []), start=1):
            print(f"{idx}. {a['description']:<50} {a['minutes']} min")
        print(f"Total: {d.get('estimated_minutes', minutes)} min")

    print("\n" + "=" * 70)


def cmd_revision(subject: Optional[str] = None, limit: int = 5) -> None:
    """Displays prioritized revision recommendations."""
    print("=" * 70)
    print("STUDENTAI REVISION RECOMMENDATIONS")
    print("=" * 70)
    if subject:
        print(f"Subject Filter: {subject}")
        print("-" * 70)

    rev_res = get_revision_recommendations(
        subject=subject,
        limit=limit,
    )

    items = rev_res.get("revision_items", [])
    if not items:
        msg = rev_res.get("message", "No revision items available.")
        print(f"\n{msg}\n")
        print("=" * 70)
        return

    print()
    for idx, item in enumerate(items, start=1):
        acc_str = f"{item['accuracy'] * 100:.0f}%" if item["accuracy"] is not None else "N/A"
        print(f"{idx}. {item['topic']} / {item['subtopic']}")
        print(f"   Priority: {item['priority']} (Quiz Accuracy: {acc_str})")
        print(f"   Reason:   {item['reason']}")
        print(f"   Action:   {item['recommended_activity']}")
        if item.get("source_documents"):
            print(f"   Sources:  {', '.join(item['source_documents'])}")
        print()
    print("=" * 70)


def cmd_study_plans() -> None:
    """Lists saved study plans."""
    print("=" * 70)
    print("SAVED STUDY PLANS")
    print("=" * 70)
    plans = get_study_plans()
    if not plans:
        print("\nNo saved study plans found.\n")
    else:
        for idx, p in enumerate(plans, start=1):
            ts = p.get("created_at", "")[:19].replace("T", " ")
            print(f"{idx}. [{ts}] Subject: {p.get('subject')} | Duration: {p.get('days')} days ({p.get('daily_minutes')} min/day)")
            print(f"   Plan ID: {p.get('plan_id')}")
            print()
    print("=" * 70)


def cmd_career_docs() -> None:
    """Lists indexed career documents (resumes, job descriptions)."""
    sr = get_sparse_retriever()
    career_docs: Dict[str, Dict[str, Any]] = {}
    for doc in sr.documents.values():
        if doc.get("category") == "career" or doc.get("document_type") in ("resume", "job_description"):
            d_id = doc.get("document_id") or doc.get("file_name")
            if d_id not in career_docs:
                career_docs[d_id] = {
                    "document_id": d_id,
                    "file_name": doc.get("file_name"),
                    "document_type": doc.get("document_type"),
                    "total_pages": doc.get("total_pages"),
                    "category": doc.get("category"),
                }

    print("=" * 70)
    print("INDEXED CAREER DOCUMENTS")
    print("=" * 70)
    if not career_docs:
        print("\nNo career documents indexed yet.\nIngest resumes or job descriptions with:")
        print("  python -m src.main ingest <path-to-pdf> --category career --type resume")
        print("  python -m src.main ingest <path-to-pdf> --category career --type job_description\n")
    else:
        for idx, (d_id, d_info) in enumerate(career_docs.items(), start=1):
            print(f"{idx}. File: {d_info['file_name']}")
            print(f"   Document ID:   {d_info['document_id']}")
            print(f"   Document Type: {d_info['document_type']}")
            print(f"   Category:      {d_info['category']}")
            print()
    print("=" * 70)


def cmd_career_ask(
    query: str,
    top_k: int = 5,
    document_type: Optional[str] = None,
    document_id: Optional[str] = None,
) -> None:
    """Answers questions strictly using career documents (resumes, job descriptions)."""
    require_api_key()
    print("=" * 70)
    print("STUDENTAI CAREER RAG")
    print("=" * 70)
    print(f"Question:      {query}")
    if document_type:
        print(f"Document Type: {document_type}")
    if document_id:
        print(f"Document ID:   {document_id}")
    print("-" * 70)

    res = answer_career_question(
        query=query,
        top_k=top_k,
        document_type=document_type,
        document_id=document_id,
    )

    print("\nANSWER:")
    print(res["answer"])
    print("\n" + "-" * 70)
    print(f"SOURCES ({len(res['sources'])} retrieved):")
    if not res["sources"]:
        print("None")
    for idx, s in enumerate(res["sources"], start=1):
        score_val = s.get("score", 0.0)
        print(f"{idx}. [{s['file_name']}] Page {s['page_number']} (Score: {score_val:.4f}, Type: {s['document_type']})")
    print("=" * 70)


def cmd_career_skills(document_id: str) -> None:
    """Extracts explicit skills from an indexed resume or job description."""
    require_api_key()
    sr = get_sparse_retriever()
    try:
        doc = get_career_document_text(document_id, sparse_retriever=sr)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 70)
    print("CAREER SKILL EXTRACTION")
    print("=" * 70)
    print(f"File:          {doc['file_name']}")
    print(f"Document ID:   {doc['document_id']}")
    print(f"Document Type: {doc['document_type']}")
    print("-" * 70)

    extracted = extract_skills(text=doc["full_text"], document_type=doc["document_type"])
    skills = extracted.get("skills", [])
    print(f"\nExtracted Skills ({len(skills)} found):\n")

    if not skills:
        print("No explicitly mentioned skills detected in document text.")
    else:
        for idx, s in enumerate(skills, start=1):
            print(f"{idx}. {s['name']} (Type: {s['skill_type']}, Confidence: {s['confidence']:.2f})")
            if s.get("evidence_quote"):
                print(f"   Evidence: \"{s['evidence_quote']}\"")
    print("\n" + "=" * 70)


def cmd_match_job(resume_id: str, job_id: str) -> None:
    """Performs deterministic match between a resume and job description."""
    require_api_key()
    sr = get_sparse_retriever()
    try:
        fit = analyze_job_fit(resume_document_id=resume_id, job_document_id=job_id, sparse_retriever=sr)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    res_info = fit["resume"]
    job_info = fit["job"]
    m = fit["match"]

    print("=" * 70)
    print("DETERMINISTIC JOB FIT ANALYSIS")
    print("=" * 70)
    print(f"Resume:  {res_info['file_name']} (ID: {res_info['document_id']}) - {res_info['skill_count']} skills")
    print(f"Job:     {job_info['file_name']} (ID: {job_info['document_id']}) - {job_info['requirement_count']} requirements")
    print("-" * 70)
    print(f"Coverage Score:     {m['coverage_percentage']:.1f}%")
    print(f"Matched Skills:     {m['matched_count']} / {m['requirement_count']}")
    print(f"Partial Matches:    {m['partial_count']}")
    print(f"Missing Skills:     {m['missing_count']}")
    print("-" * 70)

    print("\nMATCHED SKILLS (Full / Exact Equivalent):")
    if not m["matched_skills"]:
        print("None")
    for s in m["matched_skills"]:
        print(f"- {s['requirement']} (Matched: '{s['matched_resume_skill']}', Type: {s['skill_type']})")
        if s.get("resume_evidence"):
            print(f"  Resume Quote: \"{s['resume_evidence']}\"")

    if m["partial_matches"]:
        print("\nPARTIAL / RELATED MENTIONS (NOT counted in coverage score):")
        for p in m["partial_matches"]:
            print(f"- {p['requirement']}: candidate mentions '{p['candidate_mention']}'")
            if p.get("resume_evidence"):
                print(f"  Resume Quote: \"{p['resume_evidence']}\"")

    print("\nMISSING REQUIREMENTS:")
    if not m["missing_skills"]:
        print("None (All requirements matched)")
    for s in m["missing_skills"]:
        print(f"- {s['requirement']} (Type: {s['skill_type']})")
        if s.get("job_evidence"):
            print(f"  Job Quote: \"{s['job_evidence']}\"")

    print("\n" + "-" * 70)
    print("IMPORTANT DISCLAIMER & LIMITATIONS:")
    for lim in m["limitations"]:
        print(f"* {lim}")
    print("=" * 70)


def cmd_projects() -> None:
    """Lists indexed student software projects and their documented files."""
    sr = get_sparse_retriever()
    projs = list_projects(sparse_retriever=sr)
    print("=" * 70)
    print("INDEXED SOFTWARE PROJECTS")
    print("=" * 70)
    if not projs:
        print("\nNo project documentation indexed yet.\nIngest project files with:")
        print("  python -m src.main ingest <path-to-file> --category project --type project_documentation --subject <project_id>\n")
    else:
        for idx, p in enumerate(projs, start=1):
            print(f"{idx}. Project:    {p['project_name']}")
            print(f"   Project ID: {p['project_id']}")
            print(f"   Documents:  {p['document_count']} ({', '.join(p['filenames'])})")
            print()
    print("=" * 70)


def cmd_project_ask(
    query: str,
    project_id: Optional[str] = None,
    top_k: int = 5,
) -> None:
    """Answers questions grounded strictly in project documentation."""
    require_api_key()
    print("=" * 70)
    print("STUDENTAI PROJECT RAG")
    print("=" * 70)
    print(f"Question:   {query}")
    if project_id:
        print(f"Project ID: {project_id}")
    print("-" * 70)

    res = answer_project_question(
        query=query,
        project_id=project_id,
        top_k=top_k,
    )

    if res.get("ambiguous"):
        print("\nAMBIGUITY NOTICE:")
        print(res["answer"])
        print("=" * 70)
        return

    print("\nANSWER:")
    print(res["answer"])
    print("\n" + "-" * 70)
    print(f"SOURCES ({len(res['sources'])} retrieved):")
    if not res["sources"]:
        print("None")
    for idx, s in enumerate(res["sources"], start=1):
        score_val = s.get("score", 0.0)
        print(f"{idx}. [{s['file_name']}] Page {s['page_number']} (Score: {score_val:.4f}, Type: {s['document_type']})")
    print("=" * 70)


def cmd_project_summary(project_id: str) -> None:
    """Generates a structured technical summary of a project."""
    require_api_key()
    print("=" * 70)
    print("PROJECT TECHNICAL SUMMARY")
    print("=" * 70)
    print(f"Project ID: {project_id}")
    print("-" * 70)

    try:
        summary = summarize_project(project_id)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nProject Name:   {summary.get('project_name', project_id)}")
    print(f"Purpose:        {summary.get('documented_purpose', 'Unknown')}")
    print(f"Architecture:   {summary.get('documented_architecture', 'Unknown')}")
    print("\nDocumented Technologies:")
    for tech in summary.get("documented_technologies", []):
        print(f"- {tech}")
    if not summary.get("documented_technologies"):
        print("- None documented")

    print("\nDocumented Features:")
    for feat in summary.get("documented_features", []):
        print(f"- {feat}")
    if not summary.get("documented_features"):
        print("- None documented")

    print("\nDocumented Limitations / Tradeoffs:")
    for lim in summary.get("documented_limitations", []):
        print(f"- {lim}")
    if not summary.get("documented_limitations"):
        print("- None documented")

    print(f"\nSources: {', '.join(summary.get('sources', []))}")
    print("=" * 70)


def cmd_project_interview(
    project_id: str,
    count: int = 5,
    difficulty: str = "MIXED",
) -> None:
    """Generates grounded technical interview questions for a project."""
    require_api_key()
    print("=" * 70)
    print("PROJECT-AWARE INTERVIEW PREPARATION")
    print("=" * 70)
    print(f"Project ID:  {project_id}")
    print(f"Questions:   {count}")
    print(f"Difficulty:  {difficulty}")
    print("-" * 70)

    res = generate_project_interview_questions(
        project_id=project_id,
        count=count,
        difficulty=difficulty,
    )

    if res.get("insufficient_evidence"):
        print(f"\nNotice: {res.get('message', 'Insufficient documentation to generate questions.')}")
        print("=" * 70)
        return

    print(f"\nGenerated {res['generated_count']} Questions for {res['project_name']}:\n")
    for idx, q in enumerate(res["questions"], start=1):
        print(f"{idx}. [{q['difficulty']}] ({q['topic'].upper()})")
        print(f"   Question: {q['question']}")
        if q.get("answer_outline"):
            print(f"   Expected Key Points: {q['answer_outline']}")
        print(f"   Evidence: {q['evidence_summary']} ([{q['source_filename']}] Page {q['page_number']})")
        print()
    print("=" * 70)


def cmd_project_compare(project_id_a: str, project_id_b: str) -> None:
    """Compares two documented software projects."""
    require_api_key()
    print("=" * 70)
    print("PROJECT ARCHITECTURE & STACK COMPARISON")
    print("=" * 70)
    print(f"Project A: {project_id_a}")
    print(f"Project B: {project_id_b}")
    print("-" * 70)

    try:
        cmp_res = compare_projects(project_id_a, project_id_b)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    tbl = cmp_res["comparison_table"]
    pa_name = cmp_res["project_a"].get("project_name", project_id_a)
    pb_name = cmp_res["project_b"].get("project_name", project_id_b)

    print(f"\n1. Purpose:")
    print(f"   {pa_name}: {tbl['purpose'].get(pa_name, 'Unknown')}")
    print(f"   {pb_name}: {tbl['purpose'].get(pb_name, 'Unknown')}")

    print(f"\n2. Documented Architecture:")
    print(f"   {pa_name}: {tbl['architecture'].get(pa_name, 'Unknown')}")
    print(f"   {pb_name}: {tbl['architecture'].get(pb_name, 'Unknown')}")

    print(f"\n3. Technologies:")
    print(f"   {pa_name}: {', '.join(tbl['technologies'].get(pa_name, []))}")
    print(f"   {pb_name}: {', '.join(tbl['technologies'].get(pb_name, []))}")

    print(f"\n4. Limitations & Tradeoffs:")
    print(f"   {pa_name}: {', '.join(tbl['limitations'].get(pa_name, []))}")
    print(f"   {pb_name}: {', '.join(tbl['limitations'].get(pb_name, []))}")
    print("\n" + "=" * 70)


def cmd_evaluate(
    eval_file: str = "data/evaluation/rag_eval.json",
    top_k: int = 5,
    skip_generation: bool = False,
) -> None:
    """Executes full RAG evaluation benchmark across retrieval modes and generation."""
    require_api_key()
    path = Path(eval_file)
    if not path.is_file():
        print(f"Error: Evaluation file not found at: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    q_client = initialize_vector_store()
    sr = get_sparse_retriever()
    settings = get_settings()

    q_info = get_collection_info(q_client)
    total_docs = len(set(d["file_name"] for d in sr.documents.values()))
    total_chunks = q_info.get("points_count", sr.total_docs)

    print("=" * 60)
    print("STUDENTAI RAG EVALUATION")
    print("=" * 60)
    print(f"\nDocuments:\n{total_docs}\n")
    print(f"Chunks:\n{total_chunks}\n")
    print(f"Questions:\n{len(eval_dataset)}\n")

    # 1. Retrieval Evaluation across modes
    modes = ["dense", "sparse", "hybrid", "hybrid_rerank", "hybrid_rewrite"]
    mode_labels = {
        "dense": "Dense",
        "sparse": "Sparse",
        "hybrid": "Hybrid",
        "hybrid_rerank": "Hybrid + Reranker",
        "hybrid_rewrite": "Hybrid + Rewriter",
    }
    mode_eval_results = {}

    import time
    t_ret_start = time.perf_counter()

    for m in modes:
        mode_eval_results[m] = evaluate_retrieval_mode(
            eval_dataset=eval_dataset,
            mode=m,
            top_k=top_k,
            client=q_client,
            sparse_retriever=sr,
        )

    t_ret_total = time.perf_counter() - t_ret_start

    print("-" * 60)
    print("RETRIEVAL RESULTS")
    print("-" * 60)
    print(f"{'Mode':<22}{'R@1':<8}{'R@3':<8}{'R@5':<8}{'MRR':<8}")
    for m in modes:
        met = mode_eval_results[m]["metrics"]
        lbl = mode_labels[m]
        print(f"{lbl:<22}{met['recall_at_1']:<8.4f}{met['recall_at_3']:<8.4f}{met['recall_at_5']:<8.4f}{met['mrr']:<8.4f}")

    print("\n" + "-" * 60)
    print("DOCUMENT RECALL")
    print("-" * 60)
    print(f"{'Mode':<22}{'Document Recall@5':<18}")
    for m in modes:
        met = mode_eval_results[m]["metrics"]
        lbl = mode_labels[m]
        print(f"{lbl:<22}{met['doc_recall_at_5']:<18.4f}")

    # 2. Generation Evaluation
    t_gen_start = time.perf_counter()
    if not skip_generation:
        gen_eval = evaluate_generation_pipeline(
            eval_dataset=eval_dataset,
            client=q_client,
            hybrid=True,
            sparse_retriever=sr,
        )
    else:
        gen_eval = None
    t_gen_total = time.perf_counter() - t_gen_start

    if gen_eval:
        print("\n" + "-" * 60)
        print("GENERATION CHECKS")
        print("-" * 60)
        print(f"Citation correctness:\n{gen_eval['citation_accuracy'] * 100:.1f}%\n")
        print(f"Expected document correctness:\n{gen_eval['document_accuracy'] * 100:.1f}%\n")
        print(f"Refusal correctness:\n{gen_eval['refusal_accuracy'] * 100:.1f}%\n")

    # 3. Per-Question Detailed Reporting for Hybrid Mode
    print("-" * 60)
    print("PER-QUESTION RESULTS (HYBRID)")
    print("-" * 60)
    hybrid_details = mode_eval_results["hybrid"]["per_question"]
    for idx, item in enumerate(hybrid_details):
        q_text = item["question"]
        is_ans = item["answerable"]
        top_res = item["top_result"] or "None"
        first_rank = item["first_relevant_rank"] if is_ans else "N/A"
        
        # Generation check for this question
        gen_check = gen_eval["checks"][idx] if gen_eval else None
        cites = gen_check["cited_pages"] if gen_check else []
        passed = gen_check["passed"] if gen_check else item["is_hit_top5"]
        status_str = "PASS" if passed else "FAIL"

        print(f"Question: {q_text}")
        print(f"Mode:     Hybrid")
        print(f"Top result: {top_res}")
        print(f"First relevant rank: {first_rank}")
        if is_ans and cites:
            print(f"Citation: [Page {', '.join(str(p) for p in cites)}]")
        elif not is_ans:
            print(f"Refusal verified: {gen_check['refusal_correct'] if gen_check else 'Yes'}")
        print(f"Status:   {status_str}")
        print()

    print("=" * 60)
    print(f"Evaluation Runtime:")
    print(f"  Retrieval (4 modes):  {t_ret_total:.2f}s")
    if gen_eval:
        print(f"  Generation pipeline:  {t_gen_total:.2f}s")
    print("=" * 60)
    q_client.close()


def cmd_agent(query: str) -> None:
    """Executes controlled agentic orchestration for a multi-domain or complex user goal."""
    from src.agent_orchestrator import run_agent

    print("=" * 70)
    print("STUDENTAI AGENT ORCHESTRATOR")
    print("=" * 70)
    print(f"User Goal: {query.strip()}\n")

    result = run_agent(user_query=query.strip())

    print(f"Classification:   {result.classification}")
    print(f"Workflow:         {result.workflow}")
    print(f"Execution Time:   {result.total_duration_ms:.2f} ms")
    print("-" * 70)
    print("EXECUTION TRACE:")
    for tr in result.execution_trace:
        status_str = tr.status.upper()
        print(f"Step {tr.step}: [{tr.tool_name}] -> {status_str} ({tr.duration_ms:.1f} ms)")
        if tr.args:
            print(f"  Args: {tr.args}")
        if tr.error:
            print(f"  Error: {tr.error}")
    print("-" * 70)

    print("\nSYNTHESIZED ANSWER:")
    print(result.answer)
    print()

    if result.sources:
        print("-" * 70)
        print(f"SOURCES ({len(result.sources)} chunks cited/retrieved):")
        for idx, s in enumerate(result.sources[:5], start=1):
            print(f"{idx}. [{s.get('file_name', 'Unknown')}] Page {s.get('page_number', 0)} (Score: {s.get('score', 0.0):.4f})")

    if result.citation_report and result.citation_report.get("has_citations"):
        print("-" * 70)
        valid_mark = "VALID" if result.citation_report["is_valid"] else "INVALID"
        print(f"Citation Verification: {valid_mark}")
        print(f"Cited Pages:           {result.citation_report['cited_pages']}")
        if result.citation_report["invalid_pages"]:
            print(f"Invalid Pages:         {result.citation_report['invalid_pages']}")

    if result.limitations:
        print("-" * 70)
        print("LIMITATIONS & WARNINGS:")
        for lim in result.limitations:
            print(f"* {lim}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="StudentAI - Personal Knowledge, Learning & Career RAG")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Check configuration and local vector database status")

    # Extract command (Step 2)
    extract_parser = subparsers.add_parser("extract", help="Extract and preview text from a PDF without embeddings or LLM")
    extract_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    extract_parser.add_argument("--preview-chars", type=int, default=250, help="Number of characters to preview per page")
    extract_parser.add_argument("--max-pages", type=int, default=5, help="Maximum number of pages to preview")

    # Chunk command (Step 3)
    chunk_parser = subparsers.add_parser("chunk", help="Chunk an extracted PDF without embeddings or LLM")
    chunk_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    chunk_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    chunk_parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters")

    # Ingest / Index command (Step 8 & 12)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a single PDF into Qdrant with metadata")
    ingest_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    ingest_parser.add_argument("--category", type=str, default=None, help="Document category (e.g. academic, career, dsa)")
    ingest_parser.add_argument("--subject", type=str, default=None, help="Document subject (e.g. computer_networks, database)")
    ingest_parser.add_argument("--type", type=str, default=None, help="Document type (e.g. notes, resume, textbook)")

    # Index command (alias for ingest)
    index_parser = subparsers.add_parser("index", help="Alias for ingest")
    index_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    index_parser.add_argument("--category", type=str, default=None, help="Document category (e.g. academic, career, dsa)")
    index_parser.add_argument("--subject", type=str, default=None, help="Document subject (e.g. computer_networks, database)")
    index_parser.add_argument("--type", type=str, default=None, help="Document type (e.g. notes, resume, textbook)")

    # Ingest-dir command (Step 8 & 12)
    ingest_dir_parser = subparsers.add_parser("ingest-dir", help="Ingest all PDFs in a directory into Qdrant")
    ingest_dir_parser.add_argument("directory_path", type=str, help="Path to the directory containing PDFs")
    ingest_dir_parser.add_argument("--category", type=str, default=None, help="Default category for directory documents")
    ingest_dir_parser.add_argument("--subject", type=str, default=None, help="Default subject for directory documents")
    ingest_dir_parser.add_argument("--type", type=str, default=None, help="Default type for directory documents")

    # Search command (Step 6, Step 9, Step 10, Step 12)
    search_parser = subparsers.add_parser("search", help="Execute semantic, sparse, or hybrid retrieval")
    search_parser.add_argument("query", type=str, help="Search query string")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    search_parser.add_argument("--rerank", action="store_true", help="Enable second-stage relevance reranking")
    search_parser.add_argument("--candidates", type=int, default=10, help="Number of initial candidates for reranking/hybrid")
    search_parser.add_argument("--hybrid", action="store_true", help="Enable Hybrid retrieval (Dense + Sparse BM25 RRF)")
    search_parser.add_argument("--sparse", action="store_true", help="Run BM25 sparse retrieval only")
    search_parser.add_argument("--category", type=str, default=None, help="Filter by document category")
    search_parser.add_argument("--subject", type=str, default=None, help="Filter by document subject")
    search_parser.add_argument("--type", type=str, default=None, help="Filter by document type")

    # Ask command (Step 7, Step 9, Step 10, Step 12)
    ask_parser = subparsers.add_parser("ask", help="Ask a question and generate a grounded, cited RAG answer")
    ask_parser.add_argument("question", type=str, help="Question to answer")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of context passages to retrieve")
    ask_parser.add_argument("--rerank", action="store_true", help="Enable second-stage relevance reranking")
    ask_parser.add_argument("--candidates", type=int, default=10, help="Number of initial candidates")
    ask_parser.add_argument("--hybrid", action="store_true", help="Enable Hybrid retrieval (Dense + Sparse BM25 RRF)")
    ask_parser.add_argument("--category", type=str, default=None, help="Filter context by document category")
    ask_parser.add_argument("--subject", type=str, default=None, help="Filter context by document subject")
    ask_parser.add_argument("--type", type=str, default=None, help="Filter context by document type")
    ask_parser.add_argument("--rewrite", action="store_true", help="Enable conservative query rewriting before retrieval")

    # Ask-auto command (Step 13: Query Routing)
    ask_auto_parser = subparsers.add_parser("ask-auto", help="Ask a question with automated query routing and metadata filtering")
    ask_auto_parser.add_argument("question", type=str, help="Question to answer")
    ask_auto_parser.add_argument("--top-k", type=int, default=5, help="Number of context passages to retrieve")
    ask_auto_parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold to accept route (default: 0.70)")

    # Learn command (Step 14: Learning RAG)
    learn_parser = subparsers.add_parser("learn", help="Ask an academic question with pedagogical intent classification (explain, summarize, compare, quiz, exam-answer)")
    learn_parser.add_argument("question", type=str, help="Academic question, concept, or topic")
    learn_parser.add_argument("--top-k", type=int, default=5, help="Number of context passages to retrieve")
    learn_parser.add_argument("--intent", type=str, default=None, choices=["EXPLAIN", "SUMMARIZE", "COMPARE", "QUIZ", "EXAM_ANSWER", "GENERAL_LEARNING"], help="Explicitly specify learning intent override")
    learn_parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold to accept intent (default: 0.70)")

    # Learning-stats command (Step 15: Learning Analytics)
    stats_parser = subparsers.add_parser("learning-stats", help="Display topic and subtopic learning analytics and weak topics")
    stats_parser.add_argument("--topic", type=str, default=None, help="Filter analytics down to a specific topic")

    # Learning-events command (Step 15: Interaction Tracking)
    events_parser = subparsers.add_parser("learning-events", help="List recent learning interaction events")
    events_parser.add_argument("--topic", type=str, default=None, help="Filter events by topic")

    # Record-quiz command (Step 15: Quiz Outcome Tracking)
    quiz_parser = subparsers.add_parser("record-quiz", help="Record student quiz performance outcome")
    quiz_parser.add_argument("--topic", type=str, required=True, help="Academic topic")
    quiz_parser.add_argument("--subtopic", type=str, required=True, help="Academic subtopic")
    quiz_parser.add_argument("--attempted", type=int, required=True, help="Total questions attempted")
    quiz_parser.add_argument("--correct", type=int, required=True, help="Total questions answered correctly")
    quiz_parser.add_argument("--incorrect", type=int, required=True, help="Total questions answered incorrectly")

    # Recommendations command (Step 16: Personalized Learning Recommendations)
    recs_parser = subparsers.add_parser("recommendations", help="Display actionable personalized learning recommendations")
    recs_parser.add_argument("--limit", type=int, default=5, help="Maximum number of recommendations (default: 5)")
    recs_parser.add_argument("--topic", type=str, default=None, help="Filter recommendations down to a specific topic")

    # Learning-profile command (Step 16: Personalized Learning Profile)
    subparsers.add_parser("learning-profile", help="Display structured personalized learning profile")

    # Study-plan command (Step 17: Study Planner)
    plan_parser = subparsers.add_parser("study-plan", help="Generate a structured multi-day academic study plan")
    plan_parser.add_argument("--subject", type=str, required=True, help="Academic subject (e.g. DBMS, Computer Networks)")
    plan_parser.add_argument("--days", type=int, default=7, help="Plan duration in days (default: 7)")
    plan_parser.add_argument("--minutes", type=int, default=60, help="Daily study minutes (default: 60)")

    # Revision command (Step 17: Revision Engine)
    rev_parser = subparsers.add_parser("revision", help="Display prioritized revision recommendations")
    rev_parser.add_argument("--subject", type=str, default=None, help="Filter revision items by academic subject")
    rev_parser.add_argument("--limit", type=int, default=5, help="Maximum revision items (default: 5)")

    # Study-plans command (Step 17: Study Plan Persistence)
    subparsers.add_parser("study-plans", help="List saved study plans")

    # Evaluate command (Step 11)
    eval_parser = subparsers.add_parser("evaluate", help="Run full RAG benchmark evaluation")
    eval_parser.add_argument("--file", type=str, default="data/evaluation/rag_eval.json", help="Path to evaluation dataset")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Top-K for evaluation")
    eval_parser.add_argument("--skip-generation", action="store_true", help="Evaluate retrieval only")

    # Career-docs command (Step 18: Career RAG)
    subparsers.add_parser("career-docs", help="List indexed career documents (resumes, job descriptions)")

    # Career-ask command (Step 18: Career RAG)
    career_ask_parser = subparsers.add_parser("career-ask", help="Ask a question scoped strictly to career documents (resumes, job descriptions)")
    career_ask_parser.add_argument("question", type=str, help="Career question or candidate query")
    career_ask_parser.add_argument("--top-k", type=int, default=5, help="Number of context passages to retrieve")
    career_ask_parser.add_argument("--type", type=str, default=None, choices=["resume", "job_description"], help="Filter by career document type")
    career_ask_parser.add_argument("--document-id", type=str, default=None, help="Scope query strictly to a single document ID")

    # Career-skills command (Step 18: Skill Extraction)
    career_skills_parser = subparsers.add_parser("career-skills", help="Extract explicit skills from an indexed career document")
    career_skills_parser.add_argument("--document-id", type=str, required=True, help="Document ID or filename of indexed resume or job description")

    # Match-job command (Step 18: Deterministic Job Matcher)
    match_job_parser = subparsers.add_parser("match-job", help="Perform deterministic matching between a resume and job description")
    match_job_parser.add_argument("--resume-id", type=str, required=True, help="Document ID or filename of the resume")
    match_job_parser.add_argument("--job-id", type=str, required=True, help="Document ID or filename of the job description")

    # Projects command (Step 19: Project RAG)
    subparsers.add_parser("projects", help="List indexed student software projects and their documented files")

    # Project-ask command (Step 19: Project RAG)
    project_ask_parser = subparsers.add_parser("project-ask", help="Ask a question grounded in software project documentation")
    project_ask_parser.add_argument("question", type=str, help="Question about project architecture, tech stack, or implementation")
    project_ask_parser.add_argument("--project-id", type=str, default=None, help="Specific project identifier to isolate retrieval")
    project_ask_parser.add_argument("--top-k", type=int, default=5, help="Number of context passages to retrieve")

    # Project-summary command (Step 19: Project Summary)
    project_sum_parser = subparsers.add_parser("project-summary", help="Generate a structured technical summary of a project")
    project_sum_parser.add_argument("--project-id", type=str, required=True, help="Project identifier or project name")

    # Project-interview command (Step 19: Project-Aware Interview Preparation)
    project_int_parser = subparsers.add_parser("project-interview", help="Generate grounded technical interview questions from project documentation")
    project_int_parser.add_argument("--project-id", type=str, required=True, help="Project identifier or project name")
    project_int_parser.add_argument("--count", type=int, default=5, help="Number of interview questions to generate (default: 5)")
    project_int_parser.add_argument("--difficulty", type=str, default="MIXED", choices=["EASY", "MEDIUM", "HARD", "MIXED"], help="Target question difficulty (default: MIXED)")

    # Project-compare command (Step 19: Project Comparison)
    project_cmp_parser = subparsers.add_parser("project-compare", help="Compare two documented software projects")
    project_cmp_parser.add_argument("--project-a", type=str, required=True, help="First project identifier")
    project_cmp_parser.add_argument("--project-b", type=str, required=True, help="Second project identifier")

    # Agent command (Step 21: Agentic Orchestration Layer)
    agent_parser = subparsers.add_parser("agent", help="Execute multi-step goal using controlled agentic orchestrator")
    agent_parser.add_argument("--query", "-q", type=str, required=True, help="User goal, question, or task")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "extract":
        cmd_extract(args.pdf_path, preview_chars=args.preview_chars, max_pages_preview=args.max_pages)
    elif args.command == "chunk":
        cmd_chunk(args.pdf_path, chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.command in ("ingest", "index"):
        cmd_ingest_single(
            args.pdf_path,
            category=args.category,
            subject=args.subject,
            document_type=args.type,
        )
    elif args.command == "ingest-dir":
        cmd_ingest_dir(args.directory_path)
    elif args.command == "search":
        cmd_search(
            args.query,
            top_k=args.top_k,
            rerank=args.rerank,
            initial_candidates=args.candidates,
            hybrid=args.hybrid,
            sparse_only=args.sparse,
            category=args.category,
            subject=args.subject,
            document_type=args.type,
        )
    elif args.command == "ask":
        cmd_ask(
            args.question,
            top_k=args.top_k,
            rerank=args.rerank,
            initial_candidates=args.candidates,
            hybrid=args.hybrid,
            category=args.category,
            subject=args.subject,
            document_type=args.type,
            rewrite=args.rewrite,
        )
    elif args.command == "ask-auto":
        cmd_ask_auto(
            args.question,
            top_k=args.top_k,
            threshold=args.threshold,
        )
    elif args.command == "learn":
        cmd_learn(
            args.question,
            top_k=args.top_k,
            intent=args.intent,
            threshold=args.threshold,
        )
    elif args.command == "learning-stats":
        cmd_learning_stats(topic=args.topic)
    elif args.command == "learning-events":
        cmd_learning_events(topic=args.topic)
    elif args.command == "record-quiz":
        cmd_record_quiz(
            topic=args.topic,
            subtopic=args.subtopic,
            attempted=args.attempted,
            correct=args.correct,
            incorrect=args.incorrect,
        )
    elif args.command == "recommendations":
        cmd_recommendations(limit=args.limit, topic=args.topic)
    elif args.command == "learning-profile":
        cmd_learning_profile()
    elif args.command == "study-plan":
        cmd_study_plan(
            subject=args.subject,
            days=args.days,
            minutes=args.minutes,
        )
    elif args.command == "revision":
        cmd_revision(
            subject=args.subject,
            limit=args.limit,
        )
    elif args.command == "study-plans":
        cmd_study_plans()
    elif args.command == "career-docs":
        cmd_career_docs()
    elif args.command == "career-ask":
        cmd_career_ask(
            args.question,
            top_k=args.top_k,
            document_type=args.type,
            document_id=args.document_id,
        )
    elif args.command == "career-skills":
        cmd_career_skills(document_id=args.document_id)
    elif args.command == "match-job":
        cmd_match_job(resume_id=args.resume_id, job_id=args.job_id)
    elif args.command == "projects":
        cmd_projects()
    elif args.command == "project-ask":
        cmd_project_ask(
            args.question,
            project_id=args.project_id,
            top_k=args.top_k,
        )
    elif args.command == "project-summary":
        cmd_project_summary(project_id=args.project_id)
    elif args.command == "project-interview":
        cmd_project_interview(
            project_id=args.project_id,
            count=args.count,
            difficulty=args.difficulty,
        )
    elif args.command == "project-compare":
        cmd_project_compare(
            project_id_a=args.project_a,
            project_id_b=args.project_b,
        )
    elif args.command == "evaluate":
        cmd_evaluate(
            eval_file=args.file,
            top_k=args.top_k,
            skip_generation=args.skip_generation,
        )
    elif args.command == "agent":
        cmd_agent(args.query)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
