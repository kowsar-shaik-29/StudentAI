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
from src.rag import answer_question
from src.ingestion import ingest_document, ingest_directory
from src.evaluator import evaluate_retrieval_mode
from src.generation_evaluator import evaluate_generation_pipeline



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


def cmd_ingest_single(pdf_path: str) -> None:
    """Ingests a single PDF document into Qdrant."""
    require_api_key()
    try:
        stats = ingest_document(pdf_path)
    except Exception as err:
        print(f"Ingestion Error: {err}")
        sys.exit(1)

    print("=" * 60)
    print("DOCUMENT INGESTION")
    print("=" * 60)
    print(f"\nDocument:\n{stats['file_name']}\n")
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
) -> None:
    """Executes search (dense, sparse, or hybrid) with optional reranking."""
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
            )
        finally:
            q_client.close()
    elif sparse_only:
        sr = get_sparse_retriever()
        results = sr.search(query=clean_query, top_k=top_k)
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
            )
        finally:
            q_client.close()

    if not rerank:
        print(f"Results: {len(results)}\n")

    for i, res in enumerate(results, start=1):
        doc = res["file_name"]
        page = res["page_number"]
        chunk_id = res["chunk_id"]
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
            print(f"Document:          {doc}")
            print(f"Page:              {page}")
            print(f"Chunk:             {chunk_id}")
            print("Text:")
            print(preview_text)
        elif sparse_only:
            print(f"Sparse score (BM25): {res.get('sparse_score', 0.0):.4f}")
            print(f"Document:            {doc}")
            print(f"Page:                {page}")
            print(f"Chunk:               {chunk_id}")
            print("Text:")
            print(preview_text)
        elif rerank:
            print(f"Qdrant score:\n{res.get('score', 0.0):.4f}\n")
            rerank_score = res.get("rerank_score", res.get("score", 0.0))
            print(f"Reranker score:\n{rerank_score:.4f}\n")
            print(f"Document:\n{doc}\n")
            print(f"Page:\n{page}\n")
            print(f"Chunk:\n{chunk_id}\n")
            print("Text:")
            print(preview_text)
        else:
            print(f"Score:    {res.get('score', 0.0):.4f}")
            print(f"Document: {doc}")
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
) -> None:
    """Executes end-to-end RAG question answering with optional reranking or hybrid retrieval."""
    require_api_key()

    print("=" * 60)
    print("STUDENTAI RAG")
    print("=" * 60)
    print(f"\nQuestion:\n{question.strip()}\n")
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
    )

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
        print(f"   Page:  {src['page_number']}")
        print(f"   Score: {src['score']:.4f}")
        if "rerank_score" in src:
            print(f"   Rerank Score: {src['rerank_score']:.4f}")
        print(f"   Chunk: {src['chunk_id']}")
        print()

    print("=" * 60)


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
    modes = ["dense", "sparse", "hybrid", "hybrid_rerank"]
    mode_labels = {
        "dense": "Dense",
        "sparse": "Sparse",
        "hybrid": "Hybrid",
        "hybrid_rerank": "Hybrid + Reranker",
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

    # Ingest / Index command (Step 8)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a single PDF into Qdrant")
    ingest_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")

    # Index command (alias for ingest)
    index_parser = subparsers.add_parser("index", help="Alias for ingest")
    index_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")

    # Ingest-dir command (Step 8)
    ingest_dir_parser = subparsers.add_parser("ingest-dir", help="Ingest all PDFs in a directory into Qdrant")
    ingest_dir_parser.add_argument("directory_path", type=str, help="Path to the directory containing PDFs")

    # Search command (Step 6, Step 9, Step 10)
    search_parser = subparsers.add_parser("search", help="Execute semantic, sparse, or hybrid retrieval")
    search_parser.add_argument("query", type=str, help="Search query string")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    search_parser.add_argument("--rerank", action="store_true", help="Enable second-stage relevance reranking")
    search_parser.add_argument("--candidates", type=int, default=10, help="Number of initial candidates for reranking/hybrid")
    search_parser.add_argument("--hybrid", action="store_true", help="Enable Hybrid retrieval (Dense + Sparse BM25 RRF)")
    search_parser.add_argument("--sparse", action="store_true", help="Run BM25 sparse retrieval only")

    # Ask command (Step 7, Step 9, Step 10)
    ask_parser = subparsers.add_parser("ask", help="Ask a question and generate a grounded, cited RAG answer")
    ask_parser.add_argument("question", type=str, help="Question to answer")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of context passages to retrieve")
    ask_parser.add_argument("--rerank", action="store_true", help="Enable second-stage relevance reranking")
    ask_parser.add_argument("--candidates", type=int, default=10, help="Number of initial candidates")
    ask_parser.add_argument("--hybrid", action="store_true", help="Enable Hybrid retrieval (Dense + Sparse BM25 RRF)")

    # Evaluate command (Step 11)
    eval_parser = subparsers.add_parser("evaluate", help="Run full RAG benchmark evaluation")
    eval_parser.add_argument("--file", type=str, default="data/evaluation/rag_eval.json", help="Path to evaluation dataset")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Top-K for evaluation")
    eval_parser.add_argument("--skip-generation", action="store_true", help="Evaluate retrieval only")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "extract":
        cmd_extract(args.pdf_path, preview_chars=args.preview_chars, max_pages_preview=args.max_pages)
    elif args.command == "chunk":
        cmd_chunk(args.pdf_path, chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.command in ("ingest", "index"):
        cmd_ingest_single(args.pdf_path)
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
        )
    elif args.command == "ask":
        cmd_ask(
            args.question,
            top_k=args.top_k,
            rerank=args.rerank,
            initial_candidates=args.candidates,
            hybrid=args.hybrid,
        )
    elif args.command == "evaluate":
        cmd_evaluate(
            eval_file=args.file,
            top_k=args.top_k,
            skip_generation=args.skip_generation,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
