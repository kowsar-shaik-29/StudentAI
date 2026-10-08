"""
CLI Entry Point for StudentAI.
Provides commands to inspect configuration, extract PDF text, chunk documents, index vectors, and query.

Usage:
  python -m src.main status
  python -m src.main extract <path-to-pdf>
  python -m src.main chunk <path-to-pdf>
  python -m src.main index <path-to-pdf>
  python -m src.main query "Your question here"
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import get_settings, require_api_key
from src.extractor import extract_pdf, extract_pdf_pages
from src.chunker import create_chunks
from src.embedder import embed_chunks, get_genai_client
from src.vector_store import (
    initialize_vector_store,
    ensure_collection,
    upsert_chunks,
    count_vectors,
    get_collection_info,
)
from src.retriever import retrieve_context
from src.generator import generate_rag_answer


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


def cmd_index(pdf_path: str) -> None:
    """
    Step 5: Executes full indexing pipeline:
    PDF -> extract -> chunk -> embed -> store in local Qdrant.
    """
    require_api_key()
    settings = get_settings()
    path = Path(pdf_path)

    print("=" * 60)
    print(f"Indexing Document: {path.name}")
    print("=" * 60)

    # 1. Extraction
    print(f"[1/4] Extracting text from: {path.name}")
    pages = extract_pdf(path, include_empty=True)
    pages_with_text = sum(1 for p in pages if p["has_text"])
    print(f"      Extracted {len(pages)} pages ({pages_with_text} with text).")

    # 2. Chunking
    print(f"[2/4] Chunking pages (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    chunks = create_chunks(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    print(f"      Created {len(chunks)} chunks.")

    # 3. Embedding
    print(f"[3/4] Generating embeddings using {settings.embedding_model} (dim={settings.embedding_dim})...")
    enriched_chunks = embed_chunks(chunks, batch_size=20)
    print(f"      Generated {len(enriched_chunks)} vector embeddings.")

    # 4. Storage in local Qdrant
    print(f"[4/4] Storing vectors in Qdrant collection '{settings.qdrant_collection_name}'...")
    q_client = initialize_vector_store()
    try:
        stored_count = upsert_chunks(q_client, enriched_chunks)
        total_in_collection = count_vectors(q_client)
    finally:
        q_client.close()

    print("=" * 60)
    print("INDEXING SUMMARY:")
    print("=" * 60)
    print(f"Document:         {path.name}")
    print(f"Pages:            {len(pages)}")
    print(f"Chunks:           {len(chunks)}")
    print(f"Embeddings:       {len(enriched_chunks)}")
    print(f"Vectors stored:   {stored_count}")
    print(f"Collection:       {settings.qdrant_collection_name}")
    print(f"Vector dimension: {settings.embedding_dim}")
    print(f"Total in DB:      {total_in_collection}")
    print("=" * 60)
    print("Indexing completed successfully!")


def cmd_query(question: str, top_k: int = 4) -> None:
    """Executes query embedding -> Qdrant retrieval -> Gemini generation."""
    require_api_key()

    print(f"Question: {question}")
    print(f"[1/2] Retrieving top {top_k} relevant passages from Qdrant...")
    q_client = initialize_vector_store()
    try:
        context_chunks = retrieve_context(question, top_k=top_k, client=q_client)
    finally:
        q_client.close()

    if not context_chunks:
        print("No relevant context found in vector database.")
        return

    print(f"      Retrieved {len(context_chunks)} chunks.")
    for idx, c in enumerate(context_chunks, 1):
        print(f"      - [{idx}] {c['file_name']} (Page {c['page_number']}) - Score: {c['score']:.4f}")

    print(f"[2/2] Generating answer using Gemini...")
    genai_client = get_genai_client()
    result = generate_rag_answer(question, context_chunks, client=genai_client)

    print("\n" + "=" * 60)
    print("STUDENTAI ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)


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

    # Index command (Step 5)
    index_parser = subparsers.add_parser("index", help="Extract, chunk, embed, and index a PDF into Qdrant")
    index_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")

    # Ingest command (alias to index)
    ingest_parser = subparsers.add_parser("ingest", help="Alias for index")
    ingest_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question against indexed documents")
    query_parser.add_argument("question", type=str, help="Question to answer")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "extract":
        cmd_extract(args.pdf_path, preview_chars=args.preview_chars, max_pages_preview=args.max_pages)
    elif args.command == "chunk":
        cmd_chunk(args.pdf_path, chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.command in ("index", "ingest"):
        cmd_index(args.pdf_path)
    elif args.command == "query":
        cmd_query(args.question, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
