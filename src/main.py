"""
CLI Entry Point for StudentAI Phase 1.
Provides commands to inspect configuration, ingest documents, and ask questions.

Usage:
  python -m src.main status
  python -m src.main ingest <path-to-pdf>
  python -m src.main query "Your question here"
"""

import sys
import argparse
from pathlib import Path
from src.config import get_settings, require_api_key
from src.extractor import extract_pdf_pages
from src.chunker import chunk_pages
from src.embedder import embed_batch, get_genai_client
from src.vector_store import get_qdrant_client, init_collection, upsert_chunks
from src.retriever import retrieve_context
from src.generator import generate_rag_answer


def cmd_status() -> None:
    """Checks configuration and local database status."""
    settings = get_settings()
    print("=" * 60)
    print("StudentAI - Phase 1 Environment Status")
    print("=" * 60)
    print(f"Embedding Model:    {settings.embedding_model} (Dim: {settings.embedding_dim})")
    print(f"Generation Model:   {settings.generation_model}")
    print(f"Qdrant Path:        {settings.qdrant_storage_path}")
    print(f"Collection Name:    {settings.qdrant_collection_name}")
    print(f"Chunk Size/Overlap: {settings.chunk_size} / {settings.chunk_overlap} chars")

    # Check API key configuration without printing it
    if settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here":
        masked_key = settings.gemini_api_key[:4] + "..." + settings.gemini_api_key[-4:]
        print(f"Gemini API Key:     Configured ({masked_key})")
    else:
        print("Gemini API Key:     [NOT CONFIGURED] (Set GEMINI_API_KEY in .env)")

    # Check local Qdrant collection
    try:
        q_client = get_qdrant_client()
        collections = [c.name for c in q_client.get_collections().collections]
        print(f"Qdrant Collections: {collections if collections else 'None (Collection not yet created)'}")
        q_client.close()
    except Exception as e:
        print(f"Qdrant Status:      Error accessing local store: {e}")
    print("=" * 60)


def cmd_ingest(pdf_path: str) -> None:
    """Executes PDF extraction -> chunking -> embedding -> Qdrant storage."""
    require_api_key()
    settings = get_settings()

    print(f"[1/4] Extracting text from: {pdf_path}")
    pages = extract_pdf_pages(pdf_path)
    print(f"      Extracted {len(pages)} pages with text.")

    print(f"[2/4] Chunking pages (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    chunks = chunk_pages(pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    print(f"      Created {len(chunks)} chunks.")

    print(f"[3/4] Generating embeddings using {settings.embedding_model}...")
    texts_to_embed = [c["text"] for c in chunks]
    genai_client = get_genai_client()
    embeddings = embed_batch(texts_to_embed, client=genai_client)
    print(f"      Generated {len(embeddings)} vectors.")

    print(f"[4/4] Upserting vectors into Qdrant collection '{settings.qdrant_collection_name}'...")
    q_client = get_qdrant_client()
    try:
        init_collection(q_client)
        count = upsert_chunks(q_client, chunks, embeddings)
        print(f"      Successfully stored {count} chunks in Qdrant!")
    finally:
        q_client.close()


def cmd_query(question: str, top_k: int = 4) -> None:
    """Executes query embedding -> Qdrant retrieval -> Gemini generation."""
    require_api_key()

    print(f"Question: {question}")
    print(f"[1/2] Retrieving top {top_k} relevant passages from Qdrant...")
    q_client = get_qdrant_client()
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
    parser = argparse.ArgumentParser(description="StudentAI Phase 1 - Foundational RAG")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Check configuration and local database status")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and index a PDF file")
    ingest_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question against indexed documents")
    query_parser.add_argument("question", type=str, help="Question to answer")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "ingest":
        cmd_ingest(args.pdf_path)
    elif args.command == "query":
        cmd_query(args.question, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
