"""
Document Ingestion Module.
Orchestrates PDF extraction, chunking, embedding generation, and persistent vector storage
for individual files and multi-document directories.

Core Concepts:
1. Multi-Document Knowledge Base:
   All documents are indexed into the single shared Qdrant collection (`student_knowledge_base`).
2. Document Isolation:
   Deterministic chunk IDs (`<filename>_p<page>_c<idx>`) and deterministic RFC 4122 UUID5 point IDs
   prevent cross-document collisions.
3. Idempotent Ingestion:
   Re-ingesting an existing document updates its vector points in place without duplicating records.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import qdrant_client
from google import genai
from src.config import get_settings, require_api_key
from src.extractor import extract_pdf, extract_document
from src.chunker import create_chunks
from src.embedder import embed_chunks
from src.vector_store import (
    initialize_vector_store,
    ensure_collection,
    upsert_chunks,
    count_vectors,
)
from src.sparse_retriever import get_sparse_retriever
from src.metadata import create_document_metadata, DocumentMetadata



def ingest_document(
    file_path: Union[str, Path],
    client: Optional[qdrant_client.QdrantClient] = None,
    genai_client: Optional[genai.Client] = None,
    collection_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    document_type: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingests a single PDF document into the Qdrant vector store.
    Pipeline: extract -> attach metadata -> chunk -> embed -> upsert.

    Args:
        file_path: Path to the target PDF file.
        client: Optional pre-initialized QdrantClient instance.
        genai_client: Optional pre-authenticated GenAI client.
        collection_name: Optional target collection override.
        chunk_size: Optional character chunk size override.
        overlap: Optional character overlap override.
        category: Optional document category (e.g., 'academic', 'career', 'dsa').
        subject: Optional document subject (e.g., 'computer_networks', 'database').
        document_type: Optional document type (e.g., 'notes', 'resume', 'textbook').
        owner_id: Optional owner ID (defaults to 'system').
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF document not found at: {path}")

    valid_exts = (".pdf", ".md", ".txt", ".markdown")
    if path.suffix.lower() not in valid_exts:
        raise ValueError(f"Expected a supported document ({', '.join(valid_exts)}), got: {path.name}")

    settings = get_settings()
    c_size = chunk_size or settings.chunk_size
    c_overlap = overlap or settings.chunk_overlap
    target_coll = collection_name or settings.qdrant_collection_name

    # 1. Page Extraction & Metadata Setup
    doc_meta = create_document_metadata(
        file_name=path.name,
        file_path=str(path),
        category=category,
        subject=subject,
        document_type=document_type,
        owner_id=owner_id,
    )

    pages = extract_document(path, include_empty=True)
    pages_with_text = sum(1 for p in pages if p["has_text"])

    # Attach document metadata to each page so chunker propagates it
    for p in pages:
        p["document_id"] = doc_meta.document_id
        p["category"] = doc_meta.category
        p["subject"] = doc_meta.subject
        p["document_type"] = doc_meta.document_type
        p["owner_id"] = doc_meta.owner_id

    # 2. Chunking
    chunks = create_chunks(pages, chunk_size=c_size, overlap=c_overlap)
    if not chunks:
        raise ValueError(f"Document '{path.name}' produced 0 text chunks (may be empty or unscannable).")

    # 3. Dense Embeddings
    enriched_chunks = embed_chunks(chunks, client=genai_client, batch_size=20)

    # 4. Storage in Qdrant
    q_client = client or initialize_vector_store()
    try:
        ensure_collection(q_client, collection_name=target_coll, vector_size=settings.embedding_dim)
        stored_count = upsert_chunks(
            q_client,
            enriched_chunks,
            collection_name=target_coll,
            vector_size=settings.embedding_dim,
        )
    finally:
        if client is None:
            q_client.close()

    # 5. Storage in Sparse BM25 Index (Same chunks and metadata)
    sparse_retriever = get_sparse_retriever()
    sparse_retriever.index_chunks(chunks, auto_save=True)

    return {
        "file_name": path.name,
        "file_path": str(path),
        "document_id": doc_meta.document_id,
        "category": doc_meta.category,
        "subject": doc_meta.subject,
        "document_type": doc_meta.document_type,
        "pages": len(pages),
        "pages_with_text": pages_with_text,
        "chunks": len(chunks),
        "embeddings": len(enriched_chunks),
        "vectors_stored": stored_count,
        "collection": target_coll,
    }


def ingest_directory(
    directory_path: Union[str, Path],
    client: Optional[qdrant_client.QdrantClient] = None,
    genai_client: Optional[genai.Client] = None,
    collection_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    document_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Scans a directory for all PDF documents and indexes them independently
    into the shared Qdrant knowledge base.

    Args:
        directory_path: Path to the directory containing PDFs.
        client: Optional pre-initialized QdrantClient instance.
        genai_client: Optional pre-authenticated GenAI client.
        collection_name: Optional collection override.
        chunk_size: Optional chunk size override.
        overlap: Optional overlap override.

    Returns:
        Structured summary dictionary:
        {
            "total_files_found": int,
            "successful_files": int,
            "failed_files": List[Dict[str, str]],
            "total_pages": int,
            "total_chunks": int,
            "total_vectors_stored": int,
            "details": List[Dict[str, Any]],
        }

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If path is not a directory.
        ValueError: If no PDF files are found in the directory.
    """
    dir_path = Path(directory_path).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, but got file: {dir_path}")

    # Discover all PDF files (case-insensitive)
    pdf_files = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])

    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {dir_path}")

    q_client = client or initialize_vector_store()

    successful_details: List[Dict[str, Any]] = []
    failed_details: List[Dict[str, str]] = []

    total_pages = 0
    total_chunks = 0
    total_vectors = 0

    try:
        for pdf_path in pdf_files:
            try:
                stats = ingest_document(
                    file_path=pdf_path,
                    client=q_client,
                    genai_client=genai_client,
                    collection_name=collection_name,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    category=category,
                    subject=subject,
                    document_type=document_type,
                )
                successful_details.append(stats)
                total_pages += stats["pages"]
                total_chunks += stats["chunks"]
                total_vectors += stats["vectors_stored"]
            except Exception as e:
                failed_details.append({
                    "file_name": pdf_path.name,
                    "error": str(e),
                })
    finally:
        if client is None:
            q_client.close()

    return {
        "total_files_found": len(pdf_files),
        "successful_files": len(successful_details),
        "failed_files": failed_details,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "total_vectors_stored": total_vectors,
        "details": successful_details,
    }
