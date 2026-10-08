"""
Document Chunking Module.
Splits extracted document pages into sliding-window text chunks while preserving page boundaries and metadata.

Core Concepts:
1. Why Chunking:
   - High-dimensional embeddings capture focused semantic meaning better over paragraphs/sections
     than over entire multi-page documents.
   - Retrieval returns precise, relevant snippets rather than overwhelming the LLM context.
2. Why Overlap:
   - Splitting at fixed character limits can split a critical sentence in half.
   - Overlap ensures that context across boundaries is present in adjacent chunks.
3. Why Preserve Page Boundaries:
   - Academic RAG requires verifiable, single-page attribution (e.g. Page 3, not Page 3-4 mixed).
"""

from typing import List, Dict, Any


def create_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Splits page-level documents into text chunks while preserving page boundaries.

    Args:
        pages: List of page dictionaries produced by extract_pdf().
        chunk_size: Maximum character length of each chunk (must be > 0).
        overlap: Character overlap between consecutive chunks on the same page (0 <= overlap < chunk_size).

    Returns:
        List of chunk dictionaries with deterministic chunk IDs and complete metadata.

    Raises:
        ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    # 1. Parameter Validation
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")

    if overlap < 0:
        raise ValueError(f"overlap must be greater than or equal to 0, got {overlap}")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )

    all_chunks: List[Dict[str, Any]] = []
    step = chunk_size - overlap

    for page_doc in pages:
        # Extract text without mutating the original dictionary
        raw_text = page_doc.get("text", "")
        if not raw_text or not raw_text.strip():
            # Skip empty or whitespace-only pages
            continue

        text = raw_text.strip()
        text_len = len(text)
        page_num = page_doc.get("page_number", 1)
        total_pages = page_doc.get("total_pages", 1)
        file_name = page_doc.get("file_name", "unknown")
        file_path = page_doc.get("file_path", "")

        # Case A: Page text fits entirely within one chunk
        if text_len <= chunk_size:
            chunk_id = f"{file_name}_p{page_num}_c0"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "page_number": page_num,
                "total_pages": total_pages,
                "file_name": file_name,
                "file_path": file_path,
                "chunk_index": 0,
                "char_count": text_len,
            })
            continue

        # Case B: Page text exceeds chunk_size -> apply sliding window
        start = 0
        chunk_idx = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]

            chunk_id = f"{file_name}_p{page_num}_c{chunk_idx}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_number": page_num,
                "total_pages": total_pages,
                "file_name": file_name,
                "file_path": file_path,
                "chunk_index": chunk_idx,
                "char_count": len(chunk_text),
            })

            chunk_idx += 1

            # If we've reached the end of the text, stop
            if end >= text_len:
                break

            start += step

    return all_chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Backward-compatibility alias for create_chunks.
    """
    return create_chunks(pages, chunk_size=chunk_size, overlap=chunk_overlap)
