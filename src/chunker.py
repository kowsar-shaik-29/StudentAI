"""
Text Chunking Module.
Implements a sliding-window chunking strategy with character overlap from scratch.

Why chunking is needed:
1. LLM context limits & cost: Embedding entire books at once destroys retrieval granularity.
2. Embedding precision: A vector embedding captures the general gist of a block of text.
   Smaller, focused chunks yield more semantically distinct embeddings.
3. Overlap: A small overlap (e.g. 100-200 characters) prevents crucial sentences from being
   severed in half across chunk boundaries.
"""

import uuid
from typing import List, Dict, Any


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Splits page-level documents into smaller overlapping text chunks while
    preserving source metadata.

    Args:
        pages: List of page dictionaries produced by extractor.py
        chunk_size: Maximum character length of each chunk
        chunk_overlap: Overlapping characters between consecutive chunks

    Returns:
        List of chunk dictionaries with unique IDs and rich metadata.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    chunks: List[Dict[str, Any]] = []

    for page_doc in pages:
        text = page_doc["text"]
        page_num = page_doc["page_number"]
        file_name = page_doc["file_name"]

        text_length = len(text)
        start = 0
        chunk_index = 0

        # When page text is shorter than chunk size, store as a single chunk
        if text_length <= chunk_size:
            chunk_id = f"{file_name}_p{page_num}_c0_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": text_length,
                "page_number": page_num,
                "total_pages": page_doc["total_pages"],
                "file_name": file_name,
                "file_path": page_doc["file_path"],
            })
            continue

        # Sliding window over the text
        step = chunk_size - chunk_overlap
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            # Try to snap to the nearest space or newline if not at end of text
            if end < text_length:
                last_space = chunk_text.rfind(" ")
                last_newline = chunk_text.rfind("\n")
                split_point = max(last_space, last_newline)
                # Only snap if the break point is not too close to the beginning
                if split_point > (chunk_size // 2):
                    end = start + split_point + 1
                    chunk_text = text[start:end]

            chunk_id = f"{file_name}_p{page_num}_c{chunk_index}_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text.strip(),
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
                "page_number": page_num,
                "total_pages": page_doc["total_pages"],
                "file_name": file_name,
                "file_path": page_doc["file_path"],
            })

            chunk_index += 1
            start += step

    return chunks
