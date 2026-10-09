"""
Context Builder Module.
Transforms retrieved Qdrant and BM25 search results into a clean, structured context string for LLM generation.

Core Concepts:
1. Verifiable Page Attribution:
   Formats each passage with clear document and page tags ([Document: X | Page: Y | Chunk: Z])
   so the LLM can cite exact pages while answering.
2. Deduplication:
   Removes duplicate chunks (same chunk_id or identical text content).
3. Context Budget & Overlap Management:
   Enforces a configurable character budget (max_chars) while prioritizing top-ranked evidence.
4. Multi-Document Support:
   Preserves evidence across multiple distinct documents without discarding relevant cross-document passages.
"""

from typing import List, Dict, Any, Optional, Set
import re


def build_context(
    results: List[Dict[str, Any]],
    max_chars: Optional[int] = None,
    deduplicate: bool = True,
) -> str:
    """
    Constructs an augmented context string from retrieved search results.

    Args:
        results: List of retrieved chunk dictionaries from hybrid_search() or search()
                 containing 'file_name', 'page_number', 'chunk_id', and 'text'.
        max_chars: Optional maximum total characters in the output context string.
        deduplicate: Whether to filter out duplicate chunks by ID or exact text.

    Returns:
        Structured string containing all retrieved passages with clear source demarcations.
        Returns an empty string if results list is empty.
    """
    if not results:
        return ""

    passages: List[str] = []
    seen_chunk_ids: Set[str] = set()
    seen_texts: Set[str] = set()
    current_chars = 0

    for res in results:
        cid = res.get("chunk_id", "unknown")
        raw_text = res.get("text", "").strip()

        if not raw_text:
            continue

        if deduplicate:
            if cid in seen_chunk_ids:
                continue
            # Also check text hash / normalized string to prevent identical text under different IDs
            norm_text = re.sub(r"\s+", " ", raw_text)
            if norm_text in seen_texts:
                continue
            seen_chunk_ids.add(cid)
            seen_texts.add(norm_text)

        file_name = res.get("file_name", "unknown")
        page_number = res.get("page_number", "?")

        header = f"[Document: {file_name} | Page: {page_number} | Chunk: {cid}]"
        passage = f"{header}\n{raw_text}"

        # Check character budget if specified
        added_len = len(passage) + (len("\n\n---\n\n") if passages else 0)
        if max_chars is not None and (current_chars + added_len > max_chars):
            # If we already have at least one passage, stop adding
            if passages:
                break
            # If first passage itself exceeds budget, truncate passage
            passage = passage[:max_chars]

        passages.append(passage)
        current_chars += len(passage)

    return "\n\n---\n\n".join(passages)


def extract_page_citations(text: str) -> List[int]:
    """
    Extracts page numbers cited in bracketed format like [Page 1], [Page 2, 3], or [Page 1][Page 2].

    Returns:
        Sorted unique list of cited integer page numbers.
    """
    if not text:
        return []

    # Matches patterns like [Page 1], [Page 12], [Page 1, 2]
    matches = re.findall(r"\[Page\s+([0-9,\s]+)\]", text, re.IGNORECASE)
    pages = set()
    for m in matches:
        for num in re.findall(r"\d+", m):
            try:
                pages.add(int(num))
            except ValueError:
                pass
    return sorted(list(pages))
