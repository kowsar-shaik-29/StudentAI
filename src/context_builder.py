"""
Context Builder Module.
Transforms retrieved Qdrant search results into a clean, structured context string for LLM generation.

Core Concept:
In RAG, the context builder acts as the bridge between retrieval and generation.
It formats each retrieved chunk with clear document and page attribution so the LLM
can cite specific pages while answering.
"""

from typing import List, Dict, Any


def build_context(results: List[Dict[str, Any]]) -> str:
    """
    Constructs an augmented context string from retrieved search results.

    Args:
        results: List of retrieved chunk dictionaries from src.retriever.search()
                 containing 'file_name', 'page_number', 'chunk_id', and 'text'.

    Returns:
        Structured string containing all retrieved passages with clear source demarcations.
        Returns an empty string if results list is empty.
    """
    if not results:
        return ""

    passages = []
    for res in results:
        file_name = res.get("file_name", "unknown")
        page_number = res.get("page_number", "?")
        chunk_id = res.get("chunk_id", "unknown")
        text = res.get("text", "").strip()

        header = f"[Document: {file_name} | Page: {page_number} | Chunk: {chunk_id}]"
        passage = f"{header}\n{text}"
        passages.append(passage)

    return "\n\n---\n\n".join(passages)


def extract_page_citations(text: str) -> List[int]:
    """
    Extracts page numbers cited in bracketed format like [Page 1], [Page 2, 3], or [Page 1][Page 2].

    Returns:
        Sorted unique list of cited integer page numbers.
    """
    import re
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

