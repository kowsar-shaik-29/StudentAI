"""
PDF Text Extraction Module.
Uses PyMuPDF (pymupdf) to extract text while preserving page-level metadata.

Why page-level metadata matters in RAG:
In academic and textbook RAG, knowing which exact page a chunk originated from
is essential for source attribution, verifiable citations, and student review.
"""

from pathlib import Path
from typing import List, Dict, Any
import pymupdf


def extract_pdf_pages(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF document.

    Args:
        pdf_path: Path to the target PDF file.

    Returns:
        List of dictionaries containing extracted page content and metadata:
        [
            {
                "text": str,          # Cleaned text of the page
                "page_number": int,   # 1-indexed page number
                "total_pages": int,   # Total pages in document
                "file_name": str,     # Source filename
                "file_path": str      # Source path
            }, ...
        ]
    """
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    extracted_pages = []

    # Open document using PyMuPDF
    doc = pymupdf.open(str(path))
    total_pages = len(doc)

    try:
        for page_idx in range(total_pages):
            page = doc[page_idx]
            # Extract plain text from page
            page_text = page.get_text()

            # Normalizing whitespace while preserving paragraph breaks
            cleaned_text = page_text.strip()

            if cleaned_text:
                extracted_pages.append({
                    "text": cleaned_text,
                    "page_number": page_idx + 1,  # Human-readable 1-based indexing
                    "total_pages": total_pages,
                    "file_name": path.name,
                    "file_path": str(path),
                })
    finally:
        doc.close()

    return extracted_pages
