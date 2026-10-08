"""
PDF Text Extraction Module.
Uses PyMuPDF (pymupdf) to extract text page-by-page while preserving metadata.

Why page-level metadata matters in RAG:
In student notes, textbooks, and documentation, knowing the exact source page
enables verifiable citations, direct reference lookups, and precise context attribution.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import pymupdf


def extract_pdf(
    file_path: Union[str, Path],
    include_empty: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF document and attaches metadata.

    Args:
        file_path: Path to the target PDF file.
        include_empty: Whether to include pages that contain no extractable text.
                      Defaults to True so page indices remain contiguous.

    Returns:
        List of dictionaries with the following structure per page:
        [
            {
                "page_number": int,   # 1-indexed human-readable page number
                "total_pages": int,   # Total page count of document
                "file_name": str,     # Basename of the file (e.g. "notes.pdf")
                "file_path": str,     # Resolved absolute path string
                "text": str,          # Extracted page text (empty string if blank)
                "has_text": bool,     # True if text contains non-whitespace content
                "char_count": int     # Number of characters extracted
            },
            ...
        ]

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path points to a directory instead of a file.
        ValueError: If the file is 0 bytes or not a valid PDF.
        PermissionError: If the PDF is password-protected or encrypted.
        RuntimeError: If PyMuPDF encounters a parsing or corruption error.
    """
    path = Path(file_path).resolve()

    # 1. Path existence and type validation
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    if path.is_dir():
        raise IsADirectoryError(f"Expected a PDF file, but path points to a directory: {path}")

    # 2. Check for empty (0 bytes) file
    if path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty (0 bytes): {path.name}")

    # 3. Open PDF document with PyMuPDF
    try:
        doc = pymupdf.open(str(path))
    except pymupdf.FileDataError as e:
        raise RuntimeError(f"Corrupted or invalid PDF format in '{path.name}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{path.name}': {e}") from e

    try:
        # 4. Check for encryption / password requirement
        if doc.is_encrypted and not doc.authenticate(""):
            raise PermissionError(
                f"PDF '{path.name}' is password-protected or encrypted and cannot be read."
            )

        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError(f"PDF '{path.name}' has 0 pages.")

        extracted_pages: List[Dict[str, Any]] = []

        # 5. Extract text page-by-page
        for page_idx in range(total_pages):
            page = doc[page_idx]
            raw_text = page.get_text()

            # Normalize whitespace
            cleaned_text = raw_text.strip()
            has_text = len(cleaned_text) > 0

            page_data = {
                "page_number": page_idx + 1,
                "total_pages": total_pages,
                "file_name": path.name,
                "file_path": str(path),
                "text": cleaned_text,
                "has_text": has_text,
                "char_count": len(cleaned_text),
            }

            if has_text or include_empty:
                extracted_pages.append(page_data)

        return extracted_pages

    finally:
        # Always ensure the document handle is cleanly closed
        doc.close()


def extract_pdf_pages(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Convenience alias returning only pages that contain text.
    Maintains backward compatibility with Phase 1 modules.
    """
    return extract_pdf(file_path, include_empty=False)
