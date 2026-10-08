"""
Sparse Retriever Module for StudentAI.
Implements Okapi BM25 lexical ranking for document chunks with local persistence.

Core Principles:
1. Lexical Relevance:
   Scores query terms against document terms using BM25 with term frequency saturation (k1=1.5)
   and document length normalization (b=0.75).
2. Shared Knowledge Base:
   Indexes the exact same chunk texts and metadata produced during ingestion, without duplicating pipelines.
3. Multi-Document Support:
   Supports indexing across multiple files with deterministic chunk IDs.
4. Local Persistence:
   Persists indexed chunks and vocabulary statistics to data/sparse_index/bm25_index.json.
"""

import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
from collections import Counter
from src.config import get_settings


def tokenize(text: str) -> List[str]:
    """
    Extracts lowercase alphanumeric tokens from text.
    Handles alphanumeric codes, technical acronyms (e.g. ACID, SQL, IP), and punctuation removal.
    """
    if not text:
        return []
    # Match alphanumeric words (including underscores or single digits)
    return [t.lower() for t in re.findall(r"\b[a-zA-Z0-9]+\b", text)]


class SparseRetriever:
    """
    BM25 Sparse Lexical Retriever with disk persistence.
    """

    def __init__(
        self,
        index_dir: Optional[Union[str, Path]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initializes BM25 Sparse Retriever.

        Args:
            index_dir: Directory where the sparse index file is saved/loaded.
            k1: BM25 term frequency saturation parameter (default 1.5).
            b: BM25 document length normalization parameter (default 0.75).
        """
        settings = get_settings()
        self.index_dir = Path(index_dir) if index_dir else settings.sparse_index_dir
        self.index_file = self.index_dir / "bm25_index.json"
        self.k1 = k1
        self.b = b

        # In-memory index state
        # chunk_id -> chunk dictionary
        self.documents: Dict[str, Dict[str, Any]] = {}
        # chunk_id -> list of tokens
        self.doc_tokens: Dict[str, List[str]] = {}
        # chunk_id -> term count Counter
        self.doc_term_freqs: Dict[str, Counter] = {}
        # chunk_id -> length (number of tokens)
        self.doc_lengths: Dict[str, int] = {}
        # term -> count of documents containing term (Document Frequency)
        self.doc_frequencies: Dict[str, int] = {}
        self.total_docs: int = 0
        self.avg_doc_length: float = 0.0

        # Load persisted index if it exists
        self.load()

    def index_chunks(self, chunks: List[Dict[str, Any]], auto_save: bool = True) -> int:
        """
        Indexes a list of chunk dictionaries into the sparse index.
        Updates existing chunks in-place if chunk_id matches (idempotent).

        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'text', etc.
            auto_save: If True, immediately serializes updated index to disk.

        Returns:
            Total count of indexed documents.
        """
        if not chunks:
            return self.total_docs

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue

            # Clean payload (exclude vectors)
            clean_chunk = {
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
                "page_number": chunk.get("page_number", 0),
                "total_pages": chunk.get("total_pages", 0),
                "file_name": chunk.get("file_name", "unknown"),
                "file_path": chunk.get("file_path", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "char_count": chunk.get("char_count", len(chunk.get("text", ""))),
                "document_id": chunk.get("document_id", ""),
                "category": chunk.get("category", "other"),
                "subject": chunk.get("subject", "general"),
                "document_type": chunk.get("document_type", "other"),
            }

            tokens = tokenize(clean_chunk["text"])
            self.documents[chunk_id] = clean_chunk
            self.doc_tokens[chunk_id] = tokens
            self.doc_term_freqs[chunk_id] = Counter(tokens)
            self.doc_lengths[chunk_id] = len(tokens)

        # Recompute corpus-wide statistics
        self._recompute_statistics()

        if auto_save:
            self.save()

        return self.total_docs

    def _recompute_statistics(self) -> None:
        """Recomputes DF, N, and avgdl across all stored documents."""
        self.total_docs = len(self.documents)
        df_counter: Counter = Counter()

        for tokens in self.doc_tokens.values():
            unique_terms = set(tokens)
            df_counter.update(unique_terms)

        self.doc_frequencies = dict(df_counter)

        if self.total_docs > 0:
            total_tokens = sum(self.doc_lengths.values())
            self.avg_doc_length = total_tokens / self.total_docs
        else:
            self.avg_doc_length = 0.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        subject: Optional[str] = None,
        document_type: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 search over indexed document chunks with optional metadata filtering.

        Args:
            query: Natural language query string (must be non-empty)
            top_k: Number of highest scoring results to return (must be > 0)
            category: Optional category filter (e.g. 'academic', 'career')
            subject: Optional subject filter (e.g. 'database', 'computer_networks')
            document_type: Optional document type filter (e.g. 'notes', 'resume')
            document_id: Optional exact document ID filter
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

        if self.total_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Normalize filter criteria
        cat_filter = category.strip().lower() if category else None
        subj_filter = subject.strip().lower().replace(" ", "_") if subject else None
        dtype_filter = document_type.strip().lower() if document_type else None
        doc_id_filter = document_id.strip() if document_id else None

        scores: Dict[str, float] = {}

        # Precompute IDF for each unique query token
        # Standard Lucene/BM25 IDF formula: ln(1 + (N - DF + 0.5) / (DF + 0.5))
        idfs: Dict[str, float] = {}
        for token in set(query_tokens):
            df = self.doc_frequencies.get(token, 0)
            idf = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
            # Floor negative IDF to 0.0 (occurs only if a term is in > ~90% of documents)
            idfs[token] = max(0.0, idf)

        for chunk_id, term_freqs in self.doc_term_freqs.items():
            doc_meta = self.documents[chunk_id]

            # Apply metadata filters
            if cat_filter and doc_meta.get("category", "other").lower() != cat_filter:
                continue
            if subj_filter and doc_meta.get("subject", "general").lower() != subj_filter:
                continue
            if dtype_filter and doc_meta.get("document_type", "other").lower() != dtype_filter:
                continue
            if doc_id_filter and doc_meta.get("document_id") != doc_id_filter:
                continue

            doc_len = self.doc_lengths.get(chunk_id, 0)
            score = 0.0

            for q_term in query_tokens:
                tf = term_freqs.get(q_term, 0)
                if tf > 0:
                    idf = idfs.get(q_term, 0.0)
                    # BM25 TF saturation formula with document length normalization
                    norm_denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0)))
                    term_score = idf * (tf * (self.k1 + 1.0)) / norm_denom
                    score += term_score

            if score > 0.0:
                scores[chunk_id] = score

        # Rank by score descending, breaking ties by chunk_id
        sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: (scores[cid], cid), reverse=True)

        results: List[Dict[str, Any]] = []
        for cid in sorted_chunk_ids[:top_k]:
            doc_meta = self.documents[cid]
            results.append({
                "sparse_score": float(scores[cid]),
                "chunk_id": doc_meta["chunk_id"],
                "text": doc_meta["text"],
                "page_number": doc_meta["page_number"],
                "total_pages": doc_meta["total_pages"],
                "file_name": doc_meta["file_name"],
                "file_path": doc_meta.get("file_path", ""),
                "chunk_index": doc_meta["chunk_index"],
                "char_count": doc_meta["char_count"],
                "document_id": doc_meta.get("document_id", ""),
                "category": doc_meta.get("category", "other"),
                "subject": doc_meta.get("subject", "general"),
                "document_type": doc_meta.get("document_type", "other"),
            })

        return results

    def save(self) -> None:
        """Persists indexed documents and metadata to disk as JSON."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0",
            "k1": self.k1,
            "b": self.b,
            "total_docs": self.total_docs,
            "documents": list(self.documents.values()),
        }
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load(self) -> bool:
        """Loads index from disk if present. Returns True if loaded successfully."""
        if not self.index_file.is_file():
            return False

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            docs = payload.get("documents", [])
            self.k1 = payload.get("k1", self.k1)
            self.b = payload.get("b", self.b)
            self.index_chunks(docs, auto_save=False)
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clears index in memory and on disk."""
        self.documents.clear()
        self.doc_tokens.clear()
        self.doc_term_freqs.clear()
        self.doc_lengths.clear()
        self.doc_frequencies.clear()
        self.total_docs = 0
        self.avg_doc_length = 0.0
        if self.index_file.exists():
            try:
                self.index_file.unlink()
            except OSError:
                pass


# Global singleton instance helper
_sparse_instance: Optional[SparseRetriever] = None


def get_sparse_retriever() -> SparseRetriever:
    """Returns a shared SparseRetriever instance."""
    global _sparse_instance
    if _sparse_instance is None:
        _sparse_instance = SparseRetriever()
    return _sparse_instance
