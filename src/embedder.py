"""
Embedding Generation Module.
Directly interfaces with Google GenAI SDK (google-genai) to generate dense vector embeddings.

Core Concept:
A text embedding maps natural language strings into high-dimensional numerical vectors.
Texts with similar semantic meaning are situated closer together according to geometric distance (Cosine Similarity).

Model Details:
- Model: `gemini-embedding-2`
- Dimensionality: Supports Matryoshka Representation Learning (MRL), allowing flexible truncation
  to 768 dimensions for optimal balance between storage, search speed, and accuracy.
- Task Types:
  - `RETRIEVAL_DOCUMENT`: Used when embedding documents/chunks to be stored in the vector index.
  - `RETRIEVAL_QUERY`: Used when embedding the student's question for similarity search.
"""

import math
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
from src.config import get_settings, require_api_key


def _sanitize_error(error_msg: str) -> str:
    """Removes any accidental API key patterns from error messages."""
    settings = get_settings()
    sanitized = error_msg
    if settings.gemini_api_key and len(settings.gemini_api_key) > 6:
        sanitized = sanitized.replace(settings.gemini_api_key, "[REDACTED_API_KEY]")
    return sanitized


def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """Creates and returns an authenticated Google GenAI client instance."""
    key = api_key or require_api_key()
    return genai.Client(api_key=key)


def embed_text(
    text: str,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    output_dim: Optional[int] = None,
    task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
) -> List[float]:
    """
    Generates a dense vector embedding for a single text string using Google GenAI SDK.

    Args:
        text: Input string to embed.
        client: Optional pre-authenticated GenAI client.
        model: Optional model name override (defaults to gemini-embedding-2).
        output_dim: Optional embedding dimension override (defaults to 768 via MRL).
        task_type: Optional task type (e.g. RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY).

    Returns:
        List of floats representing the embedding vector.
    """
    settings = get_settings()
    active_client = client or get_genai_client()
    active_model = model or settings.embedding_model
    dim = output_dim or settings.embedding_dim

    config = types.EmbedContentConfig(
        output_dimensionality=dim,
        task_type=task_type,
    )

    try:
        response = active_client.models.embed_content(
            model=active_model,
            contents=text,
            config=config,
        )
    except APIError as e:
        safe_msg = _sanitize_error(str(e))
        raise RuntimeError(f"Google GenAI API error while generating embedding: {safe_msg}") from e
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise RuntimeError(f"Unexpected error while generating embedding: {safe_msg}") from e

    if not response.embeddings or len(response.embeddings) == 0:
        raise ValueError("Received empty embeddings response from Google GenAI SDK")

    vector = list(response.embeddings[0].values)

    # Validate vector properties
    if len(vector) != dim:
        raise ValueError(f"Expected embedding dimension {dim}, but received {len(vector)}")

    if not all(math.isfinite(v) for v in vector):
        raise ValueError("Embedding vector contains non-finite values (NaN or Inf)")

    return vector


def _embed_single_batch(
    texts: List[str],
    client: genai.Client,
    model: str,
    dim: int,
    task_type: Optional[str],
) -> List[List[float]]:
    """Helper that executes a single API batch call."""
    config = types.EmbedContentConfig(
        output_dimensionality=dim,
        task_type=task_type,
    )

    # Wrap each text string in its own Content object to ensure separate embeddings
    content_items = [
        types.Content(parts=[types.Part(text=t)])
        for t in texts
    ]

    try:
        response = client.models.embed_content(
            model=model,
            contents=content_items,
            config=config,
        )
    except APIError as e:
        safe_msg = _sanitize_error(str(e))
        raise RuntimeError(f"Google GenAI API error while generating batch embeddings: {safe_msg}") from e
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise RuntimeError(f"Unexpected error while generating batch embeddings: {safe_msg}") from e

    if not response.embeddings:
        raise ValueError("Received empty embeddings response from Google GenAI SDK")

    if len(response.embeddings) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} embeddings, but received {len(response.embeddings)}"
        )

    vectors: List[List[float]] = []
    for emb in response.embeddings:
        vec = list(emb.values)
        if len(vec) != dim:
            raise ValueError(f"Expected embedding dimension {dim}, but received {len(vec)}")
        if not all(math.isfinite(v) for v in vec):
            raise ValueError("Embedding vector contains non-finite values (NaN or Inf)")
        vectors.append(vec)

    return vectors


def embed_batch(
    texts: List[str],
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    output_dim: Optional[int] = None,
    task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
    batch_size: int = 20,
) -> List[List[float]]:
    """
    Generates embeddings for a batch of text strings, chunking requests into
    sub-batches if necessary to respect API limits.

    Args:
        texts: List of strings to embed.
        client: Optional pre-authenticated GenAI client.
        model: Optional model name override.
        output_dim: Optional embedding dimension override.
        task_type: Optional task type (defaults to RETRIEVAL_DOCUMENT).
        batch_size: Max texts per individual API request (defaults to 20).

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    settings = get_settings()
    active_client = client or get_genai_client()
    active_model = model or settings.embedding_model
    dim = output_dim or settings.embedding_dim

    all_vectors: List[List[float]] = []

    # Process in manageable sub-batches
    for i in range(0, len(texts), batch_size):
        sub_batch = texts[i : i + batch_size]
        sub_vectors = _embed_single_batch(
            texts=sub_batch,
            client=active_client,
            model=active_model,
            dim=dim,
            task_type=task_type,
        )
        all_vectors.extend(sub_vectors)

    return all_vectors


def embed_documents(
    texts: List[str],
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    output_dim: Optional[int] = None,
    batch_size: int = 20,
) -> List[List[float]]:
    """
    Generates dense embeddings for documents/chunks to be indexed.
    Enforces task_type='RETRIEVAL_DOCUMENT'.
    """
    return embed_batch(
        texts=texts,
        client=client,
        model=model,
        output_dim=output_dim,
        task_type="RETRIEVAL_DOCUMENT",
        batch_size=batch_size,
    )


def embed_chunks(
    chunks: List[Dict[str, Any]],
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    output_dim: Optional[int] = None,
    batch_size: int = 20,
) -> List[Dict[str, Any]]:
    """
    Generates embeddings for a list of document chunks and attaches each embedding
    to a copy of the corresponding chunk dictionary while preserving all metadata.

    Args:
        chunks: List of chunk dictionaries produced by chunker.py
        client: Optional pre-authenticated GenAI client
        model: Optional model override
        output_dim: Optional dimension override
        batch_size: Max chunks per API request

    Returns:
        List of chunk dictionaries with all original metadata plus the 'embedding' key.
    """
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(
        texts=texts,
        client=client,
        model=model,
        output_dim=output_dim,
        batch_size=batch_size,
    )

    enriched_chunks: List[Dict[str, Any]] = []
    for chunk, emb in zip(chunks, embeddings):
        item = dict(chunk)  # Preserve original dictionary and copy
        item["embedding"] = emb
        enriched_chunks.append(item)

    return enriched_chunks


def embed_query(
    query: str,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    output_dim: Optional[int] = None,
) -> List[float]:
    """
    Generates dense embedding for a user search query.
    Enforces task_type='RETRIEVAL_QUERY'.
    """
    return embed_text(
        text=query,
        client=client,
        model=model,
        output_dim=output_dim,
        task_type="RETRIEVAL_QUERY",
    )
