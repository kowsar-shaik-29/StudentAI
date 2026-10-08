"""
Embedding Generation Module.
Directly interfaces with Google GenAI SDK (google-genai) to generate dense vector embeddings.

Core Concept:
A text embedding maps natural language strings into high-dimensional numerical vectors.
Texts with similar semantic meaning are situated closer together according to geometric distance (Cosine Similarity).

Model Details:
- Model: `gemini-embedding-2` (Official GA embedding model).
- Dimensionality: Supports Matryoshka Representation Learning (MRL), allowing flexible truncation
  to 768 dimensions for optimal balance between storage, search speed, and accuracy.
"""

from typing import List, Optional
from google import genai
from google.genai import types
from src.config import get_settings, require_api_key


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

    response = active_client.models.embed_content(
        model=active_model,
        contents=text,
        config=config,
    )

    if response.embeddings and len(response.embeddings) > 0:
        return list(response.embeddings[0].values)
    else:
        raise ValueError("Received empty embeddings response from Google GenAI SDK")


def embed_batch(
    texts: List[str],
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    output_dim: Optional[int] = None,
    task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
) -> List[List[float]]:
    """
    Generates embeddings for a batch of text strings.

    Args:
        texts: List of strings to embed.
        client: Optional pre-authenticated GenAI client.
        model: Optional model name override.
        output_dim: Optional embedding dimension override.
        task_type: Optional task type.

    Returns:
        List of embedding vectors.
    """
    settings = get_settings()
    active_client = client or get_genai_client()
    active_model = model or settings.embedding_model
    dim = output_dim or settings.embedding_dim

    config = types.EmbedContentConfig(
        output_dimensionality=dim,
        task_type=task_type,
    )

    response = active_client.models.embed_content(
        model=active_model,
        contents=texts,
        config=config,
    )

    if response.embeddings:
        return [list(emb.values) for emb in response.embeddings]
    else:
        raise ValueError("Received empty embeddings response from Google GenAI SDK")
