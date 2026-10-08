"""
Embedding Generation Module.
Directly interfaces with Google GenAI SDK (google-genai) to generate dense vector embeddings.

Core Concept:
A text embedding maps natural language strings into high-dimensional numerical vectors (points in space).
Texts with similar semantic meaning are situated closer together according to geometric distance (e.g. Cosine Similarity).
"""

from typing import List, Optional
from google import genai
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
) -> List[float]:
    """
    Generates a dense vector embedding for a single text string using Google GenAI SDK.

    Args:
        text: Input string to embed.
        client: Optional pre-authenticated GenAI client.
        model: Optional model name override.
        output_dim: Optional embedding dimension override.

    Returns:
        List of floats representing the embedding vector.
    """
    settings = get_settings()
    active_client = client or get_genai_client()
    active_model = model or settings.embedding_model
    dim = output_dim or settings.embedding_dim

    config = {}
    if dim:
        config["output_dimensionality"] = dim

    response = active_client.models.embed_content(
        model=active_model,
        contents=text,
        config=config if config else None,
    )

    # In google-genai SDK, response.embedding contains values: list[float]
    return list(response.embedding.values)


def embed_batch(
    texts: List[str],
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    output_dim: Optional[int] = None,
) -> List[List[float]]:
    """
    Generates embeddings for a batch of text strings.

    Args:
        texts: List of strings to embed.
        client: Optional pre-authenticated GenAI client.
        model: Optional model name override.
        output_dim: Optional embedding dimension override.

    Returns:
        List of embedding vectors.
    """
    settings = get_settings()
    active_client = client or get_genai_client()
    active_model = model or settings.embedding_model
    dim = output_dim or settings.embedding_dim

    config = {}
    if dim:
        config["output_dimensionality"] = dim

    response = active_client.models.embed_content(
        model=active_model,
        contents=texts,
        config=config if config else None,
    )

    # When contents is a list, response.embeddings is a list of ContentEmbedding objects
    if hasattr(response, "embeddings") and response.embeddings:
        return [list(emb.values) for emb in response.embeddings]
    elif hasattr(response, "embedding") and response.embedding:
        return [list(response.embedding.values)]
    else:
        raise ValueError("Unexpected embedding response format from Google GenAI SDK")
