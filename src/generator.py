"""
Generation Module.
Synthesizes retrieved context and student question into a grounded answer using Gemini LLM.

Core Concept:
RAG prevents hallucination by constraining the LLM to answer using only the retrieved
context passages, and explicitly citing the source document and page number.
"""

from typing import List, Dict, Any, Optional
from google import genai
from src.config import get_settings
from src.embedder import get_genai_client


def format_context_prompt(context_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into clear, cited context passages for the LLM."""
    formatted_passages = []
    for idx, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("file_name", "Unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "").strip()
        passage = f"[Passage {idx} | Source: {source}, Page: {page}]\n{text}"
        formatted_passages.append(passage)

    return "\n\n---\n\n".join(formatted_passages)


def generate_rag_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prompts the Gemini generation model with the student question augmented by retrieved context.

    Args:
        query: Student question
        context_chunks: Retrieved relevant chunks from retriever.py
        client: Optional pre-authenticated GenAI client
        model: Optional model name override

    Returns:
        Dictionary containing answer, cited passages, and prompt details.
    """
    settings = get_settings()
    active_client = client or get_genai_client()
    active_model = model or settings.generation_model

    context_str = format_context_prompt(context_chunks)

    prompt = f"""You are StudentAI, an academic and technical learning tutor.
Answer the student's question accurately using ONLY the provided context passages below.
For every claim or key point in your explanation, cite the source document and page number (e.g. [Filename, Page X]).
If the provided context does not contain sufficient information to answer the question, clearly state:
"I cannot find the answer to this question in the provided materials." Do NOT invent facts.

Context Passages:
{context_str}

Student Question:
{query}

Academic Answer with Citations:
"""

    response = active_client.models.generate_content(
        model=active_model,
        contents=prompt,
    )

    return {
        "answer": response.text,
        "model": active_model,
        "citations": [
            {
                "file_name": c.get("file_name"),
                "page_number": c.get("page_number"),
                "score": c.get("score"),
            }
            for c in context_chunks
        ],
    }
