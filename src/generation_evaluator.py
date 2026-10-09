"""
Generation Evaluator Module (Step 11).
Implements deterministic generation checks:
1. Citation exists (e.g. [Page X])
2. Citation page belongs to expected page(s)
3. Expected document appears in sources
4. Answer is non-empty
5. Refusal correctness for unanswerable/out-of-domain questions.
"""

from typing import List, Dict, Any, Optional
import qdrant_client
from google import genai
from src.rag import answer_question
from src.context_builder import extract_page_citations
from src.sparse_retriever import SparseRetriever
from src.citation_verifier import verify_citations


STANDARD_REFUSAL_PHRASE = "I couldn't find the answer in the provided documents."


def check_generation_response(
    eval_item: Dict[str, Any],
    rag_response: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Performs deterministic evaluation checks on a RAG generation response.

    Returns:
        Structured check results:
        {
            "id": str,
            "question": str,
            "answerable": bool,
            "non_empty_answer": bool,
            "has_citation": bool,
            "citation_matches_expected_page": bool,
            "expected_doc_in_sources": bool,
            "refusal_correct": bool,
            "passed": bool,
            "answer": str,
            "cited_pages": List[int],
            "sources": List[str],
        }
    """
    item_id = eval_item["id"]
    question = eval_item["question"]
    is_answerable = eval_item.get("answerable", True)
    expected_docs = set(eval_item.get("expected_documents", []))
    expected_pages = set(eval_item.get("expected_pages", []))

    answer_text = rag_response.get("answer", "").strip()
    sources = rag_response.get("sources", [])
    source_docs = set(s.get("file_name") for s in sources)

    # Extract cited pages from answer
    cited_pages = extract_page_citations(answer_text)

    non_empty = len(answer_text) > 0
    refusal_correct = False
    has_citation = len(cited_pages) > 0
    citation_page_matches = False
    doc_in_sources = False

    if not is_answerable:
        # For out-of-domain / unanswerable questions:
        # Must contain standard refusal phrase or refusal intent, and not hallucinate facts
        refusal_correct = (STANDARD_REFUSAL_PHRASE.lower() in answer_text.lower())
        passed = non_empty and refusal_correct
    else:
        # For answerable questions:
        # 1. Non-empty
        # 2. Contains citations
        # 3. At least one cited page is in expected_pages
        # 4. Expected document is present in retrieved sources
        citation_page_matches = any(p in expected_pages for p in cited_pages)
        doc_in_sources = any(d in source_docs for d in expected_docs)
        passed = non_empty and has_citation and citation_page_matches and doc_in_sources

    # Full citation verification check
    citation_report = verify_citations(
        answer=answer_text,
        sources=sources,
        expected_documents=list(expected_docs) if expected_docs else None,
    )

    return {
        "id": item_id,
        "question": question,
        "answerable": is_answerable,
        "non_empty_answer": non_empty,
        "has_citation": has_citation,
        "citation_matches_expected_page": citation_page_matches,
        "expected_doc_in_sources": doc_in_sources,
        "refusal_correct": refusal_correct,
        "citation_report": citation_report,
        "passed": passed,
        "answer": answer_text,
        "cited_pages": cited_pages,
        "sources": list(source_docs),
    }


def evaluate_generation_pipeline(
    eval_dataset: List[Dict[str, Any]],
    client: Optional[qdrant_client.QdrantClient] = None,
    genai_client: Optional[genai.Client] = None,
    hybrid: bool = True,
    sparse_retriever: Optional[SparseRetriever] = None,
) -> Dict[str, Any]:
    """
    Evaluates end-to-end RAG generation for all dataset items.
    """
    detailed_checks: List[Dict[str, Any]] = []

    import time
    for item in eval_dataset:
        response = answer_question(
            question=item["question"],
            top_k=5,
            client=client,
            genai_client=genai_client,
            hybrid=hybrid,
            sparse_retriever=sparse_retriever,
        )
        check = check_generation_response(item, response)
        detailed_checks.append(check)
        # 1.5s pacing to stay comfortably under 15 RPM free-tier limit
        time.sleep(1.5)


    total = len(eval_dataset)
    answerable_items = [c for c in detailed_checks if c["answerable"]]
    unanswerable_items = [c for c in detailed_checks if not c["answerable"]]

    citation_accuracy = (
        sum(1 for c in answerable_items if c["has_citation"] and c["citation_matches_expected_page"])
        / len(answerable_items)
        if answerable_items
        else 0.0
    )

    doc_accuracy = (
        sum(1 for c in answerable_items if c["expected_doc_in_sources"])
        / len(answerable_items)
        if answerable_items
        else 0.0
    )

    refusal_accuracy = (
        sum(1 for c in unanswerable_items if c["refusal_correct"])
        / len(unanswerable_items)
        if unanswerable_items
        else 0.0
    )

    overall_pass_rate = (
        sum(1 for c in detailed_checks if c["passed"]) / total
        if total > 0
        else 0.0
    )

    return {
        "total_questions": total,
        "answerable_count": len(answerable_items),
        "unanswerable_count": len(unanswerable_items),
        "citation_accuracy": citation_accuracy,
        "document_accuracy": doc_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "overall_pass_rate": overall_pass_rate,
        "checks": detailed_checks,
    }
