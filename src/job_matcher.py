"""
Job Matcher Module for StudentAI Career RAG.
Performs deterministic matching between candidate resume skills and job requirements.

Design Principles:
1. Deterministic Matching:
   Uses exact normalized string matching with a conservative, explicit alias dictionary.
2. No LLM for Skill Gap:
   Does NOT ask an LLM to decide if skills match or calculate percentages.
3. Strict Categories:
   - matched_skills: full exact match or approved exact alias match.
   - partial_matches: related or partial mention (reported separately, NEVER counted toward coverage).
   - missing_skills: requirements not found in candidate skills.
4. Coverage Formula:
   coverage_percentage = (matched_count / requirement_count) * 100.0 (or 0.0 if requirements == 0).
5. Transparency:
   Explicitly includes disclaimer that this is a deterministic keyword match, not an ATS score,
   not a hiring probability, and not a verified suitability score.
"""

from typing import Dict, Any, List, Optional, Set
import re
from src.skill_extractor import normalize_skill_name

# Conservative, exact-equivalent alias mapping (strictly identical technologies)
EXACT_ALIASES: Dict[str, str] = {
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "py": "python",
    "python": "python",
    "py3": "python",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "node": "node.js",
    "nodejs": "node.js",
    "node.js": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "react": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "vue": "vue",
    "golang": "go",
    "go lang": "go",
    "amazon web services": "aws",
    "aws": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "gcp": "gcp",
    "azure": "azure",
    "microsoft azure": "azure",
    "restful api": "rest api",
    "restful apis": "rest api",
    "rest api": "rest api",
    "rest apis": "rest api",
    "rest": "rest api",
}


def _canonical_key(name: str) -> str:
    """Computes canonical lookup key via alias table or normalized name."""
    norm = normalize_skill_name(name)
    if norm in EXACT_ALIASES:
        return EXACT_ALIASES[norm]
    # Remove interior spaces and hyphens for canonical token comparison
    compact = re.sub(r"[\s\-_]", "", norm)
    if compact in EXACT_ALIASES:
        return EXACT_ALIASES[compact]
    return norm


def match_resume_to_job(
    resume_skills: List[Dict[str, Any]],
    job_requirements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministically matches resume skills against job requirements.

    Args:
        resume_skills: List of skill dicts extracted from a candidate resume.
        job_requirements: List of skill dicts extracted from a job description.

    Returns:
        Structured match result:
        {
            "requirement_count": int,
            "matched_count": int,
            "missing_count": int,
            "partial_count": int,
            "coverage_percentage": float,
            "matched_skills": List[Dict[str, Any]],
            "partial_matches": List[Dict[str, Any]],
            "missing_skills": List[Dict[str, Any]],
            "evidence": Dict[str, Any],
            "limitations": List[str],
        }
    """
    # Deduplicate candidate skills by canonical key
    resume_by_canonical: Dict[str, Dict[str, Any]] = {}
    for r_skill in (resume_skills or []):
        r_name = str(r_skill.get("name", "")).strip()
        if not r_name:
            continue
        c_key = _canonical_key(r_skill.get("normalized_name") or r_name)
        if c_key not in resume_by_canonical:
            resume_by_canonical[c_key] = r_skill

    # Deduplicate job requirements by canonical key
    unique_job_requirements: List[Dict[str, Any]] = []
    seen_req_keys: Set[str] = set()

    for j_skill in (job_requirements or []):
        j_name = str(j_skill.get("name", "")).strip()
        if not j_name:
            continue
        c_key = _canonical_key(j_skill.get("normalized_name") or j_name)
        if c_key not in seen_req_keys:
            seen_req_keys.add(c_key)
            unique_job_requirements.append(j_skill)

    matched_skills: List[Dict[str, Any]] = []
    partial_matches: List[Dict[str, Any]] = []
    missing_skills: List[Dict[str, Any]] = []

    for req in unique_job_requirements:
        req_name = str(req.get("name", "")).strip()
        req_norm = req.get("normalized_name") or normalize_skill_name(req_name)
        req_key = _canonical_key(req_norm)

        # 1. Exact or Alias Match
        if req_key in resume_by_canonical:
            matched_r = resume_by_canonical[req_key]
            matched_skills.append({
                "requirement": req_name,
                "matched_resume_skill": matched_r.get("name", req_name),
                "skill_type": req.get("skill_type") or matched_r.get("skill_type", "other"),
                "resume_evidence": matched_r.get("evidence_quote", ""),
                "job_evidence": req.get("evidence_quote", ""),
                "page_number": matched_r.get("page_number"),
            })
            continue

        # 2. Check for partial or related mention (conservative heuristic)
        found_partial = False
        for c_key, matched_r in resume_by_canonical.items():
            r_norm = normalize_skill_name(matched_r.get("name", ""))
            # Partial condition: one is a full word-boundary substring of another (e.g. "aws lambda" vs "aws")
            # But avoid false positives like "java" vs "javascript"
            if req_norm != r_norm and (
                re.search(rf"\b{re.escape(req_norm)}\b", r_norm) or
                re.search(rf"\b{re.escape(r_norm)}\b", req_norm)
            ):
                # Disallow known false positives (e.g. java and javascript)
                if ("java" in {req_norm, r_norm} and "javascript" in {req_norm, r_norm}):
                    continue
                if ("c" in {req_norm, r_norm} and ("c++" in {req_norm, r_norm} or "c#" in {req_norm, r_norm})):
                    continue

                partial_matches.append({
                    "requirement": req_name,
                    "candidate_mention": matched_r.get("name", ""),
                    "note": f"Partial or related match between '{req_name}' and '{matched_r.get('name', '')}'",
                    "resume_evidence": matched_r.get("evidence_quote", ""),
                    "job_evidence": req.get("evidence_quote", ""),
                })
                found_partial = True
                break

        # 3. Missing skill
        if not found_partial:
            missing_skills.append({
                "requirement": req_name,
                "skill_type": req.get("skill_type", "other"),
                "job_evidence": req.get("evidence_quote", ""),
            })

    requirement_count = len(unique_job_requirements)
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)
    partial_count = len(partial_matches)

    if requirement_count == 0:
        coverage_percentage = 0.0
    else:
        coverage_percentage = round((matched_count / requirement_count) * 100.0, 2)

    limitations = [
        "This analysis performs deterministic keyword and explicit alias matching only.",
        "Partial matches are reported for transparency but are NOT counted toward the coverage percentage.",
        "This score is NOT an ATS screening score, hiring probability, or certified proficiency metric.",
        "Skills demonstrated through unlisted project experience or implicit context are not detected.",
    ]

    return {
        "requirement_count": requirement_count,
        "matched_count": matched_count,
        "missing_count": missing_count,
        "partial_count": partial_count,
        "coverage_percentage": coverage_percentage,
        "matched_skills": matched_skills,
        "partial_matches": partial_matches,
        "missing_skills": missing_skills,
        "evidence": {
            "matched": [m.get("resume_evidence") for m in matched_skills if m.get("resume_evidence")],
            "missing": [m.get("job_evidence") for m in missing_skills if m.get("job_evidence")],
        },
        "limitations": limitations,
    }
