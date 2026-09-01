"""Fact-preserving resume bullet improvement for CareerLens V3."""

import json
import os
import re
import time
from difflib import SequenceMatcher

from .embeddings import generate_embeddings
from typing import Any, TypedDict

from google import genai
from google.genai import types

from .resume_prompts import RESUME_COACH_SYSTEM_PROMPT, build_resume_improvement_prompt


class ResumeImprovement(TypedDict):
    """One original bullet and its fact-preserving improvement."""

    original_bullet: str
    improved_bullet: str
    changes_made: list[str]
    preserved_facts: list[str]
    missing_information: list[str]


class ResumeImprovementReport(TypedDict):
    """Structured V3 response, including an explicit not-found state."""

    improvements: list[ResumeImprovement]
    not_found: list[str]


COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "built", "by", "for", "from",
    "in", "into", "is", "of", "on", "the", "to", "using", "with",
}


def _get_client(api_key: str | None = None) -> genai.Client:
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment.")
    return genai.Client(api_key=resolved_api_key)


def _content_tokens(text: str) -> set[str]:
    """Extract meaningful lowercase tokens for a lightweight evidence check."""
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", text.lower()))
    return tokens - COMMON_WORDS


def _normalized_text(text: str) -> str:
    """Normalize punctuation and whitespace for exact/near-match checks."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.-]+", " ", text.lower())).strip()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cv_evidence_candidates(cv_text: str) -> list[str]:
    """Extract usable CV lines/sentences for semantic evidence checking."""
    candidates = re.split(r"\n+|(?<=[.!?])\s+", cv_text)
    return [candidate.strip(" •-\t") for candidate in candidates if len(candidate.strip()) >= 25]


def _semantic_bullet_has_cv_evidence(bullet: str, cv_text: str) -> bool:
    """Use document/query embeddings to verify a paraphrased CV bullet."""
    candidates = _cv_evidence_candidates(cv_text)
    if not candidates:
        return False
    try:
        query_embedding = generate_embeddings(
            [bullet], task_type="RETRIEVAL_QUERY"
        )[0]
        candidate_embeddings = generate_embeddings(
            candidates, task_type="RETRIEVAL_DOCUMENT"
        )
    except Exception:
        # Evidence verification must fail closed if the embedding service is unavailable.
        return False

    best_similarity = max(
        _cosine_similarity(query_embedding, candidate_embedding)
        for candidate_embedding in candidate_embeddings
    )
    return best_similarity >= 0.78


def bullet_has_cv_evidence(bullet: str, cv_text: str) -> bool:
    """Return True for exact, near-match, or semantically supported CV bullets."""
    normalized_bullet = _normalized_text(bullet)
    normalized_cv = _normalized_text(cv_text)
    if not normalized_bullet or not normalized_cv:
        return False

    if normalized_bullet in normalized_cv:
        return True

    bullet_tokens = _content_tokens(bullet)
    if len(bullet_tokens) >= 3:
        for cv_line in cv_text.splitlines():
            normalized_line = _normalized_text(cv_line)
            if not normalized_line:
                continue
            similarity = SequenceMatcher(None, normalized_bullet, normalized_line).ratio()
            line_tokens = _content_tokens(cv_line)
            overlap_ratio = len(bullet_tokens.intersection(line_tokens)) / len(bullet_tokens)
            if similarity >= 0.90 and overlap_ratio >= 0.90:
                return True

    return _semantic_bullet_has_cv_evidence(bullet, cv_text)


def _normalize_improvement(item: Any, original_bullet: str) -> ResumeImprovement:
    if not isinstance(item, dict):
        raise ValueError("Each resume improvement must be an object.")

    improved_bullet = str(item.get("improved_bullet") or item.get("rewrite") or "").strip()
    if not improved_bullet:
        raise ValueError("Resume improvement is missing improved_bullet.")

    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(entry) for entry in value]
        return [str(value)]

    return {
        "original_bullet": str(item.get("original_bullet") or original_bullet),
        "improved_bullet": improved_bullet,
        "changes_made": as_list(item.get("changes_made")),
        "preserved_facts": as_list(item.get("preserved_facts")),
        "missing_information": as_list(item.get("missing_information")),
    }


def _is_temporary_unavailable(error: Exception) -> bool:
    error_text = str(error).upper()
    return "503" in error_text or "UNAVAILABLE" in error_text or "HIGH DEMAND" in error_text


def validate_resume_improvement_report(
    report: dict[str, Any],
    original_bullet: str,
) -> ResumeImprovementReport:
    improvements = report.get("improvements")
    if not isinstance(improvements, list):
        raise ValueError("Resume response must contain an improvements list.")
    not_found = report.get("not_found", [])
    if not isinstance(not_found, list):
        raise ValueError("Resume response field 'not_found' must be a list.")
    return {
        "improvements": [
            _normalize_improvement(item, original_bullet) for item in improvements
        ],
        "not_found": [str(item) for item in not_found],
    }


def improve_resume_bullet(
    bullet: str,
    cv_text: str,
    job_description: str,
    focus: str = "",
    api_key: str | None = None,
    model: str | None = None,
) -> ResumeImprovementReport:
    """Generate a grounded, fact-preserving improvement for one CV bullet."""
    if not bullet.strip():
        raise ValueError("Resume bullet cannot be empty.")
    if not cv_text.strip() or not job_description.strip():
        raise ValueError("CV text and job description are required.")

    if not bullet_has_cv_evidence(bullet, cv_text):
        return {
            "improvements": [],
            "not_found": [
                "This bullet was not found or sufficiently supported in the uploaded CV. "
                "Please paste an existing CV bullet before requesting an improvement."
            ],
        }

    client = _get_client(api_key)
    generation_model = model or os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.6-flash")
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=generation_model,
                contents=build_resume_improvement_prompt(
                    bullet, cv_text, job_description, focus
                ),
                config=types.GenerateContentConfig(
                    system_instruction=RESUME_COACH_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.15,
                ),
            )
            return validate_resume_improvement_report(
                json.loads(response.text or "{}"), bullet
            )
        except Exception as error:
            last_error = error
            if not _is_temporary_unavailable(error) or attempt == 2:
                break
            time.sleep(2**attempt)

    if last_error and _is_temporary_unavailable(last_error):
        raise RuntimeError(
            "Gemini is temporarily busy. Please wait a few seconds and try again."
        ) from last_error
    raise last_error or RuntimeError("Resume improvement failed.")
