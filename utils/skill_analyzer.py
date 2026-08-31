"""Skill-gap analysis models and Gemini integration for CareerLens V2."""

import json
import os
import time
from typing import Any, TypedDict

from google import genai
from google.genai import types

from .skill_prompts import SKILL_GAP_SYSTEM_PROMPT, build_skill_gap_prompt


class SkillItem(TypedDict):
    """One skill with status and supporting document evidence."""

    skill: str
    status: str
    evidence: str
    source: str
    section: str


class SkillGapReport(TypedDict):
    """Expected structured output of the V2 Skill Gap Analyzer."""

    matching_skills: list[SkillItem]
    related_skills: list[SkillItem]
    missing_skills: list[SkillItem]
    learning_recommendations: list[str]


REPORT_KEYS = {
    "matching_skills",
    "related_skills",
    "missing_skills",
    "learning_recommendations",
}


def _get_client(api_key: str | None = None) -> genai.Client:
    """Create a Gemini client from an explicit key or environment variable."""
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment.")
    return genai.Client(api_key=resolved_api_key)


def _normalize_skill_item(item: Any, category: str) -> SkillItem:
    """Normalize a model-returned string or partial object into a SkillItem."""
    if isinstance(item, str):
        return {
            "skill": item.strip(),
            "status": category.removesuffix("_skills"),
            "evidence": "The skill was identified from the CV and/or job description context.",
            "source": "cv_and_job_description",
            "section": "analysis",
        }

    if not isinstance(item, dict):
        raise ValueError(f"Each item in '{category}' must be a string or object.")

    skill = str(item.get("skill") or item.get("name") or "").strip()
    if not skill:
        raise ValueError(f"Each item in '{category}' must include a skill name.")

    return {
        "skill": skill,
        "status": str(item.get("status") or category.removesuffix("_skills")),
        "evidence": str(
            item.get("evidence")
            or item.get("reason")
            or item.get("explanation")
            or "Evidence was identified in the provided document context."
        ),
        "source": str(item.get("source") or "cv_and_job_description"),
        "section": str(item.get("section") or "analysis"),
    }


def validate_skill_gap_report(report: dict[str, Any]) -> SkillGapReport:
    """Normalize and validate Gemini's structured skill-gap response."""
    missing_keys = REPORT_KEYS - report.keys()
    if missing_keys:
        raise ValueError(f"Skill-gap report is missing keys: {sorted(missing_keys)}")

    normalized_report: dict[str, Any] = {}
    for category in ("matching_skills", "related_skills", "missing_skills"):
        if not isinstance(report[category], list):
            raise ValueError(f"'{category}' must be a list.")
        normalized_report[category] = [
            _normalize_skill_item(item, category) for item in report[category]
        ]

    recommendations = report["learning_recommendations"]
    if not isinstance(recommendations, list):
        raise ValueError("'learning_recommendations' must be a list.")
    normalized_report["learning_recommendations"] = [str(item) for item in recommendations]

    return normalized_report  # type: ignore[return-value]


def _is_temporary_unavailable(error: Exception) -> bool:
    """Identify temporary service errors that are safe to retry."""
    error_text = str(error).upper()
    return "503" in error_text or "UNAVAILABLE" in error_text or "HIGH DEMAND" in error_text


def analyze_skill_gap(
    cv_text: str,
    job_description: str,
    api_key: str | None = None,
    model: str | None = None,
) -> SkillGapReport:
    """Compare CV and job description and return a grounded skill-gap report."""
    if not cv_text.strip() or not job_description.strip():
        raise ValueError("Both CV text and job description are required.")

    client = _get_client(api_key)
    generation_model = model or os.getenv(
        "GEMINI_GENERATION_MODEL",
        "gemini-3.6-flash",
    )
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=generation_model,
                contents=build_skill_gap_prompt(cv_text, job_description),
                config=types.GenerateContentConfig(
                    system_instruction=SKILL_GAP_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            raw_response = response.text or "{}"
            return validate_skill_gap_report(json.loads(raw_response))
        except Exception as error:
            last_error = error
            if not _is_temporary_unavailable(error) or attempt == 2:
                break
            time.sleep(2**attempt)

    if last_error and _is_temporary_unavailable(last_error):
        raise RuntimeError(
            "Gemini is temporarily busy. Please wait a few seconds and try again."
        ) from last_error
    raise last_error or RuntimeError("Skill-gap analysis failed.")
