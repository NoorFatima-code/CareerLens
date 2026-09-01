"""Grounded tailored-resume generation for CareerLens V3."""

import json
import os
import time
from typing import Any, TypedDict

from google import genai
from google.genai import types

from .tailored_resume_prompts import (
    TAILORED_RESUME_SYSTEM_PROMPT,
    build_tailored_resume_prompt,
)


class TailoredResumeReport(TypedDict):
    """Structured output for a tailored resume draft."""

    tailored_resume: str
    changes_made: list[str]
    preserved_facts: list[str]
    unsupported_requirements: list[str]


def _get_client(api_key: str | None = None) -> genai.Client:
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment.")
    return genai.Client(api_key=resolved_api_key)


def _as_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Response field '{field_name}' must be a list.")
    return [str(item) for item in value]


def validate_tailored_resume_report(report: dict[str, Any]) -> TailoredResumeReport:
    """Validate the structured tailored-resume response."""
    required_keys = {
        "tailored_resume",
        "changes_made",
        "preserved_facts",
        "unsupported_requirements",
    }
    missing_keys = required_keys - report.keys()
    if missing_keys:
        raise ValueError(f"Response is missing keys: {sorted(missing_keys)}")
    if not isinstance(report["tailored_resume"], str) or not report["tailored_resume"].strip():
        raise ValueError("Response field 'tailored_resume' must be a non-empty string.")
    return {
        "tailored_resume": report["tailored_resume"].strip(),
        "changes_made": _as_string_list(report["changes_made"], "changes_made"),
        "preserved_facts": _as_string_list(report["preserved_facts"], "preserved_facts"),
        "unsupported_requirements": _as_string_list(
            report["unsupported_requirements"], "unsupported_requirements"
        ),
    }


def _is_temporary_unavailable(error: Exception) -> bool:
    error_text = str(error).upper()
    return "503" in error_text or "UNAVAILABLE" in error_text or "HIGH DEMAND" in error_text


def generate_tailored_resume(
    cv_text: str,
    job_description: str,
    focus: str = "",
    api_key: str | None = None,
    model: str | None = None,
) -> TailoredResumeReport:
    """Generate a new job-specific resume draft without changing original facts."""
    if not cv_text.strip():
        raise ValueError("CV text is required.")
    if not job_description.strip():
        raise ValueError("Job description is required.")

    client = _get_client(api_key)
    generation_model = model or os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.6-flash")
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=generation_model,
                contents=build_tailored_resume_prompt(cv_text, job_description, focus),
                config=types.GenerateContentConfig(
                    system_instruction=TAILORED_RESUME_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            return validate_tailored_resume_report(
                json.loads(response.text or "{}")
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
    raise last_error or RuntimeError("Tailored resume generation failed.")
