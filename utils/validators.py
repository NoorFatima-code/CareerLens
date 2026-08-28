"""Validation helpers for structured CareerLens responses."""

import json
from typing import Any


REQUIRED_RESPONSE_KEYS = {
    "answer",
    "key_points",
    "evidence",
    "recommendations",
    "not_found",
}


def validate_grounded_response(response: dict[str, Any] | str) -> dict[str, Any]:
    """Parse and validate the expected grounded-response structure."""
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini response was not valid JSON.") from exc

    if not isinstance(response, dict):
        raise ValueError("Grounded response must be a JSON object.")

    missing_keys = REQUIRED_RESPONSE_KEYS - response.keys()
    if missing_keys:
        raise ValueError(f"Grounded response is missing keys: {sorted(missing_keys)}")

    for key in ("key_points", "recommendations", "not_found", "evidence"):
        if not isinstance(response[key], list):
            raise ValueError(f"Response field '{key}' must be a list.")

    if not isinstance(response["answer"], str):
        raise ValueError("Response field 'answer' must be a string.")

    for evidence_item in response["evidence"]:
        if not isinstance(evidence_item, dict):
            raise ValueError("Each evidence item must be an object.")
        required_evidence_keys = {"source", "section", "quote"}
        if not required_evidence_keys.issubset(evidence_item.keys()):
            raise ValueError("Each evidence item needs source, section, and quote.")

    return response
