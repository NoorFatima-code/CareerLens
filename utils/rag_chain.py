"""Manual grounded generation layer for the CareerLens RAG pipeline."""

import json
import os
from typing import Any

from google import genai
from google.genai import types

from .prompts import GROUNDED_SYSTEM_PROMPT, build_grounded_prompt
from .retriever import format_retrieved_context
from .validators import validate_grounded_response


def _get_client(api_key: str | None = None) -> genai.Client:
    """Create a Gemini client from an explicit key or the environment."""
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment.")
    return genai.Client(api_key=resolved_api_key)


def generate_grounded_answer(
    question: str,
    retrieved_results: list[dict[str, Any]],
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate and validate a JSON answer from retrieved context only."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_results:
        return {
            "answer": "The retrieved documents do not contain enough information to answer this question.",
            "key_points": [],
            "evidence": [],
            "recommendations": [],
            "not_found": [question.strip()],
        }

    client = _get_client(api_key)
    generation_model = model or os.getenv(
        "GEMINI_GENERATION_MODEL",
        "gemini-3.6-flash",
    )
    response = client.models.generate_content(
        model=generation_model,
        contents=build_grounded_prompt(
            question=question,
            context=format_retrieved_context(retrieved_results),
        ),
        config=types.GenerateContentConfig(
            system_instruction=GROUNDED_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    raw_response = response.text or "{}"
    return validate_grounded_response(json.loads(raw_response))
