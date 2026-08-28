"""Gemini embedding helpers for the manual RAG pipeline."""

import os
from typing import Literal

from google import genai
from google.genai import types


EmbeddingTask = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def _get_client(api_key: str | None = None) -> genai.Client:
    """Create a Gemini client from an explicit key or the environment."""
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment.")
    return genai.Client(api_key=resolved_api_key)


def _get_model(model: str | None = None) -> str:
    """Return the configured embedding model name."""
    return model or os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )


def generate_embeddings(
    texts: list[str],
    task_type: EmbeddingTask = "RETRIEVAL_DOCUMENT",
    api_key: str | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Generate one embedding vector for each input text."""
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Embedding input cannot contain empty text.")

    client = _get_client(api_key)
    response = client.models.embed_content(
        model=_get_model(model),
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )

    return [list(embedding.values) for embedding in response.embeddings]


def generate_embedding(
    text: str,
    task_type: EmbeddingTask = "RETRIEVAL_QUERY",
    api_key: str | None = None,
    model: str | None = None,
) -> list[float]:
    """Generate one embedding vector for a single text input."""
    embeddings = generate_embeddings(
        texts=[text],
        task_type=task_type,
        api_key=api_key,
        model=model,
    )
    return embeddings[0]
