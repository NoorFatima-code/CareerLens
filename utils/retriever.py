"""Query retrieval helpers for the manual CareerLens RAG pipeline."""

from typing import Any

from .embeddings import generate_embedding
from .vector_store import ChromaVectorStore



def retrieve_relevant_chunks(
    query: str,
    vector_store: ChromaVectorStore,
    top_k: int = 5,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and return balanced CV/JD evidence from the vector store."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if vector_store.count() == 0:
        return []

    query_embedding = generate_embedding(
        text=cleaned_query,
        task_type="RETRIEVAL_QUERY",
        api_key=api_key,
    )

    per_source_results = {
        source: vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"source": source},
        )
        for source in ("cv", "job_description")
    }

    # Reserve the strongest result from each available source first.
    balanced_results: list[dict[str, Any]] = []
    for source_results in per_source_results.values():
        if source_results:
            balanced_results.append(source_results[0])

    selected_ids = {result["id"] for result in balanced_results}
    remaining_results = sorted(
        (
            result
            for source_results in per_source_results.values()
            for result in source_results
            if result["id"] not in selected_ids
        ),
        key=lambda result: result["distance"],
    )

    return (balanced_results + remaining_results)[:top_k]



def format_retrieved_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into labelled context for a generation prompt."""
    context_blocks: list[str] = []

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        section = metadata.get("section", "general")
        file_name = metadata.get("file_name", "unknown")

        context_blocks.append(
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"File: {file_name}\n"
            f"Text: {result.get('text', '')}"
        )

    return "\n\n".join(context_blocks)
