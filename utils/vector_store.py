"""ChromaDB storage helpers for CareerLens document chunks."""

from pathlib import Path
from typing import Any

import chromadb # type: ignore


class ChromaVectorStore:
    """Small wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma_db",
        collection_name: str = "career_lens_documents",
    ) -> None:
        self.persist_directory = str(Path(persist_directory))
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunk text, IDs, metadata and matching vectors to ChromaDB."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks],
            embeddings=embeddings,
        )

    def _matching_ids(self, where: dict[str, Any] | None = None) -> list[str]:
        """Get IDs matching a metadata filter without using count(where=...)."""
        result = self.collection.get(where=where, include=[])
        return result.get("ids", [])

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Return the closest chunks, optionally filtered by metadata."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        matching_ids = self._matching_ids(where)
        if not matching_ids:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(matching_ids)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
            )
        ]

    def reset(self) -> None:
        """Delete all indexed chunks from the current collection."""
        existing_ids = self._matching_ids()
        if existing_ids:
            self.collection.delete(ids=existing_ids)

    def count(self, where: dict[str, Any] | None = None) -> int:
        """Return the number of indexed chunks, optionally filtered by metadata."""
        return len(self._matching_ids(where))
