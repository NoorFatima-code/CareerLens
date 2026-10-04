"""HTTP adapter for the existing CareerLens Streamlit/RAG implementation.

This module intentionally contains no new retrieval or generation logic. It only
translates web requests into calls to the existing functions under ``utils/`` so
the web application can use the same evidence-first pipeline.
"""

from __future__ import annotations

import base64
import binascii
import os
import tempfile
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


AnalysisKind = Literal["v1_rag", "v2_skill_gap", "v3_resume_coach", "v3_tailored_resume"]


class DocumentPayload(BaseModel):
    kind: Literal["cv", "job_description"]
    file_name: str = "document"
    content_base64: str | None = None
    text: str | None = None
    content_hash: str | None = None


class AnalyzeRequest(BaseModel):
    application_id: int = Field(gt=0)
    kind: AnalysisKind
    documents: list[DocumentPayload] = Field(min_length=1)
    question: str | None = None
    bullet: str | None = None
    focus: str = ""
    top_k: int = Field(default=5, ge=1, le=10)


class AnalyzeResponse(BaseModel):
    summary: str
    result: dict[str, Any]
    sources: list[dict[str, Any]] = Field(default_factory=list)


app = FastAPI(title="CareerLens RAG adapter", version="1.0.0")


def _authorize(service_key: str | None) -> None:
    expected_key = os.getenv("RAG_API_KEY", "").strip()
    if expected_key and service_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid RAG service key")


def _decode_pdf(payload: DocumentPayload) -> bytes:
    if not payload.content_base64:
        raise HTTPException(status_code=400, detail="CV content_base64 is required")
    try:
        return base64.b64decode(payload.content_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="CV content_base64 is invalid") from error


def _load_documents(documents: list[DocumentPayload]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the two supported documents through the existing document loaders."""
    # Imports stay inside the request path so /health can run before optional RAG
    # dependencies are installed and so this adapter never duplicates the loader.
    from utils.document_loader import load_cv_document, load_job_description

    by_kind = {document.kind: document for document in documents}
    cv_payload = by_kind.get("cv")
    jd_payload = by_kind.get("job_description")
    if not cv_payload or not jd_payload:
        raise HTTPException(
            status_code=400,
            detail="Both a CV PDF and a job description are required",
        )

    cv_document = load_cv_document(
        _decode_pdf(cv_payload),
        cv_payload.file_name,
    )
    jd_text = jd_payload.text
    if not jd_text and jd_payload.content_base64:
        try:
            jd_text = base64.b64decode(jd_payload.content_base64, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError) as error:
            raise HTTPException(status_code=400, detail="Job description text is invalid") from error
    if not jd_text:
        raise HTTPException(status_code=400, detail="Job description text is required")
    jd_document = load_job_description(jd_text, jd_payload.file_name)
    return cv_document, jd_document


def _run_v1(
    cv_document: dict[str, Any],
    jd_document: dict[str, Any],
    request: AnalyzeRequest,
) -> AnalyzeResponse:
    """Run the existing manual RAG pipeline in an isolated temporary index."""
    from utils.chunker import chunk_document
    from utils.embeddings import generate_embeddings
    from utils.rag_chain import generate_grounded_answer
    from utils.retriever import retrieve_relevant_chunks
    from utils.vector_store import ChromaVectorStore

    cv_chunks = chunk_document(cv_document)
    jd_chunks = chunk_document(jd_document)
    all_chunks = cv_chunks + jd_chunks
    embeddings = generate_embeddings(
        [chunk["text"] for chunk in all_chunks],
        task_type="RETRIEVAL_DOCUMENT",
    )
    question = (request.question or "What skills and experience from my CV are relevant to this target role?").strip()

    # The Streamlit app keeps a local persistent collection. The web adapter uses
    # a request-scoped collection so different users/applications never share data.
    with tempfile.TemporaryDirectory(prefix="career-lens-rag-") as directory:
        vector_store = ChromaVectorStore(
            persist_directory=directory,
            collection_name=f"application_{request.application_id}",
        )
        vector_store.add_chunks(all_chunks, embeddings)
        sources = retrieve_relevant_chunks(
            query=question,
            vector_store=vector_store,
            top_k=request.top_k,
        )
        answer = generate_grounded_answer(question, sources)

    result = {"question": question, **answer}
    return AnalyzeResponse(
        summary=str(answer.get("answer", "Grounded analysis completed.")),
        result=result,
        sources=sources,
    )


def _run_v2_or_v3(
    cv_document: dict[str, Any],
    jd_document: dict[str, Any],
    request: AnalyzeRequest,
) -> AnalyzeResponse:
    """Delegate applicant tools to their existing implementations."""
    cv_text = cv_document["text"]
    jd_text = jd_document["text"]

    if request.kind == "v2_skill_gap":
        from utils.skill_analyzer import analyze_skill_gap

        result = analyze_skill_gap(cv_text=cv_text, job_description=jd_text)
        return AnalyzeResponse(summary="Skill-gap analysis completed.", result=dict(result))

    if request.kind == "v3_resume_coach":
        if not request.bullet or not request.bullet.strip():
            raise HTTPException(status_code=400, detail="A resume bullet is required for resume coaching")
        from utils.resume_coach import improve_resume_bullet

        result = improve_resume_bullet(
            bullet=request.bullet,
            cv_text=cv_text,
            job_description=jd_text,
            focus=request.focus,
        )
        return AnalyzeResponse(summary="Resume-coach analysis completed.", result=dict(result))

    from utils.tailored_resume import generate_tailored_resume

    result = generate_tailored_resume(
        cv_text=cv_text,
        job_description=jd_text,
        focus=request.focus,
    )
    return AnalyzeResponse(summary="Tailored resume generated.", result=dict(result))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "CareerLens existing RAG adapter"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    x_rag_service_key: str | None = Header(default=None),
) -> AnalyzeResponse:
    _authorize(x_rag_service_key)
    try:
        cv_document, jd_document = _load_documents(request.documents)
        if request.kind == "v1_rag":
            return _run_v1(cv_document, jd_document, request)
        return _run_v2_or_v3(cv_document, jd_document, request)
    except HTTPException:
        raise
    except Exception as error:
        # Do not leak provider credentials or stack traces to the website client.
        raise HTTPException(status_code=502, detail=str(error)) from error
