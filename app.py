"""CareerLens - integrated manual RAG Streamlit application."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.chunker import chunk_document
from utils.document_loader import load_cv_document, load_job_description
from utils.embeddings import generate_embeddings
from utils.rag_chain import generate_grounded_answer
from utils.retriever import retrieve_relevant_chunks
from utils.vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="CareerLens",
    page_icon="CL",
    layout="wide",
)


@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    """Create one reusable persistent ChromaDB store for the app session."""
    return ChromaVectorStore(
        persist_directory=PROJECT_ROOT / "data" / "chroma_db",
        collection_name="career_lens_documents",
    )


def show_document_preview() -> None:
    """Display extracted documents and their generated chunks."""
    st.divider()
    st.markdown("### Document preview")

    cv_document = st.session_state["cv_document"]
    jd_document = st.session_state["jd_document"]
    preview_tab_cv, preview_tab_jd = st.tabs(["CV text", "Job description text"])

    with preview_tab_cv:
        st.caption(f"Characters extracted: {len(cv_document['text'])}")
        st.text_area("Extracted CV text", cv_document["text"], height=250, disabled=True)

    with preview_tab_jd:
        st.caption(f"Characters loaded: {len(jd_document['text'])}")
        st.text_area(
            "Cleaned job description",
            jd_document["text"],
            height=250,
            disabled=True,
        )

    st.markdown("### Chunk preview")
    all_chunks = [
        ("CV", chunk) for chunk in st.session_state["cv_chunks"]
    ] + [
        ("Job description", chunk)
        for chunk in st.session_state["jd_chunks"]
    ]

    for display_source, chunk in all_chunks:
        metadata = chunk["metadata"]
        with st.expander(
            f"{display_source} | {metadata['section']} | chunk {metadata['chunk_index']}"
        ):
            st.write(chunk["text"])
            st.caption(f"Chunk ID: {chunk['id']}")


def show_retrieval_results(results: list[dict]) -> None:
    """Display retrieved chunks as source evidence."""
    st.markdown("### Retrieved evidence")
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        title = (
            f"{index}. {metadata.get('source', 'unknown')} | "
            f"{metadata.get('section', 'general')} | "
            f"distance: {result.get('distance', 0):.4f}"
        )
        with st.expander(title):
            st.write(result["text"])
            st.caption(metadata.get("file_name", "unknown"))


st.title("CareerLens")
st.subheader("RAG-Powered Resume and Job Application Copilot")
st.write(
    "Upload your CV and add a target job description. CareerLens will index "
    "both documents and answer questions using retrieved evidence."
)
st.info(
    "Manual RAG flow: load → clean → chunk → embed → store → retrieve → generate"
)

st.divider()

left_column, right_column = st.columns(2)

with left_column:
    st.markdown("### 1. Upload your CV")
    cv_file = st.file_uploader(
        "Choose a CV PDF",
        type=["pdf"],
        help="The PDF will be converted into text before indexing.",
    )

with right_column:
    st.markdown("### 2. Add the target job description")
    job_description = st.text_area(
        "Paste the job description",
        height=220,
        placeholder="Paste the complete job description here...",
    )

process_documents = st.button(
    "Load, chunk and index documents",
    type="primary",
    disabled=cv_file is None or not job_description.strip(),
)

if process_documents and cv_file is not None:
    try:
        with st.spinner("Loading, chunking and indexing documents..."):
            cv_document = load_cv_document(cv_file, cv_file.name)
            jd_document = load_job_description(job_description)
            cv_chunks = chunk_document(cv_document)
            jd_chunks = chunk_document(jd_document)
            all_chunks = cv_chunks + jd_chunks
            embeddings = generate_embeddings(
                [chunk["text"] for chunk in all_chunks],
                task_type="RETRIEVAL_DOCUMENT",
            )

            vector_store = get_vector_store()
            vector_store.reset()
            vector_store.add_chunks(all_chunks, embeddings)

        st.session_state["cv_document"] = cv_document
        st.session_state["jd_document"] = jd_document
        st.session_state["cv_chunks"] = cv_chunks
        st.session_state["jd_chunks"] = jd_chunks
        st.session_state["indexed_count"] = vector_store.count()
        st.session_state["last_results"] = []
        st.success(f"Indexed {vector_store.count()} document chunks successfully.")
    except Exception as error:
        st.error(f"Could not process the documents: {error}")

if "cv_document" in st.session_state:
    show_document_preview()

    st.divider()
    st.markdown("### Ask CareerLens")
    top_k_default = int(os.getenv("TOP_K", "5"))
    top_k = st.slider("Number of chunks to retrieve", 1, 10, top_k_default)
    question = st.text_area(
        "Ask a question about your CV and target role",
        placeholder="Which skills are missing for this role?",
    )

    ask_question = st.button(
        "Retrieve evidence and generate answer",
        type="primary",
        disabled=not question.strip(),
    )

    if ask_question:
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                results = retrieve_relevant_chunks(
                    query=question,
                    vector_store=get_vector_store(),
                    top_k=top_k,
                )
                answer = generate_grounded_answer(
                    question=question,
                    retrieved_results=results,
                )

            st.session_state["last_results"] = results
            st.session_state["last_answer"] = answer
        except Exception as error:
            st.error(f"Could not generate an answer: {error}")

    if st.session_state.get("last_results"):
        show_retrieval_results(st.session_state["last_results"])

    if st.session_state.get("last_answer"):
        answer = st.session_state["last_answer"]
        st.markdown("### Grounded answer")
        st.write(answer["answer"])

        if answer["key_points"]:
            st.markdown("#### Key points")
            for point in answer["key_points"]:
                st.write(f"- {point}")

        if answer["recommendations"]:
            st.markdown("#### Recommendations")
            for recommendation in answer["recommendations"]:
                st.write(f"- {recommendation}")

        if answer["evidence"]:
            st.markdown("#### Evidence cited by Gemini")
            for evidence in answer["evidence"]:
                st.caption(
                    f"{evidence['source']} | {evidence['section']}"
                )
                st.write(f"> {evidence['quote']}")

        if answer["not_found"]:
            st.markdown("#### Not found in the uploaded documents")
            for item in answer["not_found"]:
                st.write(f"- {item}")

    if st.button("Reset local vector index"):
        get_vector_store().reset()
        for key in (
            "cv_document",
            "jd_document",
            "cv_chunks",
            "jd_chunks",
            "indexed_count",
            "last_results",
            "last_answer",
        ):
            st.session_state.pop(key, None)
        st.success("Local vector index reset.")
