"""CareerLens - integrated manual RAG application with V2 skill-gap analysis."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.chunker import chunk_document
from utils.document_loader import load_cv_document, load_job_description
from utils.embeddings import generate_embeddings
from utils.rag_chain import generate_grounded_answer
from utils.retriever import retrieve_relevant_chunks
from utils.skill_analyzer import analyze_skill_gap
from utils.vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(page_title="CareerLens", page_icon="CL", layout="wide")


@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    """Create one reusable persistent ChromaDB store."""
    return ChromaVectorStore(
        persist_directory=PROJECT_ROOT / "data" / "chroma_db",
        collection_name="career_lens_documents",
    )


def show_document_preview() -> None:
    """Display extracted documents and chunks."""
    st.markdown("### Document preview")
    cv_document = st.session_state["cv_document"]
    jd_document = st.session_state["jd_document"]
    cv_tab, jd_tab = st.tabs(["CV text", "Job description text"])

    with cv_tab:
        st.caption(f"Characters extracted: {len(cv_document['text'])}")
        st.text_area("Extracted CV text", cv_document["text"], height=220, disabled=True)
    with jd_tab:
        st.caption(f"Characters loaded: {len(jd_document['text'])}")
        st.text_area("Cleaned job description", jd_document["text"], height=220, disabled=True)

    st.markdown("### Chunk preview")
    all_chunks = [("CV", c) for c in st.session_state["cv_chunks"]]
    all_chunks += [("Job description", c) for c in st.session_state["jd_chunks"]]
    for source_label, chunk in all_chunks:
        metadata = chunk["metadata"]
        with st.expander(f"{source_label} | {metadata['section']} | chunk {metadata['chunk_index']}"):
            st.write(chunk["text"])
            st.caption(f"Chunk ID: {chunk['id']}")


def show_retrieval_results(results: list[dict]) -> None:
    """Display retrieved evidence chunks."""
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


def show_skill_item(item: dict) -> None:
    """Render one skill item with its evidence."""
    st.markdown(f"**{item['skill']}** — {item['status']}")
    st.caption(f"{item['source']} | {item['section']}")
    st.write(item["evidence"])


st.title("CareerLens")
st.subheader("RAG-Powered Resume and Job Application Copilot")
st.write(
    "Analyze your CV against a target job using grounded retrieval and evidence-based skill guidance."
)
st.info("V1 RAG + V2 Skill Gap Analyzer")

left_column, right_column = st.columns(2)
with left_column:
    st.markdown("### 1. Upload your CV")
    cv_file = st.file_uploader("Choose a CV PDF", type=["pdf"])
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

        st.session_state.update(
            {
                "cv_document": cv_document,
                "jd_document": jd_document,
                "cv_chunks": cv_chunks,
                "jd_chunks": jd_chunks,
                "last_results": [],
                "last_answer": None,
                "skill_gap_report": None,
            }
        )
        st.success(f"Indexed {vector_store.count()} document chunks successfully.")
    except Exception as error:
        st.error(f"Could not process the documents: {error}")

if "cv_document" in st.session_state:
    show_document_preview()

    st.divider()
    st.markdown("## V2 — Skill Gap Analyzer")
    st.write(
        "Compare explicit CV evidence with the target role. Missing means not found in the uploaded CV; it does not prove that you cannot perform the skill."
    )
    analyze_button = st.button("Analyze skill gaps", type="primary")
    if analyze_button:
        try:
            with st.spinner("Comparing CV and job description..."):
                report = analyze_skill_gap(
                    cv_text=st.session_state["cv_document"]["text"],
                    job_description=st.session_state["jd_document"]["text"],
                )
            st.session_state["skill_gap_report"] = report
        except Exception as error:
            st.error(f"Could not analyze skill gaps: {error}")

    report = st.session_state.get("skill_gap_report")
    if report:
        match_tab, related_tab, missing_tab, learning_tab = st.tabs(
            ["Matching", "Related", "Missing", "Learning recommendations"]
        )
        with match_tab:
            for item in report["matching_skills"]:
                show_skill_item(item)
            if not report["matching_skills"]:
                st.info("No direct matching skills were identified.")
        with related_tab:
            for item in report["related_skills"]:
                show_skill_item(item)
            if not report["related_skills"]:
                st.info("No related skills were identified.")
        with missing_tab:
            for item in report["missing_skills"]:
                show_skill_item(item)
            if not report["missing_skills"]:
                st.success("No potential skill gaps were identified.")
        with learning_tab:
            for recommendation in report["learning_recommendations"]:
                st.write(f"- {recommendation}")

    st.divider()
    st.markdown("## V1 — Ask CareerLens")
    top_k_default = min(max(int(os.getenv("TOP_K", "5")), 1), 10)
    top_k = st.slider("Number of chunks to retrieve", 1, 10, top_k_default)
    question = st.text_area(
        "Ask a question about your CV and target role",
        placeholder="Which projects are relevant to this role?",
    )
    ask_question = st.button("Retrieve evidence and generate answer", disabled=not question.strip())

    if ask_question:
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                results = retrieve_relevant_chunks(
                    query=question,
                    vector_store=get_vector_store(),
                    top_k=top_k,
                )
                answer = generate_grounded_answer(question, results)
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
                st.caption(f"{evidence['source']} | {evidence['section']}")
                st.write(f"> {evidence['quote']}")
        if answer["not_found"]:
            st.markdown("#### Not found in the uploaded documents")
            for item in answer["not_found"]:
                st.write(f"- {item}")

    if st.button("Reset local vector index"):
        get_vector_store().reset()
        for key in (
            "cv_document", "jd_document", "cv_chunks", "jd_chunks",
            "last_results", "last_answer", "skill_gap_report",
        ):
            st.session_state.pop(key, None)
        st.success("Local vector index reset.")
