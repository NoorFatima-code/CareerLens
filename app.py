"""CareerLens - integrated manual RAG application with V2 skill-gap analysis."""

import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from utils.chunker import chunk_document
from utils.document_loader import load_cv_document, load_job_description
from utils.embeddings import generate_embeddings
from utils.rag_chain import generate_grounded_answer
from utils.retriever import retrieve_relevant_chunks
from utils.resume_coach import improve_resume_bullet
from utils.resume_template import render_resume_html, tailored_resume_to_pdf
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
                "resume_report": None,
                "tailored_resume_report": None,
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
    st.markdown("## V3 — Resume Improvement Coach")
    st.write(
        "Improve one CV bullet for the target role without inventing or changing facts."
    )
    bullet = st.text_area(
        "Paste one resume bullet",
        height=120,
        placeholder="Built a Flask expense tracker with SQLite.",
    )
    focus = st.text_input(
        "Optional improvement focus",
        placeholder="Emphasize backend development and technical clarity",
    )
    improve_button = st.button(
        "Improve resume bullet",
        type="primary",
        disabled=not bullet.strip(),
    )

    if improve_button:
        try:
            with st.spinner("Improving bullet while preserving facts..."):
                resume_report = improve_resume_bullet(
                    bullet=bullet,
                    cv_text=st.session_state["cv_document"]["text"],
                    job_description=st.session_state["jd_document"]["text"],
                    focus=focus,
                )
            st.session_state["resume_report"] = resume_report
        except Exception as error:
            st.error(f"Could not improve resume bullet: {error}")

    resume_report = st.session_state.get("resume_report")
    if resume_report:
        if resume_report.get("not_found"):
            for item in resume_report["not_found"]:
                st.warning(item)
        for index, improvement in enumerate(resume_report["improvements"], start=1):
            st.markdown(f"### Improvement {index}")
            st.markdown("**Original bullet**")
            st.write(improvement["original_bullet"])
            st.markdown("**Improved bullet**")
            st.success(improvement["improved_bullet"])
            if improvement["changes_made"]:
                st.markdown("**Changes made**")
                for change in improvement["changes_made"]:
                    st.write(f"- {change}")
            if improvement["preserved_facts"]:
                st.markdown("**Preserved facts**")
                for fact in improvement["preserved_facts"]:
                    st.write(f"- {fact}")
            if improvement["missing_information"]:
                st.markdown("**Missing information**")
                for missing in improvement["missing_information"]:
                    st.write(f"- {missing}")

    st.divider()
    st.markdown("## V3 — Generate Tailored Resume")
    st.write(
        "Create a job-specific resume draft from your original CV without inventing unsupported facts."
    )
    tailored_focus = st.text_input(
        "Optional tailoring focus",
        placeholder="Prioritize backend projects and API experience",
        key="tailored_resume_focus",
    )
    tailor_button = st.button("Generate tailored resume", type="primary")

    if tailor_button:
        try:
            with st.spinner("Generating a fact-preserving tailored resume..."):
                from utils.tailored_resume import generate_tailored_resume

                tailored_report = generate_tailored_resume(
                    cv_text=st.session_state["cv_document"]["text"],
                    job_description=st.session_state["jd_document"]["text"],
                    focus=tailored_focus,
                )
            st.session_state["tailored_resume_report"] = tailored_report
        except Exception as error:
            st.error(f"Could not generate tailored resume: {error}")

    tailored_report = st.session_state.get("tailored_resume_report")
    if tailored_report:
        st.markdown("### Tailored resume preview")
        st.caption("Original CV remains unchanged. Review the tailored draft before downloading.")
        components.html(
            render_resume_html(tailored_report["tailored_resume"]),
            height=900,
            scrolling=True,
        )

        with st.expander("Show editable Markdown text"):
            st.text_area(
                "Tailored resume source",
                tailored_report["tailored_resume"],
                height=260,
                key="tailored_resume_source_preview",
            )

        markdown_sections = [
            "# Tailored Resume",
            "",
            tailored_report["tailored_resume"],
        ]
        if tailored_report["changes_made"]:
            markdown_sections.extend(
                ["", "## Changes Made", *[f"- {item}" for item in tailored_report["changes_made"]]]
            )
        if tailored_report["preserved_facts"]:
            markdown_sections.extend(
                ["", "## Preserved Facts", *[f"- {item}" for item in tailored_report["preserved_facts"]]]
            )
        if tailored_report["unsupported_requirements"]:
            markdown_sections.extend(
                ["", "## Unsupported Requirements Not Added", *[f"- {item}" for item in tailored_report["unsupported_requirements"]]]
            )

        pdf_bytes = tailored_resume_to_pdf(tailored_report["tailored_resume"])
        download_column, pdf_column = st.columns(2)
        with download_column:
            st.download_button(
                "Download Markdown",
                data="\\n".join(markdown_sections),
                file_name="tailored_resume.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with pdf_column:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="tailored_resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        change_tab, facts_tab, safety_tab = st.tabs(
            ["Changes made", "Preserved facts", "Not added"]
        )
        with change_tab:
            if tailored_report["changes_made"]:
                for change in tailored_report["changes_made"]:
                    st.write(f"- {change}")
            else:
                st.info("No structural changes were reported.")
        with facts_tab:
            if tailored_report["preserved_facts"]:
                for fact in tailored_report["preserved_facts"]:
                    st.write(f"- {fact}")
            else:
                st.info("No preserved facts were listed by the model.")
        with safety_tab:
            if tailored_report["unsupported_requirements"]:
                for requirement in tailored_report["unsupported_requirements"]:
                    st.warning(requirement)
            else:
                st.success("No unsupported requirements were added to the draft.")

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
            "last_results", "last_answer", "skill_gap_report", "resume_report",
        ):
            st.session_state.pop(key, None)
        st.success("Local vector index reset.")
