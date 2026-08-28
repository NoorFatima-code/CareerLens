"""Document loading helpers for CVs and job descriptions."""

from typing import BinaryIO

from .pdf_utils import clean_extracted_text, extract_text_from_pdf


def create_document(
    text: str,
    source: str,
    file_name: str,
    document_type: str,
) -> dict:
    """Create a normalized document record with text and source metadata."""
    cleaned_text = clean_extracted_text(text)

    if not cleaned_text:
        raise ValueError(f"No readable text found in {file_name}.")

    return {
        "text": cleaned_text,
        "metadata": {
            "source": source,
            "file_name": file_name,
            "document_type": document_type,
        },
    }


def load_cv_document(pdf_file: BinaryIO | bytes, file_name: str) -> dict:
    """Load a CV PDF and return normalized text with metadata."""
    extracted_text = extract_text_from_pdf(pdf_file)
    return create_document(
        text=extracted_text,
        source="cv",
        file_name=file_name,
        document_type="pdf",
    )


def load_job_description(text: str, file_name: str = "job_description.txt") -> dict:
    """Load a pasted job description and return normalized text with metadata."""
    return create_document(
        text=text,
        source="job_description",
        file_name=file_name,
        document_type="text",
    )
