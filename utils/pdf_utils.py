"""Utilities for extracting readable text from PDF documents."""

import re
from io import BytesIO
from typing import BinaryIO

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph-level line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_file: BinaryIO | bytes) -> str:
    """Extract and clean text from a PDF file or raw PDF bytes.

    The function accepts Streamlit's UploadedFile object because it behaves
    like a binary file and can be passed directly to PdfReader.
    """
    if isinstance(pdf_file, bytes):
        pdf_stream = BytesIO(pdf_file)
    else:
        pdf_stream = pdf_file
        pdf_stream.seek(0)

    reader = PdfReader(pdf_stream)
    page_texts: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            page_texts.append(page_text)

    return clean_extracted_text("\n\n".join(page_texts))
