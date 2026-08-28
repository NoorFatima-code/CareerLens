"""Section-aware and fallback text chunking utilities."""

import re


SECTION_ALIASES = {
    "professional summary": "professional_summary",
    "summary": "professional_summary",
    "profile": "professional_summary",
    "technical skills": "technical_skills",
    "skills": "technical_skills",
    "projects": "projects",
    "experience": "experience",
    "work experience": "experience",
    "education": "education",
    "certifications": "certifications",
    "responsibilities": "responsibilities",
    "required skills": "required_skills",
    "required qualifications": "required_qualifications",
    "preferred skills": "preferred_skills",
    "qualifications": "qualifications",
}


def _normalize_heading(line: str) -> str:
    """Normalize a possible heading for alias matching."""
    line = line.strip().lower()
    line = re.sub(r"^[#\-•\s]+|[:\-\s]+$", "", line)
    return re.sub(r"\s+", " ", line)


def detect_section(line: str) -> str | None:
    """Return a canonical section name if the line looks like a known heading."""
    normalized_line = _normalize_heading(line)
    return SECTION_ALIASES.get(normalized_line)


def split_text_fallback(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """Split text by paragraphs, then use overlapping character windows if needed."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1.")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = start + chunk_size
            chunks.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = end - chunk_overlap
        current = ""

    if current:
        chunks.append(current.strip())

    return chunks


def _section_blocks(text: str) -> list[tuple[str, str]]:
    """Group lines under detected headings and return section/text blocks."""
    blocks: list[tuple[str, str]] = []
    current_section = "general"
    current_lines: list[str] = []

    def flush() -> None:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            blocks.append((current_section, section_text))

    for line in text.splitlines():
        detected_section = detect_section(line)
        if detected_section:
            flush()
            current_lines.clear()
            current_section = detected_section
        else:
            current_lines.append(line)

    flush()
    return blocks


def chunk_document(
    document: dict,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    """Create chunk records from a normalized document."""
    text = document.get("text", "").strip()
    base_metadata = document.get("metadata", {})

    if not text:
        raise ValueError("Document text cannot be empty.")

    blocks = _section_blocks(text)
    has_detected_heading = any(section != "general" for section, _ in blocks)
    if not has_detected_heading:
        blocks = [("general", text)]

    chunk_records: list[dict] = []
    chunk_index = 0

    for section, section_text in blocks:
        section_chunks = split_text_fallback(
            section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk_text in section_chunks:
            source = base_metadata.get("source", "document")
            chunk_records.append(
                {
                    "id": f"{source}_{section}_{chunk_index}",
                    "text": chunk_text,
                    "metadata": {
                        **base_metadata,
                        "section": section,
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1

    return chunk_records
