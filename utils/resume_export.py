"""Export helpers for tailored resume drafts."""

from fpdf import FPDF


def _safe_pdf_text(text: str) -> str:
    """Keep generated text compatible with the default PDF font."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _safe_segments(pdf: FPDF, text: str, max_width: float) -> list[str]:
    """Split text by measured character width so every segment fits the page."""
    text = _safe_pdf_text(text)
    if not text:
        return [""]

    segments: list[str] = []
    current = ""

    for character in text:
        candidate = current + character
        if current and pdf.get_string_width(candidate) > max_width:
            segments.append(current)
            current = character
        else:
            current = candidate

    if current:
        segments.append(current)
    return segments


def _write_wrapped(pdf: FPDF, text: str, line_height: int = 6) -> None:
    """Write measured-width segments using an explicit available width."""
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    for segment in _safe_segments(pdf, text, usable_width):
        # FPDF2 may leave the cursor at the right edge after multi_cell().
        # Reset it before every segment so subsequent lines are not clipped.
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, line_height, segment)
    pdf.set_x(pdf.l_margin)


def tailored_resume_to_pdf(report: dict) -> bytes:
    """Convert a tailored-resume report into a downloadable PDF."""
    pdf = FPDF(format="A4")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.set_title("CareerLens Tailored Resume")
    pdf.set_author("CareerLens")

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Tailored Resume", ln=1)
    pdf.ln(3)

    pdf.set_font("Helvetica", size=10)
    for line in report["tailored_resume"].splitlines():
        clean_line = line.strip()
        if not clean_line:
            pdf.ln(3)
            continue
        if clean_line.startswith("#"):
            heading = clean_line.lstrip("# ")
            pdf.set_font("Helvetica", "B", 13)
            _write_wrapped(pdf, heading, line_height=7)
            pdf.set_font("Helvetica", size=10)
        elif clean_line.startswith("-"):
            _write_wrapped(pdf, f"- {clean_line[1:].strip()}")
        else:
            _write_wrapped(pdf, clean_line)

    sections = [
        ("Changes Made", report.get("changes_made", [])),
        ("Preserved Facts", report.get("preserved_facts", [])),
        ("Unsupported Requirements Not Added", report.get("unsupported_requirements", [])),
    ]
    for heading, items in sections:
        if not items:
            continue
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        _write_wrapped(pdf, heading, line_height=7)
        pdf.set_font("Helvetica", size=10)
        for item in items:
            _write_wrapped(pdf, f"- {item}")

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)
