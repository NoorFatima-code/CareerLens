"""Original-style resume template and PDF rendering helpers."""

from __future__ import annotations

from html import escape
import re

from fpdf import FPDF

from .resume_export import _safe_segments


SECTION_NAMES = {
    "professional summary": "summary",
    "summary": "summary",
    "education": "education",
    "projects": "projects",
    "technical skills": "skills",
    "skills": "skills",
    "certifications": "certifications",
}


def _clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#{1,6}\s*", "", line)
    return line.strip()


def _section_key(line: str) -> str | None:
    normalized = re.sub(r"[:\s]+$", "", _clean_line(line).lower())
    return SECTION_NAMES.get(normalized)


def _parse_resume_text(text: str) -> dict[str, object]:
    """Parse the model's resume draft into template-friendly sections."""
    sections: dict[str, list[str]] = {
        "header": [],
        "summary": [],
        "education": [],
        "projects": [],
        "skills": [],
        "certifications": [],
    }
    current = "header"

    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if not line or line.lower() == "tailored resume":
            continue
        detected = _section_key(line)
        if detected:
            current = detected
            continue
        sections[current].append(line)

    return sections


def _render_project_block(lines: list[str]) -> str:
    projects: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for line in lines:
        is_bullet = line.startswith(("-", "•", "*") )
        if is_bullet and current:
            current["bullets"].append(line.lstrip("-•* "))
        elif is_bullet:
            if current is None:
                current = {"title": "Projects", "bullets": [line.lstrip("-•* ")]}
                projects.append(current)
        else:
            if current is not None and current.get("bullets"):
                current = None
            current = {"title": line, "bullets": []}
            projects.append(current)

    blocks: list[str] = []
    for project in projects:
        blocks.append('<div class="project">')
        blocks.append(f'<div class="project-title">{escape(str(project["title"]))}</div>')
        bullets = project["bullets"]
        if bullets:
            blocks.append("<ul>")
            blocks.extend(f"<li>{escape(str(bullet))}</li>" for bullet in bullets)
            blocks.append("</ul>")
        blocks.append("</div>")
    return "\n".join(blocks)


def _render_skills(lines: list[str]) -> str:
    rendered: list[str] = []
    for line in lines:
        if ":" in line:
            label, value = line.split(":", 1)
            rendered.append(
                f'<div class="skill-row"><strong>{escape(label.strip())}:</strong>'
                f' {escape(value.strip())}</div>'
            )
        else:
            rendered.append(f'<div class="skill-row">{escape(line)}</div>')
    return "\n".join(rendered)


def render_resume_html(resume_text: str) -> str:
    """Render a tailored resume draft using the original CV visual style."""
    sections = _parse_resume_text(resume_text)
    header = sections["header"]
    name = header[0] if header else "Tailored Resume"
    subtitle = header[1] if len(header) > 1 else ""
    contact = header[2] if len(header) > 2 else ""

    summary = " ".join(str(item) for item in sections["summary"])
    education = "<br>".join(escape(str(item)) for item in sections["education"])
    certifications = "<br>".join(
        f'<div class="cert-row">{escape(str(item))}</div>'
        for item in sections["certifications"]
    )

    def section(title: str, content: str) -> str:
        if not content.strip():
            return ""
        return (
            f'<section><h2>{escape(title)}</h2>'
            f'<div class="section-content">{content}</div></section>'
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{ size: A4; margin: 0.42in 0.56in 0.38in; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: #161616; font-family: Arial, Helvetica, sans-serif; font-size: 8.8pt; line-height: 1.18; }}
.resume {{ width: 100%; }}
.header {{ text-align: center; padding-bottom: 7px; border-bottom: 1px solid #2b638d; }}
.name {{ font-size: 19pt; font-weight: 700; letter-spacing: .2px; margin-bottom: 3px; }}
.subtitle {{ font-size: 8.4pt; color: #555; margin-bottom: 3px; }}
.contact {{ font-size: 7.7pt; color: #444; }}
section {{ margin-top: 6px; page-break-inside: avoid; }}
h2 {{ color: #1d5a83; font-size: 9.2pt; font-weight: 700; border-top: 1px solid #2b638d; padding-top: 4px; margin: 0 0 4px; letter-spacing: .1px; }}
.section-content {{ padding-left: 0; }}
.project {{ page-break-inside: avoid; margin-bottom: 4px; }}
.project-title {{ font-weight: 700; margin-bottom: 1px; }}
ul {{ margin: 1px 0 0 13px; padding: 0; }}
li {{ margin: 0 0 1px; padding-left: 1px; }}
.skill-row {{ margin-bottom: 1px; }}
.cert-row {{ margin-bottom: 2px; }}
strong {{ font-weight: 700; }}
</style>
</head>
<body>
<div class="resume">
<header class="header">
<div class="name">{escape(name)}</div>
<div class="subtitle">{escape(subtitle)}</div>
<div class="contact">{escape(contact)}</div>
</header>
{section("PROFESSIONAL SUMMARY", escape(summary))}
{section("EDUCATION", education)}
{section("PROJECTS", _render_project_block(sections["projects"]))}
{section("TECHNICAL SKILLS", _render_skills(sections["skills"]))}
{section("CERTIFICATIONS", certifications)}
</div>
</body>
</html>"""


def tailored_resume_to_pdf(resume_text: str) -> bytes:
    """Render a styled, original-style resume into PDF bytes without native GTK dependencies."""
    sections = _parse_resume_text(resume_text)
    pdf = FPDF(format="A4")
    pdf.set_margins(left=15, top=13, right=15)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    def write_text(
        text: str,
        size: int = 8,
        bold: bool = False,
        color: tuple[int, int, int] = (22, 22, 22),
        line_height: int = 5,
        centered: bool = False,
    ) -> None:
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_text_color(*color)
        for segment in _safe_segments(pdf, text, usable_width):
            pdf.set_x(pdf.l_margin)
            if centered:
                pdf.cell(usable_width, line_height, segment, align="C", ln=1)
            else:
                pdf.multi_cell(usable_width, line_height, segment)
        pdf.set_x(pdf.l_margin)

    def section_heading(title: str) -> None:
        pdf.ln(2)
        pdf.set_draw_color(43, 99, 141)
        pdf.set_line_width(0.25)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(1)
        write_text(title, size=9, bold=True, color=(29, 90, 131), line_height=5)

    header = sections["header"]
    if header:
        write_text(header[0], size=18, bold=True, line_height=8, centered=True)
    if len(header) > 1:
        write_text(header[1], size=8, color=(85, 85, 85), line_height=4, centered=True)
    if len(header) > 2:
        write_text(header[2], size=7, color=(70, 70, 70), line_height=4, centered=True)

    pdf.set_draw_color(43, 99, 141)
    pdf.set_line_width(0.35)
    pdf.line(pdf.l_margin, pdf.get_y() + 2, pdf.w - pdf.r_margin, pdf.get_y() + 2)
    pdf.ln(5)

    if sections["summary"]:
        section_heading("PROFESSIONAL SUMMARY")
        write_text(" ".join(sections["summary"]), size=8, line_height=4)

    if sections["education"]:
        section_heading("EDUCATION")
        for line in sections["education"]:
            write_text(line, size=8, line_height=4)

    if sections["projects"]:
        section_heading("PROJECTS")
        current_title = None
        for line in sections["projects"]:
            is_bullet = line.startswith(("-", "•", "*"))
            if is_bullet:
                write_text(f"• {line.lstrip('-•* ').strip()}", size=7.5, line_height=4)
            else:
                current_title = line
                write_text(current_title, size=8, bold=True, line_height=4.5)

    if sections["skills"]:
        section_heading("TECHNICAL SKILLS")
        for line in sections["skills"]:
            if ":" in line:
                label, value = line.split(":", 1)
                write_text(f"{label.strip()}: {value.strip()}", size=7.5, line_height=4)
            else:
                write_text(line, size=7.5, line_height=4)

    if sections["certifications"]:
        section_heading("CERTIFICATIONS")
        for line in sections["certifications"]:
            write_text(line, size=7.5, line_height=4)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)
