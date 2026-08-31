"""Prompt templates for the V2 Skill Gap Analyzer."""

SKILL_GAP_SYSTEM_PROMPT = """
You are CareerLens Skill Gap Analyzer for job applicants.
Compare the job description against the CV using only the provided text.
Do not invent skills, experience, achievements, certifications, or job titles.
A skill is missing only when it is not explicitly supported by the CV text.
Use related_skills when the CV shows adjacent or transferable experience.
Every skill item must include evidence, source, and section.
Learning recommendations must be suggestions, not claims about completed work.
Return valid JSON with exactly these keys:
matching_skills, related_skills, missing_skills, learning_recommendations.
""".strip()


def build_skill_gap_prompt(cv_text: str, job_description: str) -> str:
    """Build the comparison prompt from cleaned CV and job-description text."""
    return f"""
CV TEXT:
{cv_text}

JOB DESCRIPTION TEXT:
{job_description}

Compare the two documents and return the requested structured skill-gap report.
""".strip()
