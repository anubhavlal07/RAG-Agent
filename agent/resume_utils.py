def extract_resume_text(candidate_data):
    parts = [
        f"Name: {candidate_data.get('name', '')}",
        f"Experience: {candidate_data.get('experience_years', 0)} years",
        f"Skills: {', '.join(candidate_data.get('skills', []))}",
        f"Current Role: {candidate_data.get('current_role', '')} at {candidate_data.get('company', '')}",
        f"Education: {candidate_data.get('education', '')}",
    ]
    for project in candidate_data.get("projects", []):
        parts.append(f"Project: {project['title']} - {project['description']}")
    return "\n".join(parts)