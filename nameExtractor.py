import re
from typing import Optional

# Regular expression to match email addresses
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def extract_name(resume_text: str) -> Optional[str]:
    # Search for an email address in the resume text
    match = EMAIL_REGEX.search(resume_text)
    if not match:
        return None

    # Extract the local part of the email (before the @ symbol)
    local = match.group(0).split("@", 1)[0]
    # Split the local part into components, ignoring digits and empty parts
    parts = [p for p in re.split(r"[._\-]+", local) if p and not p.isdigit()]
    # Capitalize each part to form title case
    title_parts = [p.capitalize() for p in parts]
    if not title_parts:
        return None

    # Join the parts to form a candidate name
    candidate = " ".join(title_parts)
    if len(title_parts) > 1:
        return candidate

    # Handle single-part names by matching against full names in the text
    single = title_parts[0].lower()
    name_pattern = re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b")
    for m in name_pattern.finditer(resume_text):
        first, last = m.group(1), m.group(2)
        # Check if the single part matches the concatenated first and last names
        if (first + last).lower() == single:
            return f"{first} {last}"

    # Return the single-part candidate name if no match is found
    return candidate
