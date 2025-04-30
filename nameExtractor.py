# nameExtractor.py

import re
from typing import Optional

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def extract_name(resume_text: str) -> Optional[str]:
    """
    1) Find email; extract its local-part.
    2) Split on . _ - ; drop numeric-only bits.
    3) Title-case each.
    4) If that produces a single chunk, try to match a two-word Title-case name
       in the resume whose lowercase concatenation == that chunk lowercased.
    5) Return the best match, or the chunked name.
    """
    match = EMAIL_REGEX.search(resume_text)
    if not match:
        return None

    local = match.group(0).split("@", 1)[0]
    # 2️⃣ Split into pieces, drop numeric-only
    parts = [p for p in re.split(r"[._\-]+", local) if p and not p.isdigit()]
    # Title-case each piece
    title_parts = [p.capitalize() for p in parts]
    if not title_parts:
        return None

    # Join into candidate string
    candidate = " ".join(title_parts)
    # If multi-part (e.g. ["Anubhav","Lal"]), we're done.
    if len(title_parts) > 1:
        return candidate

    # 4️⃣ Single chunk—attempt to find proper two-word name in resume
    single = title_parts[0].lower()
    # Find all adjacent Title-case pairs in the resume
    name_pattern = re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b")
    for m in name_pattern.finditer(resume_text):
        first, last = m.group(1), m.group(2)
        if (first + last).lower() == single:
            return f"{first} {last}"

    # No multi-word match found; return the single chunk
    return candidate
