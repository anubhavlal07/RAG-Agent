import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

# Toggle between Pinecone and Postgres via environment variable
# True for Pinecone, False for Postgres
# True and False are case-sensitive
USE_PINECONE = os.getenv("USE_PINECONE")

if USE_PINECONE:
    # ----- Pinecone Imports -----
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    from agent.config import PINECONE_INDEX_NAME

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment.")

    # Initialize Pinecone client and embedding model
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    model = SentenceTransformer("all-MiniLM-L6-v2")

else:
    # ----- Postgres Imports -----
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
    PG_PORT = os.getenv("POSTGRES_PORT", "5432")
    PG_USER = os.getenv("POSTGRES_USER")
    PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    PG_DATABASE = os.getenv("POSTGRES_DB")
    PG_TABLE_NAME = os.getenv("POSTGRES_TABLENAME")

    if not all([PG_USER, PG_PASSWORD, PG_DATABASE]):
        raise ValueError("Postgres credentials (PG_USER, PG_PASSWORD, PG_DATABASE) must be set in environment.")

def _normalize_phone(phone: str) -> str:
    digits = "".join(filter(str.isdigit, phone))
    # Take the last 10 digits if length > 10, else use as-is
    return digits[-10:] if len(digits) >= 10 else digits


def get_candidate_by_phone(raw_phone: str) -> dict:
    """
    Fetch candidate metadata by phone number.
    If USE_PINECONE is True, query the Pinecone vector index.
    Otherwise, query the local Postgres database (parser_parsedresume).
    Returns a dict of metadata or None if not found.
    """
    normalized = _normalize_phone(raw_phone)
    if not normalized:
        return None

    if USE_PINECONE:
        # ----- Pinecone-based lookup -----
        # Only pass the normalized last-10-digits to the model prompt
        query = f"Phone number: {normalized}"
        query_vector = model.encode(query).astype("float32").tolist()

        try:
            result = index.query(
                vector=query_vector,
                top_k=1,
                include_metadata=True
            )
            if result.matches:
                match = result.matches[0]
                meta = match.metadata
                if meta.get("is_phone_entry") == "true":
                    return meta
            return None
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return None

    else:
        # ----- Postgres-based lookup -----
        try:
            conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASSWORD,
                database=PG_DATABASE
            )
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Compare last 10 digits of stored phone (strip non-digits) to normalized input
            query = f"""
            SELECT
                name,
                email,
                phone,
                location,
                experience_years,
                skills,
                current_role,
                company,
                education,
                projects,
                work_experience
            FROM {PG_TABLE_NAME}
            WHERE RIGHT(regexp_replace(phone, '\\D', '', 'g'), 10) = %s
            LIMIT 1
            """
            cur.execute(query, (normalized,))
            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                # Convert JSONB fields to Python structures as needed
                candidate = dict(row)
                return candidate

            return None

        except psycopg2.errors.UndefinedTable:
            print(f"Error: Postgres table '{PG_TABLE_NAME}' does not exist.")
            return None

        except Exception as e:
            print(f"Error querying Postgres: {e}")
            return None


def load_full_resume_text(raw_phone: str) -> str:
    """
    Retrieve the candidate's full resume text as a single concatenated string.
    - Pinecone: gather all chunks by candidate_id.
    - Postgres: reconstruct a text blob from fields in parser_parsedresume.
    Returns the resume text or None if not found.
    """
    normalized = _normalize_phone(raw_phone)
    if not normalized:
        return None

    if USE_PINECONE:
        # ----- Pinecone-based retrieval -----
        query = f"Phone number: {normalized}"
        query_vector = model.encode(query).astype("float32").tolist()

        try:
            # 1) Find the candidate_id via the phone entry
            phone_result = index.query(
                vector=query_vector,
                top_k=1,
                include_metadata=True
            )
            if not phone_result.matches:
                return None

            phone_meta = phone_result.matches[0].metadata
            if phone_meta.get("is_phone_entry") != "true":
                return None

            candidate_id = phone_meta.get("candidate_id")

            # 2) Fetch all chunks for that candidate_id
            slices = index.query(
                vector=query_vector,
                top_k=100,
                include_metadata=True,
                filter={"candidate_id": {"$eq": candidate_id}}
            )

            if not slices.matches:
                return None

            # 3) Sort by chunk_id and concatenate
            chunks = []
            for m in slices.matches:
                meta = m.metadata
                if meta.get("chunk_id") is not None and meta.get("text"):
                    try:
                        cid = int(meta.get("chunk_id"))
                    except (ValueError, TypeError):
                        cid = -1
                    chunks.append((cid, meta["text"]))

            chunks.sort(key=lambda x: x[0])
            full_text = "\n".join(chunk_text for _, chunk_text in chunks)
            return full_text

        except Exception as e:
            print(f"Error retrieving resume from Pinecone: {e}")
            return None

    else:
        # ----- Postgres-based retrieval -----
        try:
            conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASSWORD,
                database=PG_DATABASE
            )
            cur = conn.cursor(cursor_factory=RealDictCursor)

            query = f"""
            SELECT 
                name,
                email,
                location,
                experience_years,
                skills,
                current_role,
                company,
                education,
                projects,
                work_experience
            FROM {PG_TABLE_NAME}
            WHERE RIGHT(regexp_replace(phone, '\\D', '', 'g'), 10) = %s
            LIMIT 1
            """
            cur.execute(query, (normalized,))
            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return None

            # Build a text blob from fields
            name = row.get("name", "")
            email = row.get("email", "")
            location = row.get("location", "")
            exp_years = row.get("experience_years", 0)
            skills = row.get("skills", [])
            current_role = row.get("current_role", "")
            company = row.get("company", "")
            education = row.get("education", [])
            projects = row.get("projects", [])
            work_experience = row.get("work_experience", [])

            parts = [
                f"Name: {name}",
                f"Email: {email}",
                f"Location: {location}",
                f"Experience: {exp_years} years",
                f"Current Role: {current_role} at {company}",
                f"Skills: {', '.join(skills) if isinstance(skills, list) else str(skills)}",
                f"Education: {json.dumps(education)}",
                f"Projects: {json.dumps(projects)}",
                f"Work Experience: {json.dumps(work_experience)}"
            ]
            return "\n".join(parts)

        except psycopg2.errors.UndefinedTable:
            print(f"Error: Postgres table '{PG_TABLE_NAME}' does not exist.")
            return None

        except Exception as e:
            print(f"Error retrieving resume from Postgres: {e}")
            return None