import os
import time
import requests
import numpy as np
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# Set the URL of your resume parsing API
API_URL = "http://127.0.0.1:8000/api/parse/"
# Folder containing all resumes to ingest
RESUME_FOLDER = "resumes"

def generate_resume_text(data):
    """
    Fallback text if the parsing API did not supply a combined resume_text.
    """
    parts = [
        f"Name: {data.get('name', '')}",
        f"Phone: {data.get('phone', '')}",
        f"Email: {data.get('email', '')}",
        f"Location: {data.get('location', '')}",
        f"Experience: {data.get('experience_years', 0)} years",
        f"Skills: {', '.join(data.get('skills', []))}",
        f"Current Role: {data.get('current_role', '')} at {data.get('company', '')}",
        f"Education: {data.get('education', '')}",
    ]
    for project in data.get("projects", []):
        parts.append(f"Project: {project.get('title', '')} - {project.get('description', '')}")
    return "\n".join(parts)

def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        # Wait briefly for index readiness
        time.sleep(60)
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    return pc.Index(PINECONE_INDEX_NAME)

def _sanitize_metadata(value):
    """
    Ensure the metadata value is one of:
      - primitive: str, int, float, bool
      - list of strings
    Otherwise, convert to JSON string.
    """
    # Primitive types are fine
    if isinstance(value, (str, int, float, bool)):
        return value

    # List of strings is fine
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value

    # Otherwise, convert to JSON string
    try:
        return json.dumps(value)
    except Exception:
        return str(value)

def process_candidate(model, pinecone_index, candidate_id, candidate_data):
    """
    Embeds and upserts all chunks of the candidate's resume_text,
    and also stores a phone-entry vector. 
    Sanitizes metadata to keep it compatible with Pinecone.
    """
    name = candidate_data.get("name", "")
    phone_number = candidate_data.get("phone", "")
    resume_text = candidate_data.get("resume_text", "")

    # Must have at least name and phone to index
    if not name or not phone_number:
        print(f"Missing required data (name or phone) for candidate {candidate_id}")
        return False

    # If parsing API didn't supply a combined resume_text, generate one
    if not resume_text:
        resume_text = generate_resume_text(candidate_data)

    # Split into 1000-char chunks
    chunk_size = 1000
    chunks = [resume_text[i:i+chunk_size] for i in range(0, len(resume_text), chunk_size)]
    if not chunks:
        chunks = [resume_text]

    vectors = []

    # Build sanitized base metadata from all parsed fields
    base_metadata = {}
    for key, val in candidate_data.items():
        base_metadata[key] = _sanitize_metadata(val)

    # For each chunk, compute embedding and create a vector entry
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).astype("float32").tolist()
        vector_id = f"{candidate_id}_chunk_{i}"

        metadata = {
            "candidate_id": candidate_id,
            "chunk_id": str(i),
            "text": chunk,
            "is_phone_entry": "false"
        }

        # Merge sanitized parsed fields
        metadata.update(base_metadata)

        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })

    # Now add a separate 'phone' vector so we can look up by phone number
    phone_query = f"Phone number: {phone_number}"
    phone_embedding = model.encode(phone_query).astype("float32").tolist()
    phone_vector_id = f"{candidate_id}_phone"

    phone_metadata = {
        "candidate_id": candidate_id,
        "chunk_id": "-1",          # Sentinel for phone entry
        "text": phone_query,
        "is_phone_entry": "true"
    }
    # Merge sanitized parsed fields
    phone_metadata.update(base_metadata)

    vectors.append({
        "id": phone_vector_id,
        "values": phone_embedding,
        "metadata": phone_metadata
    })

    # Upsert vectors to Pinecone in batches of 100
    batch_size = 100
    for start_idx in range(0, len(vectors), batch_size):
        batch = vectors[start_idx:start_idx + batch_size]
        try:
            pinecone_index.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting batch {start_idx//batch_size}: {e}")
            return False

    return True

def upload_resume_and_get_data(api_url, resume_path):
    """
    Sends a resume file to your parsing API and returns the parsed JSON as a Python dict.
    """
    try:
        with open(resume_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(api_url, files=files)
            response.raise_for_status()
            return response.json()  # Should be a dict with all resume fields
    except Exception as e:
        print(f"Error parsing {resume_path}: {e}")
        return None

def ingest_all_resumes(folder_path, api_url):
    """
    Loops through all PDF/DOCX files in `folder_path`, parses each via your API,
    and indexes every parsed field + chunks into Pinecone.
    """
    # Auto-create resume folder if missing
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created missing folder: {folder_path}")
        print("⚠️ Please add resume files (.pdf or .docx) to this folder and re-run the script.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Initializing Pinecone...")
    pinecone_index = initialize_pinecone()

    resume_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.pdf', '.docx'))
    ]

    if not resume_files:
        print(f"⚠️ No .pdf or .docx files found in: {folder_path}")
        return

    print(f"Found {len(resume_files)} resumes to process.\n")

    success_count = 0
    for i, resume_file in enumerate(tqdm(resume_files, desc="Processing Resumes")):
        full_path = os.path.join(folder_path, resume_file)
        parsed_data = upload_resume_and_get_data(api_url, full_path)

        if parsed_data:
            # Use time.time() to generate a unique candidate_id
            candidate_id = f"candidate_{int(time.time())}_{i+1}"
            if process_candidate(model, pinecone_index, candidate_id, parsed_data):
                success_count += 1
            else:
                print(f"❌ Failed to index: {resume_file}")
        else:
            print(f"⚠️ Skipped: {resume_file}")
    if success_count == 0:
        print("❗ No resumes were successfully indexed. Please check the logs for errors.")
        return
    else:
        print(f"\nIndexed {success_count} resumes successfully out of {len(resume_files)} total files.")

if __name__ == "__main__":
    ingest_all_resumes(RESUME_FOLDER, API_URL)
