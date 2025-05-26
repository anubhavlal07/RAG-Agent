import os
import time
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# Set the URL of your resume parsing API
API_URL = "http://127.0.0.1:8000/api/parse/"
# Folder containing all resumes to ingest
RESUME_FOLDER = "resumes"

def generate_resume_text(data):
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
        parts.append(f"Project: {project['title']} - {project['description']}")
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
        time.sleep(60)
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    return pc.Index(PINECONE_INDEX_NAME)

def process_candidate(model, pinecone_index, candidate_id, candidate_data):
    name = candidate_data.get("name", "")
    phone_number = candidate_data.get("phone", "")
    resume_text = candidate_data.get("resume_text", "")

    if not name or not phone_number:
        print(f"Missing required data for candidate {candidate_id}")
        return False

    if not resume_text:
        resume_text = generate_resume_text(candidate_data)

    chunk_size = 1000
    chunks = [resume_text[i:i+chunk_size] for i in range(0, len(resume_text), chunk_size)]
    if not chunks:
        chunks = [resume_text]

    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).astype("float32").tolist()
        vector_id = f"{candidate_id}_chunk_{i}"

        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "candidate_id": candidate_id,
                "name": name,
                "phone": phone_number,
                "chunk_id": str(i),
                "text": chunk,
                "is_phone_entry": "false"
            }
        })

    # Phone embedding
    phone_query = f"Phone number: {phone_number}"
    phone_embedding = model.encode(phone_query).astype("float32").tolist()
    phone_vector_id = f"{candidate_id}_phone"

    vectors.append({
        "id": phone_vector_id,
        "values": phone_embedding,
        "metadata": {
            "candidate_id": candidate_id,
            "name": name,
            "phone": phone_number,
            "chunk_id": "-1",
            "text": phone_query,
            "is_phone_entry": "true"
        }
    })

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            pinecone_index.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting batch {i//batch_size}: {e}")
            return False

    return True

def upload_resume_and_get_data(api_url, resume_path):
    try:
        with open(resume_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(api_url, files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error parsing {resume_path}: {e}")
        return None

def ingest_all_resumes(folder_path, api_url):
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
            candidate_id = f"candidate_{int(time.time())}_{i+1}"
            if process_candidate(model, pinecone_index, candidate_id, parsed_data):
                success_count += 1
            else:
                print(f"❌ Failed to index: {resume_file}")
        else:
            print(f"⚠️ Skipped: {resume_file}")

    print(f"\n✅ Successfully indexed {success_count} out of {len(resume_files)} resumes.")

if __name__ == "__main__":
    ingest_all_resumes(RESUME_FOLDER, API_URL)
