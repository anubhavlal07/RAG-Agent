import os
import time
import json
import requests
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

API_URL = "http://127.0.0.1:8000/api/parse/"
RESUME_FOLDER = "resumes"
USE_PINECONE = os.getenv("USE_PINECONE")

# Pinecone config vars (will only be used if USE_PINECONE=True)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resumes-index")

def _sanitize_metadata(value):
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return str(value)

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
        parts.append(f"Project: {project.get('title', '')} - {project.get('description', '')}")
    return "\n".join(parts)

def upload_resume_and_get_data(api_url, resume_path):
    try:
        with open(resume_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(api_url, files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"❌ Error parsing {resume_path}: {e}")
        return None

class PineconeIngestor:
    def __init__(self, api_key, index_name):
        from pinecone import Pinecone, ServerlessSpec

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(60)
        else:
            print(f"Using existing Pinecone index: {self.index_name}")

        self.index = self.pc.Index(self.index_name)

    def embed_and_upsert(self, model, candidate_id, candidate_data):
        phone_number = candidate_data.get("phone", "")
        resume_text = candidate_data.get("resume_text", "") or generate_resume_text(candidate_data)

        chunk_size = 1000
        chunks = [resume_text[i:i+chunk_size] for i in range(0, len(resume_text), chunk_size)] or [resume_text]
        base_metadata = {k: _sanitize_metadata(v) for k, v in candidate_data.items()}

        vectors = []
        for i, chunk in enumerate(chunks):
            vector_id = f"{candidate_id}_chunk_{i}"
            embedding = model.encode(chunk).astype("float32").tolist()
            metadata = {
                "candidate_id": candidate_id,
                "chunk_id": str(i),
                "text": chunk,
                "is_phone_entry": "false"
            }
            metadata.update(base_metadata)
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

        phone_vector_id = f"{candidate_id}_phone"
        phone_text = f"Phone number: {phone_number}"
        phone_embedding = model.encode(phone_text).astype("float32").tolist()
        phone_metadata = {
            "candidate_id": candidate_id,
            "chunk_id": "-1",
            "text": phone_text,
            "is_phone_entry": "true"
        }
        phone_metadata.update(base_metadata)
        vectors.append({
            "id": phone_vector_id,
            "values": phone_embedding,
            "metadata": phone_metadata
        })

        # Upsert in batches
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start:start+batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"❌ Error upserting to Pinecone: {e}")
                return False
        return True

def ingest_all_resumes(folder_path, api_url):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created missing folder: {folder_path}")
        print("⚠️ Please add resumes to this folder.")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')

    pinecone_ingestor = None
    if USE_PINECONE:
        print("🔗 Pinecone enabled. Initializing...")
        pinecone_ingestor = PineconeIngestor(PINECONE_API_KEY, PINECONE_INDEX_NAME)
    else:
        print("🚫 Pinecone disabled. Will only parse resumes.")

    resume_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.pdf', '.docx'))
    ]

    if not resume_files:
        print("⚠️ No resumes found in the folder.")
        return

    print(f"📄 Found {len(resume_files)} resumes.\n")

    success_count = 0
    for i, resume_file in enumerate(tqdm(resume_files, desc="Processing Resumes")):
        full_path = os.path.join(folder_path, resume_file)
        parsed_data = upload_resume_and_get_data(api_url, full_path)

        if parsed_data:
            candidate_id = f"candidate_{int(time.time())}_{i+1}"
            if USE_PINECONE:
                ok = pinecone_ingestor.embed_and_upsert(model, candidate_id, parsed_data)
                if ok:
                    success_count += 1
                else:
                    print(f"⚠️ Failed indexing {resume_file}")
            else:
                print(f"✅ Parsed (no Pinecone): {resume_file}")
                success_count += 1
        else:
            print(f"❌ Failed to parse {resume_file}")

    print(f"\n🎉 Done. Processed {success_count}/{len(resume_files)} resume(s).")

if __name__ == "__main__":
    ingest_all_resumes(RESUME_FOLDER, API_URL)