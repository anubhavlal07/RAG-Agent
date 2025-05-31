import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from agent.config import PINECONE_INDEX_NAME

load_dotenv()  # Load env vars from .env file

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment.")

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize the same embedding model used during ingestion
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_candidate_by_phone(phone_number):
    """
    Retrieves candidate metadata for the given phone number (as stored under is_phone_entry="true").
    Returns the metadata dict if found; otherwise, None.
    """
    query = f"Phone number: {phone_number}"
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
        print(f"Error querying Pinecone for phone entry: {e}")
        return None

def load_full_resume_text(phone_number):
    """
    Retrieves and concatenates all resume chunks for a candidate by phone number.
    1. Finds candidate_id via the phone-vector lookup.
    2. Queries Pinecone with a metadata filter to fetch all chunks (where is_phone_entry="false").
    3. Sorts chunks by chunk_id and joins their 'text' fields into a single string.
    Returns the concatenated resume text, or an empty string if not found.
    """
    # Step 1: Find candidate_id using phone lookup
    candidate_meta = get_candidate_by_phone(phone_number)
    if not candidate_meta:
        return ""

    candidate_id = candidate_meta.get("candidate_id")
    if not candidate_id:
        return ""

    # Step 2: Query all chunks for this candidate_id (filter on metadata)
    try:
        # We need a dummy vector since we're only using metadata filtering:
        # Use a zero‐vector of the same dimension (384) as during ingestion.
        zero_vector = np.zeros(384, dtype="float32").tolist()

        # Filter for resume chunks (is_phone_entry="false") for this candidate_id
        filter_criteria = {
            "candidate_id": {"$eq": candidate_id},
            "is_phone_entry": {"$eq": "false"}
        }

        # top_k: large enough to retrieve all chunks (e.g., 100)
        response = index.query(
            vector=zero_vector,
            top_k=100,
            include_metadata=True,
            filter=filter_criteria
        )

        # Collect chunk metadata
        chunks = []
        for match in response.matches:
            meta = match.metadata
            # Retrieve the chunk text and chunk_id
            text = meta.get("text", "")
            chunk_id = int(meta.get("chunk_id", "0"))
            chunks.append((chunk_id, text))

        if not chunks:
            return ""

        # Step 3: Sort by chunk_id and concatenate
        chunks.sort(key=lambda x: x[0])
        full_text = "\n".join(text for _, text in chunks)
        return full_text

    except Exception as e:
        print(f"Error retrieving resume chunks from Pinecone: {e}")
        return ""
