import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from agent.config import PINECONE_INDEX_NAME

load_dotenv()  # Load env vars from .env file

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment.")

# Initialize Pinecone and model
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_candidate_by_phone(phone_number):
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
        print(f"Error querying Pinecone: {e}")
        return None