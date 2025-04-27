import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

PINECONE_INDEX_NAME = "rag-agent"  # Name of the Pinecone index
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Name of the embedding model
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}  # Configuration for the embedding model
CHUNK_SIZE = 1000  # Size of each text chunk
CHUNK_OVERLAP = 200  # Overlap size between text chunks
GROQ_MODEL_NAME = "llama3-8b-8192"  # Name of the Groq model

def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")  # Retrieve the Groq API key from environment variables
    if not key:
        raise ValueError("GROQ_API_KEY must be set")  # Raise an error if the API key is not set
    return key
