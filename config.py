# file: config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pinecone index name for vector database
PINECONE_INDEX_NAME = "rag-agent"

# Embedding model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Name of the embedding model
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}  # Model arguments (e.g., device to run on)

# Text splitter configuration
CHUNK_SIZE = 1000  # Maximum size of each text chunk
CHUNK_OVERLAP = 200  # Overlap size between consecutive chunks

# Groq LLM configuration
GROQ_MODEL_NAME = "llama3-8b-8192"  # Name of the Groq LLM model

def get_groq_api_key():
    """Retrieves the Groq API key from environment variables."""
    groq_api_key = os.getenv("GROQ_API_KEY")  # Fetch API key from environment
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY must be set in .env file")  # Raise error if not set
    return groq_api_key
