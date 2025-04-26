# file: embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings  # Import HuggingFace embeddings class
from config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_KWARGS  # Import configuration variables

def initialize_embeddings():
    """Initialize and return the HuggingFace embeddings model."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,  # Specify the model name from config
        model_kwargs=EMBEDDING_MODEL_KWARGS  # Pass additional model arguments from config
    )
    return embeddings  # Return the initialized embeddings model
