from langchain_huggingface import HuggingFaceEmbeddings  # Import the HuggingFaceEmbeddings class
from config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_KWARGS  # Import configuration constants

def initialize_embeddings():
    # Initialize and return HuggingFace embeddings with the specified model and parameters
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,  # Model name from configuration
        model_kwargs=EMBEDDING_MODEL_KWARGS  # Additional model parameters from configuration
    )
