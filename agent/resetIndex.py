import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-resume-index")  # Pinecone index name from env
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model name from env
EMBEDDING_MODEL_KWARGS = {"device": "cpu"}  # Embedding model config

def initialize_embeddings():
    # Initialize HuggingFace embeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=EMBEDDING_MODEL_KWARGS
    )

def main():
    # Initialize Pinecone client
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    existing = pc.list_indexes().names()  # List existing indexes
    if PINECONE_INDEX_NAME in existing:
        print(f"Deleting existing index '{PINECONE_INDEX_NAME}'...")
        pc.delete_index(name=PINECONE_INDEX_NAME)  # Delete existing index
    else:
        print(f"No existing index named '{PINECONE_INDEX_NAME}' found. Creating new one.")

    embeddings = initialize_embeddings()  # Create embedding model
    test_vec = embeddings.embed_query("test")  # Get embedding vector for test query
    dim = len(test_vec)  # Determine embedding dimension
    print(f"Embedding dimension detected: {dim}")

    spec = ServerlessSpec(cloud="aws", region="us-east-1")  # Pinecone serverless spec
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=spec
    )  # Create new index
    print(f"Index '{PINECONE_INDEX_NAME}' created with dimension={dim}, metric='cosine', spec={spec}")

if __name__ == "__main__":
    main()
