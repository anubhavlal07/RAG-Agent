#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from embeddings import initialize_embeddings
from config import PINECONE_INDEX_NAME

def main():
    # Load environment variables
    load_dotenv()

    # 1️⃣ Instantiate the Pinecone client
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    # 2️⃣ Delete the existing index if it exists
    existing = pc.list_indexes().names()
    if PINECONE_INDEX_NAME in existing:
        print(f"Deleting existing index '{PINECONE_INDEX_NAME}'...")
        pc.delete_index(name=PINECONE_INDEX_NAME)
    else:
        print(f"No existing index named '{PINECONE_INDEX_NAME}' found. Creating new one.")

    # 3️⃣ Determine the embedding dimension dynamically
    embeddings = initialize_embeddings()
    # embed_query returns a vector; its length is the dimension
    test_vec = embeddings.embed_query("test")
    dim = len(test_vec)
    print(f"Embedding dimension detected: {dim}")

    # 4️⃣ Re-create the index with the same spec
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=spec
    )
    print(f"Index '{PINECONE_INDEX_NAME}' created with dimension={dim}, metric='cosine', spec={spec}")

if __name__ == "__main__":
    main()
