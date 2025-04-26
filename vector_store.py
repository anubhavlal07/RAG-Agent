# file: vector_store.py
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone as LCPinecone
from config import PINECONE_INDEX_NAME

def process_uploaded_pdfs(uploaded_files, embeddings):
    docs = []  # List to store all loaded documents
    for f in uploaded_files:
        # Write uploaded file to a temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())  # Write file content to temp file
            loader = PyPDFLoader(tmp.name)  # Initialize PDF loader
            docs.extend(loader.load())  # Load documents from PDF
        os.remove(tmp.name)  # Clean up temporary file

    # Split documents into smaller chunks for processing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum size of each chunk
        chunk_overlap=200  # Overlap between chunks
    )
    chunks = splitter.split_documents(docs)  # Split documents into chunks

    # Upsert document embeddings into Pinecone index
    vector_store = LCPinecone.from_documents(
        index_name=PINECONE_INDEX_NAME,  # Name of the Pinecone index
        documents=chunks,  # Chunks of documents to upsert
        embedding=embeddings  # Embedding model to use
    )
    return vector_store  # Return the vector store instance