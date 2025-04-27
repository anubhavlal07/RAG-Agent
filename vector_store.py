import os, tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone as LCPinecone
from config import PINECONE_INDEX_NAME, CHUNK_SIZE, CHUNK_OVERLAP

def process_uploaded_pdfs(uploaded_files, embeddings):
    docs = []  # List to store all documents
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())  # Write uploaded file content to a temporary file
            loader = PyPDFLoader(tmp.name)  # Load PDF using PyPDFLoader
            docs.extend(loader.load())  # Append loaded documents to the list
        os.remove(tmp.name)  # Remove the temporary file

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # Define chunk size for splitting
        chunk_overlap=CHUNK_OVERLAP  # Define overlap between chunks
    )
    chunks = splitter.split_documents(docs)  # Split documents into chunks

    return LCPinecone.from_documents(
        index_name=PINECONE_INDEX_NAME,  # Pinecone index name
        documents=chunks,  # Chunks to be stored in Pinecone
        embedding=embeddings  # Embedding model to use
    )
