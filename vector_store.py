import os
import re
import tempfile
from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone as LCPinecone
from config import PINECONE_INDEX_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from dotenv import load_dotenv

load_dotenv()

# Create an instance of Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

def _sanitize(name: str) -> str:
    """Sanitize candidate name to make it compatible with namespaces."""
    return re.sub(r"[^0-9a-z]+", "-", name.strip().lower())

def process_uploaded_pdfs(uploaded_files, embeddings, candidate_name):
    """
    Load PDFs, chunk them, and upsert into the existing Pinecone index under a candidate-specific namespace.
    """
    # Check if the index exists; if not, raise an error
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Index '{PINECONE_INDEX_NAME}' does not exist in Pinecone.")

    # Load and chunk documents
    docs = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            docs.extend(PyPDFLoader(tmp.name).load())
        os.remove(tmp.name)

    # Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # Upsert the chunks into the Pinecone index under the candidate's namespace
    namespace = _sanitize(candidate_name)
    return LCPinecone.from_documents(
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
        documents=chunks,
        embedding=embeddings
    )
