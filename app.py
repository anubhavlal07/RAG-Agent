import os
import tempfile
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import Pinecone as LCPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from config import PINECONE_INDEX_NAME, get_groq_api_key
from embeddings import initialize_embeddings
from hr_agent import create_hr_crew, ask_next_question_with_retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="📄 RAG Chat + HR Interviewer", layout="wide")
st.title("📄 RAG Chat with Groq, Pinecone & CrewAI HR Interviewer")

# Initialize embeddings if not already in session state
if "embeddings" not in st.session_state:
    st.session_state.embeddings = initialize_embeddings()

# File uploader for PDF documents
uploaded = st.file_uploader(
    "Upload PDF(s) – candidate’s resume",
    type="pdf", accept_multiple_files=True
)

# Process and load uploaded PDFs
if uploaded and st.button("📚 Process & Load"):
    try:
        with st.spinner("Processing PDFs..."):
            docs = []
            for f in uploaded:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    docs.extend(PyPDFLoader(tmp.name).load())  # Load PDF content
                os.remove(tmp.name)  # Clean up temporary file

            # Split documents into smaller chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            # Create a vector store using Pinecone
            vector_store = LCPinecone.from_documents(
                index_name=PINECONE_INDEX_NAME,
                documents=chunks,
                embedding=st.session_state.embeddings
            )

            # Initialize the language model
            llm = ChatGroq(
                groq_api_key=get_groq_api_key(),
                model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
            )

            # Set up conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

            # Create a conversational retrieval chain
            conv_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(),
                memory=memory
            )

            # Store objects in session state
            st.session_state.vector_store = vector_store
            st.session_state.conversation = conv_chain
            st.session_state.messages = []
            st.session_state.hr_crew = create_hr_crew(conv_chain)  # Initialize HR agent
            st.session_state.hr_history = []

        st.success("✅ Documents loaded—ask away!")
        st.info("⚙️ HR Interviewer ready—click below!")

    except Exception as e:
        # Log and display any errors during processing
        logger.exception("Error during PDF processing")
        st.exception(e)

# RAG Chat interface
if "conversation" in st.session_state:
    st.markdown("### 📖 RAG Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])  # Display chat messages

    # Handle user input for RAG chat
    if prompt := st.chat_input("Ask about your PDFs"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            # Invoke the conversational chain
            resp = st.session_state.conversation.invoke({"question": prompt})
            answer = resp["answer"]
        except Exception as e:
            # Log and display any errors during invocation
            logger.exception("Error in RAG chain invoke")
            st.exception(e)
            answer = "⚠️ Sorry, something went wrong."

        # Append assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# HR Interviewer interface
if "hr_crew" in st.session_state:
    st.markdown("---\n### 🤖 HR Interviewer")
    if st.button("📝 Ask Next HR Question"):
        try:
            # Generate the next HR question
            hr_resp = ask_next_question_with_retry(st.session_state.hr_crew, retries=2)
        except Exception as e:
            # Log and display any errors during HR question generation
            logger.exception("HR agent failed")
            st.exception(e)
            hr_resp = "⚠️ Unable to generate HR question."
        st.session_state.hr_history.append({"role": "assistant", "content": hr_resp})

    # Display HR interview history
    for msg in st.session_state.hr_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
