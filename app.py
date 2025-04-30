import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import time

from embeddings import initialize_embeddings
from vector_store import process_uploaded_pdfs
from llm_chain import create_conversation_chain
from hr_agent import create_hr_agent_chain
from hr_utils import HRInterview
from langchain.document_loaders import PyPDFLoader
from nameExtractor import extract_name

load_dotenv()  # Load environment variables from .env file

def main():
    st.set_page_config(page_title="RAG+HR Interview Simulator", layout="wide")  # Set Streamlit page configuration
    st.title("📄 RAG & HR Interview Simulator")  # Display the app title

    # Initialize embeddings if not already in session state
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = initialize_embeddings()

    # File uploader for PDF resumes
    uploaded = st.file_uploader("Upload a PDF resume", type="pdf")
    if uploaded and st.button("Load Resume"):  # Process the uploaded resume when button is clicked
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())  # Write uploaded file to a temporary file
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)  # Load the PDF using PyPDFLoader
        docs = loader.load()  # Extract documents from the PDF
        os.remove(tmp_path)  # Remove the temporary file
        uploaded.seek(0)  # Reset the file pointer

        # Combine all pages' content into a single string
        full_text = "\n\n".join(d.page_content for d in docs)
        name = extract_name(full_text) or "candidate"  # Extract candidate's name or use a default value
        vs = process_uploaded_pdfs(
            uploaded_files=[uploaded],
            embeddings=st.session_state.embeddings,
            candidate_name=name
        )  # Process the uploaded PDFs into a vector store
        st.session_state.vs = vs

        # Create RAG conversation chain
        st.session_state.rag_chain = create_conversation_chain(vs)

        # Create HR agent chain and memory
        hr_chain, hr_memory = create_hr_agent_chain(vs)
        docs_for_snip = vs.as_retriever().get_relevant_documents("overview")  # Retrieve relevant documents
        snippets = "\n".join(d.page_content[:300] for d in docs_for_snip)  # Extract snippets for HR interview

        # Initialize HRInterview object
        hr = HRInterview(
            resume_text=full_text,
            hr_chain=hr_chain,
            hr_memory=hr_memory,
            resume_snippets=snippets
        )
        st.session_state.hr = hr

        # Initialize chat messages and stage
        st.session_state.messages = [
            {"role": "assistant", "content": hr.confirm_name_question()}
        ]
        st.session_state.stage = "confirm_name"

    # Display chat messages if available in session state
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):  # Display each message in the chat
                st.markdown(msg["content"])

        # Handle user input in the chat
        if reply := st.chat_input("Your answer…"):
            st.session_state.messages.append({"role": "user", "content": reply})  # Append user reply to messages
            with st.chat_message("user"):
                st.markdown(reply)

            hr: HRInterview = st.session_state.hr  # Retrieve HRInterview object from session state

            if hr.stage == "confirm_name":  # Handle name confirmation stage
                ok, resp = hr.process_confirmation(reply)  # Process user's confirmation reply
                time.sleep(1)  # Simulate thinking
                st.session_state.messages.append({"role": "assistant", "content": resp})  # Add assistant's response
                with st.chat_message("assistant"):
                    st.markdown(resp)
                if not ok:  # If confirmation fails, reset session state
                    for k in ["hr", "vs", "rag_chain", "stage", "messages"]:
                        st.session_state.pop(k, None)
                else:
                    next_qs = hr.next_question()  # Get the next set of questions
                    if not next_qs:  # If no more questions, end the conversation
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "That’s all I had for now. Feel free to restart or upload another resume!"
                        })
                        with st.chat_message("assistant"):
                            st.markdown("That’s all I had for now. Feel free to restart or upload another resume!")
                    else:
                        for q in next_qs:  # Display the next set of questions
                            time.sleep(1)  # Simulate thinking
                            st.session_state.messages.append({"role": "assistant", "content": q})
                            with st.chat_message("assistant"):
                                st.markdown(q)

            else:  # Handle other stages of the HR interview
                next_qs = hr.next_question(last_user_reply=reply)  # Get the next set of questions
                if not next_qs:  # If no more questions, end the conversation
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "That’s all I had for now. Feel free to restart or upload another resume!"
                    })
                    with st.chat_message("assistant"):
                        st.markdown("That’s all I had for now. Feel free to restart or upload another resume!")
                else:
                    for q in next_qs:  # Display the next set of questions
                        time.sleep(1)  # Simulate thinking
                        st.session_state.messages.append({"role": "assistant", "content": q})
                        with st.chat_message("assistant"):
                            st.markdown(q)

if __name__ == "__main__":
    main()  # Run the main function
