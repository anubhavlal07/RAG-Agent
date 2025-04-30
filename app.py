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

load_dotenv()

def main():
    st.set_page_config(page_title="RAG+HR Interview Simulator", layout="wide")
    st.title("📄 RAG & HR Interview Simulator")

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = initialize_embeddings()

    uploaded = st.file_uploader("Upload a PDF resume", type="pdf")
    if uploaded and st.button("Load Resume"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.remove(tmp_path)
        uploaded.seek(0)

        full_text = "\n\n".join(d.page_content for d in docs)
        name = extract_name(full_text) or "candidate"
        vs = process_uploaded_pdfs(
            uploaded_files=[uploaded],
            embeddings=st.session_state.embeddings,
            candidate_name=name
        )
        st.session_state.vs = vs

        st.session_state.rag_chain = create_conversation_chain(vs)

        hr_chain, hr_memory = create_hr_agent_chain(vs)
        docs_for_snip = vs.as_retriever().get_relevant_documents("overview")
        snippets = "\n".join(d.page_content[:300] for d in docs_for_snip)

        hr = HRInterview(
            resume_text=full_text,
            hr_chain=hr_chain,
            hr_memory=hr_memory,
            resume_snippets=snippets
        )
        st.session_state.hr = hr

        st.session_state.messages = [
            {"role": "assistant", "content": hr.confirm_name_question()}
        ]
        st.session_state.stage = "confirm_name"

    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if reply := st.chat_input("Your answer…"):
            st.session_state.messages.append({"role": "user", "content": reply})
            with st.chat_message("user"):
                st.markdown(reply)

            hr: HRInterview = st.session_state.hr

            if hr.stage == "confirm_name":
                ok, resp = hr.process_confirmation(reply)
                time.sleep(1)  # Simulate thinking
                st.session_state.messages.append({"role": "assistant", "content": resp})
                with st.chat_message("assistant"):
                    st.markdown(resp)
                if not ok:
                    for k in ["hr", "vs", "rag_chain", "stage", "messages"]:
                        st.session_state.pop(k, None)
                else:
                    next_qs = hr.next_question()
                    if not next_qs:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "That’s all I had for now. Feel free to restart or upload another resume!"
                        })
                        with st.chat_message("assistant"):
                            st.markdown("That’s all I had for now. Feel free to restart or upload another resume!")
                    else:
                        for q in next_qs:
                            time.sleep(1)  # Simulate thinking
                            st.session_state.messages.append({"role": "assistant", "content": q})
                            with st.chat_message("assistant"):
                                st.markdown(q)

            else:
                next_qs = hr.next_question(last_user_reply=reply)
                if not next_qs:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "That’s all I had for now. Feel free to restart or upload another resume!"
                    })
                    with st.chat_message("assistant"):
                        st.markdown("That’s all I had for now. Feel free to restart or upload another resume!")
                else:
                    for q in next_qs:
                        time.sleep(1)  # Simulate thinking
                        st.session_state.messages.append({"role": "assistant", "content": q})
                        with st.chat_message("assistant"):
                            st.markdown(q)

if __name__ == "__main__":
    main()
