import streamlit as st
from dotenv import load_dotenv

from embeddings import initialize_embeddings
from vector_store import process_uploaded_pdfs
from llm_chain import create_conversation_chain
from hr_agent import create_hr_agent_chain

load_dotenv()

def main():
    st.set_page_config(page_title="RAG+HR Interview", layout="wide")
    st.title("📄 RAG & HR Interview Simulator")

    # 1️⃣ Initialize embeddings once
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = initialize_embeddings()

    # 2️⃣ Candidate selector
    candidate = st.text_input("Enter Candidate Name", key="candidate_name")

    # 3️⃣ PDF upload
    uploaded = st.file_uploader(
        "Upload one or more PDF resumes",
        type="pdf",
        accept_multiple_files=True,
        key="uploaded_files"
    )

    # 4️⃣ Process & load into Pinecone (namespaced)
    if candidate and uploaded and st.button("📚 Load Resume"):
        with st.spinner(f"Loading resume for {candidate}…"):
            try:
                vs = process_uploaded_pdfs(
                    uploaded_files=uploaded,
                    embeddings=st.session_state.embeddings,
                    candidate_name=candidate
                )
                # RAG chain for doc Q&A
                st.session_state.rag = create_conversation_chain(vs)
                # HR interviewer chain + memory
                chain, memory = create_hr_agent_chain(vs)
                st.session_state.hr_chain   = chain
                st.session_state.hr_memory  = memory

                # Preload a snippet from the resume for context
                docs = vs.as_retriever().get_relevant_documents("overview")
                snippets = "\n".join(d.page_content[:300] for d in docs)
                st.session_state.resume_snippets = snippets

                # Seed first HR question
                first_q = chain.predict(
                    chat_history="",
                    resume_snippets=snippets
                )
                st.session_state.messages = [{"role":"assistant","content":first_q}]
                st.success(f"✅ {candidate}'s resume loaded. Switch to HR Interview mode.")
            except Exception as e:
                st.error(f"Error loading resume: {e}")

    # 5️⃣ Mode selector
    mode = st.sidebar.radio("Mode", ["Document Chat", "HR Interview"])

    # 6️⃣ Render conversation
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if mode == "HR Interview":
            # Candidate’s answer
            if reply := st.chat_input("Your answer…"):
                st.session_state.messages.append({"role":"user","content":reply})
                st.session_state.hr_memory.chat_memory.add_user_message(reply)

                # Generate next HR question
                q = st.session_state.hr_chain.predict(
                    chat_history=st.session_state.hr_memory.load_memory_variables({})["chat_history"],
                    resume_snippets=st.session_state.resume_snippets
                )
                st.session_state.messages.append({"role":"assistant","content":q})
                st.session_state.hr_memory.chat_memory.add_ai_message(q)

        else:  # Document Chat mode
            if q := st.chat_input("Ask about the documents…"):
                st.session_state.messages.append({"role":"user","content":q})
                resp = st.session_state.rag.invoke({"question":q})
                st.session_state.messages.append({"role":"assistant","content":resp["answer"]})

if __name__ == "__main__":
    main()
