import streamlit as st
from embeddings import initialize_embeddings
from vector_store import process_uploaded_pdfs
from llm_chain import create_conversation_chain
from hr_agent import create_hr_agent_chain

def main():
    st.set_page_config(page_title="RAG + HR Interview", layout="wide")  # Set page configuration
    st.title("📄 RAG Chat & HR Interview Simulator")  # Set page title

    # Initialize embeddings if not already in session state
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = initialize_embeddings()

    # Handle PDF uploads and processing
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("📚 Process & Load"):
        vs = process_uploaded_pdfs(uploaded, st.session_state.embeddings)  # Process uploaded PDFs
        st.session_state.vector_store = vs  # Store vector store in session state
        st.session_state.rag = create_conversation_chain(vs)  # Create RAG conversation chain
        chain, memory = create_hr_agent_chain(vs)  # Create HR agent chain and memory
        st.session_state.hr_chain = chain  # Store HR chain in session state
        st.session_state.hr_memory = memory  # Store HR memory in session state
        docs = vs.as_retriever().get_relevant_documents("overview")  # Retrieve relevant documents
        snippets = "\n".join(d.page_content[:300] for d in docs)  # Extract snippets from documents
        st.session_state.resume_snippets = snippets  # Store resume snippets in session state
        st.session_state.messages = []  # Initialize messages list
        first_q = chain.predict(  # Generate the first HR question
            chat_history="",
            resume_snippets=st.session_state.resume_snippets
        )
        st.session_state.messages.append({"role": "assistant", "content": first_q})  # Add first question to messages
        st.success("✅ Loaded! Switch to HR Interview below.")  # Display success message

    mode = st.sidebar.radio("Mode", ["Document Chat", "HR Interview"])  # Mode selector

    # Render chat messages
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):  # Display chat messages
                st.markdown(msg["content"])

        # HR Interview flow
        if mode == "HR Interview":
            if reply := st.chat_input("Your answer..."):  # Capture user input for HR interview
                st.session_state.messages.append({"role": "user", "content": reply})  # Add user reply to messages
                st.session_state.hr_memory.chat_memory.add_user_message(reply)  # Update memory with user message
                next_q = st.session_state.hr_chain.predict(  # Generate next HR question
                    chat_history=st.session_state.hr_memory.load_memory_variables({})["chat_history"],
                    resume_snippets=st.session_state.resume_snippets
                )
                st.session_state.messages.append({"role": "assistant", "content": next_q})  # Add next question to messages
                st.session_state.hr_memory.chat_memory.add_ai_message(next_q)  # Update memory with AI message

        # RAG Chat flow
        else:
            if prompt := st.chat_input("Ask about your documents..."):  # Capture user input for RAG chat
                st.session_state.messages.append({"role": "user", "content": prompt})  # Add user prompt to messages
                resp = st.session_state.rag.invoke({"question": prompt})  # Get response from RAG chain
                answer = resp["answer"]  # Extract answer from response
                st.session_state.messages.append({"role": "assistant", "content": answer})  # Add answer to messages

if __name__ == "__main__":
    main()
