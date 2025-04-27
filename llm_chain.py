from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from config import get_groq_api_key, GROQ_MODEL_NAME

def create_conversation_chain(vector_store):
    llm = ChatGroq(
        groq_api_key=get_groq_api_key(),  # Initialize the ChatGroq LLM with API key and model name
        model_name=GROQ_MODEL_NAME
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Store conversation history
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),  # Use vector store as retriever
        memory=memory
    )
