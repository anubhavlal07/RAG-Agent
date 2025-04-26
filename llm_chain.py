from langchain_groq import ChatGroq  # Importing the Groq LLM integration
from langchain.memory import ConversationBufferMemory  # Importing memory for conversation history
from langchain.chains import ConversationalRetrievalChain  # Importing the conversational chain
from config import get_groq_api_key, GROQ_MODEL_NAME  # Importing configuration utilities

def create_conversation_chain(vector_store):
    groq_api_key = get_groq_api_key()  # Fetching the Groq API key from configuration
    llm = ChatGroq(
        groq_api_key=groq_api_key,  # Setting the API key for the Groq LLM
        model_name=GROQ_MODEL_NAME  # Specifying the Groq model name
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Key to store chat history in memory
        return_messages=True  # Enable returning messages from memory
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,  # Using the Groq LLM
        retriever=vector_store.as_retriever(),  # Setting the retriever from the vector store
        memory=memory  # Attaching memory to the chain
    )
