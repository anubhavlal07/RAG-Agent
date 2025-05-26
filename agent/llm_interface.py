from langchain_groq import ChatGroq

def initialize_groq_llm(api_key):
    return ChatGroq(api_key=api_key, model_name="llama3-8b-8192")