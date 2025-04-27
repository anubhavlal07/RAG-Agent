from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from config import get_groq_api_key, GROQ_MODEL_NAME

PROMPT = PromptTemplate(
    input_variables=["chat_history", "resume_snippets"],
    template="""
You are an HR interviewer. You have the following resume details:
{resume_snippets}

Conversation so far:
{chat_history}

Ask the candidate one clear follow-up question next. Do not repeat any earlier question.  
Use second-person pronouns (you, your).

HR Interviewer:
"""
)

def create_hr_agent_chain(vector_store):
    """Returns (llm_chain, memory) ready to generate interview questions."""
    llm = ChatGroq(
        groq_api_key=get_groq_api_key(),  # Initialize ChatGroq with API key and model name
        model_name=GROQ_MODEL_NAME
    )
    chain = LLMChain(llm=llm, prompt=PROMPT, output_key="question")  # Create LLM chain with prompt
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Store conversation history
    return chain, memory
