import os
import sys
from time import time
from dotenv import load_dotenv

# LangChain + Groq imports
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Pinecone lookup
from agent.data_loader import get_candidate_by_phone

load_dotenv()

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------

COMPANY_NAME = "Tech Innovators Inc."
INTERVIEWER_NAME = "Sophia"
MAX_GENERAL_QUESTIONS = 7  # Number of HR questions before completion

# Initialize the ChatGroq LLM
llm = ChatGroq(
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL_NAME")
)

# In‐memory store of chat histories
session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieve or create a ChatMessageHistory for the given session_id."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# ----------------------------------------------------------------
# STEP 1: Look up candidate’s data from Pinecone (by phone #)
# ----------------------------------------------------------------

def fetch_and_confirm_candidate():
    """
    1. Prompt for a phone number.
    2. Use get_candidate_by_phone() to retrieve metadata.
    3. Display key fields and ask for confirmation.
       If “no,” loop again. If “yes,” return (phone_number, metadata_dict).
    """
    print("\n===== GENERAL HR INTERVIEW (STEP 1: Candidate Lookup) =====\n")
    while True:
        phone = input("Please enter your phone number (digits only or formatted): ").strip()
        if not phone:
            continue

        # Extract digits; Pinecone metadata stores phone as digits-only
        digits = "".join(filter(str.isdigit, phone))
        if len(digits) < 10:
            print("❗ We need at least 10 digits to look up your record. Please try again.")
            continue

        metadata = get_candidate_by_phone(digits)
        if not metadata:
            print(f"❗ No candidate found for phone number: {digits}. Try again or type 'exit'.")
            if phone.lower() in ["exit", "quit"]:
                sys.exit(0)
            continue

        # Display key fields and ask for confirmation
        print("\nWe found the following information for you:\n")
        display_fields = {
            "Name": metadata.get("name"),
            "Email": metadata.get("email", "<not provided>"),
            "Education": metadata.get("education", "<not provided>"),
            "Skills": ", ".join(metadata.get("skills", [])) if metadata.get("skills") else "<not provided>",
            "Experience (years)": metadata.get("experience_years", "<not provided>"),
            "Current Role": metadata.get("current_role", "<not provided>")
        }
        for label, val in display_fields.items():
            print(f"  • {label}: {val}")
        print()

        confirm = input("Is this information correct? (yes/no): ").strip().lower()
        if confirm in ["yes", "y"]:
            return digits, metadata
        elif confirm in ["no", "n"]:
            print("Okay, let's try again. Please re-enter your phone number or correct any mistakes.\n")
            continue
        else:
            print("Please answer 'yes' or 'no'.\n")

# ----------------------------------------------------------------
# STEP 2: Build a ChatPromptTemplate for the general HR flow
# ----------------------------------------------------------------

RAW_SYSTEM_PROMPT = """
You are an AI HR interviewer from {company}. Your name is {name}.
All candidate responses are stored securely.

Below are the candidate’s details (pulled from our records):
{details}

Now, conduct a short general HR screening by asking exactly one question per turn.
After the candidate answers, ask the next HR question, keeping it concise (1 sentence).
Stop after you have asked {max_questions} questions and then say: "GENERAL INTERVIEW COMPLETE."
Return ONLY the question or final completion sentence, never extra commentary.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAW_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# ----------------------------------------------------------------
# STEP 3: The interactive loop for general HR questions
# ----------------------------------------------------------------

def run_general_hr_interview(phone: str, metadata: dict) -> str:
    """
    1. Create a session_id and retrieve its chat history.
    2. Build a short summary of metadata fields for {details}.
    3. Loop through up to MAX_GENERAL_QUESTIONS:
       • On the first turn, invoke with input="" plus template variables (company, name, details, max_questions).
       • On subsequent turns, solicit candidate response, add to history, and invoke the LLM again.
       • Print existing AI question or completion message.
       • Break once AI says “GENERAL INTERVIEW COMPLETE.”
    4. Return session_id (without saving to disk). 
       The manager will handle final saving.
    """
    session_id = f"hr_{phone}_{int(time())}"
    history = get_session_history(session_id)

    # Summarize metadata fields for prompt
    summary_fields = []
    if metadata.get("name"):
        summary_fields.append(f"Name: {metadata['name']}")
    if metadata.get("education"):
        summary_fields.append(f"Education: {metadata['education']}")
    if metadata.get("skills"):
        summary_fields.append(f"Skills: {', '.join(metadata['skills'])}")
    if metadata.get("experience_years") is not None:
        summary_fields.append(f"Experience (years): {metadata['experience_years']}")
    if metadata.get("current_role"):
        summary_fields.append(f"Current Role: {metadata['current_role']}")
    details_str = "\n".join(summary_fields)

    # Build the chain (prompt + llm) with a Runnable that tracks history
    chain = PROMPT_TEMPLATE | llm
    runnable = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    print("\n===== GENERAL HR INTERVIEW (STEP 2: Q&A) =====\n")
    print(f"AI: Hello, I am {INTERVIEWER_NAME} from {COMPANY_NAME}. Let's begin.\n")

    questions_asked = 0
    while True:
        if questions_asked == 0:
            # First turn: no candidate input yet, provide template variables
            ai_response = runnable.invoke(
                {
                    "company": COMPANY_NAME,
                    "name": INTERVIEWER_NAME,
                    "details": details_str,
                    "max_questions": MAX_GENERAL_QUESTIONS,
                    "input": ""
                },
                config={"configurable": {"session_id": session_id}}
            )
        else:
            user_input = input("Candidate: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("AI: Thank you for your time. Goodbye!")
                break

            # Add candidate's message to history
            history.add_message(HumanMessage(content=user_input))

            # Provide the same template variables + candidate input
            ai_response = runnable.invoke(
                {
                    "company": COMPANY_NAME,
                    "name": INTERVIEWER_NAME,
                    "details": details_str,
                    "max_questions": MAX_GENERAL_QUESTIONS,
                    "input": user_input
                },
                config={"configurable": {"session_id": session_id}}
            )

        # Print AI response and add to history
        ai_text = ai_response.content.strip()
        print(f"\nAI: {ai_text}\n")
        history.add_message(AIMessage(content=ai_text))

        # Count questions and detect completion
        if ai_text.endswith("?"):
            questions_asked += 1
        if "GENERAL INTERVIEW COMPLETE" in ai_text:
            break

    # Do NOT save here. Return session_id for the manager to later fetch history.
    return session_id
