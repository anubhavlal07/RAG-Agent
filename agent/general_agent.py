import os
import sys
from time import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------

COMPANY_NAME = "Tech Innovators Inc."
INTERVIEWER_NAME = "Sophia"
MAX_GENERAL_QUESTIONS = 10

JOB_ROLE = "Software Development Engineer"
EXPERIENCE_YEARS_REQUIRED = 2
EXPERIENCE_AREA = "full-stack development"

# ----------------------------------------------------------------
# LLM INITIALIZATION
# ----------------------------------------------------------------

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
# STEP 1: PROMPT TEMPLATE FOR HR FLOW
# ----------------------------------------------------------------

RAW_SYSTEM_PROMPT = """
You are a recruiter named {name} from {company}. You're calling candidates for a quick conversation.

Use the candidate's resume details below to ask recruiter-style questions one at a time:

{details}

Flow:
- Ask “Am I speaking with {candidate_name}?” to confirm you have the right person. If they provide a different name, ask for clarification.
- Greet them and introduce yourself: "Hi {candidate_name}, I'm {name} from {company}."
- Ask if they have a few minutes to talk. If not, say “No worries—thank you for your time. Goodbye.” and end.
- Ask if they're actively looking for new roles or open to a career change. If they deny say “Thank you for letting me know—have a great day.” and end.
- Ask if they’re a fresher, currently studying, or experienced.
- Introduce the job: "{job_role}" (requires ~{experience_years_required} years in {experience_area}).
- Confirm if this matches their background and interests and if they are interested. If they say “no,” say “Thanks for your time.” and end.
- If they say “yes,” share one sentence mentioning 1–2 key responsibilities and a benefit of the role:
  “This role involves [RESPONSIBILITY 1] and [RESPONSIBILITY 2], and offers the chance to [BENEFIT].”
- Ask what they’re seeking in their next opportunity.
- Ask about their current role and responsibilities.
- Ask about their key skills and technologies they are comfortable with.
- Ask about their career goals and where they see themselves in 2–3 years.
- Ask about their preferred work environment (remote, hybrid, in-office).
- Ask general questions about their work style: like “How do you approach problem-solving and collaboration?”
- Ask about their experience with teamwork and communication.
- Ask about a challenging project they worked on and how they overcame obstacles.
- Ask about their availability for a follow-up interview or next steps.
- Ask their salary expectations: “What are your salary expectations for this role?”
- Ask if they have any questions for you about the role or company.
- If they ask questions, answer them briefly and positively.
- After {max_questions} questions (count only questions ending with “?”), say “GENERAL INTERVIEW COMPLETE.”

Analyze their responses to assess and ask relevant follow-up questions. Always be polite, professional, and concise.
Questions should be open-ended to encourage discussion, but also specific enough to gather useful information.
Always return only the next question or the final sentence. Do not provide explanations, lists, or extra commentary. Keep each prompt to 1–2 sentences.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAW_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# ----------------------------------------------------------------
# STEP 2: The interactive loop for general HR questions
# ----------------------------------------------------------------

def run_general_hr_interview(phone: str, metadata: dict) -> str:
    """
    Conducts the general HR interview. Returns:
    - session_id (string) if the interview completed normally.
    - None if the candidate declines or exits at any point.
    """
    session_id = f"hr_{phone}_{int(time())}"
    history = get_session_history(session_id)

    # Build {details} string from metadata
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
    candidate_name = metadata.get("name", "Candidate")

    # Build runnable chain
    chain = PROMPT_TEMPLATE | llm
    runnable = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    print("\n===== GENERAL HR INTERVIEW (STEP 2: Q&A) =====\n")

    questions_asked = 0
    while True:
        if questions_asked == 0:
            # First turn: no user input yet
            ai_response = runnable.invoke(
                {
                    "company": COMPANY_NAME,
                    "name": INTERVIEWER_NAME,
                    "details": details_str,
                    "candidate_name": candidate_name,
                    "max_questions": MAX_GENERAL_QUESTIONS,
                    "job_role": JOB_ROLE,
                    "experience_years_required": EXPERIENCE_YEARS_REQUIRED,
                    "experience_area": EXPERIENCE_AREA,
                    "input": ""
                },
                config={"configurable": {"session_id": session_id}}
            )
        else:
            user_input = input("Candidate: ").strip()
            if not user_input:
                continue

            # If candidate explicitly declines or isn’t looking:
            if any(neg in user_input.lower() for neg in ["no", "not", "busy", "exit", "quit"]):
                print("AI: Thank you for your time. Goodbye!")
                return None  # Signal early exit

            history.add_message(HumanMessage(content=user_input))
            ai_response = runnable.invoke(
                {
                    "company": COMPANY_NAME,
                    "name": INTERVIEWER_NAME,
                    "details": details_str,
                    "candidate_name": candidate_name,
                    "max_questions": MAX_GENERAL_QUESTIONS,
                    "job_role": JOB_ROLE,
                    "experience_years_required": EXPERIENCE_YEARS_REQUIRED,
                    "experience_area": EXPERIENCE_AREA,
                    "input": user_input
                },
                config={"configurable": {"session_id": session_id}}
            )

        ai_text = ai_response.content.strip()
        print(f"\nAI: {ai_text}\n")
        history.add_message(AIMessage(content=ai_text))

        # If AI explicitly ends the interview, stop
        if "GENERAL INTERVIEW COMPLETE" in ai_text:
            return session_id

        # Only count if AI’s last line ends with “?”
        if ai_text.endswith("?"):
            questions_asked += 1
            if questions_asked >= MAX_GENERAL_QUESTIONS:
                print("AI: GENERAL INTERVIEW COMPLETE.")
                history.add_message(AIMessage(content="GENERAL INTERVIEW COMPLETE."))
                return session_id

    # If somehow we exit loop unexpectedly, return None
    return None
