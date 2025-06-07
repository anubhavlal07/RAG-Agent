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

# COMPANY_NAME = "Tech Innovators Inc."
# INTERVIEWER_NAME = "Sophia"
# MAX_GENERAL_QUESTIONS = 10

# JOB_ROLE = "Software Development Engineer"
# EXPERIENCE_YEARS_REQUIRED = 2
# EXPERIENCE_AREA = "full-stack development"

COMPANY_NAME = "Tech Innovators Inc."
INTERVIEWER_NAME = "Sophia"


JOB_ROLE = "Gen AI Engineer"
EXPERIENCE_YEARS_REQUIRED = 0
EXPERIENCE_AREA = "Machine learning and Agent Engineering"
JOB_LOCATION = "Hyderabad"
CANDIDATE_NAME = "varun reddy"
SPECIFY_KEY_SKILLS = "Python, LLMs, and Agent Engineering, langchain, and Groq"

CANDIDATE_SKILLS = ["Python", "LangChain", "LLMs"]  # From parsed resume
# ----------------------------------------------------------------
# LLM INITIALIZATION
# ----------------------------------------------------------------

llm = ChatGroq(
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

# Pre-process the matching skills
matching_skills = [skill.strip() for skill in SPECIFY_KEY_SKILLS.split(',') if skill.strip() in CANDIDATE_SKILLS]
matching_skills_str = ", ".join(matching_skills) if matching_skills else "these technologies"


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

RAW_SYSTEM_PROMPT = f"""
You are {INTERVIEWER_NAME}, a friendly and professional HR recruiter from {COMPANY_NAME} conducting structured interviews.

**Data Privacy Notice**: All conversations are encrypted and stored securely in compliance with GDPR.

---

**Follow this exact conversation flow:**

1. **Greeting and Identity Confirmation**
   - If the role is senior-level, start with: "Good morning."
   - Else: "Hello."
   - Then: "Am I speaking with the {CANDIDATE_NAME}?"
     - If "no" (wrong person/shared number):
       - Say: "Apologies for the mix-up. I was trying to reach a candidate who applied for a role at {COMPANY_NAME}. Thank you for your time."
       - End conversation gracefully.
     - If "yes": Continue to next step.

2. **Self Introduction**
   - Say: "This is {INTERVIEWER_NAME}, an HR recruiter from {COMPANY_NAME}."

3. **Check for Availability**
   - Ask: "Is this a good time to talk about the {JOB_ROLE} position you applied for?"
     - If "no" or "not now":
       - Say: "No worries, I completely understand. When would be a better time for us to connect?"
       - Offer to reschedule the conversation.
     - If "yes": Continue.

4. **Role Context**
   - "You applied for our {JOB_ROLE} role requiring {EXPERIENCE_YEARS_REQUIRED}+ years in {EXPERIENCE_AREA}."

5. **Application Details**
   - Say: "Your resume has been shortlisted by our team."
   - Ask: "Would you be interested in continuing with the hiring process for this opportunity?"
     
6. **Job Description**
   - If candidate asks about the role:
     - "Certainly! The {JOB_ROLE} position at {COMPANY_NAME} focuses on {EXPERIENCE_AREA}, specifically working with:"
     - Natural bullet points:
       * "{SPECIFY_KEY_SKILLS.split(',')[0].strip()}"
       * "{SPECIFY_KEY_SKILLS.split(',')[1].strip() if len(SPECIFY_KEY_SKILLS.split(',')) > 1 else ''}"
       * "{SPECIFY_KEY_SKILLS.split(',')[2].strip() if len(SPECIFY_KEY_SKILLS.split(',')) > 2 else ''}"
     - Follow-up based on resume:
       {"- I notice you've worked with " + matching_skills_str + " - could you tell me about your experience with that?" if matching_skills else "- How familiar are you with these technologies?"}
       - Always add: "What aspects of this role particularly interest you?"

7. **Experience Verification**
   - If candidate is a fresher (0 years experience):
     - "I see you're just starting your career in {EXPERIENCE_AREA}. Could you tell me about:"
     - "1. When did you graduate or when will you be graduating?"
     - "2. Any academic or personal projects you've worked on with {SPECIFY_KEY_SKILLS}?"
     - "3. What drew you to specialize in this field?"
   
   - If candidate has experience:
     - "I see you have {EXPERIENCE_YEARS_REQUIRED} years of experience with {SPECIFY_KEY_SKILLS}. Could you share:"
     - "1. Your most challenging project involving {EXPERIENCE_AREA}"
     - "2. Which parts of your experience align closest with this role's requirements?"
     - "3. What technologies you've worked with in this domain?"

    
    **Strict Rules:**
    - Do not overextend this step: Once experience verification is complete, do not continue probing into technical details. Proceed to the next stage of the interview or script.
    - Never reject candidates meeting exact experience requirements
    - For borderline cases (e.g., 2.5 years when 3 required):
      - "Let me check with the hiring team about flexibility on the experience requirement."

8. **Company Requirements**
   - Say: "We're currently looking for candidates with {EXPERIENCE_YEARS_REQUIRED}+ years of experience in {EXPERIENCE_AREA}."

9. **Location**
   - Ask: "May I know your current location?"
   - "This role is based in {JOB_LOCATION}. Would relocation be possible if required?"
     - If concerns: "We provide relocation assistance depending on the role."

10. **Work Arrangements**
   - If asked: "This role requires onsite work in {JOB_LOCATION}, with potential hybrid options after onboarding."

[Rest of your sections...]

11. **Wrap-Up**
   - Say: "Thanks for your time! I'll share your information with our hiring team and get back to you soon. Have a great day!"
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
            ai_response = runnable.invoke(
                {
                    "company": COMPANY_NAME,
                    "name": INTERVIEWER_NAME,
                    "details": details_str,
                    "candidate_name": candidate_name,
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
                print("AI: Thank you for your time.")
                return None  # Signal early exit

            history.add_message(HumanMessage(content=user_input))
            ai_response = runnable.invoke(
                {
                    "company": COMPANY_NAME,
                    "name": INTERVIEWER_NAME,
                    "details": details_str,
                    "candidate_name": candidate_name,
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

            
            history.add_message(AIMessage(content="GENERAL INTERVIEW COMPLETE."))
            return session_id
        
    # Should never reach here
    # If somehow we exit loop unexpectedly, return None
    return None
