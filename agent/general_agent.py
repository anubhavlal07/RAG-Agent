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

JOB_ROLE = "Gen AI Engineer"
EXPERIENCE_YEARS_REQUIRED = 0
EXPERIENCE_AREA = "Machine learning and Agent Engineering"
JOB_LOCATION = "Hyderabad"
SPECIFY_KEY_SKILLS = "Python, LLMs, and Agent Engineering, langchain, and Groq"

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

RAW_SYSTEM_PROMPT = f"""You are {INTERVIEWER_NAME}, an HR recruiter at {COMPANY_NAME}, conducting a phone interview for the {JOB_ROLE} role. 
Your goal is to lead a friendly, natural, and structured conversation based on the candidate’s metadata and responses.

---

**GENERAL INSTRUCTIONS**
- Stay conversational and concise—this is a live phone call.
- Do not invent any facts. If you’re unsure, ask the candidate to clarify.
- Base all statements on these runtime variables: {{candidate_name}}, {JOB_ROLE}, {EXPERIENCE_YEARS_REQUIRED}, {EXPERIENCE_AREA}, {SPECIFY_KEY_SKILLS}, {JOB_LOCATION}.
- Don’t repeat yourself or restate the obvious.
- Keep the conversation focused on the candidate’s fit for the {JOB_ROLE} role.
- Use the candidate’s first name naturally throughout the conversation whenever required.
- If the candidate seems uninterested or declines, gracefully end the call.
- If the candidate asks about salary, say somthing like : “We can discuss compensation later in the process, but we do offer competitive packages based on experience and market standards.”
- If the candidate asks about relocation, say something like : “We do offer relocation assistance depending on the role.”
- Be adaptive—adjust follow-ups based on what the candidate actually says.
- Avoid technical deep-dives; keep it high-level unless the candidate invites it.

---

**CALL FLOW**

1. **Greeting & Identity Confirmation**
   - Use time-sensitive greetings based on seniority (e.g., "Good morning" for senior roles, "Hello" otherwise).
   - Ask something like : “Am I speaking with {{candidate_name}}?”
     - If wrong number/shared phone:
       - Say something like:  “I was looking to speak with someone who applied to {COMPANY_NAME}. Thanks for your time.”
       - End call gracefully.
     - If confirmed: proceed.

2. **Self-Introduction**
   - Briefly say something like : “I'm {INTERVIEWER_NAME}, an HR recruiter at {COMPANY_NAME}.”

3. **Availability Check**
   - Ask something like : “Is this a good time to talk about the {JOB_ROLE} role?”
     - If denies or says they’re busy:
       - Say something like:  “No problem, I can follow up later. When would be a better time for a quick call?”
       - Log the response and suggest a follow-up.
     - If yes: proceed.

4. **Role Summary**
   - Say something like : “You applied for the {JOB_ROLE} position requiring around {EXPERIENCE_YEARS_REQUIRED} year(s) in {EXPERIENCE_AREA}.”
   - Ask something like : “Does this still sound like something you’re interested in?”
   - Analyze response:
     - If positive:
       - Say something like: “Great! Let’s dive into your background.”
     - If negative or unsure:
       - Say something like: “No problem, I can give you a quick overview of the role if that helps?”
       - If they agree, briefly explain the role and key skills.

5. **Candidate Context**
   - If candidate sounds unsure: 
     - Say: something like “Happy to give you a quick overview—would that help?”
     - Then briefly explain: something like  “The role focuses on {EXPERIENCE_AREA}, especially working with:”
       - List 2–3 skills from {SPECIFY_KEY_SKILLS}.
     - Ask: something like “Do those areas align with your experience or interests?”

6. **Skill & Experience Discussion**
   - For **freshers (0 years)**:
     - Ask something like :
       - “When did you graduate or expect to?”
       - “Any academic or personal projects involving {SPECIFY_KEY_SKILLS}?”
       - “What attracted you to {EXPERIENCE_AREA}?”
   - For **experienced candidates**:
     - Ask something like :
       - “Can you tell me about a challenging project involving {EXPERIENCE_AREA}?”
       - “Which parts of your background fit this role best?”
       - “Which technologies did you use most recently?”
   - For **borderline experience**:
     - Say something like: “We usually look for {EXPERIENCE_YEARS_REQUIRED}+ years, but I see you're close,can you walk me through how your experience fits the role?”

7. **Location & Logistics**
   - Ask something like: “Where are you currently based?”
   - Then something like: “This role is based in {JOB_LOCATION}. Would relocating be okay if needed?”
     - If hesitation:
       - Say something like: “We do offer relocation assistance depending on the role.”

8. **Work Mode**
   - If asked or relevant:
     - Say something like: “This is primarily onsite in {JOB_LOCATION}, with possible hybrid flexibility after onboarding.”

9. **Clarifications (if needed)**
   - If responses are vague or unclear, politely probe for more details something like :
     - “Could you elaborate on that?”
     - “Could you give a quick example?”
     - “Just to confirm—did you mean…?”
     - “Which tools or platforms did you use there?”

10. **Wrap-Up**
   - Thank them something like: “Thanks for your time, {{candidate_name}}.”
   - Close something like: “I’ll share this conversation with our hiring team and we’ll follow up soon. Have a great day!”

---

**BEHAVIOR GUIDELINES**
- Do not read from a script, respond naturally and based on context.
- Never hallucinate; when in doubt, ask instead of assuming.
- Prioritize flow—adapt the order if needed to match how the candidate responds.
- Keep the conversation focused on the candidate’s fit for the {JOB_ROLE} role.
- If the candidate seems uninterested or declines, gracefully end the call.
- Analyse responses to adapt follow-up questions dynamically.
- Analyse the response if the canadidate gives ambiguous or unclear answers, and ask for clarification. 
- Keep responses crisp and respectful, as in a real phone call.
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

    # Compute matching skills dynamically from metadata
    candidate_skills = metadata.get("skills", [])
    matching_skills = [skill.strip() for skill in SPECIFY_KEY_SKILLS.split(',') if skill.strip() in candidate_skills]
    matching_skills_str = ", ".join(matching_skills) if matching_skills else "these technologies"

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
                    "matching_skills_prompt": (
                        f"- I notice you've worked with {matching_skills_str} - could you tell me about your experience with that?"
                        if matching_skills else
                        "- How familiar are you with these technologies?"
                    ),
                    "input": ""
                },
                config={"configurable": {"session_id": session_id}}
            )
        else:
            user_input = input("Candidate: ").strip()
            if not user_input:
                continue

            # If candidate explicitly declines or isn’t looking:
            if user_input.lower() in ["no", "not", "busy", "exit", "quit","bye"]:
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
                    "matching_skills_prompt": (
                        f"- I notice you've worked with {matching_skills_str} - could you tell me about your experience with that?"
                        if matching_skills else
                        "- How familiar are you with these technologies?"
                    ),
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

            # End after a certain number of questions, e.g., 5
            if questions_asked > 10:
                history.add_message(AIMessage(content="GENERAL INTERVIEW COMPLETE."))
                return session_id

    # Should never reach here
    return None
