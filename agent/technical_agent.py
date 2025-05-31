import os
from time import time
from dotenv import load_dotenv

# LangChain and Groq LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# In-memory store of chat histories per session
session_store = {}
# Tracks which categories have been used in a session
category_history = {}

# Dictionary of prompt templates for each technical question category
CATEGORY_TEMPLATES = {
    "tech_most_challenging_project": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE broad question about the candidate’s most challenging project across their experience—such as what made it challenging, how they overcame obstacles, and what they learned. Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_project_deep_dive": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE focused question about a specific project the candidate has listed, probing implementation details—such as architecture choices, libraries used, or performance considerations. Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_project_impact": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE question about the measurable impact of a specific project the candidate listed—such as cost savings, efficiency gains, or performance improvements. Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_platform_choice": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE question about why the candidate chose a particular platform or technology (for example, Firebase) over alternatives, probing their trade‐offs and decision criteria. Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_scalability_decision": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE question about how the candidate designed their system or data pipeline to scale—specifically, how they balanced throughput, latency, and resource costs. Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_error_handling_followup": """You are a technical interviewer at XYZ. The candidate answered: "{last_response}".
Ask ONE targeted follow‐up question about their approach to error handling or missing data in that scenario. Return ONLY the single question, in plain text, no bullet points or numbering.
""",
    "tech_performance_tuning": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE question about how the candidate identified and optimized a performance bottleneck—such as in a database query, data processing job, or front‐end rendering. Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_function_design": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE question about how the candidate would design a particular function or module related to their stated skills (e.g., data processing, API endpoint, algorithm). Inquire about inputs, outputs, and error handling. Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_syntax_and_language": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE question testing the candidate’s knowledge of syntax or idioms in a language they listed (e.g., Python list comprehensions, JavaScript async/await, SQL JOINs). Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_problem_solving": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE concise, problem‐solving question that requires the candidate to outline their approach to solving a real‐world scenario relevant to their role or projects (e.g., scaling a service, debugging a memory leak). Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
    "tech_education_application": """You are a technical interviewer at XYZ. Use the candidate’s resume (below).
Ask ONE question that connects the candidate’s formal education to practical application—such as applying a data structure, mathematical concept, or theory they learned during their degree. Return ONLY the single question, in plain text, no bullet points or numbering.

Here is the resume for context:
{resume}
""",
}

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL_NAME")
)

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieve or create chat history for the given session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def _escape_braces(text: str) -> str:
    """
    Replace every single brace with double braces so Python's str.format()
    will treat them literally (rather than as placeholders).
    """
    return text.replace("{", "{{").replace("}", "}}")

def choose_next_category(session_id: str) -> str:
    """
    Pick the next technical question category, avoiding immediate repeats
    and limiting back-to-back project questions.
    """
    all_categories = [
        "tech_syntax_and_language",
        "tech_function_design",
        "tech_problem_solving",
        "tech_education_application",
        "tech_platform_choice",
        "tech_scalability_decision",
        "tech_performance_tuning",
        "tech_error_handling_followup",
        "tech_most_challenging_project",
        "tech_project_deep_dive",
        "tech_project_impact",
    ]
    project_cats = {
        "tech_most_challenging_project",
        "tech_project_deep_dive",
        "tech_project_impact",
    }

    recent = category_history.get(session_id, [])
    recent_proj_count = sum(1 for c in recent[-2:] if c in project_cats)
    last_cat = recent[-1] if recent else None

    candidates = []
    for cat in all_categories:
        if cat == last_cat:
            continue
        if recent_proj_count >= 2 and cat in project_cats:
            continue
        candidates.append(cat)

    if not candidates:
        candidates = [c for c in all_categories if c != last_cat]

    next_cat = candidates[0]
    category_history.setdefault(session_id, []).append(next_cat)
    return next_cat

def build_category_prompt(category: str, resume: str, last_response: str) -> ChatPromptTemplate:
    """
    Build a ChatPromptTemplate for the selected category, filling in resume and last response.
    We must escape braces in 'resume' and 'last_response' so that str.format() won't misinterpret JSON.
    """
    template_text = CATEGORY_TEMPLATES[category]

    # Escape braces in resume and last_response
    esc_resume = _escape_braces(resume)
    esc_last = _escape_braces(last_response)

    # Format once to inject the escaped resume (and last_response, if needed)
    filled = template_text.format(resume=esc_resume, last_response=esc_last)

    return ChatPromptTemplate.from_messages([
        ("system", filled),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def interview_loop(resume_text: str, general_history: list) -> str:
    """
    Main loop for the technical interview. Receives:
      - resume_text: the full text of the candidate’s resume
      - general_history: a list of dicts {speaker, text, timestamp}
    1. Seed the session history with all general_history messages.
    2. Ask an initial project question.
    3. Iteratively choose categories and ask follow‐up technical questions.
    4. When the candidate says “exit” or “bye,” stop.
    5. Return this technical session’s session_id.
    """
    session_id = f"tech_{int(time())}"
    session_hist = get_session_history(session_id)

    # Seed the history with the general‐interview messages
    for entry in general_history:
        sp = entry["speaker"]
        txt = entry["text"]
        if sp.lower() == "system":
            session_hist.add_message(SystemMessage(content=txt))
        elif sp.lower() in ["ai", "assistant"]:
            session_hist.add_message(AIMessage(content=txt))
        else:
            session_hist.add_message(HumanMessage(content=txt))

    # Print a welcome message
    print("\nAI: Thank you. Now moving on to the technical portion.\n")

    # Ask the first question manually (most challenging project)
    initial_question = (
        "Based on your resume, can you explain the most technically challenging project you've worked on, "
        "including any architecture or design decisions?"
    )
    session_hist.add_message(AIMessage(content=initial_question))
    print("AI:", initial_question)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["no", "exit", "quit", "bye"]:
            print("\nAI: Thank you for your time. The technical interview is now complete.")
            break

        # Add candidate response to history
        session_hist.add_message(HumanMessage(content=user_input))

        # Pick the next category
        next_category = choose_next_category(session_id)

        # Build a prompt for that category
        prompt = build_category_prompt(next_category, resume_text, user_input)
        chain = prompt | llm
        runnable = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        # Invoke LLM with empty human input (the system prompt already contains instructions)
        response = runnable.invoke(
            {"input": ""},
            config={"configurable": {"session_id": session_id}}
        )
        ai_text = response.content.strip()
        session_hist.add_message(AIMessage(content=ai_text))

        # Print the AI’s generated question
        print("\nAI:", ai_text)

    # Do NOT save here. Return session_id so the manager can fetch history later.
    return session_id
