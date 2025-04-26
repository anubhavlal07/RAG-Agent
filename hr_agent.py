import os
import time
import logging
import traceback
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    load_dotenv()  # Load environment variables from .env file
    logger.info("Attempted to load environment variables from .env file.")
except Exception as e:
    logger.warning(f"Could not load .env file (this is fine if variables are set externally): {e}")

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    logger.error("FATAL: GROQ_API_KEY not found in environment variables. Please ensure it is set.")
    raise ValueError("GROQ_API_KEY not found in environment variables.")
else:
    logger.info(f"GROQ_API_KEY loaded successfully (Key starts with: {groq_api_key[:4]}...).")
    os.environ['GROQ_API_KEY'] = groq_api_key

MODEL_NAME = "llama-3.1-8b-instant"  # Define the model name
LLM_TEMPERATURE = 0.7  # Set the temperature for the LLM
AGENT_MAX_TOKENS = 800  # Maximum tokens for the agent

try:
    from langchain_groq import ChatGroq  # Import ChatGroq from langchain_groq
    logger.info(f"Imported ChatGroq successfully.")

    llm = ChatGroq(
        model_name=MODEL_NAME,  # Specify the model name
        groq_api_key=groq_api_key,  # Pass the API key
        temperature=LLM_TEMPERATURE,  # Set the temperature
    )
    logger.info(f"ChatGroq language model instantiated for model: {MODEL_NAME}")

except ImportError:
    logger.error("Failed to import ChatGroq. Please install langchain-groq: pip install langchain-groq")
    raise
except Exception as e:
    logger.error(f"Error instantiating ChatGroq: {e}", exc_info=True)
    raise

try:
    hr_interviewer = Agent(
        role="HR Interviewer",  # Define the agent's role
        goal="Conduct an initial HR interview assessing candidate's soft skills and cultural fit.",  # Define the agent's goal
        backstory="You are an experienced HR professional specializing in identifying top talent through insightful conversations and behavioral questions.",  # Provide the agent's backstory
        allow_delegation=False,  # Disable delegation
        verbose=True,  # Enable verbose output
        llm=llm,  # Pass the language model
        max_tokens=AGENT_MAX_TOKENS  # Set the maximum tokens
    )
    logger.info(f"CrewAI Agent '{hr_interviewer.role}' created successfully.")

except Exception as e:
    logger.error(f"Error creating CrewAI Agent: {e}", exc_info=True)
    raise

try:
    hr_interview_task = Task(
        description=(  # Define the task description
            "1. Start by introducing yourself briefly.\n"
            "2. Ask the candidate 2-3 relevant questions to evaluate their communication skills, "
            "teamwork ability, adaptability, and general alignment with a collaborative company culture.\n"
            "3. Based *only* on the interaction (as there are no prior responses in this simple setup), "
            "formulate a concise initial assessment."
        ),
        expected_output=(  # Define the expected output
            "Provide a brief response containing:\n"
            "a. Initial assessment summary (2-3 sentences) covering perceived communication, teamwork, adaptability based on the hypothetical interaction.\n"
            "b. A preliminary recommendation on cultural fit (e.g., Seems promising, Concerns regarding X, Need more info)."
        ),
        agent=hr_interviewer,  # Assign the agent to the task
    )
    logger.info("CrewAI Task created successfully.")

except Exception as e:
    logger.error(f"Error creating CrewAI Task: {e}", exc_info=True)
    raise

try:
    hr_crew = Crew(
        agents=[hr_interviewer],  # Add the HR interviewer agent
        tasks=[hr_interview_task],  # Add the HR interview task
        process=Process.sequential,  # Set the process to sequential
        verbose=True,  # Enable verbose output
    )
    logger.info(f"CrewAI Crew created successfully (ID: {hr_crew.id}).")

except Exception as e:
    if "ValidationError" in str(type(e)):  # Check for Pydantic validation errors
         logger.error(f"Pydantic Validation Error creating Crew: {e}")
    else:
        logger.error(f"Error creating CrewAI Crew: {e}", exc_info=True)
    raise

def run_interview_crew_with_retry(crew_instance, max_retries=2, retry_delay_seconds=5):
    """Runs the crew's kickoff method with retry logic."""
    logger.info(f"Attempting to kickoff crew '{crew_instance.id}' (Max Retries: {max_retries})")
    attempt = 0
    last_exception = None
    while attempt <= max_retries:
        try:
            logger.info(f"Kickoff Attempt {attempt + 1}/{max_retries + 1}...")
            result = crew_instance.kickoff()  # Execute the crew's kickoff method
            logger.info("Crew kickoff completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error during crew kickoff (Attempt {attempt + 1}): {e}")
            logger.error(traceback.format_exc())
            last_exception = e
            attempt += 1
            if attempt <= max_retries:
                logger.warning(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)  # Wait before retrying
            else:
                logger.error("Maximum retries reached. Crew execution failed.")
                raise RuntimeError(f"HR interview failed after {max_retries} retries.") from last_exception

def create_hr_crew(conversation_chain):
    """Return the initialized HR Crew."""
    return hr_crew  # Return the HR crew instance

def ask_next_question_with_retry(crew_instance, retries=2):
    """Run the HR interview process with retries."""
    return run_interview_crew_with_retry(crew_instance, max_retries=retries)

if __name__ == "__main__":
    logger.info("Executing hr_agent.py as main script.")
    print("\n" + "="*30 + " Starting HR Crew Execution " + "="*30)
    try:
        if 'hr_crew' not in locals() or hr_crew is None:  # Check if the HR crew is initialized
            raise NameError("hr_crew object was not successfully created.")

        hr_interview_result = run_interview_crew_with_retry(hr_crew, max_retries=1)  # Run the HR interview

        print("\n" + "="*30 + " HR Interview Result " + "="*30)
        print(hr_interview_result)  # Print the interview result
        logger.info("Script finished successfully.")

    except Exception as main_exception:
        print(f"\n" + "="*30 + " Execution Failed " + "="*30)
        print(f"An error occurred: {main_exception}")  # Print the error message
        logger.critical("Script execution failed in the main block.", exc_info=True)

    print("\n" + "="*30 + " End of Execution " + "="*30)
