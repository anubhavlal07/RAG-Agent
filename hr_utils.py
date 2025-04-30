from nameExtractor import extract_name
from typing import Optional

class HRInterview:
    def __init__(
        self,
        resume_text: str,
        hr_chain,
        hr_memory,
        resume_snippets: str
    ):
        # Extract candidate name from the resume text
        name = extract_name(resume_text)
        self.candidate_name = name if name else "Candidate"  # Default to "Candidate" if no name is found
        self.resume_text = resume_text  # Store the full resume text
        self.resume_snippets = resume_snippets  # Store specific snippets of the resume
        self.hr_chain = hr_chain  # Chain for generating responses
        self.hr_memory = hr_memory  # Memory for storing chat history
        self.stage = "confirm_name"  # Initial stage of the interview process

    def confirm_name_question(self) -> str:
        # Generate a question to confirm the candidate's name
        return f"Is your name {self.candidate_name}? (Yes/No)"

    def process_confirmation(self, reply: str):
        # Process the candidate's response to the name confirmation question
        ok = reply.strip().lower() in ("yes", "y", "correct")
        if ok:
            self.stage = "verification"  # Move to the verification stage
            return True, "Great—thank you! Let's get started."
        else:
            self.stage = "restart"  # Restart the process if the name is incorrect
            return False, "Wrong candidate. Please re-upload your resume."

    def _generate_response(self) -> str:
        # Generate a response using the HR chain and chat history
        chat_hist = self.hr_memory.load_memory_variables({})["chat_history"]
        response = self.hr_chain.predict(
            chat_history=chat_hist,
            resume_snippets=self.resume_snippets
        )
        self.hr_memory.chat_memory.add_ai_message(response)  # Add AI response to chat memory
        return response

    def generate_verification_questions(self):
        # Generate questions for the verification stage
        return [self._generate_response()]

    def generate_skill_questions(self, skills: str):
        # Generate questions related to the candidate's skills
        return [self._generate_response()]

    def generate_role_exploration_question(self, candidate_info: dict):
        # Generate questions to explore the candidate's interest in the role
        return [self._generate_response()]

    def generate_follow_up(self, last_response: str, current_topic: str):
        # Generate follow-up questions based on the last response and current topic
        self.hr_memory.chat_memory.add_user_message(last_response)  # Add user response to chat memory
        return [self._generate_response()]

    def wrap_up_interview(self) -> str:
        # Generate a response to wrap up the interview
        return self._generate_response()

    def next_question(self, last_user_reply: Optional[str] = None):
        # Determine the next question based on the current stage
        if self.stage == "verification":
            self.stage = "skill"  # Move to the skill stage
            return self.generate_verification_questions()

        if self.stage == "skill":
            self.stage = "role_explore"  # Move to the role exploration stage
            return self.generate_skill_questions(skills="")

        if self.stage == "role_explore":
            self.stage = "rag"  # Move to the RAG (retrieval-augmented generation) stage
            return self.generate_role_exploration_question({})

        if self.stage == "rag":
            self.stage = "followup"  # Move to the follow-up stage
            return self.generate_follow_up(last_user_reply or "", current_topic="")

        if self.stage == "followup":
            self.stage = "wrap_up"  # Move to the wrap-up stage
            return [self.wrap_up_interview()]

        return []  # Return an empty list if no valid stage is found
