# hr_utils.py

from nameExtractor import extract_name
from typing import Optional

class HRInterview:
    def __init__(
        self,
        resume_text: str,
        hr_chain,           # LLMChain from create_hr_agent_chain
        hr_memory,          # ConversationBufferMemory from create_hr_agent_chain
        resume_snippets: str
    ):
        # 1. Extract the candidate’s name
        name = extract_name(resume_text)
        self.candidate_name = name if name else "Candidate"
        self.resume_text = resume_text
        self.resume_snippets = resume_snippets

        # 2. Store the HR‐specific chain & memory
        self.hr_chain = hr_chain
        self.hr_memory = hr_memory

        # 3. Interview state
        self.stage = "confirm_name"

    def confirm_name_question(self) -> str:
        return f"Is your name {self.candidate_name}? (Yes/No)"

    def process_confirmation(self, reply: str):
        ok = reply.strip().lower() in ("yes", "y", "correct")
        if ok:
            self.stage = "verification"
            return True, "Great—thank you! Let's get started."
        else:
            self.stage = "restart"
            return False, "Wrong candidate. Please re-upload your resume."

    def _generate_response(self) -> str:
        """Call your real HR LLMChain with history & resume context."""
        chat_hist = self.hr_memory.load_memory_variables({})["chat_history"]
        # Prompt template in hr_agent uses `resume_snippets` and `chat_history`
        response = self.hr_chain.predict(
            chat_history=chat_hist,
            resume_snippets=self.resume_snippets
        )
        # Save into memory
        self.hr_memory.chat_memory.add_ai_message(response)
        return response

    def generate_verification_questions(self):
        return [self._generate_response()]

    def generate_skill_questions(self, skills: str):
        # you can pass skills via memory or append to snippets if desired
        return [self._generate_response()]

    def generate_role_exploration_question(self, candidate_info: dict):
        return [self._generate_response()]

    def generate_follow_up(self, last_response: str, current_topic: str):
        # add the candidate’s last message into memory first
        self.hr_memory.chat_memory.add_user_message(last_response)
        return [self._generate_response()]

    def wrap_up_interview(self) -> str:
        return self._generate_response()

    def next_question(self, last_user_reply: Optional[str] = None):
        """
        Alternate through stages and invoke the real HR LLM each time.
        """
        if self.stage == "verification":
            self.stage = "skill"
            return self.generate_verification_questions()

        if self.stage == "skill":
            self.stage = "role_explore"
            return self.generate_skill_questions(skills="")  # pass real skills if available

        if self.stage == "role_explore":
            self.stage = "rag"
            return self.generate_role_exploration_question({})

        if self.stage == "rag":
            self.stage = "followup"
            return self.generate_follow_up(last_user_reply or "", current_topic="")

        if self.stage == "followup":
            self.stage = "wrap_up"
            return [self.wrap_up_interview()]

        return []
