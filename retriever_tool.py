from typing import Any
from pydantic import PrivateAttr
from crewai.tools import BaseTool

class ResumeRetrieverTool(BaseTool):
    name: str = "resume_knowledge_tool"  # Name of the tool
    description: str = "Fetch relevant resume snippets for a given query."  # Description of the tool

    _conversation_chain: Any = PrivateAttr()  # Private attribute to store the conversation chain

    def __init__(self, conversation_chain, **kwargs):
        super().__init__(**kwargs)  # Initialize the base class
        self._conversation_chain = conversation_chain  # Assign the conversation chain

    def _run(self, query: str) -> str:
        response = self._conversation_chain.invoke({"question": query})  # Invoke the chain with the query
        return response["answer"]  # Return the answer from the response

    async def _arun(self, query: str) -> str:
        return self._run(query)  # Call the synchronous method for async execution
