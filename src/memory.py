from typing import List, Dict
from langchain.memory import ConversationBufferWindowMemory
from src.utils import logger

class ConversationMemoryManager:
    """
    Manages conversational memory for multi-turn chats.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.memory = ConversationBufferWindowMemory(
            k=self.window_size,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=False
        )

    def get_history_string(self) -> str:
        """
        Retrieves the formatted chat history string.
        """
        vars = self.memory.load_memory_variables({})
        return vars.get("chat_history", "")

    def add_interaction(self, question: str, answer: str):
        """
        Saves a question-answer turn to conversation memory.
        """
        logger.info(f"Adding conversational turn to memory: User='{question[:30]}...' -> AI='{answer[:30]}...'")
        self.memory.save_context(
            {"question": question},
            {"answer": answer}
        )

    def clear(self):
        """
        Resets conversation memory.
        """
        logger.info("Clearing conversational memory.")
        self.memory.clear()

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Returns memory messages in a Streamlit-compatible list format.
        """
        messages = []
        history = self.memory.chat_memory.messages
        for msg in history:
            role = "user" if msg.type == "human" else "assistant"
            messages.append({
                "role": role,
                "content": msg.content
            })
        return messages
