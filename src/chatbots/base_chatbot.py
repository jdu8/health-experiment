"""
Base Chatbot

Abstract base class for all chatbot implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseChatbot(ABC):
    """
    Abstract base class for medical chatbot implementations
    """

    def __init__(self, context: Optional[List[Dict[str, Any]]] = None, condition: str = "without_history"):
        """
        Initialize chatbot

        Args:
            context: Previous conversation history (empty for without_history condition)
            condition: 'with_history' or 'without_history'
        """
        self.context = context or []
        self.condition = condition
        self.conversation_count = len(context) if context else 0

        logger.info(f"Initialized {self.__class__.__name__} with condition: {condition}")

    @abstractmethod
    def respond(self, conversation_turns: List[Dict[str, str]]) -> str:
        """
        Generate chatbot response to current conversation

        Args:
            conversation_turns: Current conversation turns
                               [{"speaker": "patient/chatbot", "message": "..."}]

        Returns:
            Response message string
        """
        pass

    @abstractmethod
    def format_messages_for_model(self, turns: List[Dict[str, str]]) -> Any:
        """
        Format conversation turns for specific model input format

        Args:
            turns: Conversation turns

        Returns:
            Formatted input for model (format depends on implementation)
        """
        pass

    def add_to_context(self, conversation_summary: Dict[str, Any]):
        """
        Add completed conversation to context

        Args:
            conversation_summary: Summary of conversation to add to context
        """
        self.context.append(conversation_summary)
        self.conversation_count += 1
        logger.debug(f"Added conversation to context. Total: {self.conversation_count}")

    def get_context(self) -> List[Dict[str, Any]]:
        """
        Get conversation context

        Returns:
            List of conversation summaries
        """
        if self.condition == "without_history":
            return []
        return self.context

    def has_context(self) -> bool:
        """
        Check if chatbot has conversation context

        Returns:
            True if context exists and condition allows it
        """
        return self.condition == "with_history" and len(self.context) > 0

    def get_system_prompt(self) -> str:
        """
        Get system prompt for chatbot

        Returns:
            System prompt string
        """
        from src.utils.prompt_templates import format_chatbot_prompt

        return format_chatbot_prompt(
            condition=self.condition,
            conversation_history=self.context if self.has_context() else None
        )

    def is_question(self, message: str) -> bool:
        """
        Check if message is a question

        Args:
            message: Message to check

        Returns:
            True if message appears to be a question
        """
        message_lower = message.lower()

        # Check for question mark
        if "?" in message:
            return True

        # Check for question words
        question_words = [
            "what", "when", "where", "who", "why", "how",
            "can you", "could you", "would you", "are you",
            "do you", "did you", "have you", "is it", "was it"
        ]

        return any(word in message_lower for word in question_words)

    def extract_questions(self, message: str) -> List[str]:
        """
        Extract questions from a message

        Args:
            message: Message text

        Returns:
            List of question strings
        """
        # Simple implementation - split by sentence and check each
        import re

        sentences = re.split(r'[.!?]+', message)
        questions = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and self.is_question(sentence + "?"):
                questions.append(sentence)

        return questions

    def count_questions(self, conversation_turns: List[Dict[str, str]]) -> int:
        """
        Count questions asked by chatbot

        Args:
            conversation_turns: Conversation turns

        Returns:
            Number of questions asked
        """
        count = 0
        for turn in conversation_turns:
            if turn.get("speaker") == "chatbot":
                if self.is_question(turn.get("message", "")):
                    count += 1
        return count

    def get_conversation_summary(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Create summary of conversation for context

        Args:
            turns: Conversation turns

        Returns:
            Summary dictionary
        """
        # Extract key information mentioned
        all_text = " ".join([t.get("message", "") for t in turns]).lower()

        key_info = []
        if "day" in all_text or "long" in all_text:
            key_info.append("duration")
        if "fever" in all_text or "temperature" in all_text:
            key_info.append("fever")
        if "pain" in all_text or "hurt" in all_text:
            key_info.append("pain")
        if "medication" in all_text or "medicine" in all_text:
            key_info.append("medications")

        return {
            "turn_count": len(turns),
            "questions_asked": self.count_questions(turns),
            "key_info": key_info
        }
