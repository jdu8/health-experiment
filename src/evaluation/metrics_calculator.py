"""
Metrics Calculator

Calculates automatic metrics for conversation quality evaluation.
"""

import logging
from typing import List, Dict, Any, Optional
from src.models.conversation import Conversation, ConversationMetrics, AutomaticMetrics, EvaluationMetrics
from src.models.patient_state import PatientState

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates automatic metrics for conversation evaluation
    """

    def __init__(self, condition: str):
        """
        Initialize metrics calculator

        Args:
            condition: 'with_history' or 'without_history'
        """
        self.condition = condition
        logger.info(f"Initialized MetricsCalculator for condition: {condition}")

    def calculate_metrics(self, conversation: Conversation,
                         patient_history: Optional[List[Dict[str, Any]]] = None) -> ConversationMetrics:
        """
        Calculate all metrics for a conversation

        Args:
            conversation: Completed conversation
            patient_history: Previous conversations (for redundancy checking)

        Returns:
            ConversationMetrics with automatic and evaluation metrics
        """
        logger.debug(f"Calculating metrics for conversation {conversation.conversation_id}")

        # Initialize metrics
        auto_metrics = AutomaticMetrics()

        # Count bot questions
        auto_metrics.bot_questions_asked = self._count_bot_questions(conversation.turns)

        # Detect redundant questions
        auto_metrics.redundant_questions = self._count_redundant_questions(
            conversation.turns,
            patient_history
        )

        # Count references to history
        auto_metrics.references_to_history = self._count_history_references(conversation.turns)

        # Count contextual questions
        auto_metrics.contextual_questions = self._count_contextual_questions(conversation.turns)

        # Identify key information gathered
        auto_metrics.key_info_gathered = self._identify_key_info_gathered(conversation.turns)

        # Identify key information missed
        auto_metrics.key_info_missed = self._identify_key_info_missed(
            conversation.turns,
            conversation.patient_state_at_start
        )

        # Count voluntary vs prompted information
        auto_metrics.patient_volunteered_count = self._count_volunteered_info(conversation.turns)
        auto_metrics.patient_only_when_asked_count = self._count_prompted_info(conversation.turns)

        # Create evaluation metrics (empty for now)
        eval_metrics = EvaluationMetrics(evaluated=False)

        # Create full metrics object
        metrics = ConversationMetrics(
            automatic=auto_metrics,
            evaluation=eval_metrics
        )

        logger.info(f"Metrics calculated: {auto_metrics.bot_questions_asked} questions, "
                   f"{len(auto_metrics.key_info_gathered)} info gathered")

        return metrics

    def _count_bot_questions(self, turns: List) -> int:
        """
        Count questions asked by bot

        Args:
            turns: Conversation turns

        Returns:
            Number of questions
        """
        count = 0
        for turn in turns:
            if turn.speaker == "chatbot" and self._is_question(turn.message):
                count += 1
        return count

    def _is_question(self, message: str) -> bool:
        """
        Check if message is a question

        Args:
            message: Message text

        Returns:
            True if question
        """
        if "?" in message:
            return True

        question_words = [
            "what", "when", "where", "who", "why", "how",
            "can you", "could you", "would you",
            "do you", "did you", "have you", "are you",
            "is it", "was it", "tell me"
        ]

        message_lower = message.lower()
        return any(word in message_lower for word in question_words)

    def _count_redundant_questions(self, turns: List,
                                   patient_history: Optional[List[Dict[str, Any]]]) -> int:
        """
        Count questions that ask for information already in history

        Only applies to 'with_history' condition

        Args:
            turns: Current conversation turns
            patient_history: Previous conversations (can be dict or ConversationHistorySummary objects)

        Returns:
            Number of redundant questions
        """
        if self.condition != "with_history" or not patient_history:
            return 0

        # Get info from history
        history_info = set()
        for past_conv in patient_history:
            # Handle both dict and ConversationHistorySummary objects
            if hasattr(past_conv, 'key_info'):
                # ConversationHistorySummary object
                key_info = past_conv.key_info if past_conv.key_info else []
            else:
                # Dictionary
                key_info = past_conv.get("key_info", [])
            history_info.update(key_info)

        # Check bot questions against history
        redundant_count = 0
        for turn in turns:
            if turn.speaker == "chatbot" and self._is_question(turn.message):
                # Check if question asks about info in history
                if self._question_about_known_info(turn.message, history_info):
                    redundant_count += 1
                    logger.debug(f"Redundant question detected: {turn.message[:50]}")

        return redundant_count

    def _question_about_known_info(self, question: str, known_info: set) -> bool:
        """
        Check if question asks about information we already have

        Args:
            question: Question text
            known_info: Set of known information types

        Returns:
            True if question is redundant
        """
        question_lower = question.lower()

        # Map question patterns to info types
        info_patterns = {
            "duration": ["how long", "when did", "when started", "how many days"],
            "fever": ["fever", "temperature", "hot"],
            "medications": ["medication", "medicine", "taking", "tried"],
            "symptoms": ["symptom", "feeling", "experience"],
            "progression": ["better", "worse", "changing"],
        }

        for info_type, patterns in info_patterns.items():
            if info_type in known_info:
                if any(pattern in question_lower for pattern in patterns):
                    return True

        return False

    def _count_history_references(self, turns: List) -> int:
        """
        Count phrases that reference previous conversations

        Args:
            turns: Conversation turns

        Returns:
            Number of references
        """
        count = 0
        reference_phrases = [
            "you mentioned", "last time", "when we talked",
            "previously", "you said", "before", "earlier"
        ]

        for turn in turns:
            if turn.speaker == "chatbot":
                message_lower = turn.message.lower()
                if any(phrase in message_lower for phrase in reference_phrases):
                    count += 1

        return count

    def _count_contextual_questions(self, turns: List) -> int:
        """
        Count questions that build on previous answers (show good flow)

        Args:
            turns: Conversation turns

        Returns:
            Number of contextual questions
        """
        contextual_count = 0

        for i, turn in enumerate(turns):
            if turn.speaker == "chatbot" and self._is_question(turn.message):
                # Check if it references something from previous turn
                if i > 0:
                    prev_turn = turns[i-1]
                    if prev_turn.speaker == "patient":
                        # Check if question builds on patient's last answer
                        if self._builds_on_previous(turn.message, prev_turn.message):
                            contextual_count += 1

        return contextual_count

    def _builds_on_previous(self, question: str, previous_answer: str) -> bool:
        """
        Check if question builds on previous answer

        Args:
            question: Current question
            previous_answer: Previous patient answer

        Returns:
            True if contextual
        """
        # Extract key words from answer
        answer_words = set(previous_answer.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "is", "was", "been", "have", "has", "i", "you"}
        answer_words = answer_words - common_words

        if not answer_words:
            return False

        # Check if question references words from answer
        question_lower = question.lower()
        overlap = sum(1 for word in answer_words if word in question_lower)

        # If 2+ words overlap, likely contextual
        return overlap >= 2

    def _identify_key_info_gathered(self, turns: List) -> List[str]:
        """
        Identify what key information was successfully gathered

        Args:
            turns: Conversation turns

        Returns:
            List of information types gathered
        """
        all_text = " ".join([t.message for t in turns]).lower()
        gathered = []

        # Check what information was discussed
        info_indicators = {
            "duration": ["day", "days", "started", "began", "since", "ago", "week"],
            "severity": ["severe", "bad", "mild", "moderate", "terrible", "awful"],
            "fever": ["fever", "temperature", "hot", "chills", "degrees"],
            "other_symptoms": ["also", "other", "another", "along with"],
            "medications": ["medication", "medicine", "pills", "took", "taking", "tried"],
            "progression": ["better", "worse", "same", "improving", "worsening"],
            "lifestyle": ["sleep", "stress", "work", "exercise", "diet"],
            "medical_history": ["history", "before", "previous", "past", "chronic"]
        }

        for info_type, indicators in info_indicators.items():
            if any(indicator in all_text for indicator in indicators):
                gathered.append(info_type)

        return gathered

    def _identify_key_info_missed(self, turns: List, patient_state: Any) -> List[str]:
        """
        Identify important information that chatbot failed to ask about

        Args:
            turns: Conversation turns
            patient_state: Patient state at start

        Returns:
            List of information types missed
        """
        gathered = self._identify_key_info_gathered(turns)
        all_text = " ".join([t.message for t in turns]).lower()

        missed = []

        # Check for critical info that should have been asked
        # Duration
        if "duration" not in gathered and len(turns) > 4:
            missed.append("duration")

        # Fever (if fever-related symptoms mentioned)
        if any(word in all_text for word in ["hot", "chills", "sick"]):
            if "fever" not in gathered:
                missed.append("fever_temperature")

        # Other symptoms (if only one symptom mentioned)
        symptom_mentions = sum(1 for word in ["throat", "cough", "headache", "tired", "ache"] if word in all_text)
        if symptom_mentions == 1 and "other_symptoms" not in gathered:
            missed.append("other_symptoms")

        # Medications tried
        if "medications" not in gathered and len(turns) > 6:
            missed.append("medications_tried")

        # Progression
        if "progression" not in gathered and len(turns) > 6:
            missed.append("symptom_progression")

        return missed

    def _count_volunteered_info(self, turns: List) -> int:
        """
        Count information patient volunteered without being asked

        Args:
            turns: Conversation turns

        Returns:
            Count of volunteered information
        """
        volunteered = 0

        for i, turn in enumerate(turns):
            if turn.speaker == "patient":
                # Check if this info was in response to a direct question
                if i > 0:
                    prev_turn = turns[i-1]
                    if prev_turn.speaker == "chatbot" and not self._is_question(prev_turn.message):
                        # Patient added info without being asked
                        volunteered += 1

        return volunteered

    def _count_prompted_info(self, turns: List) -> int:
        """
        Count information patient only provided when asked

        Args:
            turns: Conversation turns

        Returns:
            Count of prompted information
        """
        prompted = 0

        for i, turn in enumerate(turns):
            if turn.speaker == "patient":
                # Check if this was in response to a question
                if i > 0:
                    prev_turn = turns[i-1]
                    if prev_turn.speaker == "chatbot" and self._is_question(prev_turn.message):
                        prompted += 1

        return prompted

    def get_metrics_summary(self, metrics: ConversationMetrics) -> str:
        """
        Get human-readable summary of metrics

        Args:
            metrics: Calculated metrics

        Returns:
            Summary string
        """
        auto = metrics.automatic

        summary = f"""
Automatic Metrics Summary:
- Bot Questions Asked: {auto.bot_questions_asked}
- Redundant Questions: {auto.redundant_questions}
- References to History: {auto.references_to_history}
- Contextual Questions: {auto.contextual_questions}
- Key Info Gathered: {len(auto.key_info_gathered)} ({', '.join(auto.key_info_gathered)})
- Key Info Missed: {len(auto.key_info_missed)} ({', '.join(auto.key_info_missed)})
- Patient Volunteered: {auto.patient_volunteered_count}
- Patient Prompted: {auto.patient_only_when_asked_count}
"""

        return summary.strip()
