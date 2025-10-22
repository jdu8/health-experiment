"""
Conversation Runner

Orchestrates back-and-forth between patient and chatbot.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from src.core.patient_simulator import PatientSimulator
from src.chatbots.base_chatbot import BaseChatbot
from src.models.conversation import (
    Conversation, ConversationTurn, ConversationSummary,
    PatientStateSnapshot, BiologicalState as ConvBioState,
    PsychologicalState as ConvPsychState
)
from src.models.patient_state import PatientState

logger = logging.getLogger(__name__)


class ConversationRunner:
    """
    Orchestrates conversation between patient and chatbot
    """

    def __init__(self, patient_simulator: PatientSimulator, chatbot: BaseChatbot,
                 max_turns: int = 10):
        """
        Initialize conversation runner

        Args:
            patient_simulator: Initialized patient simulator
            chatbot: Initialized chatbot (with or without history)
            max_turns: Maximum conversation turns (patient + chatbot pairs)
        """
        self.patient_simulator = patient_simulator
        self.chatbot = chatbot
        self.max_turns = max_turns
        self.start_time = None

        logger.info(f"Initialized ConversationRunner (max_turns={max_turns})")

    def run_conversation(self, conversation_id: str, patient_id: str,
                        simulation_day: int, disease_day: int,
                        patient_state: PatientState) -> Conversation:
        """
        Execute full conversation

        Args:
            conversation_id: Unique conversation identifier
            patient_id: Patient identifier
            simulation_day: Current simulation day
            disease_day: Current disease day
            patient_state: Current patient state

        Returns:
            Conversation object with all turns and metrics
        """
        logger.info(f"Starting conversation: {conversation_id}")
        self.start_time = time.time()

        # Create conversation object
        patient_state_snapshot = self._create_state_snapshot(patient_state)

        conversation = Conversation(
            conversation_id=conversation_id,
            patient_id=patient_id,
            simulation_day=simulation_day,
            disease_day=disease_day,
            real_timestamp=datetime.now().isoformat(),
            condition=self.chatbot.condition,
            patient_state_at_start=patient_state_snapshot,
            turns=[],
            conversation_summary=None,
            metrics=None
        )

        # Get patient opening message
        opening_message = self.patient_simulator.generate_opening_message()

        # Add patient's opening turn
        conversation.add_turn(
            speaker="patient",
            message=opening_message,
            internal_state={
                "anxiety_level": patient_state.psychological.anxiety_level,
                "symptoms_mentioned": [],
                "information_volunteered": []
            }
        )

        logger.debug(f"Patient opens: {opening_message[:50]}...")

        # Conversation loop
        turn_count = 0
        conversation_history = []

        while turn_count < self.max_turns:
            # Format conversation history for models
            conversation_history = [
                {"speaker": t.speaker, "message": t.message}
                for t in conversation.turns
            ]

            # Chatbot responds
            chatbot_response = self.chatbot.respond(conversation_history)

            conversation.add_turn(
                speaker="chatbot",
                message=chatbot_response,
                model_info={
                    "model_name": getattr(self.chatbot, 'model_name', 'rule_based'),
                    "temperature": 0.7,
                    "max_tokens": 512
                }
            )

            logger.debug(f"Chatbot: {chatbot_response[:50]}...")

            # Check if conversation should end after chatbot response
            if self.should_end_conversation(chatbot_response, "chatbot", conversation.turns):
                logger.info(f"Conversation ending after chatbot response (turn {turn_count})")
                break

            # Patient responds
            patient_response = self.patient_simulator.respond(chatbot_response, conversation_history)

            # Track what info patient mentioned
            symptoms_mentioned = self._extract_symptoms_mentioned(patient_response)

            conversation.add_turn(
                speaker="patient",
                message=patient_response,
                internal_state={
                    "anxiety_level": patient_state.psychological.anxiety_level,
                    "symptoms_mentioned": symptoms_mentioned,
                    "information_volunteered": []  # Could be enhanced
                }
            )

            logger.debug(f"Patient: {patient_response[:50]}...")

            # Check if conversation should end after patient response
            if self.should_end_conversation(patient_response, "patient", conversation.turns):
                logger.info(f"Conversation ending after patient response (turn {turn_count})")
                break

            turn_count += 1

        # Calculate conversation summary
        duration = time.time() - self.start_time
        end_reason = self._determine_end_reason(conversation.turns)

        conversation.conversation_summary = ConversationSummary(
            total_turns=len(conversation.turns),
            duration_seconds=duration,
            conversation_end_reason=end_reason,
            final_bot_recommendation=self._extract_final_recommendation(conversation.turns)
        )

        logger.info(f"Conversation complete: {len(conversation.turns)} turns in {duration:.1f}s")

        return conversation

    def _create_state_snapshot(self, patient_state: PatientState) -> PatientStateSnapshot:
        """
        Create snapshot of patient state at conversation start

        Args:
            patient_state: Current patient state

        Returns:
            PatientStateSnapshot
        """
        bio_state = ConvBioState(
            temperature=patient_state.biological.temperature,
            heart_rate=patient_state.biological.heart_rate,
            energy_level=patient_state.biological.energy_level,
            pain_level=patient_state.biological.pain_level
        )

        psych_state = ConvPsychState(
            anxiety_level=patient_state.psychological.anxiety_level,
            mood=patient_state.psychological.mood,
            health_concern_level=patient_state.psychological.health_concern_level
        )

        return PatientStateSnapshot(
            biological=bio_state,
            psychological=psych_state
        )

    def should_end_conversation(self, last_message: str, speaker: str,
                                turns: List[ConversationTurn]) -> bool:
        """
        Determine if conversation should end

        Criteria:
        - Max turns reached
        - Bot provided final advice/recommendation
        - Patient explicitly ends (e.g., "ok thanks")
        - Conversation became circular/repetitive

        Args:
            last_message: Most recent message
            speaker: Who sent the message
            turns: All conversation turns

        Returns:
            True if conversation should end
        """
        # Max turns reached
        if len(turns) >= self.max_turns * 2:  # *2 because turns include both patient and chatbot
            logger.debug("Max turns reached")
            return True

        # Bot provided advice/recommendation
        if speaker == "chatbot":
            advice_indicators = [
                "recommend", "suggest", "should see", "advice",
                "here's what", "you should", "best to", "would be good"
            ]
            if any(indicator in last_message.lower() for indicator in advice_indicators):
                logger.debug("Bot provided advice")
                return True

        # Patient ending conversation
        if speaker == "patient":
            end_phrases = ["thank", "thanks", "ok", "okay", "alright", "bye", "goodbye", "got it"]
            message_lower = last_message.lower().strip()

            # Short message with ending phrase
            if len(message_lower) < 50 and any(phrase in message_lower for phrase in end_phrases):
                logger.debug("Patient ending conversation")
                return True

        # Check for circular conversation (same question asked multiple times)
        if self._is_circular(turns):
            logger.debug("Conversation became circular")
            return True

        return False

    def _is_circular(self, turns: List[ConversationTurn]) -> bool:
        """
        Check if conversation has become circular/repetitive

        Args:
            turns: Conversation turns

        Returns:
            True if circular
        """
        # Get last few chatbot questions
        chatbot_turns = [t for t in turns if t.speaker == "chatbot"]
        if len(chatbot_turns) < 3:
            return False

        last_three = chatbot_turns[-3:]

        # Check for similar questions
        for i, turn1 in enumerate(last_three):
            for turn2 in last_three[i+1:]:
                # Simple similarity check - more than 50% word overlap
                words1 = set(turn1.message.lower().split())
                words2 = set(turn2.message.lower().split())

                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1 & words2) / min(len(words1), len(words2))
                    if overlap > 0.5:
                        return True

        return False

    def _determine_end_reason(self, turns: List[ConversationTurn]) -> str:
        """
        Determine why conversation ended

        Args:
            turns: Conversation turns

        Returns:
            End reason string
        """
        if len(turns) >= self.max_turns * 2:
            return "max_turns_reached"

        last_turn = turns[-1] if turns else None
        if not last_turn:
            return "unknown"

        if last_turn.speaker == "chatbot":
            if any(word in last_turn.message.lower() for word in ["recommend", "suggest", "advice"]):
                return "bot_provided_advice"

        if last_turn.speaker == "patient":
            if any(word in last_turn.message.lower() for word in ["thank", "thanks", "ok", "bye"]):
                return "patient_closed"

        if self._is_circular(turns):
            return "circular_conversation"

        return "natural_end"

    def _extract_final_recommendation(self, turns: List[ConversationTurn]) -> Optional[str]:
        """
        Extract final recommendation from bot

        Args:
            turns: Conversation turns

        Returns:
            Recommendation string or None
        """
        # Look through last few bot messages for recommendations
        bot_turns = [t for t in turns if t.speaker == "chatbot"]
        if not bot_turns:
            return None

        for turn in reversed(bot_turns[-3:]):  # Check last 3 bot messages
            message = turn.message.lower()
            if any(word in message for word in ["recommend", "suggest", "should"]):
                # Extract key recommendation
                if "see a" in message or "see your" in message:
                    return "see_doctor"
                elif "rest" in message and "hydrate" in message:
                    return "self_care_with_monitoring"
                elif "monitor" in message:
                    return "monitor_symptoms"

        return "information_provided"

    def _extract_symptoms_mentioned(self, message: str) -> List[str]:
        """
        Extract symptoms mentioned in message

        Args:
            message: Message text

        Returns:
            List of symptom names
        """
        message_lower = message.lower()
        symptoms_found = []

        symptom_keywords = {
            "fever": ["fever", "temperature", "hot", "chills"],
            "sore_throat": ["throat", "swallow"],
            "cough": ["cough", "coughing"],
            "congestion": ["stuffy", "congestion", "congested", "blocked nose"],
            "headache": ["headache", "head hurt"],
            "fatigue": ["tired", "exhausted", "fatigue", "no energy"],
            "body_aches": ["aches", "sore", "muscles hurt"]
        }

        for symptom_name, keywords in symptom_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                symptoms_found.append(symptom_name)

        return symptoms_found

    def get_conversation_topic(self, turns: List[ConversationTurn]) -> str:
        """
        Determine conversation topic from turns

        Args:
            turns: Conversation turns

        Returns:
            Topic description
        """
        all_text = " ".join([t.message for t in turns]).lower()

        # Determine main topic
        if any(word in all_text for word in ["throat", "cough", "cold", "flu"]):
            return "respiratory symptoms"
        elif "fever" in all_text or "temperature" in all_text:
            return "fever concerns"
        elif any(word in all_text for word in ["headache", "migraine"]):
            return "headache"
        elif "tired" in all_text or "fatigue" in all_text:
            return "fatigue"
        else:
            return "general symptoms"

    def extract_key_info(self, turns: List[ConversationTurn]) -> List[str]:
        """
        Extract key information discussed in conversation

        Args:
            turns: Conversation turns

        Returns:
            List of key information points
        """
        all_text = " ".join([t.message for t in turns]).lower()
        key_info = []

        # Check what was discussed
        if any(word in all_text for word in ["day", "days", "since", "started"]):
            key_info.append("duration")

        if "fever" in all_text or "temperature" in all_text:
            key_info.append("fever")

        if any(word in all_text for word in ["throat", "cough", "congestion"]):
            key_info.append("respiratory_symptoms")

        if any(word in all_text for word in ["medication", "medicine", "pills"]):
            key_info.append("medications")

        if any(word in all_text for word in ["better", "worse", "same"]):
            key_info.append("progression")

        if any(word in all_text for word in ["history", "before", "previous"]):
            key_info.append("medical_history")

        return key_info
