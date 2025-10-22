"""
Patient Simulator

LLM-based patient that generates realistic responses based on personality and symptoms.
"""

import logging
from typing import Dict, Any, List, Optional
from src.models.patient_profile import PatientProfile
from src.models.patient_state import PatientState
from src.utils.prompt_templates import format_patient_prompt, format_opening_message

logger = logging.getLogger(__name__)


class PatientSimulator:
    """
    Simulates patient behavior using LLM
    """

    def __init__(self, patient_profile: PatientProfile, current_state: PatientState,
                 llm_model: Optional[str] = None, device: str = "auto",
                 quantization: Optional[str] = "4bit"):
        """
        Initialize patient simulator

        Args:
            patient_profile: Full patient profile
            current_state: Current biological/psychological state
            llm_model: LLM model to use (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
                      If None, uses rule-based responses for testing
            device: Device to load model on ('cuda', 'cpu', 'auto')
            quantization: Quantization type ('4bit', '8bit', or None)
        """
        self.patient_profile = patient_profile
        self.current_state = current_state
        self.llm_model_name = llm_model
        self.device = device
        self.quantization = quantization
        self.system_prompt = None
        self.conversation_history = []

        # LLM components
        self.model = None
        self.tokenizer = None
        self.use_llm = False

        # Try to load LLM if model name provided
        if llm_model:
            self._try_load_llm()

        if self.use_llm:
            logger.info(f"Initialized PatientSimulator for {patient_profile.patient_id} with LLM: {llm_model}")
        else:
            logger.info(f"Initialized PatientSimulator for {patient_profile.patient_id} (rule-based fallback)")

    def _try_load_llm(self):
        """
        Attempt to load LLM model

        Sets self.use_llm = True if successful
        """
        try:
            from src.utils.llm_utils import load_model_and_tokenizer, check_transformers_available

            if not check_transformers_available():
                logger.warning("Transformers not available. Using rule-based responses.")
                return

            logger.info(f"Loading patient LLM: {self.llm_model_name}")

            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name=self.llm_model_name,
                device=self.device,
                quantization=self.quantization,
                use_cache=True
            )

            self.use_llm = True
            logger.info("✓ Patient LLM loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load LLM, using rule-based fallback: {e}")
            self.use_llm = False

    def create_system_prompt(self) -> str:
        """
        Generate comprehensive system prompt for patient LLM

        Returns:
            Formatted prompt string
        """
        # Convert patient state to dictionary format for prompt
        state_dict = {
            "symptoms": {
                name: {
                    "objective_severity": symptom.objective_severity,
                    "description": symptom.description
                }
                for name, symptom in self.current_state.current_symptoms.items()
            }
        }

        # Convert patient profile to dictionary
        profile_dict = self.patient_profile.to_dict()

        self.system_prompt = format_patient_prompt(
            patient_profile=profile_dict,
            current_state=state_dict,
            current_day=self.current_state.current_day
        )

        return self.system_prompt

    def generate_opening_message(self) -> str:
        """
        Generate patient's initial message to start conversation

        Returns:
            Opening message string
        """
        logger.debug("Generating opening message")

        symptoms_dict = {
            name: symptom.to_dict()
            for name, symptom in self.current_state.current_symptoms.items()
        }

        # Use template to generate opening
        opening = format_opening_message(
            self.patient_profile.to_dict(),
            symptoms_dict
        )

        logger.debug(f"Opening message: {opening}")
        return opening

    def respond(self, chatbot_message: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate patient response to chatbot

        Args:
            chatbot_message: Most recent message from chatbot
            conversation_history: List of previous turns [{"speaker": "patient/chatbot", "message": "..."}]

        Returns:
            Patient's response message
        """
        if self.use_llm:
            return self._respond_with_llm(chatbot_message, conversation_history)
        else:
            return self._respond_rule_based(chatbot_message, conversation_history)

    def _respond_with_llm(self, chatbot_message: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate response using LLM

        Args:
            chatbot_message: Chatbot's message
            conversation_history: Previous conversation

        Returns:
            Generated response
        """
        try:
            from src.utils.llm_utils import generate_response

            # Create or get system prompt
            if not self.system_prompt:
                self.create_system_prompt()

            # Format conversation for model
            prompt = self._format_conversation_for_llm(conversation_history, chatbot_message)

            # Generate response
            response = generate_response(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=256,  # Shorter for patient responses
                temperature=0.8,  # Higher temperature for more natural/varied responses
                top_p=0.95,
                do_sample=True,
                stop_strings=["Patient:", "Chatbot:", "Doctor:", "\n\n\n"]
            )

            # Clean up response
            response = response.strip()

            # If response is too short or empty, fallback to rule-based
            if len(response) < 5:
                logger.warning("LLM generated very short response, using fallback")
                return self._respond_rule_based(chatbot_message, conversation_history)

            logger.debug(f"LLM response: {response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}, using rule-based fallback")
            return self._respond_rule_based(chatbot_message, conversation_history)

    def _format_conversation_for_llm(self, conversation_history: List[Dict[str, str]],
                                    current_message: str) -> str:
        """
        Format conversation for LLM input

        Args:
            conversation_history: Previous conversation turns
            current_message: Current chatbot message

        Returns:
            Formatted prompt string
        """
        # Get system prompt
        if not self.system_prompt:
            self.create_system_prompt()

        # Build conversation
        formatted = f"{self.system_prompt}\n\n"
        formatted += "=== CONVERSATION ===\n\n"

        # Add conversation history
        for turn in conversation_history:
            speaker = turn.get("speaker", "unknown")
            message = turn.get("message", "")

            if speaker == "patient":
                formatted += f"You: {message}\n\n"
            elif speaker == "chatbot":
                formatted += f"Chatbot: {message}\n\n"

        # Add current message
        formatted += f"Chatbot: {current_message}\n\n"
        formatted += "You: "

        return formatted

    def _respond_rule_based(self, chatbot_message: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate response using rule-based system (for testing without LLM)

        Args:
            chatbot_message: Chatbot's message
            conversation_history: Previous conversation

        Returns:
            Generated response based on rules
        """
        message_lower = chatbot_message.lower()

        # Get communication style
        formality = self.patient_profile.communication.formality
        anxiety = self.patient_profile.psychological.health_anxiety
        verbosity = self.patient_profile.communication.verbosity

        # Determine response based on question type
        response = ""

        # Questions about symptoms
        if any(word in message_lower for word in ["symptom", "feel", "feeling", "experience"]):
            response = self._describe_current_symptoms(verbosity)

        # Questions about duration
        elif any(word in message_lower for word in ["long", "when", "start", "began"]):
            response = self._describe_duration()

        # Questions about severity
        elif any(word in message_lower for word in ["severe", "bad", "worse", "better"]):
            response = self._describe_severity()

        # Questions about other symptoms
        elif "other" in message_lower or "else" in message_lower:
            response = self._describe_other_symptoms(verbosity)

        # Questions about medications
        elif any(word in message_lower for word in ["medication", "medicine", "take", "tried"]):
            response = self._describe_medications()

        # Questions about medical history
        elif "history" in message_lower or "before" in message_lower:
            response = self._describe_history()

        # Worried/anxiety expressions
        elif any(word in message_lower for word in ["worry", "serious", "concern"]):
            if anxiety >= 7:
                response = "Yeah, I'm actually pretty worried about this. Do you think it could be something serious?"
            else:
                response = "I'm a bit concerned but trying not to worry too much."

        # Thanks/acknowledgment
        elif any(word in message_lower for word in ["thank", "ok", "alright", "understand"]):
            if formality <= 3:
                response = "Thank you for your help."
            elif formality <= 6:
                response = "Thanks, I appreciate it."
            else:
                response = "Thanks, that helps."

        # Generic greeting response
        elif any(word in message_lower for word in ["hello", "hi", "how can"]):
            response = self.generate_opening_message()

        # Default response
        else:
            if verbosity >= 7:
                response = "Well, I guess the main thing is just that I haven't been feeling right. It's hard to explain exactly."
            else:
                response = "Not really sure what else to say."

        # Add anxiety-based worry if high anxiety
        if anxiety >= 8 and len(conversation_history) > 2 and "worry" not in response.lower():
            if "?" not in response:
                response += " Should I be worried about this?"

        return response

    def _describe_current_symptoms(self, verbosity: int) -> str:
        """Describe current symptoms based on verbosity"""
        symptoms = self.current_state.current_symptoms

        if not symptoms:
            return "Actually, I'm feeling okay right now."

        # Get top symptoms by severity
        sorted_symptoms = sorted(
            symptoms.items(),
            key=lambda x: x[1].reported_severity,
            reverse=True
        )

        if verbosity <= 3:
            # Terse - just mention main symptom
            main = sorted_symptoms[0]
            return f"I have {main[0].replace('_', ' ')}. {main[1].description.capitalize()}."

        elif verbosity <= 6:
            # Medium - mention top 2-3
            descriptions = []
            for name, symptom in sorted_symptoms[:2]:
                descriptions.append(f"{symptom.description}")
            return f"I've got {' and '.join(descriptions)}."

        else:
            # Verbose - mention all symptoms
            descriptions = []
            for name, symptom in sorted_symptoms:
                severity_desc = self._severity_to_word(symptom.reported_severity)
                descriptions.append(f"{severity_desc} {symptom.description}")
            return f"I'm dealing with {', '.join(descriptions[:-1])} and {descriptions[-1]}."

    def _describe_duration(self) -> str:
        """Describe how long symptoms have been present"""
        days = self.current_state.disease_state.disease_day

        if days <= 1:
            return "Just started today actually."
        elif days == 2:
            return "Started yesterday."
        elif days <= 4:
            return f"It's been a few days, maybe {days} days?"
        else:
            return f"It's been going on for about {days} days now."

    def _describe_severity(self) -> str:
        """Describe symptom severity"""
        if not self.current_state.current_symptoms:
            return "Not too bad actually."

        avg_severity = sum(s.reported_severity for s in self.current_state.current_symptoms.values()) / len(self.current_state.current_symptoms)

        trajectory = self.current_state.disease_state.trajectory

        if trajectory == "worsening":
            return f"It's getting worse. Started out mild but now it's {self._severity_to_word(avg_severity)}."
        elif trajectory == "improving":
            return f"Actually getting a bit better. Was worse before but now it's {self._severity_to_word(avg_severity)}."
        else:
            return f"It's been pretty consistent, I'd say {self._severity_to_word(avg_severity)}."

    def _describe_other_symptoms(self, verbosity: int) -> str:
        """Describe other symptoms not yet mentioned"""
        symptoms = self.current_state.current_symptoms

        if len(symptoms) <= 1:
            return "No, that's pretty much it."

        # Get minor symptoms
        minor_symptoms = [
            (name, s) for name, s in symptoms.items()
            if s.reported_severity < 5
        ]

        if not minor_symptoms:
            return "Nothing else really."

        if verbosity >= 6:
            descriptions = [s.description for _, s in minor_symptoms]
            return f"Well, I also have {' and '.join(descriptions)}."
        else:
            return f"Just some {minor_symptoms[0][0].replace('_', ' ')} too."

    def _describe_medications(self) -> str:
        """Describe medications tried"""
        current_meds = self.patient_profile.biological.medications.get("current", [])

        if not current_meds:
            return "I haven't tried anything yet. Should I take something?"

        med_names = [m["name"] for m in current_meds]
        return f"I've tried {', '.join(med_names)} but it doesn't seem to help much."

    def _describe_history(self) -> str:
        """Describe medical history"""
        chronic = self.patient_profile.biological.chronic_conditions
        risk_factors = self.patient_profile.biological.risk_factors

        if not chronic and not risk_factors:
            return "Nothing really, I'm generally healthy."

        parts = []
        if chronic:
            parts.append(f"I have {chronic[0].type}")

        if risk_factors:
            parts.append(risk_factors[0].details)

        if parts:
            return " and ".join(parts) + "."
        return "Nothing significant."

    def _severity_to_word(self, severity: float) -> str:
        """Convert severity number to descriptive word"""
        if severity < 2:
            return "very mild"
        elif severity < 4:
            return "mild"
        elif severity < 6:
            return "moderate"
        elif severity < 8:
            return "pretty bad"
        else:
            return "really bad"

    def should_volunteer_info(self) -> bool:
        """
        Determine if patient should volunteer information without being asked

        Returns:
            True if patient should volunteer info
        """
        volunteering_level = self.patient_profile.psychological.information_volunteering

        if volunteering_level == "high":
            return True
        elif volunteering_level == "low":
            return False
        else:
            # Medium - random chance
            import random
            return random.random() < 0.5

    def get_anxiety_interjection(self) -> Optional[str]:
        """
        Get an anxiety-based interjection if anxiety is high

        Returns:
            Anxiety-related question or None
        """
        anxiety = self.current_state.psychological.anxiety_level

        if anxiety < 7:
            return None

        interjections = [
            "Do you think this could be something serious?",
            "I'm worried this might be something bad.",
            "Should I be concerned about this?",
            "Is this normal or should I see a doctor?",
        ]

        import random
        return random.choice(interjections)
