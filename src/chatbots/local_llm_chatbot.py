"""
Local LLM Chatbot

Chatbot implementation using local LLMs (with rule-based fallback).
"""

import logging
from typing import List, Dict, Any, Optional
from src.chatbots.base_chatbot import BaseChatbot

logger = logging.getLogger(__name__)


class LocalLLMChatbot(BaseChatbot):
    """
    Medical chatbot using local LLM (with rule-based fallback)
    """

    def __init__(self, model_name: Optional[str] = None, context: Optional[List[Dict[str, Any]]] = None,
                 condition: str = "without_history", device: str = "auto",
                 quantization: Optional[str] = "4bit"):
        """
        Initialize local LLM chatbot

        Args:
            model_name: HuggingFace model name (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
            context: Conversation history
            condition: 'with_history' or 'without_history'
            device: "cuda", "cpu", or "auto"
            quantization: Quantization type ('4bit', '8bit', or None)
        """
        super().__init__(context, condition)
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.use_llm = False
        self.turn_count = 0
        self.info_gathered = set()

        # LLM components
        self.model = None
        self.tokenizer = None

        # Try to load LLM if model name provided
        if model_name:
            self._try_load_llm()

        if self.use_llm:
            logger.info(f"Initialized LocalLLMChatbot with LLM: {model_name}")
        else:
            logger.info(f"Initialized LocalLLMChatbot (rule-based fallback)")

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

            logger.info(f"Loading chatbot LLM: {self.model_name}")

            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name=self.model_name,
                device=self.device,
                quantization=self.quantization,
                use_cache=True
            )

            self.use_llm = True
            logger.info("✓ Chatbot LLM loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load LLM, using rule-based fallback: {e}")
            self.use_llm = False

    def respond(self, conversation_turns: List[Dict[str, str]]) -> str:
        """
        Generate chatbot response

        Args:
            conversation_turns: Current conversation turns

        Returns:
            Response message
        """
        if self.use_llm:
            return self._respond_with_llm(conversation_turns)
        else:
            return self._respond_rule_based(conversation_turns)

    def _respond_with_llm(self, conversation_turns: List[Dict[str, str]]) -> str:
        """
        Generate response using LLM

        Args:
            conversation_turns: Conversation turns

        Returns:
            Generated response
        """
        try:
            from src.utils.llm_utils import generate_response

            # Format conversation for model
            prompt = self.format_messages_for_model(conversation_turns)

            # Generate response
            response = generate_response(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=384,  # Longer for chatbot responses
                temperature=0.7,  # Moderate temperature for medical advice
                top_p=0.9,
                do_sample=True,
                stop_strings=["Patient:", "Chatbot:", "Assistant:", "\n\n\n"]
            )

            # Clean up response
            response = response.strip()

            # If response is too short or empty, fallback to rule-based
            if len(response) < 10:
                logger.warning("LLM generated very short response, using fallback")
                return self._respond_rule_based(conversation_turns)

            logger.debug(f"LLM response: {response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}, using rule-based fallback")
            return self._respond_rule_based(conversation_turns)

    def _respond_rule_based(self, conversation_turns: List[Dict[str, str]]) -> str:
        """
        Generate response using rule-based system (medical chatbot behavior)

        Args:
            conversation_turns: Conversation turns

        Returns:
            Generated response
        """
        self.turn_count = len([t for t in conversation_turns if t.get("speaker") == "chatbot"])

        # First turn - greeting
        if self.turn_count == 0:
            return "Hello! I'm here to help. How can I assist you today?"

        # Get last patient message
        patient_messages = [t for t in conversation_turns if t.get("speaker") == "patient"]
        if not patient_messages:
            return "I'm here to help with any health concerns. What's bothering you?"

        last_patient_msg = patient_messages[-1].get("message", "").lower()

        # Analyze conversation to determine what info we have
        self._analyze_info_gathered(conversation_turns)

        # Determine response based on what we know and what to ask next
        response = self._determine_next_question(last_patient_msg, conversation_turns)

        return response

    def _analyze_info_gathered(self, conversation_turns: List[Dict[str, str]]):
        """
        Analyze what information has been gathered

        Args:
            conversation_turns: Conversation turns
        """
        all_text = " ".join([t.get("message", "") for t in conversation_turns]).lower()

        # Check what info we have
        if any(word in all_text for word in ["day", "days", "started", "began", "since", "ago"]):
            self.info_gathered.add("duration")

        if any(word in all_text for word in ["fever", "temperature", "hot", "chills"]):
            self.info_gathered.add("fever")

        if any(word in all_text for word in ["throat", "cough", "congestion", "headache", "aches", "tired", "fatigue"]):
            self.info_gathered.add("symptoms")

        if any(word in all_text for word in ["medication", "medicine", "pills", "took", "taking", "tried"]):
            self.info_gathered.add("medications")

        if any(word in all_text for word in ["better", "worse", "worsening", "improving", "same"]):
            self.info_gathered.add("progression")

        if any(word in all_text for word in ["history", "before", "previous", "past"]):
            self.info_gathered.add("history")

    def _determine_next_question(self, last_patient_msg: str, conversation_turns: List[Dict[str, str]]) -> str:
        """
        Determine what question to ask next

        Args:
            last_patient_msg: Last message from patient
            conversation_turns: All conversation turns

        Returns:
            Next question or advice
        """
        # If patient just opened, ask for more details
        if self.turn_count == 1:
            if any(word in last_patient_msg for word in ["sick", "ill", "feeling", "not well", "symptoms"]):
                return "I'm sorry to hear you're not feeling well. Can you tell me more about your symptoms?"

        # Ask about duration if we don't have it
        if "duration" not in self.info_gathered and self.turn_count <= 3:
            return "How long have you been experiencing these symptoms?"

        # Ask about other symptoms
        if "symptoms" in self.info_gathered and len(self.info_gathered) == 1 and self.turn_count <= 4:
            return "Are you experiencing any other symptoms, such as fever, headache, or body aches?"

        # Ask about fever specifically if mentioned but not detailed
        if "fever" not in self.info_gathered and self.turn_count <= 4:
            if any(word in last_patient_msg for word in ["hot", "warm", "burning"]):
                return "Have you checked your temperature? Do you have a fever?"

        # Ask about progression
        if "progression" not in self.info_gathered and self.turn_count <= 5:
            return "Have your symptoms been getting better, worse, or staying about the same?"

        # Ask about medications
        if "medications" not in self.info_gathered and self.turn_count <= 5:
            return "Have you tried any over-the-counter medications or other remedies?"

        # Check if patient is very worried (common in conversations)
        if any(word in last_patient_msg for word in ["worried", "concern", "serious", "scared"]):
            return self._provide_reassurance_and_advice(conversation_turns)

        # If we have enough info, provide advice
        if len(self.info_gathered) >= 3 or self.turn_count >= 5:
            return self._provide_advice(conversation_turns)

        # Default - ask for clarification
        return "Can you tell me a bit more about how you're feeling?"

    def _provide_reassurance_and_advice(self, conversation_turns: List[Dict[str, str]]) -> str:
        """
        Provide reassurance and advice when patient is worried

        Args:
            conversation_turns: Conversation turns

        Returns:
            Reassurance message
        """
        all_text = " ".join([t.get("message", "") for t in conversation_turns]).lower()

        # Check severity
        if any(word in all_text for word in ["really bad", "terrible", "awful", "can't", "unable"]):
            return ("I understand you're concerned. Based on what you're telling me, I'd recommend "
                   "seeing a healthcare provider to get properly evaluated, especially if symptoms are severe.")

        # General viral illness pattern
        if any(word in all_text for word in ["throat", "cough", "congestion", "cold", "flu"]):
            return ("These symptoms sound like they could be from a common viral infection. "
                   "Most viral infections improve on their own within a week or two. "
                   "However, if symptoms worsen or you develop a high fever, it would be good to see a doctor.")

        return ("I understand your concern. Most common illnesses improve with rest and self-care. "
               "However, if you're worried or symptoms get worse, it's always best to check with a healthcare provider.")

    def _provide_advice(self, conversation_turns: List[Dict[str, str]]) -> str:
        """
        Provide medical advice based on gathered information

        Args:
            conversation_turns: Conversation turns

        Returns:
            Advice message
        """
        all_text = " ".join([t.get("message", "") for t in conversation_turns]).lower()

        # Build advice based on symptoms
        advice_parts = []

        # Check for viral infection symptoms
        viral_symptoms = ["throat", "cough", "congestion", "fatigue", "aches"]
        if any(symptom in all_text for symptom in viral_symptoms):
            advice_parts.append(
                "Based on your symptoms, this sounds like it could be a viral upper respiratory infection (like a cold or flu)."
            )

            # Self-care recommendations
            advice_parts.append(
                "\nHere's what I recommend:\n"
                "- Get plenty of rest and stay hydrated\n"
                "- Over-the-counter pain relievers can help with aches and fever\n"
                "- Throat lozenges or warm tea with honey can soothe a sore throat\n"
                "- Use a humidifier if you have congestion"
            )

            # When to see a doctor
            advice_parts.append(
                "\nYou should see a healthcare provider if:\n"
                "- Symptoms persist beyond 10 days\n"
                "- You develop a high fever (>103°F/39.4°C)\n"
                "- You have difficulty breathing\n"
                "- Symptoms get significantly worse"
            )
        else:
            # Generic advice
            advice_parts.append(
                "Based on what you've told me, I'd recommend monitoring your symptoms. "
                "Make sure to rest, stay hydrated, and consider over-the-counter remedies as appropriate."
            )

        # Check context for previous conversations
        if self.has_context():
            advice_parts.append(
                "\nSince we've talked before, let me know if anything has changed or if you have other concerns."
            )

        return " ".join(advice_parts)

    def format_messages_for_model(self, turns: List[Dict[str, str]]) -> str:
        """
        Format conversation for model input

        Args:
            turns: Conversation turns

        Returns:
            Formatted string for model
        """
        # Get system prompt
        system_prompt = self.get_system_prompt()

        # Format conversation
        formatted = f"System: {system_prompt}\n\n"

        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            message = turn.get("message", "")

            if speaker == "patient":
                formatted += f"Patient: {message}\n"
            elif speaker == "chatbot":
                formatted += f"Assistant: {message}\n"

        formatted += "Assistant: "

        return formatted

    def should_end_conversation(self, last_message: str, conversation_turns: List[Dict[str, str]]) -> bool:
        """
        Determine if conversation should end

        Args:
            last_message: Last message in conversation
            conversation_turns: All turns

        Returns:
            True if conversation should end
        """
        # Check if we've given advice
        if any(word in last_message.lower() for word in ["recommend", "suggest", "should see", "advice"]):
            return True

        # Check if patient is closing
        patient_close_phrases = ["thank", "thanks", "ok", "okay", "alright", "bye", "goodbye"]
        last_patient_messages = [t for t in conversation_turns if t.get("speaker") == "patient"]
        if last_patient_messages:
            last_patient = last_patient_messages[-1].get("message", "").lower()
            if any(phrase in last_patient for phrase in patient_close_phrases) and len(last_patient) < 50:
                return True

        # Check if we've reached max turns
        if len(conversation_turns) >= 10:
            return True

        return False
