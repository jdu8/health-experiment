"""
Patient State Manager

Manages and updates patient state throughout simulation.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from src.models.patient_profile import PatientProfile
from src.models.patient_state import (
    PatientState, BiologicalState, PsychologicalState,
    DiseaseState, CurrentSymptom, ConversationHistorySummary
)
from src.models.conversation import Conversation
from src.utils.file_utils import create_state_filepath

logger = logging.getLogger(__name__)


class PatientStateManager:
    """
    Manages patient state throughout simulation
    """

    def __init__(self, patient_profile: PatientProfile, disease_id: str, condition: Optional[str] = None):
        """
        Initialize state manager for patient

        Args:
            patient_profile: Patient profile
            disease_id: Disease identifier
            condition: Optional condition ('with_history' or 'without_history')
        """
        self.patient_profile = patient_profile
        self.disease_id = disease_id
        self.condition = condition
        self.state: Optional[PatientState] = None

        logger.info(f"Initialized PatientStateManager for {patient_profile.patient_id}")

    def initialize_state(self, disease_start_day: int = 1) -> PatientState:
        """
        Create initial state at day 0 (before disease starts)

        Args:
            disease_start_day: Day when disease starts

        Returns:
            Initialized patient state
        """
        logger.info(f"Initializing state for {self.patient_profile.patient_id}")

        # Initialize biological state with baseline values
        biological = self._create_baseline_biological_state()

        # Initialize psychological state from profile
        psychological = PsychologicalState(
            anxiety_level=self.patient_profile.psychological.baseline_anxiety,
            mood=self.patient_profile.psychological.current_mood,
            health_concern_level=self.patient_profile.psychological.baseline_anxiety
        )

        # Initialize disease state (pre-disease)
        disease_state = DiseaseState(
            active_disease=self.disease_id,
            disease_day=0,
            current_phase="pre_disease",
            trajectory="stable",
            complications_occurred=False,
            expected_resolution_day=14  # Will be updated once disease engine calculates
        )

        # Create patient state
        self.state = PatientState(
            patient_id=self.patient_profile.patient_id,
            current_day=0,
            last_updated=datetime.now().isoformat(),
            biological=biological,
            psychological=psychological,
            disease_state=disease_state,
            current_symptoms={},
            conversation_history_summary=[]
        )

        self.state.validate()
        logger.debug(f"Initial state created: {self.state.get_summary()}")

        return self.state

    def _create_baseline_biological_state(self) -> BiologicalState:
        """
        Create baseline biological state from patient profile

        Returns:
            BiologicalState with baseline values
        """
        # Calculate baseline vitals
        baseline_temp = 37.0
        baseline_hr = 70
        baseline_rr = 16

        # Adjust for patient factors
        lifestyle = self.patient_profile.biological.lifestyle
        if lifestyle:
            # Stress increases heart rate
            if lifestyle.stress_level >= 7:
                baseline_hr += 5

            # Caffeine increases heart rate
            if lifestyle.caffeine_intake in ["high", "moderate"]:
                baseline_hr += 3

        return BiologicalState(
            temperature=baseline_temp,
            heart_rate=baseline_hr,
            respiratory_rate=baseline_rr,
            blood_pressure=self.patient_profile.biological.blood_pressure,
            energy_level=8,  # Baseline good energy
            pain_level=0,  # No pain at baseline
            immune_markers={"status": self.patient_profile.biological.immune_status}
        )

    def update_state(self, day: int, disease_symptoms: Dict[str, Dict[str, Any]],
                     current_phase: str, expected_resolution_day: int) -> PatientState:
        """
        Update state for new day with disease progression

        Args:
            day: Current simulation day
            disease_symptoms: Symptoms from disease engine
            current_phase: Current disease phase name
            expected_resolution_day: Expected resolution day

        Returns:
            Updated patient state
        """
        if not self.state:
            raise RuntimeError("State not initialized. Call initialize_state() first.")

        logger.debug(f"Updating state for day {day}")

        # Update day counter
        self.state.current_day = day
        self.state.disease_state.disease_day = day
        self.state.last_updated = datetime.now().isoformat()

        # Update disease state
        self.state.disease_state.current_phase = current_phase
        self.state.disease_state.expected_resolution_day = expected_resolution_day

        # Determine trajectory
        self.state.disease_state.trajectory = self._determine_trajectory(day, disease_symptoms)

        # Update symptoms
        self._update_symptoms(disease_symptoms, day)

        # Update biological state based on symptoms
        self._update_biological_state(disease_symptoms)

        # Update psychological state based on symptoms and trajectory
        self._update_psychological_state()

        logger.debug(f"State updated: {self.state.get_summary()}")

        return self.state

    def _update_symptoms(self, disease_symptoms: Dict[str, Dict[str, Any]], day: int):
        """
        Update current symptoms from disease engine

        Args:
            disease_symptoms: Symptoms from disease engine
            day: Current day
        """
        # Clear old symptoms that are no longer present
        current_symptom_names = set(self.state.current_symptoms.keys())
        new_symptom_names = set(disease_symptoms.keys())

        # Remove symptoms that are no longer present
        for symptom_name in current_symptom_names - new_symptom_names:
            self.state.remove_symptom(symptom_name)
            logger.debug(f"Removed resolved symptom: {symptom_name}")

        # Add or update symptoms
        for symptom_name, symptom_data in disease_symptoms.items():
            objective_severity = symptom_data["objective_severity"]

            # Modulate reported severity based on patient anxiety
            reported_severity = self._modulate_symptom_severity(objective_severity)

            # Check if this is a new symptom
            first_appeared_day = day
            if self.state.has_symptom(symptom_name):
                existing = self.state.get_symptom(symptom_name)
                first_appeared_day = existing.first_appeared_day

            # Create symptom object
            symptom = CurrentSymptom(
                objective_severity=objective_severity,
                reported_severity=reported_severity,
                first_appeared_day=first_appeared_day,
                description=symptom_data["description"],
                patient_notices=symptom_data["patient_notices"],
                objective_value=symptom_data.get("objective_value")
            )

            self.state.add_symptom(symptom_name, symptom)

    def _modulate_symptom_severity(self, objective_severity: float) -> float:
        """
        Apply psychological modifiers to symptom severity

        High anxiety patients report symptoms as more severe

        Args:
            objective_severity: True severity (0-10)

        Returns:
            Reported severity after anxiety/personality modulation
        """
        health_anxiety = self.patient_profile.psychological.health_anxiety
        pain_catastrophizing = self.patient_profile.psychological.pain_catastrophizing

        # Calculate multiplier based on anxiety and catastrophizing
        # High anxiety/catastrophizing increases reported severity
        anxiety_multiplier = 1.0 + ((health_anxiety - 5) * 0.1)  # ±50% based on anxiety
        catastrophize_multiplier = 1.0 + ((pain_catastrophizing - 5) * 0.05)  # ±25% based on catastrophizing

        total_multiplier = anxiety_multiplier * catastrophize_multiplier

        reported = objective_severity * total_multiplier

        # Clamp to valid range
        reported = max(0.0, min(10.0, reported))

        return round(reported, 1)

    def _update_biological_state(self, disease_symptoms: Dict[str, Dict[str, Any]]):
        """
        Update biological measurements based on symptoms

        Args:
            disease_symptoms: Current symptoms
        """
        bio = self.state.biological

        # Temperature from fever
        if "fever" in disease_symptoms:
            obj_value = disease_symptoms["fever"].get("objective_value")
            if obj_value:
                bio.temperature = obj_value
            else:
                # Calculate from severity
                fever_severity = disease_symptoms["fever"]["objective_severity"]
                bio.temperature = 37.0 + (fever_severity * 0.3)
        else:
            # Return to baseline
            bio.temperature = 37.0

        # Heart rate from fever and other factors
        baseline_hr = 70
        if "fever" in disease_symptoms:
            fever_severity = disease_symptoms["fever"]["objective_severity"]
            bio.heart_rate = int(baseline_hr + (fever_severity * 4))
        else:
            bio.heart_rate = baseline_hr

        # Add stress/caffeine modifiers
        lifestyle = self.patient_profile.biological.lifestyle
        if lifestyle:
            if lifestyle.stress_level >= 7:
                bio.heart_rate += 5
            if lifestyle.caffeine_intake in ["high", "moderate"]:
                bio.heart_rate += 3

        # Respiratory rate from respiratory symptoms
        baseline_rr = 16
        if "congestion" in disease_symptoms or "cough" in disease_symptoms:
            avg_severity = (
                disease_symptoms.get("congestion", {}).get("objective_severity", 0) +
                disease_symptoms.get("cough", {}).get("objective_severity", 0)
            ) / 2
            bio.respiratory_rate = int(baseline_rr + (avg_severity * 0.5))
        else:
            bio.respiratory_rate = baseline_rr

        # Energy level from fatigue
        if "fatigue" in disease_symptoms:
            fatigue_severity = disease_symptoms["fatigue"]["objective_severity"]
            bio.energy_level = max(1, int(10 - fatigue_severity))
        else:
            bio.energy_level = 8

        # Pain level from various symptoms
        pain_symptoms = ["sore_throat", "headache", "body_aches"]
        pain_severities = [
            disease_symptoms.get(s, {}).get("objective_severity", 0)
            for s in pain_symptoms
        ]
        bio.pain_level = int(max(pain_severities)) if pain_severities else 0

        logger.debug(f"Biological state updated - Temp: {bio.temperature}°C, HR: {bio.heart_rate}")

    def _update_psychological_state(self):
        """
        Update anxiety and mood based on symptoms and trajectory
        """
        psych = self.state.psychological
        disease = self.state.disease_state

        # Base anxiety from profile
        baseline_anxiety = self.patient_profile.psychological.baseline_anxiety
        health_anxiety = self.patient_profile.psychological.health_anxiety

        # Calculate symptom severity impact
        total_symptom_count = len(self.state.current_symptoms)
        avg_symptom_severity = 0
        if total_symptom_count > 0:
            avg_symptom_severity = sum(
                s.objective_severity for s in self.state.current_symptoms.values()
            ) / total_symptom_count

        # Anxiety increases with symptom severity and count
        anxiety_modifier = 0
        if avg_symptom_severity > 5:
            anxiety_modifier += 1
        if total_symptom_count >= 3:
            anxiety_modifier += 1

        # Trajectory affects anxiety
        if disease.trajectory == "worsening":
            anxiety_modifier += 2
        elif disease.trajectory == "improving":
            anxiety_modifier -= 1

        # Complications increase anxiety
        if disease.complications_occurred:
            anxiety_modifier += 2

        # Calculate final anxiety
        psych.anxiety_level = baseline_anxiety + anxiety_modifier
        psych.anxiety_level = max(1, min(10, psych.anxiety_level))

        # Health concern level
        psych.health_concern_level = health_anxiety + anxiety_modifier
        psych.health_concern_level = max(1, min(10, psych.health_concern_level))

        # Update mood
        if avg_symptom_severity > 7:
            psych.mood = "miserable"
        elif avg_symptom_severity > 5:
            psych.mood = "unwell"
        elif disease.trajectory == "worsening":
            psych.mood = "worried"
        elif disease.trajectory == "improving":
            psych.mood = "hopeful"
        else:
            psych.mood = self.patient_profile.psychological.current_mood

        logger.debug(f"Psychological state updated - Anxiety: {psych.anxiety_level}, Mood: {psych.mood}")

    def _determine_trajectory(self, day: int, current_symptoms: Dict[str, Dict[str, Any]]) -> str:
        """
        Determine if patient is improving, stable, or worsening

        Args:
            day: Current day
            current_symptoms: Current symptoms

        Returns:
            "improving", "stable", or "worsening"
        """
        # Compare with previous symptoms if available
        if day == 1:
            return "stable"

        # Calculate current total severity
        current_severity = sum(s["objective_severity"] for s in current_symptoms.values())

        # Compare with stored symptoms (we'd need to track this in state history)
        # For now, use heuristic based on disease phase and day
        disease_state = self.state.disease_state

        if disease_state.current_phase in ["resolution"]:
            return "improving"
        elif disease_state.current_phase in ["acute"]:
            # Peak of disease - could be worsening or stable
            if day > 6:
                return "stable"
            else:
                return "worsening"
        elif disease_state.current_phase in ["prodrome"]:
            return "worsening"
        else:
            return "stable"

    def add_conversation_to_history(self, conversation: Conversation, topic: str = "", key_info: List[str] = None):
        """
        Add conversation to history summary

        Args:
            conversation: Completed conversation
            topic: Brief topic description
            key_info: List of key information points
        """
        if not self.state:
            raise RuntimeError("State not initialized")

        conv_type = "medical" if "symptom" in topic.lower() or "sick" in topic.lower() else "casual"

        self.state.add_conversation_to_history(
            day=conversation.simulation_day,
            conv_type=conv_type,
            topic=topic,
            key_info=key_info or []
        )

        logger.debug(f"Added conversation to history: Day {conversation.simulation_day}, Type: {conv_type}")

    def get_conversation_history(self, condition: str) -> List[ConversationHistorySummary]:
        """
        Get conversation history for chatbot context

        Args:
            condition: 'with_history' returns all, 'without_history' returns empty

        Returns:
            List of conversation summaries
        """
        if not self.state:
            return []

        if condition == "with_history":
            return self.state.get_conversation_history()
        else:
            return []

    def get_current_state(self) -> PatientState:
        """
        Get current patient state

        Returns:
            Current PatientState
        """
        if not self.state:
            raise RuntimeError("State not initialized")
        return self.state

    def save_state(self, filepath: Optional[str] = None, condition: Optional[str] = None):
        """
        Persist state to JSON file

        Args:
            filepath: Optional custom filepath, otherwise uses default
            condition: Optional condition for filename (overrides instance condition)
        """
        if not self.state:
            raise RuntimeError("State not initialized")

        if filepath is None:
            # Use condition-specific filename
            cond = condition or self.condition
            if cond:
                filepath = f"data/state/{self.state.patient_id}_{cond}_state.json"
            else:
                filepath = create_state_filepath(self.state.patient_id)

        # Ensure directory exists
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        self.state.to_json(str(filepath))
        logger.info(f"State saved to {filepath}")

    def load_state(self, patient_id: Optional[str] = None, filepath: Optional[str] = None, condition: Optional[str] = None) -> PatientState:
        """
        Load state from JSON file

        Args:
            patient_id: Patient ID (uses profile if not provided)
            filepath: Optional custom filepath, otherwise uses default
            condition: Optional condition for filename (overrides instance condition)

        Returns:
            Loaded PatientState
        """
        if filepath is None:
            pid = patient_id or self.patient_profile.patient_id
            # Use condition-specific filename
            cond = condition or self.condition
            if cond:
                filepath = f"data/state/{pid}_{cond}_state.json"
            else:
                filepath = create_state_filepath(pid)

        self.state = PatientState.from_json(str(filepath))
        logger.info(f"State loaded from {filepath}")

        return self.state

    def mark_complication(self, complication_id: str):
        """
        Mark that a complication has occurred

        Args:
            complication_id: Complication identifier
        """
        if not self.state:
            raise RuntimeError("State not initialized")

        self.state.disease_state.complications_occurred = True
        if self.state.disease_state.complication_details is None:
            self.state.disease_state.complication_details = {}

        self.state.disease_state.complication_details[complication_id] = {
            "occurred_day": self.state.current_day
        }

        logger.warning(f"Complication marked: {complication_id} on day {self.state.current_day}")
