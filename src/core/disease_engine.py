"""
Disease Progression Engine

Calculates symptoms for each day based on disease model and patient factors.
"""

import random
import logging
from typing import Dict, Any, Optional, List, Tuple
from src.models.disease_model import DiseaseModel, Phase, Symptom
from src.models.patient_profile import PatientProfile

logger = logging.getLogger(__name__)


class DiseaseProgressionEngine:
    """
    Engine for calculating disease progression and symptoms over time
    """

    def __init__(self, disease_model: DiseaseModel, patient_profile: PatientProfile, random_seed: Optional[int] = None):
        """
        Initialize disease progression engine

        Args:
            disease_model: Disease model loaded from YAML
            patient_profile: Patient profile for applying modifiers
            random_seed: Optional seed for reproducibility
        """
        self.disease_model = disease_model
        self.patient_profile = patient_profile
        self.complications_triggered = []
        self.complication_day = None

        if random_seed is not None:
            random.seed(random_seed)

        logger.info(f"Initialized DiseaseProgressionEngine for {disease_model.disease_id} and {patient_profile.patient_id}")

    def get_symptoms_for_day(self, day: int) -> Dict[str, Dict[str, Any]]:
        """
        Get all symptoms and severities for a specific day

        Args:
            day: Disease day (1-indexed)

        Returns:
            Dictionary of symptoms with severity and descriptions
            {
                "symptom_name": {
                    "objective_severity": 5.5,
                    "description": "throat feels raw",
                    "patient_notices": True,
                    "objective_value": 38.5  # Optional, e.g., temperature
                }
            }
        """
        logger.debug(f"Calculating symptoms for day {day}")

        # Get current phase
        phase = self._get_current_phase(day)
        if not phase:
            logger.warning(f"No phase found for day {day}")
            return {}

        logger.debug(f"Day {day} is in phase: {phase.name}")

        # Calculate base symptoms for this phase
        symptoms = {}
        for symptom_name, symptom_config in phase.symptoms.items():
            severity = self._calculate_symptom_severity(symptom_config, day, phase)

            # Add some random variation
            severity = self._add_random_variation(severity)

            # Clamp to valid range
            severity = max(0.0, min(10.0, severity))

            # Get description
            description = self._get_symptom_description(symptom_config, severity)

            # Build symptom data
            symptom_data = {
                "objective_severity": round(severity, 1),
                "description": description,
                "patient_notices": symptom_config.patient_notices
            }

            # Add objective measure if available (e.g., temperature)
            if symptom_config.objective_measure:
                obj_value = self._calculate_objective_measure(
                    symptom_config.objective_measure,
                    severity
                )
                symptom_data["objective_value"] = obj_value

            symptoms[symptom_name] = symptom_data

        # Apply patient modifiers
        symptoms = self._apply_patient_modifiers(symptoms)

        # Check and apply complications
        if self.disease_model.has_complications():
            self._check_and_apply_complication(symptoms, day)

        # Filter out symptoms with very low severity
        symptoms = {k: v for k, v in symptoms.items() if v["objective_severity"] > 0.5}

        logger.debug(f"Day {day}: {len(symptoms)} symptoms calculated")
        return symptoms

    def _get_current_phase(self, day: int) -> Optional[Phase]:
        """
        Determine which disease phase corresponds to this day

        Args:
            day: Disease day

        Returns:
            Phase object or None if no phase matches
        """
        return self.disease_model.get_phase_for_day(day)

    def _calculate_symptom_severity(self, symptom_config: Symptom, day: int, phase: Phase) -> float:
        """
        Calculate severity based on trajectory

        Handles different trajectory types:
        - flat: constant severity
        - increasing: linear increase
        - decreasing: linear decrease
        - peak_then_decline: bell curve
        - stable: random within range

        Args:
            symptom_config: Symptom configuration
            day: Current disease day
            phase: Current phase

        Returns:
            Calculated severity (0-10)
        """
        trajectory = symptom_config.trajectory
        severity_range = symptom_config.severity_range
        min_severity, max_severity = severity_range

        # Calculate days into current phase
        phase_start = phase.day_range[0]
        days_into_phase = day - phase_start

        if trajectory == "flat":
            # Constant severity
            severity = (min_severity + max_severity) / 2

        elif trajectory == "increasing":
            # Linear increase
            phase_duration = phase.get_duration()
            progress = days_into_phase / max(1, phase_duration)
            severity = min_severity + (max_severity - min_severity) * progress
            # Apply progression rate if specified
            if symptom_config.progression_rate:
                severity = min_severity + (days_into_phase * symptom_config.progression_rate)

        elif trajectory == "decreasing":
            # Linear decrease
            phase_duration = phase.get_duration()
            progress = days_into_phase / max(1, phase_duration)
            severity = max_severity - (max_severity - min_severity) * progress
            # Apply progression rate if specified
            if symptom_config.progression_rate:
                severity = max_severity + (days_into_phase * symptom_config.progression_rate)

        elif trajectory == "peak_then_decline":
            # Bell curve with peak at specified day
            peak_day = symptom_config.peak_day or (phase_start + phase.get_duration() // 2)

            if day < peak_day:
                # Increasing to peak
                progress = (day - phase_start) / max(1, (peak_day - phase_start))
                severity = min_severity + (max_severity - min_severity) * progress
            else:
                # Decreasing from peak
                phase_end = phase.day_range[1]
                progress = (day - peak_day) / max(1, (phase_end - peak_day))
                severity = max_severity - (max_severity - min_severity) * progress

        elif trajectory == "stable":
            # Random within range
            severity = random.uniform(min_severity, max_severity)

        else:
            logger.warning(f"Unknown trajectory: {trajectory}, using midpoint")
            severity = (min_severity + max_severity) / 2

        return severity

    def _add_random_variation(self, severity: float, variation: float = 0.5) -> float:
        """
        Add random variation to severity

        Args:
            severity: Base severity
            variation: Amount of variation to add

        Returns:
            Severity with random variation
        """
        return severity + random.uniform(-variation, variation)

    def _get_symptom_description(self, symptom_config: Symptom, severity: float) -> str:
        """
        Get natural language description for symptom

        Args:
            symptom_config: Symptom configuration
            severity: Current severity

        Returns:
            Description string
        """
        templates = symptom_config.description_templates
        if not templates:
            return "experiencing symptoms"

        # Choose description based on severity if multiple templates
        if len(templates) == 1:
            return templates[0]

        # Map severity to template index
        index = int((severity / 10.0) * (len(templates) - 1))
        index = max(0, min(len(templates) - 1, index))

        return templates[index]

    def _calculate_objective_measure(self, measure_config: Dict[str, Any], severity: float) -> Any:
        """
        Calculate objective measurement (e.g., temperature from fever)

        Args:
            measure_config: Measurement configuration
            severity: Symptom severity

        Returns:
            Calculated measurement value
        """
        measure_type = measure_config.get("type")
        formula = measure_config.get("formula")

        if not formula:
            return None

        try:
            # Evaluate formula with severity as variable
            # Formula example: "37.0 + (severity * 0.3)"
            result = eval(formula, {"__builtins__": {}}, {"severity": severity})
            return round(result, 1)
        except Exception as e:
            logger.error(f"Error evaluating formula '{formula}': {e}")
            return None

    def _apply_patient_modifiers(self, symptoms: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Modify symptoms based on patient factors

        Examples:
        - Poor sleep: +1 to all severities
        - High stress: +0.5 to all severities
        - High caffeine: +1 to headache
        - Immune compromised: +2 to all severities

        Args:
            symptoms: Base symptoms

        Returns:
            Modified symptoms
        """
        modifiers = self.disease_model.patient_modifiers

        # Check patient lifestyle factors
        lifestyle = self.patient_profile.biological.lifestyle

        if lifestyle:
            # Poor sleep modifier
            if lifestyle.sleep_quality == "poor" and "poor_sleep" in modifiers:
                logger.debug("Applying poor sleep modifier")
                symptoms = self._apply_severity_modifier(symptoms, 1.0)

            # High stress modifier
            if lifestyle.stress_level >= 7 and "high_stress" in modifiers:
                logger.debug("Applying high stress modifier")
                symptoms = self._apply_severity_modifier(symptoms, 0.5)

            # High caffeine modifier (affects headaches)
            if lifestyle.caffeine_intake == "high" and "high_caffeine" in modifiers:
                if "headache" in symptoms:
                    logger.debug("Applying high caffeine modifier to headache")
                    symptoms["headache"]["objective_severity"] = min(10.0,
                        symptoms["headache"]["objective_severity"] + 1.0)

        # Immune status modifier
        if self.patient_profile.biological.immune_status == "compromised" and "poor_immune" in modifiers:
            logger.debug("Applying immune compromised modifier")
            symptoms = self._apply_severity_modifier(symptoms, 2.0)

        return symptoms

    def _apply_severity_modifier(self, symptoms: Dict[str, Dict[str, Any]], modifier: float) -> Dict[str, Dict[str, Any]]:
        """
        Apply a modifier to all symptom severities

        Args:
            symptoms: Symptoms dictionary
            modifier: Amount to add to severity

        Returns:
            Modified symptoms
        """
        for symptom_name, symptom_data in symptoms.items():
            old_severity = symptom_data["objective_severity"]
            new_severity = min(10.0, old_severity + modifier)
            symptom_data["objective_severity"] = round(new_severity, 1)

        return symptoms

    def _check_and_apply_complication(self, symptoms: Dict[str, Dict[str, Any]], day: int):
        """
        Probabilistically check if complication occurs and apply it

        Args:
            symptoms: Current symptoms (modified in place)
            day: Current disease day
        """
        if not self.disease_model.complications:
            return

        # Skip if complication already triggered
        if self.complications_triggered:
            return

        # Calculate complication probability
        base_prob = self.disease_model.complications.base_probability
        total_prob = base_prob

        # Apply risk modifiers
        for risk_modifier in self.disease_model.complications.risk_modifiers:
            if self._patient_has_risk_factor(risk_modifier.factor):
                total_prob *= risk_modifier.probability_multiplier
                logger.debug(f"Risk factor '{risk_modifier.factor}' increases complication probability")

        # Cap probability at 1.0
        total_prob = min(1.0, total_prob)

        # Check if complication triggers
        if random.random() < total_prob:
            # Select a complication type
            for complication in self.disease_model.complications.types:
                # Check if we're in the trigger day range
                trigger_start, trigger_end = complication.trigger_day_range
                if trigger_start <= day <= trigger_end:
                    # Check if this specific complication occurs
                    if random.random() < complication.probability_if_complication:
                        logger.info(f"Complication triggered: {complication.name} on day {day}")
                        self._apply_complication_effects(symptoms, complication)
                        self.complications_triggered.append(complication.complication_id)
                        self.complication_day = day
                        break

    def _patient_has_risk_factor(self, factor: str) -> bool:
        """
        Check if patient has a specific risk factor

        Args:
            factor: Risk factor name

        Returns:
            True if patient has this risk factor
        """
        lifestyle = self.patient_profile.biological.lifestyle

        if factor == "poor_sleep" and lifestyle and lifestyle.sleep_quality == "poor":
            return True
        if factor == "high_stress" and lifestyle and lifestyle.stress_level >= 7:
            return True
        if factor == "immune_compromised" and self.patient_profile.biological.immune_status == "compromised":
            return True
        if factor == "age_over_65" and self.patient_profile.demographics.age > 65:
            return True

        # Check for chronic conditions
        if factor == "chronic_lung_disease":
            for condition in self.patient_profile.biological.chronic_conditions:
                if "lung" in condition.type.lower() or "copd" in condition.type.lower() or "asthma" in condition.type.lower():
                    return True

        return False

    def _apply_complication_effects(self, symptoms: Dict[str, Dict[str, Any]], complication):
        """
        Apply complication effects to symptoms

        Args:
            symptoms: Current symptoms (modified in place)
            complication: Complication object
        """
        # Add new symptoms
        for symptom_name, symptom_data in complication.new_symptoms.items():
            symptoms[symptom_name] = {
                "objective_severity": symptom_data["severity"],
                "description": symptom_data["description"],
                "patient_notices": True
            }
            logger.debug(f"Added new symptom from complication: {symptom_name}")

        # Modify existing symptoms
        for symptom_name, changes in complication.changes_to_existing.items():
            if symptom_name in symptoms:
                modifier = changes.get("severity_modifier", 0)
                symptoms[symptom_name]["objective_severity"] = min(10.0,
                    symptoms[symptom_name]["objective_severity"] + modifier)
                logger.debug(f"Modified existing symptom: {symptom_name} +{modifier}")

    def get_expected_resolution_day(self) -> int:
        """
        Get expected resolution day based on disease duration and patient factors

        Returns:
            Expected day when disease resolves
        """
        min_duration, max_duration = self.disease_model.get_duration_range()

        # Start with average duration
        expected_duration = (min_duration + max_duration) / 2

        # Apply patient modifiers
        lifestyle = self.patient_profile.biological.lifestyle
        if lifestyle:
            if lifestyle.sleep_quality == "poor":
                expected_duration += 2
            if lifestyle.stress_level >= 7:
                expected_duration += 1

        if self.patient_profile.biological.immune_status == "compromised":
            expected_duration += 3

        return int(expected_duration)

    def has_complications(self) -> bool:
        """Check if any complications have been triggered"""
        return len(self.complications_triggered) > 0

    def get_complications(self) -> List[str]:
        """Get list of triggered complication IDs"""
        return self.complications_triggered.copy()
