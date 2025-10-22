"""
Patient State Data Model

Defines the structure for tracking patient state during simulation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class CurrentSymptom:
    """Current symptom with severity and metadata"""
    objective_severity: float  # 0-10
    reported_severity: float  # 0-10 (may differ due to anxiety)
    first_appeared_day: int
    description: str
    patient_notices: bool
    objective_value: Optional[Any] = None  # e.g., temperature in celsius

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurrentSymptom':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class BiologicalState:
    """Current biological/physiological state"""
    temperature: Optional[float] = None  # Celsius
    heart_rate: Optional[int] = None  # BPM
    respiratory_rate: Optional[int] = None  # Breaths per minute
    blood_pressure: Optional[str] = None  # "systolic/diastolic"
    energy_level: Optional[int] = None  # 1-10
    pain_level: Optional[int] = None  # 0-10
    immune_markers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiologicalState':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class PsychologicalState:
    """Current psychological state"""
    anxiety_level: int  # 1-10
    mood: str
    health_concern_level: int  # 1-10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PsychologicalState':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class DiseaseState:
    """Current disease state"""
    active_disease: str
    disease_day: int
    current_phase: str
    trajectory: str  # improving/stable/worsening
    complications_occurred: bool
    complication_details: Optional[Dict[str, Any]] = None
    expected_resolution_day: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiseaseState':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ConversationHistorySummary:
    """Summary of a past conversation"""
    day: int
    type: str  # medical/casual/medical_followup
    topic: str
    key_info: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationHistorySummary':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class PatientState:
    """Complete patient state at a given point in time"""
    patient_id: str
    current_day: int
    last_updated: str
    biological: BiologicalState
    psychological: PsychologicalState
    disease_state: DiseaseState
    current_symptoms: Dict[str, CurrentSymptom] = field(default_factory=dict)
    conversation_history_summary: List[ConversationHistorySummary] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate patient state"""
        if not self.patient_id:
            raise ValueError("patient_id is required")
        if self.current_day < 0:
            raise ValueError(f"current_day must be >= 0, got {self.current_day}")
        return True

    def update_biological(self, **kwargs):
        """Update biological state with new values"""
        for key, value in kwargs.items():
            if hasattr(self.biological, key):
                setattr(self.biological, key, value)

    def update_psychological(self, **kwargs):
        """Update psychological state with new values"""
        for key, value in kwargs.items():
            if hasattr(self.psychological, key):
                setattr(self.psychological, key, value)

    def update_disease_state(self, **kwargs):
        """Update disease state with new values"""
        for key, value in kwargs.items():
            if hasattr(self.disease_state, key):
                setattr(self.disease_state, key, value)

    def add_symptom(self, symptom_name: str, symptom: CurrentSymptom):
        """Add or update a symptom"""
        self.current_symptoms[symptom_name] = symptom

    def remove_symptom(self, symptom_name: str):
        """Remove a symptom"""
        if symptom_name in self.current_symptoms:
            del self.current_symptoms[symptom_name]

    def get_symptom(self, symptom_name: str) -> Optional[CurrentSymptom]:
        """Get a specific symptom"""
        return self.current_symptoms.get(symptom_name)

    def has_symptom(self, symptom_name: str) -> bool:
        """Check if patient has a symptom"""
        return symptom_name in self.current_symptoms

    def get_all_symptoms(self) -> List[str]:
        """Get list of all current symptom names"""
        return list(self.current_symptoms.keys())

    def add_conversation_to_history(self, day: int, conv_type: str, topic: str, key_info: List[str]):
        """Add a conversation to the history summary"""
        summary = ConversationHistorySummary(
            day=day,
            type=conv_type,
            topic=topic,
            key_info=key_info
        )
        self.conversation_history_summary.append(summary)

    def get_conversation_history(self) -> List[ConversationHistorySummary]:
        """Get conversation history"""
        return self.conversation_history_summary

    def clear_conversation_history(self):
        """Clear conversation history (for without_history condition)"""
        self.conversation_history_summary = []

    def advance_day(self):
        """Advance to next day"""
        self.current_day += 1
        self.disease_state.disease_day += 1
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert patient state to dictionary"""
        return {
            'patient_id': self.patient_id,
            'current_day': self.current_day,
            'last_updated': self.last_updated,
            'biological': self.biological.to_dict(),
            'psychological': self.psychological.to_dict(),
            'disease_state': self.disease_state.to_dict(),
            'current_symptoms': {k: v.to_dict() for k, v in self.current_symptoms.items()},
            'conversation_history_summary': [h.to_dict() for h in self.conversation_history_summary]
        }

    def to_json(self, filepath: str, indent: int = 2):
        """
        Save patient state to JSON file

        Args:
            filepath: Path to save JSON file
            indent: JSON indentation level
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatientState':
        """Create patient state from dictionary"""
        biological = BiologicalState.from_dict(data['biological'])
        psychological = PsychologicalState.from_dict(data['psychological'])
        disease_state = DiseaseState.from_dict(data['disease_state'])
        current_symptoms = {
            k: CurrentSymptom.from_dict(v) for k, v in data.get('current_symptoms', {}).items()
        }
        conversation_history_summary = [
            ConversationHistorySummary.from_dict(h)
            for h in data.get('conversation_history_summary', [])
        ]

        return cls(
            patient_id=data['patient_id'],
            current_day=data['current_day'],
            last_updated=data['last_updated'],
            biological=biological,
            psychological=psychological,
            disease_state=disease_state,
            current_symptoms=current_symptoms,
            conversation_history_summary=conversation_history_summary
        )

    @classmethod
    def from_json(cls, filepath: str) -> 'PatientState':
        """
        Load patient state from JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            PatientState object
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Patient state file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def get_summary(self) -> str:
        """Get human-readable summary of patient state"""
        symptom_list = ", ".join(self.get_all_symptoms()) if self.current_symptoms else "None"
        return (
            f"Patient {self.patient_id} - Day {self.current_day}\n"
            f"Disease: {self.disease_state.active_disease} (Day {self.disease_state.disease_day}, "
            f"Phase: {self.disease_state.current_phase})\n"
            f"Trajectory: {self.disease_state.trajectory}\n"
            f"Symptoms: {symptom_list}\n"
            f"Anxiety: {self.psychological.anxiety_level}/10, "
            f"Mood: {self.psychological.mood}\n"
            f"Temperature: {self.biological.temperature}°C, "
            f"Energy: {self.biological.energy_level}/10"
        )
