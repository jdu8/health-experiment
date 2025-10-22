"""
Conversation Data Model

Defines the structure for conversation logs and turns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    turn_number: int
    speaker: str  # "patient" or "chatbot"
    message: str
    timestamp: str
    internal_state: Optional[Dict[str, Any]] = None  # For patient
    model_info: Optional[Dict[str, Any]] = None  # For chatbot

    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary"""
        return {
            'turn_number': self.turn_number,
            'speaker': self.speaker,
            'message': self.message,
            'timestamp': self.timestamp,
            'internal_state': self.internal_state,
            'model_info': self.model_info
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create turn from dictionary"""
        return cls(**data)


@dataclass
class BiologicalState:
    """Biological state snapshot at conversation start"""
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    energy_level: Optional[int] = None
    pain_level: Optional[int] = None
    respiratory_rate: Optional[int] = None
    blood_pressure: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiologicalState':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class PsychologicalState:
    """Psychological state snapshot at conversation start"""
    anxiety_level: int
    mood: str
    health_concern_level: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PsychologicalState':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class SymptomState:
    """Individual symptom state"""
    objective_severity: float
    reported_severity: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymptomState':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class PatientStateSnapshot:
    """Patient state at the start of conversation"""
    biological: BiologicalState
    psychological: PsychologicalState
    symptoms: Dict[str, SymptomState] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'biological': self.biological.to_dict(),
            'psychological': self.psychological.to_dict(),
            'symptoms': {k: v.to_dict() for k, v in self.symptoms.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatientStateSnapshot':
        """Create from dictionary"""
        biological = BiologicalState.from_dict(data['biological'])
        psychological = PsychologicalState.from_dict(data['psychological'])
        symptoms = {
            k: SymptomState.from_dict(v) for k, v in data.get('symptoms', {}).items()
        }
        return cls(biological=biological, psychological=psychological, symptoms=symptoms)


@dataclass
class ConversationSummary:
    """Summary of conversation outcome"""
    total_turns: int
    duration_seconds: float
    conversation_end_reason: str
    final_bot_recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSummary':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class AutomaticMetrics:
    """Automatically calculated metrics"""
    bot_questions_asked: int = 0
    redundant_questions: int = 0
    references_to_history: int = 0
    contextual_questions: int = 0
    key_info_gathered: List[str] = field(default_factory=list)
    key_info_missed: List[str] = field(default_factory=list)
    patient_volunteered_count: int = 0
    patient_only_when_asked_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutomaticMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class EvaluationMetrics:
    """Human or LLM evaluation metrics"""
    evaluated: bool = False
    evaluator_id: Optional[str] = None
    scores: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ConversationMetrics:
    """All conversation metrics"""
    automatic: AutomaticMetrics
    evaluation: EvaluationMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'automatic': self.automatic.to_dict(),
            'evaluation': self.evaluation.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMetrics':
        """Create from dictionary"""
        automatic = AutomaticMetrics.from_dict(data['automatic'])
        evaluation = EvaluationMetrics.from_dict(data['evaluation'])
        return cls(automatic=automatic, evaluation=evaluation)


@dataclass
class Conversation:
    """Complete conversation log"""
    conversation_id: str
    patient_id: str
    simulation_day: int
    disease_day: int
    real_timestamp: str
    condition: str  # "with_history" or "without_history"
    patient_state_at_start: PatientStateSnapshot
    turns: List[ConversationTurn] = field(default_factory=list)
    conversation_summary: Optional[ConversationSummary] = None
    metrics: Optional[ConversationMetrics] = None

    def validate(self) -> bool:
        """Validate conversation data"""
        if not self.conversation_id:
            raise ValueError("conversation_id is required")
        if not self.patient_id:
            raise ValueError("patient_id is required")
        if self.simulation_day < 0:
            raise ValueError(f"simulation_day must be >= 0, got {self.simulation_day}")
        if self.disease_day < 0:
            raise ValueError(f"disease_day must be >= 0, got {self.disease_day}")
        if self.condition not in ["with_history", "without_history"]:
            raise ValueError(f"Invalid condition: {self.condition}")
        return True

    def add_turn(self, speaker: str, message: str,
                 internal_state: Optional[Dict[str, Any]] = None,
                 model_info: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        """
        Add a turn to the conversation

        Args:
            speaker: "patient" or "chatbot"
            message: The message content
            internal_state: Patient's internal state (if speaker is patient)
            model_info: Model information (if speaker is chatbot)

        Returns:
            The created ConversationTurn
        """
        turn_number = len(self.turns) + 1
        timestamp = datetime.now().isoformat()

        turn = ConversationTurn(
            turn_number=turn_number,
            speaker=speaker,
            message=message,
            timestamp=timestamp,
            internal_state=internal_state,
            model_info=model_info
        )

        self.turns.append(turn)
        return turn

    def get_patient_messages(self) -> List[str]:
        """Get all patient messages"""
        return [turn.message for turn in self.turns if turn.speaker == "patient"]

    def get_chatbot_messages(self) -> List[str]:
        """Get all chatbot messages"""
        return [turn.message for turn in self.turns if turn.speaker == "chatbot"]

    def get_turn_count(self) -> int:
        """Get total number of turns"""
        return len(self.turns)

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary"""
        return {
            'conversation_id': self.conversation_id,
            'patient_id': self.patient_id,
            'simulation_day': self.simulation_day,
            'disease_day': self.disease_day,
            'real_timestamp': self.real_timestamp,
            'condition': self.condition,
            'patient_state_at_start': self.patient_state_at_start.to_dict(),
            'turns': [turn.to_dict() for turn in self.turns],
            'conversation_summary': self.conversation_summary.to_dict() if self.conversation_summary else None,
            'metrics': self.metrics.to_dict() if self.metrics else None
        }

    def to_json(self, filepath: str, indent: int = 2):
        """
        Save conversation to JSON file

        Args:
            filepath: Path to save JSON file
            indent: JSON indentation level
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary"""
        patient_state = PatientStateSnapshot.from_dict(data['patient_state_at_start'])
        turns = [ConversationTurn.from_dict(t) for t in data['turns']]
        conversation_summary = (
            ConversationSummary.from_dict(data['conversation_summary'])
            if data.get('conversation_summary') else None
        )
        metrics = (
            ConversationMetrics.from_dict(data['metrics'])
            if data.get('metrics') else None
        )

        return cls(
            conversation_id=data['conversation_id'],
            patient_id=data['patient_id'],
            simulation_day=data['simulation_day'],
            disease_day=data['disease_day'],
            real_timestamp=data['real_timestamp'],
            condition=data['condition'],
            patient_state_at_start=patient_state,
            turns=turns,
            conversation_summary=conversation_summary,
            metrics=metrics
        )

    @classmethod
    def from_json(cls, filepath: str) -> 'Conversation':
        """
        Load conversation from JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            Conversation object
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Conversation file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)
