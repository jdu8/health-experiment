"""
Disease Model Data Model

Defines the structure for disease progression models loaded from YAML files.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import yaml
from pathlib import Path


@dataclass
class Symptom:
    """Individual symptom configuration"""
    severity_range: Tuple[float, float]  # [min, max]
    trajectory: str  # flat/increasing/decreasing/peak_then_decline/stable
    progression_rate: float = 0.0
    peak_day: Optional[int] = None
    subtype: Optional[str] = None
    description_templates: List[str] = field(default_factory=list)
    patient_notices: bool = True
    objective_measure: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """Validate symptom configuration"""
        if len(self.severity_range) != 2:
            raise ValueError(f"severity_range must have 2 values, got {len(self.severity_range)}")
        if self.severity_range[0] < 0 or self.severity_range[1] > 10:
            raise ValueError(f"severity_range values must be 0-10, got {self.severity_range}")
        if self.trajectory not in ["flat", "increasing", "decreasing", "peak_then_decline", "stable"]:
            raise ValueError(f"Invalid trajectory: {self.trajectory}")
        return True


@dataclass
class Phase:
    """Disease phase (e.g., incubation, prodrome, acute, resolution)"""
    name: str
    day_range: Tuple[int, int]  # [start_day, end_day]
    description: str
    symptoms: Dict[str, Symptom] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate phase configuration"""
        if len(self.day_range) != 2:
            raise ValueError(f"day_range must have 2 values, got {len(self.day_range)}")
        if self.day_range[0] < 0 or self.day_range[1] < self.day_range[0]:
            raise ValueError(f"Invalid day_range: {self.day_range}")
        for symptom_name, symptom in self.symptoms.items():
            symptom.validate()
        return True

    def get_duration(self) -> int:
        """Get phase duration in days"""
        return self.day_range[1] - self.day_range[0] + 1

    def contains_day(self, day: int) -> bool:
        """Check if day falls within this phase"""
        return self.day_range[0] <= day <= self.day_range[1]


@dataclass
class Complication:
    """Disease complication"""
    complication_id: str
    name: str
    probability_if_complication: float  # 0-1
    trigger_day_range: Tuple[int, int]
    new_symptoms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    changes_to_existing: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate complication configuration"""
        if not 0 <= self.probability_if_complication <= 1:
            raise ValueError(f"probability_if_complication must be 0-1, got {self.probability_if_complication}")
        if len(self.trigger_day_range) != 2:
            raise ValueError(f"trigger_day_range must have 2 values, got {len(self.trigger_day_range)}")
        return True


@dataclass
class RiskModifier:
    """Risk factor modifier for complications"""
    factor: str
    probability_multiplier: float

    def validate(self) -> bool:
        """Validate risk modifier"""
        if self.probability_multiplier < 0:
            raise ValueError(f"probability_multiplier must be >= 0, got {self.probability_multiplier}")
        return True


@dataclass
class Complications:
    """Complications configuration"""
    base_probability: float  # 0-1
    risk_modifiers: List[RiskModifier] = field(default_factory=list)
    types: List[Complication] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate complications configuration"""
        if not 0 <= self.base_probability <= 1:
            raise ValueError(f"base_probability must be 0-1, got {self.base_probability}")
        for modifier in self.risk_modifiers:
            modifier.validate()
        for complication in self.types:
            complication.validate()
        return True


@dataclass
class NaturalHistory:
    """Natural history of disease"""
    typical_duration: Tuple[int, int]  # [min_days, max_days]
    phases: List[Phase] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate natural history"""
        if len(self.typical_duration) != 2:
            raise ValueError(f"typical_duration must have 2 values, got {len(self.typical_duration)}")
        if self.typical_duration[0] < 0 or self.typical_duration[1] < self.typical_duration[0]:
            raise ValueError(f"Invalid typical_duration: {self.typical_duration}")
        for phase in self.phases:
            phase.validate()
        return True

    def get_phase_for_day(self, day: int) -> Optional[Phase]:
        """Get the phase for a given day"""
        for phase in self.phases:
            if phase.contains_day(day):
                return phase
        return None


@dataclass
class DiseaseModel:
    """Complete disease progression model"""
    disease_id: str
    name: str
    common_name: str
    description: str
    natural_history: NaturalHistory
    complications: Optional[Complications] = None
    patient_modifiers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate entire disease model"""
        if not self.disease_id:
            raise ValueError("disease_id is required")
        if not self.name:
            raise ValueError("name is required")

        # Validate sub-components
        self.natural_history.validate()
        if self.complications:
            self.complications.validate()

        return True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DiseaseModel':
        """
        Load disease model from YAML file

        Args:
            yaml_path: Path to disease model YAML file

        Returns:
            DiseaseModel object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Disease model not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse phases
        phases = []
        for phase_data in data['natural_history']['phases']:
            # Parse symptoms for this phase
            symptoms = {}
            for symptom_name, symptom_data in phase_data.get('symptoms', {}).items():
                # Handle severity_range as tuple
                severity_range = tuple(symptom_data['severity_range'])
                symptom_data_copy = symptom_data.copy()
                symptom_data_copy['severity_range'] = severity_range
                symptoms[symptom_name] = Symptom(**symptom_data_copy)

            # Create phase with tuple for day_range
            phase = Phase(
                name=phase_data['name'],
                day_range=tuple(phase_data['day_range']),
                description=phase_data['description'],
                symptoms=symptoms
            )
            phases.append(phase)

        # Parse natural history
        natural_history = NaturalHistory(
            typical_duration=tuple(data['natural_history']['typical_duration']),
            phases=phases
        )

        # Parse complications if present
        complications = None
        if 'complications' in data:
            comp_data = data['complications']

            # Parse risk modifiers
            risk_modifiers = [
                RiskModifier(**rm) for rm in comp_data.get('risk_modifiers', [])
            ]

            # Parse complication types
            complication_types = []
            for comp in comp_data.get('types', []):
                complication = Complication(
                    complication_id=comp['complication_id'],
                    name=comp['name'],
                    probability_if_complication=comp['probability_if_complication'],
                    trigger_day_range=tuple(comp['trigger_day_range']),
                    new_symptoms=comp.get('new_symptoms', {}),
                    changes_to_existing=comp.get('changes_to_existing', {})
                )
                complication_types.append(complication)

            complications = Complications(
                base_probability=comp_data['base_probability'],
                risk_modifiers=risk_modifiers,
                types=complication_types
            )

        # Create disease model
        model = cls(
            disease_id=data['disease_id'],
            name=data['name'],
            common_name=data['common_name'],
            description=data['description'],
            natural_history=natural_history,
            complications=complications,
            patient_modifiers=data.get('patient_modifiers', {})
        )

        # Validate
        model.validate()

        return model

    def get_all_phases(self) -> List[Phase]:
        """Get all disease phases"""
        return self.natural_history.phases

    def get_phase_for_day(self, day: int) -> Optional[Phase]:
        """Get the phase for a given disease day"""
        return self.natural_history.get_phase_for_day(day)

    def get_duration_range(self) -> Tuple[int, int]:
        """Get typical duration range"""
        return self.natural_history.typical_duration

    def has_complications(self) -> bool:
        """Check if disease has complication models"""
        return self.complications is not None

    def get_description(self) -> str:
        """Get human-readable description"""
        return (
            f"{self.name} ({self.common_name})\n"
            f"ID: {self.disease_id}\n"
            f"Description: {self.description}\n"
            f"Typical Duration: {self.natural_history.typical_duration[0]}-"
            f"{self.natural_history.typical_duration[1]} days\n"
            f"Phases: {len(self.natural_history.phases)}\n"
            f"Complications: {'Yes' if self.has_complications() else 'No'}"
        )
