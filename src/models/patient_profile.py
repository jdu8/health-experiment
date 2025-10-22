"""
Patient Profile Data Model

Defines the structure for patient profiles loaded from YAML files.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class Demographics:
    """Patient demographic information"""
    age: int
    gender: str
    occupation: str
    location: str

    def validate(self) -> bool:
        """Validate demographic data"""
        if self.age < 0 or self.age > 120:
            raise ValueError(f"Invalid age: {self.age}")
        if self.gender not in ["male", "female", "other"]:
            raise ValueError(f"Invalid gender: {self.gender}")
        if self.location not in ["urban", "suburban", "rural"]:
            raise ValueError(f"Invalid location: {self.location}")
        return True


@dataclass
class Communication:
    """Patient communication style"""
    formality: int  # 1-10
    verbosity: int  # 1-10
    vocabulary_style: str  # casual/professional/medical
    example_phrases: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate communication data"""
        if not 1 <= self.formality <= 10:
            raise ValueError(f"Formality must be 1-10, got {self.formality}")
        if not 1 <= self.verbosity <= 10:
            raise ValueError(f"Verbosity must be 1-10, got {self.verbosity}")
        if self.vocabulary_style not in ["casual", "professional", "medical"]:
            raise ValueError(f"Invalid vocabulary_style: {self.vocabulary_style}")
        return True


@dataclass
class Psychological:
    """Patient psychological profile"""
    baseline_anxiety: int  # 1-10
    health_anxiety: int  # 1-10
    pain_catastrophizing: int  # 1-10
    information_volunteering: str  # low/medium/high
    current_mood: str
    mood_triggers: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate psychological data"""
        if not 1 <= self.baseline_anxiety <= 10:
            raise ValueError(f"Baseline anxiety must be 1-10, got {self.baseline_anxiety}")
        if not 1 <= self.health_anxiety <= 10:
            raise ValueError(f"Health anxiety must be 1-10, got {self.health_anxiety}")
        if not 1 <= self.pain_catastrophizing <= 10:
            raise ValueError(f"Pain catastrophizing must be 1-10, got {self.pain_catastrophizing}")
        if self.information_volunteering not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid information_volunteering: {self.information_volunteering}")
        return True


@dataclass
class Lifestyle:
    """Patient lifestyle information"""
    sleep_hours: float
    sleep_quality: str  # poor/fair/good/excellent
    stress_level: int  # 1-10
    exercise_frequency: int  # times per week
    exercise_type: str
    diet_quality: str  # poor/moderate/good
    caffeine_intake: str  # none/low/moderate/high
    caffeine_amount: str
    alcohol: str  # none/light/moderate/heavy
    smoking: bool

    def validate(self) -> bool:
        """Validate lifestyle data"""
        if self.sleep_hours < 0 or self.sleep_hours > 24:
            raise ValueError(f"Invalid sleep_hours: {self.sleep_hours}")
        if self.sleep_quality not in ["poor", "fair", "good", "excellent"]:
            raise ValueError(f"Invalid sleep_quality: {self.sleep_quality}")
        if not 1 <= self.stress_level <= 10:
            raise ValueError(f"Stress level must be 1-10, got {self.stress_level}")
        return True


@dataclass
class RiskFactor:
    """Individual risk factor"""
    type: str
    details: str
    risk_multiplier: float


@dataclass
class ChronicCondition:
    """Chronic medical condition"""
    type: str
    diagnosis_date: str
    controlled: bool


@dataclass
class Medication:
    """Current medication"""
    name: str
    dose: str
    frequency: str
    indication: str


@dataclass
class Biological:
    """Patient biological/medical information"""
    height_cm: float
    weight_kg: float
    bmi: float
    blood_pressure: str
    immune_status: str  # normal/compromised/hyperactive
    chronic_conditions: List[ChronicCondition] = field(default_factory=list)
    risk_factors: List[RiskFactor] = field(default_factory=list)
    lifestyle: Optional[Lifestyle] = None
    medications: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate biological data"""
        if self.height_cm < 0 or self.height_cm > 300:
            raise ValueError(f"Invalid height: {self.height_cm}")
        if self.weight_kg < 0 or self.weight_kg > 500:
            raise ValueError(f"Invalid weight: {self.weight_kg}")
        if self.bmi < 0 or self.bmi > 100:
            raise ValueError(f"Invalid BMI: {self.bmi}")
        if self.immune_status not in ["normal", "compromised", "hyperactive"]:
            raise ValueError(f"Invalid immune_status: {self.immune_status}")
        return True


@dataclass
class PatientProfile:
    """Complete patient profile"""
    patient_id: str
    name: str
    demographics: Demographics
    communication: Communication
    psychological: Psychological
    biological: Biological
    assigned_disease: str
    disease_start_day: int

    def validate(self) -> bool:
        """Validate entire patient profile"""
        if not self.patient_id:
            raise ValueError("patient_id is required")
        if not self.name:
            raise ValueError("name is required")
        if self.disease_start_day < 0:
            raise ValueError(f"disease_start_day must be >= 0, got {self.disease_start_day}")

        # Validate sub-components
        self.demographics.validate()
        self.communication.validate()
        self.psychological.validate()
        self.biological.validate()

        return True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PatientProfile':
        """
        Load patient profile from YAML file

        Args:
            yaml_path: Path to patient profile YAML file

        Returns:
            PatientProfile object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Patient profile not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse demographics
        demographics = Demographics(**data['demographics'])

        # Parse communication
        communication = Communication(**data['communication'])

        # Parse psychological
        psychological = Psychological(**data['psychological'])

        # Parse lifestyle
        lifestyle_data = data['biological'].get('lifestyle', {})
        lifestyle = Lifestyle(**lifestyle_data) if lifestyle_data else None

        # Parse risk factors
        risk_factors = [
            RiskFactor(**rf) for rf in data['biological'].get('risk_factors', [])
        ]

        # Parse chronic conditions
        chronic_conditions = [
            ChronicCondition(**cc) for cc in data['biological'].get('chronic_conditions', [])
        ]

        # Parse biological
        biological_data = data['biological'].copy()
        biological_data['lifestyle'] = lifestyle
        biological_data['risk_factors'] = risk_factors
        biological_data['chronic_conditions'] = chronic_conditions
        biological = Biological(**biological_data)

        # Create patient profile
        profile = cls(
            patient_id=data['patient_id'],
            name=data['name'],
            demographics=demographics,
            communication=communication,
            psychological=psychological,
            biological=biological,
            assigned_disease=data['assigned_disease'],
            disease_start_day=data['disease_start_day']
        )

        # Validate
        profile.validate()

        return profile

    def to_dict(self) -> Dict[str, Any]:
        """Convert patient profile to dictionary"""
        return {
            'patient_id': self.patient_id,
            'name': self.name,
            'demographics': self.demographics.__dict__,
            'communication': self.communication.__dict__,
            'psychological': self.psychological.__dict__,
            'biological': {
                **self.biological.__dict__,
                'lifestyle': self.biological.lifestyle.__dict__ if self.biological.lifestyle else None,
                'risk_factors': [rf.__dict__ for rf in self.biological.risk_factors],
                'chronic_conditions': [cc.__dict__ for cc in self.biological.chronic_conditions]
            },
            'assigned_disease': self.assigned_disease,
            'disease_start_day': self.disease_start_day
        }

    def get_description(self) -> str:
        """Get human-readable description of patient"""
        return (
            f"{self.name} (ID: {self.patient_id})\n"
            f"Age: {self.demographics.age}, Gender: {self.demographics.gender}\n"
            f"Occupation: {self.demographics.occupation}\n"
            f"Anxiety: {self.psychological.baseline_anxiety}/10, "
            f"Health Anxiety: {self.psychological.health_anxiety}/10\n"
            f"Communication: {self.communication.vocabulary_style}, "
            f"Formality: {self.communication.formality}/10\n"
            f"Assigned Disease: {self.assigned_disease}"
        )
