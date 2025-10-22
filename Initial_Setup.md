# Medical Chatbot Memory Study - Technical Specification Document

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Data Schemas](#4-data-schemas)
5. [Core Components](#5-core-components)
6. [Implementation Requirements](#6-implementation-requirements)
7. [Workflow & Data Flow](#7-workflow--data-flow)
8. [Configuration](#8-configuration)
9. [Testing & Validation](#9-testing--validation)
10. [Usage Examples](#10-usage-examples)

---

## 1. Project Overview

### 1.1 Research Question
How does conversation history (memory between sessions) affect medical advice quality provided by general-purpose chatbots?

### 1.2 Goals
- Create a simulation system where virtual patients interact with chatbots
- Test chatbots in two conditions: WITH conversation history vs. WITHOUT history
- Measure differences in advice quality, continuity, and appropriateness
- Generate structured data for evaluation and analysis

### 1.3 Key Innovation
Unlike previous studies with hardcoded scenarios, this system uses:
- **Dynamic patient profiles** with personality, communication style, and anxiety levels
- **Realistic disease progression** that evolves day-by-day
- **LLM-as-patient** to enable natural, adaptive conversations
- **Multi-day timelines** mixing medical and non-medical conversations

---

## 2. System Architecture

### 2.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION ORCHESTRATOR                   │
│  - Manages experiment runs                                   │
│  - Controls day-by-day progression                          │
│  - Coordinates all components                               │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│   Patient    │      │   Disease    │     │   Medical    │
│  Simulator   │      │ Progression  │     │   Chatbot    │
│  (LLM-based) │      │    Engine    │     │  (Under Test)│
└──────────────┘      └──────────────┘     └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  State Manager   │
                    │ - Patient state  │
                    │ - Conversation   │
                    │   history        │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Data Persistence│
                    │ - YAML profiles  │
                    │ - JSON logs      │
                    │ - CSV metrics    │
                    └──────────────────┘
```

### 2.2 Technology Stack
- **Language**: Python 3.9+
- **LLM Interface**: Transformers library (HuggingFace), OpenAI API wrapper
- **Data Format**: YAML (configs), JSON (runtime data), CSV (metrics)
- **Libraries**: 
  - PyYAML (config files)
  - transformers, bitsandbytes (local LLMs)
  - pandas (data analysis)
  - numpy (numerical operations)
  - python-dateutil (timestamp handling)

---

## 3. Directory Structure

```
medical-chatbot-memory-study/
│
├── README.md                          # Project overview and setup
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── .gitignore                        # Git ignore rules
│
├── config/
│   └── experiment_config.yaml        # Main experiment configuration
│
├── profiles/                         # Patient profile definitions
│   ├── patient_001_alex_chen.yaml
│   ├── patient_002_maria_garcia.yaml
│   ├── patient_003_james_smith.yaml
│   └── template_patient.yaml         # Template for new patients
│
├── diseases/                         # Disease progression models
│   ├── viral_uri.yaml                # Common cold/flu
│   ├── migraine_progressive.yaml     # Progressive migraine
│   ├── chest_pain_cardiac.yaml       # Cardiac-related chest pain
│   └── template_disease.yaml         # Template for new diseases
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── core/                         # Core system components
│   │   ├── __init__.py
│   │   ├── simulation.py             # Main orchestrator
│   │   ├── patient_simulator.py      # LLM-based patient
│   │   ├── disease_engine.py         # Disease progression
│   │   ├── state_manager.py          # State tracking
│   │   └── conversation_runner.py    # Conversation management
│   │
│   ├── models/                       # Data models and schemas
│   │   ├── __init__.py
│   │   ├── patient_profile.py        # Patient data structure
│   │   ├── disease_model.py          # Disease data structure
│   │   ├── conversation.py           # Conversation data structure
│   │   └── metrics.py                # Metrics data structure
│   │
│   ├── chatbots/                     # Chatbot implementations
│   │   ├── __init__.py
│   │   ├── base_chatbot.py           # Abstract base class
│   │   ├── local_llm_chatbot.py      # For Mistral/Llama
│   │   └── api_chatbot.py            # For GPT/Claude (future)
│   │
│   ├── evaluation/                   # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics_calculator.py     # Auto metrics
│   │   ├── evaluator.py              # Human/LLM evaluation
│   │   └── analyzer.py               # Statistical analysis
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── file_utils.py             # File I/O helpers
│       ├── prompt_templates.py       # LLM prompts
│       └── logging_config.py         # Logging setup
│
├── data/                             # Runtime data (gitignored)
│   ├── conversations/                # Saved conversations
│   │   ├── patient_001/
│   │   │   ├── day_1_20240115_0923.json
│   │   │   ├── day_3_20240117_1445.json
│   │   │   └── metadata.json
│   │   └── patient_002/
│   │       └── ...
│   │
│   ├── state/                        # Current patient states
│   │   ├── patient_001_state.json
│   │   └── patient_002_state.json
│   │
│   └── results/                      # Evaluation results
│       ├── evaluations.csv
│       ├── metrics_summary.csv
│       └── full_results.json
│
├── scripts/                          # Executable scripts
│   ├── run_simulation.py             # Main entry point
│   ├── create_patient.py             # Create new patient profile
│   ├── create_disease.py             # Create new disease model
│   ├── evaluate_conversations.py    # Run evaluation
│   └── analyze_results.py            # Generate analysis
│
├── notebooks/                        # Jupyter/Colab notebooks
│   ├── 01_setup_and_test.ipynb      # Initial setup
│   ├── 02_run_experiment.ipynb      # Run full experiment
│   └── 03_analyze_results.ipynb     # Analyze results
│
└── tests/                            # Unit tests
    ├── __init__.py
    ├── test_patient_simulator.py
    ├── test_disease_engine.py
    └── test_conversation_runner.py
```

---

## 4. Data Schemas

### 4.1 Patient Profile Schema (YAML)

```yaml
# profiles/patient_XXX_name.yaml

patient_id: "patient_001"              # Unique identifier
name: "Alex Chen"                      # For reference only

demographics:
  age: 34                              # Years
  gender: "male"                       # male/female/other
  occupation: "high school teacher"    # Free text
  location: "urban"                    # urban/suburban/rural

communication:
  formality: 3                         # 1-10: 1=very formal, 10=very casual
  verbosity: 6                         # 1-10: 1=terse, 10=detailed
  vocabulary_style: "casual"           # casual/professional/medical
  example_phrases:                     # List of typical phrases
    - "I've been feeling pretty crappy"
    - "It's annoying af"
    - "I don't know if this matters but..."

psychological:
  baseline_anxiety: 7                  # 1-10: General anxiety level
  health_anxiety: 8                    # 1-10: Specific to health concerns
  pain_catastrophizing: 6              # 1-10: Tendency to imagine worst
  information_volunteering: "low"      # low/medium/high
  current_mood: "stressed"             # anxious/calm/stressed/depressed/neutral
  mood_triggers:                       # List of current stressors
    - "Work evaluation coming up"
    - "Family issues"

biological:
  height_cm: 175                       # Height in cm
  weight_kg: 82                        # Weight in kg
  bmi: 26.8                           # Calculated or provided
  blood_pressure: "130/85"            # Baseline BP
  immune_status: "normal"             # normal/compromised/hyperactive
  
  chronic_conditions: []              # List of ongoing conditions
  # Example:
  # - type: "type_2_diabetes"
  #   diagnosis_date: "2020-03-15"
  #   controlled: true
  
  risk_factors:                       # List of risk factors
    - type: "family_history_cvd"
      details: "Father had MI at age 55"
      risk_multiplier: 1.3
    - type: "family_history_migraine"
      details: "Mother and sister have migraines"
      risk_multiplier: 1.5
  
  lifestyle:
    sleep_hours: 5.5                 # Average hours per night
    sleep_quality: "poor"            # poor/fair/good/excellent
    stress_level: 8                  # 1-10
    exercise_frequency: 1            # Times per week
    exercise_type: "light walking"   # Free text
    diet_quality: "moderate"         # poor/moderate/good
    caffeine_intake: "high"          # none/low/moderate/high
    caffeine_amount: "4-5 cups/day"  # Details
    alcohol: "moderate"              # none/light/moderate/heavy
    smoking: false                   # Boolean
    
  medications:
    current: []                      # List of current medications
    # Example:
    # - name: "metformin"
    #   dose: "500mg"
    #   frequency: "twice daily"
    #   indication: "type 2 diabetes"
    allergies: ["penicillin"]        # List of drug allergies

assigned_disease: "viral_uri"        # Disease model ID
disease_start_day: 1                 # Which simulation day disease starts
```

### 4.2 Disease Model Schema (YAML)

```yaml
# diseases/disease_name.yaml

disease_id: "viral_uri"
name: "Viral Upper Respiratory Infection"
common_name: "Cold/Flu"
description: "Common viral infection of upper respiratory tract"

natural_history:
  typical_duration: [7, 14]          # Range in days [min, max]
  
  phases:
    - name: "incubation"
      day_range: [0, 2]              # Days [start, end]
      description: "Virus replicating, minimal symptoms"
      
      symptoms:
        fatigue:
          severity_range: [0, 2]     # Range [min, max] on 0-10 scale
          trajectory: "flat"         # flat/increasing/decreasing/peak_then_decline
          progression_rate: 0        # Change per day (if applicable)
          description_templates:     # Random choice for variety
            - "feeling a bit off"
            - "maybe slightly more tired than usual"
          patient_notices: false     # Does patient usually notice this?
    
    - name: "prodrome"
      day_range: [3, 5]
      description: "Early symptoms emerge"
      
      symptoms:
        sore_throat:
          severity_range: [2, 6]
          trajectory: "increasing"
          progression_rate: 1.5      # Increases 1.5 points per day
          description_templates:
            - "scratchy throat"
            - "throat feels raw"
            - "painful to swallow"
          patient_notices: true
        
        fatigue:
          severity_range: [2, 5]
          trajectory: "increasing"
          progression_rate: 1.0
          description_templates:
            - "feeling tired"
            - "exhausted"
            - "no energy"
          patient_notices: true
        
        headache:
          severity_range: [1, 4]
          trajectory: "stable"
          description_templates:
            - "dull headache"
            - "head pressure"
          patient_notices: true
    
    - name: "acute"
      day_range: [5, 8]
      description: "Peak illness"
      
      symptoms:
        fever:
          severity_range: [4, 7]
          trajectory: "peak_then_decline"
          peak_day: 6                # Relative to disease start
          description_templates:
            - "feeling hot"
            - "chills and fever"
            - "burning up"
          patient_notices: true
          objective_measure:
            type: "temperature_celsius"
            formula: "37.0 + (severity * 0.3)"  # Converts severity to temp
        
        cough:
          severity_range: [2, 7]
          trajectory: "increasing"
          progression_rate: 1.2
          subtype: "dry_then_productive"  # Type changes over time
          description_templates:
            - "dry cough"
            - "can't stop coughing"
            - "coughing up mucus"
          patient_notices: true
        
        congestion:
          severity_range: [4, 8]
          trajectory: "increasing"
          progression_rate: 1.0
          description_templates:
            - "stuffy nose"
            - "can't breathe through nose"
            - "congested"
          patient_notices: true
        
        body_aches:
          severity_range: [3, 6]
          trajectory: "stable"
          description_templates:
            - "body aches"
            - "muscles hurt"
            - "everything hurts"
          patient_notices: true
    
    - name: "resolution"
      day_range: [8, 14]
      description: "Recovery phase"
      
      symptoms:
        cough:
          severity_range: [5, 1]
          trajectory: "decreasing"
          progression_rate: -0.7
          description_templates:
            - "lingering cough"
            - "occasional cough"
          patient_notices: true
        
        fatigue:
          severity_range: [3, 1]
          trajectory: "decreasing"
          progression_rate: -0.5
          description_templates:
            - "still a bit tired"
            - "getting energy back"
          patient_notices: true

# Complications and risk modifiers
complications:
  base_probability: 0.05             # 5% base chance
  
  risk_modifiers:
    - factor: "immune_compromised"
      probability_multiplier: 4.0
    - factor: "chronic_lung_disease"
      probability_multiplier: 3.0
    - factor: "age_over_65"
      probability_multiplier: 2.0
    - factor: "poor_sleep"
      probability_multiplier: 1.3
    - factor: "high_stress"
      probability_multiplier: 1.2
  
  types:
    - complication_id: "secondary_bacterial"
      name: "Secondary Bacterial Infection"
      probability_if_complication: 0.6  # 60% of complications
      trigger_day_range: [7, 10]
      
      new_symptoms:
        fever_return:
          severity: 7
          description: "Fever returns after improving"
        purulent_sputum:
          severity: 5
          description: "Coughing up yellow/green mucus"
      
      changes_to_existing:
        cough:
          severity_modifier: +2
    
    - complication_id: "bronchitis"
      name: "Acute Bronchitis"
      probability_if_complication: 0.3
      trigger_day_range: [10, 21]
      
      new_symptoms:
        wheezing:
          severity: 4
          description: "Wheezing when breathing"
      
      changes_to_existing:
        cough:
          severity_modifier: +1
          duration_extension: 14  # Extra days

# Patient-specific modifiers
patient_modifiers:
  # How patient factors affect disease progression
  poor_sleep:
    effect_on_symptoms: "+1 to all severity scores"
    effect_on_duration: "+2 days to each phase"
  
  high_stress:
    effect_on_symptoms: "+0.5 to all severity scores"
    effect_on_duration: "+1 day to each phase"
  
  high_caffeine:
    effect_on_symptoms: "+1 to headache severity"
  
  poor_immune:
    effect_on_symptoms: "+2 to all severity scores"
    effect_on_duration: "+3 days to each phase"
    complication_risk_multiplier: 2.0
```

### 4.3 Conversation Log Schema (JSON)

```json
{
  "conversation_id": "patient_001_day_5_20240119_1012",
  "patient_id": "patient_001",
  "simulation_day": 5,
  "disease_day": 5,
  "real_timestamp": "2024-01-19T10:12:45Z",
  "condition": "with_history",
  
  "patient_state_at_start": {
    "biological": {
      "temperature": 38.5,
      "heart_rate": 85,
      "energy_level": 3,
      "pain_level": 6
    },
    "psychological": {
      "anxiety_level": 8,
      "mood": "worried",
      "health_concern_level": 7
    },
    "symptoms": {
      "sore_throat": {
        "objective_severity": 6,
        "reported_severity": 7.5,
        "description": "throat is killing me"
      },
      "fever": {
        "objective_severity": 5,
        "reported_severity": 6,
        "description": "feeling hot and shitty"
      }
    }
  },
  
  "turns": [
    {
      "turn_number": 1,
      "speaker": "patient",
      "message": "Hey so I've been feeling pretty sick for the past few days and I'm kinda worried about it",
      "timestamp": "2024-01-19T10:12:45Z",
      "internal_state": {
        "anxiety_level": 8,
        "symptoms_mentioned": [],
        "information_volunteered": ["duration", "concern"]
      }
    },
    {
      "turn_number": 2,
      "speaker": "chatbot",
      "message": "I'm sorry to hear you're not feeling well. Can you tell me more about your symptoms?",
      "timestamp": "2024-01-19T10:12:52Z",
      "model_info": {
        "model_name": "mistral-7b-instruct",
        "temperature": 0.7,
        "max_tokens": 512
      }
    }
  ],
  
  "conversation_summary": {
    "total_turns": 8,
    "duration_seconds": 245,
    "conversation_end_reason": "bot_provided_advice",
    "final_bot_recommendation": "self_care_with_monitoring"
  },
  
  "metrics": {
    "automatic": {
      "bot_questions_asked": 4,
      "redundant_questions": 1,
      "references_to_history": 2,
      "key_info_gathered": ["duration", "severity", "other_symptoms", "fever"],
      "key_info_missed": ["medication_tried"],
      "patient_volunteered_count": 2,
      "patient_only_when_asked_count": 6
    },
    "evaluation": {
      "evaluated": false,
      "evaluator_id": null,
      "scores": {}
    }
  }
}
```

### 4.4 Patient State Schema (JSON)

```json
{
  "patient_id": "patient_001",
  "current_day": 5,
  "last_updated": "2024-01-19T10:12:45Z",
  
  "biological": {
    "temperature": 38.5,
    "heart_rate": 85,
    "respiratory_rate": 18,
    "blood_pressure": "135/88",
    "energy_level": 3,
    "pain_level": 6,
    "immune_markers": {
      "white_blood_cells": "elevated",
      "inflammatory_markers": "high"
    }
  },
  
  "psychological": {
    "anxiety_level": 8,
    "mood": "worried",
    "health_concern_level": 7
  },
  
  "disease_state": {
    "active_disease": "viral_uri",
    "disease_day": 5,
    "current_phase": "acute",
    "trajectory": "worsening",
    "complications_occurred": false,
    "expected_resolution_day": 12
  },
  
  "current_symptoms": {
    "sore_throat": {
      "objective_severity": 6,
      "reported_severity": 7.5,
      "first_appeared_day": 3,
      "description": "throat is killing me",
      "patient_notices": true
    },
    "fever": {
      "objective_severity": 5,
      "reported_severity": 6,
      "first_appeared_day": 5,
      "description": "feeling hot and shitty",
      "patient_notices": true,
      "objective_value": 38.5
    }
  },
  
  "conversation_history_summary": [
    {
      "day": 1,
      "type": "casual",
      "topic": "Python coding help"
    },
    {
      "day": 3,
      "type": "medical",
      "topic": "Initial symptoms - sore throat",
      "key_info": ["sore_throat", "fatigue", "duration_2_days"]
    },
    {
      "day": 5,
      "type": "medical_followup",
      "topic": "Worsening symptoms",
      "key_info": ["fever_new", "sore_throat_worse", "concern_increased"]
    }
  ]
}
```

---

## 5. Core Components

### 5.1 Simulation Orchestrator

**File**: `src/core/simulation.py`

**Class**: `MedicalChatbotSimulation`

**Responsibilities**:
- Coordinate all system components
- Manage day-by-day simulation progression
- Handle both experimental conditions (with/without history)
- Save all data persistently

**Key Methods**:

```python
class MedicalChatbotSimulation:
    def __init__(self, config_path: str):
        """
        Initialize simulation from config file
        
        Args:
            config_path: Path to experiment_config.yaml
        """
        pass
    
    def load_patient(self, patient_id: str) -> PatientProfile:
        """Load patient profile from YAML"""
        pass
    
    def load_disease(self, disease_id: str) -> DiseaseModel:
        """Load disease model from YAML"""
        pass
    
    def run_experiment(self, patient_id: str, num_days: int, condition: str):
        """
        Run full experiment for one patient
        
        Args:
            patient_id: Patient identifier
            num_days: Total simulation days
            condition: 'with_history' or 'without_history'
        """
        pass
    
    def run_day(self, day: int, conversation_type: str) -> Conversation:
        """
        Run single day simulation
        
        Args:
            day: Current simulation day
            conversation_type: 'medical' or 'casual'
        
        Returns:
            Conversation object with all data
        """
        pass
    
    def generate_timeline(self, num_days: int) -> List[Dict]:
        """
        Generate conversation timeline (mix of medical and casual)
        
        Args:
            num_days: Total days to simulate
        
        Returns:
            List of dicts with day, type, and other metadata
        """
        pass
```

**Algorithm for run_day**:

```
1. Get current patient state from StateManager
2. Use DiseaseEngine to get symptoms for this day
3. Update patient state with new symptoms
4. Create PatientSimulator prompt with:
   - Patient profile
   - Current symptoms
   - Anxiety modifiers
   - Communication style
5. Initialize appropriate chatbot (with or without history)
6. Run ConversationRunner to execute back-and-forth
7. Calculate automatic metrics
8. Save conversation to JSON
9. Update StateManager
10. Return Conversation object
```

---

### 5.2 Patient Simulator

**File**: `src/core/patient_simulator.py`

**Class**: `PatientSimulator`

**Responsibilities**:
- Generate patient responses using LLM
- Modulate symptom reporting based on anxiety/personality
- Stay in character throughout conversation
- Reveal information naturally (not all at once)

**Key Methods**:

```python
class PatientSimulator:
    def __init__(self, patient_profile: PatientProfile, current_state: PatientState, llm_model: str):
        """
        Initialize patient simulator
        
        Args:
            patient_profile: Full patient profile
            current_state: Current biological/psychological state
            llm_model: LLM model to use (e.g., "mistral-7b-instruct")
        """
        pass
    
    def create_system_prompt(self) -> str:
        """
        Generate comprehensive system prompt for patient LLM
        Includes personality, symptoms, communication style, etc.
        
        Returns:
            Formatted prompt string
        """
        pass
    
    def generate_opening_message(self) -> str:
        """
        Generate patient's initial message to start conversation
        
        Returns:
            Opening message string
        """
        pass
    
    def respond(self, conversation_history: List[Dict]) -> str:
        """
        Generate patient response to chatbot
        
        Args:
            conversation_history: List of previous turns
        
        Returns:
            Patient's response message
        """
        pass
    
    def modulate_symptom_reporting(self, objective_severity: float) -> float:
        """
        Apply psychological modifiers to symptom severity
        
        Args:
            objective_severity: True severity (0-10)
        
        Returns:
            Reported severity after anxiety/personality modulation
        """
        pass
    
    def generate_symptom_description(self, symptom: str, severity: float) -> str:
        """
        Generate natural language symptom description
        
        Args:
            symptom: Symptom name
            severity: Severity level
        
        Returns:
            Natural description matching communication style
        """
        pass
```

**System Prompt Template** (stored in `src/utils/prompt_templates.py`):

```python
PATIENT_SYSTEM_PROMPT = """
You are roleplaying as a patient seeking medical advice from a chatbot.

=== YOUR IDENTITY ===
Name: {name}
Age: {age}, {gender}
Occupation: {occupation}

=== YOUR PERSONALITY ===
Communication style: {communication_description}
- Formality level: {formality}/10 (1=very formal, 10=very casual)
- You use {vocabulary_style} language
- Example phrases you might use: {example_phrases}

Anxiety and mood:
- Your baseline anxiety is {baseline_anxiety}/10
- You're specifically anxious about health: {health_anxiety}/10
- Current mood: {current_mood} because {mood_triggers}
- You tend to {catastrophizing_behavior}

=== YOUR CURRENT SITUATION ===
You're on day {current_day} of feeling unwell.

Current symptoms you're experiencing:
{symptoms_list}

Your subjective experience (anxiety is affecting perception):
{subjective_experience}

=== YOUR BACKGROUND (reveal if asked) ===
{background_info}

=== HOW YOU RESPOND ===
{response_guidelines}

=== CONVERSATION RULES ===
1. Stay in character consistently
2. Don't list all symptoms at once - reveal naturally over conversation
3. Sometimes you're unsure ("maybe", "I think", "I'm not sure")
4. Express worry when anxious: "Do you think it could be something serious?"
5. Use your communication style (casual vs formal)
6. Don't volunteer information unless it makes sense naturally
7. Answer questions but don't over-explain

Begin the conversation naturally based on your personality and symptoms.
"""
```

---

### 5.3 Disease Progression Engine

**File**: `src/core/disease_engine.py`

**Class**: `DiseaseProgressionEngine`

**Responsibilities**:
- Calculate symptoms for each day based on disease model
- Apply patient-specific modifiers
- Handle complications probabilistically
- Track disease timeline and phases

**Key Methods**:

```python
class DiseaseProgressionEngine:
    def __init__(self, disease_model: DiseaseModel, patient_profile: PatientProfile):
        """
        Initialize disease engine
        
        Args:
            disease_model: Disease model loaded from YAML
            patient_profile: Patient profile for modifiers
        """
        pass
    
    def get_symptoms_for_day(self, day: int) -> Dict[str, Dict]:
        """
        Get all symptoms and severities for specific day
        
        Args:
            day: Disease day (1-indexed)
        
        Returns:
            Dict of symptoms with severity and descriptions
            {
                "symptom_name": {
                    "objective_severity": 5.5,
                    "description_template": "throat feels raw",
                    "patient_notices": True
                }
            }
        """
        pass
    
    def _get_current_phase(self, day: int) -> Dict:
        """Determine which disease phase for this day"""
        pass
    
    def _calculate_symptom_severity(self, symptom_config: Dict, day: int, phase: Dict) -> float:
        """
        Calculate severity based on trajectory
        
        Handles:
        - flat: constant severity
        - increasing: linear increase
        - decreasing: linear decrease
        - peak_then_decline: bell curve
        - stable: random within range
        """
        pass
    
    def _apply_patient_modifiers(self, symptoms: Dict) -> Dict:
        """
        Modify symptoms based on patient factors
        
        Examples:
        - Poor sleep: +1 to all severities
        - High stress: +0.5 to all severities, longer duration
        - High caffeine: +1 to headache
        - Immune compromised: +2 to all severities
        """
        pass
    
    def _check_for_complication(self, day: int) -> Optional[str]:
        """
        Probabilistically check if complication occurs
        
        Returns:
            Complication ID if triggered, None otherwise
        """
        pass
    
    def _apply_complication(self, symptoms: Dict, complication_id: str) -> Dict:
        """Add new symptoms and modify existing ones for complication"""
        pass
```

**Trajectory Calculation Examples**:

```python
# For "increasing" trajectory
severity = base_severity + (days_into_phase * progression_rate)

# For "peak_then_decline" trajectory
if day < peak_day:
    severity = base_severity + ((day - phase_start) * increase_rate)
else:
    severity = peak_severity - ((day - peak_day) * decrease_rate)

# For "stable" trajectory
severity = random.uniform(min_severity, max_severity)

# Always add small random variation
severity += random.uniform(-0.5, 0.5)

# Clamp to valid range
severity = max(0, min(10, severity))
```

---

### 5.4 State Manager

**File**: `src/core/state_manager.py`

**Class**: `PatientStateManager`

**Responsibilities**:
- Track patient's current state (biological + psychological)
- Update state based on disease progression
- Maintain conversation history summary
- Persist state to disk

**Key Methods**:

```python
class PatientStateManager:
    def __init__(self, patient_profile: PatientProfile):
        """Initialize state manager for patient"""
        pass
    
    def initialize_state(self) -> PatientState:
        """Create initial state at day 0"""
        pass
    
    def update_state(self, day: int, disease_symptoms: Dict) -> PatientState:
        """
        Update state for new day
        
        Args:
            day: Current day
            disease_symptoms: Symptoms from disease engine
        
        Returns:
            Updated state
        """
        pass
    
    def get_current_state(self) -> PatientState:
        """Get current state"""
        pass
    
    def add_conversation_to_history(self, conversation: Conversation):
        """Add conversation summary to history"""
        pass
    
    def get_conversation_history(self, condition: str) -> List[Dict]:
        """
        Get conversation history for chatbot context
        
        Args:
            condition: 'with_history' returns all, 'without_history' returns empty
        """
        pass
    
    def save_state(self):
        """Persist state to JSON file"""
        pass
    
    def load_state(self, patient_id: str) -> PatientState:
        """Load state from JSON file"""
        pass
    
    def _calculate_biological_state(self, symptoms: Dict) -> Dict:
        """
        Calculate objective biological measurements
        
        Examples:
        - Temperature from fever severity: 37.0 + (fever_severity * 0.3)
        - Heart rate from fever: 70 + (fever_severity * 4)
        - Energy level from fatigue: 10 - fatigue_severity
        """
        pass
    
    def _update_psychological_state(self, disease_state: Dict) -> Dict:
        """
        Update anxiety/mood based on symptoms
        
        Examples:
        - High symptom severity → increased anxiety
        - Worsening trajectory → increased concern
        - Improving → decreased anxiety (toward baseline)
        """
        pass
```

---

### 5.5 Conversation Runner

**File**: `src/core/conversation_runner.py`

**Class**: `ConversationRunner`

**Responsibilities**:
- Orchestrate back-and-forth between patient and chatbot
- Track conversation metrics in real-time
- Determine when conversation should end
- Format and save conversation logs

**Key Methods**:

```python
class ConversationRunner:
    def __init__(self, patient_simulator: PatientSimulator, chatbot: BaseChatbot, max_turns: int = 10):
        """
        Initialize conversation runner
        
        Args:
            patient_simulator: Initialized patient simulator
            chatbot: Initialized chatbot (with or without history)
            max_turns: Maximum conversation turns
        """
        pass
    
    def run_conversation(self) -> Conversation:
        """
        Execute full conversation
        
        Returns:
            Conversation object with all turns and metrics
        """
        pass
    
    def should_end_conversation(self, last_message: str, speaker: str, turn_count: int) -> bool:
        """
        Determine if conversation should end
        
        Criteria:
        - Max turns reached
        - Bot provided final advice/recommendation
        - Patient explicitly ends (e.g., "ok thanks")
        - Conversation became circular
        """
        pass
    
    def calculate_turn_metrics(self, turn: Dict, conversation_history: List[Dict]) -> Dict:
        """Calculate metrics for single turn"""
        pass
    
    def calculate_conversation_metrics(self, turns: List[Dict]) -> Dict:
        """Calculate aggregate metrics for full conversation"""
        pass
```

**Conversation Flow Algorithm**:

```
1. Initialize empty turns list
2. Get patient's opening message
3. Add to turns
4. LOOP until should_end_conversation:
   a. If patient's turn:
      - Get patient simulator response
      - Calculate patient-side metrics
   b. If chatbot's turn:
      - Get chatbot response
      - Calculate chatbot-side metrics (questions asked, references to history, etc.)
   c. Add turn to turns list
   d. Check end conditions
5. Calculate final conversation metrics
6. Return Conversation object
```

---

### 5.6 Chatbot Implementations

**File**: `src/chatbots/base_chatbot.py`

**Abstract Base Class**: `BaseChatbot`

```python
from abc import ABC, abstractmethod

class BaseChatbot(ABC):
    def __init__(self, context: List[Dict] = None):
        """
        Args:
            context: Previous conversation history (empty for without_history condition)
        """
        self.context = context or []
    
    @abstractmethod
    def respond(self, conversation_turns: List[Dict]) -> str:
        """
        Generate chatbot response
        
        Args:
            conversation_turns: Current conversation turns
        
        Returns:
            Response message
        """
        pass
    
    @abstractmethod
    def format_messages_for_model(self, turns: List[Dict]) -> Any:
        """Format turns for specific model input format"""
        pass
```

**File**: `src/chatbots/local_llm_chatbot.py`

**Class**: `LocalLLMChatbot(BaseChatbot)`

```python
class LocalLLMChatbot(BaseChatbot):
    def __init__(self, model_name: str, context: List[Dict] = None, device: str = "cuda"):
        """
        Initialize local LLM chatbot
        
        Args:
            model_name: HuggingFace model name (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
            context: Conversation history
            device: "cuda" or "cpu"
        """
        super().__init__(context)
        self.model_name = model_name
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load model with 4-bit quantization for efficiency"""
        # Use bitsandbytes for quantization
        # Load tokenizer and model
        pass
    
    def respond(self, conversation_turns: List[Dict]) -> str:
        """Generate response using local LLM"""
        # Format context + current turns
        # Generate response
        # Return message
        pass
    
    def format_messages_for_model(self, turns: List[Dict]) -> str:
        """Format using chat template"""
        # Apply model's chat template
        # Return formatted string
        pass
```

---

### 5.7 Metrics Calculator

**File**: `src/evaluation/metrics_calculator.py`

**Class**: `MetricsCalculator`

**Responsibilities**:
- Calculate automatic metrics during and after conversation
- Detect redundant questions
- Count information gathering
- Identify references to history

**Key Methods**:

```python
class MetricsCalculator:
    def __init__(self, condition: str):
        """
        Args:
            condition: 'with_history' or 'without_history'
        """
        self.condition = condition
        self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict:
        """Initialize empty metrics dict"""
        return {
            'total_turns': 0,
            'bot_questions_asked': 0,
            'redundant_questions': 0,
            'references_to_history': 0,
            'contextual_questions': 0,
            'key_info_gathered': [],
            'key_info_missed': [],
            'patient_volunteered_count': 0,
            'patient_only_when_asked_count': 0
        }
    
    def analyze_turn(self, turn: Dict, conversation_history: List[Dict], patient_history: List[Dict]) -> Dict:
        """
        Analyze single turn for metrics
        
        Args:
            turn: Current turn data
            conversation_history: All previous turns in this conversation
            patient_history: All previous conversations (for redundancy check)
        """
        pass
    
    def is_question(self, message: str) -> bool:
        """Check if message contains question"""
        # Look for ? and question words
        pass
    
    def is_redundant_question(self, question: str, patient_history: List[Dict]) -> bool:
        """
        Check if question asks for information already in history
        
        Only applies to 'with_history' condition
        """
        pass
    
    def count_references_to_history(self, message: str) -> int:
        """
        Count phrases that reference previous conversations
        
        Examples: "you mentioned", "last time", "when we talked", "previously"
        """
        pass
    
    def extract_info_gathered(self, conversation: List[Dict]) -> List[str]:
        """
        Identify what key information was gathered
        
        Key info types:
        - duration
        - severity
        - location
        - other_symptoms
        - fever_measured
        - medications_tried
        - previous_episodes
        - lifestyle_factors
        """
        pass
    
    def identify_missed_info(self, conversation: List[Dict], patient_state: PatientState) -> List[str]:
        """
        Identify important information chatbot failed to ask about
        
        Examples:
        - Didn't ask about fever when patient mentions feeling hot
        - Didn't ask about duration
        - Didn't ask about other symptoms
        - Didn't ask about medications tried
        """
        pass
    
    def calculate_final_metrics(self, conversation: Conversation) -> Dict:
        """Calculate aggregate metrics for full conversation"""
        pass
```

---

## 6. Implementation Requirements

### 6.1 Dependencies (requirements.txt)

```txt
# Core
python>=3.9
pyyaml>=6.0
numpy>=1.24.0
pandas>=2.0.0
python-dateutil>=2.8.0

# LLM Libraries
transformers>=4.35.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
torch>=2.0.0
sentencepiece>=0.1.99

# Optional (for future API support)
openai>=1.0.0
anthropic>=0.7.0

# Utilities
tqdm>=4.65.0
jsonschema>=4.19.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Notebooks
jupyter>=1.0.0
ipywidgets>=8.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 6.2 Configuration File

**File**: `config/experiment_config.yaml`

```yaml
experiment:
  name: "medical_chatbot_memory_study"
  description: "Comparing chatbot performance with and without conversation history"
  version: "1.0"

simulation:
  default_max_days: 14
  default_max_turns_per_conversation: 10
  default_conversation_mix:
    - day: 1
      type: "medical"
    - day: 3
      type: "casual"
    - day: 5
      type: "medical_followup"
    - day: 7
      type: "casual"
    - day: 9
      type: "medical_followup"
    - day: 12
      type: "medical_followup"

models:
  patient_llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
    device: "cuda"
    temperature: 0.7
    max_tokens: 512
    quantization: "4bit"
  
  doctor_llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
    device: "cuda"
    temperature: 0.7
    max_tokens: 512
    quantization: "4bit"

conditions:
  - "with_history"
  - "without_history"

evaluation:
  automatic_metrics: true
  human_evaluation: false
  llm_as_judge: false

paths:
  profiles: "profiles/"
  diseases: "diseases/"
  conversations: "data/conversations/"
  state: "data/state/"
  results: "data/results/"

logging:
  level: "INFO"
  file: "experiment.log"
  console: true
```

### 6.3 Error Handling

All components should implement proper error handling:

```python
# Example error handling pattern
try:
    patient_profile = load_patient_profile(patient_id)
except FileNotFoundError:
    logger.error(f"Patient profile {patient_id} not found")
    raise
except yaml.YAMLError as e:
    logger.error(f"Invalid YAML in patient profile: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading patient: {e}")
    raise

# Validation example
def validate_patient_profile(profile: Dict) -> bool:
    """Validate patient profile has all required fields"""
    required_fields = ['patient_id', 'demographics', 'communication', 'psychological', 'biological']
    for field in required_fields:
        if field not in profile:
            raise ValueError(f"Missing required field: {field}")
    return True
```

### 6.4 Logging

Use Python's logging module throughout:

```python
import logging

logger = logging.getLogger(__name__)

# In code
logger.info(f"Starting simulation for patient {patient_id}, day {day}")
logger.debug(f"Disease symptoms for day {day}: {symptoms}")
logger.warning(f"Redundant question detected: {question}")
logger.error(f"Failed to generate patient response: {error}")
```

---

## 7. Workflow & Data Flow

### 7.1 Complete Simulation Workflow

```
START
  │
  ├─ Load experiment config
  │
  ├─ FOR EACH patient in experiment:
  │   │
  │   ├─ Load patient profile (YAML)
  │   ├─ Load disease model (YAML)
  │   ├─ Initialize DiseaseProgressionEngine
  │   ├─ Initialize PatientStateManager
  │   │
  │   ├─ FOR EACH condition (with_history, without_history):
  │   │   │
  │   │   ├─ Reset patient state
  │   │   ├─ Generate conversation timeline
  │   │   │
  │   │   ├─ FOR EACH day in timeline:
  │   │   │   │
  │   │   │   ├─ Get symptoms for this day (DiseaseEngine)
  │   │   │   ├─ Update patient state (StateManager)
  │   │   │   ├─ Create patient simulator prompt
  │   │   │   ├─ Initialize chatbot with/without context
  │   │   │   ├─ Run conversation (ConversationRunner)
  │   │   │   ├─ Calculate metrics
  │   │   │   ├─ Save conversation (JSON)
  │   │   │   └─ Update conversation history
  │   │   │
  │   │   └─ Save condition results
  │   │
  │   └─ Generate patient summary
  │
  ├─ Aggregate all results
  ├─ Calculate comparative statistics
  └─ Generate final report

END
```

### 7.2 Single Day Data Flow

```
Day N Start
    │
    ▼
┌─────────────────────────┐
│  DiseaseProgressionEngine│
│  - Calculate symptoms   │
│  - Apply modifiers      │
│  - Check complications  │
└───────────┬─────────────┘
            │
            ▼ (symptoms dict)
┌─────────────────────────┐
│   PatientStateManager   │
│  - Update bio state     │
│  - Update psych state   │
│  - Calculate vital signs│
└───────────┬─────────────┘
            │
            ▼ (current state)
┌─────────────────────────┐
│   PatientSimulator      │
│  - Create system prompt │
│  - Modulate symptoms    │
│  - Generate responses   │
└───────────┬─────────────┘
            │
            ▼ (patient LLM ready)
┌─────────────────────────┐
│    MedicalChatbot       │
│  - Load context         │
│  - Generate responses   │
└───────────┬─────────────┘
            │
            ▼ (chatbot ready)
┌─────────────────────────┐
│   ConversationRunner    │
│  - Alternate turns      │
│  - Track metrics        │
│  - Determine end        │
└───────────┬─────────────┘
            │
            ▼ (conversation object)
┌─────────────────────────┐
│   MetricsCalculator     │
│  - Calculate metrics    │
│  - Identify patterns    │
└───────────┬─────────────┘
            │
            ▼ (metrics)
┌─────────────────────────┐
│   Save to Disk          │
│  - Conversation JSON    │
│  - State JSON           │
│  - Update metadata      │
└─────────────────────────┘
```

---

## 8. Configuration

### 8.1 Creating New Patient Profile

**Script**: `scripts/create_patient.py`

```python
#!/usr/bin/env python3
"""
Interactive script to create new patient profile
"""

def create_patient_interactive():
    """Guide user through creating patient profile"""
    
    print("=== Patient Profile Creator ===\n")
    
    # Basic info
    patient_id = input("Patient ID (e.g., patient_004): ")
    name = input("Patient name: ")
    age = int(input("Age: "))
    gender = input("Gender (male/female/other): ")
    occupation = input("Occupation: ")
    
    # Communication style
    print("\n--- Communication Style ---")
    formality = int(input("Formality (1-10, 1=formal, 10=casual): "))
    verbosity = int(input("Verbosity (1-10, 1=terse, 10=detailed): "))
    vocabulary_style = input("Vocabulary (casual/professional/medical): ")
    
    # ... continue for all fields
    
    # Generate YAML
    profile = {
        'patient_id': patient_id,
        'name': name,
        # ... all fields
    }
    
    # Save
    filename = f"profiles/{patient_id}_{name.lower().replace(' ', '_')}.yaml"
    with open(filename, 'w') as f:
        yaml.dump(profile, f, default_flow_style=False)
    
    print(f"\n✓ Patient profile saved to {filename}")

if __name__ == "__main__":
    create_patient_interactive()
```

### 8.2 Creating New Disease Model

**Script**: `scripts/create_disease.py`

Similar interactive script for disease models.

---

## 9. Testing & Validation

### 9.1 Unit Tests

**File**: `tests/test_disease_engine.py`

```python
import pytest
from src.core.disease_engine import DiseaseProgressionEngine
from src.models.disease_model import DiseaseModel
from src.models.patient_profile import PatientProfile

def test_symptom_severity_calculation():
    """Test that symptom severity calculates correctly for different trajectories"""
    # Load test disease model
    disease_model = DiseaseModel.from_yaml("diseases/test_viral_uri.yaml")
    patient_profile = PatientProfile.from_yaml("profiles/test_patient.yaml")
    
    engine = DiseaseProgressionEngine(disease_model, patient_profile)
    
    # Test day 1 (incubation)
    symptoms_day1 = engine.get_symptoms_for_day(1)
    assert 'fatigue' in symptoms_day1
    assert 0 <= symptoms_day1['fatigue']['objective_severity'] <= 2
    
    # Test day 6 (acute phase)
    symptoms_day6 = engine.get_symptoms_for_day(6)
    assert 'fever' in symptoms_day6
    assert symptoms_day6['fever']['objective_severity'] > 4

def test_patient_modifiers():
    """Test that patient factors correctly modify symptoms"""
    # Test with poor sleep
    # Test with high stress
    # Assert modifiers are applied
    pass

def test_complications():
    """Test complication triggering"""
    # Set high risk factors
    # Run multiple simulations
    # Check complication occurs at appropriate rate
    pass
```

### 9.2 Integration Tests

**File**: `tests/test_conversation_flow.py`

```python
def test_full_conversation():
    """Test complete conversation flow"""
    # Initialize all components
    # Run one full day
    # Assert conversation is saved
    # Assert metrics are calculated
    # Assert state is updated
    pass

def test_with_vs_without_history():
    """Test that history is correctly included/excluded"""
    # Run same scenario twice
    # Once with history, once without
    # Assert chatbot has/doesn't have context
    pass
```

### 9.3 Validation Functions

```python
def validate_conversation_json(conversation_path: str) -> bool:
    """Validate conversation JSON against schema"""
    # Load JSON
    # Check all required fields
    # Check data types
    # Return True if valid
    pass

def validate_patient_profile(profile_path: str) -> bool:
    """Validate patient profile YAML"""
    # Load YAML
    # Check required fields
    # Check value ranges
    pass

def validate_disease_model(disease_path: str) -> bool:
    """Validate disease model YAML"""
    # Check phases are sequential
    # Check severity ranges
    # Check formulas are valid
    pass
```

---

## 10. Usage Examples

### 10.1 Running Single Patient Simulation

```python
# scripts/run_simulation.py

from src.core.simulation import MedicalChatbotSimulation

# Initialize
sim = MedicalChatbotSimulation(config_path="config/experiment_config.yaml")

# Run for one patient, one condition
sim.run_experiment(
    patient_id="patient_001",
    num_days=14,
    condition="with_history"
)

# Results automatically saved to data/conversations/patient_001/
```

### 10.2 Running Full Experiment

```python
# scripts/run_full_experiment.py

from src.core.simulation import MedicalChatbotSimulation
import glob

# Load all patients
patient_files = glob.glob("profiles/patient_*.yaml")
patients = [f.split('/')[-1].split('_')[0] + '_' + f.split('_')[1].split('.')[0] 
            for f in patient_files]

# Initialize simulation
sim = MedicalChatbotSimulation("config/experiment_config.yaml")

# Run all combinations
conditions = ["with_history", "without_history"]

for patient_id in patients:
    for condition in conditions:
        print(f"\n=== Running {patient_id} - {condition} ===")
        sim.run_experiment(
            patient_id=patient_id,
            num_days=14,
            condition=condition
        )

print("\n✓ Experiment complete! Results in data/conversations/")
```

### 10.3 Using in Google Colab

**Notebook**: `notebooks/02_run_experiment.ipynb`

```python
# Cell 1: Clone and setup
!git clone https://github.com/yourusername/medical-chatbot-memory-study.git
%cd medical-chatbot-memory-study
!pip install -r requirements.txt

# Cell 2: Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 3: Load and test
from src.core.simulation import MedicalChatbotSimulation

sim = MedicalChatbotSimulation("config/experiment_config.yaml")

# Cell 4: Run one patient (quick test)
sim.run_experiment(
    patient_id="patient_001",
    num_days=5,  # Shorter for testing
    condition="with_history"
)

# Cell 5: Download results
from google.colab import files

# Zip conversations
!zip -r conversations.zip data/conversations/

# Download
files.download('conversations.zip')
```

---

## 11. Implementation Priority

### Phase 1: Core Infrastructure (Week 1)
1. Set up directory structure
2. Implement data models (PatientProfile, DiseaseModel, Conversation)
3. Implement file I/O utilities
4. Create configuration system
5. Write validation functions

### Phase 2: Disease & Patient Simulation (Week 2)
1. Implement DiseaseProgressionEngine
2. Implement PatientStateManager
3. Implement PatientSimulator (LLM-based)
4. Create prompt templates
5. Test with simple examples

### Phase 3: Chatbot & Conversation (Week 3)
1. Implement BaseChatbot and LocalLLMChatbot
2. Implement ConversationRunner
3. Test patient-chatbot interactions
4. Debug and refine prompts

### Phase 4: Orchestration & Metrics (Week 4)
1. Implement MedicalChatbotSimulation orchestrator
2. Implement MetricsCalculator
3. Create timeline generation
4. Test full day simulation

### Phase 5: Scripts & Notebooks (Week 5)
1. Create patient/disease creation scripts
2. Create main experiment script
3. Create Colab notebooks
4. Write documentation
5. Create example profiles and diseases

### Phase 6: Testing & Validation (Week 6)
1. Write unit tests
2. Write integration tests
3. Run pilot experiments
4. Fix bugs and refine
5. Finalize documentation

---

## 12. Key Design Decisions & Rationale

### 12.1 Why YAML for Profiles?
- Human-readable and editable
- Supports complex nested structures
- Easy to version control
- Can add comments

### 12.2 Why JSON for Runtime Data?
- Standard format
- Easy to parse in any language
- Preserves data types
- Good for structured logs

### 12.3 Why LLM-as-Patient?
- More realistic than hardcoded responses
- Adapts to different chatbot questions
- Can handle unexpected conversation flows
- Scalable to many scenarios

### 12.4 Why Separate Disease Engine?
- Reusable across different patients
- Easy to add new diseases
- Clear separation of concerns
- Testable independently

### 12.5 Why Two Conditions?
- Direct comparison of with/without history
- Controls for other variables
- Clear causal inference
- Simple experimental design

---

## 13. Expected Outputs

After running full experiment, you will have:

1. **Conversation Logs**: JSON files with complete conversations
2. **Metrics CSV**: Quantitative metrics for each conversation
3. **State Snapshots**: Patient state evolution over time
4. **Summary Statistics**: Aggregate results comparing conditions
5. **Evaluation Forms**: Templates for human evaluation
6. **Visualizations**: (Optional) Plots of key metrics

---

## 14. Success Criteria

The implementation is successful when:

1. ✅ Can load patient profiles and disease models from YAML
2. ✅ Disease progression generates realistic symptom evolution
3. ✅ Patient simulator produces natural, in-character responses
4. ✅ Chatbot can run with and without conversation history
5. ✅ Conversations are saved with complete metadata
6. ✅ Automatic metrics are calculated correctly
7. ✅ Full experiment runs on Google Colab without crashes
8. ✅ Results are reproducible with same random seed
9. ✅ Code is documented and tested
10. ✅ Can easily add new patients and diseases

---

## 15. Common Issues & Solutions

### Issue: Out of Memory on Colab
**Solution**: 
- Use 4-bit quantization (already in config)
- Reduce max_tokens
- Use smaller models (7B instead of 13B)
- Clear GPU memory between runs

### Issue: Patient Responses Don't Stay in Character
**Solution**:
- Refine system prompts
- Add more examples in prompt
- Increase temperature slightly (more creativity)
- Add character consistency checks

### Issue: Conversations Too Short/Long
**Solution**:
- Adjust max_turns parameter
- Improve end detection logic
- Add explicit conversation goals to prompts

### Issue: Disease Progression Unrealistic
**Solution**:
- Review disease model YAML
- Adjust severity ranges
- Check modifiers are applied
- Add more variation/randomness

---

## 16. Future Enhancements

1. **LLM-as-Judge Evaluation**: Automated evaluation using GPT-4/Claude
2. **More Diseases**: Add 5-10 more disease models
3. **Complications**: More sophisticated complication models
4. **Multi-turn Memory**: Test longer-term memory (weeks/months)
5. **Real Patient Data**: Validate against real medical conversation data
6. **Interactive Demo**: Web interface for researchers to explore
7. **API Support**: Add GPT-4/Claude API for comparison
8. **Multilingual**: Test in different languages

---

## END OF SPECIFICATION

This document should provide everything needed for an AI code writer to implement the complete system. All components, data formats, algorithms, and workflows are specified in detail.