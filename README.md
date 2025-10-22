# Medical Chatbot Memory Study

A simulation framework to study how conversation history affects medical advice quality in chatbot systems.

## Overview

This project investigates whether chatbots provide better medical advice when they have access to conversation history (memory) versus when each interaction is isolated. The system uses:

- **Virtual Patients**: LLM-powered patient simulators with realistic personalities and symptom progression
- **Disease Models**: Day-by-day disease progression with natural symptom evolution
- **Experimental Design**: Compare chatbot performance WITH vs. WITHOUT conversation history
- **Automated Metrics**: Track conversation quality, information gathering, and advice appropriateness

## Key Features

- Dynamic patient profiles with personality, anxiety levels, and communication styles
- Realistic disease progression models with complications
- LLM-as-patient for natural, adaptive conversations
- Multi-day timelines mixing medical and non-medical conversations
- Comprehensive metrics and evaluation framework

## Project Structure

```
medical-chatbot-memory-study/
├── config/                    # Experiment configuration
├── profiles/                  # Patient profile definitions (YAML)
├── diseases/                  # Disease progression models (YAML)
├── src/                      # Source code
│   ├── core/                 # Core simulation components
│   ├── models/               # Data models and schemas
│   ├── chatbots/             # Chatbot implementations
│   ├── evaluation/           # Metrics and evaluation
│   └── utils/                # Utility functions
├── data/                     # Runtime data (gitignored)
│   ├── conversations/        # Saved conversation logs
│   ├── state/               # Patient state snapshots
│   └── results/             # Evaluation results
├── scripts/                  # Executable scripts
├── notebooks/                # Jupyter notebooks
└── tests/                    # Unit and integration tests
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for local LLMs)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd medical-chatbot-memory-study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run a Simple Simulation

```python
from src.core.simulation import MedicalChatbotSimulation

# Initialize simulation
sim = MedicalChatbotSimulation(config_path="config/experiment_config.yaml")

# Run experiment for one patient
sim.run_experiment(
    patient_id="patient_001",
    num_days=14,
    condition="with_history"
)
```

### 2. Create a Patient Profile

```bash
python scripts/create_patient.py
```

### 3. Create a Disease Model

```bash
python scripts/create_disease.py
```

## Using Google Colab

For users without local GPU resources, see [notebooks/02_run_experiment.ipynb](notebooks/02_run_experiment.ipynb) for a complete Colab-ready implementation.

## Components

### Patient Simulator
LLM-based patient that exhibits realistic:
- Communication styles (formal vs. casual)
- Anxiety levels affecting symptom reporting
- Information volunteering behavior
- Natural conversational patterns

### Disease Progression Engine
Simulates realistic disease evolution with:
- Multiple phases (incubation, prodrome, acute, resolution)
- Dynamic symptom severity trajectories
- Patient-specific modifiers (stress, sleep, immune status)
- Probabilistic complications

### Conversation Runner
Orchestrates patient-chatbot interactions:
- Manages turn-by-turn conversation flow
- Tracks real-time metrics
- Determines natural conversation endpoints
- Saves comprehensive conversation logs

### Metrics Calculator
Automatically evaluates:
- Information gathering completeness
- Redundant questions (especially important for WITH history condition)
- References to previous conversations
- Advice quality and appropriateness

## Data Formats

- **Patient Profiles**: YAML files with demographics, personality, medical history
- **Disease Models**: YAML files with symptom trajectories and phases
- **Conversations**: JSON logs with complete turn-by-turn data
- **Metrics**: CSV files for statistical analysis

## Research Applications

This framework enables research into:
- Impact of conversation memory on medical advice quality
- Chatbot information gathering strategies
- Patient communication patterns and their effects
- Comparison of different LLM architectures for medical conversations

## Development Status

Current implementation phase: **Phase 1 - Core Infrastructure**

See [Initial_Setup.md](Initial_Setup.md) for complete technical specification.

## Contributing

Contributions welcome! Please see development roadmap in the technical specification document.

## License

[Add your license here]

## Citation

[Add citation information if this becomes a published research project]

## Contact

[Add contact information]
