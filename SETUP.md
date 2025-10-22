# Setup Guide

## Quick Start

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install pyyaml numpy pandas python-dateutil
```

For full installation (including LLM support):
```bash
pip install -r requirements.txt
```

Note: Installing all dependencies including PyTorch and transformers can take significant time and disk space. For Phase 1 testing, only the basic dependencies above are needed.

### 3. Test Phase 1

```bash
python scripts/test_phase1.py
```

## Phase 1 Components

Phase 1 (Core Infrastructure) includes:

- ✅ Data Models
  - PatientProfile
  - DiseaseModel
  - Conversation
  - PatientState

- ✅ Utilities
  - file_utils.py (YAML/JSON I/O)
  - prompt_templates.py (LLM prompts)
  - logging_config.py (Logging setup)

- ✅ Templates & Examples
  - Template patient profile
  - Template disease model
  - Example patient (Alex Chen)
  - Example disease (Viral URI)

## Next Steps

After confirming Phase 1 works:
- **Phase 2**: Disease & Patient Simulation (disease_engine.py, state_manager.py, patient_simulator.py)
- **Phase 3**: Chatbot & Conversation (chatbots, conversation_runner.py)
- **Phase 4**: Orchestration & Metrics (simulation.py, metrics_calculator.py)
- **Phase 5**: Scripts & Notebooks

## File Structure

```
health-experiment/
├── config/
│   └── experiment_config.yaml       ✅
├── profiles/
│   ├── template_patient.yaml        ✅
│   └── patient_001_alex_chen.yaml   ✅
├── diseases/
│   ├── template_disease.yaml        ✅
│   └── viral_uri.yaml               ✅
├── src/
│   ├── models/
│   │   ├── patient_profile.py       ✅
│   │   ├── disease_model.py         ✅
│   │   ├── conversation.py          ✅
│   │   └── patient_state.py         ✅
│   └── utils/
│       ├── file_utils.py            ✅
│       ├── prompt_templates.py      ✅
│       └── logging_config.py        ✅
└── scripts/
    └── test_phase1.py               ✅
```
