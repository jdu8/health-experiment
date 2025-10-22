# LLM Integration Guide

## Overview

The system now supports **actual LLM integration** for both the Patient Simulator and Medical Chatbot. This document explains how the LLM system works and how to use it.

## Architecture

### Two LLMs Working Together

1. **Patient LLM** (`PatientSimulator`)
   - Simulates realistic patient behavior
   - Uses personality, anxiety, and symptom data to generate responses
   - Makes conversations dynamic and adaptive

2. **Doctor/Chatbot LLM** (`LocalLLMChatbot`)
   - The medical chatbot being evaluated
   - Can access conversation history (WITH condition) or not (WITHOUT condition)
   - Provides medical advice and asks questions

### Fallback System

- **With LLMs**: Uses transformers to load and run models
- **Without LLMs**: Falls back to rule-based responses
- **Automatic**: System tries to load LLMs, falls back if unavailable

This allows development on laptops WITHOUT LLMs, then deployment to Google Colab WITH LLMs.

## Configuration

### Model Settings (`config/experiment_config.yaml`)

```yaml
models:
  patient_llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"  # HuggingFace model
    device: "auto"  # 'cuda', 'cpu', or 'auto'
    temperature: 0.7
    max_tokens: 512
    quantization: "4bit"  # '4bit', '8bit', or null

  doctor_llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
    device: "auto"
    temperature: 0.7
    max_tokens: 512
    quantization: "4bit"
```

### Model Options

**Recommended Models:**
- **Mistral-7B-Instruct-v0.2**: Good balance of quality and speed
- **Llama-2-7B-chat**: Alternative, similar performance
- **GPT-2** (for testing): Very small, fast, lower quality

**Quantization:**
- **4-bit**: ~3.5 GB GPU memory per model, minimal quality loss
- **8-bit**: ~7 GB GPU memory, slightly better quality
- **None**: ~14 GB GPU memory, full precision

## Running with LLMs

### Option 1: Google Colab (Recommended)

**Why Colab?**
- Free GPU access (T4 with 15GB VRAM)
- No local setup needed
- Perfect for running experiments

**Steps:**
1. Open `notebooks/02_run_experiment_colab.ipynb` in Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → GPU)
3. Run all cells
4. Wait ~10 minutes for model download (first time only)
5. Experiment runs automatically

**Memory Requirements:**
- 2 models × 3.5GB (4-bit) = ~7GB GPU memory
- T4 GPU (15GB) has plenty of headroom
- Can run full 7-day experiments

### Option 2: Local with GPU

**Requirements:**
- NVIDIA GPU with 8GB+ VRAM
- CUDA installed
- Python 3.9+

**Installation:**
```bash
# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and quantization
pip3 install transformers>=4.35.0
pip3 install accelerate>=0.24.0
pip3 install bitsandbytes>=0.41.0
pip3 install sentencepiece>=0.1.99
```

**Run:**
```python
from src.core.simulation import MedicalChatbotSimulation

sim = MedicalChatbotSimulation(config_path="config/experiment_config.yaml")
results = sim.run_experiment(
    patient_id="patient_001",
    num_days=7,
    condition="with_history"
)
```

### Option 3: Local CPU Only (Not Recommended)

LLMs run VERY slowly on CPU (minutes per response). Only use for testing:

```yaml
models:
  patient_llm:
    model_name: "gpt2"  # Use small model
    device: "cpu"
    quantization: null  # No quantization on CPU
```

## LLM System Components

### 1. LLM Utilities (`src/utils/llm_utils.py`)

Core functions for model management:

```python
from src.utils.llm_utils import (
    load_model_and_tokenizer,  # Load models
    generate_response,          # Generate text
    clear_model_cache,          # Free memory
    check_cuda_available,       # Check GPU
    test_model_loading          # Test system
)
```

**Features:**
- Model caching (loads once, reuses)
- Automatic device selection
- 4-bit/8-bit quantization support
- Error handling with fallbacks

### 2. Patient Simulator (`src/core/patient_simulator.py`)

**LLM Integration:**
```python
simulator = PatientSimulator(
    patient_profile=patient,
    current_state=state,
    llm_model="mistralai/Mistral-7B-Instruct-v0.2",
    device="auto",
    quantization="4bit"
)
```

**How it works:**
1. Creates detailed system prompt with personality, symptoms, anxiety
2. Formats conversation history
3. Generates natural patient responses using LLM
4. Falls back to rule-based if LLM unavailable

**System Prompt Example:**
```
You are roleplaying as a patient seeking medical advice.

=== YOUR IDENTITY ===
Name: Alex Chen
Age: 34, male
Occupation: high school teacher

=== YOUR PERSONALITY ===
Communication style: casual (formality 6/10)
Anxiety: High health anxiety (8/10)
Current mood: stressed

=== YOUR CURRENT SITUATION ===
Day 5 of feeling unwell
Current symptoms:
- Sore throat (severity 6/10): "throat feels raw"
- Fever (severity 5/10): "feeling hot and shitty"
- Fatigue (severity 4/10): "exhausted"

[... detailed instructions on how to respond ...]
```

### 3. Chatbot (`src/chatbots/local_llm_chatbot.py`)

**LLM Integration:**
```python
chatbot = LocalLLMChatbot(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    context=conversation_history,  # For WITH history condition
    condition="with_history",
    device="auto",
    quantization="4bit"
)
```

**System Prompt:**
```
You are a helpful medical chatbot providing health advice.

Guidelines:
- Ask relevant questions to understand symptoms
- Provide appropriate medical advice
- Be empathetic and professional
- Recognize when to recommend seeing a doctor

[WITH HISTORY condition includes previous conversations]
```

## Testing LLMs

### Quick Test (No Model Download)

```python
from src.utils.llm_utils import test_model_loading

# Test with tiny GPT-2 model
success = test_model_loading("gpt2")
print(f"LLM system working: {success}")
```

### Check GPU Availability

```python
from src.utils.llm_utils import check_cuda_available, get_device

has_gpu = check_cuda_available()
device = get_device()

print(f"GPU available: {has_gpu}")
print(f"Using device: {device}")
```

### Memory Estimation

```python
from src.utils.llm_utils import estimate_model_memory

memory = estimate_model_memory(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization="4bit"
)

print(f"Model size: {memory['quantized_size_gb']:.2f} GB")
print(f"Recommended VRAM: {memory['recommended_vram_gb']:.2f} GB")
```

## Performance

### With LLMs (Google Colab T4 GPU)

- **Model loading**: 5-10 minutes (first time, then cached)
- **Per conversation**: ~2-3 minutes (10 turns)
- **Full 7-day experiment**: ~30-45 minutes
- **Memory usage**: ~7 GB GPU memory (2 models)

### Without LLMs (Rule-Based Fallback)

- **Model loading**: Instant
- **Per conversation**: <1 second
- **Full 7-day experiment**: ~30 seconds
- **Memory usage**: Minimal

## Troubleshooting

### "Out of Memory" Error

**Problem:** GPU runs out of memory

**Solutions:**
1. Use 4-bit quantization (already default)
2. Use smaller model: `"gpt2"` or `"mistral-7b"` instead of larger models
3. Clear cache between experiments:
   ```python
   from src.utils.llm_utils import clear_model_cache
   clear_model_cache()
   ```
4. Restart Colab runtime

### "Transformers not found"

**Problem:** transformers library not installed

**Solution:**
```bash
pip install transformers accelerate bitsandbytes sentencepiece
```

### Models Download Slowly

**Problem:** First-time model download is slow

**Solution:**
- This is normal (7B models are ~15GB)
- Only happens once, then cached
- Use Colab's fast internet connection

### LLM Responses Are Gibberish

**Problem:** Model generating nonsense

**Solutions:**
1. Check temperature (0.7 is good, >1.0 can be chaotic)
2. Ensure using instruct-tuned model (ends with `-Instruct` or `-chat`)
3. Check system prompt is properly formatted

### Falling Back to Rule-Based

**Problem:** System uses rule-based despite LLM specified

**Check logs:**
```python
import logging
logging.basicConfig(level=logging.INFO)
# Run simulation and check for warnings
```

**Common causes:**
- Transformers not installed
- Model name typo
- Insufficient GPU memory
- Model download failed

## Model Recommendations

### For Research (High Quality)

```yaml
models:
  patient_llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  doctor_llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
```

### For Testing (Fast)

```yaml
models:
  patient_llm:
    model_name: "gpt2"
  doctor_llm:
    model_name: "gpt2"
```

### For Low Memory

```yaml
models:
  patient_llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
    quantization: "4bit"
  doctor_llm:
    model_name: null  # Use rule-based for chatbot
```

## API Alternative

Instead of local LLMs, you can use OpenAI/Anthropic APIs:

```python
# Not yet implemented, but planned for future
# Would allow using GPT-4, Claude, etc.
```

## Next Steps

1. **Run Quick Test**: Open Colab notebook, test with 1 day
2. **Verify Quality**: Check if conversations look natural
3. **Run Full Experiment**: 7-14 days, both conditions
4. **Analyze Results**: Compare with_history vs without_history
5. **Iterate**: Adjust models, prompts, parameters as needed

## Summary

✅ **LLMs are integrated** - Patient and chatbot both use real models
✅ **Automatic fallback** - Works without LLMs for development
✅ **Colab-ready** - Optimized for Google Colab with GPU
✅ **Memory efficient** - 4-bit quantization uses ~7GB total
✅ **Production-ready** - Can run full experiments now

**The system is ready to generate real research data!**
