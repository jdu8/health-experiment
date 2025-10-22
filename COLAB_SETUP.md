# Google Colab Setup Fix

## The Import Error You're Seeing

```
ModuleNotFoundError: No module named 'src.models'
```

This happens because Python can't find the `src` module after cloning the repo.

## Quick Fix for Colab Notebook

**Add this cell RIGHT AFTER cloning the repo and changing directory:**

```python
# === FIX PYTHON PATH ===
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verify it worked
print(f"✓ Current directory: {project_root}")
print(f"✓ Added to Python path")

# Check that src directory exists
src_dir = project_root / "src"
if src_dir.exists():
    print(f"✓ Found 'src' directory: {src_dir}")
else:
    print(f"✗ ERROR: 'src' not found!")
    print(f"   Are you in the right directory?")
```

**Then your imports should work:**

```python
from src.core.simulation import MedicalChatbotSimulation
# ✓ Should work now!
```

## Complete Working Colab Cell Sequence

Here's the exact sequence that works:

### Cell 1: Clone and Setup Path
```python
# Clone repo
!git clone https://github.com/yourusername/health-experiment.git
%cd health-experiment

# CRITICAL: Fix Python path
import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

print(f"✓ Setup complete")
print(f"  Directory: {project_root}")
print(f"  src exists: {(project_root / 'src').exists()}")
```

### Cell 2: Install Dependencies
```python
!pip install -q torch torchvision torchaudio
!pip install -q transformers>=4.35.0 accelerate>=0.24.0
!pip install -q bitsandbytes>=0.41.0 sentencepiece>=0.1.99
!pip install -q pyyaml numpy pandas python-dateutil tqdm

print("✓ Dependencies installed")
```

### Cell 3: Verify Environment
```python
import torch

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU! Go to Runtime → Change runtime type → GPU")
```

### Cell 4: Test Import (Should Work Now!)
```python
from src.core.simulation import MedicalChatbotSimulation

print("✓ Import successful!")

# Initialize simulation
sim = MedicalChatbotSimulation(
    config_path="config/experiment_config.yaml",
    random_seed=42
)

print(f"✓ Simulation initialized")
print(f"  Patient LLM: {sim.patient_model_name}")
print(f"  Doctor LLM: {sim.doctor_model_name}")
```

## Alternative: Use the Python Script

Instead of the notebook, you can run the Python script which handles paths automatically:

```python
# After cloning repo and installing dependencies:
!python scripts/run_colab_experiment.py --quick-test
```

Or for full experiment:
```python
!python scripts/run_colab_experiment.py
```

## Why This Happens

1. When you `%cd health-experiment`, you change the working directory
2. But Python doesn't automatically add that directory to `sys.path`
3. So when you do `from src.core...`, Python can't find `src`
4. Adding the current directory to `sys.path` fixes it

## Debugging Checklist

If imports still fail, check:

```python
import sys
from pathlib import Path

print("Working directory:", Path.cwd())
print("Python path (first 3):", sys.path[:3])
print("src exists?", (Path.cwd() / "src").exists())
print("src/core exists?", (Path.cwd() / "src" / "core").exists())
print("simulation.py exists?", (Path.cwd() / "src" / "core" / "simulation.py").exists())
```

All of these should return `True` or show the correct paths.

## Summary

**The fix is simple:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

**Add this after `%cd health-experiment` and before any `from src...` imports.**

That's it! Your imports should work now.
