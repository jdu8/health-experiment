#!/usr/bin/env python3
"""
Google Colab Experiment Runner

Run this script in Google Colab to execute the full experiment with LLMs.

Usage:
    python scripts/run_colab_experiment.py [--quick-test]
"""

import sys
import os
from pathlib import Path
import argparse

# CRITICAL: Add project root to Python path
# This must happen before any src imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✓ Added to Python path: {project_root}\n")

# Change to project root
os.chdir(project_root)
print(f"✓ Working directory: {Path.cwd()}\n")

# Now we can import from src
try:
    from src.core.simulation import MedicalChatbotSimulation
    from src.utils.llm_utils import check_cuda_available, test_model_loading
    print("✓ Successfully imported src modules\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"\nCurrent directory: {Path.cwd()}")
    print(f"Python path: {sys.path[:3]}")
    print(f"Does 'src' exist? {(Path.cwd() / 'src').exists()}")
    sys.exit(1)


def check_environment():
    """Check GPU and environment"""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    try:
        import torch
        print(f"Python version: {sys.version.split()[0]}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("\n✓ GPU detected - ready for LLMs!")
        else:
            print("\n⚠️  WARNING: No GPU detected!")
            print("   LLMs will be VERY slow on CPU.")
            print("   In Colab: Runtime → Change runtime type → GPU")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    except ImportError:
        print("⚠️  PyTorch not installed")
        print("Installing PyTorch...")
        os.system("pip install -q torch torchvision torchaudio")


def install_dependencies():
    """Install required packages"""
    print("\n" + "=" * 70)
    print("INSTALLING DEPENDENCIES")
    print("=" * 70)

    packages = [
        "torch torchvision torchaudio",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "pyyaml numpy pandas python-dateutil tqdm"
    ]

    for package in packages:
        print(f"\nInstalling {package.split()[0]}...")
        os.system(f"pip install -q {package}")

    print("\n✓ All dependencies installed")


def test_llm_loading():
    """Test LLM system with small model"""
    print("\n" + "=" * 70)
    print("TESTING LLM SYSTEM")
    print("=" * 70)
    print("Loading small GPT-2 model for testing...")
    print("(This verifies transformers is working)\n")

    success = test_model_loading("gpt2")

    if success:
        print("\n✓ LLM system working!")
    else:
        print("\n✗ LLM test failed")
        print("  Continuing anyway - will use rule-based fallback")

    return success


def run_quick_test(sim):
    """Run quick 1-day test"""
    print("\n" + "=" * 70)
    print("QUICK TEST: 1 day, both conditions")
    print("=" * 70)

    # Test WITHOUT history
    print("\n--- Test 1: WITHOUT history ---")
    results_without = sim.run_experiment(
        patient_id="patient_001",
        num_days=1,
        condition="without_history",
        random_seed=42
    )

    print(f"\n✓ Test 1 complete:")
    print(f"  Conversations: {results_without['total_conversations']}")
    print(f"  Total turns: {results_without['total_turns']}")

    # Test WITH history
    print("\n--- Test 2: WITH history ---")
    results_with = sim.run_experiment(
        patient_id="patient_001",
        num_days=1,
        condition="with_history",
        random_seed=43
    )

    print(f"\n✓ Test 2 complete:")
    print(f"  Conversations: {results_with['total_conversations']}")
    print(f"  Total turns: {results_with['total_turns']}")

    return True


def run_full_experiment(sim):
    """Run full 7-day experiment"""
    print("\n" + "=" * 70)
    print("FULL EXPERIMENT: 7 days, both conditions")
    print("=" * 70)
    print("This will take approximately 30-45 minutes with LLMs...")
    print("(or ~30 seconds with rule-based fallback)\n")

    # Run full experiment
    results = sim.run_full_experiment(
        patient_ids=["patient_001"],
        conditions=["without_history", "with_history"],
        num_days=7
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)

    successful = [r for r in results['results'] if 'error' not in r]
    failed = [r for r in results['results'] if 'error' in r]

    print(f"Successful: {len(successful)}/{len(results['results'])}")
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['patient_id']} ({r['condition']}): {r.get('error', 'unknown error')}")

    # Show results
    for result in successful:
        print(f"\n{result['patient_id']} - {result['condition']}:")
        print(f"  Conversations: {result['total_conversations']}")
        print(f"  Total turns: {result['total_turns']}")

    return results


def show_sample_conversation():
    """Display a sample conversation"""
    import json

    print("\n" + "=" * 70)
    print("SAMPLE CONVERSATION")
    print("=" * 70)

    conv_dir = Path("data/conversations/patient_001/without_history")
    if not conv_dir.exists():
        print("No conversations found yet")
        return

    conv_files = sorted(conv_dir.glob("day_*.json"))
    if not conv_files:
        print("No conversation files found")
        return

    # Load first conversation
    with open(conv_files[0], 'r') as f:
        conv = json.load(f)

    print(f"Conversation ID: {conv['conversation_id']}")
    print(f"Day: {conv['simulation_day']}")
    print(f"Turns: {len(conv['turns'])}\n")
    print("-" * 70)

    # Show conversation (limit to 10 turns for display)
    for i, turn in enumerate(conv['turns'][:10], 1):
        speaker = "PATIENT" if turn['speaker'] == 'patient' else "CHATBOT"
        print(f"\n[{i}] {speaker}:")
        print(f"{turn['message']}")

    if len(conv['turns']) > 10:
        print(f"\n... ({len(conv['turns']) - 10} more turns)")


def main():
    parser = argparse.ArgumentParser(description="Run Colab experiment with LLMs")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick 1-day test instead of full experiment")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip LLM loading test")
    args = parser.parse_args()

    print("=" * 70)
    print("MEDICAL CHATBOT MEMORY STUDY - COLAB RUNNER")
    print("=" * 70)

    # Check environment
    check_environment()

    # Install dependencies
    if not args.skip_install:
        install_dependencies()

    # Test LLM system
    if not args.skip_test:
        test_llm_loading()

    # Initialize simulation
    print("\n" + "=" * 70)
    print("INITIALIZING SIMULATION")
    print("=" * 70)
    print("Loading models from config...")
    print("(First time: downloads models, may take 5-10 minutes)\n")

    sim = MedicalChatbotSimulation(
        config_path="config/experiment_config.yaml",
        random_seed=42
    )

    print(f"\n✓ Simulation initialized!")
    print(f"  Patient model: {sim.patient_model_name or 'rule-based'}")
    print(f"  Doctor model: {sim.doctor_model_name or 'rule-based'}")

    # Run experiment
    if args.quick_test:
        run_quick_test(sim)
    else:
        run_full_experiment(sim)

    # Show sample
    show_sample_conversation()

    # Done
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print("\nResults saved to:")
    print(f"  - Conversations: data/conversations/")
    print(f"  - State files: data/state/")
    print(f"  - Summaries: data/results/")

    print("\nTo download results from Colab:")
    print("  from google.colab import files")
    print("  import shutil")
    print("  shutil.make_archive('results', 'zip', 'data')")
    print("  files.download('results.zip')")


if __name__ == "__main__":
    main()
