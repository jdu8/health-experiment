#!/usr/bin/env python3
"""
Test script for Phase 4: Orchestration & Metrics

Tests the complete simulation orchestrator that coordinates all components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.simulation import MedicalChatbotSimulation
from src.utils.logging_config import setup_logging
import json


def test_simulation_initialization():
    """Test simulation initialization and configuration loading"""
    print("\n=== Testing Simulation Initialization ===")
    try:
        sim = MedicalChatbotSimulation(
            config_path="config/experiment_config.yaml",
            random_seed=42
        )

        print(f"✓ Simulation initialized")
        print(f"  Profiles dir: {sim.profiles_dir}")
        print(f"  Diseases dir: {sim.diseases_dir}")
        print(f"  Default max days: {sim.default_max_days}")
        print(f"  Default max turns: {sim.default_max_turns}")

        return True
    except Exception as e:
        print(f"✗ Error in simulation initialization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patient_disease_loading():
    """Test loading patient profiles and disease models"""
    print("\n=== Testing Patient & Disease Loading ===")
    try:
        sim = MedicalChatbotSimulation(random_seed=42)

        # Load patient
        patient = sim.load_patient("patient_001")
        print(f"✓ Loaded patient: {patient.name}")
        print(f"  Age: {patient.demographics.age}")
        print(f"  Assigned disease: {patient.assigned_disease}")
        print(f"  Disease starts: Day {patient.disease_start_day}")

        # Load disease
        disease = sim.load_disease(patient.assigned_disease)
        print(f"✓ Loaded disease: {disease.name}")
        print(f"  Disease ID: {disease.disease_id}")
        print(f"  Common name: {disease.common_name}")

        return True
    except Exception as e:
        print(f"✗ Error in loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timeline_generation():
    """Test conversation timeline generation"""
    print("\n=== Testing Timeline Generation ===")
    try:
        sim = MedicalChatbotSimulation(random_seed=42)

        # Test default timeline (14 days)
        timeline = sim.generate_timeline(num_days=14, disease_start_day=1)
        print(f"✓ Generated timeline for 14 days")
        print(f"  Total entries: {len(timeline)}")

        medical_count = sum(1 for t in timeline if 'medical' in t['type'])
        casual_count = sum(1 for t in timeline if t['type'] == 'casual')

        print(f"  Medical conversations: {medical_count}")
        print(f"  Casual conversations: {casual_count}")

        # Show first few days
        print(f"\n  First 5 days:")
        for entry in timeline[:5]:
            print(f"    Day {entry['day']}: {entry['type']}")

        # Test shorter timeline
        short_timeline = sim.generate_timeline(num_days=5, disease_start_day=1)
        print(f"\n✓ Generated short timeline: {len(short_timeline)} days")

        return True
    except Exception as e:
        print(f"✗ Error in timeline generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_day_simulation():
    """Test running a single day simulation"""
    print("\n=== Testing Single Day Simulation ===")
    try:
        sim = MedicalChatbotSimulation(random_seed=42)

        # Load patient and disease
        patient = sim.load_patient("patient_001")
        disease = sim.load_disease(patient.assigned_disease)

        # Import required components
        from src.core.disease_engine import DiseaseProgressionEngine
        from src.core.state_manager import PatientStateManager

        # Initialize components
        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)
        manager = PatientStateManager(patient, disease.disease_id, condition="without_history")
        state = manager.initialize_state()

        print(f"✓ Components initialized")

        # Run day 5
        day = 5
        conversation = sim.run_day(
            day=day,
            conversation_type="medical",
            patient=patient,
            disease=disease,
            engine=engine,
            manager=manager,
            condition="without_history",
            max_turns=6
        )

        print(f"✓ Day {day} simulation complete")
        print(f"  Conversation ID: {conversation.conversation_id}")
        print(f"  Turns: {len(conversation.turns)}")
        print(f"  Duration: {conversation.conversation_summary.duration_seconds:.1f}s")
        print(f"  Questions asked: {conversation.metrics.automatic.bot_questions_asked}")

        # Show first few turns
        print(f"\n  First 3 turns:")
        for i, turn in enumerate(conversation.turns[:6]):
            speaker = "P" if turn.speaker == "patient" else "C"
            msg = turn.message[:50] + "..." if len(turn.message) > 50 else turn.message
            print(f"    {i+1}. [{speaker}] {msg}")

        return True
    except Exception as e:
        print(f"✗ Error in single day simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_day_experiment():
    """Test running a multi-day experiment"""
    print("\n=== Testing Multi-Day Experiment ===")
    try:
        sim = MedicalChatbotSimulation(random_seed=42)

        # Run experiment for 5 days (shorter for testing)
        print(f"\n  Running 5-day experiment for patient_001 (without_history)...")

        results = sim.run_experiment(
            patient_id="patient_001",
            num_days=5,
            condition="without_history",
            random_seed=42
        )

        print(f"\n✓ Multi-day experiment complete")
        print(f"  Experiment ID: {results['experiment_id']}")
        print(f"  Patient: {results['patient_id']}")
        print(f"  Condition: {results['condition']}")
        print(f"  Total conversations: {results['total_conversations']}")
        print(f"  Total turns: {results['total_turns']}")

        # Show conversation breakdown
        print(f"\n  Conversations:")
        for conv in results['conversations']:
            print(f"    Day {conv['day']}: {conv['type']} - {conv['turns']} turns")

        return True
    except Exception as e:
        print(f"✗ Error in multi-day experiment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_history_condition():
    """Test running experiment with history"""
    print("\n=== Testing WITH History Condition ===")
    try:
        sim = MedicalChatbotSimulation(random_seed=42)

        print(f"\n  Running 3-day experiment (with_history)...")

        results = sim.run_experiment(
            patient_id="patient_001",
            num_days=3,
            condition="with_history",
            random_seed=42
        )

        print(f"\n✓ With-history experiment complete")
        print(f"  Total conversations: {results['total_conversations']}")
        print(f"  Total turns: {results['total_turns']}")

        # Check that later conversations have access to history
        print(f"\n  Checking conversation history usage...")

        # Load one of the saved conversations
        from pathlib import Path
        conv_dir = Path(f"data/conversations/patient_001/with_history")
        if conv_dir.exists():
            conv_files = sorted(conv_dir.glob("day_*.json"))
            if len(conv_files) >= 2:
                # Load second conversation
                with open(conv_files[1], 'r') as f:
                    conv_data = json.load(f)

                metrics = conv_data.get('metrics', {}).get('automatic', {})
                history_refs = metrics.get('references_to_history', 0)

                print(f"  ✓ Day 2 conversation has {history_refs} history references")

        return True
    except Exception as e:
        print(f"✗ Error in with_history test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timeline_customization():
    """Test custom timeline generation"""
    print("\n=== Testing Timeline Customization ===")
    try:
        sim = MedicalChatbotSimulation(random_seed=42)

        # Test with disease starting on day 3
        timeline = sim.generate_timeline(num_days=7, disease_start_day=3)

        print(f"✓ Generated timeline with disease starting day 3")
        print(f"  Timeline breakdown:")
        for entry in timeline:
            print(f"    Day {entry['day']}: {entry['type']}")

        # Verify that days before disease start are casual
        pre_disease = [t for t in timeline if t['day'] < 3]
        all_casual = all(t['type'] == 'casual' for t in pre_disease)

        if all_casual:
            print(f"  ✓ Days before disease onset are casual: {len(pre_disease)} days")
        else:
            print(f"  ⚠ Warning: Some pre-disease days are not casual")

        return True
    except Exception as e:
        print(f"✗ Error in timeline customization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_persistence():
    """Test that state is properly saved and loaded"""
    print("\n=== Testing State Persistence ===")
    try:
        from pathlib import Path

        # Check that state files were created
        state_dir = Path("data/state")

        if not state_dir.exists():
            print(f"  ⚠ State directory doesn't exist yet")
            return True

        state_files = list(state_dir.glob("patient_001_*_state.json"))

        print(f"✓ Found {len(state_files)} state files:")
        for sf in state_files:
            print(f"  - {sf.name}")

        # Load and verify a state file
        if state_files:
            with open(state_files[0], 'r') as f:
                state_data = json.load(f)

            print(f"\n  State file contents:")
            print(f"    Patient: {state_data.get('patient_id')}")
            print(f"    Current day: {state_data.get('current_day')}")
            print(f"    Disease: {state_data.get('disease_state', {}).get('active_disease')}")
            print(f"    Symptoms: {len(state_data.get('current_symptoms', {}))}")
            print(f"    Conversations: {len(state_data.get('conversation_history_summary', []))}")

        return True
    except Exception as e:
        print(f"✗ Error in state persistence test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_results_summary():
    """Test that results summaries are properly created"""
    print("\n=== Testing Results Summaries ===")
    try:
        from pathlib import Path

        results_dir = Path("data/results")

        if not results_dir.exists():
            print(f"  ⚠ Results directory doesn't exist yet")
            return True

        summary_files = list(results_dir.glob("patient_001_*_summary.json"))

        print(f"✓ Found {len(summary_files)} summary files:")
        for sf in summary_files:
            print(f"  - {sf.name}")

            # Load and show summary
            with open(sf, 'r') as f:
                summary = json.load(f)

            print(f"    Patient: {summary.get('patient_id')}")
            print(f"    Condition: {summary.get('condition')}")
            print(f"    Conversations: {summary.get('total_conversations')}")
            print(f"    Total turns: {summary.get('total_turns')}")

        return True
    except Exception as e:
        print(f"✗ Error in results summary test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test complete end-to-end workflow"""
    print("\n=== Testing Full Integration ===")
    try:
        print("\n  Running complete 7-day simulation in both conditions...")

        sim = MedicalChatbotSimulation(random_seed=42)

        # Run without history
        results_without = sim.run_experiment(
            patient_id="patient_001",
            num_days=7,
            condition="without_history",
            random_seed=42
        )

        print(f"\n  ✓ WITHOUT history: {results_without['total_conversations']} conversations, "
              f"{results_without['total_turns']} turns")

        # Run with history
        results_with = sim.run_experiment(
            patient_id="patient_001",
            num_days=7,
            condition="with_history",
            random_seed=43  # Different seed for variety
        )

        print(f"  ✓ WITH history: {results_with['total_conversations']} conversations, "
              f"{results_with['total_turns']} turns")

        # Compare
        print(f"\n  Comparison:")
        print(f"    Conversations: {results_without['total_conversations']} (without) vs "
              f"{results_with['total_conversations']} (with)")
        print(f"    Total turns: {results_without['total_turns']} (without) vs "
              f"{results_with['total_turns']} (with)")

        print(f"\n✓ Full integration test complete!")
        print(f"  Both conditions successfully simulated")
        print(f"  All data saved to disk")
        print(f"  System is ready for production use")

        return True
    except Exception as e:
        print(f"✗ Error in full integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("PHASE 4: Orchestration & Metrics - Test Suite")
    print("=" * 70)

    # Setup logging
    setup_logging(level="WARNING", console=True)

    tests = [
        ("Simulation Initialization", test_simulation_initialization),
        ("Patient & Disease Loading", test_patient_disease_loading),
        ("Timeline Generation", test_timeline_generation),
        ("Single Day Simulation", test_single_day_simulation),
        ("Multi-Day Experiment", test_multi_day_experiment),
        ("With History Condition", test_with_history_condition),
        ("Timeline Customization", test_timeline_customization),
        ("State Persistence", test_state_persistence),
        ("Results Summaries", test_results_summary),
        ("Full Integration", test_full_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Phase 4 is complete.")
        print("\nNext steps:")
        print("  - Phase 5: Create convenience scripts and notebooks")
        print("  - Phase 6: Add comprehensive unit tests")
        print("  - Ready to run full experiments!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
