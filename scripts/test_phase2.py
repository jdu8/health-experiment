#!/usr/bin/env python3
"""
Test script for Phase 2: Disease & Patient Simulation

Tests disease progression, state management, and patient simulation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.patient_profile import PatientProfile
from src.models.disease_model import DiseaseModel
from src.core.disease_engine import DiseaseProgressionEngine
from src.core.state_manager import PatientStateManager
from src.core.patient_simulator import PatientSimulator
from src.utils.logging_config import setup_logging


def test_disease_progression_engine():
    """Test disease progression engine"""
    print("\n=== Testing Disease Progression Engine ===")
    try:
        # Load patient and disease
        patient = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")

        # Create engine
        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)
        print(f"✓ Created disease progression engine")

        # Test progression over several days
        print("\nTesting symptom progression:")
        for day in [1, 3, 5, 7, 10]:
            symptoms = engine.get_symptoms_for_day(day)
            print(f"\n  Day {day}:")
            print(f"    Phase: {disease.get_phase_for_day(day).name if disease.get_phase_for_day(day) else 'None'}")
            print(f"    Symptoms ({len(symptoms)}):")
            for name, data in symptoms.items():
                print(f"      - {name}: severity {data['objective_severity']}/10")
                print(f"        '{data['description']}'")

        # Test expected resolution
        resolution_day = engine.get_expected_resolution_day()
        print(f"\n✓ Expected resolution day: {resolution_day}")

        # Test complication checking
        print(f"✓ Complications triggered: {engine.has_complications()}")

        return True
    except Exception as e:
        print(f"✗ Error in disease progression engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patient_state_manager():
    """Test patient state manager"""
    print("\n=== Testing Patient State Manager ===")
    try:
        # Load patient and disease
        patient = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")

        # Create manager
        manager = PatientStateManager(patient, disease.disease_id)
        print(f"✓ Created state manager")

        # Initialize state
        state = manager.initialize_state()
        print(f"✓ Initialized state")
        print(f"  {state.get_summary()}")

        # Create disease engine
        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)

        # Simulate progression
        print("\nSimulating state updates:")
        for day in [1, 3, 5]:
            symptoms = engine.get_symptoms_for_day(day)
            phase = disease.get_phase_for_day(day)
            resolution_day = engine.get_expected_resolution_day()

            state = manager.update_state(
                day=day,
                disease_symptoms=symptoms,
                current_phase=phase.name if phase else "unknown",
                expected_resolution_day=resolution_day
            )

            print(f"\n  Day {day}:")
            print(f"    Symptoms: {len(state.current_symptoms)}")
            print(f"    Temperature: {state.biological.temperature}°C")
            print(f"    Heart Rate: {state.biological.heart_rate} bpm")
            print(f"    Anxiety: {state.psychological.anxiety_level}/10")
            print(f"    Mood: {state.psychological.mood}")
            print(f"    Trajectory: {state.disease_state.trajectory}")

        # Test save/load
        manager.save_state()
        print(f"\n✓ Saved state to file")

        loaded_state = manager.load_state()
        print(f"✓ Loaded state from file")
        print(f"  Loaded day: {loaded_state.current_day}")

        return True
    except Exception as e:
        print(f"✗ Error in patient state manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patient_simulator():
    """Test patient simulator"""
    print("\n=== Testing Patient Simulator ===")
    try:
        # Load patient and disease
        patient = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")

        # Create state manager and initialize
        manager = PatientStateManager(patient, disease.disease_id)
        state = manager.initialize_state()

        # Create disease engine and update state for day 5
        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)
        symptoms = engine.get_symptoms_for_day(5)
        phase = disease.get_phase_for_day(5)

        state = manager.update_state(
            day=5,
            disease_symptoms=symptoms,
            current_phase=phase.name,
            expected_resolution_day=engine.get_expected_resolution_day()
        )

        # Create patient simulator
        simulator = PatientSimulator(patient, state)
        print(f"✓ Created patient simulator")

        # Test system prompt generation
        system_prompt = simulator.create_system_prompt()
        print(f"✓ Generated system prompt ({len(system_prompt)} chars)")

        # Test opening message
        opening = simulator.generate_opening_message()
        print(f"\n✓ Opening message: \"{opening}\"")

        # Test responses to various questions
        print("\nTesting patient responses:")

        test_questions = [
            "How are you feeling?",
            "How long have you been sick?",
            "What are your symptoms?",
            "Is it getting worse or better?",
            "Have you tried any medications?",
        ]

        conversation_history = []
        for question in test_questions:
            response = simulator.respond(question, conversation_history)
            print(f"\n  Chatbot: {question}")
            print(f"  Patient: {response}")

            conversation_history.append({"speaker": "chatbot", "message": question})
            conversation_history.append({"speaker": "patient", "message": response})

        # Test anxiety interjection
        anxiety_q = simulator.get_anxiety_interjection()
        if anxiety_q:
            print(f"\n✓ Anxiety interjection: \"{anxiety_q}\"")

        return True
    except Exception as e:
        print(f"✗ Error in patient simulator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integrated workflow"""
    print("\n=== Testing Integrated Workflow ===")
    try:
        # Load models
        patient = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")
        print(f"✓ Loaded patient: {patient.name}")
        print(f"✓ Loaded disease: {disease.name}")

        # Initialize components
        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)
        manager = PatientStateManager(patient, disease.disease_id)
        state = manager.initialize_state()
        print(f"✓ Initialized all components")

        # Simulate one full day
        day = 5
        print(f"\nSimulating Day {day}:")

        # 1. Get symptoms from disease engine
        symptoms = engine.get_symptoms_for_day(day)
        print(f"  1. Disease engine calculated {len(symptoms)} symptoms")

        # 2. Update patient state
        phase = disease.get_phase_for_day(day)
        state = manager.update_state(
            day=day,
            disease_symptoms=symptoms,
            current_phase=phase.name,
            expected_resolution_day=engine.get_expected_resolution_day()
        )
        print(f"  2. State manager updated patient state")
        print(f"     - Temperature: {state.biological.temperature}°C")
        print(f"     - Anxiety: {state.psychological.anxiety_level}/10")

        # 3. Create patient simulator
        simulator = PatientSimulator(patient, state)
        print(f"  3. Patient simulator created")

        # 4. Generate opening message
        opening = simulator.generate_opening_message()
        print(f"  4. Patient opens conversation:")
        print(f"     \"{opening}\"")

        # 5. Simulate conversation turn
        chatbot_msg = "I'm sorry to hear that. Can you tell me more about your symptoms?"
        patient_response = simulator.respond(chatbot_msg, [])
        print(f"  5. Chatbot responds, patient replies:")
        print(f"     \"{patient_response}\"")

        print(f"\n✓ Full day simulation complete!")

        return True
    except Exception as e:
        print(f"✗ Error in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("PHASE 2: Disease & Patient Simulation - Test Suite")
    print("=" * 60)

    # Setup logging
    setup_logging(level="WARNING", console=True)

    tests = [
        ("Disease Progression Engine", test_disease_progression_engine),
        ("Patient State Manager", test_patient_state_manager),
        ("Patient Simulator", test_patient_simulator),
        ("Integrated Workflow", test_integration),
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
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Phase 2 is complete.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
