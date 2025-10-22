#!/usr/bin/env python3
"""
Test script for Phase 1: Core Infrastructure

Tests that all data models and utilities work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.patient_profile import PatientProfile
from src.models.disease_model import DiseaseModel
from src.models.patient_state import PatientState, BiologicalState, PsychologicalState, DiseaseState
from src.models.conversation import Conversation, PatientStateSnapshot, BiologicalState as ConvBioState, PsychologicalState as ConvPsychState
from src.utils.logging_config import setup_logging
from src.utils.file_utils import list_patient_profiles, list_disease_models

def test_patient_profile():
    """Test loading patient profile"""
    print("\n=== Testing Patient Profile ===")
    try:
        profile = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        print(f"✓ Loaded patient profile: {profile.patient_id}")
        print(f"✓ Validation passed: {profile.validate()}")
        print(f"\n{profile.get_description()}")
        return True
    except Exception as e:
        print(f"✗ Error loading patient profile: {e}")
        return False

def test_disease_model():
    """Test loading disease model"""
    print("\n=== Testing Disease Model ===")
    try:
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")
        print(f"✓ Loaded disease model: {disease.disease_id}")
        print(f"✓ Validation passed: {disease.validate()}")
        print(f"\n{disease.get_description()}")

        # Test getting phase for day
        phase = disease.get_phase_for_day(5)
        if phase:
            print(f"\n✓ Day 5 phase: {phase.name}")
            print(f"  Symptoms in this phase: {list(phase.symptoms.keys())}")
        return True
    except Exception as e:
        print(f"✗ Error loading disease model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_patient_state():
    """Test creating patient state"""
    print("\n=== Testing Patient State ===")
    try:
        from datetime import datetime

        bio_state = BiologicalState(
            temperature=37.5,
            heart_rate=75,
            energy_level=6,
            pain_level=2
        )

        psych_state = PsychologicalState(
            anxiety_level=7,
            mood="worried",
            health_concern_level=6
        )

        disease_state = DiseaseState(
            active_disease="viral_uri",
            disease_day=1,
            current_phase="incubation",
            trajectory="stable",
            complications_occurred=False,
            expected_resolution_day=10
        )

        state = PatientState(
            patient_id="patient_001",
            current_day=1,
            last_updated=datetime.now().isoformat(),
            biological=bio_state,
            psychological=psych_state,
            disease_state=disease_state
        )

        print(f"✓ Created patient state")
        print(f"✓ Validation passed: {state.validate()}")
        print(f"\n{state.get_summary()}")

        # Test saving and loading
        state.to_json("data/state/test_state.json")
        print(f"✓ Saved state to JSON")

        loaded_state = PatientState.from_json("data/state/test_state.json")
        print(f"✓ Loaded state from JSON")

        return True
    except Exception as e:
        print(f"✗ Error with patient state: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation():
    """Test creating conversation"""
    print("\n=== Testing Conversation ===")
    try:
        from datetime import datetime

        bio_state = ConvBioState(temperature=37.5, heart_rate=75)
        psych_state = ConvPsychState(anxiety_level=7, mood="worried", health_concern_level=6)
        patient_state = PatientStateSnapshot(biological=bio_state, psychological=psych_state)

        conv = Conversation(
            conversation_id="test_conv_001",
            patient_id="patient_001",
            simulation_day=1,
            disease_day=1,
            real_timestamp=datetime.now().isoformat(),
            condition="with_history",
            patient_state_at_start=patient_state
        )

        print(f"✓ Created conversation")
        print(f"✓ Validation passed: {conv.validate()}")

        # Add turns
        conv.add_turn("patient", "Hi, I'm not feeling well")
        conv.add_turn("chatbot", "I'm sorry to hear that. Can you tell me more?")

        print(f"✓ Added {conv.get_turn_count()} turns")

        # Test saving
        conv.to_json("data/conversations/test_conv.json")
        print(f"✓ Saved conversation to JSON")

        # Test loading
        loaded_conv = Conversation.from_json("data/conversations/test_conv.json")
        print(f"✓ Loaded conversation from JSON")
        print(f"  Loaded {loaded_conv.get_turn_count()} turns")

        return True
    except Exception as e:
        print(f"✗ Error with conversation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_utils():
    """Test file utilities"""
    print("\n=== Testing File Utils ===")
    try:
        # List patient profiles
        profiles = list_patient_profiles()
        print(f"✓ Found {len(profiles)} patient profiles:")
        for p in profiles:
            print(f"  - {p.name}")

        # List disease models
        diseases = list_disease_models()
        print(f"✓ Found {len(diseases)} disease models:")
        for d in diseases:
            print(f"  - {d.name}")

        return True
    except Exception as e:
        print(f"✗ Error with file utils: {e}")
        return False

def test_logging():
    """Test logging configuration"""
    print("\n=== Testing Logging ===")
    try:
        logger = setup_logging(level="INFO", console=True)
        print(f"✓ Logging configured")
        logger.info("Test log message")
        print(f"✓ Log message written")
        return True
    except Exception as e:
        print(f"✗ Error with logging: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PHASE 1: Core Infrastructure - Test Suite")
    print("=" * 60)

    tests = [
        ("Logging", test_logging),
        ("Patient Profile", test_patient_profile),
        ("Disease Model", test_disease_model),
        ("Patient State", test_patient_state),
        ("Conversation", test_conversation),
        ("File Utils", test_file_utils),
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
        print("\n🎉 All tests passed! Phase 1 is complete.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
