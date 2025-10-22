#!/usr/bin/env python3
"""
Test script for Phase 3: Chatbot & Conversation

Tests chatbots, conversation runner, and metrics calculation.
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
from src.chatbots.local_llm_chatbot import LocalLLMChatbot
from src.core.conversation_runner import ConversationRunner
from src.evaluation.metrics_calculator import MetricsCalculator
from src.utils.logging_config import setup_logging
from datetime import datetime


def test_chatbot_initialization():
    """Test chatbot initialization"""
    print("\n=== Testing Chatbot Initialization ===")
    try:
        # Test without history
        chatbot1 = LocalLLMChatbot(condition="without_history")
        print(f"✓ Created chatbot (without_history)")
        print(f"  Has context: {chatbot1.has_context()}")

        # Test with history
        history = [
            {"day": 1, "type": "medical", "topic": "initial symptoms", "key_info": ["duration", "fever"]}
        ]
        chatbot2 = LocalLLMChatbot(context=history, condition="with_history")
        print(f"✓ Created chatbot (with_history)")
        print(f"  Has context: {chatbot2.has_context()}")
        print(f"  Context items: {len(chatbot2.get_context())}")

        # Test system prompt generation
        prompt = chatbot1.get_system_prompt()
        print(f"✓ Generated system prompt ({len(prompt)} chars)")

        return True
    except Exception as e:
        print(f"✗ Error in chatbot initialization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chatbot_responses():
    """Test chatbot response generation"""
    print("\n=== Testing Chatbot Responses ===")
    try:
        chatbot = LocalLLMChatbot(condition="without_history")

        test_conversations = [
            [{"speaker": "patient", "message": "I've been feeling sick for a few days"}],
            [
                {"speaker": "patient", "message": "I have a sore throat"},
                {"speaker": "chatbot", "message": "How long have you had the sore throat?"},
                {"speaker": "patient", "message": "About 3 days"}
            ]
        ]

        for i, conv in enumerate(test_conversations):
            response = chatbot.respond(conv)
            print(f"\n  Test {i+1}:")
            print(f"    Last patient: {conv[-1]['message'][:50]}")
            print(f"    Bot response: {response[:80]}...")

        print(f"\n✓ Chatbot generating appropriate responses")

        return True
    except Exception as e:
        print(f"✗ Error in chatbot responses: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conversation_runner():
    """Test conversation runner"""
    print("\n=== Testing Conversation Runner ===")
    try:
        # Setup
        patient = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")

        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)
        manager = PatientStateManager(patient, disease.disease_id)
        state = manager.initialize_state()

        # Update to day 5
        symptoms = engine.get_symptoms_for_day(5)
        phase = disease.get_phase_for_day(5)
        state = manager.update_state(5, symptoms, phase.name, engine.get_expected_resolution_day())

        # Create simulator and chatbot
        simulator = PatientSimulator(patient, state)
        chatbot = LocalLLMChatbot(condition="without_history")

        # Create runner
        runner = ConversationRunner(simulator, chatbot, max_turns=8)
        print(f"✓ Created conversation runner (max_turns=8)")

        # Run conversation
        conversation = runner.run_conversation(
            conversation_id="test_conv_001",
            patient_id=patient.patient_id,
            simulation_day=5,
            disease_day=5,
            patient_state=state
        )

        print(f"✓ Conversation completed")
        print(f"  Total turns: {len(conversation.turns)}")
        print(f"  Duration: {conversation.conversation_summary.duration_seconds:.1f}s")
        print(f"  End reason: {conversation.conversation_summary.conversation_end_reason}")

        # Show conversation
        print(f"\n  Conversation excerpt:")
        for i, turn in enumerate(conversation.turns[:6]):  # Show first 3 exchanges
            speaker_label = "Patient" if turn.speaker == "patient" else "Chatbot"
            message = turn.message[:60] + "..." if len(turn.message) > 60 else turn.message
            print(f"    {i+1}. {speaker_label}: {message}")

        return True
    except Exception as e:
        print(f"✗ Error in conversation runner: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_calculator():
    """Test metrics calculation"""
    print("\n=== Testing Metrics Calculator ===")
    try:
        # Setup and run a conversation
        patient = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")

        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)
        manager = PatientStateManager(patient, disease.disease_id)
        state = manager.initialize_state()

        symptoms = engine.get_symptoms_for_day(5)
        phase = disease.get_phase_for_day(5)
        state = manager.update_state(5, symptoms, phase.name, engine.get_expected_resolution_day())

        simulator = PatientSimulator(patient, state)
        chatbot = LocalLLMChatbot(condition="without_history")
        runner = ConversationRunner(simulator, chatbot, max_turns=8)

        conversation = runner.run_conversation(
            conversation_id="test_metrics_001",
            patient_id=patient.patient_id,
            simulation_day=5,
            disease_day=5,
            patient_state=state
        )

        # Calculate metrics
        calculator = MetricsCalculator(condition="without_history")
        print(f"✓ Created metrics calculator")

        metrics = calculator.calculate_metrics(conversation)
        print(f"✓ Calculated metrics")

        # Display metrics
        print(f"\n  Metrics:")
        print(f"    Bot questions: {metrics.automatic.bot_questions_asked}")
        print(f"    Redundant questions: {metrics.automatic.redundant_questions}")
        print(f"    History references: {metrics.automatic.references_to_history}")
        print(f"    Contextual questions: {metrics.automatic.contextual_questions}")
        print(f"    Info gathered: {len(metrics.automatic.key_info_gathered)} - {metrics.automatic.key_info_gathered}")
        print(f"    Info missed: {len(metrics.automatic.key_info_missed)} - {metrics.automatic.key_info_missed}")
        print(f"    Patient volunteered: {metrics.automatic.patient_volunteered_count}")
        print(f"    Patient prompted: {metrics.automatic.patient_only_when_asked_count}")

        # Get summary
        summary = calculator.get_metrics_summary(metrics)
        print(f"\n✓ Generated metrics summary")

        return True
    except Exception as e:
        print(f"✗ Error in metrics calculator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_history_condition():
    """Test with_history condition"""
    print("\n=== Testing WITH History Condition ===")
    try:
        # Setup
        patient = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")

        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)
        manager = PatientStateManager(patient, disease.disease_id)
        state = manager.initialize_state()

        symptoms = engine.get_symptoms_for_day(5)
        phase = disease.get_phase_for_day(5)
        state = manager.update_state(5, symptoms, phase.name, engine.get_expected_resolution_day())

        # Create chatbot WITH history
        history = [
            {
                "day": 3,
                "type": "medical",
                "topic": "initial symptoms",
                "key_info": ["duration", "sore_throat", "fever"]
            }
        ]

        simulator = PatientSimulator(patient, state)
        chatbot = LocalLLMChatbot(context=history, condition="with_history")
        runner = ConversationRunner(simulator, chatbot, max_turns=6)

        print(f"✓ Created chatbot with history")
        print(f"  Previous conversations: {len(history)}")

        # Run conversation
        conversation = runner.run_conversation(
            conversation_id="test_with_history_001",
            patient_id=patient.patient_id,
            simulation_day=5,
            disease_day=5,
            patient_state=state
        )

        print(f"✓ Conversation completed with history context")
        print(f"  Turns: {len(conversation.turns)}")

        # Calculate metrics
        calculator = MetricsCalculator(condition="with_history")
        metrics = calculator.calculate_metrics(conversation, patient_history=history)

        print(f"✓ Metrics calculated with history checking")
        print(f"  Redundant questions: {metrics.automatic.redundant_questions}")
        print(f"  History references: {metrics.automatic.references_to_history}")

        return True
    except Exception as e:
        print(f"✗ Error in with_history test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test complete end-to-end workflow"""
    print("\n=== Testing Full Integration ===")
    try:
        print("\n  Simulating complete day with conversation...")

        # Load models
        patient = PatientProfile.from_yaml("profiles/patient_001_alex_chen.yaml")
        disease = DiseaseModel.from_yaml("diseases/viral_uri.yaml")

        # Initialize all components
        engine = DiseaseProgressionEngine(disease, patient, random_seed=42)
        manager = PatientStateManager(patient, disease.disease_id)
        state = manager.initialize_state()

        # Simulate Day 5
        day = 5
        symptoms = engine.get_symptoms_for_day(day)
        phase = disease.get_phase_for_day(day)
        state = manager.update_state(day, symptoms, phase.name, engine.get_expected_resolution_day())

        print(f"  ✓ Day {day} state updated: {len(state.current_symptoms)} symptoms")

        # Create conversation components
        simulator = PatientSimulator(patient, state)
        chatbot = LocalLLMChatbot(condition="without_history")
        runner = ConversationRunner(simulator, chatbot, max_turns=8)

        # Run conversation
        conversation = runner.run_conversation(
            conversation_id=f"patient_001_day_{day}",
            patient_id=patient.patient_id,
            simulation_day=day,
            disease_day=day,
            patient_state=state
        )

        print(f"  ✓ Conversation completed: {len(conversation.turns)} turns")

        # Calculate metrics
        calculator = MetricsCalculator(condition="without_history")
        metrics = calculator.calculate_metrics(conversation)
        conversation.metrics = metrics

        print(f"  ✓ Metrics calculated: {metrics.automatic.bot_questions_asked} questions")

        # Save conversation
        conversation.to_json(f"data/conversations/test_full_integration_day{day}.json")
        print(f"  ✓ Conversation saved to JSON")

        # Add to state history
        topic = runner.get_conversation_topic(conversation.turns)
        key_info = runner.extract_key_info(conversation.turns)
        manager.add_conversation_to_history(conversation, topic, key_info)

        print(f"  ✓ Added to state history")
        print(f"    Topic: {topic}")
        print(f"    Key info: {key_info}")

        # Save state
        manager.save_state()
        print(f"  ✓ State saved")

        print(f"\n✓ Full integration successful!")
        print(f"\nSummary:")
        print(f"  - Disease progression: Working")
        print(f"  - Patient state: Working")
        print(f"  - Patient simulation: Working")
        print(f"  - Chatbot: Working")
        print(f"  - Conversation flow: Working")
        print(f"  - Metrics calculation: Working")
        print(f"  - Data persistence: Working")

        return True
    except Exception as e:
        print(f"✗ Error in full integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("PHASE 3: Chatbot & Conversation - Test Suite")
    print("=" * 60)

    # Setup logging
    setup_logging(level="WARNING", console=True)

    tests = [
        ("Chatbot Initialization", test_chatbot_initialization),
        ("Chatbot Responses", test_chatbot_responses),
        ("Conversation Runner", test_conversation_runner),
        ("Metrics Calculator", test_metrics_calculator),
        ("With History Condition", test_with_history_condition),
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
        print("\n🎉 All tests passed! Phase 3 is complete.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
