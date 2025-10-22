#!/usr/bin/env python3
"""
Medical Chatbot Simulation Orchestrator

Main orchestrator that coordinates all system components to run
complete multi-day patient-chatbot simulations.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.models.patient_profile import PatientProfile
from src.models.disease_model import DiseaseModel
from src.models.conversation import Conversation
from src.models.patient_state import PatientState
from src.core.disease_engine import DiseaseProgressionEngine
from src.core.state_manager import PatientStateManager
from src.core.patient_simulator import PatientSimulator
from src.core.conversation_runner import ConversationRunner
from src.chatbots.local_llm_chatbot import LocalLLMChatbot
from src.evaluation.metrics_calculator import MetricsCalculator
from src.utils.file_utils import load_yaml, ensure_directory

logger = logging.getLogger(__name__)


class MedicalChatbotSimulation:
    """
    Main simulation orchestrator that manages multi-day patient-chatbot experiments.

    Coordinates:
    - Disease progression across days
    - Patient state evolution
    - Conversation scheduling (medical vs casual)
    - Chatbot initialization (with/without history)
    - Metrics calculation
    - Data persistence
    """

    def __init__(self, config_path: str = "config/experiment_config.yaml", random_seed: Optional[int] = None):
        """
        Initialize simulation from configuration file.

        Args:
            config_path: Path to experiment configuration YAML
            random_seed: Optional random seed for reproducibility
        """
        logger.info(f"Initializing MedicalChatbotSimulation with config: {config_path}")

        # Load configuration
        self.config = load_yaml(config_path)

        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            logger.info(f"Set random seed: {random_seed}")

        # Extract paths
        self.profiles_dir = Path(self.config.get('paths', {}).get('profiles', 'profiles/'))
        self.diseases_dir = Path(self.config.get('paths', {}).get('diseases', 'diseases/'))
        self.conversations_dir = Path(self.config.get('paths', {}).get('conversations', 'data/conversations/'))
        self.state_dir = Path(self.config.get('paths', {}).get('state', 'data/state/'))
        self.results_dir = Path(self.config.get('paths', {}).get('results', 'data/results/'))

        # Create directories if needed
        ensure_directory(self.conversations_dir)
        ensure_directory(self.state_dir)
        ensure_directory(self.results_dir)

        # Extract simulation parameters
        sim_config = self.config.get('simulation', {})
        self.default_max_days = sim_config.get('default_max_days', 14)
        self.default_max_turns = sim_config.get('default_max_turns_per_conversation', 10)
        self.default_timeline = sim_config.get('default_conversation_mix', [])

        # Extract LLM model configuration
        models_config = self.config.get('models', {})
        patient_llm = models_config.get('patient_llm', {})
        doctor_llm = models_config.get('doctor_llm', {})

        self.patient_model_name = patient_llm.get('model_name')
        self.patient_model_device = patient_llm.get('device', 'auto')
        self.patient_model_quantization = patient_llm.get('quantization')

        self.doctor_model_name = doctor_llm.get('model_name')
        self.doctor_model_device = doctor_llm.get('device', 'auto')
        self.doctor_model_quantization = doctor_llm.get('quantization')

        logger.info(f"Simulation initialized: max_days={self.default_max_days}, max_turns={self.default_max_turns}")
        logger.info(f"Patient LLM: {self.patient_model_name or 'rule-based'}")
        logger.info(f"Doctor LLM: {self.doctor_model_name or 'rule-based'}")

    def load_patient(self, patient_id: str) -> PatientProfile:
        """
        Load patient profile from YAML file.

        Args:
            patient_id: Patient identifier (e.g., 'patient_001')

        Returns:
            PatientProfile object
        """
        # Find patient file
        patient_files = list(self.profiles_dir.glob(f"{patient_id}_*.yaml"))

        if not patient_files:
            raise FileNotFoundError(f"No patient profile found for {patient_id} in {self.profiles_dir}")

        patient_file = patient_files[0]
        logger.info(f"Loading patient profile: {patient_file}")

        return PatientProfile.from_yaml(str(patient_file))

    def load_disease(self, disease_id: str) -> DiseaseModel:
        """
        Load disease model from YAML file.

        Args:
            disease_id: Disease identifier (e.g., 'viral_uri')

        Returns:
            DiseaseModel object
        """
        disease_file = self.diseases_dir / f"{disease_id}.yaml"

        if not disease_file.exists():
            raise FileNotFoundError(f"Disease model not found: {disease_file}")

        logger.info(f"Loading disease model: {disease_file}")

        return DiseaseModel.from_yaml(str(disease_file))

    def generate_timeline(self, num_days: int, disease_start_day: int = 1) -> List[Dict]:
        """
        Generate conversation timeline mixing medical and casual conversations.

        Args:
            num_days: Total number of simulation days
            disease_start_day: Day when disease symptoms begin

        Returns:
            List of timeline entries with day, type, and metadata
        """
        logger.info(f"Generating timeline for {num_days} days (disease starts day {disease_start_day})")

        # Use default timeline if available and appropriate length
        if self.default_timeline and len(self.default_timeline) > 0:
            # Use configured timeline as a template
            timeline = []
            for entry in self.default_timeline:
                day = entry.get('day', 1)
                if day <= num_days:
                    timeline.append({
                        'day': day,
                        'type': entry.get('type', 'medical'),
                        'scheduled': True
                    })

            # Fill in any missing days as needed
            scheduled_days = {entry['day'] for entry in timeline}
            for day in range(1, num_days + 1):
                if day not in scheduled_days:
                    # Default to medical if symptoms present, casual otherwise
                    conv_type = 'medical' if day >= disease_start_day else 'casual'
                    timeline.append({
                        'day': day,
                        'type': conv_type,
                        'scheduled': False
                    })

            timeline.sort(key=lambda x: x['day'])
            return timeline

        # Generate timeline from scratch
        timeline = []

        for day in range(1, num_days + 1):
            # Determine conversation type based on disease progression
            if day < disease_start_day:
                # Before symptoms: mostly casual
                conv_type = 'casual'
            elif day == disease_start_day:
                # First symptom day: medical
                conv_type = 'medical'
            else:
                # After symptoms start: mix with bias toward medical
                # Every 2-3 days do medical follow-up
                days_since_start = day - disease_start_day
                if days_since_start % 2 == 0:
                    conv_type = 'medical_followup'
                elif random.random() < 0.3:  # 30% chance of casual conversation
                    conv_type = 'casual'
                else:
                    conv_type = 'medical_followup'

            timeline.append({
                'day': day,
                'type': conv_type,
                'scheduled': False
            })

        logger.info(f"Generated timeline: {sum(1 for t in timeline if 'medical' in t['type'])} medical, "
                   f"{sum(1 for t in timeline if t['type'] == 'casual')} casual")

        return timeline

    def run_day(self,
                day: int,
                conversation_type: str,
                patient: PatientProfile,
                disease: DiseaseModel,
                engine: DiseaseProgressionEngine,
                manager: PatientStateManager,
                condition: str,
                max_turns: Optional[int] = None) -> Conversation:
        """
        Run simulation for a single day.

        Args:
            day: Current simulation day
            conversation_type: 'medical', 'medical_followup', or 'casual'
            patient: Patient profile
            disease: Disease model
            engine: Disease progression engine
            manager: Patient state manager
            condition: 'with_history' or 'without_history'
            max_turns: Maximum conversation turns (uses default if None)

        Returns:
            Conversation object with all data
        """
        logger.info(f"Running day {day} - {conversation_type} conversation ({condition})")

        # 1. Get current patient state
        state = manager.get_current_state()

        # 2. Get symptoms for this day from disease engine
        disease_day = day - patient.disease_start_day + 1

        if disease_day >= 1:
            symptoms = engine.get_symptoms_for_day(disease_day)
            phase = disease.get_phase_for_day(disease_day)

            # 3. Update patient state with new symptoms
            state = manager.update_state(
                day=day,
                disease_symptoms=symptoms,
                current_phase=phase.name,
                expected_resolution_day=engine.get_expected_resolution_day() + patient.disease_start_day - 1
            )
        else:
            # Before disease starts - no symptoms
            logger.info(f"Day {day} is before disease onset (starts day {patient.disease_start_day})")

        # 4. Create PatientSimulator (with LLM if configured)
        simulator = PatientSimulator(
            patient,
            state,
            llm_model=self.patient_model_name,
            device=self.patient_model_device,
            quantization=self.patient_model_quantization
        )

        # 5. Initialize chatbot (with or without history and LLM)
        if condition == 'with_history':
            # Get conversation history from state manager
            context = manager.get_conversation_history(condition=condition)
            chatbot = LocalLLMChatbot(
                model_name=self.doctor_model_name,
                context=context,
                condition='with_history',
                device=self.doctor_model_device,
                quantization=self.doctor_model_quantization
            )
            logger.debug(f"Chatbot initialized WITH history: {len(context)} previous conversations")
        else:
            chatbot = LocalLLMChatbot(
                model_name=self.doctor_model_name,
                condition='without_history',
                device=self.doctor_model_device,
                quantization=self.doctor_model_quantization
            )
            logger.debug(f"Chatbot initialized WITHOUT history")

        # 6. Run ConversationRunner
        runner = ConversationRunner(
            patient_simulator=simulator,
            chatbot=chatbot,
            max_turns=max_turns or self.default_max_turns
        )

        conversation_id = f"{patient.patient_id}_day_{day}_{condition}"

        conversation = runner.run_conversation(
            conversation_id=conversation_id,
            patient_id=patient.patient_id,
            simulation_day=day,
            disease_day=disease_day if disease_day >= 1 else 0,
            patient_state=state
        )

        # 7. Calculate metrics
        calculator = MetricsCalculator(condition=condition)
        patient_history = manager.get_conversation_history(condition=condition) if condition == 'with_history' else []
        metrics = calculator.calculate_metrics(conversation, patient_history=patient_history)
        conversation.metrics = metrics

        logger.info(f"Day {day} complete: {len(conversation.turns)} turns, "
                   f"{metrics.automatic.bot_questions_asked} questions asked")

        # 8. Save conversation
        patient_conv_dir = self.conversations_dir / patient.patient_id / condition
        ensure_directory(patient_conv_dir)

        conv_filename = patient_conv_dir / f"day_{day:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        conversation.to_json(str(conv_filename))
        logger.info(f"Conversation saved: {conv_filename}")

        # 9. Update state manager with conversation
        topic = runner.get_conversation_topic(conversation.turns)
        key_info = runner.extract_key_info(conversation.turns)
        manager.add_conversation_to_history(conversation, topic, key_info)

        # Save updated state
        manager.save_state()

        # 10. Return conversation object
        return conversation

    def run_experiment(self,
                      patient_id: str,
                      num_days: Optional[int] = None,
                      condition: str = 'without_history',
                      random_seed: Optional[int] = None) -> Dict:
        """
        Run complete experiment for one patient under one condition.

        Args:
            patient_id: Patient identifier
            num_days: Number of simulation days (uses default if None)
            condition: 'with_history' or 'without_history'
            random_seed: Optional random seed for this experiment

        Returns:
            Dictionary with experiment results and metadata
        """
        if random_seed is not None:
            random.seed(random_seed)

        num_days = num_days or self.default_max_days

        logger.info("=" * 80)
        logger.info(f"STARTING EXPERIMENT: {patient_id} - {condition} - {num_days} days")
        logger.info("=" * 80)

        # Load patient and disease
        patient = self.load_patient(patient_id)
        disease = self.load_disease(patient.assigned_disease)

        logger.info(f"Patient: {patient.name} ({patient.patient_id})")
        logger.info(f"Disease: {disease.name} ({disease.disease_id})")
        logger.info(f"Disease starts: Day {patient.disease_start_day}")

        # Initialize simulation components
        engine = DiseaseProgressionEngine(disease, patient, random_seed=random_seed)
        manager = PatientStateManager(patient, disease.disease_id, condition=condition)

        # Initialize or load state
        try:
            state = manager.load_state(patient.patient_id, condition=condition)
            logger.info(f"Loaded existing state from day {state.current_day}")
        except (FileNotFoundError, Exception):
            state = manager.initialize_state()
            logger.info(f"Initialized new patient state")

        # Generate timeline
        timeline = self.generate_timeline(num_days, patient.disease_start_day)

        # Run each day
        conversations = []

        for timeline_entry in timeline:
            day = timeline_entry['day']
            conv_type = timeline_entry['type']

            try:
                conversation = self.run_day(
                    day=day,
                    conversation_type=conv_type,
                    patient=patient,
                    disease=disease,
                    engine=engine,
                    manager=manager,
                    condition=condition
                )

                conversations.append({
                    'day': day,
                    'type': conv_type,
                    'conversation_id': conversation.conversation_id,
                    'turns': len(conversation.turns),
                    'metrics': conversation.metrics
                })

                logger.info(f"✓ Day {day}/{num_days} complete")

            except Exception as e:
                logger.error(f"Error on day {day}: {e}", exc_info=True)
                raise

        # Generate experiment summary
        results = {
            'experiment_id': f"{patient_id}_{condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'patient_id': patient_id,
            'condition': condition,
            'num_days': num_days,
            'disease_id': disease.disease_id,
            'start_time': conversations[0]['conversation_id'] if conversations else None,
            'end_time': conversations[-1]['conversation_id'] if conversations else None,
            'conversations': conversations,
            'total_conversations': len(conversations),
            'total_turns': sum(c['turns'] for c in conversations),
            'timeline': timeline
        }

        # Save experiment summary
        summary_file = self.results_dir / f"{patient_id}_{condition}_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("=" * 80)
        logger.info(f"EXPERIMENT COMPLETE: {patient_id} - {condition}")
        logger.info(f"Total conversations: {len(conversations)}")
        logger.info(f"Total turns: {sum(c['turns'] for c in conversations)}")
        logger.info(f"Summary saved: {summary_file}")
        logger.info("=" * 80)

        return results

    def run_full_experiment(self,
                           patient_ids: Optional[List[str]] = None,
                           conditions: Optional[List[str]] = None,
                           num_days: Optional[int] = None) -> Dict:
        """
        Run complete experiment across multiple patients and conditions.

        Args:
            patient_ids: List of patient IDs (discovers all if None)
            conditions: List of conditions to test (uses config if None)
            num_days: Number of days per experiment

        Returns:
            Dictionary with all experiment results
        """
        # Discover patients if not specified
        if patient_ids is None:
            patient_files = list(self.profiles_dir.glob("patient_*.yaml"))
            patient_ids = [f.stem.split('_')[0] + '_' + f.stem.split('_')[1]
                          for f in patient_files if not f.stem.startswith('template')]
            logger.info(f"Discovered {len(patient_ids)} patients: {patient_ids}")

        # Use configured conditions if not specified
        if conditions is None:
            conditions = self.config.get('conditions', ['with_history', 'without_history'])

        logger.info("=" * 80)
        logger.info("RUNNING FULL EXPERIMENT")
        logger.info(f"Patients: {len(patient_ids)}")
        logger.info(f"Conditions: {conditions}")
        logger.info(f"Days per experiment: {num_days or self.default_max_days}")
        logger.info("=" * 80)

        all_results = {
            'experiment_date': datetime.now().isoformat(),
            'patients': patient_ids,
            'conditions': conditions,
            'num_days': num_days or self.default_max_days,
            'results': []
        }

        for patient_id in patient_ids:
            for condition in conditions:
                logger.info(f"\n>>> Running: {patient_id} - {condition}")

                try:
                    result = self.run_experiment(
                        patient_id=patient_id,
                        num_days=num_days,
                        condition=condition
                    )
                    all_results['results'].append(result)

                except Exception as e:
                    logger.error(f"Failed: {patient_id} - {condition}: {e}", exc_info=True)
                    all_results['results'].append({
                        'patient_id': patient_id,
                        'condition': condition,
                        'error': str(e),
                        'status': 'failed'
                    })

        # Save complete results
        complete_results_file = self.results_dir / f"full_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(complete_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info("=" * 80)
        logger.info("FULL EXPERIMENT COMPLETE")
        logger.info(f"Total experiments: {len(all_results['results'])}")
        logger.info(f"Successful: {sum(1 for r in all_results['results'] if 'error' not in r)}")
        logger.info(f"Failed: {sum(1 for r in all_results['results'] if 'error' in r)}")
        logger.info(f"Results saved: {complete_results_file}")
        logger.info("=" * 80)

        return all_results
