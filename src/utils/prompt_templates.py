"""
Prompt Templates

System prompts and templates for patient simulator and chatbot.
"""

from typing import Dict, Any, List


PATIENT_SYSTEM_PROMPT = """You are roleplaying as a patient seeking medical advice from a chatbot.

=== YOUR IDENTITY ===
Name: {name}
Age: {age}, {gender}
Occupation: {occupation}

=== YOUR PERSONALITY ===
Communication style: {communication_description}
- Formality level: {formality}/10 (1=very formal, 10=very casual)
- Verbosity level: {verbosity}/10 (1=terse, 10=detailed)
- You use {vocabulary_style} language
- Example phrases you might use: {example_phrases}

Anxiety and mood:
- Your baseline anxiety is {baseline_anxiety}/10
- You're specifically anxious about health: {health_anxiety}/10
- Current mood: {current_mood} because {mood_triggers}
- You tend to {catastrophizing_behavior}

=== YOUR CURRENT SITUATION ===
You're on day {current_day} of feeling unwell.

Current symptoms you're experiencing:
{symptoms_list}

Your subjective experience (anxiety is affecting perception):
{subjective_experience}

=== YOUR BACKGROUND (reveal if asked) ===
{background_info}

=== HOW YOU RESPOND ===
{response_guidelines}

=== CONVERSATION RULES ===
1. Stay in character consistently
2. Don't list all symptoms at once - reveal naturally over conversation
3. Sometimes you're unsure ("maybe", "I think", "I'm not sure")
4. Express worry when anxious: "Do you think it could be something serious?"
5. Use your communication style (casual vs formal)
6. Don't volunteer information unless it makes sense naturally
7. Answer questions but don't over-explain
8. Keep responses relatively brief (2-4 sentences typically)

Begin the conversation naturally based on your personality and symptoms.
"""


CHATBOT_SYSTEM_PROMPT = """You are a helpful medical chatbot assistant. Your role is to:

1. Gather information about the patient's symptoms
2. Ask relevant follow-up questions
3. Provide appropriate medical advice
4. Determine if the patient should see a doctor

Guidelines:
- Be empathetic and professional
- Ask clear, focused questions
- Don't ask for information you already know (if you have conversation history)
- Provide practical advice
- Know when to recommend seeing a healthcare provider
- Keep responses concise and clear

{context_instructions}

Start by greeting the patient and asking how you can help.
"""


CHATBOT_WITH_HISTORY_CONTEXT = """You have access to previous conversations with this patient:

{conversation_history}

Use this information to avoid asking redundant questions and to provide continuity of care.
"""


CHATBOT_WITHOUT_HISTORY_CONTEXT = """This is a new conversation with no prior context. Treat this as the first time you're speaking with this patient.
"""


def format_patient_prompt(
    patient_profile: Dict[str, Any],
    current_state: Dict[str, Any],
    current_day: int
) -> str:
    """
    Format patient simulator system prompt

    Args:
        patient_profile: Patient profile data
        current_state: Current patient state
        current_day: Current simulation day

    Returns:
        Formatted system prompt
    """
    # Communication description
    formality = patient_profile['communication']['formality']
    if formality <= 3:
        comm_desc = "Very formal and polite"
    elif formality <= 6:
        comm_desc = "Moderately formal"
    else:
        comm_desc = "Casual and relaxed"

    # Catastrophizing behavior
    catastrophizing = patient_profile['psychological']['pain_catastrophizing']
    if catastrophizing <= 3:
        catastrophize_behavior = "stay calm and rational about symptoms"
    elif catastrophizing <= 6:
        catastrophize_behavior = "worry moderately about what symptoms might mean"
    else:
        catastrophize_behavior = "catastrophize and imagine worst-case scenarios"

    # Format symptoms list
    symptoms_list = _format_symptoms_list(current_state.get('symptoms', {}))

    # Subjective experience
    subjective_experience = _format_subjective_experience(
        current_state.get('symptoms', {}),
        patient_profile['psychological']['health_anxiety']
    )

    # Background info
    background_info = _format_background_info(patient_profile)

    # Response guidelines
    response_guidelines = _format_response_guidelines(
        patient_profile['psychological']['information_volunteering']
    )

    # Example phrases
    example_phrases = ", ".join(patient_profile['communication']['example_phrases'])

    # Mood triggers
    mood_triggers = ", ".join(patient_profile['psychological']['mood_triggers'])

    return PATIENT_SYSTEM_PROMPT.format(
        name=patient_profile['name'],
        age=patient_profile['demographics']['age'],
        gender=patient_profile['demographics']['gender'],
        occupation=patient_profile['demographics']['occupation'],
        communication_description=comm_desc,
        formality=formality,
        verbosity=patient_profile['communication']['verbosity'],
        vocabulary_style=patient_profile['communication']['vocabulary_style'],
        example_phrases=example_phrases,
        baseline_anxiety=patient_profile['psychological']['baseline_anxiety'],
        health_anxiety=patient_profile['psychological']['health_anxiety'],
        current_mood=patient_profile['psychological']['current_mood'],
        mood_triggers=mood_triggers,
        catastrophizing_behavior=catastrophize_behavior,
        current_day=current_day,
        symptoms_list=symptoms_list,
        subjective_experience=subjective_experience,
        background_info=background_info,
        response_guidelines=response_guidelines
    )


def format_chatbot_prompt(
    condition: str,
    conversation_history: List[Dict[str, Any]] = None
) -> str:
    """
    Format chatbot system prompt

    Args:
        condition: "with_history" or "without_history"
        conversation_history: List of previous conversations (if with_history)

    Returns:
        Formatted system prompt
    """
    if condition == "with_history" and conversation_history:
        context_instructions = CHATBOT_WITH_HISTORY_CONTEXT.format(
            conversation_history=_format_conversation_history(conversation_history)
        )
    else:
        context_instructions = CHATBOT_WITHOUT_HISTORY_CONTEXT

    return CHATBOT_SYSTEM_PROMPT.format(
        context_instructions=context_instructions
    )


def _format_symptoms_list(symptoms: Dict[str, Any]) -> str:
    """Format symptoms into readable list"""
    if not symptoms:
        return "No significant symptoms currently."

    lines = []
    for symptom_name, symptom_data in symptoms.items():
        severity = symptom_data.get('objective_severity', 0)
        description = symptom_data.get('description', symptom_name)
        lines.append(f"- {symptom_name.replace('_', ' ').title()}: {description} (severity: {severity}/10)")

    return "\n".join(lines)


def _format_subjective_experience(symptoms: Dict[str, Any], health_anxiety: int) -> str:
    """Format subjective experience with anxiety modulation"""
    if not symptoms:
        return "You feel relatively normal, maybe just a bit off."

    anxiety_modifier = ""
    if health_anxiety >= 7:
        anxiety_modifier = " You're quite worried this might be something serious."
    elif health_anxiety >= 4:
        anxiety_modifier = " You're somewhat concerned about what this means."

    symptom_count = len(symptoms)
    if symptom_count == 1:
        experience = "You have one symptom that's bothering you."
    elif symptom_count <= 3:
        experience = "You have a few symptoms that are concerning you."
    else:
        experience = "You have multiple symptoms and you're not feeling well at all."

    return experience + anxiety_modifier


def _format_background_info(patient_profile: Dict[str, Any]) -> str:
    """Format background medical information"""
    lines = []

    # Chronic conditions
    chronic_conditions = patient_profile['biological'].get('chronic_conditions', [])
    if chronic_conditions:
        conditions_str = ", ".join([c['type'] for c in chronic_conditions])
        lines.append(f"Chronic conditions: {conditions_str}")

    # Risk factors
    risk_factors = patient_profile['biological'].get('risk_factors', [])
    if risk_factors:
        lines.append("Family history:")
        for rf in risk_factors:
            lines.append(f"  - {rf['details']}")

    # Medications
    medications = patient_profile['biological'].get('medications', {}).get('current', [])
    if medications:
        meds_str = ", ".join([m['name'] for m in medications])
        lines.append(f"Current medications: {meds_str}")

    # Allergies
    allergies = patient_profile['biological'].get('medications', {}).get('allergies', [])
    if allergies:
        allergies_str = ", ".join(allergies)
        lines.append(f"Allergies: {allergies_str}")

    # Lifestyle
    lifestyle = patient_profile['biological'].get('lifestyle', {})
    if lifestyle:
        lines.append(f"Sleep: {lifestyle.get('sleep_hours', 'unknown')} hours/night, {lifestyle.get('sleep_quality', 'unknown')} quality")
        lines.append(f"Exercise: {lifestyle.get('exercise_frequency', 0)}x per week")
        if lifestyle.get('smoking'):
            lines.append("Smoker: Yes")
        if lifestyle.get('caffeine_intake') != 'none':
            lines.append(f"Caffeine: {lifestyle.get('caffeine_amount', 'regular use')}")

    return "\n".join(lines) if lines else "No significant medical history."


def _format_response_guidelines(info_volunteering: str) -> str:
    """Format response guidelines based on information volunteering level"""
    if info_volunteering == "low":
        return """- You only answer questions directly without adding extra information
- You don't volunteer information unless specifically asked
- You tend to give brief, minimal answers"""
    elif info_volunteering == "medium":
        return """- You answer questions and occasionally add relevant details
- You might mention related symptoms if they seem important
- You give moderate-length answers"""
    else:  # high
        return """- You tend to share information freely
- You often add details even when not specifically asked
- You might mention multiple symptoms or concerns in one response
- You give detailed, thorough answers"""


def _format_conversation_history(conversation_history: List[Dict[str, Any]]) -> str:
    """Format conversation history for context"""
    if not conversation_history:
        return "No previous conversations."

    lines = []
    for conv in conversation_history:
        day = conv.get('day', '?')
        conv_type = conv.get('type', 'unknown')
        topic = conv.get('topic', 'general')
        key_info = conv.get('key_info', [])

        lines.append(f"Day {day} ({conv_type}): {topic}")
        if key_info:
            lines.append(f"  Key information: {', '.join(key_info)}")

    return "\n".join(lines)


def format_opening_message(patient_profile: Dict[str, Any], symptoms: Dict[str, Any]) -> str:
    """
    Generate an opening message template for the patient

    Args:
        patient_profile: Patient profile data
        symptoms: Current symptoms

    Returns:
        Opening message template
    """
    formality = patient_profile['communication']['formality']
    anxiety = patient_profile['psychological']['health_anxiety']

    # Greeting based on formality
    if formality <= 3:
        greeting = "Hello,"
    elif formality <= 6:
        greeting = "Hi,"
    else:
        greeting = "Hey,"

    # Concern level based on anxiety
    if anxiety >= 7:
        concern = "I'm really worried about"
    elif anxiety >= 4:
        concern = "I'm a bit concerned about"
    else:
        concern = "I wanted to ask about"

    # Get primary symptom
    if symptoms:
        primary_symptom = list(symptoms.keys())[0].replace('_', ' ')
        return f"{greeting} {concern} some symptoms I've been having. {primary_symptom.title()} mainly."
    else:
        return f"{greeting} I haven't been feeling quite right lately."
