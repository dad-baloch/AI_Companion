"""Prompt Templates — builds context-aware prompts for the LLM."""

from typing import List, Dict

# Emotion-aware system instructions
EMOTION_INSTRUCTIONS = {
    "happy": "The user seems happy. Be enthusiastic and share in their positive energy.",
    "sad": "The user seems sad. Be empathetic, gentle, and comforting.",
    "angry": "The user seems frustrated. Be calm, patient, and understanding.",
    "fearful": "The user seems anxious. Be reassuring and supportive.",
    "surprised": "The user seems surprised. Be engaging and curious about what surprised them.",
    "disgusted": "The user seems uncomfortable. Be understanding and try to help.",
    "neutral": "The user seems calm. Be friendly and conversational.",
}

SYSTEM_PROMPT = """You are a friendly, empathetic AI companion. You are helpful, warm, and conversational.
You speak naturally and keep your responses concise (2-3 sentences unless more detail is needed).
You remember the context of the conversation and refer back to it when appropriate.
{emotion_instruction}"""


def build_prompt(
    user_message: str,
    emotion: str = "neutral",
    history: List[Dict[str, str]] = None,
    max_history: int = 10,
) -> str:
    """Build a full prompt with system instructions, history, and current message."""
    emotion_instruction = EMOTION_INSTRUCTIONS.get(emotion, EMOTION_INSTRUCTIONS["neutral"])
    system = SYSTEM_PROMPT.format(emotion_instruction=emotion_instruction)

    parts = [f"System: {system}"]

    # Add recent conversation history
    if history:
        for msg in history[-max_history:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")

    parts.append(f"User: {user_message}")
    parts.append("Assistant:")

    return "\n\n".join(parts)
