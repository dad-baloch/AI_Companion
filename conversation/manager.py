"""Conversation Manager — the central switchboard that coordinates ASR, LLM, TTS, emotion, and memory."""

import logging
from typing import Optional

from .prompt_templates import build_prompt

logger = logging.getLogger(__name__)


class ConversationManager:
    """Orchestrates the full conversation loop:
    listen → detect emotion → build prompt → generate → speak.
    """

    def __init__(self, llm=None, asr=None, tts=None, emotion=None, memory=None):
        self.llm = llm
        self.asr = asr
        self.tts = tts
        self.emotion = emotion
        self.memory = memory
        self.current_emotion = "neutral"
        self.is_active = False
        logger.info("ConversationManager initialized.")

    def start(self):
        """Start the conversation loop."""
        self.is_active = True
        logger.info("Conversation started.")

    def stop(self):
        """Stop the conversation loop."""
        self.is_active = False
        logger.info("Conversation stopped.")

    def process_turn(self, user_text: str, detected_emotion: str = "neutral") -> str:
        """Process a single conversation turn.

        Args:
            user_text: Transcribed user speech.
            detected_emotion: Emotion detected from the user's face.

        Returns:
            The assistant's response text.
        """
        self.current_emotion = detected_emotion

        # Store user message in memory
        if self.memory:
            self.memory.add_message("user", user_text)

        # Build context-aware prompt
        history = self.memory.get_history() if self.memory else []
        prompt = build_prompt(
            user_message=user_text,
            emotion=detected_emotion,
            history=history,
        )

        # Generate response via LLM
        if self.llm:
            response = self.llm.generate_response(prompt)
        else:
            response = "I'm sorry, the language model is not loaded."

        # Store assistant response in memory
        if self.memory:
            self.memory.add_message("assistant", response)

        logger.info(f"Turn processed — emotion: {detected_emotion}")
        return response

    def full_turn(self) -> Optional[str]:
        """Execute a full turn: listen → detect → think → speak."""
        if not self.is_active:
            logger.warning("Conversation not active.")
            return None

        # Step 1: Listen
        user_text = None
        if self.asr:
            user_text = self.asr.recognize()
        if not user_text:
            return None

        # Step 2: Detect emotion
        emotion = "neutral"
        if self.emotion:
            emotion = self.emotion.predict()

        # Step 3: Process turn (prompt + LLM)
        response = self.process_turn(user_text, emotion)

        # Step 4: Speak
        if self.tts:
            self.tts.speak(response)

        return response
