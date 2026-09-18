"""Emotion Smoothing — applies temporal smoothing to reduce emotion prediction noise."""

from collections import deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmotionSmoother:
    """Smooths emotion predictions over a sliding window to reduce flickering."""

    def __init__(self, window_size: int = 10, confidence_threshold: float = 0.4):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.history: deque = deque(maxlen=window_size)
        self.current_emotion = "neutral"

    def update(self, emotion_probs: Dict[str, float]) -> str:
        """Update with new emotion probabilities and return smoothed emotion.
        
        Args:
            emotion_probs: Dict mapping emotion labels to probabilities.
            
        Returns:
            The smoothed (most stable) emotion label.
        """
        self.history.append(emotion_probs)

        if len(self.history) == 0:
            return self.current_emotion

        # Average probabilities across the window
        avg_probs = {}
        for emotion in emotion_probs.keys():
            avg_probs[emotion] = sum(
                h.get(emotion, 0.0) for h in self.history
            ) / len(self.history)

        # Get the dominant emotion
        dominant = max(avg_probs, key=avg_probs.get)
        dominant_confidence = avg_probs[dominant]

        # Only switch if confidence exceeds threshold
        if dominant_confidence >= self.confidence_threshold:
            self.current_emotion = dominant

        logger.debug(f"[Smoother] Smoothed emotion: {self.current_emotion} ({dominant_confidence:.2f})")
        return self.current_emotion

    def get_current(self) -> str:
        """Get the current smoothed emotion."""
        return self.current_emotion

    def reset(self):
        """Reset the smoothing history."""
        self.history.clear()
        self.current_emotion = "neutral"
        logger.info("[Smoother] History reset.")
