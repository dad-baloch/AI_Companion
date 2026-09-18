"""Emotion Classifier — classifies facial expressions into emotion categories."""

import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


class EmotionClassifier:
    """Classifies facial emotions using a lightweight CNN model (FER-based)."""

    def __init__(self, model_path: str = "models/emotion/emotion_model.onnx"):
        self.model_path = model_path
        self.model = None
        self.input_size = (48, 48)

    def load(self):
        """Load the ONNX emotion classification model."""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
            logger.info(f"[Emotion] Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"[Emotion] Failed to load model: {e}")
            raise

    def preprocess(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess a face ROI for the emotion model."""
        import cv2
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        resized = cv2.resize(gray, self.input_size)
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, 1, *self.input_size)

    def predict(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Predict emotion probabilities from a face ROI.
        
        Args:
            face_roi: Cropped face image (BGR or grayscale).
            
        Returns:
            Dictionary mapping emotion labels to probabilities.
        """
        if self.model is None:
            self.load()

        input_data = self.preprocess(face_roi)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_data})

        # Softmax
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        result = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}
        logger.debug(f"[Emotion] Predictions: {result}")
        return result

    def predict_label(self, face_roi: np.ndarray) -> str:
        """Get the top predicted emotion label."""
        probs = self.predict(face_roi)
        return max(probs, key=probs.get)

    def unload(self):
        """Unload the model."""
        self.model = None
        logger.info("[Emotion] Model unloaded.")
