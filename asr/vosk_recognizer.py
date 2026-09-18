"""Vosk-based Offline Speech Recognizer."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VoskRecognizer:
    """Offline speech-to-text using the Vosk library."""

    def __init__(self, model_path: str = "models/vosk/vosk-model-small-en-us", sample_rate: int = 16000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.model = None
        self.recognizer = None

    def load(self):
        """Load the Vosk model."""
        try:
            from vosk import Model, KaldiRecognizer
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Vosk model not found at: {self.model_path}")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            logger.info(f"[ASR] Vosk model loaded from {self.model_path}")
        except ImportError:
            logger.error("[ASR] vosk package not installed. Run: pip install vosk")
            raise

    def recognize(self, audio_data: bytes = None) -> str:
        """Recognize speech from audio data.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono, 16kHz).
                       If None, records from microphone.
                       
        Returns:
            Recognized text string.
        """
        if self.recognizer is None:
            self.load()

        if audio_data is None:
            audio_data = self._record_from_mic()

        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
        else:
            result = json.loads(self.recognizer.PartialResult())
            text = result.get("partial", "").strip()

        logger.info(f"[ASR] Recognized: '{text}'")
        return text

    def recognize_stream(self, audio_stream):
        """Recognize speech from a continuous audio stream.
        
        Args:
            audio_stream: Iterator yielding audio chunks.
            
        Yields:
            Recognized text segments.
        """
        if self.recognizer is None:
            self.load()

        for chunk in audio_stream:
            if self.recognizer.AcceptWaveform(chunk):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    yield text

        # Get final result
        final = json.loads(self.recognizer.FinalResult())
        text = final.get("text", "").strip()
        if text:
            yield text

    def _record_from_mic(self, duration: float = 5.0) -> bytes:
        """Record audio from the microphone."""
        try:
            import pyaudio
            import struct

            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=4096,
            )
            logger.info(f"[ASR] Recording for {duration}s...")
            frames = []
            for _ in range(0, int(self.sample_rate / 4096 * duration)):
                data = stream.read(4096, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            pa.terminate()
            return b"".join(frames)
        except Exception as e:
            logger.error(f"[ASR] Microphone error: {e}")
            return b""

    def unload(self):
        """Unload the model."""
        self.model = None
        self.recognizer = None
        logger.info("[ASR] Model unloaded.")
