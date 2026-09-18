"""Piper-based Offline Text-to-Speech Synthesizer."""

import subprocess
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PiperSynth:
    """Offline text-to-speech using Piper TTS."""

    def __init__(
        self,
        model_path: str = "models/tts/en_US-lessac-medium.onnx",
        piper_binary: str = "piper",
        output_dir: str = "temp_audio",
        sample_rate: int = 22050,
    ):
        self.model_path = model_path
        self.piper_binary = piper_binary
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, output_path: str = None) -> str:
        """Synthesize speech from text and save to a WAV file.
        
        Args:
            text: The text to synthesize.
            output_path: Optional output file path. Auto-generated if None.
            
        Returns:
            Path to the generated WAV file.
        """
        if not output_path:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", dir=str(self.output_dir), delete=False
            )
            output_path = tmp.name
            tmp.close()

        try:
            cmd = [
                self.piper_binary,
                "--model", self.model_path,
                "--output_file", output_path,
            ]
            process = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if process.returncode != 0:
                logger.error(f"[TTS] Piper error: {process.stderr}")
                return ""

            logger.info(f"[TTS] Synthesized audio saved to {output_path}")
            return output_path

        except FileNotFoundError:
            logger.error(f"[TTS] Piper binary not found at: {self.piper_binary}")
            return ""
        except subprocess.TimeoutExpired:
            logger.error("[TTS] Synthesis timed out.")
            return ""

    def speak(self, text: str):
        """Synthesize and immediately play the audio."""
        wav_path = self.synthesize(text)
        if wav_path:
            self._play_audio(wav_path)

    def _play_audio(self, wav_path: str):
        """Play a WAV file using available system tools."""
        try:
            import sounddevice as sd
            import soundfile as sf

            data, samplerate = sf.read(wav_path)
            sd.play(data, samplerate)
            sd.wait()
        except ImportError:
            # Fallback to aplay on Linux
            try:
                subprocess.run(["aplay", wav_path], check=True, capture_output=True)
            except Exception as e:
                logger.error(f"[TTS] Playback error: {e}")

    def cleanup(self):
        """Remove temporary audio files."""
        for f in self.output_dir.glob("*.wav"):
            try:
                f.unlink()
            except OSError:
                pass
        logger.info("[TTS] Temporary audio files cleaned up.")
