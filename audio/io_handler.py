"""Audio I/O Handler — manages microphone input and speaker output."""

import logging
import wave
import struct
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class AudioIOHandler:
    """Handles low-level audio input/output operations."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 4096):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.pa = None
        self.input_stream = None
        self.output_stream = None

    def initialize(self):
        """Initialize PyAudio."""
        try:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            logger.info("[Audio] PyAudio initialized.")
        except ImportError:
            logger.error("[Audio] pyaudio not installed. Run: pip install pyaudio")
            raise

    def start_recording(self):
        """Open the microphone input stream."""
        import pyaudio
        if self.pa is None:
            self.initialize()

        self.input_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        logger.info("[Audio] Recording started.")

    def stop_recording(self):
        """Close the microphone input stream."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            logger.info("[Audio] Recording stopped.")

    def read_chunk(self) -> Optional[bytes]:
        """Read a single chunk of audio from the microphone."""
        if self.input_stream is None:
            logger.warning("[Audio] No active input stream.")
            return None
        try:
            return self.input_stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception as e:
            logger.error(f"[Audio] Read error: {e}")
            return None

    def audio_stream(self, duration: float = None) -> Generator[bytes, None, None]:
        """Generate a stream of audio chunks.

        Args:
            duration: Recording duration in seconds. None for indefinite.

        Yields:
            Audio data chunks as bytes.
        """
        self.start_recording()
        chunks_recorded = 0
        max_chunks = None
        if duration:
            max_chunks = int(self.sample_rate / self.chunk_size * duration)

        try:
            while True:
                chunk = self.read_chunk()
                if chunk:
                    yield chunk
                    chunks_recorded += 1
                if max_chunks and chunks_recorded >= max_chunks:
                    break
        finally:
            self.stop_recording()

    def play_wav(self, wav_path: str):
        """Play a WAV file through the speakers."""
        import pyaudio
        if self.pa is None:
            self.initialize()

        try:
            wf = wave.open(wav_path, "rb")
            stream = self.pa.open(
                format=self.pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            data = wf.readframes(self.chunk_size)
            while data:
                stream.write(data)
                data = wf.readframes(self.chunk_size)

            stream.stop_stream()
            stream.close()
            wf.close()
            logger.info(f"[Audio] Played: {wav_path}")
        except Exception as e:
            logger.error(f"[Audio] Playback error: {e}")

    def get_energy(self, audio_chunk: bytes) -> float:
        """Calculate the RMS energy of an audio chunk (for VAD)."""
        samples = struct.unpack(f"<{len(audio_chunk)//2}h", audio_chunk)
        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
        return rms

    def cleanup(self):
        """Clean up all audio resources."""
        self.stop_recording()
        if self.pa:
            self.pa.terminate()
            self.pa = None
            logger.info("[Audio] Resources cleaned up.")
