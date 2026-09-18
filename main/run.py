#!/usr/bin/env python3
"""Main Entry Point — wires all modules together and starts the AI companion."""

import sys
import logging
import yaml
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm import ModelLoader
from conversation import ConversationManager
from asr import VoskRecognizer
from tts import PiperSynth
from vision import FaceDetector
from emotion import EmotionClassifier, EmotionSmoother
from memory import SessionMemory
from audio import AudioIOHandler


def setup_logging(config: dict):
    """Configure logging."""
    log_level = config.get("log_level", "INFO")
    log_file = config.get("log_file", "logs/companion.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def load_config() -> dict:
    """Load central configuration."""
    config_path = project_root / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Initialize all modules and start the companion."""
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger("main")
    logger.info(f"Starting {config.get('app_name', 'AI Companion')}...")

    try:
        # Initialize modules
        logger.info("Loading LLM...")
        llm = ModelLoader(config["llm"]["model_path"])
        llm.load()

        logger.info("Loading ASR...")
        asr = VoskRecognizer(
            model_path=config["asr"]["model_path"],
            sample_rate=config["asr"]["sample_rate"],
        )
        asr.load()

        logger.info("Loading TTS...")
        tts = PiperSynth(
            model_path=config["tts"]["model_path"],
            piper_binary=config["tts"]["piper_binary"],
        )

        logger.info("Initializing vision...")
        face_detector = FaceDetector(
            camera_index=config["vision"]["camera_index"],
            use_dnn=config["vision"]["use_dnn"],
        )
        face_detector.initialize()

        logger.info("Loading emotion classifier...")
        emotion_classifier = EmotionClassifier(
            model_path=config["emotion"]["model_path"]
        )
        emotion_classifier.load()
        emotion_smoother = EmotionSmoother(
            window_size=config["emotion"]["smoothing_window"],
            confidence_threshold=config["emotion"]["confidence_threshold"],
        )

        logger.info("Setting up memory...")
        memory = SessionMemory(
            max_history=config["memory"]["max_history"],
            persist_path=f"{config['memory']['session_dir']}/current_session.json"
            if config["memory"]["persist_sessions"]
            else None,
        )

        # Wire everything together
        conversation = ConversationManager(
            llm=llm,
            asr=asr,
            tts=tts,
            emotion=emotion_classifier,
            memory=memory,
        )

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down...")
            conversation.stop()
            face_detector.release()
            llm.unload()
            asr.unload()
            emotion_classifier.unload()
            tts.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the conversation loop
        logger.info("AI Companion is ready! Listening...")
        conversation.start()

        while conversation.is_active:
            response = conversation.full_turn()
            if response:
                logger.info(f"Response: {response}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
