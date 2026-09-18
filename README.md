# 🤖 Offline AI Companion

An offline, privacy-first AI companion robot that combines speech recognition, emotion detection, natural language generation, and text-to-speech — all running locally without internet.

## 📁 Project Structure

```
ai-companion-robot/
├── llm/                    # LLM wrapper (generate(prompt) interface)
├── conversation/           # Conversation manager — the switchboard
├── asr/                    # Offline speech-to-text (Vosk)
├── tts/                    # Offline text-to-speech (Piper)
├── vision/                 # Camera capture + face detection
├── emotion/                # Emotion classification + smoothing
├── memory/                 # Conversation history storage
├── audio/                  # Low-level mic/speaker I/O
├── config/                 # Central settings
├── benchmarks/             # Benchmark scripts + results
├── tests/                  # Unit tests
├── docs/                   # Project documentation
├── main/                   # Entry point
├── models/                 # Model weights (gitignored)
└── logs/                   # Runtime logs (gitignored)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Microphone and camera
- ~4GB RAM (for LLM inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Offline_Companion.git
cd Offline_Companion

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
```

### Download Models

Place model files in the `models/` directory:
- **LLM**: GGUF model → `models/llm/model.gguf`
- **ASR**: Vosk model → `models/vosk/vosk-model-small-en-us/`
- **TTS**: Piper ONNX model → `models/tts/en_US-lessac-medium.onnx`
- **Emotion**: ONNX classifier → `models/emotion/emotion_model.onnx`

### Run

```bash
python main/run.py
```

## 🧪 Testing

```bash
python -m pytest tests/
```

## 📊 Benchmarks

```bash
python benchmarks/bench_llm.py
python benchmarks/bench_asr.py
python benchmarks/bench_tts.py
```

## 📄 License

This project is part of a Final Year Project (FYP) for BSCS.
