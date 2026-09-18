#!/usr/bin/env python3
"""Benchmark script for ASR (speech-to-text) performance."""

import sys
import time
import json
import wave
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from asr import VoskRecognizer


def benchmark_asr(test_wav_dir: str = "benchmarks/test_audio"):
    """Run ASR benchmarks."""
    print("=" * 60)
    print("ASR (Speech-to-Text) Benchmark")
    print("=" * 60)

    recognizer = VoskRecognizer()

    # Measure load time
    start = time.perf_counter()
    recognizer.load()
    load_time = time.perf_counter() - start
    print(f"\nModel load time: {load_time:.2f}s")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": recognizer.model_path,
        "load_time_s": round(load_time, 3),
        "files": [],
    }

    test_dir = Path(test_wav_dir)
    if not test_dir.exists():
        print(f"\nNo test audio directory found at: {test_wav_dir}")
        print("Create WAV files (16kHz, mono, 16-bit) in that directory to run benchmarks.")
        return

    wav_files = sorted(test_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found for benchmarking.")
        return

    for wav_path in wav_files:
        print(f"\nProcessing: {wav_path.name}")
        with wave.open(str(wav_path), "rb") as wf:
            audio_data = wf.readframes(wf.getnframes())
            duration = wf.getnframes() / wf.getframerate()

        start = time.perf_counter()
        text = recognizer.recognize(audio_data)
        elapsed = time.perf_counter() - start

        result = {
            "file": wav_path.name,
            "audio_duration_s": round(duration, 2),
            "recognition_time_s": round(elapsed, 3),
            "real_time_factor": round(elapsed / duration, 3) if duration > 0 else 0,
            "recognized_text": text,
        }
        results["files"].append(result)
        print(f"   Duration: {duration:.1f}s | Recognition: {elapsed:.2f}s | RTF: {result['real_time_factor']:.3f}")
        print(f"   Text: {text[:80]}")

    # Save results
    output_path = Path(__file__).parent / "results" / f"asr_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    recognizer.unload()


if __name__ == "__main__":
    benchmark_asr()
