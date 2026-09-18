#!/usr/bin/env python3
"""Benchmark script for TTS (text-to-speech) performance."""

import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from tts import PiperSynth


TEST_TEXTS = [
    "Hello, how are you doing today?",
    "The quick brown fox jumps over the lazy dog.",
    "I really enjoyed our conversation about science and technology.",
    "Could you tell me more about your favorite hobbies and interests?",
    "That's a wonderful idea! Let me think about how we can make it happen together.",
]


def benchmark_tts():
    """Run TTS benchmarks."""
    print("=" * 60)
    print("TTS (Text-to-Speech) Benchmark")
    print("=" * 60)

    synth = PiperSynth(output_dir="benchmarks/results/tts_output")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": synth.model_path,
        "texts": [],
    }

    for i, text in enumerate(TEST_TEXTS, 1):
        print(f"\n[{i}/{len(TEST_TEXTS)}] Text: {text[:50]}...")
        start = time.perf_counter()
        wav_path = synth.synthesize(text)
        elapsed = time.perf_counter() - start

        file_size = os.path.getsize(wav_path) if wav_path and os.path.exists(wav_path) else 0

        result = {
            "text": text,
            "text_length_chars": len(text),
            "synthesis_time_s": round(elapsed, 3),
            "output_file_size_bytes": file_size,
            "chars_per_second": round(len(text) / elapsed, 1) if elapsed > 0 else 0,
        }
        results["texts"].append(result)
        print(f"   Time: {elapsed:.2f}s | {result['chars_per_second']} chars/s | Size: {file_size} bytes")

    # Calculate averages
    avg_time = sum(r["synthesis_time_s"] for r in results["texts"]) / len(results["texts"])
    avg_cps = sum(r["chars_per_second"] for r in results["texts"]) / len(results["texts"])
    results["avg_synthesis_time_s"] = round(avg_time, 3)
    results["avg_chars_per_second"] = round(avg_cps, 1)

    # Save results
    output_path = Path(__file__).parent / "results" / f"tts_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Average synthesis time: {avg_time:.2f}s")
    print(f"Average chars/second:  {avg_cps:.1f}")
    print(f"Results saved to: {output_path}")

    synth.cleanup()


if __name__ == "__main__":
    benchmark_tts()
