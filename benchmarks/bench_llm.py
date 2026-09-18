#!/usr/bin/env python3
"""Benchmark script for LLM inference performance."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import ModelLoader


TEST_PROMPTS = [
    "Hello, how are you today?",
    "Tell me a short joke.",
    "What is the meaning of life in one sentence?",
    "Explain photosynthesis to a five-year-old.",
    "Write a haiku about artificial intelligence.",
]


def benchmark_llm():
    """Run LLM benchmarks and record results."""
    print("=" * 60)
    print("LLM Inference Benchmark")
    print("=" * 60)

    loader = ModelLoader()

    # Measure load time
    start = time.perf_counter()
    loader.load()
    load_time = time.perf_counter() - start
    print(f"\nModel load time: {load_time:.2f}s")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": loader.model_path,
        "load_time_s": round(load_time, 3),
        "prompts": [],
    }

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: {prompt[:50]}...")
        start = time.perf_counter()
        response = loader.generate_response(prompt)
        elapsed = time.perf_counter() - start
        tokens_est = len(response.split())

        result = {
            "prompt": prompt,
            "response_length_chars": len(response),
            "response_length_tokens_est": tokens_est,
            "inference_time_s": round(elapsed, 3),
            "tokens_per_second_est": round(tokens_est / elapsed, 1) if elapsed > 0 else 0,
        }
        results["prompts"].append(result)
        print(f"   Time: {elapsed:.2f}s | ~{tokens_est} tokens | ~{result['tokens_per_second_est']} tok/s")

    # Calculate averages
    avg_time = sum(r["inference_time_s"] for r in results["prompts"]) / len(results["prompts"])
    avg_tps = sum(r["tokens_per_second_est"] for r in results["prompts"]) / len(results["prompts"])
    results["avg_inference_time_s"] = round(avg_time, 3)
    results["avg_tokens_per_second"] = round(avg_tps, 1)

    # Save results
    output_path = Path(__file__).parent / "results" / f"llm_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Average inference time: {avg_time:.2f}s")
    print(f"Average tokens/second:  {avg_tps:.1f}")
    print(f"Results saved to: {output_path}")

    loader.unload()


if __name__ == "__main__":
    benchmark_llm()
