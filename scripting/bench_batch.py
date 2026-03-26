#!/usr/bin/env python3
"""Benchmark batch processing: sequential vs parallel (Rayon)."""

import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import rust_signals
except ImportError:
    print("Error: rust_signals module not found.")
    print("Build it with: cd rust-signals && maturin develop --release")
    exit(1)


def benchmark_sequential(file_pairs, position):
    """Process files one at a time using the single-file API."""
    start = time.perf_counter()
    for base, add, output in file_pairs:
        rust_signals.combine_signals_from_files(
            base, add, output, position, "mix", 0.5, 0, False
        )
    return time.perf_counter() - start


def benchmark_batch(file_pairs, position, num_threads=None):
    """Process files in parallel with Rayon batch API."""
    start = time.perf_counter()
    results = rust_signals.batch_mix_files(
        file_pairs, position, 0.5, 0, False, num_threads
    )
    elapsed = time.perf_counter() - start
    failures = sum(1 for _, success, _ in results if not success)
    return elapsed, failures


def main():
    input_dir = Path(__file__).parent.parent / "test-signals" / "inputSignals"
    output_dir = Path("/tmp/batch_bench")
    output_dir.mkdir(exist_ok=True)

    # Get available test files
    base_files = sorted(input_dir.glob("*.wav"))
    if len(base_files) < 2:
        print(f"Error: Need at least 2 WAV files in {input_dir}")
        print("Run: python scripting/generate_test_signals.py")
        exit(1)

    print("=" * 60)
    print("Batch Processing Benchmark: Sequential vs Parallel (Rayon)")
    print("=" * 60)
    print(f"\nInput files found: {len(base_files)}")
    print(f"Output directory: {output_dir}\n")

    # Test with different batch sizes
    for batch_size in [10, 50, 100]:
        # Generate file pairs by cycling through available files
        file_pairs = []
        for i in range(batch_size):
            base = str(base_files[i % len(base_files)])
            add = str(base_files[(i + 1) % len(base_files)])
            output = str(output_dir / f"output_{i}.wav")
            file_pairs.append((base, add, output))

        print(f"--- Batch size: {batch_size} files ---")

        # Sequential benchmark
        seq_time = benchmark_sequential(file_pairs, 50000)
        seq_rate = batch_size / seq_time
        print(f"  Sequential:      {seq_time:6.3f}s  ({seq_rate:6.1f} files/sec)")

        # Parallel benchmark (auto threads)
        par_time, failures = benchmark_batch(file_pairs, 50000)
        par_rate = batch_size / par_time
        speedup = seq_time / par_time
        print(f"  Parallel (auto): {par_time:6.3f}s  ({par_rate:6.1f} files/sec)  -> {speedup:.2f}x speedup")

        if failures > 0:
            print(f"    Warning: {failures} files failed")

        # Test specific thread counts
        for threads in [2, 4, 8]:
            par_time, _ = benchmark_batch(file_pairs, 50000, threads)
            par_rate = batch_size / par_time
            speedup = seq_time / par_time
            print(f"  Parallel ({threads} thr): {par_time:6.3f}s  ({par_rate:6.1f} files/sec)  -> {speedup:.2f}x speedup")

        print()

    # Cleanup
    for f in output_dir.glob("*.wav"):
        f.unlink()

    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
