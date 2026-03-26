#!/usr/bin/env python3
"""Benchmark: Python (multiprocessing) vs Rust (Rayon) parallel processing."""

import time
import wave
import numpy as np
from multiprocessing import Pool
from pathlib import Path

def python_mix_file(args):
    """Process a single file pair with Python."""
    base_path, add_path, output_path, position = args
    with wave.open(base_path, 'rb') as w:
        sr = w.getframerate()
        base = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    with wave.open(add_path, 'rb') as w:
        add = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    result = base.copy().astype(np.int32)
    end_pos = min(position + len(add), len(result))
    add_len = end_pos - position
    result[position:end_pos] = (result[position:end_pos] + add[:add_len].astype(np.int32)) // 2
    result = np.clip(result, -32768, 32767).astype(np.int16)

    with wave.open(output_path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(result.tobytes())

def main():
    import rust_signals

    input_dir = Path(__file__).parent.parent / "test-signals" / "inputSignals"
    base_files = sorted(input_dir.glob("*.wav"))
    output_dir = Path("/tmp/bench_parallel")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Benchmark: Python (multiprocessing) vs Rust (Rayon)")
    print("=" * 70)

    for batch_size in [10, 100, 1000]:
        print(f"\n--- Batch size: {batch_size} files ---")

        file_pairs = []
        for i in range(batch_size):
            base = str(base_files[i % len(base_files)])
            add = str(base_files[(i + 1) % len(base_files)])
            output = str(output_dir / f"out_{i}.wav")
            file_pairs.append((base, add, output))

        # Python parallel
        args = [(b, a, o, 10000) for b, a, o in file_pairs]
        start = time.perf_counter()
        with Pool(8) as pool:
            pool.map(python_mix_file, args)
        py_time = time.perf_counter() - start
        py_rate = batch_size / py_time
        print(f"  Python (8 workers):  {py_time:7.3f}s  ({py_rate:7.1f} files/sec)")

        # Rust parallel
        start = time.perf_counter()
        rust_signals.batch_mix_files(file_pairs, 10000, 0.5, 0, False, 8)
        rust_time = time.perf_counter() - start
        rust_rate = batch_size / rust_time
        speedup = py_time / rust_time
        print(f"  Rust (8 threads):    {rust_time:7.3f}s  ({rust_rate:7.1f} files/sec)  -> {speedup:.1f}x faster")

    # Cleanup
    for f in output_dir.glob("*.wav"):
        f.unlink()

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
