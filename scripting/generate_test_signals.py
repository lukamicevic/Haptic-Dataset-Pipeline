#!/usr/bin/env python3
"""Generate synthetic test signals of various sizes for benchmarking."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wav_IO import save_wav

SAMPLE_RATE = 44100

def generate_signal(num_samples: int, seed: int = 42) -> np.ndarray:
    """Generate a synthetic signal with sine waves and noise."""
    np.random.seed(seed)
    t = np.arange(num_samples) / SAMPLE_RATE

    # Mix of frequencies typical in haptic signals
    signal = (
        0.3 * np.sin(2 * np.pi * 100 * t) +   # 100 Hz
        0.2 * np.sin(2 * np.pi * 250 * t) +   # 250 Hz
        0.15 * np.sin(2 * np.pi * 500 * t) +  # 500 Hz
        0.1 * np.random.randn(num_samples)     # noise
    )

    # Normalize to i16 range with some headroom
    signal = signal / np.max(np.abs(signal)) * 28000
    return signal.astype(np.int16)


def main():
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'test-signals', 'inputSignals'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Size definitions (approximate file sizes)
    # WAV file size ~= 2 * num_samples + 44 bytes header
    sizes = {
        'medium': 5 * 1024 * 1024 // 2,   # ~5 MB -> ~2.5M samples
        'large': 25 * 1024 * 1024 // 2,   # ~25 MB -> ~12.5M samples
    }

    for size_name, num_samples in sizes.items():
        for i in [1, 2]:
            filename = f'bench_{size_name}_{i}.wav'
            filepath = os.path.join(output_dir, filename)

            print(f"Generating {filename} ({num_samples:,} samples, ~{num_samples * 2 / 1024 / 1024:.1f} MB)...")
            signal = generate_signal(num_samples, seed=42 + i)
            save_wav(filepath, signal, SAMPLE_RATE)
            print(f"  Saved: {filepath}")

    print("\nDone! Generated test signals:")
    print("  - medium: ~5 MB files (bench_medium_1.wav, bench_medium_2.wav)")
    print("  - large: ~25 MB files (bench_large_1.wav, bench_large_2.wav)")
    print("\nExisting small files can be used from:")
    print("  - 1-19840-A-36.wav (~430 KB)")
    print("  - 1-9887-A-49.wav (~430 KB)")


if __name__ == "__main__":
    main()
