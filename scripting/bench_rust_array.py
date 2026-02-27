#!/usr/bin/env python3
"""Benchmark Rust with NumPy array API (no file I/O in Rust)."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wav_IO import load_wav, save_wav

try:
    import rust_signals
except ImportError:
    print("Error: rust_signals module not found.")
    print("Build it with: cd rust-signals && maturin develop --release")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Rust array-based signal processing')
    parser.add_argument('--op', required=True, choices=['mix', 'insert', 'unmix', 'remove'],
                        help='Operation to perform')
    parser.add_argument('--pos', type=int, required=True, help='Position in samples')
    parser.add_argument('--balance', type=float, default=0.5, help='Mix balance (0.0-1.0)')
    parser.add_argument('--offset', type=int, default=0, help='Add signal offset')
    parser.add_argument('--input1', required=True, help='Base/combined signal path')
    parser.add_argument('--input2', required=True, help='Add/remove signal path')
    parser.add_argument('--output', required=True, help='Output path')
    args = parser.parse_args()

    # Load with Python/NumPy (optimized)
    base, sr = load_wav(args.input1)
    add, _ = load_wav(args.input2)

    # Process with Rust (pure computation, no I/O)
    if args.op == 'mix':
        result = rust_signals.mix_signals(base, add, args.pos, args.balance, args.offset)
    elif args.op == 'insert':
        result = rust_signals.insert_signal(base, add, args.pos, args.offset)
    elif args.op == 'unmix':
        result = rust_signals.unmix_signal(base, add, args.pos, args.balance)
    elif args.op == 'remove':
        result = rust_signals.remove_signal(base, add, args.pos)

    # Save with Python/NumPy (optimized)
    save_wav(args.output, result, sr)


if __name__ == "__main__":
    main()
