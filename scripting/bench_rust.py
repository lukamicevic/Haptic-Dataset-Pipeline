#!/usr/bin/env python3
"""Benchmark script for Rust signal processing implementation."""

import argparse

try:
    import rust_signals
except ImportError:
    print("Error: rust_signals module not found.")
    print("Build it with: cd rust-signals && maturin develop --release")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Rust signal processing')
    parser.add_argument('--op', required=True, choices=['mix', 'insert', 'unmix', 'remove'],
                        help='Operation to perform')
    parser.add_argument('--pos', type=int, required=True, help='Position in samples')
    parser.add_argument('--balance', type=float, default=0.5, help='Mix balance (0.0-1.0)')
    parser.add_argument('--offset', type=int, default=0, help='Add signal offset')
    parser.add_argument('--input1', required=True, help='Base/combined signal path')
    parser.add_argument('--input2', required=True, help='Add/remove signal path')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--normalize', action='store_true', help='Normalize mix output')
    args = parser.parse_args()

    # Process
    if args.op in ['mix', 'insert']:
        rust_signals.combine_signals_from_files(
            args.input1,
            args.input2,
            args.output,
            args.pos,
            args.op,
            args.balance,
            args.offset,
            args.normalize
        )
    else:  # unmix, remove
        rust_signals.separate_signals_from_files(
            args.input1,
            args.input2,
            args.output,
            args.pos,
            args.op,
            args.balance
        )


if __name__ == "__main__":
    main()
