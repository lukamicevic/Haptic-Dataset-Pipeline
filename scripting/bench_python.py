#!/usr/bin/env python3
"""Benchmark script for Python signal processing implementation."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wav_IO import load_wav, save_wav

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Signal-Augmentations'))
from addSignals import combine_signals
from dropSignals import separate_signals


def main():
    parser = argparse.ArgumentParser(description='Benchmark Python signal processing')
    parser.add_argument('--op', required=True, choices=['mix', 'insert', 'unmix', 'remove'],
                        help='Operation to perform')
    parser.add_argument('--pos', type=int, required=True, help='Position in samples')
    parser.add_argument('--balance', type=float, default=0.5, help='Mix balance (0.0-1.0)')
    parser.add_argument('--offset', type=int, default=0, help='Add signal offset')
    parser.add_argument('--input1', required=True, help='Base/combined signal path')
    parser.add_argument('--input2', required=True, help='Add/remove signal path')
    parser.add_argument('--output', required=True, help='Output path')
    args = parser.parse_args()

    # Load signals
    signal1, sample_rate = load_wav(args.input1)
    signal2, _ = load_wav(args.input2)

    # Process
    if args.op in ['mix', 'insert']:
        result = combine_signals(
            base=signal1,
            add=signal2,
            mode=args.op,
            position=args.pos,
            mix_balance=args.balance,
            add_offset=args.offset
        )
    else:  # unmix, remove
        result = separate_signals(
            combined=signal1,
            signal_to_remove=signal2,
            mode=args.op,
            position=args.pos,
            mix_balance=args.balance
        )

    # Save
    save_wav(args.output, result, sample_rate)


if __name__ == "__main__":
    main()
