import numpy as np
from typing import Literal
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from wav_IO import load_wav, save_wav


def separate_signals(
    combined: np.ndarray,
    signal_to_remove: np.ndarray,
    mode: Literal["remove", "unmix"],
    position: int,
    mix_balance: float = 0.5
) -> np.ndarray:

    if mode == "remove":
        end = position + len(signal_to_remove)
        return np.concatenate([combined[:position], combined[end:]])

    if mode == "unmix":
        result = combined.copy()
        end = min(position + len(signal_to_remove), len(combined))

        overlap_length = end - position
        signal_slice = signal_to_remove[:overlap_length]

        weight_add = mix_balance
        weight_base = 1.0 - mix_balance

        if weight_base == 0:
            result[position:end] = 0.0
        else:
            result[position:end] = (combined[position:end] - weight_add * signal_slice) / weight_base

        return result


if __name__ == "__main__":
    MODE = "unmix"
    POSITION = 0
    MIX_BALANCE = 0.5

    combined_path = os.path.join(parent_dir, 'test-signals', 'outputSignals', 'output_mix_pos50000_bal0.5.wav')
    remove_path = os.path.join(parent_dir, 'test-signals', 'inputSignals', '1-19840-A-36.wav')

    combined_signal, sample_rate = load_wav(combined_path)
    remove_signal, _ = load_wav(remove_path)

    result = separate_signals(combined_signal, remove_signal, MODE, POSITION, MIX_BALANCE)

    output_dir = os.path.join(parent_dir, 'test-signals', 'outputSignals')
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"output_{MODE}_pos{POSITION}_bal{MIX_BALANCE}.wav"
    output_path = os.path.join(output_dir, output_filename)

    save_wav(output_path, result, sample_rate)
