import numpy as np
from typing import Literal
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from wav_IO import load_wav, save_wav


def combine_signals(
    base: np.ndarray,
    add: np.ndarray,
    mode: Literal["insert", "mix"],
    position: int,
    mix_balance: float = 0.5,
    add_offset: int = 0
) -> np.ndarray:

    add = add[add_offset:]

    if mode == "insert":
        return np.concatenate([base[:position], add, base[position:]])

    if mode == "mix":
        result = base.copy()
        end = position + len(add)
        
        if end > len(result):
            result = np.pad(result, (0, end - len(result)))
        
        weight_base = 1.0 - mix_balance
        weight_add = mix_balance
        
        result[position:end] = weight_base * result[position:end] + weight_add * add

        return result


if __name__ == "__main__":
    MODE = "mix"
    POSITION = 50000
    MIX_BALANCE = 0.25
    ADD_OFFSET = 88200

    base_path = os.path.join(parent_dir, 'test-signals', 'inputSignals', '1-19840-A-36.wav')
    add_path = os.path.join(parent_dir, 'test-signals', 'inputSignals', '1-9887-A-49.wav')

    base_signal, sample_rate = load_wav(base_path)
    add_signal, _ = load_wav(add_path)

    result = combine_signals(base_signal, add_signal, MODE, POSITION, MIX_BALANCE, ADD_OFFSET)

    output_dir = os.path.join(parent_dir, 'test-signals', 'outputSignals')
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"output_{MODE}_pos{POSITION}_bal{MIX_BALANCE}_off{ADD_OFFSET}.wav"
    output_path = os.path.join(output_dir, output_filename)

    save_wav(output_path, result, sample_rate)   