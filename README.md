# Haptic Dataset Pipeline

A high-performance data augmentation pipeline for generating synthetic haptic signal datasets for AI training. 
## Overview

This pipeline provides fast, reversible signal augmentation operations for haptic data:
- **Mix/Unmix**: Blend signals with weighted balance (reversible)
- **Insert/Remove**: Concatenate signals at specific positions (reversible)

Available in both **Python** (reference implementation) and **Rust** (production-ready, 10-100x faster).

## Quick Start

### Option 1: Rust (Recommended for Performance)

```bash
# 1. Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Install maturin
pip install maturin

# 3. Build Rust module
cd rust-signals
maturin develop --release
cd ..

# 4. Use in Python
python
>>> import rust_signals
>>> rust_signals.combine_signals_from_files(
...     base_path="test-signals/inputSignals/1-19840-A-36.wav",
...     add_path="test-signals/inputSignals/1-9887-A-49.wav",
...     output_path="test-signals/outputSignals/output.wav",
...     position=50000,
...     operation="mix",
...     mix_balance=0.5
... )
```

### Option 2: Python (Simpler Setup)

```python
from Signal_Augmentations.addSignals import combine_signals
from Signal_Augmentations.dropSignals import separate_signals
from wav_IO import load_wav, save_wav

# Load signals
base, sr = load_wav("test-signals/inputSignals/1-19840-A-36.wav")
add, _ = load_wav("test-signals/inputSignals/1-9887-A-49.wav")

# Mix signals
result = combine_signals(base, add, "mix", position=50000, mix_balance=0.5)
save_wav("output.wav", result, sr)
```

## Testing

```bash
# Test Rust implementation
python test_rust_signals.py
```

## Project Structure

```
Haptic-Dataset-Pipeline/
├── rust-signals/              # Rust implementation (fast, production-ready)
│   ├── src/
│   │   ├── lib.rs            # Python bindings
│   │   ├── wav_io.rs         # WAV file I/O
│   │   ├── signal_ops.rs     # Signal processing
│   │   └── types.rs          # Operation types
│   ├── Cargo.toml
│   └── pyproject.toml
├── Signal-Augmentations/      # Python implementation (reference)
│   ├── addSignals.py         # Combine operations
│   └── dropSignals.py        # Separate operations
├── wav_IO.py                  # Python WAV utilities
├── test-signals/              # Test data
│   ├── inputSignals/
│   └── outputSignals/
└── test_rust_signals.py       # Test suite
```

## Operations

### Combine Signals

**Mix** - Weighted blend of two signals:
```python
# Rust
rust_signals.combine_signals_from_files(
    base_path="base.wav",
    add_path="add.wav",
    output_path="mixed.wav",
    position=10000,
    operation="mix",
    mix_balance=0.3  # 70% base, 30% add
)

# Python
result = combine_signals(base, add, "mix", 10000, mix_balance=0.3)
```

**Insert** - Concatenate signals:
```python
# Rust
rust_signals.combine_signals_from_files(
    base_path="base.wav",
    add_path="add.wav",
    output_path="inserted.wav",
    position=5000,
    operation="insert"
)

# Python
result = combine_signals(base, add, "insert", 5000)
```

### Separate Signals

**Unmix** - Reverse a mix operation:
```python
# Rust
rust_signals.separate_signals_from_files(
    combined_path="mixed.wav",
    signal_path="add.wav",
    output_path="recovered.wav",
    position=10000,
    operation="unmix",
    mix_balance=0.3  # Must match original mix
)

# Python
result = separate_signals(mixed, add, "unmix", 10000, mix_balance=0.3)
```

**Remove** - Reverse an insert operation:
```python
# Rust
rust_signals.separate_signals_from_files(
    combined_path="inserted.wav",
    signal_path="add.wav",
    output_path="recovered.wav",
    position=5000,
    operation="remove"
)

# Python
result = separate_signals(inserted, add, "remove", 5000)
```

## Performance

Rust implementation performance vs Python:
- Small signals (<1MB): **10-50x faster**
- Large signals (>10MB): **50-100x faster**
- Zero-copy operations where possible
- Efficient memory management

## Requirements

### Rust Version
- Rust toolchain (latest stable)
- Python 3.8+
- maturin

### Python Version
- Python 3.8+
- NumPy
- SciPy (optional)

## Use Cases

- **Data Augmentation**: Generate synthetic training data for haptic ML models
- **Signal Analysis**: Study effects of signal mixing on haptic perception
- **Dataset Generation**: Create large-scale augmented haptic datasets efficiently
- **Research**: Experiment with different augmentation strategies

## Future Extensions

The enum-based architecture makes it easy to add new operations:
- Crossfade
- Time-stretch
- Pitch-shift
- Noise injection
- Frequency filtering

## Acknowledgments

Based on AUDIT research methodology, adapted for haptic signal processing.
