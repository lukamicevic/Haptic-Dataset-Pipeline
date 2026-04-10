# Haptic Dataset Pipeline

## Setup

```bash
# Install maturin
pip install maturin

# Build and install
cd rust-signals
maturin develop --release
cd ..
```

## Usage

```python
import rust_signals
import numpy as np

# Load a signal (16-bit PCM)
signal = np.array([...], dtype=np.int16)
```

## Operations

### Signal Combining

| Function | Parameters | Description |
|----------|------------|-------------|
| `mix_signals` | `base, add, position, mix_balance=0.5, add_offset=0, normalize=False` | Weighted blend of two signals |
| `insert_signal` | `base, add, position, add_offset=0` | Insert signal at position (expands length) |
| `replace_signal` | `base, add, position, add_offset=0` | Overwrite samples at position |

### Signal Separating

| Function | Parameters | Description |
|----------|------------|-------------|
| `unmix_signal` | `combined, signal, position, mix_balance=0.5` | Reverse a mix operation |
| `remove_signal` | `combined, signal, position` | Reverse an insert operation |

### Signal Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `butterworth_lowpass` | `signal, cutoff_hz, sample_rate=44100, resonance=1.0` | Biquad lowpass filter |
| `add_noise` | `signal, noise_level=0.1` | Add white noise (0.0-1.0) |
| `roughen_signal` | `signal, phase_shift=5, intensity=0.5` | Add texture via phase-shifted copy |
| `scale_amplitude` | `signal, factor` | Scale amplitude (>1 louder, <1 quieter) |
| `normalize_signal` | `signal, target_peak=None` | Normalize to peak (default: 32767) |
| `mask_signal` | `signal, start=None, length, mask_value=0` | Silence a portion of signal |
| `downsample_signal` | `signal, factor` | Reduce resolution by factor |

### File-Based Convenience

| Function | Parameters |
|----------|------------|
| `combine_signals_from_files` | `base_path, add_path, output_path, position, operation, mix_balance=0.5, add_offset=0, normalize=False` |
| `separate_signals_from_files` | `combined_path, signal_path, output_path, position, operation, mix_balance=0.5` |

### Batch Processing (Parallel)

| Function | Parameters |
|----------|------------|
| `batch_mix_files` | `file_pairs, position, mix_balance=0.5, add_offset=0, normalize=False, num_threads=None` |
| `batch_insert_files` | `file_pairs, position, add_offset=0, num_threads=None` |
| `batch_replace_files` | `file_pairs, position, add_offset=0, num_threads=None` |
| `batch_unmix_files` | `file_pairs, position, mix_balance=0.5, num_threads=None` |
| `batch_remove_files` | `file_pairs, position, num_threads=None` |

`file_pairs` format: `[(base_path, add_path, output_path), ...]`
