use crate::types::{CombineOp, SeparateOp};

pub fn combine_signals(
    base: &[i16],
    add: &[i16],
    position: usize,
    op: CombineOp,
) -> Vec<i16> {
    match op {
        CombineOp::Insert { add_offset } => {
            let add_slice = &add[add_offset..];
            let mut result = Vec::with_capacity(base.len() + add_slice.len());
            result.extend_from_slice(&base[..position]);
            result.extend_from_slice(add_slice);
            result.extend_from_slice(&base[position..]);
            result
        }
        CombineOp::Mix { mix_balance, add_offset, normalize } => {
            let add_slice = &add[add_offset..];
            let mut result = base.to_vec();
            let end = position + add_slice.len();

            // Pad if necessary
            if end > result.len() {
                result.resize(end, 0);
            }

            let weight_base = 1.0 - mix_balance;
            let weight_add = mix_balance;

            for (i, &add_sample) in add_slice.iter().enumerate() {
                let base_sample = result[position + i];
                result[position + i] = (
                    weight_base * base_sample as f32 +
                    weight_add * add_sample as f32
                ) as i16;
            }

            // Normalize to full dynamic range if requested
            if normalize {
                let max_abs = result.iter()
                    .map(|&x| x.abs())
                    .max()
                    .unwrap_or(1);

                if max_abs > 0 {
                    let scale = 32767.0 / max_abs as f32;
                    for sample in result.iter_mut() {
                        *sample = (*sample as f32 * scale) as i16;
                    }
                }
            }

            result
        }
        CombineOp::Replace { add_offset } => {
            let add_slice = &add[add_offset..];
            let mut result = base.to_vec();
            let end = (position + add_slice.len()).min(result.len());
            let copy_len = end - position;
            result[position..end].copy_from_slice(&add_slice[..copy_len]);
            result
        }
    }
}

pub fn separate_signals(
    combined: &[i16],
    signal: &[i16],
    position: usize,
    op: SeparateOp,
) -> Vec<i16> {
    match op {
        SeparateOp::Remove => {
            let end = position + signal.len();
            let mut result = Vec::with_capacity(combined.len() - signal.len());
            result.extend_from_slice(&combined[..position]);
            result.extend_from_slice(&combined[end..]);
            result
        }
        SeparateOp::Unmix { mix_balance } => {
            let mut result = combined.to_vec();
            let end = (position + signal.len()).min(combined.len());
            let overlap_length = end - position;

            let weight_add = mix_balance;
            let weight_base = 1.0 - mix_balance;

            if weight_base == 0.0 {
                for i in position..end {
                    result[i] = 0;
                }
            } else {
                for i in 0..overlap_length {
                    let combined_sample = combined[position + i] as f32;
                    let signal_sample = signal[i] as f32;
                    result[position + i] = (
                        (combined_sample - weight_add * signal_sample) / weight_base
                    ) as i16;
                }
            }

            result
        }
    }
}

// =============================================================================
// Training Data Generation - Inpainting and Super-resolution
// These create degraded versions of signals for AI training pairs:
// (degraded_signal, original_signal)
// =============================================================================

use rand::Rng;

/// Mask a portion of the signal (for inpainting training data)
/// Sets samples in the specified range to zero (or mask_value)
/// If start is None, picks a random position
pub fn mask_signal(signal: &[i16], start: Option<usize>, length: usize, mask_value: i16) -> (Vec<i16>, usize) {
    let mut result = signal.to_vec();

    // Determine start position (random if None)
    let actual_start = match start {
        Some(s) => s,
        None => {
            if signal.len() <= length {
                0
            } else {
                rand::thread_rng().gen_range(0..signal.len() - length)
            }
        }
    };

    let end = (actual_start + length).min(result.len());
    for i in actual_start..end {
        result[i] = mask_value;
    }
    (result, actual_start)
}

/// Downsample signal by integer factor (for super-resolution training data)
/// Takes every Nth sample, reducing length by factor
pub fn downsample_signal(signal: &[i16], factor: usize) -> Vec<i16> {
    if factor <= 1 || signal.is_empty() {
        return signal.to_vec();
    }
    signal.iter().step_by(factor).copied().collect()
}

/// Low-pass filter with cutoff frequency
/// cutoff: normalized frequency (0.0 to 0.5)
///   - 0.0 = DC only
///   - 0.5 = Nyquist (no filtering)
pub fn lowpass_filter(signal: &[i16], cutoff: f32) -> Vec<i16> {
    if signal.is_empty() || cutoff >= 0.5 {
        return signal.to_vec();
    }

    // Convert cutoff to alpha coefficient
    let alpha = 1.0 - (-2.0 * std::f32::consts::PI * cutoff).exp();

    let mut result = Vec::with_capacity(signal.len());
    let mut prev = signal[0] as f32;

    for &sample in signal {
        let filtered = alpha * sample as f32 + (1.0 - alpha) * prev;
        prev = filtered;
        result.push(filtered as i16);
    }

    result
}

/// Smooth signal using moving average (noise reduction)
/// window_size: number of samples to average (larger = more smoothing)
pub fn smooth_signal(signal: &[i16], window_size: usize) -> Vec<i16> {
    if signal.is_empty() || window_size <= 1 {
        return signal.to_vec();
    }

    let half_window = window_size / 2;
    let mut result = Vec::with_capacity(signal.len());

    for i in 0..signal.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(signal.len());

        let sum: i32 = signal[start..end].iter().map(|&x| x as i32).sum();
        let avg = sum / (end - start) as i32;
        result.push(avg as i16);
    }

    result
}

/// Add white noise to signal
/// noise_level: amplitude of noise relative to signal (0.0 to 1.0)
///   - 0.1 = subtle noise (10% of max amplitude)
///   - 0.5 = moderate noise
///   - 1.0 = full scale noise
pub fn add_noise(signal: &[i16], noise_level: f32) -> Vec<i16> {
    if signal.is_empty() || noise_level <= 0.0 {
        return signal.to_vec();
    }

    let mut rng = rand::thread_rng();

    // Scale noise relative to i16 max (32767)
    let noise_amplitude = (noise_level * 32767.0).min(32767.0);

    signal.iter()
        .map(|&sample| {
            // Generate white noise: uniform random in [-1, 1] * amplitude
            let noise = (rng.gen::<f32>() * 2.0 - 1.0) * noise_amplitude;
            let noisy = sample as f32 + noise;
            noisy.clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

/// Biquad Butterworth lowpass filter (matches dsp.js IIRFilter)
/// cutoff_hz: cutoff frequency in Hz (e.g., 1000.0 for 1kHz)
/// sample_rate: sample rate in Hz (e.g., 44100)
/// resonance: Q factor (1.0 = standard Butterworth, higher = sharper cutoff)
pub fn butterworth_lowpass(signal: &[i16], cutoff_hz: f32, sample_rate: u32, resonance: f32) -> Vec<i16> {
    if signal.is_empty() {
        return signal.to_vec();
    }

    let sample_rate_f = sample_rate as f32;

    // Clamp cutoff to valid range (must be below Nyquist)
    let cutoff = cutoff_hz.min(sample_rate_f * 0.5 - 1.0).max(1.0);
    let q = resonance.max(0.001); // Avoid division by zero

    // Biquad coefficients for lowpass
    let omega = 2.0 * std::f32::consts::PI * cutoff / sample_rate_f;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q);

    // Lowpass coefficients (before normalization)
    let b0 = (1.0 - cos_omega) / 2.0;
    let b1 = 1.0 - cos_omega;
    let b2 = (1.0 - cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b0 = b0 / a0;
    let b1 = b1 / a0;
    let b2 = b2 / a0;
    let a1 = a1 / a0;
    let a2 = a2 / a0;

    // Filter state (previous samples)
    let mut x1: f32 = 0.0; // x[n-1]
    let mut x2: f32 = 0.0; // x[n-2]
    let mut y1: f32 = 0.0; // y[n-1]
    let mut y2: f32 = 0.0; // y[n-2]

    let mut result = Vec::with_capacity(signal.len());

    for &sample in signal {
        let x0 = sample as f32;

        // Biquad difference equation: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

        // Shift state
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;

        result.push(y0.clamp(-32768.0, 32767.0) as i16);
    }

    result
}

/// Scale signal amplitude by a factor
/// factor > 1.0 = increase amplitude (louder)
/// factor < 1.0 = decrease amplitude (quieter)
/// factor = 1.0 = no change
pub fn scale_amplitude(signal: &[i16], factor: f32) -> Vec<i16> {
    if signal.is_empty() || factor == 1.0 {
        return signal.to_vec();
    }

    signal.iter()
        .map(|&sample| {
            let scaled = sample as f32 * factor;
            scaled.clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

/// Normalize signal to a target peak amplitude
/// If target_peak is None, normalizes to full dynamic range (32767)
pub fn normalize_signal(signal: &[i16], target_peak: Option<i16>) -> Vec<i16> {
    if signal.is_empty() {
        return signal.to_vec();
    }

    let current_max = signal.iter().map(|&x| x.abs()).max().unwrap_or(0);
    if current_max == 0 {
        return signal.to_vec();
    }

    let target = target_peak.unwrap_or(i16::MAX);
    let scale = target as f32 / current_max as f32;

    signal.iter()
        .map(|&x| (x as f32 * scale).clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// Make signal feel rougher by adding a phase-shifted copy on top
/// phase_shift: number of samples to shift (creates comb filter effect)
///   - Small shifts (1-10 samples) = subtle texture
///   - Larger shifts (10-100 samples) = more pronounced roughness
/// intensity: how much of the shifted signal to add (0.0 to 1.0)
///   - 0.0 = no change
///   - 0.5 = add half of shifted signal
///   - 1.0 = add full shifted signal (doubles amplitude where aligned)
pub fn roughen_signal(signal: &[i16], phase_shift: usize, intensity: f32) -> Vec<i16> {
    if signal.is_empty() || phase_shift == 0 || intensity <= 0.0 {
        return signal.to_vec();
    }

    let intensity = intensity.clamp(0.0, 1.0);

    signal.iter()
        .enumerate()
        .map(|(i, &sample)| {
            // Get phase-shifted sample (use zero for out-of-bounds)
            let shifted_sample = if i >= phase_shift {
                signal[i - phase_shift]
            } else {
                0
            };

            // Add the shifted signal on top of the original
            let result = sample as f32 + (shifted_sample as f32 * intensity);
            result.clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}
