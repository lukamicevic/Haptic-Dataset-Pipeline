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
