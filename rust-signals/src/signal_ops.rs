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
// WORK IN PROGRESS - Inpainting and Super-resolution
// These functions use simple interpolation which doesn't produce meaningful
// results for audio/haptic signals. Proper implementation requires ML-based
// approaches (e.g., latent diffusion models like AUDIT).
// =============================================================================

// /// Inpaint signal by filling gaps (zeros) with interpolation
// /// method: "linear" or "cubic"
// pub fn inpaint_signal(signal: &[i16], method: &str) -> Vec<i16> {
//     let mut result = signal.to_vec();
//
//     // Find gaps (runs of zeros)
//     let mut i = 0;
//     while i < result.len() {
//         if result[i] == 0 {
//             // Find gap boundaries
//             let gap_start = i;
//             while i < result.len() && result[i] == 0 {
//                 i += 1;
//             }
//             let gap_end = i;
//
//             // Get boundary values for interpolation
//             let left_val = if gap_start > 0 { result[gap_start - 1] as f64 } else { 0.0 };
//             let right_val = if gap_end < result.len() { result[gap_end] as f64 } else { 0.0 };
//
//             let gap_len = gap_end - gap_start;
//
//             match method {
//                 "cubic" => {
//                     // Cubic interpolation using Catmull-Rom spline
//                     let left_left = if gap_start > 1 { result[gap_start - 2] as f64 } else { left_val };
//                     let right_right = if gap_end + 1 < result.len() { result[gap_end + 1] as f64 } else { right_val };
//
//                     for j in 0..gap_len {
//                         let t = (j + 1) as f64 / (gap_len + 1) as f64;
//                         let t2 = t * t;
//                         let t3 = t2 * t;
//
//                         // Catmull-Rom coefficients
//                         let val = 0.5 * (
//                             (2.0 * left_val) +
//                             (-left_left + right_val) * t +
//                             (2.0 * left_left - 5.0 * left_val + 4.0 * right_val - right_right) * t2 +
//                             (-left_left + 3.0 * left_val - 3.0 * right_val + right_right) * t3
//                         );
//                         result[gap_start + j] = val.clamp(-32768.0, 32767.0) as i16;
//                     }
//                 }
//                 _ => {
//                     // Linear interpolation (default)
//                     for j in 0..gap_len {
//                         let t = (j + 1) as f64 / (gap_len + 1) as f64;
//                         let val = left_val + t * (right_val - left_val);
//                         result[gap_start + j] = val as i16;
//                     }
//                 }
//             }
//         } else {
//             i += 1;
//         }
//     }
//
//     result
// }

// /// Supersample signal by integer factor using interpolation
// /// method: "linear" or "sinc"
// pub fn supersample_signal(signal: &[i16], factor: usize, method: &str) -> Vec<i16> {
//     if factor <= 1 || signal.is_empty() {
//         return signal.to_vec();
//     }
//
//     let new_len = (signal.len() - 1) * factor + 1;
//     let mut result = Vec::with_capacity(new_len);
//
//     match method {
//         "sinc" => {
//             // Windowed sinc interpolation (Lanczos-like)
//             let window_size = 4; // Number of neighboring samples to consider
//
//             for i in 0..new_len {
//                 let pos = i as f64 / factor as f64;
//                 let base_idx = pos.floor() as isize;
//                 let frac = pos - base_idx as f64;
//
//                 if frac.abs() < 1e-10 {
//                     // Exact sample position
//                     result.push(signal[base_idx as usize]);
//                 } else {
//                     // Sinc interpolation
//                     let mut sum = 0.0;
//                     let mut weight_sum = 0.0;
//
//                     for k in -(window_size as isize)..=(window_size as isize) {
//                         let idx = base_idx + k;
//                         if idx >= 0 && (idx as usize) < signal.len() {
//                             let x = frac - k as f64;
//                             // Lanczos window
//                             let sinc = if x.abs() < 1e-10 {
//                                 1.0
//                             } else {
//                                 let pi_x = std::f64::consts::PI * x;
//                                 (pi_x.sin() / pi_x) * ((pi_x / window_size as f64).sin() / (pi_x / window_size as f64))
//                             };
//                             sum += signal[idx as usize] as f64 * sinc;
//                             weight_sum += sinc;
//                         }
//                     }
//
//                     let val = if weight_sum.abs() > 1e-10 { sum / weight_sum } else { 0.0 };
//                     result.push(val.clamp(-32768.0, 32767.0) as i16);
//                 }
//             }
//         }
//         _ => {
//             // Linear interpolation (default)
//             for i in 0..signal.len() - 1 {
//                 result.push(signal[i]);
//                 let diff = signal[i + 1] as f64 - signal[i] as f64;
//                 for j in 1..factor {
//                     let t = j as f64 / factor as f64;
//                     let val = signal[i] as f64 + t * diff;
//                     result.push(val as i16);
//                 }
//             }
//             result.push(signal[signal.len() - 1]);
//         }
//     }
//
//     result
// }
