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
        CombineOp::Mix { mix_balance, add_offset } => {
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
