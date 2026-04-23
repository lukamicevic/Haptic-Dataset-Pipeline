use crate::wav_io::{load_wav, save_wav};
use crate::signal_ops::{
    combine_signals, separate_signals,
    add_noise, butterworth_lowpass, mask_signal,
    downsample_signal, scale_amplitude, normalize_signal, roughen_signal,
    transfer_texture,
};
use crate::types::{CombineOp, SeparateOp};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

/// Result of a batch operation on a single file pair
#[derive(Debug)]
pub struct BatchResult {
    pub output_path: String,
    pub success: bool,
    pub error: Option<String>,
}

/// Process multiple combine operations (mix/insert) in parallel using Rayon
pub fn batch_combine(
    file_pairs: &[(String, String, String)],  // (base_path, add_path, output_path)
    position: usize,
    op: CombineOp,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(base_path, add_path, output_path)| {
                process_combine_pair(base_path, add_path, output_path, position, &op)
            })
            .collect()
    })
}

fn process_combine_pair(
    base_path: &str,
    add_path: &str,
    output_path: &str,
    position: usize,
    op: &CombineOp,
) -> BatchResult {
    let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let base_data = load_wav(base_path)?;
        let add_data = load_wav(add_path)?;

        let result_samples = combine_signals(
            &base_data.samples,
            &add_data.samples,
            position,
            op.clone(),
        );

        save_wav(output_path, &result_samples, base_data.sample_rate)?;
        Ok(())
    })();

    BatchResult {
        output_path: output_path.to_string(),
        success: result.is_ok(),
        error: result.err().map(|e| e.to_string()),
    }
}

/// Process multiple separate operations (unmix/remove) in parallel using Rayon
pub fn batch_separate(
    file_pairs: &[(String, String, String)],  // (combined_path, signal_path, output_path)
    position: usize,
    op: SeparateOp,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(combined_path, signal_path, output_path)| {
                process_separate_pair(combined_path, signal_path, output_path, position, &op)
            })
            .collect()
    })
}

fn process_separate_pair(
    combined_path: &str,
    signal_path: &str,
    output_path: &str,
    position: usize,
    op: &SeparateOp,
) -> BatchResult {
    let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let combined_data = load_wav(combined_path)?;
        let signal_data = load_wav(signal_path)?;

        let result_samples = separate_signals(
            &combined_data.samples,
            &signal_data.samples,
            position,
            op.clone(),
        );

        save_wav(output_path, &result_samples, combined_data.sample_rate)?;
        Ok(())
    })();

    BatchResult {
        output_path: output_path.to_string(),
        success: result.is_ok(),
        error: result.err().map(|e| e.to_string()),
    }
}

// =============================================================================
// Batch Single-Signal Operations
// =============================================================================

/// Batch add noise to multiple files
pub fn batch_add_noise(
    file_pairs: &[(String, String)],  // (input_path, output_path)
    noise_level: f32,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(input_path, output_path)| {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let data = load_wav(input_path)?;
                    let processed = add_noise(&data.samples, noise_level);
                    save_wav(output_path, &processed, data.sample_rate)?;
                    Ok(())
                })();

                BatchResult {
                    output_path: output_path.to_string(),
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                }
            })
            .collect()
    })
}

/// Batch butterworth lowpass filter on multiple files
pub fn batch_butterworth_lowpass(
    file_pairs: &[(String, String)],
    cutoff_hz: f32,
    sample_rate: u32,
    resonance: f32,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(input_path, output_path)| {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let data = load_wav(input_path)?;
                    let processed = butterworth_lowpass(&data.samples, cutoff_hz, sample_rate, resonance);
                    save_wav(output_path, &processed, data.sample_rate)?;
                    Ok(())
                })();

                BatchResult {
                    output_path: output_path.to_string(),
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                }
            })
            .collect()
    })
}

/// Batch mask signal on multiple files
pub fn batch_mask_signal(
    file_pairs: &[(String, String)],
    length: usize,
    mask_value: i16,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(input_path, output_path)| {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let data = load_wav(input_path)?;
                    let (processed, _) = mask_signal(&data.samples, None, length, mask_value);
                    save_wav(output_path, &processed, data.sample_rate)?;
                    Ok(())
                })();

                BatchResult {
                    output_path: output_path.to_string(),
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                }
            })
            .collect()
    })
}

/// Batch downsample signal on multiple files
pub fn batch_downsample_signal(
    file_pairs: &[(String, String)],
    factor: usize,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(input_path, output_path)| {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let data = load_wav(input_path)?;
                    let processed = downsample_signal(&data.samples, factor);
                    save_wav(output_path, &processed, data.sample_rate)?;
                    Ok(())
                })();

                BatchResult {
                    output_path: output_path.to_string(),
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                }
            })
            .collect()
    })
}

/// Batch scale amplitude on multiple files
pub fn batch_scale_amplitude(
    file_pairs: &[(String, String)],
    factor: f32,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(input_path, output_path)| {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let data = load_wav(input_path)?;
                    let processed = scale_amplitude(&data.samples, factor);
                    save_wav(output_path, &processed, data.sample_rate)?;
                    Ok(())
                })();

                BatchResult {
                    output_path: output_path.to_string(),
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                }
            })
            .collect()
    })
}

/// Batch normalize signal on multiple files
pub fn batch_normalize_signal(
    file_pairs: &[(String, String)],
    target_peak: Option<i16>,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(input_path, output_path)| {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let data = load_wav(input_path)?;
                    let processed = normalize_signal(&data.samples, target_peak);
                    save_wav(output_path, &processed, data.sample_rate)?;
                    Ok(())
                })();

                BatchResult {
                    output_path: output_path.to_string(),
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                }
            })
            .collect()
    })
}

/// Batch roughen signal on multiple files
pub fn batch_roughen_signal(
    file_pairs: &[(String, String)],
    phase_shift: usize,
    intensity: f32,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_pairs
            .par_iter()
            .map(|(input_path, output_path)| {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let data = load_wav(input_path)?;
                    let processed = roughen_signal(&data.samples, phase_shift, intensity);
                    save_wav(output_path, &processed, data.sample_rate)?;
                    Ok(())
                })();

                BatchResult {
                    output_path: output_path.to_string(),
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                }
            })
            .collect()
    })
}

/// Batch transfer texture from one set of files to another in parallel
/// file_triples: (texture_path, base_path, output_path)
/// texture_path: source of high-frequency texture
/// base_path: source of amplitude envelope
pub fn batch_transfer_texture(
    file_triples: &[(String, String, String)],
    sample_rate: u32,
    num_threads: Option<usize>,
) -> Vec<BatchResult> {
    let pool = match num_threads {
        Some(n) => ThreadPoolBuilder::new().num_threads(n).build().unwrap(),
        None => ThreadPoolBuilder::new().build().unwrap(),
    };

    pool.install(|| {
        file_triples
            .par_iter()
            .map(|(texture_path, base_path, output_path)| {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let texture_data = load_wav(texture_path)?;
                    let base_data = load_wav(base_path)?;
                    let processed = transfer_texture(&texture_data.samples, &base_data.samples, sample_rate);
                    save_wav(output_path, &processed, sample_rate)?;
                    Ok(())
                })();

                BatchResult {
                    output_path: output_path.to_string(),
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                }
            })
            .collect()
    })
}
