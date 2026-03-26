use crate::wav_io::{load_wav, save_wav};
use crate::signal_ops::{combine_signals, separate_signals};
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
