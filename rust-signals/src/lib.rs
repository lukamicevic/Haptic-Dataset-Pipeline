use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;

mod wav_io;
mod signal_ops;
mod types;

use types::{CombineOp, SeparateOp};

/// Combine two WAV files using specified operation
///
/// Parameters:
/// - base_path: Path to base WAV file
/// - add_path: Path to signal to add
/// - output_path: Path for output WAV file
/// - position: Sample position where operation occurs
/// - operation: "insert" or "mix"
/// - mix_balance: Weight for mixing (0.0-1.0), only used for "mix"
/// - add_offset: Sample offset in add signal
#[pyfunction]
#[pyo3(signature = (base_path, add_path, output_path, position, operation, mix_balance=0.5, add_offset=0))]
fn combine_signals_from_files(
    base_path: &str,
    add_path: &str,
    output_path: &str,
    position: usize,
    operation: &str,
    mix_balance: f32,
    add_offset: usize,
) -> PyResult<()> {
    // Load WAV files
    let base_data = wav_io::load_wav(base_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load base: {}", e)))?;
    let add_data = wav_io::load_wav(add_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load add: {}", e)))?;

    // Parse operation
    let op = match operation {
        "insert" => CombineOp::Insert { add_offset },
        "mix" => CombineOp::Mix { mix_balance, add_offset },
        _ => return Err(PyValueError::new_err(
            format!("Unknown operation: {}. Use 'insert' or 'mix'", operation)
        )),
    };

    // Process
    let result = signal_ops::combine_signals(
        &base_data.samples,
        &add_data.samples,
        position,
        op,
    );

    // Save
    wav_io::save_wav(output_path, &result, base_data.sample_rate)
        .map_err(|e| PyValueError::new_err(format!("Failed to save: {}", e)))?;

    Ok(())
}

/// Separate signals from combined WAV file
#[pyfunction]
#[pyo3(signature = (combined_path, signal_path, output_path, position, operation, mix_balance=0.5))]
fn separate_signals_from_files(
    combined_path: &str,
    signal_path: &str,
    output_path: &str,
    position: usize,
    operation: &str,
    mix_balance: f32,
) -> PyResult<()> {
    // Load WAV files
    let combined_data = wav_io::load_wav(combined_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load combined: {}", e)))?;
    let signal_data = wav_io::load_wav(signal_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load signal: {}", e)))?;

    // Parse operation
    let op = match operation {
        "remove" => SeparateOp::Remove,
        "unmix" => SeparateOp::Unmix { mix_balance },
        _ => return Err(PyValueError::new_err(
            format!("Unknown operation: {}. Use 'remove' or 'unmix'", operation)
        )),
    };

    // Process
    let result = signal_ops::separate_signals(
        &combined_data.samples,
        &signal_data.samples,
        position,
        op,
    );

    // Save
    wav_io::save_wav(output_path, &result, combined_data.sample_rate)
        .map_err(|e| PyValueError::new_err(format!("Failed to save: {}", e)))?;

    Ok(())
}

/// Python module
#[pymodule]
fn rust_signals(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(combine_signals_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(separate_signals_from_files, m)?)?;
    Ok(())
}
