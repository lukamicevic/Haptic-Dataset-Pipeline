use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;
use numpy::{PyArray1, PyReadonlyArray1};

mod wav_io;
mod signal_ops;
mod types;
mod batch_ops;

use types::{CombineOp, SeparateOp};

// =============================================================================
// Array-based functions (zero-copy, high performance)
// =============================================================================

/// Mix two signals (array-based, zero-copy input)
#[pyfunction]
#[pyo3(signature = (base, add, position, mix_balance=0.5, add_offset=0, normalize=false))]
fn mix_signals<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, i16>,
    add: PyReadonlyArray1<'py, i16>,
    position: usize,
    mix_balance: f32,
    add_offset: usize,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let base_slice = base.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read base array: {}", e)))?;
    let add_slice = add.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read add array: {}", e)))?;

    let op = CombineOp::Mix { mix_balance, add_offset, normalize };
    let result = signal_ops::combine_signals(base_slice, add_slice, position, op);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Insert signal into base (array-based)
#[pyfunction]
#[pyo3(signature = (base, add, position, add_offset=0))]
fn insert_signal<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, i16>,
    add: PyReadonlyArray1<'py, i16>,
    position: usize,
    add_offset: usize,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let base_slice = base.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read base array: {}", e)))?;
    let add_slice = add.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read add array: {}", e)))?;

    let op = CombineOp::Insert { add_offset };
    let result = signal_ops::combine_signals(base_slice, add_slice, position, op);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Unmix signal from combined (array-based)
#[pyfunction]
#[pyo3(signature = (combined, signal, position, mix_balance=0.5))]
fn unmix_signal<'py>(
    py: Python<'py>,
    combined: PyReadonlyArray1<'py, i16>,
    signal: PyReadonlyArray1<'py, i16>,
    position: usize,
    mix_balance: f32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let combined_slice = combined.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read combined array: {}", e)))?;
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let op = SeparateOp::Unmix { mix_balance };
    let result = signal_ops::separate_signals(combined_slice, signal_slice, position, op);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Remove signal from combined (array-based)
#[pyfunction]
fn remove_signal<'py>(
    py: Python<'py>,
    combined: PyReadonlyArray1<'py, i16>,
    signal: PyReadonlyArray1<'py, i16>,
    position: usize,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let combined_slice = combined.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read combined array: {}", e)))?;
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let op = SeparateOp::Remove;
    let result = signal_ops::separate_signals(combined_slice, signal_slice, position, op);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Replace samples in base signal with add signal (array-based)
#[pyfunction]
#[pyo3(signature = (base, add, position, add_offset=0))]
fn replace_signal<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, i16>,
    add: PyReadonlyArray1<'py, i16>,
    position: usize,
    add_offset: usize,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let base_slice = base.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read base array: {}", e)))?;
    let add_slice = add.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read add array: {}", e)))?;

    let op = CombineOp::Replace { add_offset };
    let result = signal_ops::combine_signals(base_slice, add_slice, position, op);

    Ok(PyArray1::from_vec_bound(py, result))
}

// =============================================================================
// Training Data Generation - Inpainting and Super-resolution
// These create degraded versions of signals for AI training pairs
// =============================================================================

/// Mask a portion of signal (for inpainting training data)
/// If start is None, picks a random position
/// Returns (masked_signal, actual_start_position)
#[pyfunction]
#[pyo3(signature = (signal, length, start=None, mask_value=0))]
fn mask_signal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    length: usize,
    start: Option<usize>,
    mask_value: i16,
) -> PyResult<(Bound<'py, PyArray1<i16>>, usize)> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let (result, actual_start) = signal_ops::mask_signal(signal_slice, start, length, mask_value);

    Ok((PyArray1::from_vec_bound(py, result), actual_start))
}

/// Downsample signal by factor (for super-resolution training data)
#[pyfunction]
fn downsample_signal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    factor: usize,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let result = signal_ops::downsample_signal(signal_slice, factor);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Low-pass filter with cutoff frequency
/// cutoff: normalized frequency (0.0 to 0.5, fraction of sample rate)
#[pyfunction]
#[pyo3(signature = (signal, cutoff=0.1))]
fn lowpass_filter<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    cutoff: f32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let result = signal_ops::lowpass_filter(signal_slice, cutoff);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Smooth signal using moving average (noise reduction)
/// window_size: number of samples to average (larger = more smoothing)
#[pyfunction]
#[pyo3(signature = (signal, window_size=5))]
fn smooth_signal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    window_size: usize,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let result = signal_ops::smooth_signal(signal_slice, window_size);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Add white noise to signal
/// noise_level: amplitude of noise (0.0 to 1.0, relative to max amplitude)
#[pyfunction]
#[pyo3(signature = (signal, noise_level=0.1))]
fn add_noise<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    noise_level: f32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let result = signal_ops::add_noise(signal_slice, noise_level);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Biquad Butterworth lowpass filter (matches dsp.js IIRFilter)
/// cutoff_hz: cutoff frequency in Hz (e.g., 1000.0)
/// sample_rate: sample rate in Hz (e.g., 44100)
/// resonance: Q factor (1.0 = standard Butterworth)
#[pyfunction]
#[pyo3(signature = (signal, cutoff_hz, sample_rate=44100, resonance=1.0))]
fn butterworth_lowpass<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    cutoff_hz: f32,
    sample_rate: u32,
    resonance: f32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let result = signal_ops::butterworth_lowpass(signal_slice, cutoff_hz, sample_rate, resonance);

    Ok(PyArray1::from_vec_bound(py, result))
}

// =============================================================================
// File-based functions (convenience API)
// =============================================================================

/// Combine two WAV files using specified operation
#[pyfunction]
#[pyo3(signature = (base_path, add_path, output_path, position, operation, mix_balance=0.5, add_offset=0, normalize=false))]
fn combine_signals_from_files(
    base_path: &str,
    add_path: &str,
    output_path: &str,
    position: usize,
    operation: &str,
    mix_balance: f32,
    add_offset: usize,
    normalize: bool,
) -> PyResult<()> {
    let base_data = wav_io::load_wav(base_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load base: {}", e)))?;
    let add_data = wav_io::load_wav(add_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load add: {}", e)))?;

    let op = match operation {
        "insert" => CombineOp::Insert { add_offset },
        "mix" => CombineOp::Mix { mix_balance, add_offset, normalize },
        "replace" => CombineOp::Replace { add_offset },
        _ => return Err(PyValueError::new_err(
            format!("Unknown operation: {}. Use 'insert', 'mix', or 'replace'", operation)
        )),
    };

    let result = signal_ops::combine_signals(
        &base_data.samples,
        &add_data.samples,
        position,
        op,
    );

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
    let combined_data = wav_io::load_wav(combined_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load combined: {}", e)))?;
    let signal_data = wav_io::load_wav(signal_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load signal: {}", e)))?;

    let op = match operation {
        "remove" => SeparateOp::Remove,
        "unmix" => SeparateOp::Unmix { mix_balance },
        _ => return Err(PyValueError::new_err(
            format!("Unknown operation: {}. Use 'remove' or 'unmix'", operation)
        )),
    };

    let result = signal_ops::separate_signals(
        &combined_data.samples,
        &signal_data.samples,
        position,
        op,
    );

    wav_io::save_wav(output_path, &result, combined_data.sample_rate)
        .map_err(|e| PyValueError::new_err(format!("Failed to save: {}", e)))?;

    Ok(())
}

// =============================================================================
// Batch Processing Functions (Rayon parallel)
// =============================================================================

/// Batch process multiple file pairs with mix operation in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, position, mix_balance=0.5, add_offset=0, normalize=false, num_threads=None))]
fn batch_mix_files(
    file_pairs: Vec<(String, String, String)>,
    position: usize,
    mix_balance: f32,
    add_offset: usize,
    normalize: bool,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let op = CombineOp::Mix { mix_balance, add_offset, normalize };
    let results = batch_ops::batch_combine(&file_pairs, position, op, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch process multiple file pairs with insert operation in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, position, add_offset=0, num_threads=None))]
fn batch_insert_files(
    file_pairs: Vec<(String, String, String)>,
    position: usize,
    add_offset: usize,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let op = CombineOp::Insert { add_offset };
    let results = batch_ops::batch_combine(&file_pairs, position, op, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch process multiple unmix operations in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, position, mix_balance=0.5, num_threads=None))]
fn batch_unmix_files(
    file_pairs: Vec<(String, String, String)>,
    position: usize,
    mix_balance: f32,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let op = SeparateOp::Unmix { mix_balance };
    let results = batch_ops::batch_separate(&file_pairs, position, op, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch process multiple remove operations in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, position, num_threads=None))]
fn batch_remove_files(
    file_pairs: Vec<(String, String, String)>,
    position: usize,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let op = SeparateOp::Remove;
    let results = batch_ops::batch_separate(&file_pairs, position, op, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch process multiple file pairs with replace operation in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, position, add_offset=0, num_threads=None))]
fn batch_replace_files(
    file_pairs: Vec<(String, String, String)>,
    position: usize,
    add_offset: usize,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let op = CombineOp::Replace { add_offset };
    let results = batch_ops::batch_combine(&file_pairs, position, op, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

// =============================================================================
// Python module registration
// =============================================================================

#[pymodule]
fn rust_signals(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Array-based functions (high performance)
    m.add_function(wrap_pyfunction!(mix_signals, m)?)?;
    m.add_function(wrap_pyfunction!(insert_signal, m)?)?;
    m.add_function(wrap_pyfunction!(replace_signal, m)?)?;
    m.add_function(wrap_pyfunction!(unmix_signal, m)?)?;
    m.add_function(wrap_pyfunction!(remove_signal, m)?)?;
    // Training data generation (for AI training pairs)
    m.add_function(wrap_pyfunction!(mask_signal, m)?)?;
    m.add_function(wrap_pyfunction!(downsample_signal, m)?)?;
    m.add_function(wrap_pyfunction!(lowpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_signal, m)?)?;
    m.add_function(wrap_pyfunction!(add_noise, m)?)?;
    m.add_function(wrap_pyfunction!(butterworth_lowpass, m)?)?;
    // File-based functions (convenience)
    m.add_function(wrap_pyfunction!(combine_signals_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(separate_signals_from_files, m)?)?;
    // Batch processing functions (Rayon parallel)
    m.add_function(wrap_pyfunction!(batch_mix_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_insert_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_replace_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_unmix_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_remove_files, m)?)?;
    Ok(())
}
