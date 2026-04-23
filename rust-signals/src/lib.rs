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

/// Scale signal amplitude by a factor
/// factor > 1.0 = increase (louder), factor < 1.0 = decrease (quieter)
#[pyfunction]
#[pyo3(signature = (signal, factor))]
fn scale_amplitude<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    factor: f32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let result = signal_ops::scale_amplitude(signal_slice, factor);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Normalize signal to target peak amplitude
/// If target_peak is None, normalizes to full dynamic range (32767)
#[pyfunction]
#[pyo3(signature = (signal, target_peak=None))]
fn normalize_signal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    target_peak: Option<i16>,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let result = signal_ops::normalize_signal(signal_slice, target_peak);

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

/// Make signal feel rougher by adding phase-shifted copy on top
/// phase_shift: samples to shift (1-10 = subtle, 10-100 = pronounced)
/// intensity: how much of shifted signal to add (0.0 to 1.0)
#[pyfunction]
#[pyo3(signature = (signal, phase_shift=5, intensity=0.5))]
fn roughen_signal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
    phase_shift: usize,
    intensity: f32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read signal array: {}", e)))?;

    let result = signal_ops::roughen_signal(signal_slice, phase_shift, intensity);

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Transfer texture from one signal to another
/// Takes high-frequency texture (>50Hz) from texture_signal
/// and applies it to amplitude envelope (<15Hz) from base_signal
#[pyfunction]
#[pyo3(signature = (texture_signal, base_signal, sample_rate=44100))]
fn transfer_texture<'py>(
    py: Python<'py>,
    texture_signal: PyReadonlyArray1<'py, i16>,
    base_signal: PyReadonlyArray1<'py, i16>,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let texture_slice = texture_signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read texture signal: {}", e)))?;
    let base_slice = base_signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read base signal: {}", e)))?;

    let result = signal_ops::transfer_texture(texture_slice, base_slice, sample_rate);

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
// Batch Single-Signal Operations (Rayon parallel)
// =============================================================================

/// Batch add noise to multiple files in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, noise_level=0.1, num_threads=None))]
fn batch_add_noise(
    file_pairs: Vec<(String, String)>,
    noise_level: f32,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let results = batch_ops::batch_add_noise(&file_pairs, noise_level, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch butterworth lowpass filter on multiple files in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, cutoff_hz, sample_rate=44100, resonance=1.0, num_threads=None))]
fn batch_butterworth_lowpass(
    file_pairs: Vec<(String, String)>,
    cutoff_hz: f32,
    sample_rate: u32,
    resonance: f32,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let results = batch_ops::batch_butterworth_lowpass(&file_pairs, cutoff_hz, sample_rate, resonance, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch mask signal on multiple files in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, length, mask_value=0, num_threads=None))]
fn batch_mask_signal(
    file_pairs: Vec<(String, String)>,
    length: usize,
    mask_value: i16,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let results = batch_ops::batch_mask_signal(&file_pairs, length, mask_value, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch downsample signal on multiple files in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, factor, num_threads=None))]
fn batch_downsample_signal(
    file_pairs: Vec<(String, String)>,
    factor: usize,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let results = batch_ops::batch_downsample_signal(&file_pairs, factor, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch scale amplitude on multiple files in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, factor, num_threads=None))]
fn batch_scale_amplitude(
    file_pairs: Vec<(String, String)>,
    factor: f32,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let results = batch_ops::batch_scale_amplitude(&file_pairs, factor, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch normalize signal on multiple files in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, target_peak=None, num_threads=None))]
fn batch_normalize_signal(
    file_pairs: Vec<(String, String)>,
    target_peak: Option<i16>,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let results = batch_ops::batch_normalize_signal(&file_pairs, target_peak, num_threads);

    Ok(results.into_iter()
        .map(|r| (r.output_path, r.success, r.error))
        .collect())
}

/// Batch roughen signal on multiple files in parallel
#[pyfunction]
#[pyo3(signature = (file_pairs, phase_shift=5, intensity=0.5, num_threads=None))]
fn batch_roughen_signal(
    file_pairs: Vec<(String, String)>,
    phase_shift: usize,
    intensity: f32,
    num_threads: Option<usize>,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let results = batch_ops::batch_roughen_signal(&file_pairs, phase_shift, intensity, num_threads);

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
    m.add_function(wrap_pyfunction!(add_noise, m)?)?;
    m.add_function(wrap_pyfunction!(scale_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_signal, m)?)?;
    m.add_function(wrap_pyfunction!(butterworth_lowpass, m)?)?;
    m.add_function(wrap_pyfunction!(roughen_signal, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_texture, m)?)?;
    // File-based functions (convenience)
    m.add_function(wrap_pyfunction!(combine_signals_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(separate_signals_from_files, m)?)?;
    // Batch processing functions (Rayon parallel)
    m.add_function(wrap_pyfunction!(batch_mix_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_insert_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_replace_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_unmix_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_remove_files, m)?)?;
    // Batch single-signal operations
    m.add_function(wrap_pyfunction!(batch_add_noise, m)?)?;
    m.add_function(wrap_pyfunction!(batch_butterworth_lowpass, m)?)?;
    m.add_function(wrap_pyfunction!(batch_mask_signal, m)?)?;
    m.add_function(wrap_pyfunction!(batch_downsample_signal, m)?)?;
    m.add_function(wrap_pyfunction!(batch_scale_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(batch_normalize_signal, m)?)?;
    m.add_function(wrap_pyfunction!(batch_roughen_signal, m)?)?;
    Ok(())
}
