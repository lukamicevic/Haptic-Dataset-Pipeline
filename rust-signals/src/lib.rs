use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;
use numpy::{PyArray1, PyReadonlyArray1};

mod wav_io;
mod signal_ops;
mod types;

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

// =============================================================================
// File-based functions (convenience API, uses hound for WAV I/O)
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
        _ => return Err(PyValueError::new_err(
            format!("Unknown operation: {}. Use 'insert' or 'mix'", operation)
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
// Python module registration
// =============================================================================

#[pymodule]
fn rust_signals(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Array-based functions (high performance)
    m.add_function(wrap_pyfunction!(mix_signals, m)?)?;
    m.add_function(wrap_pyfunction!(insert_signal, m)?)?;
    m.add_function(wrap_pyfunction!(unmix_signal, m)?)?;
    m.add_function(wrap_pyfunction!(remove_signal, m)?)?;
    // File-based functions (convenience)
    m.add_function(wrap_pyfunction!(combine_signals_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(separate_signals_from_files, m)?)?;
    Ok(())
}
