pub mod algorithms;
pub mod bindings;
pub mod core;
pub mod threading;

use pyo3::prelude::*;

#[pymodule]
fn _ferrum(_py: Python<'_>, _module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
