pub mod bindings;

use pyo3::prelude::*;

#[pymodule]
fn _ferrum(_py: Python<'_>, _module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
