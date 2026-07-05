use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::core::matrix::RealMatrix;

#[pyclass(name = "RealMatrix")]
#[derive(Debug, Clone)]
pub struct PyRealMatrix {
    pub matrix: RealMatrix,
}
