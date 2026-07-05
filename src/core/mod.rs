//! Core numeric data structures and invariants.

pub mod complex;
pub mod dual;
pub mod elementary_functions;
pub mod matrix;

pub use complex::Complex;
pub use dual::Dual;
pub use elementary_functions::*;
pub use matrix::Matrix;
pub use matrix::RealMatrix;
