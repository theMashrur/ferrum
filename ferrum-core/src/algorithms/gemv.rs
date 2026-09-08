use std::any::TypeId;
use std::env;
use std::ops;

use crate::core::matrix::{MatrixRead, MatrixWrite};

use crate::core::vector::VectorRead;
use crate::core::vector::VectorWrite;

pub fn basic_gemv_kernel<A, B, C, T>(a: &A, b: &B, out: &mut C, alpha: Option<T>, beta: Option<T>)
where
    A: MatrixRead<T>,
    B: VectorRead<T>,
    C: VectorWrite<T>,
    T: Copy
        + ops::Add<Output = T>
        + ops::Mul<Output = T>
        + Default
        + PartialEq
        + From<f64>
        + ops::AddAssign<T>
        + ops::Sub<Output = T>,
{
    assert_eq!(
        a.cols(),
        b.size(),
        "The Number of cols for the Matrix {} must match the size of the vector {}",
        a.cols(),
        b.size()
    );
}
