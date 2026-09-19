#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::any::TypeId;
use std::env;
use std::ops;

use crate::core::matrix::{MatrixRead, MatrixWrite};

use crate::core::vector::VectorRead;
use crate::core::vector::VectorWrite;

type GemvMicrokernelFn =
    unsafe fn(K: usize, alpha: f64, beta: f64, a: *const f64, x: *const f64, y: *mut f64);

#[derive(Debug, Clone, Copy)]
pub struct GemvBlocking {
    pub m_block: usize,
    pub k_block: usize,
}

impl Default for GemvBlocking {
    fn default() -> Self {
        Self {
            m_block: 64,
            k_block: 64,
        }
    }
}

#[derive(Clone, Copy)]
struct GemvMicrokernel {
    m_unroll: usize,
    k_unroll: usize,
    func: GemvMicrokernelFn,
}

impl GemvMicrokernel {
    unsafe fn run(
        &self,
        K: usize,
        alpha: f64,
        beta: f64,
        a: *const f64,
        x: *const f64,
        y: *mut f64,
    ) {
        (self.func)(K, alpha, beta, a, x, y);
    }
}

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

    let m = a.rows();
    let k_dim = b.size();
    let beta = beta.unwrap_or(T::from(0.0));
    let alpha = alpha.unwrap_or(T::from(1.0));

    // beta scaling block: skip if zero
    if beta != T::from(1.0) {
        for i in 0..m {
            out.accumulate(i, (beta - T::from(1.0)) * (*out.get(i)));
        }
    }

    // Matvec Multiplication block: skip if zero
    if alpha != T::from(0.0) {
        for i in 0..m {
            let mut sum = T::default();
            for j in 0..k_dim {
                sum += *a.get(i, j) * (*b.get(j));
            }
            out.accumulate(i, alpha * sum);
        }
    }
}

macro_rules! gemv_row_update {
    ($a: expr, $x_vec: expr, $acc: expr, $row: expr, $k: expr, $K: expr, $load: expr, $fmadd: expr) => {
        unsafe {
            let a_vec = $load($a.add($row * $K + $k));
            $acc[$row] = $fmadd(a_vec, $x_vec, $acc[$row]);
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_microkernel_f64_4_x86_64(
    K: usize,
    alpha: f64,
    beta: f64,
    a: *const f64,
    x: *const f64,
    y: *mut f64,
) {
    let mut y_accumulators = [_mm256_setzero_pd(); 4];

    unsafe {
        for k in (0..K).step_by(4) {
            let x_vec = _mm256_loadu_pd(x.add(k));

            for i in 0..4 {
                gemv_row_update!(
                    a,
                    x_vec,
                    y_accumulators,
                    i,
                    k,
                    K,
                    _mm256_loadu_pd,
                    _mm256_fmadd_pd
                );
            }
        }

        let v_alpha = _mm256_set1_pd(alpha);
        if beta == 0.0 {
            for i in 0..4 {
                _mm256_storeu_pd(y.add(i), _mm256_mul_pd(v_alpha, y_accumulators[i]));
            }
        } else {
            let v_beta = _mm256_set1_pd(beta);
            for i in 0..4 {
                let y_vec = _mm256_loadu_pd(y.add(i));
                let y_result =
                    _mm256_fmadd_pd(v_alpha, y_accumulators[i], _mm256_mul_pd(v_beta, y_vec));
                _mm256_storeu_pd(y.add(i), y_result);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemv_microkernel_f64_8_x86_64(
    K: usize,
    alpha: f64,
    beta: f64,
    a: *const f64,
    x: *const f64,
    y: *mut f64,
) {
    let mut y_accumulators = [_mm512_setzero_pd(); 8];

    unsafe {
        for k in (0..K).step_by(8) {
            let x_vec = _mm512_loadu_pd(x.add(k));

            for i in 0..8 {
                gemv_row_update!(
                    a,
                    x_vec,
                    y_accumulators,
                    i,
                    k,
                    K,
                    _mm512_loadu_pd,
                    _mm512_fmadd_pd
                );
            }
        }

        let v_alpha = _mm512_set1_pd(alpha);
        if beta == 0.0 {
            for i in 0..8 {
                _mm512_storeu_pd(y.add(i), _mm512_mul_pd(v_alpha, y_accumulators[i]));
            }
        } else {
            let v_beta = _mm512_set1_pd(beta);
            for i in 0..8 {
                let y_vec = _mm512_loadu_pd(y.add(i));
                let y_result =
                    _mm512_fmadd_pd(v_alpha, y_accumulators[i], _mm512_mul_pd(v_beta, y_vec));
                _mm512_storeu_pd(y.add(i), y_result);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemv_microkernel_f64_2_aarch64(
    K: usize,
    alpha: f64,
    beta: f64,
    a: *const f64,
    x: *const f64,
    y: *mut f64,
) {
    let mut y_accumulators = [vdupq_n_f64(0.0); 2];

    unsafe {
        for k in (0..K).step_by(2) {
            let x_vec = vld1q_f64(x.add(k));

            for i in 0..2 {
                gemv_row_update!(a, x_vec, y_accumulators, i, k, K, vld1q_f64, vfmaq_f64);
            }
        }

        let v_alpha = vdupq_n_f64(alpha);
        if beta == 0.0 {
            for i in 0..2 {
                vst1q_f64(y.add(i), vmulq_f64(v_alpha, y_accumulators[i]));
            }
        } else {
            let v_beta = vdupq_n_f64(beta);
            for i in 0..2 {
                let y_vec = vld1q_f64(y.add(i));
                let y_result = vfmaq_f64(vmulq_f64(v_beta, y_vec), v_alpha, y_accumulators[i]);
                vst1q_f64(y.add(i), y_result);
            }
        }
    }
}
