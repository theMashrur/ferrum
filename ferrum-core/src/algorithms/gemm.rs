#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::any::TypeId;
use std::env;
use std::ops;

use crate::core::matrix::{MatrixRead, MatrixWrite};

const SCALAR_MR_F64: usize = 4;
const SCALAR_NR_F64: usize = 4;
const MAX_F64_KERNEL_MR: usize = 16;
const MAX_F64_KERNEL_NR: usize = 14;
const MAX_F64_KERNEL_TILE: usize = MAX_F64_KERNEL_MR * MAX_F64_KERNEL_NR;
const BLOCKED_F64_VOLUME_THRESHOLD: usize = 96 * 96 * 96;

type GemmMicrokernelFn = unsafe fn(usize, f64, *const f64, *const f64, f64, *mut f64, usize, usize);

#[derive(Clone, Copy)]
struct GemmMicrokernel {
    mr: usize,
    nr: usize,
    kc: usize,
    func: GemmMicrokernelFn,
}

impl GemmMicrokernel {
    unsafe fn run(
        &self,
        k: usize,
        alpha: f64,
        a: *const f64,
        b: *const f64,
        beta: f64,
        c: *mut f64,
        rs_c: usize,
        cs_c: usize,
    ) {
        unsafe {
            (self.func)(k, alpha, a, b, beta, c, rs_c, cs_c);
        }
    }
}

pub fn basic_gemm_kernel<A, B, C, T>(a: &A, b: &B, out: &mut C, alpha: Option<T>, beta: Option<T>)
where
    A: MatrixRead<T>,
    B: MatrixRead<T>,
    C: MatrixWrite<T>,
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
        b.rows(),
        "Inner dimensions {} and {} must match for matrix multiplication",
        a.cols(),
        b.rows()
    );
    let m = a.rows();
    let k_dim = a.cols();
    let n = b.cols();
    let beta = beta.unwrap_or(T::from(0.0));
    let alpha = alpha.unwrap_or(T::from(1.0));

    // beta scaling block: skip if beta is zero
    if beta != T::from(1.0) {
        for i in 0..m {
            for j in 0..n {
                out.accumulate(i, j, (beta - T::from(1.0)) * (*out.get(i, j)));
            }
        }
    }

    // Matrix multiplication block: skip if alpha is zero
    if alpha != T::from(0.0) {
        for i in 0..m {
            for k in 0..k_dim {
                let a_ik = a.get(i, k);
                for j in 0..n {
                    let b_kj = b.get(k, j);
                    out.accumulate(i, j, (*a_ik) * (*b_kj) * alpha);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GemmBlocking {
    pub mc: usize,
    pub kc: usize,
    pub nc: usize,
}

impl Default for GemmBlocking {
    fn default() -> Self {
        Self {
            mc: 128,
            kc: 256,
            nc: 128,
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_microkernel_f64_4x4_x86_64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    _cs_c: usize,
) {
    // Initialize accumulators in registers to zero
    let mut c0 = _mm256_setzero_pd();
    let mut c1 = _mm256_setzero_pd();
    let mut c2 = _mm256_setzero_pd();
    let mut c3 = _mm256_setzero_pd();

    unsafe {
        for p in 0..k {
            // Load a 4x1 column from A
            let a_p = _mm256_loadu_pd(a.add(p * 4));

            // Load a 1x4 row from B and broadcast each element to a 4-element vector
            let b0 = _mm256_broadcast_sd(&*b.add(p * 4 + 0));
            let b1 = _mm256_broadcast_sd(&*b.add(p * 4 + 1));
            let b2 = _mm256_broadcast_sd(&*b.add(p * 4 + 2));
            let b3 = _mm256_broadcast_sd(&*b.add(p * 4 + 3));

            // Perform the fused multiply-add operation: c += a * b
            c0 = _mm256_fmadd_pd(a_p, b0, c0);
            c1 = _mm256_fmadd_pd(a_p, b1, c1);
            c2 = _mm256_fmadd_pd(a_p, b2, c2);
            c3 = _mm256_fmadd_pd(a_p, b3, c3);
        }

        let v_alpha = _mm256_set1_pd(alpha);
        if beta == 0.0 {
            // If beta is zero, we can directly store the results scaled by alpha
            _mm256_storeu_pd(c.add(0 * rs_c), _mm256_mul_pd(v_alpha, c0));
            _mm256_storeu_pd(c.add(1 * rs_c), _mm256_mul_pd(v_alpha, c1));
            _mm256_storeu_pd(c.add(2 * rs_c), _mm256_mul_pd(v_alpha, c2));
            _mm256_storeu_pd(c.add(3 * rs_c), _mm256_mul_pd(v_alpha, c3));
        } else {
            // If beta is not zero, we need to scale the existing values in C by beta and add the new results
            let vbeta = _mm256_set1_pd(beta);
            for (j, c_j) in [c0, c1, c2, c3].into_iter().enumerate() {
                let c_ptr = c.add(j * rs_c);
                let c_old = _mm256_loadu_pd(c_ptr);
                let c_new = _mm256_fmadd_pd(v_alpha, c_j, _mm256_mul_pd(vbeta, c_old));
                _mm256_storeu_pd(c_ptr, c_new);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_microkernel_f64_8x4_x86_64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    _cs_c: usize,
) {
    let mut c_top = [_mm256_setzero_pd(); 4];
    let mut c_bottom = [_mm256_setzero_pd(); 4];

    unsafe {
        for p in 0..k {
            let a_top = _mm256_loadu_pd(a.add(p * 8));
            let a_bottom = _mm256_loadu_pd(a.add(p * 8 + 4));

            for j in 0..4 {
                let b_pj = _mm256_set1_pd(*b.add(p * 4 + j));
                c_top[j] = _mm256_fmadd_pd(a_top, b_pj, c_top[j]);
                c_bottom[j] = _mm256_fmadd_pd(a_bottom, b_pj, c_bottom[j]);
            }
        }

        let v_alpha = _mm256_set1_pd(alpha);
        if beta == 0.0 {
            for j in 0..4 {
                let c_ptr = c.add(j * rs_c);
                _mm256_storeu_pd(c_ptr, _mm256_mul_pd(v_alpha, c_top[j]));
                _mm256_storeu_pd(c_ptr.add(4), _mm256_mul_pd(v_alpha, c_bottom[j]));
            }
        } else {
            let v_beta = _mm256_set1_pd(beta);
            for j in 0..4 {
                let c_ptr = c.add(j * rs_c);
                let c_old_top = _mm256_loadu_pd(c_ptr);
                let c_old_bottom = _mm256_loadu_pd(c_ptr.add(4));
                let c_new_top =
                    _mm256_fmadd_pd(v_alpha, c_top[j], _mm256_mul_pd(v_beta, c_old_top));
                let c_new_bottom =
                    _mm256_fmadd_pd(v_alpha, c_bottom[j], _mm256_mul_pd(v_beta, c_old_bottom));
                _mm256_storeu_pd(c_ptr, c_new_top);
                _mm256_storeu_pd(c_ptr.add(4), c_new_bottom);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_microkernel_f64_8x8_x86_64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    _cs_c: usize,
) {
    let mut c_accumulators = [_mm512_setzero_pd(); 8];

    unsafe {
        for p in 0..k {
            let a_p = _mm512_loadu_pd(a.add(p * 8));

            for j in 0..8 {
                let b_pj = _mm512_set1_pd(*b.add(p * 8 + j));
                c_accumulators[j] = _mm512_fmadd_pd(a_p, b_pj, c_accumulators[j]);
            }
        }

        let v_alpha = _mm512_set1_pd(alpha);
        if beta == 0.0 {
            for j in 0..8 {
                let c_ptr = c.add(j * rs_c);
                _mm512_storeu_pd(c_ptr, _mm512_mul_pd(v_alpha, c_accumulators[j]));
            }
        } else {
            let vbeta = _mm512_set1_pd(beta);
            for j in 0..8 {
                let c_ptr = c.add(j * rs_c);
                let c_old = _mm512_loadu_pd(c_ptr);
                let c_new =
                    _mm512_fmadd_pd(v_alpha, c_accumulators[j], _mm512_mul_pd(vbeta, c_old));
                _mm512_storeu_pd(c_ptr, c_new);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemm_microkernel_f64_4x4_aarch64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    _cs_c: usize,
) {
    // Initialize accumulators in NEON registers to zero
    let mut c0 = vdupq_n_f64(0.0);
    let mut c1 = vdupq_n_f64(0.0);
    let mut c2 = vdupq_n_f64(0.0);
    let mut c3 = vdupq_n_f64(0.0);

    unsafe {
        for p in 0..k {
            // Load a 4x1 column from A
            let a_p = vld1q_f64(a.add(p * 4));

            // Load a 1x4 row from B and broadcast each element to a 4-element vector
            let b0 = vdupq_n_f64(*b.add(p * 4 + 0));
            let b1 = vdupq_n_f64(*b.add(p * 4 + 1));
            let b2 = vdupq_n_f64(*b.add(p * 4 + 2));
            let b3 = vdupq_n_f64(*b.add(p * 4 + 3));

            // Perform the fused multiply-add operation: c += a * b
            c0 = vfmaq_f64(c0, a_p, b0);
            c1 = vfmaq_f64(c1, a_p, b1);
            c2 = vfmaq_f64(c2, a_p, b2);
            c3 = vfmaq_f64(c3, a_p, b3);
        }
    }

    unsafe {
        let v_alpha = vdupq_n_f64(alpha);
        if beta == 0.0 {
            // If beta is zero, we can directly store the results scaled by alpha
            vst1q_f64(c.add(0 * rs_c), vmulq_f64(v_alpha, c0));
            vst1q_f64(c.add(1 * rs_c), vmulq_f64(v_alpha, c1));
            vst1q_f64(c.add(2 * rs_c), vmulq_f64(v_alpha, c2));
            vst1q_f64(c.add(3 * rs_c), vmulq_f64(v_alpha, c3));
        } else {
            // If beta is not zero, we need to scale the existing values in C by beta and add the new results
            let vbeta = vdupq_n_f64(beta);
            for (j, c_j) in [c0, c1, c2, c3].into_iter().enumerate() {
                let c_ptr = c.add(j * rs_c);
                let c_old = vld1q_f64(c_ptr);
                let c_new = vfmaq_f64(vmulq_f64(vbeta, c_old), v_alpha, c_j);
                vst1q_f64(c_ptr, c_new);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemm_microkernel_f64_6x8_aarch64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    _cs_c: usize,
) {
    let mut c0 = [vdupq_n_f64(0.0); 8];
    let mut c1 = [vdupq_n_f64(0.0); 8];
    let mut c2 = [vdupq_n_f64(0.0); 8];

    unsafe {
        for p in 0..k {
            let a0 = vld1q_f64(a.add(p * 6));
            let a1 = vld1q_f64(a.add(p * 6 + 2));
            let a2 = vld1q_f64(a.add(p * 6 + 4));
            for j in 0..8 {
                let b_pj = vdupq_n_f64(*b.add(p * 8 + j));
                c0[j] = vfmaq_f64(c0[j], a0, b_pj);
                c1[j] = vfmaq_f64(c1[j], a1, b_pj);
                c2[j] = vfmaq_f64(c2[j], a2, b_pj);
            }
        }

        let v_alpha = vdupq_n_f64(alpha);
        if beta == 0.0 {
            for j in 0..8 {
                let c_ptr = c.add(j * rs_c);
                vst1q_f64(c_ptr, vmulq_f64(v_alpha, c0[j]));
                vst1q_f64(c_ptr.add(2), vmulq_f64(v_alpha, c1[j]));
                vst1q_f64(c_ptr.add(4), vmulq_f64(v_alpha, c2[j]));
            }
        } else {
            let vbeta = vdupq_n_f64(beta);
            for j in 0..8 {
                let c_ptr = c.add(j * rs_c);
                let c_old0 = vld1q_f64(c_ptr);
                let c_old1 = vld1q_f64(c_ptr.add(2));
                let c_old2 = vld1q_f64(c_ptr.add(4));
                let c_new0 = vfmaq_f64(vmulq_f64(vbeta, c_old0), v_alpha, c0[j]);
                let c_new1 = vfmaq_f64(vmulq_f64(vbeta, c_old1), v_alpha, c1[j]);
                let c_new2 = vfmaq_f64(vmulq_f64(vbeta, c_old2), v_alpha, c2[j]);
                vst1q_f64(c_ptr, c_new0);
                vst1q_f64(c_ptr.add(2), c_new1);
                vst1q_f64(c_ptr.add(4), c_new2);
            }
        }
    }
}

unsafe fn gemm_microkernel_f64_4x4_scalar(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    _cs_c: usize,
) {
    unsafe {
        for i in 0..SCALAR_MR_F64 {
            for j in 0..SCALAR_NR_F64 {
                let mut acc = 0.0f64;
                for p in 0..k {
                    let a_ip = *a.add(p * SCALAR_MR_F64 + i);
                    let b_pj = *b.add(p * SCALAR_NR_F64 + j);
                    acc += a_ip * b_pj;
                }

                let c_ptr = c.add(j * rs_c + i);
                let updated = if beta == 0.0 {
                    alpha * acc
                } else {
                    beta * (*c_ptr) + alpha * acc
                };
                *c_ptr = updated;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn call_gemm_microkernel_f64_4x4_x86_64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    cs_c: usize,
) {
    unsafe { gemm_microkernel_f64_4x4_x86_64(k, alpha, a, b, beta, c, rs_c, cs_c) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn call_gemm_microkernel_f64_4x4_aarch64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    cs_c: usize,
) {
    unsafe { gemm_microkernel_f64_4x4_aarch64(k, alpha, a, b, beta, c, rs_c, cs_c) }
}

fn scalar_4x4_kernel() -> GemmMicrokernel {
    GemmMicrokernel {
        mr: SCALAR_MR_F64,
        nr: SCALAR_NR_F64,
        kc: 256,
        func: gemm_microkernel_f64_4x4_scalar,
    }
}

fn gemm_microkernel_override() -> Option<String> {
    env::var("FERRUM_GEMM_F64_KERNEL")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty() && value != "auto")
}

fn gemm_microkernel_default() -> GemmMicrokernel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
            return GemmMicrokernel {
                mr: 8,
                nr: 8,
                kc: 128,
                func: call_gemm_microkernel_f64_8x8_x86_64,
            };
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return GemmMicrokernel {
                mr: 8,
                nr: 4,
                kc: 192,
                func: call_gemm_microkernel_f64_8x4_x86_64,
            };
        }

        return scalar_4x4_kernel();
    }

    #[cfg(target_arch = "aarch64")]
    {
        return GemmMicrokernel {
            mr: 6,
            nr: 8,
            kc: 192,
            func: call_gemm_microkernel_f64_6x8_aarch64,
        };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar_4x4_kernel()
}

fn gemm_microkernel_from_override(override_name: &str) -> Option<GemmMicrokernel> {
    #[cfg(target_arch = "x86_64")]
    {
        if override_name == "scalar-4x4" {
            return Some(scalar_4x4_kernel());
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            if override_name == "avx2-4x4" {
                return Some(GemmMicrokernel {
                    mr: 4,
                    nr: 4,
                    kc: 256,
                    func: call_gemm_microkernel_f64_4x4_x86_64,
                });
            }

            if override_name == "avx2-8x4" {
                return Some(GemmMicrokernel {
                    mr: 8,
                    nr: 4,
                    kc: 192,
                    func: call_gemm_microkernel_f64_8x4_x86_64,
                });
            }
        }

        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("fma")
            && override_name == "avx512-8x8"
        {
            return Some(GemmMicrokernel {
                mr: 8,
                nr: 8,
                kc: 128,
                func: call_gemm_microkernel_f64_8x8_x86_64,
            });
        }

        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("fma")
            && override_name == "avx512-16x14"
        {
            return Some(GemmMicrokernel {
                mr: 8,
                nr: 8,
                kc: 128,
                func: call_gemm_microkernel_f64_8x8_x86_64,
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if override_name == "scalar-4x4" {
            return Some(scalar_4x4_kernel());
        }

        if override_name == "neon-4x4" {
            return Some(GemmMicrokernel {
                mr: 4,
                nr: 4,
                kc: 256,
                func: call_gemm_microkernel_f64_4x4_aarch64,
            });
        }

        if override_name == "neon-6x8" {
            return Some(GemmMicrokernel {
                mr: 6,
                nr: 8,
                kc: 192,
                func: call_gemm_microkernel_f64_6x8_aarch64,
            });
        }
    }

    None
}

#[cfg(target_arch = "x86_64")]
unsafe fn call_gemm_microkernel_f64_8x4_x86_64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    cs_c: usize,
) {
    unsafe { gemm_microkernel_f64_8x4_x86_64(k, alpha, a, b, beta, c, rs_c, cs_c) }
}

#[cfg(target_arch = "x86_64")]
unsafe fn call_gemm_microkernel_f64_8x8_x86_64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    cs_c: usize,
) {
    unsafe { gemm_microkernel_f64_8x8_x86_64(k, alpha, a, b, beta, c, rs_c, cs_c) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn call_gemm_microkernel_f64_6x8_aarch64(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    rs_c: usize,
    cs_c: usize,
) {
    unsafe { gemm_microkernel_f64_6x8_aarch64(k, alpha, a, b, beta, c, rs_c, cs_c) }
}

fn gemm_microkernel_dispatcher() -> GemmMicrokernel {
    if let Some(override_name) = gemm_microkernel_override() {
        if let Some(kernel) = gemm_microkernel_from_override(&override_name) {
            return kernel;
        }
    }

    gemm_microkernel_default()
}

pub fn matmul_blocked<A, B, C, T>(
    a: &A,
    b: &B,
    out: &mut C,
    alpha: Option<T>,
    beta: Option<T>,
    blocking: GemmBlocking,
) where
    A: MatrixRead<T>,
    B: MatrixRead<T>,
    C: MatrixWrite<T>,
    T: Copy
        + ops::Add<Output = T>
        + ops::Mul<Output = T>
        + Default
        + PartialEq
        + From<f64>
        + ops::AddAssign<T>
        + ops::Sub<Output = T>
        + 'static,
{
    assert_eq!(
        a.cols(),
        b.rows(),
        "Inner dimensions {} and {} must match for matrix multiplication",
        a.cols(),
        b.rows()
    );

    let m = a.rows();
    let k = a.cols();
    let n = b.cols();
    assert_eq!(out.rows(), m, "Output rows must match A rows");
    assert_eq!(out.cols(), n, "Output cols must match B cols");

    let alpha = alpha.unwrap_or(T::from(1.0));
    let beta = beta.unwrap_or(T::from(0.0));

    // The optimized microkernel currently operates on f64-packed tiles.
    // For other element types, keep behavior correct via the scalar kernel.
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        basic_gemm_kernel(a, b, out, Some(alpha), Some(beta));
        return;
    }

    // For very small problems, packing overhead can dominate; the basic kernel
    // is typically faster than blocked dispatch in this regime.
    if m.saturating_mul(n).saturating_mul(k) < BLOCKED_F64_VOLUME_THRESHOLD {
        basic_gemm_kernel(a, b, out, Some(alpha), Some(beta));
        return;
    }

    let alpha_f64 = unsafe { *(&alpha as *const T as *const f64) };
    let beta_f64 = unsafe { *(&beta as *const T as *const f64) };

    let kernel = gemm_microkernel_dispatcher();

    let mc = (blocking.mc / kernel.mr).max(1) * kernel.mr;
    let kc = blocking.kc.min(kernel.kc).max(1);
    let nc = (blocking.nc / kernel.nr).max(1) * kernel.nr;
    let mut a_pack = Vec::new();
    let mut b_packs = Vec::new();

    let to_f64 = |v: &T| -> f64 {
        // Safe because this closure is used only in the TypeId::<T>() == TypeId::<f64>() branch.
        unsafe { *(v as *const T as *const f64) }
    };

    for jc in (0..n).step_by(nc) {
        let j_end = (jc + nc).min(n);

        for pc in (0..k).step_by(kc) {
            let p_end = (pc + kc).min(k);
            let k_panel = p_end - pc;
            let beta_panel = if pc == 0 { beta_f64 } else { 1.0 };

            // Reuse packing buffers within this k-panel to avoid per-tile allocations.
            a_pack.resize(kernel.mr * k_panel, 0.0);
            let b_pack_stride = kernel.nr * k_panel;
            let full_j_tiles = (j_end - jc) / kernel.nr;
            let full_j_end = jc + full_j_tiles * kernel.nr;
            b_packs.resize(full_j_tiles * b_pack_stride, 0.0);

            // Pack each full NR tile of B once and reuse across all i-tiles.
            for tile_idx in 0..full_j_tiles {
                let j0 = jc + tile_idx * kernel.nr;
                let b_base = tile_idx * b_pack_stride;
                for p in 0..k_panel {
                    for jj in 0..kernel.nr {
                        b_packs[b_base + p * kernel.nr + jj] = to_f64(b.get(pc + p, j0 + jj));
                    }
                }
            }

            for ic in (0..m).step_by(mc) {
                let i_end = (ic + mc).min(m);
                let mut c_tile = [0.0f64; MAX_F64_KERNEL_TILE];

                let mut i0 = ic;
                while i0 + kernel.mr <= i_end {
                    for p in 0..k_panel {
                        for ii in 0..kernel.mr {
                            a_pack[p * kernel.mr + ii] = to_f64(a.get(i0 + ii, pc + p));
                        }
                    }

                    for tile_idx in 0..full_j_tiles {
                        let j0 = jc + tile_idx * kernel.nr;
                        let b_ptr = unsafe { b_packs.as_ptr().add(tile_idx * b_pack_stride) };

                        for ii in 0..kernel.mr {
                            for jj in 0..kernel.nr {
                                c_tile[jj * kernel.mr + ii] = to_f64(out.get(i0 + ii, j0 + jj));
                            }
                        }

                        unsafe {
                            kernel.run(
                                k_panel,
                                alpha_f64,
                                a_pack.as_ptr(),
                                b_ptr,
                                beta_panel,
                                c_tile.as_mut_ptr(),
                                kernel.mr,
                                1,
                            );
                        }

                        for ii in 0..kernel.mr {
                            for jj in 0..kernel.nr {
                                *out.get_mut(i0 + ii, j0 + jj) =
                                    T::from(c_tile[jj * kernel.mr + ii]);
                            }
                        }
                    }

                    if full_j_end < j_end {
                        for i in i0..(i0 + kernel.mr) {
                            for j in full_j_end..j_end {
                                let mut acc = 0.0f64;
                                for p in pc..p_end {
                                    acc += to_f64(a.get(i, p)) * to_f64(b.get(p, j));
                                }

                                let c_old = to_f64(out.get(i, j));
                                let c_new = if pc == 0 {
                                    beta_f64 * c_old + alpha_f64 * acc
                                } else {
                                    c_old + alpha_f64 * acc
                                };
                                *out.get_mut(i, j) = T::from(c_new);
                            }
                        }
                    }

                    i0 += kernel.mr;
                }

                if i0 < i_end {
                    for i in i0..i_end {
                        for j in jc..j_end {
                            let mut acc = 0.0f64;
                            for p in pc..p_end {
                                acc += to_f64(a.get(i, p)) * to_f64(b.get(p, j));
                            }

                            let c_old = to_f64(out.get(i, j));
                            let c_new = if pc == 0 {
                                beta_f64 * c_old + alpha_f64 * acc
                            } else {
                                c_old + alpha_f64 * acc
                            };
                            *out.get_mut(i, j) = T::from(c_new);
                        }
                    }
                }
            }
        }
    }
}
