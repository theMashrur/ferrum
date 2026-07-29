use std::hint::black_box;
use std::time::{Duration, Instant};

use ferrum_core::algorithms::gemm::{GemmBlocking, basic_gemm_kernel, matmul_blocked};
use ferrum_core::core::matrix::Matrix;

fn lcg_next(state: &mut u64) -> u64 {
    // Deterministic pseudo-random stream for reproducible inputs.
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn make_matrix(rows: usize, cols: usize, seed: u64) -> Matrix<f64> {
    let mut state = seed;
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        let x = lcg_next(&mut state);
        let v = ((x >> 11) as f64) / ((1u64 << 53) as f64); // [0, 1)
        data.push(v - 0.5);
    }
    Matrix::from_data(rows, cols, data)
}

fn checksum(m: &Matrix<f64>) -> f64 {
    m.data.iter().fold(0.0, |acc, &v| acc + v)
}

fn time_basic(a: &Matrix<f64>, b: &Matrix<f64>, runs: usize) -> (Duration, f64) {
    let mut total = Duration::ZERO;
    let mut chk = 0.0;
    for _ in 0..runs {
        let mut out = Matrix::<f64>::zeros(a.rows, b.cols);
        let start = Instant::now();
        basic_gemm_kernel(a, b, &mut out, None, None);
        total += start.elapsed();
        chk = checksum(&out);
        black_box(chk);
    }
    (total / (runs as u32), chk)
}

fn time_blocked(a: &Matrix<f64>, b: &Matrix<f64>, runs: usize) -> (Duration, f64) {
    let mut total = Duration::ZERO;
    let mut chk = 0.0;
    for _ in 0..runs {
        let mut out = Matrix::<f64>::zeros(a.rows, b.cols);
        let start = Instant::now();
        matmul_blocked(a, b, &mut out, None, None, GemmBlocking::default());
        total += start.elapsed();
        chk = checksum(&out);
        black_box(chk);
    }
    (total / (runs as u32), chk)
}

fn benchmark_size(n: usize, warmup_runs: usize, runs: usize) {
    println!("\n== Size: {n} x {n} ==");

    let a = make_matrix(n, n, 0xDEADBEEF_u64 ^ (n as u64));
    let b = make_matrix(n, n, 0xABCDEF01_u64 ^ (n as u64));

    for _ in 0..warmup_runs {
        let mut out1 = Matrix::<f64>::zeros(n, n);
        basic_gemm_kernel(&a, &b, &mut out1, None, None);
        black_box(checksum(&out1));

        let mut out2 = Matrix::<f64>::zeros(n, n);
        matmul_blocked(&a, &b, &mut out2, None, None, GemmBlocking::default());
        black_box(checksum(&out2));
    }

    let (basic_avg, basic_chk) = time_basic(&a, &b, runs);
    let (blocked_avg, blocked_chk) = time_blocked(&a, &b, runs);

    let speedup = basic_avg.as_secs_f64() / blocked_avg.as_secs_f64();
    let flops = 2.0 * (n as f64) * (n as f64) * (n as f64);
    let basic_gflops = flops / basic_avg.as_secs_f64() / 1e9;
    let blocked_gflops = flops / blocked_avg.as_secs_f64() / 1e9;

    println!(
        "basic   avg: {:>10.4} ms  ({:>7.2} GFLOP/s)",
        basic_avg.as_secs_f64() * 1e3,
        basic_gflops
    );
    println!(
        "blocked avg: {:>10.4} ms  ({:>7.2} GFLOP/s)",
        blocked_avg.as_secs_f64() * 1e3,
        blocked_gflops
    );
    println!("speedup: {:>10.2}x", speedup);
    println!(
        "checksums (basic/blocked): {:.6} / {:.6}",
        basic_chk, blocked_chk
    );
}

fn main() {
    // Use release mode for meaningful numbers:
    // cargo run -p ferrum-core --release --example gemm_compare
    let warmup_runs = 2;
    let runs = 6;

    println!("GEMM benchmark: basic_gemm_kernel vs matmul_blocked");
    println!("warmup runs: {warmup_runs}, measured runs: {runs}");

    benchmark_size(128, warmup_runs, runs);
    benchmark_size(256, warmup_runs, runs);
    benchmark_size(384, warmup_runs, runs);
}
