//! The `heavy_matmul` workload computed with rustnum's fused GEMM epilogue.
//!
//! Same math as `heavy_matmul.rs` and `heavy_matmul.py`:
//!
//! ```python
//! a = np.tanh((a @ b) / scale)
//! b = np.tanh((b @ a) / scale)
//! ```
//!
//! but `np::linalg::matmul_map` applies `/ scale` and `tanh` to each output tile while it is
//! still in cache, so the 512 MB result is written once instead of three times.
//!
//! Run with:
//! `cargo run --release -p rustnum --example heavy_matmul_fused`
//!
//! Tune with `SIZE` and `ITERATIONS`, e.g. `$env:SIZE=4096; $env:ITERATIONS=6`.

use std::env;
use std::time::Instant;

use rustnum as np;

fn read_positive_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(value) => value
            .parse()
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or_else(|| panic!("{name} must be a positive integer")),
        Err(_) => default,
    }
}

fn main() {
    let size = read_positive_usize("SIZE", 8192);
    let iterations = read_positive_usize("ITERATIONS", 12);
    let inv_scale = 1.0 / (size as f64).sqrt();

    let mut rng = np::random::Rng::seed(42);
    let mut a = rng.standard_normal(&[size, size]);
    let mut b = rng.standard_normal(&[size, size]);

    println!(
        "workload: {size}x{size}, {iterations} iterations ({} fused matmuls)",
        iterations * 2
    );

    let start = Instant::now();
    for _ in 0..iterations {
        a = np::linalg::matmul_map(&a, &b, inv_scale, f64::tanh);
        b = np::linalg::matmul_map(&b, &a, inv_scale, f64::tanh);
    }
    let elapsed = start.elapsed();

    println!("elapsed: {:.3}s", elapsed.as_secs_f64());
    println!(
        "checksums: a.mean() = {:.9}, b.mean() = {:.9}",
        a.mean(),
        b.mean()
    );
}
