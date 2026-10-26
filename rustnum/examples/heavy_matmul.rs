//! A deliberately heavy matrix workload for comparing NumPy and rustnum.
//!
//! Run with:
//! `cargo run --release -p rustnum --example heavy_matmul`
//!
//! The default workload performs six rounds of two 2048 x 2048 matrix multiplications.
//! Set `SIZE` and `ITERATIONS` to tune the workload:
//! ` $env:SIZE=1536; $env:ITERATIONS=5; cargo run --release -p rustnum --example heavy_matmul`
//!
//! The equivalent NumPy workload is:
//!
//! ```python
//! import os
//! import time
//! import numpy as np
//!
//! size = int(os.environ.get("SIZE", 1024))
//! iterations = int(os.environ.get("ITERATIONS", 5))
//! rng = np.random.default_rng(42)
//! a = rng.standard_normal((size, size))
//! b = rng.standard_normal((size, size))
//!
//! start = time.perf_counter()
//! for _ in range(iterations):
//!     a = np.tanh(a @ b / size**0.5)
//!     b = np.tanh(b @ a / size**0.5)
//! print(f"elapsed: {time.perf_counter() - start:.3f}s")
//! print(a.mean(), b.mean())
//! ```
//!
//! Run the Python version with the same `SIZE` and `ITERATIONS` values.
//! Timing depends on the CPU, BLAS implementation, and number of available
//! threads, so the workload is intentionally configurable rather than tied to
//! an exact duration.

use std::env;
use std::time::Instant;

use rustnum as np;

fn read_positive_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a positive integer")),
        Err(_) => default,
    }
}

fn main() {
    let size = read_positive_usize("SIZE", 8192);
    let iterations = read_positive_usize("ITERATIONS", 12);
    let scale = (size as f64).sqrt();

    let mut rng = np::random::Rng::seed(42);
    let mut a = rng.standard_normal(&[size, size]);
    let mut b = rng.standard_normal(&[size, size]);

    println!(
        "workload: {size}x{size}, {iterations} iterations ({} matmuls)",
        iterations * 2
    );

    let start = Instant::now();
    for _ in 0..iterations {
        a = (a.matmul(&b) / scale).tanh();
        b = (b.matmul(&a) / scale).tanh();
    }
    let elapsed = start.elapsed();

    println!("elapsed: {:.3}s", elapsed.as_secs_f64());
    println!(
        "checksums: a.mean() = {:.9}, b.mean() = {:.9}",
        a.mean(),
        b.mean()
    );
}
