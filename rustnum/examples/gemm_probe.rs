//! GEMM tuning probe: best-of-N GF/s of `a.matmul(&a)` for f64 and f32, on a one-thread rayon
//! pool and on all threads. It is much lighter than `heavy_matmul` (at most 4096x4096).
//!
//! ```text
//! cargo run --release --example gemm_probe              # 1024, 2048, 4096
//! cargo run --release --example gemm_probe -- --small   # 64..512, to find the parallel crossover
//! cargo run --release --example gemm_probe -- --n=2048 --threads=1
//! ```
//!
//! Pin the process (for example `start /affinity 1` on Windows) to compare P-cores fairly; other
//! running programs make the numbers noisy, so take the best of several runs.
use rustnum as np;
use std::hint::black_box;
use std::time::Instant;

fn measure<F: np::linalg::LinalgFloat>(n: usize, threads: usize, repeats: usize) {
    let a = np::Array::from_shape_fn([n, n], |c| {
        F::from_f64(((c[0] * 17 + c[1] * 7) % 101) as f64 / 101.0 - 0.5)
    });
    black_box(a.matmul(&a));
    let mut samples = Vec::new();
    for _ in 0..if n >= 4096 { 2 } else { 6 } {
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(a.matmul(&a));
        }
        samples.push(start.elapsed().as_secs_f64() / repeats as f64);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{} n={n} threads={threads} ms={:.3} GF/s={:.2}",
        F::DTYPE,
        samples[0] * 1000.0,
        2.0 * (n as f64).powi(3) / samples[0] / 1e9
    );
}

fn sweep(threads: usize, small: bool) {
    let sizes: &[usize] = if small {
        &[64, 96, 128, 192, 256, 384, 512]
    } else {
        &[1024, 2048, 4096]
    };
    // `--n=N` measures one size.
    let one: Option<usize> = std::env::args().find_map(|a| a.strip_prefix("--n=")?.parse().ok());
    let single = [one.unwrap_or(0)];
    for &n in if one.is_some() { &single[..] } else { sizes } {
        let repeats = if small {
            (16_777_216 / (n * n * n)).max(3)
        } else {
            1
        };
        measure::<f64>(n, threads, repeats);
        measure::<f32>(n, threads, repeats);
    }
}

fn main() {
    let small = std::env::args().any(|arg| arg == "--small");
    // `--threads=N` restricts the sweep to one pool size.
    let only: Option<usize> =
        std::env::args().find_map(|a| a.strip_prefix("--threads=")?.parse().ok());
    #[cfg(feature = "rayon")]
    for threads in [
        1,
        std::thread::available_parallelism().map_or(1, usize::from),
    ] {
        if only.is_some_and(|t| t != threads) {
            continue;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| sweep(threads, small));
    }
    #[cfg(not(feature = "rayon"))]
    {
        let _ = only;
        sweep(1, small);
    }
}
