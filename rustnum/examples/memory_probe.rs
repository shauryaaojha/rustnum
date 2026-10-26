//! Small, bounded memory benchmarks; never runs the heavy_matmul example.
//!
//! * `memory_probe` prints G2/G3/G4 timings with the buffer pool off and on.
//! * `memory_probe pool` times alloc/free loops at 4096^2 and 8192^2 with the pool off and on.
//! * `memory_probe thresholds` prints `op n ms`; run it once with `RAYON_NUM_THREADS=1` and once
//!   without, and the crossover where the parallel run wins is the per-op threshold in
//!   `src/kernels_mem.rs`.
use rustnum as np;
use std::{hint::black_box, time::Instant};
/// Minimum over `n` runs after warm-up; the minimum is robust to other processes using the CPU.
fn time(mut f: impl FnMut(), n: usize) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut best = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.);
    }
    best
}
fn thresholds() {
    for n in [
        8192usize, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
    ] {
        let side = (n as f64).sqrt() as usize;
        let mut a = np::full([n], 0.5f64);
        let b = np::full([n], 0.25f64);
        println!("unary_tanh {n} {:.4}", time(|| a.tanh_inplace(), 40));
        println!(
            "sum_all {n} {:.4}",
            time(
                || {
                    black_box(a.sum());
                },
                40
            )
        );
        println!(
            "max_all {n} {:.4}",
            time(
                || {
                    black_box(a.max());
                },
                40
            )
        );
        println!("cheap_mul_scalar {n} {:.4}", time(|| a *= 1.0000001, 40));
        println!("binary_add_assign {n} {:.4}", time(|| a += &b, 40));
        let m = np::Array::from_shape_fn([side, side], |i| (i[0] + i[1]) as f64);
        let t = m.t();
        println!(
            "transpose_copy {} {:.4}",
            side * side,
            time(
                || {
                    black_box(t.to_contiguous());
                },
                20
            )
        );
        println!(
            "sum_axis0 {} {:.4}",
            side * side,
            time(
                || {
                    black_box(m.sum_axis(0));
                },
                20
            )
        );
        println!(
            "sum_axis1 {} {:.4}",
            side * side,
            time(
                || {
                    black_box(m.sum_axis(1));
                },
                20
            )
        );
        let row = np::ones::<f64>([side]);
        println!(
            "broadcast_add {} {:.4}",
            side * side,
            time(
                || {
                    black_box(&m + &row);
                },
                20
            )
        );
    }
}
fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    if arg == "thresholds" {
        thresholds();
        return;
    }
    if arg == "pool" {
        for n in [4096usize, 8192] {
            let bytes = n * n * 8;
            for cap in [0, bytes] {
                np::memory::set_pool_limit(cap);
                np::memory::clear_pool();
                let ms = time(
                    || {
                        let a = np::full([n, n], black_box(1f64));
                        black_box(a.as_slice().unwrap());
                    },
                    8,
                );
                println!(
                    "pool n={n} cap={cap}: {ms:.3} ms {:?}",
                    np::memory::pool_stats()
                );
            }
        }
        return;
    }
    let a = np::Array::from_shape_fn([1000, 1000], |i| (i[0] + i[1]) as f64 / 1000.);
    let row = np::ones::<f64>([1000]);
    for cap in [0, 512 << 20] {
        np::memory::set_pool_limit(cap);
        np::memory::clear_pool();
        println!(
            "cap={cap} G2 mean_axis0: {:.3} ms",
            time(
                || {
                    black_box(a.mean_axis(0));
                },
                100
            )
        );
        println!(
            "cap={cap} G3 transpose: {:.3} ms",
            time(
                || {
                    black_box(a.t().to_contiguous());
                },
                50
            )
        );
        println!(
            "cap={cap} G4 broadcast: {:.3} ms",
            time(
                || {
                    black_box(&a + &row);
                },
                50
            )
        );
    }
}
