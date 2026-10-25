//! Benchmarks that are runnable before the Array-facing linalg adapters land.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

#[allow(dead_code)]
#[path = "../src/linalg/kernels.rs"]
mod kernels;

fn data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| ((i * 17 % 101) as f64 - 50.0) / 17.0)
        .collect()
}

fn gemm_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f64");
    for dimension in [256_usize, 512, 1024] {
        let left = data(dimension * dimension);
        let right = data(dimension * dimension);
        group.bench_with_input(
            BenchmarkId::from_parameter(dimension),
            &dimension,
            |bench, &n| {
                bench.iter(|| {
                    let mut output = vec![0.0; n * n];
                    kernels::gemm_f64(
                        n,
                        n,
                        n,
                        1.0,
                        &left,
                        n as isize,
                        1,
                        &right,
                        n as isize,
                        1,
                        0.0,
                        &mut output,
                        n as isize,
                        1,
                    );
                    black_box(output);
                });
            },
        );
    }
    group.finish();
}

fn decomposition_benches(c: &mut Criterion) {
    let inverse_input = {
        let mut values = data(200 * 200);
        for i in 0..200 {
            values[i * 200 + i] += 250.0;
        }
        values
    };
    c.bench_function("inv_f64_200", |bench| {
        bench.iter(|| black_box(kernels::inv(black_box(&inverse_input), 200).unwrap()))
    });

    let svd_input = data(100 * 100);
    c.bench_function("svd_f64_100", |bench| {
        bench.iter(|| black_box(kernels::svd(black_box(&svd_input), 100, 100).unwrap()))
    });
}

criterion_group!(core, gemm_benches, decomposition_benches);
criterion_main!(core);
