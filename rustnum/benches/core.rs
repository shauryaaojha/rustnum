//! Criterion benchmarks. Array-level names match `tools/bench_numpy.py`; `kernel_*` benchmarks
//! time the slice-level linear-algebra kernels directly.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rustnum as np;
use rustnum::Num;
use std::hint::black_box;

// `kernels.rs` is included below as benchmark-local source, so its crate-local helper imports
// need equivalents in this benchmark crate.
mod element {
    use super::Num;

    macro_rules! arithmetic {
        ($name:ident, $method:ident, $op:tt) => {
            pub(crate) fn $name<T: Num>(x: T, y: T) -> T {
                use std::any::Any;
                macro_rules! cast {
                    ($ty:ty) => {
                        if let (Some(a), Some(b)) = (
                            (&x as &dyn Any).downcast_ref::<$ty>(),
                            (&y as &dyn Any).downcast_ref::<$ty>(),
                        ) {
                            let value = a.$method(*b);
                            return *(&value as &dyn Any)
                                .downcast_ref::<T>()
                                .expect("same numeric type");
                        }
                    };
                }
                cast!(i32);
                cast!(i64);
                cast!(u8);
                cast!(usize);
                x $op y
            }
        };
    }

    arithmetic!(add, wrapping_add, +);
    arithmetic!(mul, wrapping_mul, *);
}

#[allow(dead_code)]
#[path = "../src/linalg/gemm/mod.rs"]
mod packed_gemm;

#[allow(dead_code)]
#[path = "../src/linalg/kernels.rs"]
mod kernels;

#[path = "../src/simd_math.rs"]
mod simd_math;

fn data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| ((i * 17 % 101) as f64 - 50.0) / 17.0)
        .collect()
}

fn gemm_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_gemm_f64");
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
    c.bench_function("kernel_inv_f64_200", |bench| {
        bench.iter(|| black_box(kernels::inv(black_box(&inverse_input), 200).unwrap()))
    });

    let svd_input = data(100 * 100);
    c.bench_function("kernel_svd_f64_100", |bench| {
        bench.iter(|| black_box(kernels::svd(black_box(&svd_input), 100, 100).unwrap()))
    });
}

fn array_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 7919 % 10007) as f64) / 5003.5 - 1.0)
        .collect()
}

fn array_benches(c: &mut Criterion) {
    let million = np::Array::from_vec(array_data(1_000_000), &[1_000_000]);
    let ten_million = np::Array::from_vec(array_data(10_000_000), &[10_000_000]);
    let matrix = np::Array::from_vec(array_data(1_000_000), &[1_000, 1_000]);
    let row = np::Array::from_vec(array_data(1_000), &[1_000]);
    c.bench_function("add (1e6 f64, contiguous)", |b| {
        b.iter(|| black_box(&million + &million))
    });
    c.bench_function("add broadcast ([1000,1000] + [1000])", |b| {
        b.iter(|| black_box(&matrix + &row))
    });
    c.bench_function("sum (1e7 f64)", |b| b.iter(|| black_box(ten_million.sum())));
    c.bench_function("mean_axis(0) [1000,1000]", |b| {
        b.iter(|| black_box(matrix.mean_axis(0)))
    });
    c.bench_function("transpose().to_contiguous() [1000,1000]", |b| {
        b.iter(|| black_box(matrix.t().to_contiguous()))
    });
    c.bench_function("exp (1e6 f64)", |b| b.iter(|| black_box(million.exp())));

    let raw_million = array_data(1_000_000);
    let mut raw_out = vec![0.0f64; 1_000_000];
    c.bench_function("exp (1e6 f64) simd kernel", |b| {
        b.iter(|| {
            simd_math::exp_f64(&raw_million, &mut raw_out);
            black_box(&raw_out);
        })
    });
    c.bench_function("tanh (1e6 f64) simd kernel", |b| {
        b.iter(|| {
            simd_math::tanh_f64(&raw_million, &mut raw_out);
            black_box(&raw_out);
        })
    });
    let mut rng = np::random::Rng::seed(42);
    c.bench_function("standard_normal (4096x4096)", |b| {
        b.iter(|| black_box(rng.standard_normal(&[4096, 4096])))
    });

    c.bench_function("sort (1e6 f64)", |b| b.iter(|| black_box(million.sort())));
    c.bench_function("sort (1e7 f64)", |b| {
        b.iter(|| black_box(ten_million.sort()))
    });

    let mut inv_values = array_data(200 * 200);
    for i in 0..200 {
        inv_values[i * 200 + i] += 250.0;
    }
    let inv_input = np::Array::from_vec(inv_values, &[200, 200]);
    c.bench_function("inv (200x200)", |b| {
        b.iter(|| black_box(np::linalg::inv(&inv_input).unwrap()))
    });
    let svd_input = np::Array::from_vec(array_data(100 * 100), &[100, 100]);
    c.bench_function("svd (100x100)", |b| {
        b.iter(|| black_box(np::linalg::svd(&svd_input)))
    });
    for n in [256, 512, 1024, 2048, 4096] {
        let left = np::Array::from_vec(array_data(n * n), &[n, n]);
        let right = left.t();
        c.bench_function(&format!("matmul ({n}x{n} f64)"), |b| {
            b.iter(|| black_box(left.matmul(&left)))
        });
        if n == 256 {
            c.bench_function("matmul transposed view (256x256 f64)", |b| {
                b.iter(|| black_box(left.matmul(&right)))
            });
        }
    }
    for n in [1024, 2048, 4096] {
        let left = np::Array::from_vec(array_data(n * n), &[n, n]).astype::<f32>();
        c.bench_function(&format!("matmul ({n}x{n} f32)"), |b| {
            b.iter(|| black_box(left.matmul(&left)))
        });
    }
    let left = np::Array::from_vec(array_data(2048 * 2048), &[2048, 2048]);
    c.bench_function("matmul_map tanh (2048x2048 f64)", |b| {
        b.iter(|| {
            black_box(np::linalg::matmul_map(
                &left,
                &left,
                1.0 / 2048.0,
                f64::tanh,
            ))
        })
    });
}

criterion_group!(core, gemm_benches, decomposition_benches, array_benches);
criterion_main!(core);
