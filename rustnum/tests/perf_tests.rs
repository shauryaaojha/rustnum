//! Performance and accuracy integration tests for SIMD math, fast random numbers, and fast sorting.

#[path = "../src/simd_math.rs"]
mod simd_math;

use np::random::Rng;
use rustnum as np;

fn f64_ulp_diff(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    if a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive() {
        return 0.0;
    }
    if a == b {
        return 0.0;
    }
    let diff = (a - b).abs();
    let ulp = if b.abs() > 0.0 {
        let exp = (b.to_bits() >> 52) & 0x7FF;
        if exp == 0 {
            f64::from_bits(1)
        } else {
            f64::from_bits(exp << 52) * f64::EPSILON
        }
    } else {
        f64::from_bits(1)
    };
    diff / ulp
}

fn f32_ulp_diff(a: f32, b: f32) -> f32 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    if a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive() {
        return 0.0;
    }
    if a == b {
        return 0.0;
    }
    let diff = (a - b).abs();
    let ulp = if b.abs() > 0.0 {
        let exp = (b.to_bits() >> 23) & 0xFF;
        if exp == 0 {
            f32::from_bits(1)
        } else {
            f32::from_bits(exp << 23) * f32::EPSILON
        }
    } else {
        f32::from_bits(1)
    };
    diff / ulp
}

#[test]
fn test_simd_math_accuracy_and_edge_cases() {
    for force_scalar in [false, true] {
        simd_math::set_force_scalar(force_scalar);

        // 1. exp
        {
            let mut inputs = vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
                1e-300,
                -1e-300,
                709.7,
                710.0,
                -708.0,
                -750.0,
                1.0,
                -1.0,
                0.5,
                -0.5,
            ];
            for i in 0..10_000 {
                inputs.push(((i as f64) / 1000.0) * 100.0 - 50.0);
            }
            let mut out = vec![0.0f64; inputs.len()];
            simd_math::exp_f64(&inputs, &mut out);

            let mut max_ulp = 0.0;
            for (&x, &y) in inputs.iter().zip(&out) {
                if x.is_nan() {
                    assert!(y.is_nan());
                } else if x == f64::INFINITY {
                    assert_eq!(y, f64::INFINITY);
                } else if x == f64::NEG_INFINITY {
                    assert_eq!(y, 0.0);
                } else if x == 0.0 {
                    assert_eq!(y, 1.0);
                } else if x > 709.8 {
                    assert!(y.is_infinite());
                } else if x < -745.0 {
                    assert_eq!(y, 0.0);
                } else {
                    let expected = x.exp();
                    let ulp = f64_ulp_diff(y, expected);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                    }
                }
            }
            assert!(max_ulp <= 2.0, "exp_f64 max ulp: {max_ulp}");
        }

        // 2. log
        {
            let mut inputs = vec![
                f64::NAN,
                f64::INFINITY,
                0.0,
                -0.0,
                -1.0,
                1e-300,
                1.0,
                2.0,
                10.0,
                1e10,
                1e100,
                1e-100,
            ];
            for i in 1..10_000 {
                inputs.push((i as f64) * 0.1);
            }
            let mut out = vec![0.0f64; inputs.len()];
            simd_math::log_f64(&inputs, &mut out);

            let mut max_ulp = 0.0;
            for (&x, &y) in inputs.iter().zip(&out) {
                if x.is_nan() || x < 0.0 {
                    assert!(y.is_nan());
                } else if x == 0.0 {
                    assert_eq!(y, f64::NEG_INFINITY);
                } else if x == f64::INFINITY {
                    assert_eq!(y, f64::INFINITY);
                } else if x == 1.0 {
                    assert_eq!(y, 0.0);
                } else {
                    let expected = x.ln();
                    let ulp = f64_ulp_diff(y, expected);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                    }
                }
            }
            assert!(max_ulp <= 2.0, "log_f64 max ulp: {max_ulp}");
        }

        // 3. sin and cos
        {
            let mut inputs = vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
                std::f64::consts::PI,
                std::f64::consts::FRAC_PI_2,
                -std::f64::consts::PI,
            ];
            for i in 0..10_000 {
                inputs.push(((i as f64) / 100.0) - 50.0);
            }
            let mut out_sin = vec![0.0f64; inputs.len()];
            let mut out_cos = vec![0.0f64; inputs.len()];
            simd_math::sin_f64(&inputs, &mut out_sin);
            simd_math::cos_f64(&inputs, &mut out_cos);

            let mut max_ulp_sin = 0.0;
            let mut max_ulp_cos = 0.0;
            for (&x, (&s, &c)) in inputs.iter().zip(out_sin.iter().zip(&out_cos)) {
                if x.is_nan() || x.is_infinite() {
                    assert!(s.is_nan());
                    assert!(c.is_nan());
                } else if x == 0.0 {
                    assert_eq!(s, 0.0);
                    assert_eq!(c, 1.0);
                } else {
                    let expected_sin = x.sin();
                    let expected_cos = x.cos();
                    let ulp_s = f64_ulp_diff(s, expected_sin);
                    let ulp_c = f64_ulp_diff(c, expected_cos);
                    if ulp_s > max_ulp_sin {
                        max_ulp_sin = ulp_s;
                    }
                    if ulp_c > max_ulp_cos {
                        max_ulp_cos = ulp_c;
                    }
                }
            }
            assert!(max_ulp_sin <= 2.0, "sin_f64 max ulp: {max_ulp_sin}");
            assert!(max_ulp_cos <= 2.0, "cos_f64 max ulp: {max_ulp_cos}");
        }

        // 4. tanh
        {
            let mut inputs = vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
                20.0,
                -20.0,
                1.0,
                -1.0,
                0.5,
                -0.5,
            ];
            for i in 0..10_000 {
                inputs.push(((i as f64) / 500.0) - 10.0);
            }
            let mut out = vec![0.0f64; inputs.len()];
            simd_math::tanh_f64(&inputs, &mut out);

            let mut max_ulp = 0.0;
            for (&x, &y) in inputs.iter().zip(&out) {
                if x.is_nan() {
                    assert!(y.is_nan());
                } else if x == f64::INFINITY || x > 20.0 {
                    assert_eq!(y, 1.0);
                } else if x == f64::NEG_INFINITY || x < -20.0 {
                    assert_eq!(y, -1.0);
                } else if x == 0.0 {
                    assert_eq!(y, 0.0);
                } else {
                    let expected = x.tanh();
                    let ulp = f64_ulp_diff(y, expected);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                    }
                }
            }
            assert!(max_ulp <= 3.0, "tanh_f64 max ulp: {max_ulp}");
        }

        // 5. sigmoid
        {
            let mut inputs = vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
                40.0,
                -40.0,
            ];
            for i in 0..10_000 {
                inputs.push(((i as f64) / 500.0) - 10.0);
            }
            let mut out = vec![0.0f64; inputs.len()];
            simd_math::sigmoid_f64(&inputs, &mut out);

            let mut max_ulp = 0.0;
            for (&x, &y) in inputs.iter().zip(&out) {
                if x.is_nan() {
                    assert!(y.is_nan());
                } else if x == f64::INFINITY || x > 37.0 {
                    assert_eq!(y, 1.0);
                } else if x == f64::NEG_INFINITY || x < -745.0 {
                    assert_eq!(y, 0.0);
                } else if x == 0.0 {
                    assert_eq!(y, 0.5);
                } else {
                    let expected = 1.0 / (1.0 + (-x).exp());
                    let ulp = f64_ulp_diff(y, expected);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                    }
                }
            }
            assert!(max_ulp <= 3.0, "sigmoid_f64 max ulp: {max_ulp}");
        }

        // 6. f32 exp & log
        {
            let mut in32 = vec![
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0,
                -0.0,
                1.0,
                -1.0,
                88.0,
                -88.0,
            ];
            for i in 0..5_000 {
                in32.push(((i as f32) / 100.0) - 25.0);
            }
            let mut out32 = vec![0.0f32; in32.len()];
            simd_math::exp_f32(&in32, &mut out32);
            let mut max_ulp = 0.0;
            for (&x, &y) in in32.iter().zip(&out32) {
                if !x.is_nan() && x.is_finite() && x < 88.0 && x > -87.0 {
                    let exp = (x as f64).exp() as f32;
                    let ulp = f32_ulp_diff(y, exp);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                    }
                }
            }
            assert!(max_ulp <= 3.0, "exp_f32 max ulp: {max_ulp}");
        }
    }
    // Restore default dispatch
    simd_math::set_force_scalar(false);
}

#[test]
fn test_sort_and_argsort_nan_inf_stability() {
    let raw: Vec<f64> = vec![
        f64::NAN,
        3.0,
        -0.0,
        1.0,
        f64::INFINITY,
        -5.0,
        0.0,
        f64::NEG_INFINITY,
        1.0,
        f64::NAN,
        2.0,
        -0.0,
    ];
    let arr = np::Array::from_vec(raw.clone(), &[raw.len()]);

    let sorted = arr.sort().to_vec();
    // Verify all NaNs are at the end
    let nan_start = sorted.iter().position(|x| x.is_nan()).unwrap();
    assert!(sorted[nan_start..].iter().all(|x| x.is_nan()));
    // Verify non-NaNs are in ascending order
    for i in 1..nan_start {
        assert!(sorted[i - 1] <= sorted[i]);
    }

    let argsorted = arr.argsort().to_vec();
    assert_eq!(argsorted.len(), raw.len());
    // Verify argsort places NaNs at the end and preserves relative order of NaNs
    let nan_indices: Vec<usize> = raw
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_nan())
        .map(|(i, _)| i)
        .collect();
    let end_indices = &argsorted[argsorted.len() - nan_indices.len()..];
    assert_eq!(end_indices, &nan_indices[..]);

    // Verify argsort sorts non-NaNs in ascending order
    for i in 1..nan_start {
        let v_prev = raw[argsorted[i - 1]];
        let v_curr = raw[argsorted[i]];
        assert!(v_prev <= v_curr);
    }

    // Verify stability on equal elements (1.0 at index 3 and 8)
    let idx_1_first = argsorted.iter().position(|&idx| idx == 3).unwrap();
    let idx_1_second = argsorted.iter().position(|&idx| idx == 8).unwrap();
    assert!(idx_1_first < idx_1_second);
}

#[test]
fn test_large_sort_radix_parity() {
    let n = 100_000;
    let data: Vec<f64> = (0..n)
        .map(|i| {
            if i % 1000 == 0 {
                f64::NAN
            } else {
                ((i * 7919 % 10007) as f64) / 5003.5 - 1.0
            }
        })
        .collect();
    let arr = np::Array::from_vec(data.clone(), &[n]);
    let sorted = arr.sort().to_vec();

    let nan_start = sorted.iter().position(|x| x.is_nan()).unwrap();
    assert_eq!(sorted.len() - nan_start, 100);
    for i in 1..nan_start {
        assert!(sorted[i - 1] <= sorted[i]);
    }

    let argsorted = arr.argsort().to_vec();
    assert_eq!(argsorted.len(), n);
    for i in 1..nan_start {
        assert!(data[argsorted[i - 1]] <= data[argsorted[i]]);
    }
}

/// Large fills use one jumped xoshiro stream per 65536-element chunk, so the output depends only
/// on the seed: not on the thread count and not on the `rayon` feature.
const CHUNK: usize = 1 << 16;

fn draws(n: usize) -> [Vec<f64>; 3] {
    let mut rng = Rng::seed(42);
    let normal = rng.standard_normal(&[n]).to_vec();
    let exponential = rng.exponential(1.5, &[n]).to_vec();
    let uniform = rng.uniform(-2.0, 5.0, &[n]).to_vec();
    [normal, exponential, uniform]
}

#[test]
fn chunked_rng_is_independent_of_thread_count() {
    let n = 3 * CHUNK + 1234;
    let reference = draws(n);
    #[cfg(feature = "rayon")]
    for threads in [1, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(|| draws(n)), reference, "{threads} threads");
    }
    assert_eq!(draws(n), reference);
}

#[test]
fn first_chunk_continues_the_scalar_stream() {
    // Chunk 0 uses the generator's own stream, so it equals one-at-a-time draws.
    let mut chunked = Rng::seed(42);
    let mut scalar = Rng::seed(42);
    let big = chunked.standard_normal(&[2 * CHUNK]).to_vec();
    let one_by_one: Vec<f64> = (0..CHUNK)
        .map(|_| scalar.standard_normal(&[]).get(&[]))
        .collect();
    assert_eq!(&big[..CHUNK], &one_by_one[..]);
}
