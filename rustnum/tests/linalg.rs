//! Integration tests of the public linear-algebra API.
use rustnum as np;
use rustnum::linalg::{self, NormOrd};

fn sample(rows: usize, cols: usize, seed: usize) -> np::Array<f64> {
    np::Array::from_shape_fn([rows, cols], |c| {
        (((c[0] * 31 + c[1] * 17 + seed * 7) % 23) as f64 - 11.0) / 4.0
    })
}
fn assert_close(a: &np::Array<f64>, b: &np::Array<f64>, tol: f64) {
    assert_eq!(a.shape(), b.shape());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() <= tol * y.abs().max(1.0), "{x} != {y}");
    }
}
/// Textbook triple loop on logical indexes, independent of strides.
fn naive(a: &np::Array<f64>, b: &np::Array<f64>) -> np::Array<f64> {
    let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    np::Array::from_shape_fn([m, n], |c| {
        (0..k).map(|p| a[[c[0], p]] * b[[p, c[1]]]).sum::<f64>()
    })
}

#[test]
fn matmul_literal() {
    let a = np::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = np::array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    assert_eq!(a.matmul(&b).to_vec(), [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_transposed_views_match_contiguous() {
    let a = sample(5, 7, 1);
    let b = sample(5, 6, 2);
    assert!(!a.t().is_contiguous());
    assert_close(&a.t().matmul(&b), &naive(&a.t().to_contiguous(), &b), 1e-12);
    // both operands transposed
    let c = sample(6, 5, 3);
    assert_close(
        &a.t().matmul(&c.t()),
        &naive(&a.t().copy(), &c.t().copy()),
        1e-12,
    );
    // f32 views give the same numbers as their contiguous copies
    let a32 = a.astype::<f32>();
    let b32 = b.astype::<f32>();
    let got = a32.t().matmul(&b32);
    let want = a32.t().to_contiguous().matmul(&b32);
    assert_eq!(got.to_vec(), want.to_vec());
}

#[test]
fn matmul_sliced_views_match_contiguous() {
    let big = sample(10, 12, 4);
    let other = sample(12, 9, 5);
    let a = big.slice(&np::s![1..8;2, 2..11;3]);
    let b = other.slice(&np::s![2..5, ..;2]);
    assert!(!a.is_contiguous() && !b.is_contiguous());
    assert_close(&a.matmul(&b), &naive(&a.copy(), &b.copy()), 1e-12);
    // reversed rows (negative strides) are packed directly
    let rev = big.slice(&np::s![..;-1, ..]);
    let sq = sample(12, 3, 6);
    assert_close(&rev.matmul(&sq), &naive(&rev.copy(), &sq), 1e-12);
}

#[test]
fn matmul_batched_shapes() {
    let a = np::arange(0.0, 24.0, 1.0).reshape(&[2, 3, 4]);
    let b = sample(4, 5, 7);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), [2, 3, 5]);
    for i in 0..2 {
        let block = a.index_axis(0, i);
        assert_close(&c.index_axis(0, i), &naive(&block, &b), 1e-12);
    }

    let a = np::arange(0.0, 24.0, 1.0).reshape(&[2, 1, 3, 4]);
    let b = (np::arange(0.0, 40.0, 1.0) / 7.0).reshape(&[5, 4, 2]);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), [2, 5, 3, 2]);
    for i in 0..2 {
        for j in 0..5 {
            let x = a.index_axis(0, i).index_axis(0, 0);
            let y = b.index_axis(0, j);
            let got = c.index_axis(0, i).index_axis(0, j);
            assert_close(&got, &naive(&x, &y), 1e-12);
        }
    }
}

#[test]
fn matmul_batched_strided_batch() {
    let a = np::arange(0.0, 24.0, 1.0).reshape(&[2, 3, 4]);
    let swapped = a.swapaxes(1, 2); // [2, 4, 3], strided
    let c = swapped.matmul(&a);
    assert_eq!(c.shape(), [2, 4, 4]);
    assert_eq!(c.to_vec(), swapped.copy().matmul(&a).to_vec());
    // batch axis taken from a sliced view
    let every_other = np::ones::<f64>([4, 3, 4]).slice(&np::s![..;2]);
    assert_eq!(
        every_other.matmul(&np::ones::<f64>([4, 2])).sum(),
        2.0 * 3.0 * 2.0 * 4.0
    );
}

#[test]
fn matmul_one_dimensional_cases() {
    let m = np::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let v = np::array![1.0, 0.0, -1.0];
    let w = np::array![2.0, 1.0];
    let dot = v.matmul(&v);
    assert_eq!(dot.shape(), []);
    assert_eq!(dot.item(), 2.0);
    assert_eq!(m.matmul(&v).to_vec(), [-2.0, -2.0]);
    assert_eq!(w.matmul(&m).to_vec(), [6.0, 9.0, 12.0]);
    // 1-d with a batch
    let batch = np::ones::<f64>([4, 2, 3]);
    assert_eq!(batch.matmul(&v).shape(), [4, 2]);
    assert_eq!(w.matmul(&batch).shape(), [4, 3]);
    // strided vector
    let strided = np::array![1.0, 9.0, 0.0, 9.0, -1.0, 9.0].slice(&np::s![..;2]);
    assert_eq!(m.matmul(&strided).to_vec(), [-2.0, -2.0]);
}

#[test]
fn matmul_integers() {
    let a = np::array![[1, 2], [3, 4]];
    let b = np::array![[5, 6], [7, 8]];
    assert_eq!(a.matmul(&b).to_vec(), [19, 22, 43, 50]);
    assert_eq!(a.t().matmul(&b).to_vec(), [26, 30, 38, 44]);
    assert_eq!(
        np::array![1i64, 2, 3]
            .matmul(&np::array![4i64, 5, 6])
            .item(),
        32
    );
    // wraps like NumPy
    let big = np::array![[i32::MAX, 1]];
    assert_eq!(big.matmul(&np::array![[2], [0]]).item(), -2);
    assert_eq!(a.dot(&b).to_vec(), a.matmul(&b).to_vec());
    assert_eq!(a.trace(), 5);
}

#[test]
fn integer_linalg_products_wrap_on_overflow() {
    let a = np::array![[i32::MAX]];
    let two = np::array![[2_i32]];
    let one = np::array![[1_i32]];
    assert_eq!(linalg::multi_dot(&[&a, &two, &one]).item(), -2);
    assert_eq!(linalg::vdot(&a, &two), -2);

    let x = np::array![i64::MAX];
    let y = np::array![2_i64];
    assert_eq!(linalg::tensordot(&x, &y, &[0], &[0]).item(), -2);
}

#[test]
fn matmul_empty_dimensions() {
    let a = np::zeros::<f64>([3, 0]);
    let b = np::zeros::<f64>([0, 2]);
    assert_eq!(a.matmul(&b).to_vec(), [0.0; 6]);
    let empty = np::zeros::<f64>([0, 4]).matmul(&np::zeros::<f64>([4, 2]));
    assert_eq!(empty.shape(), [0, 2]);
}

#[test]
#[should_panic(expected = "mismatch in its core dimension")]
fn matmul_shape_mismatch_panics() {
    np::zeros::<f64>([2, 3]).matmul(&np::zeros::<f64>([2, 3]));
}

#[test]
fn dot_and_trace() {
    let a = np::array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(
        a.dot(&np::Array::scalar(2.0)).to_vec(),
        [2.0, 4.0, 6.0, 8.0]
    );
    assert_eq!(a.t().trace(), 5.0);
    assert_eq!(np::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].trace(), 6.0);
}

#[test]
fn inverse_times_matrix_is_identity() {
    let a = sample(6, 6, 9) + np::eye::<f64>(6) * 10.0;
    let inv = linalg::inv(&a).unwrap();
    assert_close(&a.matmul(&inv), &np::eye::<f64>(6), 1e-12);
    assert_close(&inv.matmul(&a), &np::eye::<f64>(6), 1e-12);
    let a32 = a.astype::<f32>();
    let p = a32.matmul(&linalg::inv(&a32).unwrap());
    assert!(p.allclose(np::eye::<f32>(6), 1e-4, 1e-4));
}

#[test]
fn solve_recovers_solution() {
    let a = sample(5, 5, 11) + np::eye::<f64>(5) * 9.0;
    let x = sample(5, 1, 12);
    let rhs = a.matmul(&x);
    assert_close(&linalg::solve(&a, &rhs).unwrap(), &x, 1e-12);
    let x1 = x.reshape(&[5]);
    let solved = linalg::solve(&a, &a.matmul(&x1)).unwrap();
    assert_close(&solved.reshape(&[5, 1]), &x, 1e-12);
    assert!(matches!(
        linalg::solve(&a, &np::zeros::<f64>([4])),
        Err(np::Error::Shape(_))
    ));
}

#[test]
fn singular_matrices_return_error() {
    let s = np::array![[1.0, 2.0], [2.0, 4.0]];
    assert!(matches!(linalg::inv(&s), Err(np::Error::Singular)));
    assert!(matches!(
        linalg::solve(&s, &np::array![1.0, 2.0]),
        Err(np::Error::Singular)
    ));
    assert!(matches!(
        linalg::matrix_power(&s, -1),
        Err(np::Error::Singular)
    ));
    assert_eq!(linalg::det(&s), 0.0);
}

#[test]
fn cholesky_and_not_positive_definite() {
    let b = sample(4, 4, 13);
    let spd = b.t().matmul(&b) + np::eye::<f64>(4);
    let l = linalg::cholesky(&spd).unwrap();
    assert_close(&l.matmul(&l.t()), &spd, 1e-12);
    assert!(matches!(
        linalg::cholesky(&np::array![[1.0, 2.0], [2.0, 1.0]]),
        Err(np::Error::NotPositiveDefinite)
    ));
}

#[test]
fn qr_reconstruction() {
    for (m, n) in [(6, 4), (4, 6), (5, 5)] {
        let a = sample(m, n, 14);
        let (q, r) = linalg::qr(&a);
        let k = m.min(n);
        assert_eq!(q.shape(), [m, k]);
        assert_eq!(r.shape(), [k, n]);
        assert_close(&q.matmul(&r), &a, 1e-12);
        assert_close(&q.t().matmul(&q), &np::eye::<f64>(k), 1e-12);
    }
}

#[test]
fn eigh_reconstruction() {
    let b = sample(5, 5, 15);
    let a = &b + &b.t();
    let (w, v) = linalg::eigh(&a);
    for i in 1..5 {
        assert!(w[[i - 1]] <= w[[i]]);
    }
    assert_close(&(&v * &w).matmul(&v.t()), &a, 1e-12);
    assert_close(&linalg::eigvalsh(&a), &w, 1e-12);
}

#[test]
fn svd_reconstruction() {
    for (m, n) in [(6, 4), (4, 6), (5, 5)] {
        let a = sample(m, n, 16);
        let (u, s, vt) = linalg::svd(&a);
        let k = m.min(n);
        assert_eq!(u.shape(), [m, k]);
        assert_eq!(s.shape(), [k]);
        assert_eq!(vt.shape(), [k, n]);
        assert_close(&(&u * &s).matmul(&vt), &a, 1e-12);
        for i in 1..k {
            assert!(s[[i - 1]] >= s[[i]]);
        }
    }
    assert_eq!(linalg::matrix_rank(&np::array![[1.0, 2.0], [2.0, 4.0]]), 1);
}

#[test]
fn eig_general() {
    let a = np::array![[0.0, -1.0], [1.0, 0.0]];
    let (re, im, _, _) = linalg::eig(&a).unwrap();
    assert_eq!(re.to_vec(), [0.0, 0.0]);
    assert_eq!(im.to_vec(), [1.0, -1.0]);
    // a v = w v, split into real and imaginary parts
    let b = sample(5, 5, 17);
    let (wr, wi, vr, vi) = linalg::eig(&b).unwrap();
    assert_close(&b.matmul(&vr), &(&vr * &wr - &vi * &wi), 1e-9);
    assert_close(&b.matmul(&vi), &(&vr * &wi + &vi * &wr), 1e-9);
}

fn panic_message<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> String {
    let payload = std::panic::catch_unwind(f).err().expect("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap()
}

#[test]
fn non_finite_input_does_not_converge() {
    let bad = np::array![[1.0, f64::NAN], [f64::NAN, 4.0]];
    match linalg::eig(&bad) {
        Err(np::Error::NoConvergence(m)) => assert_eq!(m, "Eigenvalues did not converge"),
        other => panic!("unexpected {other:?}"),
    }
    assert!(matches!(
        linalg::eigvals(&bad),
        Err(np::Error::NoConvergence(_))
    ));
    assert_eq!(panic_message(|| linalg::svd(&bad)), "SVD did not converge");
    assert_eq!(
        panic_message(|| linalg::eigh(&bad)),
        "Eigenvalues did not converge"
    );
}

#[test]
fn error_display_wording() {
    assert_eq!(
        np::Error::NoConvergence("SVD did not converge".into()).to_string(),
        "SVD did not converge"
    );
}

#[test]
fn norms_and_products() {
    let a = np::array![[1.0, -2.0], [3.0, -4.0]];
    assert_eq!(linalg::matrix_norm(&a, NormOrd::One), 6.0);
    assert_eq!(
        linalg::vector_norm(&np::array![3.0, 4.0], NormOrd::Fro),
        5.0
    );
    assert!((linalg::norm(&a) - 30.0f64.sqrt()).abs() < 1e-12);
    let stack = np::arange(0.0, 24.0, 1.0).reshape(&[2, 3, 4]);
    let n = linalg::norm_axis(&stack, NormOrd::Two, -1);
    assert_eq!(n.shape(), [2, 3]);
    assert!((n[[0, 0]] - 14.0f64.sqrt()).abs() < 1e-12);
    let o = linalg::outer(&np::array![1.0, 2.0], &np::array![3.0, 4.0]);
    assert_eq!(o.to_vec(), [3.0, 4.0, 6.0, 8.0]);
    assert_eq!(linalg::inner(&np::array![1, 2], &np::array![3, 4]), 11);
    let x = sample(3, 4, 18);
    let y = sample(4, 2, 19);
    let z = sample(2, 5, 20);
    assert_close(
        &linalg::multi_dot(&[&x, &y, &z]),
        &x.matmul(&y).matmul(&z),
        1e-12,
    );
    assert_close(&linalg::tensordot_n(&x, &y, 1), &x.matmul(&y), 1e-12);
    assert_eq!(linalg::vdot(&x, &x), x.iter().map(|v| v * v).sum::<f64>());
}

/// Naive triple loop used as an independent oracle for the tiled parallel GEMM.
fn naive_matmul(a: &np::Array<f64>, b: &np::Array<f64>) -> Vec<f64> {
    let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let x = a.get(&[i, p]);
            for j in 0..n {
                out[i * n + j] += x * b.get(&[p, j]);
            }
        }
    }
    out
}

#[test]
fn tiled_parallel_matmul_matches_naive_on_views_and_edge_tiles() {
    // 600 x 300 @ 300 x 530 spans several 256-wide tiles with ragged edges, so the
    // parallel path runs; A is a transposed view and B a sliced view.
    let a_base = np::Array::from_shape_fn(&[300, 600], |ix| {
        ((ix[0] * 7 + ix[1] * 3) % 17) as f64 - 8.0
    });
    let b_base = np::Array::from_shape_fn(&[300, 540], |ix| {
        ((ix[0] * 5 + ix[1] * 11) % 13) as f64 - 6.0
    });
    let a = a_base.t();
    let b = b_base.slice(&np::s![.., 5..535]);
    let got = a.matmul(&b);
    assert_eq!(got.shape(), &[600, 530]);
    assert_eq!(got.to_vec(), naive_matmul(&a, &b));
}

#[test]
fn tiled_parallel_matmul_f32_matches_f64() {
    let a = np::Array::from_shape_fn(&[520, 260], |ix| ((ix[0] + 2 * ix[1]) % 9) as f64 - 4.0);
    let b = np::Array::from_shape_fn(&[260, 700], |ix| ((3 * ix[0] + ix[1]) % 7) as f64 - 3.0);
    let exact = a.matmul(&b);
    let single = a.astype::<f32>().matmul(&b.astype::<f32>());
    // Small integers: every product and partial sum is exact in f32 too.
    assert_eq!(single.astype::<f64>().to_vec(), exact.to_vec());
}

#[test]
fn packed_gemm_prime_and_degenerate_shapes() {
    for (m, k, n) in [
        (1, 1031, 257),
        (257, 509, 1),
        (7, 1031, 19),
        (257, 17, 509),
        (0, 7, 19),
        (13, 0, 17),
        (13, 7, 0),
    ] {
        let a = sample(m, k, 21);
        let b = sample(k, n, 22);
        let want = naive(&a, &b);
        assert_close(
            &linalg::gemm(1.0, &a, &b, 0.0, &np::zeros::<f64>([m, n])),
            &want,
            1e-12,
        );
        assert_close(&a.matmul(&b), &want, 1e-12);
        assert_close(
            &a.astype::<f32>().matmul(&b.astype::<f32>()).astype::<f64>(),
            &want,
            1e-5,
        );
    }
}

#[test]
fn gemm_scales_signed_views_and_strided_addend() {
    let a = sample(19, 1034, 23).slice(&np::s![..;-2, 1..1033;2]);
    let b = sample(33, 516, 24).t().slice(&np::s![..;-1, ..;-2]);
    let c = sample(34, 20, 25).t().slice(&np::s![..;-2, ..;-2]);
    assert_eq!(c.shape(), [10, 17]);
    let product = naive(&a, &b);
    for (alpha, beta) in [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, -2.0),
        (1.5, -0.75),
        (-1.0, 1.0),
    ] {
        let expected = &product * alpha + &c * beta;
        assert_close(&linalg::gemm(alpha, &a, &b, beta, &c), &expected, 1e-12);
        let got32 = linalg::gemm(
            alpha as f32,
            &a.astype::<f32>(),
            &b.astype::<f32>(),
            beta as f32,
            &c.astype::<f32>(),
        );
        assert_close(&got32.astype::<f64>(), &expected, 1e-5);
    }
}

#[test]
fn gemm_zero_coefficients_ignore_nan_and_zero_depth_maps() {
    let nan = np::full([7, 9], f64::NAN);
    let b = np::ones::<f64>([9, 11]);
    let c = np::full([7, 11], f64::NAN);
    assert_eq!(linalg::gemm(0.0, &nan, &b, 0.0, &c).to_vec(), vec![0.0; 77]);
    let a = np::ones::<f64>([7, 9]);
    assert_eq!(linalg::gemm(1.0, &a, &b, 0.0, &c).to_vec(), vec![9.0; 77]);
    let a = np::zeros::<f64>([7, 0]);
    let b = np::zeros::<f64>([0, 11]);
    let c = np::ones::<f64>([7, 11]);
    assert_eq!(linalg::gemm(2.0, &a, &b, -3.0, &c).to_vec(), vec![-3.0; 77]);
    assert_eq!(
        linalg::matmul_map(&a, &b, 2.0, |x| x + 7.0).to_vec(),
        vec![7.0; 77]
    );
}

#[test]
fn matmul_map_applies_once_after_all_kc_blocks() {
    let a = sample(127, 1031, 26);
    let b = sample(1031, 269, 27);
    let expected = (naive(&a, &b) * 0.003).tanh();
    assert_close(
        &linalg::matmul_map(&a, &b, 0.003, f64::tanh),
        &expected,
        1e-12,
    );
    assert_close(
        &linalg::matmul_map(&a.astype::<f32>(), &b.astype::<f32>(), 0.003, f32::tanh)
            .astype::<f64>(),
        &expected,
        1e-5,
    );
    // Empty core reductions must still execute the nonlinear epilogue in every batch.
    let batch = np::zeros::<f32>([3, 2, 0]);
    let right = np::zeros::<f32>([0, 4]);
    assert_eq!(
        linalg::matmul_map(&batch, &right, 1.0, |x| x + 1.0).to_vec(),
        vec![1.0; 24]
    );
    let v = np::array![1.0f64, 2.0];
    assert_eq!(linalg::matmul_map(&v, &v, 2.0, |x| x + 1.0).item(), 11.0);
}

#[test]
fn matmul_into_reuses_output_and_preserves_cow() {
    let a = sample(13, 17, 28);
    let b = sample(17, 19, 29);
    let mut out = np::full([13, 19], f64::NAN);
    let ptr = out.as_slice().unwrap().as_ptr();
    a.matmul_into(&b, &mut out);
    assert_eq!(ptr, out.as_slice().unwrap().as_ptr());
    assert_close(&out, &naive(&a, &b), 1e-12);
    let saved = out.clone();
    np::zeros::<f64>([13, 17]).matmul_into(&b, &mut out);
    assert_eq!(out.to_vec(), vec![0.0; 13 * 19]);
    assert_close(&saved, &naive(&a, &b), 1e-12);
    let a = np::ones::<i32>([2, 3, 4]);
    let b = np::ones::<i32>([4]);
    let mut out = np::zeros::<i32>([2, 3]);
    a.matmul_into(&b, &mut out);
    assert_eq!(out.to_vec(), vec![4; 6]);
    // A contiguous view with an offset also reuses its unique backing allocation.
    let mut out = np::zeros::<f64>([15, 19]).slice(&np::s![1..14, ..]);
    let ptr = out.as_slice().unwrap().as_ptr();
    sample(13, 17, 28).matmul_into(&sample(17, 19, 29), &mut out);
    assert_eq!(ptr, out.as_slice().unwrap().as_ptr());
    assert_close(&out, &saved, 1e-12);
}

#[test]
#[should_panic(expected = "output operand has a mismatch")]
fn matmul_into_rejects_wrong_shape() {
    np::eye::<f32>(3).matmul_into(&np::eye::<f32>(3), &mut np::zeros::<f32>([9]));
}

#[test]
#[should_panic(expected = "not C-contiguous")]
fn matmul_into_rejects_strided_output() {
    np::eye::<f64>(3).matmul_into(&np::eye::<f64>(3), &mut np::eye::<f64>(3).t());
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(64))]
    #[test]
    fn gemm_shapes_and_strides(m in 1usize..25, k in 1usize..40, n in 1usize..25,
        sa in 1usize..4, sb in 1usize..4, reverse_a in proptest::bool::ANY,
        reverse_b in proptest::bool::ANY, transpose in proptest::bool::ANY) {
        let step_a = sa as isize * if reverse_a { -1 } else { 1 };
        let step_b = sb as isize * if reverse_b { -1 } else { 1 };
        let a = if transpose {
            sample(k * sa, m, 30).t().slice(&np::s![.., ..;step_a])
        } else {
            sample(m, k * sa, 30).slice(&np::s![.., ..;step_a])
        };
        let b = sample(k * sb, n, 31).slice(&np::s![..;step_b, ..]);
        let c = sample(n * 2, m, 32).t().slice(&np::s![.., ..;-2]);
        let want = naive(&a, &b) * -0.75 + &c * 0.5;
        assert_close(&linalg::gemm(-0.75, &a, &b, 0.5, &c), &want, 1e-12);
        assert_close(&a.astype::<f32>().matmul(&b.astype::<f32>()).astype::<f64>(), &naive(&a, &b), 1e-5);
    }
}

#[test]
fn f32_signed_and_transposed_views_and_reversed_batches() {
    let a = sample(19, 522, 33)
        .astype::<f32>()
        .slice(&np::s![..;-2, ..;-2]);
    let b = sample(35, 261, 34)
        .astype::<f32>()
        .t()
        .slice(&np::s![..;-1, ..;2]);
    let c = sample(36, 20, 35)
        .astype::<f32>()
        .t()
        .slice(&np::s![..;2, ..;-2]);
    let expected = naive(&a.astype::<f64>(), &b.astype::<f64>()) * 0.5 + c.astype::<f64>() * -2.0;
    assert_close(
        &linalg::gemm(0.5, &a, &b, -2.0, &c).astype::<f64>(),
        &expected,
        1e-5,
    );
    let batch = np::arange(0.0, 120.0, 1.0)
        .reshape(&[4, 5, 6])
        .slice(&np::s![..;-1, ..;-1, ..;-1]);
    let right = sample(6, 7, 36).slice(&np::s![..;-1, ..]);
    let got = batch.matmul(&right);
    for i in 0..4 {
        assert_close(
            &got.index_axis(0, i),
            &naive(&batch.index_axis(0, i), &right),
            1e-12,
        );
    }
}
