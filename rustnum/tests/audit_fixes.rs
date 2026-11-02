//! Regression tests for the 2026-11 code audit. Expected values come from NumPy 2.3.
use rustnum as np;

np::record! { pub struct P { x: f64, y: f64 } }

#[test]
fn integer_division_by_zero_and_min_over_minus_one_do_not_panic() {
    // NumPy: np.int64([5, 0, MIN]) // np.int64([0, 0, -1]) == [0, 0, MIN]; % gives [0, 0, 0].
    let a = np::array![5i64, 0, i64::MIN];
    let b = np::array![0i64, 0, -1];
    assert_eq!((&a / &b).to_vec(), [0, 0, i64::MIN]);
    assert_eq!((&a % &b).to_vec(), [0, 0, 0]);
    assert_eq!((&a / 0).to_vec(), [0, 0, 0]);
    assert_eq!((7i32 / &np::array![0i32, 2]).to_vec(), [0, 3]);
    let mut c = np::array![9u8, 4];
    c /= 0u8;
    assert_eq!(c.to_vec(), [0, 0]);
    let mut d = np::array![i32::MIN, 5];
    d %= -1;
    assert_eq!(d.to_vec(), [0, 0]);
}

#[test]
fn float_division_by_zero_is_ieee() {
    let r = &np::array![1.0f64, -1.0, 0.0] / 0.0;
    assert_eq!(r.get(&[0]), f64::INFINITY);
    assert_eq!(r.get(&[1]), f64::NEG_INFINITY);
    assert!(r.get(&[2]).is_nan());
}

#[test]
fn masked_integer_division_masks_zero_and_wraps_min() {
    // np.ma.array(int32 [MIN, 7, 3]) // [-1, 7, 0] -> [MIN, 1, --]
    let a = np::ma::masked_array(np::array![i32::MIN, 7, 3], None);
    let b = np::ma::masked_array(np::array![-1i32, 7, 0], None);
    let r = &a / &b;
    assert_eq!(r.filled(-99).to_vec(), [i32::MIN, 1, -99]);
}

#[test]
fn record_sort_puts_nan_fields_last_field_by_field() {
    // np.sort of [(1,nan),(5,0),(nan,2),(nan,1),(1,-3)] with fields ('x','f8'),('y','f8')
    // == [(1,-3),(1,nan),(5,0),(nan,1),(nan,2)], argsort == [4,0,1,3,2].
    let nan = f64::NAN;
    let a = np::Array::from(vec![
        P { x: 1.0, y: nan },
        P { x: 5.0, y: 0.0 },
        P { x: nan, y: 2.0 },
        P { x: nan, y: 1.0 },
        P { x: 1.0, y: -3.0 },
    ]);
    assert_eq!(a.argsort().to_vec(), [4, 0, 1, 3, 2]);
    let s = a.sort();
    let xs = s.field::<f64>("x").to_vec();
    let ys = s.field::<f64>("y").to_vec();
    assert_eq!(&xs[..3], [1.0, 1.0, 5.0]);
    assert!(xs[3].is_nan() && xs[4].is_nan());
    assert_eq!(ys[0], -3.0);
    assert!(ys[1].is_nan());
    assert_eq!(&ys[2..], [0.0, 1.0, 2.0]);
    assert!(
        P { x: nan, y: 1.0 }
            .partial_cmp(&P { x: 0.0, y: 0.0 })
            .is_some()
    );
}

#[test]
fn masked_unary_keeps_masked_data_and_masks_domain() {
    // np.ma.sqrt(np.ma.array([4, -1, 9], mask=[0, 0, 1])): data [2, -1, 9], mask [F, T, T]
    let m = np::ma::masked_array(
        np::array![4.0, -1.0, 9.0],
        Some(np::array![false, false, true]),
    );
    let r = m.sqrt();
    assert_eq!(r.data().to_vec(), [2.0, -1.0, 9.0]);
    assert_eq!(np::ma::getmask(&r).to_vec(), [false, true, true]);
}

#[test]
fn nan_statistics_on_strided_views_match_contiguous() {
    // np.nanmedian / nanvar / nanpercentile of a[:, 1] for a = [[1, nan], [2, 4], [3, 8], [4, nan]]
    let a = np::array![[1.0, f64::NAN], [2.0, 4.0], [3.0, 8.0], [4.0, f64::NAN]];
    let col = a.slice(&np::s![.., 1]);
    assert_eq!(col.nanmedian(), 6.0);
    assert_eq!(col.nanvar(), 4.0);
    assert_eq!(col.nanpercentile(25.0), 5.0);
    let row = a.slice(&np::s![.., 0]);
    assert_eq!(row.nanmedian(), 2.5);
}

/// Runs `f` and reports a panic as `Err(message)` so one probe failing does not hide the rest.
fn probe<R: std::fmt::Debug>(name: &str, f: impl FnOnce() -> R + std::panic::UnwindSafe) -> String {
    match std::panic::catch_unwind(f) {
        Ok(v) => format!("{name}: {v:?}"),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            format!("{name}: PANIC {msg}")
        }
    }
}

#[test]
fn integer_overflow_wraps_like_numpy_everywhere() {
    // Expected values: NumPy 2.3 int64 results (wrapping, no exception).
    let max = i64::MAX;
    let min = i64::MIN;
    let got = [
        probe("sum", || np::array![max, 1].sum()),
        probe("prod", || np::array![max, 2].prod()),
        probe("cumsum", || np::array![max, 1].cumsum().to_vec()),
        probe("cumprod", || np::array![max, 2].cumprod().to_vec()),
        probe("diff", || np::diff(&np::array![min, 1], 1, -1).to_vec()),
        probe("ptp", || np::array![min, max].ptp()),
        probe("dot", || {
            np::array![max, 1].dot(&np::array![max, 1]).to_vec()
        }),
        probe("trace", || np::array![[max, 0], [0, 1]].trace()),
        probe("square", || np::array![max].square().to_vec()),
        probe("abs", || np::array![min].abs().to_vec()),
        probe("power", || {
            np::power(np::array![3i64], np::array![50i64]).to_vec()
        }),
        probe("u8 sum", || np::array![200u8, 100].sum()),
        probe("u8 diff", || np::diff(&np::array![1u8, 0], 1, -1).to_vec()),
        probe("matmul", || {
            np::array![[max, 1]]
                .matmul(&np::array![[2i64], [2]])
                .to_vec()
        }),
        probe("outer", || {
            np::linalg::outer(&np::array![max], &np::array![2i64]).to_vec()
        }),
        probe("convolve", || {
            np::convolve(&np::array![max], &np::array![2i64], np::ConvolveMode::Full).to_vec()
        }),
        probe("kron", || {
            np::kron(&np::array![max], &np::array![2i64]).to_vec()
        }),
        probe("cross", || {
            np::cross(&np::array![max, 0, 0], &np::array![0i64, 2, 0]).to_vec()
        }),
        probe("ediff1d", || np::ediff1d(np::array![min, 1]).to_vec()),
        probe("neg", || (-np::array![min]).to_vec()),
        probe("einsum", || {
            np::einsum("i,i->", &[&np::array![max, 1], &np::array![2i64, 2]]).to_vec()
        }),
        probe("sum_axis", || np::array![[max], [1]].sum_axis(0).to_vec()),
        probe("prod_axis", || np::array![[max], [2]].prod_axis(0).to_vec()),
        probe("inner", || {
            np::linalg::inner(&np::array![max, 1], &np::array![2i64, 2])
        }),
        probe("vdot", || {
            np::linalg::vdot(&np::array![max, 1], &np::array![2i64, 2])
        }),
        probe("tensordot", || {
            np::linalg::tensordot_n(&np::array![max, 1], &np::array![2i64, 2], 1).to_vec()
        }),
        probe("shl", || (np::array![1i64] << np::array![70i64]).to_vec()),
        probe("shr", || (np::array![-8i64] >> np::array![70i64]).to_vec()),
        probe("ma sum", || {
            np::ma::masked_array(np::array![max, 1], None).sum()
        }),
        probe("ma prod", || {
            np::ma::masked_array(np::array![max, 2], None).prod()
        }),
        probe("u8 sum_axis", || {
            np::array![[200u8], [100]].sum_axis(0).to_vec()
        }),
    ];
    let want = [
        "sum: -9223372036854775808",
        "prod: -2",
        "cumsum: [9223372036854775807, -9223372036854775808]",
        "cumprod: [9223372036854775807, -2]",
        "diff: [-9223372036854775807]",
        "ptp: -1",
        "dot: [2]",
        "trace: -9223372036854775808",
        "square: [1]",
        "abs: [-9223372036854775808]",
        "power: [6048575297968530377]",
        "u8 sum: 44",
        "u8 diff: [255]",
        "matmul: [0]",
        "outer: [-2]",
        "convolve: [-2]",
        "kron: [-2]",
        "cross: [0, 0, -2]",
        "ediff1d: [-9223372036854775807]",
        "neg: [-9223372036854775808]",
        "einsum: [0]",
        "sum_axis: [-9223372036854775808]",
        "prod_axis: [-2]",
        "inner: 0",
        "vdot: 0",
        "tensordot: [0]",
        "shl: [0]",
        "shr: [-1]",
        "ma sum: Some(-9223372036854775808)",
        "ma prod: Some(-2)",
        "u8 sum_axis: [44]",
    ];
    let bad: Vec<String> = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.as_str() != *w)
        .map(|(g, w)| format!("got `{g}`, want `{w}`"))
        .collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn edge_cases_match_numpy() {
    // Expected values: NumPy 2.3 (see the audit notes); Debug formatting of the Rust results.
    let nan = f64::NAN;
    let e = || np::Array::<f64>::from(Vec::<f64>::new());
    let r = || np::linspace(0.0, 9.0, 10);
    let got = [
        probe("empty_sum", || e().sum()),
        probe("empty_prod", || e().prod()),
        probe("empty_mean", || e().mean()),
        probe("empty_var", || e().var()),
        probe("empty_std", || e().std()),
        probe("empty_median", || e().median()),
        probe("empty_cumsum", || e().cumsum().to_vec()),
        probe("empty_sort", || e().sort().to_vec()),
        probe("empty_unique", || np::unique(&e()).to_vec()),
        probe("linspace0", || np::linspace(0.0, 1.0, 0).to_vec()),
        probe("linspace1", || np::linspace(0.0, 1.0, 1).to_vec()),
        probe("arange_empty", || np::arange(5i64, 0, 1).to_vec()),
        probe("arange_neg", || np::arange(5i64, 0, -2).to_vec()),
        probe("nan_max", || np::array![1.0, nan, 3.0].max()),
        probe("nan_min", || np::array![1.0, nan, 3.0].min()),
        probe("nan_argmax", || np::array![1.0, nan, 3.0].argmax()),
        probe("nan_argmin", || np::array![1.0, nan, 3.0].argmin()),
        probe("nan_sort", || {
            np::array![3.0, nan, 1.0, nan].sort().to_vec()
        }),
        probe("nan_unique", || {
            np::unique(&np::array![nan, 1.0, nan, 1.0]).to_vec()
        }),
        probe("nan_median", || np::array![1.0, nan, 3.0].median()),
        probe("nan_maximum", || {
            np::maximum(np::array![nan, 1.0], np::array![1.0, nan]).to_vec()
        }),
        probe("nan_clip", || {
            np::array![nan, 5.0, -5.0].clip(-1.0, 1.0).to_vec()
        }),
        probe("rev_sum", || r().slice(&np::s![..;-1]).sum()),
        probe("rev_cumsum", || {
            np::linspace(0.0, 4.0, 5)
                .slice(&np::s![..;-1])
                .cumsum()
                .to_vec()
        }),
        probe("rev_argsort", || {
            np::array![3.0, 1.0, 2.0]
                .slice(&np::s![..;-1])
                .argsort()
                .to_vec()
        }),
        probe("step_mean", || r().slice(&np::s![1..;3]).mean()),
        probe("t_sum0", || {
            np::linspace(0.0, 5.0, 6)
                .reshape(&[2, 3])
                .t()
                .sum_axis(0)
                .to_vec()
        }),
        probe("zero_d_sum", || np::Array::scalar(5.0).sum()),
        probe("zero_d_reshape", || {
            np::Array::scalar(5.0).reshape(&[1, 1]).to_vec()
        }),
        probe("mm_empty", || {
            np::zeros::<f64>([2, 0])
                .matmul(&np::zeros::<f64>([0, 3]))
                .to_vec()
        }),
        probe("sum_axis_empty", || {
            np::zeros::<f64>([0, 3]).sum_axis(0).to_vec()
        }),
        probe("argmax_ties", || np::array![2.0, 5.0, 5.0].argmax()),
        probe("round_half", || {
            np::array![0.5, 1.5, 2.5, -0.5].round().to_vec()
        }),
    ];
    let want = [
        "empty_sum: 0.0",
        "empty_prod: 1.0",
        "empty_mean: NaN",
        "empty_var: NaN",
        "empty_std: NaN",
        "empty_median: NaN",
        "empty_cumsum: []",
        "empty_sort: []",
        "empty_unique: []",
        "linspace0: []",
        "linspace1: [0.0]",
        "arange_empty: []",
        "arange_neg: [5, 3, 1]",
        "nan_max: NaN",
        "nan_min: NaN",
        "nan_argmax: 1",
        "nan_argmin: 1",
        "nan_sort: [1.0, 3.0, NaN, NaN]",
        "nan_unique: [1.0, NaN]",
        "nan_median: NaN",
        "nan_maximum: [NaN, NaN]",
        "nan_clip: [NaN, 1.0, -1.0]",
        "rev_sum: 45.0",
        "rev_cumsum: [4.0, 7.0, 9.0, 10.0, 10.0]",
        "rev_argsort: [1, 0, 2]",
        "step_mean: 4.0",
        "t_sum0: [3.0, 12.0]",
        "zero_d_sum: 5.0",
        "zero_d_reshape: [5.0]",
        "mm_empty: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
        "sum_axis_empty: [0.0, 0.0, 0.0]",
        "argmax_ties: 1",
        "round_half: [0.0, 2.0, 2.0, -0.0]",
    ];
    let bad: Vec<String> = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.as_str() != *w)
        .map(|(g, w)| format!("got `{g}`, want `{w}`"))
        .collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn linalg_hard_inputs_return_errors_or_nan_never_panic() {
    use np::linalg as la;
    let nan = f64::NAN;
    let sing = || np::array![[1.0f64, 2.0], [2.0, 4.0]];
    let nanm = || np::array![[1.0, nan], [nan, 1.0]];
    let notpd = || np::array![[1.0, 2.0], [2.0, 1.0]];
    let e = || np::zeros::<f64>([0, 0]);
    let got = [
        probe("inv sing", || la::inv(&sing()).is_err()),
        probe("solve sing", || {
            la::solve(&sing(), &np::array![1.0, 2.0]).is_err()
        }),
        probe("det sing", || la::det(&sing())),
        probe("chol notpd", || la::cholesky(&notpd()).is_err()),
        probe("rank sing", || la::matrix_rank(&sing())),
        probe("pinv sing", || {
            la::pinv(&sing())
                .to_vec()
                .iter()
                .map(|x| (x * 25.0).round())
                .collect::<Vec<_>>()
        }),
        probe("lstsq sing", || {
            la::lstsq(&sing(), &np::array![1.0, 2.0]).map(|x| {
                x.to_vec()
                    .iter()
                    .map(|v| (v * 5.0).round())
                    .collect::<Vec<_>>()
            })
        }),
        probe("svd nan", || {
            la::svd(&nanm()).1.to_vec().iter().any(|x| x.is_nan())
        }),
        probe("eigh nan", || {
            la::eigh(&nanm()).0.to_vec().iter().any(|x| x.is_nan())
        }),
        probe("qr nan", || {
            la::qr(&nanm()).1.to_vec().iter().any(|x| x.is_nan())
        }),
        probe("eig nan", || la::eig(&nanm()).is_err()),
        probe("det nan", || la::det(&nanm()).is_nan()),
        probe("inv nan", || {
            la::inv(&nanm()).map(|x| x.to_vec().iter().any(|v| v.is_nan()))
        }),
        probe("det empty", || la::det(&e())),
        probe("inv empty", || la::inv(&e()).map(|x| x.shape().to_vec())),
        probe("svd empty", || {
            la::svd(&np::zeros::<f64>([0, 3])).1.to_vec()
        }),
        probe("eigh empty", || la::eigh(&e()).0.to_vec()),
        probe("qr empty", || {
            la::qr(&np::zeros::<f64>([0, 3])).1.shape().to_vec()
        }),
        probe("norm empty", || la::norm(&np::zeros::<f64>([0]))),
        probe("cond sing", || la::cond(&sing()) > 1e15),
        probe("slogdet sing", || la::slogdet(&sing())),
        probe("matpow neg sing", || la::matrix_power(&sing(), -1).is_err()),
        probe("svd big", || {
            la::svd(&np::array![[1e300f64, 1e300], [1e300, -1e300]])
                .1
                .to_vec()
                .iter()
                .all(|x| x.is_finite())
        }),
        probe("svd tiny", || {
            la::svd(&np::array![[1e-300f64, 0.0], [0.0, 1e-300]])
                .1
                .to_vec()
        }),
    ];
    let want = [
        "inv sing: true",
        "solve sing: true",
        "det sing: 0.0",
        "chol notpd: true",
        "rank sing: 1",
        "pinv sing: [1.0, 2.0, 2.0, 4.0]",
        "lstsq sing: Ok([1.0, 2.0])",
        "svd nan: PANIC SVD did not converge", // NumPy raises LinAlgError too
        "eigh nan: PANIC Eigenvalues did not converge",
        "qr nan: true",
        "eig nan: true",
        "det nan: true",
        "inv nan: Ok(true)",
        "det empty: 1.0",
        "inv empty: Ok([0, 0])",
        "svd empty: []",
        "eigh empty: []",
        "qr empty: [0, 3]",
        "norm empty: 0.0",
        "cond sing: true",
        "slogdet sing: (0.0, -inf)",
        "matpow neg sing: true",
        "svd big: true",
        "svd tiny: [1e-300, 1e-300]",
    ];
    let bad: Vec<String> = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.as_str() != *w)
        .map(|(g, w)| format!("got `{g}`, want `{w}`"))
        .collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn chebyshev_degree_one_root_matches_numpy() {
    // np.polynomial.chebyshev.chebroots([1, 2]) == [-0.5]  (1 + 2*T_1(x) = 1 + 2x)
    assert_eq!(
        np::polynomial::chebyshev::chebroots(&[1.0, 2.0]).0.to_vec(),
        vec![-0.5]
    );
    assert_eq!(
        np::polynomial::chebyshev::chebcompanion(&[3.0, -6.0]).to_vec(),
        vec![0.5]
    );
}

#[test]
fn huge_slice_steps_match_numpy() {
    // NumPy accepts any step; with |step| >= len the slice has at most one element.
    let a = np::Array::from_vec((0..15i64).collect(), &[3, 5]);
    let big = np::SliceArg::Range {
        start: None,
        end: None,
        step: isize::MAX,
    };
    let r = a.slice(&[big]);
    assert_eq!(r.shape(), &[1, 5]);
    assert_eq!(r.to_vec(), vec![0, 1, 2, 3, 4]); // a[::2**63-1]
    let neg = np::SliceArg::Range {
        start: Some(6),
        end: Some(-3),
        step: isize::MIN + 1,
    };
    assert_eq!(a.slice(&[neg]).to_vec(), vec![10, 11, 12, 13, 14]); // a[6:-3:-(2**63-1)]
    let empty = np::SliceArg::Range {
        start: Some(3),
        end: Some(5),
        step: isize::MAX,
    };
    assert_eq!(a.slice(&[empty]).shape(), &[0, 5]);
}

#[test]
fn half_precision_power_is_a_float_power() {
    // np.power(np.float16([2, 4, 2]), np.float16([.5, -1, 3])) == [1.414, 0.25, 8.]
    let h = |v: &[f32]| {
        np::Array::from_vec(
            v.iter().map(|&x| np::f16::from_f32(x)).collect(),
            &[v.len()],
        )
    };
    let r: Vec<f32> = np::power(h(&[2.0, 4.0, 2.0]), h(&[0.5, -1.0, 3.0]))
        .to_vec()
        .iter()
        .map(|x| x.to_f32())
        .collect();
    assert_eq!(r, vec![1.4140625, 0.25, 8.0]);
    let b = |v: &[f32]| {
        np::Array::from_vec(
            v.iter().map(|&x| np::bf16::from_f32(x)).collect(),
            &[v.len()],
        )
    };
    let r: Vec<f32> = np::power(b(&[9.0, 2.0]), b(&[0.5, -2.0]))
        .to_vec()
        .iter()
        .map(|x| x.to_f32())
        .collect();
    assert_eq!(r, vec![3.0, 0.25]);
}
