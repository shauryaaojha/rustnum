//! The SIMD-backed ufuncs agree with the scalar std functions on large contiguous arrays
//! (parallel chunks with ragged edges) and fall back exactly on strided views.
use rustnum as np;

/// Relative distance in units of the result's ulp.
fn ulps(got: f64, want: f64) -> f64 {
    if got == want || (got.is_nan() && want.is_nan()) {
        return 0.0;
    }
    let spacing = f64::from_bits(want.abs().to_bits() + 1) - want.abs();
    (got - want).abs() / spacing.max(f64::MIN_POSITIVE)
}

fn ulps32(got: f32, want: f32) -> f64 {
    if got == want || (got.is_nan() && want.is_nan()) {
        return 0.0;
    }
    let spacing = (f32::from_bits(want.abs().to_bits() + 1) - want.abs()) as f64;
    ((got as f64) - (want as f64)).abs() / spacing.max(f32::MIN_POSITIVE as f64)
}

fn check(
    name: &str,
    n: usize,
    lo: f64,
    hi: f64,
    f: fn(&np::Array<f64>) -> np::Array<f64>,
    reference: fn(f64) -> f64,
    max_ulp: f64,
) {
    // n is odd and above the parallel threshold, so chunks and the tail both run.
    let x = np::linspace(lo, hi, n);
    let got = f(&x);
    let worst = x
        .iter()
        .zip(got.iter())
        .map(|(x, y)| ulps(y, reference(x)))
        .fold(0.0, f64::max);
    assert!(worst <= max_ulp, "{name}: {worst} ulp");

    // A strided view takes the scalar path, so it matches std exactly.
    let view = x.slice(&np::s![..;3]);
    let strided = f(&view);
    for (x, y) in view.iter().zip(strided.iter()) {
        assert_eq!(y.to_bits(), reference(x).to_bits(), "{name} strided at {x}");
    }
}

fn check32(
    name: &str,
    n: usize,
    lo: f32,
    hi: f32,
    f: fn(&np::Array<f32>) -> np::Array<f32>,
    reference: fn(f32) -> f32,
    max_ulp: f64,
) {
    let x = np::linspace(lo, hi, n);
    let got = f(&x);
    let worst = x
        .iter()
        .zip(got.iter())
        .map(|(x, y)| ulps32(y, reference(x)))
        .fold(0.0, f64::max);
    assert!(worst <= max_ulp, "{name} f32: {worst} ulp");

    let view = x.slice(&np::s![..;3]);
    let strided = f(&view);
    for (x, y) in view.iter().zip(strided.iter()) {
        assert_eq!(
            y.to_bits(),
            reference(x).to_bits(),
            "{name} f32 strided at {x}"
        );
    }
}

#[test]
fn simd_ufuncs_match_std() {
    let n = 100_003;
    check("exp", n, -700.0, 700.0, |a| a.exp(), f64::exp, 2.0);
    check("exp2", n, -1000.0, 1000.0, |a| a.exp2(), f64::exp2, 2.0);
    check("expm1", n, -40.0, 700.0, |a| a.expm1(), f64::exp_m1, 2.0);
    check("log", n, 1e-300, 1e300, |a| a.log(), f64::ln, 2.0);
    check("log2", n, 1e-300, 1e300, |a| a.log2(), f64::log2, 3.0);
    check("log10", n, 1e-300, 1e300, |a| a.log10(), f64::log10, 3.0);
    check("log1p", n, -0.999, 1e300, |a| a.log1p(), f64::ln_1p, 2.0);
    check("sin", n, -1e4, 1e4, |a| a.sin(), f64::sin, 1.0);
    check("cos", n, -1e4, 1e4, |a| a.cos(), f64::cos, 1.0);
    check("tan", n, -1e4, 1e4, |a| a.tan(), f64::tan, 3.0);
    check("tanh", n, -30.0, 30.0, |a| a.tanh(), f64::tanh, 3.0);
}

#[test]
fn simd_ufuncs_match_std_f32() {
    let n = 100_003;
    check32("exp", n, -87.0, 88.0, |a| a.exp(), f32::exp, 3.0);
    check32("exp2", n, -126.0, 127.0, |a| a.exp2(), f32::exp2, 3.0);
    check32("expm1", n, -40.0, 88.0, |a| a.expm1(), f32::exp_m1, 3.0);
    check32("log", n, 1e-37, 1e37, |a| a.log(), f32::ln, 3.0);
    check32("log2", n, 1e-37, 1e37, |a| a.log2(), f32::log2, 3.0);
    check32("log10", n, 1e-37, 1e37, |a| a.log10(), f32::log10, 4.0);
    check32("log1p", n, -0.999, 1e37, |a| a.log1p(), f32::ln_1p, 2.0);
    check32("sin", n, -1e4, 1e4, |a| a.sin(), f32::sin, 2.0);
    check32("cos", n, -1e4, 1e4, |a| a.cos(), f32::cos, 2.0);
    check32("tan", n, -1e4, 1e4, |a| a.tan(), f32::tan, 3.0);
    check32("tanh", n, -30.0, 30.0, |a| a.tanh(), f32::tanh, 2.0);
}

#[test]
fn inplace_and_consuming_forms_match_the_allocating_form() {
    let x = np::linspace(-20.0, 20.0, 70_001);

    let mut inplace = x.copy();
    inplace.tanh_inplace();
    assert_eq!(inplace.to_vec(), x.tanh().to_vec());
    assert_eq!(x.copy().into_tanh().to_vec(), x.tanh().to_vec());

    let mut s = x.copy();
    s.sigmoid_inplace();
    assert_eq!(s.to_vec(), x.sigmoid().to_vec());
    assert_eq!(x.copy().into_sigmoid().to_vec(), x.sigmoid().to_vec());

    let mut em1 = x.copy();
    em1.expm1_inplace();
    assert_eq!(em1.to_vec(), x.expm1().to_vec());
    assert_eq!(x.copy().into_expm1().to_vec(), x.expm1().to_vec());

    let pos = np::linspace(0.1, 100.0, 70_001);
    let mut l2 = pos.copy();
    l2.log2_inplace();
    assert_eq!(l2.to_vec(), pos.log2().to_vec());
    assert_eq!(pos.copy().into_log2().to_vec(), pos.log2().to_vec());

    let mut l10 = pos.copy();
    l10.log10_inplace();
    assert_eq!(l10.to_vec(), pos.log10().to_vec());
    assert_eq!(pos.copy().into_log10().to_vec(), pos.log10().to_vec());

    let mut l1p = pos.copy();
    l1p.log1p_inplace();
    assert_eq!(l1p.to_vec(), pos.log1p().to_vec());
    assert_eq!(pos.copy().into_log1p().to_vec(), pos.log1p().to_vec());

    let mut t = x.copy();
    t.tan_inplace();
    assert_eq!(t.to_vec(), x.tan().to_vec());
    assert_eq!(x.copy().into_tan().to_vec(), x.tan().to_vec());

    // Shared storage is detached, not written through.
    let shared = x.clone();
    let mut y = x.clone();
    y.exp_inplace();
    assert_eq!(shared.to_vec(), x.to_vec());
}

#[test]
fn inplace_and_consuming_forms_match_the_allocating_form_f32() {
    let x = np::linspace(-10.0f32, 10.0f32, 70_001);

    let mut inplace = x.copy();
    inplace.tanh_inplace();
    assert_eq!(inplace.to_vec(), x.tanh().to_vec());
    assert_eq!(x.copy().into_tanh().to_vec(), x.tanh().to_vec());

    let mut s = x.copy();
    s.sigmoid_inplace();
    assert_eq!(s.to_vec(), x.sigmoid().to_vec());
    assert_eq!(x.copy().into_sigmoid().to_vec(), x.sigmoid().to_vec());

    let mut em1 = x.copy();
    em1.expm1_inplace();
    assert_eq!(em1.to_vec(), x.expm1().to_vec());
    assert_eq!(x.copy().into_expm1().to_vec(), x.expm1().to_vec());

    let pos = np::linspace(0.1f32, 50.0f32, 70_001);
    let mut l2 = pos.copy();
    l2.log2_inplace();
    assert_eq!(l2.to_vec(), pos.log2().to_vec());
    assert_eq!(pos.copy().into_log2().to_vec(), pos.log2().to_vec());

    let mut l10 = pos.copy();
    l10.log10_inplace();
    assert_eq!(l10.to_vec(), pos.log10().to_vec());
    assert_eq!(pos.copy().into_log10().to_vec(), pos.log10().to_vec());

    let mut l1p = pos.copy();
    l1p.log1p_inplace();
    assert_eq!(l1p.to_vec(), pos.log1p().to_vec());
    assert_eq!(pos.copy().into_log1p().to_vec(), pos.log1p().to_vec());

    let mut t = x.copy();
    t.tan_inplace();
    assert_eq!(t.to_vec(), x.tan().to_vec());
    assert_eq!(x.copy().into_tan().to_vec(), x.tan().to_vec());

    let shared = x.clone();
    let mut y = x.clone();
    y.exp_inplace();
    assert_eq!(shared.to_vec(), x.to_vec());
}

#[test]
fn sigmoid_is_accurate_for_large_negative_inputs() {
    let x = np::array![-700.0f64, -246.76, -40.0, -10.0, 0.0, 10.0, 40.0];
    for (x, y) in x.iter().zip(x.sigmoid().iter()) {
        let want = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            x.exp() / (1.0 + x.exp())
        };
        assert!(ulps(y, want) <= 3.0, "sigmoid({x}) = {y}, want {want}");
    }
}

#[test]
fn edge_cases_f64_exact() {
    // NaN propagation
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    let neg_inf = f64::NEG_INFINITY;

    let x = np::array![nan, inf, neg_inf, 0.0, -0.0];
    assert!(x.exp().to_vec()[0].is_nan());
    assert_eq!(x.exp().to_vec()[1], inf);
    assert_eq!(x.exp().to_vec()[2], 0.0);

    assert!(x.exp2().to_vec()[0].is_nan());
    assert_eq!(x.exp2().to_vec()[1], inf);
    assert_eq!(x.exp2().to_vec()[2], 0.0);

    assert!(x.expm1().to_vec()[0].is_nan());
    assert_eq!(x.expm1().to_vec()[1], inf);
    assert_eq!(x.expm1().to_vec()[2], -1.0);
    assert_eq!(x.expm1().to_vec()[3].to_bits(), 0.0f64.to_bits());
    assert_eq!(x.expm1().to_vec()[4].to_bits(), (-0.0f64).to_bits());

    // Log edges
    let log_x = np::array![nan, inf, 0.0, -0.0, -1.0];
    assert!(log_x.log().to_vec()[0].is_nan());
    assert_eq!(log_x.log().to_vec()[1], inf);
    assert_eq!(log_x.log().to_vec()[2], neg_inf);
    assert_eq!(log_x.log().to_vec()[3], neg_inf);
    assert!(log_x.log().to_vec()[4].is_nan());

    assert!(log_x.log2().to_vec()[0].is_nan());
    assert_eq!(log_x.log2().to_vec()[1], inf);
    assert_eq!(log_x.log2().to_vec()[2], neg_inf);
    assert!(log_x.log2().to_vec()[4].is_nan());

    assert!(log_x.log10().to_vec()[0].is_nan());
    assert_eq!(log_x.log10().to_vec()[1], inf);
    assert_eq!(log_x.log10().to_vec()[2], neg_inf);
    assert!(log_x.log10().to_vec()[4].is_nan());

    let log1p_x = np::array![nan, inf, 0.0, -0.0, -1.0, -2.0];
    assert!(log1p_x.log1p().to_vec()[0].is_nan());
    assert_eq!(log1p_x.log1p().to_vec()[1], inf);
    assert_eq!(log1p_x.log1p().to_vec()[2].to_bits(), 0.0f64.to_bits());
    assert_eq!(log1p_x.log1p().to_vec()[3].to_bits(), (-0.0f64).to_bits());
    assert_eq!(log1p_x.log1p().to_vec()[4], neg_inf);
    assert!(log1p_x.log1p().to_vec()[5].is_nan());

    // Trig signed zeros and inf
    let trig_x = np::array![nan, inf, neg_inf, 0.0, -0.0];
    assert!(trig_x.sin().to_vec()[0].is_nan());
    assert!(trig_x.sin().to_vec()[1].is_nan());
    assert!(trig_x.sin().to_vec()[2].is_nan());
    assert_eq!(trig_x.sin().to_vec()[3].to_bits(), 0.0f64.to_bits());
    assert_eq!(trig_x.sin().to_vec()[4].to_bits(), (-0.0f64).to_bits());

    assert!(trig_x.cos().to_vec()[0].is_nan());
    assert!(trig_x.cos().to_vec()[1].is_nan());
    assert_eq!(trig_x.cos().to_vec()[3], 1.0);
    assert_eq!(trig_x.cos().to_vec()[4], 1.0);

    assert!(trig_x.tan().to_vec()[0].is_nan());
    assert!(trig_x.tan().to_vec()[1].is_nan());
    assert_eq!(trig_x.tan().to_vec()[3].to_bits(), 0.0f64.to_bits());
    assert_eq!(trig_x.tan().to_vec()[4].to_bits(), (-0.0f64).to_bits());

    assert!(trig_x.tanh().to_vec()[0].is_nan());
    assert_eq!(trig_x.tanh().to_vec()[1], 1.0);
    assert_eq!(trig_x.tanh().to_vec()[2], -1.0);
    assert_eq!(trig_x.tanh().to_vec()[3].to_bits(), 0.0f64.to_bits());
    assert_eq!(trig_x.tanh().to_vec()[4].to_bits(), (-0.0f64).to_bits());

    assert!(trig_x.sigmoid().to_vec()[0].is_nan());
    assert_eq!(trig_x.sigmoid().to_vec()[1], 1.0);
    assert_eq!(trig_x.sigmoid().to_vec()[2], 0.0);
    assert_eq!(trig_x.sigmoid().to_vec()[3], 0.5);

    // Subnormals & tiny inputs |x| < 1e-10
    let tiny = np::array![1e-15f64, -1e-15, 1e-300, -1e-300];
    for val in tiny.iter() {
        let single = np::array![val];
        assert_eq!(single.expm1().to_vec()[0], val.exp_m1());
        assert_eq!(single.log1p().to_vec()[0], val.ln_1p());
        assert_eq!(single.sin().to_vec()[0], val.sin());
        assert_eq!(single.tan().to_vec()[0], val.tan());
        assert_eq!(single.tanh().to_vec()[0], val.tanh());
    }
}

#[test]
fn edge_cases_f32_exact() {
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;

    let x = np::array![nan, inf, neg_inf, 0.0f32, -0.0f32];
    assert!(x.exp().to_vec()[0].is_nan());
    assert_eq!(x.exp().to_vec()[1], inf);
    assert_eq!(x.exp().to_vec()[2], 0.0);

    assert!(x.exp2().to_vec()[0].is_nan());
    assert_eq!(x.exp2().to_vec()[1], inf);
    assert_eq!(x.exp2().to_vec()[2], 0.0);

    assert!(x.expm1().to_vec()[0].is_nan());
    assert_eq!(x.expm1().to_vec()[1], inf);
    assert_eq!(x.expm1().to_vec()[2], -1.0);
    assert_eq!(x.expm1().to_vec()[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(x.expm1().to_vec()[4].to_bits(), (-0.0f32).to_bits());

    let log_x = np::array![nan, inf, 0.0f32, -0.0f32, -1.0f32];
    assert!(log_x.log().to_vec()[0].is_nan());
    assert_eq!(log_x.log().to_vec()[1], inf);
    assert_eq!(log_x.log().to_vec()[2], neg_inf);
    assert_eq!(log_x.log().to_vec()[3], neg_inf);
    assert!(log_x.log().to_vec()[4].is_nan());

    assert!(log_x.log2().to_vec()[0].is_nan());
    assert_eq!(log_x.log2().to_vec()[1], inf);
    assert_eq!(log_x.log2().to_vec()[2], neg_inf);
    assert!(log_x.log2().to_vec()[4].is_nan());

    assert!(log_x.log10().to_vec()[0].is_nan());
    assert_eq!(log_x.log10().to_vec()[1], inf);
    assert_eq!(log_x.log10().to_vec()[2], neg_inf);
    assert!(log_x.log10().to_vec()[4].is_nan());

    let log1p_x = np::array![nan, inf, 0.0f32, -0.0f32, -1.0f32, -2.0f32];
    assert!(log1p_x.log1p().to_vec()[0].is_nan());
    assert_eq!(log1p_x.log1p().to_vec()[1], inf);
    assert_eq!(log1p_x.log1p().to_vec()[2].to_bits(), 0.0f32.to_bits());
    assert_eq!(log1p_x.log1p().to_vec()[3].to_bits(), (-0.0f32).to_bits());
    assert_eq!(log1p_x.log1p().to_vec()[4], neg_inf);
    assert!(log1p_x.log1p().to_vec()[5].is_nan());

    let trig_x = np::array![nan, inf, neg_inf, 0.0f32, -0.0f32];
    assert!(trig_x.sin().to_vec()[0].is_nan());
    assert!(trig_x.sin().to_vec()[1].is_nan());
    assert_eq!(trig_x.sin().to_vec()[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(trig_x.sin().to_vec()[4].to_bits(), (-0.0f32).to_bits());

    assert!(trig_x.cos().to_vec()[0].is_nan());
    assert!(trig_x.cos().to_vec()[1].is_nan());
    assert_eq!(trig_x.cos().to_vec()[3], 1.0);
    assert_eq!(trig_x.cos().to_vec()[4], 1.0);

    assert!(trig_x.tan().to_vec()[0].is_nan());
    assert!(trig_x.tan().to_vec()[1].is_nan());
    assert_eq!(trig_x.tan().to_vec()[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(trig_x.tan().to_vec()[4].to_bits(), (-0.0f32).to_bits());

    assert!(trig_x.tanh().to_vec()[0].is_nan());
    assert_eq!(trig_x.tanh().to_vec()[1], 1.0);
    assert_eq!(trig_x.tanh().to_vec()[2], -1.0);
    assert_eq!(trig_x.tanh().to_vec()[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(trig_x.tanh().to_vec()[4].to_bits(), (-0.0f32).to_bits());

    assert!(trig_x.sigmoid().to_vec()[0].is_nan());
    assert_eq!(trig_x.sigmoid().to_vec()[1], 1.0);
    assert_eq!(trig_x.sigmoid().to_vec()[2], 0.0);
    assert_eq!(trig_x.sigmoid().to_vec()[3], 0.5);

    // Subnormals & tiny inputs |x| < 1e-10
    let tiny = np::array![1e-12f32, -1e-12, 1e-37, -1e-37];
    for val in tiny.iter() {
        let single = np::array![val];
        assert_eq!(single.expm1().to_vec()[0], val.exp_m1());
        assert_eq!(single.log1p().to_vec()[0], val.ln_1p());
        assert_eq!(single.sin().to_vec()[0], val.sin());
        assert_eq!(single.tan().to_vec()[0], val.tan());
        assert_eq!(single.tanh().to_vec()[0], val.tanh());
    }
}
