//! The SIMD-backed ufuncs agree with the scalar std functions on large contiguous arrays
//! (parallel chunks with ragged edges) and fall back exactly on strided views.
use rustnum as np;

/// Relative distance in units of the result's ulp.
fn ulps(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    let spacing = f64::from_bits(want.abs().to_bits() + 1) - want.abs();
    (got - want).abs() / spacing.max(f64::MIN_POSITIVE)
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

#[test]
fn simd_ufuncs_match_std() {
    let n = 100_003;
    check("exp", n, -700.0, 700.0, |a| a.exp(), f64::exp, 2.0);
    check("exp2", n, -1000.0, 1000.0, |a| a.exp2(), f64::exp2, 2.0);
    check("log", n, 1e-300, 1e300, |a| a.log(), f64::ln, 2.0);
    check("sin", n, -1e4, 1e4, |a| a.sin(), f64::sin, 1.0);
    check("cos", n, -1e4, 1e4, |a| a.cos(), f64::cos, 1.0);
    check("tanh", n, -30.0, 30.0, |a| a.tanh(), f64::tanh, 3.0);
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
    // Shared storage is detached, not written through.
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
