//! Shared assertions for the NumPy-parity tests.
//!
//! Every test creates a [`Report`] first; failed comparisons are collected and printed together
//! when the report is dropped, so one run shows all disagreements with NumPy.

use std::cell::RefCell;

use super::complex::Complex;
use crate::Float;

thread_local! {
    static FAILS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

fn fail(msg: String) {
    FAILS.with(|f| f.borrow_mut().push(msg));
}

/// Panics on drop if any comparison of this test failed.
pub struct Report;

impl Drop for Report {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        let fails = FAILS.with(|f| core::mem::take(&mut *f.borrow_mut()));
        if !fails.is_empty() {
            let shown = fails
                .iter()
                .take(40)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            panic!(
                "{} comparison(s) disagree with NumPy:\n{shown}",
                fails.len()
            );
        }
    }
}

/// Compare `got` with NumPy's `want`.
///
/// NaN matches NaN (payload ignored). Infinities and zeros must match exactly, including the
/// sign of zero. Other values must agree within `atol + rtol * |want|`; with `rtol = atol = 0`
/// the bit patterns must be identical.
pub fn check<F: Float>(what: &str, input: &str, got: F, want: F, rtol: f64, atol: f64) {
    let ok = if want.is_nan() {
        got.is_nan()
    } else if want.is_infinite() || want == F::ZERO || (rtol == 0.0 && atol == 0.0) {
        got.to_bits64() == want.to_bits64()
    } else {
        let (g, w) = (got.to_f64(), want.to_f64());
        (g - w).abs() <= atol + rtol * w.abs()
    };
    if !ok {
        fail(format!("{what}({input}): got {got:?}, NumPy says {want:?}"));
    }
}

/// Like [`check`] for the two parts of a complex result.
///
/// Finite parts are compared against the modulus of the expected value, so cancellation in one
/// part does not cause false alarms. When `loose_signs` is set (used for non-finite inputs,
/// where C99 Annex G leaves the sign of a zero or infinity unspecified and libms disagree),
/// zeros and infinities only have to match in magnitude.
pub fn check_c<F: Float>(
    what: &str,
    input: &str,
    got: Complex<F>,
    want: (F, F),
    rtol: f64,
    loose_signs: bool,
) {
    let norm = if want.0.is_finite() && want.1.is_finite() {
        want.0.to_f64().hypot(want.1.to_f64())
    } else {
        0.0
    };
    for (part, g, w) in [("re", got.re, want.0), ("im", got.im, want.1)] {
        let ok = if w.is_nan() {
            g.is_nan()
        } else if w.is_infinite() || w == F::ZERO || rtol == 0.0 {
            if loose_signs {
                g.abs().to_bits64() == w.abs().to_bits64()
            } else {
                g.to_bits64() == w.to_bits64()
            }
        } else {
            (g.to_f64() - w.to_f64()).abs() <= rtol * norm.max(w.to_f64().abs())
        };
        if !ok {
            fail(format!(
                "{what}({input}).{part}: got {g:?}, NumPy says {w:?} (full got {got:?}, want {want:?})"
            ));
        }
    }
}

/// True when both parts of `z` are finite (the strict-sign regime).
pub fn finite_input<F: Float>(z: Complex<F>) -> bool {
    z.re.is_finite() && z.im.is_finite()
}
