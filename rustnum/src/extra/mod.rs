//! Kernels for NumPy's statistics, signal, set and tensor-contraction functions.
//!
//! Everything here is pure slice code: `&[T]` plus a shape, no `Array`. The public
//! `Array`-level wrappers live in `api.rs.pending` and are wired up at merge time, so until
//! then nothing outside the tests calls these functions (hence `allow(dead_code)`).
//!
//! Conventions shared by all kernels:
//! * multi-dimensional buffers are C-ordered (row-major) and contiguous;
//! * misuse panics with NumPy's wording where NumPy has one;
//! * NaN sorts last (NumPy's sort order).

#![allow(dead_code)]

mod einsum_kernels;
mod set_kernels;
mod signal_kernels;
mod stats_kernels;

use std::cmp::Ordering;

/// `true` for values that are not equal to themselves (NaN), without a float-only API.
pub(crate) fn is_nan_like<T: PartialOrd>(x: &T) -> bool {
    x.partial_cmp(x).is_none()
}

/// Total order used by NumPy's `sort`: ordinary order, NaN after everything, NaNs equal.
pub(crate) fn cmp_nan_last<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    match a.partial_cmp(b) {
        Some(o) => o,
        None => is_nan_like(a).cmp(&is_nan_like(b)),
    }
}

/// `np.searchsorted(sorted, v, side="left")` for NaN-last sorted input.
pub(crate) fn searchsorted_left<T: PartialOrd>(sorted: &[T], v: &T) -> usize {
    sorted.partition_point(|e| cmp_nan_last(e, v) == Ordering::Less)
}

/// `np.searchsorted(sorted, v, side="right")` for NaN-last sorted input.
pub(crate) fn searchsorted_right<T: PartialOrd>(sorted: &[T], v: &T) -> usize {
    sorted.partition_point(|e| cmp_nan_last(e, v) != Ordering::Greater)
}

/// Resolve a possibly negative axis, panicking like NumPy.
pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> usize {
    let n = ndim as isize;
    if axis < -n || axis >= n {
        panic!("axis {axis} is out of bounds for array of dimension {ndim}");
    }
    if axis < 0 {
        (axis + n) as usize
    } else {
        axis as usize
    }
}

/// Split a C-ordered shape around `axis` into `(outer, len, inner)`.
///
/// Element `(o, k, i)` lives at flat index `(o * len + k) * inner + i`.
pub(crate) fn axis_split(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    let outer = shape[..axis].iter().product();
    let inner = shape[axis + 1..].iter().product();
    (outer, shape[axis], inner)
}

/// Test helper: elementwise comparison with 1e-12 relative tolerance; NaN equals NaN.
#[cfg(test)]
pub(crate) fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "length: got {got:?}, want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = (g.is_nan() && w.is_nan()) || g == w || (g - w).abs() <= 1e-12 * w.abs().max(1.0);
        assert!(
            ok,
            "element {i}: got {g}, want {w}
 got  {got:?}
 want {want:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_last_order() {
        let mut v = [3.0, f64::NAN, 1.0, f64::NAN, 2.0];
        v.sort_by(cmp_nan_last);
        assert_eq!(&v[..3], &[1.0, 2.0, 3.0]);
        assert!(v[3].is_nan() && v[4].is_nan());
    }

    #[test]
    fn searchsorted_sides() {
        let s = [1.0, 2.0, 2.0, 4.0, f64::NAN];
        assert_eq!(searchsorted_left(&s, &2.0), 1);
        assert_eq!(searchsorted_right(&s, &2.0), 3);
        assert_eq!(searchsorted_left(&s, &9.0), 4);
        assert_eq!(searchsorted_left(&s, &f64::NAN), 4);
        assert_eq!(searchsorted_right(&s, &f64::NAN), 5);
    }

    #[test]
    fn axes() {
        assert_eq!(normalize_axis(-1, 3), 2);
        assert_eq!(axis_split(&[2, 3, 4], 1), (2, 3, 4));
    }

    #[test]
    #[should_panic(expected = "axis 2 is out of bounds for array of dimension 2")]
    fn bad_axis() {
        normalize_axis(2, 2);
    }
}
