//! Signal and numeric kernels: `convolve`, `correlate`, `diff`, `gradient`, `interp`,
//! `trapezoid`, `cross`, `kron`.
//!
//! Buffers are C-ordered slices. See the module docs of `extra` for conventions.

use super::axis_split;
use crate::Num;
use crate::element::{add, mul, sub};

/// Output size selector of `convolve` / `correlate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Every overlap: length `n + m - 1`.
    Full,
    /// Length `max(n, m)`, centred with respect to `Full`.
    Same,
    /// Only complete overlaps: length `max(n, m) - min(n, m) + 1`.
    Valid,
}

/// Cross-correlation of `a` (the longer) with `v` as `np.core.multiarray.correlate` does:
/// `out[k] = sum_j a[k - n_left + j] * v[j]` with zero padding.
fn correlate_core<T>(a: &[T], v: &[T], mode: Mode) -> Vec<T>
where
    T: Num,
{
    let (n1, n2) = (a.len(), v.len());
    let (n_left, len) = match mode {
        Mode::Valid => (0, n1 - n2 + 1),
        Mode::Same => (n2 / 2, n1),
        Mode::Full => (n2 - 1, n1 + n2 - 1),
    };
    (0..len)
        .map(|k| {
            // Window of `a` overlapping `v` for output position k.
            let start = k as isize - n_left as isize;
            let j0 = (-start).max(0) as usize;
            let j1 = ((n1 as isize - start).min(n2 as isize)).max(0) as usize;
            let mut acc = T::default();
            for j in j0..j1 {
                acc = add(acc, mul(a[(start + j as isize) as usize], v[j]));
            }
            acc
        })
        .collect()
}

fn check_nonempty(a: usize, v: usize) {
    if a == 0 {
        panic!("a cannot be empty");
    }
    if v == 0 {
        panic!("v cannot be empty");
    }
}

/// `np.convolve(a, v, mode)`. When `v` is longer the arguments are swapped, as in NumPy.
pub(crate) fn convolve<T>(a: &[T], v: &[T], mode: Mode) -> Vec<T>
where
    T: Num,
{
    check_nonempty(a.len(), v.len());
    let (a, v) = if v.len() > a.len() { (v, a) } else { (a, v) };
    let rev: Vec<T> = v.iter().rev().copied().collect();
    correlate_core(a, &rev, mode)
}

/// `np.correlate(a, v, mode)`: `c[k] = sum_n a[n + k] * v[n]` (real input, no conjugation).
///
/// When `v` is longer, the arguments are swapped and the result reversed, as in NumPy.
pub(crate) fn correlate<T>(a: &[T], v: &[T], mode: Mode) -> Vec<T>
where
    T: Num,
{
    check_nonempty(a.len(), v.len());
    if v.len() > a.len() {
        let mut out = correlate_core(v, a, mode);
        out.reverse();
        out
    } else {
        correlate_core(a, v, mode)
    }
}

/// `np.diff(x, n, axis)` for a C-ordered buffer; returns the data and the new shape.
///
/// The axis shrinks by `n` (to 0 at most, giving an empty result).
pub(crate) fn diff<T>(x: &[T], shape: &[usize], n: usize, axis: usize) -> (Vec<T>, Vec<usize>)
where
    T: Num,
{
    let mut cur = x.to_vec();
    let mut shp = shape.to_vec();
    for _ in 0..n {
        let (outer, len, inner) = axis_split(&shp, axis);
        let new_len = len.saturating_sub(1);
        let mut next = Vec::with_capacity(outer * new_len * inner);
        for o in 0..outer {
            for k in 0..new_len {
                for i in 0..inner {
                    let base = (o * len + k) * inner + i;
                    next.push(sub(cur[base + inner], cur[base]));
                }
            }
        }
        shp[axis] = new_len;
        cur = next;
    }
    (cur, shp)
}

/// Sample spacing for `gradient`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Spacing<'a> {
    /// Constant distance between samples.
    Uniform(f64),
    /// Coordinates of the samples along the axis (length `shape[axis]`).
    Coords(&'a [f64]),
}

/// `np.gradient(x, spacing, axis=axis, edge_order=edge_order)` for a C-ordered buffer.
///
/// Interior points use second-order central differences; edges use one-sided differences of
/// order `edge_order` (1 or 2). Output has the shape of the input.
pub(crate) fn gradient(
    x: &[f64],
    shape: &[usize],
    axis: usize,
    spacing: Spacing<'_>,
    edge_order: usize,
) -> Vec<f64> {
    if edge_order != 1 && edge_order != 2 {
        panic!("'edge_order' greater than 2 not supported");
    }
    let (outer, n, inner) = axis_split(shape, axis);
    if n < edge_order + 1 {
        panic!(
            "Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required."
        );
    }
    if let Spacing::Coords(c) = spacing
        && c.len() != n
    {
        panic!("when 1d, distances must match the length of the corresponding dimension");
    }
    // Distance between sample k and k + 1.
    let dx = |k: usize| match spacing {
        Spacing::Uniform(h) => h,
        Spacing::Coords(c) => c[k + 1] - c[k],
    };
    let mut out = vec![0.0; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| (o * n + k) * inner + i;
            let f = |k: usize| x[at(k)];
            for k in 1..n - 1 {
                let (d1, d2) = (dx(k - 1), dx(k));
                out[at(k)] = match spacing {
                    Spacing::Uniform(_) => (f(k + 1) - f(k - 1)) / (2.0 * d1),
                    Spacing::Coords(_) => {
                        let a = -d2 / (d1 * (d1 + d2));
                        let b = (d2 - d1) / (d1 * d2);
                        let c = d1 / (d2 * (d1 + d2));
                        a * f(k - 1) + b * f(k) + c * f(k + 1)
                    }
                };
            }
            if edge_order == 1 {
                out[at(0)] = (f(1) - f(0)) / dx(0);
                out[at(n - 1)] = (f(n - 1) - f(n - 2)) / dx(n - 2);
            } else {
                let (d1, d2) = (dx(0), dx(1));
                let a = -(2.0 * d1 + d2) / (d1 * (d1 + d2));
                let b = (d1 + d2) / (d1 * d2);
                let c = -d1 / (d2 * (d1 + d2));
                out[at(0)] = a * f(0) + b * f(1) + c * f(2);
                let (d1, d2) = (dx(n - 3), dx(n - 2));
                let a = d2 / (d1 * (d1 + d2));
                let b = -(d2 + d1) / (d1 * d2);
                let c = (2.0 * d2 + d1) / (d2 * (d1 + d2));
                out[at(n - 1)] = a * f(n - 3) + b * f(n - 2) + c * f(n - 1);
            }
        }
    }
    out
}

/// `np.interp(x, xp, fp, left, right)`: piecewise-linear interpolation.
///
/// `xp` must be increasing (not checked, like NumPy). Outside `[xp[0], xp[-1]]` the result is
/// `left` / `right`, defaulting to `fp[0]` / `fp[-1]`. NaN in `x` gives NaN (except for a
/// single-point `xp`, where NumPy returns `fp[0]`).
pub(crate) fn interp(
    x: &[f64],
    xp: &[f64],
    fp: &[f64],
    left: Option<f64>,
    right: Option<f64>,
) -> Vec<f64> {
    if xp.is_empty() {
        panic!("array of sample points is empty");
    }
    if xp.len() != fp.len() {
        panic!("fp and xp are not of the same length.");
    }
    let n = xp.len();
    let lval = left.unwrap_or(fp[0]);
    let rval = right.unwrap_or(fp[n - 1]);
    if n == 1 {
        return x
            .iter()
            .map(|&v| {
                if v < xp[0] {
                    lval
                } else if v > xp[0] {
                    rval
                } else {
                    fp[0]
                }
            })
            .collect();
    }
    x.iter()
        .map(|&v| {
            if v.is_nan() {
                return v;
            }
            if v < xp[0] {
                return lval;
            }
            if v > xp[n - 1] {
                return rval;
            }
            // Largest j with xp[j] <= v.
            let j = xp.partition_point(|&e| e <= v) - 1;
            if j == n - 1 || xp[j] == v {
                return fp[j];
            }
            let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
            let mut r = slope * (v - xp[j]) + fp[j];
            if r.is_nan() {
                r = slope * (v - xp[j + 1]) + fp[j + 1];
                if r.is_nan() && fp[j] == fp[j + 1] {
                    r = fp[j];
                }
            }
            r
        })
        .collect()
}

/// `np.trapezoid(y, x, dx, axis)` for a C-ordered buffer; output has `axis` removed.
///
/// `Spacing::Uniform(dx)` or `Spacing::Coords(x)` with `x.len() == shape[axis]`.
pub(crate) fn trapezoid(y: &[f64], shape: &[usize], axis: usize, spacing: Spacing<'_>) -> Vec<f64> {
    let (outer, n, inner) = axis_split(shape, axis);
    if let Spacing::Coords(c) = spacing
        && c.len() != n
    {
        panic!(
            "operands could not be broadcast together with shapes [{n}] [{}]",
            c.len()
        );
    }
    let mut out = Vec::with_capacity(outer * inner);
    let mut terms = vec![0.0; n.saturating_sub(1)];
    for o in 0..outer {
        for i in 0..inner {
            for (k, t) in terms.iter_mut().enumerate() {
                let d = match spacing {
                    Spacing::Uniform(h) => h,
                    Spacing::Coords(c) => c[k + 1] - c[k],
                };
                let (a, b) = (y[(o * n + k) * inner + i], y[(o * n + k + 1) * inner + i]);
                *t = d * (b + a) / 2.0;
            }
            out.push(super::stats_kernels::pairwise_sum(&terms));
        }
    }
    out
}

/// `np.cross(a, b)` over the last axis, for batches of 2-vectors and 3-vectors.
///
/// `a` holds rows of length `da` and `b` rows of length `db` (each 2 or 3), same row count.
/// A 2-vector is a 3-vector with `z = 0`. Two 2-vectors give one scalar (the z component) per
/// row; otherwise each row has 3 components.
pub(crate) fn cross<T>(a: &[T], da: usize, b: &[T], db: usize) -> Vec<T>
where
    T: Num,
{
    if !(da == 2 || da == 3) || !(db == 2 || db == 3) {
        panic!("incompatible dimensions for cross product\n(dimension must be 2 or 3)");
    }
    let rows = a.len() / da;
    if b.len() / db != rows {
        panic!(
            "operands could not be broadcast together with shapes [{rows}] [{}]",
            b.len() / db
        );
    }
    let z = T::default();
    let mut out = Vec::with_capacity(rows * if da == 2 && db == 2 { 1 } else { 3 });
    for r in 0..rows {
        let (u, v) = (&a[r * da..(r + 1) * da], &b[r * db..(r + 1) * db]);
        let (u0, u1, u2) = (u[0], u[1], if da == 3 { u[2] } else { z });
        let (v0, v1, v2) = (v[0], v[1], if db == 3 { v[2] } else { z });
        let cz = sub(mul(u0, v1), mul(u1, v0));
        if da == 2 && db == 2 {
            out.push(cz);
        } else {
            out.push(sub(mul(u1, v2), mul(u2, v1)));
            out.push(sub(mul(u2, v0), mul(u0, v2)));
            out.push(cz);
        }
    }
    out
}

/// `np.kron(a, b)` for C-ordered buffers; the shorter shape is padded with leading 1s.
///
/// Result shape is `a_shape[i] * b_shape[i]` per axis.
pub(crate) fn kron<T>(
    a: &[T],
    a_shape: &[usize],
    b: &[T],
    b_shape: &[usize],
) -> (Vec<T>, Vec<usize>)
where
    T: Num,
{
    let nd = a_shape.len().max(b_shape.len());
    let pad = |s: &[usize]| -> Vec<usize> {
        let mut p = vec![1; nd - s.len()];
        p.extend_from_slice(s);
        p
    };
    let (sa, sb) = (pad(a_shape), pad(b_shape));
    let out_shape: Vec<usize> = sa.iter().zip(&sb).map(|(x, y)| x * y).collect();
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; nd];
    for _ in 0..total {
        let (mut ia, mut ib) = (0, 0);
        for d in 0..nd {
            ia = ia * sa[d] + idx[d] / sb[d];
            ib = ib * sb[d] + idx[d] % sb[d];
        }
        out.push(mul(a[ia], b[ib]));
        for d in (0..nd).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    (out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::assert_close;

    fn check_conv(a: &[f64], b: &[f64], mode: Mode, conv: &[f64], corr: &[f64]) {
        assert_close(&convolve(a, b, mode), conv);
        assert_close(&correlate(a, b, mode), corr);
    }

    fn check_diff(
        x: &[f64],
        shape: &[usize],
        n: usize,
        axis: usize,
        want: &[f64],
        want_shape: &[usize],
    ) {
        let (got, got_shape) = diff(x, shape, n, axis);
        assert_eq!(got_shape, want_shape);
        assert_close(&got, want);
    }

    #[test]
    fn convolve_correlate_all_modes_and_length_combos() {
        check_conv(
            &[5.0, 1.0, 2.0],
            &[4.0, 1.0, 3.0],
            Mode::Full,
            &[20.0, 9.0, 24.0, 5.0, 6.0],
            &[15.0, 8.0, 27.0, 6.0, 8.0],
        );
        check_conv(
            &[5.0, 1.0, 2.0],
            &[4.0, 1.0, 3.0],
            Mode::Same,
            &[9.0, 24.0, 5.0],
            &[8.0, 27.0, 6.0],
        );
        check_conv(
            &[5.0, 1.0, 2.0],
            &[4.0, 1.0, 3.0],
            Mode::Valid,
            &[24.0],
            &[27.0],
        );
        check_conv(
            &[4.0, -3.0, -5.0, -2.0],
            &[-2.0, 4.0],
            Mode::Full,
            &[-8.0, 22.0, -2.0, -16.0, -8.0],
            &[16.0, -20.0, -14.0, 2.0, 4.0],
        );
        check_conv(
            &[4.0, -3.0, -5.0, -2.0],
            &[-2.0, 4.0],
            Mode::Same,
            &[-8.0, 22.0, -2.0, -16.0],
            &[16.0, -20.0, -14.0, 2.0],
        );
        check_conv(
            &[4.0, -3.0, -5.0, -2.0],
            &[-2.0, 4.0],
            Mode::Valid,
            &[22.0, -2.0, -16.0],
            &[-20.0, -14.0, 2.0],
        );
        check_conv(
            &[5.0, -5.0, 0.0, 4.0],
            &[-4.0, 3.0, -4.0],
            Mode::Full,
            &[-20.0, 35.0, -35.0, 4.0, 12.0, -16.0],
            &[-20.0, 35.0, -35.0, 4.0, 12.0, -16.0],
        );
        check_conv(
            &[5.0, -5.0, 0.0, 4.0],
            &[-4.0, 3.0, -4.0],
            Mode::Same,
            &[35.0, -35.0, 4.0, 12.0],
            &[35.0, -35.0, 4.0, 12.0],
        );
        check_conv(
            &[5.0, -5.0, 0.0, 4.0],
            &[-4.0, 3.0, -4.0],
            Mode::Valid,
            &[-35.0, 4.0],
            &[-35.0, 4.0],
        );
        check_conv(
            &[0.0, 3.0],
            &[-2.0, -2.0, -2.0, 2.0],
            Mode::Full,
            &[0.0, -6.0, -6.0, -6.0, 6.0],
            &[0.0, 6.0, -6.0, -6.0, -6.0],
        );
        check_conv(
            &[0.0, 3.0],
            &[-2.0, -2.0, -2.0, 2.0],
            Mode::Same,
            &[0.0, -6.0, -6.0, -6.0],
            &[6.0, -6.0, -6.0, -6.0],
        );
        check_conv(
            &[0.0, 3.0],
            &[-2.0, -2.0, -2.0, 2.0],
            Mode::Valid,
            &[-6.0, -6.0, -6.0],
            &[6.0, -6.0, -6.0],
        );
        check_conv(
            &[-3.0, 5.0],
            &[-1.0, 0.0, 0.0, 1.0, 1.0],
            Mode::Full,
            &[3.0, -5.0, 0.0, -3.0, 2.0, 5.0],
            &[-3.0, 2.0, 5.0, 0.0, 3.0, -5.0],
        );
        check_conv(
            &[-3.0, 5.0],
            &[-1.0, 0.0, 0.0, 1.0, 1.0],
            Mode::Same,
            &[3.0, -5.0, 0.0, -3.0, 2.0],
            &[2.0, 5.0, 0.0, 3.0, -5.0],
        );
        check_conv(
            &[-3.0, 5.0],
            &[-1.0, 0.0, 0.0, 1.0, 1.0],
            Mode::Valid,
            &[-5.0, 0.0, -3.0, 2.0],
            &[2.0, 5.0, 0.0, 3.0],
        );
        check_conv(
            &[0.0, 5.0, 3.0, 3.0, 2.0],
            &[1.0, -2.0],
            Mode::Full,
            &[0.0, 5.0, -7.0, -3.0, -4.0, -4.0],
            &[0.0, -10.0, -1.0, -3.0, -1.0, 2.0],
        );
        check_conv(
            &[0.0, 5.0, 3.0, 3.0, 2.0],
            &[1.0, -2.0],
            Mode::Same,
            &[0.0, 5.0, -7.0, -3.0, -4.0],
            &[0.0, -10.0, -1.0, -3.0, -1.0],
        );
        check_conv(
            &[0.0, 5.0, 3.0, 3.0, 2.0],
            &[1.0, -2.0],
            Mode::Valid,
            &[5.0, -7.0, -3.0, -4.0],
            &[-10.0, -1.0, -3.0, -1.0],
        );
        check_conv(
            &[5.0],
            &[0.0, -3.0, 4.0, -4.0],
            Mode::Full,
            &[0.0, -15.0, 20.0, -20.0],
            &[-20.0, 20.0, -15.0, 0.0],
        );
        check_conv(
            &[5.0],
            &[0.0, -3.0, 4.0, -4.0],
            Mode::Same,
            &[0.0, -15.0, 20.0, -20.0],
            &[-20.0, 20.0, -15.0, 0.0],
        );
        check_conv(
            &[5.0],
            &[0.0, -3.0, 4.0, -4.0],
            Mode::Valid,
            &[0.0, -15.0, 20.0, -20.0],
            &[-20.0, 20.0, -15.0, 0.0],
        );
        check_conv(
            &[4.0, 1.0, -4.0, -5.0],
            &[-1.0],
            Mode::Full,
            &[-4.0, -1.0, 4.0, 5.0],
            &[-4.0, -1.0, 4.0, 5.0],
        );
        check_conv(
            &[4.0, 1.0, -4.0, -5.0],
            &[-1.0],
            Mode::Same,
            &[-4.0, -1.0, 4.0, 5.0],
            &[-4.0, -1.0, 4.0, 5.0],
        );
        check_conv(
            &[4.0, 1.0, -4.0, -5.0],
            &[-1.0],
            Mode::Valid,
            &[-4.0, -1.0, 4.0, 5.0],
            &[-4.0, -1.0, 4.0, 5.0],
        );
        check_conv(&[-5.0], &[-4.0], Mode::Full, &[20.0], &[20.0]);
        check_conv(&[-5.0], &[-4.0], Mode::Same, &[20.0], &[20.0]);
        check_conv(&[-5.0], &[-4.0], Mode::Valid, &[20.0], &[20.0]);
        check_conv(
            &[0.0, 5.0, 0.0, 3.0, 5.0, 4.0],
            &[1.0, -1.0, 0.0, -3.0, 0.0, -1.0],
            Mode::Full,
            &[
                0.0, 5.0, -5.0, 3.0, -13.0, -1.0, -18.0, -15.0, -15.0, -5.0, -4.0,
            ],
            &[
                0.0, -5.0, 0.0, -18.0, -5.0, -18.0, -10.0, -15.0, -2.0, 1.0, 4.0,
            ],
        );
        check_conv(
            &[0.0, 5.0, 0.0, 3.0, 5.0, 4.0],
            &[1.0, -1.0, 0.0, -3.0, 0.0, -1.0],
            Mode::Same,
            &[-5.0, 3.0, -13.0, -1.0, -18.0, -15.0],
            &[0.0, -18.0, -5.0, -18.0, -10.0, -15.0],
        );
        check_conv(
            &[0.0, 5.0, 0.0, 3.0, 5.0, 4.0],
            &[1.0, -1.0, 0.0, -3.0, 0.0, -1.0],
            Mode::Valid,
            &[-1.0],
            &[-18.0],
        );
        check_conv(
            &[-3.0, 5.0, -5.0, -4.0, -3.0, 5.0, 2.0],
            &[4.0, -3.0, 2.0, -1.0],
            Mode::Full,
            &[-12.0, 29.0, -41.0, 12.0, -15.0, 26.0, -9.0, 7.0, -1.0, -2.0],
            &[3.0, -11.0, 24.0, -33.0, 30.0, -19.0, 1.0, -23.0, 14.0, 8.0],
        );
        check_conv(
            &[-3.0, 5.0, -5.0, -4.0, -3.0, 5.0, 2.0],
            &[4.0, -3.0, 2.0, -1.0],
            Mode::Same,
            &[29.0, -41.0, 12.0, -15.0, 26.0, -9.0, 7.0],
            &[-11.0, 24.0, -33.0, 30.0, -19.0, 1.0, -23.0],
        );
        check_conv(
            &[-3.0, 5.0, -5.0, -4.0, -3.0, 5.0, 2.0],
            &[4.0, -3.0, 2.0, -1.0],
            Mode::Valid,
            &[12.0, -15.0, 26.0, -9.0],
            &[-33.0, 30.0, -19.0, 1.0],
        );
        check_conv(
            &[0.0, -5.0, 1.0],
            &[4.0, 2.0, -4.0, 0.0, -3.0, 5.0, 4.0, -3.0],
            Mode::Full,
            &[0.0, -20.0, -6.0, 22.0, -4.0, 15.0, -28.0, -15.0, 19.0, -3.0],
            &[0.0, 15.0, -23.0, -21.0, 20.0, -3.0, 20.0, -14.0, -18.0, 4.0],
        );
        check_conv(
            &[0.0, -5.0, 1.0],
            &[4.0, 2.0, -4.0, 0.0, -3.0, 5.0, 4.0, -3.0],
            Mode::Same,
            &[-20.0, -6.0, 22.0, -4.0, 15.0, -28.0, -15.0, 19.0],
            &[15.0, -23.0, -21.0, 20.0, -3.0, 20.0, -14.0, -18.0],
        );
        check_conv(
            &[0.0, -5.0, 1.0],
            &[4.0, 2.0, -4.0, 0.0, -3.0, 5.0, 4.0, -3.0],
            Mode::Valid,
            &[-6.0, 22.0, -4.0, 15.0, -28.0, -15.0],
            &[-23.0, -21.0, 20.0, -3.0, 20.0, -14.0],
        );
    }

    #[test]
    fn convolve_integers_and_errors() {
        assert_eq!(
            convolve(&[1i64, 2, 3], &[0, 1, 1], Mode::Full),
            vec![0, 1, 3, 5, 3]
        );
        assert_eq!(
            correlate(&[1i64, 2, 3], &[0, 1, 1], Mode::Full),
            vec![1, 3, 5, 3, 0]
        );
        assert!(std::panic::catch_unwind(|| convolve::<f64>(&[], &[1.0], Mode::Full)).is_err());
        assert!(std::panic::catch_unwind(|| correlate::<f64>(&[1.0], &[], Mode::Same)).is_err());
    }

    #[test]
    fn diff_orders_and_axes() {
        check_diff(&[1.0, 4.0, 9.0, 16.0], &[4], 1, 0, &[3.0, 5.0, 7.0], &[3]);
        check_diff(&[1.0, 4.0, 9.0, 16.0], &[4], 2, 0, &[2.0, 2.0], &[2]);
        check_diff(&[1.0, 4.0, 9.0, 16.0], &[4], 4, 0, &[], &[0]);
        check_diff(&[1.0, 4.0, 9.0, 16.0], &[4], 7, 0, &[], &[0]);
        check_diff(
            &[1.0, 4.0, 9.0, 16.0],
            &[4],
            0,
            0,
            &[1.0, 4.0, 9.0, 16.0],
            &[4],
        );
        check_diff(&[5.0], &[1], 1, 0, &[], &[0]);
        check_diff(&[], &[0], 1, 0, &[], &[0]);
        check_diff(
            &[1.0, 4.0, 9.0, 16.0, 2.0, 3.0, 5.0, 8.0, 0.0, 0.0, 1.0, 1.0],
            &[3, 4],
            1,
            0,
            &[1.0, -1.0, -4.0, -8.0, -2.0, -3.0, -4.0, -7.0],
            &[2, 4],
        );
        check_diff(
            &[1.0, 4.0, 9.0, 16.0, 2.0, 3.0, 5.0, 8.0, 0.0, 0.0, 1.0, 1.0],
            &[3, 4],
            2,
            0,
            &[-3.0, -2.0, 0.0, 1.0],
            &[1, 4],
        );
        check_diff(
            &[1.0, 4.0, 9.0, 16.0, 2.0, 3.0, 5.0, 8.0, 0.0, 0.0, 1.0, 1.0],
            &[3, 4],
            1,
            1,
            &[3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
            &[3, 3],
        );
        check_diff(
            &[1.0, 4.0, 9.0, 16.0, 2.0, 3.0, 5.0, 8.0, 0.0, 0.0, 1.0, 1.0],
            &[3, 4],
            2,
            1,
            &[2.0, 2.0, 1.0, 1.0, 1.0, -1.0],
            &[3, 2],
        );
        check_diff(
            &[
                0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0,
                196.0, 225.0, 256.0, 289.0, 324.0, 361.0, 400.0, 441.0, 484.0, 529.0,
            ],
            &[2, 3, 4],
            1,
            0,
            &[
                144.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.0, 360.0, 384.0, 408.0,
            ],
            &[1, 3, 4],
        );
        check_diff(
            &[
                0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0,
                196.0, 225.0, 256.0, 289.0, 324.0, 361.0, 400.0, 441.0, 484.0, 529.0,
            ],
            &[2, 3, 4],
            1,
            1,
            &[
                16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 112.0, 120.0, 128.0, 136.0, 144.0,
                152.0, 160.0, 168.0,
            ],
            &[2, 2, 4],
        );
        check_diff(
            &[
                0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0,
                196.0, 225.0, 256.0, 289.0, 324.0, 361.0, 400.0, 441.0, 484.0, 529.0,
            ],
            &[2, 3, 4],
            1,
            2,
            &[
                1.0, 3.0, 5.0, 9.0, 11.0, 13.0, 17.0, 19.0, 21.0, 25.0, 27.0, 29.0, 33.0, 35.0,
                37.0, 41.0, 43.0, 45.0,
            ],
            &[2, 3, 3],
        );
        check_diff(
            &[
                0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0,
                196.0, 225.0, 256.0, 289.0, 324.0, 361.0, 400.0, 441.0, 484.0, 529.0,
            ],
            &[2, 3, 4],
            2,
            1,
            &[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0],
            &[2, 1, 4],
        );
        check_diff(
            &[
                0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0,
                196.0, 225.0, 256.0, 289.0, 324.0, 361.0, 400.0, 441.0, 484.0, 529.0,
            ],
            &[2, 3, 4],
            5,
            1,
            &[],
            &[2, 0, 4],
        );
    }

    #[test]
    fn gradient_1d() {
        let y = [1.0, 2.0, 4.0, 8.0, 3.0, 7.0];
        let u = |h, eo| gradient(&y, &[6], 0, Spacing::Uniform(h), eo);
        assert_close(&u(1.0, 1), &[1.0, 1.5, 3.0, -0.5, -0.5, 4.0]);
        assert_close(&u(0.5, 1), &[2.0, 3.0, 6.0, -1.0, -1.0, 8.0]);
        assert_close(&u(1.0, 2), &[0.5, 1.5, 3.0, -0.5, -0.5, 8.5]);
        assert_close(
            &u(2.5, 2),
            &[0.19999999999999996, 0.6, 1.2, -0.2, -0.2, 3.4],
        );
        let x = [0.0, 1.0, 3.0, 7.0, 8.0, 12.0];
        assert_close(
            &gradient(&y, &[6], 0, Spacing::Coords(&x), 1),
            &[1.0, 1.0, 1.0, -3.8, -3.8000000000000003, 1.0],
        );
        assert_close(
            &gradient(&y, &[6], 0, Spacing::Coords(&x), 2),
            &[1.0, 1.0, 1.0, -3.8, -3.8000000000000003, 5.800000000000001],
        );
        // two samples: only first-order edges are possible
        assert_close(
            &gradient(&[1.0, 4.0], &[2], 0, Spacing::Uniform(1.0), 1),
            &[3.0, 3.0],
        );
        assert!(
            std::panic::catch_unwind(|| gradient(&[1.0, 4.0], &[2], 0, Spacing::Uniform(1.0), 2))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| gradient(&[1.0], &[1], 0, Spacing::Uniform(1.0), 1))
                .is_err()
        );
    }

    #[test]
    fn gradient_axis() {
        let y = [1.0, 2.0, 4.0, 8.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert_close(
            &gradient(&y, &[3, 4], 0, Spacing::Uniform(1.0), 1),
            &[
                2.0, -1.0, 0.0, -7.0, 2.0, 3.5, -1.0, -1.0, 2.0, 8.0, -2.0, 5.0,
            ],
        );
        assert_close(
            &gradient(&y, &[3, 4], 1, Spacing::Uniform(1.0), 1),
            &[
                1.0, 1.5, 3.0, 4.0, -2.0, 0.5, 0.0, -3.0, 4.0, -1.5, -1.5, 4.0,
            ],
        );
        assert_close(
            &gradient(&y, &[3, 4], 1, Spacing::Uniform(0.5), 2),
            &[
                1.0, 3.0, 6.0, 10.0, -9.0, 1.0, 0.0, -12.0, 19.0, -3.0, -3.0, 19.0,
            ],
        );
        assert_close(
            &gradient(&y, &[3, 4], 0, Spacing::Coords(&[0.0, 1.0, 4.0]), 1),
            &[
                2.0,
                -1.0,
                0.0,
                -7.0,
                1.6666666666666665,
                -0.08333333333333337,
                -0.16666666666666682,
                -4.833333333333333,
                0.6666666666666666,
                2.6666666666666665,
                -0.6666666666666666,
                1.6666666666666667,
            ],
        );
    }

    #[test]
    fn interp_basic_and_fill() {
        let xp = [1.0, 2.0, 4.0, 8.0];
        let fp = [3.0, 5.0, -1.0, 0.0];
        let x = [
            f64::NAN,
            0.0,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            7.0,
            8.0,
            9.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ];
        assert_close(
            &interp(&x, &xp, &fp, None, None),
            &[
                f64::NAN,
                3.0,
                3.0,
                4.0,
                5.0,
                2.0,
                -1.0,
                -0.25,
                0.0,
                0.0,
                3.0,
                0.0,
            ],
        );
        assert_close(
            &interp(&x, &xp, &fp, Some(-9.0), Some(99.0)),
            &[
                f64::NAN,
                -9.0,
                3.0,
                4.0,
                5.0,
                2.0,
                -1.0,
                -0.25,
                0.0,
                99.0,
                -9.0,
                99.0,
            ],
        );
        assert_close(
            &interp(&x, &xp, &fp, Some(f64::NAN), None),
            &[
                f64::NAN,
                f64::NAN,
                3.0,
                4.0,
                5.0,
                2.0,
                -1.0,
                -0.25,
                0.0,
                0.0,
                f64::NAN,
                0.0,
            ],
        );
    }

    #[test]
    fn interp_edge_cases() {
        // single sample point: NaN maps to fp[0]
        assert_close(
            &interp(&[f64::NAN, 0.0, 1.0, 5.0], &[1.0], &[3.0], None, None),
            &[3.0, 3.0, 3.0, 3.0],
        );
        assert_close(
            &interp(&[0.0, 5.0], &[1.0], &[3.0], Some(-1.0), Some(7.0)),
            &[-1.0, 7.0],
        );
        // infinities among the samples
        assert_close(
            &interp(
                &[2.5],
                &[1.0, 2.0, 3.0],
                &[f64::INFINITY, 5.0, 6.0],
                None,
                None,
            ),
            &[5.5],
        );
        assert_close(
            &interp(
                &[1.5],
                &[1.0, 2.0, 3.0],
                &[f64::INFINITY, f64::INFINITY, 6.0],
                None,
                None,
            ),
            &[f64::INFINITY],
        );
        assert!(interp(&[], &[1.0, 2.0], &[1.0, 2.0], None, None).is_empty());
        assert!(std::panic::catch_unwind(|| interp(&[1.0], &[], &[], None, None)).is_err());
        assert!(
            std::panic::catch_unwind(|| interp(&[1.0], &[1.0, 2.0], &[1.0], None, None)).is_err()
        );
    }

    #[test]
    fn trapezoid_cases() {
        assert_close(
            &trapezoid(&[1.0, 2.0, 3.0], &[3], 0, Spacing::Uniform(1.0)),
            &[4.0],
        );
        assert_close(
            &trapezoid(&[1.0, 2.0, 3.0], &[3], 0, Spacing::Uniform(0.5)),
            &[2.0],
        );
        assert_close(
            &trapezoid(&[1.0, 2.0, 3.0], &[3], 0, Spacing::Coords(&[0.0, 1.0, 3.0])),
            &[6.5],
        );
        assert_eq!(trapezoid(&[5.0], &[1], 0, Spacing::Uniform(1.0)), vec![0.0]);
        assert_eq!(trapezoid(&[], &[0], 0, Spacing::Uniform(1.0)), vec![0.0]);
        let y = [1.0, 2.0, 3.0, 5.0, 0.0, 4.0, 4.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        assert_close(
            &trapezoid(&y, &[3, 4], 1, Spacing::Uniform(1.0)),
            &[8.0, 8.5, 6.0],
        );
        assert_close(
            &trapezoid(&y, &[3, 4], 0, Spacing::Uniform(2.0)),
            &[3.0, 12.0, 13.0, 9.0],
        );
        assert_close(
            &trapezoid(&y, &[3, 4], 1, Spacing::Coords(&[0.0, 0.5, 2.0, 2.25])),
            &[5.5, 7.625, 4.5],
        );
        assert!(
            std::panic::catch_unwind(|| trapezoid(&[1.0, 2.0], &[2], 0, Spacing::Coords(&[0.0])))
                .is_err()
        );
    }

    #[test]
    fn cross_cases() {
        let a3 = [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, -2.0, 5.0, 0.5];
        let b3 = [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 3.0, 3.0, -1.0];
        assert_close(
            &cross(&a3, 3, &b3, 3),
            &[-3.0, 6.0, -3.0, 0.0, 1.0, 0.0, -6.5, -0.5, -21.0],
        );
        assert_close(&cross(&a3[..3], 3, &b3[..3], 3), &[-3.0, 6.0, -3.0]);
        let a2 = [1.0, 2.0, 3.0, -4.0, 0.0, 5.0];
        let b2 = [3.0, 4.0, 1.0, 1.0, 2.0, 0.0];
        assert_close(&cross(&a2, 2, &b2, 2), &[-2.0, 7.0, -10.0]);
        assert_close(
            &cross(&a2, 2, &b3, 3),
            &[12.0, -6.0, -3.0, -0.0, -0.0, 4.0, -5.0, 0.0, -15.0],
        );
        assert_close(
            &cross(&a3, 3, &b2, 2),
            &[-12.0, 9.0, -2.0, -1.0, 1.0, 0.0, -0.0, 1.0, -10.0],
        );
        assert_eq!(cross(&[1i64, 2], 2, &[3, 4], 2), vec![-2]);
        assert_eq!(cross(&[1i64, 0, 0], 3, &[0, 1, 0], 3), vec![0, 0, 1]);
        assert!(cross::<f64>(&[], 3, &[], 3).is_empty());
        assert!(
            std::panic::catch_unwind(|| cross(&[1.0, 2.0, 3.0, 4.0], 4, &[1.0, 2.0, 3.0, 4.0], 4))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| cross(
                &[1.0, 2.0, 3.0],
                3,
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                3
            ))
            .is_err()
        );
    }

    #[test]
    fn kron_cases() {
        let check =
            |a: &[f64], sa: &[usize], b: &[f64], sb: &[usize], want: &[f64], ws: &[usize]| {
                let (got, shape) = kron(a, sa, b, sb);
                assert_eq!(shape, ws);
                assert_close(&got, want);
            };
        check(
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            &[0.0, 5.0, 1.0, 6.0, 7.0, 2.0],
            &[2, 3],
            &[
                0.0, 5.0, 1.0, 0.0, 10.0, 2.0, 6.0, 7.0, 2.0, 12.0, 14.0, 4.0, 0.0, 15.0, 3.0, 0.0,
                20.0, 4.0, 18.0, 21.0, 6.0, 24.0, 28.0, 8.0,
            ],
            &[4, 6],
        );
        check(
            &[0.0, 5.0, 1.0, 6.0, 7.0, 2.0],
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            &[
                0.0, 0.0, 5.0, 10.0, 1.0, 2.0, 0.0, 0.0, 15.0, 20.0, 3.0, 4.0, 6.0, 12.0, 7.0,
                14.0, 2.0, 4.0, 18.0, 24.0, 21.0, 28.0, 6.0, 8.0,
            ],
            &[4, 6],
        );
        check(
            &[1.0, 2.0, 3.0],
            &[3],
            &[1.0, 10.0],
            &[2],
            &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            &[6],
        );
        // different ndim: the shorter shape is padded with leading ones
        check(
            &[1.0, 2.0],
            &[2],
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            &[1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 6.0, 8.0],
            &[2, 4],
        );
        check(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &[2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            &[
                0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 2.0, 4.0, 3.0, 6.0, 6.0, 8.0, 9.0, 12.0,
                4.0, 8.0, 5.0, 10.0, 12.0, 16.0, 15.0, 20.0, 6.0, 12.0, 7.0, 14.0, 18.0, 24.0,
                21.0, 28.0,
            ],
            &[2, 4, 4],
        );
        check(
            &[3.0],
            &[],
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            &[3.0, 6.0, 9.0, 12.0],
            &[2, 2],
        );
        check(&[2.0], &[], &[5.0], &[], &[10.0], &[]);
        let (e, s) = kron(&[] as &[f64], &[0], &[1.0, 2.0], &[2]);
        assert!(e.is_empty());
        assert_eq!(s, vec![0]);
        let (g, _) = kron(&[1i64, 2], &[2], &[1, 1], &[2]);
        assert_eq!(g, vec![1, 1, 2, 2]);
    }
}
