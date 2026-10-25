//! Statistics kernels: `cov`, `corrcoef`, `histogram`, `histogram_bin_edges`, `bincount`,
//! `digitize`, `average`, `quantile`, `nanquantile`.
//!
//! Buffers are C-ordered `f64` slices. See the module docs of `extra` for conventions.

use super::{axis_split, cmp_nan_last, is_nan_like, searchsorted_left, searchsorted_right};

/// NumPy's pairwise summation (blocks of 8 accumulators below 128 elements).
pub(crate) fn pairwise_sum(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 8 {
        let mut s = 0.0;
        for v in x {
            s += v;
        }
        s
    } else if n <= 128 {
        let mut r = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]];
        let mut i = 8;
        while i + 8 <= n {
            for (k, rk) in r.iter_mut().enumerate() {
                *rk += x[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += x[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&x[..n2]) + pairwise_sum(&x[n2..])
    }
}

/// Mean of a slice (NaN for an empty slice, like `np.mean`).
pub(crate) fn mean(x: &[f64]) -> f64 {
    pairwise_sum(x) / x.len() as f64
}

/// `np.cov(m, bias=bias, ddof=ddof)` for `m` of shape `(nvars, nobs)`, one variable per row.
///
/// Returns the `nvars x nvars` covariance matrix. `ddof` defaults to 1 (0 when `bias`).
/// When `nobs - ddof <= 0` the result is `inf`/`NaN`, as in NumPy (which also warns).
pub(crate) fn cov(
    x: &[f64],
    nvars: usize,
    nobs: usize,
    bias: bool,
    ddof: Option<usize>,
) -> Vec<f64> {
    assert_eq!(x.len(), nvars * nobs, "cov: buffer does not match shape");
    let ddof = ddof.unwrap_or(usize::from(!bias));
    let fact = if nobs > ddof {
        (nobs - ddof) as f64
    } else {
        0.0
    };
    let mut xc = x.to_vec();
    if nobs > 0 {
        for row in xc.chunks_mut(nobs) {
            let m = mean(row);
            for v in row.iter_mut() {
                *v -= m;
            }
        }
    }
    let mut out = vec![0.0; nvars * nvars];
    for i in 0..nvars {
        for j in i..nvars {
            let (ri, rj) = (&xc[i * nobs..(i + 1) * nobs], &xc[j * nobs..(j + 1) * nobs]);
            let prod: Vec<f64> = ri.iter().zip(rj).map(|(a, b)| a * b).collect();
            let c = pairwise_sum(&prod) / fact;
            out[i * nvars + j] = c;
            out[j * nvars + i] = c;
        }
    }
    out
}

/// `np.corrcoef(m)` for `m` of shape `(nvars, nobs)`; entries are clipped to `[-1, 1]`.
pub(crate) fn corrcoef(x: &[f64], nvars: usize, nobs: usize) -> Vec<f64> {
    let mut c = cov(x, nvars, nobs, false, None);
    let sd: Vec<f64> = (0..nvars).map(|i| c[i * nvars + i].sqrt()).collect();
    for i in 0..nvars {
        for j in 0..nvars {
            let v = c[i * nvars + j] / sd[i] / sd[j];
            c[i * nvars + j] = if v.is_nan() { v } else { v.clamp(-1.0, 1.0) };
        }
    }
    c
}

/// The `bins` argument of `histogram`: a number of equal-width bins or explicit edges.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Bins<'a> {
    Count(usize),
    Edges(&'a [f64]),
}

/// `np.linspace(start, stop, num)` (endpoint included, last value exactly `stop`).
pub(crate) fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    match num {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let div = (num - 1) as f64;
            let delta = stop - start;
            let step = delta / div;
            let mut y: Vec<f64> = (0..num)
                .map(|i| {
                    let i = i as f64;
                    if step == 0.0 {
                        i / div * delta + start
                    } else {
                        i * step + start
                    }
                })
                .collect();
            y[num - 1] = stop;
            y
        }
    }
}

fn check_edges(edges: &[f64]) {
    if edges.len() < 2 {
        panic!("`bins` must have at least 2 edges when an array");
    }
    if edges
        .windows(2)
        .any(|w| w[0] > w[1] || w[0].is_nan() || w[1].is_nan())
    {
        panic!("`bins` must increase monotonically, when an array");
    }
}

/// First and last edge for equal-width bins (NumPy's `_get_outer_edges`).
fn outer_edges(x: &[f64], range: Option<(f64, f64)>) -> (f64, f64) {
    let (first, last) = match range {
        Some((lo, hi)) => {
            if lo > hi {
                panic!("max must be larger than min in range parameter.");
            }
            if !(lo.is_finite() && hi.is_finite()) {
                panic!("supplied range of [{lo}, {hi}] is not finite");
            }
            (lo, hi)
        }
        None if x.is_empty() => (0.0, 1.0),
        None => {
            let mut lo = x[0];
            let mut hi = x[0];
            for &v in x {
                if v.is_nan() {
                    lo = v;
                    hi = v;
                    break;
                }
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if !(lo.is_finite() && hi.is_finite()) {
                panic!("autodetected range of [{lo}, {hi}] is not finite");
            }
            (lo, hi)
        }
    };
    if first == last {
        (first - 0.5, last + 0.5)
    } else {
        (first, last)
    }
}

/// `np.histogram_bin_edges(x, bins, range)` for a bin count or explicit edges.
pub(crate) fn histogram_bin_edges(
    x: &[f64],
    bins: Bins<'_>,
    range: Option<(f64, f64)>,
) -> Vec<f64> {
    match bins {
        Bins::Count(n) => {
            if n < 1 {
                panic!("`bins` must be positive, when an integer");
            }
            let (first, last) = outer_edges(x, range);
            linspace(first, last, n + 1)
        }
        Bins::Edges(e) => {
            check_edges(e);
            e.to_vec()
        }
    }
}

/// `np.histogram(x, bins, range, density, weights)`; returns `(counts, bin_edges)`.
///
/// The last bin is closed on the right, all others are half open. NaN values are ignored.
/// Counts are returned as `f64` (integer valued when unweighted).
pub(crate) fn histogram(
    x: &[f64],
    bins: Bins<'_>,
    range: Option<(f64, f64)>,
    density: bool,
    weights: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    if let Some(w) = weights
        && w.len() != x.len()
    {
        panic!("weights should have the same shape as a.");
    }
    let edges = histogram_bin_edges(x, bins, range);
    let nbins = edges.len() - 1;
    let mut counts = vec![0.0; nbins];
    let wt = |i: usize| weights.map_or(1.0, |w| w[i]);
    match bins {
        Bins::Count(_) => {
            let (first, last) = (edges[0], edges[nbins]);
            let denom = last - first;
            let nb = nbins as f64;
            for (i, &v) in x.iter().enumerate() {
                if !(v >= first && v <= last) {
                    continue;
                }
                let mut idx = ((v - first) / denom * nb) as usize;
                if idx == nbins {
                    idx -= 1;
                }
                // Fix rounding so that edges[idx] <= v < edges[idx + 1] exactly.
                if v < edges[idx] {
                    idx -= 1;
                }
                if v >= edges[idx + 1] && idx != nbins - 1 {
                    idx += 1;
                }
                counts[idx] += wt(i);
            }
        }
        Bins::Edges(_) => {
            for (i, &v) in x.iter().enumerate() {
                if v.is_nan() || v < edges[0] || v > edges[nbins] {
                    continue;
                }
                let idx = searchsorted_right(&edges, &v).min(nbins);
                counts[idx - 1] += wt(i);
            }
        }
    }
    if density {
        let total = pairwise_sum(&counts);
        for (c, w) in counts.iter_mut().zip(edges.windows(2)) {
            *c = *c / (w[1] - w[0]) / total;
        }
    }
    (counts, edges)
}

fn check_bincount(x: &[i64], minlength: usize) -> usize {
    if x.iter().any(|&v| v < 0) {
        panic!("'list' argument must have no negative elements");
    }
    let m = x.iter().max().map_or(0, |&m| m as usize + 1);
    m.max(minlength)
}

/// `np.bincount(x, minlength=minlength)`.
pub(crate) fn bincount(x: &[i64], minlength: usize) -> Vec<usize> {
    let mut out = vec![0; check_bincount(x, minlength)];
    for &v in x {
        out[v as usize] += 1;
    }
    out
}

/// `np.bincount(x, weights=weights, minlength=minlength)`.
pub(crate) fn bincount_weighted(x: &[i64], weights: &[f64], minlength: usize) -> Vec<f64> {
    if weights.len() != x.len() {
        panic!("The weights and list don't have the same length.");
    }
    let mut out = vec![0.0; check_bincount(x, minlength)];
    for (&v, &w) in x.iter().zip(weights) {
        out[v as usize] += w;
    }
    out
}

/// `np.digitize(x, bins, right)`: index of the bin each value falls into.
///
/// `bins` must be monotonic (all-equal counts as increasing). NaN goes past the last bin.
pub(crate) fn digitize(x: &[f64], bins: &[f64], right: bool) -> Vec<usize> {
    let increasing = bins.windows(2).all(|w| w[0] <= w[1]);
    let decreasing = bins.windows(2).all(|w| w[0] >= w[1]);
    if !increasing && !decreasing {
        panic!("bins must be monotonically increasing or decreasing");
    }
    if increasing {
        x.iter()
            .map(|v| {
                if right {
                    searchsorted_left(bins, v)
                } else {
                    searchsorted_right(bins, v)
                }
            })
            .collect()
    } else {
        let rev: Vec<f64> = bins.iter().rev().copied().collect();
        x.iter()
            .map(|v| {
                let pos = if right {
                    searchsorted_left(&rev, v)
                } else {
                    searchsorted_right(&rev, v)
                };
                bins.len() - pos
            })
            .collect()
    }
}

/// `np.average(x, weights=weights)` over the whole buffer.
///
/// Panics with NumPy's message when the weights sum to zero.
pub(crate) fn average(x: &[f64], weights: Option<&[f64]>) -> f64 {
    match weights {
        None => mean(x),
        Some(w) => {
            if w.len() != x.len() {
                panic!("Axis must be specified when shapes of a and weights differ.");
            }
            let scl = pairwise_sum(w);
            if scl == 0.0 {
                panic!("Weights sum to zero, can't be normalized");
            }
            let prod: Vec<f64> = x.iter().zip(w).map(|(a, b)| a * b).collect();
            pairwise_sum(&prod) / scl
        }
    }
}

/// `np.average(x, axis=axis, weights=weights)`; output has `axis` removed.
///
/// `weights` is either the same shape as `x` or 1-d with length `shape[axis]`.
pub(crate) fn average_axis(
    x: &[f64],
    shape: &[usize],
    axis: usize,
    weights: Option<&[f64]>,
) -> Vec<f64> {
    let (outer, n, inner) = axis_split(shape, axis);
    if let Some(w) = weights
        && w.len() != x.len()
        && w.len() != n
    {
        panic!("Length of weights not compatible with specified axis.");
    }
    let full = weights.is_some_and(|w| w.len() == x.len());
    let mut out = Vec::with_capacity(outer * inner);
    let mut lane = vec![0.0; n];
    let mut wl = vec![0.0; n];
    for o in 0..outer {
        for i in 0..inner {
            for k in 0..n {
                lane[k] = x[(o * n + k) * inner + i];
                if let Some(w) = weights {
                    wl[k] = if full {
                        w[(o * n + k) * inner + i]
                    } else {
                        w[k]
                    };
                }
            }
            out.push(average(&lane, weights.map(|_| wl.as_slice())));
        }
    }
    out
}

fn check_q(q: f64) {
    if !(0.0..=1.0).contains(&q) {
        panic!("Quantiles must be in the range [0, 1]");
    }
}

/// NumPy's linear-method interpolation on a NaN-free, non-empty sorted lane.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let idx = (n - 1) as f64 * q;
    let (a, b, gamma) = if idx >= (n - 1) as f64 {
        (sorted[n - 1], sorted[n - 1], 1.0)
    } else {
        let lo = idx.floor();
        (sorted[lo as usize], sorted[lo as usize + 1], idx - lo)
    };
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

fn quantile_lane(lane: &[f64], q: f64, skip_nan: bool) -> f64 {
    let mut v: Vec<f64> = lane.to_vec();
    v.sort_by(cmp_nan_last);
    let nans = v.iter().rev().take_while(|e| is_nan_like(*e)).count();
    if skip_nan {
        v.truncate(v.len() - nans);
        if v.is_empty() {
            return f64::NAN;
        }
    } else {
        if v.is_empty() {
            panic!("index -1 is out of bounds for axis 0 with size 0");
        }
        if nans > 0 {
            return f64::NAN;
        }
    }
    quantile_sorted(&v, q)
}

/// `np.quantile(x, q)` (method `"linear"`) over the whole buffer. NaN in `x` gives NaN.
pub(crate) fn quantile(x: &[f64], q: f64) -> f64 {
    check_q(q);
    quantile_lane(x, q, false)
}

/// `np.nanquantile(x, q)`: NaNs are ignored; all-NaN or empty input gives NaN.
pub(crate) fn nanquantile(x: &[f64], q: f64) -> f64 {
    check_q(q);
    quantile_lane(x, q, true)
}

fn quantile_axis_impl(x: &[f64], shape: &[usize], axis: usize, q: f64, skip_nan: bool) -> Vec<f64> {
    check_q(q);
    let (outer, n, inner) = axis_split(shape, axis);
    let mut out = Vec::with_capacity(outer * inner);
    let mut lane = vec![0.0; n];
    for o in 0..outer {
        for i in 0..inner {
            for (k, l) in lane.iter_mut().enumerate() {
                *l = x[(o * n + k) * inner + i];
            }
            out.push(quantile_lane(&lane, q, skip_nan));
        }
    }
    out
}

/// `np.quantile(x, q, axis=axis)` for a scalar `q`; output has `axis` removed.
pub(crate) fn quantile_axis(x: &[f64], shape: &[usize], axis: usize, q: f64) -> Vec<f64> {
    quantile_axis_impl(x, shape, axis, q, false)
}

/// `np.nanquantile(x, q, axis=axis)` for a scalar `q`; output has `axis` removed.
pub(crate) fn nanquantile_axis(x: &[f64], shape: &[usize], axis: usize, q: f64) -> Vec<f64> {
    quantile_axis_impl(x, shape, axis, q, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::assert_close;

    #[test]
    fn cov_default_rows_are_variables() {
        let x = [
            2.1, 4.5, 3.3, 8.0, 1.2, 1.0, -2.0, 0.5, 3.5, 2.0, 10.0, 9.0, 7.5, 8.25, 12.0,
        ];
        assert_close(
            &cov(&x, 3, 5, false, None),
            &[
                7.007,
                1.5125000000000002,
                -2.9837499999999997,
                1.5125000000000002,
                4.125,
                0.46875,
                -2.9837499999999997,
                0.46875,
                3.0500000000000003,
            ],
        );
    }

    #[test]
    fn cov_bias_and_ddof() {
        let x = [
            2.1, 4.5, 3.3, 8.0, 1.2, 1.0, -2.0, 0.5, 3.5, 2.0, 10.0, 9.0, 7.5, 8.25, 12.0,
        ];
        assert_close(
            &cov(&x, 3, 5, true, None),
            &[
                5.6056,
                1.2100000000000002,
                -2.387,
                1.2100000000000002,
                3.3000000000000003,
                0.375,
                -2.387,
                0.375,
                2.4400000000000004,
            ],
        );
        assert_close(
            &cov(&x, 3, 5, false, Some(0)),
            &[
                5.6056,
                1.2100000000000002,
                -2.387,
                1.2100000000000002,
                3.3000000000000003,
                0.375,
                -2.387,
                0.375,
                2.4400000000000004,
            ],
        );
        assert_close(
            &cov(&x, 3, 5, false, Some(2)),
            &[
                9.342666666666666,
                2.0166666666666666,
                -3.9783333333333326,
                2.0166666666666666,
                5.5,
                0.625,
                -3.9783333333333326,
                0.625,
                4.066666666666666,
            ],
        );
        // explicit ddof wins over bias, like NumPy
        assert_close(
            &cov(&x, 3, 5, true, Some(2)),
            &[
                9.342666666666666,
                2.0166666666666666,
                -3.9783333333333326,
                2.0166666666666666,
                5.5,
                0.625,
                -3.9783333333333326,
                0.625,
                4.066666666666666,
            ],
        );
    }

    #[test]
    fn cov_degenerate() {
        // one observation: 0/0
        assert!(
            cov(&[1.0, 2.0], 2, 1, false, None)
                .iter()
                .all(|v| v.is_nan())
        );
        // ddof >= nobs: division by zero gives inf, like NumPy
        let c = cov(&[1.0, 2.0, 3.0, 5.0], 2, 2, false, Some(2));
        assert_eq!(c, vec![f64::INFINITY; 4]);
        // single variable
        assert_close(
            &cov(&[1.0, 2.0, 3.0, 4.0], 1, 4, false, None),
            &[1.6666666666666665],
        );
        assert!(cov(&[], 0, 0, false, None).is_empty());
    }

    #[test]
    fn corrcoef_matches_numpy() {
        let x = [
            2.1, 4.5, 3.3, 8.0, 1.2, 1.0, -2.0, 0.5, 3.5, 2.0, 10.0, 9.0, 7.5, 8.25, 12.0,
        ];
        assert_close(
            &corrcoef(&x, 3, 5),
            &[
                1.0,
                0.28133084337263436,
                -0.6454259959649434,
                0.2813308433726344,
                0.9999999999999998,
                0.13215371756808084,
                -0.6454259959649434,
                0.13215371756808084,
                1.0,
            ],
        );
        let neg = corrcoef(&[1.0, 2.0, 3.0, 3.0, 2.0, 1.0], 2, 3);
        assert_eq!(neg, vec![1.0, -1.0, -1.0, 1.0]);
        // constant row: NaN off the diagonal and for itself, like NumPy
        let c = corrcoef(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], 2, 3);
        assert_eq!(c[0], 1.0);
        assert!(c[1].is_nan() && c[2].is_nan() && c[3].is_nan());
    }

    #[test]
    fn histogram_uniform_bins_rounding_at_edges() {
        let x = [0.3, 0.7, 0.1, 0.9, 1.0, 0.5, 0.6];
        let (c, e) = histogram(&x, Bins::Count(10), Some((0.0, 1.0)), false, None);
        assert_eq!(c, [0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0]);
        assert_close(
            &e,
            &[
                0.0,
                0.1,
                0.2,
                0.30000000000000004,
                0.4,
                0.5,
                0.6000000000000001,
                0.7000000000000001,
                0.8,
                0.9,
                1.0,
            ],
        );
    }

    #[test]
    fn histogram_with_range() {
        let (c, e) = histogram(
            &[1.0, 2.0, 1.0],
            Bins::Count(4),
            Some((0.0, 3.0)),
            false,
            None,
        );
        assert_eq!(c, [0.0, 2.0, 1.0, 0.0]);
        assert_close(&e, &[0.0, 0.75, 1.5, 2.25, 3.0]);
        // values outside the range are dropped, the right edge is included
        let (c, _) = histogram(
            &[-1.0, 0.0, 3.0, 3.5],
            Bins::Count(3),
            Some((0.0, 3.0)),
            false,
            None,
        );
        assert_eq!(c, [1.0, 0.0, 1.0]);
    }

    #[test]
    fn histogram_auto_range_closed_last_bin() {
        let x = [2.5, 3.5, 3.5, 1.0, 9.75, 4.0, 4.0, 7.0];
        let (c, e) = histogram(&x, Bins::Count(4), None, false, None);
        assert_eq!(c, [2.0, 4.0, 1.0, 1.0]);
        assert_close(&e, &[1.0, 3.1875, 5.375, 7.5625, 9.75]);
    }

    #[test]
    fn histogram_empty_and_constant() {
        let (c, e) = histogram(&[], Bins::Count(3), None, false, None);
        assert_eq!(c, [0.0, 0.0, 0.0]);
        assert_close(&e, &[0.0, 0.3333333333333333, 0.6666666666666666, 1.0]);
        let (c, e) = histogram(&[5.0, 5.0], Bins::Count(2), None, false, None);
        assert_eq!(c, [0.0, 2.0]);
        assert_close(&e, &[4.5, 5.0, 5.5]);
        // length-1
        let (c, e) = histogram(&[2.0], Bins::Count(1), None, false, None);
        assert_eq!(c, [1.0]);
        assert_close(&e, &[1.5, 2.5]);
    }

    #[test]
    fn histogram_explicit_edges() {
        let (c, e) = histogram(
            &[1.0, 2.0, 3.0, 4.0],
            Bins::Edges(&[1.0, 2.0, 4.0]),
            None,
            false,
            None,
        );
        assert_eq!(c, [1.0, 3.0]);
        assert_eq!(e, [1.0, 2.0, 4.0]);
        // NaN and out-of-range values are ignored
        let (c, _) = histogram(
            &[f64::NAN, 1.0, 2.0, 5.0, -1.0],
            Bins::Edges(&[0.0, 1.0, 2.0, 3.0]),
            None,
            false,
            None,
        );
        assert_eq!(c, [0.0, 1.0, 1.0]);
        // zero-width bins
        let (c, _) = histogram(
            &[1.0, 1.0, 2.0, 2.0],
            Bins::Edges(&[1.0, 1.0, 2.0, 2.0]),
            None,
            false,
            None,
        );
        assert_eq!(c, [0.0, 2.0, 2.0]);
    }

    #[test]
    fn histogram_density_and_weights() {
        let (c, _) = histogram(&[1.0, 2.0, 3.0, 4.0], Bins::Count(3), None, true, None);
        assert_close(&c, &[0.25, 0.25, 0.5]);
        let (c, e) = histogram(
            &[0.5, 1.5, 1.5],
            Bins::Count(2),
            None,
            false,
            Some(&[1.0, 2.0, 3.0]),
        );
        assert_eq!(c, [1.0, 5.0]);
        assert_close(&e, &[0.5, 1.0, 1.5]);
        let (c, _) = histogram(
            &[0.5, 1.5, 1.5],
            Bins::Edges(&[0.0, 1.0, 2.0]),
            None,
            true,
            Some(&[1.0, 2.0, 3.0]),
        );
        assert_close(&c, &[0.16666666666666666, 0.8333333333333334]);
    }

    #[test]
    fn histogram_bin_edges_kernel() {
        let e = histogram_bin_edges(&[1.0, 2.0, 7.0], Bins::Count(3), None);
        assert_close(&e, &[1.0, 3.0, 5.0, 7.0]);
        let e = histogram_bin_edges(&[1.0, 2.0, 7.0], Bins::Count(2), Some((0.0, 10.0)));
        assert_close(&e, &[0.0, 5.0, 10.0]);
        let e = histogram_bin_edges(&[], Bins::Count(4), None);
        assert_close(&e, &[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(
            histogram_bin_edges(&[1.0], Bins::Edges(&[0.0, 5.0]), None),
            vec![0.0, 5.0]
        );
    }

    #[test]
    fn histogram_errors() {
        let r = std::panic::catch_unwind(|| {
            histogram(&[1.0, f64::NAN], Bins::Count(2), None, false, None)
        });
        assert!(r.is_err());
        let r = std::panic::catch_unwind(|| {
            histogram(&[1.0], Bins::Edges(&[0.0, 2.0, 1.0]), None, false, None)
        });
        assert!(r.is_err());
        let r = std::panic::catch_unwind(|| histogram(&[1.0], Bins::Count(0), None, false, None));
        assert!(r.is_err());
        let r = std::panic::catch_unwind(|| {
            histogram(&[1.0], Bins::Count(2), Some((2.0, 1.0)), false, None)
        });
        assert!(r.is_err());
    }

    #[test]
    fn bincount_cases() {
        assert_eq!(bincount(&[1, 1, 3, 0, 3, 3], 0), vec![1, 2, 0, 3]);
        assert_eq!(bincount(&[], 0), Vec::<usize>::new());
        assert_eq!(bincount(&[], 3), vec![0, 0, 0]);
        assert_eq!(bincount(&[1], 4), vec![0, 1, 0, 0]);
        assert_eq!(bincount(&[5, 0], 2), vec![1, 0, 0, 0, 0, 1]);
        assert_eq!(
            bincount_weighted(&[0, 2, 2], &[0.5, 1.0, 1.5], 0),
            vec![0.5, 0.0, 2.5]
        );
        assert_eq!(bincount_weighted(&[1], &[2.0], 3), vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn bincount_errors() {
        assert!(std::panic::catch_unwind(|| bincount(&[1, -1], 0)).is_err());
        assert!(std::panic::catch_unwind(|| bincount_weighted(&[1, 2], &[1.0], 0)).is_err());
    }

    #[test]
    fn digitize_increasing() {
        let x = [f64::NAN, 1.0, 0.0, 5.0, 2.0, 3.0, 1.5];
        let bins = [1.0, 2.0, 3.0];
        assert_eq!(digitize(&x, &bins, false), vec![3, 1, 0, 3, 2, 3, 1]);
        assert_eq!(digitize(&x, &bins, true), vec![3, 0, 0, 3, 1, 2, 1]);
    }

    #[test]
    fn digitize_decreasing_and_edge_cases() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 2.5, f64::NAN];
        let bins = [3.0, 2.0, 1.0];
        assert_eq!(digitize(&x, &bins, false), vec![3, 2, 1, 0, 0, 1, 0]);
        assert_eq!(digitize(&x, &bins, true), vec![3, 3, 2, 1, 0, 1, 0]);
        assert_eq!(digitize(&[1.0, 2.0], &[], false), vec![0, 0]);
        assert_eq!(digitize(&[], &[1.0], false), Vec::<usize>::new());
        let eq = [2.0, 2.0, 2.0];
        assert_eq!(digitize(&[1.0, 2.0, 3.0], &eq, false), vec![0, 3, 3]);
        assert!(std::panic::catch_unwind(|| digitize(&[1.0], &[1.0, 3.0, 2.0], false)).is_err());
    }

    #[test]
    fn average_weighted() {
        assert_close(
            &[average(&[1.0, 2.0, 3.0, 4.0], Some(&[4.0, 3.0, 2.0, 1.0]))],
            &[2.0],
        );
        assert_close(&[average(&[1.0, 2.0, 3.0, 4.0], None)], &[2.5]);
        assert!(average(&[], None).is_nan());
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = [1.0, 0.0, 2.0, 3.0, 1.0, 1.0];
        assert_close(
            &average_axis(&a, &[2, 3], 0, Some(&[1.0, 3.0])),
            &[3.25, 4.25, 5.25],
        );
        assert_close(
            &average_axis(&a, &[2, 3], 1, Some(&[1.0, 0.0, 2.0])),
            &[2.3333333333333335, 5.333333333333333],
        );
        assert_close(
            &average_axis(&a, &[2, 3], 1, Some(&w)),
            &[2.3333333333333335, 4.6],
        );
        assert_close(&average_axis(&a, &[2, 3], 0, Some(&w)), &[3.25, 5.0, 4.0]);
        assert_close(&average_axis(&a, &[2, 3], 1, None), &[2.0, 5.0]);
        assert!(std::panic::catch_unwind(|| average(&[1.0, 2.0], Some(&[1.0, -1.0]))).is_err());
        assert!(
            std::panic::catch_unwind(|| average_axis(&[1.0, 2.0], &[2], 0, Some(&[1.0]))).is_err()
        );
    }

    #[test]
    fn quantile_linear() {
        let x = [3.5, 1.0, 7.25, 2.0, 9.0, 4.5];
        let qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 0.3333333333333333];
        let want = [1.0, 1.5, 2.375, 4.0, 6.5625, 8.125, 9.0, 3.0];
        for (q, w) in qs.into_iter().zip(want) {
            assert_close(&[quantile(&x, q)], &[w]);
        }
        assert_eq!(quantile(&[5.0], 0.3), 5.0);
        assert_eq!(quantile(&[1.0, 2.0], 0.5), 1.5);
    }

    #[test]
    fn quantile_nan_inf_and_errors() {
        assert!(quantile(&[1.0, f64::NAN, 3.0], 0.5).is_nan());
        assert!(quantile(&[1.0, f64::INFINITY], 0.5).is_nan());
        assert!(quantile(&[1.0, f64::INFINITY], 1.0).is_nan());
        assert!(quantile(&[1.0, f64::INFINITY, f64::INFINITY], 0.5).is_nan());
        assert_eq!(quantile(&[f64::NEG_INFINITY, 1.0, 2.0], 0.5), 1.0);
        assert!(std::panic::catch_unwind(|| quantile(&[1.0], 1.5)).is_err());
        assert!(std::panic::catch_unwind(|| quantile(&[1.0], -0.1)).is_err());
        assert!(std::panic::catch_unwind(|| quantile(&[], 0.5)).is_err());
        assert_eq!(nanquantile(&[1.0, f64::NAN, 3.0], 0.5), 2.0);
        assert_eq!(nanquantile(&[f64::NAN, 2.0, 8.0, f64::NAN, 4.0], 0.25), 3.0);
        assert!(nanquantile(&[f64::NAN, f64::NAN], 0.5).is_nan());
        assert!(nanquantile(&[], 0.5).is_nan());
    }

    #[test]
    fn quantile_axis_variants() {
        let m = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0];
        let n = [
            f64::NAN,
            1.0,
            4.0,
            1.0,
            5.0,
            9.0,
            f64::NAN,
            6.0,
            5.0,
            3.0,
            5.0,
            8.0,
        ];
        assert_close(&quantile_axis(&m, &[3, 4], 0, 0.3), &[4.2, 2.2, 3.2, 4.0]);
        assert_close(
            &quantile_axis(&m, &[3, 4], 1, 0.3),
            &[1.0, 4.699999999999999, 4.8],
        );
        assert_close(
            &quantile_axis(&m, &[3, 4], 1, 0.9),
            &[3.7, 8.100000000000001, 7.1000000000000005],
        );
        assert_close(
            &quantile_axis(&n, &[3, 4], 0, 0.5),
            &[f64::NAN, 3.0, f64::NAN, 6.0],
        );
        assert_close(
            &nanquantile_axis(&n, &[3, 4], 0, 0.5),
            &[5.0, 3.0, 4.5, 6.0],
        );
        assert_close(&nanquantile_axis(&n, &[3, 4], 1, 0.5), &[1.0, 6.0, 5.0]);
        let a3 = [
            0.0,
            1.0,
            2.8284271247461903,
            5.196152422706632,
            1.0,
            4.180339887498949,
            0.6969384566990691,
            4.520259177452136,
            1.6274169979695223,
            6.0,
            3.6227766016837926,
            1.4828726939093997,
            6.569219381653056,
            4.872166581031863,
            3.3832034148351795,
            2.0947501931112527,
            1.0,
            0.09279563550022374,
            6.367532368147138,
            5.819079927272796,
            5.442719099991592,
            5.234089594072643,
            5.189146716115445,
            5.304125036192545,
        ];
        assert_close(
            &quantile_axis(&a3, &[2, 3, 4], 1, 0.4),
            &[
                0.8,
                3.5442719099991593,
                2.402129391136766,
                3.912781880743589,
                4.554175279993274,
                3.916292391925535,
                4.827958055859392,
                4.662250067576286,
            ],
        );
        assert_close(
            &quantile_axis(&a3, &[2, 3, 4], 2, 0.4),
            &[
                1.3656854249492385,
                1.6360679774997904,
                2.026488918712377,
                3.6809960480745163,
                1.96381598545456,
                5.248096682496623,
            ],
        );
        assert_close(
            &quantile_axis(&a3, &[2, 3, 4], 0, 0.4),
            &[
                2.6276877526612226,
                2.548866632412745,
                3.050337640781786,
                3.3353110849494048,
                1.0,
                1.727813336299714,
                2.965176021278297,
                5.0397874773804,
                3.1535378387783504,
                5.540453756443585,
                4.249324647456453,
                3.011373630822658,
            ],
        );
    }

    #[test]
    fn pairwise_sum_and_mean() {
        let x: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        let naive: f64 = x.iter().sum();
        assert!((pairwise_sum(&x) - naive).abs() < 1e-10);
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(mean(&[1.0, 2.0, 6.0]), 3.0);
        // many small values after a big one: pairwise stays accurate
        let mut y = vec![1e8];
        y.extend(vec![1.0; 100_000]);
        assert_eq!(pairwise_sum(&y), 1e8 + 100_000.0);
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace(2.0, 2.0, 3), vec![2.0, 2.0, 2.0]);
    }
}
