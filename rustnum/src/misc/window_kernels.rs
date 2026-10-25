//! Window functions exactly as NumPy computes them (always `float64`).
//!
//! `m == 0` gives an empty window and `m == 1` gives `[1.0]`, like NumPy.

use super::math_kernels::i0;

use core::f64::consts::PI;

/// Shared symmetric grid `n = arange(1 - M, M, 2)` as floats, plus the divisor `M - 1`.
fn grid(m: usize) -> (Vec<f64>, f64) {
    let mf = m as f64;
    let n = (0..m).map(|k| (1.0 - mf) + 2.0 * k as f64).collect();
    (n, mf - 1.0)
}

fn trivial(m: usize) -> Option<Vec<f64>> {
    match m {
        0 => Some(Vec::new()),
        1 => Some(vec![1.0]),
        _ => None,
    }
}

/// `np.hanning(M)`: `0.5 + 0.5 cos(pi n / (M - 1))` for `n = -(M-1), -(M-3), ..., M-1`.
pub fn hanning(m: usize) -> Vec<f64> {
    if let Some(v) = trivial(m) {
        return v;
    }
    let (n, d) = grid(m);
    n.iter().map(|&n| 0.5 + 0.5 * (PI * n / d).cos()).collect()
}

/// `np.hamming(M)`: `0.54 + 0.46 cos(pi n / (M - 1))`.
pub fn hamming(m: usize) -> Vec<f64> {
    if let Some(v) = trivial(m) {
        return v;
    }
    let (n, d) = grid(m);
    n.iter()
        .map(|&n| 0.54 + 0.46 * (PI * n / d).cos())
        .collect()
}

/// `np.blackman(M)`: `0.42 + 0.5 cos(pi n / (M - 1)) + 0.08 cos(2 pi n / (M - 1))`.
pub fn blackman(m: usize) -> Vec<f64> {
    if let Some(v) = trivial(m) {
        return v;
    }
    let (n, d) = grid(m);
    n.iter()
        .map(|&n| 0.42 + 0.5 * (PI * n / d).cos() + 0.08 * (2.0 * PI * n / d).cos())
        .collect()
}

/// `np.bartlett(M)`: triangular window with zero end points.
pub fn bartlett(m: usize) -> Vec<f64> {
    if let Some(v) = trivial(m) {
        return v;
    }
    let (n, d) = grid(m);
    n.iter()
        .map(|&n| if n <= 0.0 { 1.0 + n / d } else { 1.0 - n / d })
        .collect()
}

/// `np.kaiser(M, beta)`: `I0(beta sqrt(1 - ((n - alpha) / alpha)^2)) / I0(beta)` with `alpha = (M-1)/2`.
pub fn kaiser(m: usize, beta: f64) -> Vec<f64> {
    if let Some(v) = trivial(m) {
        return v;
    }
    let alpha = (m as f64 - 1.0) / 2.0;
    let denom = i0(beta);
    (0..m)
        .map(|k| {
            let r = (k as f64 - alpha) / alpha;
            i0(beta * (1.0 - r * r).sqrt()) / denom
        })
        .collect()
}
