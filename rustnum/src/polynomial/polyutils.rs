//! Utility functions for polynomials, like `numpy.polynomial.polyutils`.

use crate::Array;

/// Get the domain from an array of sample points, like `pu.getdomain(x)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polyutils::getdomain;
/// assert_eq!(getdomain(&[0.0, 1.0, 5.0]), [0.0, 5.0]);
/// assert_eq!(getdomain(&[2.0]), [1.0, 3.0]);
/// ```
pub fn getdomain(x: &[f64]) -> [f64; 2] {
    if x.is_empty() {
        return [-1.0, 1.0];
    }
    let mut min = x[0];
    let mut max = x[0];
    for &val in &x[1..] {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }
    if min == max {
        [min - 1.0, max + 1.0]
    } else {
        [min, max]
    }
}

/// Compute the linear mapping parameters (offset, scale) between two domains, like `pu.mapparms(old, new)`.
///
/// The mapping is `y = off + scl * x`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polyutils::mapparms;
/// let (off, scl) = mapparms([-1.0, 1.0], [0.0, 10.0]);
/// assert_eq!(off, 5.0);
/// assert_eq!(scl, 5.0);
/// ```
pub fn mapparms(old: [f64; 2], new: [f64; 2]) -> (f64, f64) {
    let span_old = old[1] - old[0];
    assert!(span_old != 0.0, "domain cannot have zero width");
    let scl = (new[1] - new[0]) / span_old;
    let off = new[0] - scl * old[0];
    (off, scl)
}

/// Apply a linear domain map to a scalar point, like `pu.mapdomain(x, old, new)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polyutils::mapdomain;
/// assert_eq!(mapdomain(0.0, [-1.0, 1.0], [0.0, 10.0]), 5.0);
/// ```
pub fn mapdomain(x: f64, old: [f64; 2], new: [f64; 2]) -> f64 {
    let (off, scl) = mapparms(old, new);
    off + scl * x
}

/// Apply a linear domain map to a slice of points, like `pu.mapdomain(x, old, new)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polyutils::mapdomain_slice;
/// let y = mapdomain_slice(&[-1.0, 0.0, 1.0], [-1.0, 1.0], [0.0, 10.0]);
/// assert_eq!(y, vec![0.0, 5.0, 10.0]);
/// ```
pub fn mapdomain_slice(x: &[f64], old: [f64; 2], new: [f64; 2]) -> Vec<f64> {
    let (off, scl) = mapparms(old, new);
    x.iter().map(|&v| off + scl * v).collect()
}

/// Apply a linear domain map to an array of points, like `pu.mapdomain(x, old, new)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::polyutils::mapdomain_array;
/// let x = np::array![-1.0, 0.0, 1.0];
/// let y = mapdomain_array(&x, [-1.0, 1.0], [0.0, 10.0]);
/// assert_eq!(y.to_vec(), vec![0.0, 5.0, 10.0]);
/// ```
pub fn mapdomain_array(x: &Array<f64>, old: [f64; 2], new: [f64; 2]) -> Array<f64> {
    let (off, scl) = mapparms(old, new);
    let mapped: Vec<f64> = x.to_vec().into_iter().map(|v| off + scl * v).collect();
    Array::from_vec(mapped, x.shape())
}

/// Trim trailing coefficients smaller than tolerance, like `pu.trimseq(seq)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polyutils::trimseq;
/// assert_eq!(trimseq(&[1.0, 2.0, 0.0, 0.0], 0.0), vec![1.0, 2.0]);
/// assert_eq!(trimseq(&[0.0, 0.0], 0.0), vec![0.0]);
/// ```
pub fn trimseq(seq: &[f64], tol: f64) -> Vec<f64> {
    if let Some(pos) = seq.iter().rposition(|&c| c.abs() > tol) {
        seq[..=pos].to_vec()
    } else {
        vec![0.0]
    }
}

pub(crate) fn generic_div<F: Fn(&[f64], &[f64]) -> Vec<f64>>(
    mul_f: F,
    c1: &[f64],
    c2: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let trimmed_c1 = trimseq(c1, 0.0);
    let trimmed_c2 = trimseq(c2, 0.0);
    assert!(
        !(trimmed_c2.len() == 1 && trimmed_c2[0] == 0.0),
        "division by zero polynomial"
    );
    let lc1 = trimmed_c1.len();
    let lc2 = trimmed_c2.len();
    if lc1 < lc2 {
        (vec![0.0], trimmed_c1)
    } else if lc2 == 1 {
        let scl = trimmed_c2[0];
        (trimmed_c1.iter().map(|&x| x / scl).collect(), vec![0.0])
    } else {
        let mut quo = vec![0.0; lc1 - lc2 + 1];
        let mut rem = trimmed_c1;
        for i in (0..=lc1 - lc2).rev() {
            let mut basis_i = vec![0.0; i + 1];
            basis_i[i] = 1.0;
            let p = mul_f(&basis_i, &trimmed_c2);
            let p_last = p.last().copied().unwrap_or(1.0);
            let rem_last = rem.last().copied().unwrap_or(0.0);
            let q = rem_last / p_last;
            let rem_len = rem.len() - 1;
            let mut next_rem = Vec::with_capacity(rem_len);
            for j in 0..rem_len {
                let pj = if j < p.len() - 1 { p[j] } else { 0.0 };
                next_rem.push(rem[j] - q * pj);
            }
            rem = next_rem;
            quo[i] = q;
        }
        (quo, trimseq(&rem, 0.0))
    }
}

pub(crate) fn generic_pow<F: Fn(&[f64], &[f64]) -> Vec<f64>>(
    mul_f: F,
    c: &[f64],
    power: usize,
) -> Vec<f64> {
    if power == 0 {
        return vec![1.0];
    }
    let mut base = c.to_vec();
    let mut exp = power;
    let mut result = vec![1.0];
    while exp > 0 {
        if exp % 2 == 1 {
            result = mul_f(&result, &base);
        }
        if exp > 1 {
            base = mul_f(&base, &base);
        }
        exp /= 2;
    }
    result
}

pub(crate) fn generic_fit<F: Fn(&[f64], usize) -> Array<f64>>(
    vander_fn: F,
    x: &[f64],
    y: &[f64],
    deg: usize,
) -> Vec<f64> {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    let v = vander_fn(x, deg);
    let y_arr = Array::from_vec(y.to_vec(), &[y.len()]);
    let c = crate::linalg::lstsq(&v, &y_arr).expect("least-squares fit failed");
    c.to_vec()
}
