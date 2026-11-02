//! Standard power-series polynomials, like `numpy.polynomial.polynomial`.

use super::polyutils;
use crate::Array;

/// Evaluate a polynomial at points x using Horner's method, like `np.polynomial.polynomial.polyval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyval;
/// // 1 + 2x + 3x^2 at x=2 -> 1 + 4 + 12 = 17
/// assert_eq!(polyval(2.0, &[1.0, 2.0, 3.0]), 17.0);
/// ```
pub fn polyval(x: f64, c: &[f64]) -> f64 {
    let mut val = 0.0;
    for &coeff in c.iter().rev() {
        val = coeff + x * val;
    }
    val
}

/// Evaluate a polynomial over an array of points x, like `np.polynomial.polynomial.polyval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::polynomial::polyval_array;
/// let x = np::array![0.0, 1.0, 2.0];
/// let y = polyval_array(&x, &[1.0, 2.0, 3.0]);
/// assert_eq!(y.to_vec(), vec![1.0, 6.0, 17.0]);
/// ```
pub fn polyval_array(x: &Array<f64>, c: &[f64]) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(|xi| polyval(xi, c)).collect();
    Array::from_vec(data, x.shape())
}

/// Vandermonde matrix of given degree, like `np.polynomial.polynomial.polyvander(x, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyvander;
/// let v = polyvander(&[1.0, 2.0], 2);
/// assert_eq!(v.shape(), &[2, 3]);
/// assert_eq!(v.to_vec(), vec![1.0, 1.0, 1.0, 1.0, 2.0, 4.0]);
/// ```
pub fn polyvander(x: &[f64], deg: usize) -> Array<f64> {
    let m = x.len();
    let n = deg + 1;
    let mut data = vec![0.0; m * n];
    for (i, &xi) in x.iter().enumerate() {
        let mut p = 1.0;
        for j in 0..n {
            data[i * n + j] = p;
            p *= xi;
        }
    }
    Array::from_vec(data, &[m, n])
}

/// Companion matrix of polynomial coefficients, like `np.polynomial.polynomial.polycompanion(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polycompanion;
/// let m = polycompanion(&[1.0, 2.0, 3.0]);
/// assert_eq!(m.shape(), &[2, 2]);
/// ```
pub fn polycompanion(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    assert!(
        c.len() >= 2,
        "Series must have maximum degree of at least 1."
    );
    let n = c.len() - 1;
    let mut mat = vec![0.0; n * n];
    let last = c[n];
    for i in 1..n {
        mat[i * n + (i - 1)] = 1.0;
    }
    for i in 0..n {
        mat[i * n + (n - 1)] -= c[i] / last;
    }
    Array::from_vec(mat, &[n, n])
}

/// Least-squares fit of a polynomial to data, like `np.polynomial.polynomial.polyfit(x, y, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyfit;
/// let c = polyfit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0], 1).to_vec();
/// assert!((c[0] - 1.0).abs() < 1e-10);
/// assert!((c[1] - 2.0).abs() < 1e-10);
/// ```
pub fn polyfit(x: &[f64], y: &[f64], deg: usize) -> Array<f64> {
    let c = polyutils::generic_fit(polyvander, x, y, deg);
    Array::from_vec(c.clone(), &[c.len()])
}

/// Compute the roots of a polynomial, like `np.polynomial.polynomial.polyroots(c)`.
///
/// Returns `(re, im)` arrays.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyroots;
/// // Roots of x^2 - 1 = 0 -> 1, -1
/// let (re, im) = polyroots(&[-1.0, 0.0, 1.0]);
/// assert_eq!(re.len(), 2);
/// ```
pub fn polyroots(c: &[f64]) -> (Array<f64>, Array<f64>) {
    let trimmed = polyutils::trimseq(c, 0.0);
    if trimmed.len() <= 1 {
        return (
            Array::from_vec(Vec::new(), &[0]),
            Array::from_vec(Vec::new(), &[0]),
        );
    }
    let comp = polycompanion(&trimmed);
    crate::linalg::eigvals(&comp).expect("eigvals failed")
}

/// Differentiate a polynomial m times, like `np.polynomial.polynomial.polyder(c, m)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyder;
/// // (1 + 2x + 3x^2)' = 2 + 6x
/// let d = polyder(&[1.0, 2.0, 3.0], 1);
/// assert_eq!(d.to_vec(), vec![2.0, 6.0]);
/// ```
pub fn polyder(c: &[f64], m: usize) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(c.to_vec(), &[c.len()]);
    }
    let mut current = c.to_vec();
    for _ in 0..m {
        if current.len() <= 1 {
            return Array::from_vec(vec![0.0], &[1]);
        }
        let mut next = Vec::with_capacity(current.len() - 1);
        for (i, &coeff) in current.iter().enumerate().skip(1) {
            next.push(coeff * i as f64);
        }
        current = next;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Integrate a polynomial m times, like `np.polynomial.polynomial.polyint(c, m, k, lbnd)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyint;
/// // int(2 + 6x) = 2x + 3x^2
/// let res = polyint(&[2.0, 6.0], 1, &[0.0], 0.0);
/// assert_eq!(res.to_vec(), vec![0.0, 2.0, 3.0]);
/// ```
pub fn polyint(c: &[f64], m: usize, k: &[f64], lbnd: f64) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(c.to_vec(), &[c.len()]);
    }
    let mut current = c.to_vec();
    for step in 0..m {
        let k_val = if step < k.len() { k[step] } else { 0.0 };
        let mut next = Vec::with_capacity(current.len() + 1);
        next.push(0.0);
        for (i, &coeff) in current.iter().enumerate() {
            next.push(coeff / (i + 1) as f64);
        }
        let val_at_lbnd = polyval(lbnd, &next);
        next[0] = k_val - val_at_lbnd;
        current = next;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Add one polynomial to another, like `np.polynomial.polynomial.polyadd(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyadd;
/// let s = polyadd(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
/// assert_eq!(s.to_vec(), vec![4.0, 6.0, 5.0]);
/// ```
pub fn polyadd(c1: &[f64], c2: &[f64]) -> Array<f64> {
    let len = c1.len().max(c2.len());
    let mut res = vec![0.0; len];
    for (i, &v) in c1.iter().enumerate() {
        res[i] += v;
    }
    for (i, &v) in c2.iter().enumerate() {
        res[i] += v;
    }
    let trimmed = polyutils::trimseq(&res, 0.0);
    Array::from_vec(trimmed.clone(), &[trimmed.len()])
}

/// Subtract one polynomial from another, like `np.polynomial.polynomial.polysub(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polysub;
/// let diff = polysub(&[3.0, 5.0, 5.0], &[1.0, 2.0]);
/// assert_eq!(diff.to_vec(), vec![2.0, 3.0, 5.0]);
/// ```
pub fn polysub(c1: &[f64], c2: &[f64]) -> Array<f64> {
    let len = c1.len().max(c2.len());
    let mut res = vec![0.0; len];
    for (i, &v) in c1.iter().enumerate() {
        res[i] += v;
    }
    for (i, &v) in c2.iter().enumerate() {
        res[i] -= v;
    }
    let trimmed = polyutils::trimseq(&res, 0.0);
    Array::from_vec(trimmed.clone(), &[trimmed.len()])
}

/// Multiply one polynomial by another, like `np.polynomial.polynomial.polymul(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polymul;
/// // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
/// let prod = polymul(&[1.0, 2.0], &[3.0, 4.0]);
/// assert_eq!(prod.to_vec(), vec![3.0, 10.0, 8.0]);
/// ```
pub fn polymul(c1: &[f64], c2: &[f64]) -> Array<f64> {
    let t1 = polyutils::trimseq(c1, 0.0);
    let t2 = polyutils::trimseq(c2, 0.0);
    if (t1.len() == 1 && t1[0] == 0.0) || (t2.len() == 1 && t2[0] == 0.0) {
        return Array::from_vec(vec![0.0], &[1]);
    }
    let mut res = vec![0.0; t1.len() + t2.len() - 1];
    for (i, &a) in t1.iter().enumerate() {
        for (j, &b) in t2.iter().enumerate() {
            res[i + j] += a * b;
        }
    }
    let trimmed = polyutils::trimseq(&res, 0.0);
    Array::from_vec(trimmed.clone(), &[trimmed.len()])
}

/// Multiply a polynomial by x, like `np.polynomial.polynomial.polymulx(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polymulx;
/// let px = polymulx(&[1.0, 2.0]);
/// assert_eq!(px.to_vec(), vec![0.0, 1.0, 2.0]);
/// ```
pub fn polymulx(c: &[f64]) -> Array<f64> {
    let t = polyutils::trimseq(c, 0.0);
    if t.len() == 1 && t[0] == 0.0 {
        return Array::from_vec(vec![0.0], &[1]);
    }
    let mut res = Vec::with_capacity(t.len() + 1);
    res.push(0.0);
    res.extend_from_slice(&t);
    Array::from_vec(res.clone(), &[res.len()])
}

/// Divide one polynomial by another, returning quotient and remainder, like `np.polynomial.polynomial.polydiv(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polydiv;
/// let (q, r) = polydiv(&[1.0, 2.0, 1.0], &[1.0, 1.0]);
/// assert_eq!(q.to_vec(), vec![1.0, 1.0]);
/// assert_eq!(r.to_vec(), vec![0.0]);
/// ```
pub fn polydiv(c1: &[f64], c2: &[f64]) -> (Array<f64>, Array<f64>) {
    let (quo, rem) = polyutils::generic_div(|a, b| polymul(a, b).to_vec(), c1, c2);
    (
        Array::from_vec(quo.clone(), &[quo.len()]),
        Array::from_vec(rem.clone(), &[rem.len()]),
    )
}

/// Raise a polynomial to an integer power, like `np.polynomial.polynomial.polypow(c, power)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polypow;
/// let p2 = polypow(&[1.0, 1.0], 2);
/// assert_eq!(p2.to_vec(), vec![1.0, 2.0, 1.0]);
/// ```
pub fn polypow(c: &[f64], power: usize) -> Array<f64> {
    let res = polyutils::generic_pow(|a, b| polymul(a, b).to_vec(), c, power);
    Array::from_vec(res.clone(), &[res.len()])
}

/// Linear polynomial with given intercept and slope, like `np.polynomial.polynomial.polyline(off, scl)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyline;
/// let l = polyline(2.0, 3.0);
/// assert_eq!(l.to_vec(), vec![2.0, 3.0]);
/// ```
pub fn polyline(off: f64, scl: f64) -> Array<f64> {
    Array::from_vec(vec![off, scl], &[2])
}

/// Generate a monic polynomial with the specified roots, like `np.polynomial.polynomial.polyfromroots(roots)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polyfromroots;
/// // Roots 1 and 2 -> (x - 1)(x - 2) = x^2 - 3x + 2
/// let p = polyfromroots(&[1.0, 2.0]);
/// assert_eq!(p.to_vec(), vec![2.0, -3.0, 1.0]);
/// ```
pub fn polyfromroots(roots: &[f64]) -> Array<f64> {
    let mut prod = vec![1.0];
    for &r in roots {
        let factor = [-r, 1.0];
        prod = polymul(&prod, &factor).to_vec();
    }
    Array::from_vec(prod.clone(), &[prod.len()])
}

/// Trim small trailing coefficients from a polynomial, like `np.polynomial.polynomial.polytrim(c, tol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::polynomial::polytrim;
/// let t = polytrim(&[1.0, 2.0, 0.0, 0.0], 0.0);
/// assert_eq!(t.to_vec(), vec![1.0, 2.0]);
/// ```
pub fn polytrim(c: &[f64], tol: f64) -> Array<f64> {
    let t = polyutils::trimseq(c, tol);
    Array::from_vec(t.clone(), &[t.len()])
}
