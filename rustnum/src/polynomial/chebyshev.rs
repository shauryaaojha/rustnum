//! Chebyshev series, like `numpy.polynomial.chebyshev`.

use super::polyutils;
use crate::Array;

/// Evaluate a Chebyshev series at points x using Clenshaw recurrence, like `np.polynomial.chebyshev.chebval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebval;
/// // T_0(0.5) = 1, T_1(0.5) = 0.5, T_2(0.5) = 2*0.5^2 - 1 = -0.5
/// // 1*T_0 + 2*T_1 + 3*T_2 at 0.5 = 1 + 1 - 1.5 = 0.5
/// assert_eq!(chebval(0.5, &[1.0, 2.0, 3.0]), 0.5);
/// ```
pub fn chebval(x: f64, c: &[f64]) -> f64 {
    let n = c.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return c[0];
    }
    if n == 2 {
        return c[0] + c[1] * x;
    }
    let mut c0 = c[n - 2];
    let mut c1 = c[n - 1];
    let x2 = 2.0 * x;
    for i in 3..=n {
        let tmp = c0;
        c0 = c[n - i] - c1;
        c1 = tmp + c1 * x2;
    }
    c0 + c1 * x
}

/// Evaluate a Chebyshev series over an array of points, like `np.polynomial.chebyshev.chebval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::chebyshev::chebval_array;
/// let x = np::array![0.0, 1.0];
/// let y = chebval_array(&x, &[1.0, 2.0]);
/// assert_eq!(y.to_vec(), vec![1.0, 3.0]);
/// ```
pub fn chebval_array(x: &Array<f64>, c: &[f64]) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(|xi| chebval(xi, c)).collect();
    Array::from_vec(data, x.shape())
}

/// Pseudo-Vandermonde matrix of Chebyshev polynomials, like `np.polynomial.chebyshev.chebvander(x, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebvander;
/// let v = chebvander(&[0.5], 2);
/// assert_eq!(v.to_vec(), vec![1.0, 0.5, -0.5]);
/// ```
pub fn chebvander(x: &[f64], deg: usize) -> Array<f64> {
    let m = x.len();
    let n = deg + 1;
    let mut data = vec![0.0; m * n];
    for (i, &xi) in x.iter().enumerate() {
        data[i * n] = 1.0;
        if deg > 0 {
            data[i * n + 1] = xi;
            for j in 2..n {
                data[i * n + j] = 2.0 * xi * data[i * n + (j - 1)] - data[i * n + (j - 2)];
            }
        }
    }
    Array::from_vec(data, &[m, n])
}

/// Scaled companion matrix of Chebyshev series, like `np.polynomial.chebyshev.chebcompanion(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebcompanion;
/// let m = chebcompanion(&[1.0, 2.0, 3.0]);
/// assert_eq!(m.shape(), &[2, 2]);
/// ```
pub fn chebcompanion(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    assert!(
        c.len() >= 2,
        "Series must have maximum degree of at least 1."
    );
    let n = c.len() - 1;
    let mut mat = vec![0.0; n * n];
    let mut scl = vec![1.0];
    for _ in 0..n - 1 {
        scl.push((0.5f64).sqrt());
    }
    if n > 1 {
        mat[1] = (0.5f64).sqrt();
        mat[n] = (0.5f64).sqrt();
        for i in 1..n - 1 {
            mat[i * n + (i + 1)] = 0.5;
            mat[(i + 1) * n + i] = 0.5;
        }
    }
    let last = c[n];
    let last_scl = scl[n - 1];
    for i in 0..n {
        mat[i * n + (n - 1)] -= (c[i] / last) * (scl[i] / last_scl) * 0.5;
    }
    Array::from_vec(mat, &[n, n])
}

/// Least-squares fit of Chebyshev series to data, like `np.polynomial.chebyshev.chebfit(x, y, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebfit;
/// let c = chebfit(&[-1.0, 0.0, 1.0], &[1.0, 0.0, 1.0], 2).to_vec();
/// assert!((c[0] - 0.5).abs() < 1e-10);
/// assert!((c[2] - 0.5).abs() < 1e-10);
/// ```
pub fn chebfit(x: &[f64], y: &[f64], deg: usize) -> Array<f64> {
    let c = polyutils::generic_fit(chebvander, x, y, deg);
    Array::from_vec(c.clone(), &[c.len()])
}

/// Compute the roots of a Chebyshev series, like `np.polynomial.chebyshev.chebroots(c)`.
///
/// Returns `(re, im)` arrays.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebroots;
/// // T_2(x) = 2x^2 - 1 = 0 -> roots +-1/sqrt(2) ~ +-0.7071
/// let (re, _) = chebroots(&[0.0, 0.0, 1.0]);
/// assert_eq!(re.len(), 2);
/// ```
pub fn chebroots(c: &[f64]) -> (Array<f64>, Array<f64>) {
    let trimmed = polyutils::trimseq(c, 0.0);
    if trimmed.len() <= 1 {
        return (
            Array::from_vec(Vec::new(), &[0]),
            Array::from_vec(Vec::new(), &[0]),
        );
    }
    let comp = chebcompanion(&trimmed);
    crate::linalg::eigvals(&comp).expect("eigvals failed")
}

/// Differentiate a Chebyshev series m times, like `np.polynomial.chebyshev.chebder(c, m)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebder;
/// // (T_2(x))' = 4x = 4*T_1(x)
/// let d = chebder(&[0.0, 0.0, 1.0], 1);
/// assert_eq!(d.to_vec(), vec![0.0, 4.0]);
/// ```
pub fn chebder(c: &[f64], m: usize) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(c.to_vec(), &[c.len()]);
    }
    let mut current = c.to_vec();
    for _ in 0..m {
        let n = current.len();
        if n <= 1 {
            return Array::from_vec(vec![0.0], &[1]);
        }
        let mut der = vec![0.0; n - 1];
        for j in (3..n).rev() {
            der[j - 1] = (2 * j) as f64 * current[j];
            current[j - 2] += (j as f64 * current[j]) / (j - 2) as f64;
        }
        if n > 2 {
            der[1] = 4.0 * current[2];
        }
        der[0] = current[1];
        current = der;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Integrate a Chebyshev series m times, like `np.polynomial.chebyshev.chebint(c, m, k, lbnd)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebint;
/// let res = chebint(&[1.0, 2.0, 3.0], 1, &[0.0], 0.0);
/// assert_eq!(res.shape(), &[4]);
/// ```
pub fn chebint(c: &[f64], m: usize, k: &[f64], lbnd: f64) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(c.to_vec(), &[c.len()]);
    }
    let mut current = c.to_vec();
    for step in 0..m {
        let k_val = if step < k.len() { k[step] } else { 0.0 };
        let n = current.len();
        let mut tmp = vec![0.0; n + 1];
        tmp[1] = current[0];
        if n > 1 {
            tmp[2] = current[1] / 4.0;
        }
        for j in 2..n {
            tmp[j + 1] = current[j] / (2.0 * (j + 1) as f64);
            tmp[j - 1] -= current[j] / (2.0 * (j - 1) as f64);
        }
        let val_at_lbnd = chebval(lbnd, &tmp);
        tmp[0] += k_val - val_at_lbnd;
        current = tmp;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Add two Chebyshev series, like `np.polynomial.chebyshev.chebadd(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebadd;
/// let res = chebadd(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
/// assert_eq!(res.to_vec(), vec![4.0, 6.0, 5.0]);
/// ```
pub fn chebadd(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polyadd(c1, c2)
}

/// Subtract two Chebyshev series, like `np.polynomial.chebyshev.chebsub(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebsub;
/// let res = chebsub(&[3.0, 5.0, 5.0], &[1.0, 2.0]);
/// assert_eq!(res.to_vec(), vec![2.0, 3.0, 5.0]);
/// ```
pub fn chebsub(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polysub(c1, c2)
}

/// Multiply one Chebyshev series by another, like `np.polynomial.chebyshev.chebmul(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebmul;
/// let res = chebmul(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
/// assert_eq!(res.to_vec(), vec![6.5, 12.0, 12.0, 4.0, 1.5]);
/// ```
pub fn chebmul(c1: &[f64], c2: &[f64]) -> Array<f64> {
    let t1 = polyutils::trimseq(c1, 0.0);
    let t2 = polyutils::trimseq(c2, 0.0);
    if (t1.len() == 1 && t1[0] == 0.0) || (t2.len() == 1 && t2[0] == 0.0) {
        return Array::from_vec(vec![0.0], &[1]);
    }
    let mut res = vec![0.0; t1.len() + t2.len() - 1];
    for (i, &a) in t1.iter().enumerate() {
        for (j, &b) in t2.iter().enumerate() {
            if i == 0 {
                res[j] += a * b;
            } else if j == 0 {
                res[i] += a * b;
            } else {
                res[i + j] += 0.5 * a * b;
                let diff = i.abs_diff(j);
                res[diff] += 0.5 * a * b;
            }
        }
    }
    let trimmed = polyutils::trimseq(&res, 0.0);
    Array::from_vec(trimmed.clone(), &[trimmed.len()])
}

/// Multiply a Chebyshev series by x, like `np.polynomial.chebyshev.chebmulx(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebmulx;
/// let res = chebmulx(&[1.0, 2.0, 3.0]);
/// assert_eq!(res.to_vec(), vec![1.0, 2.5, 1.0, 1.5]);
/// ```
pub fn chebmulx(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    if c.len() == 1 && c[0] == 0.0 {
        return Array::from_vec(vec![0.0], &[1]);
    }
    let mut prd = vec![0.0; c.len() + 1];
    prd[1] = c[0];
    if c.len() > 1 {
        for i in 1..c.len() {
            let tmp = c[i] * 0.5;
            prd[i + 1] += tmp;
            prd[i - 1] += tmp;
        }
    }
    Array::from_vec(prd.clone(), &[prd.len()])
}

/// Divide two Chebyshev series, returning quotient and remainder, like `np.polynomial.chebyshev.chebdiv(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebdiv;
/// let (q, r) = chebdiv(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
/// assert_eq!(q.to_vec(), vec![3.0]);
/// assert_eq!(r.to_vec(), vec![-8.0, -4.0]);
/// ```
pub fn chebdiv(c1: &[f64], c2: &[f64]) -> (Array<f64>, Array<f64>) {
    let (quo, rem) = polyutils::generic_div(|a, b| chebmul(a, b).to_vec(), c1, c2);
    (
        Array::from_vec(quo.clone(), &[quo.len()]),
        Array::from_vec(rem.clone(), &[rem.len()]),
    )
}

/// Raise a Chebyshev series to an integer power, like `np.polynomial.chebyshev.chebpow(c, power)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebpow;
/// let p = chebpow(&[0.0, 1.0], 2);
/// assert_eq!(p.to_vec(), vec![0.5, 0.0, 0.5]);
/// ```
pub fn chebpow(c: &[f64], power: usize) -> Array<f64> {
    let res = polyutils::generic_pow(|a, b| chebmul(a, b).to_vec(), c, power);
    Array::from_vec(res.clone(), &[res.len()])
}

/// Linear Chebyshev series off + scl*x, like `np.polynomial.chebyshev.chebline(off, scl)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebline;
/// let l = chebline(2.0, 3.0);
/// assert_eq!(l.to_vec(), vec![2.0, 3.0]);
/// ```
pub fn chebline(off: f64, scl: f64) -> Array<f64> {
    Array::from_vec(vec![off, scl], &[2])
}

/// Generate a Chebyshev series with the specified roots, like `np.polynomial.chebyshev.chebfromroots(roots)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebfromroots;
/// let p = chebfromroots(&[1.0, -1.0, 0.0]);
/// assert_eq!(p.to_vec(), vec![0.0, -0.25, 0.0, 0.25]);
/// ```
pub fn chebfromroots(roots: &[f64]) -> Array<f64> {
    let p = super::polynomial::polyfromroots(roots);
    poly2cheb(&p.to_vec())
}

/// Trim small trailing coefficients from Chebyshev series, like `np.polynomial.chebyshev.chebtrim(c, tol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebtrim;
/// let t = chebtrim(&[1.0, 2.0, 0.0], 0.0);
/// assert_eq!(t.to_vec(), vec![1.0, 2.0]);
/// ```
pub fn chebtrim(c: &[f64], tol: f64) -> Array<f64> {
    let t = polyutils::trimseq(c, tol);
    Array::from_vec(t.clone(), &[t.len()])
}

/// Gauss-Chebyshev quadrature nodes and weights, like `np.polynomial.chebyshev.chebgauss(deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebgauss;
/// let (x, w) = chebgauss(2);
/// assert_eq!(x.len(), 2);
/// assert_eq!(w.len(), 2);
/// ```
pub fn chebgauss(deg: usize) -> (Array<f64>, Array<f64>) {
    assert!(deg >= 1, "deg must be a positive integer");
    let mut x = Vec::with_capacity(deg);
    let n = deg as f64;
    for i in 1..=deg {
        let angle = std::f64::consts::PI * (2.0 * i as f64 - 1.0) / (2.0 * n);
        x.push(angle.cos());
    }
    let w = vec![std::f64::consts::PI / n; deg];
    (Array::from_vec(x, &[deg]), Array::from_vec(w, &[deg]))
}

/// Chebyshev weight function 1 / sqrt(1 - x^2), like `np.polynomial.chebyshev.chebweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::chebweight;
/// assert_eq!(chebweight(0.0), 1.0);
/// ```
pub fn chebweight(x: f64) -> f64 {
    1.0 / (1.0 - x * x).sqrt()
}

/// Evaluate Chebyshev weight function over array, like `np.polynomial.chebyshev.chebweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::chebyshev::chebweight_array;
/// let x = np::array![0.0];
/// assert_eq!(chebweight_array(&x).to_vec(), vec![1.0]);
/// ```
pub fn chebweight_array(x: &Array<f64>) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(chebweight).collect();
    Array::from_vec(data, x.shape())
}

/// Convert a polynomial in standard basis to Chebyshev series, like `np.polynomial.chebyshev.poly2cheb(pol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::poly2cheb;
/// let c = poly2cheb(&[0.0, 1.0, 2.0, 3.0]);
/// assert_eq!(c.to_vec(), vec![1.0, 3.25, 1.0, 0.75]);
/// ```
pub fn poly2cheb(pol: &[f64]) -> Array<f64> {
    let pol = polyutils::trimseq(pol, 0.0);
    let mut res = vec![0.0];
    for &coeff in pol.iter().rev() {
        let mx = chebmulx(&res).to_vec();
        res = chebadd(&mx, &[coeff]).to_vec();
    }
    Array::from_vec(res.clone(), &[res.len()])
}

/// Convert a Chebyshev series to a standard polynomial, like `np.polynomial.chebyshev.cheb2poly(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::chebyshev::cheb2poly;
/// let p = cheb2poly(&[0.0, 1.0, 2.0, 3.0]);
/// assert_eq!(p.to_vec(), vec![-2.0, -8.0, 4.0, 12.0]);
/// ```
pub fn cheb2poly(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    let n = c.len();
    if n < 3 {
        return Array::from_vec(c.clone(), &[c.len()]);
    }
    let mut c0 = vec![c[n - 2]];
    let mut c1 = vec![c[n - 1]];
    for i in (2..=n - 1).rev() {
        let tmp = c0;
        c0 = super::polynomial::polysub(&[c[i - 2]], &c1).to_vec();
        let x2_c1 = super::polynomial::polymul(&[0.0, 2.0], &c1).to_vec();
        c1 = super::polynomial::polyadd(&tmp, &x2_c1).to_vec();
    }
    let x_c1 = super::polynomial::polymulx(&c1).to_vec();
    super::polynomial::polyadd(&c0, &x_c1)
}
