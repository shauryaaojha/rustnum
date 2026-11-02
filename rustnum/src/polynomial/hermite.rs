//! Physicists' Hermite series, like `numpy.polynomial.hermite`.

use super::polyutils;
use crate::Array;

/// Evaluate a Hermite series at points x using Clenshaw recurrence, like `np.polynomial.hermite.hermval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermval;
/// // H_0(0.5) = 1, H_1(0.5) = 2*0.5 = 1, H_2(0.5) = 4*0.25 - 2 = -1
/// assert_eq!(hermval(0.5, &[1.0, 2.0]), 3.0);
/// ```
pub fn hermval(x: f64, c: &[f64]) -> f64 {
    let n = c.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return c[0];
    }
    if n == 2 {
        return c[0] + c[1] * 2.0 * x;
    }
    let mut nd = n;
    let mut c0 = c[n - 2];
    let mut c1 = c[n - 1];
    let x2 = 2.0 * x;
    for i in 3..=n {
        let tmp = c0;
        nd -= 1;
        c0 = c[n - i] - c1 * (2 * (nd - 1)) as f64;
        c1 = tmp + c1 * x2;
    }
    c0 + c1 * x2
}

/// Evaluate a Hermite series over an array of points, like `np.polynomial.hermite.hermval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::hermite::hermval_array;
/// let x = np::array![0.0, 1.0];
/// let y = hermval_array(&x, &[1.0, 2.0]);
/// assert_eq!(y.to_vec(), vec![1.0, 5.0]);
/// ```
pub fn hermval_array(x: &Array<f64>, c: &[f64]) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(|xi| hermval(xi, c)).collect();
    Array::from_vec(data, x.shape())
}

/// Pseudo-Vandermonde matrix of Hermite polynomials, like `np.polynomial.hermite.hermvander(x, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermvander;
/// let v = hermvander(&[-1.0, 0.0, 1.0], 2);
/// assert_eq!(v.shape(), &[3, 3]);
/// ```
pub fn hermvander(x: &[f64], deg: usize) -> Array<f64> {
    let m = x.len();
    let n = deg + 1;
    let mut data = vec![0.0; m * n];
    for (i, &xi) in x.iter().enumerate() {
        data[i * n] = 1.0;
        if deg > 0 {
            let x2 = 2.0 * xi;
            data[i * n + 1] = x2;
            for j in 2..n {
                data[i * n + j] =
                    data[i * n + (j - 1)] * x2 - data[i * n + (j - 2)] * (2 * (j - 1)) as f64;
            }
        }
    }
    Array::from_vec(data, &[m, n])
}

/// Scaled companion matrix of Hermite series, like `np.polynomial.hermite.hermcompanion(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermcompanion;
/// let m = hermcompanion(&[1.0, 0.0, 1.0]);
/// assert_eq!(m.shape(), &[2, 2]);
/// ```
pub fn hermcompanion(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    assert!(
        c.len() >= 2,
        "Series must have maximum degree of at least 1."
    );
    if c.len() == 2 {
        return Array::from_vec(vec![-0.5 * c[0] / c[1]], &[1, 1]);
    }
    let n = c.len() - 1;
    let mut mat = vec![0.0; n * n];
    let mut scl = vec![0.0; n];
    scl[n - 1] = 1.0;
    for i in (0..n - 1).rev() {
        scl[i] = scl[i + 1] / (2.0 * (i + 1) as f64).sqrt();
    }
    for i in 1..n {
        let v = (0.5 * i as f64).sqrt();
        mat[(i - 1) * n + i] = v;
        mat[i * n + (i - 1)] = v;
    }
    let last = 2.0 * c[n];
    for i in 0..n {
        mat[i * n + (n - 1)] -= scl[i] * c[i] / last;
    }
    Array::from_vec(mat, &[n, n])
}

/// Least-squares fit of Hermite series to data, like `np.polynomial.hermite.hermfit(x, y, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermfit;
/// let c = hermfit(&[-1.0, 0.0, 1.0], &[1.0, 0.0, 1.0], 2);
/// assert_eq!(c.len(), 3);
/// ```
pub fn hermfit(x: &[f64], y: &[f64], deg: usize) -> Array<f64> {
    let c = polyutils::generic_fit(hermvander, x, y, deg);
    Array::from_vec(c.clone(), &[c.len()])
}

/// Compute the roots of a Hermite series, like `np.polynomial.hermite.hermroots(c)`.
///
/// Returns `(re, im)` arrays.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermroots;
/// let (re, _) = hermroots(&[0.0, 0.0, 1.0]);
/// assert_eq!(re.len(), 2);
/// ```
pub fn hermroots(c: &[f64]) -> (Array<f64>, Array<f64>) {
    let trimmed = polyutils::trimseq(c, 0.0);
    if trimmed.len() <= 1 {
        return (
            Array::from_vec(Vec::new(), &[0]),
            Array::from_vec(Vec::new(), &[0]),
        );
    }
    let comp = hermcompanion(&trimmed);
    crate::linalg::eigvals(&comp).expect("eigvals failed")
}

/// Differentiate a Hermite series m times, like `np.polynomial.hermite.hermder(c, m)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermder;
/// let d = hermder(&[1.0, 0.5, 0.5, 0.5], 1);
/// assert_eq!(d.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub fn hermder(c: &[f64], m: usize) -> Array<f64> {
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
        for j in 1..n {
            der[j - 1] = (2 * j) as f64 * current[j];
        }
        current = der;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Integrate a Hermite series m times, like `np.polynomial.hermite.hermint(c, m, k, lbnd)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermint;
/// let res = hermint(&[1.0, 2.0, 3.0], 1, &[0.0], 0.0);
/// assert_eq!(res.len(), 4);
/// ```
pub fn hermint(c: &[f64], m: usize, k: &[f64], lbnd: f64) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(c.to_vec(), &[c.len()]);
    }
    let mut current = c.to_vec();
    for step in 0..m {
        let k_val = if step < k.len() { k[step] } else { 0.0 };
        let n = current.len();
        let mut tmp = vec![0.0; n + 1];
        tmp[1] = current[0] / 2.0;
        for j in 1..n {
            tmp[j + 1] = current[j] / (2.0 * (j + 1) as f64);
        }
        let val_at_lbnd = hermval(lbnd, &tmp);
        tmp[0] += k_val - val_at_lbnd;
        current = tmp;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Add two Hermite series, like `np.polynomial.hermite.hermadd(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermadd;
/// let res = hermadd(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
/// assert_eq!(res.to_vec(), vec![4.0, 6.0, 5.0]);
/// ```
pub fn hermadd(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polyadd(c1, c2)
}

/// Subtract two Hermite series, like `np.polynomial.hermite.hermsub(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermsub;
/// let res = hermsub(&[3.0, 5.0, 5.0], &[1.0, 2.0]);
/// assert_eq!(res.to_vec(), vec![2.0, 3.0, 5.0]);
/// ```
pub fn hermsub(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polysub(c1, c2)
}

/// Multiply a Hermite series by x, like `np.polynomial.hermite.hermmulx(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermmulx;
/// let res = hermmulx(&[1.0, 2.0, 3.0]);
/// assert_eq!(res.to_vec(), vec![2.0, 6.5, 1.0, 1.5]);
/// ```
pub fn hermmulx(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    if c.len() == 1 && c[0] == 0.0 {
        return Array::from_vec(vec![0.0], &[1]);
    }
    let mut prd = vec![0.0; c.len() + 1];
    prd[1] = c[0] / 2.0;
    for i in 1..c.len() {
        prd[i + 1] += c[i] / 2.0;
        prd[i - 1] += c[i] * i as f64;
    }
    let trimmed = polyutils::trimseq(&prd, 0.0);
    Array::from_vec(trimmed.clone(), &[trimmed.len()])
}

/// Multiply one Hermite series by another, like `np.polynomial.hermite.hermmul(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermmul;
/// let res = hermmul(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]);
/// assert_eq!(res.to_vec(), vec![52.0, 29.0, 52.0, 7.0, 6.0]);
/// ```
pub fn hermmul(c1: &[f64], c2: &[f64]) -> Array<f64> {
    let t1 = polyutils::trimseq(c1, 0.0);
    let t2 = polyutils::trimseq(c2, 0.0);
    let (c, xs) = if t1.len() > t2.len() {
        (t2, t1)
    } else {
        (t1, t2)
    };
    if (c.len() == 1 && c[0] == 0.0) || (xs.len() == 1 && xs[0] == 0.0) {
        return Array::from_vec(vec![0.0], &[1]);
    }
    if c.len() == 1 {
        let res: Vec<f64> = xs.iter().map(|&x| c[0] * x).collect();
        return Array::from_vec(res.clone(), &[res.len()]);
    }
    if c.len() == 2 {
        let c0: Vec<f64> = xs.iter().map(|&x| c[0] * x).collect();
        let c1: Vec<f64> = xs.iter().map(|&x| c[1] * x).collect();
        let mx = hermmulx(&c1).to_vec();
        let mx2: Vec<f64> = mx.iter().map(|&x| 2.0 * x).collect();
        return hermadd(&c0, &mx2);
    }
    let mut nd = c.len();
    let mut c0: Vec<f64> = xs.iter().map(|&x| c[nd - 2] * x).collect();
    let mut c1: Vec<f64> = xs.iter().map(|&x| c[nd - 1] * x).collect();
    for i in 3..=c.len() {
        let tmp = c0;
        nd -= 1;
        let c_scale = (2 * (nd - 1)) as f64;
        let c1_scaled: Vec<f64> = c1.iter().map(|&x| x * c_scale).collect();
        let c_i_xs: Vec<f64> = xs.iter().map(|&x| c[c.len() - i] * x).collect();
        c0 = hermsub(&c_i_xs, &c1_scaled).to_vec();

        let mx = hermmulx(&c1).to_vec();
        let mx2: Vec<f64> = mx.iter().map(|&x| 2.0 * x).collect();
        c1 = hermadd(&tmp, &mx2).to_vec();
    }
    let mx = hermmulx(&c1).to_vec();
    let mx2: Vec<f64> = mx.iter().map(|&x| 2.0 * x).collect();
    hermadd(&c0, &mx2)
}

/// Divide two Hermite series, returning quotient and remainder, like `np.polynomial.hermite.hermdiv(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermdiv;
/// let (q, r) = hermdiv(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
/// assert_eq!(q.len(), 1);
/// ```
pub fn hermdiv(c1: &[f64], c2: &[f64]) -> (Array<f64>, Array<f64>) {
    let (quo, rem) = polyutils::generic_div(|a, b| hermmul(a, b).to_vec(), c1, c2);
    (
        Array::from_vec(quo.clone(), &[quo.len()]),
        Array::from_vec(rem.clone(), &[rem.len()]),
    )
}

/// Raise a Hermite series to an integer power, like `np.polynomial.hermite.hermpow(c, power)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermpow;
/// let p = hermpow(&[0.0, 1.0], 2);
/// assert_eq!(p.len(), 3);
/// ```
pub fn hermpow(c: &[f64], power: usize) -> Array<f64> {
    let res = polyutils::generic_pow(|a, b| hermmul(a, b).to_vec(), c, power);
    Array::from_vec(res.clone(), &[res.len()])
}

/// Linear Hermite series off + scl*x, like `np.polynomial.hermite.hermline(off, scl)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermline;
/// let l = hermline(2.0, 3.0);
/// assert_eq!(l.to_vec(), vec![2.0, 1.5]);
/// ```
pub fn hermline(off: f64, scl: f64) -> Array<f64> {
    Array::from_vec(vec![off, 0.5 * scl], &[2])
}

/// Generate a Hermite series with the specified roots, like `np.polynomial.hermite.hermfromroots(roots)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermfromroots;
/// let p = hermfromroots(&[0.0]);
/// assert_eq!(p.to_vec(), vec![0.0, 0.5]);
/// ```
pub fn hermfromroots(roots: &[f64]) -> Array<f64> {
    let p = super::polynomial::polyfromroots(roots);
    poly2herm(&p.to_vec())
}

/// Trim small trailing coefficients from Hermite series, like `np.polynomial.hermite.hermtrim(c, tol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermtrim;
/// let t = hermtrim(&[1.0, 2.0, 0.0], 0.0);
/// assert_eq!(t.to_vec(), vec![1.0, 2.0]);
/// ```
pub fn hermtrim(c: &[f64], tol: f64) -> Array<f64> {
    let t = polyutils::trimseq(c, tol);
    Array::from_vec(t.clone(), &[t.len()])
}

fn normed_hermite_n(x: f64, n: usize) -> f64 {
    let pi_fourth = std::f64::consts::PI.powf(0.25);
    if n == 0 {
        return 1.0 / pi_fourth;
    }
    let mut c0 = 0.0;
    let mut c1 = 1.0 / pi_fourth;
    let mut nd = n as f64;
    for _ in 0..n - 1 {
        let tmp = c0;
        c0 = -c1 * ((nd - 1.0) / nd).sqrt();
        c1 = tmp + c1 * x * (2.0 / nd).sqrt();
        nd -= 1.0;
    }
    c0 + c1 * x * (2.0f64).sqrt()
}

/// Gauss-Hermite quadrature nodes and weights, like `np.polynomial.hermite.hermgauss(deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermgauss;
/// let (x, w) = hermgauss(2);
/// assert_eq!(x.len(), 2);
/// assert_eq!(w.len(), 2);
/// ```
pub fn hermgauss(deg: usize) -> (Array<f64>, Array<f64>) {
    assert!(deg >= 1, "deg must be a positive integer");
    let mut c = vec![0.0; deg + 1];
    c[deg] = 1.0;
    let m = hermcompanion(&c);
    let mut x = crate::linalg::eigvalsh(&m).to_vec();

    for xi in x.iter_mut() {
        let dy = normed_hermite_n(*xi, deg);
        let df = normed_hermite_n(*xi, deg - 1) * (2.0 * deg as f64).sqrt();
        *xi -= dy / df;
    }

    let mut fm = vec![0.0; deg];
    for i in 0..deg {
        fm[i] = normed_hermite_n(x[i], deg - 1);
    }
    let max_fm = fm.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut w = vec![0.0; deg];
    for i in 0..deg {
        let f_scaled = fm[i] / max_fm;
        w[i] = 1.0 / (f_scaled * f_scaled);
    }

    for i in 0..deg / 2 {
        let j = deg - 1 - i;
        let w_sym = (w[i] + w[j]) / 2.0;
        w[i] = w_sym;
        w[j] = w_sym;
        let x_sym = (x[i] - x[j]) / 2.0;
        x[i] = x_sym;
        x[j] = -x_sym;
    }
    if deg % 2 == 1 {
        x[deg / 2] = 0.0;
    }

    let w_sum: f64 = w.iter().sum();
    let target = std::f64::consts::PI.sqrt();
    for wi in w.iter_mut() {
        *wi *= target / w_sum;
    }

    (Array::from_vec(x, &[deg]), Array::from_vec(w, &[deg]))
}

/// Hermite weight function exp(-x^2), like `np.polynomial.hermite.hermweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::hermweight;
/// assert_eq!(hermweight(0.0), 1.0);
/// ```
pub fn hermweight(x: f64) -> f64 {
    (-x * x).exp()
}

/// Evaluate Hermite weight function over array, like `np.polynomial.hermite.hermweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::hermite::hermweight_array;
/// let x = np::array![0.0];
/// assert_eq!(hermweight_array(&x).to_vec(), vec![1.0]);
/// ```
pub fn hermweight_array(x: &Array<f64>) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(hermweight).collect();
    Array::from_vec(data, x.shape())
}

/// Convert a polynomial in standard basis to Hermite series, like `np.polynomial.hermite.poly2herm(pol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::poly2herm;
/// let c = poly2herm(&[0.0, 1.0, 2.0, 3.0]);
/// assert_eq!(c.to_vec(), vec![1.0, 2.75, 0.5, 0.375]);
/// ```
pub fn poly2herm(pol: &[f64]) -> Array<f64> {
    let pol = polyutils::trimseq(pol, 0.0);
    let mut res = vec![0.0];
    for &coeff in pol.iter().rev() {
        let mx = hermmulx(&res).to_vec();
        res = hermadd(&mx, &[coeff]).to_vec();
    }
    Array::from_vec(res.clone(), &[res.len()])
}

/// Convert a Hermite series to a standard polynomial, like `np.polynomial.hermite.herm2poly(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite::herm2poly;
/// let p = herm2poly(&[1.0, 2.75, 0.5, 0.375]);
/// assert_eq!(p.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
/// ```
pub fn herm2poly(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    let n = c.len();
    if n <= 1 {
        return Array::from_vec(c.clone(), &[c.len()]);
    }
    if n == 2 {
        return Array::from_vec(vec![c[0], 2.0 * c[1]], &[2]);
    }
    let mut c0 = vec![c[n - 2]];
    let mut c1 = vec![c[n - 1]];
    for i in (2..=n - 1).rev() {
        let tmp = c0;
        let c1_scaled: Vec<f64> = c1.iter().map(|&x| x * (2 * (i - 1)) as f64).collect();
        c0 = super::polynomial::polysub(&[c[i - 2]], &c1_scaled).to_vec();

        let mx = super::polynomial::polymulx(&c1).to_vec();
        let mx2: Vec<f64> = mx.iter().map(|&x| 2.0 * x).collect();
        c1 = super::polynomial::polyadd(&tmp, &mx2).to_vec();
    }
    let mx = super::polynomial::polymulx(&c1).to_vec();
    let mx2: Vec<f64> = mx.iter().map(|&x| 2.0 * x).collect();
    super::polynomial::polyadd(&c0, &mx2)
}
