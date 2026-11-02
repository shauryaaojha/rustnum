//! Probabilists' Hermite series, like `numpy.polynomial.hermite_e`.

use super::polyutils;
use crate::Array;

/// Evaluate a HermiteE series at points x using Clenshaw recurrence, like `np.polynomial.hermite_e.hermeval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermeval;
/// // He_0(0.5) = 1, He_1(0.5) = 0.5, He_2(0.5) = 0.25 - 1 = -0.75
/// assert_eq!(hermeval(0.5, &[1.0, 2.0]), 2.0);
/// ```
pub fn hermeval(x: f64, c: &[f64]) -> f64 {
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
    let mut nd = n;
    let mut c0 = c[n - 2];
    let mut c1 = c[n - 1];
    for i in 3..=n {
        let tmp = c0;
        nd -= 1;
        c0 = c[n - i] - c1 * (nd - 1) as f64;
        c1 = tmp + c1 * x;
    }
    c0 + c1 * x
}

/// Evaluate a HermiteE series over an array of points, like `np.polynomial.hermite_e.hermeval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::hermite_e::hermeval_array;
/// let x = np::array![0.0, 1.0];
/// let y = hermeval_array(&x, &[1.0, 2.0]);
/// assert_eq!(y.to_vec(), vec![1.0, 3.0]);
/// ```
pub fn hermeval_array(x: &Array<f64>, c: &[f64]) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(|xi| hermeval(xi, c)).collect();
    Array::from_vec(data, x.shape())
}

/// Pseudo-Vandermonde matrix of HermiteE polynomials, like `np.polynomial.hermite_e.hermevander(x, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermevander;
/// let v = hermevander(&[-1.0, 0.0, 1.0], 2);
/// assert_eq!(v.shape(), &[3, 3]);
/// ```
pub fn hermevander(x: &[f64], deg: usize) -> Array<f64> {
    let m = x.len();
    let n = deg + 1;
    let mut data = vec![0.0; m * n];
    for (i, &xi) in x.iter().enumerate() {
        data[i * n] = 1.0;
        if deg > 0 {
            data[i * n + 1] = xi;
            for j in 2..n {
                data[i * n + j] =
                    data[i * n + (j - 1)] * xi - data[i * n + (j - 2)] * (j - 1) as f64;
            }
        }
    }
    Array::from_vec(data, &[m, n])
}

/// Scaled companion matrix of HermiteE series, like `np.polynomial.hermite_e.hermecompanion(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermecompanion;
/// let m = hermecompanion(&[1.0, 0.0, 1.0]);
/// assert_eq!(m.shape(), &[2, 2]);
/// ```
pub fn hermecompanion(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    assert!(
        c.len() >= 2,
        "Series must have maximum degree of at least 1."
    );
    if c.len() == 2 {
        return Array::from_vec(vec![-c[0] / c[1]], &[1, 1]);
    }
    let n = c.len() - 1;
    let mut mat = vec![0.0; n * n];
    let mut scl = vec![0.0; n];
    scl[n - 1] = 1.0;
    for i in (0..n - 1).rev() {
        scl[i] = scl[i + 1] / ((i + 1) as f64).sqrt();
    }
    for i in 1..n {
        let v = (i as f64).sqrt();
        mat[(i - 1) * n + i] = v;
        mat[i * n + (i - 1)] = v;
    }
    let last = c[n];
    for i in 0..n {
        mat[i * n + (n - 1)] -= scl[i] * c[i] / last;
    }
    Array::from_vec(mat, &[n, n])
}

/// Least-squares fit of HermiteE series to data, like `np.polynomial.hermite_e.hermefit(x, y, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermefit;
/// let c = hermefit(&[-1.0, 0.0, 1.0], &[1.0, 0.0, 1.0], 2);
/// assert_eq!(c.len(), 3);
/// ```
pub fn hermefit(x: &[f64], y: &[f64], deg: usize) -> Array<f64> {
    let c = polyutils::generic_fit(hermevander, x, y, deg);
    Array::from_vec(c.clone(), &[c.len()])
}

/// Compute the roots of a HermiteE series, like `np.polynomial.hermite_e.hermeroots(c)`.
///
/// Returns `(re, im)` arrays.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermeroots;
/// let (re, _) = hermeroots(&[0.0, 0.0, 1.0]);
/// assert_eq!(re.len(), 2);
/// ```
pub fn hermeroots(c: &[f64]) -> (Array<f64>, Array<f64>) {
    let trimmed = polyutils::trimseq(c, 0.0);
    if trimmed.len() <= 1 {
        return (
            Array::from_vec(Vec::new(), &[0]),
            Array::from_vec(Vec::new(), &[0]),
        );
    }
    let comp = hermecompanion(&trimmed);
    crate::linalg::eigvals(&comp).expect("eigvals failed")
}

/// Differentiate a HermiteE series m times, like `np.polynomial.hermite_e.hermeder(c, m)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermeder;
/// let d = hermeder(&[1.0, 1.0, 1.0, 1.0], 1);
/// assert_eq!(d.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub fn hermeder(c: &[f64], m: usize) -> Array<f64> {
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
            der[j - 1] = j as f64 * current[j];
        }
        current = der;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Integrate a HermiteE series m times, like `np.polynomial.hermite_e.hermeint(c, m, k, lbnd)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermeint;
/// let res = hermeint(&[1.0, 2.0, 3.0], 1, &[0.0], 0.0);
/// assert_eq!(res.len(), 4);
/// ```
pub fn hermeint(c: &[f64], m: usize, k: &[f64], lbnd: f64) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(c.to_vec(), &[c.len()]);
    }
    let mut current = c.to_vec();
    for step in 0..m {
        let k_val = if step < k.len() { k[step] } else { 0.0 };
        let n = current.len();
        let mut tmp = vec![0.0; n + 1];
        tmp[1] = current[0];
        for j in 1..n {
            tmp[j + 1] = current[j] / (j + 1) as f64;
        }
        let val_at_lbnd = hermeval(lbnd, &tmp);
        tmp[0] += k_val - val_at_lbnd;
        current = tmp;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Add two HermiteE series, like `np.polynomial.hermite_e.hermeadd(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermeadd;
/// let res = hermeadd(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
/// assert_eq!(res.to_vec(), vec![4.0, 6.0, 5.0]);
/// ```
pub fn hermeadd(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polyadd(c1, c2)
}

/// Subtract two HermiteE series, like `np.polynomial.hermite_e.hermesub(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermesub;
/// let res = hermesub(&[3.0, 5.0, 5.0], &[1.0, 2.0]);
/// assert_eq!(res.to_vec(), vec![2.0, 3.0, 5.0]);
/// ```
pub fn hermesub(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polysub(c1, c2)
}

/// Multiply a HermiteE series by x, like `np.polynomial.hermite_e.hermemulx(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermemulx;
/// let res = hermemulx(&[1.0, 2.0, 3.0]);
/// assert_eq!(res.to_vec(), vec![2.0, 7.0, 2.0, 3.0]);
/// ```
pub fn hermemulx(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    if c.len() == 1 && c[0] == 0.0 {
        return Array::from_vec(vec![0.0], &[1]);
    }
    let mut prd = vec![0.0; c.len() + 1];
    prd[1] = c[0];
    for i in 1..c.len() {
        prd[i + 1] += c[i];
        prd[i - 1] += c[i] * i as f64;
    }
    let trimmed = polyutils::trimseq(&prd, 0.0);
    Array::from_vec(trimmed.clone(), &[trimmed.len()])
}

/// Multiply one HermiteE series by another, like `np.polynomial.hermite_e.hermemul(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermemul;
/// let res = hermemul(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]);
/// assert_eq!(res.to_vec(), vec![14.0, 15.0, 28.0, 7.0, 6.0]);
/// ```
pub fn hermemul(c1: &[f64], c2: &[f64]) -> Array<f64> {
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
        let mx = hermemulx(&c1).to_vec();
        return hermeadd(&c0, &mx);
    }
    let mut nd = c.len();
    let mut c0: Vec<f64> = xs.iter().map(|&x| c[nd - 2] * x).collect();
    let mut c1: Vec<f64> = xs.iter().map(|&x| c[nd - 1] * x).collect();
    for i in 3..=c.len() {
        let tmp = c0;
        nd -= 1;
        let c_scale = (nd - 1) as f64;
        let c1_scaled: Vec<f64> = c1.iter().map(|&x| x * c_scale).collect();
        let c_i_xs: Vec<f64> = xs.iter().map(|&x| c[c.len() - i] * x).collect();
        c0 = hermesub(&c_i_xs, &c1_scaled).to_vec();

        let mx = hermemulx(&c1).to_vec();
        c1 = hermeadd(&tmp, &mx).to_vec();
    }
    let mx = hermemulx(&c1).to_vec();
    hermeadd(&c0, &mx)
}

/// Divide two HermiteE series, returning quotient and remainder, like `np.polynomial.hermite_e.hermediv(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermediv;
/// let (q, r) = hermediv(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
/// assert_eq!(q.len(), 1);
/// ```
pub fn hermediv(c1: &[f64], c2: &[f64]) -> (Array<f64>, Array<f64>) {
    let (quo, rem) = polyutils::generic_div(|a, b| hermemul(a, b).to_vec(), c1, c2);
    (
        Array::from_vec(quo.clone(), &[quo.len()]),
        Array::from_vec(rem.clone(), &[rem.len()]),
    )
}

/// Raise a HermiteE series to an integer power, like `np.polynomial.hermite_e.hermepow(c, power)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermepow;
/// let p = hermepow(&[0.0, 1.0], 2);
/// assert_eq!(p.len(), 3);
/// ```
pub fn hermepow(c: &[f64], power: usize) -> Array<f64> {
    let res = polyutils::generic_pow(|a, b| hermemul(a, b).to_vec(), c, power);
    Array::from_vec(res.clone(), &[res.len()])
}

/// Linear HermiteE series off + scl*x, like `np.polynomial.hermite_e.hermeline(off, scl)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermeline;
/// let l = hermeline(2.0, 3.0);
/// assert_eq!(l.to_vec(), vec![2.0, 3.0]);
/// ```
pub fn hermeline(off: f64, scl: f64) -> Array<f64> {
    Array::from_vec(vec![off, scl], &[2])
}

/// Generate a HermiteE series with the specified roots, like `np.polynomial.hermite_e.hermefromroots(roots)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermefromroots;
/// let p = hermefromroots(&[0.0]);
/// assert_eq!(p.to_vec(), vec![0.0, 1.0]);
/// ```
pub fn hermefromroots(roots: &[f64]) -> Array<f64> {
    let p = super::polynomial::polyfromroots(roots);
    poly2herme(&p.to_vec())
}

/// Trim small trailing coefficients from HermiteE series, like `np.polynomial.hermite_e.hermetrim(c, tol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermetrim;
/// let t = hermetrim(&[1.0, 2.0, 0.0], 0.0);
/// assert_eq!(t.to_vec(), vec![1.0, 2.0]);
/// ```
pub fn hermetrim(c: &[f64], tol: f64) -> Array<f64> {
    let t = polyutils::trimseq(c, tol);
    Array::from_vec(t.clone(), &[t.len()])
}

fn normed_hermite_e_n(x: f64, n: usize) -> f64 {
    let factor = (2.0 * std::f64::consts::PI).powf(0.25);
    if n == 0 {
        return 1.0 / factor;
    }
    let mut c0 = 0.0;
    let mut c1 = 1.0 / factor;
    let mut nd = n as f64;
    for _ in 0..n - 1 {
        let tmp = c0;
        c0 = -c1 * ((nd - 1.0) / nd).sqrt();
        c1 = tmp + c1 * x * (1.0 / nd).sqrt();
        nd -= 1.0;
    }
    c0 + c1 * x
}

/// Gauss-HermiteE quadrature nodes and weights, like `np.polynomial.hermite_e.hermegauss(deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermegauss;
/// let (x, w) = hermegauss(2);
/// assert_eq!(x.len(), 2);
/// assert_eq!(w.len(), 2);
/// ```
pub fn hermegauss(deg: usize) -> (Array<f64>, Array<f64>) {
    assert!(deg >= 1, "deg must be a positive integer");
    let mut c = vec![0.0; deg + 1];
    c[deg] = 1.0;
    let m = hermecompanion(&c);
    let mut x = crate::linalg::eigvalsh(&m).to_vec();

    for xi in x.iter_mut() {
        let dy = normed_hermite_e_n(*xi, deg);
        let df = normed_hermite_e_n(*xi, deg - 1) * (deg as f64).sqrt();
        *xi -= dy / df;
    }

    let mut fm = vec![0.0; deg];
    for i in 0..deg {
        fm[i] = normed_hermite_e_n(x[i], deg - 1);
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
    let target = (2.0 * std::f64::consts::PI).sqrt();
    for wi in w.iter_mut() {
        *wi *= target / w_sum;
    }

    (Array::from_vec(x, &[deg]), Array::from_vec(w, &[deg]))
}

/// HermiteE weight function exp(-x^2 / 2), like `np.polynomial.hermite_e.hermeweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::hermeweight;
/// assert_eq!(hermeweight(0.0), 1.0);
/// ```
pub fn hermeweight(x: f64) -> f64 {
    (-0.5 * x * x).exp()
}

/// Evaluate HermiteE weight function over array, like `np.polynomial.hermite_e.hermeweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::hermite_e::hermeweight_array;
/// let x = np::array![0.0];
/// assert_eq!(hermeweight_array(&x).to_vec(), vec![1.0]);
/// ```
pub fn hermeweight_array(x: &Array<f64>) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(hermeweight).collect();
    Array::from_vec(data, x.shape())
}

/// Convert a polynomial in standard basis to HermiteE series, like `np.polynomial.hermite_e.poly2herme(pol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::poly2herme;
/// let c = poly2herme(&[0.0, 1.0, 2.0, 3.0]);
/// assert_eq!(c.to_vec(), vec![2.0, 10.0, 2.0, 3.0]);
/// ```
pub fn poly2herme(pol: &[f64]) -> Array<f64> {
    let pol = polyutils::trimseq(pol, 0.0);
    let mut res = vec![0.0];
    for &coeff in pol.iter().rev() {
        let mx = hermemulx(&res).to_vec();
        res = hermeadd(&mx, &[coeff]).to_vec();
    }
    Array::from_vec(res.clone(), &[res.len()])
}

/// Convert a HermiteE series to a standard polynomial, like `np.polynomial.hermite_e.herme2poly(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::hermite_e::herme2poly;
/// let p = herme2poly(&[2.0, 10.0, 2.0, 3.0]);
/// assert_eq!(p.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
/// ```
pub fn herme2poly(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    let n = c.len();
    if n <= 2 {
        return Array::from_vec(c.clone(), &[c.len()]);
    }
    let mut c0 = vec![c[n - 2]];
    let mut c1 = vec![c[n - 1]];
    for i in (2..=n - 1).rev() {
        let tmp = c0;
        let c1_scaled: Vec<f64> = c1.iter().map(|&x| x * (i - 1) as f64).collect();
        c0 = super::polynomial::polysub(&[c[i - 2]], &c1_scaled).to_vec();

        let mx = super::polynomial::polymulx(&c1).to_vec();
        c1 = super::polynomial::polyadd(&tmp, &mx).to_vec();
    }
    let mx = super::polynomial::polymulx(&c1).to_vec();
    super::polynomial::polyadd(&c0, &mx)
}
