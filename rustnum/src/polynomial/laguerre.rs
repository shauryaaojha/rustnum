//! Laguerre series, like `numpy.polynomial.laguerre`.

use super::polyutils;
use crate::Array;

/// Evaluate a Laguerre series at points x using Clenshaw recurrence, like `np.polynomial.laguerre.lagval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagval;
/// // L_0(0.5) = 1, L_1(0.5) = 1 - 0.5 = 0.5
/// assert_eq!(lagval(0.5, &[1.0, 2.0]), 2.0);
/// ```
pub fn lagval(x: f64, c: &[f64]) -> f64 {
    let n = c.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return c[0];
    }
    if n == 2 {
        return c[0] + c[1] * (1.0 - x);
    }
    let mut nd = n;
    let mut c0 = c[n - 2];
    let mut c1 = c[n - 1];
    for i in 3..=n {
        let tmp = c0;
        nd -= 1;
        c0 = c[n - i] - (c1 * (nd - 1) as f64) / nd as f64;
        c1 = tmp + (c1 * (2 * nd - 1) as f64 - c1 * x) / nd as f64;
    }
    c0 + c1 * (1.0 - x)
}

/// Evaluate a Laguerre series over an array of points, like `np.polynomial.laguerre.lagval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::laguerre::lagval_array;
/// let x = np::array![0.0, 1.0];
/// let y = lagval_array(&x, &[1.0, 2.0]);
/// assert_eq!(y.to_vec(), vec![3.0, 1.0]);
/// ```
pub fn lagval_array(x: &Array<f64>, c: &[f64]) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(|xi| lagval(xi, c)).collect();
    Array::from_vec(data, x.shape())
}

/// Pseudo-Vandermonde matrix of Laguerre polynomials, like `np.polynomial.laguerre.lagvander(x, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagvander;
/// let v = lagvander(&[0.0, 1.0], 1);
/// assert_eq!(v.to_vec(), vec![1.0, 1.0, 1.0, 0.0]);
/// ```
pub fn lagvander(x: &[f64], deg: usize) -> Array<f64> {
    let m = x.len();
    let n = deg + 1;
    let mut data = vec![0.0; m * n];
    for (i, &xi) in x.iter().enumerate() {
        data[i * n] = 1.0;
        if deg > 0 {
            data[i * n + 1] = 1.0 - xi;
            for j in 2..n {
                data[i * n + j] = (data[i * n + (j - 1)] * ((2 * j - 1) as f64 - xi)
                    - data[i * n + (j - 2)] * (j - 1) as f64)
                    / j as f64;
            }
        }
    }
    Array::from_vec(data, &[m, n])
}

/// Companion matrix of Laguerre series, like `np.polynomial.laguerre.lagcompanion(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagcompanion;
/// let m = lagcompanion(&[1.0, 2.0, 3.0]);
/// assert_eq!(m.shape(), &[2, 2]);
/// ```
pub fn lagcompanion(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    assert!(
        c.len() >= 2,
        "Series must have maximum degree of at least 1."
    );
    if c.len() == 2 {
        return Array::from_vec(vec![1.0 + c[0] / c[1]], &[1, 1]);
    }
    let n = c.len() - 1;
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        mat[i * n + i] = 2.0 * i as f64 + 1.0;
    }
    for i in 1..n {
        let v = -(i as f64);
        mat[(i - 1) * n + i] = v;
        mat[i * n + (i - 1)] = v;
    }
    let last = c[n];
    for i in 0..n {
        mat[i * n + (n - 1)] += (c[i] / last) * n as f64;
    }
    Array::from_vec(mat, &[n, n])
}

/// Least-squares fit of Laguerre series to data, like `np.polynomial.laguerre.lagfit(x, y, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagfit;
/// let c = lagfit(&[0.0, 1.0, 2.0], &[1.0, 0.0, 1.0], 2);
/// assert_eq!(c.len(), 3);
/// ```
pub fn lagfit(x: &[f64], y: &[f64], deg: usize) -> Array<f64> {
    let c = polyutils::generic_fit(lagvander, x, y, deg);
    Array::from_vec(c.clone(), &[c.len()])
}

/// Compute the roots of a Laguerre series, like `np.polynomial.laguerre.lagroots(c)`.
///
/// Returns `(re, im)` arrays.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagroots;
/// let (re, _) = lagroots(&[1.0, -1.0]); // L_1(x) = 1 - x -> root = 1
/// assert_eq!(re.len(), 1);
/// ```
pub fn lagroots(c: &[f64]) -> (Array<f64>, Array<f64>) {
    let trimmed = polyutils::trimseq(c, 0.0);
    if trimmed.len() <= 1 {
        return (
            Array::from_vec(Vec::new(), &[0]),
            Array::from_vec(Vec::new(), &[0]),
        );
    }
    let comp = lagcompanion(&trimmed);
    crate::linalg::eigvals(&comp).expect("eigvals failed")
}

/// Differentiate a Laguerre series m times, like `np.polynomial.laguerre.lagder(c, m)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagder;
/// let d = lagder(&[1.0, 1.0, 1.0, -3.0], 1);
/// assert_eq!(d.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub fn lagder(c: &[f64], m: usize) -> Array<f64> {
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
        for j in (2..n).rev() {
            der[j - 1] = -current[j];
            current[j - 1] += current[j];
        }
        der[0] = -current[1];
        current = der;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Integrate a Laguerre series m times, like `np.polynomial.laguerre.lagint(c, m, k, lbnd)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagint;
/// let res = lagint(&[1.0, 2.0, 3.0], 1, &[1.0], 0.0);
/// assert_eq!(res.len(), 4);
/// ```
pub fn lagint(c: &[f64], m: usize, k: &[f64], lbnd: f64) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(c.to_vec(), &[c.len()]);
    }
    let mut current = c.to_vec();
    for step in 0..m {
        let k_val = if step < k.len() { k[step] } else { 0.0 };
        let n = current.len();
        let mut tmp = vec![0.0; n + 1];
        tmp[0] = current[0];
        tmp[1] = -current[0];
        for j in 1..n {
            tmp[j] += current[j];
            tmp[j + 1] = -current[j];
        }
        let val_at_lbnd = lagval(lbnd, &tmp);
        tmp[0] += k_val - val_at_lbnd;
        current = tmp;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Add two Laguerre series, like `np.polynomial.laguerre.lagadd(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagadd;
/// let res = lagadd(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
/// assert_eq!(res.to_vec(), vec![4.0, 6.0, 5.0]);
/// ```
pub fn lagadd(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polyadd(c1, c2)
}

/// Subtract two Laguerre series, like `np.polynomial.laguerre.lagsub(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagsub;
/// let res = lagsub(&[3.0, 5.0, 5.0], &[1.0, 2.0]);
/// assert_eq!(res.to_vec(), vec![2.0, 3.0, 5.0]);
/// ```
pub fn lagsub(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polysub(c1, c2)
}

/// Multiply a Laguerre series by x, like `np.polynomial.laguerre.lagmulx(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagmulx;
/// let res = lagmulx(&[1.0, 2.0, 3.0]);
/// assert_eq!(res.to_vec(), vec![-1.0, -1.0, 11.0, -9.0]);
/// ```
pub fn lagmulx(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    if c.len() == 1 && c[0] == 0.0 {
        return Array::from_vec(vec![0.0], &[1]);
    }
    let mut prd = vec![0.0; c.len() + 1];
    prd[0] = c[0];
    prd[1] = -c[0];
    for i in 1..c.len() {
        prd[i + 1] = -c[i] * (i + 1) as f64;
        prd[i] += c[i] * (2 * i + 1) as f64;
        prd[i - 1] -= c[i] * i as f64;
    }
    let trimmed = polyutils::trimseq(&prd, 0.0);
    Array::from_vec(trimmed.clone(), &[trimmed.len()])
}

/// Multiply one Laguerre series by another, like `np.polynomial.laguerre.lagmul(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagmul;
/// let res = lagmul(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]);
/// assert_eq!(res.to_vec(), vec![8.0, -13.0, 38.0, -51.0, 36.0]);
/// ```
pub fn lagmul(c1: &[f64], c2: &[f64]) -> Array<f64> {
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
        let mx = lagmulx(&c1).to_vec();
        let diff = lagsub(&c1, &mx).to_vec();
        return lagadd(&c0, &diff);
    }
    let mut nd = c.len();
    let mut c0: Vec<f64> = xs.iter().map(|&x| c[nd - 2] * x).collect();
    let mut c1: Vec<f64> = xs.iter().map(|&x| c[nd - 1] * x).collect();
    for i in 3..=c.len() {
        let tmp = c0;
        nd -= 1;
        let c_scale = (nd - 1) as f64 / nd as f64;
        let c1_scaled: Vec<f64> = c1.iter().map(|&x| x * c_scale).collect();
        let c_i_xs: Vec<f64> = xs.iter().map(|&x| c[c.len() - i] * x).collect();
        c0 = lagsub(&c_i_xs, &c1_scaled).to_vec();

        let c1_factor = (2 * nd - 1) as f64;
        let c1_term: Vec<f64> = c1.iter().map(|&x| x * c1_factor).collect();
        let mx = lagmulx(&c1).to_vec();
        let diff = lagsub(&c1_term, &mx).to_vec();
        let diff_scaled: Vec<f64> = diff.iter().map(|&x| x / nd as f64).collect();
        c1 = lagadd(&tmp, &diff_scaled).to_vec();
    }
    let mx = lagmulx(&c1).to_vec();
    let diff = lagsub(&c1, &mx).to_vec();
    lagadd(&c0, &diff)
}

/// Divide two Laguerre series, returning quotient and remainder, like `np.polynomial.laguerre.lagdiv(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagdiv;
/// let (q, r) = lagdiv(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
/// assert_eq!(q.len(), 1);
/// ```
pub fn lagdiv(c1: &[f64], c2: &[f64]) -> (Array<f64>, Array<f64>) {
    let (quo, rem) = polyutils::generic_div(|a, b| lagmul(a, b).to_vec(), c1, c2);
    (
        Array::from_vec(quo.clone(), &[quo.len()]),
        Array::from_vec(rem.clone(), &[rem.len()]),
    )
}

/// Raise a Laguerre series to an integer power, like `np.polynomial.laguerre.lagpow(c, power)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagpow;
/// let p = lagpow(&[0.0, 1.0], 2);
/// assert_eq!(p.len(), 3);
/// ```
pub fn lagpow(c: &[f64], power: usize) -> Array<f64> {
    let res = polyutils::generic_pow(|a, b| lagmul(a, b).to_vec(), c, power);
    Array::from_vec(res.clone(), &[res.len()])
}

/// Linear Laguerre series off + scl*x, like `np.polynomial.laguerre.lagline(off, scl)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagline;
/// let l = lagline(2.0, 3.0);
/// assert_eq!(l.to_vec(), vec![5.0, -3.0]);
/// ```
pub fn lagline(off: f64, scl: f64) -> Array<f64> {
    // 2 + 3*x = 2*L_0 + 3*(L_0 - L_1) = 5*L_0 - 3*L_1
    Array::from_vec(vec![off + scl, -scl], &[2])
}

/// Generate a Laguerre series with the specified roots, like `np.polynomial.laguerre.lagfromroots(roots)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagfromroots;
/// let p = lagfromroots(&[1.0]);
/// assert_eq!(p.to_vec(), vec![0.0, -1.0]);
/// ```
pub fn lagfromroots(roots: &[f64]) -> Array<f64> {
    let p = super::polynomial::polyfromroots(roots);
    poly2lag(&p.to_vec())
}

/// Trim small trailing coefficients from Laguerre series, like `np.polynomial.laguerre.lagtrim(c, tol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagtrim;
/// let t = lagtrim(&[1.0, 2.0, 0.0], 0.0);
/// assert_eq!(t.to_vec(), vec![1.0, 2.0]);
/// ```
pub fn lagtrim(c: &[f64], tol: f64) -> Array<f64> {
    let t = polyutils::trimseq(c, tol);
    Array::from_vec(t.clone(), &[t.len()])
}

/// Gauss-Laguerre quadrature nodes and weights, like `np.polynomial.laguerre.laggauss(deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::laggauss;
/// let (x, w) = laggauss(2);
/// assert_eq!(x.len(), 2);
/// assert_eq!(w.len(), 2);
/// ```
pub fn laggauss(deg: usize) -> (Array<f64>, Array<f64>) {
    assert!(deg >= 1, "deg must be a positive integer");
    let mut c = vec![0.0; deg + 1];
    c[deg] = 1.0;
    let m = lagcompanion(&c);
    let mut x = crate::linalg::eigvalsh(&m).to_vec();

    let c_der = lagder(&c, 1).to_vec();
    for xi in x.iter_mut() {
        let dy = lagval(*xi, &c);
        let df = lagval(*xi, &c_der);
        *xi -= dy / df;
    }

    let mut fm = vec![0.0; deg];
    let mut df = vec![0.0; deg];
    for i in 0..deg {
        fm[i] = lagval(x[i], &c[1..]);
        df[i] = lagval(x[i], &c_der);
    }
    let max_fm = fm.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let max_df = df.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut w = vec![0.0; deg];
    for i in 0..deg {
        let f_scaled = fm[i] / max_fm;
        let d_scaled = df[i] / max_df;
        w[i] = 1.0 / (f_scaled * d_scaled);
    }

    let w_sum: f64 = w.iter().sum();
    for wi in w.iter_mut() {
        *wi /= w_sum;
    }

    (Array::from_vec(x, &[deg]), Array::from_vec(w, &[deg]))
}

/// Laguerre weight function exp(-x), like `np.polynomial.laguerre.lagweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lagweight;
/// assert_eq!(lagweight(0.0), 1.0);
/// ```
pub fn lagweight(x: f64) -> f64 {
    (-x).exp()
}

/// Evaluate Laguerre weight function over array, like `np.polynomial.laguerre.lagweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::laguerre::lagweight_array;
/// let x = np::array![0.0];
/// assert_eq!(lagweight_array(&x).to_vec(), vec![1.0]);
/// ```
pub fn lagweight_array(x: &Array<f64>) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(lagweight).collect();
    Array::from_vec(data, x.shape())
}

/// Convert a polynomial in standard basis to Laguerre series, like `np.polynomial.laguerre.poly2lag(pol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::poly2lag;
/// let c = poly2lag(&[0.0, 1.0, 2.0, 3.0]);
/// assert_eq!(c.to_vec(), vec![23.0, -63.0, 58.0, -18.0]);
/// ```
pub fn poly2lag(pol: &[f64]) -> Array<f64> {
    let pol = polyutils::trimseq(pol, 0.0);
    let mut res = vec![0.0];
    for &coeff in pol.iter().rev() {
        let mx = lagmulx(&res).to_vec();
        res = lagadd(&mx, &[coeff]).to_vec();
    }
    Array::from_vec(res.clone(), &[res.len()])
}

/// Convert a Laguerre series to a standard polynomial, like `np.polynomial.laguerre.lag2poly(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::laguerre::lag2poly;
/// let p = lag2poly(&[23.0, -63.0, 58.0, -18.0]);
/// assert_eq!(p.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
/// ```
pub fn lag2poly(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    let n = c.len();
    if n <= 1 {
        return Array::from_vec(c.clone(), &[c.len()]);
    }
    let mut c0 = vec![c[n - 2]];
    let mut c1 = vec![c[n - 1]];
    for i in (2..=n - 1).rev() {
        let tmp = c0;
        let c1_scaled: Vec<f64> = c1
            .iter()
            .map(|&x| (x * (i - 1) as f64) / i as f64)
            .collect();
        c0 = super::polynomial::polysub(&[c[i - 2]], &c1_scaled).to_vec();

        let c1_term: Vec<f64> = c1.iter().map(|&x| x * (2 * i - 1) as f64).collect();
        let mx = super::polynomial::polymulx(&c1).to_vec();
        let diff = super::polynomial::polysub(&c1_term, &mx).to_vec();
        let diff_scaled: Vec<f64> = diff.iter().map(|&x| x / i as f64).collect();
        c1 = super::polynomial::polyadd(&tmp, &diff_scaled).to_vec();
    }
    let mx = super::polynomial::polymulx(&c1).to_vec();
    let diff = super::polynomial::polysub(&c1, &mx).to_vec();
    super::polynomial::polyadd(&c0, &diff)
}
