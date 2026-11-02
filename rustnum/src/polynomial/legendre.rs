//! Legendre series, like `numpy.polynomial.legendre`.

use super::polyutils;
use crate::Array;

/// Evaluate a Legendre series at points x using Clenshaw recurrence, like `np.polynomial.legendre.legval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legval;
/// // P_0(0.5) = 1, P_1(0.5) = 0.5, P_2(0.5) = 0.5*(3*0.25 - 1) = -0.125
/// assert_eq!(legval(0.5, &[1.0, 2.0]), 2.0);
/// ```
pub fn legval(x: f64, c: &[f64]) -> f64 {
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
        c0 = c[n - i] - c1 * ((nd - 1) as f64 / nd as f64);
        c1 = tmp + c1 * x * ((2 * nd - 1) as f64 / nd as f64);
    }
    c0 + c1 * x
}

/// Evaluate a Legendre series over an array of points, like `np.polynomial.legendre.legval(x, c)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::legendre::legval_array;
/// let x = np::array![0.0, 1.0];
/// let y = legval_array(&x, &[1.0, 2.0]);
/// assert_eq!(y.to_vec(), vec![1.0, 3.0]);
/// ```
pub fn legval_array(x: &Array<f64>, c: &[f64]) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(|xi| legval(xi, c)).collect();
    Array::from_vec(data, x.shape())
}

/// Pseudo-Vandermonde matrix of Legendre polynomials, like `np.polynomial.legendre.legvander(x, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legvander;
/// let v = legvander(&[0.5], 2);
/// assert_eq!(v.to_vec(), vec![1.0, 0.5, -0.125]);
/// ```
pub fn legvander(x: &[f64], deg: usize) -> Array<f64> {
    let m = x.len();
    let n = deg + 1;
    let mut data = vec![0.0; m * n];
    for (i, &xi) in x.iter().enumerate() {
        data[i * n] = 1.0;
        if deg > 0 {
            data[i * n + 1] = xi;
            for j in 2..n {
                data[i * n + j] = (data[i * n + (j - 1)] * xi * (2 * j - 1) as f64
                    - data[i * n + (j - 2)] * (j - 1) as f64)
                    / j as f64;
            }
        }
    }
    Array::from_vec(data, &[m, n])
}

/// Scaled companion matrix of Legendre series, like `np.polynomial.legendre.legcompanion(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legcompanion;
/// let m = legcompanion(&[1.0, 2.0, 3.0]);
/// assert_eq!(m.shape(), &[2, 2]);
/// ```
pub fn legcompanion(c: &[f64]) -> Array<f64> {
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
    for (i, item) in scl.iter_mut().enumerate() {
        *item = 1.0 / ((2 * i + 1) as f64).sqrt();
    }
    for i in 1..n {
        let v = (i as f64) * scl[i - 1] * scl[i];
        mat[(i - 1) * n + i] = v;
        mat[i * n + (i - 1)] = v;
    }
    let last = c[n];
    let last_scl = scl[n - 1];
    let factor = (n as f64) / (2 * n - 1) as f64;
    for i in 0..n {
        mat[i * n + (n - 1)] -= (c[i] / last) * (scl[i] / last_scl) * factor;
    }
    Array::from_vec(mat, &[n, n])
}

/// Least-squares fit of Legendre series to data, like `np.polynomial.legendre.legfit(x, y, deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legfit;
/// let c = legfit(&[-1.0, 0.0, 1.0], &[1.0, 0.0, 1.0], 2);
/// assert_eq!(c.len(), 3);
/// ```
pub fn legfit(x: &[f64], y: &[f64], deg: usize) -> Array<f64> {
    let c = polyutils::generic_fit(legvander, x, y, deg);
    Array::from_vec(c.clone(), &[c.len()])
}

/// Compute the roots of a Legendre series, like `np.polynomial.legendre.legroots(c)`.
///
/// Returns `(re, im)` arrays.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legroots;
/// let (re, _) = legroots(&[0.0, 0.0, 1.0]);
/// assert_eq!(re.len(), 2);
/// ```
pub fn legroots(c: &[f64]) -> (Array<f64>, Array<f64>) {
    let trimmed = polyutils::trimseq(c, 0.0);
    if trimmed.len() <= 1 {
        return (
            Array::from_vec(Vec::new(), &[0]),
            Array::from_vec(Vec::new(), &[0]),
        );
    }
    let comp = legcompanion(&trimmed);
    crate::linalg::eigvals(&comp).expect("eigvals failed")
}

/// Differentiate a Legendre series m times, like `np.polynomial.legendre.legder(c, m)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legder;
/// let d = legder(&[1.0, 2.0, 3.0, 4.0], 1);
/// assert_eq!(d.to_vec(), vec![6.0, 9.0, 20.0]);
/// ```
pub fn legder(c: &[f64], m: usize) -> Array<f64> {
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
            der[j - 1] = (2 * j - 1) as f64 * current[j];
            current[j - 2] += current[j];
        }
        if n > 2 {
            der[1] = 3.0 * current[2];
        }
        der[0] = current[1];
        current = der;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Integrate a Legendre series m times, like `np.polynomial.legendre.legint(c, m, k, lbnd)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legint;
/// let res = legint(&[1.0, 2.0, 3.0], 1, &[0.0], 0.0);
/// assert_eq!(res.shape(), &[4]);
/// ```
pub fn legint(c: &[f64], m: usize, k: &[f64], lbnd: f64) -> Array<f64> {
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
            tmp[2] = current[1] / 3.0;
        }
        for j in 2..n {
            let t = current[j] / (2 * j + 1) as f64;
            tmp[j + 1] = t;
            tmp[j - 1] -= t;
        }
        let val_at_lbnd = legval(lbnd, &tmp);
        tmp[0] += k_val - val_at_lbnd;
        current = tmp;
    }
    Array::from_vec(current.clone(), &[current.len()])
}

/// Add two Legendre series, like `np.polynomial.legendre.legadd(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legadd;
/// let res = legadd(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
/// assert_eq!(res.to_vec(), vec![4.0, 6.0, 5.0]);
/// ```
pub fn legadd(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polyadd(c1, c2)
}

/// Subtract two Legendre series, like `np.polynomial.legendre.legsub(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legsub;
/// let res = legsub(&[3.0, 5.0, 5.0], &[1.0, 2.0]);
/// assert_eq!(res.to_vec(), vec![2.0, 3.0, 5.0]);
/// ```
pub fn legsub(c1: &[f64], c2: &[f64]) -> Array<f64> {
    super::polynomial::polysub(c1, c2)
}

/// Multiply a Legendre series by x, like `np.polynomial.legendre.legmulx(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legmulx;
/// let res = legmulx(&[1.0, 2.0, 3.0]);
/// assert_eq!(res.len(), 4);
/// ```
pub fn legmulx(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    if c.len() == 1 && c[0] == 0.0 {
        return Array::from_vec(vec![0.0], &[1]);
    }
    let mut prd = vec![0.0; c.len() + 1];
    prd[1] = c[0];
    for (i, &c_i) in c.iter().enumerate().skip(1) {
        let j = i + 1;
        let k = i - 1;
        let s = (i + j) as f64;
        prd[j] += (c_i * j as f64) / s;
        prd[k] += (c_i * i as f64) / s;
    }
    let trimmed = polyutils::trimseq(&prd, 0.0);
    Array::from_vec(trimmed.clone(), &[trimmed.len()])
}

/// Multiply one Legendre series by another, like `np.polynomial.legendre.legmul(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legmul;
/// let res = legmul(&[1.0, 2.0, 3.0], &[3.0, 2.0]);
/// assert_eq!(res.len(), 4);
/// ```
pub fn legmul(c1: &[f64], c2: &[f64]) -> Array<f64> {
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
        let mx = legmulx(&c1);
        return legadd(&c0, &mx.to_vec());
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
        c0 = legsub(&c_i_xs, &c1_scaled).to_vec();

        let mx_scale = (2 * nd - 1) as f64 / nd as f64;
        let mx = legmulx(&c1).to_vec();
        let mx_scaled: Vec<f64> = mx.iter().map(|&x| x * mx_scale).collect();
        c1 = legadd(&tmp, &mx_scaled).to_vec();
    }
    let mx = legmulx(&c1).to_vec();
    legadd(&c0, &mx)
}

/// Divide two Legendre series, returning quotient and remainder, like `np.polynomial.legendre.legdiv(c1, c2)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legdiv;
/// let (q, r) = legdiv(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
/// assert_eq!(q.len(), 1);
/// ```
pub fn legdiv(c1: &[f64], c2: &[f64]) -> (Array<f64>, Array<f64>) {
    let (quo, rem) = polyutils::generic_div(|a, b| legmul(a, b).to_vec(), c1, c2);
    (
        Array::from_vec(quo.clone(), &[quo.len()]),
        Array::from_vec(rem.clone(), &[rem.len()]),
    )
}

/// Raise a Legendre series to an integer power, like `np.polynomial.legendre.legpow(c, power)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legpow;
/// let p = legpow(&[0.0, 1.0], 2);
/// assert_eq!(p.len(), 3);
/// ```
pub fn legpow(c: &[f64], power: usize) -> Array<f64> {
    let res = polyutils::generic_pow(|a, b| legmul(a, b).to_vec(), c, power);
    Array::from_vec(res.clone(), &[res.len()])
}

/// Linear Legendre series off + scl*x, like `np.polynomial.legendre.legline(off, scl)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legline;
/// let l = legline(2.0, 3.0);
/// assert_eq!(l.to_vec(), vec![2.0, 3.0]);
/// ```
pub fn legline(off: f64, scl: f64) -> Array<f64> {
    Array::from_vec(vec![off, scl], &[2])
}

/// Generate a Legendre series with the specified roots, like `np.polynomial.legendre.legfromroots(roots)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legfromroots;
/// let p = legfromroots(&[1.0, -1.0]);
/// assert_eq!(p.len(), 3);
/// ```
pub fn legfromroots(roots: &[f64]) -> Array<f64> {
    let p = super::polynomial::polyfromroots(roots);
    poly2leg(&p.to_vec())
}

/// Trim small trailing coefficients from Legendre series, like `np.polynomial.legendre.legtrim(c, tol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legtrim;
/// let t = legtrim(&[1.0, 2.0, 0.0], 0.0);
/// assert_eq!(t.to_vec(), vec![1.0, 2.0]);
/// ```
pub fn legtrim(c: &[f64], tol: f64) -> Array<f64> {
    let t = polyutils::trimseq(c, tol);
    Array::from_vec(t.clone(), &[t.len()])
}

/// Gauss-Legendre quadrature nodes and weights, like `np.polynomial.legendre.leggauss(deg)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::leggauss;
/// let (x, w) = leggauss(2);
/// assert_eq!(x.len(), 2);
/// assert_eq!(w.len(), 2);
/// ```
pub fn leggauss(deg: usize) -> (Array<f64>, Array<f64>) {
    assert!(deg >= 1, "deg must be a positive integer");
    let mut c = vec![0.0; deg + 1];
    c[deg] = 1.0;
    let m = legcompanion(&c);
    let mut x = crate::linalg::eigvalsh(&m).to_vec();

    let c_der = legder(&c, 1).to_vec();
    for xi in x.iter_mut() {
        let dy = legval(*xi, &c);
        let df = legval(*xi, &c_der);
        *xi -= dy / df;
    }

    let mut fm = vec![0.0; deg];
    let mut df = vec![0.0; deg];
    for i in 0..deg {
        fm[i] = legval(x[i], &c[1..]);
        df[i] = legval(x[i], &c_der);
    }
    let max_fm = fm.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let max_df = df.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut w = vec![0.0; deg];
    for i in 0..deg {
        let f_scaled = fm[i] / max_fm;
        let d_scaled = df[i] / max_df;
        w[i] = 1.0 / (f_scaled * d_scaled);
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
    for wi in w.iter_mut() {
        *wi *= 2.0 / w_sum;
    }

    (Array::from_vec(x, &[deg]), Array::from_vec(w, &[deg]))
}

/// Legendre weight function f(x) = 1, like `np.polynomial.legendre.legweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::legweight;
/// assert_eq!(legweight(0.5), 1.0);
/// ```
pub fn legweight(_x: f64) -> f64 {
    1.0
}

/// Evaluate Legendre weight function over array, like `np.polynomial.legendre.legweight(x)`.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// use rustnum::polynomial::legendre::legweight_array;
/// let x = np::array![0.0, 0.5];
/// assert_eq!(legweight_array(&x).to_vec(), vec![1.0, 1.0]);
/// ```
pub fn legweight_array(x: &Array<f64>) -> Array<f64> {
    let data: Vec<f64> = x.to_vec().into_iter().map(legweight).collect();
    Array::from_vec(data, x.shape())
}

/// Convert a polynomial in standard basis to Legendre series, like `np.polynomial.legendre.poly2leg(pol)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::poly2leg;
/// let c = poly2leg(&[0.0, 1.0, 2.0, 3.0]);
/// assert_eq!(c.len(), 4);
/// ```
pub fn poly2leg(pol: &[f64]) -> Array<f64> {
    let pol = polyutils::trimseq(pol, 0.0);
    let mut res = vec![0.0];
    for &coeff in pol.iter().rev() {
        let mx = legmulx(&res).to_vec();
        res = legadd(&mx, &[coeff]).to_vec();
    }
    Array::from_vec(res.clone(), &[res.len()])
}

/// Convert a Legendre series to a standard polynomial, like `np.polynomial.legendre.leg2poly(c)`.
///
/// # Examples
///
/// ```
/// use rustnum::polynomial::legendre::leg2poly;
/// let p = leg2poly(&[-1.0, -3.5, 3.0, 7.5]);
/// assert_eq!(p.len(), 4);
/// ```
pub fn leg2poly(c: &[f64]) -> Array<f64> {
    let c = polyutils::trimseq(c, 0.0);
    let n = c.len();
    if n < 3 {
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

        let mx = super::polynomial::polymulx(&c1).to_vec();
        let mx_scaled: Vec<f64> = mx
            .iter()
            .map(|&x| (x * (2 * i - 1) as f64) / i as f64)
            .collect();
        c1 = super::polynomial::polyadd(&tmp, &mx_scaled).to_vec();
    }
    let x_c1 = super::polynomial::polymulx(&c1).to_vec();
    super::polynomial::polyadd(&c0, &x_c1)
}
