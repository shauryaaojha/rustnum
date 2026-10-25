//! Slice-level polynomial kernels on `f64`. Coefficients are highest power first.
//!
//! No dependency on `Array` or on any linalg module: the least-squares solver is a small
//! column-pivoted Householder QR and the root finder is the Francis double-shift QR
//! algorithm on the (balanced) companion matrix, with an Aberth-Ehrlich fallback.

use std::ops::{Index, IndexMut};

/// Evaluate `p(x)` with Horner's rule, like `np.polyval(p, x)`. An empty `p` is the zero polynomial.
pub(crate) fn polyval(p: &[f64], x: f64) -> f64 {
    p.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// `a + b`, right-aligned (the constant terms line up), like `np.polyadd`. Leading zeros are kept.
pub(crate) fn polyadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    combine(a, b, 1.0)
}

/// `a - b`, like `np.polysub`. Leading zeros are kept.
pub(crate) fn polysub(a: &[f64], b: &[f64]) -> Vec<f64> {
    combine(a, b, -1.0)
}

fn combine(a: &[f64], b: &[f64], sign: f64) -> Vec<f64> {
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    out[n - a.len()..].copy_from_slice(a);
    for (o, &v) in out[n - b.len()..].iter_mut().zip(b) {
        *o += sign * v;
    }
    out
}

/// Product of two polynomials (discrete convolution), like `np.polymul`.
pub(crate) fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert!(!a.is_empty(), "a cannot be empty");
    assert!(!b.is_empty(), "v cannot be empty");
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// Polynomial long division `u / v` returning `(quotient, remainder)`, like `np.polydiv`.
///
/// Mirrors NumPy exactly, including trimming leading remainder coefficients that satisfy
/// `isclose(r[0], 0, rtol=1e-14)` (which, with NumPy's default `atol=1e-8`, means `|r[0]| <= 1e-8`).
pub(crate) fn polydiv(u: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    assert!(!u.is_empty() && !v.is_empty(), "polydiv: empty polynomial");
    assert!(
        v[0] != 0.0,
        "polydiv: leading coefficient of the divisor is zero"
    );
    let m = u.len() - 1;
    let n = v.len() - 1;
    let scale = 1.0 / v[0];
    let steps = if m >= n { m - n + 1 } else { 0 };
    let mut q = vec![0.0; steps.max(1)];
    let mut r = u.to_vec();
    for k in 0..steps {
        let d = scale * r[k];
        q[k] = d;
        for (ri, &vi) in r[k..=k + n].iter_mut().zip(v) {
            *ri -= d * vi;
        }
    }
    let mut lead = 0;
    while lead + 1 < r.len() && r[lead].abs() <= 1e-8 {
        lead += 1;
    }
    r.drain(..lead);
    (q, r)
}

/// `m`-th derivative, like `np.polyder(p, m)`. The derivative of a constant is empty, as in NumPy.
pub(crate) fn polyder(p: &[f64], m: usize) -> Vec<f64> {
    let mut y = p.to_vec();
    for _ in 0..m {
        if y.is_empty() {
            break;
        }
        y.pop();
        let n = y.len();
        for (i, c) in y.iter_mut().enumerate() {
            *c *= (n - i) as f64;
        }
    }
    y
}

/// `m`-fold antiderivative with integration constants `k`, like `np.polyint(p, m, k)`.
///
/// `k` empty means all zeros, `k.len() == 1` is broadcast to every integration, and
/// `k.len() < m` otherwise panics (NumPy raises `ValueError`).
pub(crate) fn polyint(p: &[f64], m: usize, k: &[f64]) -> Vec<f64> {
    let consts: Vec<f64> = match k.len() {
        0 => vec![0.0; m],
        1 => vec![k[0]; m],
        len if len >= m => k[..m].to_vec(),
        _ => panic!("k must be a scalar or a rank-1 array of length 1 or >m."),
    };
    let mut y = p.to_vec();
    for c in consts {
        let n = y.len();
        for (i, v) in y.iter_mut().enumerate() {
            *v /= (n - i) as f64;
        }
        y.push(c);
    }
    y
}

// ---------------------------------------------------------------------------------------
// Least squares
// ---------------------------------------------------------------------------------------

/// Least-squares polynomial fit of degree `deg`, like `np.polyfit(x, y, deg)`.
///
/// Builds the Vandermonde matrix, scales its columns to unit norm (as NumPy does), and solves
/// by column-pivoted Householder QR. If the matrix is rank deficient (for example fewer points
/// than `deg + 1`) NumPy warns and returns the minimum-norm solution; this kernel returns the
/// basic solution instead (free coefficients set to zero).
pub(crate) fn polyfit(x: &[f64], y: &[f64], deg: usize) -> Vec<f64> {
    assert!(!x.is_empty(), "expected non-empty vector for x");
    assert_eq!(x.len(), y.len(), "expected x and y to have same length");
    let m = x.len();
    let n = deg + 1;

    // Column j holds x^(deg - j).
    let mut cols: Vec<Vec<f64>> = (0..n)
        .map(|j| x.iter().map(|&xi| xi.powi((deg - j) as i32)).collect())
        .collect();
    let mut scale: Vec<f64> = cols
        .iter()
        .map(|c| c.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect();
    for s in scale.iter_mut() {
        if *s == 0.0 {
            *s = 1.0;
        }
    }
    for (c, &s) in cols.iter_mut().zip(&scale) {
        c.iter_mut().for_each(|v| *v /= s);
    }

    let mut rhs = y.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let steps = m.min(n);
    let mut diag = vec![0.0; steps];
    for k in 0..steps {
        // Pivot: remaining column with the largest norm below row k.
        let norm2 = |c: &Vec<f64>| c[k..].iter().map(|v| v * v).sum::<f64>();
        let mut best = k;
        let mut best_norm = norm2(&cols[k]);
        for (j, col) in cols.iter().enumerate().skip(k + 1) {
            let nj = norm2(col);
            if nj > best_norm {
                best = j;
                best_norm = nj;
            }
        }
        cols.swap(k, best);
        perm.swap(k, best);

        let norm = best_norm.sqrt();
        if norm == 0.0 {
            diag[k] = 0.0;
            continue;
        }
        // Householder vector for column k, rows k..m.
        let alpha = if cols[k][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = cols[k][k..].to_vec();
        v[0] -= alpha;
        let vnorm2: f64 = v.iter().map(|t| t * t).sum();
        diag[k] = alpha;
        if vnorm2 > 0.0 {
            for c in cols.iter_mut().skip(k) {
                reflect(&mut c[k..], &v, vnorm2);
            }
            reflect(&mut rhs[k..], &v, vnorm2);
        }
        cols[k][k] = alpha;
    }

    // Numerical rank from the R diagonal, with NumPy's lstsq default rcond = len(x) * eps.
    let rcond = m as f64 * f64::EPSILON;
    let top = diag.first().map_or(0.0, |d| d.abs());
    let rank = diag.iter().take_while(|d| d.abs() > rcond * top).count();

    // Back substitution on the leading rank x rank block of R.
    let mut sol = vec![0.0; n];
    for i in (0..rank).rev() {
        let mut acc = rhs[i];
        for j in i + 1..rank {
            acc -= cols[j][i] * sol[j];
        }
        sol[i] = acc / cols[i][i];
    }
    let mut coef = vec![0.0; n];
    for (i, &pj) in perm.iter().enumerate() {
        coef[pj] = sol[i] / scale[pj];
    }
    coef
}

/// Apply `I - 2 v v^T / (v^T v)` to `target`.
fn reflect(target: &mut [f64], v: &[f64], vnorm2: f64) {
    let dot: f64 = target.iter().zip(v).map(|(a, b)| a * b).sum();
    let f = 2.0 * dot / vnorm2;
    for (t, &vi) in target.iter_mut().zip(v) {
        *t -= f * vi;
    }
}

// ---------------------------------------------------------------------------------------
// Roots
// ---------------------------------------------------------------------------------------

/// Roots of the polynomial `p` as `(re, im)` pairs, like `np.roots(p)`.
///
/// Leading zeros are stripped and trailing zeros become zero roots appended at the end (as in
/// NumPy). The rest are the eigenvalues of the companion matrix, found with the Francis
/// double-shift QR algorithm. Conjugate pairs are returned positive imaginary part first, and the
/// order otherwise follows the position on the diagonal of the real Schur form (the order
/// LAPACK's `dgeev` produces in practice). If QR fails to converge, Aberth-Ehrlich iteration is used.
pub(crate) fn roots(p: &[f64]) -> Vec<(f64, f64)> {
    let first = match p.iter().position(|&c| c != 0.0) {
        Some(i) => i,
        None => return Vec::new(),
    };
    let last = p.iter().rposition(|&c| c != 0.0).unwrap_or(first);
    let trailing_zeros = p.len() - 1 - last;
    let q = &p[first..=last];

    let mut out = if q.len() > 1 {
        let n = q.len() - 1;
        let mut a = Mat::zeros(n);
        for j in 1..=n {
            a[(1, j)] = -q[j] / q[0];
        }
        for i in 2..=n {
            a[(i, i - 1)] = 1.0;
        }
        balance(&mut a);
        hqr(&mut a).unwrap_or_else(|| aberth(q))
    } else {
        Vec::new()
    };
    out.extend(std::iter::repeat_n((0.0, 0.0), trailing_zeros));
    out
}

/// Square matrix with 1-based `(row, col)` indexing, to follow the textbook algorithms literally.
struct Mat {
    n: usize,
    d: Vec<f64>,
}

impl Mat {
    fn zeros(n: usize) -> Self {
        Mat {
            n,
            d: vec![0.0; (n + 1) * (n + 1)],
        }
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.d[i * (self.n + 1) + j]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.d[i * (self.n + 1) + j]
    }
}

/// Diagonal similarity scaling (powers of two) so rows and columns have comparable 2-norms,
/// following LAPACK's `dgebal` (job `S`, diagonal included in the norms) so that the eigenvalue
/// order matches NumPy's.
fn balance(a: &mut Mat) {
    const RADIX: f64 = 2.0;
    const FACTOR: f64 = 0.95;
    let n = a.n;
    let mut noconv = true;
    let mut sweeps = 0;
    while noconv && sweeps < 100 {
        noconv = false;
        sweeps += 1;
        for i in 1..=n {
            let mut c = (1..=n).map(|j| a[(j, i)] * a[(j, i)]).sum::<f64>().sqrt();
            let mut r = (1..=n).map(|j| a[(i, j)] * a[(i, j)]).sum::<f64>().sqrt();
            if c == 0.0 || r == 0.0 {
                continue;
            }
            let s = c + r;
            let mut f = 1.0;
            let mut g = r / RADIX;
            while c < g && f < 1e150 {
                f *= RADIX;
                c *= RADIX;
                r /= RADIX;
                g /= RADIX;
            }
            g = c / RADIX;
            while g >= r && f > 1e-150 {
                f /= RADIX;
                c /= RADIX;
                g /= RADIX;
                r *= RADIX;
            }
            if c + r >= FACTOR * s {
                continue;
            }
            noconv = true;
            let g = 1.0 / f;
            for j in 1..=n {
                a[(i, j)] *= g;
                a[(j, i)] *= f;
            }
        }
    }
}

fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

/// Eigenvalues of an upper Hessenberg matrix by the Francis double-shift QR algorithm
/// (EISPACK `hqr`). Returns `None` if an eigenvalue needs more than 30 iterations.
fn hqr(a: &mut Mat) -> Option<Vec<(f64, f64)>> {
    let n = a.n;
    let mut wr = vec![0.0; n + 1];
    let mut wi = vec![0.0; n + 1];
    let mut anorm = 0.0;
    for i in 1..=n {
        for j in i.saturating_sub(1).max(1)..=n {
            anorm += a[(i, j)].abs();
        }
    }
    let (mut p, mut q, mut r): (f64, f64, f64);
    let (mut x, mut y, mut z, mut w);
    let mut nn = n;
    let mut t = 0.0;
    while nn >= 1 {
        let mut its = 0;
        loop {
            // Look for a single small subdiagonal element.
            let mut l = nn;
            while l >= 2 {
                let mut s = a[(l - 1, l - 1)].abs() + a[(l, l)].abs();
                if s == 0.0 {
                    s = anorm;
                }
                if a[(l, l - 1)].abs() + s == s {
                    a[(l, l - 1)] = 0.0;
                    break;
                }
                l -= 1;
            }
            x = a[(nn, nn)];
            if l == nn {
                // One root found.
                wr[nn] = x + t;
                wi[nn] = 0.0;
                nn -= 1;
            } else {
                y = a[(nn - 1, nn - 1)];
                w = a[(nn, nn - 1)] * a[(nn - 1, nn)];
                if l == nn - 1 {
                    // Two roots found.
                    p = 0.5 * (y - x);
                    q = p * p + w;
                    z = q.abs().sqrt();
                    x += t;
                    if q >= 0.0 {
                        z = p + sign(z, p);
                        wr[nn - 1] = x + z;
                        wr[nn] = x + z;
                        if z != 0.0 {
                            wr[nn] = x - w / z;
                        }
                        wi[nn - 1] = 0.0;
                        wi[nn] = 0.0;
                    } else {
                        wr[nn - 1] = x + p;
                        wr[nn] = x + p;
                        wi[nn - 1] = z;
                        wi[nn] = -z;
                    }
                    nn -= 2;
                } else {
                    // No roots yet: do a QR sweep.
                    if its == 30 {
                        return None;
                    }
                    if its == 10 || its == 20 {
                        // Exceptional shift.
                        t += x;
                        for i in 1..=nn {
                            a[(i, i)] -= x;
                        }
                        let s = a[(nn, nn - 1)].abs() + a[(nn - 1, nn - 2)].abs();
                        x = 0.75 * s;
                        y = x;
                        w = -0.4375 * s * s;
                    }
                    its += 1;
                    // Like LAPACK's dlahqr: when the trailing 2x2 block has real eigenvalues,
                    // shift twice by the one closer to the corner entry (not by both).
                    let disc = 0.25 * (y - x) * (y - x) + w;
                    if disc > 0.0 {
                        let mid = 0.5 * (x + y);
                        let (e1, e2) = (mid + disc.sqrt(), mid - disc.sqrt());
                        let sigma = if (e1 - x).abs() <= (e2 - x).abs() {
                            e1
                        } else {
                            e2
                        };
                        x = sigma;
                        y = sigma;
                        w = 0.0;
                    }
                    let mut m = nn - 2;
                    loop {
                        z = a[(m, m)];
                        r = x - z;
                        let mut s = y - z;
                        p = (r * s - w) / a[(m + 1, m)] + a[(m, m + 1)];
                        q = a[(m + 1, m + 1)] - z - r - s;
                        r = a[(m + 2, m + 1)];
                        s = p.abs() + q.abs() + r.abs();
                        p /= s;
                        q /= s;
                        r /= s;
                        if m == l {
                            break;
                        }
                        let u = a[(m, m - 1)].abs() * (q.abs() + r.abs());
                        let v =
                            p.abs() * (a[(m - 1, m - 1)].abs() + z.abs() + a[(m + 1, m + 1)].abs());
                        if u + v == v {
                            break;
                        }
                        m -= 1;
                    }
                    for i in m + 2..=nn {
                        a[(i, i - 2)] = 0.0;
                        if i != m + 2 {
                            a[(i, i - 3)] = 0.0;
                        }
                    }
                    for k in m..nn {
                        if k != m {
                            p = a[(k, k - 1)];
                            q = a[(k + 1, k - 1)];
                            r = if k != nn - 1 { a[(k + 2, k - 1)] } else { 0.0 };
                            x = p.abs() + q.abs() + r.abs();
                            if x != 0.0 {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        let s = sign((p * p + q * q + r * r).sqrt(), p);
                        if s != 0.0 {
                            if k == m {
                                if l != m {
                                    a[(k, k - 1)] = -a[(k, k - 1)];
                                }
                            } else {
                                a[(k, k - 1)] = -s * x;
                            }
                            p += s;
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q /= p;
                            r /= p;
                            for j in k..=nn {
                                let mut pp = a[(k, j)] + q * a[(k + 1, j)];
                                if k != nn - 1 {
                                    pp += r * a[(k + 2, j)];
                                    a[(k + 2, j)] -= pp * z;
                                }
                                a[(k + 1, j)] -= pp * y;
                                a[(k, j)] -= pp * x;
                            }
                            let mmin = if nn < k + 3 { nn } else { k + 3 };
                            for i in l..=mmin {
                                let mut pp = x * a[(i, k)] + y * a[(i, k + 1)];
                                if k != nn - 1 {
                                    pp += z * a[(i, k + 2)];
                                    a[(i, k + 2)] -= pp * r;
                                }
                                a[(i, k + 1)] -= pp * q;
                                a[(i, k)] -= pp;
                            }
                        }
                    }
                }
            }
            if l + 1 >= nn {
                break;
            }
        }
    }
    Some((1..=n).map(|i| (wr[i], wi[i])).collect())
}

type Cx = (f64, f64);

fn cmul(a: Cx, b: Cx) -> Cx {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cdiv(a: Cx, b: Cx) -> Cx {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

/// Aberth-Ehrlich simultaneous iteration; fallback when QR does not converge.
/// `p` has a non-zero leading and trailing coefficient and at least two entries.
fn aberth(p: &[f64]) -> Vec<(f64, f64)> {
    let n = p.len() - 1;
    let monic: Vec<f64> = p.iter().map(|&c| c / p[0]).collect();
    // Cauchy bound on the root magnitudes.
    let radius = 1.0 + monic[1..].iter().fold(0.0f64, |m, c| m.max(c.abs()));
    let mut z: Vec<Cx> = (0..n)
        .map(|k| {
            let ang = 2.0 * std::f64::consts::PI * (k as f64 + 0.25) / n as f64;
            (radius * 0.5 * ang.cos(), radius * 0.5 * ang.sin())
        })
        .collect();
    for _ in 0..500 {
        let mut worst = 0.0f64;
        for k in 0..n {
            let (mut f, mut df) = ((0.0, 0.0), (0.0, 0.0));
            for &c in &monic {
                df = cmul(df, z[k]);
                df = (df.0 + f.0, df.1 + f.1);
                f = cmul(f, z[k]);
                f = (f.0 + c, f.1);
            }
            if f == (0.0, 0.0) {
                continue;
            }
            let ratio = cdiv(f, df);
            let mut s = (0.0, 0.0);
            for j in 0..n {
                if j != k {
                    let inv = cdiv((1.0, 0.0), (z[k].0 - z[j].0, z[k].1 - z[j].1));
                    s = (s.0 + inv.0, s.1 + inv.1);
                }
            }
            let rs = cmul(ratio, s);
            let w = cdiv(ratio, (1.0 - rs.0, -rs.1));
            z[k] = (z[k].0 - w.0, z[k].1 - w.1);
            worst = worst.max(w.0.hypot(w.1) / z[k].0.hypot(z[k].1).max(1e-300));
        }
        if worst < 1e-15 {
            break;
        }
    }
    // Snap numerically-real roots and order conjugate pairs positive-imaginary first.
    for v in z.iter_mut() {
        if v.1.abs() <= 1e-13 * v.0.hypot(v.1).max(1.0) {
            v.1 = 0.0;
        }
    }
    z.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    z
}

/// Aberth-Ehrlich roots, exposed so tests can check the fallback path directly.
#[cfg(test)]
pub(crate) fn roots_aberth(p: &[f64]) -> Vec<(f64, f64)> {
    aberth(p)
}
