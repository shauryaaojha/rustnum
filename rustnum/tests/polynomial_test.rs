#![allow(clippy::approx_constant)]

use rustnum::polynomial::*;

fn assert_allclose(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "mismatch at index {}: {} vs {} (diff {})",
            i,
            x,
            y,
            (x - y).abs()
        );
    }
}

// -------------------------------------------------------------
// 1. Module function tests for each polynomial family
// -------------------------------------------------------------

#[test]
fn test_polynomial_module_functions() {
    let c1 = [1.0, 2.0, 3.0];
    let c2 = [3.0, 2.0, 1.0];

    assert!((polyval(0.5, &c1) - 2.75).abs() < 1e-12);
    assert_allclose(&polyder(&c1, 1).to_vec(), &[2.0, 6.0], 1e-12);
    assert_allclose(
        &polyint(&c1, 1, &[0.0], 0.0).to_vec(),
        &[0.0, 1.0, 1.0, 1.0],
        1e-12,
    );
    assert_allclose(
        &polymul(&c1, &c2).to_vec(),
        &[3.0, 8.0, 14.0, 8.0, 3.0],
        1e-12,
    );
    assert_allclose(&polymulx(&c1).to_vec(), &[0.0, 1.0, 2.0, 3.0], 1e-12);
    assert_allclose(&polyadd(&c1, &c2).to_vec(), &[4.0, 4.0, 4.0], 1e-12);
    assert_allclose(&polysub(&c1, &c2).to_vec(), &[-2.0, 0.0, 2.0], 1e-12);

    let (q, r) = polydiv(&c1, &c2);
    assert_allclose(&q.to_vec(), &[3.0], 1e-12);
    assert_allclose(&r.to_vec(), &[-8.0, -4.0], 1e-12);

    let p2 = polypow(&[1.0, 1.0], 2);
    assert_allclose(&p2.to_vec(), &[1.0, 2.0, 1.0], 1e-12);

    let l = polyline(2.0, 3.0);
    assert_allclose(&l.to_vec(), &[2.0, 3.0], 1e-12);

    let fr = polyfromroots(&[1.0, 2.0]);
    assert_allclose(&fr.to_vec(), &[2.0, -3.0, 1.0], 1e-12);

    let v = polyvander(&[0.5], 2);
    assert_allclose(&v.to_vec(), &[1.0, 0.5, 0.25], 1e-12);

    let (re, im) = polyroots(&[-2.0, 1.0, 1.0]);
    let mut re_vec = re.to_vec();
    re_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_allclose(&re_vec, &[-2.0, 1.0], 1e-12);
    assert!(im.to_vec().iter().all(|&x| x.abs() < 1e-12));
}

#[test]
fn test_chebyshev_module_functions() {
    let c1 = [1.0, 2.0, 3.0];
    let c2 = [3.0, 2.0, 1.0];

    assert!((chebval(0.5, &c1) - 0.5).abs() < 1e-12);
    assert_allclose(&chebder(&c1, 1).to_vec(), &[2.0, 12.0], 1e-12);
    assert_allclose(
        &chebint(&c1, 1, &[0.0], 0.0).to_vec(),
        &[0.5, -0.5, 0.5, 0.5],
        1e-12,
    );
    assert_allclose(
        &chebmul(&c1, &c2).to_vec(),
        &[6.5, 12.0, 12.0, 4.0, 1.5],
        1e-12,
    );

    let (q, r) = chebdiv(&c1, &c2);
    assert_allclose(&q.to_vec(), &[3.0], 1e-12);
    assert_allclose(&r.to_vec(), &[-8.0, -4.0], 1e-12);

    let (x, w) = chebgauss(2);
    assert_allclose(
        &x.to_vec(),
        &[0.7071067811865476, -0.7071067811865476],
        1e-10,
    );
    assert_allclose(
        &w.to_vec(),
        &[std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2],
        1e-10,
    );

    assert!((chebweight(0.0) - 1.0).abs() < 1e-12);

    // Roundtrip poly2cheb / cheb2poly
    let p = [1.0, 2.0, 3.0];
    let c = poly2cheb(&p);
    let p_back = cheb2poly(&c.to_vec());
    assert_allclose(&p_back.to_vec(), &p, 1e-12);
}

#[test]
fn test_legendre_module_functions() {
    let c1 = [1.0, 2.0, 3.0];
    let c2 = [3.0, 2.0, 1.0];

    assert!((legval(0.5, &c1) - 1.625).abs() < 1e-12);
    assert_allclose(&legder(&c1, 1).to_vec(), &[2.0, 9.0], 1e-12);
    assert_allclose(
        &legint(&c1, 1, &[0.0], 0.0).to_vec(),
        &[0.3333333333333333, 0.4, 0.6666666666666666, 0.6],
        1e-12,
    );
    assert_allclose(
        &legmul(&c1, &c2).to_vec(),
        &[
            4.933333333333333,
            11.2,
            13.523809523809522,
            4.8,
            1.542857142857143,
        ],
        1e-12,
    );

    let (q, r) = legdiv(&c1, &c2);
    assert_allclose(&q.to_vec(), &[3.0], 1e-12);
    assert_allclose(&r.to_vec(), &[-8.0, -4.0], 1e-12);

    let (x, w) = leggauss(2);
    assert_allclose(
        &x.to_vec(),
        &[-0.5773502691896257, 0.5773502691896257],
        1e-10,
    );
    assert_allclose(&w.to_vec(), &[1.0, 1.0], 1e-10);

    assert_eq!(legweight(0.5), 1.0);

    let p = [1.0, 2.0, 3.0];
    let l = poly2leg(&p);
    let p_back = leg2poly(&l.to_vec());
    assert_allclose(&p_back.to_vec(), &p, 1e-12);
}

#[test]
fn test_laguerre_module_functions() {
    let c1 = [1.0, 2.0, 3.0];
    let c2 = [3.0, 2.0, 1.0];

    assert!((lagval(0.5, &c1) - 2.375).abs() < 1e-12);
    assert_allclose(&lagder(&c1, 1).to_vec(), &[-5.0, -3.0], 1e-12);
    assert_allclose(
        &lagint(&c1, 1, &[0.0], 0.0).to_vec(),
        &[1.0, 1.0, 1.0, -3.0],
        1e-12,
    );
    assert_allclose(
        &lagmul(&c1, &c2).to_vec(),
        &[10.0, 4.0, 16.0, -12.0, 18.0],
        1e-12,
    );

    let (q, r) = lagdiv(&c1, &c2);
    assert_allclose(&q.to_vec(), &[3.0], 1e-12);
    assert_allclose(&r.to_vec(), &[-8.0, -4.0], 1e-12);

    let (x, w) = laggauss(2);
    assert_allclose(&x.to_vec(), &[0.585786437626905, 3.414213562373095], 1e-10);
    assert_allclose(
        &w.to_vec(),
        &[0.8535533905932738, 0.14644660940672624],
        1e-10,
    );

    assert!((lagweight(1.0) - (-1.0f64).exp()).abs() < 1e-12);

    let p = [1.0, 2.0, 3.0];
    let l = poly2lag(&p);
    let p_back = lag2poly(&l.to_vec());
    assert_allclose(&p_back.to_vec(), &p, 1e-12);
}

#[test]
fn test_hermite_module_functions() {
    let c1 = [1.0, 2.0, 3.0];
    let c2 = [3.0, 2.0, 1.0];

    assert!((hermval(0.5, &c1) - 0.0).abs() < 1e-12);
    assert_allclose(&hermder(&c1, 1).to_vec(), &[4.0, 12.0], 1e-12);
    assert_allclose(
        &hermint(&c1, 1, &[0.0], 0.0).to_vec(),
        &[1.0, 0.5, 0.5, 0.5],
        1e-12,
    );
    assert_allclose(
        &hermmul(&c1, &c2).to_vec(),
        &[35.0, 40.0, 38.0, 8.0, 3.0],
        1e-12,
    );

    let (q, r) = hermdiv(&c1, &c2);
    assert_allclose(&q.to_vec(), &[3.0], 1e-12);
    assert_allclose(&r.to_vec(), &[-8.0, -4.0], 1e-12);

    let (x, w) = hermgauss(2);
    assert_allclose(
        &x.to_vec(),
        &[-0.7071067811865476, 0.7071067811865476],
        1e-10,
    );
    assert_allclose(&w.to_vec(), &[0.886226925452758, 0.886226925452758], 1e-10);

    assert!((hermweight(1.0) - (-1.0f64).exp()).abs() < 1e-12);

    let p = [1.0, 2.0, 3.0];
    let h = poly2herm(&p);
    let p_back = herm2poly(&h.to_vec());
    assert_allclose(&p_back.to_vec(), &p, 1e-12);
}

#[test]
fn test_hermite_e_module_functions() {
    let c1 = [1.0, 2.0, 3.0];
    let c2 = [3.0, 2.0, 1.0];

    assert!((hermeval(0.5, &c1) - (-0.25)).abs() < 1e-12);
    assert_allclose(&hermeder(&c1, 1).to_vec(), &[2.0, 6.0], 1e-12);
    assert_allclose(
        &hermeint(&c1, 1, &[0.0], 0.0).to_vec(),
        &[1.0, 1.0, 1.0, 1.0],
        1e-12,
    );
    assert_allclose(
        &hermemul(&c1, &c2).to_vec(),
        &[13.0, 24.0, 26.0, 8.0, 3.0],
        1e-12,
    );

    let (q, r) = hermediv(&c1, &c2);
    assert_allclose(&q.to_vec(), &[3.0], 1e-12);
    assert_allclose(&r.to_vec(), &[-8.0, -4.0], 1e-12);

    let (x, w) = hermegauss(2);
    assert_allclose(&x.to_vec(), &[-1.0, 1.0], 1e-10);
    assert_allclose(&w.to_vec(), &[1.2533141373155, 1.2533141373155], 1e-10);

    assert!((hermeweight(1.0) - (-0.5f64).exp()).abs() < 1e-12);

    let p = [1.0, 2.0, 3.0];
    let he = poly2herme(&p);
    let p_back = herme2poly(&he.to_vec());
    assert_allclose(&p_back.to_vec(), &p, 1e-12);
}

// -------------------------------------------------------------
// 2. Convenience classes: evaluation, roots, fit, calculus, operators
// -------------------------------------------------------------

#[test]
fn test_classes_roots_of_known_polynomials() {
    // 1. Polynomial roots: x^2 + x - 2 = (x - 1)(x + 2) -> roots: -2, 1
    let p = Polynomial::new(&[-2.0, 1.0, 1.0]);
    let (re, _) = p.roots();
    let mut re_vec = re.to_vec();
    re_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_allclose(&re_vec, &[-2.0, 1.0], 1e-10);

    // 2. Chebyshev roots: T_2(x) = 2x^2 - 1 -> +-1/sqrt(2) ~ +-0.707107
    let c = Chebyshev::new(&[0.0, 0.0, 1.0]);
    let (re, _) = c.roots();
    let mut re_vec = re.to_vec();
    re_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_allclose(&re_vec, &[-0.70710678, 0.70710678], 1e-6);

    // 3. Legendre roots: P_2(x) = 0.5*(3x^2 - 1) -> +-1/sqrt(3) ~ +-0.577350
    let l = Legendre::new(&[0.0, 0.0, 1.0]);
    let (re, _) = l.roots();
    let mut re_vec = re.to_vec();
    re_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_allclose(&re_vec, &[-0.57735027, 0.57735027], 1e-6);

    // 4. Laguerre roots: L_2(x) = 0.5*(x^2 - 4x + 2) -> 2 +- sqrt(2) ~ 0.585786, 3.414214
    let lag = Laguerre::new(&[0.0, 0.0, 1.0]);
    let (re, _) = lag.roots();
    let mut re_vec = re.to_vec();
    re_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_allclose(&re_vec, &[0.58578644, 3.41421356], 1e-6);

    // 5. Hermite roots: H_2(x) = 4x^2 - 2 -> +-1/sqrt(2) ~ +-0.707107
    let h = Hermite::new(&[0.0, 0.0, 1.0]);
    let (re, _) = h.roots();
    let mut re_vec = re.to_vec();
    re_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_allclose(&re_vec, &[-0.70710678, 0.70710678], 1e-6);

    // 6. HermiteE roots: He_2(x) = x^2 - 1 -> +-1.0
    let he = HermiteE::new(&[0.0, 0.0, 1.0]);
    let (re, _) = he.roots();
    let mut re_vec = re.to_vec();
    re_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_allclose(&re_vec, &[-1.0, 1.0], 1e-6);
}

#[test]
fn test_classes_fitting_on_noisy_data() {
    let x = [
        -1.0, -0.8947, -0.7895, -0.6842, -0.5789, -0.4737, -0.3684, -0.2632, -0.1579, -0.0526,
        0.0526, 0.1579, 0.2632, 0.3684, 0.4737, 0.5789, 0.6842, 0.7895, 0.8947, 1.0,
    ];
    let y = [
        -2.4752, -1.9972, -1.4815, -0.9945, -0.6724, -0.2956, 0.1385, 0.4082, 0.6233, 0.9177,
        1.0779, 1.2551, 1.4345, 1.4376, 1.5246, 1.627, 1.6156, 1.6598, 1.5432, 1.4294,
    ];

    let p = Polynomial::fit(&x, &y, 2);
    let c = Chebyshev::fit(&x, &y, 2);
    let l = Legendre::fit(&x, &y, 2);

    assert_allclose(p.coef(), &[0.994492, 1.952707, -1.508297], 1e-4);
    assert_allclose(c.coef(), &[0.240343, 1.952707, -0.754148], 1e-4);
    assert_allclose(l.coef(), &[0.491726, 1.952707, -1.005531], 1e-4);

    // Predictions of all three fitted models should match closely
    for &xi in &[0.0, 0.5, -0.5] {
        let yp = p.eval(xi);
        let yc = c.eval(xi);
        let yl = l.eval(xi);
        assert!((yp - yc).abs() < 1e-4);
        assert!((yp - yl).abs() < 1e-4);
    }
}

#[test]
fn test_classes_operators_and_calculus() {
    let p1 = Polynomial::new(&[1.0, 2.0, 3.0]);
    let p2 = Polynomial::new(&[2.0, 1.0]);

    // + - *
    let p_add = &p1 + &p2;
    assert_allclose(p_add.coef(), &[3.0, 3.0, 3.0], 1e-12);
    let p_sub = &p1 - &p2;
    assert_allclose(p_sub.coef(), &[-1.0, 1.0, 3.0], 1e-12);
    let p_mul = &p1 * &p2;
    assert_allclose(p_mul.coef(), &[2.0, 5.0, 8.0, 3.0], 1e-12);

    // scalar ops
    let p_sc = &p1 + 5.0;
    assert_allclose(p_sc.coef(), &[6.0, 2.0, 3.0], 1e-12);
    let p_sc2 = &p1 * 2.0;
    assert_allclose(p_sc2.coef(), &[2.0, 4.0, 6.0], 1e-12);
    let p_sc3 = &p1 / 2.0;
    assert_allclose(p_sc3.coef(), &[0.5, 1.0, 1.5], 1e-12);

    // / and %
    let (q, r) = p1.divmod(&p2);
    assert_allclose((&q * &p2 + &r).coef(), p1.coef(), 1e-12);
    assert_eq!(&p1 / &p2, q);
    assert_eq!(&p1 % &p2, r);

    // pow
    let p_pow = p2.pow(2);
    assert_allclose(p_pow.coef(), &[4.0, 4.0, 1.0], 1e-12);

    // deriv & integ
    let d = p1.deriv(1);
    assert_allclose(d.coef(), &[2.0, 6.0], 1e-12);
    let integ = d.integ(1, &[1.0], None);
    assert_allclose(integ.coef(), &[1.0, 2.0, 3.0], 1e-12);
}

#[test]
fn test_classes_conversions_between_kinds() {
    let p = Polynomial::new(&[-2.0, 2.0, 6.0]);
    let c: Chebyshev = p.convert();
    assert_allclose(c.coef(), &[1.0, 2.0, 3.0], 1e-12);

    let l: Legendre = c.convert();
    let c_roundtrip: Chebyshev = l.convert();
    assert_allclose(c_roundtrip.coef(), c.coef(), 1e-12);

    let h: Hermite = p.convert();
    let he: HermiteE = h.convert();
    let lag: Laguerre = he.convert();
    let p_back: Polynomial = lag.convert();

    // Verify values match at sample points
    for &xi in &[-0.8, 0.0, 0.5] {
        let val_orig = p.eval(xi);
        let val_back = p_back.eval(xi);
        assert!((val_orig - val_back).abs() < 1e-10);
    }
}

#[test]
fn test_classes_display() {
    let p = Polynomial::new(&[1.0, 2.0, 3.0]);
    assert_eq!(format!("{}", p), "1.0 + 2.0 x + 3.0 x**2");

    let p_neg = Polynomial::new(&[-1.0, -2.0, 3.0]);
    assert_eq!(format!("{}", p_neg), "-1.0 - 2.0 x + 3.0 x**2");

    let c = Chebyshev::new(&[1.0, 2.0, 3.0]);
    assert_eq!(format!("{}", c), "1.0 + 2.0 T_1(x) + 3.0 T_2(x)");

    let l = Legendre::new(&[1.0, 2.0, 3.0]);
    assert_eq!(format!("{}", l), "1.0 + 2.0 P_1(x) + 3.0 P_2(x)");

    let lag = Laguerre::new(&[1.0, 2.0, 3.0]);
    assert_eq!(format!("{}", lag), "1.0 + 2.0 L_1(x) + 3.0 L_2(x)");

    let h = Hermite::new(&[1.0, 2.0, 3.0]);
    assert_eq!(format!("{}", h), "1.0 + 2.0 H_1(x) + 3.0 H_2(x)");

    let he = HermiteE::new(&[1.0, 2.0, 3.0]);
    assert_eq!(format!("{}", he), "1.0 + 2.0 He_1(x) + 3.0 He_2(x)");

    let zero = Polynomial::new(&[0.0]);
    assert_eq!(format!("{}", zero), "0.0");
}

#[test]
fn test_classes_linspace_cutdeg_trim_truncate_basis_identity() {
    let p = Polynomial::new(&[1.0, 2.0, 3.0, 0.0]);
    assert_eq!(p.degree(), 3);

    let trimmed = p.trim(0.0);
    assert_eq!(trimmed.degree(), 2);
    assert_allclose(trimmed.coef(), &[1.0, 2.0, 3.0], 1e-12);

    let cut = p.cutdeg(1);
    assert_eq!(cut.degree(), 1);
    assert_allclose(cut.coef(), &[1.0, 2.0], 1e-12);

    let trunc = p.truncate(2);
    assert_allclose(trunc.coef(), &[1.0, 2.0], 1e-12);

    let id = Polynomial::identity();
    assert_eq!(id.degree(), 1);
    assert_allclose(id.coef(), &[0.0, 1.0], 1e-12);

    let b = Chebyshev::basis(3);
    assert_eq!(b.degree(), 3);
    assert_allclose(b.coef(), &[0.0, 0.0, 0.0, 1.0], 1e-12);

    let (xs, ys) = p.linspace(5, None);
    assert_eq!(xs.len(), 5);
    assert_eq!(ys.len(), 5);
    for i in 0..5 {
        assert!((ys.to_vec()[i] - p.eval(xs.to_vec()[i])).abs() < 1e-12);
    }
}
