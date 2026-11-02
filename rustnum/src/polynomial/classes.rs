//! Convenience polynomial classes, like `numpy.polynomial`.

use super::polyutils;
use crate::Array;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

fn identity_poly(c: &[f64]) -> Array<f64> {
    Array::from_vec(c.to_vec(), &[c.len()])
}

fn format_float(val: f64) -> String {
    if val.fract() == 0.0 {
        format!("{:.1}", val)
    } else {
        format!("{}", val)
    }
}

pub(crate) fn format_polynomial_display(
    coef: &[f64],
    domain: [f64; 2],
    window: [f64; 2],
    basis_name: &str,
) -> String {
    if coef.is_empty() {
        return "0.0".to_string();
    }
    let (off, scl) = polyutils::mapparms(domain, window);
    let arg = if (off.abs() < 1e-14) && ((scl - 1.0).abs() < 1e-14) {
        "x".to_string()
    } else if off.abs() < 1e-14 {
        format!("({}*x)", format_float(scl))
    } else if (scl - 1.0).abs() < 1e-14 {
        format!("({} + x)", format_float(off))
    } else {
        format!("({} + {}*x)", format_float(off), format_float(scl))
    };

    let mut out = format_float(coef[0]);
    if coef.len() == 1 {
        return out;
    }

    for (idx, &c) in coef.iter().enumerate().skip(1) {
        let (sign, abs_c) = if c >= 0.0 { ("+", c) } else { ("-", -c) };
        let term_str = if basis_name.is_empty() {
            if idx == 1 {
                arg.clone()
            } else {
                format!("{}**{}", arg, idx)
            }
        } else {
            format!("{}_{}({})", basis_name, idx, arg)
        };
        out.push_str(&format!(" {} {} {}", sign, format_float(abs_c), term_str));
    }
    out
}

/// Trait implemented by all polynomial series kinds to support generic conversion.
pub trait PolyKind: Sized {
    /// Return the default domain for this kind.
    fn default_domain() -> [f64; 2];
    /// Return the default window for this kind.
    fn default_window() -> [f64; 2];
    /// Create an instance from coefficients, domain, and window.
    fn from_coef_domain_window(coef: &[f64], domain: [f64; 2], window: [f64; 2]) -> Self;
    /// Convert coefficients from this basis to the standard polynomial basis.
    fn to_poly_basis(c: &[f64]) -> Vec<f64>;
    /// Convert coefficients from the standard polynomial basis to this basis.
    fn from_poly_basis(c: &[f64]) -> Vec<f64>;
}

macro_rules! define_poly_class {
    (
        $name:ident,
        $doc_name:literal,
        $default_dom:expr,
        $default_win:expr,
        $basis:expr,
        $val_fn:path,
        $val_arr_fn:path,
        $vander_fn:path,
        $fit_fn:path,
        $roots_fn:path,
        $der_fn:path,
        $int_fn:path,
        $add_fn:path,
        $sub_fn:path,
        $mul_fn:path,
        $div_fn:path,
        $pow_fn:path,
        $line_fn:path,
        $fromroots_fn:path,
        $trim_fn:path,
        $to_poly:path,
        $from_poly:path
    ) => {
        #[doc = concat!($doc_name, " series class, like `numpy.polynomial.", stringify!($name), "`.")]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name {
            pub(crate) coef: Vec<f64>,
            pub(crate) domain: [f64; 2],
            pub(crate) window: [f64; 2],
        }

        impl PolyKind for $name {
            fn default_domain() -> [f64; 2] {
                $default_dom
            }
            fn default_window() -> [f64; 2] {
                $default_win
            }
            fn from_coef_domain_window(coef: &[f64], domain: [f64; 2], window: [f64; 2]) -> Self {
                let c = if coef.is_empty() {
                    vec![0.0]
                } else {
                    coef.to_vec()
                };
                Self {
                    coef: c,
                    domain,
                    window,
                }
            }
            fn to_poly_basis(c: &[f64]) -> Vec<f64> {
                $to_poly(c).to_vec()
            }
            fn from_poly_basis(c: &[f64]) -> Vec<f64> {
                $from_poly(c).to_vec()
            }
        }

        impl $name {
            #[doc = concat!("Create a new ", stringify!($name), " instance with default domain and window, like `np.polynomial.", stringify!($name), "(coef)`.")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0, 3.0]);")]
            /// assert_eq!(p.coef(), &[1.0, 2.0, 3.0]);
            /// ```
            pub fn new(coef: &[f64]) -> Self {
                Self::from_coef_domain_window(coef, $default_dom, $default_win)
            }

            #[doc = concat!("Create a new ", stringify!($name), " instance with specified domain and window, like `np.polynomial.", stringify!($name), "(coef, domain, window)`.")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::with_domain_and_window(&[1.0, 2.0], [0.0, 1.0], [-1.0, 1.0]);")]
            /// assert_eq!(p.domain(), [0.0, 1.0]);
            /// assert_eq!(p.window(), [-1.0, 1.0]);
            /// ```
            pub fn with_domain_and_window(coef: &[f64], domain: [f64; 2], window: [f64; 2]) -> Self {
                Self::from_coef_domain_window(coef, domain, window)
            }

            /// Return the coefficients slice, like `p.coef`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0]);")]
            /// assert_eq!(p.coef(), &[1.0, 2.0]);
            /// ```
            pub fn coef(&self) -> &[f64] {
                &self.coef
            }

            /// Return the domain of the series, like `p.domain`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0]);")]
            /// assert_eq!(p.domain(), p.domain());
            /// ```
            pub fn domain(&self) -> [f64; 2] {
                self.domain
            }

            /// Return the window of the series, like `p.window`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0]);")]
            /// assert_eq!(p.window(), p.window());
            /// ```
            pub fn window(&self) -> [f64; 2] {
                self.window
            }

            /// Return the degree of the series, like `p.degree()`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0, 3.0]);")]
            /// assert_eq!(p.degree(), 2);
            /// ```
            pub fn degree(&self) -> usize {
                if self.coef.len() <= 1 {
                    0
                } else {
                    self.coef.len() - 1
                }
            }

            /// Least squares fit to data with minimal domain covering x, like `p.fit(x, y, deg)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::fit(&[-1.0, 0.0, 1.0], &[1.0, 0.0, 1.0], 2);")]
            /// assert_eq!(p.degree(), 2);
            /// ```
            pub fn fit(x: &[f64], y: &[f64], deg: usize) -> Self {
                let domain = polyutils::getdomain(x);
                Self::fit_with_domain(x, y, deg, domain, $default_win)
            }

            /// Least squares fit to data with specified domain and window, like `p.fit(x, y, deg, domain=..., window=...)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::fit_with_domain(&[-1.0, 0.0, 1.0], &[1.0, 0.0, 1.0], 2, [-1.0, 1.0], [-1.0, 1.0]);")]
            /// assert_eq!(p.degree(), 2);
            /// ```
            pub fn fit_with_domain(x: &[f64], y: &[f64], deg: usize, domain: [f64; 2], window: [f64; 2]) -> Self {
                let xnew = polyutils::mapdomain_slice(x, domain, window);
                let coef = $fit_fn(&xnew, y, deg).to_vec();
                Self::from_coef_domain_window(&coef, domain, window)
            }

            /// Evaluate series at a scalar point x, like `p(x)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0]);")]
            /// assert_eq!(p.eval(0.0), p.eval(0.0));
            /// ```
            pub fn eval(&self, x: f64) -> f64 {
                let x_win = polyutils::mapdomain(x, self.domain, self.window);
                $val_fn(x_win, &self.coef)
            }

            /// Evaluate series over an array of points, like `p(x)`.
            ///
            /// # Examples
            ///
            /// ```
            /// use rustnum as np;
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0]);")]
            /// let x = np::array![0.0, 1.0];
            /// let y = p.eval_array(&x);
            /// assert_eq!(y.shape(), &[2]);
            /// ```
            pub fn eval_array(&self, x: &Array<f64>) -> Array<f64> {
                let x_win = polyutils::mapdomain_array(x, self.domain, self.window);
                $val_arr_fn(&x_win, &self.coef)
            }

            /// Return the roots of the series mapped to the domain, like `p.roots()`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[0.0, 1.0]);")]
            /// let (re, _) = p.roots();
            /// assert_eq!(re.len(), 1);
            /// ```
            pub fn roots(&self) -> (Array<f64>, Array<f64>) {
                let (re, im) = $roots_fn(&self.coef);
                let (off, scl) = polyutils::mapparms(self.window, self.domain);
                let re_mapped: Vec<f64> = re.to_vec().into_iter().map(|r| off + scl * r).collect();
                let im_mapped: Vec<f64> = im.to_vec().into_iter().map(|i| scl * i).collect();
                (Array::from_vec(re_mapped, re.shape()), Array::from_vec(im_mapped, im.shape()))
            }

            /// Differentiate series m times, like `p.deriv(m)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0, 3.0]);")]
            /// let d = p.deriv(1);
            /// assert_eq!(d.degree(), 1);
            /// ```
            pub fn deriv(&self, m: usize) -> Self {
                if m == 0 {
                    return self.clone();
                }
                let (_off, scl) = polyutils::mapparms(self.domain, self.window);
                let mut d = $der_fn(&self.coef, m).to_vec();
                let scale = scl.powi(m as i32);
                for x in d.iter_mut() {
                    *x *= scale;
                }
                Self::from_coef_domain_window(&d, self.domain, self.window)
            }

            /// Integrate series m times, like `p.integ(m, k, lbnd)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0]);")]
            /// let pi = p.integ(1, &[0.0], None);
            /// assert_eq!(pi.degree(), 2);
            /// ```
            pub fn integ(&self, m: usize, k: &[f64], lbnd: Option<f64>) -> Self {
                if m == 0 {
                    return self.clone();
                }
                let (off, scl) = polyutils::mapparms(self.domain, self.window);
                let effective_lbnd = match lbnd {
                    Some(val) => off + scl * val,
                    None => 0.0,
                };
                let mut current = self.coef.clone();
                for step in 0..m {
                    let k_val = if step < k.len() { k[step] } else { 0.0 };
                    for x in current.iter_mut() {
                        *x /= scl;
                    }
                    let integrated = $int_fn(&current, 1, &[k_val], effective_lbnd).to_vec();
                    current = integrated;
                }
                Self::from_coef_domain_window(&current, self.domain, self.window)
            }

            /// Divide series by another, returning quotient and remainder, like `divmod(p1, p2)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p1 = ", stringify!($name), "::new(&[1.0, 2.0, 3.0]);")]
            #[doc = concat!("let p2 = ", stringify!($name), "::new(&[1.0, 1.0]);")]
            /// let (q, r) = p1.divmod(&p2);
            /// assert_eq!(q.degree(), 1);
            /// ```
            pub fn divmod(&self, other: &Self) -> (Self, Self) {
                assert_eq!(self.domain, other.domain, "domains of the two series are not compatible");
                assert_eq!(self.window, other.window, "windows of the two series are not compatible");
                let (q, r) = $div_fn(&self.coef, &other.coef);
                (
                    Self::from_coef_domain_window(&q.to_vec(), self.domain, self.window),
                    Self::from_coef_domain_window(&r.to_vec(), self.domain, self.window),
                )
            }

            /// Raise series to an integer power, like `p**n`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0]);")]
            /// let p2 = p.pow(2);
            /// assert_eq!(p2.degree(), 2);
            /// ```
            pub fn pow(&self, n: usize) -> Self {
                let coef = $pow_fn(&self.coef, n).to_vec();
                Self::from_coef_domain_window(&coef, self.domain, self.window)
            }

            /// Convert series to target kind with specified domain and window, like `p.convert(domain=..., kind=..., window=...)`.
            pub fn convert_with_domain<Target: PolyKind>(&self, target_domain: Option<[f64; 2]>, target_window: Option<[f64; 2]>) -> Target {
                let t_domain = target_domain.unwrap_or_else(Target::default_domain);
                let t_window = target_window.unwrap_or_else(Target::default_window);

                let (off1, scl1) = polyutils::mapparms(t_window, t_domain);
                let (off2, scl2) = polyutils::mapparms(self.domain, self.window);
                let c0 = off2 + scl2 * off1;
                let c1 = scl2 * scl1;

                let p_u = Self::to_poly_basis(&self.coef);

                let mut p_t = vec![0.0];
                let u_poly = vec![c0, c1];
                for &a in p_u.iter().rev() {
                    let mul = super::polynomial::polymul(&p_t, &u_poly).to_vec();
                    p_t = super::polynomial::polyadd(&mul, &[a]).to_vec();
                }

                let target_coef = Target::from_poly_basis(&p_t);
                Target::from_coef_domain_window(&target_coef, t_domain, t_window)
            }

            /// Convert series to target kind with its default domain and window, like `p.convert(kind=Target)`.
            ///
            /// # Examples
            ///
            /// ```
            /// use rustnum::polynomial::*;
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0]);")]
            /// let poly: Polynomial = p.convert();
            /// assert_eq!(poly.degree(), 1);
            /// ```
            pub fn convert<Target: PolyKind>(&self) -> Target {
                self.convert_with_domain::<Target>(None, None)
            }

            /// Convert to standard Polynomial series, like `p.convert(kind=Polynomial)`.
            pub fn convert_to_polynomial(&self) -> Polynomial {
                self.convert::<Polynomial>()
            }

            /// Convert to Chebyshev series, like `p.convert(kind=Chebyshev)`.
            pub fn convert_to_chebyshev(&self) -> Chebyshev {
                self.convert::<Chebyshev>()
            }

            /// Convert to Legendre series, like `p.convert(kind=Legendre)`.
            pub fn convert_to_legendre(&self) -> Legendre {
                self.convert::<Legendre>()
            }

            /// Convert to Laguerre series, like `p.convert(kind=Laguerre)`.
            pub fn convert_to_laguerre(&self) -> Laguerre {
                self.convert::<Laguerre>()
            }

            /// Convert to Hermite series, like `p.convert(kind=Hermite)`.
            pub fn convert_to_hermite(&self) -> Hermite {
                self.convert::<Hermite>()
            }

            /// Convert to HermiteE series, like `p.convert(kind=HermiteE)`.
            pub fn convert_to_hermite_e(&self) -> HermiteE {
                self.convert::<HermiteE>()
            }

            /// Linearly spaced points and values across domain, like `p.linspace(n, domain)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0]);")]
            /// let (x, y) = p.linspace(5, None);
            /// assert_eq!(x.len(), 5);
            /// assert_eq!(y.len(), 5);
            /// ```
            pub fn linspace(&self, n: usize, domain: Option<[f64; 2]>) -> (Array<f64>, Array<f64>) {
                let dom = domain.unwrap_or(self.domain);
                let xs = crate::linspace(dom[0], dom[1], n);
                let ys = self.eval_array(&xs);
                (xs, ys)
            }

            /// Truncate series to the given degree, like `p.cutdeg(deg)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0, 3.0]);")]
            /// let cut = p.cutdeg(1);
            /// assert_eq!(cut.degree(), 1);
            /// ```
            pub fn cutdeg(&self, deg: usize) -> Self {
                let limit = (deg + 1).min(self.coef.len());
                let mut c = self.coef[..limit].to_vec();
                if c.is_empty() {
                    c.push(0.0);
                }
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }

            /// Trim trailing small coefficients below tolerance, like `p.trim(tol)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0, 0.0]);")]
            /// let trimmed = p.trim(0.0);
            /// assert_eq!(trimmed.degree(), 1);
            /// ```
            pub fn trim(&self, tol: f64) -> Self {
                let c = $trim_fn(&self.coef, tol).to_vec();
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }

            /// Truncate series to given number of coefficients, like `p.truncate(size)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::new(&[1.0, 2.0, 3.0]);")]
            /// let trunc = p.truncate(2);
            /// assert_eq!(trunc.coef().len(), 2);
            /// ```
            pub fn truncate(&self, size: usize) -> Self {
                let limit = size.min(self.coef.len());
                let mut c = self.coef[..limit].to_vec();
                if c.is_empty() {
                    c.push(0.0);
                }
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }

            /// Identity polynomial p(x) = x, like `p.identity()`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let id = ", stringify!($name), "::identity();")]
            /// assert!((id.eval(0.5) - 0.5).abs() < 1e-12);
            /// ```
            pub fn identity() -> Self {
                Self::identity_with_domain($default_dom, $default_win)
            }

            /// Identity polynomial with custom domain and window, like `p.identity(domain, window)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let id = ", stringify!($name), "::identity_with_domain([0.0, 1.0], [-1.0, 1.0]);")]
            /// assert!((id.eval(0.5) - 0.5).abs() < 1e-12);
            /// ```
            pub fn identity_with_domain(domain: [f64; 2], window: [f64; 2]) -> Self {
                let (off, scl) = polyutils::mapparms(window, domain);
                let coef = $line_fn(off, scl).to_vec();
                Self::from_coef_domain_window(&coef, domain, window)
            }

            /// Basis polynomial of degree deg, like `p.basis(deg)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let b = ", stringify!($name), "::basis(2);")]
            /// assert_eq!(b.degree(), 2);
            /// ```
            pub fn basis(deg: usize) -> Self {
                Self::basis_with_domain(deg, $default_dom, $default_win)
            }

            /// Basis polynomial with custom domain and window, like `p.basis(deg, domain, window)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let b = ", stringify!($name), "::basis_with_domain(2, [0.0, 1.0], [-1.0, 1.0]);")]
            /// assert_eq!(b.degree(), 2);
            /// ```
            pub fn basis_with_domain(deg: usize, domain: [f64; 2], window: [f64; 2]) -> Self {
                let mut coef = vec![0.0; deg + 1];
                coef[deg] = 1.0;
                Self::from_coef_domain_window(&coef, domain, window)
            }

            /// Construct series with specified roots, like `p.fromroots(roots)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::fromroots(&[1.0]);")]
            /// assert_eq!(p.degree(), 1);
            /// ```
            pub fn fromroots(roots: &[f64]) -> Self {
                Self::fromroots_with_domain(roots, $default_dom, $default_win)
            }

            /// Construct series with specified roots, domain, and window, like `p.fromroots(roots, domain, window)`.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use rustnum::polynomial::", stringify!($name), ";")]
            #[doc = concat!("let p = ", stringify!($name), "::fromroots_with_domain(&[1.0], [0.0, 2.0], [-1.0, 1.0]);")]
            /// assert_eq!(p.degree(), 1);
            /// ```
            pub fn fromroots_with_domain(roots: &[f64], domain: [f64; 2], window: [f64; 2]) -> Self {
                let deg = roots.len();
                let (off, scl) = polyutils::mapparms(domain, window);
                let rnew: Vec<f64> = roots.iter().map(|&r| off + scl * r).collect();
                let mut coef = $fromroots_fn(&rnew).to_vec();
                let scale = scl.powi(deg as i32);
                for c in coef.iter_mut() {
                    *c /= scale;
                }
                Self::from_coef_domain_window(&coef, domain, window)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", format_polynomial_display(&self.coef, self.domain, self.window, $basis))
            }
        }

        impl Neg for $name {
            type Output = Self;
            fn neg(self) -> Self {
                let c: Vec<f64> = self.coef.iter().map(|&x| -x).collect();
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Neg for &$name {
            type Output = $name;
            fn neg(self) -> $name {
                let c: Vec<f64> = self.coef.iter().map(|&x| -x).collect();
                $name::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Add for $name {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $add_fn(&self.coef, &other.coef).to_vec();
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Add<&Self> for $name {
            type Output = Self;
            fn add(self, other: &Self) -> Self {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $add_fn(&self.coef, &other.coef).to_vec();
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Add for &$name {
            type Output = $name;
            fn add(self, other: Self) -> $name {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $add_fn(&self.coef, &other.coef).to_vec();
                $name::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Add<f64> for $name {
            type Output = Self;
            fn add(mut self, scalar: f64) -> Self {
                if self.coef.is_empty() {
                    self.coef.push(scalar);
                } else {
                    self.coef[0] += scalar;
                }
                self
            }
        }

        impl Add<f64> for &$name {
            type Output = $name;
            fn add(self, scalar: f64) -> $name {
                let mut c = self.coef.clone();
                if c.is_empty() {
                    c.push(scalar);
                } else {
                    c[0] += scalar;
                }
                $name::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Add<$name> for f64 {
            type Output = $name;
            fn add(self, mut poly: $name) -> $name {
                if poly.coef.is_empty() {
                    poly.coef.push(self);
                } else {
                    poly.coef[0] += self;
                }
                poly
            }
        }

        impl Add<&$name> for f64 {
            type Output = $name;
            fn add(self, poly: &$name) -> $name {
                poly + self
            }
        }

        impl Sub for $name {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $sub_fn(&self.coef, &other.coef).to_vec();
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Sub<&Self> for $name {
            type Output = Self;
            fn sub(self, other: &Self) -> Self {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $sub_fn(&self.coef, &other.coef).to_vec();
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Sub for &$name {
            type Output = $name;
            fn sub(self, other: Self) -> $name {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $sub_fn(&self.coef, &other.coef).to_vec();
                $name::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Sub<f64> for $name {
            type Output = Self;
            fn sub(mut self, scalar: f64) -> Self {
                if self.coef.is_empty() {
                    self.coef.push(-scalar);
                } else {
                    self.coef[0] -= scalar;
                }
                self
            }
        }

        impl Sub<f64> for &$name {
            type Output = $name;
            fn sub(self, scalar: f64) -> $name {
                let mut c = self.coef.clone();
                if c.is_empty() {
                    c.push(-scalar);
                } else {
                    c[0] -= scalar;
                }
                $name::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Sub<$name> for f64 {
            type Output = $name;
            fn sub(self, poly: $name) -> $name {
                -poly + self
            }
        }

        impl Sub<&$name> for f64 {
            type Output = $name;
            fn sub(self, poly: &$name) -> $name {
                -poly + self
            }
        }

        impl Mul for $name {
            type Output = Self;
            fn mul(self, other: Self) -> Self {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $mul_fn(&self.coef, &other.coef).to_vec();
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Mul<&Self> for $name {
            type Output = Self;
            fn mul(self, other: &Self) -> Self {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $mul_fn(&self.coef, &other.coef).to_vec();
                Self::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Mul for &$name {
            type Output = $name;
            fn mul(self, other: Self) -> $name {
                assert_eq!(self.domain, other.domain, "domains and windows of the two series are not compatible");
                assert_eq!(self.window, other.window, "domains and windows of the two series are not compatible");
                let c = $mul_fn(&self.coef, &other.coef).to_vec();
                $name::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Mul<f64> for $name {
            type Output = Self;
            fn mul(mut self, scalar: f64) -> Self {
                for x in self.coef.iter_mut() {
                    *x *= scalar;
                }
                self
            }
        }

        impl Mul<f64> for &$name {
            type Output = $name;
            fn mul(self, scalar: f64) -> $name {
                let c: Vec<f64> = self.coef.iter().map(|&x| x * scalar).collect();
                $name::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Mul<$name> for f64 {
            type Output = $name;
            fn mul(self, mut poly: $name) -> $name {
                for x in poly.coef.iter_mut() {
                    *x *= self;
                }
                poly
            }
        }

        impl Mul<&$name> for f64 {
            type Output = $name;
            fn mul(self, poly: &$name) -> $name {
                poly * self
            }
        }

        impl Div for $name {
            type Output = Self;
            fn div(self, other: Self) -> Self {
                let (q, _) = self.divmod(&other);
                q
            }
        }

        impl Div<&Self> for $name {
            type Output = Self;
            fn div(self, other: &Self) -> Self {
                let (q, _) = self.divmod(other);
                q
            }
        }

        impl Div for &$name {
            type Output = $name;
            fn div(self, other: Self) -> $name {
                let (q, _) = self.divmod(other);
                q
            }
        }

        impl Div<f64> for $name {
            type Output = Self;
            fn div(mut self, scalar: f64) -> Self {
                for x in self.coef.iter_mut() {
                    *x /= scalar;
                }
                self
            }
        }

        impl Div<f64> for &$name {
            type Output = $name;
            fn div(self, scalar: f64) -> $name {
                let c: Vec<f64> = self.coef.iter().map(|&x| x / scalar).collect();
                $name::from_coef_domain_window(&c, self.domain, self.window)
            }
        }

        impl Rem for $name {
            type Output = Self;
            fn rem(self, other: Self) -> Self {
                let (_, r) = self.divmod(&other);
                r
            }
        }

        impl Rem<&Self> for $name {
            type Output = Self;
            fn rem(self, other: &Self) -> Self {
                let (_, r) = self.divmod(other);
                r
            }
        }

        impl Rem for &$name {
            type Output = $name;
            fn rem(self, other: Self) -> $name {
                let (_, r) = self.divmod(other);
                r
            }
        }
    };
}

define_poly_class!(
    Polynomial,
    "Standard power",
    [-1.0, 1.0],
    [-1.0, 1.0],
    "",
    super::polynomial::polyval,
    super::polynomial::polyval_array,
    super::polynomial::polyvander,
    super::polynomial::polyfit,
    super::polynomial::polyroots,
    super::polynomial::polyder,
    super::polynomial::polyint,
    super::polynomial::polyadd,
    super::polynomial::polysub,
    super::polynomial::polymul,
    super::polynomial::polydiv,
    super::polynomial::polypow,
    super::polynomial::polyline,
    super::polynomial::polyfromroots,
    super::polynomial::polytrim,
    identity_poly,
    identity_poly
);

define_poly_class!(
    Chebyshev,
    "Chebyshev",
    [-1.0, 1.0],
    [-1.0, 1.0],
    "T",
    super::chebyshev::chebval,
    super::chebyshev::chebval_array,
    super::chebyshev::chebvander,
    super::chebyshev::chebfit,
    super::chebyshev::chebroots,
    super::chebyshev::chebder,
    super::chebyshev::chebint,
    super::chebyshev::chebadd,
    super::chebyshev::chebsub,
    super::chebyshev::chebmul,
    super::chebyshev::chebdiv,
    super::chebyshev::chebpow,
    super::chebyshev::chebline,
    super::chebyshev::chebfromroots,
    super::chebyshev::chebtrim,
    super::chebyshev::cheb2poly,
    super::chebyshev::poly2cheb
);

define_poly_class!(
    Legendre,
    "Legendre",
    [-1.0, 1.0],
    [-1.0, 1.0],
    "P",
    super::legendre::legval,
    super::legendre::legval_array,
    super::legendre::legvander,
    super::legendre::legfit,
    super::legendre::legroots,
    super::legendre::legder,
    super::legendre::legint,
    super::legendre::legadd,
    super::legendre::legsub,
    super::legendre::legmul,
    super::legendre::legdiv,
    super::legendre::legpow,
    super::legendre::legline,
    super::legendre::legfromroots,
    super::legendre::legtrim,
    super::legendre::leg2poly,
    super::legendre::poly2leg
);

define_poly_class!(
    Laguerre,
    "Laguerre",
    [0.0, 1.0],
    [0.0, 1.0],
    "L",
    super::laguerre::lagval,
    super::laguerre::lagval_array,
    super::laguerre::lagvander,
    super::laguerre::lagfit,
    super::laguerre::lagroots,
    super::laguerre::lagder,
    super::laguerre::lagint,
    super::laguerre::lagadd,
    super::laguerre::lagsub,
    super::laguerre::lagmul,
    super::laguerre::lagdiv,
    super::laguerre::lagpow,
    super::laguerre::lagline,
    super::laguerre::lagfromroots,
    super::laguerre::lagtrim,
    super::laguerre::lag2poly,
    super::laguerre::poly2lag
);

define_poly_class!(
    Hermite,
    "Hermite",
    [-1.0, 1.0],
    [-1.0, 1.0],
    "H",
    super::hermite::hermval,
    super::hermite::hermval_array,
    super::hermite::hermvander,
    super::hermite::hermfit,
    super::hermite::hermroots,
    super::hermite::hermder,
    super::hermite::hermint,
    super::hermite::hermadd,
    super::hermite::hermsub,
    super::hermite::hermmul,
    super::hermite::hermdiv,
    super::hermite::hermpow,
    super::hermite::hermline,
    super::hermite::hermfromroots,
    super::hermite::hermtrim,
    super::hermite::herm2poly,
    super::hermite::poly2herm
);

define_poly_class!(
    HermiteE,
    "HermiteE",
    [-1.0, 1.0],
    [-1.0, 1.0],
    "He",
    super::hermite_e::hermeval,
    super::hermite_e::hermeval_array,
    super::hermite_e::hermevander,
    super::hermite_e::hermefit,
    super::hermite_e::hermeroots,
    super::hermite_e::hermeder,
    super::hermite_e::hermeint,
    super::hermite_e::hermeadd,
    super::hermite_e::hermesub,
    super::hermite_e::hermemul,
    super::hermite_e::hermediv,
    super::hermite_e::hermepow,
    super::hermite_e::hermeline,
    super::hermite_e::hermefromroots,
    super::hermite_e::hermetrim,
    super::hermite_e::herme2poly,
    super::hermite_e::poly2herme
);
