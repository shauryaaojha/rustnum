//! A NumPy-compatible complex number type, generic over `f32` and `f64`.
//!
//! Arithmetic follows NumPy's loops (a real scalar is promoted to `x + 0j` first, division is
//! Smith's algorithm, `powc` uses NumPy's integer fast path). The elementary functions follow the
//! C99 Annex G special values (FreeBSD msun), so branch cuts honour the sign of a zero
//! imaginary part: `sqrt(-4 + 0j) = 2j` but `sqrt(-4 - 0j) = -2j`.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::Float;

/// A complex number `re + im*j`, like NumPy `complex128` (`complex64` for `f32`).
///
/// Supports `+ - * /` and unary `-` between complex numbers and with a real scalar on either
/// side, and `Display` prints like `str(np.complex128(...))`. It is not an array element type yet.
///
/// ```
/// use rustnum as np;
/// let z = np::Complex::new(1.0, 2.0) * np::Complex::new(3.0, -1.0);
/// assert_eq!(z, np::Complex::new(5.0, 5.0));
/// assert_eq!(z.to_string(), "(5+5j)");
/// assert_eq!((2.0f64 / np::Complex::new(0.0, 1.0)).to_string(), "-2j");
/// ```
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Complex<F> {
    /// Real part.
    pub re: F,
    /// Imaginary part.
    pub im: F,
}

impl<F: Float> Complex<F> {
    /// Builds `re + im*j`, like `np.complex128(re, im)` (`np.complex64` for `f32`).
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::Complex::new(1.0, 2.0);
    /// assert_eq!((z.re, z.im), (1.0, 2.0));
    /// ```
    pub fn new(re: F, im: F) -> Self {
        Complex { re, im }
    }

    /// Builds `re + 0j`, like `np.complex128(re)`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::from_real(3.0), np::Complex::new(3.0, 0.0));
    /// ```
    pub fn from_real(re: F) -> Self {
        Complex { re, im: F::ZERO }
    }

    /// The imaginary unit `0 + 1j`, like NumPy's `1j`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::<f64>::i() * np::Complex::i(), np::Complex::new(-1.0, 0.0));
    /// ```
    pub fn i() -> Self {
        Complex {
            re: F::ZERO,
            im: F::ONE,
        }
    }

    /// Complex conjugate, like `np.conj(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(1.0, 2.0).conj(), np::Complex::new(1.0, -2.0));
    /// ```
    pub fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }

    /// Modulus, like `np.abs(z)`, computed with `hypot` (no overflow, `inf` beats NaN).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(3.0, 4.0).norm(), 5.0);
    /// ```
    pub fn norm(self) -> F {
        self.re.hypot(self.im)
    }

    /// Squared modulus `re*re + im*im`, like `np.abs(z)**2` without the root.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(3.0, 4.0).norm_sqr(), 25.0);
    /// ```
    pub fn norm_sqr(self) -> F {
        self.re * self.re + self.im * self.im
    }

    /// Argument in `[-pi, pi]`, like `np.angle(z)`: `atan2(im, re)`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(0.0, 1.0).arg(), std::f64::consts::FRAC_PI_2);
    /// ```
    pub fn arg(self) -> F {
        self.im.atan2(self.re)
    }

    /// Builds `r * (cos(theta) + j sin(theta))`, like `r * np.exp(1j * theta)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::Complex::from_polar(2.0, 0.0);
    /// assert_eq!(z, np::Complex::new(2.0, 0.0));
    /// ```
    pub fn from_polar(r: F, theta: F) -> Self {
        Complex::new(r * theta.cos(), r * theta.sin())
    }

    /// True when either part is NaN, like `np.isnan(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::Complex::new(f64::NAN, 0.0).is_nan());
    /// assert!(!np::Complex::new(1.0, 0.0).is_nan());
    /// ```
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// True when either part is infinite, like `np.isinf(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::Complex::new(0.0, f64::INFINITY).is_infinite());
    /// ```
    pub fn is_infinite(self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }

    /// True when both parts are finite, like `np.isfinite(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::Complex::new(1.0, 2.0).is_finite());
    /// assert!(!np::Complex::new(1.0, f64::NAN).is_finite());
    /// ```
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Complex exponential, like `np.exp(z)` (C99 `cexp`).
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::Complex::new(0.0, std::f64::consts::PI).exp();
    /// assert_eq!(z.re, -1.0);
    /// assert!(z.im.abs() < 1e-15);
    /// ```
    pub fn exp(self) -> Self {
        let (x, y) = (self.re, self.im);
        if y == F::ZERO {
            return Complex::new(x.exp(), y);
        }
        if x == F::ZERO {
            return Complex::new(y.cos(), y.sin());
        }
        if !y.is_finite() {
            return if x.is_finite() || x.is_nan() {
                Complex::new(F::NAN, F::NAN)
            } else if x < F::ZERO {
                Complex::new(F::ZERO, F::ZERO)
            } else {
                Complex::new(x, F::NAN)
            };
        }
        if x > F::MAX_VALUE.ln() {
            // exp(x) would overflow; split it so a tiny cos/sin can still give a finite part.
            let t = (x * F::HALF).exp();
            return Complex::new((t * y.cos()) * t, (t * y.sin()) * t);
        }
        let e = x.exp();
        Complex::new(e * y.cos(), e * y.sin())
    }

    /// Principal natural logarithm, like `np.log(z)`: `log(-1 + 0j) = pi j`, `log(-1 - 0j) = -pi j`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::Complex::new(-1.0, 0.0).ln();
    /// assert_eq!(z, np::Complex::new(0.0, std::f64::consts::PI));
    /// ```
    pub fn ln(self) -> Self {
        let (x, y) = (self.re, self.im);
        let (mut ax, mut ay) = (x.abs(), y.abs());
        if ax < ay {
            core::mem::swap(&mut ax, &mut ay);
        }
        let re = if ax.is_finite() && ay.is_finite() {
            let half = F::HALF;
            if ax >= half && ax <= F::from_f64(1.5) {
                // close to the unit circle: log1p keeps the precision of log|z| near 0
                half * (((ax - F::ONE) * (ax + F::ONE)) + ay * ay).ln_1p()
            } else {
                let h = ax.hypot(ay);
                if h.is_finite() {
                    h.ln()
                } else {
                    (ax * half).hypot(ay * half).ln() + F::LN_2
                }
            }
        } else {
            x.hypot(y).ln() // inf beats NaN, NaN otherwise
        };
        Complex::new(re, y.atan2(x))
    }

    /// Principal square root, like `np.sqrt(z)` (C99 `csqrt`, scaled against overflow).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(-4.0, 0.0).sqrt(), np::Complex::new(0.0, 2.0));
    /// assert_eq!(np::Complex::new(-4.0, -0.0).sqrt(), np::Complex::new(0.0, -2.0));
    /// ```
    pub fn sqrt(self) -> Self {
        let (mut a, mut b) = (self.re, self.im);
        if a == F::ZERO && b == F::ZERO {
            return Complex::new(F::ZERO, b);
        }
        if b.is_infinite() {
            return Complex::new(F::INFINITY, b);
        }
        if a.is_nan() {
            return Complex::new(a, F::NAN);
        }
        if a.is_infinite() {
            return if a.sign_bit() {
                Complex::new(zero_or_nan(b), a.copy_sign(b))
            } else {
                Complex::new(a, zero_or_nan(b).copy_sign(b))
            };
        }
        let scale = a.abs() >= F::SQRT_THRESH || b.abs() >= F::SQRT_THRESH;
        if scale {
            let q = F::from_f64(0.25);
            a = a * q;
            b = b * q;
        }
        let r = if a >= F::ZERO {
            let t = ((a + a.hypot(b)) * F::HALF).sqrt();
            Complex::new(t, b / (t + t))
        } else {
            let t = ((-a + a.hypot(b)) * F::HALF).sqrt();
            Complex::new(b.abs() / (t + t), t.copy_sign(b))
        };
        if scale {
            Complex::new(r.re * F::TWO, r.im * F::TWO)
        } else {
            r
        }
    }

    /// Complex power, like `z ** w` / `np.power(z, w)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::Complex::new(1.0, 2.0);
    /// assert_eq!(z.powc(np::Complex::new(2.0, 0.0)), z * z);
    /// assert_eq!(z.powc(np::Complex::new(0.0, 0.0)), np::Complex::new(1.0, 0.0));
    /// ```
    pub fn powc(self, w: Self) -> Self {
        let one = Complex::from_real(F::ONE);
        if w.re == F::ZERO && w.im == F::ZERO {
            return one;
        }
        if self.re == F::ZERO && self.im == F::ZERO {
            return if w.re > F::ZERO {
                Complex::new(F::ZERO, F::ZERO)
            } else {
                Complex::new(F::NAN, F::NAN)
            };
        }
        if w.im == F::ZERO && w.re == w.re.trunc() && w.re.abs() < F::from_f64(100.0) {
            let n = w.re.to_f64() as i64;
            match n {
                1 => return self,
                2 => return self * self,
                3 => return self * (self * self),
                _ => {}
            }
            let mut k = n.unsigned_abs();
            let mut acc = one;
            let mut p = self;
            loop {
                if k & 1 == 1 {
                    acc *= p;
                }
                k >>= 1;
                if k == 0 {
                    break;
                }
                p = p * p;
            }
            return if n < 0 { one / acc } else { acc };
        }
        (w * self.ln()).exp()
    }

    /// Complex raised to a real power, like `z ** p` (NumPy promotes `p` to `p + 0j`).
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::Complex::new(3.0, 4.0);
    /// assert_eq!(z.powf(2.0), np::Complex::new(-7.0, 24.0));
    /// ```
    pub fn powf(self, p: F) -> Self {
        self.powc(Complex::from_real(p))
    }

    /// Complex sine, like `np.sin(z)` (C99 `csin`).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(0.0, 0.0).sin(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn sin(self) -> Self {
        let w = csinh(Complex::new(-self.im, self.re));
        Complex::new(w.im, -w.re)
    }

    /// Complex cosine, like `np.cos(z)` (C99 `ccos`).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(0.0, 0.0).cos().re, 1.0);
    /// ```
    pub fn cos(self) -> Self {
        ccosh(Complex::new(-self.im, self.re))
    }

    /// Complex tangent, like `np.tan(z)` (C99 `ctan`).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(0.0, 0.0).tan(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn tan(self) -> Self {
        let w = ctanh(Complex::new(-self.im, self.re));
        Complex::new(w.im, -w.re)
    }

    /// Complex hyperbolic sine, like `np.sinh(z)` (C99 `csinh`).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(0.0, 0.0).sinh(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn sinh(self) -> Self {
        csinh(self)
    }

    /// Complex hyperbolic cosine, like `np.cosh(z)` (C99 `ccosh`).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(0.0, 0.0).cosh().re, 1.0);
    /// ```
    pub fn cosh(self) -> Self {
        ccosh(self)
    }

    /// Complex hyperbolic tangent, like `np.tanh(z)` (C99 `ctanh`, Kahan's algorithm).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Complex::new(0.0, 0.0).tanh(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn tanh(self) -> Self {
        ctanh(self)
    }

    /// Ordering used by `np.sort` on complex values: lexicographic on `(re, im)` with NaNs last.
    ///
    /// ```
    /// use rustnum as np;
    /// use std::cmp::Ordering;
    /// let (a, b) = (np::Complex::new(1.0, 5.0), np::Complex::new(2.0, 0.0));
    /// assert_eq!(a.cmp_numpy(&b), Ordering::Less);
    /// assert_eq!(np::Complex::new(f64::NAN, 0.0).cmp_numpy(&b), Ordering::Greater);
    /// ```
    pub fn cmp_numpy(&self, other: &Self) -> Ordering {
        if clt(self, other) {
            Ordering::Less
        } else if clt(other, self) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

/// NumPy's `CLT` for complex sorting.
fn clt<F: Float>(a: &Complex<F>, b: &Complex<F>) -> bool {
    if a.re < b.re {
        !a.im.is_nan() || b.im.is_nan()
    } else if a.re > b.re {
        b.im.is_nan() && !a.im.is_nan()
    } else if a.re == b.re || (a.re.is_nan() && b.re.is_nan()) {
        a.im < b.im || (b.im.is_nan() && !a.im.is_nan())
    } else {
        b.re.is_nan()
    }
}

/// `b - b` of C: zero for finite `b`, NaN for NaN or infinity.
fn zero_or_nan<F: Float>(b: F) -> F {
    if b.is_finite() { F::ZERO } else { F::NAN }
}

fn csinh<F: Float>(z: Complex<F>) -> Complex<F> {
    let (x, y) = (z.re, z.im);
    if x.is_finite() && y.is_finite() {
        if y == F::ZERO {
            return Complex::new(x.sinh(), y);
        }
        let ax = x.abs();
        if ax < F::HYPERBOLIC_DIRECT {
            return Complex::new(x.sinh() * y.cos(), x.cosh() * y.sin());
        }
        if ax < F::MAX_VALUE.ln() {
            let h = ax.exp() * F::HALF;
            return Complex::new(h.copy_sign(x) * y.cos(), h * y.sin());
        }
        let s = (ax * F::HALF).exp();
        let h = s * F::HALF;
        return Complex::new((h.copy_sign(x) * y.cos()) * s, (h * y.sin()) * s);
    }
    if x == F::ZERO {
        return Complex::new(x, F::NAN);
    }
    if y == F::ZERO {
        return Complex::new(x, y);
    }
    if x.is_finite() {
        return Complex::new(F::NAN, F::NAN);
    }
    if x.is_infinite() {
        if !y.is_finite() {
            return Complex::new(x * x, F::NAN);
        }
        return Complex::new(x * y.cos(), F::INFINITY * y.sin());
    }
    Complex::new(F::NAN, F::NAN)
}

fn ccosh<F: Float>(z: Complex<F>) -> Complex<F> {
    let (x, y) = (z.re, z.im);
    if x.is_finite() && y.is_finite() {
        if y == F::ZERO {
            return Complex::new(x.cosh(), x * y);
        }
        let ax = x.abs();
        if ax < F::HYPERBOLIC_DIRECT {
            return Complex::new(x.cosh() * y.cos(), x.sinh() * y.sin());
        }
        if ax < F::MAX_VALUE.ln() {
            let h = ax.exp() * F::HALF;
            return Complex::new(h * y.cos(), h.copy_sign(x) * y.sin());
        }
        let s = (ax * F::HALF).exp();
        let h = s * F::HALF;
        return Complex::new((h * y.cos()) * s, (h.copy_sign(x) * y.sin()) * s);
    }
    if x == F::ZERO {
        return Complex::new(F::NAN, x * F::ZERO.copy_sign(y));
    }
    if y == F::ZERO {
        return Complex::new(x * x, F::ZERO.copy_sign(x) * y);
    }
    if x.is_finite() {
        return Complex::new(F::NAN, F::NAN);
    }
    if x.is_infinite() {
        if !y.is_finite() {
            return Complex::new(x * x, F::NAN);
        }
        return Complex::new((x * x) * y.cos(), x * y.sin());
    }
    Complex::new(F::NAN, F::NAN)
}

fn ctanh<F: Float>(z: Complex<F>) -> Complex<F> {
    let (x, y) = (z.re, z.im);
    if !x.is_finite() {
        if x.is_nan() {
            return Complex::new(x, if y == F::ZERO { y } else { x * y });
        }
        let t = if y.is_infinite() {
            y
        } else {
            y.sin() * y.cos()
        };
        return Complex::new(F::ONE.copy_sign(x), F::ZERO.copy_sign(t));
    }
    if !y.is_finite() {
        return Complex::new(F::NAN, F::NAN);
    }
    if x.abs() >= F::HYPERBOLIC_DIRECT {
        let e = (-x.abs()).exp();
        let four = F::from_f64(4.0);
        return Complex::new(F::ONE.copy_sign(x), four * y.sin() * y.cos() * e * e);
    }
    let t = y.tan();
    let beta = F::ONE + t * t;
    let s = x.sinh();
    let rho = (F::ONE + s * s).sqrt();
    let denom = F::ONE + beta * s * s;
    Complex::new((beta * rho * s) / denom, t / denom)
}

// ---------------------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------------------

impl<F: Float> Add for Complex<F> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl<F: Float> Sub for Complex<F> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl<F: Float> Mul for Complex<F> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// Smith's algorithm, exactly as in NumPy's `complex division` loop.
impl<F: Float> Div for Complex<F> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let (ar, ai, br, bi) = (self.re, self.im, o.re, o.im);
        let (br_abs, bi_abs) = (br.abs(), bi.abs());
        if br_abs >= bi_abs {
            if br_abs == F::ZERO && bi_abs == F::ZERO {
                // division by zero gives a complex inf or nan
                Complex::new(ar / br_abs, ai / br_abs)
            } else {
                let rat = bi / br;
                let scl = F::ONE / (br + bi * rat);
                Complex::new((ar + ai * rat) * scl, (ai - ar * rat) * scl)
            }
        } else {
            let rat = br / bi;
            let scl = F::ONE / (bi + br * rat);
            Complex::new((ar * rat + ai) * scl, (ai * rat - ar) * scl)
        }
    }
}

impl<F: Float> Neg for Complex<F> {
    type Output = Self;
    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}

// Real scalar on the right. NumPy promotes the scalar to `s + 0j` first, and so do we, so
// results with infinities and signed zeros match NumPy bit for bit.
macro_rules! scalar_rhs {
    ($($tr:ident $m:ident $atr:ident $am:ident),*) => {$(
        impl<F: Float> $tr<F> for Complex<F> {
            type Output = Complex<F>;
            fn $m(self, s: F) -> Complex<F> {
                $tr::$m(self, Complex::from_real(s))
            }
        }
        impl<F: Float> $atr for Complex<F> {
            fn $am(&mut self, o: Complex<F>) {
                *self = $tr::$m(*self, o);
            }
        }
        impl<F: Float> $atr<F> for Complex<F> {
            fn $am(&mut self, s: F) {
                *self = $tr::$m(*self, Complex::from_real(s));
            }
        }
    )*};
}
scalar_rhs!(Add add AddAssign add_assign, Sub sub SubAssign sub_assign,
            Mul mul MulAssign mul_assign, Div div DivAssign div_assign);

// Real scalar on the left (cannot be generic because of the orphan rule).
macro_rules! scalar_lhs {
    ($($t:ty),*) => {$(
        impl Add<Complex<$t>> for $t {
            type Output = Complex<$t>;
            fn add(self, z: Complex<$t>) -> Complex<$t> { Complex::from_real(self) + z }
        }
        impl Sub<Complex<$t>> for $t {
            type Output = Complex<$t>;
            fn sub(self, z: Complex<$t>) -> Complex<$t> { Complex::from_real(self) - z }
        }
        impl Mul<Complex<$t>> for $t {
            type Output = Complex<$t>;
            fn mul(self, z: Complex<$t>) -> Complex<$t> { Complex::from_real(self) * z }
        }
        impl Div<Complex<$t>> for $t {
            type Output = Complex<$t>;
            fn div(self, z: Complex<$t>) -> Complex<$t> { Complex::from_real(self) / z }
        }
    )*};
}
scalar_lhs!(f32, f64);

// ---------------------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------------------

/// Python / NumPy shortest repr of `|x|` without sign and without a trailing `.0`.
fn py_abs_repr<F: Float>(x: F) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    let x = x.abs();
    if x.is_infinite() {
        return "inf".to_string();
    }
    if x == F::ZERO {
        return "0".to_string();
    }
    let sci = format!("{x:e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp10: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp10 + 1;
    // NumPy: positional notation when 1e-4 <= |x| < 1e16 (1e6 for float32), else scientific.
    let xf = x.to_f64();
    if xf >= 1e-4 && xf < F::REPR_SCI_UPPER {
        let n = digits.len() as i32;
        if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt >= n {
            format!("{}{}", digits, "0".repeat((decpt - n) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..decpt as usize],
                &digits[decpt as usize..]
            )
        }
    } else {
        let rest = &digits[1..];
        let sign = if exp10 < 0 { '-' } else { '+' };
        let dot = if rest.is_empty() { "" } else { "." };
        format!("{}{}{}e{}{:02}", &digits[..1], dot, rest, sign, exp10.abs())
    }
}

/// Like `str(np.complex128(...))`: `(1+2j)`, `(-0.5-1j)`, `2j` when the real part is `+0`.
impl<F: Float> fmt::Display for Complex<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let im_neg = self.im.sign_bit() && !self.im.is_nan();
        let re_neg = self.re.sign_bit() && !self.re.is_nan();
        let im = py_abs_repr(self.im);
        if self.re == F::ZERO && !re_neg {
            let sign = if im_neg { "-" } else { "" };
            return write!(f, "{sign}{im}j");
        }
        let re = py_abs_repr(self.re);
        let rs = if re_neg { "-" } else { "" };
        let is = if im_neg { "-" } else { "+" };
        write!(f, "({rs}{re}{is}{im}j)")
    }
}
