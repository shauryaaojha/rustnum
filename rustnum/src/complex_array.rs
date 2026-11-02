//! Complex arrays: `Complex<F>` as an array element (`complex64` / `complex128`).
//!
//! `Complex<f32>` is NumPy's `complex64` and `Complex<f64>` is `complex128`. The scalar
//! arithmetic and elementary functions live in `misc/complex.rs`; this file makes the type an
//! [`Element`] and [`Num`], adds mixed real/complex operators and the complex-only array
//! methods (`real`, `imag`, `conj`, `angle`, `exp`, `log`, `sqrt`, ...).

use crate::element::add;
use crate::{Array, Complex, Element, Float, Num};
use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Rem, Sub};

/// Lexicographic order on `(re, im)`, like NumPy's complex comparison and `np.sort`.
///
/// NaN in either compared part gives `None`, so `max`/`min`/`sort` treat NaN-containing values
/// like NaN floats.
/// ```
/// use rustnum as np;
/// assert!(np::Complex::new(1.0, 5.0) < np::Complex::new(2.0, 0.0));
/// assert!(np::Complex::new(1.0, 1.0) < np::Complex::new(1.0, 2.0));
/// ```
impl<F: Float> PartialOrd for Complex<F> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        match self.re.partial_cmp(&o.re)? {
            Ordering::Equal => self.im.partial_cmp(&o.im),
            ord => Some(ord),
        }
    }
}

/// `Complex<f32>` is `complex64` and `Complex<f64>` is `complex128`.
///
/// `astype` keeps both parts between complex dtypes and takes the real part when casting to a
/// real dtype, like NumPy (which also warns). Casting a real value gives `x + 0j`.
/// ```
/// use rustnum as np;
/// assert_eq!(<np::Complex<f64> as np::Element>::DTYPE, "complex128");
/// assert_eq!(<np::Complex<f32> as np::Element>::DTYPE, "complex64");
/// let z = np::array![1.5f64, -2.0].astype::<np::Complex<f64>>();
/// assert_eq!(z.to_vec(), [np::Complex::new(1.5, 0.0), np::Complex::new(-2.0, 0.0)]);
/// assert_eq!(z.astype::<f64>().to_vec(), [1.5, -2.0]);
/// ```
impl<F: Float> Element for Complex<F> {
    const DTYPE: &'static str = F::COMPLEX_DTYPE;
    fn to_f64(self) -> f64 {
        self.re.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        Complex::from_real(F::from_f64(v))
    }
    fn to_complex_parts(self) -> (f64, f64) {
        (self.re.to_f64(), self.im.to_f64())
    }
    fn from_complex_parts(re: f64, im: f64) -> Self {
        Complex::new(F::from_f64(re), F::from_f64(im))
    }
    fn npy_descr() -> Option<String> {
        F::COMPLEX_DESCR.map(str::to_string)
    }
    fn npy_itemsize() -> usize {
        2 * F::BYTES
    }
    fn npy_write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.re.to_bits64().to_le_bytes()[..F::BYTES]);
        out.extend_from_slice(&self.im.to_bits64().to_le_bytes()[..F::BYTES]);
    }
    fn npy_read(b: &[u8]) -> Self {
        let part = |s: &[u8]| {
            let mut w = [0u8; 8];
            w[..F::BYTES].copy_from_slice(&s[..F::BYTES]);
            F::from_bits64(u64::from_le_bytes(w))
        };
        Complex::new(part(b), part(&b[F::BYTES..]))
    }
}

/// Complex numbers have no remainder; like NumPy this raises (panics).
impl<F: Float> Rem for Complex<F> {
    type Output = Self;
    fn rem(self, _: Self) -> Self {
        panic!(
            "ufunc 'remainder' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''"
        )
    }
}

/// Complex arrays take part in every `Num` routine (`sum`, `prod`, `cumsum`, `matmul`, `dot`,
/// `trace`, ...). `Num::abs` returns `|z| + 0j`; use [`absolute`] for the real modulus array.
/// ```
/// use rustnum as np;
/// let z = np::array![np::Complex::new(3.0f64, 4.0)];
/// assert_eq!(z.abs().item(), np::Complex::new(5.0, 0.0));
/// assert_eq!(np::absolute(&z).item(), 5.0);
/// ```
impl<F: Float> Num for Complex<F> {
    const ZERO: Self = Complex {
        re: F::ZERO,
        im: F::ZERO,
    };
    const ONE: Self = Complex {
        re: F::ONE,
        im: F::ZERO,
    };
    const MIN_VALUE: Self = Complex {
        re: F::MIN_VALUE,
        im: F::MIN_VALUE,
    };
    const MAX_VALUE: Self = Complex {
        re: F::MAX_VALUE,
        im: F::MAX_VALUE,
    };
    fn abs(self) -> Self {
        Complex::from_real(self.norm())
    }
    fn from_usize(v: usize) -> Self {
        Complex::from_real(F::from_usize(v))
    }
}

// ---------------------------------------------------------------------------------------
// Mixed real / complex operators (the real operand is promoted to `x + 0j`, like NumPy)
// ---------------------------------------------------------------------------------------

macro_rules! mixed_ops {
    ($Tr:ident, $m:ident) => {
        impl<F: Float> $Tr<Array<F>> for Array<Complex<F>> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: Array<F>) -> Self::Output {
                self.par_zip(&r, |x, y| $Tr::$m(x, Complex::from_real(y)))
            }
        }
        impl<F: Float> $Tr<&Array<F>> for Array<Complex<F>> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: &Array<F>) -> Self::Output {
                self.par_zip(r, |x, y| $Tr::$m(x, Complex::from_real(y)))
            }
        }
        impl<F: Float> $Tr<Array<F>> for &Array<Complex<F>> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: Array<F>) -> Self::Output {
                self.par_zip(&r, |x, y| $Tr::$m(x, Complex::from_real(y)))
            }
        }
        impl<F: Float> $Tr<&Array<F>> for &Array<Complex<F>> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: &Array<F>) -> Self::Output {
                self.par_zip(r, |x, y| $Tr::$m(x, Complex::from_real(y)))
            }
        }
        impl<F: Float> $Tr<Array<Complex<F>>> for Array<F> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: Array<Complex<F>>) -> Self::Output {
                self.par_zip(&r, |x, y| $Tr::$m(Complex::from_real(x), y))
            }
        }
        impl<F: Float> $Tr<&Array<Complex<F>>> for Array<F> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: &Array<Complex<F>>) -> Self::Output {
                self.par_zip(r, |x, y| $Tr::$m(Complex::from_real(x), y))
            }
        }
        impl<F: Float> $Tr<Array<Complex<F>>> for &Array<F> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: Array<Complex<F>>) -> Self::Output {
                self.par_zip(&r, |x, y| $Tr::$m(Complex::from_real(x), y))
            }
        }
        impl<F: Float> $Tr<&Array<Complex<F>>> for &Array<F> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: &Array<Complex<F>>) -> Self::Output {
                self.par_zip(r, |x, y| $Tr::$m(Complex::from_real(x), y))
            }
        }
        impl<F: Float> $Tr<F> for Array<Complex<F>> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: F) -> Self::Output {
                self.into_map(|x| $Tr::$m(x, Complex::from_real(r)))
            }
        }
        impl<F: Float> $Tr<F> for &Array<Complex<F>> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: F) -> Self::Output {
                self.par_map_cheap(|x| $Tr::$m(x, Complex::from_real(r)))
            }
        }
        impl<F: Float> $Tr<Array<Complex<F>>> for Complex<F> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: Array<Complex<F>>) -> Self::Output {
                r.into_map(|y| $Tr::$m(self, y))
            }
        }
        impl<F: Float> $Tr<&Array<Complex<F>>> for Complex<F> {
            type Output = Array<Complex<F>>;
            fn $m(self, r: &Array<Complex<F>>) -> Self::Output {
                r.par_map_cheap(|y| $Tr::$m(self, y))
            }
        }
    };
}
mixed_ops!(Add, add);
mixed_ops!(Sub, sub);
mixed_ops!(Mul, mul);
mixed_ops!(Div, div);

macro_rules! real_scalar_left {
    ($t:ty) => {
        real_scalar_left!($t, Add, add);
        real_scalar_left!($t, Sub, sub);
        real_scalar_left!($t, Mul, mul);
        real_scalar_left!($t, Div, div);
    };
    ($t:ty, $Tr:ident, $m:ident) => {
        impl $Tr<Array<Complex<$t>>> for $t {
            type Output = Array<Complex<$t>>;
            fn $m(self, r: Array<Complex<$t>>) -> Self::Output {
                r.into_map(|y| $Tr::$m(Complex::from_real(self), y))
            }
        }
        impl $Tr<&Array<Complex<$t>>> for $t {
            type Output = Array<Complex<$t>>;
            fn $m(self, r: &Array<Complex<$t>>) -> Self::Output {
                r.par_map_cheap(|y| $Tr::$m(Complex::from_real(self), y))
            }
        }
    };
}
real_scalar_left!(f32);
real_scalar_left!(f64);

// ---------------------------------------------------------------------------------------
// Complex-only array methods
// ---------------------------------------------------------------------------------------

impl<F: Float> Array<Complex<F>> {
    /// Real parts, like `a.real` / `np.real(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(1.0, 2.0), np::Complex::new(3.0, -4.0)];
    /// assert_eq!(z.real().to_vec(), [1.0, 3.0]);
    /// ```
    pub fn real(&self) -> Array<F> {
        self.par_map_cheap(|z| z.re)
    }

    /// Imaginary parts, like `a.imag` / `np.imag(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(1.0, 2.0), np::Complex::new(3.0, -4.0)];
    /// assert_eq!(z.imag().to_vec(), [2.0, -4.0]);
    /// ```
    pub fn imag(&self) -> Array<F> {
        self.par_map_cheap(|z| z.im)
    }

    /// Complex conjugate, like `np.conj(a)` / `np.conjugate(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(1.0, 2.0)];
    /// assert_eq!(z.conj().item(), np::Complex::new(1.0, -2.0));
    /// ```
    pub fn conj(&self) -> Self {
        self.par_map_cheap(Complex::conj)
    }

    /// Argument in radians, like `np.angle(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, 1.0), np::Complex::new(-1.0, 0.0)];
    /// assert_eq!(z.angle().to_vec(), [std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
    /// ```
    pub fn angle(&self) -> Array<F> {
        self.par_map(Complex::arg)
    }

    /// Argument in degrees, like `np.angle(z, deg=True)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0f64, 1.0)];
    /// assert!((z.angle_deg().item() - 90.0).abs() < 1e-12);
    /// ```
    pub fn angle_deg(&self) -> Array<F> {
        let k = F::from_f64(180.0) / F::PI;
        self.par_map(move |z| z.arg() * k)
    }

    /// Complex exponential, like `np.exp(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, 0.0)];
    /// assert_eq!(z.exp().item(), np::Complex::new(1.0, 0.0));
    /// ```
    pub fn exp(&self) -> Self {
        self.par_map(Complex::exp)
    }

    /// Principal natural logarithm, like `np.log(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(1.0, 0.0)];
    /// assert_eq!(z.log().item(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn log(&self) -> Self {
        self.par_map(Complex::ln)
    }

    /// Base-2 logarithm, like `np.log2(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(8.0f64, 0.0)];
    /// assert!((z.log2().item().re - 3.0).abs() < 1e-14);
    /// ```
    pub fn log2(&self) -> Self {
        self.par_map(|z| {
            let l = z.ln();
            Complex::new(l.re * F::LOG2_E, l.im * F::LOG2_E)
        })
    }

    /// Base-10 logarithm, like `np.log10(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(100.0f64, 0.0)];
    /// assert!((z.log10().item().re - 2.0).abs() < 1e-14);
    /// ```
    pub fn log10(&self) -> Self {
        self.par_map(|z| {
            let l = z.ln();
            let k = F::from_f64(std::f64::consts::LOG10_E);
            Complex::new(l.re * k, l.im * k)
        })
    }

    /// Principal square root, like `np.sqrt(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(-4.0, 0.0)];
    /// assert_eq!(z.sqrt().item(), np::Complex::new(0.0, 2.0));
    /// ```
    pub fn sqrt(&self) -> Self {
        self.par_map(Complex::sqrt)
    }

    /// Complex sine, like `np.sin(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, 0.0)];
    /// assert_eq!(z.sin().item(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn sin(&self) -> Self {
        self.par_map(Complex::sin)
    }

    /// Complex cosine, like `np.cos(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, 0.0)];
    /// assert_eq!(z.cos().item().re, 1.0);
    /// ```
    pub fn cos(&self) -> Self {
        self.par_map(Complex::cos)
    }

    /// Complex tangent, like `np.tan(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, 0.0)];
    /// assert_eq!(z.tan().item(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn tan(&self) -> Self {
        self.par_map(Complex::tan)
    }

    /// Complex hyperbolic sine, like `np.sinh(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, 0.0)];
    /// assert_eq!(z.sinh().item(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn sinh(&self) -> Self {
        self.par_map(Complex::sinh)
    }

    /// Complex hyperbolic cosine, like `np.cosh(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, 0.0)];
    /// assert_eq!(z.cosh().item().re, 1.0);
    /// ```
    pub fn cosh(&self) -> Self {
        self.par_map(Complex::cosh)
    }

    /// Complex hyperbolic tangent, like `np.tanh(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, 0.0)];
    /// assert_eq!(z.tanh().item(), np::Complex::new(0.0, 0.0));
    /// ```
    pub fn tanh(&self) -> Self {
        self.par_map(Complex::tanh)
    }

    /// True where either part is NaN, like `np.isnan(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(f64::NAN, 0.0), np::Complex::new(1.0, 0.0)];
    /// assert_eq!(z.is_nan().to_vec(), [true, false]);
    /// ```
    pub fn is_nan(&self) -> Array<bool> {
        self.par_map(Complex::is_nan)
    }

    /// True where either part is infinite, like `np.isinf(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(0.0, f64::INFINITY), np::Complex::new(1.0, 0.0)];
    /// assert_eq!(z.is_inf().to_vec(), [true, false]);
    /// ```
    pub fn is_inf(&self) -> Array<bool> {
        self.par_map(Complex::is_infinite)
    }

    /// True where both parts are finite, like `np.isfinite(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(f64::NAN, 0.0), np::Complex::new(1.0, 0.0)];
    /// assert_eq!(z.is_finite().to_vec(), [false, true]);
    /// ```
    pub fn is_finite(&self) -> Array<bool> {
        self.par_map(Complex::is_finite)
    }

    /// Mean of all elements, like `np.mean(z)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![np::Complex::new(1.0, 2.0), np::Complex::new(3.0, 4.0)];
    /// assert_eq!(z.mean(), np::Complex::new(2.0, 3.0));
    /// ```
    pub fn mean(&self) -> Complex<F> {
        self.sum() / Complex::from_real(F::from_usize(self.size()))
    }

    /// Mean along one axis, like `np.mean(z, axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let z = np::array![[np::Complex::new(1.0, 1.0), np::Complex::new(3.0, 3.0)]];
    /// assert_eq!(z.mean_axis(1).to_vec(), [np::Complex::new(2.0, 2.0)]);
    /// ```
    pub fn mean_axis(&self, axis: isize) -> Array<Complex<F>> {
        let n = self.shape()[crate::slicing::axis_index(axis, self.ndim())];
        let k = Complex::from_real(F::from_usize(n));
        self.sum_axis(axis).par_map_cheap(move |z| z / k)
    }

    /// Dot product that conjugates the first argument, like `np.vdot(a, b)`.
    ///
    /// Both arrays are flattened. Panics when the sizes differ. (`np::linalg::vdot` is generic
    /// over `Num` and does not conjugate; use this method for complex data.)
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::array![np::Complex::new(1.0, 2.0), np::Complex::new(3.0, 4.0)];
    /// let b = np::array![np::Complex::new(5.0, 6.0), np::Complex::new(7.0, 8.0)];
    /// assert_eq!(a.vdot(&b), np::Complex::new(70.0, -8.0));
    /// ```
    pub fn vdot(&self, other: &Array<Complex<F>>) -> Complex<F> {
        assert!(
            self.size() == other.size(),
            "vectors have different lengths"
        );
        self.iter()
            .zip(other.iter())
            .fold(Complex::from_real(F::ZERO), |s, (x, y)| {
                add(s, x.conj() * y)
            })
    }
}

/// Element types whose `np.absolute` is defined; the output is real for complex input.
///
/// ```
/// use rustnum as np;
/// let z = np::array![np::Complex::new(3.0, 4.0), np::Complex::new(0.0, -1.0)];
/// assert_eq!(np::absolute(&z).to_vec(), [5.0, 1.0]);
/// assert_eq!(np::absolute(&np::array![-2, 3]).to_vec(), [2, 3]);
/// ```
pub trait Absolute {
    /// Real element type of the result.
    type Output: Element;
    /// Absolute value, like `np.absolute(a)`.
    /// ```
    /// use rustnum as np;
    /// use np::Absolute;
    /// assert_eq!(np::array![-1.5].absolute().to_vec(), [1.5]);
    /// ```
    fn absolute(&self) -> Array<Self::Output>;
}
macro_rules! absolute_real {
    ($($t:ty),*) => {$(
        impl Absolute for Array<$t> {
            type Output = $t;
            fn absolute(&self) -> Array<$t> { self.abs() }
        }
    )*};
}
absolute_real!(f32, f64, i32, i64, u8, usize, crate::f16, crate::bf16);
impl<F: Float> Absolute for Array<Complex<F>> {
    type Output = F;
    fn absolute(&self) -> Array<F> {
        self.par_map(Complex::norm)
    }
}
/// Absolute value, like `np.absolute(a)` / `np.abs(a)`; complex input gives a real array.
///
/// ```
/// use rustnum as np;
/// let z = np::array![np::Complex::new(3.0f32, 4.0)];
/// assert_eq!(np::absolute(&z).to_vec(), [5.0f32]);
/// ```
pub fn absolute<A: Absolute>(a: &A) -> Array<A::Output> {
    a.absolute()
}

#[cfg(test)]
mod tests {
    use super::*;
    type C = Complex<f64>;
    fn c(re: f64, im: f64) -> C {
        Complex::new(re, im)
    }

    #[test]
    fn broadcast_complex_arithmetic() {
        let a = crate::array![[c(1., 2.)], [c(3., 4.)]];
        let b = crate::array![c(1., 0.), c(0., 1.)];
        let p = &a * &b;
        assert_eq!(p.shape(), [2, 2]);
        assert_eq!(p.to_vec(), [c(1., 2.), c(-2., 1.), c(3., 4.), c(-4., 3.)]);
    }

    #[test]
    fn mixed_real_complex() {
        let z = crate::array![c(1., 2.), c(3., -4.)];
        let r = crate::array![2.0, 0.5];
        assert_eq!((&z + &r).to_vec(), [c(3., 2.), c(3.5, -4.)]);
        assert_eq!((&r * &z).to_vec(), [c(2., 4.), c(1.5, -2.)]);
        assert_eq!((&z * 2.0).to_vec(), [c(2., 4.), c(6., -8.)]);
        assert_eq!((2.0 * &z).to_vec(), [c(2., 4.), c(6., -8.)]);
        assert_eq!((c(0., 1.) * &z).to_vec(), [c(-2., 1.), c(4., 3.)]);
    }

    #[test]
    fn negation_keeps_signed_zero() {
        let z = crate::array![c(0., 0.)];
        let n = -&z;
        assert!(n.item().re.is_sign_negative() && n.item().im.is_sign_negative());
    }

    #[test]
    fn display_like_numpy() {
        let z = crate::array![c(1., 2.), c(3., -4.)];
        assert_eq!(z.to_string(), "[1.+2.j 3.-4.j]");
        let z = crate::array![c(1.5, 2.), c(3., -4.)];
        assert_eq!(z.to_string(), "[1.5+2.j 3. -4.j]");
        assert_eq!(
            format!("{:?}", crate::array![c(0., 1.)]),
            "array([0.+1.j], dtype=complex128)"
        );
    }

    #[test]
    fn astype_between_complex_widths() {
        let z = crate::array![c(1.25, -2.5)];
        let w = z.astype::<Complex<f32>>();
        assert_eq!(w.item(), Complex::new(1.25f32, -2.5));
        assert_eq!(w.astype::<C>().item(), c(1.25, -2.5));
    }

    #[test]
    fn complex_sort_is_lexicographic() {
        let z = crate::array![c(2., 0.), c(1., 5.), c(1., -1.)];
        assert_eq!(z.sort().to_vec(), [c(1., -1.), c(1., 5.), c(2., 0.)]);
        assert_eq!(z.argmax(), 0);
        assert_eq!(z.argmin(), 2);
    }

    #[test]
    #[should_panic(expected = "remainder")]
    fn remainder_is_unsupported() {
        let z = crate::array![c(1., 1.)];
        let _ = &z % &z;
    }
}
