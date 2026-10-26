//! Public `Array` methods and free functions for NumPy's remaining element-wise math, the
//! window functions, `ediff1d` and `angle`. They map the kernels of `math_kernels.rs` and
//! `window_kernels.rs` over arrays.
//!
//! Everything is bounded by the core [`Float`] (or [`Num`]) trait, so generic user code over
//! `F: rustnum::Float` can call all of it.
//!
//! Functionality the core already exports is not repeated here: `rint`, `fix`, `fabs`, `positive`
//! and `negative` are the NumPy names for the core `round`, `trunc`, `abs`, a copy and unary `-`,
//! and delegate to them (no second implementation).

use core::f64::consts::TAU;

use super::math_kernels as k;
use super::window_kernels as w;
use crate::{Array, Float, IntoArray, Num};

// ---------------------------------------------------------------------------------------
// Unary methods
// ---------------------------------------------------------------------------------------

impl<F: Float> Array<F> {
    /// Equivalent of `np.sinc(a)`: normalized sinc `sin(pi x) / (pi x)`, with `sinc(0) = 1`.
    ///
    /// ```
    /// use rustnum as np;
    /// let v: Vec<f64> = np::array![0.0, 0.5, 1.5].sinc().to_vec();
    /// assert_eq!(v[0], 1.0);
    /// assert!((v[1] - 0.6366197723675814).abs() < 1e-15);
    /// assert!((v[2] + 0.2122065907891938).abs() < 1e-15);
    /// ```
    pub fn sinc(&self) -> Array<F> {
        self.par_map(k::sinc::<F>)
    }

    /// Equivalent of `np.i0(a)`: modified Bessel function of the first kind, order 0.
    ///
    /// ```
    /// use rustnum as np;
    /// let v: Vec<f64> = np::array![0.0, 1.0, 2.0].i0().to_vec();
    /// assert_eq!(v[0], 1.0);
    /// assert!((v[1] - 1.2660658777520082).abs() < 1e-14);
    /// assert!((v[2] - 2.279585302336067).abs() < 1e-14);
    /// ```
    pub fn i0(&self) -> Array<F> {
        self.par_map(k::i0::<F>)
    }

    /// Equivalent of `np.deg2rad(a)`: degrees to radians.
    ///
    /// ```
    /// use rustnum as np;
    /// use std::f64::consts::{FRAC_PI_2, PI};
    /// assert_eq!(np::array![0.0, 90.0, 180.0].deg2rad().to_vec(), [0.0, FRAC_PI_2, PI]);
    /// ```
    pub fn deg2rad(&self) -> Array<F> {
        self.par_map(k::deg2rad::<F>)
    }

    /// Equivalent of `np.rad2deg(a)`: radians to degrees.
    ///
    /// ```
    /// use rustnum as np;
    /// use std::f64::consts::{FRAC_PI_2, PI};
    /// assert_eq!(np::array![0.0, FRAC_PI_2, PI].rad2deg().to_vec(), [0.0, 90.0, 180.0]);
    /// ```
    pub fn rad2deg(&self) -> Array<F> {
        self.par_map(k::rad2deg::<F>)
    }

    /// Equivalent of `np.rint(a)`: round to the nearest integer, halves to even, keeping `-0.0`.
    ///
    /// This is the core [`Array::round`] under its other NumPy name.
    ///
    /// ```
    /// use rustnum as np;
    /// let r: Vec<f64> = np::array![0.5, 1.5, 2.5, -0.5, -1.5].rint().to_vec();
    /// assert_eq!(r, [0.0, 2.0, 2.0, 0.0, -2.0]);
    /// assert!(r[3].is_sign_negative());
    /// ```
    pub fn rint(&self) -> Array<F> {
        self.round()
    }

    /// Equivalent of `np.fix(a)`: round towards zero (`fix(-0.4)` is `-0.0`).
    ///
    /// This is the core [`Array::trunc`] under its other NumPy name.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![2.7, -2.7, -0.4].fix().to_vec(), [2.0, -2.0, 0.0]);
    /// ```
    pub fn fix(&self) -> Array<F> {
        self.trunc()
    }

    /// Equivalent of `np.fabs(a)`: absolute value of floats.
    ///
    /// This is the core [`Array::abs`] under its other NumPy name.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-1.5, 2.0, -0.0].fabs().to_vec(), [1.5, 2.0, 0.0]);
    /// ```
    pub fn fabs(&self) -> Array<F> {
        self.abs()
    }

    /// Equivalent of `np.positive(a)`: unary plus, a copy of the array.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-1.0, 2.0].positive().to_vec(), [-1.0, 2.0]);
    /// ```
    pub fn positive(&self) -> Array<F> {
        self.copy()
    }

    /// Equivalent of `np.negative(a)`: unary minus (`negative(0.0)` is `-0.0`).
    ///
    /// This is the core unary `-` operator under its NumPy name.
    ///
    /// ```
    /// use rustnum as np;
    /// let r: Vec<f64> = np::array![1.0, -2.0, 0.0].negative().to_vec();
    /// assert_eq!(r, [-1.0, 2.0, 0.0]);
    /// assert!(r[2].is_sign_negative());
    /// ```
    pub fn negative(&self) -> Array<F> {
        -self
    }

    /// Equivalent of `np.signbit(a)`: true where the sign bit is set (so `-0.0` is true).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-0.0, 0.0, -1.0].signbit().to_vec(), [true, false, true]);
    /// ```
    pub fn signbit(&self) -> Array<bool> {
        self.par_map(k::signbit::<F>)
    }

    /// Equivalent of `np.spacing(a)`: distance to the next float away from zero.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.0, 1e16].spacing().to_vec(), [f64::EPSILON, 2.0]);
    /// assert!(np::array![f64::INFINITY].spacing().item().is_nan());
    /// ```
    pub fn spacing(&self) -> Array<F> {
        self.par_map(k::spacing::<F>)
    }

    /// Equivalent of `np.modf(a)`: fractional and integral parts `(frac, int)`, both carrying the
    /// sign of the input.
    ///
    /// ```
    /// use rustnum as np;
    /// let (frac, int) = np::array![3.75, -3.75].modf();
    /// assert_eq!(frac.to_vec(), [0.75, -0.75]);
    /// assert_eq!(int.to_vec(), [3.0, -3.0]);
    /// ```
    pub fn modf(&self) -> (Array<F>, Array<F>) {
        let (frac, int): (Vec<F>, Vec<F>) = self.iter().map(k::modf::<F>).unzip();
        (
            Array::from_vec(frac, self.shape()),
            Array::from_vec(int, self.shape()),
        )
    }

    /// Equivalent of `np.frexp(a)`: mantissa and exponent `(m, e)` with `a = m * 2**e` and
    /// `0.5 <= |m| < 1`.
    ///
    /// ```
    /// use rustnum as np;
    /// let (m, e) = np::array![8.0, 0.1, -3.0].frexp();
    /// assert_eq!(m.to_vec(), [0.5, 0.8, -0.75]);
    /// assert_eq!(e.to_vec(), [4, -3, 2]);
    /// ```
    pub fn frexp(&self) -> (Array<F>, Array<i32>) {
        let (m, e): (Vec<F>, Vec<i32>) = self.iter().map(k::frexp::<F>).unzip();
        (
            Array::from_vec(m, self.shape()),
            Array::from_vec(e, self.shape()),
        )
    }

    /// Equivalent of `np.nan_to_num(a)`: NaN becomes `0`, `+inf`/`-inf` become the largest/smallest
    /// finite float of the element type.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0];
    /// assert_eq!(a.nan_to_num().to_vec(), [0.0, f64::MAX, f64::MIN, 1.0]);
    /// ```
    pub fn nan_to_num(&self) -> Array<F> {
        self.nan_to_num_with(F::ZERO, None, None)
    }

    /// Equivalent of `np.nan_to_num(a, nan=nan, posinf=posinf, neginf=neginf)`; `None` keeps the
    /// default (largest / smallest finite float).
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0];
    /// let r = a.nan_to_num_with(-1.0, Some(9.0), None);
    /// assert_eq!(r.to_vec(), [-1.0, 9.0, f64::MIN, 1.0]);
    /// ```
    pub fn nan_to_num_with(&self, nan: F, posinf: Option<F>, neginf: Option<F>) -> Array<F> {
        self.par_map(|x| k::nan_to_num(x, nan, posinf, neginf))
    }

    /// Equivalent of `np.unwrap(a)`: undo `2 pi` jumps along the last axis.
    ///
    /// ```
    /// use rustnum as np;
    /// let v: Vec<f64> = np::array![0.0, 3.0, 6.5].unwrap().to_vec();
    /// assert_eq!(&v[..2], &[0.0, 3.0]);
    /// assert!((v[2] - 0.21681469282041377).abs() < 1e-12);
    /// ```
    pub fn unwrap(&self) -> Array<F> {
        self.unwrap_with(None, -1, F::from_f64(TAU))
    }

    /// Equivalent of `np.unwrap(a, discont=discont, axis=axis, period=period)` for float input.
    ///
    /// `discont = None` means `period / 2`. Panics for 0-d arrays and out-of-range axes.
    ///
    /// ```
    /// use rustnum as np;
    /// // shape [3, 2]; unwrap down the columns (axis 0)
    /// let a = np::Array::from_vec(vec![0.0, 0.0, 3.0, 3.0, 6.5, 6.5], &[3, 2]);
    /// let u = a.unwrap_with(None, 0, std::f64::consts::TAU).to_vec();
    /// assert_eq!(&u[..4], &[0.0, 0.0, 3.0, 3.0]);
    /// assert!((u[4] - 0.21681469282041377).abs() < 1e-12);
    /// // a custom period: np.unwrap([0, 3, 6.5], discont=1, period=4)
    /// let b = np::array![0.0, 3.0, 6.5].unwrap_with(Some(1.0), -1, 4.0);
    /// assert_eq!(b.to_vec(), [0.0, -1.0, -1.5]);
    /// ```
    pub fn unwrap_with(&self, discont: Option<F>, axis: isize, period: F) -> Array<F> {
        let nd = self.ndim();
        assert!(
            nd > 0,
            "diff requires input that is at least one dimensional"
        );
        let ax = if axis < 0 { axis + nd as isize } else { axis };
        assert!(
            ax >= 0 && (ax as usize) < nd,
            "axis {axis} is out of bounds for array of dimension {nd}"
        );
        let moved = self.moveaxis(ax, -1);
        let n = moved.shape()[nd - 1];
        let data = moved.to_vec();
        let out = if n == 0 {
            data
        } else {
            data.chunks(n)
                .flat_map(|row| k::unwrap(row, discont, period))
                .collect()
        };
        Array::from_vec(out, moved.shape()).moveaxis(-1, ax)
    }
}

// ---------------------------------------------------------------------------------------
// Free functions of one array: ediff1d and angle
// ---------------------------------------------------------------------------------------

/// Equivalent of `np.ediff1d(a)`: differences between consecutive elements of the flattened array.
///
/// Integer differences wrap on overflow like NumPy.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::ediff1d(np::array![1, 2, 4, 7]).to_vec(), [1, 2, 3]);
/// assert_eq!(np::ediff1d(np::array![[1, 3], [6, 10]]).to_vec(), [2, 3, 4]);
/// ```
pub fn ediff1d<T: Num>(a: impl IntoArray<T>) -> Array<T> {
    ediff1d_with(a, &[], &[])
}

/// Equivalent of `np.ediff1d(a, to_end=to_end, to_begin=to_begin)`.
///
/// ```
/// use rustnum as np;
/// let r = np::ediff1d_with(np::array![1, 2, 4, 7], &[99], &[-1]);
/// assert_eq!(r.to_vec(), [-1, 1, 2, 3, 99]);
/// ```
pub fn ediff1d_with<T: Num>(a: impl IntoArray<T>, to_end: &[T], to_begin: &[T]) -> Array<T> {
    Array::from(k::ediff1d(&a.into_array().to_vec(), to_end, to_begin))
}

/// Equivalent of `np.angle(a)` for real input: `0` for `x >= 0` and `pi` for `x < 0`
/// (`angle(-0.0)` is `pi`, NaN stays NaN).
///
/// ```
/// use rustnum as np;
/// let r = np::angle(np::array![1.0, -1.0, 0.0, -0.0]).to_vec();
/// assert_eq!(r, [0.0, std::f64::consts::PI, 0.0, std::f64::consts::PI]);
/// ```
pub fn angle<F: Float>(a: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_map(|x| k::angle(x, F::ZERO))
}

/// Equivalent of `np.angle(a, deg=True)` for real input: the angle in degrees.
///
/// ```
/// use rustnum as np;
/// let d: Vec<f64> = np::angle_deg(np::array![1.0, -1.0]).to_vec();
/// assert_eq!(d[0], 0.0);
/// assert!((d[1] - 180.0).abs() < 1e-12);
/// ```
pub fn angle_deg<F: Float>(a: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_map(|x| k::angle_deg(x, F::ZERO))
}

// ---------------------------------------------------------------------------------------
// Binary free functions (broadcasting, scalars allowed anywhere)
// ---------------------------------------------------------------------------------------

/// Equivalent of `np.logaddexp(a, b)`: `log(exp(a) + exp(b))` without overflow.
///
/// ```
/// use rustnum as np;
/// let a = np::array![0.0, 1.0, f64::NEG_INFINITY];
/// let r = np::logaddexp(&a, &a).to_vec();
/// assert_eq!(r, [std::f64::consts::LN_2, 1.0 + std::f64::consts::LN_2, f64::NEG_INFINITY]);
/// // adding probabilities that would underflow as plain numbers
/// let s = np::logaddexp(1e-50f64.ln(), 2.5e-50f64.ln()).item();
/// assert!((s - 3.5e-50f64.ln()).abs() < 1e-12);
/// ```
pub fn logaddexp<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, k::logaddexp::<F>)
}

/// Equivalent of `np.logaddexp2(a, b)`: `log2(2**a + 2**b)` without overflow.
///
/// ```
/// use rustnum as np;
/// let a = np::array![1.0, 3.0];
/// assert_eq!(np::logaddexp2(&a, &a).to_vec(), [2.0, 4.0]);
/// ```
pub fn logaddexp2<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, k::logaddexp2::<F>)
}

/// Equivalent of `np.fmod(a, b)`: C `fmod`, the remainder has the sign of the dividend.
///
/// ```
/// use rustnum as np;
/// let r = np::fmod(np::array![-7.0, 7.0, 7.0], np::array![3.0, -3.0, 3.0]);
/// assert_eq!(r.to_vec(), [-1.0, 1.0, 1.0]);
/// ```
pub fn fmod<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, k::fmod::<F>)
}

/// Equivalent of `np.remainder(a, b)` / `np.mod(a, b)`: the remainder has the sign of the divisor
/// (Python `%`).
///
/// `mod` is a Rust keyword, so there is no `np::mod_` alias; use `remainder`.
///
/// ```
/// use rustnum as np;
/// let r = np::remainder(np::array![-7.0, 7.0, 7.0], np::array![3.0, -3.0, 3.0]);
/// assert_eq!(r.to_vec(), [2.0, -2.0, 1.0]);
/// ```
pub fn remainder<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, k::remainder::<F>)
}

/// Equivalent of `np.floor_divide(a, b)` (`a // b`): the floor of the quotient.
///
/// ```
/// use rustnum as np;
/// let r = np::floor_divide(np::array![-7.0, 7.0, 7.0], np::array![3.0, -3.0, 3.0]);
/// assert_eq!(r.to_vec(), [-3.0, -3.0, 2.0]);
/// ```
pub fn floor_divide<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, k::floor_divide::<F>)
}

/// Equivalent of `np.divmod(a, b)`: `(floor_divide(a, b), remainder(a, b))` in one pass.
///
/// ```
/// use rustnum as np;
/// let (q, r) = np::divmod(np::array![-7.0, 7.0], 3.0f64);
/// assert_eq!(q.to_vec(), [-3.0, 2.0]);
/// assert_eq!(r.to_vec(), [2.0, 1.0]);
/// ```
pub fn divmod<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> (Array<F>, Array<F>) {
    let (a, b) = (a.into_array(), b.into_array());
    let shape = a.common_shape(&b);
    let (a, b) = (a.broadcast_to(&shape), b.broadcast_to(&shape));
    let (q, r): (Vec<F>, Vec<F>) = a.iter().zip(b.iter()).map(|(x, y)| k::divmod(x, y)).unzip();
    (Array::from_vec(q, &shape), Array::from_vec(r, &shape))
}

/// Equivalent of `np.copysign(a, b)`: the magnitude of `a` with the sign of `b`.
///
/// ```
/// use rustnum as np;
/// let r = np::copysign(np::array![1.0, -2.0, 3.0], np::array![-1.0, 1.0, -0.0]);
/// assert_eq!(r.to_vec(), [-1.0, 2.0, -3.0]);
/// ```
pub fn copysign<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, k::copysign::<F>)
}

/// Equivalent of `np.heaviside(x, h0)`: `0` for `x < 0`, `h0` at `0`, `1` for `x > 0`, NaN for NaN.
///
/// ```
/// use rustnum as np;
/// let r = np::heaviside(np::array![-1.5, 0.0, 2.0], 0.5);
/// assert_eq!(r.to_vec(), [0.0, 0.5, 1.0]);
/// ```
pub fn heaviside<F: Float>(x: impl IntoArray<F>, h0: impl IntoArray<F>) -> Array<F> {
    x.into_array().par_zip(h0, k::heaviside::<F>)
}

/// Equivalent of `np.nextafter(a, b)`: the next representable float after `a` towards `b`.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::nextafter(1.0f64, 2.0).item(), 1.0 + f64::EPSILON);
/// assert_eq!(np::nextafter(1.0f64, 0.0).item(), 1.0 - f64::EPSILON / 2.0);
/// ```
pub fn nextafter<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, k::nextafter::<F>)
}

/// Equivalent of `np.ldexp(a, e)`: `a * 2**e` with an integer exponent array.
///
/// ```
/// use rustnum as np;
/// let r = np::ldexp(np::array![1.0, 0.5, 3.0], np::array![1, 2, -1]);
/// assert_eq!(r.to_vec(), [2.0, 2.0, 1.5]);
/// ```
pub fn ldexp<F: Float>(a: impl IntoArray<F>, e: impl IntoArray<i32>) -> Array<F> {
    a.into_array().par_zip(e, k::ldexp::<F>)
}

/// Equivalent of `np.float_power(a, b)`: a power always computed in `float64`.
///
/// Unlike the core `power`, which keeps the input type, the result is an `Array<f64>`.
///
/// ```
/// use rustnum as np;
/// let r = np::float_power(np::array![2.0f32, 3.0], np::array![2.0f32, 0.5]).to_vec();
/// assert_eq!(r[0], 4.0);
/// assert!((r[1] - 1.7320508075688772).abs() < 1e-15);
/// ```
pub fn float_power<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<f64> {
    a.into_array().par_zip(b, k::float_power::<F>)
}

/// Equivalent of `np.gcd(a, b)` for `int64` arrays: the greatest common divisor, never negative.
///
/// ```
/// use rustnum as np;
/// let r = np::gcd(np::array![12i64, 18, -4, 0], np::array![8i64, 27, 6, 0]);
/// assert_eq!(r.to_vec(), [4, 9, 2, 0]);
/// ```
pub fn gcd(a: impl IntoArray<i64>, b: impl IntoArray<i64>) -> Array<i64> {
    a.into_array().par_zip(b, k::gcd)
}

/// Equivalent of `np.lcm(a, b)` for `int64` arrays: the least common multiple, never negative
/// (wraps on overflow like NumPy).
///
/// ```
/// use rustnum as np;
/// let r = np::lcm(np::array![4i64, 6, -3, 0], np::array![6i64, 10, 5, 7]);
/// assert_eq!(r.to_vec(), [12, 30, 15, 0]);
/// ```
pub fn lcm(a: impl IntoArray<i64>, b: impl IntoArray<i64>) -> Array<i64> {
    a.into_array().par_zip(b, k::lcm)
}

// ---------------------------------------------------------------------------------------
// Windows (float64, like NumPy)
// ---------------------------------------------------------------------------------------

/// Equivalent of `np.hanning(M)`: the Hann window of `m` points (`m = 0` is empty, `m = 1` is `[1]`).
///
/// ```
/// use rustnum as np;
/// let w = np::hanning(5).to_vec();
/// for (g, e) in w.iter().zip([0.0, 0.5, 1.0, 0.5, 0.0]) {
///     assert!((g - e).abs() < 1e-15);
/// }
/// assert!(np::hanning(0).is_empty());
/// assert_eq!(np::hanning(1).to_vec(), [1.0]);
/// ```
pub fn hanning(m: usize) -> Array<f64> {
    Array::from(w::hanning(m))
}

/// Equivalent of `np.hamming(M)`: the Hamming window of `m` points.
///
/// ```
/// use rustnum as np;
/// let w = np::hamming(5).to_vec();
/// for (g, e) in w.iter().zip([0.08, 0.54, 1.0, 0.54, 0.08]) {
///     assert!((g - e).abs() < 1e-15);
/// }
/// ```
pub fn hamming(m: usize) -> Array<f64> {
    Array::from(w::hamming(m))
}

/// Equivalent of `np.blackman(M)`: the Blackman window of `m` points.
///
/// ```
/// use rustnum as np;
/// let w = np::blackman(5).to_vec();
/// for (g, e) in w.iter().zip([-1.3877787807814457e-17, 0.34, 1.0, 0.34, -1.3877787807814457e-17]) {
///     assert!((g - e).abs() < 1e-15);
/// }
/// ```
pub fn blackman(m: usize) -> Array<f64> {
    Array::from(w::blackman(m))
}

/// Equivalent of `np.bartlett(M)`: the Bartlett (triangular) window of `m` points.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::bartlett(5).to_vec(), [0.0, 0.5, 1.0, 0.5, 0.0]);
/// ```
pub fn bartlett(m: usize) -> Array<f64> {
    Array::from(w::bartlett(m))
}

/// Equivalent of `np.kaiser(M, beta)`: the Kaiser window of `m` points with shape parameter `beta`.
///
/// ```
/// use rustnum as np;
/// let w = np::kaiser(5, 14.0).to_vec();
/// assert!((w[0] - 7.726866835270368e-06).abs() < 1e-15);
/// assert!((w[1] - 0.16493218754795197).abs() < 1e-14);
/// assert!((w[2] - 1.0).abs() < 1e-15);
/// ```
pub fn kaiser(m: usize, beta: f64) -> Array<f64> {
    Array::from(w::kaiser(m, beta))
}

impl<F: Float> Array<F> {
    /// Applies np.sinc with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.sinc_inplace();
    /// ```
    pub fn sinc_inplace(&mut self) {
        self.mapv_inplace_parallel(k::sinc::<F>);
    }
    /// Applies np.sinc with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_sinc();
    /// ```
    pub fn into_sinc(mut self) -> Self {
        self.sinc_inplace();
        self
    }
    /// Applies np.i0 with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.i0_inplace();
    /// ```
    pub fn i0_inplace(&mut self) {
        self.mapv_inplace_parallel(k::i0::<F>);
    }
    /// Applies np.i0 with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_i0();
    /// ```
    pub fn into_i0(mut self) -> Self {
        self.i0_inplace();
        self
    }
    /// Applies np.deg2rad with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.deg2rad_inplace();
    /// ```
    pub fn deg2rad_inplace(&mut self) {
        self.mapv_inplace_parallel(k::deg2rad::<F>);
    }
    /// Applies np.deg2rad with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_deg2rad();
    /// ```
    pub fn into_deg2rad(mut self) -> Self {
        self.deg2rad_inplace();
        self
    }
    /// Applies np.rad2deg with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.rad2deg_inplace();
    /// ```
    pub fn rad2deg_inplace(&mut self) {
        self.mapv_inplace_parallel(k::rad2deg::<F>);
    }
    /// Applies np.rad2deg with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_rad2deg();
    /// ```
    pub fn into_rad2deg(mut self) -> Self {
        self.rad2deg_inplace();
        self
    }
    /// Applies np.rint with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.rint_inplace();
    /// ```
    pub fn rint_inplace(&mut self) {
        self.mapv_inplace_parallel(F::round);
    }
    /// Applies np.rint with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_rint();
    /// ```
    pub fn into_rint(mut self) -> Self {
        self.rint_inplace();
        self
    }
    /// Applies np.fix with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.fix_inplace();
    /// ```
    pub fn fix_inplace(&mut self) {
        self.mapv_inplace_parallel(F::trunc);
    }
    /// Applies np.fix with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_fix();
    /// ```
    pub fn into_fix(mut self) -> Self {
        self.fix_inplace();
        self
    }
    /// Applies np.fabs with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.fabs_inplace();
    /// ```
    pub fn fabs_inplace(&mut self) {
        self.mapv_inplace_parallel(F::abs);
    }
    /// Applies np.fabs with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_fabs();
    /// ```
    pub fn into_fabs(mut self) -> Self {
        self.fabs_inplace();
        self
    }
    /// Applies np.positive with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.positive_inplace();
    /// ```
    pub fn positive_inplace(&mut self) {
        self.mapv_inplace_parallel(|x| x);
    }
    /// Applies np.positive with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_positive();
    /// ```
    pub fn into_positive(mut self) -> Self {
        self.positive_inplace();
        self
    }
    /// Applies np.negative with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.negative_inplace();
    /// ```
    pub fn negative_inplace(&mut self) {
        self.mapv_inplace_parallel(|x| -x);
    }
    /// Applies np.negative with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_negative();
    /// ```
    pub fn into_negative(mut self) -> Self {
        self.negative_inplace();
        self
    }
    /// Applies np.spacing with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.spacing_inplace();
    /// ```
    pub fn spacing_inplace(&mut self) {
        self.mapv_inplace_parallel(k::spacing::<F>);
    }
    /// Applies np.spacing with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_spacing();
    /// ```
    pub fn into_spacing(mut self) -> Self {
        self.spacing_inplace();
        self
    }
    /// Applies np.nan_to_num with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.nan_to_num_inplace();
    /// ```
    pub fn nan_to_num_inplace(&mut self) {
        self.mapv_inplace_parallel(|x| k::nan_to_num(x, F::ZERO, None, None));
    }
    /// Applies np.nan_to_num with out=a, preserving copy-on-write semantics.
    /// ```
    /// let mut a=rustnum::array![0.5f64]; a.into_nan_to_num();
    /// ```
    pub fn into_nan_to_num(mut self) -> Self {
        self.nan_to_num_inplace();
        self
    }
    /// Replaces non-finite values in place, like np.nan_to_num(a, copy=False).
    /// ```
    /// let mut a=rustnum::array![f64::NAN]; a.nan_to_num_with_inplace(2.,None,None);
    /// assert_eq!(a.item(),2.);
    /// ```
    pub fn nan_to_num_with_inplace(&mut self, nan: F, posinf: Option<F>, neginf: Option<F>) {
        self.mapv_inplace_parallel(|x| k::nan_to_num(x, nan, posinf, neginf));
    }
    /// Consumes and replaces non-finite values, like np.nan_to_num(a, copy=False).
    /// ```
    /// assert_eq!(rustnum::array![f64::NAN].into_nan_to_num_with(2.,None,None).item(),2.);
    /// ```
    pub fn into_nan_to_num_with(mut self, nan: F, posinf: Option<F>, neginf: Option<F>) -> Self {
        self.nan_to_num_with_inplace(nan, posinf, neginf);
        self
    }
}
