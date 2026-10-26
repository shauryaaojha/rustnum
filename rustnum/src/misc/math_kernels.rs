//! Scalar kernels for NumPy's remaining element-wise math.
//!
//! Every function here works on one value (or one 1-d slice) and follows the NumPy 2.3
//! algorithm, including NaN, infinity and signed-zero behaviour. The public `Array`
//! wrappers in `api.rs` only map these kernels over arrays.

use core::fmt::{Display, LowerExp};

use crate::{Float, Num};

/// Extra per-type constants and bit-level hooks the misc kernels need on top of [`Float`].
///
/// This is the sealed private supertrait of [`Float`]: it lives in a private module, so it cannot be
/// named (or implemented) outside the crate, and `f32` / `f64` are its only implementors. It is a
/// supertrait, so any generic code bounded by `F: Float` can use the kernels. Every item has a name
/// that does not exist on `Float`, `Num` or `Element`, so method and constant lookups on a generic
/// `F: Float` stay unambiguous.
#[doc(hidden)]
pub trait FloatHooks: Copy + Display + LowerExp {
    /// `2`.
    const TWO: Self;
    /// `0.5`.
    const HALF: Self;
    /// `ln 2`.
    const LN_2: Self;
    /// `log2 e`.
    const LOG2_E: Self;
    /// Number of explicit mantissa bits (52 or 23).
    const MANT_BITS: u32;
    /// Number of exponent bits (11 or 8).
    const EXP_BITS: u32;
    /// `|x|` above which `csqrt` scales its argument (C99 msun `THRESH`).
    const SQRT_THRESH: Self;
    /// `|x|` below which complex sinh/cosh/tanh use the direct formulas (msun: 22 or 9).
    const HYPERBOLIC_DIRECT: Self;
    /// NumPy prints values with `|x| >= REPR_SCI_UPPER` (or `< 1e-4`) in scientific notation.
    const REPR_SCI_UPPER: f64;
    /// NumPy dtype name of `Complex<Self>` (`"complex128"`, `"complex64"`).
    const COMPLEX_DTYPE: &'static str;
    /// `.npy` descriptor of `Complex<Self>` (`"<c16"`, `"<c8"`), `None` when NumPy has no such dtype.
    const COMPLEX_DESCR: Option<&'static str>;
    /// Bytes of one real value (8 or 4).
    const BYTES: usize;

    /// The IEEE bit pattern, zero-extended to 64 bits.
    fn to_bits64(self) -> u64;
    /// Inverse of [`FloatHooks::to_bits64`] (the upper bits are ignored for `f32`).
    fn from_bits64(b: u64) -> Self;
    /// True when the sign bit is set (`-0.0` and negative NaN included), like `np.signbit`.
    fn sign_bit(self) -> bool;
    /// Magnitude of `self` with the sign bit of `sign`, like `np.copysign`.
    fn copy_sign(self, sign: Self) -> Self;
}

macro_rules! impl_float_hooks {
    ($t:ident, $bits:ident, $mant:expr, $expb:expr, $thresh:expr, $direct:expr, $sci:expr, $cd:literal, $cdescr:expr, $bytes:expr) => {
        impl FloatHooks for $t {
            const TWO: Self = 2.0;
            const HALF: Self = 0.5;
            const LN_2: Self = core::$t::consts::LN_2;
            const LOG2_E: Self = core::$t::consts::LOG2_E;
            const MANT_BITS: u32 = $mant;
            const EXP_BITS: u32 = $expb;
            const SQRT_THRESH: Self = $thresh;
            const HYPERBOLIC_DIRECT: Self = $direct;
            const REPR_SCI_UPPER: f64 = $sci;
            const COMPLEX_DTYPE: &'static str = $cd;
            const COMPLEX_DESCR: Option<&'static str> = $cdescr;
            const BYTES: usize = $bytes;

            fn to_bits64(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn from_bits64(b: u64) -> Self {
                <$t>::from_bits(b as $bits)
            }
            fn sign_bit(self) -> bool {
                <$t>::is_sign_negative(self)
            }
            fn copy_sign(self, sign: Self) -> Self {
                <$t>::copysign(self, sign)
            }
        }
    };
}

// 1.6568542494923802 = 4 * (sqrt(2) - 1); THRESH = that * 2^(emax - 1).
impl_float_hooks!(
    f64,
    u64,
    52,
    11,
    1.656_854_249_492_380_2 * f64::from_bits(0x7FD0_0000_0000_0000),
    22.0,
    1e16,
    "complex128",
    Some("<c16"),
    8
);
impl_float_hooks!(
    f32,
    u32,
    23,
    8,
    1.656_854_2 * f32::from_bits(0x7E80_0000),
    9.0,
    1e6,
    "complex64",
    Some("<c8"),
    4
);

#[inline]
fn bias<F: Float>() -> i32 {
    (1i32 << (F::EXP_BITS - 1)) - 1
}

/// 2^k for k within the normal range of `F`.
#[inline]
fn pow2<F: Float>(k: i32) -> F {
    F::from_bits64(((bias::<F>() + k) as u64) << F::MANT_BITS)
}

// ---------------------------------------------------------------------------------------
// Trigonometric / exponential helpers
// ---------------------------------------------------------------------------------------

/// `np.sinc(x)`: normalized `sin(pi x) / (pi x)`. NumPy 2.3 replaces a zero argument by `eps`.
pub fn sinc<F: Float>(x: F) -> F {
    let px = F::PI * x;
    let y = if px == F::ZERO { F::EPSILON } else { px };
    y.sin() / y
}

const I0_A: [f64; 30] = [
    -4.4153416464793395e-18,
    3.3307945188222384e-17,
    -2.431279846547955e-16,
    1.715391285555133e-15,
    -1.1685332877993451e-14,
    7.676185498604936e-14,
    -4.856446783111929e-13,
    2.95505266312964e-12,
    -1.726826291441556e-11,
    9.675809035373237e-11,
    -5.189795601635263e-10,
    2.6598237246823866e-09,
    -1.300025009986248e-08,
    6.046995022541919e-08,
    -2.670793853940612e-07,
    1.1173875391201037e-06,
    -4.4167383584587505e-06,
    1.6448448070728896e-05,
    -5.754195010082104e-05,
    0.00018850288509584165,
    -0.0005763755745385824,
    0.0016394756169413357,
    -0.004324309995050576,
    0.010546460394594998,
    -0.02373741480589947,
    0.04930528423967071,
    -0.09490109704804764,
    0.17162090152220877,
    -0.3046826723431984,
    0.6767952744094761,
];

const I0_B: [f64; 25] = [
    -7.233180487874754e-18,
    -4.830504485944182e-18,
    4.46562142029676e-17,
    3.461222867697461e-17,
    -2.8276239805165836e-16,
    -3.425485619677219e-16,
    1.7725601330565263e-15,
    3.8116806693526224e-15,
    -9.554846698828307e-15,
    -4.150569347287222e-14,
    1.54008621752141e-14,
    3.8527783827421426e-13,
    7.180124451383666e-13,
    -1.7941785315068062e-12,
    -1.3215811840447713e-11,
    -3.1499165279632416e-11,
    1.1889147107846439e-11,
    4.94060238822497e-10,
    3.3962320257083865e-09,
    2.266668990498178e-08,
    2.0489185894690638e-07,
    2.8913705208347567e-06,
    6.889758346916825e-05,
    0.0033691164782556943,
    0.8044904110141088,
];

fn chbevl<F: Float>(x: F, coeffs: &[f64]) -> F {
    let mut b0 = F::from_f64(coeffs[0]);
    let mut b1 = F::ZERO;
    let mut b2 = F::ZERO;
    for &c in &coeffs[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + F::from_f64(c);
    }
    F::HALF * (b0 - b2)
}

/// `np.i0(x)`: modified Bessel function of the first kind, order 0 (Cephes Chebyshev
/// coefficients, exactly the ones NumPy uses). `i0(inf)` is NaN like NumPy.
pub fn i0<F: Float>(x: F) -> F {
    let x = x.abs();
    if x <= F::from_f64(8.0) {
        x.exp() * chbevl(x / F::TWO - F::TWO, &I0_A)
    } else {
        x.exp() * chbevl(F::from_f64(32.0) / x - F::TWO, &I0_B) / x.sqrt()
    }
}

/// `np.deg2rad(x)`: `x * (pi / 180)`.
pub fn deg2rad<F: Float>(x: F) -> F {
    x * (F::PI / F::from_f64(180.0))
}

/// `np.rad2deg(x)`: `x * (180 / pi)`.
pub fn rad2deg<F: Float>(x: F) -> F {
    x * (F::from_f64(180.0) / F::PI)
}

/// `np.logaddexp(x, y)`: `log(exp(x) + exp(y))` without overflow.
pub fn logaddexp<F: Float>(x: F, y: F) -> F {
    if x == y {
        // also covers x = y = +-inf
        return x + F::LN_2;
    }
    let tmp = x - y;
    if tmp > F::ZERO {
        x + (-tmp).exp().ln_1p()
    } else if tmp <= F::ZERO {
        y + tmp.exp().ln_1p()
    } else {
        tmp // NaN
    }
}

/// `np.logaddexp2(x, y)`: `log2(2**x + 2**y)` without overflow.
pub fn logaddexp2<F: Float>(x: F, y: F) -> F {
    if x == y {
        return x + F::ONE;
    }
    let tmp = x - y;
    if tmp > F::ZERO {
        x + F::LOG2_E * (-tmp).exp2().ln_1p()
    } else if tmp <= F::ZERO {
        y + F::LOG2_E * tmp.exp2().ln_1p()
    } else {
        tmp
    }
}

// ---------------------------------------------------------------------------------------
// Division family
// ---------------------------------------------------------------------------------------

/// `np.fmod(a, b)`: C `fmod`, the result has the sign of the dividend `a`.
pub fn fmod<F: Float>(a: F, b: F) -> F {
    a % b
}

/// `np.remainder(a, b)` / Python `%`: the result has the sign of the divisor `b`.
pub fn remainder<F: Float>(a: F, b: F) -> F {
    let mut m = a % b;
    if b == F::ZERO {
        return m; // NaN
    }
    if m != F::ZERO {
        // NaN also lands here and stays NaN.
        if (b < F::ZERO) != (m < F::ZERO) {
            m = m + b;
        }
    } else {
        m = F::ZERO.copy_sign(b);
    }
    m
}

/// `np.divmod(a, b)`: `(floor_divide(a, b), remainder(a, b))` computed together like NumPy.
pub fn divmod<F: Float>(a: F, b: F) -> (F, F) {
    if b == F::ZERO {
        return (a / b, a % b);
    }
    let mut m = a % b;
    let mut div = (a - m) / b;
    if m != F::ZERO {
        if (b < F::ZERO) != (m < F::ZERO) {
            m = m + b;
            div = div - F::ONE;
        }
    } else {
        m = F::ZERO.copy_sign(b);
    }
    let floordiv = if div != F::ZERO {
        let f = div.floor();
        if div - f > F::HALF { f + F::ONE } else { f }
    } else {
        F::ZERO.copy_sign(a / b)
    };
    (floordiv, m)
}

/// `np.floor_divide(a, b)`: `floor(a / b)` with NumPy's rounding fix-ups.
pub fn floor_divide<F: Float>(a: F, b: F) -> F {
    divmod(a, b).0
}

/// `np.modf(x)`: `(fractional, integral)` parts, both with the sign of `x`.
pub fn modf<F: Float>(x: F) -> (F, F) {
    if x.is_nan() {
        return (x, x);
    }
    if x.is_infinite() {
        return (F::ZERO.copy_sign(x), x);
    }
    let ip = x.trunc();
    ((x - ip).copy_sign(x), ip)
}

/// `np.frexp(x)`: `(m, e)` with `x = m * 2**e` and `0.5 <= |m| < 1`. Zero, inf and NaN give `(x, 0)`.
pub fn frexp<F: Float>(x: F) -> (F, i32) {
    if x == F::ZERO || !x.is_finite() {
        return (x, 0);
    }
    let mant = F::MANT_BITS;
    let exp_mask = (1u64 << F::EXP_BITS) - 1;
    let b = bias::<F>();
    let mut adj = 0;
    let mut v = x;
    if (x.to_bits64() >> mant) & exp_mask == 0 {
        // subnormal: scale into the normal range first
        v = x * pow2::<F>(mant as i32 + 1);
        adj = -(mant as i32 + 1);
    }
    let bits = v.to_bits64();
    let e = ((bits >> mant) & exp_mask) as i32;
    let kept = bits & !(exp_mask << mant);
    let m = F::from_bits64(kept | (((b - 1) as u64) << mant));
    (m, e - (b - 1) + adj)
}

/// `np.ldexp(x, n)`: `x * 2**n`, correctly rounded even into the subnormal range.
pub fn ldexp<F: Float>(x: F, n: i32) -> F {
    let b = bias::<F>();
    let emin = 1 - b;
    let m = F::MANT_BITS as i32;
    let mut y = x;
    let mut n = n;
    if n > b {
        y = y * pow2::<F>(b);
        n -= b;
        if n > b {
            y = y * pow2::<F>(b);
            n -= b;
            if n > b {
                n = b;
            }
        }
    } else if n < emin {
        let f = pow2::<F>(emin) * pow2::<F>(m + 1);
        let step = -emin - (m + 1);
        y = y * f;
        n += step;
        if n < emin {
            y = y * f;
            n += step;
            if n < emin {
                n = emin;
            }
        }
    }
    y * pow2::<F>(n)
}

/// `np.copy_sign(x, y)`: magnitude of `x` with the sign bit of `y`.
pub fn copysign<F: Float>(x: F, y: F) -> F {
    x.copy_sign(y)
}

/// `np.nextafter(x, y)`: next representable value after `x` towards `y`.
pub fn nextafter<F: Float>(x: F, y: F) -> F {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x == y {
        return y;
    }
    if x == F::ZERO {
        let tiny = F::from_bits64(1);
        return if y > F::ZERO { tiny } else { -tiny };
    }
    let bits = x.to_bits64();
    if (x < y) == (x > F::ZERO) {
        F::from_bits64(bits + 1)
    } else {
        F::from_bits64(bits - 1)
    }
}

/// `np.spacing(x)`: distance from `x` to the next float away from zero. `spacing(+-inf)` is NaN.
/// `spacing(-0.0)` is `+tiny` like NumPy (only `x < 0` looks towards `-inf`).
pub fn spacing<F: Float>(x: F) -> F {
    if x.is_infinite() {
        return F::NAN;
    }
    let toward = if x < F::ZERO {
        F::NEG_INFINITY
    } else {
        F::INFINITY
    };
    nextafter(x, toward) - x
}

/// `np.heaviside(x, h0)`: 0 for `x < 0`, `h0` for `x == 0`, 1 for `x > 0`, NaN for NaN `x`.
pub fn heaviside<F: Float>(x: F, h0: F) -> F {
    if x.is_nan() {
        F::NAN
    } else if x == F::ZERO {
        h0
    } else if x < F::ZERO {
        F::ZERO
    } else {
        F::ONE
    }
}

/// `np.signbit(x)`: true when the sign bit is set (so `-0.0` and `-nan` are true).
pub fn signbit<F: Float>(x: F) -> bool {
    x.sign_bit()
}

/// `np.float_power(x, y)`: always computed in `float64`, like NumPy.
pub fn float_power<F: Float>(x: F, y: F) -> f64 {
    x.to_f64().powf(y.to_f64())
}

// ---------------------------------------------------------------------------------------
// Integers
// ---------------------------------------------------------------------------------------

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `np.gcd(a, b)` for `int64`: non-negative result, `gcd(0, 0) == 0`.
/// `gcd(i64::MIN, 0)` wraps to `i64::MIN` like NumPy.
pub fn gcd(a: i64, b: i64) -> i64 {
    gcd_u64(a.unsigned_abs(), b.unsigned_abs()) as i64
}

/// `np.lcm(a, b)` for `int64`: non-negative result, `lcm(0, x) == 0`, wraps on overflow like NumPy.
pub fn lcm(a: i64, b: i64) -> i64 {
    let (ua, ub) = (a.unsigned_abs(), b.unsigned_abs());
    let g = gcd_u64(ua, ub);
    ua.checked_div(g).map_or(0, |q| q.wrapping_mul(ub) as i64)
}

// ---------------------------------------------------------------------------------------
// Slice helpers
// ---------------------------------------------------------------------------------------

/// `np.unwrap(p, discont, period=period)` on one 1-d slice (float input).
///
/// `discont = None` means `period / 2`. NaN differences propagate through the running sum.
pub fn unwrap<F: Float>(p: &[F], discont: Option<F>, period: F) -> Vec<F> {
    let mut out = Vec::with_capacity(p.len());
    if p.is_empty() {
        return out;
    }
    let high = period / F::TWO;
    let low = -high;
    let discont = discont.unwrap_or(high);
    out.push(p[0]);
    let mut cum = F::ZERO;
    for i in 1..p.len() {
        let dd = p[i] - p[i - 1];
        let mut ddmod = remainder(dd - low, period) + low;
        if ddmod == low && dd > F::ZERO {
            ddmod = high;
        }
        let mut correct = ddmod - dd;
        if dd.abs() < discont {
            correct = F::ZERO;
        }
        cum = cum + correct;
        out.push(p[i] + cum);
    }
    out
}

/// `np.nan_to_num(x, nan, posinf, neginf)` for one value.
///
/// `posinf`/`neginf` default to the largest/smallest finite float, like NumPy.
pub fn nan_to_num<F: Float>(x: F, nan: F, posinf: Option<F>, neginf: Option<F>) -> F {
    if x.is_nan() {
        nan
    } else if x == F::INFINITY {
        posinf.unwrap_or(F::MAX_VALUE)
    } else if x == F::NEG_INFINITY {
        neginf.unwrap_or(-F::MAX_VALUE)
    } else {
        x
    }
}

/// `np.ediff1d(a, to_end, to_begin)`: differences of consecutive elements of the flattened
/// input, with optional values prepended and appended.
pub fn ediff1d<T: Num>(a: &[T], to_end: &[T], to_begin: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len().saturating_sub(1) + to_end.len() + to_begin.len());
    out.extend_from_slice(to_begin);
    out.extend(a.windows(2).map(|w| crate::element::sub(w[1], w[0])));
    out.extend_from_slice(to_end);
    out
}

/// `np.angle(z)` for `z = re + im*j`: `atan2(im, re)` (real input uses `im = 0`).
pub fn angle<F: Float>(re: F, im: F) -> F {
    im.atan2(re)
}

/// `np.angle(z, deg=True)`: the angle in degrees.
pub fn angle_deg<F: Float>(re: F, im: F) -> F {
    angle(re, im) * F::from_f64(180.0 / core::f64::consts::PI)
}

// `real_if_close` needs complex arrays, which wait for `Complex<F>: Element` (docs/api/misc.md, M5).
// These two kernels are tested against NumPy and kept for that; nothing else in the module is unused.
/// Tolerance used by `np.real_if_close(a, tol)`: values above 1 are multiples of machine epsilon.
#[allow(dead_code)]
pub fn real_if_close_tol<F: Float>(tol: F) -> F {
    if tol > F::ONE { tol * F::EPSILON } else { tol }
}

/// Decision of `np.real_if_close(a, tol)`: true when every `|imag|` is below the tolerance
/// (an empty slice counts as true, a NaN never does).
#[allow(dead_code)]
pub fn real_if_close_ok<F: Float>(imag: &[F], tol: F) -> bool {
    let tol = real_if_close_tol(tol);
    imag.iter().all(|v| v.abs() < tol)
}
