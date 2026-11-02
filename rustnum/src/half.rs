//! `f16` (IEEE 754 binary16, NumPy `float16`) and `bf16` (bfloat16) array elements.
//!
//! Both are plain 16-bit storage types. Arithmetic and elementary functions compute in `f32`
//! and round the result back once (round to nearest, ties to even), which is what NumPy does for
//! `float16`. Conversions handle NaN, infinities, signed zeros and subnormals exactly.
//!
//! Bulk `f32 <-> f16` conversion (used by `astype`) runs eight lanes at a time with the x86
//! F16C instructions when the CPU has them (checked at run time) and falls back to the scalar
//! algorithms otherwise.

use crate::{Array, Element, Float, Num};
use std::any::{Any, TypeId};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::str::FromStr;

// ---------------------------------------------------------------------------------------
// Bit-level conversions
// ---------------------------------------------------------------------------------------

/// Rounds an `f32` to binary16 bits (round to nearest, ties to even).
pub(crate) const fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let abs = b & 0x7fff_ffff;
    if abs > 0x7f80_0000 {
        // NaN: keep the top payload bits, never collapse to infinity.
        let ret = 0x7c00 | ((abs >> 13) & 0x3ff) as u16;
        return sign | if ret == 0x7c00 { 0x7c01 } else { ret };
    }
    if abs >= 0x4780_0000 {
        return sign | 0x7c00; // infinity and everything >= 65536
    }
    if abs < 0x3880_0000 {
        // below 2^-14: subnormal or zero in binary16
        if abs <= 0x3300_0000 {
            return sign; // <= 2^-25 rounds to zero (the tie goes to even)
        }
        let e = abs >> 23; // 102..=112
        let m = (abs & 0x7f_ffff) | 0x80_0000;
        let shift = 126 - e;
        let q = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let q = if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        };
        return sign | q as u16;
    }
    let mut h = (abs - 0x3800_0000) >> 13;
    let rest = abs & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1; // a carry out of the mantissa correctly bumps the exponent (or reaches infinity)
    }
    sign | h as u16
}

/// Rounds an `f64` to binary16 bits directly (no double rounding through `f32`).
pub(crate) const fn f64_to_f16_bits(x: f64) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 48) & 0x8000) as u16;
    let abs = b & 0x7fff_ffff_ffff_ffff;
    if abs > 0x7ff0_0000_0000_0000 {
        let ret = 0x7c00 | ((abs >> 42) & 0x3ff) as u16;
        return sign | if ret == 0x7c00 { 0x7c01 } else { ret };
    }
    if abs >= 0x40f0_0000_0000_0000 {
        return sign | 0x7c00;
    }
    if abs < 0x3f10_0000_0000_0000 {
        if abs <= 0x3e60_0000_0000_0000 {
            return sign;
        }
        let e = abs >> 52; // 999..=1008
        let m = (abs & 0xf_ffff_ffff_ffff) | (1 << 52);
        let shift = 1051 - e;
        let q = m >> shift;
        let rem = m & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        let q = if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        };
        return sign | q as u16;
    }
    let mut h = (abs - (1008u64 << 52)) >> 42;
    let rest = abs & 0x3ff_ffff_ffff;
    let half = 1u64 << 41;
    if rest > half || (rest == half && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// Exact binary16 bits to `f32`.
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let exp = u32::from(h >> 10) & 0x1f;
    let man = u32::from(h) & 0x3ff;
    match exp {
        0 => {
            // zero or subnormal: man * 2^-24 is exact in f32
            let v = man as f32 * f32::from_bits(0x3380_0000);
            f32::from_bits(v.to_bits() | sign)
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (man << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (man << 13)),
    }
}

/// Exact bfloat16 bits to `f32`.
pub(crate) fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Rounds an `f32` to bfloat16 bits (round to nearest, ties to even; NaN stays NaN).
pub(crate) const fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x0040) as u16;
    }
    let rounded = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    (rounded >> 16) as u16
}

/// Rounds an `f64` to bfloat16 bits with a single rounding (via a round-to-odd `f32`).
pub(crate) const fn f64_to_bf16_bits(x: f64) -> u16 {
    if x.is_nan() {
        return (((x.to_bits() >> 48) as u16) & 0x8000) | 0x7fc0;
    }
    let t = x as f32;
    if t.is_infinite() || t as f64 == x {
        return f32_to_bf16_bits(t);
    }
    // Round to odd: truncate toward zero and force the sticky bit.
    let mut bits = t.to_bits();
    if (t as f64).abs() > x.abs() {
        bits -= 1;
    }
    f32_to_bf16_bits(f32::from_bits(bits | 1))
}

// ---------------------------------------------------------------------------------------
// Bulk conversion (F16C fast path)
// ---------------------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f16c {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Converts `src` to `dst` eight lanes at a time; returns how many elements were converted
    /// (a multiple of 8), the tail is left to the caller.
    ///
    /// # Safety
    /// The CPU must support AVX and F16C.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn f16_to_f32(src: &[u16], dst: &mut [f32]) -> usize {
        let n = src.len().min(dst.len()) / 8 * 8;
        let mut i = 0;
        while i < n {
            // SAFETY: i + 8 <= n <= both lengths; unaligned loads/stores need no alignment.
            unsafe {
                let v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(v));
            }
            i += 8;
        }
        n
    }

    /// Converts `src` to `dst` with round-to-nearest-even; see [`f16_to_f32`].
    ///
    /// # Safety
    /// The CPU must support AVX and F16C.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn f32_to_f16(src: &[f32], dst: &mut [u16]) -> usize {
        let n = src.len().min(dst.len()) / 8 * 8;
        let mut i = 0;
        while i < n {
            // SAFETY: i + 8 <= n <= both lengths; unaligned loads/stores need no alignment.
            unsafe {
                let v = _mm256_loadu_ps(src.as_ptr().add(i));
                let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), h);
            }
            i += 8;
        }
        n
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn have_f16c() -> bool {
    std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx")
}

/// `f16` bits to `f32`, F16C when available.
pub(crate) fn bulk_f16_to_f32(src: &[u16], dst: &mut [f32]) {
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if have_f16c() {
        // SAFETY: the F16C and AVX features were detected at run time just above.
        done = unsafe { f16c::f16_to_f32(src, dst) };
    }
    for (d, &s) in dst[done..].iter_mut().zip(&src[done..]) {
        *d = f16_bits_to_f32(s);
    }
}

/// `f32` to `f16` bits, F16C when available.
pub(crate) fn bulk_f32_to_f16(src: &[f32], dst: &mut [u16]) {
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if have_f16c() {
        // SAFETY: the F16C and AVX features were detected at run time just above.
        done = unsafe { f16c::f32_to_f16(src, dst) };
    }
    for (d, &s) in dst[done..].iter_mut().zip(&src[done..]) {
        *d = f32_to_f16_bits(s);
    }
}

// ---------------------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------------------

/// Shortest decimal digits (and decimal exponent of the first digit) that round-trip through
/// the 16-bit type, like NumPy's Dragon4 `unique` mode.
fn shortest_digits(x: f64, round_trip: impl Fn(f64) -> f64) -> (String, i32) {
    for n in 1..=9usize {
        let s = format!("{:.*e}", n - 1, x);
        let back: f64 = s.parse().unwrap();
        if round_trip(back).to_bits() == x.to_bits() {
            return split_sci(&s);
        }
    }
    split_sci(&format!("{x:e}"))
}

fn split_sci(s: &str) -> (String, i32) {
    let (mant, exp) = s.split_once('e').unwrap();
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let trimmed = digits.trim_end_matches('0');
    (
        if trimmed.is_empty() { "0" } else { trimmed }.to_string(),
        exp.parse().unwrap(),
    )
}

fn positional(neg: bool, digits: &str, exp10: i32) -> String {
    let n = digits.len() as i32;
    let decpt = exp10 + 1;
    let body = if decpt <= 0 {
        format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt >= n {
        format!("{}{}", digits, "0".repeat((decpt - n) as usize))
    } else {
        format!(
            "{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    };
    if neg { format!("-{body}") } else { body }
}

/// Defines `Float` methods that compute through `f32` and round back once.
macro_rules! via_f32 {
    ($($m:ident),*) => {$(
        fn $m(self) -> Self {
            Self::from_f32(self.to_f32().$m())
        }
    )*};
}

macro_rules! half_type {
    (
        $(#[$meta:meta])*
        $name:ident, $dtype:literal, $complex:literal,
        from_f32 = $from_f32:ident, from_f64 = $from_f64:ident, to_f32 = $to_f32:ident,
        eps = $eps:expr, max_bits = $max_bits:expr, inf_bits = $inf_bits:expr,
        mant_bits = $mant:expr, exp_bits = $expb:expr,
        sci_upper = $sci:expr, sqrt_thresh = $thresh:expr, direct = $direct:expr
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Default)]
        #[repr(transparent)]
        pub struct $name(u16);

        impl $name {
            #[doc = concat!("Builds a `", stringify!($name), "` from its raw bits.\n\n```\nuse rustnum as np;\nassert_eq!(np::", stringify!($name), "::from_bits(0).to_f32(), 0.0);\n```")]
            pub const fn from_bits(bits: u16) -> Self {
                Self(bits)
            }
            #[doc = concat!("The raw bits of a `", stringify!($name), "`.\n\n```\nuse rustnum as np;\nassert_ne!(np::", stringify!($name), "::from_f32(1.0).to_bits(), 0);\n```")]
            pub const fn to_bits(self) -> u16 {
                self.0
            }
            #[doc = concat!("Rounds an `f32` to `", stringify!($name), "` (nearest, ties to even), like `np.float16(x)`.\n\n```\nuse rustnum as np;\nassert_eq!(np::", stringify!($name), "::from_f32(1.5).to_f32(), 1.5);\nassert!(np::", stringify!($name), "::from_f32(f32::NAN).is_nan());\n```")]
            pub fn from_f32(x: f32) -> Self {
                Self($from_f32(x))
            }
            #[doc = concat!("Rounds an `f64` to `", stringify!($name), "` with a single rounding.\n\n```\nuse rustnum as np;\nassert_eq!(np::", stringify!($name), "::from_f64(0.5).to_f64(), 0.5);\n```")]
            pub const fn from_f64(x: f64) -> Self {
                Self($from_f64(x))
            }
            #[doc = concat!("Exact conversion to `f32`.\n\n```\nuse rustnum as np;\nassert_eq!(np::", stringify!($name), "::from_f32(2.0).to_f32(), 2.0);\n```")]
            pub fn to_f32(self) -> f32 {
                $to_f32(self.0)
            }
            #[doc = concat!("Exact conversion to `f64`.\n\n```\nuse rustnum as np;\nassert_eq!(np::", stringify!($name), "::from_f32(-2.5).to_f64(), -2.5);\n```")]
            pub fn to_f64(self) -> f64 {
                f64::from(self.to_f32())
            }
            #[doc = concat!("True for NaN, like `np.isnan`.\n\n```\nuse rustnum as np;\nassert!(np::", stringify!($name), "::NAN.is_nan());\n```")]
            pub fn is_nan(self) -> bool {
                self.to_f32().is_nan()
            }
            #[doc = concat!("True for positive or negative infinity, like `np.isinf`.\n\n```\nuse rustnum as np;\nassert!(np::", stringify!($name), "::INFINITY.is_infinite());\n```")]
            pub fn is_infinite(self) -> bool {
                self.to_f32().is_infinite()
            }
            #[doc = concat!("True when neither NaN nor infinite, like `np.isfinite`.\n\n```\nuse rustnum as np;\nassert!(np::", stringify!($name), "::MAX.is_finite());\n```")]
            pub fn is_finite(self) -> bool {
                self.to_f32().is_finite()
            }
            #[doc = concat!("The largest finite value, like `np.finfo(...).max`.\n\n```\nuse rustnum as np;\nassert!(np::", stringify!($name), "::MAX.to_f32() > 60000.0);\n```")]
            pub const MAX: Self = Self($max_bits);
            #[doc = concat!("The most negative finite value, like `np.finfo(...).min`.\n\n```\nuse rustnum as np;\nassert!(np::", stringify!($name), "::MIN.to_f32() < -60000.0);\n```")]
            pub const MIN: Self = Self($max_bits | 0x8000);
            #[doc = concat!("Machine epsilon, like `np.finfo(...).eps`.\n\n```\nuse rustnum as np;\nassert!(np::", stringify!($name), "::EPSILON.to_f32() > 0.0);\n```")]
            pub const EPSILON: Self = Self::from_f64($eps);
            #[doc = concat!("Positive infinity, like `np.inf`.\n\n```\nuse rustnum as np;\nassert_eq!(np::", stringify!($name), "::INFINITY.to_f32(), f32::INFINITY);\n```")]
            pub const INFINITY: Self = Self($inf_bits);
            #[doc = concat!("Negative infinity.\n\n```\nuse rustnum as np;\nassert_eq!(np::", stringify!($name), "::NEG_INFINITY.to_f32(), f32::NEG_INFINITY);\n```")]
            pub const NEG_INFINITY: Self = Self($inf_bits | 0x8000);
            #[doc = concat!("A quiet NaN, like `np.nan`.\n\n```\nuse rustnum as np;\nassert!(np::", stringify!($name), "::NAN.is_nan());\n```")]
            pub const NAN: Self = Self($inf_bits | (1 << ($mant - 1)));
        }

        impl PartialEq for $name {
            fn eq(&self, o: &Self) -> bool {
                self.to_f32() == o.to_f32()
            }
        }
        impl PartialOrd for $name {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                self.to_f32().partial_cmp(&o.to_f32())
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let x = self.to_f64();
                if x.is_nan() {
                    return f.pad("NaN");
                }
                if x.is_infinite() {
                    return f.pad(if x < 0.0 { "-inf" } else { "inf" });
                }
                if x == 0.0 {
                    return f.pad(if x.is_sign_negative() { "-0" } else { "0" });
                }
                let (digits, exp10) = shortest_digits(x.abs(), |v| $name::from_f64(v).to_f64());
                f.pad(&positional(x < 0.0, &digits, exp10))
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let s = self.to_string();
                if s.contains(|c: char| c == '.' || c.is_alphabetic()) {
                    f.pad(&s)
                } else {
                    f.pad(&format!("{s}.0"))
                }
            }
        }
        impl fmt::LowerExp for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let x = self.to_f64();
                if !x.is_finite() || x == 0.0 {
                    return fmt::Display::fmt(self, f);
                }
                let (digits, exp10) = shortest_digits(x.abs(), |v| $name::from_f64(v).to_f64());
                let sign = if x < 0.0 { "-" } else { "" };
                let frac = if digits.len() > 1 {
                    format!(".{}", &digits[1..])
                } else {
                    String::new()
                };
                f.pad(&format!("{sign}{}{frac}e{exp10}", &digits[..1]))
            }
        }
        impl FromStr for $name {
            type Err = std::num::ParseFloatError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                s.parse::<f64>().map($name::from_f64)
            }
        }
        impl From<$name> for f32 {
            fn from(x: $name) -> f32 {
                x.to_f32()
            }
        }
        impl From<$name> for f64 {
            fn from(x: $name) -> f64 {
                x.to_f64()
            }
        }
        impl Neg for $name {
            type Output = Self;
            fn neg(self) -> Self {
                Self(self.0 ^ 0x8000)
            }
        }
        impl Add for $name {
            type Output = Self;
            fn add(self, o: Self) -> Self {
                Self::from_f32(self.to_f32() + o.to_f32())
            }
        }
        impl Sub for $name {
            type Output = Self;
            fn sub(self, o: Self) -> Self {
                Self::from_f32(self.to_f32() - o.to_f32())
            }
        }
        impl Mul for $name {
            type Output = Self;
            fn mul(self, o: Self) -> Self {
                Self::from_f32(self.to_f32() * o.to_f32())
            }
        }
        impl Div for $name {
            type Output = Self;
            fn div(self, o: Self) -> Self {
                Self::from_f32(self.to_f32() / o.to_f32())
            }
        }
        impl Rem for $name {
            type Output = Self;
            fn rem(self, o: Self) -> Self {
                Self::from_f32(self.to_f32() % o.to_f32())
            }
        }

        impl Num for $name {
            const ZERO: Self = Self(0);
            const ONE: Self = Self::from_f64(1.0);
            const MIN_VALUE: Self = Self::MIN;
            const MAX_VALUE: Self = Self::MAX;
            fn abs(self) -> Self {
                Self(self.0 & 0x7fff)
            }
            fn from_usize(v: usize) -> Self {
                Self::from_f64(v as f64)
            }
        }
        impl crate::misc::FloatHooks for $name {
            const TWO: Self = Self::from_f64(2.0);
            const HALF: Self = Self::from_f64(0.5);
            const LN_2: Self = Self::from_f64(core::f64::consts::LN_2);
            const LOG2_E: Self = Self::from_f64(core::f64::consts::LOG2_E);
            const MANT_BITS: u32 = $mant;
            const EXP_BITS: u32 = $expb;
            const SQRT_THRESH: Self = Self::from_f64($thresh);
            const HYPERBOLIC_DIRECT: Self = Self::from_f64($direct);
            const REPR_SCI_UPPER: f64 = $sci;
            const COMPLEX_DTYPE: &'static str = $complex;
            const COMPLEX_DESCR: Option<&'static str> = None;
            const BYTES: usize = 2;
            fn to_bits64(self) -> u64 {
                u64::from(self.0)
            }
            fn from_bits64(b: u64) -> Self {
                Self(b as u16)
            }
            fn sign_bit(self) -> bool {
                self.0 & 0x8000 != 0
            }
            fn copy_sign(self, sign: Self) -> Self {
                Self((self.0 & 0x7fff) | (sign.0 & 0x8000))
            }
        }
        impl Float for $name {
            const NAN: Self = Self::NAN;
            const INFINITY: Self = Self::INFINITY;
            const NEG_INFINITY: Self = Self::NEG_INFINITY;
            const EPSILON: Self = Self::EPSILON;
            const PI: Self = Self::from_f64(core::f64::consts::PI);
            const E: Self = Self::from_f64(core::f64::consts::E);
            via_f32!(
                exp, exp2, exp_m1, ln, log2, log10, ln_1p, sqrt, cbrt, sin, cos, tan, asin, acos,
                atan, sinh, cosh, tanh, floor, ceil, trunc
            );
            fn round(self) -> Self {
                Self::from_f32(self.to_f32().round_ties_even())
            }
            fn powf(self, p: Self) -> Self {
                Self::from_f32(self.to_f32().powf(p.to_f32()))
            }
            fn powi(self, p: i32) -> Self {
                Self::from_f32(self.to_f32().powi(p))
            }
            fn is_nan(self) -> bool {
                $name::is_nan(self)
            }
            fn is_infinite(self) -> bool {
                $name::is_infinite(self)
            }
            fn is_finite(self) -> bool {
                $name::is_finite(self)
            }
            fn atan2(self, o: Self) -> Self {
                Self::from_f32(self.to_f32().atan2(o.to_f32()))
            }
            fn hypot(self, o: Self) -> Self {
                Self::from_f32(self.to_f32().hypot(o.to_f32()))
            }
            fn max(self, o: Self) -> Self {
                if self.is_nan() || o.is_nan() {
                    Self::NAN
                } else if self.to_f32() >= o.to_f32() {
                    self
                } else {
                    o
                }
            }
            fn min(self, o: Self) -> Self {
                if self.is_nan() || o.is_nan() {
                    Self::NAN
                } else if self.to_f32() <= o.to_f32() {
                    self
                } else {
                    o
                }
            }
        }
    };
}

half_type!(
    /// IEEE 754 binary16 ("half precision"), NumPy `float16`.
    ///
    /// 1 sign bit, 5 exponent bits, 10 mantissa bits. Arithmetic computes in `f32` and rounds
    /// once, like NumPy. `Array<f16>` supports every `Num` and `Float` routine; `sum`, `mean`,
    /// `var` and `std` accumulate in `f32` and round once, as NumPy does. `.npy` dtype `<f2`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::array![1.5f32, 0.1, 65504.0].astype::<np::f16>();
    /// assert_eq!(a.dtype(), "float16");
    /// assert_eq!(np::f16::from_f32(0.1).to_bits(), 0x2e66);
    /// assert_eq!(np::f16::from_f32(0.1).to_string(), "0.1");
    /// ```
    f16, "float16", "complex32",
    from_f32 = f32_to_f16_bits, from_f64 = f64_to_f16_bits, to_f32 = f16_bits_to_f32,
    eps = 0.0009765625, max_bits = 0x7bff, inf_bits = 0x7c00, mant_bits = 10, exp_bits = 5,
    sci_upper = 1e3, sqrt_thresh = 1.0e4, direct = 4.0
);
half_type!(
    /// bfloat16: the top 16 bits of an `f32` (8 exponent bits, 7 mantissa bits).
    ///
    /// Same computing model as [`struct@f16`]. NumPy has no bfloat16 dtype, so there is no `.npy`
    /// encoding; `to_npy_bytes` on a `bf16` array reports an `Error::DType`.
    ///
    /// ```
    /// use rustnum as np;
    /// let x = np::bf16::from_f32(1.0 + 1.0 / 256.0 + 1.0 / 1024.0);
    /// assert_eq!(x.to_f32(), 1.0078125);
    /// assert_eq!(np::bf16::from_f32(1.0).to_bits(), 0x3f80);
    /// ```
    bf16, "bfloat16", "bcomplex32",
    from_f32 = f32_to_bf16_bits, from_f64 = f64_to_bf16_bits, to_f32 = bf16_bits_to_f32,
    eps = 0.0078125, max_bits = 0x7f7f, inf_bits = 0x7f80, mant_bits = 7, exp_bits = 8,
    sci_upper = 1e3, sqrt_thresh = 1.0e30, direct = 4.0
);

impl Element for f16 {
    const DTYPE: &'static str = "float16";
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
    fn from_f64(v: f64) -> Self {
        f16::from_f64(v)
    }
    fn npy_descr() -> Option<String> {
        Some("<f2".to_string())
    }
    fn npy_itemsize() -> usize {
        2
    }
    fn npy_write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
    fn npy_read(b: &[u8]) -> Self {
        f16(u16::from_le_bytes([b[0], b[1]]))
    }
}
impl Element for bf16 {
    const DTYPE: &'static str = "bfloat16";
    fn to_f64(self) -> f64 {
        bf16::to_f64(self)
    }
    fn from_f64(v: f64) -> Self {
        bf16::from_f64(v)
    }
}

impl f16 {
    /// Converts a slice of `f32` to `f16` (nearest, ties to even), like `a.astype(np.float16)`.
    ///
    /// Uses the F16C instructions eight lanes at a time when the CPU supports them. Panics when
    /// the slices differ in length.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut out = vec![np::f16::from_bits(0); 3];
    /// np::f16::slice_from_f32(&[1.0, 0.5, -2.0], &mut out);
    /// assert_eq!(out[2].to_f32(), -2.0);
    /// ```
    pub fn slice_from_f32(src: &[f32], dst: &mut [f16]) {
        assert_eq!(src.len(), dst.len(), "slices must have the same length");
        // SAFETY: f16 is repr(transparent) over u16, so the slices have identical layout.
        let bits: &mut [u16] =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u16>(), dst.len()) };
        bulk_f32_to_f16(src, bits);
    }

    /// Converts a slice of `f16` to `f32` exactly, like `a.astype(np.float32)`.
    ///
    /// Uses the F16C instructions when available. Panics when the slices differ in length.
    ///
    /// ```
    /// use rustnum as np;
    /// let src = [np::f16::from_f32(1.5), np::f16::from_f32(-0.25)];
    /// let mut out = vec![0.0f32; 2];
    /// np::f16::slice_to_f32(&src, &mut out);
    /// assert_eq!(out, [1.5, -0.25]);
    /// ```
    pub fn slice_to_f32(src: &[f16], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len(), "slices must have the same length");
        // SAFETY: f16 is repr(transparent) over u16, so the slices have identical layout.
        let bits: &[u16] =
            unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u16>(), src.len()) };
        bulk_f16_to_f32(bits, dst);
    }

    /// Scalar-only `f32` to `f16` conversion of a slice, the reference path for tests and benchmarks.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut out = vec![np::f16::from_bits(0); 2];
    /// np::f16::slice_from_f32_scalar(&[1.0, 2.0], &mut out);
    /// assert_eq!(out[1].to_f32(), 2.0);
    /// ```
    pub fn slice_from_f32_scalar(src: &[f32], dst: &mut [f16]) {
        assert_eq!(src.len(), dst.len(), "slices must have the same length");
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = f16(f32_to_f16_bits(s));
        }
    }
}

/// `astype` fast path for contiguous `f32 <-> f16 / bf16` data; `None` for every other pair.
pub(crate) fn bulk_astype<T: Element, U: Element>(a: &Array<T>) -> Option<Array<U>> {
    let (from, to) = (TypeId::of::<T>(), TypeId::of::<U>());
    let (f32_id, f16_id, bf16_id) = (
        TypeId::of::<f32>(),
        TypeId::of::<f16>(),
        TypeId::of::<bf16>(),
    );
    let supported = (from == f32_id && (to == f16_id || to == bf16_id))
        || (to == f32_id && (from == f16_id || from == bf16_id));
    if !supported {
        return None;
    }
    let src = a.as_slice()?;
    macro_rules! go {
        ($s:ty, $d:ty, $conv:expr) => {
            if from == TypeId::of::<$s>() && to == TypeId::of::<$d>() {
                // SAFETY: the TypeId checks above prove T == $s.
                let s: &[$s] =
                    unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<$s>(), src.len()) };
                let mut out: Vec<$d> = vec![<$d>::default(); s.len()];
                ($conv)(s, &mut out[..]);
                let boxed: Box<dyn Any> = Box::new(Array::from_vec(out, a.shape()));
                return boxed.downcast::<Array<U>>().ok().map(|b| *b);
            }
        };
    }
    go!(f32, f16, f16::slice_from_f32);
    go!(f16, f32, f16::slice_to_f32);
    go!(f32, bf16, |s: &[f32], d: &mut [bf16]| {
        for (o, &x) in d.iter_mut().zip(s) {
            *o = bf16(f32_to_bf16_bits(x));
        }
    });
    go!(bf16, f32, |s: &[bf16], d: &mut [f32]| {
        for (o, &x) in d.iter_mut().zip(s) {
            *o = bf16_bits_to_f32(x.0);
        }
    });
    None
}

/// True for the 16-bit float element types, whose reductions accumulate in `f32`.
pub(crate) fn is_half<T: Element>() -> bool {
    T::DTYPE == "float16" || T::DTYPE == "bfloat16"
}
