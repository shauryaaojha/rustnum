use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// A value which can be stored in an array, like a NumPy dtype.
/// ```
/// use rustnum as np;
/// assert_eq!(<f64 as np::Element>::DTYPE,"float64");
/// ```
pub trait Element:
    Copy + Clone + Send + Sync + Debug + Default + PartialEq + PartialOrd + 'static
{
    /// Scalar equivalent of np.dtype.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<f64 as np::Element>::DTYPE,"float64");
    /// ```
    const DTYPE: &'static str;
    /// Scalar equivalent of np.float64.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<bool as np::Element>::to_f64(true),1.0);
    /// ```
    fn to_f64(self) -> f64;
    /// Scalar equivalent of np.astype.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<i32 as np::Element>::from_f64(2.9),2);
    /// ```
    fn from_f64(value: f64) -> Self;
}
/// A numeric array element, like NumPy numeric dtypes.
/// ```
/// use rustnum as np;
/// assert_eq!(<i32 as np::Num>::ONE,1);
/// ```
pub trait Num:
    Element
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
{
    /// Scalar equivalent of np.zeros.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<i32 as np::Num>::ZERO,0);
    /// ```
    const ZERO: Self;
    /// Scalar equivalent of np.ones.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<i32 as np::Num>::ONE,1);
    /// ```
    const ONE: Self;
    /// Scalar equivalent of np.iinfo / finfo.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<i32 as np::Num>::MIN_VALUE,i32::MIN);
    /// ```
    const MIN_VALUE: Self;
    /// Scalar equivalent of np.iinfo / finfo.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<i32 as np::Num>::MAX_VALUE,i32::MAX);
    /// ```
    const MAX_VALUE: Self;
    /// Scalar equivalent of np.abs.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<i32 as np::Num>::abs(-2),2);
    /// ```
    fn abs(self) -> Self;
    /// Scalar equivalent of np.astype.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<i32 as np::Num>::from_usize(3),3);
    /// ```
    fn from_usize(value: usize) -> Self;
}
/// A floating point array element, like NumPy floating dtypes.
/// ```
/// use rustnum as np;
/// assert!(<f64 as np::Float>::PI>3.0);
/// ```
///
/// `Float` has a hidden, sealed supertrait (`FloatHooks`) that carries the bit-level helpers and
/// constants of the `f32`/`f64` kernels. Its items never clash with `Float`/`Num`/`Element`, so
/// generic code bounded by `F: Float` can call every `Array<F>` method.
pub trait Float: Num + Neg<Output = Self> + crate::misc::FloatHooks {
    /// Scalar equivalent of np.nan.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::NAN.is_nan());
    /// ```
    const NAN: Self;
    /// Scalar equivalent of np.inf.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::INFINITY.is_infinite());
    /// ```
    const INFINITY: Self;
    /// Scalar equivalent of np.negative(inf).
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::NEG_INFINITY.is_infinite());
    /// ```
    const NEG_INFINITY: Self;
    /// Scalar equivalent of np.finfo.eps.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::EPSILON>0.0);
    /// ```
    const EPSILON: Self;
    /// Scalar equivalent of np.pi.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::PI>0.0);
    /// ```
    const PI: Self;
    /// Scalar equivalent of np.e.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::E>0.0);
    /// ```
    const E: Self;
    /// Scalar equivalent of np.exp.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::exp(0.5).is_finite());
    /// ```
    fn exp(self) -> Self;
    /// Scalar equivalent of np.exp2.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::exp2(0.5).is_finite());
    /// ```
    fn exp2(self) -> Self;
    /// Scalar equivalent of np.expm1.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::exp_m1(0.5).is_finite());
    /// ```
    fn exp_m1(self) -> Self;
    /// Scalar equivalent of np.log.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::ln(0.5).is_finite());
    /// ```
    fn ln(self) -> Self;
    /// Scalar equivalent of np.log2.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::log2(0.5).is_finite());
    /// ```
    fn log2(self) -> Self;
    /// Scalar equivalent of np.log10.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::log10(0.5).is_finite());
    /// ```
    fn log10(self) -> Self;
    /// Scalar equivalent of np.log1p.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::ln_1p(0.5).is_finite());
    /// ```
    fn ln_1p(self) -> Self;
    /// Scalar equivalent of np.sqrt.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::sqrt(0.5).is_finite());
    /// ```
    fn sqrt(self) -> Self;
    /// Scalar equivalent of np.cbrt.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::cbrt(0.5).is_finite());
    /// ```
    fn cbrt(self) -> Self;
    /// Scalar equivalent of np.sin.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::sin(0.5).is_finite());
    /// ```
    fn sin(self) -> Self;
    /// Scalar equivalent of np.cos.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::cos(0.5).is_finite());
    /// ```
    fn cos(self) -> Self;
    /// Scalar equivalent of np.tan.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::tan(0.5).is_finite());
    /// ```
    fn tan(self) -> Self;
    /// Scalar equivalent of np.arcsin.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::asin(0.5).is_finite());
    /// ```
    fn asin(self) -> Self;
    /// Scalar equivalent of np.arccos.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::acos(0.5).is_finite());
    /// ```
    fn acos(self) -> Self;
    /// Scalar equivalent of np.arctan.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::atan(0.5).is_finite());
    /// ```
    fn atan(self) -> Self;
    /// Scalar equivalent of np.sinh.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::sinh(0.5).is_finite());
    /// ```
    fn sinh(self) -> Self;
    /// Scalar equivalent of np.cosh.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::cosh(0.5).is_finite());
    /// ```
    fn cosh(self) -> Self;
    /// Scalar equivalent of np.tanh.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::tanh(0.5).is_finite());
    /// ```
    fn tanh(self) -> Self;
    /// Scalar equivalent of np.floor.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::floor(0.5).is_finite());
    /// ```
    fn floor(self) -> Self;
    /// Scalar equivalent of np.ceil.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::ceil(0.5).is_finite());
    /// ```
    fn ceil(self) -> Self;
    /// Scalar equivalent of np.round.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::round(0.5).is_finite());
    /// ```
    fn round(self) -> Self;
    /// Scalar equivalent of np.trunc.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::trunc(0.5).is_finite());
    /// ```
    fn trunc(self) -> Self;
    /// Scalar equivalent of np.power.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::powf(0.5,2.0).is_finite());
    /// ```
    fn powf(self, p: Self) -> Self;
    /// Scalar equivalent of np.power.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::powi(0.5,2).is_finite());
    /// ```
    fn powi(self, p: i32) -> Self;
    /// Scalar equivalent of np.isnan.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<f64 as np::Float>::is_nan(0.5),false);
    /// ```
    fn is_nan(self) -> bool;
    /// Scalar equivalent of np.isinf.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<f64 as np::Float>::is_infinite(0.5),false);
    /// ```
    fn is_infinite(self) -> bool;
    /// Scalar equivalent of np.isfinite.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(<f64 as np::Float>::is_finite(0.5),true);
    /// ```
    fn is_finite(self) -> bool;
    /// Scalar equivalent of np.atan2.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::atan2(0.5,1.0).is_finite());
    /// ```
    fn atan2(self, o: Self) -> Self;
    /// Scalar equivalent of np.hypot.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::hypot(0.5,1.0).is_finite());
    /// ```
    fn hypot(self, o: Self) -> Self;
    /// Scalar equivalent of np.maximum.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::max(0.5,1.0).is_finite());
    /// ```
    fn max(self, o: Self) -> Self;
    /// Scalar equivalent of np.minimum.
    /// ```
    /// use rustnum as np;
    /// assert!(<f64 as np::Float>::min(0.5,1.0).is_finite());
    /// ```
    fn min(self, o: Self) -> Self;
}
macro_rules! element {
    ($t:ty,$n:literal) => {
        impl Element for $t {
            const DTYPE: &'static str = $n;
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_f64(v: f64) -> Self {
                v as $t
            }
        }
    };
}
element!(f32, "float32");
element!(f64, "float64");
element!(i32, "int32");
element!(i64, "int64");
element!(u8, "uint8");
element!(usize, "uint64");
impl Element for bool {
    const DTYPE: &'static str = "bool";
    fn to_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
}
macro_rules! num {
    ($t:ty,$abs:ident) => {
        impl Num for $t {
            const ZERO: Self = 0 as $t;
            const ONE: Self = 1 as $t;
            const MIN_VALUE: Self = <$t>::MIN;
            const MAX_VALUE: Self = <$t>::MAX;
            fn abs(self) -> Self {
                self.$abs()
            }
            fn from_usize(v: usize) -> Self {
                v as $t
            }
        }
    };
}
num!(f32, abs);
num!(f64, abs);
num!(i32, wrapping_abs);
num!(i64, wrapping_abs);
impl Num for u8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN_VALUE: Self = u8::MIN;
    const MAX_VALUE: Self = u8::MAX;
    fn abs(self) -> Self {
        self
    }
    fn from_usize(v: usize) -> Self {
        v as u8
    }
}
impl Num for usize {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN_VALUE: Self = usize::MIN;
    const MAX_VALUE: Self = usize::MAX;
    fn abs(self) -> Self {
        self
    }
    fn from_usize(v: usize) -> Self {
        v
    }
}
macro_rules! float {
    ($t:ty) => {
        impl Float for $t {
            const NAN: Self = <$t>::NAN;
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            const EPSILON: Self = <$t>::EPSILON;
            const PI: Self = core::f64::consts::PI as $t;
            const E: Self = core::f64::consts::E as $t;
            fn exp(self) -> Self {
                self.exp()
            }
            fn exp2(self) -> Self {
                self.exp2()
            }
            fn exp_m1(self) -> Self {
                self.exp_m1()
            }
            fn ln(self) -> Self {
                self.ln()
            }
            fn log2(self) -> Self {
                self.log2()
            }
            fn log10(self) -> Self {
                self.log10()
            }
            fn ln_1p(self) -> Self {
                self.ln_1p()
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn cbrt(self) -> Self {
                self.cbrt()
            }
            fn sin(self) -> Self {
                self.sin()
            }
            fn cos(self) -> Self {
                self.cos()
            }
            fn tan(self) -> Self {
                self.tan()
            }
            fn asin(self) -> Self {
                self.asin()
            }
            fn acos(self) -> Self {
                self.acos()
            }
            fn atan(self) -> Self {
                self.atan()
            }
            fn sinh(self) -> Self {
                self.sinh()
            }
            fn cosh(self) -> Self {
                self.cosh()
            }
            fn tanh(self) -> Self {
                self.tanh()
            }
            fn floor(self) -> Self {
                self.floor()
            }
            fn ceil(self) -> Self {
                self.ceil()
            }
            fn round(self) -> Self {
                self.round_ties_even()
            }
            fn trunc(self) -> Self {
                self.trunc()
            }
            fn powf(self, p: Self) -> Self {
                self.powf(p)
            }
            fn powi(self, p: i32) -> Self {
                self.powi(p)
            }
            fn is_nan(self) -> bool {
                self.is_nan()
            }
            fn is_infinite(self) -> bool {
                self.is_infinite()
            }
            fn is_finite(self) -> bool {
                self.is_finite()
            }
            fn atan2(self, o: Self) -> Self {
                self.atan2(o)
            }
            fn hypot(self, o: Self) -> Self {
                self.hypot(o)
            }
            fn max(self, o: Self) -> Self {
                if self.is_nan() || o.is_nan() {
                    Self::NAN
                } else {
                    self.max(o)
                }
            }
            fn min(self, o: Self) -> Self {
                if self.is_nan() || o.is_nan() {
                    Self::NAN
                } else {
                    self.min(o)
                }
            }
        }
    };
}
float!(f32);
float!(f64);

pub(crate) fn isnan<T: Element>(x: T) -> bool {
    x.partial_cmp(&x).is_none()
}
pub(crate) fn maximum_value<T: Element>(x: T, y: T) -> T {
    if isnan(x) || x > y { x } else { y }
}
pub(crate) fn minimum_value<T: Element>(x: T, y: T) -> T {
    if isnan(x) || x < y { x } else { y }
}
macro_rules! arithmetic {
    ($name:ident,$method:ident,$op:tt) => {
        pub(crate) fn $name<T:Num>(x:T,y:T)->T {
            use std::any::Any;
            macro_rules! cast {($t:ty)=> {
                if let (Some(a),Some(b))=((&x as &dyn Any).downcast_ref::<$t>(),(&y as &dyn Any).downcast_ref::<$t>()) {
                    let value=a.$method(*b);
                    return *(&value as &dyn Any).downcast_ref::<T>().expect("same numeric type");
                }
            }}
            cast!(i32);cast!(i64);cast!(u8);cast!(usize);
            x $op y
        }
    };
}
arithmetic!(add,wrapping_add,+);
arithmetic!(sub,wrapping_sub,-);
arithmetic!(mul,wrapping_mul,*);
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dtype_casts() {
        assert!(bool::from_f64(-1.0));
        assert_eq!(i64::DTYPE, "int64");
        assert_eq!(u8::from_f64(2.9), 2);
    }
    #[test]
    fn bankers_rounding() {
        assert_eq!(Float::round(2.5f64), 2.0);
        assert_eq!(Float::round(3.5f64), 4.0);
    }
    #[test]
    fn nan_extrema() {
        assert!(Float::max(f64::NAN, 1.0).is_nan());
    }
}
