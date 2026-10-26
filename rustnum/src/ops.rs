use crate::element::{add, mul, sub};
use crate::{Array, Element, Float, IntoArray, Num};
use std::ops::*;
macro_rules! binary {
    ($trait:ident,$method:ident,$f:expr) => {
        impl<T: Num> $trait<Array<T>> for Array<T> {
            type Output = Array<T>;
            fn $method(self, r: Array<T>) -> Self::Output {
                self.into_zip_owned(r, $f)
            }
        }
        impl<T: Num> $trait<&Array<T>> for Array<T> {
            type Output = Array<T>;
            fn $method(self, r: &Array<T>) -> Self::Output {
                self.into_zip(r, $f)
            }
        }
        impl<T: Num> $trait<Array<T>> for &Array<T> {
            type Output = Array<T>;
            fn $method(self, r: Array<T>) -> Self::Output {
                r.into_zip(self, |y, x| ($f)(x, y))
            }
        }
        impl<T: Num> $trait<&Array<T>> for &Array<T> {
            type Output = Array<T>;
            fn $method(self, r: &Array<T>) -> Self::Output {
                self.par_zip(r, $f)
            }
        }
        impl<T: Num> $trait<T> for Array<T> {
            type Output = Array<T>;
            fn $method(self, r: T) -> Self::Output {
                self.into_map(|x| ($f)(x, r))
            }
        }
        impl<T: Num> $trait<T> for &Array<T> {
            type Output = Array<T>;
            fn $method(self, r: T) -> Self::Output {
                self.par_zip(r, $f)
            }
        }
    };
}
binary!(Add, add, add);
binary!(Sub, sub, sub);
binary!(Mul, mul, mul);
binary!(Div, div, |a, b| a / b);
binary!(Rem, rem, |a, b| a % b);
macro_rules! scalar_left {
    ($t:ty,$trait:ident,$method:ident,$f:expr) => {
        impl $trait<Array<$t>> for $t {
            type Output = Array<$t>;
            fn $method(self, r: Array<$t>) -> Self::Output {
                r.into_map(|x| ($f)(self, x))
            }
        }
        impl $trait<&Array<$t>> for $t {
            type Output = Array<$t>;
            fn $method(self, r: &Array<$t>) -> Self::Output {
                r.par_map(|x| ($f)(self, x))
            }
        }
    };
    ($t:ty) => {
        scalar_left!($t, Add, add, add);
        scalar_left!($t, Sub, sub, sub);
        scalar_left!($t, Mul, mul, mul);
        scalar_left!($t, Div, div, |a, b| a / b);
        scalar_left!($t, Rem, rem, |a, b| a % b);
    };
}
scalar_left!(f32);
scalar_left!(f64);
scalar_left!(i32);
scalar_left!(i64);
scalar_left!(u8);
scalar_left!(usize);
macro_rules! assign {
    ($trait:ident,$method:ident,$f:expr) => {
        impl<T: Num, R: IntoArray<T>> $trait<R> for Array<T> {
            fn $method(&mut self, r: R) {
                if let Some(value) = r.scalar_value() {
                    self.mapv_inplace_parallel(|x| ($f)(x, value));
                } else {
                    self.zip_inplace(&r.into_array(), $f);
                }
            }
        }
    };
}
assign!(AddAssign, add_assign, add);
assign!(SubAssign, sub_assign, sub);
assign!(MulAssign, mul_assign, mul);
assign!(DivAssign, div_assign, |a, b| a / b);
assign!(RemAssign, rem_assign, |a, b| a % b);
impl<T: Num + Neg<Output = T>> Neg for Array<T> {
    type Output = Self;
    fn neg(self) -> Self {
        self.into_map(|x| {
            if T::DTYPE.starts_with("float") || T::DTYPE.starts_with("complex") {
                -x
            } else {
                sub(T::ZERO, x)
            }
        })
    }
}
impl<T: Num + Neg<Output = T>> Neg for &Array<T> {
    type Output = Array<T>;
    fn neg(self) -> Self::Output {
        self.par_map(|x| {
            if T::DTYPE.starts_with("float") || T::DTYPE.starts_with("complex") {
                -x
            } else {
                sub(T::ZERO, x)
            }
        })
    }
}
impl<T: Element> Array<T> {
    /// Implements np.equal with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].eq_elem(2).to_vec(),[false,true]);
    /// ```
    pub fn eq_elem(&self, b: impl IntoArray<T>) -> Array<bool> {
        self.par_zip(b, |x, y| x == y)
    }
    /// Implements np.not_equal with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].ne_elem(2).to_vec(),[true,false]);
    /// ```
    pub fn ne_elem(&self, b: impl IntoArray<T>) -> Array<bool> {
        self.par_zip(b, |x, y| x != y)
    }
    /// Implements np.less with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].lt(2).to_vec(),[true,false]);
    /// ```
    pub fn lt(&self, b: impl IntoArray<T>) -> Array<bool> {
        self.par_zip(b, |x, y| x < y)
    }
    /// Implements np.less_equal with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].le(2).to_vec(),[true,true]);
    /// ```
    pub fn le(&self, b: impl IntoArray<T>) -> Array<bool> {
        self.par_zip(b, |x, y| x <= y)
    }
    /// Implements np.greater with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].gt(1).to_vec(),[false,true]);
    /// ```
    pub fn gt(&self, b: impl IntoArray<T>) -> Array<bool> {
        self.par_zip(b, |x, y| x > y)
    }
    /// Implements np.greater_equal with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].ge(1).to_vec(),[true,true]);
    /// ```
    pub fn ge(&self, b: impl IntoArray<T>) -> Array<bool> {
        self.par_zip(b, |x, y| x >= y)
    }
    /// Implements np.array_equal with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::array![1,2].array_equal([1,2]));
    /// ```
    pub fn array_equal(&self, b: impl IntoArray<T>) -> bool {
        self == &b.into_array()
    }
}
impl<F: Float> Array<F> {
    /// Implements np.allclose with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::array![1.0f64,2.].allclose([1.000001,2.],1e-5,1e-8));
    /// ```
    pub fn allclose(&self, b: impl IntoArray<F>, rtol: F, atol: F) -> bool {
        isclose(self, b, rtol, atol).all()
    }
}
macro_rules! bool_binary {
 ($trait:ident,$method:ident,$op:tt)=>{
 impl $trait<Array<bool>> for Array<bool>{type Output=Array<bool>;fn $method(self,r:Array<bool>)->Self::Output{self.into_zip_owned(r,|a,b|a $op b)}}
 impl $trait<&Array<bool>> for Array<bool>{type Output=Array<bool>;fn $method(self,r:&Array<bool>)->Self::Output{self.into_zip(r,|a,b|a $op b)}}
 impl $trait<Array<bool>> for &Array<bool>{type Output=Array<bool>;fn $method(self,r:Array<bool>)->Self::Output{r.into_zip(self,|b,a|a $op b)}}
 impl $trait<&Array<bool>> for &Array<bool>{type Output=Array<bool>;fn $method(self,r:&Array<bool>)->Self::Output{self.par_zip(r,|a,b|a $op b)}}
 impl $trait<bool> for Array<bool>{type Output=Array<bool>;fn $method(self,r:bool)->Self::Output{self.into_map(|a|a $op r)}}
 impl $trait<bool> for &Array<bool>{type Output=Array<bool>;fn $method(self,r:bool)->Self::Output{self.par_zip(r,|a,b|a $op b)}}
 };
}
bool_binary!(BitAnd,bitand,&);
bool_binary!(BitOr,bitor,|);
bool_binary!(BitXor,bitxor,^);
impl Not for Array<bool> {
    type Output = Self;
    fn not(self) -> Self {
        self.into_map(|x| !x)
    }
}
impl Not for &Array<bool> {
    type Output = Array<bool>;
    fn not(self) -> Self::Output {
        self.par_map(|x| !x)
    }
}
/// Implements np.logical_and with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::logical_and([true,false],true).to_vec(),[true,false]);
/// ```
pub fn logical_and(a: impl IntoArray<bool>, b: impl IntoArray<bool>) -> Array<bool> {
    a.into_array().par_zip(b, |a, b| a && b)
}
/// Implements np.logical_or with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::logical_or([true,false],true).to_vec(),[true,true]);
/// ```
pub fn logical_or(a: impl IntoArray<bool>, b: impl IntoArray<bool>) -> Array<bool> {
    a.into_array().par_zip(b, |a, b| a || b)
}
/// Implements np.logical_xor with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::logical_xor([true,false],true).to_vec(),[false,true]);
/// ```
pub fn logical_xor(a: impl IntoArray<bool>, b: impl IntoArray<bool>) -> Array<bool> {
    a.into_array().par_zip(b, |a, b| a ^ b)
}
/// Implements np.logical_not with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::logical_not([true,false]).to_vec(),[false,true]);
/// ```
pub fn logical_not(a: impl IntoArray<bool>) -> Array<bool> {
    a.into_array().par_map(|x| !x)
}
/// Implements np.isclose with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::isclose([1.0,2.],[1.000001,3.],1e-5,1e-8).to_vec(),[true,false]);
/// ```
pub fn isclose<F: Float>(
    a: impl IntoArray<F>,
    b: impl IntoArray<F>,
    rtol: F,
    atol: F,
) -> Array<bool> {
    a.into_array().par_zip(b, |x, y| {
        x == y || (x.is_finite() && y.is_finite() && (x - y).abs() <= atol + rtol * y.abs())
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_left_float() {
        let a: Array<f64> = crate::array![1.0, 2.0];
        let b = 2.0 * &a;
        assert_eq!(b.to_vec(), [2.0, 4.0]);
        let c = crate::array![1f32, 2.];
        assert_eq!((&c * 2f32).to_vec(), [2., 4.]);
    }
    #[test]
    fn close_infinities() {
        let a = crate::array![f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        assert_eq!(isclose(&a, &a, 1e-5, 1e-8).to_vec(), [true, true, false]);
    }
    #[test]
    fn compound_broadcast() {
        let mut a = crate::array![[1, 2], [3, 4]];
        a += crate::array![10, 20];
        a *= 2;
        assert_eq!(a.to_vec(), [22, 44, 26, 48]);
    }
}
