use crate::{Array, Float, IntoArray, Num};
impl<F: Float> Array<F> {
    /// Implements np.exp with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].exp().item(),1f64);
    /// ```
    pub fn exp(&self) -> Self {
        self.par_map(F::exp)
    }
    /// Implements np.exp2 with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![3f64].exp2().item(),8f64);
    /// ```
    pub fn exp2(&self) -> Self {
        self.par_map(F::exp2)
    }
    /// Implements np.expm1 with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].expm1().item(),0f64);
    /// ```
    pub fn expm1(&self) -> Self {
        self.par_map(F::exp_m1)
    }
    /// Implements np.log with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1f64].log().item(),0f64);
    /// ```
    pub fn log(&self) -> Self {
        self.par_map(F::ln)
    }
    /// Implements np.log2 with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![8f64].log2().item(),3f64);
    /// ```
    pub fn log2(&self) -> Self {
        self.par_map(F::log2)
    }
    /// Implements np.log10 with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![100f64].log10().item(),2f64);
    /// ```
    pub fn log10(&self) -> Self {
        self.par_map(F::log10)
    }
    /// Implements np.log1p with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].log1p().item(),0f64);
    /// ```
    pub fn log1p(&self) -> Self {
        self.par_map(F::ln_1p)
    }
    /// Implements np.sqrt with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![4f64].sqrt().item(),2f64);
    /// ```
    pub fn sqrt(&self) -> Self {
        self.par_map(F::sqrt)
    }
    /// Implements np.cbrt with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![8f64].cbrt().item(),2f64);
    /// ```
    pub fn cbrt(&self) -> Self {
        self.par_map(F::cbrt)
    }
    /// Implements np.sin with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].sin().item(),0f64);
    /// ```
    pub fn sin(&self) -> Self {
        self.par_map(F::sin)
    }
    /// Implements np.cos with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].cos().item(),1f64);
    /// ```
    pub fn cos(&self) -> Self {
        self.par_map(F::cos)
    }
    /// Implements np.tan with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].tan().item(),0f64);
    /// ```
    pub fn tan(&self) -> Self {
        self.par_map(F::tan)
    }
    /// Implements np.arcsin with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].arcsin().item(),0f64);
    /// ```
    pub fn arcsin(&self) -> Self {
        self.par_map(F::asin)
    }
    /// Implements np.arccos with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1f64].arccos().item(),0f64);
    /// ```
    pub fn arccos(&self) -> Self {
        self.par_map(F::acos)
    }
    /// Implements np.arctan with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].arctan().item(),0f64);
    /// ```
    pub fn arctan(&self) -> Self {
        self.par_map(F::atan)
    }
    /// Implements np.sinh with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].sinh().item(),0f64);
    /// ```
    pub fn sinh(&self) -> Self {
        self.par_map(F::sinh)
    }
    /// Implements np.cosh with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].cosh().item(),1f64);
    /// ```
    pub fn cosh(&self) -> Self {
        self.par_map(F::cosh)
    }
    /// Implements np.tanh with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0f64].tanh().item(),0f64);
    /// ```
    pub fn tanh(&self) -> Self {
        self.par_map(F::tanh)
    }
    /// Implements np.floor with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.9f64].floor().item(),1f64);
    /// ```
    pub fn floor(&self) -> Self {
        self.par_map(F::floor)
    }
    /// Implements np.ceil with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.1f64].ceil().item(),2f64);
    /// ```
    pub fn ceil(&self) -> Self {
        self.par_map(F::ceil)
    }
    /// Implements np.round with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![2.5f64].round().item(),2f64);
    /// ```
    pub fn round(&self) -> Self {
        self.par_map(F::round)
    }
    /// Implements np.round with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.25f64,1.35].round_decimals(1).to_vec(),[1.2,1.4]);
    /// ```
    pub fn round_decimals(&self, d: i32) -> Self {
        let scale = F::from_usize(10).powf(F::from_f64((d as f64).abs()));
        self.par_map(|x| {
            if d >= 0 {
                (x * scale).round() / scale
            } else {
                (x / scale).round() * scale
            }
        })
    }
    /// Implements np.trunc with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-1.9f64].trunc().item(),-1f64);
    /// ```
    pub fn trunc(&self) -> Self {
        self.par_map(F::trunc)
    }
    /// Implements np.reciprocal with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,2.].reciprocal().to_vec(),[1.,0.5]);
    /// ```
    pub fn reciprocal(&self) -> Self {
        self.par_map(|x| F::ONE / x)
    }
    /// Implements np.power with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![4.0f64,9.].powf(0.5).to_vec(),[2.,3.]);
    /// ```
    pub fn powf(&self, p: F) -> Self {
        self.par_map(|x| x.powf(p))
    }
    /// Implements np.power with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![2.0f64,3.].powi(3).to_vec(),[8.,27.]);
    /// ```
    pub fn powi(&self, p: i32) -> Self {
        self.par_map(|x| x.powi(p))
    }
    /// Implements np.isnan with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![f64::NAN,0.].is_nan().to_vec(),[true,false]);
    /// ```
    pub fn is_nan(&self) -> Array<bool> {
        self.par_map(F::is_nan)
    }
    /// Implements np.isinf with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![f64::INFINITY,0.].is_inf().to_vec(),[true,false]);
    /// ```
    pub fn is_inf(&self) -> Array<bool> {
        self.par_map(F::is_infinite)
    }
    /// Implements np.isfinite with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![f64::NAN,0.].is_finite().to_vec(),[false,true]);
    /// ```
    pub fn is_finite(&self) -> Array<bool> {
        self.par_map(F::is_finite)
    }
    /// Logistic activation (non-NumPy extra, expressible with np.exp).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0.0f64].sigmoid().item(),0.5);
    /// ```
    pub fn sigmoid(&self) -> Self {
        self.par_map(|x| {
            if x >= F::ZERO {
                F::ONE / (F::ONE + (-x).exp())
            } else {
                let e = x.exp();
                e / (F::ONE + e)
            }
        })
    }
    /// Rectified linear activation (non-NumPy extra, equivalent to np.maximum(a, 0)).
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-2.0f64,1.].relu().to_vec(),[0.,1.]);
    /// ```
    pub fn relu(&self) -> Self {
        self.par_map(|x| x.max(F::ZERO))
    }
}
impl<T: Num> Array<T> {
    /// Implements np.abs with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-2,0,3].abs().to_vec(),[2,0,3]);
    /// ```
    pub fn abs(&self) -> Self {
        self.par_map(Num::abs)
    }
    /// Implements np.square with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-2,3].square().to_vec(),[4,9]);
    /// ```
    pub fn square(&self) -> Self {
        self.par_map(|x| crate::element::mul(x, x))
    }
    /// Implements np.clip with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-2,1,9].clip(0,3).to_vec(),[0,1,3]);
    /// ```
    pub fn clip(&self, lo: T, hi: T) -> Self {
        self.par_map(|x| crate::element::minimum_value(crate::element::maximum_value(x, lo), hi))
    }
    /// Implements np.sign with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![-2,0,9].sign().to_vec(),[-1,0,1]);
    /// ```
    pub fn sign(&self) -> Self {
        self.par_map(|x| {
            if x > T::ZERO {
                T::ONE
            } else if x < T::ZERO {
                crate::element::sub(T::ZERO, T::ONE)
            } else if crate::element::isnan(x) {
                x
            } else {
                T::ZERO
            }
        })
    }
}

/// Implements np.maximum with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::maximum(2,[1,3]).to_vec(),[2,3]);
/// ```
pub fn maximum<T: Num>(a: impl IntoArray<T>, b: impl IntoArray<T>) -> Array<T> {
    a.into_array().par_zip(b, crate::element::maximum_value)
}
/// Implements np.minimum with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::minimum([1,3],2).to_vec(),[1,2]);
/// ```
pub fn minimum<T: Num>(a: impl IntoArray<T>, b: impl IntoArray<T>) -> Array<T> {
    a.into_array().par_zip(b, crate::element::minimum_value)
}
/// Implements np.power with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::power([2.0f64,3.],2.).to_vec(),[4.,9.]);
/// ```
pub fn power<T: Num>(a: impl IntoArray<T>, b: impl IntoArray<T>) -> Array<T> {
    a.into_array().par_zip(b, |x, y| {
        use std::any::Any;
        if let (Some(&x), Some(&y)) = (
            (&x as &dyn Any).downcast_ref::<f32>(),
            (&y as &dyn Any).downcast_ref::<f32>(),
        ) {
            let v = x.powf(y);
            return *(&v as &dyn Any).downcast_ref::<T>().unwrap();
        }
        if let (Some(&x), Some(&y)) = (
            (&x as &dyn Any).downcast_ref::<f64>(),
            (&y as &dyn Any).downcast_ref::<f64>(),
        ) {
            let v = x.powf(y);
            return *(&v as &dyn Any).downcast_ref::<T>().unwrap();
        }
        assert!(
            y >= T::ZERO,
            "Integers to negative integer powers are not allowed."
        );
        let mut exponent = y;
        let mut base = x;
        let mut result = T::ONE;
        let two = T::from_usize(2);
        while exponent > T::ZERO {
            if exponent % two != T::ZERO {
                result = crate::element::mul(result, base)
            }
            exponent = exponent / two;
            if exponent > T::ZERO {
                base = crate::element::mul(base, base)
            }
        }
        result
    })
}
/// Implements np.arctan2 with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::arctan2(0.0f64,[1.,2.]).to_vec(),[0.,0.]);
/// ```
pub fn arctan2<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, F::atan2)
}
/// Implements np.hypot with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::hypot(3.0f64,4.).item(),5.);
/// ```
pub fn hypot<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F> {
    a.into_array().par_zip(b, F::hypot)
}
/// Implements np.where with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::where_(&np::array![true,false],1,[2,3]).to_vec(),[1,3]);
/// ```
pub fn where_<T: crate::Element>(
    c: &Array<bool>,
    x: impl IntoArray<T>,
    y: impl IntoArray<T>,
) -> Array<T> {
    let x = x.into_array();
    let y = y.into_array();
    let shape = crate::broadcast_shapes(&x.common_shape(&y), c.shape())
        .expect("operands could not be broadcast together");
    #[cfg(feature = "rayon")]
    let parallel = c.is_contiguous() && x.is_contiguous() && y.is_contiguous();
    let c = c.broadcast_to(&shape);
    let x = x.broadcast_to(&shape);
    let y = y.broadcast_to(&shape);
    #[cfg(feature = "rayon")]
    if parallel && crate::layout::size(&shape) >= crate::layout::PAR_THRESHOLD {
        use rayon::prelude::*;
        return Array::from_vec(
            (0..crate::layout::size(&shape))
                .into_par_iter()
                .map(|i| {
                    if c.flat_value(i) {
                        x.flat_value(i)
                    } else {
                        y.flat_value(i)
                    }
                })
                .collect(),
            &shape,
        );
    }
    Array::from_vec(
        c.iter()
            .zip(x.iter())
            .zip(y.iter())
            .map(|((c, x), y)| if c { x } else { y })
            .collect(),
        &shape,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nan_propagation() {
        assert!(maximum(f64::NAN, 1.).item().is_nan());
        assert!(minimum(1., f64::NAN).item().is_nan());
        assert!(crate::array![f64::NAN].sign().item().is_nan());
    }
    #[test]
    fn ties_even_decimals() {
        assert_eq!(
            crate::array![1.25, 1.35, 2.5].round_decimals(1).to_vec(),
            [1.2, 1.4, 2.5]
        );
    }
    #[test]
    fn stable_sigmoid() {
        assert_eq!(
            crate::array![-1000., 0., 1000.].sigmoid().to_vec(),
            [0., 0.5, 1.]
        );
    }
}
