//! Masked arrays, like `numpy.ma`.
//!
//! A [`MaskedArray`] couples a data array with a boolean mask (`true` = masked / invalid).
//! Arrays without any masked value carry no mask at all (`nomask`), so unmasked work costs the
//! same as plain arrays. Arithmetic propagates masks, domain errors (division by zero, `sqrt` of a
//! negative number, ...) become masked values instead of NaN or panics, and reductions ignore
//! masked values.
//!
//! ```
//! use rustnum as np;
//! use np::ma;
//! let a = ma::masked_array(np::array![1.0, 2.0, 3.0, 4.0], Some(np::array![false, true, false, false]));
//! assert_eq!(a.sum(), Some(8.0));
//! assert_eq!(a.mean(), Some(8.0 / 3.0));
//! assert_eq!(a.to_string(), "[1.0 -- 3.0 4.0]");
//! let b = ma::masked_greater(&np::array![1, 5, 2, 9], 4);
//! assert_eq!(b.compressed().to_vec(), [1, 2]);
//! ```

use crate::broadcast::broadcast_shapes;
use crate::element::{add, mul, sub};
use crate::slicing::axis_index;
use crate::{Array, Complex, Element, Float, Num, SliceArg};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The value `filled` uses by default for masked entries, like `np.ma.default_fill_value`.
///
/// `1e20` for floats, `999999` for integers, `True` for booleans.
///
/// ```
/// use rustnum::ma;
/// assert_eq!(ma::default_fill_value::<f64>(), 1e20);
/// assert_eq!(ma::default_fill_value::<i32>(), 999999);
/// assert_eq!(ma::default_fill_value::<bool>(), true);
/// ```
pub fn default_fill_value<T: Element>() -> T {
    let d = T::DTYPE;
    if d.starts_with("float") || d.starts_with("bfloat") || d.starts_with("complex") {
        T::from_f64(1e20)
    } else if d == "bool" {
        T::from_f64(1.0)
    } else if d.starts_with("int") || d.starts_with("uint") {
        T::from_f64(999_999.0)
    } else {
        T::default()
    }
}

/// An array with a mask, like `numpy.ma.MaskedArray`.
///
/// The mask is `None` (`nomask`) or a boolean array of the data's shape. Construct one with
/// [`masked_array`], [`masked_where`] and friends.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_array(np::array![[1, 2], [3, 4]], Some(np::array![[false, true], [false, false]]));
/// assert_eq!(a.shape(), [2, 2]);
/// assert_eq!(a.count(), 3);
/// assert_eq!(a.to_string(), "[[1 --]\n [3 4]]");
/// ```
#[derive(Clone)]
pub struct MaskedArray<T: Element> {
    data: Array<T>,
    mask: Option<Array<bool>>,
    fill_value: T,
}

/// Anything that can be seen as a masked array: plain arrays (no mask) and masked arrays.
///
/// ```
/// use rustnum as np;
/// use np::ma::IntoMasked;
/// let m = np::array![1.0, 2.0].into_masked();
/// assert!(!m.is_masked());
/// ```
pub trait IntoMasked<T: Element> {
    /// Converts to a masked array without a mask (or keeps the existing one).
    /// ```
    /// use rustnum as np;
    /// use np::ma::IntoMasked;
    /// assert_eq!((&np::array![1, 2]).into_masked().count(), 2);
    /// ```
    fn into_masked(self) -> MaskedArray<T>;
}
impl<T: Element> IntoMasked<T> for Array<T> {
    fn into_masked(self) -> MaskedArray<T> {
        masked_array(self, None)
    }
}
impl<T: Element> IntoMasked<T> for &Array<T> {
    fn into_masked(self) -> MaskedArray<T> {
        masked_array(self.clone(), None)
    }
}
impl<T: Element> IntoMasked<T> for MaskedArray<T> {
    fn into_masked(self) -> MaskedArray<T> {
        self
    }
}
impl<T: Element> IntoMasked<T> for &MaskedArray<T> {
    fn into_masked(self) -> MaskedArray<T> {
        self.clone()
    }
}

/// Drops an all-`false` mask, like NumPy's `shrink`.
fn shrink(mask: Array<bool>) -> Option<Array<bool>> {
    if mask.iter().any(|b| b) {
        Some(mask)
    } else {
        None
    }
}

/// Builds a masked array from data and an optional mask, like `np.ma.masked_array(data, mask)`.
///
/// `None` means `nomask`. The mask must have the data's shape (panics otherwise, like NumPy's
/// `MaskError`). The fill value is [`default_fill_value`].
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_array(np::array![1, 2, 3], Some(np::array![false, true, false]));
/// assert_eq!(a.count(), 2);
/// assert_eq!(np::ma::getmask(&a).to_vec(), [false, true, false]);
/// ```
pub fn masked_array<T: Element>(data: Array<T>, mask: Option<Array<bool>>) -> MaskedArray<T> {
    if let Some(m) = &mask {
        assert!(
            m.shape() == data.shape(),
            "Mask and data not compatible: data size is {}, mask size is {}.",
            data.size(),
            m.size()
        );
    }
    MaskedArray {
        data,
        mask,
        fill_value: default_fill_value(),
    }
}

/// Masks `a` where `condition` is true, like `np.ma.masked_where(condition, a)`.
///
/// An existing mask on `a` is kept (or-ed in). Panics when the shapes differ.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_where(&np::array![true, false, true], &np::array![1, 2, 3]);
/// assert_eq!(a.compressed().to_vec(), [2]);
/// ```
pub fn masked_where<T: Element>(condition: &Array<bool>, a: impl IntoMasked<T>) -> MaskedArray<T> {
    let a = a.into_masked();
    assert!(
        condition.shape() == a.data.shape(),
        "Inconsistent shape between the condition and the input (got {:?} and {:?})",
        condition.shape(),
        a.data.shape()
    );
    let mask = match &a.mask {
        Some(m) => m | condition,
        None => condition.clone(),
    };
    MaskedArray {
        mask: shrink(mask),
        ..a
    }
}

macro_rules! where_fn {
    ($(#[$m:meta])* $name:ident, $op:tt) => {
        $(#[$m])*
        pub fn $name<T: Element>(x: impl IntoMasked<T>, value: T) -> MaskedArray<T> {
            let x = x.into_masked();
            let cond = x.data.map(|v| v $op value);
            masked_where(&cond, x)
        }
    };
}
where_fn!(
    /// Masks where `x > value`, like `np.ma.masked_greater(x, value)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5, 2, 9], 4);
    /// assert_eq!(a.compressed().to_vec(), [1, 2]);
    /// ```
    masked_greater, >
);
where_fn!(
    /// Masks where `x >= value`, like `np.ma.masked_greater_equal(x, value)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater_equal(&np::array![1, 4, 5], 4);
    /// assert_eq!(a.compressed().to_vec(), [1]);
    /// ```
    masked_greater_equal, >=
);
where_fn!(
    /// Masks where `x < value`, like `np.ma.masked_less(x, value)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_less(&np::array![1, 5, 2, 9], 4);
    /// assert_eq!(a.compressed().to_vec(), [5, 9]);
    /// ```
    masked_less, <
);
where_fn!(
    /// Masks where `x <= value`, like `np.ma.masked_less_equal(x, value)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_less_equal(&np::array![1, 4, 5], 4);
    /// assert_eq!(a.compressed().to_vec(), [5]);
    /// ```
    masked_less_equal, <=
);
where_fn!(
    /// Masks where `x != value`, like `np.ma.masked_not_equal(x, value)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_not_equal(&np::array![1, 4, 1], 1);
    /// assert_eq!(a.compressed().to_vec(), [1, 1]);
    /// ```
    masked_not_equal, !=
);

/// Masks where `x == value` and sets the fill value to `value`, like `np.ma.masked_equal`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_equal(&np::array![1, 2, 1, 3], 1);
/// assert_eq!(a.compressed().to_vec(), [2, 3]);
/// assert_eq!(a.fill_value(), 1);
/// ```
pub fn masked_equal<T: Element>(x: impl IntoMasked<T>, value: T) -> MaskedArray<T> {
    let x = x.into_masked();
    let cond = x.data.map(|v| v == value);
    let mut r = masked_where(&cond, x);
    r.fill_value = value;
    r
}

/// Masks the closed interval `[v1, v2]` (order irrelevant), like `np.ma.masked_inside`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_inside(&np::array![0.0, 1.0, 2.0, 3.0], 1.0, 2.0);
/// assert_eq!(a.compressed().to_vec(), [0.0, 3.0]);
/// ```
pub fn masked_inside<T: Element>(x: impl IntoMasked<T>, v1: T, v2: T) -> MaskedArray<T> {
    let (lo, hi) = if v2 < v1 { (v2, v1) } else { (v1, v2) };
    let x = x.into_masked();
    let cond = x.data.map(|v| v >= lo && v <= hi);
    masked_where(&cond, x)
}

/// Masks everything outside the closed interval `[v1, v2]`, like `np.ma.masked_outside`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_outside(&np::array![0.0, 1.0, 2.0, 3.0], 1.0, 2.0);
/// assert_eq!(a.compressed().to_vec(), [1.0, 2.0]);
/// ```
pub fn masked_outside<T: Element>(x: impl IntoMasked<T>, v1: T, v2: T) -> MaskedArray<T> {
    let (lo, hi) = if v2 < v1 { (v2, v1) } else { (v1, v2) };
    let x = x.into_masked();
    let cond = x.data.map(|v| v < lo || v > hi);
    masked_where(&cond, x)
}

/// Masks values close to `value`, like `np.ma.masked_values(x, value, rtol, atol)`.
///
/// Uses `isclose` semantics and sets the fill value to `value`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_values(&np::array![1.0, 1.1, 0.9999999999999], 1.0, 1e-5, 1e-8);
/// assert_eq!(a.compressed().to_vec(), [1.1]);
/// ```
pub fn masked_values<F: Float>(
    x: impl IntoMasked<F>,
    value: F,
    rtol: F,
    atol: F,
) -> MaskedArray<F> {
    let x = x.into_masked();
    let cond = x
        .data
        .map(|v| (v - value).abs() <= atol + rtol * value.abs());
    let mut r = masked_where(&cond, x);
    r.fill_value = value;
    r
}

/// Masks NaN and infinite values, like `np.ma.masked_invalid(x)`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_invalid(&np::array![1.0, f64::NAN, f64::INFINITY, 4.0]);
/// assert_eq!(a.compressed().to_vec(), [1.0, 4.0]);
/// ```
pub fn masked_invalid<F: Float>(x: impl IntoMasked<F>) -> MaskedArray<F> {
    let x = x.into_masked();
    let cond = x.data.map(|v| !v.is_finite());
    masked_where(&cond, x)
}

/// The data array (masked entries included), like `np.ma.getdata(a)`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_greater(&np::array![1, 5], 4);
/// assert_eq!(np::ma::getdata(&a).to_vec(), [1, 5]);
/// ```
pub fn getdata<T: Element>(a: &MaskedArray<T>) -> Array<T> {
    a.data.clone()
}

/// The mask as a full boolean array, like `np.ma.getmaskarray(a)`.
///
/// `np.ma.getmask` returns the `nomask` singleton for unmasked arrays; here an unmasked array
/// gives all-`false`. Use [`MaskedArray::mask`] to distinguish `nomask`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_array(np::array![1, 2], None);
/// assert_eq!(np::ma::getmask(&a).to_vec(), [false, false]);
/// ```
pub fn getmask<T: Element>(a: &MaskedArray<T>) -> Array<bool> {
    a.mask_array()
}

/// Fills masked entries with `fill_value`, like `np.ma.filled(a, fill_value)`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_greater(&np::array![1, 5, 2], 4);
/// assert_eq!(np::ma::filled(&a, -1).to_vec(), [1, -1, 2]);
/// ```
pub fn filled<T: Element>(a: &MaskedArray<T>, fill_value: T) -> Array<T> {
    a.filled(fill_value)
}

/// The unmasked values as a 1-d array, like `np.ma.compressed(a)`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_less(&np::array![[1, 5], [7, 2]], 3);
/// assert_eq!(np::ma::compressed(&a).to_vec(), [5, 7]);
/// ```
pub fn compressed<T: Element>(a: &MaskedArray<T>) -> Array<T> {
    a.compressed()
}

/// Number of unmasked elements, like `np.ma.count(a)`.
///
/// ```
/// use rustnum as np;
/// let a = np::ma::masked_equal(&np::array![1, 2, 1], 1);
/// assert_eq!(np::ma::count(&a), 1);
/// ```
pub fn count<T: Element>(a: &MaskedArray<T>) -> usize {
    a.count()
}

/// True when at least one element is masked, like `np.ma.is_masked(a)`.
///
/// ```
/// use rustnum as np;
/// assert!(np::ma::is_masked(&np::ma::masked_greater(&np::array![1, 5], 4)));
/// assert!(!np::ma::is_masked(&np::ma::masked_array(np::array![1, 5], None)));
/// ```
pub fn is_masked<T: Element>(a: &MaskedArray<T>) -> bool {
    a.is_masked()
}

impl<T: Element> MaskedArray<T> {
    fn mask_array(&self) -> Array<bool> {
        match &self.mask {
            Some(m) => m.clone(),
            None => crate::zeros(self.data.shape()),
        }
    }

    /// The data array including masked entries, like `a.data`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![1, 2], Some(np::array![true, false]));
    /// assert_eq!(a.data().to_vec(), [1, 2]);
    /// ```
    pub fn data(&self) -> &Array<T> {
        &self.data
    }

    /// The mask, `None` for `nomask`, like `a.mask` (which is `np.ma.nomask` when unmasked).
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![1, 2], None);
    /// assert!(a.mask().is_none());
    /// ```
    pub fn mask(&self) -> Option<&Array<bool>> {
        self.mask.as_ref()
    }

    /// Replaces the mask (`None` clears it), like `a.mask = mask`.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut a = np::ma::masked_array(np::array![1, 2], None);
    /// a.set_mask(Some(np::array![false, true]));
    /// assert_eq!(a.count(), 1);
    /// ```
    pub fn set_mask(&mut self, mask: Option<Array<bool>>) {
        if let Some(m) = &mask {
            assert!(
                m.shape() == self.data.shape(),
                "Mask and data not compatible: data size is {}, mask size is {}.",
                self.data.size(),
                m.size()
            );
        }
        self.mask = mask.and_then(shrink);
    }

    /// The value used by `filled` for masked entries, like `a.fill_value`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::ma::masked_array(np::array![1.0], None).fill_value(), 1e20);
    /// ```
    pub fn fill_value(&self) -> T {
        self.fill_value
    }

    /// Sets the fill value, like `a.fill_value = v`.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut a = np::ma::masked_array(np::array![1.0], None);
    /// a.set_fill_value(-1.0);
    /// assert_eq!(a.fill_value(), -1.0);
    /// ```
    pub fn set_fill_value(&mut self, value: T) {
        self.fill_value = value;
    }

    /// True when at least one element is masked.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::ma::masked_greater(&np::array![1, 5], 4).is_masked());
    /// ```
    pub fn is_masked(&self) -> bool {
        self.mask.is_some()
    }

    /// Shape, like `a.shape`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::ma::masked_array(np::array![[1, 2]], None).shape(), [1, 2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Number of dimensions, like `a.ndim`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::ma::masked_array(np::array![[1, 2]], None).ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Total number of elements, masked or not, like `a.size`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::ma::masked_greater(&np::array![1, 5], 4).size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Dtype name, like `a.dtype`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::ma::masked_array(np::array![1.0], None).dtype(), "float64");
    /// ```
    pub fn dtype(&self) -> &'static str {
        T::DTYPE
    }

    /// Number of unmasked elements, like `a.count()`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::ma::masked_greater(&np::array![1, 5], 4).count(), 1);
    /// ```
    pub fn count(&self) -> usize {
        match &self.mask {
            Some(m) => m.iter().filter(|b| !b).count(),
            None => self.data.size(),
        }
    }

    /// Unmasked count along `axis`, like `a.count(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[1, 5], [2, 9]], 4);
    /// assert_eq!(a.count_axis(1).to_vec(), [1, 1]);
    /// ```
    pub fn count_axis(&self, axis: isize) -> Array<usize> {
        let ax = axis_index(axis, self.ndim());
        match &self.mask {
            Some(m) => (!m).astype::<usize>().sum_axis(ax as isize),
            None => crate::full(
                {
                    let mut s = self.data.shape().to_vec();
                    s.remove(ax);
                    s
                },
                self.data.shape()[ax],
            ),
        }
    }

    /// The array with masked entries replaced by `fill_value`, like `a.filled(fill_value)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1.0, 5.0], 4.0);
    /// assert_eq!(a.filled(0.0).to_vec(), [1.0, 0.0]);
    /// ```
    pub fn filled(&self, fill_value: T) -> Array<T> {
        match &self.mask {
            Some(m) => crate::where_(m, Array::scalar(fill_value), &self.data),
            None => self.data.clone(),
        }
    }

    /// The unmasked values as a 1-d array in C order, like `a.compressed()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5, 2], 4);
    /// assert_eq!(a.compressed().to_vec(), [1, 2]);
    /// ```
    pub fn compressed(&self) -> Array<T> {
        match &self.mask {
            Some(m) => self.data.mask(&m.map(|b| !b)),
            None => self.data.ravel(),
        }
    }

    /// The element at `index`, `None` when it is masked, like `a[index]`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5], 4);
    /// assert_eq!(a.get(&[0]), Some(1));
    /// assert_eq!(a.get(&[1]), None);
    /// ```
    pub fn get(&self, index: &[usize]) -> Option<T> {
        match &self.mask {
            Some(m) if m.get(index) => None,
            _ => Some(self.data.get(index)),
        }
    }

    /// Masks the element at `index`, like `a[index] = np.ma.masked`.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut a = np::ma::masked_array(np::array![1, 2], None);
    /// a.set_masked(&[1]);
    /// assert_eq!(a.count(), 1);
    /// ```
    pub fn set_masked(&mut self, index: &[usize]) {
        let mut m = self.mask_array();
        m.set(index, true);
        self.mask = Some(m);
    }

    /// Stores `value` at `index` and unmasks it, like `a[index] = value`.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut a = np::ma::masked_greater(&np::array![1, 5], 4);
    /// a.set(&[1], 3);
    /// assert_eq!(a.count(), 2);
    /// ```
    pub fn set(&mut self, index: &[usize], value: T) {
        self.data.set(index, value);
        if let Some(m) = &mut self.mask {
            m.set(index, false);
            if !m.iter().any(|b| b) {
                self.mask = None;
            }
        }
    }

    /// Converts the element type, keeping the mask, like `a.astype(dtype)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5], 4).astype::<f64>();
    /// assert_eq!(a.compressed().to_vec(), [1.0]);
    /// ```
    pub fn astype<U: Element>(&self) -> MaskedArray<U> {
        MaskedArray {
            data: self.data.astype::<U>(),
            mask: self.mask.clone(),
            fill_value: {
                let (re, im) = self.fill_value.to_complex_parts();
                U::from_complex_parts(re, im)
            },
        }
    }

    /// Basic slice of data and mask together, like `a[...]` with slices.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5, 2, 3], 4);
    /// assert_eq!(a.slice(&np::s![1..3]).to_string(), "[-- 2]");
    /// ```
    pub fn slice(&self, args: &[SliceArg]) -> MaskedArray<T> {
        let mask = self.mask.as_ref().map(|m| m.slice(args));
        MaskedArray {
            data: self.data.slice(args),
            mask: mask.and_then(shrink),
            fill_value: self.fill_value,
        }
    }

    /// Reshapes data and mask, like `a.reshape(shape)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5, 2, 3], 4).reshape(&[2, 2]);
    /// assert_eq!(a.shape(), [2, 2]);
    /// ```
    pub fn reshape(&self, shape: &[usize]) -> MaskedArray<T> {
        MaskedArray {
            data: self.data.reshape(shape),
            mask: self.mask.as_ref().map(|m| m.reshape(shape)),
            fill_value: self.fill_value,
        }
    }

    /// Transposes data and mask, like `a.T`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![[1, 2, 3]], None).t();
    /// assert_eq!(a.shape(), [3, 1]);
    /// ```
    pub fn t(&self) -> MaskedArray<T> {
        MaskedArray {
            data: self.data.t(),
            mask: self.mask.as_ref().map(Array::t),
            fill_value: self.fill_value,
        }
    }

    /// Flattens to 1-d, like `a.ravel()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![[1, 2], [3, 4]], None).ravel();
    /// assert_eq!(a.shape(), [4]);
    /// ```
    pub fn ravel(&self) -> MaskedArray<T> {
        MaskedArray {
            data: self.data.ravel(),
            mask: self.mask.as_ref().map(Array::ravel),
            fill_value: self.fill_value,
        }
    }

    /// Copies data and mask, like `a.copy()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![1], None);
    /// assert_eq!(a.copy().count(), 1);
    /// ```
    pub fn copy(&self) -> MaskedArray<T> {
        MaskedArray {
            data: self.data.copy(),
            mask: self.mask.as_ref().map(Array::copy),
            fill_value: self.fill_value,
        }
    }

    /// Masked entries of a lane-wise reduction: all-masked along `axis`.
    fn lane_mask(&self, axis: isize) -> Option<Array<bool>> {
        self.mask.as_ref().and_then(|m| shrink(m.all_axis(axis)))
    }
}

// ---------------------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------------------

impl<T: Num> MaskedArray<T> {
    /// Sum of the unmasked values, `None` when everything is masked, like `a.sum()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5, 2], 4);
    /// assert_eq!(a.sum(), Some(3));
    /// assert_eq!(np::ma::masked_greater(&np::array![5], 4).sum(), None);
    /// ```
    pub fn sum(&self) -> Option<T> {
        (self.count() > 0).then(|| self.filled(T::ZERO).sum())
    }

    /// Sum along an axis; lanes that are entirely masked stay masked, like `a.sum(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[1, 5], [2, 3]], 4);
    /// assert_eq!(a.sum_axis(1).to_vec_filled(-1), [1, 5]);
    /// ```
    pub fn sum_axis(&self, axis: isize) -> MaskedArray<T> {
        MaskedArray {
            data: self.filled(T::ZERO).sum_axis(axis),
            mask: self.lane_mask(axis),
            fill_value: self.fill_value,
        }
    }

    /// Product of the unmasked values, like `a.prod()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![2, 5, 3], 4);
    /// assert_eq!(a.prod(), Some(6));
    /// ```
    pub fn prod(&self) -> Option<T> {
        (self.count() > 0).then(|| self.filled(T::ONE).prod())
    }

    /// Product along an axis; lanes that are entirely masked stay masked, like `a.prod(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[2, 5], [3, 4]], 4);
    /// assert_eq!(a.prod_axis(1).to_vec_filled(-1), [2, 12]);
    /// ```
    pub fn prod_axis(&self, axis: isize) -> MaskedArray<T> {
        MaskedArray {
            data: self.filled(T::ONE).prod_axis(axis),
            mask: self.lane_mask(axis),
            fill_value: self.fill_value,
        }
    }

    /// Smallest unmasked value, `None` when everything is masked, like `a.min()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_less(&np::array![1, 5, 2], 2);
    /// assert_eq!(a.min(), Some(2));
    /// ```
    pub fn min(&self) -> Option<T> {
        (self.count() > 0).then(|| self.filled(T::from_f64(f64::INFINITY)).min())
    }

    /// Largest unmasked value, `None` when everything is masked, like `a.max()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5, 2], 4);
    /// assert_eq!(a.max(), Some(2));
    /// ```
    pub fn max(&self) -> Option<T> {
        (self.count() > 0).then(|| self.filled(T::from_f64(f64::NEG_INFINITY)).max())
    }

    /// Minimum along an axis, like `a.min(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_less(&np::array![[1, 5], [7, 2]], 3);
    /// assert_eq!(a.min_axis(1).to_vec_filled(-1), [5, 7]);
    /// ```
    pub fn min_axis(&self, axis: isize) -> MaskedArray<T> {
        MaskedArray {
            data: self.filled(T::from_f64(f64::INFINITY)).min_axis(axis),
            mask: self.lane_mask(axis),
            fill_value: self.fill_value,
        }
    }

    /// Maximum along an axis, like `a.max(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[1, 5], [7, 2]], 4);
    /// assert_eq!(a.max_axis(1).to_vec_filled(-1), [1, 2]);
    /// ```
    pub fn max_axis(&self, axis: isize) -> MaskedArray<T> {
        MaskedArray {
            data: self.filled(T::from_f64(f64::NEG_INFINITY)).max_axis(axis),
            mask: self.lane_mask(axis),
            fill_value: self.fill_value,
        }
    }

    /// Flat index of the smallest unmasked value, like `a.argmin()`.
    ///
    /// Masked entries are filled with the largest value first, so a fully masked array gives 0.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_less(&np::array![1, 5, 2], 2);
    /// assert_eq!(a.argmin(), 2);
    /// ```
    pub fn argmin(&self) -> usize {
        self.filled(T::from_f64(f64::INFINITY)).argmin()
    }

    /// Flat index of the largest unmasked value, like `a.argmax()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 9, 2], 4);
    /// assert_eq!(a.argmax(), 2);
    /// ```
    pub fn argmax(&self) -> usize {
        self.filled(T::from_f64(f64::NEG_INFINITY)).argmax()
    }

    /// Index of the smallest unmasked value along an axis, like `a.argmin(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_less(&np::array![[1, 5, 2]], 2);
    /// assert_eq!(a.argmin_axis(1).to_vec(), [2]);
    /// ```
    pub fn argmin_axis(&self, axis: isize) -> Array<usize> {
        self.filled(T::from_f64(f64::INFINITY)).argmin_axis(axis)
    }

    /// Index of the largest unmasked value along an axis, like `a.argmax(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[1, 9, 2]], 4);
    /// assert_eq!(a.argmax_axis(1).to_vec(), [2]);
    /// ```
    pub fn argmax_axis(&self, axis: isize) -> Array<usize> {
        self.filled(T::from_f64(f64::NEG_INFINITY))
            .argmax_axis(axis)
    }

    /// The array with masked entries replaced by `fill`, flattened, for compact comparisons.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1, 5], 4);
    /// assert_eq!(a.to_vec_filled(0), [1, 0]);
    /// ```
    pub fn to_vec_filled(&self, fill: T) -> Vec<T> {
        self.filled(fill).to_vec()
    }
}

impl<F: Float> MaskedArray<F> {
    /// Mean of the unmasked values, `None` when everything is masked, like `a.mean()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1.0, 5.0, 2.0], 4.0);
    /// assert_eq!(a.mean(), Some(1.5));
    /// ```
    pub fn mean(&self) -> Option<F> {
        let n = self.count();
        (n > 0).then(|| self.filled(F::ZERO).sum() * F::ONE / F::from_usize(n))
    }

    /// Mean along an axis, like `a.mean(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[1.0, 5.0], [2.0, 3.0]], 4.0);
    /// assert_eq!(a.mean_axis(1).to_vec_filled(-1.0), [1.0, 2.5]);
    /// ```
    pub fn mean_axis(&self, axis: isize) -> MaskedArray<F> {
        let sum = self.filled(F::ZERO).sum_axis(axis);
        let cnt = self.count_axis(axis).astype::<F>();
        let data = (&sum * F::ONE).par_zip(&cnt, |s, c| if c == F::ZERO { s } else { s / c });
        MaskedArray {
            data,
            mask: self.lane_mask(axis),
            fill_value: self.fill_value,
        }
    }

    /// Variance of the unmasked values, `None` when everything is masked, like `a.var()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1.0, 3.0, 9.0], 4.0);
    /// assert_eq!(a.var(), Some(1.0));
    /// ```
    pub fn var(&self) -> Option<F> {
        self.var_ddof(0)
    }

    /// Variance with `ddof` degrees of freedom removed, like `a.var(ddof=ddof)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1.0, 3.0, 9.0], 4.0);
    /// assert_eq!(a.var_ddof(1), Some(2.0));
    /// ```
    pub fn var_ddof(&self, ddof: usize) -> Option<F> {
        let n = self.count();
        if n == 0 || n <= ddof {
            return None;
        }
        let mean = self.mean()?;
        let danom = self.data.map(|v| (v - mean) * (v - mean));
        let danom = match &self.mask {
            Some(m) => crate::where_(m, Array::scalar(F::ZERO), &danom),
            None => danom,
        };
        Some(danom.sum() / F::from_usize(n - ddof))
    }

    /// Standard deviation of the unmasked values, like `a.std()`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1.0, 3.0, 9.0], 4.0);
    /// assert_eq!(a.std(), Some(1.0));
    /// ```
    pub fn std(&self) -> Option<F> {
        self.var().map(Float::sqrt)
    }

    /// Standard deviation with `ddof` degrees of freedom removed, like `a.std(ddof=ddof)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![1.0, 3.0, 9.0], 4.0);
    /// assert_eq!(a.std_ddof(1), Some(2.0f64.sqrt()));
    /// ```
    pub fn std_ddof(&self, ddof: usize) -> Option<F> {
        self.var_ddof(ddof).map(Float::sqrt)
    }

    /// Variance along an axis, like `a.var(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[1.0, 3.0], [2.0, 9.0]], 4.0);
    /// assert_eq!(a.var_axis(1).to_vec_filled(-1.0)[0], 1.0);
    /// ```
    pub fn var_axis(&self, axis: isize) -> MaskedArray<F> {
        self.var_ddof_axis(axis, 0)
    }

    /// Variance along an axis with `ddof` removed, like `a.var(axis=axis, ddof=ddof)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[1.0, 3.0], [2.0, 9.0]], 4.0);
    /// assert_eq!(a.var_ddof_axis(1, 1).to_vec_filled(-1.0)[0], 2.0);
    /// ```
    pub fn var_ddof_axis(&self, axis: isize, ddof: usize) -> MaskedArray<F> {
        let ax = axis_index(axis, self.ndim());
        let mean = self.mean_axis(axis);
        let mean_b = mean.data.expand_dims(ax as isize);
        let danom = (&self.data - &mean_b).square();
        let danom = match &self.mask {
            Some(m) => crate::where_(m, Array::scalar(F::ZERO), &danom),
            None => danom,
        };
        let cnt = self.count_axis(axis);
        let sum = danom.sum_axis(axis);
        let data = sum.par_zip(&cnt, |s, c| {
            let d = F::from_usize(c.saturating_sub(ddof));
            if d == F::ZERO { s } else { s / d }
        });
        let invalid = cnt.map(|c| c <= ddof);
        let mut mask = self.mask.as_ref().map(|m| m.all_axis(axis));
        let bad = invalid.iter().any(|b| b);
        if bad {
            mask = Some(match mask {
                Some(m) => &m | &invalid,
                None => invalid,
            });
        }
        MaskedArray {
            data,
            mask: mask.and_then(shrink),
            fill_value: self.fill_value,
        }
    }

    /// Standard deviation along an axis, like `a.std(axis=axis)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_greater(&np::array![[1.0, 3.0], [2.0, 9.0]], 4.0);
    /// assert_eq!(a.std_axis(1).to_vec_filled(-1.0)[0], 1.0);
    /// ```
    pub fn std_axis(&self, axis: isize) -> MaskedArray<F> {
        let v = self.var_axis(axis);
        MaskedArray {
            data: v.data.sqrt(),
            mask: v.mask,
            fill_value: v.fill_value,
        }
    }
}

// ---------------------------------------------------------------------------------------
// Arithmetic with mask propagation
// ---------------------------------------------------------------------------------------

fn finite<T: Element>(x: T) -> bool {
    let (re, im) = x.to_complex_parts();
    re.is_finite() && im.is_finite()
}

fn combined_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    broadcast_shapes(a, b).unwrap_or_else(|| {
        panic!("operands could not be broadcast together with shapes {a:?} {b:?}")
    })
}

/// Element-wise binary operation: masks propagate, `domain` (when given) masks invalid operand
/// pairs and non-finite results; masked positions keep the first operand's data, like NumPy.
fn binary<T: Num>(
    a: &MaskedArray<T>,
    b: &MaskedArray<T>,
    f: impl Fn(T, T) -> T,
    domain: Option<&dyn Fn(T, T) -> bool>,
) -> MaskedArray<T> {
    let shape = combined_shape(a.data.shape(), b.data.shape());
    let da = a.data.broadcast_to(&shape).to_vec();
    let db = b.data.broadcast_to(&shape).to_vec();
    let ma = a.mask.as_ref().map(|m| m.broadcast_to(&shape).to_vec());
    let mb = b.mask.as_ref().map(|m| m.broadcast_to(&shape).to_vec());
    let mut mask = vec![false; da.len()];
    let mut data = Vec::with_capacity(da.len());
    for i in 0..da.len() {
        let mut m = ma.as_ref().is_some_and(|v| v[i]) || mb.as_ref().is_some_and(|v| v[i]);
        let mut value = da[i];
        if !m {
            if let Some(d) = domain {
                if d(da[i], db[i]) {
                    m = true;
                } else {
                    value = f(da[i], db[i]);
                    if !finite(value) {
                        m = true;
                        value = da[i];
                    }
                }
            } else {
                value = f(da[i], db[i]);
            }
        }
        mask[i] = m;
        data.push(value);
    }
    let mask = shrink(Array::from_vec(mask, &shape));
    MaskedArray {
        data: Array::from_vec(data, &shape),
        mask,
        fill_value: a.fill_value,
    }
}

fn safe_divide<T: Num>(a: T, b: T) -> bool {
    a.abs() * T::from_f64(f64::MIN_POSITIVE) >= b.abs()
}

macro_rules! masked_binary {
    ($Tr:ident, $m:ident, $f:expr, $dom:expr) => {
        impl<T: Num> $Tr<&MaskedArray<T>> for &MaskedArray<T> {
            type Output = MaskedArray<T>;
            fn $m(self, r: &MaskedArray<T>) -> MaskedArray<T> {
                binary(self, r, $f, $dom)
            }
        }
        impl<T: Num> $Tr<MaskedArray<T>> for MaskedArray<T> {
            type Output = MaskedArray<T>;
            fn $m(self, r: MaskedArray<T>) -> MaskedArray<T> {
                binary(&self, &r, $f, $dom)
            }
        }
        impl<T: Num> $Tr<&MaskedArray<T>> for MaskedArray<T> {
            type Output = MaskedArray<T>;
            fn $m(self, r: &MaskedArray<T>) -> MaskedArray<T> {
                binary(&self, r, $f, $dom)
            }
        }
        impl<T: Num> $Tr<MaskedArray<T>> for &MaskedArray<T> {
            type Output = MaskedArray<T>;
            fn $m(self, r: MaskedArray<T>) -> MaskedArray<T> {
                binary(self, &r, $f, $dom)
            }
        }
        impl<T: Num> $Tr<T> for &MaskedArray<T> {
            type Output = MaskedArray<T>;
            fn $m(self, r: T) -> MaskedArray<T> {
                binary(self, &masked_array(Array::scalar(r), None), $f, $dom)
            }
        }
        impl<T: Num> $Tr<T> for MaskedArray<T> {
            type Output = MaskedArray<T>;
            fn $m(self, r: T) -> MaskedArray<T> {
                binary(&self, &masked_array(Array::scalar(r), None), $f, $dom)
            }
        }
        impl<T: Num> $Tr<&Array<T>> for &MaskedArray<T> {
            type Output = MaskedArray<T>;
            fn $m(self, r: &Array<T>) -> MaskedArray<T> {
                binary(self, &masked_array(r.clone(), None), $f, $dom)
            }
        }
        impl<T: Num> $Tr<&MaskedArray<T>> for &Array<T> {
            type Output = MaskedArray<T>;
            fn $m(self, r: &MaskedArray<T>) -> MaskedArray<T> {
                binary(&masked_array(self.clone(), None), r, $f, $dom)
            }
        }
    };
}
masked_binary!(Add, add, add, None);
masked_binary!(Sub, sub, sub, None);
masked_binary!(Mul, mul, mul, None);
masked_binary!(Div, div, crate::element::div, Some(&safe_divide::<T>));

macro_rules! masked_scalar_left {
    ($($t:ty),*) => {$(
        impl Add<&MaskedArray<$t>> for $t {
            type Output = MaskedArray<$t>;
            fn add(self, r: &MaskedArray<$t>) -> MaskedArray<$t> { &masked_array(Array::scalar(self), None) + r }
        }
        impl Sub<&MaskedArray<$t>> for $t {
            type Output = MaskedArray<$t>;
            fn sub(self, r: &MaskedArray<$t>) -> MaskedArray<$t> { &masked_array(Array::scalar(self), None) - r }
        }
        impl Mul<&MaskedArray<$t>> for $t {
            type Output = MaskedArray<$t>;
            fn mul(self, r: &MaskedArray<$t>) -> MaskedArray<$t> { &masked_array(Array::scalar(self), None) * r }
        }
        impl Div<&MaskedArray<$t>> for $t {
            type Output = MaskedArray<$t>;
            fn div(self, r: &MaskedArray<$t>) -> MaskedArray<$t> { &masked_array(Array::scalar(self), None) / r }
        }
    )*};
}
masked_scalar_left!(f32, f64, i32, i64);

impl<T: Num + Neg<Output = T>> Neg for &MaskedArray<T> {
    type Output = MaskedArray<T>;
    fn neg(self) -> MaskedArray<T> {
        MaskedArray {
            data: -&self.data,
            mask: self.mask.clone(),
            fill_value: self.fill_value,
        }
    }
}

impl<T: Num> MaskedArray<T> {
    /// Absolute value, like `np.ma.abs(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![-1, -2], Some(np::array![false, true]));
    /// assert_eq!(a.abs().to_vec_filled(0), [1, 0]);
    /// ```
    pub fn abs(&self) -> MaskedArray<T> {
        let abs = self.data.abs();
        // Like NumPy, the data under the mask stays as it was.
        let data = match &self.mask {
            Some(m) => crate::where_(m, &self.data, &abs),
            None => abs,
        };
        MaskedArray {
            data,
            mask: self.mask.clone(),
            fill_value: self.fill_value,
        }
    }

    /// Element-wise square, like `np.ma.square(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![3, 4], Some(np::array![false, true]));
    /// assert_eq!(a.square().to_vec_filled(0), [9, 0]);
    /// ```
    pub fn square(&self) -> MaskedArray<T> {
        MaskedArray {
            data: self.data.square(),
            mask: self.mask.clone(),
            fill_value: self.fill_value,
        }
    }
}

/// Unary function with an optional domain: masked where the input is outside the domain or the
/// result is not finite (when a domain exists); data under the mask is the input.
fn unary<F: Float>(
    a: &MaskedArray<F>,
    f: impl Fn(F) -> F,
    domain: Option<&dyn Fn(F) -> bool>,
) -> MaskedArray<F> {
    let shape = a.data.shape().to_vec();
    let values = a.data.to_vec();
    let old = a.mask.as_ref().map(Array::to_vec);
    let mut mask = vec![false; values.len()];
    let mut data = Vec::with_capacity(values.len());
    for (i, &x) in values.iter().enumerate() {
        // Masked inputs keep their data and skip `f`; with a domain, out-of-domain inputs and
        // non-finite results are masked too.
        let masked_in = old.as_ref().is_some_and(|m| m[i]);
        let (m, v) = if masked_in {
            (true, x)
        } else {
            let v = f(x);
            match domain {
                Some(d) if d(x) || !v.is_finite() => (true, x),
                _ => (false, v),
            }
        };
        mask[i] = m;
        data.push(v);
    }
    MaskedArray {
        data: Array::from_vec(data, &shape),
        mask: shrink(Array::from_vec(mask, &shape)),
        fill_value: a.fill_value,
    }
}

macro_rules! unary_methods {
    ($($(#[$m:meta])* $name:ident => $f:expr, $dom:expr;)*) => {
        impl<F: Float> MaskedArray<F> {$(
            $(#[$m])*
            pub fn $name(&self) -> MaskedArray<F> {
                unary(self, $f, $dom)
            }
        )*}
    };
}
unary_methods! {
    /// Exponential, like `np.ma.exp(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![0.0, 1.0], Some(np::array![false, true]));
    /// assert_eq!(a.exp().to_vec_filled(0.0), [1.0, 0.0]);
    /// ```
    exp => Float::exp, None;
    /// Sine, like `np.ma.sin(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![0.0], None);
    /// assert_eq!(a.sin().to_vec_filled(0.0), [0.0]);
    /// ```
    sin => Float::sin, None;
    /// Cosine, like `np.ma.cos(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![0.0], None);
    /// assert_eq!(a.cos().to_vec_filled(0.0), [1.0]);
    /// ```
    cos => Float::cos, None;
    /// Hyperbolic tangent, like `np.ma.tanh(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![0.0], None);
    /// assert_eq!(a.tanh().to_vec_filled(0.0), [0.0]);
    /// ```
    tanh => Float::tanh, None;
    /// Square root; negative inputs become masked, like `np.ma.sqrt(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![4.0, -1.0], None).sqrt();
    /// assert_eq!(a.compressed().to_vec(), [2.0]);
    /// ```
    sqrt => Float::sqrt, Some(&|x: F| x < F::ZERO);
    /// Natural logarithm; non-positive inputs become masked, like `np.ma.log(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![1.0, 0.0, -2.0], None).log();
    /// assert_eq!(a.compressed().to_vec(), [0.0]);
    /// ```
    log => Float::ln, Some(&|x: F| x <= F::ZERO);
    /// Base-2 logarithm; non-positive inputs become masked, like `np.ma.log2(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![8.0, 0.0], None).log2();
    /// assert_eq!(a.compressed().to_vec(), [3.0]);
    /// ```
    log2 => Float::log2, Some(&|x: F| x <= F::ZERO);
    /// Base-10 logarithm; non-positive inputs become masked, like `np.ma.log10(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![100.0, -1.0], None).log10();
    /// assert_eq!(a.compressed().to_vec(), [2.0]);
    /// ```
    log10 => Float::log10, Some(&|x: F| x <= F::ZERO);
    /// Arc sine; inputs outside `[-1, 1]` become masked, like `np.ma.arcsin(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![0.0, 2.0], None).arcsin();
    /// assert_eq!(a.compressed().to_vec(), [0.0]);
    /// ```
    arcsin => Float::asin, Some(&|x: F| x < -F::ONE || x > F::ONE);
    /// Arc cosine; inputs outside `[-1, 1]` become masked, like `np.ma.arccos(a)`.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::ma::masked_array(np::array![1.0, 2.0], None).arccos();
    /// assert_eq!(a.compressed().to_vec(), [0.0]);
    /// ```
    arccos => Float::acos, Some(&|x: F| x < -F::ONE || x > F::ONE);
}

// ---------------------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------------------

/// Python `repr(float)`: shortest round-trip digits, `1e-05`, `1e+16`, `inf`, `nan`.
fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp10: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp10) {
        let n = digits.len() as i32;
        let decpt = exp10 + 1;
        let body = if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt >= n {
            format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..decpt as usize],
                &digits[decpt as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let frac = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        format!(
            "{sign}{}{frac}e{}{:02}",
            &digits[..1],
            if exp10 < 0 { '-' } else { '+' },
            exp10.abs()
        )
    }
}

/// How NumPy prints one element of an object array built with `astype("O")`.
fn element_text<T: Element>(x: T) -> String {
    let d = T::DTYPE;
    if d.starts_with("float") || d.starts_with("bfloat") {
        py_float_repr(x.to_f64())
    } else if d == "bool" {
        if x.to_f64() != 0.0 { "True" } else { "False" }.into()
    } else if d.starts_with("complex") {
        let (re, im) = x.to_complex_parts();
        Complex::new(re, im).to_string()
    } else {
        let s = format!("{x:?}");
        s
    }
}

impl<T: Element> fmt::Display for MaskedArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mask = self.mask.as_ref().map(Array::to_vec);
        let values = self.data.to_vec();
        let text = |i: usize| {
            if mask.as_ref().is_some_and(|m| m[i]) {
                "--".to_string()
            } else {
                element_text(values[i])
            }
        };
        f.write_str(&crate::display::render_custom(self.data.shape(), &text))
    }
}

impl<T: Element> fmt::Debug for MaskedArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mask = self.mask_array();
        let fill = element_text(self.fill_value);
        write!(
            f,
            "masked_array(data={},\n             mask={},\n       fill_value={})",
            self, mask, fill
        )
    }
}
