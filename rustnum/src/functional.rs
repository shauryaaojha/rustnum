//! Functional helpers: `vectorize`, `apply_along_axis`, `apply_over_axes`.
use crate::broadcast::broadcast_shapes;
use crate::layout::size;
use crate::slicing::axis_index;
use crate::{Array, Element, IntoArray};

/// A scalar closure lifted over arrays with broadcasting, like the object np.vectorize(f) returns.
///
/// Build it with [`vectorize`] and call it with [`Vectorize::call`], [`Vectorize::call2`] or
/// [`Vectorize::call3`] depending on the closure's arity.
/// ```
/// use rustnum as np;
/// let hyp = np::vectorize(|a: f64, b: f64| (a * a + b * b).sqrt());
/// assert_eq!(hyp.call2(np::array![3.0, 5.0], 4.0).to_vec(), [5.0, 41f64.sqrt()]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Vectorize<F> {
    f: F,
}
/// Wrap a scalar closure into an array function with broadcasting, like np.vectorize(pyfunc).
///
/// The result element type may differ from the input type.
/// ```
/// use rustnum as np;
/// let sign_label = np::vectorize(|x: i32| if x < 0 { 0u8 } else { 1u8 });
/// assert_eq!(sign_label.call(np::array![-2, 0, 5]).to_vec(), [0, 1, 1]);
/// ```
pub fn vectorize<F>(pyfunc: F) -> Vectorize<F> {
    Vectorize { f: pyfunc }
}
impl<F> Vectorize<F> {
    /// Apply a one-argument closure to every element, like calling the np.vectorize object with one array.
    /// ```
    /// use rustnum as np;
    /// let f = np::vectorize(|x: i32| x * 2);
    /// assert_eq!(f.call(np::array![[1, 2], [3, 4]]).to_vec(), [2, 4, 6, 8]);
    /// ```
    pub fn call<A: Element, R: Element>(&self, a: impl IntoArray<A>) -> Array<R>
    where
        F: Fn(A) -> R,
    {
        a.into_array().map(&self.f)
    }
    /// Apply a two-argument closure with broadcasting, like calling the np.vectorize object with two arrays.
    /// ```
    /// use rustnum as np;
    /// let f = np::vectorize(|a: i32, b: i32| a.max(b));
    /// assert_eq!(f.call2(np::array![[1], [5]], np::array![2, 4]).to_vec(), [2, 4, 5, 5]);
    /// ```
    pub fn call2<A: Element, B: Element, R: Element>(
        &self,
        a: impl IntoArray<A>,
        b: impl IntoArray<B>,
    ) -> Array<R>
    where
        F: Fn(A, B) -> R,
    {
        a.into_array().zip_map(b, &self.f)
    }
    /// Apply a three-argument closure with broadcasting, like calling the np.vectorize object with three arrays.
    /// ```
    /// use rustnum as np;
    /// let f = np::vectorize(|a: i32, b: i32, c: i32| a * b + c);
    /// assert_eq!(f.call3(np::array![1, 2], 10, np::array![5, 6]).to_vec(), [15, 26]);
    /// ```
    pub fn call3<A: Element, B: Element, C: Element, R: Element>(
        &self,
        a: impl IntoArray<A>,
        b: impl IntoArray<B>,
        c: impl IntoArray<C>,
    ) -> Array<R>
    where
        F: Fn(A, B, C) -> R,
    {
        let (a, b, c) = (a.into_array(), b.into_array(), c.into_array());
        let shape = broadcast_shapes(a.shape(), b.shape())
            .and_then(|s| broadcast_shapes(&s, c.shape()))
            .unwrap_or_else(|| {
                panic!(
                    "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape {:?} and arg 1 with shape {:?}.",
                    a.shape(),
                    b.shape()
                )
            });
        let (a, b, c) = (
            a.broadcast_to(&shape),
            b.broadcast_to(&shape),
            c.broadcast_to(&shape),
        );
        Array::from_shape_fn(&shape, |i| (self.f)(a.get(i), b.get(i), c.get(i)))
    }
}

/// Apply a function to 1-d lanes along an axis, like np.apply_along_axis(func1d, axis, arr).
///
/// `func1d` receives each lane as a 1-d array and returns an array (0-d for a scalar). The result
/// has the shape of `arr` with the `axis` dimension replaced by the shape `func1d` returns.
/// Panics if `func1d` returns differently shaped results.
/// ```
/// use rustnum as np;
/// let a = np::array![[1, 2, 3], [4, 5, 6]];
/// let sums = np::apply_along_axis(|lane| np::Array::scalar(lane.sum()), 1, &a);
/// assert_eq!(sums.to_vec(), [6, 15]);
/// let sorted_desc = np::apply_along_axis(|lane| lane.sort().flip(0), 0, &a);
/// assert_eq!(sorted_desc.to_vec(), [4, 5, 6, 1, 2, 3]);
/// ```
pub fn apply_along_axis<T: Element, R: Element>(
    mut func1d: impl FnMut(&Array<T>) -> Array<R>,
    axis: isize,
    arr: &Array<T>,
) -> Array<R> {
    let ax = axis_index(axis, arr.ndim());
    let c = arr.to_contiguous();
    let src = c.as_slice().expect("contiguous");
    let n = arr.shape()[ax];
    let outer = size(&arr.shape()[..ax]);
    let inner = size(&arr.shape()[ax + 1..]);
    assert!(
        outer * inner > 0,
        "Cannot apply_along_axis when any iteration dimensions are 0"
    );
    let mut res_shape: Option<Vec<usize>> = None;
    let mut results: Vec<Vec<R>> = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for j in 0..inner {
            let lane: Array<T> = (0..n).map(|i| src[(o * n + i) * inner + j]).collect();
            let r = func1d(&lane);
            match &res_shape {
                None => res_shape = Some(r.shape().to_vec()),
                Some(s) => assert!(
                    s.as_slice() == r.shape(),
                    "could not broadcast input array from shape {:?} into shape {:?}",
                    r.shape(),
                    s
                ),
            }
            results.push(r.to_vec());
        }
    }
    let rs = res_shape.expect("at least one lane");
    let m = size(&rs);
    let mut shape = arr.shape()[..ax].to_vec();
    shape.extend_from_slice(&rs);
    shape.extend_from_slice(&arr.shape()[ax + 1..]);
    let mut out = vec![R::default(); outer * m * inner];
    for o in 0..outer {
        for j in 0..inner {
            for (k, &v) in results[o * inner + j].iter().enumerate() {
                out[(o * m + k) * inner + j] = v;
            }
        }
    }
    Array::from_vec(out, &shape)
}

/// Apply a function repeatedly over several axes, like np.apply_over_axes(func, a, axes).
///
/// `func(array, axis)` must return an array with the same number of dimensions as `a` or one fewer
/// (the removed axis is re-inserted with length 1). Panics otherwise.
/// ```
/// use rustnum as np;
/// let a = np::arange(0, 24, 1).reshape(&[2, 3, 4]);
/// let s = np::apply_over_axes(|x, ax| x.sum_axis(ax), &a, &[0, 2]);
/// assert_eq!(s.shape(), [1, 3, 1]);
/// assert_eq!(s.to_vec(), [60, 92, 124]);
/// ```
pub fn apply_over_axes<T: Element>(
    mut func: impl FnMut(&Array<T>, isize) -> Array<T>,
    a: &Array<T>,
    axes: &[isize],
) -> Array<T> {
    let mut val = a.clone();
    for &axis in axes {
        let ax = if axis < 0 {
            axis + a.ndim() as isize
        } else {
            axis
        };
        let res = func(&val, ax);
        val = if res.ndim() == val.ndim() {
            res
        } else if res.ndim() + 1 == val.ndim() {
            res.expand_dims(ax)
        } else {
            panic!("function is not returning an array of the correct shape")
        };
    }
    val
}
