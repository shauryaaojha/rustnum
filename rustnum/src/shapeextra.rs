//! Shape and creation extras: `rot90`, `dstack`, `atleast_nd`, `geomspace`, `vander`, `frombuffer`, ...
use crate::element::mul;
use crate::lanes::view;
use crate::slicing::axis_index;
use crate::{Array, Element, Float, Num, concatenate, logspace, vstack};
use std::any::{Any, TypeId};

/// Insert a length-1 axis at each position of `at` (positions in the output), as a view.
fn with_unit_axes<T: Element>(a: &Array<T>, at: &[usize]) -> Array<T> {
    let mut shape = a.shape.clone();
    let mut strides = a.strides.clone();
    for &p in at {
        shape.insert(p, 1);
        strides.insert(p, 0);
    }
    view(a, shape, strides, a.offset)
}
/// View an array with at least one dimension, like np.atleast_1d(a).
/// ```
/// use rustnum as np;
/// assert_eq!(np::atleast_1d(&np::Array::scalar(3)).shape(), [1]);
/// assert_eq!(np::atleast_1d(&np::array![1, 2]).shape(), [2]);
/// ```
pub fn atleast_1d<T: Element>(a: &Array<T>) -> Array<T> {
    if a.ndim() == 0 {
        with_unit_axes(a, &[0])
    } else {
        a.clone()
    }
}
/// View an array with at least two dimensions, like np.atleast_2d(a).
///
/// A 1-d array of length N becomes a row of shape `[1, N]`.
/// ```
/// use rustnum as np;
/// assert_eq!(np::atleast_2d(&np::Array::scalar(3)).shape(), [1, 1]);
/// assert_eq!(np::atleast_2d(&np::array![1, 2]).shape(), [1, 2]);
/// ```
pub fn atleast_2d<T: Element>(a: &Array<T>) -> Array<T> {
    match a.ndim() {
        0 => with_unit_axes(a, &[0, 0]),
        1 => with_unit_axes(a, &[0]),
        _ => a.clone(),
    }
}
/// View an array with at least three dimensions, like np.atleast_3d(a).
///
/// A 1-d array of length N becomes `[1, N, 1]`; a 2-d array `[M, N]` becomes `[M, N, 1]`.
/// ```
/// use rustnum as np;
/// assert_eq!(np::atleast_3d(&np::array![1, 2]).shape(), [1, 2, 1]);
/// assert_eq!(np::atleast_3d(&np::array![[1, 2]]).shape(), [1, 2, 1]);
/// ```
pub fn atleast_3d<T: Element>(a: &Array<T>) -> Array<T> {
    match a.ndim() {
        0 => with_unit_axes(a, &[0, 0, 0]),
        1 => with_unit_axes(a, &[0, 2]),
        2 => with_unit_axes(a, &[2]),
        _ => a.clone(),
    }
}
/// Rotate an array by 90 degrees in the plane of two axes, like np.rot90(m, k, axes).
///
/// Rotation is from the first towards the second axis; `k` may be negative. The result is a view.
/// ```
/// use rustnum as np;
/// let m = np::array![[1, 2], [3, 4]];
/// assert_eq!(np::rot90(&m, 1, (0, 1)).to_vec(), [2, 4, 1, 3]);
/// assert_eq!(np::rot90(&m, -1, (0, 1)).to_vec(), [3, 1, 4, 2]);
/// ```
pub fn rot90<T: Element>(m: &Array<T>, k: isize, axes: (isize, isize)) -> Array<T> {
    let nd = m.ndim() as isize;
    assert!(
        nd >= 2 && axes.0 < nd && axes.0 >= -nd && axes.1 < nd && axes.1 >= -nd,
        "Axes={:?} out of range for array of ndim={}.",
        axes,
        nd
    );
    let (a0, a1) = (axis_index(axes.0, m.ndim()), axis_index(axes.1, m.ndim()));
    assert!(a0 != a1, "Axes must be different.");
    let (a0, a1) = (a0 as isize, a1 as isize);
    match k.rem_euclid(4) {
        0 => m.clone(),
        1 => m.flip(a1).swapaxes(a0, a1),
        2 => m.flip(a0).flip(a1),
        _ => m.swapaxes(a0, a1).flip(a1),
    }
}
/// Stack arrays depth-wise along the third axis, like np.dstack(tup).
/// ```
/// use rustnum as np;
/// let a = np::array![1, 2];
/// let b = np::array![3, 4];
/// let s = np::dstack(&[&a, &b]);
/// assert_eq!(s.shape(), [1, 2, 2]);
/// assert_eq!(s.to_vec(), [1, 3, 2, 4]);
/// ```
pub fn dstack<T: Element>(arrays: &[&Array<T>]) -> Array<T> {
    let v: Vec<Array<T>> = arrays.iter().map(|a| atleast_3d(a)).collect();
    let r: Vec<&Array<T>> = v.iter().collect();
    concatenate(&r, 2)
}
/// Stack 1-d arrays as columns of a 2-d array, like np.column_stack(tup).
///
/// 2-d (and higher) arrays are stacked along their second axis unchanged.
/// ```
/// use rustnum as np;
/// let a = np::array![1, 2, 3];
/// let b = np::array![4, 5, 6];
/// assert_eq!(np::column_stack(&[&a, &b]).to_vec(), [1, 4, 2, 5, 3, 6]);
/// ```
pub fn column_stack<T: Element>(arrays: &[&Array<T>]) -> Array<T> {
    let v: Vec<Array<T>> = arrays
        .iter()
        .map(|a| {
            if a.ndim() < 2 {
                let n = a.size();
                a.reshape(&[n, 1])
            } else {
                (*a).clone()
            }
        })
        .collect();
    let r: Vec<&Array<T>> = v.iter().collect();
    concatenate(&r, 1)
}
/// Stack arrays vertically, like np.row_stack(tup) (NumPy's deprecated alias of np.vstack).
/// ```
/// use rustnum as np;
/// let a = np::array![1, 2];
/// let b = np::array![3, 4];
/// assert_eq!(np::row_stack(&[&a, &b]).to_vec(), [1, 2, 3, 4]);
/// ```
pub fn row_stack<T: Element>(arrays: &[&Array<T>]) -> Array<T> {
    vstack(arrays)
}
/// Numbers spaced evenly on a log scale (a geometric progression), like np.geomspace(start, stop, num).
///
/// Both endpoints are included and returned exactly. Panics with
/// "Geometric sequence cannot include zero" if either endpoint is 0.
/// ```
/// use rustnum as np;
/// let g = np::geomspace(1.0, 1000.0, 4);
/// assert_eq!(g.to_vec(), [1.0, 10.0, 100.0, 1000.0]);
/// ```
pub fn geomspace<F: Float>(start: F, stop: F, num: usize) -> Array<F> {
    assert!(
        start != F::ZERO && stop != F::ZERO,
        "Geometric sequence cannot include zero"
    );
    let neg = start < F::ZERO && stop < F::ZERO;
    let (s, e) = if neg { (-start, -stop) } else { (start, stop) };
    let mut v = logspace(s.log10(), e.log10(), num).to_vec();
    if num > 0 {
        v[0] = s;
    }
    if num > 1 {
        v[num - 1] = e;
    }
    if neg {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
    Array::from(v)
}
/// Vandermonde matrix of a 1-d array, like np.vander(x, N, increasing).
///
/// `n` defaults to `x.len()` columns. Powers decrease from left to right unless `increasing`.
/// Integer powers wrap on overflow.
/// ```
/// use rustnum as np;
/// let v = np::vander(&np::array![1, 2, 3], None, false);
/// assert_eq!(v.to_vec(), [1, 1, 1, 4, 2, 1, 9, 3, 1]);
/// ```
pub fn vander<T: Num>(x: &Array<T>, n: Option<usize>, increasing: bool) -> Array<T> {
    assert!(
        x.ndim() == 1,
        "x must be a one-dimensional array or sequence."
    );
    let cols = n.unwrap_or(x.size());
    let mut out = Vec::with_capacity(x.size() * cols);
    for xi in x.iter() {
        let mut powers = Vec::with_capacity(cols);
        let mut p = T::ONE;
        for _ in 0..cols {
            powers.push(p);
            p = mul(p, xi);
        }
        if !increasing {
            powers.reverse();
        }
        out.extend(powers);
    }
    Array::from_vec(out, &[x.size(), cols])
}
/// Build a 1-d array from an iterator, like np.fromiter(iter, dtype, count).
///
/// With `count` set only that many items are read; panics if the iterator is shorter.
/// ```
/// use rustnum as np;
/// assert_eq!(np::fromiter((0..5).map(|i| i * i), None).to_vec(), [0, 1, 4, 9, 16]);
/// assert_eq!(np::fromiter(0..10, Some(3)).to_vec(), [0, 1, 2]);
/// ```
pub fn fromiter<T: Element>(iter: impl IntoIterator<Item = T>, count: Option<usize>) -> Array<T> {
    let v: Vec<T> = match count {
        None => iter.into_iter().collect(),
        Some(c) => {
            let v: Vec<T> = iter.into_iter().take(c).collect();
            assert!(
                v.len() == c,
                "iterator too short: Expected {} but iterator had only {} items.",
                c,
                v.len()
            );
            v
        }
    };
    Array::from(v)
}
fn decode<T: Element>(bytes: &[u8]) -> Vec<T> {
    macro_rules! go {
        ($ty:ty) => {
            if TypeId::of::<T>() == TypeId::of::<$ty>() {
                let v: Vec<$ty> = bytes
                    .chunks_exact(std::mem::size_of::<$ty>())
                    .map(|c| <$ty>::from_ne_bytes(c.try_into().expect("chunk size")))
                    .collect();
                return *(Box::new(v) as Box<dyn Any>)
                    .downcast::<Vec<T>>()
                    .expect("type checked");
            }
        };
    }
    go!(f32);
    go!(f64);
    go!(i32);
    go!(i64);
    go!(u8);
    go!(usize);
    let v: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
    *(Box::new(v) as Box<dyn Any>)
        .downcast::<Vec<T>>()
        .expect("bool is the last element type")
}
/// Interpret raw bytes as a 1-d array, like np.frombuffer(buffer, dtype, count, offset).
///
/// The element type is the type parameter and bytes are read in native byte order. The data is
/// copied. `count` `None` reads to the end.
/// ```
/// use rustnum as np;
/// let bytes = [1u8, 0, 0, 0, 2, 0, 0, 0];
/// assert_eq!(np::frombuffer::<i32>(&bytes, None, 0).to_vec(), [i32::from_ne_bytes([1, 0, 0, 0]), i32::from_ne_bytes([2, 0, 0, 0])]);
/// assert_eq!(np::frombuffer::<u8>(&bytes, Some(2), 4).to_vec(), [2, 0]);
/// ```
pub fn frombuffer<T: Element>(buffer: &[u8], count: Option<usize>, offset: usize) -> Array<T> {
    assert!(
        offset <= buffer.len(),
        "offset must be non-negative and no greater than buffer length ({})",
        buffer.len()
    );
    let item = std::mem::size_of::<T>();
    let rest = &buffer[offset..];
    let n = match count {
        None => {
            assert!(
                rest.len().is_multiple_of(item),
                "buffer size must be a multiple of element size"
            );
            rest.len() / item
        }
        Some(c) => {
            assert!(
                c * item <= rest.len(),
                "buffer is smaller than requested size"
            );
            c
        }
    };
    Array::from(decode::<T>(&rest[..n * item]))
}
/// Round to the given number of decimals, like np.around(a, decimals).
/// ```
/// use rustnum as np;
/// assert_eq!(np::around(&np::array![1.26f64, 2.74], 1).to_vec(), [1.3, 2.7]);
/// assert_eq!(np::around(&np::array![1234.0f64], -2).to_vec(), [1200.0]);
/// ```
pub fn around<F: Float>(a: &Array<F>, decimals: i32) -> Array<F> {
    a.round_decimals(decimals)
}
/// Drop a negligible imaginary part, like np.real_if_close(a, tol) on a `(re, im)` pair.
///
/// rustnum represents complex data as two real arrays. Returns `(re, None)` when every
/// `|im| < tol` (a `tol` above 1 is a multiple of machine epsilon, NumPy's default is 100), and
/// `(re, Some(im))` otherwise.
/// ```
/// use rustnum as np;
/// let re = np::array![2.1, 5.2];
/// let tiny = np::array![1e-16, -1e-15];
/// assert!(np::real_if_close(&re, &tiny, 100.0).1.is_none());
/// let big = np::array![0.0, 0.5];
/// assert!(np::real_if_close(&re, &big, 100.0).1.is_some());
/// ```
pub fn real_if_close<F: Float>(
    re: &Array<F>,
    im: &Array<F>,
    tol: F,
) -> (Array<F>, Option<Array<F>>) {
    assert!(
        re.shape() == im.shape(),
        "operands could not be broadcast together with shapes {:?} {:?}",
        re.shape(),
        im.shape()
    );
    let tol = if tol > F::ONE { tol * F::EPSILON } else { tol };
    if im.iter().all(|x| x.abs() < tol) {
        (re.clone(), None)
    } else {
        (re.clone(), Some(im.clone()))
    }
}
impl<F: Float> Array<F> {
    /// True where the value is positive infinity, like np.isposinf(a).
    /// ```
    /// use rustnum as np;
    /// let a = np::array![f64::INFINITY, f64::NEG_INFINITY, 1.0];
    /// assert_eq!(a.is_posinf().to_vec(), [true, false, false]);
    /// ```
    pub fn is_posinf(&self) -> Array<bool> {
        self.map(|x| x == F::INFINITY)
    }
    /// True where the value is negative infinity, like np.isneginf(a).
    /// ```
    /// use rustnum as np;
    /// let a = np::array![f64::INFINITY, f64::NEG_INFINITY, 1.0];
    /// assert_eq!(a.is_neginf().to_vec(), [false, true, false]);
    /// ```
    pub fn is_neginf(&self) -> Array<bool> {
        self.map(|x| x == F::NEG_INFINITY)
    }
}
