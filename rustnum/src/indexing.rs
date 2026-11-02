//! Index helpers: triangle and diagonal indices, `fill_diagonal`, `ndindex`, `ndenumerate`.
use crate::lanes::for_each_mut;
use crate::layout::{advance, size};
use crate::{Array, Element, IntoArray};

fn tri_indices(n: usize, k: isize, m: Option<usize>, lower: bool) -> (Array<usize>, Array<usize>) {
    let m = m.unwrap_or(n);
    let (mut rows, mut cols) = (Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..m {
            let keep = if lower {
                j as isize - i as isize <= k
            } else {
                j as isize - i as isize >= k
            };
            if keep {
                rows.push(i);
                cols.push(j);
            }
        }
    }
    (Array::from(rows), Array::from(cols))
}
/// Indices of the lower triangle of an `n` x `m` array, like np.tril_indices(n, k, m).
///
/// `m` defaults to `n` when `None`. Returns `(rows, cols)` in row-major order.
/// ```
/// use rustnum as np;
/// let (r, c) = np::tril_indices(3, -1, None);
/// assert_eq!(r.to_vec(), [1, 2, 2]);
/// assert_eq!(c.to_vec(), [0, 0, 1]);
/// ```
pub fn tril_indices(n: usize, k: isize, m: Option<usize>) -> (Array<usize>, Array<usize>) {
    tri_indices(n, k, m, true)
}
/// Indices of the upper triangle of an `n` x `m` array, like np.triu_indices(n, k, m).
///
/// `m` defaults to `n` when `None`. Returns `(rows, cols)` in row-major order.
/// ```
/// use rustnum as np;
/// let (r, c) = np::triu_indices(3, 1, None);
/// assert_eq!(r.to_vec(), [0, 0, 1]);
/// assert_eq!(c.to_vec(), [1, 2, 2]);
/// ```
pub fn triu_indices(n: usize, k: isize, m: Option<usize>) -> (Array<usize>, Array<usize>) {
    tri_indices(n, k, m, false)
}
fn need_2d<T: Element>(arr: &Array<T>) {
    assert!(arr.ndim() == 2, "input array must be 2-d");
}
/// Lower-triangle indices for the shape of a 2-d array, like np.tril_indices_from(arr, k).
/// ```
/// use rustnum as np;
/// let a = np::zeros::<f64>([2, 3]);
/// let (r, c) = np::tril_indices_from(&a, 0);
/// assert_eq!((r.to_vec(), c.to_vec()), (vec![0, 1, 1], vec![0, 0, 1]));
/// ```
pub fn tril_indices_from<T: Element>(arr: &Array<T>, k: isize) -> (Array<usize>, Array<usize>) {
    need_2d(arr);
    tril_indices(arr.shape()[0], k, Some(arr.shape()[1]))
}
/// Upper-triangle indices for the shape of a 2-d array, like np.triu_indices_from(arr, k).
/// ```
/// use rustnum as np;
/// let a = np::zeros::<f64>([2, 3]);
/// let (r, c) = np::triu_indices_from(&a, 1);
/// assert_eq!((r.to_vec(), c.to_vec()), (vec![0, 0, 1], vec![1, 2, 2]));
/// ```
pub fn triu_indices_from<T: Element>(arr: &Array<T>, k: isize) -> (Array<usize>, Array<usize>) {
    need_2d(arr);
    triu_indices(arr.shape()[0], k, Some(arr.shape()[1]))
}
/// Indices of the main diagonal of an `ndim`-dimensional cube, like np.diag_indices(n, ndim).
///
/// Returns `ndim` index arrays of length `n` (NumPy's default `ndim` is 2).
/// ```
/// use rustnum as np;
/// let idx = np::diag_indices(3, 2);
/// assert_eq!(idx[0].to_vec(), [0, 1, 2]);
/// assert_eq!(idx[1].to_vec(), [0, 1, 2]);
/// ```
pub fn diag_indices(n: usize, ndim: usize) -> Vec<Array<usize>> {
    (0..ndim).map(|_| (0..n).collect()).collect()
}
/// Diagonal indices for a cube-shaped array, like np.diag_indices_from(arr).
///
/// Panics unless `arr` has at least 2 axes of equal length.
/// ```
/// use rustnum as np;
/// let a = np::zeros::<f64>([4, 4]);
/// assert_eq!(np::diag_indices_from(&a)[1].to_vec(), [0, 1, 2, 3]);
/// ```
pub fn diag_indices_from<T: Element>(arr: &Array<T>) -> Vec<Array<usize>> {
    assert!(arr.ndim() >= 2, "input array must be at least 2-d");
    assert!(
        arr.shape().iter().all(|&d| d == arr.shape()[0]),
        "All dimensions of input must be of equal length"
    );
    diag_indices(arr.shape()[0], arr.ndim())
}
/// Fill the main diagonal of an array in place, like np.fill_diagonal(a, val, wrap).
///
/// `val` is repeated cyclically if it has several elements. For tall 2-d arrays `wrap` restarts
/// the diagonal after `ncols` rows (NumPy's flat-index behaviour). Arrays with more than two axes
/// must have all axes of equal length.
/// ```
/// use rustnum as np;
/// let mut a = np::zeros::<i32>([3, 3]);
/// np::fill_diagonal(&mut a, 5, false);
/// assert_eq!(a.to_vec(), [5, 0, 0, 0, 5, 0, 0, 0, 5]);
/// let mut t = np::zeros::<i32>([5, 2]);
/// np::fill_diagonal(&mut t, 1, true);
/// assert_eq!(t.to_vec(), [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
/// ```
pub fn fill_diagonal<T: Element>(a: &mut Array<T>, val: impl IntoArray<T>, wrap: bool) {
    assert!(a.ndim() >= 2, "array must be at least 2-d");
    let shape = a.shape().to_vec();
    let (step, end) = if a.ndim() == 2 {
        let end = if wrap { a.size() } else { shape[1] * shape[1] };
        (shape[1] + 1, end)
    } else {
        assert!(
            shape.iter().all(|&d| d == shape[0]),
            "All dimensions of input must be of equal length"
        );
        let mut step = 0;
        let mut acc = 1;
        for &d in shape[..shape.len() - 1].iter() {
            acc *= d;
            step += acc;
        }
        (step + 1, a.size())
    };
    let vals = val.into_array().to_vec();
    if vals.is_empty() {
        return;
    }
    let mut next = 0usize;
    for_each_mut(a, |i, x| {
        if i < end && i % step == 0 {
            *x = vals[next % vals.len()];
            next += 1;
        }
    });
}
/// Iterate over every index of an array of the given shape, like np.ndindex(*shape).
///
/// Indices come in row-major order; an empty shape yields one empty index.
/// ```
/// use rustnum as np;
/// let all: Vec<Vec<usize>> = np::ndindex(&[2, 2]).collect();
/// assert_eq!(all, [vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
/// assert_eq!(np::ndindex(&[]).count(), 1);
/// ```
pub fn ndindex(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + use<> {
    let shape = shape.to_vec();
    let mut coord = vec![0usize; shape.len()];
    let mut left = size(&shape);
    std::iter::from_fn(move || {
        if left == 0 {
            return None;
        }
        left -= 1;
        let out = coord.clone();
        advance(&mut coord, &shape);
        Some(out)
    })
}
/// Iterate over `(index, value)` pairs in row-major order, like np.ndenumerate(a).
/// ```
/// use rustnum as np;
/// let a = np::array![[1, 2], [3, 4]];
/// let pairs: Vec<(Vec<usize>, i32)> = np::ndenumerate(&a).collect();
/// assert_eq!(pairs[2], (vec![1, 0], 3));
/// ```
pub fn ndenumerate<T: Element>(a: &Array<T>) -> impl Iterator<Item = (Vec<usize>, T)> + use<T> {
    ndindex(a.shape()).zip(a.iter())
}
/// Indices of the elements selected by a triangle-mask function, like np.mask_indices(n, mask_func, k).
///
/// `mask_func(ones((n, n)), k)` should zero the unselected entries; the non-zero positions are returned.
/// ```
/// use rustnum as np;
/// let upper = |m: &np::Array<i64>, k: isize| {
///     np::Array::from_shape_fn(m.shape(), |c| if c[1] as isize - c[0] as isize >= k { m.get(c) } else { 0 })
/// };
/// let (r, c) = np::mask_indices(3, upper, 1);
/// assert_eq!((r.to_vec(), c.to_vec()), (vec![0, 0, 1], vec![1, 2, 2]));
/// ```
pub fn mask_indices(
    n: usize,
    mask_func: impl Fn(&Array<i64>, isize) -> Array<i64>,
    k: isize,
) -> (Array<usize>, Array<usize>) {
    let m = mask_func(&crate::ones::<i64>([n, n]), k);
    let nz = crate::nonzero(&m);
    let mut it = nz.into_iter();
    (it.next().unwrap_or_default(), it.next().unwrap_or_default())
}
