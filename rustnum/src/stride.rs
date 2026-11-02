//! Stride tricks: `sliding_window_view` and `as_strided` (zero-copy, bounds-checked).
use crate::lanes::view;
use crate::slicing::axis_index;
use crate::{Array, Element};

/// Rolling windows over an array as a zero-copy view, like np.lib.stride_tricks.sliding_window_view(x, window_shape, axis).
///
/// With `axis` `None`, `window_shape` must give one length per axis of `x`. Otherwise it gives one
/// length per entry of `axis`. The result has the shape of `x` with each windowed axis shortened to
/// `n - w + 1`, followed by the window axes. Like every rustnum view it is copy-on-write: writing
/// to it never changes `x`.
/// ```
/// use rustnum as np;
/// let x = np::arange(0, 5, 1);
/// let w = np::sliding_window_view(&x, &[3], None);
/// assert_eq!(w.shape(), [3, 3]);
/// assert_eq!(w.to_vec(), [0, 1, 2, 1, 2, 3, 2, 3, 4]);
/// ```
pub fn sliding_window_view<T: Element>(
    x: &Array<T>,
    window_shape: &[usize],
    axis: Option<&[isize]>,
) -> Array<T> {
    let axes: Vec<usize> = match axis {
        None => {
            assert!(
                window_shape.len() == x.ndim(),
                "Since axis is `None`, must provide window_shape for all dimensions of `x`; got {} window_shape elements and `x.ndim` is {}.",
                window_shape.len(),
                x.ndim()
            );
            (0..x.ndim()).collect()
        }
        Some(ax) => {
            assert!(
                ax.len() == window_shape.len(),
                "Must provide matching length window_shape and axis; got {} window_shape elements and {} axes elements.",
                window_shape.len(),
                ax.len()
            );
            ax.iter().map(|&a| axis_index(a, x.ndim())).collect()
        }
    };
    let mut shape = x.shape.clone();
    let mut strides = x.strides.clone();
    for (&a, &w) in axes.iter().zip(window_shape) {
        assert!(
            shape[a] >= w,
            "window shape cannot be larger than input array shape"
        );
        shape[a] = shape[a] + 1 - w;
    }
    for (&a, &w) in axes.iter().zip(window_shape) {
        shape.push(w);
        strides.push(x.strides[a]);
    }
    let empty = shape.contains(&0);
    view(x, shape, strides, if empty { 0 } else { x.offset })
}
/// Create a view with the given shape and element strides, like np.lib.stride_tricks.as_strided(x, shape, strides).
///
/// Strides count elements (not bytes) and start from the first element of `x`; they may be zero or
/// negative. The view is bounds-checked: if any element it addresses lies outside the underlying
/// buffer, this panics, so it can never read outside the buffer. It is copy-on-write like every view.
/// ```
/// use rustnum as np;
/// let x = np::arange(0, 6, 1);
/// let v = np::as_strided(&x, &[2, 3], &[3, 1]);
/// assert_eq!(v.to_vec(), [0, 1, 2, 3, 4, 5]);
/// let overlap = np::as_strided(&x, &[3, 2], &[1, 1]);
/// assert_eq!(overlap.to_vec(), [0, 1, 1, 2, 2, 3]);
/// ```
pub fn as_strided<T: Element>(x: &Array<T>, shape: &[usize], strides: &[isize]) -> Array<T> {
    assert!(
        shape.len() == strides.len(),
        "mismatch in length of strides and shape"
    );
    if shape.contains(&0) {
        return view(x, shape.to_vec(), strides.to_vec(), 0);
    }
    let (mut lo, mut hi) = (x.offset as i128, x.offset as i128);
    for (&n, &s) in shape.iter().zip(strides) {
        let reach = (n as i128 - 1) * s as i128;
        if reach < 0 {
            lo += reach;
        } else {
            hi += reach;
        }
    }
    assert!(
        lo >= 0 && hi < x.data.len() as i128,
        "as_strided: shape {:?} with strides {:?} would access memory outside the array buffer",
        shape,
        strides
    );
    view(x, shape.to_vec(), strides.to_vec(), x.offset)
}
