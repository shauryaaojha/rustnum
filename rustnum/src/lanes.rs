//! Shared helper: run a kernel over every 1-d lane of an array along one axis.
use crate::layout::size;
use crate::{Array, Element};

/// Apply `f(lane_in, lane_out)` to every lane along `axis` (already normalised).
///
/// The result has the input shape with `shape[axis]` replaced by `out_n`.
pub(crate) fn map_lanes<T: Element, U: Element>(
    a: &Array<T>,
    axis: usize,
    out_n: usize,
    mut f: impl FnMut(&[T], &mut [U]),
) -> Array<U> {
    let c = a.to_contiguous();
    let src = c.as_slice().expect("contiguous");
    let n = a.shape[axis];
    let outer = size(&a.shape[..axis]);
    let inner = size(&a.shape[axis + 1..]);
    let mut out_shape = a.shape.clone();
    out_shape[axis] = out_n;
    let mut out = vec![U::default(); outer * out_n * inner];
    let mut lane_in = vec![T::default(); n];
    let mut lane_out = vec![U::default(); out_n];
    for o in 0..outer {
        for j in 0..inner {
            for (i, x) in lane_in.iter_mut().enumerate() {
                *x = src[(o * n + i) * inner + j];
            }
            f(&lane_in, &mut lane_out);
            for (i, &x) in lane_out.iter().enumerate() {
                out[(o * out_n + i) * inner + j] = x;
            }
        }
    }
    Array::from_owned_shape(out, out_shape)
}

/// Python-style tuple repr of a shape, as used in NumPy error messages: `(3,)`, `(2, 3)`.
pub(crate) fn pyshape(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Visit every element in C order with its flat index, writing through copy-on-write.
pub(crate) fn for_each_mut<T: Element>(a: &mut Array<T>, mut f: impl FnMut(usize, &mut T)) {
    a.prepare_write();
    let offsets = crate::layout::Offsets::new(&a.shape, &a.strides, a.offset);
    let data = std::sync::Arc::make_mut(&mut a.data);
    for (i, off) in offsets.enumerate() {
        f(i, &mut data[off]);
    }
}

/// A view of `a`'s storage with a new shape, strides and offset (no bounds check: callers verify).
pub(crate) fn view<T: Element>(
    a: &Array<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
) -> Array<T> {
    Array {
        data: a.data.clone(),
        recycle: a.recycle,
        shape,
        strides,
        offset,
    }
}
