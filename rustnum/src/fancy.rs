//! Advanced (integer-array) indexing, like `a[[0, 2], [1, 3]]`, and the helpers around it:
//! `ix_`, `take_along_axis`, `put_along_axis`, `put`, `choose`, `ravel_multi_index`,
//! `unravel_index`.
//!
//! Index arrays are `i64` (NumPy's `intp`); negative values count from the end of the axis.
//! In an index tuple built with `s!`, an integer array (`SliceArg::Fancy`) selects along its axis;
//! several arrays broadcast together and, as in NumPy, plain integers next to an array also
//! take part in the broadcast. The advanced axes land where NumPy puts them: at the position of the
//! first advanced index when all advanced indices are adjacent, at the front otherwise.

use crate::broadcast::broadcast_shapes;
use crate::layout::size;
use crate::slicing::{axis_index, normalize};
use crate::{Array, Element, IntoArray, SliceArg};
use std::sync::Arc;

fn shape_text(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

/// Normalizes `index` against an axis of length `n`, panicking like NumPy when out of range.
fn wrap_index(index: isize, n: usize, axis: usize) -> usize {
    let k = if index < 0 { index + n as isize } else { index };
    assert!(
        k >= 0 && (k as usize) < n,
        "index {index} is out of bounds for axis {axis} with size {n}"
    );
    k as usize
}

/// One entry of an index tuple after normalization.
enum Slot {
    /// A slice on `axis`: `count` elements starting at `start` with `step`.
    Basic {
        axis: usize,
        start: isize,
        step: isize,
        count: usize,
    },
    /// `np.newaxis`.
    New,
    /// An advanced index on `axis`, values already wrapped and broadcast to the common shape.
    Adv { axis: usize, values: Vec<usize> },
}

/// The result of planning an advanced index: its shape and, per result element in C order,
/// the storage offset it reads or writes.
pub(crate) struct Plan {
    pub shape: Vec<usize>,
    pub offsets: Vec<usize>,
}

impl SliceArg {
    /// True for an integer-array index (see [`SliceArg::Fancy`]).
    pub(crate) fn is_fancy(&self) -> bool {
        matches!(self, SliceArg::Fancy { .. })
    }
}

impl<T: Element> Array<T> {
    /// Plans `a[args]` when `args` contains at least one integer array.
    pub(crate) fn advanced_plan(&self, args: &[SliceArg]) -> Plan {
        let consumed = args
            .iter()
            .filter(|a| !matches!(a, SliceArg::NewAxis))
            .count();
        assert!(
            consumed <= self.ndim(),
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            self.ndim(),
            consumed
        );
        // 1. Raw slots; advanced ones keep their unbroadcast shape for now.
        enum Raw {
            Slot(Slot),
            Adv {
                axis: usize,
                shape: Vec<usize>,
                values: Vec<isize>,
            },
        }
        let mut raw = Vec::new();
        let mut axis = 0;
        for arg in args {
            match arg {
                SliceArg::NewAxis => raw.push(Raw::Slot(Slot::New)),
                SliceArg::Index(i) => {
                    raw.push(Raw::Adv {
                        axis,
                        shape: vec![],
                        values: vec![*i],
                    });
                    axis += 1;
                }
                SliceArg::Fancy { shape, values } => {
                    raw.push(Raw::Adv {
                        axis,
                        shape: shape.clone(),
                        values: values.clone(),
                    });
                    axis += 1;
                }
                SliceArg::Range { start, end, step } => {
                    let (s, count) = normalize(*start, *end, *step, self.shape[axis], false);
                    raw.push(Raw::Slot(Slot::Basic {
                        axis,
                        start: s,
                        step: *step,
                        count,
                    }));
                    axis += 1;
                }
                SliceArg::Inclusive { start, end, step } => {
                    let (s, count) = normalize(*start, Some(*end), *step, self.shape[axis], true);
                    raw.push(Raw::Slot(Slot::Basic {
                        axis,
                        start: s,
                        step: *step,
                        count,
                    }));
                    axis += 1;
                }
            }
        }
        for a in axis..self.ndim() {
            raw.push(Raw::Slot(Slot::Basic {
                axis: a,
                start: 0,
                step: 1,
                count: self.shape[a],
            }));
        }
        // 2. Broadcast the advanced index arrays against each other.
        let mut bshape: Vec<usize> = vec![];
        let mut first = true;
        for r in &raw {
            if let Raw::Adv { shape, .. } = r {
                if first {
                    bshape = shape.clone();
                    first = false;
                } else {
                    bshape = broadcast_shapes(&bshape, shape).unwrap_or_else(|| {
                        let all: Vec<String> = raw
                            .iter()
                            .filter_map(|r| match r {
                                Raw::Adv { shape, .. } => Some(shape_text(shape)),
                                _ => None,
                            })
                            .collect();
                        panic!(
                            "shape mismatch: indexing arrays could not be broadcast together with shapes {}",
                            all.join(" ")
                        )
                    });
                }
            }
        }
        let bsize = size(&bshape);
        let mut slots = Vec::new();
        for r in raw {
            match r {
                Raw::Slot(s) => slots.push(s),
                Raw::Adv {
                    axis,
                    shape,
                    values,
                } => {
                    let n = self.shape[axis];
                    let wide: Vec<i64> = values.iter().map(|&v| v as i64).collect();
                    let src = Array::from_vec(wide, &shape).broadcast_to(&bshape);
                    let wrapped: Vec<usize> = (0..bsize)
                        .map(|i| wrap_index(src.flat_value(i) as isize, n, axis))
                        .collect();
                    slots.push(Slot::Adv {
                        axis,
                        values: wrapped,
                    });
                }
            }
        }
        // 3. Where do the advanced dimensions go?
        let adv_positions: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| matches!(s, Slot::Adv { .. }).then_some(i))
            .collect();
        let adjacent = adv_positions.windows(2).all(|w| w[1] == w[0] + 1);
        // Result dimensions: (slot index, extent); a slot index of usize::MAX marks a broadcast dim.
        let mut dims: Vec<(usize, usize)> = Vec::new();
        let broadcast_dims: Vec<(usize, usize)> = bshape.iter().map(|&n| (usize::MAX, n)).collect();
        if !adjacent {
            dims.extend(&broadcast_dims);
        }
        for (i, s) in slots.iter().enumerate() {
            match s {
                Slot::Basic { count, .. } => dims.push((i, *count)),
                Slot::New => dims.push((i, 1)),
                Slot::Adv { .. } => {
                    if adjacent && i == adv_positions[0] {
                        dims.extend(&broadcast_dims);
                    }
                }
            }
        }
        let shape: Vec<usize> = dims.iter().map(|d| d.1).collect();
        let total = size(&shape);
        // 4. Enumerate result elements.
        let bdim_pos: Vec<usize> = dims
            .iter()
            .enumerate()
            .filter_map(|(p, d)| (d.0 == usize::MAX).then_some(p))
            .collect();
        let mut bstrides = vec![1usize; bshape.len()];
        for j in (0..bshape.len().saturating_sub(1)).rev() {
            bstrides[j] = bstrides[j + 1] * bshape[j + 1];
        }
        let mut offsets = Vec::with_capacity(total);
        let mut coord = vec![0usize; shape.len()];
        for _ in 0..total {
            let mut p = 0;
            for (j, &pos) in bdim_pos.iter().enumerate() {
                p += coord[pos] * bstrides[j];
            }
            let mut off = self.offset as isize;
            for (i, s) in slots.iter().enumerate() {
                match s {
                    Slot::Basic {
                        axis, start, step, ..
                    } => {
                        let c = dims
                            .iter()
                            .position(|d| d.0 == i)
                            .map_or(0, |pos| coord[pos]);
                        off += (start + c as isize * step) * self.strides[*axis];
                    }
                    Slot::New => {}
                    Slot::Adv { axis, values } => {
                        off += values[p] as isize * self.strides[*axis];
                    }
                }
            }
            offsets.push(off as usize);
            for d in (0..shape.len()).rev() {
                coord[d] += 1;
                if coord[d] < shape[d] {
                    break;
                }
                coord[d] = 0;
            }
        }
        Plan { shape, offsets }
    }

    /// Reads `a[args]` for an index tuple containing integer arrays; always a copy.
    pub(crate) fn advanced_get(&self, args: &[SliceArg]) -> Array<T> {
        let plan = self.advanced_plan(args);
        let data = plan.offsets.iter().map(|&o| self.data[o]).collect();
        Array::from_vec(data, &plan.shape)
    }

    /// Writes `a[args] = values` for an index tuple containing integer arrays.
    pub(crate) fn advanced_assign(&mut self, args: &[SliceArg], values: Array<T>) {
        let plan = self.advanced_plan(args);
        assert!(
            broadcast_shapes(values.shape(), &plan.shape).as_deref() == Some(&plan.shape[..])
                && values.ndim() <= plan.shape.len(),
            "shape mismatch: value array of shape {} could not be broadcast to indexing result of shape {}",
            shape_text(values.shape()),
            shape_text(&plan.shape)
        );
        let values = values.broadcast_to(&plan.shape).to_vec();
        self.prepare_write();
        let data = Arc::make_mut(&mut self.data);
        for (o, v) in plan.offsets.into_iter().zip(values) {
            data[o] = v;
        }
    }
}

macro_rules! fancy_from {
    ($($t:ty),*) => {$(
        impl From<Vec<$t>> for SliceArg {
            fn from(v: Vec<$t>) -> Self {
                SliceArg::Fancy {
                    shape: vec![v.len()],
                    values: v.into_iter().map(|x| isize::try_from(x).expect("index is too large")).collect(),
                }
            }
        }
        impl From<&[$t]> for SliceArg {
            fn from(v: &[$t]) -> Self {
                SliceArg::from(v.to_vec())
            }
        }
        impl<const N: usize> From<[$t; N]> for SliceArg {
            fn from(v: [$t; N]) -> Self {
                SliceArg::from(v.to_vec())
            }
        }
    )*};
}
fancy_from!(i32, i64, isize, usize);

macro_rules! fancy_from_array {
    ($($t:ty),*) => {$(
        impl From<&Array<$t>> for SliceArg {
            fn from(a: &Array<$t>) -> Self {
                SliceArg::Fancy {
                    shape: a.shape().to_vec(),
                    values: a.iter().map(|x| isize::try_from(x).expect("index is too large")).collect(),
                }
            }
        }
        impl From<Array<$t>> for SliceArg {
            fn from(a: Array<$t>) -> Self {
                SliceArg::from(&a)
            }
        }
    )*};
}
fancy_from_array!(i32, i64, usize);

/// A 1-d boolean mask selects the positions where it is `true`, like `a[mask]` on one axis.
impl From<&Array<bool>> for SliceArg {
    fn from(m: &Array<bool>) -> Self {
        assert!(
            m.ndim() == 1,
            "boolean index did not match indexed array; only 1-d masks are supported inside an index tuple"
        );
        let values: Vec<isize> = m
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.then_some(i as isize))
            .collect();
        SliceArg::Fancy {
            shape: vec![values.len()],
            values,
        }
    }
}

/// Open mesh from several 1-d index sequences, like `np.ix_(*args)`.
///
/// The `k`-th result has shape `(1, .., n_k, .., 1)`, so indexing with all of them selects the
/// cross product: `a[np.ix_([0, 2], [1, 3])]` is the 2x2 sub-matrix on rows 0, 2 and columns 1, 3.
///
/// ```
/// use rustnum as np;
/// let a = np::arange(0, 12, 1).reshape(&[3, 4]);
/// let ix = np::ix_(&[&[0, 2], &[1, 3]]);
/// assert_eq!(ix[0].shape(), [2, 1]);
/// assert_eq!(ix[1].shape(), [1, 2]);
/// let sub = a.slice(&np::s![&ix[0], &ix[1]]);
/// assert_eq!(sub.to_vec(), [1, 3, 9, 11]);
/// ```
pub fn ix_(args: &[&[i64]]) -> Vec<Array<i64>> {
    let k = args.len();
    args.iter()
        .enumerate()
        .map(|(i, a)| {
            let mut shape = vec![1; k];
            shape[i] = a.len();
            Array::from_vec(a.to_vec(), &shape)
        })
        .collect()
}

/// Sets of coordinates for taking values from `a` along `axis`.
fn along_axis_plan(
    a_shape: &[usize],
    idx: &Array<i64>,
    axis: isize,
    what: &str,
) -> (usize, Vec<usize>) {
    assert!(
        a_shape.len() == idx.ndim(),
        "`indices` and `{what}` must have the same number of dimensions"
    );
    let axis = axis_index(axis, a_shape.len());
    let mut shape = Vec::with_capacity(a_shape.len());
    for d in 0..a_shape.len() {
        if d == axis {
            shape.push(idx.shape()[d]);
        } else {
            let (x, y) = (a_shape[d], idx.shape()[d]);
            assert!(
                x == y || x == 1 || y == 1,
                "shape mismatch: indexing arrays could not be broadcast together with shapes {} {}",
                shape_text(a_shape),
                shape_text(idx.shape())
            );
            shape.push(if x == 1 { y } else { x });
        }
    }
    (axis, shape)
}

fn along_axis_offsets<T: Element>(
    a: &Array<T>,
    idx: &Array<i64>,
    axis: usize,
    shape: &[usize],
) -> Vec<usize> {
    let total = size(shape);
    let mut coord = vec![0usize; shape.len()];
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        let mut off = a.offset as isize;
        let icoord: Vec<usize> = coord
            .iter()
            .zip(idx.shape())
            .map(|(&c, &n)| if n == 1 { 0 } else { c })
            .collect();
        let v = idx.get(&icoord);
        for (d, &cd) in coord.iter().enumerate() {
            let c = if d == axis {
                wrap_index(v as isize, a.shape()[d], d)
            } else if a.shape()[d] == 1 {
                0
            } else {
                cd
            };
            off += c as isize * a.strides[d];
        }
        out.push(off as usize);
        for d in (0..shape.len()).rev() {
            coord[d] += 1;
            if coord[d] < shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    out
}

/// Takes values from `a` along `axis` at the positions in `indices`, like `np.take_along_axis`.
///
/// `indices` must have the same number of dimensions as `a`; the other axes broadcast. The result
/// has the shape of `indices` broadcast against `a` outside `axis`. Typical use is applying
/// `argsort`/`argmax` results. Panics on out-of-range indices.
///
/// ```
/// use rustnum as np;
/// let a = np::array![[10, 30, 20], [60, 40, 50]];
/// let order = np::array![[0i64, 2, 1], [1, 2, 0]];
/// let sorted = np::take_along_axis(&a, &order, 1);
/// assert_eq!(sorted.to_vec(), [10, 20, 30, 40, 50, 60]);
/// ```
pub fn take_along_axis<T: Element>(a: &Array<T>, indices: &Array<i64>, axis: isize) -> Array<T> {
    let (axis, shape) = along_axis_plan(a.shape(), indices, axis, "arr");
    let offsets = along_axis_offsets(a, indices, axis, &shape);
    Array::from_vec(offsets.into_iter().map(|o| a.data[o]).collect(), &shape)
}

/// Puts `values` into `a` along `axis` at the positions in `indices`, like `np.put_along_axis`.
///
/// `indices` and `values` broadcast against `a` outside `axis`; later writes win when an index
/// repeats. Panics on out-of-range indices.
///
/// ```
/// use rustnum as np;
/// let mut a = np::zeros::<i32>([2, 3]);
/// np::put_along_axis(&mut a, &np::array![[1i64], [2]], 7, 1);
/// assert_eq!(a.to_vec(), [0, 7, 0, 0, 0, 7]);
/// ```
pub fn put_along_axis<T: Element>(
    a: &mut Array<T>,
    indices: &Array<i64>,
    values: impl IntoArray<T>,
    axis: isize,
) {
    let (axis, shape) = along_axis_plan(a.shape(), indices, axis, "arr");
    let values = values.into_array();
    assert!(
        broadcast_shapes(values.shape(), &shape).as_deref() == Some(&shape[..]),
        "shape mismatch: value array of shape {} could not be broadcast to indexing result of shape {}",
        shape_text(values.shape()),
        shape_text(&shape)
    );
    let values = values.broadcast_to(&shape).to_vec();
    a.prepare_write();
    let offsets = along_axis_offsets(a, indices, axis, &shape);
    let data = Arc::make_mut(&mut a.data);
    for (o, v) in offsets.into_iter().zip(values) {
        data[o] = v;
    }
}

/// Replaces elements of the flattened `a` at `ind` with `v`, like `np.put(a, ind, v)`.
///
/// `v` repeats cyclically when shorter than `ind`. Negative indices count from the end; an
/// out-of-range index panics like NumPy's default `mode='raise'`.
///
/// ```
/// use rustnum as np;
/// let mut a = np::zeros::<i32>([2, 2]);
/// np::put(&mut a, [0i64, 3], [5, 6]);
/// assert_eq!(a.to_vec(), [5, 0, 0, 6]);
/// ```
pub fn put<T: Element>(a: &mut Array<T>, ind: impl IntoArray<i64>, v: impl IntoArray<T>) {
    let ind = ind.into_array().to_vec();
    let v = v.into_array().to_vec();
    if ind.is_empty() {
        return;
    }
    assert!(!v.is_empty(), "cannot replace elements of an empty array");
    let n = a.size();
    let mut flat = a.to_vec();
    for (k, &i) in ind.iter().enumerate() {
        flat[wrap_index(i as isize, n, 0)] = v[k % v.len()];
    }
    let shape = a.shape().to_vec();
    *a = Array::from_vec(flat, &shape);
}

/// Builds an array from `choices` picked by the index array `a`, like `np.choose(a, choices)`.
///
/// `a` and every choice broadcast to a common shape; element `i` of the result is
/// `choices[a[i]][i]`. Panics with "invalid entry in choice array" for an out-of-range index.
///
/// ```
/// use rustnum as np;
/// let pick = np::array![0i64, 1, 0, 1];
/// let r = np::choose(&pick, &[&np::array![1, 2, 3, 4], &np::array![10, 20, 30, 40]]);
/// assert_eq!(r.to_vec(), [1, 20, 3, 40]);
/// ```
pub fn choose<T: Element>(a: &Array<i64>, choices: &[&Array<T>]) -> Array<T> {
    assert!(!choices.is_empty(), "choices must not be empty");
    let mut shape = a.shape().to_vec();
    for c in choices {
        shape = broadcast_shapes(&shape, c.shape()).unwrap_or_else(|| {
            panic!("shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape {} and arg 1 with shape {}.", shape_text(&shape), shape_text(c.shape()))
        });
    }
    let idx = a.broadcast_to(&shape).to_vec();
    let views: Vec<Vec<T>> = choices
        .iter()
        .map(|c| c.broadcast_to(&shape).to_vec())
        .collect();
    let out = idx
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            assert!(
                k >= 0 && (k as usize) < views.len(),
                "invalid entry in choice array"
            );
            views[k as usize][i]
        })
        .collect();
    Array::from_vec(out, &shape)
}

/// Flat C-order index from a tuple of coordinate arrays, like `np.ravel_multi_index(multi_index, dims)`.
///
/// The coordinate arrays broadcast together. Panics with "invalid entry in coordinates array"
/// when a coordinate is outside `0..dims[k]`.
///
/// ```
/// use rustnum as np;
/// let r = np::array![0i64, 1, 2];
/// let c = np::array![1i64, 2, 3];
/// assert_eq!(np::ravel_multi_index(&[&r, &c], &[3, 4]).to_vec(), [1, 6, 11]);
/// ```
pub fn ravel_multi_index(multi_index: &[&Array<i64>], dims: &[usize]) -> Array<i64> {
    assert!(
        multi_index.len() == dims.len(),
        "parameter multi_index must be a sequence of length {}",
        dims.len()
    );
    let mut shape: Vec<usize> = vec![];
    for m in multi_index {
        shape = broadcast_shapes(&shape, m.shape()).unwrap_or_else(|| {
            panic!("shape mismatch: objects cannot be broadcast to a single shape")
        });
    }
    let parts: Vec<Vec<i64>> = multi_index
        .iter()
        .map(|m| m.broadcast_to(&shape).to_vec())
        .collect();
    let n = size(&shape);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut flat = 0i64;
        for (k, p) in parts.iter().enumerate() {
            let c = p[i];
            assert!(
                c >= 0 && (c as usize) < dims[k],
                "invalid entry in coordinates array"
            );
            flat = flat * dims[k] as i64 + c;
        }
        out.push(flat);
    }
    Array::from_vec(out, &shape)
}

/// Coordinates of flat C-order indices in an array of `shape`, like `np.unravel_index(indices, shape)`.
///
/// Returns one array per dimension, each with the shape of `indices`. Panics when an index is
/// outside `0..prod(shape)`.
///
/// ```
/// use rustnum as np;
/// let rc = np::unravel_index(&np::array![1i64, 6, 11], &[3, 4]);
/// assert_eq!(rc[0].to_vec(), [0, 1, 2]);
/// assert_eq!(rc[1].to_vec(), [1, 2, 3]);
/// ```
pub fn unravel_index(indices: &Array<i64>, shape: &[usize]) -> Vec<Array<i64>> {
    let total = size(shape);
    let mut cols: Vec<Vec<i64>> = vec![Vec::with_capacity(indices.size()); shape.len()];
    for i in indices.iter() {
        assert!(
            i >= 0 && (i as usize) < total,
            "index {i} is out of bounds for array with size {total}"
        );
        let mut rem = i as usize;
        for d in (0..shape.len()).rev() {
            cols[d].push((rem % shape[d]) as i64);
            rem /= shape[d];
        }
    }
    cols.into_iter()
        .map(|c| Array::from_vec(c, indices.shape()))
        .collect()
}
