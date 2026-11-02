//! Sorting, searching and conditional selection: `lexsort`, `partition`, `select`, `place`, ...
use crate::lanes::{for_each_mut, map_lanes, pyshape, view};
use crate::layout::size;
use crate::manip::compare;
use crate::slicing::axis_index;
use crate::{Array, Element, IntoArray, Num, broadcast_shapes};
use std::cmp::Ordering;

// ---------------------------------------------------------------- introselect

fn insertion_sort<T: Copy>(v: &mut [T], cmp: &impl Fn(&T, &T) -> Ordering) {
    for i in 1..v.len() {
        let x = v[i];
        let mut j = i;
        while j > 0 && cmp(&v[j - 1], &x) == Ordering::Greater {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = x;
    }
}
fn median3<T: Copy>(a: T, b: T, c: T, cmp: &impl Fn(&T, &T) -> Ordering) -> T {
    let (lo, hi) = if cmp(&a, &b) == Ordering::Greater {
        (b, a)
    } else {
        (a, b)
    };
    if cmp(&c, &lo) == Ordering::Less {
        lo
    } else if cmp(&c, &hi) == Ordering::Greater {
        hi
    } else {
        c
    }
}
/// Median of medians (groups of five): a pivot guaranteeing a linear-time worst case.
fn median_of_medians<T: Copy>(v: &mut [T], cmp: &impl Fn(&T, &T) -> Ordering) -> T {
    let groups = v.len().div_ceil(5);
    for g in 0..groups {
        let lo = g * 5;
        let hi = (lo + 5).min(v.len());
        insertion_sort(&mut v[lo..hi], cmp);
        v.swap(g, lo + (hi - lo - 1) / 2);
    }
    let mid = groups / 2;
    introselect(&mut v[..groups], mid, cmp);
    v[mid]
}
/// Rearranges `v` so that `v[k]` is the element sorted order would put there, everything before it
/// is `<=` and everything after is `>=`. Introselect: quickselect with a median-of-three pivot that
/// falls back to median-of-medians when the recursion depth budget is spent, so it is O(n) overall.
fn introselect<T: Copy>(v: &mut [T], k: usize, cmp: &impl Fn(&T, &T) -> Ordering) {
    let (mut lo, mut hi) = (0usize, v.len());
    let mut budget = 2 * (usize::BITS - v.len().max(1).leading_zeros()) as usize;
    while hi - lo > 12 {
        let n = hi - lo;
        let pivot = if budget > 0 {
            budget -= 1;
            median3(v[lo], v[lo + n / 2], v[hi - 1], cmp)
        } else {
            median_of_medians(&mut v[lo..hi], cmp)
        };
        // three-way partition: [lo, lt) < pivot, [lt, i) == pivot, [gt, hi) > pivot
        let (mut lt, mut i, mut gt) = (lo, lo, hi);
        while i < gt {
            match cmp(&v[i], &pivot) {
                Ordering::Less => {
                    v.swap(lt, i);
                    lt += 1;
                    i += 1;
                }
                Ordering::Greater => {
                    gt -= 1;
                    v.swap(i, gt);
                }
                Ordering::Equal => i += 1,
            }
        }
        if k < lt {
            hi = lt;
        } else if k >= gt {
            lo = gt;
        } else {
            return;
        }
    }
    insertion_sort(&mut v[lo..hi], cmp);
}
fn check_kth(kth: isize, n: usize) -> usize {
    let k = if kth < 0 { kth + n as isize } else { kth };
    assert!(
        k >= 0 && (k as usize) < n,
        "kth(={}) out of bounds ({})",
        k,
        n
    );
    k as usize
}
impl<T: Element> Array<T> {
    /// Partial sort around the `kth` element along `axis`, like np.partition(a, kth, axis).
    ///
    /// Element `kth` of each lane is the value a full sort would put there; smaller-or-equal values
    /// come before it, larger-or-equal after it (order within each side is unspecified). NaNs sort
    /// last. Average and worst case O(n) (introselect). Panics with `kth(=k) out of bounds (n)`.
    /// ```
    /// use rustnum as np;
    /// let p = np::array![3, 4, 2, 1].partition(2, -1);
    /// assert_eq!(p[[2]], 3);
    /// assert!(p[[0]] <= 3 && p[[1]] <= 3 && p[[3]] >= 3);
    /// ```
    pub fn partition(&self, kth: isize, axis: isize) -> Array<T> {
        let ax = axis_index(axis, self.ndim());
        let n = self.shape[ax];
        let k = check_kth(kth, n);
        map_lanes(self, ax, n, |l, o| {
            o.copy_from_slice(l);
            introselect(o, k, &compare::<T>);
        })
    }
    /// Indices that would partition the array around `kth` along `axis`, like np.argpartition(a, kth, axis).
    ///
    /// `a.take`-ing the result gives a valid partition; see [`Array::partition`].
    /// ```
    /// use rustnum as np;
    /// let a = np::array![3, 4, 2, 1];
    /// let i = a.argpartition(1, -1);
    /// assert_eq!(a[[i[[1]]]], 2);
    /// ```
    pub fn argpartition(&self, kth: isize, axis: isize) -> Array<usize> {
        let ax = axis_index(axis, self.ndim());
        let n = self.shape[ax];
        let k = check_kth(kth, n);
        map_lanes(self, ax, n, |l, o| {
            let mut pairs: Vec<(T, usize)> = l.iter().copied().zip(0..).collect();
            introselect(&mut pairs, k, &|a: &(T, usize), b: &(T, usize)| {
                compare(&a.0, &b.0)
            });
            for (d, p) in o.iter_mut().zip(pairs) {
                *d = p.1;
            }
        })
    }
}

/// Indirect stable sort on several keys, like np.lexsort(keys, axis).
///
/// The last key is the primary sort key. All keys must have the same shape; the result holds
/// indices along `axis`. NaNs sort last.
/// ```
/// use rustnum as np;
/// let surname = np::array![1, 2, 1, 2, 1];
/// let first = np::array![5, 5, 3, 3, 3];
/// // sort by `first`, then by `surname`
/// assert_eq!(np::lexsort(&[&surname, &first], -1).to_vec(), [2, 4, 3, 0, 1]);
/// ```
pub fn lexsort<T: Element>(keys: &[&Array<T>], axis: isize) -> Array<usize> {
    assert!(
        !keys.is_empty(),
        "need sequence of keys with len > 0 in lexsort"
    );
    let shape = keys[0].shape().to_vec();
    assert!(
        keys.iter().all(|k| k.shape() == shape),
        "all keys need to be the same shape"
    );
    let ax = axis_index(axis, shape.len());
    let n = shape[ax];
    let outer = size(&shape[..ax]);
    let inner = size(&shape[ax + 1..]);
    let cs: Vec<Array<T>> = keys.iter().map(|k| k.to_contiguous()).collect();
    let mut out = vec![0usize; size(&shape)];
    let mut idx: Vec<usize> = Vec::with_capacity(n);
    for o in 0..outer {
        for j in 0..inner {
            let at = |i: usize| (o * n + i) * inner + j;
            idx.clear();
            idx.extend(0..n);
            idx.sort_by(|&a, &b| {
                for k in cs.iter().rev() {
                    let s = k.as_slice().expect("contiguous");
                    let ord = compare(&s[at(a)], &s[at(b)]);
                    if ord != Ordering::Equal {
                        return ord;
                    }
                }
                Ordering::Equal
            });
            for (i, &v) in idx.iter().enumerate() {
                out[at(i)] = v;
            }
        }
    }
    Array::from_owned_shape(out, shape)
}

/// Indices of the non-zero elements of the flattened array, like np.flatnonzero(a).
/// ```
/// use rustnum as np;
/// assert_eq!(np::flatnonzero(&np::array![[0, 1], [2, 0]]).to_vec(), [1, 2]);
/// ```
pub fn flatnonzero<T: Element>(a: &Array<T>) -> Array<usize> {
    a.iter()
        .enumerate()
        .filter_map(|(i, x)| (x != T::default()).then_some(i))
        .collect()
}

fn broadcast_all(shapes: &[Vec<usize>]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for (i, s) in shapes.iter().enumerate() {
        match broadcast_shapes(&out, s) {
            Some(b) => out = b,
            None => {
                let j = (0..i)
                    .find(|&j| broadcast_shapes(&shapes[j], s).is_none())
                    .unwrap_or(0);
                panic!(
                    "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg {} with shape {} and arg {} with shape {}.",
                    j,
                    pyshape(&shapes[j]),
                    i,
                    pyshape(s)
                );
            }
        }
    }
    out
}

/// Choose elements from several arrays by the first true condition, like np.select(condlist, choicelist, default).
///
/// All inputs broadcast together; where no condition holds the result is `default`.
/// ```
/// use rustnum as np;
/// let x = np::arange(0, 6, 1);
/// let out = np::select(&[&x.lt(2), &x.gt(3)], &[&x, &x.map(|v| v * 10)], 0);
/// assert_eq!(out.to_vec(), [0, 1, 0, 0, 40, 50]);
/// ```
pub fn select<T: Element>(
    condlist: &[&Array<bool>],
    choicelist: &[&Array<T>],
    default: T,
) -> Array<T> {
    assert!(
        !condlist.is_empty(),
        "select with an empty condition list is not possible"
    );
    assert!(
        condlist.len() == choicelist.len(),
        "list of cases must be same length as list of conditions"
    );
    let shapes: Vec<Vec<usize>> = condlist
        .iter()
        .map(|c| c.shape().to_vec())
        .chain(choicelist.iter().map(|c| c.shape().to_vec()))
        .collect();
    let shape = broadcast_all(&shapes);
    let conds: Vec<Array<bool>> = condlist.iter().map(|c| c.broadcast_to(&shape)).collect();
    let choices: Vec<Array<T>> = choicelist.iter().map(|c| c.broadcast_to(&shape)).collect();
    Array::from_shape_fn(&shape, |c| {
        for (cond, choice) in conds.iter().zip(&choices) {
            if cond.get(c) {
                return choice.get(c);
            }
        }
        default
    })
}

/// Evaluate a piecewise-defined function, like np.piecewise(x, condlist, funclist).
///
/// `funclist` has one function per condition, or one extra last function used where no condition
/// holds (default: zero). Later conditions overwrite earlier ones where several are true. Use
/// `&|_| c` for a constant piece.
/// ```
/// use rustnum as np;
/// let x = np::linspace(-2.5, 2.5, 6);
/// let y = np::piecewise(&x, &[&x.lt(0.0), &x.ge(0.0)], &[&|v: f64| -v, &|v: f64| v]);
/// assert_eq!(y.to_vec(), [2.5, 1.5, 0.5, 0.5, 1.5, 2.5]);
/// ```
pub fn piecewise<T: Num>(
    x: &Array<T>,
    condlist: &[&Array<bool>],
    funclist: &[&dyn Fn(T) -> T],
) -> Array<T> {
    let k = condlist.len();
    assert!(
        funclist.len() == k || funclist.len() == k + 1,
        "with {} condition(s), either {} or {} functions are expected",
        k,
        k,
        k + 1
    );
    let conds: Vec<Array<bool>> = condlist
        .iter()
        .map(|c| {
            assert!(
                broadcast_shapes(c.shape(), x.shape()).as_deref() == Some(x.shape()),
                "boolean index did not match indexed array along axis 0; size of axis is {} but size of corresponding boolean axis is {}",
                x.shape().first().copied().unwrap_or(1),
                c.shape().first().copied().unwrap_or(1)
            );
            c.broadcast_to(x.shape())
        })
        .collect();
    Array::from_shape_fn(x.shape(), |c| {
        let v = x.get(c);
        let mut out = T::ZERO;
        let mut hit = false;
        for (cond, f) in conds.iter().zip(funclist) {
            if cond.get(c) {
                out = f(v);
                hit = true;
            }
        }
        if !hit && funclist.len() == k + 1 {
            out = funclist[k](v);
        }
        out
    })
}

/// Elements of `arr` where `condition` is true, as a 1-d array, like np.extract(condition, arr).
/// ```
/// use rustnum as np;
/// let a = np::arange(0, 6, 1).reshape(&[2, 3]);
/// assert_eq!(np::extract(&a.gt(2), &a).to_vec(), [3, 4, 5]);
/// ```
pub fn extract<T: Element>(condition: &Array<bool>, arr: &Array<T>) -> Array<T> {
    let flat = arr.ravel();
    condition
        .iter()
        .enumerate()
        .filter(|&(_, c)| c)
        .map(|(i, _)| {
            assert!(
                i < flat.size(),
                "index {} is out of bounds for axis 0 with size {}",
                i,
                flat.size()
            );
            flat.get(&[i])
        })
        .collect()
}

/// Change elements of an array in place where `mask` is true, like np.place(arr, mask, vals).
///
/// `vals` is repeated cyclically if shorter than the number of true entries.
/// ```
/// use rustnum as np;
/// let mut a = np::arange(0, 5, 1);
/// let big = a.gt(2);
/// np::place(&mut a, &big, &[-1, -2]);
/// assert_eq!(a.to_vec(), [0, 1, 2, -1, -2]);
/// ```
pub fn place<T: Element>(arr: &mut Array<T>, mask: &Array<bool>, vals: &[T]) {
    assert!(
        mask.size() == arr.size(),
        "place: mask and data must be the same size"
    );
    let mask = mask.to_vec();
    if mask.iter().any(|&m| m) {
        assert!(!vals.is_empty(), "Cannot insert from an empty array!");
    }
    let mut next = 0usize;
    for_each_mut(arr, |i, x| {
        if mask[i] {
            *x = vals[next % vals.len()];
            next += 1;
        }
    });
}

/// Copy values from `src` into `dst`, broadcasting, like np.copyto(dst, src, where=where_).
///
/// Only positions where `where_` is true are written when it is given.
/// ```
/// use rustnum as np;
/// let mut a = np::zeros::<i32>([2, 3]);
/// np::copyto(&mut a, [1, 2, 3], None);
/// np::copyto(&mut a, 9, Some(&np::array![[true, false, false], [false, false, true]]));
/// assert_eq!(a.to_vec(), [9, 2, 3, 1, 2, 9]);
/// ```
pub fn copyto<T: Element>(
    dst: &mut Array<T>,
    src: impl IntoArray<T>,
    where_: Option<&Array<bool>>,
) {
    let src = src.into_array();
    assert!(
        broadcast_shapes(src.shape(), dst.shape()).as_deref() == Some(dst.shape())
            && src.ndim() <= dst.ndim(),
        "could not broadcast input array from shape {} into shape {}",
        pyshape(src.shape()),
        pyshape(dst.shape())
    );
    let src = src.broadcast_to(dst.shape()).to_vec();
    let mask = where_.map(|w| {
        assert!(
            broadcast_shapes(w.shape(), dst.shape()).as_deref() == Some(dst.shape())
                && w.ndim() <= dst.ndim(),
            "could not broadcast where mask from shape {} into shape {}",
            pyshape(w.shape()),
            pyshape(dst.shape())
        );
        w.broadcast_to(dst.shape()).to_vec()
    });
    for_each_mut(dst, |i, x| {
        if mask.as_ref().is_none_or(|m| m[i]) {
            *x = src[i];
        }
    });
}

/// Trim leading and/or trailing zeros, like np.trim_zeros(filt, trim).
///
/// `trim` is "f" (front), "b" (back) or "fb" (both, NumPy's default). For arrays with more than
/// one axis every axis is trimmed to the bounding box of the non-zero elements. The result is a view.
/// ```
/// use rustnum as np;
/// let a = np::array![0, 0, 1, 2, 0, 3, 0, 0];
/// assert_eq!(np::trim_zeros(&a, "fb").to_vec(), [1, 2, 0, 3]);
/// assert_eq!(np::trim_zeros(&a, "f").to_vec(), [1, 2, 0, 3, 0, 0]);
/// assert_eq!(np::trim_zeros(&a, "b").to_vec(), [0, 0, 1, 2, 0, 3]);
/// ```
pub fn trim_zeros<T: Element>(filt: &Array<T>, trim: &str) -> Array<T> {
    let t = trim.to_ascii_lowercase();
    assert!(
        t.chars().all(|c| c == 'f' || c == 'b'),
        "unexpected character(s) in `trim`: '{}'",
        trim
    );
    let (front, back) = (t.contains('f'), t.contains('b'));
    let nd = filt.ndim();
    if nd == 0 || filt.is_empty() {
        return filt.clone();
    }
    let mut lo = vec![usize::MAX; nd];
    let mut hi = vec![0usize; nd];
    let mut c = vec![0usize; nd];
    let mut any = false;
    for x in filt.iter() {
        if x != T::default() {
            any = true;
            for d in 0..nd {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d] + 1);
            }
        }
        crate::layout::advance(&mut c, filt.shape());
    }
    let mut shape = Vec::with_capacity(nd);
    let mut offset = filt.offset as isize;
    for d in 0..nd {
        let n = filt.shape[d];
        let (s, e) = if any {
            (if front { lo[d] } else { 0 }, if back { hi[d] } else { n })
        } else {
            (if front { n } else { 0 }, if back { 0 } else { n })
        };
        let len = e.saturating_sub(s);
        shape.push(len);
        if len > 0 {
            offset += s as isize * filt.strides[d];
        }
    }
    if shape.contains(&0) {
        offset = 0;
    }
    view(filt, shape, filt.strides.clone(), offset as usize)
}
