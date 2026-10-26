use crate::element::isnan;
use crate::layout::{Offsets, size, strides};
use crate::slicing::axis_index;
use crate::{Array, Element, SliceArg};
use std::{cmp::Ordering, sync::Arc};
pub(crate) fn compare<T: Element>(a: &T, b: &T) -> Ordering {
    match (isnan(*a), isnan(*b)) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    }
}
impl<T: Element> Array<T> {
    /// Implements np.reshape with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3,4].reshape(&[2,2]).shape(),[2,2]);
    /// ```
    pub fn reshape(&self, shape: &[usize]) -> Self {
        assert!(
            self.size() == size(shape),
            "cannot reshape array of size {} into shape {:?}",
            self.size(),
            shape
        );
        let a = self.to_contiguous();
        Self {
            data: a.data,
            shape: shape.to_vec(),
            strides: strides(shape),
            offset: a.offset,
        }
    }
    /// Implements np.reshape with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3,4].reshape_infer(&[2,-1]).shape(),[2,2]);
    /// ```
    pub fn reshape_infer(&self, shape: &[isize]) -> Self {
        let mut out = Vec::new();
        let mut infer = None;
        for (i, &d) in shape.iter().enumerate() {
            if d == -1 {
                assert!(infer.is_none(), "can only specify one unknown dimension");
                infer = Some(i);
                out.push(1)
            } else {
                assert!(d >= 0, "negative dimensions are not allowed");
                out.push(d as usize)
            }
        }
        if let Some(i) = infer {
            let known = size(&out);
            assert!(
                known > 0 && self.size().is_multiple_of(known),
                "cannot reshape array of size {} into shape {:?}",
                self.size(),
                shape
            );
            out[i] = self.size() / known;
        }
        self.reshape(&out)
    }
    /// Implements np.ravel with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].t().ravel().to_vec(),[1,3,2,4]);
    /// ```
    pub fn ravel(&self) -> Self {
        self.reshape(&[self.size()])
    }
    /// Implements np.flatten with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].flatten().shape(),[4]);
    /// ```
    pub fn flatten(&self) -> Self {
        Self::from(self.to_vec())
    }
    /// Implements np.ndarray.T with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].t().to_vec(),[1,3,2,4]);
    /// ```
    pub fn t(&self) -> Self {
        self.transpose(&(0..self.ndim()).rev().collect::<Vec<_>>())
    }
    /// Implements np.transpose with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].transpose(&[1,0]).to_vec(),[1,3,2,4]);
    /// ```
    pub fn transpose(&self, axes: &[usize]) -> Self {
        assert!(axes.len() == self.ndim(), "axes don't match array");
        let mut seen = vec![false; self.ndim()];
        for &a in axes {
            assert!(
                a < self.ndim(),
                "axis {} is out of bounds for array of dimension {}",
                a,
                self.ndim()
            );
            assert!(!seen[a], "repeated axis in transpose");
            seen[a] = true;
        }
        Self {
            data: self.data.clone(),
            shape: axes.iter().map(|&i| self.shape[i]).collect(),
            strides: axes.iter().map(|&i| self.strides[i]).collect(),
            offset: self.offset,
        }
    }
    /// Implements np.transpose with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].permute_axes(&[1,0]).to_vec(),[1,3,2,4]);
    /// ```
    pub fn permute_axes(&self, axes: &[usize]) -> Self {
        self.transpose(axes)
    }
    /// Implements np.swapaxes with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].swapaxes(0,-1).to_vec(),[1,3,2,4]);
    /// ```
    pub fn swapaxes(&self, a: isize, b: isize) -> Self {
        let mut axes: Vec<_> = (0..self.ndim()).collect();
        axes.swap(axis_index(a, self.ndim()), axis_index(b, self.ndim()));
        self.transpose(&axes)
    }
    /// Implements np.moveaxis with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::zeros::<i32>([2,3,4]).moveaxis(0,-1).shape(),[3,4,2]);
    /// ```
    pub fn moveaxis(&self, src: isize, dst: isize) -> Self {
        let s = axis_index(src, self.ndim());
        let d = axis_index(dst, self.ndim());
        let mut axes: Vec<_> = (0..self.ndim()).collect();
        let a = axes.remove(s);
        axes.insert(d, a);
        self.transpose(&axes)
    }
    /// Implements np.squeeze with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::zeros::<i32>([1,2,1]).squeeze().shape(),[2]);
    /// ```
    pub fn squeeze(&self) -> Self {
        let mut a = self.clone();
        for i in (0..a.ndim()).rev() {
            if a.shape[i] == 1 {
                a.shape.remove(i);
                a.strides.remove(i);
            }
        }
        a
    }
    /// Implements np.squeeze with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::zeros::<i32>([1,2,1]).squeeze_axis(-1).shape(),[1,2]);
    /// ```
    pub fn squeeze_axis(&self, axis: isize) -> Self {
        let axis = axis_index(axis, self.ndim());
        assert!(
            self.shape[axis] == 1,
            "cannot select an axis to squeeze out which has size not equal to one"
        );
        let mut a = self.clone();
        a.shape.remove(axis);
        a.strides.remove(axis);
        a
    }
    /// Implements np.expand_dims with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].expand_dims(-1).shape(),[2,1]);
    /// ```
    pub fn expand_dims(&self, axis: isize) -> Self {
        let axis = axis_index(axis, self.ndim() + 1);
        let mut a = self.clone();
        a.shape.insert(axis, 1);
        a.strides.insert(axis, 0);
        a
    }
    /// Implements np.flip with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3].flip(-1).to_vec(),[3,2,1]);
    /// ```
    pub fn flip(&self, axis: isize) -> Self {
        let axis = axis_index(axis, self.ndim());
        let mut a = self.clone();
        if !a.is_empty() {
            a.offset =
                (a.offset as isize + a.strides[axis] * (a.shape[axis] as isize - 1)) as usize;
        }
        a.strides[axis] = -a.strides[axis];
        a
    }
    /// Implements np.flip_all with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].flip_all().to_vec(),[4,3,2,1]);
    /// ```
    pub fn flip_all(&self) -> Self {
        let mut a = self.clone();
        for i in 0..self.ndim() {
            a = a.flip(i as isize)
        }
        a
    }
    /// Implements np.roll with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3].roll(1,0).to_vec(),[3,1,2]);
    /// ```
    pub fn roll(&self, shift: isize, axis: isize) -> Self {
        let axis = axis_index(axis, self.ndim());
        let n = self.shape[axis];
        if n == 0 {
            return self.copy();
        }
        let shift = shift.rem_euclid(n as isize) as usize;
        let indices: Vec<_> = (0..n).map(|i| (i + n - shift) % n).collect();
        self.take(&indices, axis as isize)
    }
    /// Implements np.repeat with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].repeat(2,0).to_vec(),[1,1,2,2]);
    /// ```
    pub fn repeat(&self, n: usize, axis: isize) -> Self {
        let axis = axis_index(axis, self.ndim());
        self.shape[axis].checked_mul(n).expect("array is too big");
        let indices: Vec<_> = (0..self.shape[axis])
            .flat_map(|i| std::iter::repeat_n(i, n))
            .collect();
        self.take(&indices, axis as isize)
    }
    /// Implements np.tile with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].tile(&[2]).to_vec(),[1,2,1,2]);
    /// ```
    pub fn tile(&self, reps: &[usize]) -> Self {
        let rank = self.ndim().max(reps.len());
        let mut base = vec![1; rank - self.ndim()];
        base.extend_from_slice(&self.shape);
        let mut repetitions = vec![1; rank - reps.len()];
        repetitions.extend_from_slice(reps);
        let shape: Vec<_> = base
            .iter()
            .zip(&repetitions)
            .map(|(&a, &b)| a.checked_mul(b).expect("array is too big"))
            .collect();
        let a = self.reshape(&base);
        let mut src = vec![0; rank];
        Self::from_shape_fn(&shape, |c| {
            for i in 0..rank {
                src[i] = c[i] % base[i];
            }
            a.get(&src)
        })
    }
    /// Implements np.sort with array value semantics.
    ///
    /// Returns a sorted copy; NaNs sort last.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![3,1,2].sort().to_vec(),[1,2,3]);
    /// ```
    pub fn sort(&self) -> Self {
        self.sort_axis(-1)
    }
    /// Implements np.sort with array value semantics.
    ///
    /// Returns a sorted copy; NaNs sort last.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[4,1],[2,3]].sort_axis(0).to_vec(),[2,1,4,3]);
    /// ```
    pub fn sort_axis(&self, axis: isize) -> Self {
        let axis = axis_index(axis, self.ndim());
        let mut out = self.copy();
        let mut shape = self.shape.clone();
        let n = shape.remove(axis);
        let mut st = out.strides.clone();
        let stride = st.remove(axis) as usize;
        let data = Arc::make_mut(&mut out.data);
        if stride == 1 {
            for base in Offsets::new(&shape, &st, 0) {
                sort_slice(&mut data[base..base + n]);
            }
        } else {
            let mut lane = Vec::with_capacity(n);
            for base in Offsets::new(&shape, &st, 0) {
                lane.clear();
                lane.extend((0..n).map(|i| data[base + i * stride]));
                sort_slice(&mut lane);
                for (i, &v) in lane.iter().enumerate() {
                    data[base + i * stride] = v;
                }
            }
        }
        out
    }
    /// Implements np.argsort with array value semantics.
    ///
    /// Stable: equal values keep their original relative order.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![3,1,1].argsort().to_vec(),[1,2,0]);
    /// ```
    pub fn argsort(&self) -> Array<usize> {
        self.argsort_axis(-1)
    }
    /// Implements np.argsort with array value semantics.
    ///
    /// Stable: equal values keep their original relative order.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[4,1],[2,3]].argsort_axis(0).to_vec(),[1,0,0,1]);
    /// ```
    pub fn argsort_axis(&self, axis: isize) -> Array<usize> {
        let axis = axis_index(axis, self.ndim());
        let a = self.to_contiguous();
        let mut out = crate::zeros::<usize>(&self.shape);
        let mut shape = self.shape.clone();
        let n = shape.remove(axis);
        let mut st = out.strides.clone();
        let stride = st.remove(axis) as usize;
        let values = a.as_slice().unwrap();
        let data = Arc::make_mut(&mut out.data);
        if stride == 1 {
            for base in Offsets::new(&shape, &st, 0) {
                argsort_slice(&values[base..base + n], &mut data[base..base + n]);
            }
        } else {
            let mut lane_vals = Vec::with_capacity(n);
            let mut lane_out = vec![0usize; n];
            for base in Offsets::new(&shape, &st, 0) {
                lane_vals.clear();
                lane_vals.extend((0..n).map(|i| values[base + i * stride]));
                argsort_slice(&lane_vals, &mut lane_out);
                for (i, &v) in lane_out.iter().enumerate() {
                    data[base + i * stride] = v;
                }
            }
        }
        out
    }
}
/// Implements np.concatenate with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::concatenate(&[&np::array![1,2],&np::array![3]],0).to_vec(),[1,2,3]);
/// ```
pub fn concatenate<T: Element>(arrays: &[&Array<T>], axis: isize) -> Array<T> {
    assert!(!arrays.is_empty(), "need at least one array to concatenate");
    assert!(
        arrays[0].ndim() > 0,
        "zero-dimensional arrays cannot be concatenated"
    );
    let axis = axis_index(axis, arrays[0].ndim());
    let mut shape = arrays[0].shape.clone();
    shape[axis] = 0;
    for a in arrays {
        assert!(
            a.ndim() == shape.len(),
            "all the input arrays must have same number of dimensions"
        );
        for (i, &n) in shape.iter().enumerate() {
            assert!(
                i == axis || a.shape[i] == n,
                "all the input array dimensions except for the concatenation axis must match exactly"
            );
        }
        shape[axis] = shape[axis]
            .checked_add(a.shape[axis])
            .expect("array is too big");
    }
    let mut src = vec![0; shape.len()];
    Array::from_shape_fn(&shape, |c| {
        let mut n = c[axis];
        for a in arrays {
            if n < a.shape[axis] {
                src.copy_from_slice(c);
                src[axis] = n;
                return a.get(&src);
            }
            n -= a.shape[axis];
        }
        unreachable!()
    })
}
/// Implements np.stack with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::stack(&[&np::array![1,2],&np::array![3,4]],-1).to_vec(),[1,3,2,4]);
/// ```
pub fn stack<T: Element>(arrays: &[&Array<T>], axis: isize) -> Array<T> {
    assert!(!arrays.is_empty(), "need at least one array to stack");
    let shape = arrays[0].shape();
    assert!(
        arrays.iter().all(|a| a.shape() == shape),
        "all input arrays must have the same shape"
    );
    let axis = axis_index(axis, shape.len() + 1);
    let views: Vec<_> = arrays
        .iter()
        .map(|a| a.expand_dims(axis as isize))
        .collect();
    concatenate(&views.iter().collect::<Vec<_>>(), axis as isize)
}
/// Implements np.vstack with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::vstack(&[&np::array![1,2],&np::array![3,4]]).shape(),[2,2]);
/// ```
pub fn vstack<T: Element>(arrays: &[&Array<T>]) -> Array<T> {
    let promoted: Vec<_> = arrays
        .iter()
        .map(|a| match a.ndim() {
            0 => a.reshape(&[1, 1]),
            1 => a.expand_dims(0),
            _ => (*a).clone(),
        })
        .collect();
    concatenate(&promoted.iter().collect::<Vec<_>>(), 0)
}
/// Implements np.hstack with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::hstack(&[&np::array![1,2],&np::array![3,4]]).to_vec(),[1,2,3,4]);
/// ```
pub fn hstack<T: Element>(arrays: &[&Array<T>]) -> Array<T> {
    let promoted: Vec<_> = arrays
        .iter()
        .map(|a| {
            if a.ndim() == 0 {
                a.reshape(&[1])
            } else {
                (*a).clone()
            }
        })
        .collect();
    let axis = if promoted.first().is_some_and(|a| a.ndim() > 1) {
        1
    } else {
        0
    };
    concatenate(&promoted.iter().collect::<Vec<_>>(), axis)
}
/// Implements np.split with array value semantics.
///
/// ```
/// use rustnum as np;
/// let parts=np::split(&np::array![1,2,3,4],2,0);
/// assert_eq!(parts[1].to_vec(),[3,4]);
/// ```
pub fn split<T: Element>(a: &Array<T>, sections: usize, axis: isize) -> Vec<Array<T>> {
    assert!(sections > 0, "number sections must be larger than 0");
    let ax = axis_index(axis, a.ndim());
    assert!(
        a.shape[ax].is_multiple_of(sections),
        "array split does not result in an equal division"
    );
    array_split(a, sections, axis)
}
/// Implements np.array_split with array value semantics.
///
/// ```
/// use rustnum as np;
/// let parts=np::array_split(&np::array![1,2,3,4,5],2,0);
/// assert_eq!(parts[0].to_vec(),[1,2,3]);
/// ```
pub fn array_split<T: Element>(a: &Array<T>, sections: usize, axis: isize) -> Vec<Array<T>> {
    assert!(sections > 0, "number sections must be larger than 0");
    let ax = axis_index(axis, a.ndim());
    let n = a.shape[ax];
    let mut start = 0;
    (0..sections)
        .map(|i| {
            let end = start + n / sections + usize::from(i < n % sections);
            let mut args = vec![SliceArg::all(); a.ndim()];
            args[ax] = SliceArg::Range {
                start: Some(start as isize),
                end: Some(end as isize),
                step: 1,
            };
            start = end;
            a.slice(&args)
        })
        .collect()
}
/// Implements np.unique with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::unique(&np::array![3,1,3,2]).to_vec(),[1,2,3]);
/// ```
pub fn unique<T: Element>(a: &Array<T>) -> Array<T> {
    unique_counts(a).0
}
/// Implements np.unique(return_counts=True) with array value semantics.
///
/// ```
/// use rustnum as np;
/// let (v,n)=np::unique_counts(&np::array![3,1,3,2]);
/// assert_eq!(v.to_vec(),[1,2,3]);assert_eq!(n.to_vec(),[1,1,2]);
/// ```
pub fn unique_counts<T: Element>(a: &Array<T>) -> (Array<T>, Array<usize>) {
    let mut v = a.to_vec();
    sort_slice(&mut v);
    let mut values = Vec::new();
    let mut counts = Vec::new();
    for x in v {
        if values
            .last()
            .is_some_and(|&y| x == y || (isnan(x) && isnan(y)))
        {
            *counts.last_mut().unwrap() += 1;
        } else {
            values.push(x);
            counts.push(1usize);
        }
    }
    (Array::from(values), Array::from(counts))
}
/// Implements np.nonzero with array value semantics.
///
/// ```
/// use rustnum as np;
/// let indices=np::nonzero(&np::array![[0,1],[2,0]]);
/// assert_eq!(indices[0].to_vec(),[0,1]);assert_eq!(indices[1].to_vec(),[1,0]);
/// ```
pub fn nonzero<T: Element>(a: &Array<T>) -> Vec<Array<usize>> {
    assert!(
        a.ndim() > 0,
        "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead."
    );
    let mut axes = vec![Vec::new(); a.ndim()];
    let mut c = vec![0; a.ndim()];
    for x in a.iter() {
        if x != T::default() {
            for (out, &i) in axes.iter_mut().zip(&c) {
                out.push(i);
            }
        }
        crate::layout::advance(&mut c, a.shape());
    }
    axes.into_iter().map(Array::from).collect()
}
/// Implements np.argwhere with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::argwhere(&np::array![[0,1],[2,0]]).to_vec(),[0,1,1,0]);
/// ```
pub fn argwhere<T: Element>(a: &Array<T>) -> Array<usize> {
    let mut data = Vec::new();
    let mut c = vec![0; a.ndim()];
    let mut n = 0;
    for x in a.iter() {
        if x != T::default() {
            data.extend_from_slice(&c);
            n += 1;
        }
        crate::layout::advance(&mut c, a.shape());
    }
    Array::from_vec(data, &[n, a.ndim()])
}
/// Implements np.searchsorted with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::searchsorted(&np::array![1,3,3,5],3),1);
/// ```
pub fn searchsorted<T: Element>(sorted: &Array<T>, value: T) -> usize {
    assert!(sorted.ndim() == 1, "object too deep for desired array");
    let mut lo = 0;
    let mut hi = sorted.size();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if compare(&sorted.get(&[mid]), &value) == Ordering::Less {
            lo = mid + 1
        } else {
            hi = mid
        }
    }
    lo
}
/// Implements np.pad with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::pad(&np::array![1,2],1,0).to_vec(),[0,1,2,0]);
/// ```
pub fn pad<T: Element>(a: &Array<T>, width: usize, value: T) -> Array<T> {
    let shape: Vec<_> = a
        .shape
        .iter()
        .map(|&d| {
            width
                .checked_mul(2)
                .and_then(|w| d.checked_add(w))
                .expect("array is too big")
        })
        .collect();
    let mut src = vec![0; a.ndim()];
    Array::from_shape_fn(&shape, |c| {
        if c.iter()
            .enumerate()
            .any(|(i, &v)| v < width || v - width >= a.shape[i])
        {
            return value;
        }
        for i in 0..c.len() {
            src[i] = c[i] - width;
        }
        a.get(&src)
    })
}

#[inline(always)]
fn f64_to_sort_key(x: f64) -> u64 {
    let u = x.to_bits();
    u ^ (((u as i64) >> 63) as u64 | (1u64 << 63))
}

#[inline(always)]
fn sort_key_to_f64(k: u64) -> f64 {
    let mask = (((k ^ (1u64 << 63)) as i64) >> 63) as u64 | (1u64 << 63);
    f64::from_bits(k ^ mask)
}

#[inline(always)]
fn f32_to_sort_key(x: f32) -> u32 {
    let u = x.to_bits();
    u ^ (((u as i32) >> 31) as u32 | (1u32 << 31))
}

#[inline(always)]
fn sort_key_to_f32(k: u32) -> f32 {
    let mask = (((k ^ (1u32 << 31)) as i32) >> 31) as u32 | (1u32 << 31);
    f32::from_bits(k ^ mask)
}

#[inline(always)]
fn i64_to_sort_key(x: i64) -> u64 {
    (x as u64) ^ (1u64 << 63)
}

#[inline(always)]
fn sort_key_to_i64(k: u64) -> i64 {
    (k ^ (1u64 << 63)) as i64
}

#[inline(always)]
fn i32_to_sort_key(x: i32) -> u32 {
    (x as u32) ^ (1u32 << 31)
}

#[inline(always)]
fn sort_key_to_i32(k: u32) -> i32 {
    (k ^ (1u32 << 31)) as i32
}

pub(crate) fn partition_nans_to_end<T: Element>(slice: &mut [T]) -> usize {
    let mut left = 0;
    let mut right = slice.len();
    while left < right {
        if isnan(slice[left]) {
            right -= 1;
            slice.swap(left, right);
        } else {
            left += 1;
        }
    }
    left
}

pub(crate) fn radix_sort_u64(slice: &mut [u64]) {
    let n = slice.len();
    if n <= 1 {
        return;
    }
    let mut buf = vec![0u64; n];
    let mut in_buf = false;

    for shift in (0..64).step_by(8) {
        let mut counts = [0usize; 256];
        let src = if in_buf { buf.as_slice() } else { &*slice };
        for &val in src {
            let bucket = ((val >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        let mut offsets = [0usize; 256];
        let mut sum = 0;
        for i in 0..256 {
            offsets[i] = sum;
            sum += counts[i];
        }

        if in_buf {
            for &val in buf.as_slice() {
                let bucket = ((val >> shift) & 0xFF) as usize;
                slice[offsets[bucket]] = val;
                offsets[bucket] += 1;
            }
            in_buf = false;
        } else {
            for &val in slice.iter() {
                let bucket = ((val >> shift) & 0xFF) as usize;
                buf[offsets[bucket]] = val;
                offsets[bucket] += 1;
            }
            in_buf = true;
        }
    }

    if in_buf {
        slice.copy_from_slice(&buf);
    }
}

pub(crate) fn radix_sort_u32(slice: &mut [u32]) {
    let n = slice.len();
    if n <= 1 {
        return;
    }
    let mut buf = vec![0u32; n];
    let mut in_buf = false;

    for shift in (0..32).step_by(8) {
        let mut counts = [0usize; 256];
        let src = if in_buf { buf.as_slice() } else { &*slice };
        for &val in src {
            let bucket = ((val >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        let mut offsets = [0usize; 256];
        let mut sum = 0;
        for i in 0..256 {
            offsets[i] = sum;
            sum += counts[i];
        }

        if in_buf {
            for &val in buf.as_slice() {
                let bucket = ((val >> shift) & 0xFF) as usize;
                slice[offsets[bucket]] = val;
                offsets[bucket] += 1;
            }
            in_buf = false;
        } else {
            for &val in slice.iter() {
                let bucket = ((val >> shift) & 0xFF) as usize;
                buf[offsets[bucket]] = val;
                offsets[bucket] += 1;
            }
            in_buf = true;
        }
    }

    if in_buf {
        slice.copy_from_slice(&buf);
    }
}

pub(crate) fn radix_sort_pairs_u64(slice: &mut [(u64, usize)]) {
    let n = slice.len();
    if n <= 1 {
        return;
    }
    let mut buf = vec![(0u64, 0usize); n];
    let mut in_buf = false;

    for shift in (0..64).step_by(8) {
        let mut counts = [0usize; 256];
        let src = if in_buf { buf.as_slice() } else { &*slice };
        for &(k, _) in src {
            let bucket = ((k >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        let mut offsets = [0usize; 256];
        let mut sum = 0;
        for i in 0..256 {
            offsets[i] = sum;
            sum += counts[i];
        }

        if in_buf {
            for &pair in buf.as_slice() {
                let bucket = ((pair.0 >> shift) & 0xFF) as usize;
                slice[offsets[bucket]] = pair;
                offsets[bucket] += 1;
            }
            in_buf = false;
        } else {
            for &pair in slice.iter() {
                let bucket = ((pair.0 >> shift) & 0xFF) as usize;
                buf[offsets[bucket]] = pair;
                offsets[bucket] += 1;
            }
            in_buf = true;
        }
    }

    if in_buf {
        slice.copy_from_slice(&buf);
    }
}

pub(crate) fn radix_sort_pairs_u32(slice: &mut [(u32, usize)]) {
    let n = slice.len();
    if n <= 1 {
        return;
    }
    let mut buf = vec![(0u32, 0usize); n];
    let mut in_buf = false;

    for shift in (0..32).step_by(8) {
        let mut counts = [0usize; 256];
        let src = if in_buf { buf.as_slice() } else { &*slice };
        for &(k, _) in src {
            let bucket = ((k >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        let mut offsets = [0usize; 256];
        let mut sum = 0;
        for i in 0..256 {
            offsets[i] = sum;
            sum += counts[i];
        }

        if in_buf {
            for &pair in buf.as_slice() {
                let bucket = ((pair.0 >> shift) & 0xFF) as usize;
                slice[offsets[bucket]] = pair;
                offsets[bucket] += 1;
            }
            in_buf = false;
        } else {
            for &pair in slice.iter() {
                let bucket = ((pair.0 >> shift) & 0xFF) as usize;
                buf[offsets[bucket]] = pair;
                offsets[bucket] += 1;
            }
            in_buf = true;
        }
    }

    if in_buf {
        slice.copy_from_slice(&buf);
    }
}

pub(crate) fn sort_slice<T: Element>(slice: &mut [T]) {
    let n = slice.len();
    if n <= 1 {
        return;
    }
    let valid_len = partition_nans_to_end(slice);
    if valid_len <= 1 {
        return;
    }
    let valid = &mut slice[..valid_len];

    #[cfg(feature = "rayon")]
    {
        use rayon::slice::ParallelSliceMut;
        if rayon::current_num_threads() > 1 && valid_len >= 4096 {
            valid.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            return;
        }
    }

    if valid_len >= (1 << 16) {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<f64>() {
            // SAFETY: TypeId matches f64, so T is f64 with identical size and alignment.
            let f_slice: &mut [f64] = unsafe {
                std::slice::from_raw_parts_mut(valid.as_mut_ptr() as *mut f64, valid_len)
            };
            let keys: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(f_slice.as_mut_ptr() as *mut u64, valid_len)
            };
            for x in keys.iter_mut() {
                *x = f64_to_sort_key(f64::from_bits(*x));
            }
            radix_sort_u64(keys);
            for x in keys.iter_mut() {
                let f = sort_key_to_f64(*x);
                *x = f.to_bits();
            }
            return;
        } else if tid == TypeId::of::<f32>() {
            // SAFETY: TypeId matches f32, so T is f32 with identical size and alignment.
            let f_slice: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(valid.as_mut_ptr() as *mut f32, valid_len)
            };
            let keys: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(f_slice.as_mut_ptr() as *mut u32, valid_len)
            };
            for x in keys.iter_mut() {
                *x = f32_to_sort_key(f32::from_bits(*x));
            }
            radix_sort_u32(keys);
            for x in keys.iter_mut() {
                let f = sort_key_to_f32(*x);
                *x = f.to_bits();
            }
            return;
        } else if tid == TypeId::of::<i64>() {
            // SAFETY: TypeId matches i64, so T is i64 with identical size and alignment.
            let i_slice: &mut [i64] = unsafe {
                std::slice::from_raw_parts_mut(valid.as_mut_ptr() as *mut i64, valid_len)
            };
            let keys: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(i_slice.as_mut_ptr() as *mut u64, valid_len)
            };
            for x in keys.iter_mut() {
                *x = i64_to_sort_key(*x as i64);
            }
            radix_sort_u64(keys);
            for x in keys.iter_mut() {
                *x = sort_key_to_i64(*x) as u64;
            }
            return;
        } else if tid == TypeId::of::<i32>() {
            // SAFETY: TypeId matches i32, so T is i32 with identical size and alignment.
            let i_slice: &mut [i32] = unsafe {
                std::slice::from_raw_parts_mut(valid.as_mut_ptr() as *mut i32, valid_len)
            };
            let keys: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(i_slice.as_mut_ptr() as *mut u32, valid_len)
            };
            for x in keys.iter_mut() {
                *x = i32_to_sort_key(*x as i32);
            }
            radix_sort_u32(keys);
            for x in keys.iter_mut() {
                *x = sort_key_to_i32(*x) as u32;
            }
            return;
        }
    }

    valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
}

pub(crate) fn argsort_slice<T: Element>(values: &[T], out: &mut [usize]) {
    let n = values.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        out[0] = 0;
        return;
    }

    let mut pairs = Vec::with_capacity(n);
    let mut nan_indices = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if isnan(v) {
            nan_indices.push(i);
        } else {
            pairs.push((v, i));
        }
    }

    #[cfg(feature = "rayon")]
    {
        use rayon::slice::ParallelSliceMut;
        if rayon::current_num_threads() > 1 && pairs.len() >= 4096 {
            pairs.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            for (i, &(_, idx)) in pairs.iter().enumerate() {
                out[i] = idx;
            }
            let offset = pairs.len();
            for (i, &idx) in nan_indices.iter().enumerate() {
                out[offset + i] = idx;
            }
            return;
        }
    }

    if pairs.len() >= (1 << 16) {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<f64>() {
            let mut u64_pairs: Vec<(u64, usize)> = pairs
                .iter()
                .map(|&(v, idx)| {
                    // SAFETY: T is f64
                    let f: f64 = unsafe { *(&v as *const T as *const f64) };
                    (f64_to_sort_key(f), idx)
                })
                .collect();
            radix_sort_pairs_u64(&mut u64_pairs);
            for (i, &(_, idx)) in u64_pairs.iter().enumerate() {
                out[i] = idx;
            }
            let offset = u64_pairs.len();
            for (i, &idx) in nan_indices.iter().enumerate() {
                out[offset + i] = idx;
            }
            return;
        } else if tid == TypeId::of::<f32>() {
            let mut u32_pairs: Vec<(u32, usize)> = pairs
                .iter()
                .map(|&(v, idx)| {
                    // SAFETY: T is f32
                    let f: f32 = unsafe { *(&v as *const T as *const f32) };
                    (f32_to_sort_key(f), idx)
                })
                .collect();
            radix_sort_pairs_u32(&mut u32_pairs);
            for (i, &(_, idx)) in u32_pairs.iter().enumerate() {
                out[i] = idx;
            }
            let offset = u32_pairs.len();
            for (i, &idx) in nan_indices.iter().enumerate() {
                out[offset + i] = idx;
            }
            return;
        } else if tid == TypeId::of::<i64>() {
            let mut u64_pairs: Vec<(u64, usize)> = pairs
                .iter()
                .map(|&(v, idx)| {
                    // SAFETY: T is i64
                    let val: i64 = unsafe { *(&v as *const T as *const i64) };
                    (i64_to_sort_key(val), idx)
                })
                .collect();
            radix_sort_pairs_u64(&mut u64_pairs);
            for (i, &(_, idx)) in u64_pairs.iter().enumerate() {
                out[i] = idx;
            }
            let offset = u64_pairs.len();
            for (i, &idx) in nan_indices.iter().enumerate() {
                out[offset + i] = idx;
            }
            return;
        } else if tid == TypeId::of::<i32>() {
            let mut u32_pairs: Vec<(u32, usize)> = pairs
                .iter()
                .map(|&(v, idx)| {
                    // SAFETY: T is i32
                    let val: i32 = unsafe { *(&v as *const T as *const i32) };
                    (i32_to_sort_key(val), idx)
                })
                .collect();
            radix_sort_pairs_u32(&mut u32_pairs);
            for (i, &(_, idx)) in u32_pairs.iter().enumerate() {
                out[i] = idx;
            }
            let offset = u32_pairs.len();
            for (i, &idx) in nan_indices.iter().enumerate() {
                out[offset + i] = idx;
            }
            return;
        }
    }

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    for (i, &(_, idx)) in pairs.iter().enumerate() {
        out[i] = idx;
    }
    let offset = pairs.len();
    for (i, &idx) in nan_indices.iter().enumerate() {
        out[offset + i] = idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shape_views_and_copies() {
        let a = crate::array![[1, 2], [3, 4]];
        assert!(Arc::ptr_eq(&a.data, &a.reshape(&[4]).data));
        assert!(!Arc::ptr_eq(&a.data, &a.t().reshape(&[4]).data));
    }
    #[test]
    fn stable_nan_sort() {
        let a = crate::array![2., f64::NAN, 1., 2., f64::NAN];
        assert_eq!(a.argsort().to_vec(), [2, 0, 3, 1, 4]);
    }
    #[test]
    fn stacks_promote_vectors() {
        let a = crate::array![1, 2];
        let b = crate::array![3, 4];
        assert_eq!(vstack(&[&a, &b]).shape(), [2, 2]);
        assert_eq!(hstack(&[&a, &b]).to_vec(), [1, 2, 3, 4]);
    }
    #[test]
    fn tile_alignment() {
        assert_eq!(
            crate::array![1, 2].tile(&[2, 2]).to_vec(),
            [1, 2, 1, 2, 1, 2, 1, 2]
        );
    }
    #[test]
    fn unequal_split() {
        let a = crate::array![0, 1, 2, 3, 4];
        let v = array_split(&a, 3, 0);
        assert_eq!(
            v.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
    }
}
