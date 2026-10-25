use crate::broadcast::broadcast_shapes;
#[cfg(feature = "rayon")]
use crate::layout::PAR_THRESHOLD;
use crate::layout::{Offsets, advance, contiguous, size, strides};
use crate::{Element, IntoArray};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{
    any::{Any, TypeId},
    ops::{Index, IndexMut},
    sync::Arc,
};
/// A homogeneous array, like np.ndarray, with copy-on-write value semantics.
///
/// Views share storage until mutation; writing a view never changes its parent.
/// ```
/// use rustnum as np;
/// let mut a: np::Array<i32> = (1..=4).collect();
/// let b = a.clone();
/// a[[0]] = 9;
/// a[&[1][..]] = 8;
/// assert_eq!(b.to_vec(), vec![1,2,3,4]);
/// assert_eq!((&b).into_iter().sum::<i32>(), 10);
/// assert!(np::Array::<i32>::default().is_empty());
/// ```
#[derive(Clone)]
pub struct Array<T> {
    pub(crate) data: Arc<Vec<T>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) offset: usize,
}
impl<T: Element> Array<T> {
    /// Implements np.array(...).reshape with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::Array::from_vec(vec![1,2,3,4], &[2,2]);
    /// assert_eq!(a[[1,0]],3);
    /// ```
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Self {
        assert!(
            data.len() == size(shape),
            "cannot reshape array of size {} into shape {:?}",
            data.len(),
            shape
        );
        Self {
            data: Arc::new(data),
            shape: shape.to_vec(),
            strides: strides(shape),
            offset: 0,
        }
    }
    /// Implements np.fromfunction with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::Array::from_shape_fn([2,3], |c| (c[0]*10+c[1]) as i32);
    /// assert_eq!(a.to_vec(),[0,1,2,10,11,12]);
    /// ```
    pub fn from_shape_fn(shape: impl AsRef<[usize]>, mut f: impl FnMut(&[usize]) -> T) -> Self {
        let sh = shape.as_ref();
        let mut c = vec![0; sh.len()];
        let data = (0..size(sh))
            .map(|_| {
                let v = f(&c);
                advance(&mut c, sh);
                v
            })
            .collect();
        Self::from_vec(data, sh)
    }
    /// Implements np.array with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::Array::scalar(7);
    /// assert_eq!(a.shape(), &[]);
    /// assert_eq!(a.item(),7);
    /// ```
    pub fn scalar(value: T) -> Self {
        Self::from_vec(vec![value], &[])
    }
    /// Implements np.shape with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].shape(),[2,2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Implements np.ndim with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Array::scalar(1).ndim(),0);
    /// ```
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    /// Implements np.size with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].size(),4);
    /// ```
    pub fn size(&self) -> usize {
        size(&self.shape)
    }
    /// Implements np.len with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2,3],[4,5,6]].len(),2);
    /// ```
    pub fn len(&self) -> usize {
        *self.shape.first().expect("len() of unsized object")
    }
    /// Implements np.ndarray.size == 0 with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(!np::Array::scalar(0).is_empty());
    /// assert!(np::Array::<i32>::default().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }
    /// Implements np.ndarray.strides (in elements) with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::zeros::<i32>([2,3]).strides(),[3,1]);
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }
    /// Implements np.ndarray.dtype with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.0f64].dtype(),"float64");
    /// ```
    pub fn dtype(&self) -> &'static str {
        T::DTYPE
    }
    /// Implements np.ndarray.flags.c_contiguous with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(!np::array![[1,2],[3,4]].t().is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        contiguous(&self.shape, &self.strides)
    }
    /// Storage, element offset of the first element and element strides, for zero-copy kernels.
    ///
    /// Element `index` lives at `data[offset + sum(index[i] * strides[i])]`; strides may be
    /// zero (broadcast) or negative (flipped views).
    pub(crate) fn raw_parts(&self) -> (&[T], usize, &[isize]) {
        (&self.data, self.offset, &self.strides)
    }
    pub(crate) fn offset_of(&self, index: &[usize]) -> usize {
        assert!(
            index.len() == self.ndim(),
            "incorrect number of indices for array: array is {}-dimensional, but {} were indexed",
            self.ndim(),
            index.len()
        );
        let mut off = self.offset as isize;
        for (axis, &i) in index.iter().enumerate() {
            assert!(
                i < self.shape[axis],
                "index {} is out of bounds for axis {} with size {}",
                i,
                axis,
                self.shape[axis]
            );
            off += self.strides[axis] * i as isize;
        }
        off as usize
    }
    // Materialize overlapping layouts before writes so logical elements stay independent.
    pub(crate) fn prepare_write(&mut self) {
        if self
            .shape
            .iter()
            .zip(&self.strides)
            .any(|(&d, &s)| d > 1 && s == 0)
        {
            *self = self.copy();
        }
        Arc::make_mut(&mut self.data);
    }
    /// Implements np.ndarray.__getitem__ with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].get(&[1,0]),3);
    /// ```
    pub fn get(&self, index: &[usize]) -> T {
        self.data[self.offset_of(index)]
    }
    /// Implements np.ndarray.__setitem__ with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut a=np::array![1,2]; a.set(&[0],9);
    /// assert_eq!(a.to_vec(),[9,2]);
    /// ```
    pub fn set(&mut self, index: &[usize], value: T) {
        self.offset_of(index);
        self.prepare_write();
        let off = self.offset_of(index);
        Arc::make_mut(&mut self.data)[off] = value;
    }
    /// Implements np.item with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[5]].item(),5);
    /// ```
    pub fn item(&self) -> T {
        assert!(
            self.size() == 1,
            "can only convert an array of size 1 to a Python scalar"
        );
        self.data[self.offset]
    }
    /// Implements np.ndarray.ravel().tolist with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].t().to_vec(),[1,3,2,4]);
    /// ```
    pub fn to_vec(&self) -> Vec<T> {
        if let Some(s) = self.as_slice() {
            s.to_vec()
        } else {
            self.iter().collect()
        }
    }
    /// Implements np.ascontiguousarray with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![1,2];
    /// assert_eq!(a.as_slice(),Some(&[1,2][..]));
    /// ```
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.is_empty() {
            return Some(&self.data[0..0]);
        }
        self.is_contiguous()
            .then(|| &self.data[self.offset..self.offset + self.size()])
    }
    /// Implements np.ndarray.flat with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3].iter().sum::<i32>(),6);
    /// ```
    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + use<T> {
        let data = self.data.clone();
        Offsets::new(&self.shape, &self.strides, self.offset).map(move |i| data[i])
    }
    /// Implements np.ascontiguousarray with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1,2],[3,4]].t().to_contiguous();
    /// assert!(a.is_contiguous()); assert_eq!(a.to_vec(),[1,3,2,4]);
    /// ```
    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            self.copy()
        }
    }
    /// Implements np.ndarray.astype with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.9,-2.1].astype::<i32>().to_vec(),[1,-2]);
    /// ```
    pub fn astype<U: Element>(&self) -> Array<U> {
        if TypeId::of::<T>() == TypeId::of::<U>() {
            let boxed: Box<dyn Any> = Box::new(self.copy());
            return *boxed.downcast::<Array<U>>().expect("identical type");
        }
        self.par_map(|v| U::from_f64(v.to_f64()))
    }
    /// Implements np.vectorize with array value semantics.
    ///
    /// Public callbacks execute serially and may capture non-Sync state; built-in ufuncs use parallel kernels.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].map(|x|x*3).to_vec(),[3,6]);
    /// ```
    pub fn map<U: Element>(&self, f: impl Fn(T) -> U) -> Array<U> {
        let data = if let Some(s) = self.as_slice() {
            s.iter().copied().map(f).collect()
        } else {
            self.iter().map(f).collect()
        };
        Array::from_vec(data, &self.shape)
    }
    /// Implements np.vectorize with assignment with array value semantics.
    ///
    /// Public callbacks execute serially and may capture non-Sync state; built-in ufuncs use parallel kernels.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![1,2]; let mut b=a.clone(); b.mapv_inplace(|x|x*2);
    /// assert_eq!(b.to_vec(),[2,4]); assert_eq!(a.to_vec(),[1,2]);
    /// ```
    pub fn mapv_inplace(&mut self, f: impl Fn(T) -> T) {
        self.prepare_write();
        let off = self.offset;
        let n = self.size();
        let contiguous = self.is_contiguous();
        let offsets = Offsets::new(&self.shape, &self.strides, off);
        let data = Arc::make_mut(&mut self.data);
        if contiguous {
            for v in &mut data[off..off + n] {
                *v = f(*v)
            }
        } else {
            for i in offsets {
                data[i] = f(data[i]);
            }
        }
    }
    /// Implements np.ndarray.fill with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut a=np::zeros::<i32>([2,2]); a.fill(7);
    /// assert_eq!(a.to_vec(),[7,7,7,7]);
    /// ```
    pub fn fill(&mut self, value: T) {
        #[cfg(feature = "rayon")]
        if self.is_contiguous() && self.size() >= PAR_THRESHOLD {
            self.prepare_write();
            let end = self.offset + self.size();
            Arc::make_mut(&mut self.data)[self.offset..end]
                .par_iter_mut()
                .for_each(|x| *x = value);
            return;
        }
        self.mapv_inplace(|_| value)
    }
    /// Implements np.ndarray.copy with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![1,2]; let mut b=a.copy(); b[[0]]=8;
    /// assert_eq!(a[[0]],1);
    /// ```
    pub fn copy(&self) -> Self {
        Self::from_vec(self.to_vec(), &self.shape)
    }
    pub(crate) fn replace_c_order(&mut self, values: Vec<T>) {
        self.prepare_write();
        *Arc::make_mut(&mut self.data) = values;
        self.strides = strides(&self.shape);
        self.offset = 0;
    }
    /// Implements np.vectorize with broadcasting with array value semantics.
    ///
    /// Public callbacks execute serially and may capture non-Sync state; built-in ufuncs use parallel kernels.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1],[2]].zip_map([10,20],|a,b|a+b).to_vec(),[11,21,12,22]);
    /// ```
    pub fn zip_map<U: Element, R: Element>(
        &self,
        other: impl IntoArray<U>,
        f: impl Fn(T, U) -> R,
    ) -> Array<R> {
        let other = other.into_array();
        let shape = self.common_shape(&other);
        let a = self.broadcast_to(&shape);
        let b = other.broadcast_to(&shape);
        let data = match (a.as_slice(), b.as_slice()) {
            (Some(x), Some(y)) => x
                .iter()
                .copied()
                .zip(y.iter().copied())
                .map(|(x, y)| f(x, y))
                .collect(),
            _ => a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect(),
        };
        Array::from_vec(data, &shape)
    }
    #[cfg(feature = "rayon")]
    pub(crate) fn flat_value(&self, mut i: usize) -> T {
        if self.is_contiguous() {
            return self.data[self.offset + i];
        }
        let mut offset = self.offset as isize;
        for (&n, &s) in self.shape.iter().zip(&self.strides).rev() {
            offset += (i % n) as isize * s;
            i /= n;
        }
        self.data[offset as usize]
    }
    pub(crate) fn common_shape<U: Element>(&self, b: &Array<U>) -> Vec<usize> {
        broadcast_shapes(&self.shape, &b.shape).unwrap_or_else(|| {
            panic!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                self.shape, b.shape
            )
        })
    }
    pub(crate) fn par_map<U: Element>(&self, f: impl Fn(T) -> U + Sync + Send) -> Array<U> {
        #[cfg(feature = "rayon")]
        if let Some(s) = self.as_slice()
            && s.len() >= PAR_THRESHOLD
        {
            return Array::from_vec(s.par_iter().map(|&x| f(x)).collect(), &self.shape);
        }
        self.map(f)
    }
    pub(crate) fn par_zip<U: Element, R: Element>(
        &self,
        b: impl IntoArray<U>,
        f: impl Fn(T, U) -> R + Sync + Send,
    ) -> Array<R> {
        let b = b.into_array();
        let shape = self.common_shape(&b);
        #[cfg(feature = "rayon")]
        if size(&shape) >= PAR_THRESHOLD
            && let (Some(x), Some(y)) = (self.as_slice(), b.as_slice())
        {
            if self.shape == shape && b.shape == shape {
                return Array::from_vec(
                    x.par_iter()
                        .zip(y.par_iter())
                        .map(|(&x, &y)| f(x, y))
                        .collect(),
                    &shape,
                );
            }
            if self.size() == 1 && b.shape == shape {
                let v = self.item();
                return b.par_map(|y| f(v, y));
            }
            if b.size() == 1 && self.shape == shape {
                let v = b.item();
                return self.par_map(|x| f(x, v));
            }
            let a = self.broadcast_to(&shape);
            let b = b.broadcast_to(&shape);
            return Array::from_vec(
                (0..size(&shape))
                    .into_par_iter()
                    .map(|i| f(a.flat_value(i), b.flat_value(i)))
                    .collect(),
                &shape,
            );
        }
        if self.size() == 1 && b.shape == shape {
            let v = self.item();
            return b.map(|y| f(v, y));
        }
        if b.size() == 1 && self.shape == shape {
            let v = b.item();
            return self.map(|x| f(x, v));
        }
        self.zip_map(b, f)
    }
    /// Implements np.broadcast_to with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].broadcast_to(&[2,2]).to_vec(),[1,2,1,2]);
    /// ```
    pub fn broadcast_to(&self, target: &[usize]) -> Self {
        assert!(
            target.len() >= self.ndim()
                && broadcast_shapes(&self.shape, target).as_deref() == Some(target),
            "operands could not be broadcast together with shapes {:?} {:?}",
            self.shape,
            target
        );
        let pad = target.len() - self.ndim();
        let mut st = vec![0; target.len()];
        for i in 0..self.ndim() {
            st[pad + i] = if self.shape[i] == 1 {
                0
            } else {
                self.strides[i]
            };
        }
        Self {
            data: self.data.clone(),
            shape: target.to_vec(),
            strides: st,
            offset: self.offset,
        }
    }
}
impl<T: Element> Default for Array<T> {
    fn default() -> Self {
        Self::from_vec(vec![], &[0])
    }
}
impl<T: Element> From<Vec<T>> for Array<T> {
    fn from(v: Vec<T>) -> Self {
        let n = v.len();
        Self::from_vec(v, &[n])
    }
}
impl<T: Element> FromIterator<T> for Array<T> {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Self {
        Self::from(it.into_iter().collect::<Vec<_>>())
    }
}
impl<T: Element> PartialEq for Array<T> {
    fn eq(&self, b: &Self) -> bool {
        self.shape == b.shape && self.iter().eq(b.iter())
    }
}
impl<T: Element, const N: usize> Index<[usize; N]> for Array<T> {
    type Output = T;
    fn index(&self, i: [usize; N]) -> &T {
        &self.data[self.offset_of(&i)]
    }
}
impl<T: Element, const N: usize> IndexMut<[usize; N]> for Array<T> {
    fn index_mut(&mut self, i: [usize; N]) -> &mut T {
        self.offset_of(&i);
        self.prepare_write();
        let off = self.offset_of(&i);
        &mut Arc::make_mut(&mut self.data)[off]
    }
}
impl<T: Element> Index<&[usize]> for Array<T> {
    type Output = T;
    fn index(&self, i: &[usize]) -> &T {
        &self.data[self.offset_of(i)]
    }
}
impl<T: Element> IndexMut<&[usize]> for Array<T> {
    fn index_mut(&mut self, i: &[usize]) -> &mut T {
        self.offset_of(i);
        self.prepare_write();
        let off = self.offset_of(i);
        &mut Arc::make_mut(&mut self.data)[off]
    }
}
impl<T: Element> IntoIterator for &Array<T> {
    type Item = T;
    type IntoIter = Box<dyn Iterator<Item = T>>;
    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_dtype_preserves_large_integers() {
        let a = Array::from(vec![i64::MAX]);
        assert_eq!(a.astype::<i64>().item(), i64::MAX);
    }
    #[test]
    fn broadcast_mutation_does_not_alias() {
        let a = Array::scalar(1).broadcast_to(&[3]);
        let mut b = a.clone();
        b[[1]] = 7;
        assert_eq!(b.to_vec(), [1, 7, 1]);
        assert_eq!(a.to_vec(), [1, 1, 1]);
    }
    #[test]
    fn contiguous_shares_storage() {
        let a = Array::from(vec![1, 2]);
        assert!(Arc::ptr_eq(&a.data, &a.to_contiguous().data));
        assert!(!Arc::ptr_eq(&a.data, &a.copy().data));
    }
    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_threshold_dispatch() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let small = crate::ones::<i32>([PAR_THRESHOLD - 1]);
        let _ = small.par_map(|x| {
            assert!(rayon::current_thread_index().is_none());
            x
        });
        let large = crate::ones::<i32>([PAR_THRESHOLD]);
        let output = large.par_map(|x| {
            if rayon::current_thread_index().is_some() {
                calls.fetch_add(1, Ordering::Relaxed);
            }
            x + 1
        });
        assert_eq!(calls.load(Ordering::Relaxed), PAR_THRESHOLD);
        assert_eq!(output.sum(), 2 * PAR_THRESHOLD as i32);
        calls.store(0, Ordering::Relaxed);
        let left = crate::ones::<i32>([128, 1]);
        let right = crate::ones::<i32>([256]);
        let output = left.par_zip(&right, |x, y| {
            if rayon::current_thread_index().is_some() {
                calls.fetch_add(1, Ordering::Relaxed);
            }
            x + y
        });
        assert_eq!(calls.load(Ordering::Relaxed), PAR_THRESHOLD);
        assert_eq!(output.shape(), [128, 256]);
    }
    #[cfg(not(feature = "rayon"))]
    #[test]
    fn serial_kernel_fallback() {
        let a = crate::ones::<i32>([32768]);
        assert_eq!(a.par_map(|x| x + 1).sum(), 65536);
        assert_eq!(a.par_zip(2, |x, y| x * y).sum(), 65536);
    }
}
