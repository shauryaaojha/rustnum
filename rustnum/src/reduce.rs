use crate::element::{add, isnan, maximum_value, minimum_value, mul, sub};
use crate::layout::Offsets;
use crate::slicing::axis_index;
use crate::{Array, Element, Float, Num};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Eight accumulators and a balanced tree keep f32 error comparable to NumPy.
fn pairwise<T: Num>(x: &[T]) -> T {
    if x.len() < 8 {
        return x.iter().copied().fold(T::ZERO, add);
    }
    if x.len() <= 128 {
        let mut a = [T::ZERO; 8];
        a.copy_from_slice(&x[..8]);
        let mut i = 8;
        while i + 8 <= x.len() {
            for j in 0..8 {
                a[j] = add(a[j], x[i + j]);
            }
            i += 8;
        }
        let s = add(
            add(add(a[0], a[1]), add(a[2], a[3])),
            add(add(a[4], a[5]), add(a[6], a[7])),
        );
        return x[i..].iter().copied().fold(s, add);
    }
    let mid = (x.len() / 2) / 8 * 8;
    #[cfg(feature = "rayon")]
    if x.len() >= crate::kernels_mem::REDUCE_ALL {
        let (a, b) = rayon::join(|| pairwise(&x[..mid]), || pairwise(&x[mid..]));
        return add(a, b);
    }
    add(pairwise(&x[..mid]), pairwise(&x[mid..]))
}
// Balanced summation over a virtual slice avoids materializing strided lanes.
fn pairwise_by<T: Num>(n: usize, f: &impl Fn(usize) -> T) -> T {
    fn visit<T: Num>(start: usize, n: usize, f: &impl Fn(usize) -> T) -> T {
        if n <= 128 {
            let mut acc = [T::ZERO; 8];
            for i in 0..n {
                acc[i % 8] = add(acc[i % 8], f(start + i));
            }
            add(
                add(add(acc[0], acc[1]), add(acc[2], acc[3])),
                add(add(acc[4], acc[5]), add(acc[6], acc[7])),
            )
        } else {
            let mid = n / 2 / 8 * 8;
            add(visit(start, mid, f), visit(start + mid, n - mid, f))
        }
    }
    visit(0, n, f)
}
fn sum_rows<T: Num>(data: &[T], stride: usize, rows: usize, out: &mut [T]) {
    if rows <= 16 {
        out.fill(T::ZERO);
        for row in 0..rows {
            for (x, &y) in out.iter_mut().zip(&data[row * stride..]) {
                *x = add(*x, y);
            }
        }
    } else {
        let mid = rows / 2;
        sum_rows(data, stride, mid, out);
        let mut right = [T::ZERO; 128];
        sum_rows(
            &data[mid * stride..],
            stride,
            rows - mid,
            &mut right[..out.len()],
        );
        for (x, &y) in out.iter_mut().zip(&right) {
            *x = add(*x, y);
        }
    }
}
impl<T: Element> Array<T> {
    fn axis_kernel<U: Element>(
        &self,
        axis: isize,
        f: impl Fn(usize, isize, usize) -> U + Sync + Send,
    ) -> Array<U> {
        let axis = axis_index(axis, self.ndim());
        let mut shape = self.shape.clone();
        let n = shape.remove(axis);
        let mut data = crate::pool::output(crate::layout::size(&shape));
        let step = self.strides[axis];
        let compute = |i: usize| {
            let mut index = i;
            let mut base = self.offset as isize;
            for d in (0..self.ndim()).rev() {
                if d != axis {
                    base += (index % self.shape[d]) as isize * self.strides[d];
                    index /= self.shape[d];
                }
            }
            f(base as usize, step, n)
        };
        #[cfg(feature = "rayon")]
        if self.size() >= crate::kernels_mem::REDUCE_LANES {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = compute(i));
        } else {
            for (i, x) in data.iter_mut().enumerate() {
                *x = compute(i);
            }
        }
        #[cfg(not(feature = "rayon"))]
        for (i, x) in data.iter_mut().enumerate() {
            *x = compute(i);
        }
        Array::from_owned_shape(data, shape)
    }
    fn axis_fold<U: Element>(
        &self,
        axis: isize,
        init: U,
        f: impl Fn(U, T) -> U + Sync + Send,
    ) -> Array<U> {
        self.axis_kernel(axis, |base, step, n| {
            (0..n).fold(init, |acc, i| {
                f(acc, self.data[(base as isize + i as isize * step) as usize])
            })
        })
    }
    pub(crate) fn lanes<U: Element>(
        &self,
        axis: isize,
        mut f: impl FnMut(&Array<T>) -> U,
    ) -> Array<U> {
        let axis = axis_index(axis, self.ndim());
        let mut shape = self.shape.clone();
        let n = shape.remove(axis);
        let mut strides = self.strides.clone();
        let stride = strides.remove(axis);
        let offsets = Offsets::new(&shape, &strides, self.offset);
        let mut lane = Array {
            data: self.data.clone(),
            recycle: self.recycle,
            shape: vec![n],
            strides: vec![stride],
            offset: self.offset,
        };
        let values = offsets
            .map(|offset| {
                lane.offset = if n == 0 { 0 } else { offset };
                f(&lane)
            })
            .collect();
        Array::from_vec(values, &shape)
    }
    /// Implements np.min with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![3,1,2].min(),1);
    /// ```
    pub fn min(&self) -> T {
        self.extreme(false)
    }
    /// Implements np.max with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![3,1,2].max(),3);
    /// ```
    pub fn max(&self) -> T {
        self.extreme(true)
    }
    fn extreme(&self, max: bool) -> T {
        assert!(
            !self.is_empty(),
            "zero-size array to reduction operation {} which has no identity",
            if max { "maximum" } else { "minimum" }
        );
        let combine = |a, b| {
            if max {
                maximum_value(a, b)
            } else {
                minimum_value(a, b)
            }
        };
        if let Some(s) = self.as_slice() {
            #[cfg(feature = "rayon")]
            if s.len() >= crate::kernels_mem::REDUCE_ALL {
                return s.par_iter().copied().reduce_with(combine).unwrap();
            }
            return s.iter().copied().reduce(combine).unwrap();
        }
        self.iter().reduce(combine).unwrap()
    }
    /// Implements np.argmin with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![3,1,2].argmin(),1);
    /// ```
    pub fn argmin(&self) -> usize {
        self.arg_extreme(false)
    }
    /// Implements np.argmax with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,3,2].argmax(),1);
    /// ```
    pub fn argmax(&self) -> usize {
        self.arg_extreme(true)
    }
    fn arg_extreme(&self, max: bool) -> usize {
        assert!(
            !self.is_empty(),
            "zero-size array to reduction operation {} which has no identity",
            if max { "maximum" } else { "minimum" }
        );
        let step = |best: (usize, T), (i, x): (usize, T)| {
            if !isnan(best.1) && (isnan(x) || if max { x > best.1 } else { x < best.1 }) {
                (i, x)
            } else {
                best
            }
        };
        if let Some(s) = self.as_slice() {
            return s.iter().copied().enumerate().fold((0, s[0]), step).0;
        }
        self.iter()
            .enumerate()
            .fold((0, self.data[self.offset]), step)
            .0
    }
    /// Implements np.min with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.min_axis(1).to_vec(),[1.,3.]);
    /// ```
    pub fn min_axis(&self, axis: isize) -> Array<T> {
        let a = axis_index(axis, self.ndim());
        assert!(
            self.shape[a] > 0,
            "zero-size array to reduction operation minimum which has no identity"
        );
        self.axis_kernel(axis, |base, step, n| {
            (1..n).fold(self.data[base], |acc, i| {
                minimum_value(acc, self.data[(base as isize + i as isize * step) as usize])
            })
        })
    }
    /// Implements np.max with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.max_axis(1).to_vec(),[2.,4.]);
    /// ```
    pub fn max_axis(&self, axis: isize) -> Array<T> {
        let a = axis_index(axis, self.ndim());
        assert!(
            self.shape[a] > 0,
            "zero-size array to reduction operation maximum which has no identity"
        );
        self.axis_kernel(axis, |base, step, n| {
            (1..n).fold(self.data[base], |acc, i| {
                maximum_value(acc, self.data[(base as isize + i as isize * step) as usize])
            })
        })
    }
    /// Implements np.argmin with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.argmin_axis(1).to_vec(),[0,0]);
    /// ```
    pub fn argmin_axis(&self, axis: isize) -> Array<usize> {
        let a = axis_index(axis, self.ndim());
        assert!(
            self.shape[a] > 0,
            "zero-size array to reduction operation minimum which has no identity"
        );
        self.axis_kernel(axis, |base, step, n| {
            let mut best = 0;
            let mut value = self.data[base];
            for i in 1..n {
                let x = self.data[(base as isize + i as isize * step) as usize];
                if !isnan(value) && (isnan(x) || x < value) {
                    best = i;
                    value = x;
                }
            }
            best
        })
    }
    /// Implements np.argmax with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.argmax_axis(1).to_vec(),[1,1]);
    /// ```
    pub fn argmax_axis(&self, axis: isize) -> Array<usize> {
        let a = axis_index(axis, self.ndim());
        assert!(
            self.shape[a] > 0,
            "zero-size array to reduction operation maximum which has no identity"
        );
        self.axis_kernel(axis, |base, step, n| {
            let mut best = 0;
            let mut value = self.data[base];
            for i in 1..n {
                let x = self.data[(base as isize + i as isize * step) as usize];
                if !isnan(value) && (isnan(x) || x > value) {
                    best = i;
                    value = x;
                }
            }
            best
        })
    }
    /// Implements np.any with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::array![0,2].any());
    /// ```
    pub fn any(&self) -> bool {
        if let Some(s) = self.as_slice() {
            s.iter().any(|x| *x != T::default())
        } else {
            self.iter().any(|x| x != T::default())
        }
    }
    /// Implements np.all with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(!np::array![0,2].all());
    /// ```
    pub fn all(&self) -> bool {
        if let Some(s) = self.as_slice() {
            s.iter().all(|x| *x != T::default())
        } else {
            self.iter().all(|x| x != T::default())
        }
    }
    /// Implements np.any with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.any_axis(1).to_vec(),[true,true]);
    /// ```
    pub fn any_axis(&self, axis: isize) -> Array<bool> {
        self.axis_fold(axis, false, |acc, x| acc || x != T::default())
    }
    /// Implements np.all with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.all_axis(1).to_vec(),[true,true]);
    /// ```
    pub fn all_axis(&self, axis: isize) -> Array<bool> {
        self.axis_fold(axis, true, |acc, x| acc && x != T::default())
    }
    /// Implements np.count_nonzero with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0,2,3].count_nonzero(),2);
    /// ```
    pub fn count_nonzero(&self) -> usize {
        if let Some(s) = self.as_slice() {
            s.iter().filter(|&&x| x != T::default()).count()
        } else {
            self.iter().filter(|&x| x != T::default()).count()
        }
    }
}
impl<T: Num> Array<T> {
    /// Implements np.sum with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3].sum(),6);
    /// ```
    pub fn sum(&self) -> T {
        if let Some(s) = self.as_slice() {
            pairwise(s)
        } else {
            pairwise_by(self.size(), &|i| self.flat_value(i))
        }
    }
    /// Implements np.prod with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3].prod(),6);
    /// ```
    pub fn prod(&self) -> T {
        if let Some(s) = self.as_slice() {
            #[cfg(feature = "rayon")]
            if s.len() >= crate::kernels_mem::REDUCE_ALL {
                return s.par_iter().copied().reduce(|| T::ONE, mul);
            }
            return s.iter().copied().fold(T::ONE, mul);
        }
        self.iter().fold(T::ONE, mul)
    }
    /// Implements np.ptp with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![2,5,9].ptp(),7);
    /// ```
    pub fn ptp(&self) -> T {
        sub(self.max(), self.min())
    }
    /// Implements np.sum with array value semantics.
    ///
    /// Use result.expand_dims(axis) to retain the reduced dimension.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.sum_axis(1).to_vec(),[3.,7.]);
    /// ```
    pub fn sum_axis(&self, axis: isize) -> Array<T> {
        let axis = axis_index(axis, self.ndim());
        if let Some(input) = self.as_slice() {
            let inner = crate::layout::size(&self.shape[axis + 1..]);
            let mut shape = self.shape.clone();
            let rows = shape.remove(axis);
            let mut out = crate::pool::filled(crate::layout::size(&shape), T::ZERO);
            if rows > 0 && inner > 0 {
                if inner == 1 {
                    #[cfg(feature = "rayon")]
                    if input.len() >= crate::kernels_mem::REDUCE_LANES {
                        out.par_iter_mut()
                            .zip(input.par_chunks(rows))
                            .for_each(|(o, row)| *o = pairwise(row));
                    } else {
                        for (o, row) in out.iter_mut().zip(input.chunks(rows)) {
                            *o = pairwise(row);
                        }
                    }
                    #[cfg(not(feature = "rayon"))]
                    for (o, row) in out.iter_mut().zip(input.chunks(rows)) {
                        *o = pairwise(row);
                    }
                } else {
                    // Split output into disjoint column blocks; each task streams
                    // complete row segments. Middle axes add independent outer batches.
                    let run = |outer: usize, dst: &mut [T]| {
                        let input = &input[outer * rows * inner..];
                        #[cfg(feature = "rayon")]
                        if input.len() >= crate::kernels_mem::REDUCE_ROWS {
                            dst.par_chunks_mut(128)
                                .enumerate()
                                .for_each(|(c, out)| sum_rows(&input[c * 128..], inner, rows, out));
                            return;
                        }
                        dst.chunks_mut(128)
                            .enumerate()
                            .for_each(|(c, out)| sum_rows(&input[c * 128..], inner, rows, out));
                    };
                    #[cfg(feature = "rayon")]
                    if out.len() / inner > 1 && input.len() >= crate::kernels_mem::REDUCE_ROWS {
                        out.par_chunks_mut(inner)
                            .enumerate()
                            .for_each(|(i, dst)| run(i, dst));
                    } else {
                        out.chunks_mut(inner)
                            .enumerate()
                            .for_each(|(i, dst)| run(i, dst));
                    }
                    #[cfg(not(feature = "rayon"))]
                    out.chunks_mut(inner)
                        .enumerate()
                        .for_each(|(i, dst)| run(i, dst));
                }
            }
            return Array::from_owned_shape(out, shape);
        }
        self.axis_kernel(axis as isize, |base, step, n| {
            pairwise_by(n, &|i| {
                self.data[(base as isize + i as isize * step) as usize]
            })
        })
    }
    /// Implements np.prod with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.prod_axis(1).to_vec(),[2.,12.]);
    /// ```
    pub fn prod_axis(&self, axis: isize) -> Array<T> {
        self.axis_fold(axis, T::ONE, mul)
    }
    /// Implements np.cumsum with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3].cumsum().to_vec(),[1,3,6]);
    /// ```
    pub fn cumsum(&self) -> Array<T> {
        self.cumulative_flat(false)
    }
    /// Implements np.cumprod with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3].cumprod().to_vec(),[1,2,6]);
    /// ```
    pub fn cumprod(&self) -> Array<T> {
        self.cumulative_flat(true)
    }
    fn cumulative_flat(&self, prod: bool) -> Array<T> {
        let mut s = if prod { T::ONE } else { T::ZERO };
        let mut step = |x| {
            s = if prod { mul(s, x) } else { add(s, x) };
            s
        };
        let data = if let Some(x) = self.as_slice() {
            x.iter().copied().map(&mut step).collect::<Vec<_>>()
        } else {
            self.iter().map(step).collect()
        };
        Array::from(data)
    }
    /// Implements np.cumsum with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.cumsum_axis(1).to_vec(),[1.,3.,3.,7.]);
    /// ```
    pub fn cumsum_axis(&self, axis: isize) -> Array<T> {
        self.cumulative_axis(axis, false)
    }
    /// Implements np.cumprod with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.cumprod_axis(1).to_vec(),[1.,2.,3.,12.]);
    /// ```
    pub fn cumprod_axis(&self, axis: isize) -> Array<T> {
        self.cumulative_axis(axis, true)
    }
    fn cumulative_axis(&self, axis: isize, prod: bool) -> Array<T> {
        let axis = axis_index(axis, self.ndim());
        let mut result = self.copy();
        let mut shape = self.shape.clone();
        let n = shape.remove(axis);
        let mut st = result.strides.clone();
        let stride = st.remove(axis) as usize;
        let data = std::sync::Arc::make_mut(&mut result.data);
        for base in Offsets::new(&shape, &st, 0) {
            let mut s = if prod { T::ONE } else { T::ZERO };
            for i in 0..n {
                let j = base + i * stride;
                s = if prod {
                    mul(s, data[j])
                } else {
                    add(s, data[j])
                };
                data[j] = s;
            }
        }
        result
    }
}
impl<F: Float> Array<F> {
    /// Implements np.mean with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,2.,3.].mean(),2.);
    /// ```
    pub fn mean(&self) -> F {
        self.sum() / F::from_usize(self.size())
    }
    /// Implements np.var with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,3.].var(),1.);
    /// ```
    pub fn var(&self) -> F {
        self.var_ddof(0)
    }
    /// Implements np.std with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,3.].std(),1.);
    /// ```
    pub fn std(&self) -> F {
        self.var().sqrt()
    }
    /// Implements np.var with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,3.].var_ddof(1),2.);
    /// ```
    pub fn var_ddof(&self, ddof: usize) -> F {
        let m = self.mean();
        pairwise_by(self.size(), &|i| {
            let d = self.flat_value(i) - m;
            d * d
        }) / F::from_usize(self.size().saturating_sub(ddof))
    }
    /// Implements np.std with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.0f64,3.].std_ddof(1),2f64.sqrt());
    /// ```
    pub fn std_ddof(&self, ddof: usize) -> F {
        self.var_ddof(ddof).sqrt()
    }
    /// Implements np.mean with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.mean_axis(1).to_vec(),[1.5,3.5]);
    /// ```
    pub fn mean_axis(&self, axis: isize) -> Array<F> {
        let n = self.shape[axis_index(axis, self.ndim())];
        self.sum_axis(axis).into_map(|x| x / F::from_usize(n))
    }
    /// Implements np.var with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.var_axis(1).to_vec(),[0.25,0.25]);
    /// ```
    pub fn var_axis(&self, axis: isize) -> Array<F> {
        self.var_ddof_axis(axis, 0)
    }
    /// Implements np.std with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.std_axis(1).to_vec(),[0.5,0.5]);
    /// ```
    pub fn std_axis(&self, axis: isize) -> Array<F> {
        self.std_ddof_axis(axis, 0)
    }
    /// Implements np.var with array value semantics.
    ///
    /// Argument order is (axis, ddof).
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.var_ddof_axis(1,1).to_vec(),[0.5,0.5]);
    /// ```
    pub fn var_ddof_axis(&self, axis: isize, ddof: usize) -> Array<F> {
        self.axis_kernel(axis, |base, step, n| {
            let value = |i| self.data[(base as isize + i as isize * step) as usize];
            let mean = pairwise_by(n, &value) / F::from_usize(n);
            pairwise_by(n, &|i| {
                let d = value(i) - mean;
                d * d
            }) / F::from_usize(n.saturating_sub(ddof))
        })
    }
    /// Implements np.std with array value semantics.
    ///
    /// Argument order is (axis, ddof).
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.std_ddof_axis(1,1).to_vec(),[0.5f64.sqrt(),0.5f64.sqrt()]);
    /// ```
    pub fn std_ddof_axis(&self, axis: isize, ddof: usize) -> Array<F> {
        self.var_ddof_axis(axis, ddof).into_sqrt()
    }
    /// Implements np.median with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,7.,3.,9.].median(),5.);
    /// ```
    pub fn median(&self) -> F {
        self.percentile(F::from_usize(50))
    }
    /// Implements np.percentile with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,3.,7.,9.].percentile(25.),2.5);
    /// ```
    pub fn percentile(&self, q: F) -> F {
        let q = q.to_f64();
        assert!(
            (0.0..=100.0).contains(&q),
            "Percentiles must be in the range [0, 100]"
        );
        if self.is_empty() {
            return F::NAN;
        }
        let mut v = self.to_vec();
        if v.iter().any(|x| x.is_nan()) {
            return F::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = q / 100.0 * (v.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            return v[lo];
        }
        let fraction = F::from_f64(pos - lo as f64);
        if fraction < F::from_f64(0.5) {
            v[lo] + (v[hi] - v[lo]) * fraction
        } else {
            v[hi] - (v[hi] - v[lo]) * (F::ONE - fraction)
        }
    }
    /// Implements np.median with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.median_axis(1).to_vec(),[1.5,3.5]);
    /// ```
    pub fn median_axis(&self, axis: isize) -> Array<F> {
        self.lanes(axis, |a| a.median())
    }
    /// Implements np.nansum with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nansum(),4.);
    /// ```
    pub fn nansum(&self) -> F {
        pairwise_by(self.size(), &|i| {
            let x = self.flat_value(i);
            if x.is_nan() { F::ZERO } else { x }
        })
    }
    /// Implements np.nanmean with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nanmean(),2.);
    /// ```
    pub fn nanmean(&self) -> F {
        let n = if let Some(s) = self.as_slice() {
            s.iter().filter(|x| !x.is_nan()).count()
        } else {
            self.iter().filter(|x| !x.is_nan()).count()
        };
        self.nansum() / F::from_usize(n)
    }
    /// Implements np.nanmin with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nanmin(),1.);
    /// ```
    pub fn nanmin(&self) -> F {
        assert!(
            !self.is_empty(),
            "zero-size array to reduction operation minimum which has no identity"
        );
        (0..self.size())
            .map(|i| self.flat_value(i))
            .fold(F::NAN, |acc, x| {
                if x.is_nan() {
                    acc
                } else if acc.is_nan() {
                    x
                } else {
                    minimum_value(acc, x)
                }
            })
    }
    /// Implements np.nanmax with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nanmax(),3.);
    /// ```
    pub fn nanmax(&self) -> F {
        assert!(
            !self.is_empty(),
            "zero-size array to reduction operation maximum which has no identity"
        );
        (0..self.size())
            .map(|i| self.flat_value(i))
            .fold(F::NAN, |acc, x| {
                if x.is_nan() {
                    acc
                } else if acc.is_nan() {
                    x
                } else {
                    maximum_value(acc, x)
                }
            })
    }
    /// Implements np.nansum with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.nansum_axis(1).to_vec(),[3.,7.]);
    /// ```
    pub fn nansum_axis(&self, axis: isize) -> Array<F> {
        self.axis_kernel(axis, |base, step, n| {
            pairwise_by(n, &|i| {
                let x = self.data[(base as isize + i as isize * step) as usize];
                if x.is_nan() { F::ZERO } else { x }
            })
        })
    }
    /// Implements np.nanmean with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.nanmean_axis(1).to_vec(),[1.5,3.5]);
    /// ```
    pub fn nanmean_axis(&self, axis: isize) -> Array<F> {
        self.axis_kernel(axis, |base, step, n| {
            let value = |i| self.data[(base as isize + i as isize * step) as usize];
            let count = (0..n).filter(|&i| !value(i).is_nan()).count();
            pairwise_by(n, &|i| {
                let x = value(i);
                if x.is_nan() { F::ZERO } else { x }
            }) / F::from_usize(count)
        })
    }
    /// Implements np.nanmin with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.nanmin_axis(1).to_vec(),[1.,3.]);
    /// ```
    pub fn nanmin_axis(&self, axis: isize) -> Array<F> {
        let a = axis_index(axis, self.ndim());
        assert!(
            self.shape[a] > 0,
            "zero-size array to reduction operation minimum which has no identity"
        );
        self.axis_fold(axis, F::NAN, |acc, x| {
            if x.is_nan() {
                acc
            } else if acc.is_nan() {
                x
            } else {
                minimum_value(acc, x)
            }
        })
    }
    /// Implements np.nanmax with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.0f64,2.],[3.,4.]];
    /// assert_eq!(a.nanmax_axis(1).to_vec(),[2.,4.]);
    /// ```
    pub fn nanmax_axis(&self, axis: isize) -> Array<F> {
        let a = axis_index(axis, self.ndim());
        assert!(
            self.shape[a] > 0,
            "zero-size array to reduction operation maximum which has no identity"
        );
        self.axis_fold(axis, F::NAN, |acc, x| {
            if x.is_nan() {
                acc
            } else if acc.is_nan() {
                x
            } else {
                maximum_value(acc, x)
            }
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pairwise_accuracy() {
        let mut v = vec![1f32; 100_000];
        v[0] = 100_000_000.;
        assert!((Array::from(v).sum() - 100_100_000.).abs() <= 16.);
    }
    #[test]
    fn nan_first_index() {
        assert_eq!(crate::array![1., f64::NAN, 0.].argmin(), 1);
    }
    #[test]
    fn empty_identities() {
        let a = Array::<f64>::default();
        assert_eq!(a.sum(), 0.);
        assert_eq!(a.prod(), 1.);
        assert!(a.mean().is_nan());
        assert!(a.all());
        assert!(!a.any());
    }
    #[test]
    fn ddof_exhaustion() {
        assert!(crate::array![1.].var_ddof(2).is_nan());
        assert!(crate::array![1., 2.].var_ddof(2).is_infinite());
    }
    #[test]
    fn quantiles() {
        let a = crate::array![1., 3., 7., 9.];
        assert_eq!(a.median(), 5.);
        assert_eq!(a.percentile(25.), 2.5);
    }
}
