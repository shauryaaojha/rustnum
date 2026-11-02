//! NaN-aware reductions that complete NumPy's `nan*` family.
use crate::lanes::map_lanes;
use crate::slicing::axis_index;
use crate::{Array, Float};

fn finite_sorted<F: Float>(lane: &[F]) -> Vec<F> {
    let mut v: Vec<F> = lane.iter().copied().filter(|x| !x.is_nan()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}
/// Linear-interpolated quantile `q` in [0, 1] of an ascending, NaN-free slice (NumPy "linear").
fn lerp_quantile<F: Float>(v: &[F], q: f64) -> F {
    if v.is_empty() {
        return F::NAN;
    }
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return v[lo];
    }
    let t = F::from_f64(pos - lo as f64);
    if t < F::from_f64(0.5) {
        v[lo] + (v[hi] - v[lo]) * t
    } else {
        v[hi] - (v[hi] - v[lo]) * (F::ONE - t)
    }
}
fn check_percentile(q: f64) {
    assert!(
        (0.0..=100.0).contains(&q),
        "Percentiles must be in the range [0, 100]"
    );
}
fn nan_var<F: Float>(lane: &[F], ddof: usize) -> F {
    let (mut sum, mut n) = (F::ZERO, 0usize);
    for &x in lane.iter().filter(|x| !x.is_nan()) {
        sum = sum + x;
        n += 1;
    }
    let mean = sum / F::from_usize(n);
    let mut ss = F::ZERO;
    for &x in lane.iter().filter(|x| !x.is_nan()) {
        let d = x - mean;
        ss = ss + d * d;
    }
    ss / F::from_usize(n.saturating_sub(ddof))
}
fn nan_arg<F: Float>(lane: impl Iterator<Item = F>, max: bool) -> usize {
    let mut best: Option<(usize, F)> = None;
    let mut n = 0;
    for (i, x) in lane.enumerate() {
        n += 1;
        if x.is_nan() {
            continue;
        }
        let better = match best {
            None => true,
            Some((_, b)) => {
                if max {
                    x > b
                } else {
                    x < b
                }
            }
        };
        if better {
            best = Some((i, x));
        }
    }
    assert!(n > 0, "attempt to get argmax of an empty sequence");
    best.expect("All-NaN slice encountered").0
}
impl<F: Float> Array<F> {
    /// Variance ignoring NaNs, like np.nanvar(a).
    ///
    /// An all-NaN input gives NaN (NumPy also warns).
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nanvar(),1.);
    /// ```
    pub fn nanvar(&self) -> F {
        self.nanvar_ddof(0)
    }
    /// Variance ignoring NaNs with delta degrees of freedom, like np.nanvar(a, ddof=ddof).
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nanvar_ddof(1),2.);
    /// ```
    pub fn nanvar_ddof(&self, ddof: usize) -> F {
        nan_var(&self.to_vec(), ddof)
    }
    /// Standard deviation ignoring NaNs, like np.nanstd(a).
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nanstd(),1.);
    /// ```
    pub fn nanstd(&self) -> F {
        self.nanvar().sqrt()
    }
    /// Standard deviation ignoring NaNs with ddof, like np.nanstd(a, ddof=ddof).
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nanstd_ddof(1),2f64.sqrt());
    /// ```
    pub fn nanstd_ddof(&self, ddof: usize) -> F {
        self.nanvar_ddof(ddof).sqrt()
    }
    /// Variance ignoring NaNs along an axis, like np.nanvar(a, axis=axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.,f64::NAN],[3.,5.]];
    /// assert_eq!(a.nanvar_axis(1).to_vec(),[0.,1.]);
    /// ```
    pub fn nanvar_axis(&self, axis: isize) -> Array<F> {
        self.nanvar_ddof_axis(axis, 0)
    }
    /// Variance ignoring NaNs along an axis with ddof, like np.nanvar(a, axis=axis, ddof=ddof).
    ///
    /// Argument order is (axis, ddof).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.,f64::NAN,3.],[3.,5.,7.]];
    /// assert_eq!(a.nanvar_ddof_axis(1,1).to_vec(),[2.,4.]);
    /// ```
    pub fn nanvar_ddof_axis(&self, axis: isize, ddof: usize) -> Array<F> {
        let ax = axis_index(axis, self.ndim());
        map_lanes(self, ax, 1, |l, o| o[0] = nan_var(l, ddof)).squeeze_axis(axis)
    }
    /// Standard deviation ignoring NaNs along an axis, like np.nanstd(a, axis=axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.,f64::NAN],[3.,5.]];
    /// assert_eq!(a.nanstd_axis(1).to_vec(),[0.,1.]);
    /// ```
    pub fn nanstd_axis(&self, axis: isize) -> Array<F> {
        self.nanvar_axis(axis).into_sqrt()
    }
    /// Standard deviation ignoring NaNs along an axis with ddof, like np.nanstd(a, axis=axis, ddof=ddof).
    ///
    /// Argument order is (axis, ddof).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.,f64::NAN,3.],[3.,5.,7.]];
    /// assert_eq!(a.nanstd_ddof_axis(1,1).to_vec(),[2f64.sqrt(),2.]);
    /// ```
    pub fn nanstd_ddof_axis(&self, axis: isize, ddof: usize) -> Array<F> {
        self.nanvar_ddof_axis(axis, ddof).into_sqrt()
    }
    /// Median ignoring NaNs, like np.nanmedian(a).
    ///
    /// An all-NaN or empty input gives NaN (NumPy also warns).
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.,10.].nanmedian(),3.);
    /// ```
    pub fn nanmedian(&self) -> F {
        lerp_quantile(&finite_sorted(&self.to_vec()), 0.5)
    }
    /// Median ignoring NaNs along an axis, like np.nanmedian(a, axis=axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.,f64::NAN,3.],[4.,5.,6.]];
    /// assert_eq!(a.nanmedian_axis(1).to_vec(),[2.,5.]);
    /// ```
    pub fn nanmedian_axis(&self, axis: isize) -> Array<F> {
        let ax = axis_index(axis, self.ndim());
        map_lanes(self, ax, 1, |l, o| {
            o[0] = lerp_quantile(&finite_sorted(l), 0.5)
        })
        .squeeze_axis(axis)
    }
    /// Percentile ignoring NaNs, like np.nanpercentile(a, q).
    ///
    /// Panics unless `q` is in [0, 100]. An all-NaN or empty input gives NaN.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.,7.,9.].nanpercentile(25.),2.5);
    /// ```
    pub fn nanpercentile(&self, q: F) -> F {
        let q = q.to_f64();
        check_percentile(q);
        lerp_quantile(&finite_sorted(&self.to_vec()), q / 100.0)
    }
    /// Percentile ignoring NaNs along an axis, like np.nanpercentile(a, q, axis=axis).
    ///
    /// Argument order is (q, axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.,f64::NAN,3.],[4.,5.,6.]];
    /// assert_eq!(a.nanpercentile_axis(50.,1).to_vec(),[2.,5.]);
    /// ```
    pub fn nanpercentile_axis(&self, q: F, axis: isize) -> Array<F> {
        let q = q.to_f64();
        check_percentile(q);
        let ax = axis_index(axis, self.ndim());
        map_lanes(self, ax, 1, |l, o| {
            o[0] = lerp_quantile(&finite_sorted(l), q / 100.0)
        })
        .squeeze_axis(axis)
    }
    /// Index of the maximum ignoring NaNs, like np.nanargmax(a).
    ///
    /// Panics with "All-NaN slice encountered" when every value is NaN.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![f64::NAN,1.,5.,2.].nanargmax(),2);
    /// ```
    pub fn nanargmax(&self) -> usize {
        nan_arg(self.iter(), true)
    }
    /// Index of the minimum ignoring NaNs, like np.nanargmin(a).
    ///
    /// Panics with "All-NaN slice encountered" when every value is NaN.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![f64::NAN,1.,5.,2.].nanargmin(),1);
    /// ```
    pub fn nanargmin(&self) -> usize {
        nan_arg(self.iter(), false)
    }
    /// Indices of the maxima ignoring NaNs along an axis, like np.nanargmax(a, axis=axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[f64::NAN,1.],[4.,2.]];
    /// assert_eq!(a.nanargmax_axis(1).to_vec(),[1,0]);
    /// ```
    pub fn nanargmax_axis(&self, axis: isize) -> Array<usize> {
        let ax = axis_index(axis, self.ndim());
        map_lanes(self, ax, 1, |l, o| o[0] = nan_arg(l.iter().copied(), true)).squeeze_axis(axis)
    }
    /// Indices of the minima ignoring NaNs along an axis, like np.nanargmin(a, axis=axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[f64::NAN,1.],[4.,2.]];
    /// assert_eq!(a.nanargmin_axis(1).to_vec(),[1,1]);
    /// ```
    pub fn nanargmin_axis(&self, axis: isize) -> Array<usize> {
        let ax = axis_index(axis, self.ndim());
        map_lanes(self, ax, 1, |l, o| o[0] = nan_arg(l.iter().copied(), false)).squeeze_axis(axis)
    }
    /// Product ignoring NaNs (treated as 1), like np.nanprod(a).
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![2.,f64::NAN,3.].nanprod(),6.);
    /// ```
    pub fn nanprod(&self) -> F {
        self.iter()
            .fold(F::ONE, |p, x| if x.is_nan() { p } else { p * x })
    }
    /// Product ignoring NaNs along an axis, like np.nanprod(a, axis=axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[2.,f64::NAN],[3.,4.]];
    /// assert_eq!(a.nanprod_axis(1).to_vec(),[2.,12.]);
    /// ```
    pub fn nanprod_axis(&self, axis: isize) -> Array<F> {
        let ax = axis_index(axis, self.ndim());
        map_lanes(self, ax, 1, |l, o| {
            o[0] = l
                .iter()
                .fold(F::ONE, |p, &x| if x.is_nan() { p } else { p * x })
        })
        .squeeze_axis(axis)
    }
    /// Cumulative sum treating NaN as zero over the flattened array, like np.nancumsum(a).
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1.,f64::NAN,3.].nancumsum().to_vec(),[1.,1.,4.]);
    /// ```
    pub fn nancumsum(&self) -> Array<F> {
        let mut acc = F::ZERO;
        self.iter()
            .map(|x| {
                if !x.is_nan() {
                    acc = acc + x;
                }
                acc
            })
            .collect()
    }
    /// Cumulative product treating NaN as one over the flattened array, like np.nancumprod(a).
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![2.,f64::NAN,3.].nancumprod().to_vec(),[2.,2.,6.]);
    /// ```
    pub fn nancumprod(&self) -> Array<F> {
        let mut acc = F::ONE;
        self.iter()
            .map(|x| {
                if !x.is_nan() {
                    acc = acc * x;
                }
                acc
            })
            .collect()
    }
    /// Cumulative sum treating NaN as zero along an axis, like np.nancumsum(a, axis=axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[1.,f64::NAN],[3.,4.]];
    /// assert_eq!(a.nancumsum_axis(1).to_vec(),[1.,1.,3.,7.]);
    /// ```
    pub fn nancumsum_axis(&self, axis: isize) -> Array<F> {
        let ax = axis_index(axis, self.ndim());
        map_lanes(self, ax, self.shape[ax], |l, o| {
            let mut acc = F::ZERO;
            for (d, &x) in o.iter_mut().zip(l) {
                if !x.is_nan() {
                    acc = acc + x;
                }
                *d = acc;
            }
        })
    }
    /// Cumulative product treating NaN as one along an axis, like np.nancumprod(a, axis=axis).
    /// ```
    /// use rustnum as np;
    /// let a=np::array![[2.,f64::NAN],[3.,4.]];
    /// assert_eq!(a.nancumprod_axis(1).to_vec(),[2.,2.,3.,12.]);
    /// ```
    pub fn nancumprod_axis(&self, axis: isize) -> Array<F> {
        let ax = axis_index(axis, self.ndim());
        map_lanes(self, ax, self.shape[ax], |l, o| {
            let mut acc = F::ONE;
            for (d, &x) in o.iter_mut().zip(l) {
                if !x.is_nan() {
                    acc = acc * x;
                }
                *d = acc;
            }
        })
    }
}
