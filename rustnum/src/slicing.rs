use crate::layout::Offsets;
use crate::{Array, Element, IntoArray};
use std::{ops::*, sync::Arc};
/// An index or slice, like a NumPy index tuple entry.
/// ```
/// use rustnum::{SliceArg, array};
/// assert_eq!(array![1,2,3].slice(&[SliceArg::Index(-1)]).item(),3);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SliceArg {
    /// Selects one index, like a NumPy integer index.
    /// ```
    /// let a=rustnum::SliceArg::Index(-1); assert!(matches!(a,rustnum::SliceArg::Index(-1)));
    /// ```
    Index(isize),
    /// Selects a half-open range, like a NumPy slice.
    /// ```
    /// let a=rustnum::SliceArg::Range{start:None,end:None,step:-1}; assert_eq!(rustnum::array![1,2].slice(&[a]).to_vec(),[2,1]);
    /// ```
    Range {
        /// The optional start, like NumPy slice.start.
        /// ```
        /// let s=rustnum::SliceArg::from(1..);
        /// assert_eq!(rustnum::array![1,2].slice(&[s]).item(),2);
        /// ```
        start: Option<isize>,
        /// The optional stop, like NumPy slice.stop.
        /// ```
        /// let s=rustnum::SliceArg::from(..1);
        /// assert_eq!(rustnum::array![1,2].slice(&[s]).item(),1);
        /// ```
        end: Option<isize>,
        /// The nonzero increment, like NumPy slice.step.
        /// ```
        /// assert_eq!(rustnum::array![1,2,3].slice(&rustnum::s![..;2]).to_vec(),[1,3]);
        /// ```
        step: isize,
    },
    /// Selects an inclusive Rust range, equivalent to a NumPy slice with an adjusted stop.
    /// ```
    /// let a=rustnum::SliceArg::from(0..=1); assert_eq!(rustnum::array![1,2,3].slice(&[a]).to_vec(),[1,2]);
    /// ```
    Inclusive {
        /// The optional start, like NumPy slice.start.
        /// ```
        /// let s=rustnum::SliceArg::from(1..);
        /// assert_eq!(rustnum::array![1,2].slice(&[s]).item(),2);
        /// ```
        start: Option<isize>,
        /// The included endpoint, like NumPy slice.stop adjusted by one.
        /// ```
        /// let s=rustnum::SliceArg::from(..=1);
        /// assert_eq!(rustnum::array![1,2,3].slice(&[s]).to_vec(),[1,2]);
        /// ```
        end: isize,
        /// The nonzero increment, like NumPy slice.step.
        /// ```
        /// assert_eq!(rustnum::array![1,2,3].slice(&rustnum::s![..;2]).to_vec(),[1,3]);
        /// ```
        step: isize,
    },
    /// Inserts a singleton axis, like np.newaxis.
    /// ```
    /// assert_eq!(rustnum::array![1,2].slice(&[rustnum::SliceArg::NewAxis]).shape(),[1,2]);
    /// ```
    NewAxis,
}
/// Inserts a singleton axis, like np.newaxis.
/// ```
/// use rustnum as np;
/// assert_eq!(np::array![1,2].slice(&np::s![.., np::NEW_AXIS]).shape(),[2,1]);
/// ```
pub const NEW_AXIS: SliceArg = SliceArg::NewAxis;
impl SliceArg {
    /// Implements np.all with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2].slice(&[np::SliceArg::all()]).to_vec(),[1,2]);
    /// ```
    pub fn all() -> Self {
        Self::Range {
            start: None,
            end: None,
            step: 1,
        }
    }
}
impl From<RangeFull> for SliceArg {
    fn from(_: RangeFull) -> Self {
        Self::all()
    }
}
macro_rules! conversions {
    ($($t:ty),*)=>{$(
        impl From<$t> for SliceArg {fn from(i:$t)->Self{Self::Index(isize::try_from(i).expect("index is too large"))}}
        impl From<Range<$t>> for SliceArg {fn from(r:Range<$t>)->Self{Self::Range{start:Some(isize::try_from(r.start).expect("index is too large")),end:Some(isize::try_from(r.end).expect("index is too large")),step:1}}}
        impl From<RangeFrom<$t>> for SliceArg {fn from(r:RangeFrom<$t>)->Self{Self::Range{start:Some(isize::try_from(r.start).expect("index is too large")),end:None,step:1}}}
        impl From<RangeTo<$t>> for SliceArg {fn from(r:RangeTo<$t>)->Self{Self::Range{start:None,end:Some(isize::try_from(r.end).expect("index is too large")),step:1}}}
        impl From<RangeInclusive<$t>> for SliceArg {fn from(r:RangeInclusive<$t>)->Self{Self::Inclusive{start:Some(isize::try_from(*r.start()).expect("index is too large")),end:isize::try_from(*r.end()).expect("index is too large"),step:1}}}
        impl From<RangeToInclusive<$t>> for SliceArg {fn from(r:RangeToInclusive<$t>)->Self{Self::Inclusive{start:None,end:isize::try_from(r.end).expect("index is too large"),step:1}}}
    )*}
}
conversions!(usize, isize, i32, i64);
/// Builds a slice tuple, like NumPy's a[start:stop:step, ...] syntax.
/// ```
/// use rustnum as np;
/// let i:usize=1; let j:i64=-1;
/// let a=np::arange(0,24,1).reshape(&[4,2,3]);
/// assert_eq!(a.slice(&np::s![i..i+2, .., j]).to_vec(),[8,11,14,17]);
/// assert_eq!(np::array![0,1,2,3].slice(&np::s![1..;-1]).to_vec(),[1,0]);
/// ```
#[macro_export]
macro_rules! s {
    ($($arg:expr $(; $step:expr)?),* $(,)?) => {
        ::std::vec::Vec::<$crate::SliceArg>::from([$({
            #[allow(unused_mut, clippy::reversed_empty_ranges)]
            let mut arg=$crate::SliceArg::from($arg);
            $(match &mut arg {
                $crate::SliceArg::Range{step,..}|$crate::SliceArg::Inclusive{step,..}=>{
                    *step=isize::try_from($step).expect("slice step is too large");
                }
                _=>panic!("step requires a slice"),
            })?
            arg
        }),*])
    };
}
fn normalize(
    start: Option<isize>,
    end: Option<isize>,
    step: isize,
    n: usize,
    inclusive: bool,
) -> (isize, usize) {
    assert!(step != 0, "slice step cannot be zero");
    let n = n as i128;
    let step = step as i128;
    let lo = if step < 0 { -1 } else { 0 };
    let hi = if step < 0 { n - 1 } else { n };
    let adjust = |x: isize| {
        let x = x as i128;
        if x < 0 { x + n } else { x }
    };
    let a = start.map_or(if step < 0 { n - 1 } else { 0 }, |x| {
        adjust(x).clamp(lo, hi)
    });
    let b = end.map_or(if step < 0 { -1 } else { n }, |x| {
        let raw = adjust(x) + if inclusive { step.signum() } else { 0 };
        raw.clamp(lo, hi)
    });
    let count = if step > 0 && b > a {
        (b - a - 1) / step + 1
    } else if step < 0 && a > b {
        (a - b - 1) / (-step) + 1
    } else {
        0
    };
    (a as isize, count as usize)
}
impl<T: Element> Array<T> {
    /// Implements np.ndarray.__getitem__ with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![0,1,2,3].slice(&np::s![..;-1]).to_vec(),[3,2,1,0]);
    /// ```
    pub fn slice(&self, args: &[SliceArg]) -> Self {
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
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let mut offset = self.offset as isize;
        let mut axis = 0;
        for arg in args {
            match *arg {
                SliceArg::NewAxis => {
                    shape.push(1);
                    strides.push(0);
                }
                SliceArg::Index(i) => {
                    let n = self.shape[axis] as isize;
                    let k = if i < 0 { i + n } else { i };
                    assert!(
                        k >= 0 && k < n,
                        "index {} is out of bounds for axis {} with size {}",
                        i,
                        axis,
                        n
                    );
                    offset += k * self.strides[axis];
                    axis += 1;
                }
                SliceArg::Range { start, end, step } => {
                    let (start, count) = normalize(start, end, step, self.shape[axis], false);
                    if count > 0 {
                        offset += start * self.strides[axis];
                    }
                    shape.push(count);
                    strides.push(
                        self.strides[axis]
                            .checked_mul(step)
                            .expect("slice step is too large"),
                    );
                    axis += 1;
                }
                SliceArg::Inclusive { start, end, step } => {
                    let (start, count) = normalize(start, Some(end), step, self.shape[axis], true);
                    if count > 0 {
                        offset += start * self.strides[axis];
                    }
                    shape.push(count);
                    strides.push(
                        self.strides[axis]
                            .checked_mul(step)
                            .expect("slice step is too large"),
                    );
                    axis += 1;
                }
            }
        }
        shape.extend_from_slice(&self.shape[axis..]);
        strides.extend_from_slice(&self.strides[axis..]);
        if shape.contains(&0) {
            offset = 0;
        }
        Self {
            data: self.data.clone(),
            shape,
            strides,
            offset: offset as usize,
        }
    }
    /// Implements np.ndarray.__setitem__ with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut a=np::zeros::<i32>([2,3]); a.slice_assign(&np::s![..,1..], [4,5]);
    /// assert_eq!(a.to_vec(),[0,4,5,0,4,5]);
    /// ```
    pub fn slice_assign(&mut self, args: &[SliceArg], values: impl IntoArray<T>) {
        let values = values.into_array();
        // Validate before detaching; snapshot the RHS for overlapping assignments.
        let shape = self.slice(args).shape;
        let values = values.broadcast_to(&shape).to_vec();
        self.prepare_write();
        let view = self.slice(args);
        let offsets = Offsets::new(&view.shape, &view.strides, view.offset);
        drop(view);
        let data = Arc::make_mut(&mut self.data);
        for (i, v) in offsets.zip(values) {
            data[i] = v;
        }
    }
    /// Implements np.take with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].index_axis(-1,0).to_vec(),[1,3]);
    /// ```
    pub fn index_axis(&self, axis: isize, i: usize) -> Self {
        let axis = axis_index(axis, self.ndim());
        let mut args = vec![SliceArg::all(); self.ndim()];
        args[axis] = SliceArg::Index(isize::try_from(i).expect("index is too large"));
        self.slice(&args)
    }
    /// Implements np.ndarray.__getitem__ with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].row(1).to_vec(),[3,4]);
    /// ```
    pub fn row(&self, i: usize) -> Self {
        assert!(self.ndim() == 2, "input array must be 2-d");
        self.index_axis(0, i)
    }
    /// Implements np.ndarray.__getitem__ with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].col(1).to_vec(),[2,4]);
    /// ```
    pub fn col(&self, i: usize) -> Self {
        assert!(self.ndim() == 2, "input array must be 2-d");
        self.index_axis(1, i)
    }
    /// Implements np.ndarray.__getitem__ with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![1,2,3].mask(&np::array![true,false,true]).to_vec(),[1,3]);
    /// ```
    pub fn mask(&self, m: &Array<bool>) -> Self {
        assert!(
            self.shape == m.shape,
            "boolean index did not match indexed array"
        );
        self.iter()
            .zip(m.iter())
            .filter_map(|(v, b)| b.then_some(v))
            .collect()
    }
    /// Implements np.ndarray.__setitem__ with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// let mut a=np::array![1,2,3]; a.mask_assign(&np::array![false,true,false],9);
    /// assert_eq!(a.to_vec(),[1,9,3]);
    /// ```
    pub fn mask_assign(&mut self, m: &Array<bool>, value: T) {
        assert!(
            self.shape == m.shape,
            "boolean index did not match indexed array"
        );
        let v = self
            .iter()
            .zip(m.iter())
            .map(|(v, b)| if b { value } else { v })
            .collect();
        self.replace_c_order(v);
    }
    /// Implements np.take with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![10,20,30].take(&[2,0],-1).to_vec(),[30,10]);
    /// ```
    pub fn take(&self, indices: &[usize], axis: isize) -> Self {
        let axis = axis_index(axis, self.ndim());
        for &i in indices {
            assert!(
                i < self.shape[axis],
                "index {} is out of bounds for axis {} with size {}",
                i,
                axis,
                self.shape[axis]
            );
        }
        let mut shape = self.shape.clone();
        shape[axis] = indices.len();
        let mut src = vec![0; self.ndim()];
        Self::from_shape_fn(&shape, |c| {
            src.copy_from_slice(c);
            src[axis] = indices[c[axis]];
            self.get(&src)
        })
    }
    /// Implements np.compress with array value semantics.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1,2],[3,4]].compress(&np::array![false,true],0).to_vec(),[3,4]);
    /// ```
    pub fn compress(&self, mask: &Array<bool>, axis: isize) -> Self {
        assert!(mask.ndim() == 1, "condition must be a 1-d array");
        let indices: Vec<_> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.then_some(i))
            .collect();
        self.take(&indices, axis)
    }
}
pub(crate) fn axis_index(axis: isize, n: usize) -> usize {
    let a = if axis < 0 { axis + n as isize } else { axis };
    assert!(
        a >= 0 && (a as usize) < n,
        "axis {} is out of bounds for array of dimension {}",
        axis,
        n
    );
    a as usize
}
#[cfg(test)]
mod tests {
    #[test]
    fn negative_bounds() {
        let a = crate::array![0, 1, 2, 3, 4];
        assert_eq!(a.slice(&crate::s![..;-1]).to_vec(), [4, 3, 2, 1, 0]);
        assert_eq!(a.slice(&crate::s![..-1]).to_vec(), [0, 1, 2, 3]);
        assert!(a.slice(&crate::s![..-1;-1]).is_empty());
    }
    #[test]
    fn clipped_empty() {
        let a = crate::array![1, 2];
        assert!(a.slice(&crate::s![20..30]).as_slice().unwrap().is_empty());
        assert_eq!(a.slice(&crate::s![-99..99]).to_vec(), [1, 2]);
    }
    #[test]
    fn inclusive_negative() {
        assert_eq!(
            crate::array![0, 1, 2].slice(&crate::s![..=-1]).to_vec(),
            [0, 1, 2]
        );
    }
    #[test]
    fn broadcast_assignment() {
        let mut a = crate::zeros::<i32>([2, 3]);
        a.slice_assign(&crate::s![.., 1..], crate::array![5, 6]);
        assert_eq!(a.to_vec(), [0, 5, 6, 0, 5, 6]);
    }
    #[test]
    fn stepped_literals() {
        assert_eq!(crate::s![1..3, .., -1].len(), 3);
        assert_eq!(crate::s![..;2,1..;-1].len(), 2);
    }
}
