use crate::{Array, Element, Float, Num};
/// Creates zeros, like `np.zeros`.
/// Implements np.zeros with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::zeros::<i32>([2,2]).to_vec(),[0,0,0,0]);
/// ```
pub fn zeros<T: Element>(shape: impl AsRef<[usize]>) -> Array<T> {
    let shape = shape.as_ref();
    Array::from_vec(
        crate::pool::filled(crate::layout::size(shape), T::default()),
        shape,
    )
}
/// Creates ones, like `np.ones`.
/// Implements np.ones with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::ones::<i32>([2]).to_vec(),[1,1]);
/// ```
pub fn ones<T: Num>(shape: impl AsRef<[usize]>) -> Array<T> {
    full(shape, T::ONE)
}
/// Creates a filled array, like `np.full`.
/// Implements np.full with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::full([2],7).to_vec(),[7,7]);
/// ```
pub fn full<T: Element>(shape: impl AsRef<[usize]>, v: T) -> Array<T> {
    let shape = shape.as_ref();
    Array::from_vec(crate::pool::filled(crate::layout::size(shape), v), shape)
}
/// Implements np.zeros_like with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::zeros_like(&np::array![1,2]).to_vec(),[0,0]);
/// ```
pub fn zeros_like<T: Element>(a: &Array<T>) -> Array<T> {
    zeros(a.shape())
}
/// Implements np.ones_like with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::ones_like(&np::array![2,3]).to_vec(),[1,1]);
/// ```
pub fn ones_like<T: Num>(a: &Array<T>) -> Array<T> {
    ones(a.shape())
}
/// Implements np.full_like with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::full_like(&np::array![1,2],9).to_vec(),[9,9]);
/// ```
pub fn full_like<T: Element>(a: &Array<T>, v: T) -> Array<T> {
    full(a.shape(), v)
}
/// Creates evenly stepped values, like `np.arange`.
/// Implements np.arange with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::arange(5,0,-2).to_vec(),[5,3,1]);
/// ```
pub fn arange<T: Num>(start: T, stop: T, step: T) -> Array<T> {
    assert!(step != T::ZERO, "arange: step must not be zero");
    if crate::element::is_float_dtype(T::DTYPE) {
        let length = ((stop.to_f64() - start.to_f64()) / step.to_f64()).ceil();
        assert!(!length.is_nan(), "arange: cannot compute length");
        assert!(
            length.is_finite() && length < isize::MAX as f64,
            "Maximum allowed size exceeded"
        );
        let n = length.max(0.0) as usize;
        return Array::from(
            (0..n)
                .map(|i| start + T::from_usize(i) * step)
                .collect::<Vec<_>>(),
        );
    }
    // i128 exactly represents every supported integer, including usize.
    let start = integer_value(start);
    let stop = integer_value(stop);
    let step = integer_value(step);
    let distance = if step > 0 { stop - start } else { start - stop };
    let n = if distance <= 0 {
        0
    } else {
        (distance - 1) / step.abs() + 1
    };
    let n = usize::try_from(n).expect("Maximum allowed size exceeded");
    Array::from(
        (0..n)
            .map(|i| integer_from::<T>(start + i as i128 * step))
            .collect::<Vec<_>>(),
    )
}
fn integer_value<T: Num>(v: T) -> i128 {
    use std::any::Any;
    let a = &v as &dyn Any;
    if let Some(&x) = a.downcast_ref::<i64>() {
        x as i128
    } else if let Some(&x) = a.downcast_ref::<i32>() {
        x as i128
    } else if let Some(&x) = a.downcast_ref::<usize>() {
        x as i128
    } else if let Some(&x) = a.downcast_ref::<u8>() {
        x as i128
    } else {
        unreachable!()
    }
}
fn integer_from<T: Num>(v: i128) -> T {
    use std::any::{Any, TypeId};
    macro_rules! cast {
        ($t:ty) => {
            if TypeId::of::<T>() == TypeId::of::<$t>() {
                return *(&(v as $t) as &dyn Any).downcast_ref::<T>().unwrap();
            }
        };
    }
    cast!(i64);
    cast!(i32);
    cast!(usize);
    cast!(u8);
    unreachable!()
}
/// Creates evenly spaced values, like `np.linspace`.
/// Implements np.linspace with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::linspace(0.,1.,3).to_vec(),[0.,0.5,1.]);
/// ```
pub fn linspace<F: Float>(start: F, stop: F, num: usize) -> Array<F> {
    if num == 0 {
        return Array::from(Vec::new());
    }
    if num == 1 {
        return Array::from(vec![start]);
    }
    let d = (stop - start) / F::from_usize(num - 1);
    Array::from(
        (0..num)
            .map(|i| {
                if i + 1 == num {
                    stop
                } else {
                    start + d * F::from_usize(i)
                }
            })
            .collect::<Vec<_>>(),
    )
}
/// Implements np.logspace with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::logspace(0.,2.,3).to_vec(),[1.,10.,100.]);
/// ```
pub fn logspace<F: Float>(start: F, stop: F, num: usize) -> Array<F> {
    linspace(start, stop, num).map(|x| F::from_usize(10).powf(x))
}
/// Implements np.eye with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::eye::<i32>(2).to_vec(),[1,0,0,1]);
/// ```
pub fn eye<T: Num>(n: usize) -> Array<T> {
    let mut a = zeros([n, n]);
    for i in 0..n {
        a.set(&[i, i], T::ONE)
    }
    a
}
/// Implements np.identity with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::identity::<i32>(2),np::eye::<i32>(2));
/// ```
pub fn identity<T: Num>(n: usize) -> Array<T> {
    eye(n)
}
/// Implements np.diag with array value semantics.
///
/// ```
/// use rustnum as np;
/// let a=np::diag(&np::array![2,3]);
/// assert_eq!(a.to_vec(),[2,0,0,3]); assert_eq!(np::diag(&a).to_vec(),[2,3]);
/// ```
pub fn diag<T: Element>(v: &Array<T>) -> Array<T> {
    match v.ndim() {
        1 => {
            let n = v.size();
            let mut a = zeros([n, n]);
            for i in 0..n {
                a.set(&[i, i], v.get(&[i]));
            }
            a
        }
        2 => {
            let n = v.shape()[0].min(v.shape()[1]);
            Array::from((0..n).map(|i| v.get(&[i, i])).collect::<Vec<_>>())
        }
        _ => panic!("Input must be 1- or 2-d."),
    }
}
/// Implements np.tri with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::tri::<i32>(2).to_vec(),[1,0,1,1]);
/// ```
pub fn tri<T: Num>(n: usize) -> Array<T> {
    Array::from_shape_fn([n, n], |c| if c[1] <= c[0] { T::ONE } else { T::ZERO })
}
/// Implements np.tril with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::tril(&np::array![[1,2],[3,4]]).to_vec(),[1,0,3,4]);
/// ```
pub fn tril<T: Num>(a: &Array<T>) -> Array<T> {
    triangular(a, false)
}
/// Implements np.triu with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::triu(&np::array![[1,2],[3,4]]).to_vec(),[1,2,0,4]);
/// ```
pub fn triu<T: Num>(a: &Array<T>) -> Array<T> {
    triangular(a, true)
}
fn triangular<T: Num>(a: &Array<T>, upper: bool) -> Array<T> {
    assert!(a.ndim() > 0, "input array must have at least one dimension");
    let a = if a.ndim() == 1 {
        a.broadcast_to(&[a.size(), a.size()])
    } else {
        a.clone()
    };
    let rank = a.ndim();
    Array::from_shape_fn(a.shape(), |c| {
        if if upper {
            c[rank - 1] >= c[rank - 2]
        } else {
            c[rank - 1] <= c[rank - 2]
        } {
            a.get(c)
        } else {
            T::ZERO
        }
    })
}
/// Implements np.meshgrid with array value semantics.
///
/// ```
/// use rustnum as np;
/// let (x,y)=np::meshgrid(&np::array![1,2],&np::array![3,4,5]);
/// assert_eq!(x.to_vec(),[1,2,1,2,1,2]); assert_eq!(y.to_vec(),[3,3,4,4,5,5]);
/// ```
pub fn meshgrid<T: Element>(x: &Array<T>, y: &Array<T>) -> (Array<T>, Array<T>) {
    let x = x.ravel();
    let y = y.ravel();
    let nx = x.size();
    let ny = y.size();
    (
        Array::from_shape_fn([ny, nx], |c| x.get(&[c[1]])),
        Array::from_shape_fn([ny, nx], |c| y.get(&[c[0]])),
    )
}
/// Creates array literals, like np.array, supporting one through three dimensions.
/// ```
/// use rustnum as np;
/// assert_eq!(np::array![1,2].shape(),[2]);
/// assert_eq!(np::array![[1,2],[3,4]].shape(),[2,2]);
/// assert_eq!(np::array![[[1,2]],[[3,4]]].shape(),[2,1,2]);
/// ```
#[macro_export]
macro_rules! array { ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => {{let planes=vec![$(vec![$(vec![$($x),*]),+]),+];let d0=planes.len();let d1=planes.first().map_or(0,Vec::len);let d2=planes.first().and_then(|p|p.first()).map_or(0,Vec::len);assert!(planes.iter().all(|p|p.len()==d1&&p.iter().all(|r|r.len()==d2)),"setting an array element with a sequence");$crate::Array::from_vec(planes.into_iter().flatten().flatten().collect(),&[d0,d1,d2])}}; ($([$($x:expr),* $(,)?]),+ $(,)?) => {{ let rows=vec![$(vec![$($x),*]),+];let nr=rows.len();let n=rows.first().map_or(0,Vec::len);assert!(rows.iter().all(|r|r.len()==n),"setting an array element with a sequence");$crate::Array::from_vec(rows.into_iter().flatten().collect(),&[nr,n]) }}; ($($x:expr),* $(,)?) => {$crate::Array::from(vec![$($x),*])}; }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_endpoints() {
        assert_eq!(
            arange(i64::MAX - 2, i64::MAX, 1).to_vec(),
            [i64::MAX - 2, i64::MAX - 1]
        );
        assert_eq!(arange(254u8, 255, 2).to_vec(), [254]);
    }
    #[test]
    fn float_range_length() {
        assert_eq!(arange(0.0, 1.0, 0.1).size(), 10);
    }
    #[test]
    fn three_dimensional_literal() {
        assert_eq!(
            crate::array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].shape(),
            [2, 2, 2]
        );
    }
    #[test]
    #[should_panic(expected = "setting an array element with a sequence")]
    fn ragged() {
        let _ = crate::array![[1], [2, 3]];
    }
}
