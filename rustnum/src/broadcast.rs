use crate::{Array, Element};
/// Converts an array-like value into an array, like NumPy `asarray`.
/// ```
/// use rustnum::{IntoArray, array};
/// assert_eq!(3.into_array().item(),3);
/// assert_eq!([1,2].into_array(),array![1,2]);
/// let v=vec![1,2]; assert_eq!(v.as_slice().into_array(),array![1,2]);
/// assert_eq!(v.into_array(),array![1,2]);
/// let a=array![1,2]; assert_eq!((&a).into_array(),a.clone().into_array());
/// ```
pub trait IntoArray<T: Element> {
    /// Converts to an array, like np.asarray.
    /// ```
    /// use rustnum::IntoArray;
    /// assert_eq!(7.into_array().item(),7);
    /// ```
    fn into_array(self) -> Array<T>;
}
impl<T: Element> IntoArray<T> for Array<T> {
    fn into_array(self) -> Array<T> {
        self
    }
}
impl<T: Element> IntoArray<T> for &Array<T> {
    fn into_array(self) -> Array<T> {
        self.clone()
    }
}
impl<T: Element> IntoArray<T> for T {
    fn into_array(self) -> Array<T> {
        Array::scalar(self)
    }
}
impl<T: Element> IntoArray<T> for Vec<T> {
    fn into_array(self) -> Array<T> {
        Array::from(self)
    }
}
impl<T: Element> IntoArray<T> for &[T] {
    fn into_array(self) -> Array<T> {
        Array::from(self.to_vec())
    }
}
impl<T: Element, const N: usize> IntoArray<T> for [T; N] {
    fn into_array(self) -> Array<T> {
        Array::from(self.to_vec())
    }
}
/// Computes the common shape, like NumPy broadcasting.
/// Implements np.broadcast_shapes with array value semantics.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::broadcast_shapes(&[2,1],&[3]),Some(vec![2,3]));
/// assert_eq!(np::broadcast_shapes(&[0],&[1]),Some(vec![0]));
/// ```
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let y = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if x != y && x != 1 && y != 1 {
            return None;
        }
        out.push(if x == 1 { y } else { x });
    }
    out.reverse();
    Some(out)
}
/// Broadcasts two arrays, like `np.broadcast_arrays`.
/// Implements np.broadcast_arrays with array value semantics.
///
/// ```
/// use rustnum as np;
/// let (a,b)=np::broadcast_arrays(&np::array![[1],[2]],&np::array![3,4]);
/// assert_eq!(a.to_vec(),[1,1,2,2]); assert_eq!(b.to_vec(),[3,4,3,4]);
/// ```
pub fn broadcast_arrays<T: Element, U: Element>(
    a: &Array<T>,
    b: &Array<U>,
) -> (Array<T>, Array<U>) {
    let s = broadcast_shapes(a.shape(), b.shape()).unwrap_or_else(|| {
        panic!(
            "operands could not be broadcast together with shapes {:?} {:?}",
            a.shape(),
            b.shape()
        )
    });
    (a.broadcast_to(&s), b.broadcast_to(&s))
}

#[cfg(test)]
mod tests {
    #[test]
    fn zero_broadcast() {
        assert_eq!(super::broadcast_shapes(&[0, 3], &[1, 3]), Some(vec![0, 3]));
    }
    #[test]
    fn incompatible() {
        assert_eq!(super::broadcast_shapes(&[2, 3], &[4]), None);
    }
}
