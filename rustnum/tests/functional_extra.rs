//! Group 7: vectorize, apply_along_axis, apply_over_axes. Expected values come from NumPy 2.3.2.
use rustnum as np;

#[test]
fn vectorize_broadcasts() {
    let hyp = np::vectorize(|a: f64, b: f64| (a * a + b * b).sqrt());
    assert_eq!(hyp.call2(np::array![3.0, 5.0], 4.0).to_vec()[0], 5.0);
    let mx = np::vectorize(|a: i32, b: i32| a.max(b));
    let r = mx.call2(np::array![[1], [5]], np::array![2, 4]);
    assert_eq!(r.shape(), [2, 2]);
    assert_eq!(r.to_vec(), [2, 4, 5, 5]);
    let f3 = np::vectorize(|a: i32, b: i32, c: i32| a * b + c);
    let r = f3.call3(np::array![1, 2], 10, np::array![[5], [6]]);
    assert_eq!(r.shape(), [2, 2]);
    assert_eq!(r.to_vec(), [15, 25, 16, 26]);
    // different output type, non-contiguous input
    let label = np::vectorize(|x: i32| x >= 3);
    let t = np::array![[1, 2], [3, 4]].t();
    assert_eq!(label.call(&t).to_vec(), [false, true, false, true]);
    assert!(label.call(7).item());
}

#[test]
#[should_panic(expected = "operands could not be broadcast together")]
fn vectorize_mismatch() {
    let f = np::vectorize(|a: i32, b: i32| a + b);
    f.call2(np::array![1, 2, 3], np::array![1, 2]);
}

#[test]
fn apply_along_axis_shapes() {
    let a = np::array![[1, 2, 3], [4, 5, 6]];
    let sums = np::apply_along_axis(|l| np::Array::scalar(l.sum()), 1, &a);
    assert_eq!(sums.shape(), [2]);
    assert_eq!(sums.to_vec(), [6, 15]);
    let col_sums = np::apply_along_axis(|l| np::Array::scalar(l.sum()), 0, &a);
    assert_eq!(col_sums.to_vec(), [5, 7, 9]);
    // function returning a vector of a different length: axis is replaced by its shape
    let r = np::apply_along_axis(|l| np::array![l.min(), l.max()], 1, &a);
    assert_eq!(r.shape(), [2, 2]);
    assert_eq!(r.to_vec(), [1, 3, 4, 6]);
    let r = np::apply_along_axis(|l| np::array![l.min(), l.max()], 0, &a);
    assert_eq!(r.shape(), [2, 3]);
    assert_eq!(r.to_vec(), [1, 2, 3, 4, 5, 6]);
    // same length, middle axis of a 3-d array, negative axis
    let c = np::arange(0, 24, 1).reshape(&[2, 3, 4]);
    let m = np::apply_along_axis(|l| l.flip(0), 1, &c);
    assert_eq!(m.shape(), [2, 3, 4]);
    assert_eq!(m.get(&[0, 0, 1]), 9);
    assert_eq!(m.get(&[1, 2, 3]), 15);
    let n = np::apply_along_axis(|l| np::Array::scalar(l.sum()), -1, &c);
    assert_eq!(n.shape(), [2, 3]);
    assert_eq!(n.to_vec(), [6, 22, 38, 54, 70, 86]);
    // 2-d result per lane
    let q = np::apply_along_axis(|l| l.reshape(&[2, 2]), 1, &np::array![[1, 2, 3, 4]]);
    assert_eq!(q.shape(), [1, 2, 2]);
    // non-contiguous input
    let t = np::apply_along_axis(|l| np::Array::scalar(l.sum()), 0, &a.t());
    assert_eq!(t.to_vec(), [6, 15]);
}

#[test]
#[should_panic(expected = "axis 3 is out of bounds for array of dimension 2")]
fn apply_along_axis_bad_axis() {
    np::apply_along_axis(|l| l.clone(), 3, &np::zeros::<f64>([2, 2]));
}

#[test]
#[should_panic(expected = "Cannot apply_along_axis when any iteration dimensions are 0")]
fn apply_along_axis_empty() {
    np::apply_along_axis(|l| l.clone(), 1, &np::zeros::<f64>([0, 3]));
}

#[test]
fn apply_over_axes_matches_numpy() {
    let a = np::arange(0, 24, 1).reshape(&[2, 3, 4]);
    let s = np::apply_over_axes(|x, ax| x.sum_axis(ax), &a, &[0, 2]);
    assert_eq!(s.shape(), [1, 3, 1]);
    assert_eq!(s.to_vec(), [60, 92, 124]);
    let neg = np::apply_over_axes(|x, ax| x.sum_axis(ax), &a, &[-1]);
    assert_eq!(neg.shape(), [2, 3, 1]);
    assert_eq!(neg.to_vec(), [6, 22, 38, 54, 70, 86]);
    // function that keeps the dimensionality
    let k = np::apply_over_axes(|x, _| x.map(|v| v + 1), &a, &[0, 1]);
    assert_eq!(k.get(&[0, 0, 0]), 2);
}

#[test]
#[should_panic(expected = "function is not returning an array of the correct shape")]
fn apply_over_axes_bad_shape() {
    let a = np::arange(0, 24, 1).reshape(&[2, 3, 4]);
    np::apply_over_axes(|x, ax| x.sum_axis(ax).sum_axis(0), &a, &[1]);
}
