//! Group 5: stride tricks. Expected values come from NumPy 2.3.2.
use rustnum as np;

#[test]
fn sliding_window_1d() {
    let x = np::arange(0.0, 10.0, 1.0);
    let w = np::sliding_window_view(&x, &[3], None);
    assert_eq!(w.shape(), [8, 3]);
    assert_eq!(&w.to_vec()[..6], &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
    // zero-copy: shares the same buffer, so the strides overlap
    assert_eq!(w.strides(), [1, 1]);
    assert!(!w.is_contiguous());
}

#[test]
fn sliding_window_2d_and_axes() {
    let a = np::arange(0, 12, 1).reshape(&[3, 4]);
    let w = np::sliding_window_view(&a, &[2, 2], None);
    assert_eq!(w.shape(), [2, 3, 2, 2]);
    assert_eq!(w.index_axis(0, 1).index_axis(0, 2).to_vec(), [6, 7, 10, 11]);
    assert_eq!(w.to_vec().iter().sum::<i32>(), 132);
    let w = np::sliding_window_view(&a, &[2], Some(&[1]));
    assert_eq!(w.shape(), [3, 3, 2]);
    assert_eq!(w.index_axis(0, 2).to_vec(), [8, 9, 9, 10, 10, 11]);
    assert_eq!(
        np::sliding_window_view(&a, &[2, 3], Some(&[0, 1])).shape(),
        [2, 2, 2, 3]
    );
    assert_eq!(
        np::sliding_window_view(&a, &[3], Some(&[-1])).shape(),
        [3, 2, 3]
    );
    assert_eq!(
        np::sliding_window_view(&a, &[2, 2], Some(&[1, 1])).shape(),
        [3, 2, 2, 2]
    );
    let z = np::sliding_window_view(&np::arange(0, 3, 1), &[0], None);
    assert_eq!(z.shape(), [4, 0]);
}

#[test]
#[should_panic(expected = "window shape cannot be larger than input array shape")]
fn window_too_big() {
    np::sliding_window_view(&np::arange(0, 3, 1), &[4], None);
}

#[test]
#[should_panic(
    expected = "Since axis is `None`, must provide window_shape for all dimensions of `x`; got 1 window_shape elements and `x.ndim` is 2."
)]
fn window_needs_all_dims() {
    np::sliding_window_view(&np::zeros::<i32>([2, 2]), &[2], None);
}

#[test]
#[should_panic(
    expected = "Must provide matching length window_shape and axis; got 2 window_shape elements and 1 axes elements."
)]
fn window_axis_len() {
    np::sliding_window_view(&np::zeros::<i32>([3, 3]), &[2, 2], Some(&[0]));
}

#[test]
#[should_panic(expected = "axis 3 is out of bounds for array of dimension 1")]
fn window_bad_axis() {
    np::sliding_window_view(&np::arange(0, 3, 1), &[2], Some(&[3]));
}

#[test]
fn window_writes_do_not_leak() {
    let x = np::arange(0, 6, 1);
    let mut w = np::sliding_window_view(&x, &[3], None);
    w.set(&[0, 1], 100);
    // overlapping elements of the window view are independent after the write
    assert_eq!(w.to_vec(), [0, 100, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5]);
    assert_eq!(x.to_vec(), [0, 1, 2, 3, 4, 5]);
    let mut m = np::sliding_window_view(&x, &[2], None);
    m.fill(7);
    assert!(m.iter().all(|v| v == 7));
    assert_eq!(x.to_vec(), [0, 1, 2, 3, 4, 5]);
}

#[test]
fn as_strided_views() {
    let x = np::arange(0, 6, 1);
    assert_eq!(
        np::as_strided(&x, &[3, 2], &[1, 1]).to_vec(),
        [0, 1, 1, 2, 2, 3]
    );
    assert_eq!(
        np::as_strided(&x, &[2, 3], &[3, 1]).to_vec(),
        [0, 1, 2, 3, 4, 5]
    );
    assert_eq!(
        np::as_strided(&x, &[3, 2], &[2, 1]).to_vec(),
        [0, 1, 2, 3, 4, 5]
    );
    assert_eq!(
        np::as_strided(&x, &[2, 3], &[0, 1]).to_vec(),
        [0, 1, 2, 0, 1, 2]
    );
    assert!(np::as_strided(&x, &[0, 99], &[1000, 1000]).is_empty());
    // strides start from the view's own first element; negative strides walk backwards from it
    let tail = x.slice(&np::s![3..]);
    assert_eq!(np::as_strided(&tail, &[3], &[-1]).to_vec(), [3, 2, 1]);
    let mut v = np::as_strided(&x, &[3, 2], &[1, 1]);
    v.set(&[0, 1], 50);
    assert_eq!(x.to_vec(), [0, 1, 2, 3, 4, 5]);
    assert_eq!(v.to_vec(), [0, 50, 1, 2, 2, 3]);
}

#[test]
#[should_panic(expected = "would access memory outside the array buffer")]
fn as_strided_past_end() {
    np::as_strided(&np::arange(0, 6, 1), &[3, 3], &[2, 1]);
}

#[test]
#[should_panic(expected = "would access memory outside the array buffer")]
fn as_strided_before_start() {
    np::as_strided(&np::arange(0, 6, 1), &[2], &[-1]);
}

#[test]
#[should_panic(expected = "mismatch in length of strides and shape")]
fn as_strided_rank_mismatch() {
    np::as_strided(&np::arange(0, 6, 1), &[2, 2], &[1]);
}
