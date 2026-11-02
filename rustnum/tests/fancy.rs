//! Advanced indexing with hand-computed expectations (NumPy results worked out by hand; the
//! broader NumPy comparison is in the parity suite).

use rustnum as np;

fn cube() -> np::Array<i32> {
    np::arange(0, 24, 1).reshape(&[2, 3, 4])
}

#[test]
fn adjacent_arrays_replace_their_axes_in_place() {
    let a = cube();
    let r = a.slice(&np::s![.., [0, 2], [1, 3]]);
    assert_eq!(r.shape(), [2, 2]);
    assert_eq!(r.to_vec(), [1, 11, 13, 23]);
    let r = a.slice(&np::s![.., 1, [0, 3]]);
    assert_eq!(r.shape(), [2, 2]);
    assert_eq!(r.to_vec(), [4, 7, 16, 19]);
}

#[test]
fn separated_advanced_indices_move_to_the_front() {
    let a = cube();
    let r = a.slice(&np::s![[0, 1], .., [1, 3]]);
    assert_eq!(r.shape(), [2, 3]);
    assert_eq!(r.to_vec(), [1, 5, 9, 15, 19, 23]);
    // a plain integer next to an array is part of the broadcast (NumPy's famous quirk)
    let r = a.slice(&np::s![0, .., [1, 2]]);
    assert_eq!(r.shape(), [2, 3]);
    assert_eq!(r.to_vec(), [1, 5, 9, 2, 6, 10]);
    // np.newaxis between two advanced indices also separates them
    let r = a.slice(&np::s![[0, 1], np::NEW_AXIS, [1, 2]]);
    assert_eq!(r.shape(), [2, 1, 4]);
    assert_eq!(r.to_vec(), [4, 5, 6, 7, 20, 21, 22, 23]);
}

#[test]
fn index_arrays_broadcast_and_keep_their_shape() {
    let a = cube();
    let rows = np::array![[0usize], [1]];
    let cols = np::array![[0usize, 2, 1]];
    let r = a.slice(&np::s![&rows, &cols]);
    assert_eq!(r.shape(), [2, 3, 4]);
    assert_eq!(&r.to_vec()[..4], [0, 1, 2, 3]);
    assert_eq!(&r.to_vec()[20..24], [16, 17, 18, 19]);
    let m = np::arange(0, 6, 1).reshape(&[2, 3]);
    let r = m.slice(&np::s![
        &np::array![[0usize, 1], [1, 0]],
        &np::array![[2usize, 2], [0, 1]]
    ]);
    assert_eq!(r.shape(), [2, 2]);
    assert_eq!(r.to_vec(), [2, 5, 3, 1]);
    // negative indices
    assert_eq!(m.slice(&np::s![[-1, 0], [-1, -3]]).to_vec(), [5, 0]);
}

#[test]
fn slices_steps_and_masks_mix_with_arrays() {
    let m = np::arange(0, 20, 1).reshape(&[4, 5]);
    assert_eq!(m.slice(&np::s![..;2, [4, 0]]).to_vec(), [4, 0, 14, 10]);
    let mask = np::array![true, false, true, false, true];
    assert_eq!(
        m.slice(&np::s![1..3, &mask]).to_vec(),
        [5, 7, 9, 10, 12, 14]
    );
    assert_eq!(m.slice(&np::s![[3], ..]).shape(), [1, 5]);
}

#[test]
fn advanced_indexing_copies() {
    let a = np::array![1, 2, 3];
    let mut b = a.slice(&np::s![[0, 1]]);
    b[[0]] = 99;
    assert_eq!(a.to_vec(), [1, 2, 3]);
}

#[test]
fn assignment_broadcasts_and_last_write_wins() {
    let mut a = np::zeros::<i32>([4]);
    a.slice_assign(&np::s![[0, 0, 1]], np::array![1, 2, 3]);
    assert_eq!(a.to_vec(), [2, 3, 0, 0]);
    let mut m = np::zeros::<i32>([3, 4]);
    m.slice_assign(&np::s![[0, 2], [1, 3]], 7);
    assert_eq!(m.to_vec(), [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]);
    m.slice_assign(&np::s![.., [0, 1]], np::array![1, 2]);
    assert_eq!(&m.to_vec()[..4], [1, 2, 0, 0]);
    // writing through a transposed view
    let mut t = np::zeros::<i32>([2, 3]).t();
    t.slice_assign(&np::s![[2], [1]], 5);
    assert_eq!(t.t().to_vec(), [0, 0, 0, 0, 0, 5]);
}

#[test]
#[should_panic(expected = "index 3 is out of bounds for axis 1 with size 3")]
fn out_of_bounds_matches_numpy_wording() {
    let _ = cube().slice(&np::s![.., [3]]);
}

#[test]
#[should_panic(
    expected = "shape mismatch: indexing arrays could not be broadcast together with shapes (2,) (3,)"
)]
fn unbroadcastable_arrays_panic() {
    let _ = cube().slice(&np::s![[0, 1], [0, 1, 2]]);
}

#[test]
#[should_panic(
    expected = "shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (2,)"
)]
fn assignment_shape_mismatch_panics() {
    let mut a = np::zeros::<i32>([5]);
    a.slice_assign(&np::s![[0, 1]], np::array![1, 2, 3]);
}

#[test]
fn ix_builds_an_open_mesh() {
    let m = np::arange(0, 20, 1).reshape(&[4, 5]);
    let ix = np::ix_(&[&[0, 2], &[1, 3, 4]]);
    assert_eq!(ix[0].shape(), [2, 1]);
    assert_eq!(ix[1].shape(), [1, 3]);
    let sub = m.slice(&np::s![&ix[0], &ix[1]]);
    assert_eq!(sub.shape(), [2, 3]);
    assert_eq!(sub.to_vec(), [1, 3, 4, 11, 13, 14]);
}

#[test]
fn take_and_put_along_axis() {
    let a = np::array![[10, 30, 20], [60, 40, 50]];
    let order = a.argsort_axis(1).astype::<i64>();
    assert_eq!(
        np::take_along_axis(&a, &order, 1).to_vec(),
        [10, 20, 30, 40, 50, 60]
    );
    let best = np::array![[1i64], [0]];
    assert_eq!(np::take_along_axis(&a, &best, 1).to_vec(), [30, 60]);
    let mut b = np::zeros::<i32>([2, 3]);
    np::put_along_axis(&mut b, &np::array![[2i64], [0]], np::array![[8], [9]], 1);
    assert_eq!(b.to_vec(), [0, 0, 8, 9, 0, 0]);
}

#[test]
#[should_panic(expected = "`indices` and `arr` must have the same number of dimensions")]
fn take_along_axis_needs_matching_ndim() {
    let _ = np::take_along_axis(&np::array![[1, 2]], &np::array![0i64], 0);
}

#[test]
fn put_choose_ravel_unravel() {
    let mut a = np::zeros::<i32>([2, 3]);
    np::put(&mut a, [0i64, 5, -2], [1, 2]);
    assert_eq!(a.to_vec(), [1, 0, 0, 0, 1, 2]);
    let r = np::choose(
        &np::array![[0i64, 1], [1, 0]],
        &[&np::array![1, 2], &np::array![[10, 20], [30, 40]]],
    );
    assert_eq!(r.to_vec(), [1, 20, 30, 2]);
    let flat = np::ravel_multi_index(&[&np::array![1i64, 2], &np::array![0i64, 3]], &[3, 4]);
    assert_eq!(flat.to_vec(), [4, 11]);
    let back = np::unravel_index(&flat, &[3, 4]);
    assert_eq!(back[0].to_vec(), [1, 2]);
    assert_eq!(back[1].to_vec(), [0, 3]);
}

#[test]
#[should_panic(expected = "invalid entry in choice array")]
fn choose_checks_entries() {
    let _ = np::choose(&np::array![2i64], &[&np::array![1], &np::array![2]]);
}

#[test]
#[should_panic(expected = "invalid entry in coordinates array")]
fn ravel_multi_index_checks_range() {
    let _ = np::ravel_multi_index(&[&np::array![3i64]], &[3]);
}

#[test]
#[should_panic(expected = "index 12 is out of bounds for array with size 12")]
fn unravel_index_checks_range() {
    let _ = np::unravel_index(&np::array![12i64], &[3, 4]);
}
