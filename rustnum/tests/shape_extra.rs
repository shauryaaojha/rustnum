//! Group 4: shape and creation extras. Expected values come from NumPy 2.3.2.
use rustnum as np;

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
            "{got:?} vs {want:?}"
        );
    }
}

#[test]
fn rot90_all_k() {
    let m = np::arange(0, 6, 1).reshape(&[2, 3]);
    let cases: [(isize, &[i32], &[usize]); 6] = [
        (0, &[0, 1, 2, 3, 4, 5], &[2, 3]),
        (1, &[2, 5, 1, 4, 0, 3], &[3, 2]),
        (2, &[5, 4, 3, 2, 1, 0], &[2, 3]),
        (3, &[3, 0, 4, 1, 5, 2], &[3, 2]),
        (-1, &[3, 0, 4, 1, 5, 2], &[3, 2]),
        (5, &[2, 5, 1, 4, 0, 3], &[3, 2]),
    ];
    for (k, want, shape) in cases {
        let r = np::rot90(&m, k, (0, 1));
        assert_eq!(r.to_vec(), want, "k={k}");
        assert_eq!(r.shape(), shape);
    }
}

#[test]
fn rot90_3d_axes() {
    let c = np::arange(0, 8, 1).reshape(&[2, 2, 2]);
    assert_eq!(np::rot90(&c, 1, (0, 2)).to_vec(), [1, 5, 3, 7, 0, 4, 2, 6]);
    assert_eq!(
        np::rot90(&c, -1, (-1, 1)).to_vec(),
        [1, 3, 0, 2, 5, 7, 4, 6]
    );
}

#[test]
#[should_panic(expected = "Axes must be different.")]
fn rot90_same_axes() {
    np::rot90(&np::zeros::<i32>([2, 2]), 1, (0, 0));
}

#[test]
#[should_panic(expected = "Axes=(0, 1) out of range for array of ndim=1.")]
fn rot90_1d() {
    np::rot90(&np::zeros::<i32>([3]), 1, (0, 1));
}

#[test]
fn stacks_and_atleast() {
    let a = np::arange(0, 4, 1).reshape(&[2, 2]);
    let b = np::arange(4, 8, 1).reshape(&[2, 2]);
    let d = np::dstack(&[&a, &b]);
    assert_eq!(d.shape(), [2, 2, 2]);
    assert_eq!(d.to_vec(), [0, 4, 1, 5, 2, 6, 3, 7]);
    let v = np::dstack(&[&np::arange(0, 3, 1), &np::arange(3, 6, 1)]);
    assert_eq!(v.shape(), [1, 3, 2]);
    let c = np::column_stack(&[&np::arange(0, 3, 1), &np::arange(3, 6, 1)]);
    assert_eq!(c.shape(), [3, 2]);
    assert_eq!(c.to_vec(), [0, 3, 1, 4, 2, 5]);
    let m = np::column_stack(&[&a, &np::array![9, 8]]);
    assert_eq!(m.to_vec(), [0, 1, 9, 2, 3, 8]);
    assert_eq!(np::row_stack(&[&a, &b]).shape(), [4, 2]);
    assert_eq!(np::atleast_3d(&np::arange(0, 3, 1)).shape(), [1, 3, 1]);
    assert_eq!(np::atleast_2d(&np::Array::scalar(3.0)).shape(), [1, 1]);
    assert_eq!(np::atleast_1d(&np::Array::scalar(3.0)).to_vec(), [3.0]);
    // views, even of non-contiguous input
    let t = np::atleast_2d(&a.t().index_axis(0, 1));
    assert_eq!(t.shape(), [1, 2]);
    assert_eq!(t.to_vec(), [1, 3]);
}

#[test]
#[should_panic(
    expected = "all the input array dimensions except for the concatenation axis must match exactly"
)]
fn dstack_mismatch() {
    np::dstack(&[&np::zeros::<i32>([2, 3]), &np::zeros::<i32>([2, 4])]);
}

#[test]
fn geomspace_matches_numpy() {
    close(
        &np::geomspace(1.0, 1000.0, 4).to_vec(),
        &[1.0, 10.0, 100.0, 1000.0],
    );
    close(
        &np::geomspace(2.0, 512.0, 5).to_vec(),
        &[
            2.0,
            7.999999999999999,
            32.00000000000001,
            127.99999999999999,
            512.0,
        ],
    );
    close(
        &np::geomspace(-1.0, -1000.0, 4).to_vec(),
        &[-1.0, -10.0, -100.0, -1000.0],
    );
    close(
        &np::geomspace(0.1, 5.0, 7).to_vec(),
        &[
            0.1,
            0.19193831036664843,
            0.36840314986403866,
            0.7071067811865475,
            1.3572088082974532,
            2.6050036547934563,
            5.0,
        ],
    );
    assert_eq!(np::geomspace(1.0, 10.0, 1).to_vec(), [1.0]);
    assert!(np::geomspace(1.0, 10.0, 0).is_empty());
    let mixed = np::geomspace(1.0f64, -1.0, 3).to_vec();
    assert!(mixed[1].is_nan() && mixed[0] == 1.0 && mixed[2] == -1.0);
}

#[test]
#[should_panic(expected = "Geometric sequence cannot include zero")]
fn geomspace_zero() {
    np::geomspace(0.0, 1.0, 3);
}

#[test]
fn vander_forms() {
    let x = np::array![1, 2, 3, 5];
    assert_eq!(
        np::vander(&x, None, false).to_vec(),
        [1, 1, 1, 1, 8, 4, 2, 1, 27, 9, 3, 1, 125, 25, 5, 1]
    );
    let y = np::array![1, 2, 3];
    assert_eq!(np::vander(&y, Some(2), false).to_vec(), [1, 1, 2, 1, 3, 1]);
    assert_eq!(
        np::vander(&y, Some(4), true).to_vec(),
        [1, 1, 1, 1, 1, 2, 4, 8, 1, 3, 9, 27]
    );
    assert_eq!(
        np::vander(&np::array![2.5, 3.0], Some(3), false).to_vec(),
        [6.25, 2.5, 1.0, 9.0, 3.0, 1.0]
    );
    // integer powers wrap instead of panicking
    assert_eq!(
        np::vander(&np::array![200u8], Some(3), false).to_vec(),
        [64, 200, 1]
    );
}

#[test]
#[should_panic(expected = "x must be a one-dimensional array or sequence.")]
fn vander_2d() {
    np::vander(&np::zeros::<f64>([2, 2]), None, false);
}

#[test]
fn fromiter_frombuffer() {
    assert_eq!(
        np::fromiter((0..5).map(|i| i * i), None).to_vec(),
        [0, 1, 4, 9, 16]
    );
    assert_eq!(np::fromiter(0..10, Some(3)).to_vec(), [0, 1, 2]);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1.5f64.to_ne_bytes());
    bytes.extend_from_slice(&(-2.0f64).to_ne_bytes());
    assert_eq!(np::frombuffer::<f64>(&bytes, None, 0).to_vec(), [1.5, -2.0]);
    assert_eq!(np::frombuffer::<f64>(&bytes, Some(1), 8).to_vec(), [-2.0]);
    assert_eq!(np::frombuffer::<u8>(&[1, 2, 3], None, 1).to_vec(), [2, 3]);
    assert_eq!(
        np::frombuffer::<bool>(&[0, 2, 1], None, 0).to_vec(),
        [false, true, true]
    );
    let ints: Vec<u8> = [7i32, -1].iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(np::frombuffer::<i32>(&ints, None, 0).to_vec(), [7, -1]);
    let longs: Vec<u8> = [i64::MIN, 5].iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(
        np::frombuffer::<i64>(&longs, None, 0).to_vec(),
        [i64::MIN, 5]
    );
    let f32s: Vec<u8> = [0.5f32].iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(np::frombuffer::<f32>(&f32s, None, 0).to_vec(), [0.5]);
    let us: Vec<u8> = [9usize].iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(np::frombuffer::<usize>(&us, None, 0).to_vec(), [9]);
}

#[test]
#[should_panic(expected = "iterator too short: Expected 2 but iterator had only 0 items.")]
fn fromiter_short() {
    np::fromiter(Vec::<i32>::new(), Some(2));
}

#[test]
#[should_panic(expected = "buffer size must be a multiple of element size")]
fn frombuffer_bad_size() {
    np::frombuffer::<i32>(b"abc", None, 0);
}

#[test]
#[should_panic(expected = "buffer is smaller than requested size")]
fn frombuffer_count_too_big() {
    np::frombuffer::<i32>(&[0; 8], Some(3), 0);
}

#[test]
fn around_and_inf() {
    let a = np::array![1.26, 2.74, -0.5, 0.5, 1.5, 2.5];
    assert_eq!(np::around(&a, 0).to_vec(), [1.0, 3.0, -0.0, 0.0, 2.0, 2.0]);
    assert_eq!(
        np::around(&np::array![1234.5678, -0.0049], -2).to_vec(),
        [1200.0, -0.0]
    );
    close(
        &np::around(&np::array![1.2345, 2.3456], 3).to_vec(),
        &[1.234, 2.346],
    );
    let v = np::array![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1.0];
    assert_eq!(v.is_posinf().to_vec(), [true, false, false, false]);
    assert_eq!(v.is_neginf().to_vec(), [false, true, false, false]);
}

#[test]
fn real_if_close_pairs() {
    let re = np::array![2.1, 5.2];
    let (r, im) = np::real_if_close(&re, &np::array![1e-16, -1e-15], 100.0);
    assert_eq!(r.to_vec(), [2.1, 5.2]);
    assert!(im.is_none());
    let (_, im) = np::real_if_close(&re, &np::array![0.0, 0.5], 100.0);
    assert_eq!(im.unwrap().to_vec(), [0.0, 0.5]);
    let (_, im) = np::real_if_close(&re, &np::array![0.0, 1e-3], 1e-2);
    assert!(im.is_none());
}
