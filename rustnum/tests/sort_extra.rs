//! Group 2: lexsort, partition, select, piecewise, extract, place, copyto, trim_zeros.
//! Expected values come from NumPy 2.3.2.
use rustnum as np;

const NAN: f64 = f64::NAN;

#[test]
fn lexsort_1d_and_2d() {
    let surname = np::array![1, 2, 1, 2, 1];
    let first = np::array![5, 5, 3, 3, 3];
    assert_eq!(
        np::lexsort(&[&surname, &first], -1).to_vec(),
        [2, 4, 3, 0, 1]
    );
    let k1 = np::array![[3, 1, 2], [1, 1, 0]];
    let k2 = np::array![[0, 0, 1], [2, 2, 2]];
    let r = np::lexsort(&[&k1, &k2], 1);
    assert_eq!(r.shape(), [2, 3]);
    assert_eq!(r.to_vec(), [1, 0, 2, 2, 0, 1]);
    assert_eq!(np::lexsort(&[&k1, &k2], 0).to_vec(), [0, 0, 0, 1, 1, 1]);
}

#[test]
fn lexsort_nan_last_and_single_key() {
    let k = np::array![2., NAN, 1., NAN, 0.];
    assert_eq!(np::lexsort(&[&k], 0).to_vec(), [4, 2, 0, 1, 3]);
    let primary = np::array![NAN, 2., NAN, 1., NAN];
    let secondary = np::array![1., 1., 1., 0., 0.];
    assert_eq!(
        np::lexsort(&[&secondary, &primary], 0).to_vec(),
        [3, 1, 4, 0, 2]
    );
}

#[test]
#[should_panic(expected = "all keys need to be the same shape")]
fn lexsort_shape_mismatch() {
    np::lexsort(&[&np::arange(0, 3, 1), &np::arange(0, 4, 1)], -1);
}

#[test]
#[should_panic(expected = "need sequence of keys with len > 0 in lexsort")]
fn lexsort_no_keys() {
    np::lexsort::<i32>(&[], -1);
}

fn lcg(n: usize, modulus: u64, seed: u64) -> Vec<i64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) % modulus) as i64
        })
        .collect()
}

fn check_partition(v: &[i64], k: usize) {
    let a = np::Array::from(v.to_vec());
    let p = a.partition(k as isize, -1).to_vec();
    let mut sorted = v.to_vec();
    sorted.sort();
    assert_eq!(p[k], sorted[k], "kth value, k={k}");
    assert!(p[..k].iter().all(|&x| x <= p[k]));
    assert!(p[k + 1..].iter().all(|&x| x >= p[k]));
    let mut again = p.clone();
    again.sort();
    assert_eq!(again, sorted, "partition must be a permutation");
    let i = a.argpartition(k as isize, -1).to_vec();
    assert_eq!(v[i[k]], sorted[k]);
    assert!(i[..k].iter().all(|&j| v[j] <= sorted[k]));
    assert!(i[k + 1..].iter().all(|&j| v[j] >= sorted[k]));
    let mut seen = i.clone();
    seen.sort();
    assert_eq!(seen, (0..v.len()).collect::<Vec<_>>());
}

#[test]
fn partition_random_duplicates_and_adversarial() {
    for (n, m) in [(1, 5), (2, 5), (13, 4), (100, 7), (1000, 3), (5000, 1000)] {
        let v = lcg(n, m, n as u64);
        for k in [0, n / 3, n / 2, n - 1] {
            check_partition(&v, k);
        }
    }
    let asc: Vec<i64> = (0..20000).collect();
    let desc: Vec<i64> = (0..20000).rev().collect();
    let organ: Vec<i64> = (0..10000).chain((0..10000).rev()).collect();
    let constant = vec![7i64; 5000];
    for v in [&asc, &desc, &organ, &constant] {
        for k in [0, v.len() / 2, v.len() - 1] {
            check_partition(v, k);
        }
    }
}

#[test]
fn partition_axes_negative_kth_and_nan() {
    let a = np::array![[7, 1, 5, 3], [2, 9, 4, 8], [6, 0, 3, 3]];
    let p = a.partition(2, 1);
    assert_eq!(p.index_axis(1, 2).to_vec(), [5, 8, 3]);
    assert_eq!(a.partition(-1, 0).index_axis(0, 2).to_vec(), [7, 9, 5, 8]);
    assert_eq!(a.partition(0, 0).index_axis(0, 0).to_vec(), [2, 0, 3, 3]);
    assert_eq!(
        a.t().partition(0, 1).index_axis(1, 0).to_vec(),
        [2, 0, 3, 3]
    );
    let f = np::array![3.0, NAN, 1.0, 2.0];
    let q = f.partition(2, 0).to_vec();
    assert_eq!(q[2], 3.0);
    assert!(q[3].is_nan());
}

#[test]
#[should_panic(expected = "kth(=7) out of bounds (6)")]
fn partition_kth_out_of_bounds() {
    np::arange(0, 6, 1).partition(7, 0);
}

#[test]
#[should_panic(expected = "kth(=-1) out of bounds (6)")]
fn partition_kth_too_negative() {
    np::arange(0, 6, 1).partition(-7, 0);
}

#[test]
#[should_panic(expected = "kth(=6) out of bounds (6)")]
fn argpartition_kth_out_of_bounds() {
    np::arange(0, 6, 1).argpartition(6, 0);
}

#[test]
fn flatnonzero_select() {
    let a = np::array![[0, 1, 0], [2, 0, 3]];
    assert_eq!(np::flatnonzero(&a).to_vec(), [1, 3, 5]);
    let x = np::arange(0, 8, 1);
    let ten = x.map(|v| v * 10);
    let out = np::select(&[&x.lt(2), &x.gt(5)], &[&x, &ten], -1);
    assert_eq!(out.to_vec(), [0, 1, -1, -1, -1, -1, 60, 70]);
    let rev = x.flip(0);
    assert_eq!(
        np::select(&[&x.lt(3)], &[&rev], 7).to_vec(),
        [7, 6, 5, 7, 7, 7, 7, 7]
    );
    let c0 = np::array![[true], [false]];
    let c1 = np::array![true, false, true];
    let b0 = np::array![[1], [2]];
    let b1 = np::array![10, 20, 30];
    let r = np::select(&[&c0, &c1], &[&b0, &b1], 0);
    assert_eq!(r.shape(), [2, 3]);
    assert_eq!(r.to_vec(), [1, 1, 1, 10, 0, 30]);
}

#[test]
#[should_panic(expected = "select with an empty condition list is not possible")]
fn select_empty() {
    np::select::<i32>(&[], &[], 0);
}

#[test]
#[should_panic(expected = "list of cases must be same length as list of conditions")]
fn select_len_mismatch() {
    let x = np::arange(0, 3, 1);
    np::select(&[&x.gt(0)], &[&x, &x], 0);
}

#[test]
fn piecewise_matches_numpy() {
    let x = np::linspace(-2.5, 2.5, 6);
    let y = np::piecewise(&x, &[&x.lt(0.0), &x.ge(0.0)], &[&|v: f64| -v, &|v: f64| v]);
    assert_eq!(y.to_vec(), [2.5, 1.5, 0.5, 0.5, 1.5, 2.5]);
    // k + 1 functions: the last is the "otherwise" piece
    let z = np::piecewise(
        &x,
        &[&x.lt(-1.0), &x.gt(1.0)],
        &[&|_| 1.0, &|_| 2.0, &|_| 9.0],
    );
    assert_eq!(z.to_vec(), [1.0, 1.0, 9.0, 9.0, 2.0, 2.0]);
    // k functions: zero elsewhere
    let w = np::piecewise(&x, &[&x.lt(-1.0)], &[&|_| 1.0]);
    assert_eq!(w.to_vec(), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
#[should_panic(expected = "with 2 condition(s), either 2 or 3 functions are expected")]
fn piecewise_wrong_function_count() {
    let x = np::linspace(0.0, 1.0, 4);
    np::piecewise(&x, &[&x.lt(0.5), &x.gt(0.5)], &[&|v: f64| v]);
}

#[test]
fn extract_place_copyto() {
    let a = np::arange(0, 6, 1).map(|v| v * 2);
    assert_eq!(
        np::extract(&np::array![true, false, true, false, true, false], &a).to_vec(),
        [0, 4, 8]
    );
    let m = np::array![[5, 6], [7, 8]];
    assert_eq!(
        np::extract(&np::array![[true, false], [false, true]], &m).to_vec(),
        [5, 8]
    );

    let mut b = np::arange(0.0, 6.0, 1.0);
    let big = b.gt(2.0);
    np::place(&mut b, &big, &[-1.0, -2.0]);
    assert_eq!(b.to_vec(), [0., 1., 2., -1., -2., -1.]);
    let mut c = np::arange(0, 6, 1).reshape(&[2, 3]);
    let odd = c.map(|v| v % 2 == 1);
    np::place(&mut c, &odd, &[100]);
    assert_eq!(c.to_vec(), [0, 100, 2, 100, 4, 100]);

    let mut d = np::zeros::<i32>([2, 3]);
    np::copyto(&mut d, [1, 2, 3], None);
    np::copyto(
        &mut d,
        9,
        Some(&np::array![[true, false, false], [false, false, true]]),
    );
    assert_eq!(d.to_vec(), [9, 2, 3, 1, 2, 9]);
    let mut e = np::zeros::<i32>([2, 3]);
    np::copyto(
        &mut e,
        np::array![[5], [6]],
        Some(&np::array![true, false, true]),
    );
    assert_eq!(e.to_vec(), [5, 0, 5, 6, 0, 6]);
}

#[test]
fn place_copyto_do_not_touch_clones() {
    let orig = np::arange(0, 4, 1);
    let mut b = orig.clone();
    np::place(&mut b, &orig.gt(1), &[0]);
    np::copyto(&mut b, 5, Some(&orig.lt(1)));
    assert_eq!(orig.to_vec(), [0, 1, 2, 3]);
    assert_eq!(b.to_vec(), [5, 1, 0, 0]);
    // writing through a transposed (non-contiguous) destination
    let mut t = np::arange(0, 6, 1).reshape(&[2, 3]).t();
    np::copyto(&mut t, np::array![[1], [2], [3]], None);
    assert_eq!(t.to_vec(), [1, 1, 2, 2, 3, 3]);
}

#[test]
#[should_panic(expected = "place: mask and data must be the same size")]
fn place_size_mismatch() {
    let mut a = np::arange(0, 6, 1);
    np::place(&mut a, &np::array![true, false], &[1]);
}

#[test]
#[should_panic(expected = "Cannot insert from an empty array!")]
fn place_empty_vals() {
    let mut a = np::arange(0, 3, 1);
    np::place(&mut a, &np::array![true, true, true], &[]);
}

#[test]
#[should_panic(expected = "could not broadcast input array from shape (3,) into shape (6,)")]
fn copyto_bad_shape() {
    let mut a = np::arange(0, 6, 1);
    np::copyto(&mut a, np::arange(0, 3, 1), None);
}

#[test]
#[should_panic(expected = "could not broadcast where mask from shape (3,) into shape (6,)")]
fn copyto_bad_where() {
    let mut a = np::arange(0, 6, 1);
    np::copyto(&mut a, 1, Some(&np::array![true, true, true]));
}

#[test]
fn trim_zeros_1d_and_2d() {
    let t = np::array![0, 0, 1, 2, 0, 3, 0, 0];
    assert_eq!(np::trim_zeros(&t, "fb").to_vec(), [1, 2, 0, 3]);
    assert_eq!(np::trim_zeros(&t, "f").to_vec(), [1, 2, 0, 3, 0, 0]);
    assert_eq!(np::trim_zeros(&t, "b").to_vec(), [0, 0, 1, 2, 0, 3]);
    let z = np::zeros::<f64>([3]);
    assert!(np::trim_zeros(&z, "fb").is_empty());
    assert!(np::trim_zeros(&z, "f").is_empty());
    assert!(np::trim_zeros(&z, "b").is_empty());
    let m = np::array![[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 3, 0], [0, 0, 0, 0]];
    assert_eq!(np::trim_zeros(&m, "fb").to_vec(), [1, 0, 2, 3]);
    assert_eq!(
        np::trim_zeros(&m, "f").to_vec(),
        [1, 0, 0, 2, 3, 0, 0, 0, 0]
    );
    assert_eq!(
        np::trim_zeros(&m, "b").to_vec(),
        [0, 0, 0, 0, 1, 0, 0, 2, 3]
    );
    assert_eq!(np::trim_zeros(&m, "FB").shape(), [2, 2]);
}

#[test]
#[should_panic(expected = "unexpected character(s) in `trim`: 'x'")]
fn trim_zeros_bad_trim() {
    np::trim_zeros(&np::arange(0, 3, 1), "x");
}
