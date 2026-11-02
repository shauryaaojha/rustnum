//! Group 6: bit operations on integer arrays. Expected values come from NumPy 2.3.2.
use rustnum as np;

#[test]
fn binary_ops_and_operators() {
    let a = np::array![12i64, 10, -3, 255];
    let b = np::array![6i64, 5, 7, 1];
    assert_eq!(np::bitwise_and(&a, &b).to_vec(), [4, 0, 5, 1]);
    assert_eq!(np::bitwise_or(&a, &b).to_vec(), [14, 15, -1, 255]);
    assert_eq!(np::bitwise_xor(&a, &b).to_vec(), [10, 15, -6, 254]);
    assert_eq!(np::bitwise_not(&a).to_vec(), [-13, -11, 2, -256]);
    assert_eq!(np::invert(&a).to_vec(), [-13, -11, 2, -256]);
    assert_eq!((&a & &b).to_vec(), [4, 0, 5, 1]);
    assert_eq!((&a | &b).to_vec(), [14, 15, -1, 255]);
    assert_eq!((&a ^ &b).to_vec(), [10, 15, -6, 254]);
    assert_eq!((!&a).to_vec(), [-13, -11, 2, -256]);
    assert_eq!((&a << &b).to_vec(), [768, 320, -384, 510]);
    assert_eq!((&a >> &b).to_vec(), [0, 0, -1, 127]);
    assert_eq!((a.clone() & 6).to_vec(), [4, 2, 4, 6]);
    assert_eq!((a.clone() << 1).to_vec(), [24, 20, -6, 510]);
    assert_eq!(np::left_shift(&a, &b).to_vec(), [768, 320, -384, 510]);
    assert_eq!(np::right_shift(&a, &b).to_vec(), [0, 0, -1, 127]);
}

#[test]
fn broadcasting_and_other_types() {
    let col = np::array![[1i32], [2]];
    let row = np::array![3i32, 4];
    assert_eq!((&col | &row).to_vec(), [3, 5, 3, 6]);
    assert_eq!((&col << &row).to_vec(), [8, 16, 16, 32]);
    let u = np::array![1u8, 128, 255];
    assert_eq!((&u << 1u8).to_vec(), [2, 0, 254]);
    assert_eq!((!&u).to_vec(), [254, 127, 0]);
    assert_eq!((&u >> 9u8).to_vec(), [0, 0, 0]);
    assert_eq!((&u << 8u8).to_vec(), [0, 0, 0]);
    let z = np::array![5usize, 6];
    assert_eq!((&z ^ 3usize).to_vec(), [6, 5]);
    let i = np::array![1i32, 2, 3];
    assert_eq!((i & np::array![1, 1, 1]).to_vec(), [1, 0, 1]);
}

#[test]
fn shift_counts_out_of_range() {
    let s = np::array![0i32, 1, 2, 31, 32, 33, 64, -1];
    assert_eq!(
        np::left_shift(1, &s).to_vec(),
        [1, 2, 4, i32::MIN, 0, 0, 0, 0]
    );
    assert_eq!(
        np::right_shift(-8, &s).to_vec(),
        [-8, -4, -2, -1, -1, -1, -1, -1]
    );
    assert_eq!(np::right_shift(8, &s).to_vec(), [8, 4, 2, 0, 0, 0, 0, 0]);
}

#[test]
fn bitwise_count_matches_numpy() {
    assert_eq!(
        np::bitwise_count(&np::array![0i32, 1, 255, -7, i32::MIN]).to_vec(),
        [0u8, 1, 8, 3, 1]
    );
    assert_eq!(
        np::bitwise_count(&np::array![i64::MAX, i64::MIN]).to_vec(),
        [63u8, 1]
    );
    assert_eq!(np::bitwise_count(&np::array![255u8]).to_vec(), [8u8]);
}

#[test]
fn packbits_matches_numpy() {
    let bits = np::array![1, 0, 1, 0, 0, 0, 1, 1, 1, 1];
    assert_eq!(np::packbits(&bits, None, "big").to_vec(), [163, 192]);
    assert_eq!(np::packbits(&bits, None, "little").to_vec(), [197, 3]);
    let m = np::array![[1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0]];
    let r = np::packbits(&m, Some(1), "big");
    assert_eq!(r.shape(), [2, 2]);
    assert_eq!(r.to_vec(), [177, 128, 1, 0]);
    let c = np::packbits(&m, Some(0), "big");
    assert_eq!(c.shape(), [1, 9]);
    assert_eq!(c.to_vec(), [128, 0, 128, 128, 0, 0, 0, 192, 128]);
    assert_eq!(np::packbits(&m, None, "big").to_vec(), [177, 128, 128]);
    assert_eq!(
        np::packbits(&m, Some(-1), "little").to_vec(),
        [141, 1, 128, 0]
    );
    assert_eq!(
        np::packbits(&np::array![true, false, true], None, "big").to_vec(),
        [160]
    );
    assert!(np::packbits(&np::Array::<bool>::default(), None, "big").is_empty());
}

#[test]
#[should_panic(expected = "Expected an input array of integer or boolean data type")]
fn packbits_float() {
    np::packbits(&np::array![1.0], None, "big");
}

#[test]
#[should_panic(expected = "'order' must begin with 'l' or 'b'")]
fn packbits_bad_order() {
    np::packbits(&np::array![1], None, "x");
}

#[test]
fn unpackbits_matches_numpy() {
    let p = np::array![[160u8, 1], [7, 255]];
    let flat = np::unpackbits(&p, None, None, "big");
    assert_eq!(
        flat.to_vec(),
        [
            1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1
        ]
    );
    let ax1 = np::unpackbits(&p, Some(1), None, "big");
    assert_eq!(ax1.shape(), [2, 16]);
    assert_eq!(
        ax1.index_axis(0, 0).to_vec(),
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    );
    let ax0 = np::unpackbits(&p, Some(0), Some(3), "big");
    assert_eq!(ax0.shape(), [3, 2]);
    assert_eq!(ax0.to_vec(), [1, 0, 0, 0, 1, 0]);
    let c = np::unpackbits(&p, Some(1), Some(11), "big");
    assert_eq!(
        c.index_axis(0, 0).to_vec(),
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        c.index_axis(0, 1).to_vec(),
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    );
    let neg = np::unpackbits(&p, Some(1), Some(-3), "little");
    assert_eq!(neg.shape(), [2, 13]);
    assert_eq!(
        neg.index_axis(0, 1).to_vec(),
        [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    );
    assert_eq!(
        np::unpackbits(&p, Some(1), Some(20), "big").shape(),
        [2, 20]
    );
    // round trip
    let bits = np::array![1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1];
    let packed = np::packbits(&bits, None, "big");
    let back = np::unpackbits(&packed, None, Some(11), "big");
    assert_eq!(back.to_vec(), [1u8, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1]);
}

#[test]
#[should_panic(expected = "-count larger than number of elements")]
fn unpackbits_too_negative() {
    np::unpackbits(&np::array![[160u8, 1], [7, 255]], Some(1), Some(-17), "big");
}
