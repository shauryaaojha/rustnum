//! Public-API tests for `np::fft` and `np::poly`. Expected values were computed with NumPy 2.3.

use rustnum as np;

fn close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "length: got {got:?}, want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "element {i}: got {g}, want {w}\n got  {got:?}\n want {want:?}"
        );
    }
}

#[test]
fn fft_matches_numpy_for_non_power_of_two_lengths() {
    // n = 6 (composite) and n = 7 (prime).
    let x = np::array![1.0, 2.0, -1.0, 0.5, 3.0, -2.5];
    let (re, im) = np::fft::fft(&x, &np::zeros_like(&x));
    close(
        &re.to_vec(),
        &[3.0, -0.7499999999999998, 0.75, 3.0, 0.75, -0.75],
        1e-13,
    );
    close(
        &im.to_vec(),
        &[
            0.0,
            -0.4330127018922193,
            -7.361215932167728,
            0.0,
            7.361215932167728,
            0.43301270189221924,
        ],
        1e-13,
    );
    let x = np::array![0.5, -1.0, 2.0, 3.5, -0.25, 1.0, 4.0];
    let (re, im) = np::fft::fft(&x, &np::zeros_like(&x));
    close(
        &re.to_vec(),
        &[
            9.75,
            -1.225242216975605,
            -0.8441275495353171,
            -1.0556302334890788,
            -1.0556302334890788,
            -0.8441275495353171,
            -1.225242216975605,
        ],
        1e-13,
    );
    close(
        &im.to_vec(),
        &[
            0.0,
            1.3071654784674827,
            8.240391359281787,
            -0.7047294926260177,
            0.7047294926260177,
            -8.240391359281787,
            -1.3071654784674827,
        ],
        1e-13,
    );
}

#[test]
fn fft_ifft_round_trip() {
    for n in [1usize, 2, 3, 5, 8, 12, 17, 64, 100] {
        let re = np::Array::from_vec(
            (0..n)
                .map(|i| ((i * i) as f64 * 0.37).sin() + 0.1 * i as f64)
                .collect(),
            &[n],
        );
        let im = np::Array::from_vec((0..n).map(|i| (i as f64 * 1.3).cos()).collect(), &[n]);
        let (fr, fi) = np::fft::fft(&re, &im);
        assert_eq!(fr.shape(), &[n]);
        let (br, bi) = np::fft::ifft(&fr, &fi);
        close(&br.to_vec(), &re.to_vec(), 1e-12);
        close(&bi.to_vec(), &im.to_vec(), 1e-12);
    }
}

#[test]
fn fft_batches_over_leading_axes() {
    let m = np::array![[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]];
    let (re, im) = np::fft::fft(&m, &np::zeros_like(&m));
    assert_eq!(re.shape(), &[2, 4]);
    let row = np::array![1.0, 2.0, 3.0, 4.0];
    let (r0, i0) = np::fft::fft(&row, &np::zeros_like(&row));
    assert_eq!(&re.to_vec()[..4], r0.to_vec().as_slice());
    assert_eq!(&im.to_vec()[..4], i0.to_vec().as_slice());
}

#[test]
fn rfft_shapes_and_values() {
    let x = np::array![1.0, 2.0, -1.0, 0.5, 3.0, -2.5];
    let (re, im) = np::fft::rfft(&x);
    assert_eq!(re.shape(), &[4]);
    assert_eq!(im.shape(), &[4]);
    close(&re.to_vec(), &[3.0, -0.75, 0.75, 3.0], 1e-13);
    close(
        &im.to_vec(),
        &[0.0, -0.4330127018922196, -7.361215932167728, 0.0],
        1e-13,
    );

    let x = np::array![0.5, -1.0, 2.0, 3.5, -0.25, 1.0, 4.0];
    let (re, im) = np::fft::rfft(&x);
    assert_eq!(re.shape(), &[4]);
    close(
        &re.to_vec(),
        &[
            9.75,
            -1.2252422169756045,
            -0.8441275495353162,
            -1.0556302334890784,
        ],
        1e-13,
    );
    close(
        &im.to_vec(),
        &[
            0.0,
            1.3071654784674829,
            8.240391359281787,
            -0.7047294926260181,
        ],
        1e-13,
    );

    // 2-d input: transform along the last axis.
    let m = np::array![
        [1.0, 2.0, 3.0, 4.0],
        [0.0, -1.0, 0.5, 2.0],
        [3.0, 3.0, -2.0, 1.0]
    ];
    let (re, im) = np::fft::rfft(&m);
    assert_eq!(re.shape(), &[3, 3]);
    close(
        &re.to_vec(),
        &[10.0, -2.0, -2.0, 1.5, -0.5, -0.5, 5.0, 5.0, -3.0],
        1e-13,
    );
    close(
        &im.to_vec(),
        &[0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, -2.0, 0.0],
        1e-13,
    );
}

#[test]
fn irfft_recovers_even_and_odd_lengths() {
    for x in [
        np::array![1.0, 2.0, -1.0, 0.5, 3.0, -2.5],
        np::array![0.5, -1.0, 2.0, 3.5, -0.25, 1.0, 4.0],
    ] {
        let n = x.len();
        let (re, im) = np::fft::rfft(&x);
        let back = np::fft::irfft(&re, &im, n);
        assert_eq!(back.shape(), &[n]);
        close(&back.to_vec(), &x.to_vec(), 1e-13);
    }
}

#[test]
fn fft2_and_ifft2() {
    let m = np::array![
        [1.0, 2.0, 3.0, 4.0],
        [0.0, -1.0, 0.5, 2.0],
        [3.0, 3.0, -2.0, 1.0]
    ];
    let (re, im) = np::fft::fft2(&m, &np::zeros_like(&m));
    assert_eq!(re.shape(), &[3, 4]);
    close(
        &re.to_vec(),
        &[
            16.5,
            2.5,
            -5.5,
            2.5,
            6.75,
            0.08012701892219276,
            -0.25,
            -8.580127018922193,
            6.75,
            -8.580127018922193,
            -0.25,
            0.08012701892219276,
        ],
        1e-13,
    );
    close(
        &im.to_vec(),
        &[
            0.0,
            3.0,
            0.0,
            -3.0,
            3.031088913245535,
            6.263139720814412,
            -2.1650635094610964,
            3.2631397208144124,
            -3.031088913245535,
            -3.2631397208144124,
            2.1650635094610964,
            -6.263139720814412,
        ],
        1e-13,
    );
    let (br, bi) = np::fft::ifft2(&re, &im);
    close(&br.to_vec(), &m.to_vec(), 1e-13);
    close(&bi.to_vec(), &[0.0; 12], 1e-13);
}

#[test]
fn freq_and_shift() {
    close(
        &np::fft::fftfreq(5, 0.1_f64).to_vec(),
        &[0.0, 2.0, 4.0, -4.0, -2.0],
        1e-13,
    );
    assert_eq!(
        np::fft::rfftfreq(8, 1.0_f64).to_vec(),
        vec![0.0, 0.125, 0.25, 0.375, 0.5]
    );
    let a = np::Array::from_vec((0..12).collect::<Vec<i32>>(), &[3, 4]);
    let s = np::fft::fftshift(&a);
    assert_eq!(s.shape(), &[3, 4]);
    assert_eq!(s.to_vec(), vec![10, 11, 8, 9, 2, 3, 0, 1, 6, 7, 4, 5]);
    assert_eq!(
        np::fft::ifftshift(&a).to_vec(),
        vec![6, 7, 4, 5, 10, 11, 8, 9, 2, 3, 0, 1]
    );
    assert_eq!(np::fft::ifftshift(&s), a);
}

#[test]
fn f32_arrays_work() {
    let x: np::Array<f32> = np::array![1.0f32, 2.0, 3.0, 4.0];
    let (re, im) = np::fft::fft(&x, &np::zeros_like(&x));
    assert_eq!(re.to_vec(), vec![10.0f32, -2.0, -2.0, -2.0]);
    assert_eq!(im.to_vec(), vec![0.0f32, 2.0, 0.0, -2.0]);
}

#[test]
#[should_panic(expected = "same shape")]
fn fft_rejects_mismatched_parts() {
    let _ = np::fft::fft(&np::array![1.0, 2.0], &np::array![1.0, 2.0, 3.0]);
}

#[test]
fn polyfit_and_polyval() {
    let xs = np::array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = np::array![1.0, 2.5, 7.0, 14.0, 26.0, 40.0];
    let c2 = np::poly::polyfit(&xs, &ys, 2);
    assert_eq!(c2.shape(), &[3]);
    close(
        &c2.to_vec(),
        &[1.6517857142857164, -0.4732142857142936, 1.1250000000000138],
        1e-11,
    );
    let c1 = np::poly::polyfit(&xs, &ys, 1);
    close(
        &c1.to_vec(),
        &[7.785714285714283, -4.380952380952383],
        1e-11,
    );
    close(
        &np::poly::polyval(&c2, np::array![0.5, 6.0]).to_vec(),
        &[1.301339285714296, 57.75000000000004],
        1e-11,
    );
    // an exact polynomial is recovered
    let x = np::array![0.0, 1.0, 2.0, 3.0];
    let y = np::array![1.0, 3.0, 7.0, 13.0];
    close(
        &np::poly::polyfit(&x, &y, 2).to_vec(),
        &[1.0, 1.0, 1.0],
        1e-12,
    );
    assert_eq!(
        np::poly::polyval(np::array![3.0, 0.0, 1.0], 2.0).item(),
        13.0
    );
}

#[test]
fn polynomial_arithmetic() {
    let m = np::poly::polymul(np::array![1.0, 2.0], np::array![1.0, -3.0, 4.0]);
    assert_eq!(m.to_vec(), vec![1.0, -1.0, -2.0, 8.0]);
    let (q, r) = np::poly::polydiv(np::array![1.0, -3.0, 2.0, 5.0], np::array![1.0, -1.0]);
    assert_eq!(q.to_vec(), vec![1.0, -2.0, 0.0]);
    assert_eq!(r.to_vec(), vec![5.0]);
    assert_eq!(
        np::poly::polyadd(np::array![1.0, 2.0, 3.0], np::array![4.0, 5.0]).to_vec(),
        vec![1.0, 6.0, 8.0]
    );
    assert_eq!(
        np::poly::polysub(np::array![1.0, 2.0, 3.0], np::array![4.0, 5.0]).to_vec(),
        vec![1.0, -2.0, -2.0]
    );
    assert_eq!(
        np::poly::polyder(np::array![1.0, 2.0, 3.0, 4.0], 1).to_vec(),
        vec![3.0, 4.0, 3.0]
    );
    assert_eq!(
        np::poly::polyint(np::array![3.0, 2.0, 1.0], 1, &[]).to_vec(),
        vec![1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn roots_match_numpy() {
    let (re, im) = np::poly::roots(np::array![1.0, -6.0, 11.0, -6.0]);
    close(&re.to_vec(), &[3.0, 2.0, 1.0], 1e-12);
    close(&im.to_vec(), &[0.0; 3], 1e-12);

    let (re, im) = np::poly::roots(np::array![1.0, 0.0, 1.0]);
    close(&re.to_vec(), &[0.0, 0.0], 1e-12);
    close(&im.to_vec(), &[1.0, -1.0], 1e-12);

    // a trailing zero becomes a zero root appended at the end
    let (re, _) = np::poly::roots(np::array![1.0, -3.0, 2.0, 0.0]);
    close(&re.to_vec(), &[2.0, 1.0, 0.0], 1e-12);

    // a real root plus a complex-conjugate pair
    let (re, im) = np::poly::roots(np::array![1.0, 2.0, 3.0, 4.0]);
    close(
        &re.to_vec(),
        &[
            -1.6506291914393885,
            -0.17468540428030543,
            -0.17468540428030543,
        ],
        1e-12,
    );
    close(
        &im.to_vec(),
        &[0.0, 1.5468688872313967, -1.5468688872313967],
        1e-12,
    );
}
