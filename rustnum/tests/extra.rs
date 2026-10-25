//! Public-API tests for the statistics, signal, set and einsum functions at the crate root.
//! Expected values were computed with NumPy 2.3.

use rustnum as np;
use rustnum::{Array, ConvolveMode};

fn close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "length: got {got:?}, want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "element {i}: got {g}, want {w}\n got  {got:?}\n want {want:?}"
        );
    }
}

fn sample_3x5() -> Array<f64> {
    np::array![
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [2.0, 1.5, 0.5, -1.0, 3.0],
        [1.0, 1.0, 2.0, 3.0, 5.0]
    ]
}

fn hist_data() -> Array<f64> {
    np::array![0.1, 0.5, 1.5, 2.2, 2.2, 3.9, 4.0, 4.4, -0.3, 9.0]
}

#[test]
fn cov_and_corrcoef_3x5() {
    let c = np::cov(&sample_3x5(), false, None);
    assert_eq!(c.shape(), &[3, 3]);
    close(
        &c.to_vec(),
        &[
            2.5,
            -0.1250000000000001,
            2.5,
            -0.1250000000000001,
            2.325,
            0.525,
            2.5,
            0.525,
            2.8,
        ],
        1e-12,
    );
    let cb = np::cov(&sample_3x5(), true, None);
    close(
        &cb.to_vec(),
        &[
            2.0,
            -0.10000000000000009,
            2.0,
            -0.10000000000000009,
            1.8600000000000003,
            0.42000000000000004,
            2.0,
            0.42000000000000004,
            2.2399999999999998,
        ],
        1e-12,
    );
    let r = np::corrcoef(&sample_3x5());
    assert_eq!(r.shape(), &[3, 3]);
    close(
        &r.to_vec(),
        &[
            1.0,
            -0.051847584736521304,
            0.9449111825230679,
            -0.051847584736521304,
            1.0,
            0.20576372293827538,
            0.9449111825230679,
            0.20576372293827538,
            1.0,
        ],
        1e-12,
    );
}

#[test]
fn histogram_variants() {
    let a = hist_data();
    let (counts, edges) = np::histogram(&a, 4usize, None, false, None);
    assert_eq!(counts.to_vec(), vec![4.0, 4.0, 1.0, 1.0]);
    close(
        &edges.to_vec(),
        &[
            -0.3,
            2.0250000000000004,
            4.3500000000000005,
            6.675000000000001,
            9.0,
        ],
        1e-12,
    );

    let (counts, edges) = np::histogram(&a, vec![0.0, 1.0, 2.0, 5.0, 10.0], None, false, None);
    assert_eq!(counts.to_vec(), vec![2.0, 1.0, 5.0, 1.0]);
    assert_eq!(edges.to_vec(), vec![0.0, 1.0, 2.0, 5.0, 10.0]);

    let (density, edges) = np::histogram(&a, 3usize, Some((0.0, 4.0)), true, None);
    close(
        &density.to_vec(),
        &[
            0.21428571428571427,
            0.32142857142857145,
            0.21428571428571425,
        ],
        1e-12,
    );
    close(
        &edges.to_vec(),
        &[0.0, 1.3333333333333333, 2.6666666666666665, 4.0],
        1e-12,
    );
    close(
        &np::histogram_bin_edges(&a, 3usize, Some((0.0, 4.0))).to_vec(),
        &edges.to_vec(),
        1e-12,
    );
}

#[test]
fn bincount_digitize_average_quantile() {
    let x = Array::from_vec(vec![0i64, 1, 1, 3, 5], &[5]);
    assert_eq!(
        np::bincount(&x, None, 0).to_vec(),
        vec![1.0, 2.0, 0.0, 1.0, 0.0, 1.0]
    );
    let v = np::array![0.2, 6.4, 3.0, 1.6];
    let bins = np::array![0.0, 1.0, 2.5, 4.0, 10.0];
    assert_eq!(np::digitize(&v, &bins, false).to_vec(), vec![1, 4, 3, 2]);

    let a = np::array![1.0, 2.0, 3.0, 4.0];
    let w = np::array![4.0, 3.0, 2.0, 1.0];
    assert_eq!(np::average(&a, Some(&w)), 2.0);
    assert_eq!(np::average(&a, None), 2.5);

    let d = hist_data();
    assert!((np::quantile(&d, 0.3) - 1.1999999999999997).abs() < 1e-12);
    assert!((np::quantile(&d, 0.5) - 2.2).abs() < 1e-12);
    assert!(np::nanquantile(&np::array![1.0, f64::NAN, 3.0], 0.5) == 2.0);

    let c = sample_3x5();
    let w5 = np::array![1.0, 2.0, 3.0, 4.0, 5.0];
    close(
        &np::average_axis(&c, Some(&w5), 1).to_vec(),
        &[2.6666666666666665, 1.1666666666666667, 3.066666666666667],
        1e-12,
    );
    close(
        &np::quantile_axis(&c, 0.25, 1).to_vec(),
        &[1.0, 0.5, 1.0],
        1e-12,
    );
}

#[test]
fn convolve_and_correlate_modes() {
    let a = np::array![1.0, 2.0, 3.0];
    let v = np::array![0.0, 1.0, 0.5];
    assert_eq!(
        np::convolve(&a, &v, ConvolveMode::Full).to_vec(),
        vec![0.0, 1.0, 2.5, 4.0, 1.5]
    );
    assert_eq!(
        np::convolve(&a, &v, ConvolveMode::Same).to_vec(),
        vec![1.0, 2.5, 4.0]
    );
    let long = np::array![1.0, 2.0, 3.0, 4.0, 5.0];
    let ones = np::array![1.0, 1.0, 1.0];
    assert_eq!(
        np::convolve(&long, &ones, ConvolveMode::Valid).to_vec(),
        vec![6.0, 9.0, 12.0]
    );
    assert_eq!(
        np::correlate(&a, &v, ConvolveMode::Full).to_vec(),
        vec![0.5, 2.0, 3.5, 3.0, 0.0]
    );
}

#[test]
fn interp_matches_numpy() {
    let xp = np::array![1.0, 2.0, 3.0, 4.0];
    let fp = np::array![1.0, 4.0, 9.0, 16.0];
    let x = np::array![0.0, 1.5, 2.5, 4.0, -1.0, 10.0];
    assert_eq!(
        np::interp(&x, &xp, &fp, None, None).to_vec(),
        vec![1.0, 2.5, 6.5, 16.0, 1.0, 16.0]
    );
    let r = np::interp(
        &np::array![0.0, 10.0],
        &np::array![1.0, 2.0],
        &np::array![5.0, 6.0],
        Some(-1.0),
        Some(99.0),
    );
    assert_eq!(r.to_vec(), vec![-1.0, 99.0]);
}

#[test]
fn gradient_2d_diff_trapezoid() {
    let g = np::array![
        [1.0, 4.0, 9.0, 16.0],
        [2.0, 3.0, 5.0, 8.0],
        [0.0, 1.0, 0.0, 1.0]
    ];
    let d = np::gradient(&g, 0.5);
    assert_eq!(d.len(), 2);
    assert_eq!(d[0].shape(), &[3, 4]);
    close(
        &d[0].to_vec(),
        &[
            2.0, -2.0, -8.0, -16.0, -1.0, -3.0, -9.0, -15.0, -4.0, -4.0, -10.0, -14.0,
        ],
        1e-12,
    );
    close(
        &d[1].to_vec(),
        &[6.0, 8.0, 12.0, 14.0, 2.0, 3.0, 5.0, 6.0, 2.0, 0.0, 0.0, 2.0],
        1e-12,
    );
    let d1 = np::gradient(&g, 1.0);
    close(
        &d1[1].to_vec(),
        &[3.0, 4.0, 6.0, 7.0, 1.0, 1.5, 2.5, 3.0, 1.0, 0.0, 0.0, 1.0],
        1e-12,
    );
    let gx = np::gradient_axis(&g, vec![0.0, 1.0, 3.0, 4.0], 1, 1);
    close(
        &gx.to_vec(),
        &[
            3.0,
            2.8333333333333335,
            5.499999999999999,
            7.0,
            1.0,
            1.0,
            2.333333333333333,
            3.0,
            1.0,
            0.5,
            0.5,
            1.0,
        ],
        1e-12,
    );

    assert_eq!(
        np::diff(&g, 1, 1).to_vec(),
        vec![3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 1.0, -1.0, 1.0]
    );
    let d2 = np::diff(&g, 2, 0);
    assert_eq!(d2.shape(), &[1, 4]);
    assert_eq!(d2.to_vec(), vec![-3.0, -1.0, -1.0, 1.0]);

    assert_eq!(
        np::trapezoid(&np::array![1.0, 2.0, 3.0, 4.0], 0.5, 0).item(),
        3.75
    );
    assert_eq!(
        np::trapezoid(&np::array![1.0, 4.0, 9.0], vec![0.0, 1.0, 3.0], 0).item(),
        15.5
    );
}

#[test]
fn cross_and_kron() {
    let c = np::cross(&np::array![1.0, 2.0, 3.0], &np::array![4.0, 5.0, 6.0]);
    assert_eq!(c.to_vec(), vec![-3.0, 6.0, -3.0]);
    let k = np::kron(&np::array![[1, 2], [3, 4]], &np::array![[0, 1], [1, 0]]);
    assert_eq!(k.shape(), &[4, 4]);
    assert_eq!(
        k.to_vec(),
        vec![0, 1, 0, 2, 1, 0, 2, 0, 0, 3, 0, 4, 3, 0, 4, 0]
    );
}

#[test]
fn einsum_matches_numpy() {
    let a = np::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = np::array![[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]];
    let mm = np::einsum("ij,jk->ik", &[&a, &b]);
    assert_eq!(mm.shape(), &[2, 2]);
    assert_eq!(mm.to_vec(), vec![5.0, 11.0, 14.0, 23.0]);

    let s = np::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
    let tr = np::einsum("ii", &[&s]);
    assert_eq!(tr.shape(), &[] as &[usize]);
    assert_eq!(tr.item(), 16.0);
    assert_eq!(np::einsum("ii->i", &[&s]).to_vec(), vec![1.0, 5.0, 10.0]);

    let b1 = Array::from_vec((0..12).map(|v| v as f64 - 3.0).collect(), &[2, 2, 3]);
    let b2 = Array::from_vec((0..12).map(|v| v as f64 * 0.5).collect(), &[2, 3, 2]);
    let bmm = np::einsum("bij,bjk->bik", &[&b1, &b2]);
    assert_eq!(bmm.shape(), &[2, 2, 2]);
    assert_eq!(
        bmm.to_vec(),
        vec![-4.0, -7.0, 5.0, 6.5, 50.0, 56.0, 86.0, 96.5]
    );

    let v = np::array![1.0, 2.0, 3.0];
    let w = np::array![4.0, -5.0, 6.0];
    let dot = np::einsum("i,i->", &[&v, &w]);
    assert_eq!(dot.item(), 12.0);
    assert_eq!(
        np::einsum("i,j->ij", &[&v, &w]).to_vec(),
        vec![4.0, -5.0, 6.0, 8.0, -10.0, 12.0, 12.0, -15.0, 18.0]
    );
    let t = np::einsum("ij->ji", &[&a]);
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn set_functions() {
    let a = np::array![5, 1, 3, 3, 7];
    assert_eq!(
        np::intersect1d(&a, &np::array![3, 7, 9, 1]).to_vec(),
        vec![1, 3, 7]
    );
    assert_eq!(
        np::union1d(&np::array![5, 1, 3, 3], &np::array![2, 9, 1]).to_vec(),
        vec![1, 2, 3, 5, 9]
    );
    assert_eq!(np::setdiff1d(&a, &np::array![3, 9]).to_vec(), vec![1, 5, 7]);
    assert_eq!(
        np::setxor1d(&np::array![1, 2, 3, 4], &np::array![3, 4, 5, 6]).to_vec(),
        vec![1, 2, 5, 6]
    );
    let m = np::isin(&np::array![[1, 2], [3, 4]], &np::array![2, 4, 8]);
    assert_eq!(m.shape(), &[2, 2]);
    assert_eq!(m.to_vec(), vec![false, true, false, true]);
    // the core `unique` is the only unique at the crate root
    assert_eq!(np::unique(&a).to_vec(), vec![1, 3, 5, 7]);
}
