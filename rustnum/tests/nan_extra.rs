//! Group 1: the remaining nan* reductions. Expected values come from NumPy 2.3.2.
use rustnum as np;

const NAN: f64 = f64::NAN;

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
            "{got:?} vs {want:?}"
        );
    }
}

fn a() -> np::Array<f64> {
    np::array![[1., NAN, 3., 4.], [2., 7., NAN, 1.], [NAN, NAN, NAN, NAN]]
}
fn b() -> np::Array<f64> {
    np::array![
        [[1., NAN, 3.], [4., 5., NAN]],
        [[7., 8., 9.], [NAN, 2., 1.]]
    ]
}
fn c() -> np::Array<f64> {
    np::array![[1., NAN, 3., 4.], [2., 7., NAN, 1.]]
}

#[test]
fn var_std() {
    close(&[a().nanvar()], &[4.333333333333333]);
    close(&[a().nanstd()], &[2.0816659994661326]);
    close(&[a().nanvar_ddof(1)], &[5.2]);
    close(
        &a().nanvar_axis(1).to_vec(),
        &[1.5555555555555556, 6.888888888888889, NAN],
    );
}

#[test]
fn var_std_axis() {
    close(&a().nanstd_axis(0).to_vec(), &[0.5, 0.0, 0.0, 1.5]);
    let v = b().nanvar_ddof_axis(1, 1);
    assert_eq!(v.shape(), [2, 3]);
    close(&v.to_vec(), &[4.5, NAN, NAN, NAN, 18.0, 32.0]);
    let s = b().nanstd_axis(2);
    close(&s.to_vec(), &[1.0, 0.5, 0.816496580927726, 0.5]);
    close(&b().nanstd_ddof_axis(-1, 0).to_vec(), &s.to_vec());
}

#[test]
fn median_percentile() {
    close(&[a().nanmedian()], &[2.5]);
    close(&a().nanmedian_axis(1).to_vec(), &[3.0, 2.0, NAN]);
    let m = b().nanmedian_axis(0);
    assert_eq!(m.shape(), [2, 3]);
    close(&m.to_vec(), &[4.0, 8.0, 6.0, 4.0, 3.5, 1.0]);
    close(&[a().nanpercentile(30.0)], &[1.5]);
    close(&a().nanpercentile_axis(70.0, 1).to_vec(), &[3.4, 4.0, NAN]);
    close(
        &b().nanpercentile_axis(33.0, 2).to_vec(),
        &[1.6600000000000001, 4.33, 7.66, 1.33],
    );
}

#[test]
fn argmax_argmin() {
    assert_eq!(c().nanargmax(), 5);
    assert_eq!(c().nanargmin(), 0);
    assert_eq!(c().nanargmax_axis(0).to_vec(), [1, 1, 0, 0]);
    assert_eq!(c().nanargmin_axis(1).to_vec(), [0, 3]);
}

#[test]
#[should_panic(expected = "All-NaN slice encountered")]
fn argmax_all_nan() {
    np::array![NAN, NAN].nanargmax();
}

#[test]
#[should_panic(expected = "Percentiles must be in the range [0, 100]")]
fn percentile_range() {
    np::array![1.0].nanpercentile(101.0);
}

#[test]
fn prod_cum() {
    assert_eq!(c().nanprod(), 168.0);
    close(&c().nanprod_axis(0).to_vec(), &[2., 7., 3., 4.]);
    close(
        &c().nancumsum().to_vec(),
        &[1., 1., 4., 8., 10., 17., 17., 18.],
    );
    close(
        &c().nancumprod().to_vec(),
        &[1., 1., 3., 12., 24., 168., 168., 168.],
    );
    close(
        &c().nancumsum_axis(0).to_vec(),
        &[1., 0., 3., 4., 3., 7., 3., 5.],
    );
    close(
        &c().nancumprod_axis(1).to_vec(),
        &[1., 1., 3., 12., 2., 14., 14., 14.],
    );
    let cs = b().nancumsum_axis(1);
    assert_eq!(cs.shape(), [2, 2, 3]);
    close(
        &cs.to_vec(),
        &[1., 0., 3., 5., 5., 3., 7., 8., 9., 7., 10., 10.],
    );
}

#[test]
fn non_contiguous_and_f32() {
    let t = c().t();
    close(&t.nanvar_axis(0).to_vec(), &c().nanvar_axis(1).to_vec());
    let f: np::Array<f32> = np::array![1.0f32, f32::NAN, 3.0];
    assert_eq!(f.nanmedian(), 2.0);
    assert_eq!(f.nancumsum().to_vec(), [1.0, 1.0, 4.0]);
}
