use proptest::prelude::*;
use rustnum as np;
fn reference() -> np::Array<f64> {
    np::arange(0., 24., 1.).reshape(&[2, 3, 4])
}
fn close(actual: np::Array<f64>, expected: &[f64]) {
    assert_eq!(actual.size(), expected.len());
    for (i, (a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-12 * e.abs().max(1.),
            "at {i}: {a} != {e}"
        );
    }
}
// Literal reference results checked against NumPy 2.3.2, not a second reduction algorithm.
#[test]
fn sum_axis_0() {
    let a = reference();
    let result = a.sum_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0,
        ],
    );
}
#[test]
fn prod_axis_0() {
    let a = reference();
    let result = a.prod_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            0.0, 13.0, 28.0, 45.0, 64.0, 85.0, 108.0, 133.0, 160.0, 189.0, 220.0, 253.0,
        ],
    );
}
#[test]
fn min_axis_0() {
    let a = reference();
    let result = a.min_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    );
}
#[test]
fn max_axis_0() {
    let a = reference();
    let result = a.max_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        ],
    );
}
#[test]
fn mean_axis_0() {
    let a = reference();
    let result = a.mean_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ],
    );
}
#[test]
fn median_axis_0() {
    let a = reference();
    let result = a.median_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ],
    );
}
#[test]
fn nansum_axis_0() {
    let a = reference();
    let result = a.nansum_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0,
        ],
    );
}
#[test]
fn nanmean_axis_0() {
    let a = reference();
    let result = a.nanmean_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ],
    );
}
#[test]
fn nanmin_axis_0() {
    let a = reference();
    let result = a.nanmin_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    );
}
#[test]
fn nanmax_axis_0() {
    let a = reference();
    let result = a.nanmax_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        ],
    );
}
#[test]
fn cumsum_axis_0() {
    let a = reference();
    let result = a.cumsum_axis(0);
    assert_eq!(result.shape(), [2, 3, 4]);
    close(
        result,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0,
            20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0,
        ],
    );
}
#[test]
fn cumprod_axis_0() {
    let a = reference();
    let result = a.cumprod_axis(0);
    assert_eq!(result.shape(), [2, 3, 4]);
    close(
        result,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 0.0, 13.0, 28.0, 45.0,
            64.0, 85.0, 108.0, 133.0, 160.0, 189.0, 220.0, 253.0,
        ],
    );
}
#[test]
fn var_axis_0() {
    let a = reference();
    let result = a.var_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0,
        ],
    );
}
#[test]
fn std_axis_0() {
    let a = reference();
    let result = a.std_axis(0);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
    );
}
#[test]
fn var_ddof_axis_0() {
    let a = reference();
    let result = a.var_ddof_axis(0, 1);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0,
        ],
    );
}
#[test]
fn std_ddof_axis_0() {
    let a = reference();
    let result = a.std_ddof_axis(0, 1);
    assert_eq!(result.shape(), [3, 4]);
    close(
        result,
        &[
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
            8.48528137423857,
        ],
    );
}
#[test]
fn argmin_axis_0() {
    assert_eq!(reference().argmin_axis(0).to_vec(), [0; 12]);
}
#[test]
fn argmax_axis_0() {
    assert_eq!(reference().argmax_axis(0).to_vec(), [1; 12]);
}
#[test]
fn any_axis_0() {
    assert_eq!(reference().any_axis(0).to_vec(), [true; 12]);
}
#[test]
fn all_axis_0() {
    assert_eq!(
        reference().all_axis(0).to_vec(),
        [
            false, true, true, true, true, true, true, true, true, true, true, true
        ]
    );
}
#[test]
fn sum_axis_1() {
    let a = reference();
    let result = a.sum_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0]);
}
#[test]
fn prod_axis_1() {
    let a = reference();
    let result = a.prod_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(
        result,
        &[0.0, 45.0, 120.0, 231.0, 3840.0, 4641.0, 5544.0, 6555.0],
    );
}
#[test]
fn min_axis_1() {
    let a = reference();
    let result = a.min_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0]);
}
#[test]
fn max_axis_1() {
    let a = reference();
    let result = a.max_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0]);
}
#[test]
fn mean_axis_1() {
    let a = reference();
    let result = a.mean_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]);
}
#[test]
fn median_axis_1() {
    let a = reference();
    let result = a.median_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]);
}
#[test]
fn nansum_axis_1() {
    let a = reference();
    let result = a.nansum_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0]);
}
#[test]
fn nanmean_axis_1() {
    let a = reference();
    let result = a.nanmean_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]);
}
#[test]
fn nanmin_axis_1() {
    let a = reference();
    let result = a.nanmin_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0]);
}
#[test]
fn nanmax_axis_1() {
    let a = reference();
    let result = a.nanmax_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0]);
}
#[test]
fn cumsum_axis_1() {
    let a = reference();
    let result = a.cumsum_axis(1);
    assert_eq!(result.shape(), [2, 3, 4]);
    close(
        result,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 12.0, 13.0, 14.0,
            15.0, 28.0, 30.0, 32.0, 34.0, 48.0, 51.0, 54.0, 57.0,
        ],
    );
}
#[test]
fn cumprod_axis_1() {
    let a = reference();
    let result = a.cumprod_axis(1);
    assert_eq!(result.shape(), [2, 3, 4]);
    close(
        result,
        &[
            0.0, 1.0, 2.0, 3.0, 0.0, 5.0, 12.0, 21.0, 0.0, 45.0, 120.0, 231.0, 12.0, 13.0, 14.0,
            15.0, 192.0, 221.0, 252.0, 285.0, 3840.0, 4641.0, 5544.0, 6555.0,
        ],
    );
}
#[test]
fn var_axis_1() {
    let a = reference();
    let result = a.var_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(
        result,
        &[
            10.666666666666666,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666,
        ],
    );
}
#[test]
fn std_axis_1() {
    let a = reference();
    let result = a.std_axis(1);
    assert_eq!(result.shape(), [2, 4]);
    close(
        result,
        &[
            3.265986323710904,
            3.265986323710904,
            3.265986323710904,
            3.265986323710904,
            3.265986323710904,
            3.265986323710904,
            3.265986323710904,
            3.265986323710904,
        ],
    );
}
#[test]
fn var_ddof_axis_1() {
    let a = reference();
    let result = a.var_ddof_axis(1, 1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0]);
}
#[test]
fn std_ddof_axis_1() {
    let a = reference();
    let result = a.std_ddof_axis(1, 1);
    assert_eq!(result.shape(), [2, 4]);
    close(result, &[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
}
#[test]
fn argmin_axis_1() {
    assert_eq!(reference().argmin_axis(1).to_vec(), [0; 8]);
}
#[test]
fn argmax_axis_1() {
    assert_eq!(reference().argmax_axis(1).to_vec(), [2; 8]);
}
#[test]
fn any_axis_1() {
    assert_eq!(reference().any_axis(1).to_vec(), [true; 8]);
}
#[test]
fn all_axis_1() {
    assert_eq!(
        reference().all_axis(1).to_vec(),
        [false, true, true, true, true, true, true, true]
    );
}
#[test]
fn sum_axis_2() {
    let a = reference();
    let result = a.sum_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[6.0, 22.0, 38.0, 54.0, 70.0, 86.0]);
}
#[test]
fn prod_axis_2() {
    let a = reference();
    let result = a.prod_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[0.0, 840.0, 7920.0, 32760.0, 93024.0, 212520.0]);
}
#[test]
fn min_axis_2() {
    let a = reference();
    let result = a.min_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[0.0, 4.0, 8.0, 12.0, 16.0, 20.0]);
}
#[test]
fn max_axis_2() {
    let a = reference();
    let result = a.max_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[3.0, 7.0, 11.0, 15.0, 19.0, 23.0]);
}
#[test]
fn mean_axis_2() {
    let a = reference();
    let result = a.mean_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[1.5, 5.5, 9.5, 13.5, 17.5, 21.5]);
}
#[test]
fn median_axis_2() {
    let a = reference();
    let result = a.median_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[1.5, 5.5, 9.5, 13.5, 17.5, 21.5]);
}
#[test]
fn nansum_axis_2() {
    let a = reference();
    let result = a.nansum_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[6.0, 22.0, 38.0, 54.0, 70.0, 86.0]);
}
#[test]
fn nanmean_axis_2() {
    let a = reference();
    let result = a.nanmean_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[1.5, 5.5, 9.5, 13.5, 17.5, 21.5]);
}
#[test]
fn nanmin_axis_2() {
    let a = reference();
    let result = a.nanmin_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[0.0, 4.0, 8.0, 12.0, 16.0, 20.0]);
}
#[test]
fn nanmax_axis_2() {
    let a = reference();
    let result = a.nanmax_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[3.0, 7.0, 11.0, 15.0, 19.0, 23.0]);
}
#[test]
fn cumsum_axis_2() {
    let a = reference();
    let result = a.cumsum_axis(2);
    assert_eq!(result.shape(), [2, 3, 4]);
    close(
        result,
        &[
            0.0, 1.0, 3.0, 6.0, 4.0, 9.0, 15.0, 22.0, 8.0, 17.0, 27.0, 38.0, 12.0, 25.0, 39.0,
            54.0, 16.0, 33.0, 51.0, 70.0, 20.0, 41.0, 63.0, 86.0,
        ],
    );
}
#[test]
fn cumprod_axis_2() {
    let a = reference();
    let result = a.cumprod_axis(2);
    assert_eq!(result.shape(), [2, 3, 4]);
    close(
        result,
        &[
            0.0, 0.0, 0.0, 0.0, 4.0, 20.0, 120.0, 840.0, 8.0, 72.0, 720.0, 7920.0, 12.0, 156.0,
            2184.0, 32760.0, 16.0, 272.0, 4896.0, 93024.0, 20.0, 420.0, 9240.0, 212520.0,
        ],
    );
}
#[test]
fn var_axis_2() {
    let a = reference();
    let result = a.var_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(result, &[1.25, 1.25, 1.25, 1.25, 1.25, 1.25]);
}
#[test]
fn std_axis_2() {
    let a = reference();
    let result = a.std_axis(2);
    assert_eq!(result.shape(), [2, 3]);
    close(
        result,
        &[
            1.118033988749895,
            1.118033988749895,
            1.118033988749895,
            1.118033988749895,
            1.118033988749895,
            1.118033988749895,
        ],
    );
}
#[test]
fn var_ddof_axis_2() {
    let a = reference();
    let result = a.var_ddof_axis(2, 1);
    assert_eq!(result.shape(), [2, 3]);
    close(
        result,
        &[
            1.6666666666666667,
            1.6666666666666667,
            1.6666666666666667,
            1.6666666666666667,
            1.6666666666666667,
            1.6666666666666667,
        ],
    );
}
#[test]
fn std_ddof_axis_2() {
    let a = reference();
    let result = a.std_ddof_axis(2, 1);
    assert_eq!(result.shape(), [2, 3]);
    close(
        result,
        &[
            1.2909944487358056,
            1.2909944487358056,
            1.2909944487358056,
            1.2909944487358056,
            1.2909944487358056,
            1.2909944487358056,
        ],
    );
}
#[test]
fn argmin_axis_2() {
    assert_eq!(reference().argmin_axis(2).to_vec(), [0; 6]);
}
#[test]
fn argmax_axis_2() {
    assert_eq!(reference().argmax_axis(2).to_vec(), [3; 6]);
}
#[test]
fn any_axis_2() {
    assert_eq!(reference().any_axis(2).to_vec(), [true; 6]);
}
#[test]
fn all_axis_2() {
    assert_eq!(
        reference().all_axis(2).to_vec(),
        [false, true, true, true, true, true]
    );
}

#[test]
fn whole_reductions() {
    let a = reference();
    assert_eq!(a.sum(), 276.);
    assert_eq!(a.prod(), 0.);
    assert_eq!(a.min(), 0.);
    assert_eq!(a.max(), 23.);
    assert_eq!(a.argmax(), 23);
    assert_eq!(a.argmin(), 0);
    assert_eq!(a.mean(), 11.5);
    assert_eq!(a.var(), 47.916666666666664);
    assert!((a.std() - 6.922186552431729).abs() < 1e-12);
    assert_eq!(a.var_ddof(1), 50.);
    assert!((a.std_ddof(1) - 7.0710678118654755).abs() < 1e-12);
    assert_eq!(a.median(), 11.5);
    assert_eq!(a.percentile(25.), 5.75);
    assert_eq!(a.percentile(100.), 23.);
    assert_eq!(a.ptp(), 23.);
    assert_eq!(a.count_nonzero(), 23);
    assert!(a.any());
    assert!(!a.all());
    close(
        a.cumsum(),
        &[
            0., 1., 3., 6., 10., 15., 21., 28., 36., 45., 55., 66., 78., 91., 105., 120., 136.,
            153., 171., 190., 210., 231., 253., 276.,
        ],
    );
    assert_eq!(a.cumprod().to_vec(), [0.; 24]);
    assert_eq!(a.nansum(), 276.);
    assert_eq!(a.nanmean(), 11.5);
    assert_eq!(a.nanmin(), 0.);
    assert_eq!(a.nanmax(), 23.);
}
#[test]
fn nan_axes_with_missing_data() {
    let a = np::array![[f64::NAN, 2., f64::NAN], [4., f64::NAN, f64::NAN]];
    close(a.nansum_axis(0), &[4., 2., 0.]);
    close(a.nansum_axis(1), &[2., 4.]);
    assert_eq!(a.nanmean_axis(0).to_vec()[..2], [4., 2.]);
    assert!(a.nanmean_axis(0)[[2]].is_nan());
    close(a.nanmean_axis(1), &[2., 4.]);
    close(a.nanmin_axis(1), &[2., 4.]);
    close(a.nanmax_axis(1), &[2., 4.]);
    assert!(a.nanmin_axis(0)[[2]].is_nan());
    assert!(a.nanmax_axis(0)[[2]].is_nan());
    assert!(a.min().is_nan());
    assert!(a.max().is_nan());
    assert!(a.median().is_nan());
}
#[test]
fn strided_reductions_and_negative_axes() {
    let a = np::array![[1., 2., 3.], [4., 5., 6.]].t();
    close(a.sum_axis(-1), &[5., 7., 9.]);
    close(a.prod_axis(0), &[6., 120.]);
    close(a.cumsum_axis(0), &[1., 4., 3., 9., 6., 15.]);
    close(a.cumprod_axis(0), &[1., 4., 2., 20., 6., 120.]);
    assert_eq!(a.sum(), 21.);
    assert_eq!(a.min(), 1.);
    assert_eq!(a.max(), 6.);
}
#[test]
fn slice_runtime_integer_types() {
    let a = np::arange(0, 60, 1).reshape(&[5, 3, 4]);
    let i: usize = 1;
    let j: i64 = -1;
    assert_eq!(
        a.slice(&np::s![i..i + 2, .., j]).to_vec(),
        [15, 19, 23, 27, 31, 35]
    );
    let i: isize = -2;
    let j: i32 = 2;
    assert_eq!(
        a.slice(&np::s![i.., .., j]).to_vec(),
        [38, 42, 46, 50, 54, 58]
    );
    assert_eq!(
        a.slice(&np::s![..;2,1..;-1,0]).to_vec(),
        [4, 0, 28, 24, 52, 48]
    );
    assert_eq!(a.slice(&np::s![-3..]).shape(), [3, 3, 4]);
    assert_eq!(a.slice(&np::s![.., np::NEW_AXIS]).shape(), [5, 1, 3, 4]);
}
#[test]
fn negative_step_clamping() {
    let a = np::arange(0, 6, 1);
    assert_eq!(a.slice(&np::s![..;-2]).to_vec(), [5, 3, 1]);
    assert_eq!(a.slice(&np::s![1..;-1]).to_vec(), [1, 0]);
    assert_eq!(a.slice(&np::s![99..-99;-2]).to_vec(), [5, 3, 1]);
    assert!(a.slice(&np::s![..-1;-1]).is_empty());
    assert_eq!(a.slice(&np::s![4..=1;-1]).to_vec(), [4, 3, 2, 1]);
}
#[test]
fn cow_all_mutators() {
    let a = np::array![[1, 2, 3], [4, 5, 6]];
    let mut b = a.col(1);
    b[[0]] = 10;
    assert_eq!(b.to_vec(), [10, 5]);
    let mut b = a.t();
    b[&[0, 1][..]] = 20;
    assert_eq!(b.to_vec(), [1, 20, 2, 5, 3, 6]);
    let mut b = a.flip(1);
    b.set(&[1, 0], 30);
    assert_eq!(b.to_vec(), [3, 2, 1, 30, 5, 4]);
    let mut b = a.row(0);
    b.fill(7);
    assert_eq!(b.to_vec(), [7, 7, 7]);
    let mut b = a.col(0);
    b += 2;
    assert_eq!(b.to_vec(), [3, 6]);
    let mut b = a.t();
    b.slice_assign(&np::s![.., 0], 9);
    assert_eq!(b.to_vec(), [9, 4, 9, 5, 9, 6]);
    let mut b = a.row(1);
    b.mapv_inplace(|v| v * 2);
    assert_eq!(b.to_vec(), [8, 10, 12]);
    let mut b = a.clone();
    b.mask_assign(&a.gt(3), 0);
    assert_eq!(b.to_vec(), [1, 2, 3, 0, 0, 0]);
    assert_eq!(a.to_vec(), [1, 2, 3, 4, 5, 6]);
}
#[test]
fn overlapping_rhs_snapshot() {
    let mut a = np::array![0, 1, 2, 3, 4];
    let rhs = a.slice(&np::s![..4]);
    a.slice_assign(&np::s![1..], rhs);
    assert_eq!(a.to_vec(), [0, 0, 1, 2, 3]);
}
#[test]
fn broadcast_writes_independent() {
    let mut a = np::Array::scalar(1).broadcast_to(&[2, 3]);
    a.slice_assign(&np::s![1, ..], 2);
    assert_eq!(a.to_vec(), [1, 1, 1, 2, 2, 2]);
}
#[test]
fn reshape_transposed_and_negative_stride() {
    let a = np::array![[0, 1, 2], [3, 4, 5]];
    assert_eq!(a.t().reshape(&[2, 3]).to_vec(), [0, 3, 1, 4, 2, 5]);
    assert_eq!(a.flip(1).reshape_infer(&[-1]).to_vec(), [2, 1, 0, 5, 4, 3]);
    assert!(a.reshape(&[1, 2, 3]).squeeze_axis(0).is_contiguous());
}
#[test]
fn zero_dimensions() {
    let a = np::Array::scalar(3);
    assert!(!a.is_empty());
    assert_eq!(a.get(&[]), 3);
    assert_eq!(a.t().item(), 3);
    assert_eq!(a.expand_dims(0).squeeze().item(), 3);
    assert_eq!(np::argwhere(&a).shape(), [1, 0]);
    assert_eq!(np::argwhere(&np::Array::scalar(0)).shape(), [0, 0]);
    assert_eq!(
        np::full::<i32>([0, 3], 1).broadcast_to(&[2, 0, 3]).size(),
        0
    );
    assert!(np::zeros::<i32>([2, 0, 3]).flip(0).ravel().is_empty());
    assert_eq!(np::zeros::<i32>([2, 0, 3]).sum_axis(1).to_vec(), [0; 6]);
    assert_eq!(np::zeros::<i32>([2, 0, 3]).prod_axis(1).to_vec(), [1; 6]);
}
#[test]
fn sort_nan_and_unique_counts() {
    let a = np::array![f64::NAN, 2., 1., 2., f64::NAN];
    assert_eq!(a.argsort().to_vec(), [2, 1, 3, 0, 4]);
    assert_eq!(a.sort().to_vec()[..3], [1., 2., 2.]);
    let (u, n) = np::unique_counts(&a);
    assert_eq!(u.to_vec()[..2], [1., 2.]);
    assert!(u[[2]].is_nan());
    assert_eq!(n.to_vec(), [1, 2, 2]);
    assert_eq!(np::searchsorted(&np::array![1., 2., f64::NAN], f64::NAN), 2);
}
#[test]
fn manipulation_values() {
    let a = np::array![[1, 2], [3, 4]];
    assert_eq!(a.roll(-1, -1).to_vec(), [2, 1, 4, 3]);
    assert_eq!(a.repeat(2, 0).to_vec(), [1, 2, 1, 2, 3, 4, 3, 4]);
    assert_eq!(a.tile(&[2, 1]).to_vec(), [1, 2, 3, 4, 1, 2, 3, 4]);
    assert_eq!(
        np::pad(&a, 1, 0).to_vec(),
        [0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        np::concatenate(&[&a, &a], 1).to_vec(),
        [1, 2, 1, 2, 3, 4, 3, 4]
    );
    assert_eq!(np::stack(&[&a, &a], -1).to_vec(), [1, 1, 2, 2, 3, 3, 4, 4]);
    let mut parts = np::split(&a, 2, 0);
    parts[0].fill(9);
    assert_eq!(a.to_vec(), [1, 2, 3, 4]);
}
#[test]
fn large_contiguous_kernels() {
    for n in [32767, 32768, 65536] {
        let a = np::full([n], 2i64);
        let b = np::full([n], 3i64);
        assert_eq!((&a + &b).sum(), 5 * n as i64);
        assert_eq!((2 * &a).sum(), 4 * n as i64);
        assert_eq!(a.square().sum(), 4 * n as i64);
        assert_eq!(a.min(), 2);
        assert_eq!(a.max(), 2);
        assert_eq!(np::ones::<i64>([n]).prod(), 1);
        let mut c = a.clone();
        c += &b;
        assert_eq!(c.sum(), 5 * n as i64);
        let f = np::full([n], 0.5f32);
        assert_eq!(f.sum(), n as f32 * 0.5);
        assert_eq!(f.mean(), 0.5);
        assert_eq!(f.square().sum(), n as f32 * 0.25);
        assert_eq!(np::where_(&a.gt(1), &a, &b).sum(), 2 * n as i64);
    }
}
#[test]
fn dtype_and_cast_precision() {
    assert_eq!(np::array![i64::MAX].astype::<i64>().item(), i64::MAX);
    assert_eq!(np::array![usize::MAX].astype::<usize>().item(), usize::MAX);
    assert_eq!(np::array![false, true].astype::<i32>().to_vec(), [0, 1]);
    assert_eq!(
        np::array![0., f64::NAN, -2.].astype::<bool>().to_vec(),
        [false, true, true]
    );
}
#[test]
fn float_ufunc_edges() {
    assert_eq!(
        np::array![-2.5, -1.5, 0.5, 1.5, 2.5, 3.5].round().to_vec(),
        [-2., -2., 0., 2., 2., 4.]
    );
    assert_eq!(
        np::array![15., 25., 35.].round_decimals(-1).to_vec(),
        [20., 20., 40.]
    );
    assert_eq!(np::array![2, 5].clip(4, 1).to_vec(), [1, 1]);
    assert!(np::array![f64::NAN].relu().item().is_nan());
    assert!(np::array![-1.0f64].sqrt().item().is_nan());
    assert_eq!(np::array![0f64].exp().item(), 1.);
}
#[test]
fn numpy_display_strings() {
    assert_eq!(np::array![1., 2.5, 3.].to_string(), "[1.  2.5 3. ]");
    assert_eq!(
        np::array![1., 2.123456789, 30.].to_string(),
        "[ 1.          2.12345679 30.        ]"
    );
    assert_eq!(
        np::array![1.2f32, 1.2345678, 2.].to_string(),
        "[1.2       1.2345678 2.       ]"
    );
    assert_eq!(
        np::array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.25].to_string(),
        "[ nan  inf -inf 1.25]"
    );
    assert_eq!(np::array![true, true].to_string(), "[ True  True]");
    assert_eq!(np::Array::scalar(1.).to_string(), "1.0");
    assert_eq!(np::Array::scalar(true).to_string(), "True");
    assert_eq!(
        np::arange(0, 30, 1).to_string(),
        "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n 24 25 26 27 28 29]"
    );
}
proptest! {
 #[test] fn transpose_twice(rows in 0usize..10,cols in 0usize..10){
  let a=np::Array::from_shape_fn([rows,cols],|c|(c[0]*10+c[1]) as i32);
  prop_assert_eq!(a.t().t(),a);
 }
 #[test] fn broadcast_add_commutes(x in proptest::collection::vec(-1000i32..1000,1..20),y in -1000i32..1000){
  let a=np::Array::from(x).expand_dims(1);let b=np::array![y,y+1,y+2];
  prop_assert_eq!(&a+&b,&b+&a);
 }
 #[test] fn reshape_roundtrip(rows in 0usize..10,cols in 0usize..10){
  let a=np::Array::from_shape_fn([rows,cols],|c|(c[0]*10+c[1]) as i32);
  prop_assert_eq!(a.ravel().reshape(&[rows,cols]),a);
 }
 #[test] fn reverse_twice(v in proptest::collection::vec(any::<i32>(),0..100)){
  let a=np::Array::from(v);
  prop_assert_eq!(a.slice(&np::s![..;-1]).slice(&np::s![..;-1]),a);
 }
 #[test] fn copy_on_write_preserves_parent(v in proptest::collection::vec(any::<i32>(),1..100),value in any::<i32>()){
  let a=np::Array::from(v.clone());let mut b=a.clone();b[[0]]=value;
  prop_assert_eq!(a.to_vec(),v);prop_assert_eq!(b[[0]],value);
 }
}

#[test]
#[should_panic(expected = "cannot reshape array of size 3 into shape [2, 2]")]
fn panic_from_vec_shape() {
    np::Array::from_vec(vec![1, 2, 3], &[2, 2]);
}

#[test]
#[should_panic(expected = "cannot reshape array of size 6 into shape [4, 2]")]
fn panic_reshape_shape() {
    np::arange(0, 6, 1).reshape(&[4, 2]);
}

#[test]
#[should_panic(expected = "can only specify one unknown dimension")]
fn panic_reshape_two_infers() {
    np::array![1, 2].reshape_infer(&[-1, -1]);
}

#[test]
#[should_panic(expected = "negative dimensions are not allowed")]
fn panic_reshape_negative() {
    np::array![1, 2].reshape_infer(&[-2]);
}

#[test]
#[should_panic(expected = "cannot reshape array of size 0 into shape [0, -1]")]
fn panic_reshape_zero_infer() {
    np::Array::<i32>::default().reshape_infer(&[0, -1]);
}

#[test]
#[should_panic(expected = "len() of unsized object")]
fn panic_scalar_len() {
    np::Array::scalar(1).len();
}

#[test]
#[should_panic(expected = "can only convert an array of size 1 to a Python scalar")]
fn panic_item_count() {
    np::array![1, 2].item();
}

#[test]
#[should_panic(expected = "index 5 is out of bounds for axis 0 with size 3")]
fn panic_get_bounds() {
    np::array![1, 2, 3].get(&[5]);
}

#[test]
#[should_panic(expected = "incorrect number of indices for array")]
fn panic_index_rank() {
    np::array![1, 2].get(&[0, 0]);
}

#[test]
#[should_panic(expected = "index 2 is out of bounds for axis 0 with size 2")]
fn panic_set_bounds() {
    let mut a = np::array![1, 2];
    a.set(&[2], 3);
}

#[test]
#[should_panic(expected = "axis 3 is out of bounds for array of dimension 3")]
fn panic_invalid_axis() {
    reference().sum_axis(3);
}

#[test]
#[should_panic(expected = "axis -4 is out of bounds for array of dimension 3")]
fn panic_invalid_negative_axis() {
    reference().mean_axis(-4);
}

#[test]
#[should_panic(expected = "operands could not be broadcast together with shapes [2, 3] [4]")]
fn panic_broadcast_shape() {
    let _ = np::zeros::<i32>([2, 3]) + np::zeros::<i32>([4]);
}

#[test]
#[should_panic(expected = "operands could not be broadcast together with shapes [2] [3]")]
fn panic_broadcast_to_shape() {
    np::array![1, 2].broadcast_to(&[3]);
}

#[test]
#[should_panic(expected = "operands could not be broadcast together with shapes [2] [3]")]
fn panic_broadcast_arrays_shape() {
    np::broadcast_arrays(&np::array![1, 2], &np::array![1, 2, 3]);
}

#[test]
#[should_panic(expected = "non-broadcastable output operand")]
fn panic_assignment_shape() {
    let mut a = np::array![1, 2];
    a += np::ones::<i32>([2, 2]);
}

#[test]
#[should_panic(expected = "arange: step must not be zero")]
fn panic_arange_zero() {
    np::arange(0, 3, 0);
}

#[test]
#[should_panic(expected = "arange: cannot compute length")]
fn panic_arange_nan() {
    np::arange(0., f64::NAN, 1.);
}

#[test]
#[should_panic(expected = "Maximum allowed size exceeded")]
fn panic_arange_infinite() {
    np::arange(0., f64::INFINITY, 1.);
}

#[test]
#[should_panic(expected = "array is too big")]
fn panic_shape_overflow() {
    np::Array::<i32>::from_shape_fn([usize::MAX, 2], |_| 0);
}

#[test]
#[should_panic(expected = "Input must be 1- or 2-d.")]
fn panic_diag_rank() {
    np::diag(&np::zeros::<i32>([2, 2, 2]));
}

#[test]
#[should_panic(expected = "slice step cannot be zero")]
fn panic_slice_step_zero() {
    np::array![1, 2].slice(&np::s![..;0]);
}

#[test]
#[should_panic(expected = "too many indices for array")]
fn panic_slice_too_many() {
    np::array![1, 2].slice(&np::s![0, 0]);
}

#[test]
#[should_panic(expected = "index -3 is out of bounds for axis 0 with size 2")]
fn panic_slice_negative_oob() {
    np::array![1, 2].slice(&np::s![-3]);
}

#[test]
#[should_panic(expected = "operands could not be broadcast together")]
fn panic_slice_assignment_shape() {
    let mut a = np::array![1, 2];
    a.slice_assign(&np::s![..], [1, 2, 3]);
}

#[test]
#[should_panic(expected = "input array must be 2-d")]
fn panic_row_rank() {
    np::array![1, 2].row(0);
}

#[test]
#[should_panic(expected = "input array must be 2-d")]
fn panic_col_rank() {
    np::array![1, 2].col(0);
}

#[test]
#[should_panic(expected = "boolean index did not match indexed array")]
fn panic_mask_shape() {
    np::array![1, 2].mask(&np::array![true]);
}

#[test]
#[should_panic(expected = "boolean index did not match indexed array")]
fn panic_mask_assign_shape() {
    let mut a = np::array![1, 2];
    a.mask_assign(&np::array![true], 0);
}

#[test]
#[should_panic(expected = "index 2 is out of bounds for axis 0 with size 2")]
fn panic_take_oob() {
    np::array![1, 2].take(&[2], 0);
}

#[test]
#[should_panic(expected = "condition must be a 1-d array")]
fn panic_compress_rank() {
    np::array![1, 2].compress(&np::array![[true, false]], 0);
}

#[test]
fn integer_zero_division_gives_zero_like_numpy() {
    // np.array([1, -7], np.int32) // 0 == [0, 0] (NumPy warns, never raises)
    assert_eq!((np::array![1i32, -7] / 0).to_vec(), [0, 0]);
    assert_eq!((np::array![1i32, -7] % 0).to_vec(), [0, 0]);
}

#[test]
#[should_panic(expected = "Percentiles must be in the range [0, 100]")]
fn panic_percentile_range() {
    reference().percentile(101.);
}

#[test]
#[should_panic(expected = "Percentiles must be in the range [0, 100]")]
fn panic_percentile_nan() {
    reference().percentile(f64::NAN);
}

#[test]
#[should_panic(expected = "axes don't match array")]
fn panic_transpose_rank() {
    reference().transpose(&[0, 1]);
}

#[test]
#[should_panic(expected = "repeated axis in transpose")]
fn panic_transpose_repeated() {
    reference().transpose(&[0, 1, 1]);
}

#[test]
#[should_panic(expected = "axis 3 is out of bounds for array of dimension 3")]
fn panic_transpose_oob() {
    reference().transpose(&[0, 1, 3]);
}

#[test]
#[should_panic(expected = "cannot select an axis to squeeze out which has size not equal to one")]
fn panic_squeeze_size() {
    np::array![1, 2].squeeze_axis(0);
}

#[test]
#[should_panic(expected = "axis 2 is out of bounds for array of dimension 2")]
fn panic_expand_axis() {
    np::array![1, 2].expand_dims(2);
}

#[test]
#[should_panic(expected = "need at least one array to concatenate")]
fn panic_concatenate_empty() {
    np::concatenate::<i32>(&[], 0);
}

#[test]
#[should_panic(expected = "zero-dimensional arrays cannot be concatenated")]
fn panic_concatenate_scalar() {
    np::concatenate(&[&np::Array::scalar(1)], 0);
}

#[test]
#[should_panic(expected = "all the input arrays must have same number of dimensions")]
fn panic_concatenate_rank() {
    np::concatenate(&[&np::array![1, 2], &np::array![[1, 2]]], 0);
}

#[test]
#[should_panic(
    expected = "all the input array dimensions except for the concatenation axis must match exactly"
)]
fn panic_concatenate_shape() {
    np::concatenate(&[&np::array![[1, 2]], &np::array![[1, 2, 3]]], 0);
}

#[test]
#[should_panic(expected = "need at least one array to stack")]
fn panic_stack_empty() {
    np::stack::<i32>(&[], 0);
}

#[test]
#[should_panic(expected = "all input arrays must have the same shape")]
fn panic_stack_shape() {
    np::stack(&[&np::array![1], &np::array![1, 2]], 0);
}

#[test]
#[should_panic(expected = "number sections must be larger than 0")]
fn panic_split_zero() {
    np::split(&np::array![1, 2], 0, 0);
}

#[test]
#[should_panic(expected = "number sections must be larger than 0")]
fn panic_array_split_zero() {
    np::array_split(&np::array![1, 2], 0, 0);
}

#[test]
#[should_panic(expected = "array split does not result in an equal division")]
fn panic_split_unequal() {
    np::split(&np::array![1, 2, 3], 2, 0);
}

#[test]
#[should_panic(expected = "Calling nonzero on 0d arrays is not allowed")]
fn panic_nonzero_scalar() {
    np::nonzero(&np::Array::scalar(1));
}

#[test]
#[should_panic(expected = "object too deep for desired array")]
fn panic_searchsorted_rank() {
    np::searchsorted(&np::array![[1, 2]], 1);
}

#[test]
#[should_panic(expected = "Integers to negative integer powers are not allowed.")]
fn panic_power_negative_int() {
    np::power(2i32, -1);
}

#[test]
#[should_panic(expected = "setting an array element with a sequence")]
fn panic_ragged_3d() {
    let _ = np::array![[[1], [2]], [[3, 4], [5, 6]]];
}

#[test]
#[should_panic(expected = "index is too large")]
fn panic_slice_index_overflow() {
    let _ = np::s![usize::MAX];
}

#[test]
#[should_panic(expected = "step requires a slice")]
fn panic_slice_step_index() {
    let _ = np::s![0;2];
}

#[test]
#[should_panic(expected = "slice step is too large")]
fn panic_slice_step_overflow() {
    let _ = np::s![..;usize::MAX];
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation minimum which has no identity")]
fn panic_empty_min() {
    np::Array::<f64>::default().min();
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation minimum which has no identity")]
fn panic_empty_min_axis() {
    np::zeros::<f64>([2, 0, 3]).min_axis(1);
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation maximum which has no identity")]
fn panic_empty_max() {
    np::Array::<f64>::default().max();
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation maximum which has no identity")]
fn panic_empty_max_axis() {
    np::zeros::<f64>([2, 0, 3]).max_axis(1);
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation minimum which has no identity")]
fn panic_empty_argmin() {
    np::Array::<f64>::default().argmin();
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation minimum which has no identity")]
fn panic_empty_argmin_axis() {
    np::zeros::<f64>([2, 0, 3]).argmin_axis(1);
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation maximum which has no identity")]
fn panic_empty_argmax() {
    np::Array::<f64>::default().argmax();
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation maximum which has no identity")]
fn panic_empty_argmax_axis() {
    np::zeros::<f64>([2, 0, 3]).argmax_axis(1);
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation minimum which has no identity")]
fn panic_empty_nanmin() {
    np::Array::<f64>::default().nanmin();
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation minimum which has no identity")]
fn panic_empty_nanmin_axis() {
    np::zeros::<f64>([2, 0, 3]).nanmin_axis(1);
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation maximum which has no identity")]
fn panic_empty_nanmax() {
    np::Array::<f64>::default().nanmax();
}

#[test]
#[should_panic(expected = "zero-size array to reduction operation maximum which has no identity")]
fn panic_empty_nanmax_axis() {
    np::zeros::<f64>([2, 0, 3]).nanmax_axis(1);
}

#[test]
fn operators_add_f32() {
    let a: np::Array<f32> = np::array![4f32];
    let b: np::Array<f32> = np::array![2f32];
    assert_eq!((&a + &b).item(), 6f32);
    assert_eq!((a.clone() + &b).item(), 6f32);
    assert_eq!((&a + b.clone()).item(), 6f32);
    assert_eq!((a.clone() + b.clone()).item(), 6f32);
    assert_eq!((&a + 2f32).item(), 6f32);
    assert_eq!((a.clone() + 2f32).item(), 6f32);
    assert_eq!((4f32 + &b).item(), 6f32);
    assert_eq!((4f32 + b.clone()).item(), 6f32);
    let mut c = a.clone();
    c += &b;
    assert_eq!(c.item(), 6f32);
    let mut c = a.clone();
    c += b;
    assert_eq!(c.item(), 6f32);
    let mut c = a;
    c += 2f32;
    assert_eq!(c.item(), 6f32);
}

#[test]
fn operators_sub_f32() {
    let a: np::Array<f32> = np::array![4f32];
    let b: np::Array<f32> = np::array![2f32];
    assert_eq!((&a - &b).item(), 2f32);
    assert_eq!((a.clone() - &b).item(), 2f32);
    assert_eq!((&a - b.clone()).item(), 2f32);
    assert_eq!((a.clone() - b.clone()).item(), 2f32);
    assert_eq!((&a - 2f32).item(), 2f32);
    assert_eq!((a.clone() - 2f32).item(), 2f32);
    assert_eq!((4f32 - &b).item(), 2f32);
    assert_eq!((4f32 - b.clone()).item(), 2f32);
    let mut c = a.clone();
    c -= &b;
    assert_eq!(c.item(), 2f32);
    let mut c = a.clone();
    c -= b;
    assert_eq!(c.item(), 2f32);
    let mut c = a;
    c -= 2f32;
    assert_eq!(c.item(), 2f32);
}

#[test]
fn operators_mul_f32() {
    let a: np::Array<f32> = np::array![4f32];
    let b: np::Array<f32> = np::array![2f32];
    assert_eq!((&a * &b).item(), 8f32);
    assert_eq!((a.clone() * &b).item(), 8f32);
    assert_eq!((&a * b.clone()).item(), 8f32);
    assert_eq!((a.clone() * b.clone()).item(), 8f32);
    assert_eq!((&a * 2f32).item(), 8f32);
    assert_eq!((a.clone() * 2f32).item(), 8f32);
    assert_eq!((4f32 * &b).item(), 8f32);
    assert_eq!((4f32 * b.clone()).item(), 8f32);
    let mut c = a.clone();
    c *= &b;
    assert_eq!(c.item(), 8f32);
    let mut c = a.clone();
    c *= b;
    assert_eq!(c.item(), 8f32);
    let mut c = a;
    c *= 2f32;
    assert_eq!(c.item(), 8f32);
}

#[test]
fn operators_div_f32() {
    let a: np::Array<f32> = np::array![4f32];
    let b: np::Array<f32> = np::array![2f32];
    assert_eq!((&a / &b).item(), 2f32);
    assert_eq!((a.clone() / &b).item(), 2f32);
    assert_eq!((&a / b.clone()).item(), 2f32);
    assert_eq!((a.clone() / b.clone()).item(), 2f32);
    assert_eq!((&a / 2f32).item(), 2f32);
    assert_eq!((a.clone() / 2f32).item(), 2f32);
    assert_eq!((4f32 / &b).item(), 2f32);
    assert_eq!((4f32 / b.clone()).item(), 2f32);
    let mut c = a.clone();
    c /= &b;
    assert_eq!(c.item(), 2f32);
    let mut c = a.clone();
    c /= b;
    assert_eq!(c.item(), 2f32);
    let mut c = a;
    c /= 2f32;
    assert_eq!(c.item(), 2f32);
}

#[test]
fn operators_rem_f32() {
    let a: np::Array<f32> = np::array![4f32];
    let b: np::Array<f32> = np::array![2f32];
    assert_eq!((&a % &b).item(), 0f32);
    assert_eq!((a.clone() % &b).item(), 0f32);
    assert_eq!((&a % b.clone()).item(), 0f32);
    assert_eq!((a.clone() % b.clone()).item(), 0f32);
    assert_eq!((&a % 2f32).item(), 0f32);
    assert_eq!((a.clone() % 2f32).item(), 0f32);
    assert_eq!((4f32 % &b).item(), 0f32);
    assert_eq!((4f32 % b.clone()).item(), 0f32);
    let mut c = a.clone();
    c %= &b;
    assert_eq!(c.item(), 0f32);
    let mut c = a.clone();
    c %= b;
    assert_eq!(c.item(), 0f32);
    let mut c = a;
    c %= 2f32;
    assert_eq!(c.item(), 0f32);
}

#[test]
fn operators_add_f64() {
    let a: np::Array<f64> = np::array![4f64];
    let b: np::Array<f64> = np::array![2f64];
    assert_eq!((&a + &b).item(), 6f64);
    assert_eq!((a.clone() + &b).item(), 6f64);
    assert_eq!((&a + b.clone()).item(), 6f64);
    assert_eq!((a.clone() + b.clone()).item(), 6f64);
    assert_eq!((&a + 2f64).item(), 6f64);
    assert_eq!((a.clone() + 2f64).item(), 6f64);
    assert_eq!((4f64 + &b).item(), 6f64);
    assert_eq!((4f64 + b.clone()).item(), 6f64);
    let mut c = a.clone();
    c += &b;
    assert_eq!(c.item(), 6f64);
    let mut c = a.clone();
    c += b;
    assert_eq!(c.item(), 6f64);
    let mut c = a;
    c += 2f64;
    assert_eq!(c.item(), 6f64);
}

#[test]
fn operators_sub_f64() {
    let a: np::Array<f64> = np::array![4f64];
    let b: np::Array<f64> = np::array![2f64];
    assert_eq!((&a - &b).item(), 2f64);
    assert_eq!((a.clone() - &b).item(), 2f64);
    assert_eq!((&a - b.clone()).item(), 2f64);
    assert_eq!((a.clone() - b.clone()).item(), 2f64);
    assert_eq!((&a - 2f64).item(), 2f64);
    assert_eq!((a.clone() - 2f64).item(), 2f64);
    assert_eq!((4f64 - &b).item(), 2f64);
    assert_eq!((4f64 - b.clone()).item(), 2f64);
    let mut c = a.clone();
    c -= &b;
    assert_eq!(c.item(), 2f64);
    let mut c = a.clone();
    c -= b;
    assert_eq!(c.item(), 2f64);
    let mut c = a;
    c -= 2f64;
    assert_eq!(c.item(), 2f64);
}

#[test]
fn operators_mul_f64() {
    let a: np::Array<f64> = np::array![4f64];
    let b: np::Array<f64> = np::array![2f64];
    assert_eq!((&a * &b).item(), 8f64);
    assert_eq!((a.clone() * &b).item(), 8f64);
    assert_eq!((&a * b.clone()).item(), 8f64);
    assert_eq!((a.clone() * b.clone()).item(), 8f64);
    assert_eq!((&a * 2f64).item(), 8f64);
    assert_eq!((a.clone() * 2f64).item(), 8f64);
    assert_eq!((4f64 * &b).item(), 8f64);
    assert_eq!((4f64 * b.clone()).item(), 8f64);
    let mut c = a.clone();
    c *= &b;
    assert_eq!(c.item(), 8f64);
    let mut c = a.clone();
    c *= b;
    assert_eq!(c.item(), 8f64);
    let mut c = a;
    c *= 2f64;
    assert_eq!(c.item(), 8f64);
}

#[test]
fn operators_div_f64() {
    let a: np::Array<f64> = np::array![4f64];
    let b: np::Array<f64> = np::array![2f64];
    assert_eq!((&a / &b).item(), 2f64);
    assert_eq!((a.clone() / &b).item(), 2f64);
    assert_eq!((&a / b.clone()).item(), 2f64);
    assert_eq!((a.clone() / b.clone()).item(), 2f64);
    assert_eq!((&a / 2f64).item(), 2f64);
    assert_eq!((a.clone() / 2f64).item(), 2f64);
    assert_eq!((4f64 / &b).item(), 2f64);
    assert_eq!((4f64 / b.clone()).item(), 2f64);
    let mut c = a.clone();
    c /= &b;
    assert_eq!(c.item(), 2f64);
    let mut c = a.clone();
    c /= b;
    assert_eq!(c.item(), 2f64);
    let mut c = a;
    c /= 2f64;
    assert_eq!(c.item(), 2f64);
}

#[test]
fn operators_rem_f64() {
    let a: np::Array<f64> = np::array![4f64];
    let b: np::Array<f64> = np::array![2f64];
    assert_eq!((&a % &b).item(), 0f64);
    assert_eq!((a.clone() % &b).item(), 0f64);
    assert_eq!((&a % b.clone()).item(), 0f64);
    assert_eq!((a.clone() % b.clone()).item(), 0f64);
    assert_eq!((&a % 2f64).item(), 0f64);
    assert_eq!((a.clone() % 2f64).item(), 0f64);
    assert_eq!((4f64 % &b).item(), 0f64);
    assert_eq!((4f64 % b.clone()).item(), 0f64);
    let mut c = a.clone();
    c %= &b;
    assert_eq!(c.item(), 0f64);
    let mut c = a.clone();
    c %= b;
    assert_eq!(c.item(), 0f64);
    let mut c = a;
    c %= 2f64;
    assert_eq!(c.item(), 0f64);
}

#[test]
fn operators_add_i32() {
    let a: np::Array<i32> = np::array![4i32];
    let b: np::Array<i32> = np::array![2i32];
    assert_eq!((&a + &b).item(), 6i32);
    assert_eq!((a.clone() + &b).item(), 6i32);
    assert_eq!((&a + b.clone()).item(), 6i32);
    assert_eq!((a.clone() + b.clone()).item(), 6i32);
    assert_eq!((&a + 2i32).item(), 6i32);
    assert_eq!((a.clone() + 2i32).item(), 6i32);
    assert_eq!((4i32 + &b).item(), 6i32);
    assert_eq!((4i32 + b.clone()).item(), 6i32);
    let mut c = a.clone();
    c += &b;
    assert_eq!(c.item(), 6i32);
    let mut c = a.clone();
    c += b;
    assert_eq!(c.item(), 6i32);
    let mut c = a;
    c += 2i32;
    assert_eq!(c.item(), 6i32);
}

#[test]
fn operators_sub_i32() {
    let a: np::Array<i32> = np::array![4i32];
    let b: np::Array<i32> = np::array![2i32];
    assert_eq!((&a - &b).item(), 2i32);
    assert_eq!((a.clone() - &b).item(), 2i32);
    assert_eq!((&a - b.clone()).item(), 2i32);
    assert_eq!((a.clone() - b.clone()).item(), 2i32);
    assert_eq!((&a - 2i32).item(), 2i32);
    assert_eq!((a.clone() - 2i32).item(), 2i32);
    assert_eq!((4i32 - &b).item(), 2i32);
    assert_eq!((4i32 - b.clone()).item(), 2i32);
    let mut c = a.clone();
    c -= &b;
    assert_eq!(c.item(), 2i32);
    let mut c = a.clone();
    c -= b;
    assert_eq!(c.item(), 2i32);
    let mut c = a;
    c -= 2i32;
    assert_eq!(c.item(), 2i32);
}

#[test]
fn operators_mul_i32() {
    let a: np::Array<i32> = np::array![4i32];
    let b: np::Array<i32> = np::array![2i32];
    assert_eq!((&a * &b).item(), 8i32);
    assert_eq!((a.clone() * &b).item(), 8i32);
    assert_eq!((&a * b.clone()).item(), 8i32);
    assert_eq!((a.clone() * b.clone()).item(), 8i32);
    assert_eq!((&a * 2i32).item(), 8i32);
    assert_eq!((a.clone() * 2i32).item(), 8i32);
    assert_eq!((4i32 * &b).item(), 8i32);
    assert_eq!((4i32 * b.clone()).item(), 8i32);
    let mut c = a.clone();
    c *= &b;
    assert_eq!(c.item(), 8i32);
    let mut c = a.clone();
    c *= b;
    assert_eq!(c.item(), 8i32);
    let mut c = a;
    c *= 2i32;
    assert_eq!(c.item(), 8i32);
}

#[test]
fn operators_div_i32() {
    let a: np::Array<i32> = np::array![4i32];
    let b: np::Array<i32> = np::array![2i32];
    assert_eq!((&a / &b).item(), 2i32);
    assert_eq!((a.clone() / &b).item(), 2i32);
    assert_eq!((&a / b.clone()).item(), 2i32);
    assert_eq!((a.clone() / b.clone()).item(), 2i32);
    assert_eq!((&a / 2i32).item(), 2i32);
    assert_eq!((a.clone() / 2i32).item(), 2i32);
    assert_eq!((4i32 / &b).item(), 2i32);
    assert_eq!((4i32 / b.clone()).item(), 2i32);
    let mut c = a.clone();
    c /= &b;
    assert_eq!(c.item(), 2i32);
    let mut c = a.clone();
    c /= b;
    assert_eq!(c.item(), 2i32);
    let mut c = a;
    c /= 2i32;
    assert_eq!(c.item(), 2i32);
}

#[test]
fn operators_rem_i32() {
    let a: np::Array<i32> = np::array![4i32];
    let b: np::Array<i32> = np::array![2i32];
    assert_eq!((&a % &b).item(), 0i32);
    assert_eq!((a.clone() % &b).item(), 0i32);
    assert_eq!((&a % b.clone()).item(), 0i32);
    assert_eq!((a.clone() % b.clone()).item(), 0i32);
    assert_eq!((&a % 2i32).item(), 0i32);
    assert_eq!((a.clone() % 2i32).item(), 0i32);
    assert_eq!((4i32 % &b).item(), 0i32);
    assert_eq!((4i32 % b.clone()).item(), 0i32);
    let mut c = a.clone();
    c %= &b;
    assert_eq!(c.item(), 0i32);
    let mut c = a.clone();
    c %= b;
    assert_eq!(c.item(), 0i32);
    let mut c = a;
    c %= 2i32;
    assert_eq!(c.item(), 0i32);
}

#[test]
fn operators_add_i64() {
    let a: np::Array<i64> = np::array![4i64];
    let b: np::Array<i64> = np::array![2i64];
    assert_eq!((&a + &b).item(), 6i64);
    assert_eq!((a.clone() + &b).item(), 6i64);
    assert_eq!((&a + b.clone()).item(), 6i64);
    assert_eq!((a.clone() + b.clone()).item(), 6i64);
    assert_eq!((&a + 2i64).item(), 6i64);
    assert_eq!((a.clone() + 2i64).item(), 6i64);
    assert_eq!((4i64 + &b).item(), 6i64);
    assert_eq!((4i64 + b.clone()).item(), 6i64);
    let mut c = a.clone();
    c += &b;
    assert_eq!(c.item(), 6i64);
    let mut c = a.clone();
    c += b;
    assert_eq!(c.item(), 6i64);
    let mut c = a;
    c += 2i64;
    assert_eq!(c.item(), 6i64);
}

#[test]
fn operators_sub_i64() {
    let a: np::Array<i64> = np::array![4i64];
    let b: np::Array<i64> = np::array![2i64];
    assert_eq!((&a - &b).item(), 2i64);
    assert_eq!((a.clone() - &b).item(), 2i64);
    assert_eq!((&a - b.clone()).item(), 2i64);
    assert_eq!((a.clone() - b.clone()).item(), 2i64);
    assert_eq!((&a - 2i64).item(), 2i64);
    assert_eq!((a.clone() - 2i64).item(), 2i64);
    assert_eq!((4i64 - &b).item(), 2i64);
    assert_eq!((4i64 - b.clone()).item(), 2i64);
    let mut c = a.clone();
    c -= &b;
    assert_eq!(c.item(), 2i64);
    let mut c = a.clone();
    c -= b;
    assert_eq!(c.item(), 2i64);
    let mut c = a;
    c -= 2i64;
    assert_eq!(c.item(), 2i64);
}

#[test]
fn operators_mul_i64() {
    let a: np::Array<i64> = np::array![4i64];
    let b: np::Array<i64> = np::array![2i64];
    assert_eq!((&a * &b).item(), 8i64);
    assert_eq!((a.clone() * &b).item(), 8i64);
    assert_eq!((&a * b.clone()).item(), 8i64);
    assert_eq!((a.clone() * b.clone()).item(), 8i64);
    assert_eq!((&a * 2i64).item(), 8i64);
    assert_eq!((a.clone() * 2i64).item(), 8i64);
    assert_eq!((4i64 * &b).item(), 8i64);
    assert_eq!((4i64 * b.clone()).item(), 8i64);
    let mut c = a.clone();
    c *= &b;
    assert_eq!(c.item(), 8i64);
    let mut c = a.clone();
    c *= b;
    assert_eq!(c.item(), 8i64);
    let mut c = a;
    c *= 2i64;
    assert_eq!(c.item(), 8i64);
}

#[test]
fn operators_div_i64() {
    let a: np::Array<i64> = np::array![4i64];
    let b: np::Array<i64> = np::array![2i64];
    assert_eq!((&a / &b).item(), 2i64);
    assert_eq!((a.clone() / &b).item(), 2i64);
    assert_eq!((&a / b.clone()).item(), 2i64);
    assert_eq!((a.clone() / b.clone()).item(), 2i64);
    assert_eq!((&a / 2i64).item(), 2i64);
    assert_eq!((a.clone() / 2i64).item(), 2i64);
    assert_eq!((4i64 / &b).item(), 2i64);
    assert_eq!((4i64 / b.clone()).item(), 2i64);
    let mut c = a.clone();
    c /= &b;
    assert_eq!(c.item(), 2i64);
    let mut c = a.clone();
    c /= b;
    assert_eq!(c.item(), 2i64);
    let mut c = a;
    c /= 2i64;
    assert_eq!(c.item(), 2i64);
}

#[test]
fn operators_rem_i64() {
    let a: np::Array<i64> = np::array![4i64];
    let b: np::Array<i64> = np::array![2i64];
    assert_eq!((&a % &b).item(), 0i64);
    assert_eq!((a.clone() % &b).item(), 0i64);
    assert_eq!((&a % b.clone()).item(), 0i64);
    assert_eq!((a.clone() % b.clone()).item(), 0i64);
    assert_eq!((&a % 2i64).item(), 0i64);
    assert_eq!((a.clone() % 2i64).item(), 0i64);
    assert_eq!((4i64 % &b).item(), 0i64);
    assert_eq!((4i64 % b.clone()).item(), 0i64);
    let mut c = a.clone();
    c %= &b;
    assert_eq!(c.item(), 0i64);
    let mut c = a.clone();
    c %= b;
    assert_eq!(c.item(), 0i64);
    let mut c = a;
    c %= 2i64;
    assert_eq!(c.item(), 0i64);
}

#[test]
fn operators_add_u8() {
    let a: np::Array<u8> = np::array![4u8];
    let b: np::Array<u8> = np::array![2u8];
    assert_eq!((&a + &b).item(), 6u8);
    assert_eq!((a.clone() + &b).item(), 6u8);
    assert_eq!((&a + b.clone()).item(), 6u8);
    assert_eq!((a.clone() + b.clone()).item(), 6u8);
    assert_eq!((&a + 2u8).item(), 6u8);
    assert_eq!((a.clone() + 2u8).item(), 6u8);
    assert_eq!((4u8 + &b).item(), 6u8);
    assert_eq!((4u8 + b.clone()).item(), 6u8);
    let mut c = a.clone();
    c += &b;
    assert_eq!(c.item(), 6u8);
    let mut c = a.clone();
    c += b;
    assert_eq!(c.item(), 6u8);
    let mut c = a;
    c += 2u8;
    assert_eq!(c.item(), 6u8);
}

#[test]
fn operators_sub_u8() {
    let a: np::Array<u8> = np::array![4u8];
    let b: np::Array<u8> = np::array![2u8];
    assert_eq!((&a - &b).item(), 2u8);
    assert_eq!((a.clone() - &b).item(), 2u8);
    assert_eq!((&a - b.clone()).item(), 2u8);
    assert_eq!((a.clone() - b.clone()).item(), 2u8);
    assert_eq!((&a - 2u8).item(), 2u8);
    assert_eq!((a.clone() - 2u8).item(), 2u8);
    assert_eq!((4u8 - &b).item(), 2u8);
    assert_eq!((4u8 - b.clone()).item(), 2u8);
    let mut c = a.clone();
    c -= &b;
    assert_eq!(c.item(), 2u8);
    let mut c = a.clone();
    c -= b;
    assert_eq!(c.item(), 2u8);
    let mut c = a;
    c -= 2u8;
    assert_eq!(c.item(), 2u8);
}

#[test]
fn operators_mul_u8() {
    let a: np::Array<u8> = np::array![4u8];
    let b: np::Array<u8> = np::array![2u8];
    assert_eq!((&a * &b).item(), 8u8);
    assert_eq!((a.clone() * &b).item(), 8u8);
    assert_eq!((&a * b.clone()).item(), 8u8);
    assert_eq!((a.clone() * b.clone()).item(), 8u8);
    assert_eq!((&a * 2u8).item(), 8u8);
    assert_eq!((a.clone() * 2u8).item(), 8u8);
    assert_eq!((4u8 * &b).item(), 8u8);
    assert_eq!((4u8 * b.clone()).item(), 8u8);
    let mut c = a.clone();
    c *= &b;
    assert_eq!(c.item(), 8u8);
    let mut c = a.clone();
    c *= b;
    assert_eq!(c.item(), 8u8);
    let mut c = a;
    c *= 2u8;
    assert_eq!(c.item(), 8u8);
}

#[test]
fn operators_div_u8() {
    let a: np::Array<u8> = np::array![4u8];
    let b: np::Array<u8> = np::array![2u8];
    assert_eq!((&a / &b).item(), 2u8);
    assert_eq!((a.clone() / &b).item(), 2u8);
    assert_eq!((&a / b.clone()).item(), 2u8);
    assert_eq!((a.clone() / b.clone()).item(), 2u8);
    assert_eq!((&a / 2u8).item(), 2u8);
    assert_eq!((a.clone() / 2u8).item(), 2u8);
    assert_eq!((4u8 / &b).item(), 2u8);
    assert_eq!((4u8 / b.clone()).item(), 2u8);
    let mut c = a.clone();
    c /= &b;
    assert_eq!(c.item(), 2u8);
    let mut c = a.clone();
    c /= b;
    assert_eq!(c.item(), 2u8);
    let mut c = a;
    c /= 2u8;
    assert_eq!(c.item(), 2u8);
}

#[test]
fn operators_rem_u8() {
    let a: np::Array<u8> = np::array![4u8];
    let b: np::Array<u8> = np::array![2u8];
    assert_eq!((&a % &b).item(), 0u8);
    assert_eq!((a.clone() % &b).item(), 0u8);
    assert_eq!((&a % b.clone()).item(), 0u8);
    assert_eq!((a.clone() % b.clone()).item(), 0u8);
    assert_eq!((&a % 2u8).item(), 0u8);
    assert_eq!((a.clone() % 2u8).item(), 0u8);
    assert_eq!((4u8 % &b).item(), 0u8);
    assert_eq!((4u8 % b.clone()).item(), 0u8);
    let mut c = a.clone();
    c %= &b;
    assert_eq!(c.item(), 0u8);
    let mut c = a.clone();
    c %= b;
    assert_eq!(c.item(), 0u8);
    let mut c = a;
    c %= 2u8;
    assert_eq!(c.item(), 0u8);
}

#[test]
fn operators_add_usize() {
    let a: np::Array<usize> = np::array![4usize];
    let b: np::Array<usize> = np::array![2usize];
    assert_eq!((&a + &b).item(), 6usize);
    assert_eq!((a.clone() + &b).item(), 6usize);
    assert_eq!((&a + b.clone()).item(), 6usize);
    assert_eq!((a.clone() + b.clone()).item(), 6usize);
    assert_eq!((&a + 2usize).item(), 6usize);
    assert_eq!((a.clone() + 2usize).item(), 6usize);
    assert_eq!((4usize + &b).item(), 6usize);
    assert_eq!((4usize + b.clone()).item(), 6usize);
    let mut c = a.clone();
    c += &b;
    assert_eq!(c.item(), 6usize);
    let mut c = a.clone();
    c += b;
    assert_eq!(c.item(), 6usize);
    let mut c = a;
    c += 2usize;
    assert_eq!(c.item(), 6usize);
}

#[test]
fn operators_sub_usize() {
    let a: np::Array<usize> = np::array![4usize];
    let b: np::Array<usize> = np::array![2usize];
    assert_eq!((&a - &b).item(), 2usize);
    assert_eq!((a.clone() - &b).item(), 2usize);
    assert_eq!((&a - b.clone()).item(), 2usize);
    assert_eq!((a.clone() - b.clone()).item(), 2usize);
    assert_eq!((&a - 2usize).item(), 2usize);
    assert_eq!((a.clone() - 2usize).item(), 2usize);
    assert_eq!((4usize - &b).item(), 2usize);
    assert_eq!((4usize - b.clone()).item(), 2usize);
    let mut c = a.clone();
    c -= &b;
    assert_eq!(c.item(), 2usize);
    let mut c = a.clone();
    c -= b;
    assert_eq!(c.item(), 2usize);
    let mut c = a;
    c -= 2usize;
    assert_eq!(c.item(), 2usize);
}

#[test]
fn operators_mul_usize() {
    let a: np::Array<usize> = np::array![4usize];
    let b: np::Array<usize> = np::array![2usize];
    assert_eq!((&a * &b).item(), 8usize);
    assert_eq!((a.clone() * &b).item(), 8usize);
    assert_eq!((&a * b.clone()).item(), 8usize);
    assert_eq!((a.clone() * b.clone()).item(), 8usize);
    assert_eq!((&a * 2usize).item(), 8usize);
    assert_eq!((a.clone() * 2usize).item(), 8usize);
    assert_eq!((4usize * &b).item(), 8usize);
    assert_eq!((4usize * b.clone()).item(), 8usize);
    let mut c = a.clone();
    c *= &b;
    assert_eq!(c.item(), 8usize);
    let mut c = a.clone();
    c *= b;
    assert_eq!(c.item(), 8usize);
    let mut c = a;
    c *= 2usize;
    assert_eq!(c.item(), 8usize);
}

#[test]
fn operators_div_usize() {
    let a: np::Array<usize> = np::array![4usize];
    let b: np::Array<usize> = np::array![2usize];
    assert_eq!((&a / &b).item(), 2usize);
    assert_eq!((a.clone() / &b).item(), 2usize);
    assert_eq!((&a / b.clone()).item(), 2usize);
    assert_eq!((a.clone() / b.clone()).item(), 2usize);
    assert_eq!((&a / 2usize).item(), 2usize);
    assert_eq!((a.clone() / 2usize).item(), 2usize);
    assert_eq!((4usize / &b).item(), 2usize);
    assert_eq!((4usize / b.clone()).item(), 2usize);
    let mut c = a.clone();
    c /= &b;
    assert_eq!(c.item(), 2usize);
    let mut c = a.clone();
    c /= b;
    assert_eq!(c.item(), 2usize);
    let mut c = a;
    c /= 2usize;
    assert_eq!(c.item(), 2usize);
}

#[test]
fn operators_rem_usize() {
    let a: np::Array<usize> = np::array![4usize];
    let b: np::Array<usize> = np::array![2usize];
    assert_eq!((&a % &b).item(), 0usize);
    assert_eq!((a.clone() % &b).item(), 0usize);
    assert_eq!((&a % b.clone()).item(), 0usize);
    assert_eq!((a.clone() % b.clone()).item(), 0usize);
    assert_eq!((&a % 2usize).item(), 0usize);
    assert_eq!((a.clone() % 2usize).item(), 0usize);
    assert_eq!((4usize % &b).item(), 0usize);
    assert_eq!((4usize % b.clone()).item(), 0usize);
    let mut c = a.clone();
    c %= &b;
    assert_eq!(c.item(), 0usize);
    let mut c = a.clone();
    c %= b;
    assert_eq!(c.item(), 0usize);
    let mut c = a;
    c %= 2usize;
    assert_eq!(c.item(), 0usize);
}
#[test]
fn bool_and_combinations() {
    let a = np::array![true, false];
    let b = np::array![false, true];
    assert_eq!((&a & &b).to_vec(), [false, false]);
    assert_eq!((a.clone() & &b).to_vec(), [false, false]);
    assert_eq!((&a & b.clone()).to_vec(), [false, false]);
    assert_eq!((a & b).to_vec(), [false, false]);
}
#[test]
fn bool_or_combinations() {
    let a = np::array![true, false];
    let b = np::array![false, true];
    assert_eq!((&a | &b).to_vec(), [true, true]);
    assert_eq!((a.clone() | &b).to_vec(), [true, true]);
    assert_eq!((&a | b.clone()).to_vec(), [true, true]);
    assert_eq!((a | b).to_vec(), [true, true]);
}
#[test]
fn bool_xor_combinations() {
    let a = np::array![true, false];
    let b = np::array![false, true];
    assert_eq!((&a ^ &b).to_vec(), [true, true]);
    assert_eq!((a.clone() ^ &b).to_vec(), [true, true]);
    assert_eq!((&a ^ b.clone()).to_vec(), [true, true]);
    assert_eq!((a ^ b).to_vec(), [true, true]);
}

#[test]
fn unary_and_integer_power() {
    assert_eq!((-np::array![1, -2]).to_vec(), [-1, 2]);
    assert_eq!((-&np::array![1i64, -2]).to_vec(), [-1, 2]);
    assert!((-np::array![0f64]).item().is_sign_negative());
    assert!((-&np::array![0f32]).item().is_sign_negative());
    assert_eq!((!np::array![true, false]).to_vec(), [false, true]);
    assert_eq!((!&np::array![true, false]).to_vec(), [false, true]);
    assert_eq!(np::power([2i64, 3], 3).to_vec(), [8, 27]);
    assert_eq!(np::power(2u8, 8).item(), 0);
    assert_eq!(np::array![i32::MIN].abs().item(), i32::MIN);
    assert_eq!(np::array![0u8, 255].abs().to_vec(), [0, 255]);
    assert!(!np::array![-0.0f64].abs().item().is_sign_negative());
}
#[test]
fn error_source() {
    use std::error::Error;
    let e = np::Error::from(std::io::Error::other("broken"));
    assert_eq!(e.source().unwrap().to_string(), "broken");
    assert!(np::Error::Singular.source().is_none());
}
#[test]
fn callbacks_may_capture_cell() {
    use std::cell::Cell;
    let calls = Cell::new(0);
    let a = np::array![1, 2, 3];
    let b = a.map(|x| {
        calls.set(calls.get() + 1);
        x + 1
    });
    assert_eq!(calls.get(), 3);
    assert_eq!(b.to_vec(), [2, 3, 4]);
}

#[test]
fn numpy_display_fixture_0() {
    let a = np::arange(0i64, 120i64, 1i64).reshape(&[2, 3, 4, 5]);
    assert_eq!(
        a.to_string(),
        "[[[[  0   1   2   3   4]\n   [  5   6   7   8   9]\n   [ 10  11  12  13  14]\n   [ 15  16  17  18  19]]\n\n  [[ 20  21  22  23  24]\n   [ 25  26  27  28  29]\n   [ 30  31  32  33  34]\n   [ 35  36  37  38  39]]\n\n  [[ 40  41  42  43  44]\n   [ 45  46  47  48  49]\n   [ 50  51  52  53  54]\n   [ 55  56  57  58  59]]]\n\n\n [[[ 60  61  62  63  64]\n   [ 65  66  67  68  69]\n   [ 70  71  72  73  74]\n   [ 75  76  77  78  79]]\n\n  [[ 80  81  82  83  84]\n   [ 85  86  87  88  89]\n   [ 90  91  92  93  94]\n   [ 95  96  97  98  99]]\n\n  [[100 101 102 103 104]\n   [105 106 107 108 109]\n   [110 111 112 113 114]\n   [115 116 117 118 119]]]]"
    );
}

#[test]
fn numpy_display_fixture_1() {
    let a = np::arange(0i64, 1100i64, 1i64).reshape(&[10, 110]);
    assert_eq!(
        a.to_string(),
        "[[   0    1    2 ...  107  108  109]\n [ 110  111  112 ...  217  218  219]\n [ 220  221  222 ...  327  328  329]\n ...\n [ 770  771  772 ...  877  878  879]\n [ 880  881  882 ...  987  988  989]\n [ 990  991  992 ... 1097 1098 1099]]"
    );
}

#[test]
fn numpy_display_fixture_2() {
    let a = np::Array::<i64>::from_vec(
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        &[3, 6],
    );
    assert_eq!(
        a.to_string(),
        "[[ 0  1  2  3  4  5]\n [ 6  7  8  9 10 11]\n [12 13 14 15 16 17]]"
    );
}

#[test]
fn numpy_display_fixture_3() {
    let a = np::Array::<f64>::from_vec(vec![], &[2, 0, 3]);
    assert_eq!(a.to_string(), "[]");
}

#[test]
fn numpy_display_fixture_4() {
    let a = np::Array::<f32>::from_vec(vec![1.2, 1.2345678, 2.0], &[3]);
    assert_eq!(a.to_string(), "[1.2       1.2345678 2.       ]");
}

#[test]
fn numpy_display_fixture_5() {
    let a = np::Array::<f64>::from_vec(vec![1e-05, 1.23e-05, -0.0, 1.0], &[4]);
    assert_eq!(a.to_string(), "[ 1.00e-05  1.23e-05 -0.00e+00  1.00e+00]");
}

#[test]
fn numpy_display_fixture_6() {
    let a = np::Array::<f64>::from_vec(vec![100000000.0, 1200000000.0, -300000000.0], &[3]);
    assert_eq!(a.to_string(), "[ 1.0e+08  1.2e+09 -3.0e+08]");
}

#[test]
fn numpy_display_fixture_7() {
    let a = np::Array::<f64>::from_vec(vec![0.0001, 0.0001], &[2]);
    assert_eq!(a.to_string(), "[0.0001 0.0001]");
}

#[test]
fn numpy_display_fixture_8() {
    let a = np::Array::<f64>::from_vec(vec![1e-05, 1e-05], &[2]);
    assert_eq!(a.to_string(), "[1.e-05 1.e-05]");
}

#[test]
fn numpy_display_fixture_9() {
    let a = np::Array::<f64>::from_vec(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY], &[3]);
    assert_eq!(a.to_string(), "[ nan  inf -inf]");
}

#[test]
fn numpy_display_fixture_10() {
    let a = np::Array::<f64>::from_vec(vec![f64::NAN, 1.0, 2.5], &[3]);
    assert_eq!(a.to_string(), "[nan 1.  2.5]");
}

#[test]
fn numpy_display_fixture_11() {
    let a = np::Array::<f64>::from_vec(vec![0.0, -0.0, -2.5, 1.25], &[4]);
    assert_eq!(a.to_string(), "[ 0.   -0.   -2.5   1.25]");
}

#[test]
fn numpy_display_fixture_12() {
    let a = np::Array::<f64>::from_vec(vec![1.9999999999, 0.000123456789], &[2]);
    assert_eq!(a.to_string(), "[2.00000000e+00 1.23456789e-04]");
}

#[test]
fn numpy_display_fixture_13() {
    let a = np::Array::<f64>::from_vec(vec![1.00000001, 1.000000001, 1000.0], &[3]);
    assert_eq!(a.to_string(), "[   1.00000001    1.         1000.        ]");
}

#[test]
fn numpy_display_fixture_14() {
    let a = np::Array::<f64>::from_vec(vec![1e+308, 1e-308], &[2]);
    assert_eq!(a.to_string(), "[1.e+308 1.e-308]");
}

#[test]
fn numpy_display_fixture_15() {
    let a = np::Array::<bool>::from_vec(vec![true, false, true, true], &[2, 2]);
    assert_eq!(a.to_string(), "[[ True False]\n [ True  True]]");
}

#[test]
fn numpy_display_fixture_16() {
    let a = np::Array::<f64>::from_vec(vec![1e+20], &[]);
    assert_eq!(a.to_string(), "1e+20");
}

#[test]
fn numpy_display_fixture_17() {
    let a = np::Array::<f64>::from_vec(vec![1e-05], &[]);
    assert_eq!(a.to_string(), "1e-05");
}

#[test]
fn numpy_display_fixture_18() {
    let a = np::Array::<f64>::from_vec(vec![f64::NAN], &[]);
    assert_eq!(a.to_string(), "nan");
}

#[test]
fn numpy_display_fixture_19() {
    let a = np::arange(0i64, 1001i64, 1i64).reshape(&[1, 1001]);
    assert_eq!(a.to_string(), "[[   0    1    2 ...  998  999 1000]]");
}

#[test]
fn numpy_display_fixture_20() {
    let a = np::Array::<f64>::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 1, 2, 2]);
    assert_eq!(
        a.to_string(),
        "[[[[0. 0.]\n   [0. 0.]]]\n\n\n [[[0. 0.]\n   [0. 0.]]]]"
    );
}
