//! Integration tests for the misc element-wise math, window functions and `Complex`, on the public API only.
//! Expected values are literal NumPy 2.3 output produced once by a generator script (not computed
//! by the code under test). Exact checks compare bit patterns (so `-0.0` vs `0.0` matters); NaN
//! matches any NaN.
#![allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::type_complexity
)]

use rustnum as np;
use rustnum::{Array, Complex};

/// Compare `got` with NumPy: NaN matches NaN; infinities, zeros (when `atol == 0`) and `rtol == 0`
/// need identical bits; everything else must agree within `atol + rtol * |want|`.
fn arr(v: &[f64]) -> Array<f64> {
    Array::from(v.to_vec())
}

#[track_caller]
fn close(what: &str, got: &Array<f64>, shape: &[usize], want: &[f64], rtol: f64, atol: f64) {
    assert_eq!(got.shape(), shape, "{what}: shape");
    let g = got.to_vec();
    assert_eq!(g.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in g.iter().zip(want).enumerate() {
        let ok = if w.is_nan() {
            g.is_nan()
        } else if w.is_infinite() || (w == 0.0 && atol == 0.0) || (rtol == 0.0 && atol == 0.0) {
            g.to_bits() == w.to_bits()
        } else {
            (g - w).abs() <= atol + rtol * w.abs()
        };
        assert!(
            ok,
            "{what}[{i}]: got {g:e} ({:#x}), NumPy says {w:e} ({:#x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}
#[track_caller]
fn close32(what: &str, got: &Array<f32>, want: &[f32]) {
    let g = got.to_vec();
    assert_eq!(g.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in g.iter().zip(want).enumerate() {
        let ok = if w.is_nan() {
            g.is_nan()
        } else {
            g.to_bits() == w.to_bits()
        };
        assert!(ok, "{what}[{i}]: got {g:e}, NumPy says {w:e}");
    }
}
const X: [f64; 12] = [
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    -0.0,
    0.0,
    0.5,
    -1.5,
    2.5,
    3.75,
    -3.75,
    1e+16,
    -7.0,
];

#[test]
fn sinc_method() {
    let a = arr(&X);
    close(
        "sinc",
        &a.sinc(),
        &[12],
        &[
            f64::NAN,
            f64::NAN,
            f64::NAN,
            1.0,
            1.0,
            0.6366197723675814,
            -0.2122065907891938,
            0.12732395447351627,
            -0.060021087743807156,
            -0.060021087743807156,
            -1.1943397231451733e-17,
            3.8981718325193755e-17,
        ],
        1e-13,
        1e-15,
    );
}

#[test]
fn deg2rad_method() {
    let a = arr(&X);
    close(
        "deg2rad",
        &a.deg2rad(),
        &[12],
        &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            0.008726646259971648,
            -0.026179938779914945,
            0.04363323129985824,
            0.06544984694978735,
            -0.06544984694978735,
            174532925199432.97,
            -0.12217304763960307,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn rad2deg_method() {
    let a = arr(&X);
    close(
        "rad2deg",
        &a.rad2deg(),
        &[12],
        &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            28.64788975654116,
            -85.94366926962348,
            143.2394487827058,
            214.8591731740587,
            -214.8591731740587,
            5.7295779513082323e+17,
            -401.07045659157626,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn rint_method() {
    let a = arr(&X);
    close(
        "rint",
        &a.rint(),
        &[12],
        &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            0.0,
            -2.0,
            2.0,
            4.0,
            -4.0,
            1e+16,
            -7.0,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn fix_method() {
    let a = arr(&X);
    close(
        "fix",
        &a.fix(),
        &[12],
        &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            0.0,
            -1.0,
            2.0,
            3.0,
            -3.0,
            1e+16,
            -7.0,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn nan_to_num_method() {
    let a = arr(&X);
    close(
        "nan_to_num",
        &a.nan_to_num(),
        &[12],
        &[
            0.0,
            1.7976931348623157e+308,
            -1.7976931348623157e+308,
            -0.0,
            0.0,
            0.5,
            -1.5,
            2.5,
            3.75,
            -3.75,
            1e+16,
            -7.0,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn i0_method() {
    let a = arr(&X);
    close(
        "i0",
        &a.i0(),
        &[12],
        &[
            f64::NAN,
            f64::NAN,
            f64::NAN,
            1.0,
            1.0,
            1.0634833707413234,
            1.6467231897728907,
            3.289839144050123,
            9.118945860844567,
            9.118945860844567,
            f64::INFINITY,
            168.59390851028968,
        ],
        1e-13,
        0.0,
    );
}

#[test]
fn fabs_method() {
    let a = arr(&X);
    close(
        "fabs",
        &a.fabs(),
        &[12],
        &[
            f64::NAN,
            f64::INFINITY,
            f64::INFINITY,
            0.0,
            0.0,
            0.5,
            1.5,
            2.5,
            3.75,
            3.75,
            1e+16,
            7.0,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn positive_method() {
    let a = arr(&X);
    close(
        "positive",
        &a.positive(),
        &[12],
        &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            0.5,
            -1.5,
            2.5,
            3.75,
            -3.75,
            1e+16,
            -7.0,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn negative_method() {
    let a = arr(&X);
    close(
        "negative",
        &a.negative(),
        &[12],
        &[
            f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.0,
            -0.0,
            -0.5,
            1.5,
            -2.5,
            -3.75,
            3.75,
            -1e+16,
            7.0,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn spacing_method() {
    let a = arr(&X);
    close(
        "spacing",
        &a.spacing(),
        &[12],
        &[
            f64::NAN,
            f64::NAN,
            f64::NAN,
            5e-324,
            5e-324,
            1.1102230246251565e-16,
            -2.220446049250313e-16,
            4.440892098500626e-16,
            4.440892098500626e-16,
            -4.440892098500626e-16,
            2.0,
            -8.881784197001252e-16,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn signbit_method() {
    assert_eq!(
        arr(&X).signbit().to_vec(),
        [
            false, false, true, true, false, false, true, false, false, true, false, true
        ]
    );
    assert_eq!(
        np::array![-f64::NAN, f64::NAN].signbit().to_vec(),
        [true, false]
    );
}

#[test]
fn modf_method() {
    let (frac, int) = arr(&X).modf();
    close(
        "modf frac",
        &frac,
        &[12],
        &[
            f64::NAN,
            0.0,
            -0.0,
            -0.0,
            0.0,
            0.5,
            -0.5,
            0.5,
            0.75,
            -0.75,
            0.0,
            -0.0,
        ],
        0.0,
        0.0,
    );
    close(
        "modf int",
        &int,
        &[12],
        &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            0.0,
            -1.0,
            2.0,
            3.0,
            -3.0,
            1e+16,
            -7.0,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn frexp_method() {
    let a = arr(&[
        0.0, -0.0, 0.5, 1.0, -3.0, 8.0, 0.1, 1e+300, 5e-324, -2.5e-310, 3.75,
    ]);
    let (m, e) = a.frexp();
    close(
        "frexp m",
        &m,
        &[11],
        &[
            0.0,
            -0.0,
            0.5,
            0.5,
            -0.75,
            0.5,
            0.8,
            0.7466108948025751,
            0.5,
            -0.7190772539449313,
            0.9375,
        ],
        0.0,
        0.0,
    );
    assert_eq!(e.to_vec(), [0, 0, 0, 1, 2, 4, -3, 997, -1073, -1028, 2]);
    // NumPy leaves the exponent of inf/NaN unspecified (-1 on Windows, 0 with glibc); we return 0.
    let (m, e) = np::array![f64::INFINITY, f64::NAN].frexp();
    assert!(m.to_vec()[0].is_infinite() && m.to_vec()[1].is_nan());
    assert_eq!(e.to_vec(), [0, 0]);
}

#[test]
fn nan_to_num_with_method() {
    let r = arr(&X).nan_to_num_with(-1.0, Some(9.0), Some(-9.0));
    close(
        "nan_to_num_with",
        &r,
        &[12],
        &[
            -1.0, 9.0, -9.0, -0.0, 0.0, 0.5, -1.5, 2.5, 3.75, -3.75, 1e+16, -7.0,
        ],
        0.0,
        0.0,
    );
    let r = arr(&X).nan_to_num_with(7.0, None, None);
    close(
        "nan_to_num_with defaults",
        &r,
        &[12],
        &[
            7.0,
            1.7976931348623157e+308,
            -1.7976931348623157e+308,
            -0.0,
            0.0,
            0.5,
            -1.5,
            2.5,
            3.75,
            -3.75,
            1e+16,
            -7.0,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn f32_methods() {
    let a: Array<f32> = arr(&[
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        0.0,
        0.5,
        -1.5,
        2.5,
    ])
    .astype::<f32>();
    let f = |v: f64| v as f32;
    close32(
        "nan_to_num f32",
        &a.nan_to_num(),
        &[
            0.0,
            3.4028234663852886e+38,
            -3.4028234663852886e+38,
            -0.0,
            0.0,
            0.5,
            -1.5,
            2.5,
        ]
        .map(f),
    );
    close32(
        "rint f32",
        &a.rint(),
        &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            0.0,
            -2.0,
            2.0,
        ]
        .map(f),
    );
    close32(
        "spacing f32",
        &np::array![1.0f32, 1e10].spacing(),
        &[1.1920928955078125e-07, 1024.0].map(f),
    );
    close32(
        "deg2rad f32",
        &np::array![180.0f32].deg2rad(),
        &[3.1415927410125732].map(f),
    );
}

#[test]
fn unwrap_methods() {
    let a = arr(&[0.0, 3.0, 6.5, 2.0, -5.0, 1.0]);
    close(
        "unwrap",
        &a.unwrap(),
        &[6],
        &[0.0, 3.0, 0.21681469282041377, 2.0, 1.2831853071795862, 1.0],
        1e-12,
        1e-12,
    );
    close(
        "unwrap period",
        &a.unwrap_with(Some(1.0), -1, 4.0),
        &[6],
        &[0.0, -1.0, -1.5, -2.0, -1.0, 1.0],
        1e-12,
        1e-12,
    );
    close(
        "unwrap discont",
        &a.unwrap_with(Some(5.0), -1, std::f64::consts::TAU),
        &[6],
        &[0.0, 3.0, 6.5, 2.0, 1.2831853071795862, 1.0],
        1e-12,
        1e-12,
    );
    let m = Array::from_vec([0.0, 3.0, 6.5, 0.0, 2.0, 4.0].to_vec(), &[3, 2]);
    close(
        "unwrap axis0",
        &m.unwrap_with(None, 0, std::f64::consts::TAU),
        &[3, 2],
        &[0.0, 3.0, 0.21681469282041377, 0.0, 2.0, -2.2831853071795862],
        1e-12,
        1e-12,
    );
    close(
        "unwrap axis1",
        &m.unwrap(),
        &[3, 2],
        &[0.0, 3.0, 6.5, 6.283185307179586, 2.0, 4.0],
        1e-12,
        1e-12,
    );
    let nan = np::array![0.0, f64::NAN, 3.0, 9.0].unwrap().to_vec();
    assert!(nan[0] == 0.0 && nan[1..].iter().all(|v| v.is_nan()));
    assert_eq!(np::array![1.0f64].unwrap().to_vec(), [1.0]);
}

const A: [f64; 4] = [-7.5, -0.0, f64::INFINITY, 2.0];
const B: [f64; 3] = [3.0, -0.0, f64::NAN];

#[test]
fn logaddexp_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    close(
        "logaddexp",
        &np::logaddexp(&a, &b),
        &[4, 3],
        &[
            3.000027536070229,
            0.0005529314753607964,
            f64::NAN,
            3.048587351573742,
            0.6931471805599453,
            f64::NAN,
            f64::INFINITY,
            f64::INFINITY,
            f64::NAN,
            3.313261687518223,
            2.1269280110429727,
            f64::NAN,
        ],
        1e-14,
        1e-14,
    );
    // scalar operand
    close(
        "logaddexp scalar",
        &np::logaddexp(&a, 3.0),
        &[4, 1],
        &[
            3.000027536070229,
            3.048587351573742,
            f64::INFINITY,
            3.313261687518223,
        ],
        1e-14,
        1e-14,
    );
}

#[test]
fn logaddexp2_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    close(
        "logaddexp2",
        &np::logaddexp2(&a, &b),
        &[4, 3],
        &[
            3.00099588612128,
            0.007947906386200174,
            f64::NAN,
            3.169925001442312,
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::INFINITY,
            f64::NAN,
            3.584962500721156,
            2.321928094887362,
            f64::NAN,
        ],
        1e-14,
        1e-14,
    );
    // scalar operand
    close(
        "logaddexp2 scalar",
        &np::logaddexp2(&a, 3.0),
        &[4, 1],
        &[
            3.00099588612128,
            3.169925001442312,
            f64::INFINITY,
            3.584962500721156,
        ],
        1e-14,
        1e-14,
    );
}

#[test]
fn fmod_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    close(
        "fmod",
        &np::fmod(&a, &b),
        &[4, 3],
        &[
            -1.5,
            f64::NAN,
            f64::NAN,
            -0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            2.0,
            f64::NAN,
            f64::NAN,
        ],
        0.0,
        0.0,
    );
    // scalar operand
    close(
        "fmod scalar",
        &np::fmod(&a, 3.0),
        &[4, 1],
        &[-1.5, -0.0, f64::NAN, 2.0],
        0.0,
        0.0,
    );
}

#[test]
fn remainder_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    close(
        "remainder",
        &np::remainder(&a, &b),
        &[4, 3],
        &[
            1.5,
            f64::NAN,
            f64::NAN,
            0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            2.0,
            f64::NAN,
            f64::NAN,
        ],
        0.0,
        0.0,
    );
    // scalar operand
    close(
        "remainder scalar",
        &np::remainder(&a, 3.0),
        &[4, 1],
        &[1.5, 0.0, f64::NAN, 2.0],
        0.0,
        0.0,
    );
}

#[test]
fn floor_divide_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    close(
        "floor_divide",
        &np::floor_divide(&a, &b),
        &[4, 3],
        &[
            -3.0,
            f64::INFINITY,
            f64::NAN,
            -0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NEG_INFINITY,
            f64::NAN,
            0.0,
            f64::NEG_INFINITY,
            f64::NAN,
        ],
        0.0,
        0.0,
    );
    // scalar operand
    close(
        "floor_divide scalar",
        &np::floor_divide(&a, 3.0),
        &[4, 1],
        &[-3.0, -0.0, f64::NAN, 0.0],
        0.0,
        0.0,
    );
}

#[test]
fn copysign_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    close(
        "copysign",
        &np::copysign(&a, &b),
        &[4, 3],
        &[
            7.5,
            -7.5,
            7.5,
            0.0,
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            2.0,
            -2.0,
            2.0,
        ],
        0.0,
        0.0,
    );
    // scalar operand
    close(
        "copysign scalar",
        &np::copysign(&a, 3.0),
        &[4, 1],
        &[7.5, 0.0, f64::INFINITY, 2.0],
        0.0,
        0.0,
    );
}

#[test]
fn nextafter_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    close(
        "nextafter",
        &np::nextafter(&a, &b),
        &[4, 3],
        &[
            -7.499999999999999,
            -7.499999999999999,
            f64::NAN,
            5e-324,
            -0.0,
            f64::NAN,
            1.7976931348623157e+308,
            1.7976931348623157e+308,
            f64::NAN,
            2.0000000000000004,
            1.9999999999999998,
            f64::NAN,
        ],
        0.0,
        0.0,
    );
    // scalar operand
    close(
        "nextafter scalar",
        &np::nextafter(&a, 3.0),
        &[4, 1],
        &[
            -7.499999999999999,
            5e-324,
            1.7976931348623157e+308,
            2.0000000000000004,
        ],
        0.0,
        0.0,
    );
}

#[test]
fn float_power_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    close(
        "float_power",
        &np::float_power(&a, &b),
        &[4, 3],
        &[
            -421.875,
            1.0,
            f64::NAN,
            -0.0,
            1.0,
            f64::NAN,
            f64::INFINITY,
            1.0,
            f64::NAN,
            8.0,
            1.0,
            f64::NAN,
        ],
        1e-14,
        0.0,
    );
    // scalar operand
    close(
        "float_power scalar",
        &np::float_power(&a, 3.0),
        &[4, 1],
        &[-421.875, -0.0, f64::INFINITY, 8.0],
        1e-14,
        0.0,
    );
}

#[test]
fn divmod_broadcasts() {
    let a = arr(&A).reshape(&[4, 1]);
    let b = arr(&B);
    let (q, r) = np::divmod(&a, &b);
    close(
        "divmod q",
        &q,
        &[4, 3],
        &[
            -3.0,
            f64::INFINITY,
            f64::NAN,
            -0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NEG_INFINITY,
            f64::NAN,
            0.0,
            f64::NEG_INFINITY,
            f64::NAN,
        ],
        0.0,
        0.0,
    );
    close(
        "divmod r",
        &r,
        &[4, 3],
        &[
            1.5,
            f64::NAN,
            f64::NAN,
            0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            2.0,
            f64::NAN,
            f64::NAN,
        ],
        0.0,
        0.0,
    );
    let (q, r) = np::divmod(&a, 3.0);
    close(
        "divmod q scalar",
        &q,
        &[4, 1],
        &[-3.0, -0.0, f64::NAN, 0.0],
        0.0,
        0.0,
    );
    close(
        "divmod r scalar",
        &r,
        &[4, 1],
        &[1.5, 0.0, f64::NAN, 2.0],
        0.0,
        0.0,
    );
}

#[test]
fn heaviside_broadcasts() {
    let x = np::array![-2.0, 0.0, -0.0, f64::NAN, 3.0].reshape(&[5, 1]);
    let h0 = np::array![0.5, 1.0];
    close(
        "heaviside",
        &np::heaviside(&x, &h0),
        &[5, 2],
        &[0.0, 0.0, 0.5, 1.0, 0.5, 1.0, f64::NAN, f64::NAN, 1.0, 1.0],
        0.0,
        0.0,
    );
    close(
        "heaviside scalar",
        &np::heaviside(&x, 0.25),
        &[5, 1],
        &[0.0, 0.25, 0.25, f64::NAN, 1.0],
        0.0,
        0.0,
    );
}

#[test]
fn ldexp_broadcasts() {
    let a = np::array![1.0, 0.5, -3.0, f64::INFINITY, 0.0].reshape(&[5, 1]);
    let e = np::array![2, -1, 1024, -1100, 0];
    close(
        "ldexp",
        &np::ldexp(&a, &e),
        &[5, 5],
        &[
            4.0,
            0.5,
            f64::INFINITY,
            0.0,
            1.0,
            2.0,
            0.25,
            8.98846567431158e+307,
            0.0,
            0.5,
            -12.0,
            -1.5,
            f64::NEG_INFINITY,
            -0.0,
            -3.0,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        0.0,
        0.0,
    );
    close(
        "ldexp scalar",
        &np::ldexp(&a, 3),
        &[5, 1],
        &[8.0, 4.0, -24.0, f64::INFINITY, 0.0],
        0.0,
        0.0,
    );
}

#[test]
fn gcd_lcm_broadcast() {
    let a = np::array![12i64, 18, -4, 0, 1 << 40].reshape(&[5, 1]);
    let b = np::array![8i64, 27, -6, 0];
    let g = np::gcd(&a, &b);
    assert_eq!(g.shape(), &[5, 4]);
    assert_eq!(
        g.to_vec(),
        [
            4,
            3,
            6,
            12,
            2,
            9,
            6,
            18,
            4,
            1,
            2,
            4,
            8,
            27,
            6,
            0,
            8,
            1,
            2,
            1099511627776
        ]
    );
    let l = np::lcm(&a, &b);
    assert_eq!(l.shape(), &[5, 4]);
    assert_eq!(
        l.to_vec(),
        [
            24,
            108,
            12,
            0,
            72,
            54,
            18,
            0,
            8,
            108,
            12,
            0,
            0,
            0,
            0,
            0,
            1099511627776,
            29686813949952,
            3298534883328,
            0
        ]
    );
    assert_eq!(np::gcd(12i64, 18).item(), 6);
    assert_eq!(np::lcm(i64::MAX, 2).item(), -2);
}

#[test]
fn ediff1d_and_angle() {
    assert_eq!(np::ediff1d(np::array![[1, 3], [6, 10]]).to_vec(), [2, 3, 4]);
    assert_eq!(
        np::ediff1d_with(np::array![1, 2, 4], &[9, 9], &[-1]).to_vec(),
        [-1, 1, 2, 9, 9]
    );
    assert!(np::ediff1d(np::array![5]).is_empty());
    assert_eq!(np::ediff1d(np::array![i32::MAX, i32::MIN]).to_vec(), [1]);
    close(
        "ediff1d f64",
        &np::ediff1d(np::array![1.5, f64::INFINITY, f64::NAN]),
        &[2],
        &[f64::INFINITY, f64::NAN],
        0.0,
        0.0,
    );
    close(
        "angle",
        &np::angle(arr(&X)),
        &[12],
        &[
            f64::NAN,
            0.0,
            3.141592653589793,
            3.141592653589793,
            0.0,
            0.0,
            3.141592653589793,
            0.0,
            0.0,
            3.141592653589793,
            0.0,
            3.141592653589793,
        ],
        0.0,
        0.0,
    );
    close(
        "angle_deg",
        &np::angle_deg(np::array![1.0, -1.0, 0.0]),
        &[3],
        &[0.0, 180.0, 0.0],
        1e-14,
        0.0,
    );
}

#[test]
fn hanning_window() {
    close("hanning(0)", &np::hanning(0), &[0], &[], 1e-14, 1e-15);
    close("hanning(1)", &np::hanning(1), &[1], &[1.0], 1e-14, 1e-15);
    close(
        "hanning(5)",
        &np::hanning(5),
        &[5],
        &[0.0, 0.5, 1.0, 0.5, 0.0],
        1e-14,
        1e-15,
    );
    close(
        "hanning(12)",
        &np::hanning(12),
        &[12],
        &[
            0.0,
            0.07937323358440945,
            0.29229249349905684,
            0.5711574191366425,
            0.8274303669726426,
            0.9797464868072487,
            0.9797464868072487,
            0.8274303669726426,
            0.5711574191366425,
            0.29229249349905684,
            0.07937323358440945,
            0.0,
        ],
        1e-14,
        1e-15,
    );
}

#[test]
fn hamming_window() {
    close("hamming(0)", &np::hamming(0), &[0], &[], 1e-14, 1e-15);
    close("hamming(1)", &np::hamming(1), &[1], &[1.0], 1e-14, 1e-15);
    close(
        "hamming(5)",
        &np::hamming(5),
        &[5],
        &[0.08000000000000002, 0.54, 1.0, 0.54, 0.08000000000000002],
        1e-14,
        1e-15,
    );
    close(
        "hamming(12)",
        &np::hamming(12),
        &[12],
        &[
            0.08000000000000002,
            0.15302337489765672,
            0.3489090940191323,
            0.6054648256057111,
            0.8412359376148312,
            0.9813667678626689,
            0.9813667678626689,
            0.8412359376148312,
            0.6054648256057111,
            0.3489090940191323,
            0.15302337489765672,
            0.08000000000000002,
        ],
        1e-14,
        1e-15,
    );
}

#[test]
fn blackman_window() {
    close("blackman(0)", &np::blackman(0), &[0], &[], 1e-14, 1e-15);
    close("blackman(1)", &np::blackman(1), &[1], &[1.0], 1e-14, 1e-15);
    close(
        "blackman(5)",
        &np::blackman(5),
        &[5],
        &[
            -1.3877787807814457e-17,
            0.34,
            0.9999999999999999,
            0.34,
            -1.3877787807814457e-17,
        ],
        1e-14,
        1e-15,
    );
    close(
        "blackman(12)",
        &np::blackman(12),
        &[12],
        &[
            -1.3877787807814457e-17,
            0.032606434624560324,
            0.159903634783434,
            0.4143979812474828,
            0.7360451799107798,
            0.9670467694337431,
            0.9670467694337431,
            0.7360451799107798,
            0.4143979812474828,
            0.159903634783434,
            0.032606434624560324,
            -1.3877787807814457e-17,
        ],
        1e-14,
        1e-15,
    );
}

#[test]
fn bartlett_window() {
    close("bartlett(0)", &np::bartlett(0), &[0], &[], 1e-14, 1e-15);
    close("bartlett(1)", &np::bartlett(1), &[1], &[1.0], 1e-14, 1e-15);
    close(
        "bartlett(5)",
        &np::bartlett(5),
        &[5],
        &[0.0, 0.5, 1.0, 0.5, 0.0],
        1e-14,
        1e-15,
    );
    close(
        "bartlett(12)",
        &np::bartlett(12),
        &[12],
        &[
            0.0,
            0.18181818181818177,
            0.36363636363636365,
            0.5454545454545454,
            0.7272727272727273,
            0.9090909090909091,
            0.9090909090909091,
            0.7272727272727273,
            0.5454545454545454,
            0.36363636363636365,
            0.18181818181818177,
            0.0,
        ],
        1e-14,
        1e-15,
    );
}

#[test]
fn kaiser_window() {
    close("kaiser(0)", &np::kaiser(0, 14.0), &[0], &[], 1e-14, 1e-15);
    close(
        "kaiser(1)",
        &np::kaiser(1, 14.0),
        &[1],
        &[1.0],
        1e-14,
        1e-15,
    );
    close(
        "kaiser(5)",
        &np::kaiser(5, 14.0),
        &[5],
        &[
            7.726866835270368e-06,
            0.16493218754795197,
            1.0,
            0.16493218754795197,
            7.726866835270368e-06,
        ],
        1e-14,
        1e-15,
    );
    close(
        "kaiser(12)",
        &np::kaiser(12, 14.0),
        &[12],
        &[
            7.726866835270368e-06,
            0.003460091937889429,
            0.04652001885143685,
            0.22973712022098938,
            0.5998853159552917,
            0.9456748983666177,
            0.9456748983666177,
            0.5998853159552917,
            0.22973712022098938,
            0.04652001885143685,
            0.003460091937889429,
            7.726866835270368e-06,
        ],
        1e-14,
        1e-15,
    );
}

#[test]
fn kaiser_other_betas() {
    close(
        "kaiser(7, 0)",
        &np::kaiser(7, 0.0),
        &[7],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-14,
        1e-15,
    );
    close(
        "kaiser(12, 5.5)",
        &np::kaiser(12, 5.5),
        &[12],
        &[
            0.023422141030646812,
            0.1304993220795649,
            0.3267718259566892,
            0.5830270949172955,
            0.8285951489100458,
            0.9796312670569656,
            0.9796312670569656,
            0.8285951489100458,
            0.5830270949172955,
            0.3267718259566892,
            0.1304993220795649,
            0.023422141030646812,
        ],
        1e-14,
        1e-15,
    );
}

#[test]
fn complex_arithmetic_and_display() {
    let (z1, z2) = (Complex::new(1.0, 2.0), Complex::new(3.0, -1.0));
    assert_eq!(z1 + z2, Complex::new(4.0, 1.0));
    assert_eq!((z1 + z2).to_string(), "(4+1j)");
    assert_eq!(z1 - z2, Complex::new(-2.0, 3.0));
    assert_eq!((z1 - z2).to_string(), "(-2+3j)");
    assert_eq!(z1 * z2, Complex::new(5.0, 5.0));
    assert_eq!((z1 * z2).to_string(), "(5+5j)");
    let q: Complex<f64> = z1 / z2;
    assert!((q.re - 0.1).abs() < 1e-15 && (q.im - 0.7000000000000001).abs() < 1e-15);
    assert_eq!(q.to_string(), "(0.1+0.7000000000000001j)");
    assert_eq!((-z1).to_string(), "(-1-2j)");
    assert_eq!((z1 * 2.0).to_string(), "(2+4j)");
    assert_eq!((2.0f64 * z1).to_string(), "(2+4j)");
    assert_eq!((z1 + 0.5).to_string(), "(1.5+2j)");
    assert_eq!((0.5f64 - z1).to_string(), "(-0.5-2j)");
    assert_eq!((1.0f64 / Complex::new(0.0, 1.0)).to_string(), "-1j");
    let mut z = z1;
    z += z2;
    z *= 2.0;
    z -= Complex::new(1.0, 1.0);
    z /= Complex::new(0.0, 1.0);
    assert_eq!(z.to_string(), "(1-7j)");
    assert_eq!(z1.conj().to_string(), "(1-2j)");
    assert_eq!(Complex::new(3.0, 4.0).norm(), 5.0);
    assert_eq!(Complex::new(0.0, 1.0).arg(), 1.5707963267948966);
    assert_eq!(Complex::<f64>::i().to_string(), "1j");
    assert_eq!(Complex::from_real(2.5).to_string(), "(2.5+0j)");
    assert_eq!(
        Complex::new(f64::NAN, f64::INFINITY).to_string(),
        "(nan+infj)"
    );
    assert_eq!(Complex::new(1e16, -1.0).to_string(), "(1e+16-1j)");
    assert_eq!(Complex::new(-0.5, -0.0).to_string(), "(-0.5-0j)");
    assert_eq!(Complex::new(0.0, 0.0).to_string(), "0j");
    assert_eq!(Complex::new(1.5f32, 2.0).to_string(), "(1.5+2j)");
    assert_eq!(Complex::new(1e6f32, 2.0).to_string(), "(1e+06+2j)");
}

#[test]
fn complex_functions() {
    fn near(z: Complex<f64>, re: f64, im: f64) {
        let tol = 1e-14 * (1.0 + re.abs().max(im.abs()));
        assert!(
            (z.re - re).abs() <= tol && (z.im - im).abs() <= tol,
            "{z:?} vs ({re:e}, {im:e})"
        );
    }
    let z = Complex::new(1.0, 2.0);
    near(z.exp(), -1.1312043837568135, 2.4717266720048188);
    near(z.ln(), 0.8047189562170503, 1.1071487177940904);
    near(z.sqrt(), 1.272019649514069, 0.7861513777574233);
    near(z.sin(), 3.165778513216168, 1.9596010414216063);
    near(z.cos(), 2.0327230070196656, -3.0518977991518);
    near(z.tan(), 0.03381282607989669, 1.0147936161466338);
    near(z.sinh(), -0.4890562590412937, 1.4031192506220405);
    near(z.cosh(), -0.64214812471552, 1.0686074213827783);
    near(z.tanh(), 1.16673625724092, -0.24345820118572525);
    near(
        z.powc(Complex::new(3.0, -1.0)),
        -27.436381991606034,
        19.7893103650107,
    );
    near(z.powf(2.5), -6.960664459571899, 2.7296244647840076);
    near(z.powc(Complex::new(-3.0, 0.0)), -0.08800000000000001, 0.016);
    // branch cuts follow the sign of a zero imaginary part
    assert_eq!(Complex::new(-4.0, 0.0).sqrt().to_string(), "2j");
    assert_eq!(Complex::new(-4.0, -0.0).sqrt().to_string(), "-2j");
    assert_eq!(
        Complex::new(-1.0, 0.0).ln().to_string(),
        "3.141592653589793j"
    );
    assert_eq!(
        Complex::new(-1.0, -0.0).ln().to_string(),
        "-3.141592653589793j"
    );
    assert!(Complex::new(f64::NAN, 1.0).is_nan() && !z.is_nan());
    assert!(Complex::new(f64::INFINITY, 1.0).is_infinite() && !z.is_infinite());
    assert!(z.is_finite() && !Complex::new(1.0, f64::NAN).is_finite());
    assert_eq!(z.norm_sqr(), 5.0);
    let p = Complex::from_polar(2.0, std::f64::consts::FRAC_PI_2);
    assert!(p.re.abs() < 1e-15 && (p.im - 2.0).abs() < 1e-15);
}

#[test]
fn complex_sort_order_matches_numpy() {
    let nan = f64::NAN;
    // np.sort puts [R+Rj, R+nanj, nan+Rj, nan+nanj] in that order
    let sorted = [
        Complex::new(1.0, 1.0),
        Complex::new(1.0, nan),
        Complex::new(nan, 0.0),
        Complex::new(nan, nan),
    ];
    for i in 0..sorted.len() {
        for j in 0..sorted.len() {
            assert_eq!(sorted[i].cmp_numpy(&sorted[j]), i.cmp(&j), "{i} vs {j}");
        }
    }
    assert!(
        Complex::new(1.0, 2.0)
            .cmp_numpy(&Complex::new(1.0, 3.0))
            .is_lt()
    );
    assert!(
        Complex::new(2.0, 0.0)
            .cmp_numpy(&Complex::new(1.0, 9.0))
            .is_gt()
    );
}

/// Generic user code bounded by the public `Float` trait can use every misc method and function.
fn generic<F: np::Float>(a: &Array<F>) -> (Array<F>, Array<F>, Array<F>) {
    let two = F::ONE + F::ONE;
    let s = a
        .sinc()
        .i0()
        .deg2rad()
        .rad2deg()
        .rint()
        .fix()
        .fabs()
        .positive()
        .negative();
    let (frac, int) = a.modf();
    let (m, e) = a.frexp();
    let _: Array<i32> = e;
    let b = np::logaddexp(a, &frac) + np::fmod(&int, two) + np::copysign(&m, F::ONE);
    let (q, r) = np::divmod(a, two);
    let c = np::heaviside(&q, F::ONE) + np::remainder(&r, two) + np::floor_divide(a, two);
    let d = np::nextafter(a, F::INFINITY)
        + np::logaddexp2(a, a)
        + a.spacing()
        + a.nan_to_num()
        + a.unwrap();
    let _ = (
        a.signbit(),
        np::angle(a),
        np::angle_deg(a),
        np::ediff1d(a),
        np::float_power(a, a),
    );
    (s + b, c, d + np::ldexp(a, 2))
}

fn generic_complex<F: np::Float>(re: F, im: F) -> Complex<F> {
    let z = Complex::new(re, im);
    let u = Complex::new(im, re);
    (z * z + z / u - u).exp().sqrt().ln() * re + im
}

#[test]
fn works_for_generic_float_code() {
    let a = np::array![0.5f64, 1.5, -2.25];
    let (x, y, z) = generic(&a);
    assert_eq!(
        (x.shape(), y.shape(), z.shape()),
        (&[3][..], &[3][..], &[3][..])
    );
    let b = np::array![0.5f32, 1.5, -2.25];
    let (x, _, _) = generic(&b);
    assert_eq!(x.shape(), &[3]);
    assert!(generic_complex(0.5f64, 0.25).is_finite());
    assert!(generic_complex(0.5f32, 0.25).is_finite());
}

#[test]
fn methods_do_not_mutate_input_and_keep_shape() {
    let a = np::array![[0.5, -1.5], [2.5, 3.5]];
    let before = a.to_vec();
    let r = a.rint();
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(r.to_vec(), [0.0, -2.0, 2.0, 4.0]);
    assert_eq!(a.to_vec(), before);
    assert_eq!(a.t().fix().to_vec(), [0.0, 2.0, -1.0, 3.0]);
    assert_eq!(a.modf().0.shape(), &[2, 2]);
}
