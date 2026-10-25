//! NumPy parity tests for the window functions (GENERATED ONCE with NumPy 2.3.2).
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use super::tests_util::Report;
use super::window_kernels::*;

fn close(got: f64, want: f64, what: &str, i: usize) {
    let tol = 1e-14 + 1e-13 * want.abs();
    assert!(
        (got - want).abs() <= tol,
        "{what}[{i}]: got {got:?}, NumPy says {want:?}"
    );
}

#[test]
fn hanning_0() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = hanning(0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(0)", i);
    }
}

#[test]
fn hanning_1() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = hanning(1);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(1)", i);
    }
}

#[test]
fn hanning_2() {
    let _report = Report;
    let want: &[f64] = &[0.0, 0.0];
    let got = hanning(2);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(2)", i);
    }
}

#[test]
fn hanning_3() {
    let _report = Report;
    let want: &[f64] = &[0.0, 1.0, 0.0];
    let got = hanning(3);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(3)", i);
    }
}

#[test]
fn hanning_4() {
    let _report = Report;
    let want: &[f64] = &[0.0, 0.75, 0.75, 0.0];
    let got = hanning(4);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(4)", i);
    }
}

#[test]
fn hanning_5() {
    let _report = Report;
    let want: &[f64] = &[0.0, 0.5, 1.0, 0.5, 0.0];
    let got = hanning(5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(5)", i);
    }
}

#[test]
fn hanning_10() {
    let _report = Report;
    let want: &[f64] = &[
        0.0,
        0.11697777844051105,
        0.41317591116653485,
        0.75,
        0.9698463103929542,
        0.9698463103929542,
        0.75,
        0.41317591116653485,
        0.11697777844051105,
        0.0,
    ];
    let got = hanning(10);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(10)", i);
    }
}

#[test]
fn hanning_11() {
    let _report = Report;
    let want: &[f64] = &[
        0.0,
        0.09549150281252633,
        0.34549150281252633,
        0.6545084971874737,
        0.9045084971874737,
        1.0,
        0.9045084971874737,
        0.6545084971874737,
        0.34549150281252633,
        0.09549150281252633,
        0.0,
    ];
    let got = hanning(11);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(11)", i);
    }
}

#[test]
fn hanning_51() {
    let _report = Report;
    let want: &[f64] = &[
        0.0,
        0.0039426493427611176,
        0.015708419435684517,
        0.035111757055874326,
        0.06184665997806821,
        0.09549150281252633,
        0.13551568628929433,
        0.18128800512565513,
        0.23208660251050156,
        0.28711035421746367,
        0.34549150281252644,
        0.4063093427071377,
        0.4686047402353433,
        0.5313952597646567,
        0.5936906572928624,
        0.6545084971874737,
        0.7128896457825363,
        0.7679133974894983,
        0.8187119948743449,
        0.8644843137107058,
        0.9045084971874737,
        0.9381533400219317,
        0.9648882429441257,
        0.9842915805643155,
        0.996057350657239,
        1.0,
        0.996057350657239,
        0.9842915805643155,
        0.9648882429441257,
        0.9381533400219317,
        0.9045084971874737,
        0.8644843137107058,
        0.8187119948743449,
        0.7679133974894983,
        0.7128896457825363,
        0.6545084971874737,
        0.5936906572928624,
        0.5313952597646567,
        0.4686047402353433,
        0.4063093427071377,
        0.34549150281252644,
        0.28711035421746367,
        0.23208660251050156,
        0.18128800512565513,
        0.13551568628929433,
        0.09549150281252633,
        0.06184665997806821,
        0.035111757055874326,
        0.015708419435684517,
        0.0039426493427611176,
        0.0,
    ];
    let got = hanning(51);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hanning(51)", i);
    }
}

#[test]
fn hamming_0() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = hamming(0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(0)", i);
    }
}

#[test]
fn hamming_1() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = hamming(1);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(1)", i);
    }
}

#[test]
fn hamming_2() {
    let _report = Report;
    let want: &[f64] = &[0.08000000000000002, 0.08000000000000002];
    let got = hamming(2);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(2)", i);
    }
}

#[test]
fn hamming_3() {
    let _report = Report;
    let want: &[f64] = &[0.08000000000000002, 1.0, 0.08000000000000002];
    let got = hamming(3);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(3)", i);
    }
}

#[test]
fn hamming_4() {
    let _report = Report;
    let want: &[f64] = &[
        0.08000000000000002,
        0.7700000000000001,
        0.7700000000000001,
        0.08000000000000002,
    ];
    let got = hamming(4);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(4)", i);
    }
}

#[test]
fn hamming_5() {
    let _report = Report;
    let want: &[f64] = &[0.08000000000000002, 0.54, 1.0, 0.54, 0.08000000000000002];
    let got = hamming(5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(5)", i);
    }
}

#[test]
fn hamming_10() {
    let _report = Report;
    let want: &[f64] = &[
        0.08000000000000002,
        0.1876195561652702,
        0.46012183827321207,
        0.7700000000000001,
        0.9722586055615179,
        0.9722586055615179,
        0.7700000000000001,
        0.46012183827321207,
        0.1876195561652702,
        0.08000000000000002,
    ];
    let got = hamming(10);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(10)", i);
    }
}

#[test]
fn hamming_11() {
    let _report = Report;
    let want: &[f64] = &[
        0.08000000000000002,
        0.16785218258752427,
        0.39785218258752425,
        0.6821478174124759,
        0.9121478174124759,
        1.0,
        0.9121478174124759,
        0.6821478174124759,
        0.39785218258752425,
        0.16785218258752427,
        0.08000000000000002,
    ];
    let got = hamming(11);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(11)", i);
    }
}

#[test]
fn hamming_51() {
    let _report = Report;
    let want: &[f64] = &[
        0.08000000000000002,
        0.08362723739534023,
        0.09445174588082977,
        0.11230281649140439,
        0.13689892717982277,
        0.16785218258752427,
        0.20467443138615082,
        0.24678496471560274,
        0.29351967430966147,
        0.3441415258800666,
        0.39785218258752436,
        0.4538045952905667,
        0.5111163610165159,
        0.5688836389834843,
        0.6261954047094335,
        0.6821478174124759,
        0.7358584741199334,
        0.7864803256903384,
        0.8332150352843973,
        0.8753255686138494,
        0.9121478174124759,
        0.9431010728201773,
        0.9676971835085957,
        0.9855482541191704,
        0.9963727626046599,
        1.0,
        0.9963727626046599,
        0.9855482541191704,
        0.9676971835085957,
        0.9431010728201773,
        0.9121478174124759,
        0.8753255686138494,
        0.8332150352843973,
        0.7864803256903384,
        0.7358584741199334,
        0.6821478174124759,
        0.6261954047094335,
        0.5688836389834843,
        0.5111163610165159,
        0.4538045952905667,
        0.39785218258752436,
        0.3441415258800666,
        0.29351967430966147,
        0.24678496471560274,
        0.20467443138615082,
        0.16785218258752427,
        0.13689892717982277,
        0.11230281649140439,
        0.09445174588082977,
        0.08362723739534023,
        0.08000000000000002,
    ];
    let got = hamming(51);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "hamming(51)", i);
    }
}

#[test]
fn blackman_0() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = blackman(0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(0)", i);
    }
}

#[test]
fn blackman_1() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = blackman(1);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(1)", i);
    }
}

#[test]
fn blackman_2() {
    let _report = Report;
    let want: &[f64] = &[-1.3877787807814457e-17, -1.3877787807814457e-17];
    let got = blackman(2);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(2)", i);
    }
}

#[test]
fn blackman_3() {
    let _report = Report;
    let want: &[f64] = &[
        -1.3877787807814457e-17,
        0.9999999999999999,
        -1.3877787807814457e-17,
    ];
    let got = blackman(3);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(3)", i);
    }
}

#[test]
fn blackman_4() {
    let _report = Report;
    let want: &[f64] = &[
        -1.3877787807814457e-17,
        0.6300000000000001,
        0.6300000000000001,
        -1.3877787807814457e-17,
    ];
    let got = blackman(4);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(4)", i);
    }
}

#[test]
fn blackman_5() {
    let _report = Report;
    let want: &[f64] = &[
        -1.3877787807814457e-17,
        0.34,
        0.9999999999999999,
        0.34,
        -1.3877787807814457e-17,
    ];
    let got = blackman(5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(5)", i);
    }
}

#[test]
fn blackman_10() {
    let _report = Report;
    let want: &[f64] = &[
        -1.3877787807814457e-17,
        0.05086963265386543,
        0.25800050150366216,
        0.6300000000000001,
        0.9511298658424724,
        0.9511298658424724,
        0.6300000000000001,
        0.25800050150366216,
        0.05086963265386543,
        -1.3877787807814457e-17,
    ];
    let got = blackman(10);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(10)", i);
    }
}

#[test]
fn blackman_11() {
    let _report = Report;
    let want: &[f64] = &[
        -1.3877787807814457e-17,
        0.0402128623625221,
        0.2007701432625305,
        0.5097871376374778,
        0.8492298567374694,
        0.9999999999999999,
        0.8492298567374694,
        0.5097871376374778,
        0.2007701432625305,
        0.0402128623625221,
        -1.3877787807814457e-17,
    ];
    let got = blackman(11);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(11)", i);
    }
}

#[test]
fn blackman_51() {
    let _report = Report;
    let want: &[f64] = &[
        -1.3877787807814457e-17,
        0.0014293022330515853,
        0.005812953839193558,
        0.01342924724958721,
        0.024712803576387936,
        0.0402128623625221,
        0.06053892785163935,
        0.08629749995879714,
        0.11802425918529577,
        0.15611643503756847,
        0.20077014326253062,
        0.2519272238360776,
        0.30923556413018505,
        0.3720260836594985,
        0.43930853842180223,
        0.5097871376374778,
        0.5818957266026412,
        0.6538510541642923,
        0.7237214897074868,
        0.7895075552730509,
        0.8492298567374694,
        0.9010194836202515,
        0.9432057331378386,
        0.9743961149678245,
        0.9935440035475294,
        0.9999999999999999,
        0.9935440035475294,
        0.9743961149678245,
        0.9432057331378386,
        0.9010194836202515,
        0.8492298567374694,
        0.7895075552730509,
        0.7237214897074868,
        0.6538510541642923,
        0.5818957266026412,
        0.5097871376374778,
        0.43930853842180223,
        0.3720260836594985,
        0.30923556413018505,
        0.2519272238360776,
        0.20077014326253062,
        0.15611643503756847,
        0.11802425918529577,
        0.08629749995879714,
        0.06053892785163935,
        0.0402128623625221,
        0.024712803576387936,
        0.01342924724958721,
        0.005812953839193558,
        0.0014293022330515853,
        -1.3877787807814457e-17,
    ];
    let got = blackman(51);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "blackman(51)", i);
    }
}

#[test]
fn bartlett_0() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = bartlett(0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(0)", i);
    }
}

#[test]
fn bartlett_1() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = bartlett(1);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(1)", i);
    }
}

#[test]
fn bartlett_2() {
    let _report = Report;
    let want: &[f64] = &[0.0, 0.0];
    let got = bartlett(2);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(2)", i);
    }
}

#[test]
fn bartlett_3() {
    let _report = Report;
    let want: &[f64] = &[0.0, 1.0, 0.0];
    let got = bartlett(3);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(3)", i);
    }
}

#[test]
fn bartlett_4() {
    let _report = Report;
    let want: &[f64] = &[0.0, 0.6666666666666667, 0.6666666666666667, 0.0];
    let got = bartlett(4);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(4)", i);
    }
}

#[test]
fn bartlett_5() {
    let _report = Report;
    let want: &[f64] = &[0.0, 0.5, 1.0, 0.5, 0.0];
    let got = bartlett(5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(5)", i);
    }
}

#[test]
fn bartlett_10() {
    let _report = Report;
    let want: &[f64] = &[
        0.0,
        0.2222222222222222,
        0.4444444444444444,
        0.6666666666666667,
        0.8888888888888888,
        0.8888888888888888,
        0.6666666666666667,
        0.4444444444444444,
        0.2222222222222222,
        0.0,
    ];
    let got = bartlett(10);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(10)", i);
    }
}

#[test]
fn bartlett_11() {
    let _report = Report;
    let want: &[f64] = &[
        0.0,
        0.19999999999999996,
        0.4,
        0.6,
        0.8,
        1.0,
        0.8,
        0.6,
        0.4,
        0.19999999999999996,
        0.0,
    ];
    let got = bartlett(11);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(11)", i);
    }
}

#[test]
fn bartlett_51() {
    let _report = Report;
    let want: &[f64] = &[
        0.0,
        0.040000000000000036,
        0.07999999999999996,
        0.12,
        0.16000000000000003,
        0.19999999999999996,
        0.24,
        0.28,
        0.31999999999999995,
        0.36,
        0.4,
        0.43999999999999995,
        0.48,
        0.52,
        0.56,
        0.6,
        0.64,
        0.6799999999999999,
        0.72,
        0.76,
        0.8,
        0.84,
        0.88,
        0.92,
        0.96,
        1.0,
        0.96,
        0.92,
        0.88,
        0.84,
        0.8,
        0.76,
        0.72,
        0.6799999999999999,
        0.64,
        0.6,
        0.56,
        0.52,
        0.48,
        0.43999999999999995,
        0.4,
        0.36,
        0.31999999999999995,
        0.28,
        0.24,
        0.19999999999999996,
        0.16000000000000003,
        0.12,
        0.07999999999999996,
        0.040000000000000036,
        0.0,
    ];
    let got = bartlett(51);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "bartlett(51)", i);
    }
}

#[test]
fn kaiser_0_beta_0_0() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = kaiser(0, 0.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(0, 0.0)", i);
    }
}

#[test]
fn kaiser_1_beta_0_0() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = kaiser(1, 0.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(1, 0.0)", i);
    }
}

#[test]
fn kaiser_2_beta_0_0() {
    let _report = Report;
    let want: &[f64] = &[1.0, 1.0];
    let got = kaiser(2, 0.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(2, 0.0)", i);
    }
}

#[test]
fn kaiser_3_beta_0_0() {
    let _report = Report;
    let want: &[f64] = &[1.0, 1.0, 1.0];
    let got = kaiser(3, 0.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(3, 0.0)", i);
    }
}

#[test]
fn kaiser_12_beta_0_0() {
    let _report = Report;
    let want: &[f64] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let got = kaiser(12, 0.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(12, 0.0)", i);
    }
}

#[test]
fn kaiser_51_beta_0_0() {
    let _report = Report;
    let want: &[f64] = &[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let got = kaiser(51, 0.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(51, 0.0)", i);
    }
}

#[test]
fn kaiser_0_beta_5_0() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = kaiser(0, 5.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(0, 5.0)", i);
    }
}

#[test]
fn kaiser_1_beta_5_0() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = kaiser(1, 5.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(1, 5.0)", i);
    }
}

#[test]
fn kaiser_2_beta_5_0() {
    let _report = Report;
    let want: &[f64] = &[0.036710892271286676, 0.036710892271286676];
    let got = kaiser(2, 5.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(2, 5.0)", i);
    }
}

#[test]
fn kaiser_3_beta_5_0() {
    let _report = Report;
    let want: &[f64] = &[0.036710892271286676, 1.0, 0.036710892271286676];
    let got = kaiser(3, 5.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(3, 5.0)", i);
    }
}

#[test]
fn kaiser_12_beta_5_0() {
    let _report = Report;
    let want: &[f64] = &[
        0.036710892271286676,
        0.1619952511523947,
        0.3668380606253588,
        0.6160930407139884,
        0.8445883822449818,
        0.9816782822458596,
        0.9816782822458596,
        0.8445883822449818,
        0.6160930407139884,
        0.3668380606253588,
        0.1619952511523947,
        0.036710892271286676,
    ];
    let got = kaiser(12, 5.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(12, 5.0)", i);
    }
}

#[test]
fn kaiser_51_beta_5_0() {
    let _report = Report;
    let want: &[f64] = &[
        0.036710892271286676,
        0.05702652016097732,
        0.08137001295154699,
        0.10984475054507063,
        0.1424707129794146,
        0.1791782508181855,
        0.2198039487419763,
        0.2640887828478816,
        0.31167871028318017,
        0.36212776215987214,
        0.4149036392433668,
        0.4693957368604605,
        0.5249254530938666,
        0.5807585649001086,
        0.6361193925189762,
        0.6902064155143299,
        0.7422089558704821,
        0.7913245063693937,
        0.8367762572713046,
        0.8778303620265007,
        0.9138124838692001,
        0.9441231797756305,
        0.9682517060710092,
        0.9857878701899838,
        0.9964316045822836,
        1.0,
        0.9964316045822836,
        0.9857878701899838,
        0.9682517060710092,
        0.9441231797756305,
        0.9138124838692001,
        0.8778303620265007,
        0.8367762572713046,
        0.7913245063693937,
        0.7422089558704821,
        0.6902064155143299,
        0.6361193925189762,
        0.5807585649001086,
        0.5249254530938666,
        0.4693957368604605,
        0.4149036392433668,
        0.36212776215987214,
        0.31167871028318017,
        0.2640887828478816,
        0.2198039487419763,
        0.1791782508181855,
        0.1424707129794146,
        0.10984475054507063,
        0.08137001295154699,
        0.05702652016097732,
        0.036710892271286676,
    ];
    let got = kaiser(51, 5.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(51, 5.0)", i);
    }
}

#[test]
fn kaiser_0_beta_8_6() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = kaiser(0, 8.6);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(0, 8.6)", i);
    }
}

#[test]
fn kaiser_1_beta_8_6() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = kaiser(1, 8.6);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(1, 8.6)", i);
    }
}

#[test]
fn kaiser_2_beta_8_6() {
    let _report = Report;
    let want: &[f64] = &[0.0013325139979024198, 0.0013325139979024198];
    let got = kaiser(2, 8.6);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(2, 8.6)", i);
    }
}

#[test]
fn kaiser_3_beta_8_6() {
    let _report = Report;
    let want: &[f64] = &[0.0013325139979024198, 1.0, 0.0013325139979024198];
    let got = kaiser(3, 8.6);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(3, 8.6)", i);
    }
}

#[test]
fn kaiser_12_beta_8_6() {
    let _report = Report;
    let want: &[f64] = &[
        0.0013325139979024198,
        0.034547607744245444,
        0.16021938631113028,
        0.41484344225986564,
        0.7363621417828066,
        0.9670868343930674,
        0.9670868343930674,
        0.7363621417828066,
        0.41484344225986564,
        0.16021938631113028,
        0.034547607744245444,
        0.0013325139979024198,
    ];
    let got = kaiser(12, 8.6);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(12, 8.6)", i);
    }
}

#[test]
fn kaiser_51_beta_8_6() {
    let _report = Report;
    let want: &[f64] = &[
        0.0013325139979024198,
        0.004087764836375501,
        0.008820896591883348,
        0.01620597693676516,
        0.026975683074208397,
        0.04188720541565504,
        0.06168005326298255,
        0.08702768987563986,
        0.11848563812020903,
        0.15643927053402434,
        0.20105486954533092,
        0.25223766819411697,
        0.3096004302249363,
        0.37244569345167367,
        0.43976409753498646,
        0.5102502854427691,
        0.5823367663319555,
        0.6542449325220533,
        0.724051221730045,
        0.7897652990361129,
        0.8494161890910448,
        0.9011415951483038,
        0.9432752575949441,
        0.9744271681127263,
        0.9935517775891662,
        1.0,
        0.9935517775891662,
        0.9744271681127263,
        0.9432752575949441,
        0.9011415951483038,
        0.8494161890910448,
        0.7897652990361129,
        0.724051221730045,
        0.6542449325220533,
        0.5823367663319555,
        0.5102502854427691,
        0.43976409753498646,
        0.37244569345167367,
        0.3096004302249363,
        0.25223766819411697,
        0.20105486954533092,
        0.15643927053402434,
        0.11848563812020903,
        0.08702768987563986,
        0.06168005326298255,
        0.04188720541565504,
        0.026975683074208397,
        0.01620597693676516,
        0.008820896591883348,
        0.004087764836375501,
        0.0013325139979024198,
    ];
    let got = kaiser(51, 8.6);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(51, 8.6)", i);
    }
}

#[test]
fn kaiser_0_beta_14_0() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = kaiser(0, 14.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(0, 14.0)", i);
    }
}

#[test]
fn kaiser_1_beta_14_0() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = kaiser(1, 14.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(1, 14.0)", i);
    }
}

#[test]
fn kaiser_2_beta_14_0() {
    let _report = Report;
    let want: &[f64] = &[7.726866835270368e-06, 7.726866835270368e-06];
    let got = kaiser(2, 14.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(2, 14.0)", i);
    }
}

#[test]
fn kaiser_3_beta_14_0() {
    let _report = Report;
    let want: &[f64] = &[7.726866835270368e-06, 1.0, 7.726866835270368e-06];
    let got = kaiser(3, 14.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(3, 14.0)", i);
    }
}

#[test]
fn kaiser_12_beta_14_0() {
    let _report = Report;
    let want: &[f64] = &[
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
    ];
    let got = kaiser(12, 14.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(12, 14.0)", i);
    }
}

#[test]
fn kaiser_51_beta_14_0() {
    let _report = Report;
    let want: &[f64] = &[
        7.726866835270368e-06,
        8.150948456727312e-05,
        0.00032600076740669175,
        0.0009425887506662767,
        0.002266248466391737,
        0.0048056791412168075,
        0.00927621458545023,
        0.01661643007392611,
        0.027978965679194313,
        0.044687350007222106,
        0.06815374319396057,
        0.09975740118942189,
        0.14068980496716194,
        0.19177896985813342,
        0.25331138714876045,
        0.32487421778447284,
        0.40524175565528453,
        0.4923281434670698,
        0.5832226995029494,
        0.6743154327037104,
        0.7615093990873935,
        0.8405049538342629,
        0.9071303903483532,
        0.9576856052548601,
        0.989261639090822,
        1.0,
        0.989261639090822,
        0.9576856052548601,
        0.9071303903483532,
        0.8405049538342629,
        0.7615093990873935,
        0.6743154327037104,
        0.5832226995029494,
        0.4923281434670698,
        0.40524175565528453,
        0.32487421778447284,
        0.25331138714876045,
        0.19177896985813342,
        0.14068980496716194,
        0.09975740118942189,
        0.06815374319396057,
        0.044687350007222106,
        0.027978965679194313,
        0.01661643007392611,
        0.00927621458545023,
        0.0048056791412168075,
        0.002266248466391737,
        0.0009425887506662767,
        0.00032600076740669175,
        8.150948456727312e-05,
        7.726866835270368e-06,
    ];
    let got = kaiser(51, 14.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(51, 14.0)", i);
    }
}

#[test]
fn kaiser_0_beta_neg3_0() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = kaiser(0, -3.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(0, -3.0)", i);
    }
}

#[test]
fn kaiser_1_beta_neg3_0() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = kaiser(1, -3.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(1, -3.0)", i);
    }
}

#[test]
fn kaiser_2_beta_neg3_0() {
    let _report = Report;
    let want: &[f64] = &[0.2048847564012536, 0.2048847564012536];
    let got = kaiser(2, -3.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(2, -3.0)", i);
    }
}

#[test]
fn kaiser_3_beta_neg3_0() {
    let _report = Report;
    let want: &[f64] = &[0.2048847564012536, 1.0, 0.2048847564012536];
    let got = kaiser(3, -3.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(3, -3.0)", i);
    }
}

#[test]
fn kaiser_12_beta_neg3_0() {
    let _report = Report;
    let want: &[f64] = &[
        0.2048847564012536,
        0.38807016765742613,
        0.585872002493217,
        0.7701128342191157,
        0.9124477348391573,
        0.9899941443214347,
        0.9899941443214347,
        0.9124477348391573,
        0.7701128342191157,
        0.585872002493217,
        0.38807016765742613,
        0.2048847564012536,
    ];
    let got = kaiser(12, -3.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(12, -3.0)", i);
    }
}

#[test]
fn kaiser_51_beta_neg3_0() {
    let _report = Report;
    let want: &[f64] = &[
        0.2048847564012536,
        0.24265186139428022,
        0.2820508239741365,
        0.3228502371917532,
        0.3647993860357045,
        0.40763038412652414,
        0.45106052419343956,
        0.4947948181698064,
        0.5385287003465871,
        0.5819508649290033,
        0.6247462075770618,
        0.6665988390991193,
        0.7071951384277205,
        0.7462268113529213,
        0.7833939212296187,
        0.8184078580166964,
        0.8509942125467148,
        0.8808955238601638,
        0.9078738687577339,
        0.931713264412767,
        0.9522218569243645,
        0.9692338710557488,
        0.982611299064323,
        0.9922453094578068,
        0.9980573596697762,
        1.0,
        0.9980573596697762,
        0.9922453094578068,
        0.982611299064323,
        0.9692338710557488,
        0.9522218569243645,
        0.931713264412767,
        0.9078738687577339,
        0.8808955238601638,
        0.8509942125467148,
        0.8184078580166964,
        0.7833939212296187,
        0.7462268113529213,
        0.7071951384277205,
        0.6665988390991193,
        0.6247462075770618,
        0.5819508649290033,
        0.5385287003465871,
        0.4947948181698064,
        0.45106052419343956,
        0.40763038412652414,
        0.3647993860357045,
        0.3228502371917532,
        0.2820508239741365,
        0.24265186139428022,
        0.2048847564012536,
    ];
    let got = kaiser(51, -3.0);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(51, -3.0)", i);
    }
}

#[test]
fn kaiser_0_beta_0_5() {
    let _report = Report;
    let want: &[f64] = &[];
    let got = kaiser(0, 0.5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(0, 0.5)", i);
    }
}

#[test]
fn kaiser_1_beta_0_5() {
    let _report = Report;
    let want: &[f64] = &[1.0];
    let got = kaiser(1, 0.5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(1, 0.5)", i);
    }
}

#[test]
fn kaiser_2_beta_0_5() {
    let _report = Report;
    let want: &[f64] = &[0.9403061933191574, 0.9403061933191574];
    let got = kaiser(2, 0.5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(2, 0.5)", i);
    }
}

#[test]
fn kaiser_3_beta_0_5() {
    let _report = Report;
    let want: &[f64] = &[0.9403061933191574, 1.0, 0.9403061933191574];
    let got = kaiser(3, 0.5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(3, 0.5)", i);
    }
}

#[test]
fn kaiser_12_beta_0_5() {
    let _report = Report;
    let want: &[f64] = &[
        0.9403061933191574,
        0.9598345881794499,
        0.9756027399223767,
        0.9875141588703074,
        0.9954958939201881,
        0.999499031774765,
        0.999499031774765,
        0.9954958939201881,
        0.9875141588703074,
        0.9756027399223767,
        0.9598345881794499,
        0.9403061933191574,
    ];
    let got = kaiser(12, 0.5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(12, 0.5)", i);
    }
}

#[test]
fn kaiser_51_beta_0_5() {
    let _report = Report;
    let want: &[f64] = &[
        0.9403061933191574,
        0.9449193409282346,
        0.949354820552551,
        0.953611319497149,
        0.9576875776919449,
        0.9615823881089496,
        0.9652945971619824,
        0.968823105088731,
        0.9721668663150309,
        0.9753248898012321,
        0.9782962393705391,
        0.9810800340191993,
        0.9836754482084445,
        0.9860817121380714,
        0.9882981120015747,
        0.9903239902227378,
        0.9921587456736031,
        0.9938018338737447,
        0.9952527671707686,
        0.9965111149019871,
        0.9975765035372048,
        0.9984486168025632,
        0.9991271957854116,
        0.9996120390201563,
        0.9999030025550654,
        1.0,
        0.9999030025550654,
        0.9996120390201563,
        0.9991271957854116,
        0.9984486168025632,
        0.9975765035372048,
        0.9965111149019871,
        0.9952527671707686,
        0.9938018338737447,
        0.9921587456736031,
        0.9903239902227378,
        0.9882981120015747,
        0.9860817121380714,
        0.9836754482084445,
        0.9810800340191993,
        0.9782962393705391,
        0.9753248898012321,
        0.9721668663150309,
        0.968823105088731,
        0.9652945971619824,
        0.9615823881089496,
        0.9576875776919449,
        0.953611319497149,
        0.949354820552551,
        0.9449193409282346,
        0.9403061933191574,
    ];
    let got = kaiser(51, 0.5);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, "kaiser(51, 0.5)", i);
    }
}

#[test]
fn hanning_1001_samples() {
    let _report = Report;
    const SAMPLES: &[(usize, f64)] = &[
        (0, 0.0),
        (1, 9.869571931442334e-06),
        (2, 3.947789809194413e-05),
        (249, 0.4968584280172205),
        (250, 0.5),
        (499, 0.9999901304280685),
        (500, 1.0),
        (501, 0.9999901304280685),
        (750, 0.5),
        (998, 3.947789809194413e-05),
        (999, 9.869571931442334e-06),
        (1000, 0.0),
    ];
    let got = hanning(1001);
    assert_eq!(got.len(), 1001);
    for &(i, want) in SAMPLES {
        close(got[i], want, "hanning(1001)", i);
    }
}

#[test]
fn hamming_1001_samples() {
    let _report = Report;
    const SAMPLES: &[(usize, f64)] = &[
        (0, 0.08000000000000002),
        (1, 0.08000908000617696),
        (2, 0.08003631966624458),
        (249, 0.5371097537758429),
        (250, 0.54),
        (499, 0.9999909199938231),
        (500, 1.0),
        (501, 0.9999909199938231),
        (750, 0.54),
        (998, 0.08003631966624458),
        (999, 0.08000908000617696),
        (1000, 0.08000000000000002),
    ];
    let got = hamming(1001);
    assert_eq!(got.len(), 1001);
    for &(i, want) in SAMPLES {
        close(got[i], want, "hamming(1001)", i);
    }
}

#[test]
fn blackman_1001_samples() {
    let _report = Report;
    const SAMPLES: &[(usize, f64)] = &[
        (0, -1.3877787807814457e-17),
        (1, 3.5531082367146194e-06),
        (2, 1.4213040755925443e-05),
        (249, 0.3368647444809152),
        (250, 0.34),
        (499, 0.9999838139643739),
        (500, 0.9999999999999999),
        (501, 0.9999838139643739),
        (750, 0.34),
        (998, 1.4213040755925443e-05),
        (999, 3.5531082367146194e-06),
        (1000, -1.3877787807814457e-17),
    ];
    let got = blackman(1001);
    assert_eq!(got.len(), 1001);
    for &(i, want) in SAMPLES {
        close(got[i], want, "blackman(1001)", i);
    }
}

#[test]
fn bartlett_1001_samples() {
    let _report = Report;
    const SAMPLES: &[(usize, f64)] = &[
        (0, 0.0),
        (1, 0.0020000000000000018),
        (2, 0.0040000000000000036),
        (249, 0.498),
        (250, 0.5),
        (499, 0.998),
        (500, 1.0),
        (501, 0.998),
        (750, 0.5),
        (998, 0.0040000000000000036),
        (999, 0.0020000000000000018),
        (1000, 0.0),
    ];
    let got = bartlett(1001);
    assert_eq!(got.len(), 1001);
    for &(i, want) in SAMPLES {
        close(got[i], want, "bartlett(1001)", i);
    }
}

#[test]
fn kaiser_1001_samples() {
    let _report = Report;
    const SAMPLES: &[(usize, f64)] = &[
        (0, 0.0013325139979024198),
        (1, 0.001432801776902677),
        (2, 0.001536605277121529),
        (249, 0.3372555761635363),
        (250, 0.34039362244018856),
        (499, 0.9999838334030634),
        (500, 1.0),
        (501, 0.9999838334030634),
        (750, 0.34039362244018856),
        (998, 0.001536605277121529),
        (999, 0.001432801776902677),
        (1000, 0.0013325139979024198),
    ];
    let got = kaiser(1001, 8.6);
    assert_eq!(got.len(), 1001);
    for &(i, want) in SAMPLES {
        close(got[i], want, "kaiser(1001, 8.6)", i);
    }
}

#[test]
fn windows_are_symmetric_and_peak_at_one() {
    let _report = Report;
    for f in [hanning, hamming, blackman, bartlett] {
        let w = f(33);
        for i in 0..w.len() {
            assert!((w[i] - w[w.len() - 1 - i]).abs() < 1e-15);
        }
        assert!((w[16] - 1.0).abs() < 1e-15);
    }
}
