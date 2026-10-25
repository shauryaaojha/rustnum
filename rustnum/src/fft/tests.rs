//! FFT tests: NumPy-computed constants plus mathematical properties.
// @generated fixtures: values computed with NumPy 2.3.2 (see tools in the task notes).
const FFT8_RE: [f64; 8] = [
    0.5714285714285714,
    0.4643217902420239,
    1.1714285714285713,
    -0.7356782097579766,
    0.5714285714285716,
    1.8785353526151192,
    -0.02857142857142858,
    0.6785353526151192,
];
const FFT8_IM: [f64; 8] = [
    0.20000000000000007,
    0.7760843115400643,
    -2.1714285714285717,
    -1.1923283828236018,
    0.6,
    -0.7189414543972074,
    -1.0285714285714285,
    3.5351855256807445,
];
const FFT12_RE: [f64; 12] = [
    -2.571428571428571,
    0.11073269562908206,
    4.321244472280612,
    0.6571428571428568,
    -0.5284977889122449,
    0.8035530186566321,
    0.8571428571428568,
    0.9107326956290817,
    2.242783503197959,
    1.0571428571428574,
    -2.606958757994897,
    1.6035530186566325,
];
const FFT12_IM: [f64; 12] = [
    0.0,
    -3.398900692201895,
    0.5153850220838192,
    -1.0571428571428574,
    -2.8948716593053936,
    -0.42967073636953323,
    2.0,
    0.02967073636953388,
    -1.9051283406946071,
    0.6571428571428576,
    3.4846149779161806,
    2.998900692201895,
];
const FFT17_RE: [f64; 17] = [
    3.8571428571428585,
    6.674247917655773,
    -3.0218231143519345,
    4.804092123829412,
    0.712663555711484,
    -1.2134354202904347,
    -2.9906452632115412,
    2.319581122818799,
    -1.8010151294053465,
    0.08029204992323502,
    0.20468067094439713,
    1.6095854652250805,
    -1.3637111789323175,
    2.8767690851102348,
    -6.037219440864505,
    -3.2600066876673264,
    6.263087100647848,
];
const FFT17_IM: [f64; 17] = [
    -0.6,
    2.975741930954872,
    3.8585811186561094,
    -0.2408430080691275,
    1.210052579899543,
    -0.36114815656578925,
    -3.494010845762872,
    -0.1321527145525832,
    2.942524751692494,
    0.5984306455178081,
    -0.4738280670859453,
    -0.808866963616294,
    -3.0904910792051106,
    -0.20593542358237849,
    4.365924646678295,
    -1.4069887185873822,
    -5.13699069637164,
];
const BIG97_IDX: [usize; 7] = [0, 1, 2, 3, 7, 48, 96];
const BIG97_RE: [f64; 7] = [
    28.142857142857153,
    0.4474469281561848,
    0.5172705029377229,
    0.6968007330592898,
    -0.7183567627166733,
    0.21985762094219138,
    0.44667425898787516,
];
const BIG97_IM: [f64; 7] = [
    1.1999999999999997,
    1.4812241108706055,
    1.849667676480881,
    2.4940307178864387,
    -2.155541273381271,
    0.44611386546850784,
    0.9204070341989394,
];
const BIG100_IDX: [usize; 7] = [0, 1, 2, 3, 7, 50, 99];
const BIG100_RE: [f64; 7] = [
    28.57142857142857,
    -0.0081596368311736,
    -0.07013021046994616,
    -0.24150153664378426,
    1.6746528194261667,
    1.7142857142857149,
    -0.032972528883700836,
];
const BIG100_IM: [f64; 7] = [
    6.661338147750939e-16,
    0.6548132559451993,
    1.494184832619621,
    2.903735192892355,
    -7.902278674557765,
    2.0,
    -0.654033479398249,
];
const BIG1000_IDX: [usize; 7] = [0, 1, 2, 3, 7, 500, 999];
const BIG1000_RE: [f64; 7] = [
    279.71428571428567,
    -6.002994141998668,
    -6.009470015523638,
    -6.019448805726958,
    -6.0953520078576595,
    0.8571428571428534,
    -6.000481182128517,
];
const BIG1000_IM: [f64; 7] = [
    0.40000000000000835,
    0.41257767965563774,
    0.42520748701529865,
    0.4379339879871078,
    0.49072983256955194,
    2.3999999999999986,
    0.3874302150665979,
];
const BIG1024_IDX: [usize; 7] = [0, 1, 2, 3, 7, 512, 1023];
const BIG1024_RE: [f64; 7] = [
    291.57142857142856,
    -0.9992060953992112,
    -0.9992813056074741,
    -1.00023243955078,
    -1.0130759791217632,
    -0.1428571428571388,
    -1.001660171991577,
];
const BIG1024_IM: [f64; 7] = [
    7.549516567451064e-15,
    -0.018410128897104028,
    -0.03684952750472359,
    -0.05535525453261492,
    -0.1308061318561451,
    2.0000000000000004,
    0.01841765793341621,
];
const IFFT12_RE: [f64; 12] = [
    -0.21428571428571425,
    0.13362941822138602,
    -0.21724656316624147,
    0.08809523809523812,
    0.1868986252664966,
    0.07589439130242345,
    0.07142857142857145,
    0.06696275155471934,
    -0.04404148240935374,
    0.054761904761904734,
    0.36010370602338426,
    0.009227724635756824,
];
const IFFT12_IM: [f64; 12] = [
    0.0,
    0.24990839101682455,
    0.29038458149301505,
    0.05476190476190477,
    -0.15876069505788393,
    0.0024725613641278323,
    0.16666666666666666,
    -0.03580589469746111,
    -0.24123930494211612,
    -0.08809523809523812,
    0.04294875184031828,
    -0.28324172435015793,
];
const RFFT9_RE: [f64; 5] = [
    0.0,
    0.642857142857143,
    0.6428571428571426,
    0.6428571428571428,
    0.6428571428571428,
];
const RFFT9_IM: [f64; 5] = [
    0.0,
    -1.7662354839351142,
    -0.7661273095248493,
    -0.37115374447904503,
    -0.11335305902687026,
];
const RFFT12_RE: [f64; 7] = [
    -2.571428571428571,
    0.8571428571428572,
    0.8571428571428572,
    0.857142857142857,
    0.8571428571428572,
    0.8571428571428568,
    0.8571428571428574,
];
const RFFT12_IM: [f64; 7] = [
    0.0,
    -3.198900692201895,
    -1.4846149779161808,
    -0.8571428571428572,
    -0.49487165930539345,
    -0.2296707363695336,
    0.0,
];
const IRFFT_N8: [f64; 8] = [
    0.4375,
    -1.0392135623730951,
    0.1875,
    0.6035533905932737,
    -0.0625,
    1.7892135623730951,
    -0.3125,
    -0.10355339059327373,
];
const IRFFT_N9: [f64; 9] = [
    0.2777777777777778,
    -0.7900486060630625,
    -0.2620100474698749,
    0.7743234540838294,
    -0.08968993141015592,
    0.6846940582832116,
    1.4478987681383926,
    -0.7749620009060331,
    0.23201652756591476,
];
const IRFFT_N6: [f64; 6] = [
    0.25,
    -0.8831730176127763,
    0.6391560817564839,
    0.5833333333333333,
    1.360843918243516,
    -0.450160315720557,
];
const IRFFT_IN_RE: [f64; 5] = [1.5, -2.0, 0.5, 3.0, -1.0];
const IRFFT_IN_IM: [f64; 5] = [0.7, 1.0, -0.25, 2.0, 0.5];
const FFT2_RE: [f64; 15] = [
    1.7142857142857142,
    0.9593372450173616,
    1.3653453681714085,
    -0.2592531129931226,
    -5.922572357338505,
    2.1517396894987244,
    0.8340587663551116,
    -3.214804863736397,
    3.733187307575826,
    2.718987361234031,
    0.4196888819298472,
    5.305242471320914,
    -0.45146136412285,
    0.9698438079622782,
    -1.7521963437317694,
];
const FFT2_IM: [f64; 15] = [
    0.20000000000000007,
    -2.1529305743820384,
    2.9069636426512253,
    0.41238590784831275,
    -4.366418976117499,
    3.2589732744431474,
    -2.5879103775143766,
    -1.9264783095483313,
    -0.3310611961830475,
    -0.006471261856102473,
    -4.658973274443148,
    3.5661460371058715,
    -1.0286135790667217,
    0.5668035342985628,
    6.147585152764145,
];
const IFFT2_RE: [f64; 15] = [
    0.11428571428571428,
    -0.39483815715590037,
    -0.01728354086620818,
    0.09102302454476055,
    0.06395581633449077,
    0.027979258795323165,
    -0.116813089582118,
    0.06465625386415189,
    -0.030097424274856677,
    0.35368283142139423,
    0.1434493126332483,
    0.18126582408226877,
    0.24887915383838843,
    -0.2143203242490932,
    0.05560391775700744,
];
const IFFT2_IM: [f64; 15] = [
    0.01333333333333334,
    -0.29109459840783325,
    0.027492393856554182,
    0.19379757617674834,
    -0.14352870495880257,
    -0.31059821829620987,
    0.4098390101842764,
    0.03778690228657085,
    -0.06857423860444813,
    0.23774306914039145,
    0.21726488496287655,
    -0.00043141745707349816,
    -0.02207074641220317,
    -0.12843188730322205,
    -0.1725273585009584,
];
const FFT2B_RE: [f64; 32] = [
    4.428571428571429,
    0.417926199941987,
    2.485714285714286,
    -3.8206534693564156,
    -0.4285714285714288,
    -0.8750690570848443,
    2.0857142857142863,
    2.5635106122135576,
    -4.0285714285714285,
    3.077940630324509,
    1.685714285714285,
    -2.350630798246919,
    -2.1999999999999993,
    2.350630798246919,
    2.7142857142857144,
    -3.07794063032451,
    4.7142857142857135,
    4.550630798246918,
    -2.2000000000000006,
    3.9892104675453193,
    2.7142857142857144,
    5.277940630324511,
    2.2,
    -2.960639038973891,
    -2.8285714285714287,
    -6.189210467545319,
    2.7142857142857144,
    -0.7606390389738911,
    2.2,
    0.7606390389738913,
    -7.114285714285715,
    6.18921046754532,
];
const FFT2B_IM: [f64; 32] = [
    0.3999999999999999,
    3.7983178713415504,
    0.6285714285714281,
    6.052444851419504,
    2.4000000000000004,
    1.8588249858013066,
    -0.2285714285714291,
    -2.109587708562361,
    -5.314285714285713,
    3.9892104675453197,
    -2.220446049250313e-16,
    1.4393609610261093,
    -0.5142857142857151,
    -2.960639038973891,
    -5.551115123125783e-16,
    8.38921046754532,
    1.5999999999999999,
    -3.805250462402102,
    -9.314285714285715,
    -7.710488545493102,
    -4.3999999999999995,
    -1.6233209661693269,
    -3.885714285714285,
    13.13905997406453,
    4.114285714285714,
    -0.8779406303245114,
    2.220446049250313e-16,
    -4.550630798246919,
    4.914285714285715,
    -0.15063079824691838,
    5.551115123125783e-16,
    -5.277940630324512,
];
const FFTFREQ8: [f64; 8] = [0.0, 1.25, 2.5, 3.75, -5.0, -3.75, -2.5, -1.25];
const RFFTFREQ8: [f64; 5] = [0.0, 1.25, 2.5, 3.75, 5.0];
const FFTFREQ9: [f64; 9] = [
    0.0,
    1.1111111111111112,
    2.2222222222222223,
    3.3333333333333335,
    4.444444444444445,
    -4.444444444444445,
    -3.3333333333333335,
    -2.2222222222222223,
    -1.1111111111111112,
];
const RFFTFREQ9: [f64; 5] = [
    0.0,
    1.1111111111111112,
    2.2222222222222223,
    3.3333333333333335,
    4.444444444444445,
];
const FFTFREQ1: [f64; 1] = [0.0];
const FFTSHIFT5: [f64; 5] = [3.0, 4.0, 0.0, 1.0, 2.0];
const IFFTSHIFT5: [f64; 5] = [2.0, 3.0, 4.0, 0.0, 1.0];
const FFTSHIFT6: [f64; 6] = [3.0, 4.0, 5.0, 0.0, 1.0, 2.0];
const IFFTSHIFT6: [f64; 6] = [3.0, 4.0, 5.0, 0.0, 1.0, 2.0];
const FFTSHIFT_3X4: [f64; 12] = [10.0, 11.0, 8.0, 9.0, 2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0];
const IFFTSHIFT_3X4: [f64; 12] = [6.0, 7.0, 4.0, 5.0, 10.0, 11.0, 8.0, 9.0, 2.0, 3.0, 0.0, 1.0];
const FFTSHIFT_3X5: [f64; 15] = [
    13.0, 14.0, 10.0, 11.0, 12.0, 3.0, 4.0, 0.0, 1.0, 2.0, 8.0, 9.0, 5.0, 6.0, 7.0,
];
const IFFTSHIFT_3X5: [f64; 15] = [
    7.0, 8.0, 9.0, 5.0, 6.0, 12.0, 13.0, 14.0, 10.0, 11.0, 2.0, 3.0, 4.0, 0.0, 1.0,
];

use super::kernels::*;

fn xre(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| ((k * 37 + 11) % 19) as f64 / 7.0 - 1.0)
        .collect()
}

fn xim(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| ((k * 13 + 5) % 11) as f64 / 5.0 - 1.0)
        .collect()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Largest elementwise error, relative to the largest magnitude in `want`.
fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = 1.0 + max_abs(want);
    got.iter()
        .zip(want)
        .fold(0.0, |m, (g, w)| f64::max(m, (g - w).abs() / scale))
}

fn check(got: &[f64], want: &[f64], tol: f64, what: &str) {
    let e = rel_err(got, want);
    assert!(e < tol, "{what}: relative error {e:e} exceeds {tol:e}");
}

fn pick(v: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| v[i]).collect()
}

// ------------------------------------------------------------------ NumPy parity

#[test]
fn fft_matches_numpy_small_lengths() {
    let cases: [(usize, &[f64], &[f64]); 3] = [
        (8, &FFT8_RE, &FFT8_IM),
        (12, &FFT12_RE, &FFT12_IM),
        (17, &FFT17_RE, &FFT17_IM),
    ];
    for (n, wre, wim) in cases {
        let (re, im) = fft(&xre(n), &xim(n));
        check(&re, wre, 1e-14, &format!("fft re n={n}"));
        check(&im, wim, 1e-14, &format!("fft im n={n}"));
    }
}

#[test]
fn fft_matches_numpy_large_lengths() {
    type BigCase<'a> = (usize, &'a [usize], &'a [f64], &'a [f64]);
    let cases: [BigCase; 4] = [
        (97, &BIG97_IDX, &BIG97_RE, &BIG97_IM),
        (100, &BIG100_IDX, &BIG100_RE, &BIG100_IM),
        (1000, &BIG1000_IDX, &BIG1000_RE, &BIG1000_IM),
        (1024, &BIG1024_IDX, &BIG1024_RE, &BIG1024_IM),
    ];
    for (n, idx, wre, wim) in cases {
        let (re, im) = fft(&xre(n), &xim(n));
        check(&pick(&re, idx), wre, 1e-13, &format!("fft re n={n}"));
        check(&pick(&im, idx), wim, 1e-13, &format!("fft im n={n}"));
    }
}

#[test]
fn ifft_matches_numpy() {
    let (re, im) = ifft(&xre(12), &xim(12));
    check(&re, &IFFT12_RE, 1e-14, "ifft re");
    check(&im, &IFFT12_IM, 1e-14, "ifft im");
}

#[test]
fn rfft_matches_numpy() {
    let (re, im) = rfft(&xre(9));
    assert_eq!(re.len(), 5);
    check(&re, &RFFT9_RE, 1e-14, "rfft re n=9");
    check(&im, &RFFT9_IM, 1e-14, "rfft im n=9");
    let (re, im) = rfft(&xre(12));
    assert_eq!(re.len(), 7);
    check(&re, &RFFT12_RE, 1e-14, "rfft re n=12");
    check(&im, &RFFT12_IM, 1e-14, "rfft im n=12");
}

#[test]
fn irfft_matches_numpy_including_ignored_imaginary_parts() {
    let cases: [(usize, &[f64]); 3] = [(8, &IRFFT_N8), (9, &IRFFT_N9), (6, &IRFFT_N6)];
    for (n, want) in cases {
        let got = irfft(&IRFFT_IN_RE, &IRFFT_IN_IM, n);
        check(&got, want, 1e-14, &format!("irfft n={n}"));
    }
}

#[test]
fn fft2_and_ifft2_match_numpy() {
    let (re, im) = fft2(&xre(15), &xim(15), 3, 5);
    check(&re, &FFT2_RE, 1e-14, "fft2 re");
    check(&im, &FFT2_IM, 1e-14, "fft2 im");
    let (re, im) = ifft2(&xre(15), &xim(15), 3, 5);
    check(&re, &IFFT2_RE, 1e-14, "ifft2 re");
    check(&im, &IFFT2_IM, 1e-14, "ifft2 im");
    let (re, im) = fft2(&xre(32), &xim(32), 4, 8);
    check(&re, &FFT2B_RE, 1e-14, "fft2 4x8 re");
    check(&im, &FFT2B_IM, 1e-14, "fft2 4x8 im");
}

#[test]
fn frequencies_match_numpy() {
    check(&fftfreq(8, 0.1), &FFTFREQ8, 1e-15, "fftfreq 8");
    check(&fftfreq(9, 0.1), &FFTFREQ9, 1e-15, "fftfreq 9");
    check(&fftfreq(1, 1.0), &FFTFREQ1, 1e-15, "fftfreq 1");
    check(&rfftfreq(8, 0.1), &RFFTFREQ8, 1e-15, "rfftfreq 8");
    check(&rfftfreq(9, 0.1), &RFFTFREQ9, 1e-15, "rfftfreq 9");
}

#[test]
fn shifts_match_numpy() {
    let ramp = |n: usize| (0..n).map(|v| v as f64).collect::<Vec<_>>();
    assert_eq!(fftshift(&ramp(5)), FFTSHIFT5);
    assert_eq!(ifftshift(&ramp(5)), IFFTSHIFT5);
    assert_eq!(fftshift(&ramp(6)), FFTSHIFT6);
    assert_eq!(ifftshift(&ramp(6)), IFFTSHIFT6);
    assert_eq!(shift_nd(&ramp(12), &[3, 4], false), FFTSHIFT_3X4);
    assert_eq!(shift_nd(&ramp(12), &[3, 4], true), IFFTSHIFT_3X4);
    assert_eq!(shift_nd(&ramp(15), &[3, 5], false), FFTSHIFT_3X5);
    assert_eq!(shift_nd(&ramp(15), &[3, 5], true), IFFTSHIFT_3X5);
}

#[test]
fn f32_transform_is_close_to_numpy() {
    let re32: Vec<f32> = xre(12).iter().map(|&v| v as f32).collect();
    let im32: Vec<f32> = xim(12).iter().map(|&v| v as f32).collect();
    let (re, im) = fft(&re32, &im32);
    let re64: Vec<f64> = re.iter().map(|&v| f64::from(v)).collect();
    let im64: Vec<f64> = im.iter().map(|&v| f64::from(v)).collect();
    check(&re64, &FFT12_RE, 1e-5, "f32 fft re");
    check(&im64, &FFT12_IM, 1e-5, "f32 fft im");
    // Bluestein path in f32.
    let n = 100;
    let a: Vec<f32> = xre(n).iter().map(|&v| v as f32).collect();
    let z = vec![0.0f32; n];
    let (fr, fi) = fft(&a, &z);
    let (br, bi) = ifft(&fr, &fi);
    for (g, w) in br.iter().zip(&a) {
        assert!((g - w).abs() < 1e-5, "f32 round trip {g} vs {w}");
    }
    assert!(bi.iter().all(|v| v.abs() < 1e-5));
}

// ------------------------------------------------------------------ properties

#[test]
fn ifft_of_fft_is_identity_for_every_length_up_to_64_and_some_primes() {
    let mut lengths: Vec<usize> = (1..=64).collect();
    lengths.extend([97, 101, 127, 257, 509, 1009]);
    for n in lengths {
        let (xr, xi) = (xre(n), xim(n));
        let (fr, fi) = fft(&xr, &xi);
        let (br, bi) = ifft(&fr, &fi);
        check(&br, &xr, 1e-13, &format!("round trip re n={n}"));
        check(&bi, &xi, 1e-13, &format!("round trip im n={n}"));
    }
}

#[test]
fn irfft_of_rfft_is_identity_for_even_and_odd_lengths() {
    for n in 1..=40 {
        let x = xre(n);
        let (re, im) = rfft(&x);
        assert_eq!(re.len(), n / 2 + 1);
        let back = irfft(&re, &im, n);
        check(&back, &x, 1e-13, &format!("rfft round trip n={n}"));
    }
}

#[test]
fn parseval_energy_is_preserved() {
    for n in [1usize, 2, 7, 16, 30, 64, 99, 128, 251] {
        let (xr, xi) = (xre(n), xim(n));
        let (fr, fi) = fft(&xr, &xi);
        let time: f64 = xr.iter().zip(&xi).map(|(a, b)| a * a + b * b).sum();
        let freq: f64 = fr.iter().zip(&fi).map(|(a, b)| a * a + b * b).sum::<f64>() / n as f64;
        assert!(
            (time - freq).abs() < 1e-12 * (1.0 + time),
            "Parseval n={n}: {time} vs {freq}"
        );
    }
}

#[test]
fn fft_of_delta_is_all_ones_and_shifted_delta_is_a_phase_ramp() {
    for n in [1usize, 2, 3, 5, 8, 13, 16, 31, 64, 100] {
        let mut d = vec![0.0f64; n];
        d[0] = 1.0;
        let (re, im) = fft(&d, &vec![0.0; n]);
        check(&re, &vec![1.0; n], 1e-14, &format!("delta re n={n}"));
        check(&im, &vec![0.0; n], 1e-14, &format!("delta im n={n}"));
    }
    // delta at position 1: X[k] = exp(-2 pi i k / n)
    let n = 10;
    let mut d = vec![0.0f64; n];
    d[1] = 1.0;
    let (re, im) = fft(&d, &vec![0.0; n]);
    for k in 0..n {
        let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
        assert!((re[k] - a.cos()).abs() < 1e-14 && (im[k] - a.sin()).abs() < 1e-14);
    }
}

#[test]
fn fft_is_linear() {
    let n = 45;
    let (a, b) = (xre(n), xim(n));
    let sum: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 2.0 * x - 3.0 * y).collect();
    let z = vec![0.0; n];
    let (fa, _) = fft(&a, &z);
    let (fb, _) = fft(&b, &z);
    let (fs, _) = fft(&sum, &z);
    let want: Vec<f64> = fa.iter().zip(&fb).map(|(x, y)| 2.0 * x - 3.0 * y).collect();
    check(&fs, &want, 1e-13, "linearity");
}

#[test]
fn pure_sine_lands_in_one_bin() {
    let n = 50; // non power of two
    let x: Vec<f64> = (0..n)
        .map(|k| (2.0 * std::f64::consts::PI * 5.0 * k as f64 / n as f64).sin())
        .collect();
    let (re, im) = rfft(&x);
    for (k, (r, i)) in re.iter().zip(&im).enumerate() {
        let mag = r.hypot(*i);
        if k == 5 {
            assert!((mag - 25.0).abs() < 1e-12, "peak magnitude {mag}");
        } else {
            assert!(mag < 1e-12, "leakage {mag} at bin {k}");
        }
    }
}

#[test]
fn rfft_equals_first_half_of_fft_and_fft_of_real_is_hermitian() {
    let n = 21;
    let x = xre(n);
    let (re, im) = rfft(&x);
    let (fr, fi) = fft(&x, &vec![0.0; n]);
    check(&re, &fr[..n / 2 + 1], 1e-14, "rfft re vs fft");
    check(&im, &fi[..n / 2 + 1], 1e-14, "rfft im vs fft");
    for k in 1..n {
        assert!((fr[k] - fr[n - k]).abs() < 1e-13 && (fi[k] + fi[n - k]).abs() < 1e-13);
    }
}

#[test]
fn ifft2_inverts_fft2() {
    for (r, c) in [
        (1usize, 1usize),
        (3, 5),
        (4, 8),
        (7, 6),
        (1, 9),
        (9, 1),
        (17, 3),
    ] {
        let (xr, xi) = (xre(r * c), xim(r * c));
        let (fr, fi) = fft2(&xr, &xi, r, c);
        let (br, bi) = ifft2(&fr, &fi, r, c);
        check(&br, &xr, 1e-13, &format!("fft2 round trip re {r}x{c}"));
        check(&bi, &xi, 1e-13, &format!("fft2 round trip im {r}x{c}"));
    }
}

#[test]
fn fft2_of_separable_input_is_outer_product_of_ffts() {
    let (r, c) = (6, 5);
    let a = xre(r);
    let b = xim(c);
    let mut grid = Vec::new();
    for x in &a {
        for y in &b {
            grid.push(x * y);
        }
    }
    let (gr, gi) = fft2(&grid, &vec![0.0; r * c], r, c);
    let (ar, ai) = fft(&a, &vec![0.0; r]);
    let (br, bi) = fft(&b, &vec![0.0; c]);
    for i in 0..r {
        for j in 0..c {
            let wr = ar[i] * br[j] - ai[i] * bi[j];
            let wi = ar[i] * bi[j] + ai[i] * br[j];
            assert!((gr[i * c + j] - wr).abs() < 1e-12 && (gi[i * c + j] - wi).abs() < 1e-12);
        }
    }
}

#[test]
fn shift_and_ifftshift_are_inverses() {
    for n in 0..20usize {
        let v: Vec<usize> = (0..n).collect();
        assert_eq!(ifftshift(&fftshift(&v)), v);
        assert_eq!(fftshift(&ifftshift(&v)), v);
    }
    let v: Vec<usize> = (0..30).collect();
    let s = shift_nd(&v, &[5, 6], false);
    assert_eq!(shift_nd(&s, &[5, 6], true), v);
}

#[test]
fn freq_layout_is_consistent_with_shift() {
    // fftshift(fftfreq(n)) is increasing for every n.
    for n in 1..=33 {
        let f = fftshift(&fftfreq(n, 1.0));
        assert!(f.windows(2).all(|w| w[0] < w[1]), "n={n}: {f:?}");
    }
}

#[test]
fn empty_and_single_element_inputs() {
    let (r, i) = fft::<f64>(&[], &[]);
    assert!(r.is_empty() && i.is_empty());
    let (r, i) = ifft::<f64>(&[], &[]);
    assert!(r.is_empty() && i.is_empty());
    let (r, i) = fft(&[3.5], &[-1.0]);
    assert_eq!((r[0], i[0]), (3.5, -1.0));
    let (r, i) = rfft(&[2.0]);
    assert_eq!((r.len(), r[0], i[0]), (1, 2.0, 0.0));
}

#[test]
fn resize_crops_and_pads() {
    let (r, i) = resize(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 5);
    assert_eq!(r, [1.0, 2.0, 3.0, 0.0, 0.0]);
    assert_eq!(i, [4.0, 5.0, 6.0, 0.0, 0.0]);
    let (r, i) = resize(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 2);
    assert_eq!((r, i), (vec![1.0, 2.0], vec![4.0, 5.0]));
}

#[test]
#[should_panic(expected = "same length")]
fn mismatched_parts_panic() {
    let _ = fft(&[1.0, 2.0], &[1.0]);
}

#[test]
fn small_integer_transforms_are_exact() {
    let (re, im) = fft(&[1.0, 2.0, 3.0, 4.0], &[0.0; 4]);
    assert_eq!(re, [10.0, -2.0, -2.0, -2.0]);
    assert_eq!(im, [0.0, 2.0, 0.0, -2.0]);
    let (re, im) = rfft(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!((re, im), (vec![10.0, -2.0, -2.0], vec![0.0, 2.0, 0.0]));
    assert_eq!(
        irfft(&[10.0, -2.0, -2.0], &[0.0, 2.0, 0.0], 4),
        [1.0, 2.0, 3.0, 4.0]
    );
    let (re, im) = fft2(&[1.0, 2.0, 3.0, 4.0], &[0.0; 4], 2, 2);
    assert_eq!(re, [10.0, -2.0, -4.0, 0.0]);
    assert_eq!(im, [0.0; 4]);
}
