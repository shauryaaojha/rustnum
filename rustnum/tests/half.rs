//! `f16` / `bf16` tests with hand-computed ground truth (the exhaustive NumPy comparison lives in
//! the parity suite, see `tools/gen_fixtures.py`).

use rustnum as np;
use rustnum::{bf16, f16};

fn h(x: f32) -> f16 {
    f16::from_f32(x)
}

#[test]
fn f16_known_bit_patterns() {
    assert_eq!(h(1.0).to_bits(), 0x3c00);
    assert_eq!(h(-2.0).to_bits(), 0xc000);
    assert_eq!(h(0.5).to_bits(), 0x3800);
    assert_eq!(h(65504.0).to_bits(), 0x7bff);
    assert_eq!(h(0.1).to_bits(), 0x2e66);
    assert_eq!(h(f32::INFINITY).to_bits(), 0x7c00);
    assert_eq!(h(f32::NEG_INFINITY).to_bits(), 0xfc00);
    assert_eq!(h(0.0).to_bits(), 0);
    assert_eq!(h(-0.0).to_bits(), 0x8000);
    // smallest subnormal 2^-24 and smallest normal 2^-14
    assert_eq!(h(2f32.powi(-24)).to_bits(), 1);
    assert_eq!(h(2f32.powi(-14)).to_bits(), 0x0400);
    assert!(h(f32::NAN).is_nan());
}

#[test]
fn f16_rounds_to_nearest_even() {
    // 1 + 2^-11 is exactly halfway between 1 and 1 + 2^-10: ties go to the even mantissa (1.0)
    assert_eq!(h(1.0 + 2f32.powi(-11)).to_bits(), 0x3c00);
    // 1 + 3 * 2^-11 is halfway between 1 + 2^-10 (odd) and 1 + 2^-9 (even)
    assert_eq!(h(1.0 + 3.0 * 2f32.powi(-11)).to_bits(), 0x3c02);
    // just above the tie rounds up
    assert_eq!(h(1.0 + 2f32.powi(-11) + 2f32.powi(-20)).to_bits(), 0x3c01);
    // overflow: 65520 is the halfway point to 65536 and rounds up to infinity
    assert_eq!(h(65519.0).to_bits(), 0x7bff);
    assert_eq!(h(65520.0).to_bits(), 0x7c00);
    // underflow: 2^-25 is half the smallest subnormal and rounds to zero (even), a hair more rounds up
    assert_eq!(h(2f32.powi(-25)).to_bits(), 0);
    assert_eq!(h(2f32.powi(-25) * 1.0001).to_bits(), 1);
    // direct f64 conversion does not double-round
    let tie_plus = 1.0 + 2f64.powi(-11) + 2f64.powi(-40);
    assert_eq!(f16::from_f64(tie_plus).to_bits(), 0x3c01);
    assert_eq!(f16::from_f32(tie_plus as f32).to_bits(), 0x3c00); // f32 already lost the sticky bit
}

#[test]
fn f16_all_bit_patterns_round_trip_through_f32() {
    for bits in 0..=u16::MAX {
        let x = f16::from_bits(bits);
        let back = f16::from_f32(x.to_f32());
        if x.is_nan() {
            assert!(back.is_nan(), "{bits:#06x}");
        } else {
            assert_eq!(back.to_bits(), bits, "{bits:#06x}");
        }
        // f64 path agrees with the f32 path
        if !x.is_nan() {
            assert_eq!(f16::from_f64(x.to_f64()).to_bits(), bits);
        }
    }
}

#[test]
fn f16_arithmetic_rounds_once() {
    assert_eq!((h(1.0) + h(2f32.powi(-11))).to_bits(), 0x3c00); // tie to even
    assert_eq!((h(2048.0) + h(1.0)).to_bits(), 0x6800); // 2049 is not representable, ties to even 2048
    assert_eq!((h(3.0) * h(0.1)).to_bits(), 0x34cc);
    assert_eq!((h(1.0) / h(3.0)).to_bits(), 0x3555);
    assert!((h(1.0) / h(0.0)).to_f32().is_infinite());
    assert!((h(0.0) / h(0.0)).is_nan());
    assert_eq!((-h(1.5)).to_f32(), -1.5);
    assert_eq!(np::Num::abs(h(-3.25)).to_f32(), 3.25);
}

#[test]
fn f16_nan_and_comparisons() {
    let nan = f16::NAN;
    assert!(nan != nan);
    assert!(nan.partial_cmp(&h(1.0)).is_none());
    assert!(!nan.lt(&h(1.0)) && !nan.gt(&h(1.0)));
    assert!(h(0.0) == h(-0.0));
    assert!(h(1.0) < h(2.0));
    let a = np::array![3.0f32, f32::NAN, 1.0].astype::<f16>();
    assert_eq!(a.sort().to_vec()[0].to_f32(), 1.0);
    assert!(a.sort().to_vec()[2].is_nan());
}

#[test]
fn f16_display_matches_numpy() {
    assert_eq!(h(0.1).to_string(), "0.1");
    assert_eq!(h(65504.0).to_string(), "65500");
    assert_eq!(h(1.5).to_string(), "1.5");
    assert_eq!(h(2f32.powi(-24)).to_string(), "0.00000006");
    assert_eq!(format!("{:?}", h(2.0)), "2.0");
    assert_eq!(format!("{:e}", h(0.1)), "1e-1");
    assert_eq!("0.1".parse::<f16>().unwrap().to_bits(), 0x2e66);
}

#[test]
fn f16_array_reductions_accumulate_in_f32() {
    let ones = np::ones::<f16>([10000]);
    assert_eq!(ones.sum().to_f32(), 10000.0);
    assert_eq!(ones.mean().to_f32(), 1.0);
    let a = np::array![1.0f32, 2.0, 3.0, 4.0].astype::<f16>();
    assert_eq!(a.var().to_f32(), 1.25);
    assert_eq!(a.std().to_bits(), h(1.25f32.sqrt()).to_bits());
    assert_eq!(a.sum_axis(0).item().to_f32(), 10.0);
    // cumsum rounds at every step, like NumPy's float16 accumulate
    let b = np::array![2048.0f32, 1.0, 1.0].astype::<f16>();
    assert_eq!(b.cumsum().to_vec()[2].to_f32(), 2048.0);
}

#[test]
fn f16_astype_paths_agree() {
    let src: Vec<f32> = (0..1013).map(|i| (i as f32 - 500.0) * 0.37).collect();
    let a = np::Array::from(src.clone());
    let bulk = a.astype::<f16>();
    let mut scalar = vec![f16::from_bits(0); src.len()];
    f16::slice_from_f32_scalar(&src, &mut scalar);
    assert_eq!(
        bulk.to_vec()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let back = bulk.astype::<f32>();
    for (b, s) in back.iter().zip(scalar.iter()) {
        assert_eq!(b, s.to_f32());
    }
    // non-contiguous input takes the generic path and gives the same numbers
    let t = np::Array::from_vec(src.clone(), &[1013, 1]).t();
    assert_eq!(t.astype::<f16>().to_vec(), bulk.to_vec());
}

#[test]
fn f16_float_functions_compute_through_f32() {
    let x = np::array![0.5f32, 1.0, 2.0].astype::<f16>();
    assert_eq!(x.exp().to_vec()[1].to_bits(), h(1f32.exp()).to_bits());
    assert_eq!(x.sqrt().to_vec()[2].to_bits(), h(2f32.sqrt()).to_bits());
    assert_eq!(x.sin().to_vec()[0].to_bits(), h(0.5f32.sin()).to_bits());
    assert_eq!(np::Float::round(h(2.5)).to_f32(), 2.0); // half to even
    assert!(np::Float::max(h(1.0), f16::NAN).is_nan());
}

#[test]
fn f16_npy_round_trip() {
    let a = np::array![[1.0f32, 0.1, -3.5], [65504.0, 0.0, 6.1e-5]].astype::<f16>();
    let bytes = np::io::to_npy_bytes(&a);
    assert!(String::from_utf8_lossy(&bytes[..80]).contains("'<f2'"));
    let b: np::Array<f16> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(b.shape(), [2, 3]);
    assert_eq!(
        a.to_vec().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.to_vec().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert!(matches!(
        np::io::from_npy_bytes::<f32>(&bytes),
        Err(np::Error::DType { .. })
    ));
}

#[test]
fn bf16_conversions() {
    let b = |x: f32| bf16::from_f32(x);
    assert_eq!(b(1.0).to_bits(), 0x3f80);
    assert_eq!(b(-2.0).to_bits(), 0xc000);
    assert_eq!(b(3.0e38).to_bits(), 0x7f62);
    // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even keeps 1.0
    assert_eq!(b(1.0 + 2f32.powi(-8)).to_bits(), 0x3f80);
    assert_eq!(b(1.0 + 3.0 * 2f32.powi(-8)).to_bits(), 0x3f82);
    assert_eq!(b(f32::INFINITY).to_bits(), 0x7f80);
    assert!(b(f32::NAN).is_nan());
    assert_eq!(b(f32::MAX).to_bits(), 0x7f80); // rounds up to infinity
    for bits in 0..=u16::MAX {
        let x = bf16::from_bits(bits);
        assert_eq!(x.to_f32().to_bits(), u32::from(bits) << 16);
        if !x.is_nan() {
            assert_eq!(bf16::from_f32(x.to_f32()).to_bits(), bits);
        }
    }
    // single rounding from f64: above the tie, f64 -> bf16 rounds up although f32 sees an exact tie
    let x = 1.0 + 2f64.powi(-8) + 2f64.powi(-40);
    assert_eq!(bf16::from_f64(x).to_bits(), 0x3f81);
    assert_eq!(bf16::from_f32(x as f32).to_bits(), 0x3f80);
}

#[test]
fn bf16_arrays_and_no_npy_dtype() {
    let a = np::array![1.0f32, 2.0, 3.0].astype::<bf16>();
    assert_eq!(a.dtype(), "bfloat16");
    assert_eq!(
        (&a * &a)
            .to_vec()
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>(),
        [1.0, 4.0, 9.0]
    );
    assert_eq!(a.sum().to_f32(), 6.0);
    assert_eq!(a.to_string(), "[1. 2. 3.]");
    // NumPy has no bfloat16 dtype, so there is no .npy encoding
    assert!(matches!(
        np::io::save_npy(std::env::temp_dir().join("rustnum_bf16_never.npy"), &a),
        Err(np::Error::DType { .. })
    ));
}

#[test]
fn reports_whether_the_f16c_path_runs() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    eprintln!(
        "f16c available: {}",
        std::arch::is_x86_feature_detected!("f16c")
    );
}
