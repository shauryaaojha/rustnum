//! # SIMD Transcendental Kernels
//!
//! ## Wiring Note for Orchestrator:
//! These slice-to-slice and in-place SIMD kernels are designed to wire directly into
//! `rustnum/src/ufunc.rs`.
//! For contiguous arrays:
//! - When unique/owned (`Arc::strong_count(&self.data) == 1`):
//!   call `simd_math::<op>_f64_inplace(data.as_mut_slice())` or `<op>_f32_inplace(...)`.
//! - For out-of-place evaluation:
//!   allocate an uninitialized/zeroed buffer of size `self.size()`, call
//!   `simd_math::<op>_f64(self.as_slice().unwrap(), dst.as_mut_slice().unwrap())`.
//! - Strided/non-contiguous arrays fall back to element-wise `par_map`.
//! - Parallel chunking: for slices >= 1<<15, chunks can be dispatched across rayon threads
//!   calling these kernels on each contiguous chunk.
//!
//! ## Measured accuracy (max ulp vs std, 2·10⁶ inputs per function over its full domain)
//! Produced by `cargo run --release -p rustnum --example simd_accuracy`.
//!
//! | Function | AVX2 f64 | AVX2 f32 | Vectorised today |
//! |---|---:|---:|---|
//! | exp | 2.0 | 3.0 | f64, f32, NEON |
//! | exp2 | 2.0 | 3.0 | f64, f32, NEON |
//! | expm1 | 2.0 | 3.0 | f64, f32, NEON |
//! | log | 2.0 | 3.0 | f64, f32, NEON |
//! | log2 | 3.0 | 3.0 | f64, f32, NEON |
//! | log10 | 3.0 | 4.0 | f64, f32, NEON |
//! | log1p | 1.0 | 1.0 | f64, f32, NEON |
//! | sin | 1.0 | 2.0 | f64, f32, NEON |
//! | cos | 1.0 | 2.0 | f64, f32, NEON |
//! | tan | 3.0 | 3.0 | f64, f32, NEON |
//! | tanh | 2.0 | 2.0 | f64, f32, NEON |
//! | sigmoid | 3.0 | 4.0 | f64, f32, NEON |

#![allow(dead_code, clippy::approx_constant, clippy::excessive_precision)]

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

static FORCE_SCALAR: AtomicBool = AtomicBool::new(false);
static DETECTED_AVX2_FMA: AtomicU8 = AtomicU8::new(0); // 0: uninit, 1: false, 2: true

/// Forces scalar fallback for all SIMD math operations (crate-private switch for testing).
#[inline]
pub(crate) fn set_force_scalar(force: bool) {
    FORCE_SCALAR.store(force, Ordering::Relaxed);
}

/// Returns whether scalar fallback is forced.
#[inline]
pub(crate) fn get_force_scalar() -> bool {
    FORCE_SCALAR.load(Ordering::Relaxed)
}

#[inline]
fn has_avx2_fma() -> bool {
    if get_force_scalar() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let state = DETECTED_AVX2_FMA.load(Ordering::Relaxed);
        if state == 2 {
            return true;
        } else if state == 1 {
            return false;
        }
        let available = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        DETECTED_AVX2_FMA.store(if available { 2 } else { 1 }, Ordering::Relaxed);
        available
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
#[allow(dead_code)]
fn has_neon() -> bool {
    !get_force_scalar()
}

// ---------------------------------------------------------------------------
// Mathematical Constants
// ---------------------------------------------------------------------------

const LN2_HI_F64: f64 = 0.6931471803691238;
const LN2_LO_F64: f64 = 1.9082149292705877e-10;
const INV_LN2_F64: f64 = std::f64::consts::LOG2_E;

const LN2_HI_F32: f32 = 0.693_145_75;
const LN2_LO_F32: f32 = 1.4286068e-6;
const INV_LN2_F32: f32 = std::f32::consts::LOG2_E;

const LOG10_2_HI_F64: f64 = 0.3010299956639812;
const LOG10_2_LO_F64: f64 = -2.8037281277884704e-18;

const LOG10_2_HI_F32: f32 = 0.3010299956639812;
const LOG10_2_LO_F32: f32 = -2.8037281277884704e-18;

const PIO2_1_F64: f64 = std::f64::consts::FRAC_PI_2;
const PIO2_2_F64: f64 = 6.123233995736766e-17;
const PIO2_3_F64: f64 = -1.4973849048591698e-33;
const INV_PIO2_F64: f64 = 2.0 / std::f64::consts::PI;

const PIO2_1_F32: f32 = 1.5703125;
const PIO2_2_F32: f32 = 0.0004837513;
const PIO2_3_F32: f32 = 7.5495336e-08;
const PIO2_4_F32: f32 = 2.563283e-12;
const INV_PIO2_F32: f32 = 2.0 / std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Scalar Fallback Kernels
// ---------------------------------------------------------------------------

#[inline(always)]
fn scalar_sigmoid_f64(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline(always)]
fn scalar_sigmoid_f32(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------------
// AVX2+FMA Implementation Kernels (f64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let inv_ln2 = _mm256_set1_pd(INV_LN2_F64);
        let ln2_hi = _mm256_set1_pd(LN2_HI_F64);
        let ln2_lo = _mm256_set1_pd(LN2_LO_F64);
        let one = _mm256_set1_pd(1.0);
        let max_exp = _mm256_set1_pd(709.782712893384);
        let min_exp = _mm256_set1_pd(-745.1332191019412);
        let inf = _mm256_set1_pd(f64::INFINITY);
        let zero = _mm256_setzero_pd();

        let c12 = _mm256_set1_pd(1.0 / 479001600.0);
        let c11 = _mm256_set1_pd(1.0 / 39916800.0);
        let c10 = _mm256_set1_pd(1.0 / 3628800.0);
        let c9 = _mm256_set1_pd(1.0 / 362880.0);
        let c8 = _mm256_set1_pd(1.0 / 40320.0);
        let c7 = _mm256_set1_pd(1.0 / 5040.0);
        let c6 = _mm256_set1_pd(1.0 / 720.0);
        let c5 = _mm256_set1_pd(1.0 / 120.0);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        let c3 = _mm256_set1_pd(1.0 / 6.0);
        let c2 = _mm256_set1_pd(0.5);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let k = _mm256_round_pd(
                _mm256_mul_pd(x, inv_ln2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let r1 = _mm256_fnmadd_pd(k, ln2_hi, x);
            let r = _mm256_fnmadd_pd(k, ln2_lo, r1);

            let mut p = _mm256_fmadd_pd(c12, r, c11);
            p = _mm256_fmadd_pd(p, r, c10);
            p = _mm256_fmadd_pd(p, r, c9);
            p = _mm256_fmadd_pd(p, r, c8);
            p = _mm256_fmadd_pd(p, r, c7);
            p = _mm256_fmadd_pd(p, r, c6);
            p = _mm256_fmadd_pd(p, r, c5);
            p = _mm256_fmadd_pd(p, r, c4);
            p = _mm256_fmadd_pd(p, r, c3);
            p = _mm256_fmadd_pd(p, r, c2);
            p = _mm256_fmadd_pd(p, r, one);
            let poly = _mm256_fmadd_pd(p, r, one);

            let ki = _mm256_cvtpd_epi32(k);
            let k1 = _mm_srai_epi32(ki, 1);
            let k2 = _mm_sub_epi32(ki, k1);

            let bias = _mm_set1_epi32(1023);
            let k1_bias = _mm_add_epi32(k1, bias);
            let k2_bias = _mm_add_epi32(k2, bias);

            let k1_64 = _mm256_cvtepi32_epi64(k1_bias);
            let k2_64 = _mm256_cvtepi32_epi64(k2_bias);

            let scale1 = _mm256_castsi256_pd(_mm256_slli_epi64(k1_64, 52));
            let scale2 = _mm256_castsi256_pd(_mm256_slli_epi64(k2_64, 52));

            let res = _mm256_mul_pd(_mm256_mul_pd(poly, scale1), scale2);

            let overflow_mask = _mm256_cmp_pd(x, max_exp, _CMP_GT_OQ);
            let underflow_mask = _mm256_cmp_pd(x, min_exp, _CMP_LT_OQ);

            let clamped = _mm256_blendv_pd(res, inf, overflow_mask);
            let final_res = _mm256_blendv_pd(clamped, zero, underflow_mask);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp2_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let max_exp = _mm256_set1_pd(1023.99);
        let min_exp = _mm256_set1_pd(-1074.0);
        let inf = _mm256_set1_pd(f64::INFINITY);
        let zero = _mm256_setzero_pd();
        let one = _mm256_set1_pd(1.0);

        let c12 = _mm256_set1_pd(1.0 / 479001600.0);
        let c11 = _mm256_set1_pd(1.0 / 39916800.0);
        let c10 = _mm256_set1_pd(1.0 / 3628800.0);
        let c9 = _mm256_set1_pd(1.0 / 362880.0);
        let c8 = _mm256_set1_pd(1.0 / 40320.0);
        let c7 = _mm256_set1_pd(1.0 / 5040.0);
        let c6 = _mm256_set1_pd(1.0 / 720.0);
        let c5 = _mm256_set1_pd(1.0 / 120.0);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        let c3 = _mm256_set1_pd(1.0 / 6.0);
        let c2 = _mm256_set1_pd(0.5);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let k = _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let rem = _mm256_sub_pd(x, k);
            let r = _mm256_mul_pd(rem, ln2);

            let mut p = _mm256_fmadd_pd(c12, r, c11);
            p = _mm256_fmadd_pd(p, r, c10);
            p = _mm256_fmadd_pd(p, r, c9);
            p = _mm256_fmadd_pd(p, r, c8);
            p = _mm256_fmadd_pd(p, r, c7);
            p = _mm256_fmadd_pd(p, r, c6);
            p = _mm256_fmadd_pd(p, r, c5);
            p = _mm256_fmadd_pd(p, r, c4);
            p = _mm256_fmadd_pd(p, r, c3);
            p = _mm256_fmadd_pd(p, r, c2);
            p = _mm256_fmadd_pd(p, r, one);
            let poly = _mm256_fmadd_pd(p, r, one);

            let ki = _mm256_cvtpd_epi32(k);
            let k1 = _mm_srai_epi32(ki, 1);
            let k2 = _mm_sub_epi32(ki, k1);

            let bias = _mm_set1_epi32(1023);
            let k1_bias = _mm_add_epi32(k1, bias);
            let k2_bias = _mm_add_epi32(k2, bias);

            let k1_64 = _mm256_cvtepi32_epi64(k1_bias);
            let k2_64 = _mm256_cvtepi32_epi64(k2_bias);

            let scale1 = _mm256_castsi256_pd(_mm256_slli_epi64(k1_64, 52));
            let scale2 = _mm256_castsi256_pd(_mm256_slli_epi64(k2_64, 52));

            let res = _mm256_mul_pd(_mm256_mul_pd(poly, scale1), scale2);

            let overflow_mask = _mm256_cmp_pd(x, max_exp, _CMP_GT_OQ);
            let underflow_mask = _mm256_cmp_pd(x, min_exp, _CMP_LT_OQ);

            let clamped = _mm256_blendv_pd(res, inf, overflow_mask);
            let final_res = _mm256_blendv_pd(clamped, zero, underflow_mask);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp2();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn expm1_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let inv_ln2 = _mm256_set1_pd(INV_LN2_F64);
        let ln2_hi = _mm256_set1_pd(LN2_HI_F64);
        let ln2_lo = _mm256_set1_pd(LN2_LO_F64);
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);
        let max_exp = _mm256_set1_pd(709.782712893384);
        let min_expm1 = _mm256_set1_pd(-54.0);
        let inf = _mm256_set1_pd(f64::INFINITY);
        let zero = _mm256_setzero_pd();
        let _ = zero;
        let nan = _mm256_set1_pd(f64::NAN);

        let c16 = _mm256_set1_pd(1.0 / 20922789888000.0);
        let c15 = _mm256_set1_pd(1.0 / 1307674368000.0);
        let c14 = _mm256_set1_pd(1.0 / 87178291200.0);
        let c13 = _mm256_set1_pd(1.0 / 6227020800.0);
        let c12 = _mm256_set1_pd(1.0 / 479001600.0);
        let c11 = _mm256_set1_pd(1.0 / 39916800.0);
        let c10 = _mm256_set1_pd(1.0 / 3628800.0);
        let c9 = _mm256_set1_pd(1.0 / 362880.0);
        let c8 = _mm256_set1_pd(1.0 / 40320.0);
        let c7 = _mm256_set1_pd(1.0 / 5040.0);
        let c6 = _mm256_set1_pd(1.0 / 720.0);
        let c5 = _mm256_set1_pd(1.0 / 120.0);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        let c3 = _mm256_set1_pd(1.0 / 6.0);
        let c2 = _mm256_set1_pd(0.5);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let k = _mm256_round_pd(
                _mm256_mul_pd(x, inv_ln2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let r1 = _mm256_fnmadd_pd(k, ln2_hi, x);
            let r = _mm256_fnmadd_pd(k, ln2_lo, r1);

            let mut p = _mm256_fmadd_pd(c16, r, c15);
            p = _mm256_fmadd_pd(p, r, c14);
            p = _mm256_fmadd_pd(p, r, c13);
            p = _mm256_fmadd_pd(p, r, c12);
            p = _mm256_fmadd_pd(p, r, c11);
            p = _mm256_fmadd_pd(p, r, c10);
            p = _mm256_fmadd_pd(p, r, c9);
            p = _mm256_fmadd_pd(p, r, c8);
            p = _mm256_fmadd_pd(p, r, c7);
            p = _mm256_fmadd_pd(p, r, c6);
            p = _mm256_fmadd_pd(p, r, c5);
            p = _mm256_fmadd_pd(p, r, c4);
            p = _mm256_fmadd_pd(p, r, c3);
            p = _mm256_fmadd_pd(p, r, c2);
            let poly = _mm256_fmadd_pd(_mm256_mul_pd(r, r), p, r);

            let ki = _mm256_cvtpd_epi32(k);
            let k1 = _mm_srai_epi32(ki, 1);
            let k2 = _mm_sub_epi32(ki, k1);

            let bias = _mm_set1_epi32(1023);
            let k1_bias = _mm_add_epi32(k1, bias);
            let k2_bias = _mm_add_epi32(k2, bias);

            let k1_64 = _mm256_cvtepi32_epi64(k1_bias);
            let k2_64 = _mm256_cvtepi32_epi64(k2_bias);

            let scale1 = _mm256_castsi256_pd(_mm256_slli_epi64(k1_64, 52));
            let scale2 = _mm256_castsi256_pd(_mm256_slli_epi64(k2_64, 52));
            let twok = _mm256_mul_pd(scale1, scale2);

            let twok_minus_one = _mm256_sub_pd(twok, one);
            let res_k_nonzero = _mm256_fmadd_pd(twok, poly, twok_minus_one);

            let mut p_dir = _mm256_fmadd_pd(c16, x, c15);
            p_dir = _mm256_fmadd_pd(p_dir, x, c14);
            p_dir = _mm256_fmadd_pd(p_dir, x, c13);
            p_dir = _mm256_fmadd_pd(p_dir, x, c12);
            p_dir = _mm256_fmadd_pd(p_dir, x, c11);
            p_dir = _mm256_fmadd_pd(p_dir, x, c10);
            p_dir = _mm256_fmadd_pd(p_dir, x, c9);
            p_dir = _mm256_fmadd_pd(p_dir, x, c8);
            p_dir = _mm256_fmadd_pd(p_dir, x, c7);
            p_dir = _mm256_fmadd_pd(p_dir, x, c6);
            p_dir = _mm256_fmadd_pd(p_dir, x, c5);
            p_dir = _mm256_fmadd_pd(p_dir, x, c4);
            p_dir = _mm256_fmadd_pd(p_dir, x, c3);
            p_dir = _mm256_fmadd_pd(p_dir, x, c2);
            let poly_direct = _mm256_fmadd_pd(_mm256_mul_pd(x, x), p_dir, x);

            let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
            let is_small = _mm256_cmp_pd(abs_x, _mm256_set1_pd(0.69314718056), _CMP_LT_OQ);
            let res = _mm256_blendv_pd(res_k_nonzero, poly_direct, is_small);

            let overflow_mask = _mm256_cmp_pd(x, max_exp, _CMP_GT_OQ);
            let underflow_mask = _mm256_cmp_pd(x, min_expm1, _CMP_LT_OQ);
            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_pd(res, inf, overflow_mask);
            final_res = _mm256_blendv_pd(final_res, neg_one, underflow_mask);
            final_res = _mm256_blendv_pd(final_res, nan, is_nan);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp_m1();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_pd();
        let ln2_hi = _mm256_set1_pd(LN2_HI_F64);
        let ln2_lo = _mm256_set1_pd(LN2_LO_F64);
        let sqrt2_half = _mm256_set1_pd(std::f64::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let pos_inf = _mm256_set1_pd(f64::INFINITY);
        let nan = _mm256_set1_pd(f64::NAN);
        let min_normal = _mm256_set1_pd(2.2250738585072014e-308);

        let c10 = _mm256_set1_pd(2.0 / 21.0);
        let c9 = _mm256_set1_pd(2.0 / 19.0);
        let c8 = _mm256_set1_pd(2.0 / 17.0);
        let c7 = _mm256_set1_pd(2.0 / 15.0);
        let c6 = _mm256_set1_pd(2.0 / 13.0);
        let c5 = _mm256_set1_pd(2.0 / 11.0);
        let c4 = _mm256_set1_pd(2.0 / 9.0);
        let c3 = _mm256_set1_pd(2.0 / 7.0);
        let c2 = _mm256_set1_pd(2.0 / 5.0);
        let c1 = _mm256_set1_pd(2.0 / 3.0);
        let c0 = _mm256_set1_pd(2.0);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let is_subnormal = _mm256_and_pd(
                _mm256_cmp_pd(x, min_normal, _CMP_LT_OQ),
                _mm256_cmp_pd(x, zero, _CMP_GT_OQ),
            );
            if _mm256_movemask_pd(is_subnormal) != 0 {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).ln();
                }
                i += 4;
                continue;
            }

            let x_bits = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_bits, 52);
            let low_halves =
                _mm256_permutevar8x32_epi32(exp_bits, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
            let mut k = _mm256_sub_pd(
                _mm256_cvtepi32_pd(_mm256_castsi256_si128(low_halves)),
                _mm256_set1_pd(1023.0),
            );

            let mant_mask = _mm256_set1_epi64x(0x000F_FFFF_FFFF_FFFF);
            let exp_1023 = _mm256_set1_epi64x(0x3FF0_0000_0000_0000);
            let mant_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mant_mask), exp_1023);
            let mut m = _mm256_mul_pd(_mm256_castsi256_pd(mant_bits), _mm256_set1_pd(0.5));
            k = _mm256_add_pd(k, one);

            let lt_mask = _mm256_cmp_pd(m, sqrt2_half, _CMP_LT_OQ);
            m = _mm256_blendv_pd(m, _mm256_mul_pd(m, two), lt_mask);
            k = _mm256_blendv_pd(k, _mm256_sub_pd(k, one), lt_mask);

            let u = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
            let u2 = _mm256_mul_pd(u, u);

            let mut p = _mm256_fmadd_pd(c10, u2, c9);
            p = _mm256_fmadd_pd(p, u2, c8);
            p = _mm256_fmadd_pd(p, u2, c7);
            p = _mm256_fmadd_pd(p, u2, c6);
            p = _mm256_fmadd_pd(p, u2, c5);
            p = _mm256_fmadd_pd(p, u2, c4);
            p = _mm256_fmadd_pd(p, u2, c3);
            p = _mm256_fmadd_pd(p, u2, c2);
            p = _mm256_fmadd_pd(p, u2, c1);
            p = _mm256_fmadd_pd(p, u2, c0);
            let ln_m = _mm256_mul_pd(u, p);

            let res = _mm256_fmadd_pd(k, ln2_hi, _mm256_fmadd_pd(k, ln2_lo, ln_m));

            let eq_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
            let lt_zero = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
            let is_inf = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);
            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_pd(res, neg_inf, eq_zero);
            final_res = _mm256_blendv_pd(final_res, nan, lt_zero);
            final_res = _mm256_blendv_pd(final_res, pos_inf, is_inf);
            final_res = _mm256_blendv_pd(final_res, nan, is_nan);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.ln();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log2_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_pd();
        let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
        let sqrt2_half = _mm256_set1_pd(std::f64::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let pos_inf = _mm256_set1_pd(f64::INFINITY);
        let nan = _mm256_set1_pd(f64::NAN);
        let min_normal = _mm256_set1_pd(2.2250738585072014e-308);

        let c10 = _mm256_set1_pd(2.0 / 21.0);
        let c9 = _mm256_set1_pd(2.0 / 19.0);
        let c8 = _mm256_set1_pd(2.0 / 17.0);
        let c7 = _mm256_set1_pd(2.0 / 15.0);
        let c6 = _mm256_set1_pd(2.0 / 13.0);
        let c5 = _mm256_set1_pd(2.0 / 11.0);
        let c4 = _mm256_set1_pd(2.0 / 9.0);
        let c3 = _mm256_set1_pd(2.0 / 7.0);
        let c2 = _mm256_set1_pd(2.0 / 5.0);
        let c1 = _mm256_set1_pd(2.0 / 3.0);
        let c0 = _mm256_set1_pd(2.0);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let is_subnormal = _mm256_and_pd(
                _mm256_cmp_pd(x, min_normal, _CMP_LT_OQ),
                _mm256_cmp_pd(x, zero, _CMP_GT_OQ),
            );
            if _mm256_movemask_pd(is_subnormal) != 0 {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).log2();
                }
                i += 4;
                continue;
            }

            let x_bits = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_bits, 52);
            let low_halves =
                _mm256_permutevar8x32_epi32(exp_bits, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
            let mut k = _mm256_sub_pd(
                _mm256_cvtepi32_pd(_mm256_castsi256_si128(low_halves)),
                _mm256_set1_pd(1023.0),
            );

            let mant_mask = _mm256_set1_epi64x(0x000F_FFFF_FFFF_FFFF);
            let exp_1023 = _mm256_set1_epi64x(0x3FF0_0000_0000_0000);
            let mant_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mant_mask), exp_1023);
            let mut m = _mm256_mul_pd(_mm256_castsi256_pd(mant_bits), _mm256_set1_pd(0.5));
            k = _mm256_add_pd(k, one);

            let lt_mask = _mm256_cmp_pd(m, sqrt2_half, _CMP_LT_OQ);
            m = _mm256_blendv_pd(m, _mm256_mul_pd(m, two), lt_mask);
            k = _mm256_blendv_pd(k, _mm256_sub_pd(k, one), lt_mask);

            let u = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
            let u2 = _mm256_mul_pd(u, u);

            let mut p = _mm256_fmadd_pd(c10, u2, c9);
            p = _mm256_fmadd_pd(p, u2, c8);
            p = _mm256_fmadd_pd(p, u2, c7);
            p = _mm256_fmadd_pd(p, u2, c6);
            p = _mm256_fmadd_pd(p, u2, c5);
            p = _mm256_fmadd_pd(p, u2, c4);
            p = _mm256_fmadd_pd(p, u2, c3);
            p = _mm256_fmadd_pd(p, u2, c2);
            p = _mm256_fmadd_pd(p, u2, c1);
            p = _mm256_fmadd_pd(p, u2, c0);
            let ln_m = _mm256_mul_pd(u, p);

            let res = _mm256_fmadd_pd(ln_m, log2_e, k);

            let eq_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
            let lt_zero = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
            let is_inf = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);
            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_pd(res, neg_inf, eq_zero);
            final_res = _mm256_blendv_pd(final_res, nan, lt_zero);
            final_res = _mm256_blendv_pd(final_res, pos_inf, is_inf);
            final_res = _mm256_blendv_pd(final_res, nan, is_nan);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.log2();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log10_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_pd();
        let log10_2_hi = _mm256_set1_pd(LOG10_2_HI_F64);
        let log10_2_lo = _mm256_set1_pd(LOG10_2_LO_F64);
        let log10_e = _mm256_set1_pd(std::f64::consts::LOG10_E);
        let sqrt2_half = _mm256_set1_pd(std::f64::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let pos_inf = _mm256_set1_pd(f64::INFINITY);
        let nan = _mm256_set1_pd(f64::NAN);
        let min_normal = _mm256_set1_pd(2.2250738585072014e-308);

        let c10 = _mm256_set1_pd(2.0 / 21.0);
        let c9 = _mm256_set1_pd(2.0 / 19.0);
        let c8 = _mm256_set1_pd(2.0 / 17.0);
        let c7 = _mm256_set1_pd(2.0 / 15.0);
        let c6 = _mm256_set1_pd(2.0 / 13.0);
        let c5 = _mm256_set1_pd(2.0 / 11.0);
        let c4 = _mm256_set1_pd(2.0 / 9.0);
        let c3 = _mm256_set1_pd(2.0 / 7.0);
        let c2 = _mm256_set1_pd(2.0 / 5.0);
        let c1 = _mm256_set1_pd(2.0 / 3.0);
        let c0 = _mm256_set1_pd(2.0);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let is_subnormal = _mm256_and_pd(
                _mm256_cmp_pd(x, min_normal, _CMP_LT_OQ),
                _mm256_cmp_pd(x, zero, _CMP_GT_OQ),
            );
            if _mm256_movemask_pd(is_subnormal) != 0 {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).log10();
                }
                i += 4;
                continue;
            }

            let x_bits = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_bits, 52);
            let low_halves =
                _mm256_permutevar8x32_epi32(exp_bits, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
            let mut k = _mm256_sub_pd(
                _mm256_cvtepi32_pd(_mm256_castsi256_si128(low_halves)),
                _mm256_set1_pd(1023.0),
            );

            let mant_mask = _mm256_set1_epi64x(0x000F_FFFF_FFFF_FFFF);
            let exp_1023 = _mm256_set1_epi64x(0x3FF0_0000_0000_0000);
            let mant_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mant_mask), exp_1023);
            let mut m = _mm256_mul_pd(_mm256_castsi256_pd(mant_bits), _mm256_set1_pd(0.5));
            k = _mm256_add_pd(k, one);

            let lt_mask = _mm256_cmp_pd(m, sqrt2_half, _CMP_LT_OQ);
            m = _mm256_blendv_pd(m, _mm256_mul_pd(m, two), lt_mask);
            k = _mm256_blendv_pd(k, _mm256_sub_pd(k, one), lt_mask);

            let u = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
            let u2 = _mm256_mul_pd(u, u);

            let mut p = _mm256_fmadd_pd(c10, u2, c9);
            p = _mm256_fmadd_pd(p, u2, c8);
            p = _mm256_fmadd_pd(p, u2, c7);
            p = _mm256_fmadd_pd(p, u2, c6);
            p = _mm256_fmadd_pd(p, u2, c5);
            p = _mm256_fmadd_pd(p, u2, c4);
            p = _mm256_fmadd_pd(p, u2, c3);
            p = _mm256_fmadd_pd(p, u2, c2);
            p = _mm256_fmadd_pd(p, u2, c1);
            p = _mm256_fmadd_pd(p, u2, c0);
            let ln_m = _mm256_mul_pd(u, p);

            let res = _mm256_fmadd_pd(
                k,
                log10_2_hi,
                _mm256_fmadd_pd(ln_m, log10_e, _mm256_mul_pd(k, log10_2_lo)),
            );

            let eq_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
            let lt_zero = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
            let is_inf = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);
            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_pd(res, neg_inf, eq_zero);
            final_res = _mm256_blendv_pd(final_res, nan, lt_zero);
            final_res = _mm256_blendv_pd(final_res, pos_inf, is_inf);
            final_res = _mm256_blendv_pd(final_res, nan, is_nan);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.log10();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log1p_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_pd();
        let ln2_hi = _mm256_set1_pd(LN2_HI_F64);
        let ln2_lo = _mm256_set1_pd(LN2_LO_F64);
        let sqrt2_half = _mm256_set1_pd(std::f64::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);
        let two = _mm256_set1_pd(2.0);
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let pos_inf = _mm256_set1_pd(f64::INFINITY);
        let nan = _mm256_set1_pd(f64::NAN);
        let threshold = _mm256_set1_pd(0.35);

        let c10 = _mm256_set1_pd(2.0 / 21.0);
        let c9 = _mm256_set1_pd(2.0 / 19.0);
        let c8 = _mm256_set1_pd(2.0 / 17.0);
        let c7 = _mm256_set1_pd(2.0 / 15.0);
        let c6 = _mm256_set1_pd(2.0 / 13.0);
        let c5 = _mm256_set1_pd(2.0 / 11.0);
        let c4 = _mm256_set1_pd(2.0 / 9.0);
        let c3 = _mm256_set1_pd(2.0 / 7.0);
        let c2 = _mm256_set1_pd(2.0 / 5.0);
        let c1 = _mm256_set1_pd(2.0 / 3.0);
        let c0 = _mm256_set1_pd(2.0);

        let sign_mask = _mm256_set1_pd(-0.0);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));
            let abs_x = _mm256_andnot_pd(sign_mask, x);
            let is_small = _mm256_cmp_pd(abs_x, threshold, _CMP_LT_OQ);

            // Small branch: u = x / (2 + x)
            let u_s = _mm256_div_pd(x, _mm256_add_pd(two, x));
            let u2_s = _mm256_mul_pd(u_s, u_s);
            let mut p_s = _mm256_fmadd_pd(c10, u2_s, c9);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c8);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c7);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c6);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c5);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c4);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c3);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c2);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c1);
            p_s = _mm256_fmadd_pd(p_s, u2_s, c0);
            let res_small = _mm256_mul_pd(u_s, p_s);

            // Large branch: log(1 + x)
            let y = _mm256_add_pd(one, x);
            let y_bits = _mm256_castpd_si256(y);
            let exp_bits = _mm256_srli_epi64(y_bits, 52);
            let low_halves =
                _mm256_permutevar8x32_epi32(exp_bits, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
            let mut k = _mm256_sub_pd(
                _mm256_cvtepi32_pd(_mm256_castsi256_si128(low_halves)),
                _mm256_set1_pd(1023.0),
            );

            let mant_mask = _mm256_set1_epi64x(0x000F_FFFF_FFFF_FFFF);
            let exp_1023 = _mm256_set1_epi64x(0x3FF0_0000_0000_0000);
            let mant_bits = _mm256_or_si256(_mm256_and_si256(y_bits, mant_mask), exp_1023);
            let mut m = _mm256_mul_pd(_mm256_castsi256_pd(mant_bits), _mm256_set1_pd(0.5));
            k = _mm256_add_pd(k, one);

            let lt_mask = _mm256_cmp_pd(m, sqrt2_half, _CMP_LT_OQ);
            m = _mm256_blendv_pd(m, _mm256_mul_pd(m, two), lt_mask);
            k = _mm256_blendv_pd(k, _mm256_sub_pd(k, one), lt_mask);

            let u_l = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
            let u2_l = _mm256_mul_pd(u_l, u_l);

            let mut p_l = _mm256_fmadd_pd(c10, u2_l, c9);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c8);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c7);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c6);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c5);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c4);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c3);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c2);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c1);
            p_l = _mm256_fmadd_pd(p_l, u2_l, c0);
            let ln_m = _mm256_mul_pd(u_l, p_l);
            let res_large = _mm256_fmadd_pd(k, ln2_hi, _mm256_fmadd_pd(k, ln2_lo, ln_m));

            let res = _mm256_blendv_pd(res_large, res_small, is_small);

            let eq_neg_one = _mm256_cmp_pd(x, neg_one, _CMP_EQ_OQ);
            let lt_neg_one = _mm256_cmp_pd(x, neg_one, _CMP_LT_OQ);
            let is_inf = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);
            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
            let is_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);

            let mut final_res = _mm256_blendv_pd(res, neg_inf, eq_neg_one);
            final_res = _mm256_blendv_pd(final_res, nan, lt_neg_one);
            final_res = _mm256_blendv_pd(final_res, pos_inf, is_inf);
            final_res = _mm256_blendv_pd(final_res, nan, is_nan);
            final_res = _mm256_blendv_pd(final_res, x, is_zero);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.ln_1p();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sin_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let p1 = _mm256_set1_pd(PIO2_1_F64);
        let p2 = _mm256_set1_pd(PIO2_2_F64);
        let p3 = _mm256_set1_pd(PIO2_3_F64);
        let inv_pio2 = _mm256_set1_pd(INV_PIO2_F64);
        let one = _mm256_set1_pd(1.0);
        let bound = _mm256_set1_pd(1e5);
        let sign_mask = _mm256_set1_pd(-0.0);

        let s8 = _mm256_set1_pd(1.0 / 355687428096000.0);
        let s7 = _mm256_set1_pd(-1.0 / 1307674368000.0);
        let s6 = _mm256_set1_pd(1.0 / 6227020800.0);
        let s5 = _mm256_set1_pd(-1.0 / 39916800.0);
        let s4 = _mm256_set1_pd(1.0 / 362880.0);
        let s3 = _mm256_set1_pd(-1.0 / 5040.0);
        let s2 = _mm256_set1_pd(1.0 / 120.0);
        let s1 = _mm256_set1_pd(-1.0 / 6.0);

        let c8 = _mm256_set1_pd(1.0 / 20922789888000.0);
        let c7 = _mm256_set1_pd(-1.0 / 87178291200.0);
        let c6 = _mm256_set1_pd(1.0 / 479001600.0);
        let c5 = _mm256_set1_pd(-1.0 / 3628800.0);
        let c4 = _mm256_set1_pd(1.0 / 40320.0);
        let c3 = _mm256_set1_pd(-1.0 / 720.0);
        let c2 = _mm256_set1_pd(1.0 / 24.0);
        let c1 = _mm256_set1_pd(-0.5);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let abs_x = _mm256_andnot_pd(sign_mask, x);
            let in_range = _mm256_cmp_pd(abs_x, bound, _CMP_LT_OQ);
            if _mm256_movemask_pd(in_range) != 0x0F {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).sin();
                }
                i += 4;
                continue;
            }

            let k = _mm256_round_pd(
                _mm256_mul_pd(x, inv_pio2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let mut r = _mm256_fnmadd_pd(k, p1, x);
            r = _mm256_fnmadd_pd(k, p2, r);
            r = _mm256_fnmadd_pd(k, p3, r);

            let r2 = _mm256_mul_pd(r, r);

            let mut sp = _mm256_fmadd_pd(s8, r2, s7);
            sp = _mm256_fmadd_pd(sp, r2, s6);
            sp = _mm256_fmadd_pd(sp, r2, s5);
            sp = _mm256_fmadd_pd(sp, r2, s4);
            sp = _mm256_fmadd_pd(sp, r2, s3);
            sp = _mm256_fmadd_pd(sp, r2, s2);
            sp = _mm256_fmadd_pd(sp, r2, s1);
            let sin_r = _mm256_fmadd_pd(_mm256_mul_pd(r, r2), sp, r);

            let mut cp = _mm256_fmadd_pd(c8, r2, c7);
            cp = _mm256_fmadd_pd(cp, r2, c6);
            cp = _mm256_fmadd_pd(cp, r2, c5);
            cp = _mm256_fmadd_pd(cp, r2, c4);
            cp = _mm256_fmadd_pd(cp, r2, c3);
            cp = _mm256_fmadd_pd(cp, r2, c2);
            cp = _mm256_fmadd_pd(cp, r2, c1);
            let cos_r = _mm256_fmadd_pd(r2, cp, one);

            let ki = _mm256_cvtpd_epi32(k);
            let q = _mm_and_si128(ki, _mm_set1_epi32(3));
            let q_64 = _mm256_cvtepi32_epi64(q);

            let is_q1_or_q3 = _mm256_castsi256_pd(_mm256_cmpeq_epi64(
                _mm256_and_si256(q_64, _mm256_set1_epi64x(1)),
                _mm256_set1_epi64x(1),
            ));
            let is_q2_or_q3 = _mm256_castsi256_pd(_mm256_cmpgt_epi64(q_64, _mm256_set1_epi64x(1)));

            let mag = _mm256_blendv_pd(sin_r, cos_r, is_q1_or_q3);
            let neg_mag = _mm256_xor_pd(mag, sign_mask);
            let res = _mm256_blendv_pd(mag, neg_mag, is_q2_or_q3);

            _mm256_storeu_pd(dst.add(i), res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.sin();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cos_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let p1 = _mm256_set1_pd(PIO2_1_F64);
        let p2 = _mm256_set1_pd(PIO2_2_F64);
        let p3 = _mm256_set1_pd(PIO2_3_F64);
        let inv_pio2 = _mm256_set1_pd(INV_PIO2_F64);
        let one = _mm256_set1_pd(1.0);
        let bound = _mm256_set1_pd(1e5);
        let sign_mask = _mm256_set1_pd(-0.0);

        let s8 = _mm256_set1_pd(1.0 / 355687428096000.0);
        let s7 = _mm256_set1_pd(-1.0 / 1307674368000.0);
        let s6 = _mm256_set1_pd(1.0 / 6227020800.0);
        let s5 = _mm256_set1_pd(-1.0 / 39916800.0);
        let s4 = _mm256_set1_pd(1.0 / 362880.0);
        let s3 = _mm256_set1_pd(-1.0 / 5040.0);
        let s2 = _mm256_set1_pd(1.0 / 120.0);
        let s1 = _mm256_set1_pd(-1.0 / 6.0);

        let c8 = _mm256_set1_pd(1.0 / 20922789888000.0);
        let c7 = _mm256_set1_pd(-1.0 / 87178291200.0);
        let c6 = _mm256_set1_pd(1.0 / 479001600.0);
        let c5 = _mm256_set1_pd(-1.0 / 3628800.0);
        let c4 = _mm256_set1_pd(1.0 / 40320.0);
        let c3 = _mm256_set1_pd(-1.0 / 720.0);
        let c2 = _mm256_set1_pd(1.0 / 24.0);
        let c1 = _mm256_set1_pd(-0.5);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let abs_x = _mm256_andnot_pd(sign_mask, x);
            let in_range = _mm256_cmp_pd(abs_x, bound, _CMP_LT_OQ);
            if _mm256_movemask_pd(in_range) != 0x0F {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).cos();
                }
                i += 4;
                continue;
            }

            let k = _mm256_round_pd(
                _mm256_mul_pd(x, inv_pio2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let mut r = _mm256_fnmadd_pd(k, p1, x);
            r = _mm256_fnmadd_pd(k, p2, r);
            r = _mm256_fnmadd_pd(k, p3, r);

            let r2 = _mm256_mul_pd(r, r);

            let mut sp = _mm256_fmadd_pd(s8, r2, s7);
            sp = _mm256_fmadd_pd(sp, r2, s6);
            sp = _mm256_fmadd_pd(sp, r2, s5);
            sp = _mm256_fmadd_pd(sp, r2, s4);
            sp = _mm256_fmadd_pd(sp, r2, s3);
            sp = _mm256_fmadd_pd(sp, r2, s2);
            sp = _mm256_fmadd_pd(sp, r2, s1);
            let sin_r = _mm256_fmadd_pd(_mm256_mul_pd(r, r2), sp, r);

            let mut cp = _mm256_fmadd_pd(c8, r2, c7);
            cp = _mm256_fmadd_pd(cp, r2, c6);
            cp = _mm256_fmadd_pd(cp, r2, c5);
            cp = _mm256_fmadd_pd(cp, r2, c4);
            cp = _mm256_fmadd_pd(cp, r2, c3);
            cp = _mm256_fmadd_pd(cp, r2, c2);
            cp = _mm256_fmadd_pd(cp, r2, c1);
            let cos_r = _mm256_fmadd_pd(r2, cp, one);

            let ki = _mm256_cvtpd_epi32(k);
            let q = _mm_and_si128(_mm_add_epi32(ki, _mm_set1_epi32(1)), _mm_set1_epi32(3));
            let q_64 = _mm256_cvtepi32_epi64(q);

            let is_q1_or_q3 = _mm256_castsi256_pd(_mm256_cmpeq_epi64(
                _mm256_and_si256(q_64, _mm256_set1_epi64x(1)),
                _mm256_set1_epi64x(1),
            ));
            let is_q2_or_q3 = _mm256_castsi256_pd(_mm256_cmpgt_epi64(q_64, _mm256_set1_epi64x(1)));

            let mag = _mm256_blendv_pd(sin_r, cos_r, is_q1_or_q3);
            let neg_mag = _mm256_xor_pd(mag, sign_mask);
            let res = _mm256_blendv_pd(mag, neg_mag, is_q2_or_q3);

            _mm256_storeu_pd(dst.add(i), res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.cos();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tan_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let p1 = _mm256_set1_pd(PIO2_1_F64);
        let p2 = _mm256_set1_pd(PIO2_2_F64);
        let p3 = _mm256_set1_pd(PIO2_3_F64);
        let inv_pio2 = _mm256_set1_pd(INV_PIO2_F64);
        let one = _mm256_set1_pd(1.0);
        let bound = _mm256_set1_pd(1e5);
        let sign_mask = _mm256_set1_pd(-0.0);

        let s8 = _mm256_set1_pd(1.0 / 355687428096000.0);
        let s7 = _mm256_set1_pd(-1.0 / 1307674368000.0);
        let s6 = _mm256_set1_pd(1.0 / 6227020800.0);
        let s5 = _mm256_set1_pd(-1.0 / 39916800.0);
        let s4 = _mm256_set1_pd(1.0 / 362880.0);
        let s3 = _mm256_set1_pd(-1.0 / 5040.0);
        let s2 = _mm256_set1_pd(1.0 / 120.0);
        let s1 = _mm256_set1_pd(-1.0 / 6.0);

        let c8 = _mm256_set1_pd(1.0 / 20922789888000.0);
        let c7 = _mm256_set1_pd(-1.0 / 87178291200.0);
        let c6 = _mm256_set1_pd(1.0 / 479001600.0);
        let c5 = _mm256_set1_pd(-1.0 / 3628800.0);
        let c4 = _mm256_set1_pd(1.0 / 40320.0);
        let c3 = _mm256_set1_pd(-1.0 / 720.0);
        let c2 = _mm256_set1_pd(1.0 / 24.0);
        let c1 = _mm256_set1_pd(-0.5);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));

            let abs_x = _mm256_andnot_pd(sign_mask, x);
            let in_range = _mm256_cmp_pd(abs_x, bound, _CMP_LT_OQ);
            if _mm256_movemask_pd(in_range) != 0x0F {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).tan();
                }
                i += 4;
                continue;
            }

            let k = _mm256_round_pd(
                _mm256_mul_pd(x, inv_pio2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let mut r = _mm256_fnmadd_pd(k, p1, x);
            r = _mm256_fnmadd_pd(k, p2, r);
            r = _mm256_fnmadd_pd(k, p3, r);

            let r2 = _mm256_mul_pd(r, r);

            let mut sp = _mm256_fmadd_pd(s8, r2, s7);
            sp = _mm256_fmadd_pd(sp, r2, s6);
            sp = _mm256_fmadd_pd(sp, r2, s5);
            sp = _mm256_fmadd_pd(sp, r2, s4);
            sp = _mm256_fmadd_pd(sp, r2, s3);
            sp = _mm256_fmadd_pd(sp, r2, s2);
            sp = _mm256_fmadd_pd(sp, r2, s1);
            let sin_r = _mm256_fmadd_pd(_mm256_mul_pd(r, r2), sp, r);

            let mut cp = _mm256_fmadd_pd(c8, r2, c7);
            cp = _mm256_fmadd_pd(cp, r2, c6);
            cp = _mm256_fmadd_pd(cp, r2, c5);
            cp = _mm256_fmadd_pd(cp, r2, c4);
            cp = _mm256_fmadd_pd(cp, r2, c3);
            cp = _mm256_fmadd_pd(cp, r2, c2);
            cp = _mm256_fmadd_pd(cp, r2, c1);
            let cos_r = _mm256_fmadd_pd(r2, cp, one);

            let ki = _mm256_cvtpd_epi32(k);
            let is_odd = _mm256_cvtepi32_epi64(_mm_and_si128(ki, _mm_set1_epi32(1)));
            let odd_mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(is_odd, _mm256_set1_epi64x(1)));

            let num = _mm256_blendv_pd(sin_r, _mm256_xor_pd(cos_r, sign_mask), odd_mask);
            let den = _mm256_blendv_pd(cos_r, sin_r, odd_mask);
            let res = _mm256_div_pd(num, den);

            _mm256_storeu_pd(dst.add(i), res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.tan();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_pd();
        let sign_mask = _mm256_set1_pd(-0.0);
        let twenty = _mm256_set1_pd(20.0);
        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let nan = _mm256_set1_pd(f64::NAN);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));
            let sign = _mm256_and_pd(x, sign_mask);
            let abs_x = _mm256_andnot_pd(sign_mask, x);
            let y = _mm256_add_pd(abs_x, abs_x);

            let mut em1 = [0.0; 4];
            let mut y_arr = [0.0; 4];
            _mm256_storeu_pd(y_arr.as_mut_ptr(), y);
            expm1_f64_avx2(y_arr.as_ptr(), em1.as_mut_ptr(), 4);
            let em1_vec = _mm256_loadu_pd(em1.as_ptr());

            let th = _mm256_div_pd(em1_vec, _mm256_add_pd(em1_vec, two));
            let signed_th = _mm256_xor_pd(th, sign);

            let large_mask = _mm256_cmp_pd(abs_x, twenty, _CMP_GE_OQ);
            let sat = _mm256_xor_pd(one, sign);
            let res = _mm256_blendv_pd(signed_th, sat, large_mask);

            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
            let is_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
            let mut final_res = _mm256_blendv_pd(res, nan, is_nan);
            final_res = _mm256_blendv_pd(final_res, x, is_zero);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.tanh();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_f64_avx2(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_pd();
        let sign_mask = _mm256_set1_pd(-0.0);
        let forty = _mm256_set1_pd(40.0);
        let min_exp = _mm256_set1_pd(-745.0);
        let one = _mm256_set1_pd(1.0);
        let nan = _mm256_set1_pd(f64::NAN);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));
            let abs_x = _mm256_andnot_pd(sign_mask, x);
            let neg_abs_x = _mm256_xor_pd(abs_x, sign_mask);

            let mut neg_abs = [0.0; 4];
            let mut exp_vals = [0.0; 4];
            _mm256_storeu_pd(neg_abs.as_mut_ptr(), neg_abs_x);
            exp_f64_avx2(neg_abs.as_ptr(), exp_vals.as_mut_ptr(), 4);
            let e = _mm256_loadu_pd(exp_vals.as_ptr());

            let denom = _mm256_add_pd(one, e);
            let s = _mm256_div_pd(one, denom);
            let s_neg = _mm256_mul_pd(e, s);

            let is_positive = _mm256_cmp_pd(x, zero, _CMP_GE_OQ);
            let res = _mm256_blendv_pd(s_neg, s, is_positive);

            let is_huge_pos = _mm256_cmp_pd(x, forty, _CMP_GT_OQ);
            let is_huge_neg = _mm256_cmp_pd(x, min_exp, _CMP_LT_OQ);
            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_pd(res, one, is_huge_pos);
            final_res = _mm256_blendv_pd(final_res, zero, is_huge_neg);
            final_res = _mm256_blendv_pd(final_res, nan, is_nan);

            _mm256_storeu_pd(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = scalar_sigmoid_f64(x);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2+FMA Implementation Kernels (f32)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let inv_ln2 = _mm256_set1_ps(INV_LN2_F32);
        let ln2_hi = _mm256_set1_ps(LN2_HI_F32);
        let ln2_lo = _mm256_set1_ps(LN2_LO_F32);
        let one = _mm256_set1_ps(1.0);
        let max_exp = _mm256_set1_ps(88.722_84);
        let min_exp = _mm256_set1_ps(-103.27893);
        let inf = _mm256_set1_ps(f32::INFINITY);
        let zero = _mm256_setzero_ps();

        let c6 = _mm256_set1_ps(1.0 / 720.0);
        let c5 = _mm256_set1_ps(1.0 / 120.0);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c3 = _mm256_set1_ps(1.0 / 6.0);
        let c2 = _mm256_set1_ps(0.5);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let k = _mm256_round_ps(
                _mm256_mul_ps(x, inv_ln2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let r1 = _mm256_fnmadd_ps(k, ln2_hi, x);
            let r = _mm256_fnmadd_ps(k, ln2_lo, r1);

            let mut p = _mm256_fmadd_ps(c6, r, c5);
            p = _mm256_fmadd_ps(p, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, one);
            let poly = _mm256_fmadd_ps(p, r, one);

            let ki = _mm256_cvtps_epi32(k);
            let k1 = _mm256_srai_epi32(ki, 1);
            let k2 = _mm256_sub_epi32(ki, k1);

            let bias = _mm256_set1_epi32(127);
            let k1_bias = _mm256_add_epi32(k1, bias);
            let k2_bias = _mm256_add_epi32(k2, bias);

            let scale1 = _mm256_castsi256_ps(_mm256_slli_epi32(k1_bias, 23));
            let scale2 = _mm256_castsi256_ps(_mm256_slli_epi32(k2_bias, 23));

            let res = _mm256_mul_ps(_mm256_mul_ps(poly, scale1), scale2);

            let overflow_mask = _mm256_cmp_ps(x, max_exp, _CMP_GT_OQ);
            let underflow_mask = _mm256_cmp_ps(x, min_exp, _CMP_LT_OQ);

            let clamped = _mm256_blendv_ps(res, inf, overflow_mask);
            let final_res = _mm256_blendv_ps(clamped, zero, underflow_mask);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp2_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        let max_exp = _mm256_set1_ps(128.0);
        let min_exp = _mm256_set1_ps(-149.0);
        let inf = _mm256_set1_ps(f32::INFINITY);
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);

        let c6 = _mm256_set1_ps(1.0 / 720.0);
        let c5 = _mm256_set1_ps(1.0 / 120.0);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c3 = _mm256_set1_ps(1.0 / 6.0);
        let c2 = _mm256_set1_ps(0.5);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let k = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let rem = _mm256_sub_ps(x, k);
            let r = _mm256_mul_ps(rem, ln2);

            let mut p = _mm256_fmadd_ps(c6, r, c5);
            p = _mm256_fmadd_ps(p, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, one);
            let poly = _mm256_fmadd_ps(p, r, one);

            let ki = _mm256_cvtps_epi32(k);
            let k1 = _mm256_srai_epi32(ki, 1);
            let k2 = _mm256_sub_epi32(ki, k1);

            let bias = _mm256_set1_epi32(127);
            let k1_bias = _mm256_add_epi32(k1, bias);
            let k2_bias = _mm256_add_epi32(k2, bias);

            let scale1 = _mm256_castsi256_ps(_mm256_slli_epi32(k1_bias, 23));
            let scale2 = _mm256_castsi256_ps(_mm256_slli_epi32(k2_bias, 23));

            let res = _mm256_mul_ps(_mm256_mul_ps(poly, scale1), scale2);

            let overflow_mask = _mm256_cmp_ps(x, max_exp, _CMP_GT_OQ);
            let underflow_mask = _mm256_cmp_ps(x, min_exp, _CMP_LT_OQ);

            let clamped = _mm256_blendv_ps(res, inf, overflow_mask);
            let final_res = _mm256_blendv_ps(clamped, zero, underflow_mask);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp2();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn expm1_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let inv_ln2 = _mm256_set1_ps(INV_LN2_F32);
        let ln2_hi = _mm256_set1_ps(LN2_HI_F32);
        let ln2_lo = _mm256_set1_ps(LN2_LO_F32);
        let one = _mm256_set1_ps(1.0);
        let neg_one = _mm256_set1_ps(-1.0);
        let max_exp = _mm256_set1_ps(88.722_84);
        let min_expm1 = _mm256_set1_ps(-25.0);
        let inf = _mm256_set1_ps(f32::INFINITY);
        let zero = _mm256_setzero_ps();
        let _ = zero;
        let nan = _mm256_set1_ps(f32::NAN);

        let c9 = _mm256_set1_ps(1.0 / 362880.0);
        let c8 = _mm256_set1_ps(1.0 / 40320.0);
        let c7 = _mm256_set1_ps(1.0 / 5040.0);
        let c6 = _mm256_set1_ps(1.0 / 720.0);
        let c5 = _mm256_set1_ps(1.0 / 120.0);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c3 = _mm256_set1_ps(1.0 / 6.0);
        let c2 = _mm256_set1_ps(0.5);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let k = _mm256_round_ps(
                _mm256_mul_ps(x, inv_ln2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let r1 = _mm256_fnmadd_ps(k, ln2_hi, x);
            let r = _mm256_fnmadd_ps(k, ln2_lo, r1);

            let mut p = _mm256_fmadd_ps(c9, r, c8);
            p = _mm256_fmadd_ps(p, r, c7);
            p = _mm256_fmadd_ps(p, r, c6);
            p = _mm256_fmadd_ps(p, r, c5);
            p = _mm256_fmadd_ps(p, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            let poly = _mm256_fmadd_ps(_mm256_mul_ps(r, r), p, r);

            let ki = _mm256_cvtps_epi32(k);
            let k1 = _mm256_srai_epi32(ki, 1);
            let k2 = _mm256_sub_epi32(ki, k1);

            let bias = _mm256_set1_epi32(127);
            let k1_bias = _mm256_add_epi32(k1, bias);
            let k2_bias = _mm256_add_epi32(k2, bias);

            let scale1 = _mm256_castsi256_ps(_mm256_slli_epi32(k1_bias, 23));
            let scale2 = _mm256_castsi256_ps(_mm256_slli_epi32(k2_bias, 23));
            let twok = _mm256_mul_ps(scale1, scale2);
            let twok_minus_one = _mm256_sub_ps(twok, one);
            let res_normal = _mm256_fmadd_ps(twok, poly, twok_minus_one);

            let poly_exp = _mm256_add_ps(one, poly);
            let res_large = _mm256_mul_ps(_mm256_mul_ps(poly_exp, scale1), scale2);
            let is_large_k = _mm256_cmp_ps(k, _mm256_set1_ps(24.5), _CMP_GT_OQ);
            let res_k_nonzero = _mm256_blendv_ps(res_normal, res_large, is_large_k);

            let mut p_dir = _mm256_fmadd_ps(c9, x, c8);
            p_dir = _mm256_fmadd_ps(p_dir, x, c7);
            p_dir = _mm256_fmadd_ps(p_dir, x, c6);
            p_dir = _mm256_fmadd_ps(p_dir, x, c5);
            p_dir = _mm256_fmadd_ps(p_dir, x, c4);
            p_dir = _mm256_fmadd_ps(p_dir, x, c3);
            p_dir = _mm256_fmadd_ps(p_dir, x, c2);
            let poly_direct = _mm256_fmadd_ps(_mm256_mul_ps(x, x), p_dir, x);

            let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
            let is_small = _mm256_cmp_ps(abs_x, _mm256_set1_ps(0.6931472), _CMP_LT_OQ);
            let res = _mm256_blendv_ps(res_k_nonzero, poly_direct, is_small);

            let overflow_mask = _mm256_cmp_ps(x, max_exp, _CMP_GT_OQ);
            let underflow_mask = _mm256_cmp_ps(x, min_expm1, _CMP_LT_OQ);
            let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_ps(res, inf, overflow_mask);
            final_res = _mm256_blendv_ps(final_res, neg_one, underflow_mask);
            final_res = _mm256_blendv_ps(final_res, nan, is_nan);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp_m1();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_ps();
        let ln2_hi = _mm256_set1_ps(LN2_HI_F32);
        let ln2_lo = _mm256_set1_ps(LN2_LO_F32);
        let sqrt2_half = _mm256_set1_ps(std::f32::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
        let pos_inf = _mm256_set1_ps(f32::INFINITY);
        let nan = _mm256_set1_ps(f32::NAN);
        let min_normal = _mm256_set1_ps(1.17549435e-38);

        let c4 = _mm256_set1_ps(2.0 / 9.0);
        let c3 = _mm256_set1_ps(2.0 / 7.0);
        let c2 = _mm256_set1_ps(2.0 / 5.0);
        let c1 = _mm256_set1_ps(2.0 / 3.0);
        let c0 = _mm256_set1_ps(2.0);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let is_subnormal = _mm256_and_ps(
                _mm256_cmp_ps(x, min_normal, _CMP_LT_OQ),
                _mm256_cmp_ps(x, zero, _CMP_GT_OQ),
            );
            if _mm256_movemask_ps(is_subnormal) != 0 {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).ln();
                }
                i += 8;
                continue;
            }

            let x_bits = _mm256_castps_si256(x);
            let exp_bits = _mm256_srli_epi32(x_bits, 23);
            let mut k = _mm256_cvtepi32_ps(_mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127)));

            let mant_mask = _mm256_set1_epi32(0x007F_FFFF);
            let exp_127 = _mm256_set1_epi32(0x3F80_0000);
            let mant_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mant_mask), exp_127);
            let mut m = _mm256_mul_ps(_mm256_castsi256_ps(mant_bits), _mm256_set1_ps(0.5));
            k = _mm256_add_ps(k, one);

            let lt_mask = _mm256_cmp_ps(m, sqrt2_half, _CMP_LT_OQ);
            m = _mm256_blendv_ps(m, _mm256_mul_ps(m, two), lt_mask);
            k = _mm256_blendv_ps(k, _mm256_sub_ps(k, one), lt_mask);

            let u = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
            let u2 = _mm256_mul_ps(u, u);

            let mut p = _mm256_fmadd_ps(c4, u2, c3);
            p = _mm256_fmadd_ps(p, u2, c2);
            p = _mm256_fmadd_ps(p, u2, c1);
            p = _mm256_fmadd_ps(p, u2, c0);
            let ln_m = _mm256_mul_ps(u, p);

            let res = _mm256_fmadd_ps(k, ln2_hi, _mm256_fmadd_ps(k, ln2_lo, ln_m));

            let eq_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
            let lt_zero = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            let is_inf = _mm256_cmp_ps(x, pos_inf, _CMP_EQ_OQ);
            let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_ps(res, neg_inf, eq_zero);
            final_res = _mm256_blendv_ps(final_res, nan, lt_zero);
            final_res = _mm256_blendv_ps(final_res, pos_inf, is_inf);
            final_res = _mm256_blendv_ps(final_res, nan, is_nan);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.ln();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log2_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_ps();
        let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let sqrt2_half = _mm256_set1_ps(std::f32::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
        let pos_inf = _mm256_set1_ps(f32::INFINITY);
        let nan = _mm256_set1_ps(f32::NAN);
        let min_normal = _mm256_set1_ps(1.17549435e-38);

        let c4 = _mm256_set1_ps(2.0 / 9.0);
        let c3 = _mm256_set1_ps(2.0 / 7.0);
        let c2 = _mm256_set1_ps(2.0 / 5.0);
        let c1 = _mm256_set1_ps(2.0 / 3.0);
        let c0 = _mm256_set1_ps(2.0);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let is_subnormal = _mm256_and_ps(
                _mm256_cmp_ps(x, min_normal, _CMP_LT_OQ),
                _mm256_cmp_ps(x, zero, _CMP_GT_OQ),
            );
            if _mm256_movemask_ps(is_subnormal) != 0 {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).log2();
                }
                i += 8;
                continue;
            }

            let x_bits = _mm256_castps_si256(x);
            let exp_bits = _mm256_srli_epi32(x_bits, 23);
            let mut k = _mm256_cvtepi32_ps(_mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127)));

            let mant_mask = _mm256_set1_epi32(0x007F_FFFF);
            let exp_127 = _mm256_set1_epi32(0x3F80_0000);
            let mant_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mant_mask), exp_127);
            let mut m = _mm256_mul_ps(_mm256_castsi256_ps(mant_bits), _mm256_set1_ps(0.5));
            k = _mm256_add_ps(k, one);

            let lt_mask = _mm256_cmp_ps(m, sqrt2_half, _CMP_LT_OQ);
            m = _mm256_blendv_ps(m, _mm256_mul_ps(m, two), lt_mask);
            k = _mm256_blendv_ps(k, _mm256_sub_ps(k, one), lt_mask);

            let u = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
            let u2 = _mm256_mul_ps(u, u);

            let mut p = _mm256_fmadd_ps(c4, u2, c3);
            p = _mm256_fmadd_ps(p, u2, c2);
            p = _mm256_fmadd_ps(p, u2, c1);
            p = _mm256_fmadd_ps(p, u2, c0);
            let ln_m = _mm256_mul_ps(u, p);

            let res = _mm256_fmadd_ps(ln_m, log2_e, k);

            let eq_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
            let lt_zero = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            let is_inf = _mm256_cmp_ps(x, pos_inf, _CMP_EQ_OQ);
            let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_ps(res, neg_inf, eq_zero);
            final_res = _mm256_blendv_ps(final_res, nan, lt_zero);
            final_res = _mm256_blendv_ps(final_res, pos_inf, is_inf);
            final_res = _mm256_blendv_ps(final_res, nan, is_nan);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.log2();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log10_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_ps();
        let log10_2_hi = _mm256_set1_ps(LOG10_2_HI_F32);
        let log10_2_lo = _mm256_set1_ps(LOG10_2_LO_F32);
        let log10_e = _mm256_set1_ps(std::f32::consts::LOG10_E);
        let sqrt2_half = _mm256_set1_ps(std::f32::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
        let pos_inf = _mm256_set1_ps(f32::INFINITY);
        let nan = _mm256_set1_ps(f32::NAN);
        let min_normal = _mm256_set1_ps(1.17549435e-38);

        let c4 = _mm256_set1_ps(2.0 / 9.0);
        let c3 = _mm256_set1_ps(2.0 / 7.0);
        let c2 = _mm256_set1_ps(2.0 / 5.0);
        let c1 = _mm256_set1_ps(2.0 / 3.0);
        let c0 = _mm256_set1_ps(2.0);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let is_subnormal = _mm256_and_ps(
                _mm256_cmp_ps(x, min_normal, _CMP_LT_OQ),
                _mm256_cmp_ps(x, zero, _CMP_GT_OQ),
            );
            if _mm256_movemask_ps(is_subnormal) != 0 {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).log10();
                }
                i += 8;
                continue;
            }

            let x_bits = _mm256_castps_si256(x);
            let exp_bits = _mm256_srli_epi32(x_bits, 23);
            let mut k = _mm256_cvtepi32_ps(_mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127)));

            let mant_mask = _mm256_set1_epi32(0x007F_FFFF);
            let exp_127 = _mm256_set1_epi32(0x3F80_0000);
            let mant_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mant_mask), exp_127);
            let mut m = _mm256_mul_ps(_mm256_castsi256_ps(mant_bits), _mm256_set1_ps(0.5));
            k = _mm256_add_ps(k, one);

            let lt_mask = _mm256_cmp_ps(m, sqrt2_half, _CMP_LT_OQ);
            m = _mm256_blendv_ps(m, _mm256_mul_ps(m, two), lt_mask);
            k = _mm256_blendv_ps(k, _mm256_sub_ps(k, one), lt_mask);

            let u = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
            let u2 = _mm256_mul_ps(u, u);

            let mut p = _mm256_fmadd_ps(c4, u2, c3);
            p = _mm256_fmadd_ps(p, u2, c2);
            p = _mm256_fmadd_ps(p, u2, c1);
            p = _mm256_fmadd_ps(p, u2, c0);
            let ln_m = _mm256_mul_ps(u, p);

            let res = _mm256_fmadd_ps(
                k,
                log10_2_hi,
                _mm256_fmadd_ps(ln_m, log10_e, _mm256_mul_ps(k, log10_2_lo)),
            );

            let eq_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
            let lt_zero = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            let is_inf = _mm256_cmp_ps(x, pos_inf, _CMP_EQ_OQ);
            let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_ps(res, neg_inf, eq_zero);
            final_res = _mm256_blendv_ps(final_res, nan, lt_zero);
            final_res = _mm256_blendv_ps(final_res, pos_inf, is_inf);
            final_res = _mm256_blendv_ps(final_res, nan, is_nan);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.log10();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log1p_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_ps();
        let ln2_hi = _mm256_set1_ps(LN2_HI_F32);
        let ln2_lo = _mm256_set1_ps(LN2_LO_F32);
        let sqrt2_half = _mm256_set1_ps(std::f32::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_ps(1.0);
        let neg_one = _mm256_set1_ps(-1.0);
        let two = _mm256_set1_ps(2.0);
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
        let pos_inf = _mm256_set1_ps(f32::INFINITY);
        let nan = _mm256_set1_ps(f32::NAN);
        let threshold = _mm256_set1_ps(0.35);

        let c4 = _mm256_set1_ps(2.0 / 9.0);
        let c3 = _mm256_set1_ps(2.0 / 7.0);
        let c2 = _mm256_set1_ps(2.0 / 5.0);
        let c1 = _mm256_set1_ps(2.0 / 3.0);
        let c0 = _mm256_set1_ps(2.0);

        let sign_mask = _mm256_set1_ps(-0.0);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));
            let abs_x = _mm256_andnot_ps(sign_mask, x);
            let is_small = _mm256_cmp_ps(abs_x, threshold, _CMP_LT_OQ);

            // Small branch: u = x / (2 + x)
            let u_s = _mm256_div_ps(x, _mm256_add_ps(two, x));
            let u2_s = _mm256_mul_ps(u_s, u_s);
            let mut p_s = _mm256_fmadd_ps(c4, u2_s, c3);
            p_s = _mm256_fmadd_ps(p_s, u2_s, c2);
            p_s = _mm256_fmadd_ps(p_s, u2_s, c1);
            p_s = _mm256_fmadd_ps(p_s, u2_s, c0);
            let res_small = _mm256_mul_ps(u_s, p_s);

            // Large branch: log(1 + x)
            let y = _mm256_add_ps(one, x);
            let y_bits = _mm256_castps_si256(y);
            let exp_bits = _mm256_srli_epi32(y_bits, 23);
            let mut k = _mm256_cvtepi32_ps(_mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127)));

            let mant_mask = _mm256_set1_epi32(0x007F_FFFF);
            let exp_127 = _mm256_set1_epi32(0x3F80_0000);
            let mant_bits = _mm256_or_si256(_mm256_and_si256(y_bits, mant_mask), exp_127);
            let mut m = _mm256_mul_ps(_mm256_castsi256_ps(mant_bits), _mm256_set1_ps(0.5));
            k = _mm256_add_ps(k, one);

            let lt_mask = _mm256_cmp_ps(m, sqrt2_half, _CMP_LT_OQ);
            m = _mm256_blendv_ps(m, _mm256_mul_ps(m, two), lt_mask);
            k = _mm256_blendv_ps(k, _mm256_sub_ps(k, one), lt_mask);

            let u_l = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
            let u2_l = _mm256_mul_ps(u_l, u_l);

            let mut p_l = _mm256_fmadd_ps(c4, u2_l, c3);
            p_l = _mm256_fmadd_ps(p_l, u2_l, c2);
            p_l = _mm256_fmadd_ps(p_l, u2_l, c1);
            p_l = _mm256_fmadd_ps(p_l, u2_l, c0);
            let ln_m = _mm256_mul_ps(u_l, p_l);
            let res_large = _mm256_fmadd_ps(k, ln2_hi, _mm256_fmadd_ps(k, ln2_lo, ln_m));

            let res = _mm256_blendv_ps(res_large, res_small, is_small);

            let eq_neg_one = _mm256_cmp_ps(x, neg_one, _CMP_EQ_OQ);
            let lt_neg_one = _mm256_cmp_ps(x, neg_one, _CMP_LT_OQ);
            let is_inf = _mm256_cmp_ps(x, pos_inf, _CMP_EQ_OQ);
            let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
            let is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);

            let mut final_res = _mm256_blendv_ps(res, neg_inf, eq_neg_one);
            final_res = _mm256_blendv_ps(final_res, nan, lt_neg_one);
            final_res = _mm256_blendv_ps(final_res, pos_inf, is_inf);
            final_res = _mm256_blendv_ps(final_res, nan, is_nan);
            final_res = _mm256_blendv_ps(final_res, x, is_zero);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.ln_1p();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sin_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let p1 = _mm256_set1_ps(PIO2_1_F32);
        let p2 = _mm256_set1_ps(PIO2_2_F32);
        let p3 = _mm256_set1_ps(PIO2_3_F32);
        let p4 = _mm256_set1_ps(PIO2_4_F32);
        let inv_pio2 = _mm256_set1_ps(INV_PIO2_F32);
        let one = _mm256_set1_ps(1.0);
        let bound = _mm256_set1_ps(1e5);
        let sign_mask = _mm256_set1_ps(-0.0);

        let s4 = _mm256_set1_ps(1.0 / 362880.0);
        let s3 = _mm256_set1_ps(-1.0 / 5040.0);
        let s2 = _mm256_set1_ps(1.0 / 120.0);
        let s1 = _mm256_set1_ps(-1.0 / 6.0);

        let c5 = _mm256_set1_ps(-1.0 / 3628800.0);
        let c4 = _mm256_set1_ps(1.0 / 40320.0);
        let c3 = _mm256_set1_ps(-1.0 / 720.0);
        let c2 = _mm256_set1_ps(1.0 / 24.0);
        let c1 = _mm256_set1_ps(-0.5);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let abs_x = _mm256_andnot_ps(sign_mask, x);
            let in_range = _mm256_cmp_ps(abs_x, bound, _CMP_LT_OQ);
            if _mm256_movemask_ps(in_range) != 0xFF {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).sin();
                }
                i += 8;
                continue;
            }

            let k = _mm256_round_ps(
                _mm256_mul_ps(x, inv_pio2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let mut r = _mm256_fnmadd_ps(k, p1, x);
            r = _mm256_fnmadd_ps(k, p2, r);
            r = _mm256_fnmadd_ps(k, p3, r);
            r = _mm256_fnmadd_ps(k, p4, r);

            let r_bound = _mm256_set1_ps(1e-3);
            let abs_r = _mm256_andnot_ps(sign_mask, r);
            let r_ok = _mm256_cmp_ps(abs_r, r_bound, _CMP_GE_OQ);
            if _mm256_movemask_ps(r_ok) != 0xFF {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).sin();
                }
                i += 8;
                continue;
            }

            let r2 = _mm256_mul_ps(r, r);

            let mut sp = _mm256_fmadd_ps(s4, r2, s3);
            sp = _mm256_fmadd_ps(sp, r2, s2);
            sp = _mm256_fmadd_ps(sp, r2, s1);
            let sin_r = _mm256_fmadd_ps(_mm256_mul_ps(r, r2), sp, r);

            let mut cp = _mm256_fmadd_ps(c5, r2, c4);
            cp = _mm256_fmadd_ps(cp, r2, c3);
            cp = _mm256_fmadd_ps(cp, r2, c2);
            cp = _mm256_fmadd_ps(cp, r2, c1);
            let cos_r = _mm256_fmadd_ps(r2, cp, one);

            let ki = _mm256_cvtps_epi32(k);
            let q = _mm256_and_si256(ki, _mm256_set1_epi32(3));

            let is_q1_or_q3 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                _mm256_and_si256(q, _mm256_set1_epi32(1)),
                _mm256_set1_epi32(1),
            ));
            let is_q2_or_q3 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(q, _mm256_set1_epi32(1)));

            let mag = _mm256_blendv_ps(sin_r, cos_r, is_q1_or_q3);
            let neg_mag = _mm256_xor_ps(mag, sign_mask);
            let res = _mm256_blendv_ps(mag, neg_mag, is_q2_or_q3);

            _mm256_storeu_ps(dst.add(i), res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.sin();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cos_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let p1 = _mm256_set1_ps(PIO2_1_F32);
        let p2 = _mm256_set1_ps(PIO2_2_F32);
        let p3 = _mm256_set1_ps(PIO2_3_F32);
        let p4 = _mm256_set1_ps(PIO2_4_F32);
        let inv_pio2 = _mm256_set1_ps(INV_PIO2_F32);
        let one = _mm256_set1_ps(1.0);
        let bound = _mm256_set1_ps(1e5);
        let sign_mask = _mm256_set1_ps(-0.0);

        let s4 = _mm256_set1_ps(1.0 / 362880.0);
        let s3 = _mm256_set1_ps(-1.0 / 5040.0);
        let s2 = _mm256_set1_ps(1.0 / 120.0);
        let s1 = _mm256_set1_ps(-1.0 / 6.0);

        let c5 = _mm256_set1_ps(-1.0 / 3628800.0);
        let c4 = _mm256_set1_ps(1.0 / 40320.0);
        let c3 = _mm256_set1_ps(-1.0 / 720.0);
        let c2 = _mm256_set1_ps(1.0 / 24.0);
        let c1 = _mm256_set1_ps(-0.5);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let abs_x = _mm256_andnot_ps(sign_mask, x);
            let in_range = _mm256_cmp_ps(abs_x, bound, _CMP_LT_OQ);
            if _mm256_movemask_ps(in_range) != 0xFF {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).cos();
                }
                i += 8;
                continue;
            }

            let k = _mm256_round_ps(
                _mm256_mul_ps(x, inv_pio2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let mut r = _mm256_fnmadd_ps(k, p1, x);
            r = _mm256_fnmadd_ps(k, p2, r);
            r = _mm256_fnmadd_ps(k, p3, r);
            r = _mm256_fnmadd_ps(k, p4, r);

            let r_bound = _mm256_set1_ps(1e-3);
            let abs_r = _mm256_andnot_ps(sign_mask, r);
            let r_ok = _mm256_cmp_ps(abs_r, r_bound, _CMP_GE_OQ);
            if _mm256_movemask_ps(r_ok) != 0xFF {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).cos();
                }
                i += 8;
                continue;
            }

            let r2 = _mm256_mul_ps(r, r);

            let mut sp = _mm256_fmadd_ps(s4, r2, s3);
            sp = _mm256_fmadd_ps(sp, r2, s2);
            sp = _mm256_fmadd_ps(sp, r2, s1);
            let sin_r = _mm256_fmadd_ps(_mm256_mul_ps(r, r2), sp, r);

            let mut cp = _mm256_fmadd_ps(c5, r2, c4);
            cp = _mm256_fmadd_ps(cp, r2, c3);
            cp = _mm256_fmadd_ps(cp, r2, c2);
            cp = _mm256_fmadd_ps(cp, r2, c1);
            let cos_r = _mm256_fmadd_ps(r2, cp, one);

            let ki = _mm256_cvtps_epi32(k);
            let q = _mm256_and_si256(
                _mm256_add_epi32(ki, _mm256_set1_epi32(1)),
                _mm256_set1_epi32(3),
            );

            let is_q1_or_q3 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                _mm256_and_si256(q, _mm256_set1_epi32(1)),
                _mm256_set1_epi32(1),
            ));
            let is_q2_or_q3 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(q, _mm256_set1_epi32(1)));

            let mag = _mm256_blendv_ps(sin_r, cos_r, is_q1_or_q3);
            let neg_mag = _mm256_xor_ps(mag, sign_mask);
            let res = _mm256_blendv_ps(mag, neg_mag, is_q2_or_q3);

            _mm256_storeu_ps(dst.add(i), res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.cos();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tan_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let p1 = _mm256_set1_ps(PIO2_1_F32);
        let p2 = _mm256_set1_ps(PIO2_2_F32);
        let p3 = _mm256_set1_ps(PIO2_3_F32);
        let p4 = _mm256_set1_ps(PIO2_4_F32);
        let inv_pio2 = _mm256_set1_ps(INV_PIO2_F32);
        let one = _mm256_set1_ps(1.0);
        let bound = _mm256_set1_ps(1e5);
        let sign_mask = _mm256_set1_ps(-0.0);

        let s4 = _mm256_set1_ps(1.0 / 362880.0);
        let s3 = _mm256_set1_ps(-1.0 / 5040.0);
        let s2 = _mm256_set1_ps(1.0 / 120.0);
        let s1 = _mm256_set1_ps(-1.0 / 6.0);

        let c5 = _mm256_set1_ps(-1.0 / 3628800.0);
        let c4 = _mm256_set1_ps(1.0 / 40320.0);
        let c3 = _mm256_set1_ps(-1.0 / 720.0);
        let c2 = _mm256_set1_ps(1.0 / 24.0);
        let c1 = _mm256_set1_ps(-0.5);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));

            let abs_x = _mm256_andnot_ps(sign_mask, x);
            let in_range = _mm256_cmp_ps(abs_x, bound, _CMP_LT_OQ);
            if _mm256_movemask_ps(in_range) != 0xFF {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).tan();
                }
                i += 8;
                continue;
            }

            let k = _mm256_round_ps(
                _mm256_mul_ps(x, inv_pio2),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );

            let mut r = _mm256_fnmadd_ps(k, p1, x);
            r = _mm256_fnmadd_ps(k, p2, r);
            r = _mm256_fnmadd_ps(k, p3, r);
            r = _mm256_fnmadd_ps(k, p4, r);

            let r_bound = _mm256_set1_ps(1e-3);
            let abs_r = _mm256_andnot_ps(sign_mask, r);
            let r_ok = _mm256_cmp_ps(abs_r, r_bound, _CMP_GE_OQ);
            if _mm256_movemask_ps(r_ok) != 0xFF {
                for j in 0..8 {
                    *dst.add(i + j) = (*src.add(i + j)).tan();
                }
                i += 8;
                continue;
            }

            let r2 = _mm256_mul_ps(r, r);

            let mut sp = _mm256_fmadd_ps(s4, r2, s3);
            sp = _mm256_fmadd_ps(sp, r2, s2);
            sp = _mm256_fmadd_ps(sp, r2, s1);
            let sin_r = _mm256_fmadd_ps(_mm256_mul_ps(r, r2), sp, r);

            let mut cp = _mm256_fmadd_ps(c5, r2, c4);
            cp = _mm256_fmadd_ps(cp, r2, c3);
            cp = _mm256_fmadd_ps(cp, r2, c2);
            cp = _mm256_fmadd_ps(cp, r2, c1);
            let cos_r = _mm256_fmadd_ps(r2, cp, one);

            let ki = _mm256_cvtps_epi32(k);
            let is_odd = _mm256_and_si256(ki, _mm256_set1_epi32(1));
            let odd_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(is_odd, _mm256_set1_epi32(1)));

            let num = _mm256_blendv_ps(sin_r, _mm256_xor_ps(cos_r, sign_mask), odd_mask);
            let den = _mm256_blendv_ps(cos_r, sin_r, odd_mask);
            let res = _mm256_div_ps(num, den);

            _mm256_storeu_ps(dst.add(i), res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.tan();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0);
        let ten = _mm256_set1_ps(10.0);
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);
        let nan = _mm256_set1_ps(f32::NAN);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));
            let sign = _mm256_and_ps(x, sign_mask);
            let abs_x = _mm256_andnot_ps(sign_mask, x);
            let y = _mm256_add_ps(abs_x, abs_x);

            let mut em1 = [0.0f32; 8];
            let mut y_arr = [0.0f32; 8];
            _mm256_storeu_ps(y_arr.as_mut_ptr(), y);
            expm1_f32_avx2(y_arr.as_ptr(), em1.as_mut_ptr(), 8);
            let em1_vec = _mm256_loadu_ps(em1.as_ptr());

            let th = _mm256_div_ps(em1_vec, _mm256_add_ps(em1_vec, two));
            let signed_th = _mm256_xor_ps(th, sign);

            let large_mask = _mm256_cmp_ps(abs_x, ten, _CMP_GE_OQ);
            let sat = _mm256_xor_ps(one, sign);
            let res = _mm256_blendv_ps(signed_th, sat, large_mask);

            let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
            let is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
            let mut final_res = _mm256_blendv_ps(res, nan, is_nan);
            final_res = _mm256_blendv_ps(final_res, x, is_zero);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.tanh();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_f32_avx2(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller verified AVX2/FMA CPU support and valid pointer access of length len.
    unsafe {
        let zero = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0);
        let twenty = _mm256_set1_ps(20.0);
        let min_exp = _mm256_set1_ps(-88.0);
        let one = _mm256_set1_ps(1.0);
        let nan = _mm256_set1_ps(f32::NAN);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(src.add(i));
            let abs_x = _mm256_andnot_ps(sign_mask, x);
            let neg_abs_x = _mm256_xor_ps(abs_x, sign_mask);

            let mut neg_abs = [0.0f32; 8];
            let mut exp_vals = [0.0f32; 8];
            _mm256_storeu_ps(neg_abs.as_mut_ptr(), neg_abs_x);
            exp_f32_avx2(neg_abs.as_ptr(), exp_vals.as_mut_ptr(), 8);
            let e = _mm256_loadu_ps(exp_vals.as_ptr());

            let denom = _mm256_add_ps(one, e);
            let s = _mm256_div_ps(one, denom);
            let s_neg = _mm256_mul_ps(e, s);

            let is_positive = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
            let res = _mm256_blendv_ps(s_neg, s, is_positive);

            let is_huge_pos = _mm256_cmp_ps(x, twenty, _CMP_GT_OQ);
            let is_huge_neg = _mm256_cmp_ps(x, min_exp, _CMP_LT_OQ);
            let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_ps(res, one, is_huge_pos);
            final_res = _mm256_blendv_ps(final_res, zero, is_huge_neg);
            final_res = _mm256_blendv_ps(final_res, nan, is_nan);

            _mm256_storeu_ps(dst.add(i), final_res);
            i += 8;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = scalar_sigmoid_f32(x);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// AArch64 NEON Implementation Kernels (f64 and f32)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
unsafe fn exp_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let inv_ln2 = vdupq_n_f64(INV_LN2_F64);
        let ln2_hi = vdupq_n_f64(LN2_HI_F64);
        let ln2_lo = vdupq_n_f64(LN2_LO_F64);
        let one = vdupq_n_f64(1.0);
        let max_exp = vdupq_n_f64(709.782712893384);
        let min_exp = vdupq_n_f64(-745.1332191019412);
        let inf = vdupq_n_f64(f64::INFINITY);
        let zero = vdupq_n_f64(0.0);

        let c12 = vdupq_n_f64(1.0 / 479001600.0);
        let c11 = vdupq_n_f64(1.0 / 39916800.0);
        let c10 = vdupq_n_f64(1.0 / 3628800.0);
        let c9 = vdupq_n_f64(1.0 / 362880.0);
        let c8 = vdupq_n_f64(1.0 / 40320.0);
        let c7 = vdupq_n_f64(1.0 / 5040.0);
        let c6 = vdupq_n_f64(1.0 / 720.0);
        let c5 = vdupq_n_f64(1.0 / 120.0);
        let c4 = vdupq_n_f64(1.0 / 24.0);
        let c3 = vdupq_n_f64(1.0 / 6.0);
        let c2 = vdupq_n_f64(0.5);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let k = vrndnq_f64(vmulq_f64(x, inv_ln2));

            let mut r = vfmsq_f64(x, k, ln2_hi);
            r = vfmsq_f64(r, k, ln2_lo);

            let mut p = vfmaq_f64(c11, c12, r);
            p = vfmaq_f64(c10, p, r);
            p = vfmaq_f64(c9, p, r);
            p = vfmaq_f64(c8, p, r);
            p = vfmaq_f64(c7, p, r);
            p = vfmaq_f64(c6, p, r);
            p = vfmaq_f64(c5, p, r);
            p = vfmaq_f64(c4, p, r);
            p = vfmaq_f64(c3, p, r);
            p = vfmaq_f64(c2, p, r);
            p = vfmaq_f64(one, p, r);
            let poly = vfmaq_f64(one, p, r);

            let ki = vcvtq_s64_f64(k);
            let k1 = vshrq_n_s64(ki, 1);
            let k2 = vsubq_s64(ki, k1);

            let bias = vdupq_n_s64(1023);
            let scale1 = vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(k1, bias), 52));
            let scale2 = vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(k2, bias), 52));

            let res = vmulq_f64(vmulq_f64(poly, scale1), scale2);

            let overflow_mask = vcgtq_f64(x, max_exp);
            let underflow_mask = vcltq_f64(x, min_exp);

            let clamped = vbslq_f64(overflow_mask, inf, res);
            let final_res = vbslq_f64(underflow_mask, zero, clamped);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn exp2_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let ln2 = vdupq_n_f64(std::f64::consts::LN_2);
        let max_exp = vdupq_n_f64(1023.99);
        let min_exp = vdupq_n_f64(-1074.0);
        let inf = vdupq_n_f64(f64::INFINITY);
        let zero = vdupq_n_f64(0.0);
        let one = vdupq_n_f64(1.0);

        let c12 = vdupq_n_f64(1.0 / 479001600.0);
        let c11 = vdupq_n_f64(1.0 / 39916800.0);
        let c10 = vdupq_n_f64(1.0 / 3628800.0);
        let c9 = vdupq_n_f64(1.0 / 362880.0);
        let c8 = vdupq_n_f64(1.0 / 40320.0);
        let c7 = vdupq_n_f64(1.0 / 5040.0);
        let c6 = vdupq_n_f64(1.0 / 720.0);
        let c5 = vdupq_n_f64(1.0 / 120.0);
        let c4 = vdupq_n_f64(1.0 / 24.0);
        let c3 = vdupq_n_f64(1.0 / 6.0);
        let c2 = vdupq_n_f64(0.5);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let k = vrndnq_f64(x);
            let rem = vsubq_f64(x, k);
            let r = vmulq_f64(rem, ln2);

            let mut p = vfmaq_f64(c11, c12, r);
            p = vfmaq_f64(c10, p, r);
            p = vfmaq_f64(c9, p, r);
            p = vfmaq_f64(c8, p, r);
            p = vfmaq_f64(c7, p, r);
            p = vfmaq_f64(c6, p, r);
            p = vfmaq_f64(c5, p, r);
            p = vfmaq_f64(c4, p, r);
            p = vfmaq_f64(c3, p, r);
            p = vfmaq_f64(c2, p, r);
            p = vfmaq_f64(one, p, r);
            let poly = vfmaq_f64(one, p, r);

            let ki = vcvtq_s64_f64(k);
            let k1 = vshrq_n_s64(ki, 1);
            let k2 = vsubq_s64(ki, k1);

            let bias = vdupq_n_s64(1023);
            let scale1 = vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(k1, bias), 52));
            let scale2 = vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(k2, bias), 52));

            let res = vmulq_f64(vmulq_f64(poly, scale1), scale2);

            let overflow_mask = vcgtq_f64(x, max_exp);
            let underflow_mask = vcltq_f64(x, min_exp);

            let clamped = vbslq_f64(overflow_mask, inf, res);
            let final_res = vbslq_f64(underflow_mask, zero, clamped);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp2();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn expm1_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let inv_ln2 = vdupq_n_f64(INV_LN2_F64);
        let ln2_hi = vdupq_n_f64(LN2_HI_F64);
        let ln2_lo = vdupq_n_f64(LN2_LO_F64);
        let one = vdupq_n_f64(1.0);
        let neg_one = vdupq_n_f64(-1.0);
        let max_exp = vdupq_n_f64(709.782712893384);
        let min_expm1 = vdupq_n_f64(-54.0);
        let inf = vdupq_n_f64(f64::INFINITY);
        let zero = vdupq_n_f64(0.0);
        let _ = zero;
        let nan = vdupq_n_f64(f64::NAN);

        let c16 = vdupq_n_f64(1.0 / 20922789888000.0);
        let c15 = vdupq_n_f64(1.0 / 1307674368000.0);
        let c14 = vdupq_n_f64(1.0 / 87178291200.0);
        let c13 = vdupq_n_f64(1.0 / 6227020800.0);
        let c12 = vdupq_n_f64(1.0 / 479001600.0);
        let c11 = vdupq_n_f64(1.0 / 39916800.0);
        let c10 = vdupq_n_f64(1.0 / 3628800.0);
        let c9 = vdupq_n_f64(1.0 / 362880.0);
        let c8 = vdupq_n_f64(1.0 / 40320.0);
        let c7 = vdupq_n_f64(1.0 / 5040.0);
        let c6 = vdupq_n_f64(1.0 / 720.0);
        let c5 = vdupq_n_f64(1.0 / 120.0);
        let c4 = vdupq_n_f64(1.0 / 24.0);
        let c3 = vdupq_n_f64(1.0 / 6.0);
        let c2 = vdupq_n_f64(0.5);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let k = vrndnq_f64(vmulq_f64(x, inv_ln2));

            let mut r = vfmsq_f64(x, k, ln2_hi);
            r = vfmsq_f64(r, k, ln2_lo);

            let mut p = vfmaq_f64(c15, c16, r);
            p = vfmaq_f64(c14, p, r);
            p = vfmaq_f64(c13, p, r);
            p = vfmaq_f64(c12, p, r);
            p = vfmaq_f64(c11, p, r);
            p = vfmaq_f64(c10, p, r);
            p = vfmaq_f64(c9, p, r);
            p = vfmaq_f64(c8, p, r);
            p = vfmaq_f64(c7, p, r);
            p = vfmaq_f64(c6, p, r);
            p = vfmaq_f64(c5, p, r);
            p = vfmaq_f64(c4, p, r);
            p = vfmaq_f64(c3, p, r);
            p = vfmaq_f64(c2, p, r);
            let poly = vfmaq_f64(r, vmulq_f64(r, r), p);

            let ki = vcvtq_s64_f64(k);
            let k1 = vshrq_n_s64(ki, 1);
            let k2 = vsubq_s64(ki, k1);

            let bias = vdupq_n_s64(1023);
            let scale1 = vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(k1, bias), 52));
            let scale2 = vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(k2, bias), 52));
            let twok = vmulq_f64(scale1, scale2);

            let twok_minus_one = vsubq_f64(twok, one);
            let res_k_nonzero = vfmaq_f64(twok_minus_one, twok, poly);

            let mut p_dir = vfmaq_f64(c15, c16, x);
            p_dir = vfmaq_f64(c14, p_dir, x);
            p_dir = vfmaq_f64(c13, p_dir, x);
            p_dir = vfmaq_f64(c12, p_dir, x);
            p_dir = vfmaq_f64(c11, p_dir, x);
            p_dir = vfmaq_f64(c10, p_dir, x);
            p_dir = vfmaq_f64(c9, p_dir, x);
            p_dir = vfmaq_f64(c8, p_dir, x);
            p_dir = vfmaq_f64(c7, p_dir, x);
            p_dir = vfmaq_f64(c6, p_dir, x);
            p_dir = vfmaq_f64(c5, p_dir, x);
            p_dir = vfmaq_f64(c4, p_dir, x);
            p_dir = vfmaq_f64(c3, p_dir, x);
            p_dir = vfmaq_f64(c2, p_dir, x);
            let poly_direct = vfmaq_f64(x, vmulq_f64(x, x), p_dir);

            let is_small = vcltq_f64(vabsq_f64(x), vdupq_n_f64(0.69314718056));
            let res = vbslq_f64(is_small, poly_direct, res_k_nonzero);

            let overflow_mask = vcgtq_f64(x, max_exp);
            let underflow_mask = vcltq_f64(x, min_expm1);
            let is_nan = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));

            let mut final_res = vbslq_f64(overflow_mask, inf, res);
            final_res = vbslq_f64(underflow_mask, neg_one, final_res);
            final_res = vbslq_f64(is_nan, nan, final_res);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp_m1();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn log_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let ln2_hi = vdupq_n_f64(LN2_HI_F64);
        let ln2_lo = vdupq_n_f64(LN2_LO_F64);
        let sqrt2_half = vdupq_n_f64(std::f64::consts::SQRT_2 * 0.5);
        let one = vdupq_n_f64(1.0);
        let two = vdupq_n_f64(2.0);
        let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);
        let pos_inf = vdupq_n_f64(f64::INFINITY);
        let nan = vdupq_n_f64(f64::NAN);
        let min_normal = vdupq_n_f64(2.2250738585072014e-308);

        let c10 = vdupq_n_f64(2.0 / 21.0);
        let c9 = vdupq_n_f64(2.0 / 19.0);
        let c8 = vdupq_n_f64(2.0 / 17.0);
        let c7 = vdupq_n_f64(2.0 / 15.0);
        let c6 = vdupq_n_f64(2.0 / 13.0);
        let c5 = vdupq_n_f64(2.0 / 11.0);
        let c4 = vdupq_n_f64(2.0 / 9.0);
        let c3 = vdupq_n_f64(2.0 / 7.0);
        let c2 = vdupq_n_f64(2.0 / 5.0);
        let c1 = vdupq_n_f64(2.0 / 3.0);
        let c0 = vdupq_n_f64(2.0);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));

            let is_subnormal = vandq_u64(vcltq_f64(x, min_normal), vcgtq_f64(x, zero));
            if vgetq_lane_u64::<0>(is_subnormal) != 0 || vgetq_lane_u64::<1>(is_subnormal) != 0 {
                for j in 0..2 {
                    *dst.add(i + j) = (*src.add(i + j)).ln();
                }
                i += 2;
                continue;
            }

            let x_bits = vreinterpretq_u64_f64(x);
            let exp_bits = vshrq_n_u64(x_bits, 52);
            let mut k = vsubq_f64(
                vcvtq_f64_s64(vreinterpretq_s64_u64(exp_bits)),
                vdupq_n_f64(1023.0),
            );

            let mant_mask = vdupq_n_u64(0x000F_FFFF_FFFF_FFFF);
            let exp_1023 = vdupq_n_u64(0x3FF0_0000_0000_0000);
            let mant_bits = vorrq_u64(vandq_u64(x_bits, mant_mask), exp_1023);
            let mut m = vmulq_f64(vreinterpretq_f64_u64(mant_bits), vdupq_n_f64(0.5));
            k = vaddq_f64(k, one);

            let lt_mask = vcltq_f64(m, sqrt2_half);
            m = vbslq_f64(lt_mask, vmulq_f64(m, two), m);
            k = vbslq_f64(lt_mask, vsubq_f64(k, one), k);

            let u = vdivq_f64(vsubq_f64(m, one), vaddq_f64(m, one));
            let u2 = vmulq_f64(u, u);

            let mut p = vfmaq_f64(c9, c10, u2);
            p = vfmaq_f64(c8, p, u2);
            p = vfmaq_f64(c7, p, u2);
            p = vfmaq_f64(c6, p, u2);
            p = vfmaq_f64(c5, p, u2);
            p = vfmaq_f64(c4, p, u2);
            p = vfmaq_f64(c3, p, u2);
            p = vfmaq_f64(c2, p, u2);
            p = vfmaq_f64(c1, p, u2);
            p = vfmaq_f64(c0, p, u2);
            let ln_m = vmulq_f64(u, p);

            let res = vfmaq_f64(vfmaq_f64(ln_m, k, ln2_lo), k, ln2_hi);

            let eq_zero = vceqq_f64(x, zero);
            let lt_zero = vcltq_f64(x, zero);
            let is_inf = vceqq_f64(x, pos_inf);
            let is_nan = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));

            let mut final_res = vbslq_f64(eq_zero, neg_inf, res);
            final_res = vbslq_f64(lt_zero, nan, final_res);
            final_res = vbslq_f64(is_inf, pos_inf, final_res);
            final_res = vbslq_f64(is_nan, nan, final_res);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.ln();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn log2_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let log2_e = vdupq_n_f64(std::f64::consts::LOG2_E);
        let sqrt2_half = vdupq_n_f64(std::f64::consts::SQRT_2 * 0.5);
        let one = vdupq_n_f64(1.0);
        let two = vdupq_n_f64(2.0);
        let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);
        let pos_inf = vdupq_n_f64(f64::INFINITY);
        let nan = vdupq_n_f64(f64::NAN);
        let min_normal = vdupq_n_f64(2.2250738585072014e-308);

        let c10 = vdupq_n_f64(2.0 / 21.0);
        let c9 = vdupq_n_f64(2.0 / 19.0);
        let c8 = vdupq_n_f64(2.0 / 17.0);
        let c7 = vdupq_n_f64(2.0 / 15.0);
        let c6 = vdupq_n_f64(2.0 / 13.0);
        let c5 = vdupq_n_f64(2.0 / 11.0);
        let c4 = vdupq_n_f64(2.0 / 9.0);
        let c3 = vdupq_n_f64(2.0 / 7.0);
        let c2 = vdupq_n_f64(2.0 / 5.0);
        let c1 = vdupq_n_f64(2.0 / 3.0);
        let c0 = vdupq_n_f64(2.0);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));

            let is_subnormal = vandq_u64(vcltq_f64(x, min_normal), vcgtq_f64(x, zero));
            if vgetq_lane_u64::<0>(is_subnormal) != 0 || vgetq_lane_u64::<1>(is_subnormal) != 0 {
                for j in 0..2 {
                    *dst.add(i + j) = (*src.add(i + j)).log2();
                }
                i += 2;
                continue;
            }

            let x_bits = vreinterpretq_u64_f64(x);
            let exp_bits = vshrq_n_u64(x_bits, 52);
            let mut k = vsubq_f64(
                vcvtq_f64_s64(vreinterpretq_s64_u64(exp_bits)),
                vdupq_n_f64(1023.0),
            );

            let mant_mask = vdupq_n_u64(0x000F_FFFF_FFFF_FFFF);
            let exp_1023 = vdupq_n_u64(0x3FF0_0000_0000_0000);
            let mant_bits = vorrq_u64(vandq_u64(x_bits, mant_mask), exp_1023);
            let mut m = vmulq_f64(vreinterpretq_f64_u64(mant_bits), vdupq_n_f64(0.5));
            k = vaddq_f64(k, one);

            let lt_mask = vcltq_f64(m, sqrt2_half);
            m = vbslq_f64(lt_mask, vmulq_f64(m, two), m);
            k = vbslq_f64(lt_mask, vsubq_f64(k, one), k);

            let u = vdivq_f64(vsubq_f64(m, one), vaddq_f64(m, one));
            let u2 = vmulq_f64(u, u);

            let mut p = vfmaq_f64(c9, c10, u2);
            p = vfmaq_f64(c8, p, u2);
            p = vfmaq_f64(c7, p, u2);
            p = vfmaq_f64(c6, p, u2);
            p = vfmaq_f64(c5, p, u2);
            p = vfmaq_f64(c4, p, u2);
            p = vfmaq_f64(c3, p, u2);
            p = vfmaq_f64(c2, p, u2);
            p = vfmaq_f64(c1, p, u2);
            p = vfmaq_f64(c0, p, u2);
            let ln_m = vmulq_f64(u, p);

            let res = vfmaq_f64(k, ln_m, log2_e);

            let eq_zero = vceqq_f64(x, zero);
            let lt_zero = vcltq_f64(x, zero);
            let is_inf = vceqq_f64(x, pos_inf);
            let is_nan = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));

            let mut final_res = vbslq_f64(eq_zero, neg_inf, res);
            final_res = vbslq_f64(lt_zero, nan, final_res);
            final_res = vbslq_f64(is_inf, pos_inf, final_res);
            final_res = vbslq_f64(is_nan, nan, final_res);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.log2();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn log10_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let log10_2_hi = vdupq_n_f64(LOG10_2_HI_F64);
        let log10_2_lo = vdupq_n_f64(LOG10_2_LO_F64);
        let log10_e = vdupq_n_f64(std::f64::consts::LOG10_E);
        let sqrt2_half = vdupq_n_f64(std::f64::consts::SQRT_2 * 0.5);
        let one = vdupq_n_f64(1.0);
        let two = vdupq_n_f64(2.0);
        let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);
        let pos_inf = vdupq_n_f64(f64::INFINITY);
        let nan = vdupq_n_f64(f64::NAN);
        let min_normal = vdupq_n_f64(2.2250738585072014e-308);

        let c10 = vdupq_n_f64(2.0 / 21.0);
        let c9 = vdupq_n_f64(2.0 / 19.0);
        let c8 = vdupq_n_f64(2.0 / 17.0);
        let c7 = vdupq_n_f64(2.0 / 15.0);
        let c6 = vdupq_n_f64(2.0 / 13.0);
        let c5 = vdupq_n_f64(2.0 / 11.0);
        let c4 = vdupq_n_f64(2.0 / 9.0);
        let c3 = vdupq_n_f64(2.0 / 7.0);
        let c2 = vdupq_n_f64(2.0 / 5.0);
        let c1 = vdupq_n_f64(2.0 / 3.0);
        let c0 = vdupq_n_f64(2.0);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));

            let is_subnormal = vandq_u64(vcltq_f64(x, min_normal), vcgtq_f64(x, zero));
            if vgetq_lane_u64::<0>(is_subnormal) != 0 || vgetq_lane_u64::<1>(is_subnormal) != 0 {
                for j in 0..2 {
                    *dst.add(i + j) = (*src.add(i + j)).log10();
                }
                i += 2;
                continue;
            }

            let x_bits = vreinterpretq_u64_f64(x);
            let exp_bits = vshrq_n_u64(x_bits, 52);
            let mut k = vsubq_f64(
                vcvtq_f64_s64(vreinterpretq_s64_u64(exp_bits)),
                vdupq_n_f64(1023.0),
            );

            let mant_mask = vdupq_n_u64(0x000F_FFFF_FFFF_FFFF);
            let exp_1023 = vdupq_n_u64(0x3FF0_0000_0000_0000);
            let mant_bits = vorrq_u64(vandq_u64(x_bits, mant_mask), exp_1023);
            let mut m = vmulq_f64(vreinterpretq_f64_u64(mant_bits), vdupq_n_f64(0.5));
            k = vaddq_f64(k, one);

            let lt_mask = vcltq_f64(m, sqrt2_half);
            m = vbslq_f64(lt_mask, vmulq_f64(m, two), m);
            k = vbslq_f64(lt_mask, vsubq_f64(k, one), k);

            let u = vdivq_f64(vsubq_f64(m, one), vaddq_f64(m, one));
            let u2 = vmulq_f64(u, u);

            let mut p = vfmaq_f64(c9, c10, u2);
            p = vfmaq_f64(c8, p, u2);
            p = vfmaq_f64(c7, p, u2);
            p = vfmaq_f64(c6, p, u2);
            p = vfmaq_f64(c5, p, u2);
            p = vfmaq_f64(c4, p, u2);
            p = vfmaq_f64(c3, p, u2);
            p = vfmaq_f64(c2, p, u2);
            p = vfmaq_f64(c1, p, u2);
            p = vfmaq_f64(c0, p, u2);
            let ln_m = vmulq_f64(u, p);

            let res = vfmaq_f64(
                vfmaq_f64(vmulq_f64(k, log10_2_lo), ln_m, log10_e),
                k,
                log10_2_hi,
            );

            let eq_zero = vceqq_f64(x, zero);
            let lt_zero = vcltq_f64(x, zero);
            let is_inf = vceqq_f64(x, pos_inf);
            let is_nan = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));

            let mut final_res = vbslq_f64(eq_zero, neg_inf, res);
            final_res = vbslq_f64(lt_zero, nan, final_res);
            final_res = vbslq_f64(is_inf, pos_inf, final_res);
            final_res = vbslq_f64(is_nan, nan, final_res);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.log10();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn log1p_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let ln2_hi = vdupq_n_f64(LN2_HI_F64);
        let ln2_lo = vdupq_n_f64(LN2_LO_F64);
        let sqrt2_half = vdupq_n_f64(std::f64::consts::SQRT_2 * 0.5);
        let one = vdupq_n_f64(1.0);
        let neg_one = vdupq_n_f64(-1.0);
        let two = vdupq_n_f64(2.0);
        let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);
        let pos_inf = vdupq_n_f64(f64::INFINITY);
        let nan = vdupq_n_f64(f64::NAN);
        let threshold = vdupq_n_f64(0.35);

        let c10 = vdupq_n_f64(2.0 / 21.0);
        let c9 = vdupq_n_f64(2.0 / 19.0);
        let c8 = vdupq_n_f64(2.0 / 17.0);
        let c7 = vdupq_n_f64(2.0 / 15.0);
        let c6 = vdupq_n_f64(2.0 / 13.0);
        let c5 = vdupq_n_f64(2.0 / 11.0);
        let c4 = vdupq_n_f64(2.0 / 9.0);
        let c3 = vdupq_n_f64(2.0 / 7.0);
        let c2 = vdupq_n_f64(2.0 / 5.0);
        let c1 = vdupq_n_f64(2.0 / 3.0);
        let c0 = vdupq_n_f64(2.0);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let abs_x = vabsq_f64(x);
            let is_small = vcltq_f64(abs_x, threshold);

            let u_s = vdivq_f64(x, vaddq_f64(two, x));
            let u2_s = vmulq_f64(u_s, u_s);
            let mut p_s = vfmaq_f64(c9, c10, u2_s);
            p_s = vfmaq_f64(c8, p_s, u2_s);
            p_s = vfmaq_f64(c7, p_s, u2_s);
            p_s = vfmaq_f64(c6, p_s, u2_s);
            p_s = vfmaq_f64(c5, p_s, u2_s);
            p_s = vfmaq_f64(c4, p_s, u2_s);
            p_s = vfmaq_f64(c3, p_s, u2_s);
            p_s = vfmaq_f64(c2, p_s, u2_s);
            p_s = vfmaq_f64(c1, p_s, u2_s);
            p_s = vfmaq_f64(c0, p_s, u2_s);
            let res_small = vmulq_f64(u_s, p_s);

            let y = vaddq_f64(one, x);
            let y_bits = vreinterpretq_u64_f64(y);
            let exp_bits = vshrq_n_u64(y_bits, 52);
            let mut k = vsubq_f64(
                vcvtq_f64_s64(vreinterpretq_s64_u64(exp_bits)),
                vdupq_n_f64(1023.0),
            );

            let mant_mask = vdupq_n_u64(0x000F_FFFF_FFFF_FFFF);
            let exp_1023 = vdupq_n_u64(0x3FF0_0000_0000_0000);
            let mant_bits = vorrq_u64(vandq_u64(y_bits, mant_mask), exp_1023);
            let mut m = vmulq_f64(vreinterpretq_f64_u64(mant_bits), vdupq_n_f64(0.5));
            k = vaddq_f64(k, one);

            let lt_mask = vcltq_f64(m, sqrt2_half);
            m = vbslq_f64(lt_mask, vmulq_f64(m, two), m);
            k = vbslq_f64(lt_mask, vsubq_f64(k, one), k);

            let u_l = vdivq_f64(vsubq_f64(m, one), vaddq_f64(m, one));
            let u2_l = vmulq_f64(u_l, u_l);

            let mut p_l = vfmaq_f64(c9, c10, u2_l);
            p_l = vfmaq_f64(c8, p_l, u2_l);
            p_l = vfmaq_f64(c7, p_l, u2_l);
            p_l = vfmaq_f64(c6, p_l, u2_l);
            p_l = vfmaq_f64(c5, p_l, u2_l);
            p_l = vfmaq_f64(c4, p_l, u2_l);
            p_l = vfmaq_f64(c3, p_l, u2_l);
            p_l = vfmaq_f64(c2, p_l, u2_l);
            p_l = vfmaq_f64(c1, p_l, u2_l);
            p_l = vfmaq_f64(c0, p_l, u2_l);
            let ln_m = vmulq_f64(u_l, p_l);
            let res_large = vfmaq_f64(vfmaq_f64(ln_m, k, ln2_lo), k, ln2_hi);

            let res = vbslq_f64(is_small, res_small, res_large);

            let eq_neg_one = vceqq_f64(x, neg_one);
            let lt_neg_one = vcltq_f64(x, neg_one);
            let is_inf = vceqq_f64(x, pos_inf);
            let is_nan = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));
            let is_zero = vceqq_f64(x, zero);

            let mut final_res = vbslq_f64(eq_neg_one, neg_inf, res);
            final_res = vbslq_f64(lt_neg_one, nan, final_res);
            final_res = vbslq_f64(is_inf, pos_inf, final_res);
            final_res = vbslq_f64(is_nan, nan, final_res);
            final_res = vbslq_f64(is_zero, x, final_res);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.ln_1p();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sin_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let p1 = vdupq_n_f64(PIO2_1_F64);
        let p2 = vdupq_n_f64(PIO2_2_F64);
        let p3 = vdupq_n_f64(PIO2_3_F64);
        let inv_pio2 = vdupq_n_f64(INV_PIO2_F64);
        let one = vdupq_n_f64(1.0);
        let bound = vdupq_n_f64(1e5);

        let s8 = vdupq_n_f64(1.0 / 355687428096000.0);
        let s7 = vdupq_n_f64(-1.0 / 1307674368000.0);
        let s6 = vdupq_n_f64(1.0 / 6227020800.0);
        let s5 = vdupq_n_f64(-1.0 / 39916800.0);
        let s4 = vdupq_n_f64(1.0 / 362880.0);
        let s3 = vdupq_n_f64(-1.0 / 5040.0);
        let s2 = vdupq_n_f64(1.0 / 120.0);
        let s1 = vdupq_n_f64(-1.0 / 6.0);

        let c8 = vdupq_n_f64(1.0 / 20922789888000.0);
        let c7 = vdupq_n_f64(-1.0 / 87178291200.0);
        let c6 = vdupq_n_f64(1.0 / 479001600.0);
        let c5 = vdupq_n_f64(-1.0 / 3628800.0);
        let c4 = vdupq_n_f64(1.0 / 40320.0);
        let c3 = vdupq_n_f64(-1.0 / 720.0);
        let c2 = vdupq_n_f64(1.0 / 24.0);
        let c1 = vdupq_n_f64(-0.5);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let abs_x = vabsq_f64(x);
            let in_range = vcltq_f64(abs_x, bound);
            if vgetq_lane_u64::<0>(in_range) == 0 || vgetq_lane_u64::<1>(in_range) == 0 {
                for j in 0..2 {
                    *dst.add(i + j) = (*src.add(i + j)).sin();
                }
                i += 2;
                continue;
            }

            let k = vrndnq_f64(vmulq_f64(x, inv_pio2));
            let mut r = vfmsq_f64(x, k, p1);
            r = vfmsq_f64(r, k, p2);
            r = vfmsq_f64(r, k, p3);

            let r2 = vmulq_f64(r, r);

            let mut sp = vfmaq_f64(s7, s8, r2);
            sp = vfmaq_f64(s6, sp, r2);
            sp = vfmaq_f64(s5, sp, r2);
            sp = vfmaq_f64(s4, sp, r2);
            sp = vfmaq_f64(s3, sp, r2);
            sp = vfmaq_f64(s2, sp, r2);
            sp = vfmaq_f64(s1, sp, r2);
            let sin_r = vfmaq_f64(r, vmulq_f64(r, r2), sp);

            let mut cp = vfmaq_f64(c7, c8, r2);
            cp = vfmaq_f64(c6, cp, r2);
            cp = vfmaq_f64(c5, cp, r2);
            cp = vfmaq_f64(c4, cp, r2);
            cp = vfmaq_f64(c3, cp, r2);
            cp = vfmaq_f64(c2, cp, r2);
            cp = vfmaq_f64(c1, cp, r2);
            let cos_r = vfmaq_f64(one, r2, cp);

            let ki = vcvtq_s64_f64(k);
            let q = vandq_s64(ki, vdupq_n_s64(3));

            let is_q1_or_q3 = vceqq_s64(vandq_s64(q, vdupq_n_s64(1)), vdupq_n_s64(1));
            let is_q2_or_q3 = vcgtq_s64(q, vdupq_n_s64(1));

            let mag = vbslq_f64(is_q1_or_q3, cos_r, sin_r);
            let neg_mag = vnegq_f64(mag);
            let res = vbslq_f64(is_q2_or_q3, neg_mag, mag);

            vst1q_f64(dst.add(i), res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.sin();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn cos_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let p1 = vdupq_n_f64(PIO2_1_F64);
        let p2 = vdupq_n_f64(PIO2_2_F64);
        let p3 = vdupq_n_f64(PIO2_3_F64);
        let inv_pio2 = vdupq_n_f64(INV_PIO2_F64);
        let one = vdupq_n_f64(1.0);
        let bound = vdupq_n_f64(1e5);

        let s8 = vdupq_n_f64(1.0 / 355687428096000.0);
        let s7 = vdupq_n_f64(-1.0 / 1307674368000.0);
        let s6 = vdupq_n_f64(1.0 / 6227020800.0);
        let s5 = vdupq_n_f64(-1.0 / 39916800.0);
        let s4 = vdupq_n_f64(1.0 / 362880.0);
        let s3 = vdupq_n_f64(-1.0 / 5040.0);
        let s2 = vdupq_n_f64(1.0 / 120.0);
        let s1 = vdupq_n_f64(-1.0 / 6.0);

        let c8 = vdupq_n_f64(1.0 / 20922789888000.0);
        let c7 = vdupq_n_f64(-1.0 / 87178291200.0);
        let c6 = vdupq_n_f64(1.0 / 479001600.0);
        let c5 = vdupq_n_f64(-1.0 / 3628800.0);
        let c4 = vdupq_n_f64(1.0 / 40320.0);
        let c3 = vdupq_n_f64(-1.0 / 720.0);
        let c2 = vdupq_n_f64(1.0 / 24.0);
        let c1 = vdupq_n_f64(-0.5);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let abs_x = vabsq_f64(x);
            let in_range = vcltq_f64(abs_x, bound);
            if vgetq_lane_u64::<0>(in_range) == 0 || vgetq_lane_u64::<1>(in_range) == 0 {
                for j in 0..2 {
                    *dst.add(i + j) = (*src.add(i + j)).cos();
                }
                i += 2;
                continue;
            }

            let k = vrndnq_f64(vmulq_f64(x, inv_pio2));
            let mut r = vfmsq_f64(x, k, p1);
            r = vfmsq_f64(r, k, p2);
            r = vfmsq_f64(r, k, p3);

            let r2 = vmulq_f64(r, r);

            let mut sp = vfmaq_f64(s7, s8, r2);
            sp = vfmaq_f64(s6, sp, r2);
            sp = vfmaq_f64(s5, sp, r2);
            sp = vfmaq_f64(s4, sp, r2);
            sp = vfmaq_f64(s3, sp, r2);
            sp = vfmaq_f64(s2, sp, r2);
            sp = vfmaq_f64(s1, sp, r2);
            let sin_r = vfmaq_f64(r, vmulq_f64(r, r2), sp);

            let mut cp = vfmaq_f64(c7, c8, r2);
            cp = vfmaq_f64(c6, cp, r2);
            cp = vfmaq_f64(c5, cp, r2);
            cp = vfmaq_f64(c4, cp, r2);
            cp = vfmaq_f64(c3, cp, r2);
            cp = vfmaq_f64(c2, cp, r2);
            cp = vfmaq_f64(c1, cp, r2);
            let cos_r = vfmaq_f64(one, r2, cp);

            let ki = vcvtq_s64_f64(k);
            let q = vandq_s64(vaddq_s64(ki, vdupq_n_s64(1)), vdupq_n_s64(3));

            let is_q1_or_q3 = vceqq_s64(vandq_s64(q, vdupq_n_s64(1)), vdupq_n_s64(1));
            let is_q2_or_q3 = vcgtq_s64(q, vdupq_n_s64(1));

            let mag = vbslq_f64(is_q1_or_q3, cos_r, sin_r);
            let neg_mag = vnegq_f64(mag);
            let res = vbslq_f64(is_q2_or_q3, neg_mag, mag);

            vst1q_f64(dst.add(i), res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.cos();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn tan_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let p1 = vdupq_n_f64(PIO2_1_F64);
        let p2 = vdupq_n_f64(PIO2_2_F64);
        let p3 = vdupq_n_f64(PIO2_3_F64);
        let inv_pio2 = vdupq_n_f64(INV_PIO2_F64);
        let one = vdupq_n_f64(1.0);
        let bound = vdupq_n_f64(1e5);

        let s8 = vdupq_n_f64(1.0 / 355687428096000.0);
        let s7 = vdupq_n_f64(-1.0 / 1307674368000.0);
        let s6 = vdupq_n_f64(1.0 / 6227020800.0);
        let s5 = vdupq_n_f64(-1.0 / 39916800.0);
        let s4 = vdupq_n_f64(1.0 / 362880.0);
        let s3 = vdupq_n_f64(-1.0 / 5040.0);
        let s2 = vdupq_n_f64(1.0 / 120.0);
        let s1 = vdupq_n_f64(-1.0 / 6.0);

        let c8 = vdupq_n_f64(1.0 / 20922789888000.0);
        let c7 = vdupq_n_f64(-1.0 / 87178291200.0);
        let c6 = vdupq_n_f64(1.0 / 479001600.0);
        let c5 = vdupq_n_f64(-1.0 / 3628800.0);
        let c4 = vdupq_n_f64(1.0 / 40320.0);
        let c3 = vdupq_n_f64(-1.0 / 720.0);
        let c2 = vdupq_n_f64(1.0 / 24.0);
        let c1 = vdupq_n_f64(-0.5);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let abs_x = vabsq_f64(x);
            let in_range = vcltq_f64(abs_x, bound);
            if vgetq_lane_u64::<0>(in_range) == 0 || vgetq_lane_u64::<1>(in_range) == 0 {
                for j in 0..2 {
                    *dst.add(i + j) = (*src.add(i + j)).tan();
                }
                i += 2;
                continue;
            }

            let k = vrndnq_f64(vmulq_f64(x, inv_pio2));
            let mut r = vfmsq_f64(x, k, p1);
            r = vfmsq_f64(r, k, p2);
            r = vfmsq_f64(r, k, p3);

            let r2 = vmulq_f64(r, r);

            let mut sp = vfmaq_f64(s7, s8, r2);
            sp = vfmaq_f64(s6, sp, r2);
            sp = vfmaq_f64(s5, sp, r2);
            sp = vfmaq_f64(s4, sp, r2);
            sp = vfmaq_f64(s3, sp, r2);
            sp = vfmaq_f64(s2, sp, r2);
            sp = vfmaq_f64(s1, sp, r2);
            let sin_r = vfmaq_f64(r, vmulq_f64(r, r2), sp);

            let mut cp = vfmaq_f64(c7, c8, r2);
            cp = vfmaq_f64(c6, cp, r2);
            cp = vfmaq_f64(c5, cp, r2);
            cp = vfmaq_f64(c4, cp, r2);
            cp = vfmaq_f64(c3, cp, r2);
            cp = vfmaq_f64(c2, cp, r2);
            cp = vfmaq_f64(c1, cp, r2);
            let cos_r = vfmaq_f64(one, r2, cp);

            let ki = vcvtq_s64_f64(k);
            let is_odd = vceqq_s64(vandq_s64(ki, vdupq_n_s64(1)), vdupq_n_s64(1));

            let num = vbslq_f64(is_odd, vnegq_f64(cos_r), sin_r);
            let den = vbslq_f64(is_odd, sin_r, cos_r);
            let res = vdivq_f64(num, den);

            vst1q_f64(dst.add(i), res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.tan();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn tanh_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let twenty = vdupq_n_f64(20.0);
        let one = vdupq_n_f64(1.0);
        let two = vdupq_n_f64(2.0);
        let nan = vdupq_n_f64(f64::NAN);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let abs_x = vabsq_f64(x);
            let is_neg = vcltq_f64(x, zero);
            let y = vaddq_f64(abs_x, abs_x);

            let mut em1 = [0.0; 2];
            let mut y_arr = [0.0; 2];
            vst1q_f64(y_arr.as_mut_ptr(), y);
            expm1_f64_neon(y_arr.as_ptr(), em1.as_mut_ptr(), 2);
            let em1_vec = vld1q_f64(em1.as_ptr());

            let th = vdivq_f64(em1_vec, vaddq_f64(em1_vec, two));
            let signed_th = vbslq_f64(is_neg, vnegq_f64(th), th);

            let large_mask = vcgeq_f64(abs_x, twenty);
            let sat = vbslq_f64(is_neg, vnegq_f64(one), one);
            let res = vbslq_f64(large_mask, sat, signed_th);

            let is_nan = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));
            let is_zero = vceqq_f64(x, zero);
            let mut final_res = vbslq_f64(is_nan, nan, res);
            final_res = vbslq_f64(is_zero, x, final_res);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.tanh();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sigmoid_f64_neon(src: *const f64, dst: *mut f64, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let forty = vdupq_n_f64(40.0);
        let min_exp = vdupq_n_f64(-745.0);
        let one = vdupq_n_f64(1.0);
        let nan = vdupq_n_f64(f64::NAN);

        while i + 2 <= len {
            let x = vld1q_f64(src.add(i));
            let abs_x = vabsq_f64(x);
            let neg_abs_x = vnegq_f64(abs_x);

            let mut neg_abs = [0.0; 2];
            let mut exp_vals = [0.0; 2];
            vst1q_f64(neg_abs.as_mut_ptr(), neg_abs_x);
            exp_f64_neon(neg_abs.as_ptr(), exp_vals.as_mut_ptr(), 2);
            let e = vld1q_f64(exp_vals.as_ptr());

            let denom = vaddq_f64(one, e);
            let s = vdivq_f64(one, denom);
            let s_neg = vmulq_f64(e, s);

            let is_positive = vcgeq_f64(x, zero);
            let res = vbslq_f64(is_positive, s, s_neg);

            let is_huge_pos = vcgtq_f64(x, forty);
            let is_huge_neg = vcltq_f64(x, min_exp);
            let is_nan = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));

            let mut final_res = vbslq_f64(is_huge_pos, one, res);
            final_res = vbslq_f64(is_huge_neg, zero, final_res);
            final_res = vbslq_f64(is_nan, nan, final_res);

            vst1q_f64(dst.add(i), final_res);
            i += 2;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = scalar_sigmoid_f64(x);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// NEON f32 Kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
unsafe fn exp_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let inv_ln2 = vdupq_n_f32(INV_LN2_F32);
        let ln2_hi = vdupq_n_f32(LN2_HI_F32);
        let ln2_lo = vdupq_n_f32(LN2_LO_F32);
        let one = vdupq_n_f32(1.0);
        let max_exp = vdupq_n_f32(88.722_84);
        let min_exp = vdupq_n_f32(-103.27893);
        let inf = vdupq_n_f32(f32::INFINITY);
        let zero = vdupq_n_f32(0.0);

        let c6 = vdupq_n_f32(1.0 / 720.0);
        let c5 = vdupq_n_f32(1.0 / 120.0);
        let c4 = vdupq_n_f32(1.0 / 24.0);
        let c3 = vdupq_n_f32(1.0 / 6.0);
        let c2 = vdupq_n_f32(0.5);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let k = vrndnq_f32(vmulq_f32(x, inv_ln2));

            let mut r = vfmsq_f32(x, k, ln2_hi);
            r = vfmsq_f32(r, k, ln2_lo);

            let mut p = vfmaq_f32(c5, c6, r);
            p = vfmaq_f32(c4, p, r);
            p = vfmaq_f32(c3, p, r);
            p = vfmaq_f32(c2, p, r);
            p = vfmaq_f32(one, p, r);
            let poly = vfmaq_f32(one, p, r);

            let ki = vcvtq_s32_f32(k);
            let k1 = vshrq_n_s32(ki, 1);
            let k2 = vsubq_s32(ki, k1);

            let bias = vdupq_n_s32(127);
            let scale1 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k1, bias), 23));
            let scale2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k2, bias), 23));

            let res = vmulq_f32(vmulq_f32(poly, scale1), scale2);

            let overflow_mask = vcgtq_f32(x, max_exp);
            let underflow_mask = vcltq_f32(x, min_exp);

            let clamped = vbslq_f32(overflow_mask, inf, res);
            let final_res = vbslq_f32(underflow_mask, zero, clamped);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn exp2_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let max_exp = vdupq_n_f32(128.0);
        let min_exp = vdupq_n_f32(-149.0);
        let inf = vdupq_n_f32(f32::INFINITY);
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);

        let c6 = vdupq_n_f32(1.0 / 720.0);
        let c5 = vdupq_n_f32(1.0 / 120.0);
        let c4 = vdupq_n_f32(1.0 / 24.0);
        let c3 = vdupq_n_f32(1.0 / 6.0);
        let c2 = vdupq_n_f32(0.5);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let k = vrndnq_f32(x);
            let rem = vsubq_f32(x, k);
            let r = vmulq_f32(rem, ln2);

            let mut p = vfmaq_f32(c5, c6, r);
            p = vfmaq_f32(c4, p, r);
            p = vfmaq_f32(c3, p, r);
            p = vfmaq_f32(c2, p, r);
            p = vfmaq_f32(one, p, r);
            let poly = vfmaq_f32(one, p, r);

            let ki = vcvtq_s32_f32(k);
            let k1 = vshrq_n_s32(ki, 1);
            let k2 = vsubq_s32(ki, k1);

            let bias = vdupq_n_s32(127);
            let scale1 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k1, bias), 23));
            let scale2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k2, bias), 23));

            let res = vmulq_f32(vmulq_f32(poly, scale1), scale2);

            let overflow_mask = vcgtq_f32(x, max_exp);
            let underflow_mask = vcltq_f32(x, min_exp);

            let clamped = vbslq_f32(overflow_mask, inf, res);
            let final_res = vbslq_f32(underflow_mask, zero, clamped);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp2();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn expm1_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let inv_ln2 = vdupq_n_f32(INV_LN2_F32);
        let ln2_hi = vdupq_n_f32(LN2_HI_F32);
        let ln2_lo = vdupq_n_f32(LN2_LO_F32);
        let one = vdupq_n_f32(1.0);
        let neg_one = vdupq_n_f32(-1.0);
        let max_exp = vdupq_n_f32(88.722_84);
        let min_expm1 = vdupq_n_f32(-25.0);
        let inf = vdupq_n_f32(f32::INFINITY);
        let zero = vdupq_n_f32(0.0);
        let _ = zero;
        let nan = vdupq_n_f32(f32::NAN);

        let c9 = vdupq_n_f32(1.0 / 362880.0);
        let c8 = vdupq_n_f32(1.0 / 40320.0);
        let c7 = vdupq_n_f32(1.0 / 5040.0);
        let c6 = vdupq_n_f32(1.0 / 720.0);
        let c5 = vdupq_n_f32(1.0 / 120.0);
        let c4 = vdupq_n_f32(1.0 / 24.0);
        let c3 = vdupq_n_f32(1.0 / 6.0);
        let c2 = vdupq_n_f32(0.5);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let k = vrndnq_f32(vmulq_f32(x, inv_ln2));

            let mut r = vfmsq_f32(x, k, ln2_hi);
            r = vfmsq_f32(r, k, ln2_lo);

            let mut p = vfmaq_f32(c8, c9, r);
            p = vfmaq_f32(c7, p, r);
            p = vfmaq_f32(c6, p, r);
            p = vfmaq_f32(c5, p, r);
            p = vfmaq_f32(c4, p, r);
            p = vfmaq_f32(c3, p, r);
            p = vfmaq_f32(c2, p, r);
            let poly = vfmaq_f32(r, vmulq_f32(r, r), p);

            let ki = vcvtq_s32_f32(k);
            let k1 = vshrq_n_s32(ki, 1);
            let k2 = vsubq_s32(ki, k1);

            let bias = vdupq_n_s32(127);
            let scale1 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k1, bias), 23));
            let scale2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k2, bias), 23));
            let twok = vmulq_f32(scale1, scale2);
            let twok_minus_one = vsubq_f32(twok, one);
            let res_normal = vfmaq_f32(twok_minus_one, twok, poly);

            let poly_exp = vaddq_f32(one, poly);
            let res_large = vmulq_f32(vmulq_f32(poly_exp, scale1), scale2);
            let is_large_k = vcgtq_f32(k, vdupq_n_f32(24.5));
            let res_k_nonzero = vbslq_f32(is_large_k, res_large, res_normal);

            let mut p_dir = vfmaq_f32(c8, c9, x);
            p_dir = vfmaq_f32(c7, p_dir, x);
            p_dir = vfmaq_f32(c6, p_dir, x);
            p_dir = vfmaq_f32(c5, p_dir, x);
            p_dir = vfmaq_f32(c4, p_dir, x);
            p_dir = vfmaq_f32(c3, p_dir, x);
            p_dir = vfmaq_f32(c2, p_dir, x);
            let poly_direct = vfmaq_f32(x, vmulq_f32(x, x), p_dir);

            let is_small = vcltq_f32(vabsq_f32(x), vdupq_n_f32(0.6931472));
            let res = vbslq_f32(is_small, poly_direct, res_k_nonzero);

            let overflow_mask = vcgtq_f32(x, max_exp);
            let underflow_mask = vcltq_f32(x, min_expm1);
            let is_nan = vmvnq_u32(vceqq_f32(x, x));

            let mut final_res = vbslq_f32(overflow_mask, inf, res);
            final_res = vbslq_f32(underflow_mask, neg_one, final_res);
            final_res = vbslq_f32(is_nan, nan, final_res);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.exp_m1();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn log_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let ln2_hi = vdupq_n_f32(LN2_HI_F32);
        let ln2_lo = vdupq_n_f32(LN2_LO_F32);
        let sqrt2_half = vdupq_n_f32(std::f32::consts::SQRT_2 * 0.5);
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
        let pos_inf = vdupq_n_f32(f32::INFINITY);
        let nan = vdupq_n_f32(f32::NAN);
        let min_normal = vdupq_n_f32(1.17549435e-38);

        let c4 = vdupq_n_f32(2.0 / 9.0);
        let c3 = vdupq_n_f32(2.0 / 7.0);
        let c2 = vdupq_n_f32(2.0 / 5.0);
        let c1 = vdupq_n_f32(2.0 / 3.0);
        let c0 = vdupq_n_f32(2.0);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));

            let is_subnormal = vandq_u32(vcltq_f32(x, min_normal), vcgtq_f32(x, zero));
            if vgetq_lane_u32::<0>(is_subnormal) != 0
                || vgetq_lane_u32::<1>(is_subnormal) != 0
                || vgetq_lane_u32::<2>(is_subnormal) != 0
                || vgetq_lane_u32::<3>(is_subnormal) != 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).ln();
                }
                i += 4;
                continue;
            }

            let x_bits = vreinterpretq_u32_f32(x);
            let exp_bits = vshrq_n_u32(x_bits, 23);
            let mut k = vsubq_f32(
                vcvtq_f32_s32(vreinterpretq_s32_u32(exp_bits)),
                vdupq_n_f32(127.0),
            );

            let mant_mask = vdupq_n_u32(0x007F_FFFF);
            let exp_127 = vdupq_n_u32(0x3F80_0000);
            let mant_bits = vorrq_u32(vandq_u32(x_bits, mant_mask), exp_127);
            let mut m = vmulq_f32(vreinterpretq_f32_u32(mant_bits), vdupq_n_f32(0.5));
            k = vaddq_f32(k, one);

            let lt_mask = vcltq_f32(m, sqrt2_half);
            m = vbslq_f32(lt_mask, vmulq_f32(m, two), m);
            k = vbslq_f32(lt_mask, vsubq_f32(k, one), k);

            let u = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
            let u2 = vmulq_f32(u, u);

            let mut p = vfmaq_f32(c3, c4, u2);
            p = vfmaq_f32(c2, p, u2);
            p = vfmaq_f32(c1, p, u2);
            p = vfmaq_f32(c0, p, u2);
            let ln_m = vmulq_f32(u, p);

            let res = vfmaq_f32(vfmaq_f32(ln_m, k, ln2_lo), k, ln2_hi);

            let eq_zero = vceqq_f32(x, zero);
            let lt_zero = vcltq_f32(x, zero);
            let is_inf = vceqq_f32(x, pos_inf);
            let is_nan = vmvnq_u32(vceqq_f32(x, x));

            let mut final_res = vbslq_f32(eq_zero, neg_inf, res);
            final_res = vbslq_f32(lt_zero, nan, final_res);
            final_res = vbslq_f32(is_inf, pos_inf, final_res);
            final_res = vbslq_f32(is_nan, nan, final_res);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.ln();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn log2_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let log2_e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let sqrt2_half = vdupq_n_f32(std::f32::consts::SQRT_2 * 0.5);
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
        let pos_inf = vdupq_n_f32(f32::INFINITY);
        let nan = vdupq_n_f32(f32::NAN);
        let min_normal = vdupq_n_f32(1.17549435e-38);

        let c4 = vdupq_n_f32(2.0 / 9.0);
        let c3 = vdupq_n_f32(2.0 / 7.0);
        let c2 = vdupq_n_f32(2.0 / 5.0);
        let c1 = vdupq_n_f32(2.0 / 3.0);
        let c0 = vdupq_n_f32(2.0);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));

            let is_subnormal = vandq_u32(vcltq_f32(x, min_normal), vcgtq_f32(x, zero));
            if vgetq_lane_u32::<0>(is_subnormal) != 0
                || vgetq_lane_u32::<1>(is_subnormal) != 0
                || vgetq_lane_u32::<2>(is_subnormal) != 0
                || vgetq_lane_u32::<3>(is_subnormal) != 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).log2();
                }
                i += 4;
                continue;
            }

            let x_bits = vreinterpretq_u32_f32(x);
            let exp_bits = vshrq_n_u32(x_bits, 23);
            let mut k = vsubq_f32(
                vcvtq_f32_s32(vreinterpretq_s32_u32(exp_bits)),
                vdupq_n_f32(127.0),
            );

            let mant_mask = vdupq_n_u32(0x007F_FFFF);
            let exp_127 = vdupq_n_u32(0x3F80_0000);
            let mant_bits = vorrq_u32(vandq_u32(x_bits, mant_mask), exp_127);
            let mut m = vmulq_f32(vreinterpretq_f32_u32(mant_bits), vdupq_n_f32(0.5));
            k = vaddq_f32(k, one);

            let lt_mask = vcltq_f32(m, sqrt2_half);
            m = vbslq_f32(lt_mask, vmulq_f32(m, two), m);
            k = vbslq_f32(lt_mask, vsubq_f32(k, one), k);

            let u = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
            let u2 = vmulq_f32(u, u);

            let mut p = vfmaq_f32(c3, c4, u2);
            p = vfmaq_f32(c2, p, u2);
            p = vfmaq_f32(c1, p, u2);
            p = vfmaq_f32(c0, p, u2);
            let ln_m = vmulq_f32(u, p);

            let res = vfmaq_f32(k, ln_m, log2_e);

            let eq_zero = vceqq_f32(x, zero);
            let lt_zero = vcltq_f32(x, zero);
            let is_inf = vceqq_f32(x, pos_inf);
            let is_nan = vmvnq_u32(vceqq_f32(x, x));

            let mut final_res = vbslq_f32(eq_zero, neg_inf, res);
            final_res = vbslq_f32(lt_zero, nan, final_res);
            final_res = vbslq_f32(is_inf, pos_inf, final_res);
            final_res = vbslq_f32(is_nan, nan, final_res);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.log2();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn log10_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let log10_2_hi = vdupq_n_f32(LOG10_2_HI_F32);
        let log10_2_lo = vdupq_n_f32(LOG10_2_LO_F32);
        let log10_e = vdupq_n_f32(std::f32::consts::LOG10_E);
        let sqrt2_half = vdupq_n_f32(std::f32::consts::SQRT_2 * 0.5);
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
        let pos_inf = vdupq_n_f32(f32::INFINITY);
        let nan = vdupq_n_f32(f32::NAN);
        let min_normal = vdupq_n_f32(1.17549435e-38);

        let c4 = vdupq_n_f32(2.0 / 9.0);
        let c3 = vdupq_n_f32(2.0 / 7.0);
        let c2 = vdupq_n_f32(2.0 / 5.0);
        let c1 = vdupq_n_f32(2.0 / 3.0);
        let c0 = vdupq_n_f32(2.0);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));

            let is_subnormal = vandq_u32(vcltq_f32(x, min_normal), vcgtq_f32(x, zero));
            if vgetq_lane_u32::<0>(is_subnormal) != 0
                || vgetq_lane_u32::<1>(is_subnormal) != 0
                || vgetq_lane_u32::<2>(is_subnormal) != 0
                || vgetq_lane_u32::<3>(is_subnormal) != 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).log10();
                }
                i += 4;
                continue;
            }

            let x_bits = vreinterpretq_u32_f32(x);
            let exp_bits = vshrq_n_u32(x_bits, 23);
            let mut k = vsubq_f32(
                vcvtq_f32_s32(vreinterpretq_s32_u32(exp_bits)),
                vdupq_n_f32(127.0),
            );

            let mant_mask = vdupq_n_u32(0x007F_FFFF);
            let exp_127 = vdupq_n_u32(0x3F80_0000);
            let mant_bits = vorrq_u32(vandq_u32(x_bits, mant_mask), exp_127);
            let mut m = vmulq_f32(vreinterpretq_f32_u32(mant_bits), vdupq_n_f32(0.5));
            k = vaddq_f32(k, one);

            let lt_mask = vcltq_f32(m, sqrt2_half);
            m = vbslq_f32(lt_mask, vmulq_f32(m, two), m);
            k = vbslq_f32(lt_mask, vsubq_f32(k, one), k);

            let u = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
            let u2 = vmulq_f32(u, u);

            let mut p = vfmaq_f32(c3, c4, u2);
            p = vfmaq_f32(c2, p, u2);
            p = vfmaq_f32(c1, p, u2);
            p = vfmaq_f32(c0, p, u2);
            let ln_m = vmulq_f32(u, p);

            let res = vfmaq_f32(
                vfmaq_f32(vmulq_f32(k, log10_2_lo), ln_m, log10_e),
                k,
                log10_2_hi,
            );

            let eq_zero = vceqq_f32(x, zero);
            let lt_zero = vcltq_f32(x, zero);
            let is_inf = vceqq_f32(x, pos_inf);
            let is_nan = vmvnq_u32(vceqq_f32(x, x));

            let mut final_res = vbslq_f32(eq_zero, neg_inf, res);
            final_res = vbslq_f32(lt_zero, nan, final_res);
            final_res = vbslq_f32(is_inf, pos_inf, final_res);
            final_res = vbslq_f32(is_nan, nan, final_res);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.log10();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn log1p_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let ln2_hi = vdupq_n_f32(LN2_HI_F32);
        let ln2_lo = vdupq_n_f32(LN2_LO_F32);
        let sqrt2_half = vdupq_n_f32(std::f32::consts::SQRT_2 * 0.5);
        let one = vdupq_n_f32(1.0);
        let neg_one = vdupq_n_f32(-1.0);
        let two = vdupq_n_f32(2.0);
        let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
        let pos_inf = vdupq_n_f32(f32::INFINITY);
        let nan = vdupq_n_f32(f32::NAN);
        let threshold = vdupq_n_f32(0.35);

        let c4 = vdupq_n_f32(2.0 / 9.0);
        let c3 = vdupq_n_f32(2.0 / 7.0);
        let c2 = vdupq_n_f32(2.0 / 5.0);
        let c1 = vdupq_n_f32(2.0 / 3.0);
        let c0 = vdupq_n_f32(2.0);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let abs_x = vabsq_f32(x);
            let is_small = vcltq_f32(abs_x, threshold);

            let u_s = vdivq_f32(x, vaddq_f32(two, x));
            let u2_s = vmulq_f32(u_s, u_s);
            let mut p_s = vfmaq_f32(c3, c4, u2_s);
            p_s = vfmaq_f32(c2, p_s, u2_s);
            p_s = vfmaq_f32(c1, p_s, u2_s);
            p_s = vfmaq_f32(c0, p_s, u2_s);
            let res_small = vmulq_f32(u_s, p_s);

            let y = vaddq_f32(one, x);
            let y_bits = vreinterpretq_u32_f32(y);
            let exp_bits = vshrq_n_u32(y_bits, 23);
            let mut k = vsubq_f32(
                vcvtq_f32_s32(vreinterpretq_s32_u32(exp_bits)),
                vdupq_n_f32(127.0),
            );

            let mant_mask = vdupq_n_u32(0x007F_FFFF);
            let exp_127 = vdupq_n_u32(0x3F80_0000);
            let mant_bits = vorrq_u32(vandq_u32(y_bits, mant_mask), exp_127);
            let mut m = vmulq_f32(vreinterpretq_f32_u32(mant_bits), vdupq_n_f32(0.5));
            k = vaddq_f32(k, one);

            let lt_mask = vcltq_f32(m, sqrt2_half);
            m = vbslq_f32(lt_mask, vmulq_f32(m, two), m);
            k = vbslq_f32(lt_mask, vsubq_f32(k, one), k);

            let u_l = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
            let u2_l = vmulq_f32(u_l, u_l);

            let mut p_l = vfmaq_f32(c3, c4, u2_l);
            p_l = vfmaq_f32(c2, p_l, u2_l);
            p_l = vfmaq_f32(c1, p_l, u2_l);
            p_l = vfmaq_f32(c0, p_l, u2_l);
            let ln_m = vmulq_f32(u_l, p_l);
            let res_large = vfmaq_f32(vfmaq_f32(ln_m, k, ln2_lo), k, ln2_hi);

            let res = vbslq_f32(is_small, res_small, res_large);

            let eq_neg_one = vceqq_f32(x, neg_one);
            let lt_neg_one = vcltq_f32(x, neg_one);
            let is_inf = vceqq_f32(x, pos_inf);
            let is_nan = vmvnq_u32(vceqq_f32(x, x));
            let is_zero = vceqq_f32(x, zero);

            let mut final_res = vbslq_f32(eq_neg_one, neg_inf, res);
            final_res = vbslq_f32(lt_neg_one, nan, final_res);
            final_res = vbslq_f32(is_inf, pos_inf, final_res);
            final_res = vbslq_f32(is_nan, nan, final_res);
            final_res = vbslq_f32(is_zero, x, final_res);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.ln_1p();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sin_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let p1 = vdupq_n_f32(PIO2_1_F32);
        let p2 = vdupq_n_f32(PIO2_2_F32);
        let p3 = vdupq_n_f32(PIO2_3_F32);
        let p4 = vdupq_n_f32(PIO2_4_F32);
        let inv_pio2 = vdupq_n_f32(INV_PIO2_F32);
        let one = vdupq_n_f32(1.0);
        let bound = vdupq_n_f32(1e5);

        let s4 = vdupq_n_f32(1.0 / 362880.0);
        let s3 = vdupq_n_f32(-1.0 / 5040.0);
        let s2 = vdupq_n_f32(1.0 / 120.0);
        let s1 = vdupq_n_f32(-1.0 / 6.0);

        let c5 = vdupq_n_f32(-1.0 / 3628800.0);
        let c4 = vdupq_n_f32(1.0 / 40320.0);
        let c3 = vdupq_n_f32(-1.0 / 720.0);
        let c2 = vdupq_n_f32(1.0 / 24.0);
        let c1 = vdupq_n_f32(-0.5);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let abs_x = vabsq_f32(x);
            let in_range = vcltq_f32(abs_x, bound);
            if vgetq_lane_u32::<0>(in_range) == 0
                || vgetq_lane_u32::<1>(in_range) == 0
                || vgetq_lane_u32::<2>(in_range) == 0
                || vgetq_lane_u32::<3>(in_range) == 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).sin();
                }
                i += 4;
                continue;
            }

            let k = vrndnq_f32(vmulq_f32(x, inv_pio2));
            let mut r = vfmsq_f32(x, k, p1);
            r = vfmsq_f32(r, k, p2);
            r = vfmsq_f32(r, k, p3);
            r = vfmsq_f32(r, k, p4);

            let r_bound = vdupq_n_f32(1e-3);
            let abs_r = vabsq_f32(r);
            let r_ok = vcgeq_f32(abs_r, r_bound);
            if vgetq_lane_u32::<0>(r_ok) == 0
                || vgetq_lane_u32::<1>(r_ok) == 0
                || vgetq_lane_u32::<2>(r_ok) == 0
                || vgetq_lane_u32::<3>(r_ok) == 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).sin();
                }
                i += 4;
                continue;
            }

            let r2 = vmulq_f32(r, r);

            let mut sp = vfmaq_f32(s3, s4, r2);
            sp = vfmaq_f32(s2, sp, r2);
            sp = vfmaq_f32(s1, sp, r2);
            let sin_r = vfmaq_f32(r, vmulq_f32(r, r2), sp);

            let mut cp = vfmaq_f32(c4, c5, r2);
            cp = vfmaq_f32(c3, cp, r2);
            cp = vfmaq_f32(c2, cp, r2);
            cp = vfmaq_f32(c1, cp, r2);
            let cos_r = vfmaq_f32(one, r2, cp);

            let ki = vcvtq_s32_f32(k);
            let q = vandq_s32(ki, vdupq_n_s32(3));

            let is_q1_or_q3 = vceqq_s32(vandq_s32(q, vdupq_n_s32(1)), vdupq_n_s32(1));
            let is_q2_or_q3 = vcgtq_s32(q, vdupq_n_s32(1));

            let mag = vbslq_f32(is_q1_or_q3, cos_r, sin_r);
            let neg_mag = vnegq_f32(mag);
            let res = vbslq_f32(is_q2_or_q3, neg_mag, mag);

            vst1q_f32(dst.add(i), res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.sin();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn cos_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let p1 = vdupq_n_f32(PIO2_1_F32);
        let p2 = vdupq_n_f32(PIO2_2_F32);
        let p3 = vdupq_n_f32(PIO2_3_F32);
        let p4 = vdupq_n_f32(PIO2_4_F32);
        let inv_pio2 = vdupq_n_f32(INV_PIO2_F32);
        let one = vdupq_n_f32(1.0);
        let bound = vdupq_n_f32(1e5);

        let s4 = vdupq_n_f32(1.0 / 362880.0);
        let s3 = vdupq_n_f32(-1.0 / 5040.0);
        let s2 = vdupq_n_f32(1.0 / 120.0);
        let s1 = vdupq_n_f32(-1.0 / 6.0);

        let c5 = vdupq_n_f32(-1.0 / 3628800.0);
        let c4 = vdupq_n_f32(1.0 / 40320.0);
        let c3 = vdupq_n_f32(-1.0 / 720.0);
        let c2 = vdupq_n_f32(1.0 / 24.0);
        let c1 = vdupq_n_f32(-0.5);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let abs_x = vabsq_f32(x);
            let in_range = vcltq_f32(abs_x, bound);
            if vgetq_lane_u32::<0>(in_range) == 0
                || vgetq_lane_u32::<1>(in_range) == 0
                || vgetq_lane_u32::<2>(in_range) == 0
                || vgetq_lane_u32::<3>(in_range) == 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).cos();
                }
                i += 4;
                continue;
            }

            let k = vrndnq_f32(vmulq_f32(x, inv_pio2));
            let mut r = vfmsq_f32(x, k, p1);
            r = vfmsq_f32(r, k, p2);
            r = vfmsq_f32(r, k, p3);
            r = vfmsq_f32(r, k, p4);

            let r_bound = vdupq_n_f32(1e-3);
            let abs_r = vabsq_f32(r);
            let r_ok = vcgeq_f32(abs_r, r_bound);
            if vgetq_lane_u32::<0>(r_ok) == 0
                || vgetq_lane_u32::<1>(r_ok) == 0
                || vgetq_lane_u32::<2>(r_ok) == 0
                || vgetq_lane_u32::<3>(r_ok) == 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).cos();
                }
                i += 4;
                continue;
            }

            let r2 = vmulq_f32(r, r);

            let mut sp = vfmaq_f32(s3, s4, r2);
            sp = vfmaq_f32(s2, sp, r2);
            sp = vfmaq_f32(s1, sp, r2);
            let sin_r = vfmaq_f32(r, vmulq_f32(r, r2), sp);

            let mut cp = vfmaq_f32(c4, c5, r2);
            cp = vfmaq_f32(c3, cp, r2);
            cp = vfmaq_f32(c2, cp, r2);
            cp = vfmaq_f32(c1, cp, r2);
            let cos_r = vfmaq_f32(one, r2, cp);

            let ki = vcvtq_s32_f32(k);
            let q = vandq_s32(vaddq_s32(ki, vdupq_n_s32(1)), vdupq_n_s32(3));

            let is_q1_or_q3 = vceqq_s32(vandq_s32(q, vdupq_n_s32(1)), vdupq_n_s32(1));
            let is_q2_or_q3 = vcgtq_s32(q, vdupq_n_s32(1));

            let mag = vbslq_f32(is_q1_or_q3, cos_r, sin_r);
            let neg_mag = vnegq_f32(mag);
            let res = vbslq_f32(is_q2_or_q3, neg_mag, mag);

            vst1q_f32(dst.add(i), res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.cos();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn tan_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let p1 = vdupq_n_f32(PIO2_1_F32);
        let p2 = vdupq_n_f32(PIO2_2_F32);
        let p3 = vdupq_n_f32(PIO2_3_F32);
        let p4 = vdupq_n_f32(PIO2_4_F32);
        let inv_pio2 = vdupq_n_f32(INV_PIO2_F32);
        let one = vdupq_n_f32(1.0);
        let bound = vdupq_n_f32(1e5);

        let s4 = vdupq_n_f32(1.0 / 362880.0);
        let s3 = vdupq_n_f32(-1.0 / 5040.0);
        let s2 = vdupq_n_f32(1.0 / 120.0);
        let s1 = vdupq_n_f32(-1.0 / 6.0);

        let c5 = vdupq_n_f32(-1.0 / 3628800.0);
        let c4 = vdupq_n_f32(1.0 / 40320.0);
        let c3 = vdupq_n_f32(-1.0 / 720.0);
        let c2 = vdupq_n_f32(1.0 / 24.0);
        let c1 = vdupq_n_f32(-0.5);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let abs_x = vabsq_f32(x);
            let in_range = vcltq_f32(abs_x, bound);
            if vgetq_lane_u32::<0>(in_range) == 0
                || vgetq_lane_u32::<1>(in_range) == 0
                || vgetq_lane_u32::<2>(in_range) == 0
                || vgetq_lane_u32::<3>(in_range) == 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).tan();
                }
                i += 4;
                continue;
            }

            let k = vrndnq_f32(vmulq_f32(x, inv_pio2));
            let mut r = vfmsq_f32(x, k, p1);
            r = vfmsq_f32(r, k, p2);
            r = vfmsq_f32(r, k, p3);
            r = vfmsq_f32(r, k, p4);

            let r_bound = vdupq_n_f32(1e-3);
            let abs_r = vabsq_f32(r);
            let r_ok = vcgeq_f32(abs_r, r_bound);
            if vgetq_lane_u32::<0>(r_ok) == 0
                || vgetq_lane_u32::<1>(r_ok) == 0
                || vgetq_lane_u32::<2>(r_ok) == 0
                || vgetq_lane_u32::<3>(r_ok) == 0
            {
                for j in 0..4 {
                    *dst.add(i + j) = (*src.add(i + j)).tan();
                }
                i += 4;
                continue;
            }

            let r2 = vmulq_f32(r, r);

            let mut sp = vfmaq_f32(s3, s4, r2);
            sp = vfmaq_f32(s2, sp, r2);
            sp = vfmaq_f32(s1, sp, r2);
            let sin_r = vfmaq_f32(r, vmulq_f32(r, r2), sp);

            let mut cp = vfmaq_f32(c4, c5, r2);
            cp = vfmaq_f32(c3, cp, r2);
            cp = vfmaq_f32(c2, cp, r2);
            cp = vfmaq_f32(c1, cp, r2);
            let cos_r = vfmaq_f32(one, r2, cp);

            let ki = vcvtq_s32_f32(k);
            let is_odd = vceqq_s32(vandq_s32(ki, vdupq_n_s32(1)), vdupq_n_s32(1));

            let num = vbslq_f32(is_odd, vnegq_f32(cos_r), sin_r);
            let den = vbslq_f32(is_odd, sin_r, cos_r);
            let res = vdivq_f32(num, den);

            vst1q_f32(dst.add(i), res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.tan();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn tanh_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let ten = vdupq_n_f32(10.0);
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let nan = vdupq_n_f32(f32::NAN);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let abs_x = vabsq_f32(x);
            let is_neg = vcltq_f32(x, zero);
            let y = vaddq_f32(abs_x, abs_x);

            let mut em1 = [0.0f32; 4];
            let mut y_arr = [0.0f32; 4];
            vst1q_f32(y_arr.as_mut_ptr(), y);
            expm1_f32_neon(y_arr.as_ptr(), em1.as_mut_ptr(), 4);
            let em1_vec = vld1q_f32(em1.as_ptr());

            let th = vdivq_f32(em1_vec, vaddq_f32(em1_vec, two));
            let signed_th = vbslq_f32(is_neg, vnegq_f32(th), th);

            let large_mask = vcgeq_f32(abs_x, ten);
            let sat = vbslq_f32(is_neg, vnegq_f32(one), one);
            let res = vbslq_f32(large_mask, sat, signed_th);

            let is_nan = vmvnq_u32(vceqq_f32(x, x));
            let is_zero = vceqq_f32(x, zero);
            let mut final_res = vbslq_f32(is_nan, nan, res);
            final_res = vbslq_f32(is_zero, x, final_res);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = x.tanh();
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sigmoid_f32_neon(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // SAFETY: caller guarantees valid pointer access of length len.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let twenty = vdupq_n_f32(20.0);
        let min_exp = vdupq_n_f32(-88.0);
        let one = vdupq_n_f32(1.0);
        let nan = vdupq_n_f32(f32::NAN);

        while i + 4 <= len {
            let x = vld1q_f32(src.add(i));
            let abs_x = vabsq_f32(x);
            let neg_abs_x = vnegq_f32(abs_x);

            let mut neg_abs = [0.0f32; 4];
            let mut exp_vals = [0.0f32; 4];
            vst1q_f32(neg_abs.as_mut_ptr(), neg_abs_x);
            exp_f32_neon(neg_abs.as_ptr(), exp_vals.as_mut_ptr(), 4);
            let e = vld1q_f32(exp_vals.as_ptr());

            let denom = vaddq_f32(one, e);
            let s = vdivq_f32(one, denom);
            let s_neg = vmulq_f32(e, s);

            let is_positive = vcgeq_f32(x, zero);
            let res = vbslq_f32(is_positive, s, s_neg);

            let is_huge_pos = vcgtq_f32(x, twenty);
            let is_huge_neg = vcltq_f32(x, min_exp);
            let is_nan = vmvnq_u32(vceqq_f32(x, x));

            let mut final_res = vbslq_f32(is_huge_pos, one, res);
            final_res = vbslq_f32(is_huge_neg, zero, final_res);
            final_res = vbslq_f32(is_nan, nan, final_res);

            vst1q_f32(dst.add(i), final_res);
            i += 4;
        }

        while i < len {
            let x = *src.add(i);
            *dst.add(i) = scalar_sigmoid_f32(x);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch Helper Macros
// ---------------------------------------------------------------------------

macro_rules! define_dispatch_f64 {
    (
        $name:ident,
        $inplace_name:ident,
        $avx2_fn:ident,
        $neon_fn:ident,
        $scalar_op:ident,
        $doc_str:literal
    ) => {
        #[doc = $doc_str]
        pub(crate) fn $name(src: &[f64], dst: &mut [f64]) {
            assert_eq!(
                src.len(),
                dst.len(),
                "src and dst slices must have identical length"
            );
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2_fma() {
                    // SAFETY: lengths checked equal, CPU features verified
                    unsafe {
                        $avx2_fn(src.as_ptr(), dst.as_mut_ptr(), src.len());
                    }
                    return;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if has_neon() {
                    // SAFETY: lengths checked equal, CPU features verified
                    unsafe {
                        $neon_fn(src.as_ptr(), dst.as_mut_ptr(), src.len());
                    }
                    return;
                }
            }
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.$scalar_op();
            }
        }

        #[doc = concat!("In-place version of `", stringify!($name), "`.")]
        pub(crate) fn $inplace_name(slice: &mut [f64]) {
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2_fma() {
                    // SAFETY: slice is valid for reading and writing of length len
                    unsafe {
                        $avx2_fn(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
                    }
                    return;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if has_neon() {
                    // SAFETY: slice is valid for reading and writing of length len
                    unsafe {
                        $neon_fn(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
                    }
                    return;
                }
            }
            for d in slice.iter_mut() {
                *d = d.$scalar_op();
            }
        }
    };
}

macro_rules! define_dispatch_f32 {
    (
        $name:ident,
        $inplace_name:ident,
        $avx2_fn:ident,
        $neon_fn:ident,
        $scalar_op:ident,
        $doc_str:literal
    ) => {
        #[doc = $doc_str]
        pub(crate) fn $name(src: &[f32], dst: &mut [f32]) {
            assert_eq!(
                src.len(),
                dst.len(),
                "src and dst slices must have identical length"
            );
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2_fma() {
                    // SAFETY: lengths checked equal, CPU features verified
                    unsafe {
                        $avx2_fn(src.as_ptr(), dst.as_mut_ptr(), src.len());
                    }
                    return;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if has_neon() {
                    // SAFETY: lengths checked equal, CPU features verified
                    unsafe {
                        $neon_fn(src.as_ptr(), dst.as_mut_ptr(), src.len());
                    }
                    return;
                }
            }
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.$scalar_op();
            }
        }

        #[doc = concat!("In-place version of `", stringify!($name), "`.")]
        pub(crate) fn $inplace_name(slice: &mut [f32]) {
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2_fma() {
                    // SAFETY: slice valid for reading and writing of length len
                    unsafe {
                        $avx2_fn(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
                    }
                    return;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if has_neon() {
                    // SAFETY: slice valid for reading and writing of length len
                    unsafe {
                        $neon_fn(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
                    }
                    return;
                }
            }
            for d in slice.iter_mut() {
                *d = d.$scalar_op();
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Exported Function Definitions (f64)
// ---------------------------------------------------------------------------

define_dispatch_f64!(
    exp_f64,
    exp_f64_inplace,
    exp_f64_avx2,
    exp_f64_neon,
    exp,
    "Computes the exponential function for each element, like np.exp."
);

define_dispatch_f64!(
    exp2_f64,
    exp2_f64_inplace,
    exp2_f64_avx2,
    exp2_f64_neon,
    exp2,
    "Computes base-2 exponential function for each element, like np.exp2."
);

define_dispatch_f64!(
    expm1_f64,
    expm1_f64_inplace,
    expm1_f64_avx2,
    expm1_f64_neon,
    exp_m1,
    "Computes exp(x) - 1 for each element, like np.expm1."
);

define_dispatch_f64!(
    log_f64,
    log_f64_inplace,
    log_f64_avx2,
    log_f64_neon,
    ln,
    "Computes the natural logarithm for each element, like np.log."
);

define_dispatch_f64!(
    log2_f64,
    log2_f64_inplace,
    log2_f64_avx2,
    log2_f64_neon,
    log2,
    "Computes the base-2 logarithm for each element, like np.log2."
);

define_dispatch_f64!(
    log10_f64,
    log10_f64_inplace,
    log10_f64_avx2,
    log10_f64_neon,
    log10,
    "Computes the base-10 logarithm for each element, like np.log10."
);

define_dispatch_f64!(
    log1p_f64,
    log1p_f64_inplace,
    log1p_f64_avx2,
    log1p_f64_neon,
    ln_1p,
    "Computes ln(1 + x) for each element, like np.log1p."
);

define_dispatch_f64!(
    sin_f64,
    sin_f64_inplace,
    sin_f64_avx2,
    sin_f64_neon,
    sin,
    "Computes trigonometric sine for each element, like np.sin."
);

define_dispatch_f64!(
    cos_f64,
    cos_f64_inplace,
    cos_f64_avx2,
    cos_f64_neon,
    cos,
    "Computes trigonometric cosine for each element, like np.cos."
);

define_dispatch_f64!(
    tan_f64,
    tan_f64_inplace,
    tan_f64_avx2,
    tan_f64_neon,
    tan,
    "Computes trigonometric tangent for each element, like np.tan."
);

define_dispatch_f64!(
    tanh_f64,
    tanh_f64_inplace,
    tanh_f64_avx2,
    tanh_f64_neon,
    tanh,
    "Computes hyperbolic tangent for each element, like np.tanh."
);

/// Computes the logistic sigmoid 1 / (1 + exp(-x)) for each element, like scipy.special.expit.
pub(crate) fn sigmoid_f64(src: &[f64], dst: &mut [f64]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "src and dst slices must have identical length"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: lengths checked equal, CPU features verified
            unsafe {
                sigmoid_f64_avx2(src.as_ptr(), dst.as_mut_ptr(), src.len());
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            // SAFETY: lengths checked equal, CPU features verified
            unsafe {
                sigmoid_f64_neon(src.as_ptr(), dst.as_mut_ptr(), src.len());
            }
            return;
        }
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = scalar_sigmoid_f64(*s);
    }
}

/// In-place version of `sigmoid_f64`.
pub(crate) fn sigmoid_f64_inplace(slice: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: slice valid for reading and writing of length len
            unsafe {
                sigmoid_f64_avx2(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            // SAFETY: slice valid for reading and writing of length len
            unsafe {
                sigmoid_f64_neon(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
            }
            return;
        }
    }
    for d in slice.iter_mut() {
        *d = scalar_sigmoid_f64(*d);
    }
}

// ---------------------------------------------------------------------------
// Exported Function Definitions (f32)
// ---------------------------------------------------------------------------

define_dispatch_f32!(
    exp_f32,
    exp_f32_inplace,
    exp_f32_avx2,
    exp_f32_neon,
    exp,
    "Computes the exponential function for each element, like np.exp."
);

define_dispatch_f32!(
    exp2_f32,
    exp2_f32_inplace,
    exp2_f32_avx2,
    exp2_f32_neon,
    exp2,
    "Computes base-2 exponential function for each element, like np.exp2."
);

define_dispatch_f32!(
    expm1_f32,
    expm1_f32_inplace,
    expm1_f32_avx2,
    expm1_f32_neon,
    exp_m1,
    "Computes exp(x) - 1 for each element, like np.expm1."
);

define_dispatch_f32!(
    log_f32,
    log_f32_inplace,
    log_f32_avx2,
    log_f32_neon,
    ln,
    "Computes the natural logarithm for each element, like np.log."
);

define_dispatch_f32!(
    log2_f32,
    log2_f32_inplace,
    log2_f32_avx2,
    log2_f32_neon,
    log2,
    "Computes the base-2 logarithm for each element, like np.log2."
);

define_dispatch_f32!(
    log10_f32,
    log10_f32_inplace,
    log10_f32_avx2,
    log10_f32_neon,
    log10,
    "Computes the base-10 logarithm for each element, like np.log10."
);

define_dispatch_f32!(
    log1p_f32,
    log1p_f32_inplace,
    log1p_f32_avx2,
    log1p_f32_neon,
    ln_1p,
    "Computes ln(1 + x) for each element, like np.log1p."
);

define_dispatch_f32!(
    sin_f32,
    sin_f32_inplace,
    sin_f32_avx2,
    sin_f32_neon,
    sin,
    "Computes trigonometric sine for each element, like np.sin."
);

define_dispatch_f32!(
    cos_f32,
    cos_f32_inplace,
    cos_f32_avx2,
    cos_f32_neon,
    cos,
    "Computes trigonometric cosine for each element, like np.cos."
);

define_dispatch_f32!(
    tan_f32,
    tan_f32_inplace,
    tan_f32_avx2,
    tan_f32_neon,
    tan,
    "Computes trigonometric tangent for each element, like np.tan."
);

define_dispatch_f32!(
    tanh_f32,
    tanh_f32_inplace,
    tanh_f32_avx2,
    tanh_f32_neon,
    tanh,
    "Computes hyperbolic tangent for each element, like np.tanh."
);

/// Computes the logistic sigmoid 1 / (1 + exp(-x)) for each f32 element, like scipy.special.expit.
pub(crate) fn sigmoid_f32(src: &[f32], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "src and dst slices must have identical length"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: lengths checked equal, CPU features verified
            unsafe {
                sigmoid_f32_avx2(src.as_ptr(), dst.as_mut_ptr(), src.len());
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            // SAFETY: lengths checked equal, CPU features verified
            unsafe {
                sigmoid_f32_neon(src.as_ptr(), dst.as_mut_ptr(), src.len());
            }
            return;
        }
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = scalar_sigmoid_f32(*s);
    }
}

/// In-place version of `sigmoid_f32`.
pub(crate) fn sigmoid_f32_inplace(slice: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: slice valid for reading and writing of length len
            unsafe {
                sigmoid_f32_avx2(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            // SAFETY: slice valid for reading and writing of length len
            unsafe {
                sigmoid_f32_neon(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
            }
            return;
        }
    }
    for d in slice.iter_mut() {
        *d = scalar_sigmoid_f32(*d);
    }
}
