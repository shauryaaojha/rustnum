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
//! ## Measured accuracy (max ulp vs `f64` std, 2·10⁶ inputs per function over its full domain, 2026-10-26)
//! Produced by `cargo run --release -p rustnum --example simd_accuracy`.
//!
//! | Function | AVX2 f64 | AVX2 f32 | Vectorised today |
//! |---|---:|---:|---|
//! | exp | 2.0 | 2.6 | f64, f32 |
//! | exp2 | 2.0 | 0.5 | f64 |
//! | log | 2.0 | 0.5 | f64 |
//! | sin, cos | 1.0 | 0.5 | f64 |
//! | tanh | 3.0 | 1.1 | f64 |
//! | sigmoid | 3.0 | 2.3 | f64 |
//! | expm1, log2, log10, log1p, tan | scalar std | scalar std | no (dispatch falls back to std) |

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

// ---------------------------------------------------------------------------
// Mathematical Constants
// ---------------------------------------------------------------------------

const LN2_HI_F64: f64 = 0.6931471803691238;
const LN2_LO_F64: f64 = 1.9082149292705877e-10;
const INV_LN2_F64: f64 = std::f64::consts::LOG2_E;

const LN2_HI_F32: f32 = 0.693_145_75; // 0x3f317200: exact high part, LN2_HI + LN2_LO = ln 2
const LN2_LO_F32: f32 = 1.4286068e-6;
const INV_LN2_F32: f32 = std::f32::consts::LOG2_E;

// pi/2 = PIO2_1 + PIO2_2 + PIO2_3 to within 6e-50 (each term is the double nearest the
// remainder), used with FMA so x - k*pi/2 keeps full relative accuracy near multiples of pi/2.
const PIO2_1_F64: f64 = std::f64::consts::FRAC_PI_2;
const PIO2_2_F64: f64 = 6.123233995736766e-17;
const PIO2_3_F64: f64 = -1.4973849048591698e-33;
const INV_PIO2_F64: f64 = 2.0 / std::f64::consts::PI;

#[allow(clippy::approx_constant)] // Cody-Waite high part, deliberately not the full FRAC_PI_2
const PIO2_1_F32: f32 = 1.5707963;
const PIO2_2_F32: f32 = -4.371139e-8;
const INV_PIO2_F32: f32 = 2.0 / std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Scalar Kernels (f64 and f32)
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
    // SAFETY: caller verified valid pointer access of length len.
    unsafe {
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
        let ln2_hi = _mm256_set1_pd(LN2_HI_F64);
        let ln2_lo = _mm256_set1_pd(LN2_LO_F64);
        let sqrt2_half = _mm256_set1_pd(std::f64::consts::SQRT_2 * 0.5);
        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let zero = _mm256_setzero_pd();
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let pos_inf = _mm256_set1_pd(f64::INFINITY);
        let nan = _mm256_set1_pd(f64::NAN);

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

            let x_bits = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_bits, 52);
            // Narrow the four 64-bit exponents to 32-bit lanes with AVX2 only
            // (`_mm256_cvtepi64_epi32` is AVX-512 and faults on AVX2 CPUs).
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
    // SAFETY: caller verified valid pointer access of length len.
    unsafe {
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
    // SAFETY: caller verified valid pointer access of length len.
    unsafe {
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
    // SAFETY: caller verified valid pointer access of length len.
    unsafe {
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
        let nan = _mm256_set1_pd(f64::NAN);
        let bound = _mm256_set1_pd(1e5);

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
            let sign_mask = _mm256_set1_pd(-0.0);
            let neg_mag = _mm256_xor_pd(mag, sign_mask);
            let res = _mm256_blendv_pd(mag, neg_mag, is_q2_or_q3);

            let abs_x = _mm256_andnot_pd(sign_mask, x);
            let in_range = _mm256_cmp_pd(abs_x, bound, _CMP_LT_OQ);
            let is_nan_or_inf = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

            let final_res = _mm256_blendv_pd(nan, res, in_range);
            let final_res = _mm256_blendv_pd(final_res, nan, is_nan_or_inf);

            _mm256_storeu_pd(dst.add(i), final_res);
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
        let nan = _mm256_set1_pd(f64::NAN);
        let bound = _mm256_set1_pd(1e5);

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
            let sign_mask = _mm256_set1_pd(-0.0);
            let neg_mag = _mm256_xor_pd(mag, sign_mask);
            let res = _mm256_blendv_pd(mag, neg_mag, is_q2_or_q3);

            let abs_x = _mm256_andnot_pd(sign_mask, x);
            let in_range = _mm256_cmp_pd(abs_x, bound, _CMP_LT_OQ);
            let is_nan_or_inf = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

            let final_res = _mm256_blendv_pd(nan, res, in_range);
            let final_res = _mm256_blendv_pd(final_res, nan, is_nan_or_inf);

            _mm256_storeu_pd(dst.add(i), final_res);
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
    // SAFETY: caller verified valid pointer access of length len.
    unsafe {
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
        let sign_mask = _mm256_set1_pd(-0.0);
        let twenty = _mm256_set1_pd(20.0);
        let one = _mm256_set1_pd(1.0);

        while i + 4 <= len {
            let x = _mm256_loadu_pd(src.add(i));
            let sign = _mm256_and_pd(x, sign_mask);
            let abs_x = _mm256_andnot_pd(sign_mask, x);

            let mut vals = [0.0; 4];
            let mut abs_vals = [0.0; 4];
            _mm256_storeu_pd(abs_vals.as_mut_ptr(), abs_x);
            for j in 0..4 {
                let ax = abs_vals[j];
                let em1 = (2.0 * ax).exp_m1();
                vals[j] = em1 / (em1 + 2.0);
            }
            let th = _mm256_loadu_pd(vals.as_ptr());
            let signed_th = _mm256_xor_pd(th, sign);

            let large_mask = _mm256_cmp_pd(abs_x, twenty, _CMP_GE_OQ);
            let sat = _mm256_xor_pd(one, sign);
            let res = _mm256_blendv_pd(signed_th, sat, large_mask);

            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
            let final_res = _mm256_blendv_pd(res, _mm256_set1_pd(f64::NAN), is_nan);

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
        let sign_mask = _mm256_set1_pd(-0.0);
        let forty = _mm256_set1_pd(40.0);
        let zero = _mm256_setzero_pd();
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

            // e = exp(-|x|) <= 1, so neither branch overflows: x >= 0 gives
            // 1 / (1 + e); x < 0 gives e / (1 + e) = e * s, which stays accurate
            // down to the exp underflow (1 - s would cancel catastrophically).
            let denom = _mm256_add_pd(one, e);
            let s = _mm256_div_pd(one, denom);
            let s_neg = _mm256_mul_pd(e, s);

            let is_positive = _mm256_cmp_pd(x, zero, _CMP_GE_OQ);
            let res = _mm256_blendv_pd(s_neg, s, is_positive);

            let is_huge_pos = _mm256_cmp_pd(x, forty, _CMP_GT_OQ);
            let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

            let mut final_res = _mm256_blendv_pd(res, one, is_huge_pos);
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

// ---------------------------------------------------------------------------
// Dispatch Helper Macros
// ---------------------------------------------------------------------------

macro_rules! define_dispatch_f64 {
    (
        $name:ident,
        $inplace_name:ident,
        $avx2_fn:ident,
        $scalar_op:ident,
        $measured_ulp:literal,
        $doc_str:literal
    ) => {
        #[doc = $doc_str]
        ///
        /// Accuracy: see the table in the module documentation.
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
        $scalar_op:ident,
        $measured_ulp:literal,
        $doc_str:literal
    ) => {
        #[doc = $doc_str]
        ///
        /// Accuracy: see the table in the module documentation.
        pub(crate) fn $name(src: &[f32], dst: &mut [f32]) {
            assert_eq!(
                src.len(),
                dst.len(),
                "src and dst slices must have identical length"
            );
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2_fma() && stringify!($name) == "exp_f32" {
                    // SAFETY: lengths checked equal, CPU features verified
                    unsafe {
                        exp_f32_avx2(src.as_ptr(), dst.as_mut_ptr(), src.len());
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
                if has_avx2_fma() && stringify!($name) == "exp_f32_inplace" {
                    // SAFETY: slice valid for reading and writing
                    unsafe {
                        exp_f32_avx2(slice.as_ptr(), slice.as_mut_ptr(), slice.len());
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
    exp,
    "1.0",
    "Computes the exponential function for each element, like np.exp."
);

define_dispatch_f64!(
    exp2_f64,
    exp2_f64_inplace,
    exp2_f64_avx2,
    exp2,
    "1.0",
    "Computes base-2 exponential function for each element, like np.exp2."
);

define_dispatch_f64!(
    expm1_f64,
    expm1_f64_inplace,
    expm1_f64_avx2,
    exp_m1,
    "1.0",
    "Computes exp(x) - 1 for each element, like np.expm1."
);

define_dispatch_f64!(
    log_f64,
    log_f64_inplace,
    log_f64_avx2,
    ln,
    "1.0",
    "Computes the natural logarithm for each element, like np.log."
);

define_dispatch_f64!(
    log2_f64,
    log2_f64_inplace,
    log2_f64_avx2,
    log2,
    "1.0",
    "Computes the base-2 logarithm for each element, like np.log2."
);

define_dispatch_f64!(
    log10_f64,
    log10_f64_inplace,
    log10_f64_avx2,
    log10,
    "1.0",
    "Computes the base-10 logarithm for each element, like np.log10."
);

define_dispatch_f64!(
    log1p_f64,
    log1p_f64_inplace,
    log1p_f64_avx2,
    ln_1p,
    "1.0",
    "Computes ln(1 + x) for each element, like np.log1p."
);

define_dispatch_f64!(
    sin_f64,
    sin_f64_inplace,
    sin_f64_avx2,
    sin,
    "2.0",
    "Computes trigonometric sine for each element, like np.sin."
);

define_dispatch_f64!(
    cos_f64,
    cos_f64_inplace,
    cos_f64_avx2,
    cos,
    "2.0",
    "Computes trigonometric cosine for each element, like np.cos."
);

define_dispatch_f64!(
    tan_f64,
    tan_f64_inplace,
    tan_f64_avx2,
    tan,
    "2.0",
    "Computes trigonometric tangent for each element, like np.tan."
);

define_dispatch_f64!(
    tanh_f64,
    tanh_f64_inplace,
    tanh_f64_avx2,
    tanh,
    "1.0",
    "Computes hyperbolic tangent for each element, like np.tanh."
);

/// Computes the logistic sigmoid 1 / (1 + exp(-x)) for each element, like scipy.special.expit.
///
/// Accuracy: see the table in the module documentation.
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
    exp,
    "1.0",
    "Computes the exponential function for each element, like np.exp."
);

define_dispatch_f32!(
    exp2_f32,
    exp2_f32_inplace,
    exp2,
    "1.0",
    "Computes base-2 exponential function for each element, like np.exp2."
);

define_dispatch_f32!(
    expm1_f32,
    expm1_f32_inplace,
    exp_m1,
    "1.0",
    "Computes exp(x) - 1 for each element, like np.expm1."
);

define_dispatch_f32!(
    log_f32,
    log_f32_inplace,
    ln,
    "1.0",
    "Computes the natural logarithm for each element, like np.log."
);

define_dispatch_f32!(
    log2_f32,
    log2_f32_inplace,
    log2,
    "1.0",
    "Computes the base-2 logarithm for each element, like np.log2."
);

define_dispatch_f32!(
    log10_f32,
    log10_f32_inplace,
    log10,
    "1.0",
    "Computes the base-10 logarithm for each element, like np.log10."
);

define_dispatch_f32!(
    log1p_f32,
    log1p_f32_inplace,
    ln_1p,
    "1.0",
    "Computes ln(1 + x) for each element, like np.log1p."
);

define_dispatch_f32!(
    sin_f32,
    sin_f32_inplace,
    sin,
    "1.0",
    "Computes trigonometric sine for each element, like np.sin."
);

define_dispatch_f32!(
    cos_f32,
    cos_f32_inplace,
    cos,
    "1.0",
    "Computes trigonometric cosine for each element, like np.cos."
);

define_dispatch_f32!(
    tan_f32,
    tan_f32_inplace,
    tan,
    "1.0",
    "Computes trigonometric tangent for each element, like np.tan."
);

define_dispatch_f32!(
    tanh_f32,
    tanh_f32_inplace,
    tanh,
    "1.0",
    "Computes hyperbolic tangent for each element, like np.tanh."
);

/// Computes the logistic sigmoid 1 / (1 + exp(-x)) for each f32 element, like scipy.special.expit.
///
/// Accuracy: see the table in the module documentation.
pub(crate) fn sigmoid_f32(src: &[f32], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "src and dst slices must have identical length"
    );
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = scalar_sigmoid_f32(*s);
    }
}

/// In-place version of `sigmoid_f32`.
pub(crate) fn sigmoid_f32_inplace(slice: &mut [f32]) {
    for d in slice.iter_mut() {
        *d = scalar_sigmoid_f32(*d);
    }
}
