//! Register micro-kernels: `acc = sum_p a[p] (MR) x b[p] (NR)` over packed panels.
//!
//! Each element type has a portable scalar kernel and, on x86, an AVX2+FMA kernel selected once
//! at runtime. Two entry points exist per kernel:
//! * `raw` stores the finished accumulators row-major into a small tile (used for edge tiles and
//!   non-unit column strides of C);
//! * `fused` (AVX2 only) applies `C = alpha * acc + beta * C` straight from the registers for full
//!   tiles whose C rows are contiguous, so the common case never touches an intermediate tile.

use super::kernels::KernelFloat;

/// `raw(k, a_panel, b_panel, tile)` writes `MR * NR` accumulators, row-major, into `tile`.
pub(super) type RawKernel<F> = unsafe fn(usize, *const F, *const F, *mut F);
/// `fused(k, a_panel, b_panel, alpha, beta, c, row_stride)` for a full `MR x NR` tile with unit
/// column stride. `beta == 0` must not read `c`.
pub(super) type FusedKernel<F> = unsafe fn(usize, *const F, *const F, F, F, *mut F, isize);

/// The kernels chosen for one element type.
#[derive(Clone, Copy)]
pub(super) struct Kernels<F> {
    pub raw: RawKernel<F>,
    pub fused: Option<FusedKernel<F>>,
}

/// Portable kernel; written so LLVM can vectorise the inner `j` loop for any target
/// (this is also what runs on aarch64 until a NEON kernel exists).
///
/// # Safety
/// `a` holds `k * MR` packed elements, `b` holds `k * NR`, and `tile` has `MR * NR` writable slots.
pub(super) unsafe fn scalar<F: KernelFloat, const MR: usize, const NR: usize>(
    k: usize,
    a: *const F,
    b: *const F,
    tile: *mut F,
) {
    let mut acc = [[F::ZERO; NR]; MR];
    for p in 0..k {
        for (i, row) in acc.iter_mut().enumerate() {
            // SAFETY: p < k, i < MR and j < NR stay inside the packed panels (caller contract).
            let ai = unsafe { *a.add(p * MR + i) };
            for (j, value) in row.iter_mut().enumerate() {
                // SAFETY: as above.
                *value = *value + ai * unsafe { *b.add(p * NR + j) };
            }
        }
    }
    for (i, row) in acc.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            // SAFETY: tile has MR * NR slots (caller contract).
            unsafe { *tile.add(i * NR + j) = value };
        }
    }
}

/// Runtime feature check, evaluated once and cached.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_fma() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    })
}

pub(super) fn f64_kernels() -> Kernels<f64> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if avx2_fma() {
        return Kernels {
            raw: avx2::d6x8_raw,
            fused: Some(avx2::d6x8_fused),
        };
    }
    Kernels {
        raw: scalar::<f64, 6, 8>,
        fused: None,
    }
}

pub(super) fn f32_kernels() -> Kernels<f32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if avx2_fma() {
        return Kernels {
            raw: avx2::s6x16_raw,
            fused: Some(avx2::s6x16_fused),
        };
    }
    Kernels {
        raw: scalar::<f32, 6, 16>,
        fused: None,
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // Both kernels keep 12 accumulators (6 rows x 2 vectors) in registers. The depth loop is
    // unrolled by two so the loop counter and branch are amortised over 24 FMAs.
    macro_rules! accumulate {
        ($k:expr, $a:expr, $b:expr, $nr:expr, $half:expr, $zero:ident, $load:ident,
         $set1:ident, $fma:ident) => {{
            let (k, a, b) = ($k, $a, $b);
            let mut acc = [$zero(); 12];
            let mut p = 0;
            macro_rules! step {
                ($q:expr) => {{
                    let q = $q;
                    let b0 = $load(b.add(q * $nr));
                    let b1 = $load(b.add(q * $nr + $half));
                    for i in 0..6 {
                        let ai = $set1(*a.add(q * 6 + i));
                        acc[2 * i] = $fma(ai, b0, acc[2 * i]);
                        acc[2 * i + 1] = $fma(ai, b1, acc[2 * i + 1]);
                    }
                }};
            }
            while p + 2 <= k {
                step!(p);
                step!(p + 1);
                p += 2;
            }
            if p < k {
                step!(p);
            }
            acc
        }};
    }

    /// Starts loading the six C rows of the tile (`bytes` wide each) so the long depth loop hides
    /// the cache-miss latency of the final read-modify-write. Prefetches never fault.
    #[inline]
    unsafe fn prefetch_rows<T>(c: *mut T, rs: isize, bytes: usize) {
        for i in 0..6 {
            let row = c.wrapping_offset(i as isize * rs).cast::<i8>();
            // SAFETY: prefetch hints are not memory accesses and cannot fault.
            unsafe {
                _mm_prefetch::<_MM_HINT_T0>(row);
                _mm_prefetch::<_MM_HINT_T0>(row.wrapping_add(bytes - 1));
            }
        }
    }

    /// # Safety
    /// AVX2+FMA available; `a` holds `6 * k`, `b` holds `8 * k` packed elements.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn d6x8_acc(k: usize, a: *const f64, b: *const f64) -> [__m256d; 12] {
        // SAFETY: reads stay within the packed panels (caller contract).
        unsafe {
            accumulate!(
                k,
                a,
                b,
                8,
                4,
                _mm256_setzero_pd,
                _mm256_loadu_pd,
                _mm256_set1_pd,
                _mm256_fmadd_pd
            )
        }
    }

    /// # Safety
    /// AVX2+FMA available; packed panels as for `d6x8_acc`; `tile` has 48 writable slots.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn d6x8_raw(k: usize, a: *const f64, b: *const f64, tile: *mut f64) {
        // SAFETY: tile has 6 * 8 slots; each store writes 4.
        unsafe {
            let acc = d6x8_acc(k, a, b);
            for (i, v) in acc.iter().enumerate() {
                _mm256_storeu_pd(tile.add(i * 4), *v);
            }
        }
    }

    /// # Safety
    /// AVX2+FMA available; packed panels as above; `c + i * rs + 0..8` is writable for `i < 6`
    /// (and readable when `beta != 0`).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn d6x8_fused(
        k: usize,
        a: *const f64,
        b: *const f64,
        alpha: f64,
        beta: f64,
        c: *mut f64,
        rs: isize,
    ) {
        // SAFETY: C rows are contiguous 8-element runs inside the caller-owned tile.
        unsafe {
            prefetch_rows(c, rs, 8 * 8);
            let acc = d6x8_acc(k, a, b);
            let va = _mm256_set1_pd(alpha);
            let vb = _mm256_set1_pd(beta);
            for i in 0..6 {
                let row = c.offset(i as isize * rs);
                let mut x0 = _mm256_mul_pd(va, acc[2 * i]);
                let mut x1 = _mm256_mul_pd(va, acc[2 * i + 1]);
                if beta != 0.0 {
                    x0 = _mm256_fmadd_pd(vb, _mm256_loadu_pd(row), x0);
                    x1 = _mm256_fmadd_pd(vb, _mm256_loadu_pd(row.add(4)), x1);
                }
                _mm256_storeu_pd(row, x0);
                _mm256_storeu_pd(row.add(4), x1);
            }
        }
    }

    /// # Safety
    /// AVX2+FMA available; `a` holds `6 * k`, `b` holds `16 * k` packed elements.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn s6x16_acc(k: usize, a: *const f32, b: *const f32) -> [__m256; 12] {
        // SAFETY: reads stay within the packed panels (caller contract).
        unsafe {
            accumulate!(
                k,
                a,
                b,
                16,
                8,
                _mm256_setzero_ps,
                _mm256_loadu_ps,
                _mm256_set1_ps,
                _mm256_fmadd_ps
            )
        }
    }

    /// # Safety
    /// AVX2+FMA available; packed panels as for `s6x16_acc`; `tile` has 96 writable slots.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn s6x16_raw(k: usize, a: *const f32, b: *const f32, tile: *mut f32) {
        // SAFETY: tile has 6 * 16 slots; each store writes 8.
        unsafe {
            let acc = s6x16_acc(k, a, b);
            for (i, v) in acc.iter().enumerate() {
                _mm256_storeu_ps(tile.add(i * 8), *v);
            }
        }
    }

    /// # Safety
    /// AVX2+FMA available; packed panels as above; `c + i * rs + 0..16` is writable for `i < 6`
    /// (and readable when `beta != 0`).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn s6x16_fused(
        k: usize,
        a: *const f32,
        b: *const f32,
        alpha: f32,
        beta: f32,
        c: *mut f32,
        rs: isize,
    ) {
        // SAFETY: C rows are contiguous 16-element runs inside the caller-owned tile.
        unsafe {
            prefetch_rows(c, rs, 16 * 4);
            let acc = s6x16_acc(k, a, b);
            let va = _mm256_set1_ps(alpha);
            let vb = _mm256_set1_ps(beta);
            for i in 0..6 {
                let row = c.offset(i as isize * rs);
                let mut x0 = _mm256_mul_ps(va, acc[2 * i]);
                let mut x1 = _mm256_mul_ps(va, acc[2 * i + 1]);
                if beta != 0.0 {
                    x0 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(row), x0);
                    x1 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(row.add(8)), x1);
                }
                _mm256_storeu_ps(row, x0);
                _mm256_storeu_ps(row.add(8), x1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn dispatched_kernels_match_scalar_on_exact_small_integers() {
        for k in [0usize, 1, 7, 255, 256, 257] {
            let a: Vec<f64> = (0..k * 6).map(|i| (i % 7) as f64 - 3.0).collect();
            let b: Vec<f64> = (0..k * 8).map(|i| (i % 5) as f64 - 2.0).collect();
            let (mut portable, mut native) = ([0.0; 48], [0.0; 48]);
            let kernels = f64_kernels();
            // SAFETY: buffers hold complete packed panels and a full 6 x 8 tile.
            unsafe {
                scalar::<f64, 6, 8>(k, a.as_ptr(), b.as_ptr(), portable.as_mut_ptr());
                (kernels.raw)(k, a.as_ptr(), b.as_ptr(), native.as_mut_ptr());
            }
            assert_eq!(portable, native);
            if let Some(fused) = kernels.fused {
                // c = 2 * acc + 3 * c with a padded row stride of 11.
                let mut c = vec![1.0; 6 * 11];
                // SAFETY: 6 rows of 8 writable/readable elements at stride 11.
                unsafe { fused(k, a.as_ptr(), b.as_ptr(), 2.0, 3.0, c.as_mut_ptr(), 11) };
                for i in 0..6 {
                    for j in 0..8 {
                        assert_eq!(c[i * 11 + j], 2.0 * portable[i * 8 + j] + 3.0);
                    }
                    assert_eq!(&c[i * 11 + 8..i * 11 + 11], &[1.0; 3]);
                }
            }
            let a: Vec<f32> = a.iter().map(|&x| x as f32).collect();
            let b: Vec<f32> = (0..k * 16).map(|i| (i % 5) as f32 - 2.0).collect();
            let (mut portable, mut native) = ([0.0f32; 96], [0.0f32; 96]);
            let kernels = f32_kernels();
            // SAFETY: buffers hold complete packed panels and a full 6 x 16 tile.
            unsafe {
                scalar::<f32, 6, 16>(k, a.as_ptr(), b.as_ptr(), portable.as_mut_ptr());
                (kernels.raw)(k, a.as_ptr(), b.as_ptr(), native.as_mut_ptr());
            }
            assert_eq!(portable, native);
            if let Some(fused) = kernels.fused {
                let mut c = vec![f32::NAN; 6 * 16];
                // SAFETY: beta = 0 never reads c; 6 rows of 16 writable elements.
                unsafe { fused(k, a.as_ptr(), b.as_ptr(), 1.0, 0.0, c.as_mut_ptr(), 16) };
                assert_eq!(&c[..], &portable[..]);
            }
        }
    }
}
