//! Packed, stride-aware BLIS-style GEMM for `f32` and `f64`: `C = alpha * A * B + beta * C`,
//! optionally followed by an element-wise map applied while each finished tile is in cache.
//!
//! Structure (loops from the outside in):
//! 1. `js`: column slabs of at most [`blocking::SLAB_COLUMNS`] columns of B/C.
//! 2. `pc`: depth blocks of `kc`; the packed B slab (`kc x slab`) is built once per block, in
//!    parallel, and shared read-only by every task.
//! 3. tasks over `(MC block, NC block)` rectangles of C, spread over rayon. A task packs its own
//!    A block (`mc x kc`) into a thread-local buffer, then walks its rectangle in `MR x NR`
//!    register tiles, each computed by a micro-kernel and merged into C immediately.
//!
//! Only this module owns raw output pointers; tasks write disjoint rectangles of C.

mod blocking;
mod microkernel;
mod packing;

use super::kernels::{self, KernelFloat};
use blocking::{Blocking, MR, NR_F32, NR_F64, SLAB_COLUMNS};
use microkernel::Kernels;
use std::cell::RefCell;

/// Storage plus a logical origin; unlike a tail slice this represents negative strides.
#[derive(Clone, Copy)]
pub(crate) struct Matrix<'a, F> {
    pub data: &'a [F],
    pub offset: usize,
    pub row: isize,
    pub col: isize,
}

impl<F> Matrix<'_, F> {
    fn check(self, rows: usize, cols: usize) {
        check(self.data.len(), self.offset, rows, cols, self.row, self.col);
    }

    /// Address of logical element `(i, j)`; only dereferenceable for in-range elements.
    fn ptr_at(self, i: usize, j: usize) -> *const F {
        self.data
            .as_ptr()
            .wrapping_offset(self.offset as isize + i as isize * self.row + j as isize * self.col)
    }
}

/// Panics unless every element `(i < rows, j < cols)` of the strided matrix lies inside `len`.
fn check(len: usize, offset: usize, rows: usize, cols: usize, row: isize, col: isize) {
    if rows == 0 || cols == 0 {
        return;
    }
    let r = (rows - 1) as i128 * row as i128;
    let c = (cols - 1) as i128 * col as i128;
    let low = offset as i128 + r.min(0) + c.min(0);
    let high = offset as i128 + r.max(0) + c.max(0);
    assert!(
        low >= 0 && high < len as i128,
        "matrix strides are outside the supplied slice"
    );
}

#[derive(Clone, Copy)]
struct Output<F> {
    ptr: *mut F,
    row: isize,
    col: isize,
}
// SAFETY: `run` validates bounds and injective output strides, and within one parallel region
// every task owns a disjoint (MC, NC) rectangle; regions are joined before the next begins.
unsafe impl<F: Send> Send for Output<F> {}
// SAFETY: the only concurrent accesses are to the disjoint rectangles described above.
unsafe impl<F: Send> Sync for Output<F> {}

impl<F> Output<F> {
    fn ptr_at(self, i: usize, j: usize) -> *mut F {
        self.ptr
            .wrapping_offset(i as isize * self.row + j as isize * self.col)
    }
}

/// One validated product. Fields are private; the type is only named in [`GemmFloat`].
pub(crate) struct Problem<'a, F> {
    m: usize,
    k: usize,
    n: usize,
    alpha: F,
    beta: F,
    a: Matrix<'a, F>,
    b: Matrix<'a, F>,
    c: Output<F>,
    map: Option<fn(F) -> F>,
}

/// Element types with a packed GEMM (`f32`, `f64`).
pub(crate) trait GemmFloat: KernelFloat + Send + Sync + 'static {
    #[doc(hidden)]
    const BLOCK: Blocking;
    #[doc(hidden)]
    fn drive(problem: &Problem<'_, Self>);
    #[doc(hidden)]
    fn take_a() -> Vec<Self>;
    #[doc(hidden)]
    fn give_a(buffer: Vec<Self>);
    #[doc(hidden)]
    fn take_b() -> Vec<Self>;
    #[doc(hidden)]
    fn give_b(buffer: Vec<Self>);
}

macro_rules! impl_gemm_float {
    ($t:ty, $nr:expr, $block:expr, $kernels:path, $buffers:ident) => {
        thread_local! {
            // (packed A block, packed B slab), reused across calls to avoid allocation.
            static $buffers: (RefCell<Vec<$t>>, RefCell<Vec<$t>>) =
                const { (RefCell::new(Vec::new()), RefCell::new(Vec::new())) };
        }
        impl GemmFloat for $t {
            const BLOCK: Blocking = $block;
            fn drive(problem: &Problem<'_, $t>) {
                drive::<$t, MR, $nr>(problem, $kernels());
            }
            // Buffers are taken (not borrowed) so a re-entrant call, for example from inside the
            // epilogue map, simply allocates its own instead of panicking on a RefCell.
            fn take_a() -> Vec<$t> {
                $buffers
                    .try_with(|b| std::mem::take(&mut *b.0.borrow_mut()))
                    .unwrap_or_default()
            }
            fn give_a(buffer: Vec<$t>) {
                let _ = $buffers.try_with(|b| *b.0.borrow_mut() = buffer);
            }
            fn take_b() -> Vec<$t> {
                $buffers
                    .try_with(|b| std::mem::take(&mut *b.1.borrow_mut()))
                    .unwrap_or_default()
            }
            fn give_b(buffer: Vec<$t>) {
                let _ = $buffers.try_with(|b| *b.1.borrow_mut() = buffer);
            }
        }
    };
}
impl_gemm_float!(
    f64,
    NR_F64,
    blocking::F64,
    microkernel::f64_kernels,
    F64_BUFFERS
);
impl_gemm_float!(
    f32,
    NR_F32,
    blocking::F32,
    microkernel::f32_kernels,
    F32_BUFFERS
);

/// Safe internal entry point: `C = alpha * A * B + beta * C`, then `map` on every element.
///
/// A, B and C are described by slices plus signed element strides (negative and broadcast
/// strides are fine for the inputs). `beta == 0` never reads C and `alpha == 0` never reads A or
/// B, as in BLAS. Panics when a strided matrix leaves its slice or C's strides alias.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<F: GemmFloat>(
    (m, k, n): (usize, usize, usize),
    alpha: F,
    a: Matrix<'_, F>,
    b: Matrix<'_, F>,
    beta: F,
    c: &mut [F],
    offset: usize,
    strides: (isize, isize),
    map: Option<fn(F) -> F>,
) {
    a.check(m, k);
    b.check(k, n);
    check(c.len(), offset, m, n, strides.0, strides.1);
    if m == 0 || n == 0 {
        return;
    }
    assert!(
        kernels::output_is_non_aliasing(m, n, strides.0, strides.1),
        "GEMM output strides must not alias"
    );
    let problem = Problem {
        m,
        k,
        n,
        alpha,
        beta,
        a,
        b,
        c: Output {
            ptr: c.as_mut_ptr().wrapping_add(offset),
            row: strides.0,
            col: strides.1,
        },
        map,
    };
    if k == 0 || alpha == F::ZERO {
        scale(&problem);
    } else {
        F::drive(&problem);
    }
}

/// `C = beta * C` (then map) for products with no contribution from A and B.
fn scale<F: GemmFloat>(p: &Problem<'_, F>) {
    for i in 0..p.m {
        for j in 0..p.n {
            // SAFETY: (i, j) is inside the validated, non-aliasing output; `run` holds `&mut`.
            unsafe {
                let dst = p.c.ptr_at(i, j);
                let mut value = if p.beta == F::ZERO {
                    F::ZERO
                } else {
                    p.beta * *dst
                };
                if let Some(f) = p.map {
                    value = f(value);
                }
                *dst = value;
            }
        }
    }
}

/// Task rectangle sizes `(mc, nc)` and whether to use rayon.
fn plan<const NR: usize>(
    p_m: usize,
    p_k: usize,
    p_n: usize,
    blk: Blocking,
) -> (usize, usize, bool) {
    #[cfg_attr(not(feature = "rayon"), allow(unused_mut))]
    let mut mc = blk.mc.min(p_m.next_multiple_of(MR));
    #[cfg_attr(not(feature = "rayon"), allow(unused_mut))]
    let mut nc = blk.nc.min(p_n.min(SLAB_COLUMNS).next_multiple_of(NR));
    #[cfg(feature = "rayon")]
    {
        let threads = rayon::current_num_threads();
        let work = p_m.saturating_mul(p_n).saturating_mul(p_k);
        if threads > 1 && work >= blocking::PARALLEL_WORK {
            // Split the larger side until every thread can expect several tasks, but keep
            // blocks big enough to amortise packing.
            let (min_mc, min_nc) = (4 * MR, 64usize.next_multiple_of(NR));
            let wanted = blocking::TASKS_PER_THREAD * threads;
            let slab = p_n.min(SLAB_COLUMNS);
            while p_m.div_ceil(mc) * slab.div_ceil(nc) < wanted {
                if nc > 2 * mc && nc / 2 >= min_nc {
                    nc = (nc / 2).next_multiple_of(NR);
                } else if mc / 2 >= min_mc {
                    mc = (mc / 2).next_multiple_of(MR);
                } else if nc / 2 >= min_nc {
                    nc = (nc / 2).next_multiple_of(NR);
                } else {
                    break;
                }
            }
            return (mc, nc, true);
        }
    }
    let _ = p_k;
    (mc, nc, false)
}

/// Runs `task(0..count)`, on rayon when `parallel`.
fn run_tasks(count: usize, parallel: bool, task: impl Fn(usize) + Send + Sync) {
    #[cfg(feature = "rayon")]
    if parallel && count > 1 {
        use rayon::prelude::*;
        (0..count).into_par_iter().for_each(task);
        return;
    }
    let _ = parallel;
    for t in 0..count {
        task(t);
    }
}

fn drive<F: GemmFloat, const MR: usize, const NR: usize>(p: &Problem<'_, F>, kernels: Kernels<F>) {
    let blk = F::BLOCK;
    let (mc, nc, parallel) = plan::<NR>(p.m, p.k, p.n, blk);
    let mut slab = F::take_b();
    let mut js = 0;
    while js < p.n {
        let ns = SLAB_COLUMNS.min(p.n - js);
        for pc in (0..p.k).step_by(blk.kc) {
            let kb = blk.kc.min(p.k - pc);
            let need = ns.div_ceil(NR) * NR * kb;
            if slab.len() < need {
                slab.resize(need, F::ZERO);
            }
            let packed = &mut slab[..need];
            pack_b_slab::<F, NR>(p, packed, (js, ns), (pc, kb), parallel);
            let packed: &[F] = packed;
            let mblocks = p.m.div_ceil(mc);
            let nblocks = ns.div_ceil(nc);
            run_tasks(mblocks * nblocks, parallel, |t| {
                // m varies fastest so consecutive tasks reuse the same B block.
                let (ib, jb) = (t % mblocks, t / mblocks);
                let block = Block {
                    ic: ib * mc,
                    mb: mc.min(p.m - ib * mc),
                    jc: jb * nc,
                    nb: nc.min(ns - jb * nc),
                    js,
                    pc,
                    kb,
                    first: pc == 0,
                    last: pc + kb == p.k,
                };
                run_block::<F, MR, NR>(p, &kernels, packed, &block);
            });
        }
        js += ns;
    }
    F::give_b(slab);
}

/// Packs B rows `pc..pc+kb` and columns `js..js+ns` into consecutive NR-wide panels.
fn pack_b_slab<F: GemmFloat, const NR: usize>(
    p: &Problem<'_, F>,
    packed: &mut [F],
    (js, ns): (usize, usize),
    (pc, kb): (usize, usize),
    parallel: bool,
) {
    let one = |panel: usize, chunk: &mut [F]| {
        let j0 = js + panel * NR;
        let cols = NR.min(js + ns - j0);
        // SAFETY: rows pc..pc+kb and columns j0..j0+cols are inside B (validated in `run`);
        // `chunk` holds exactly NR * kb elements.
        unsafe {
            packing::pack_b_panel::<F, NR>(p.b.ptr_at(pc, j0), p.b.row, p.b.col, cols, kb, chunk)
        };
    };
    #[cfg(feature = "rayon")]
    if parallel {
        use rayon::prelude::*;
        packed
            .par_chunks_mut(NR * kb)
            .enumerate()
            .for_each(|(panel, chunk)| one(panel, chunk));
        return;
    }
    let _ = parallel;
    for (panel, chunk) in packed.chunks_mut(NR * kb).enumerate() {
        one(panel, chunk);
    }
}

/// One `(MC, NC)` rectangle of C within the current depth block.
struct Block {
    /// First row and row count.
    ic: usize,
    mb: usize,
    /// First column (relative to `js`) and column count; `jc` is a multiple of NR.
    jc: usize,
    nb: usize,
    /// First column of the slab, first depth and depth of this block.
    js: usize,
    pc: usize,
    kb: usize,
    /// First depth block (applies `beta`) / last depth block (applies the map).
    first: bool,
    last: bool,
}

fn run_block<F: GemmFloat, const MR: usize, const NR: usize>(
    p: &Problem<'_, F>,
    kernels: &Kernels<F>,
    packed_b: &[F],
    blk: &Block,
) {
    let mut packed_a = F::take_a();
    let a_len = blk.mb.div_ceil(MR) * MR * blk.kb;
    if packed_a.len() < a_len {
        packed_a.resize(a_len, F::ZERO);
    }
    for (panel, chunk) in packed_a[..a_len].chunks_mut(MR * blk.kb).enumerate() {
        let i0 = blk.ic + panel * MR;
        let rows = MR.min(blk.ic + blk.mb - i0);
        // SAFETY: rows i0..i0+rows and depths pc..pc+kb are inside A (validated in `run`);
        // `chunk` holds exactly MR * kb elements.
        unsafe {
            packing::pack_a_panel::<F, MR>(
                p.a.ptr_at(i0, blk.pc),
                p.a.row,
                p.a.col,
                rows,
                blk.kb,
                chunk,
            )
        };
    }
    let beta = if blk.first { p.beta } else { F::ONE };
    let map = if blk.last { p.map } else { None };
    for jr in (0..blk.nb).step_by(NR) {
        let cols = NR.min(blk.nb - jr);
        let b_panel = &packed_b[(blk.jc + jr) / NR * NR * blk.kb..][..NR * blk.kb];
        for ir in (0..blk.mb).step_by(MR) {
            let rows = MR.min(blk.mb - ir);
            let a_panel = &packed_a[ir / MR * MR * blk.kb..][..MR * blk.kb];
            // SAFETY: both panels hold kb full-width columns; the destination rectangle lies in
            // C and belongs exclusively to this task for the current depth block.
            unsafe {
                micro_tile::<F, MR, NR>(
                    kernels,
                    blk.kb,
                    a_panel.as_ptr(),
                    b_panel.as_ptr(),
                    p.alpha,
                    beta,
                    p.c.ptr_at(blk.ic + ir, blk.js + blk.jc + jr),
                    (p.c.row, p.c.col),
                    (rows, cols),
                    map,
                );
            }
        }
    }
    F::give_a(packed_a);
}

/// Computes one register tile and merges it into C: `C = alpha * acc + beta * C`, then `map`.
///
/// # Safety
/// The packed panels hold `kb` columns of `MR` / `NR` values, and the `rows x cols` rectangle at
/// `c` (with the given strides) is valid for reads and writes and not accessed concurrently.
#[allow(clippy::too_many_arguments)]
unsafe fn micro_tile<F: GemmFloat, const MR: usize, const NR: usize>(
    kernels: &Kernels<F>,
    kb: usize,
    a: *const F,
    b: *const F,
    alpha: F,
    beta: F,
    c: *mut F,
    (rs, cs): (isize, isize),
    (rows, cols): (usize, usize),
    map: Option<fn(F) -> F>,
) {
    // SAFETY: see the function contract; the fused kernel needs a full tile with unit column
    // stride, otherwise the raw kernel fills a local tile that is merged element by element.
    unsafe {
        match kernels.fused {
            Some(fused) if rows == MR && cols == NR && cs == 1 => {
                fused(kb, a, b, alpha, beta, c, rs)
            }
            _ => {
                let mut tile = [[F::ZERO; NR]; MR];
                (kernels.raw)(kb, a, b, tile.as_mut_ptr().cast());
                for (r, row) in tile.iter().enumerate().take(rows) {
                    for (s, &acc) in row.iter().enumerate().take(cols) {
                        let dst = c.offset(r as isize * rs + s as isize * cs);
                        let mut value = alpha * acc;
                        if beta != F::ZERO {
                            value = value + beta * *dst;
                        }
                        *dst = value;
                    }
                }
            }
        }
        if let Some(f) = map {
            for r in 0..rows {
                for s in 0..cols {
                    let dst = c.offset(r as isize * rs + s as isize * cs);
                    *dst = f(*dst);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill<F: GemmFloat>(len: usize, seed: usize) -> Vec<F> {
        (0..len)
            .map(|i| F::from_usize((i * 31 + seed * 17) % 23) - F::from_usize(11))
            .collect()
    }

    /// Reference in f64 on the logical (strided) indices.
    #[allow(clippy::too_many_arguments)]
    fn reference<F: GemmFloat + Into<f64>>(
        (m, k, n): (usize, usize, usize),
        alpha: f64,
        a: Matrix<'_, F>,
        b: Matrix<'_, F>,
        beta: f64,
        c: &[F],
        offset: usize,
        (rs, cs): (isize, isize),
    ) -> Vec<f64> {
        let at = |mat: Matrix<'_, F>, i: usize, j: usize| -> f64 {
            mat.data[(mat.offset as isize + i as isize * mat.row + j as isize * mat.col) as usize]
                .into()
        };
        let mut out: Vec<f64> = c.iter().map(|&x| x.into()).collect();
        for i in 0..m {
            for j in 0..n {
                let dot: f64 = (0..k).map(|p| at(a, i, p) * at(b, p, j)).sum();
                let idx = (offset as isize + i as isize * rs + j as isize * cs) as usize;
                out[idx] = alpha * dot + if beta == 0.0 { 0.0 } else { beta * out[idx] };
            }
        }
        out
    }

    fn exercise<F: GemmFloat + Into<f64>>(dims: (usize, usize, usize), tol: f64) {
        let (m, k, n) = dims;
        let a_data = fill::<F>(m * k, 1);
        let b_data = fill::<F>(k * n, 2);
        // Row-major, transposed (column-major) and reversed variants of the inputs.
        let a_layouts = [
            Matrix {
                data: &a_data,
                offset: 0,
                row: k as isize,
                col: 1,
            },
            Matrix {
                data: &a_data,
                offset: 0,
                row: 1,
                col: m as isize,
            },
            Matrix {
                data: &a_data,
                offset: (m * k).saturating_sub(1),
                row: -(k as isize),
                col: -1,
            },
        ];
        let b_layouts = [
            Matrix {
                data: &b_data,
                offset: 0,
                row: n as isize,
                col: 1,
            },
            Matrix {
                data: &b_data,
                offset: 0,
                row: 1,
                col: k as isize,
            },
            Matrix {
                data: &b_data,
                offset: (k * n).saturating_sub(1),
                row: -(n as isize),
                col: -1,
            },
        ];
        for (ia, &a) in a_layouts.iter().enumerate() {
            for (ib, &b) in b_layouts.iter().enumerate() {
                if (ia == 1 && m * k == 0) || (ib == 1 && k * n == 0) {
                    continue;
                }
                // Padded row-major C and a transposed padded C.
                for (offset, strides, len) in [
                    (3usize, (n as isize + 2, 1isize), 3 + m * (n + 2)),
                    (1, (1, m as isize + 1), 1 + n * (m + 1)),
                ] {
                    for (alpha, beta) in [(1, 0), (-3, 2), (2, 1)] {
                        let c0 = fill::<F>(len, 3);
                        let want =
                            reference(dims, alpha as f64, a, b, beta as f64, &c0, offset, strides);
                        let mut c = c0.clone();
                        run(
                            dims,
                            signed::<F>(alpha),
                            a,
                            b,
                            signed::<F>(beta),
                            &mut c,
                            offset,
                            strides,
                            None,
                        );
                        for (x, y) in c.iter().zip(&want) {
                            let x: f64 = (*x).into();
                            assert!(
                                (x - y).abs() <= tol * y.abs().max(1.0),
                                "{dims:?} {ia}{ib} {x} != {y}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn signed<F: GemmFloat>(x: i32) -> F {
        let magnitude = F::from_usize(x.unsigned_abs() as usize);
        if x < 0 { -magnitude } else { magnitude }
    }

    #[test]
    fn matches_reference_across_block_boundaries_for_all_layouts() {
        for dims in [
            (1, 1, 1),
            (1, 9, 5),
            (7, 1, 3),
            (6, 8, 8),
            (13, 17, 19),
            (97, 257, 129),
            (101, 300, 41),
            (50, 3, 600),
        ] {
            exercise::<f64>(dims, 1e-12);
            exercise::<f32>(dims, 1e-4);
        }
    }

    #[test]
    fn signed_output_strides_preserve_padding() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [99.0; 15];
        run(
            (2, 2, 2),
            2.0,
            Matrix {
                data: &a,
                offset: 0,
                row: 2,
                col: 1,
            },
            Matrix {
                data: &b,
                offset: 0,
                row: 2,
                col: 1,
            },
            0.0,
            &mut c,
            9,
            (-5, -2),
            Some(|x| x + 1.0),
        );
        // [[19,22],[43,50]], scaled and mapped into reversed, padded C.
        assert_eq!(
            c,
            [
                99.0, 99.0, 101.0, 99.0, 87.0, 99.0, 99.0, 45.0, 99.0, 39.0, 99.0, 99.0, 99.0,
                99.0, 99.0
            ]
        );
    }

    #[test]
    fn map_runs_once_after_the_last_depth_block() {
        // k spans several kc blocks; a map that is not idempotent exposes repeated application.
        let (m, k, n) = (13, 700, 9);
        let a = vec![1.0f64; m * k];
        let b = vec![1.0f64; k * n];
        let mut c = vec![0.0; m * n];
        run(
            (m, k, n),
            0.5,
            Matrix {
                data: &a,
                offset: 0,
                row: k as isize,
                col: 1,
            },
            Matrix {
                data: &b,
                offset: 0,
                row: n as isize,
                col: 1,
            },
            0.0,
            &mut c,
            0,
            (n as isize, 1),
            Some(|x| x + 1.0),
        );
        assert!(c.iter().all(|&x| x == 351.0));
    }
}
