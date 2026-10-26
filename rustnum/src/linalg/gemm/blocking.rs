//! Cache-blocking table for the packed GEMM, tuned for an Intel Core Ultra 5 125H
//! (hybrid: P-cores with 48 KB L1d and 2 MB L2, E-core clusters sharing 2 MB L2, 18 MB shared L3).
//!
//! Sizes are in elements. `mc` and `nc` are the *largest* task rectangle; the driver shrinks
//! them for small problems so that every core gets several tasks. All values are multiples of the
//! register tile (`MR`, `NR`).

/// Cache block sizes for one element type.
#[derive(Clone, Copy)]
pub(crate) struct Blocking {
    /// Rows of A packed into one block (A block = `mc * kc` elements).
    pub mc: usize,
    /// Depth of one rank-`kc` update; also the C re-read/re-write frequency.
    pub kc: usize,
    /// Columns of one task rectangle (B block read by that task = `kc * nc` elements).
    pub nc: usize,
}

/// Rows of the register tile. Six broadcasts feed two B vectors: 12 accumulators + 2 B + 1 A
/// vector = 15 of the 16 ymm registers.
pub(super) const MR: usize = 6;
/// f64 register tile columns: two 4-lane AVX2 vectors.
pub(super) const NR_F64: usize = 8;
/// f32 register tile columns: two 8-lane AVX2 vectors.
pub(super) const NR_F32: usize = 16;

pub(super) const F64: Blocking = Blocking {
    // 96 * 256 * 8 B = 192 KiB packed A: fits a P-core's 2 MiB L2 with room for the streamed B
    // panel, and stays under the ~512 KiB an E-core gets when its 2 MiB cluster L2 is shared.
    mc: 96,
    // One A micro-panel (6 * 256 * 8 = 12 KiB) plus one B micro-panel (8 * 256 * 8 = 16 KiB)
    // = 28 KiB sit in the 48 KiB L1d, leaving space for C lines and prefetch.
    kc: 256,
    // 256 * 512 * 8 B = 1 MiB B block: shared (via L2/L3) by all MC blocks of the same column
    // range, large enough that packing A is amortised over 512 columns.
    nc: 512,
};

pub(super) const F32: Blocking = Blocking {
    // 192 * 256 * 4 B = 192 KiB packed A: the same byte footprint as f64.
    mc: 192,
    // 6 * 256 * 4 = 6 KiB A + 16 * 256 * 4 = 16 KiB B micro-panels = 22 KiB of L1d.
    kc: 256,
    // 256 * 512 * 4 B = 512 KiB B block.
    nc: 512,
};

/// Widest column range whose packed B slab (`kc * 4096` elements, 8 MiB for f64) is built at once.
/// Bounds the shared workspace for very wide outputs while fitting the 18 MiB L3.
pub(super) const SLAB_COLUMNS: usize = 4096;

/// `m * n * k` multiply-adds below which the product stays on the calling thread (about a 200^3
/// product, ~0.3 ms serial): below this the rayon fork/join costs more than the extra cores save.
/// Measured with `examples/gemm_probe.rs --small`: f64 breaks even near n=128-192, f32 near 200-256.
#[cfg(feature = "rayon")]
pub(super) const PARALLEL_WORK: usize = 1 << 23;

/// The driver shrinks task rectangles until there are at least this many tasks per rayon thread,
/// so that work stealing can give fast P-cores more blocks than slow E-cores. Larger values were
/// not measurably faster (1, 2, 4 and 8 tried) and pay more packing overhead.
#[cfg(feature = "rayon")]
pub(super) const TASKS_PER_THREAD: usize = 2;
