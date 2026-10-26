//! Packing of A and B micro-panels. Both read through arbitrary signed strides; the common
//! unit-stride layouts take straight-line copy paths.
//!
//! Layouts (what the micro-kernels expect):
//! * A panel: `MR` rows, for each depth `p` the `MR` row values are adjacent (`dst[p * MR + i]`).
//! * B panel: `NR` columns, for each depth `p` the `NR` column values are adjacent (`dst[p * NR + j]`).
//!
//! Missing rows/columns of an edge panel are zero-filled, so kernels never need masked loads.

use super::kernels::KernelFloat;

/// Packs `rows <= MR` rows by `depth` columns of A into `dst` (`MR * depth` elements).
///
/// `src` points at element (first row, first depth); `rs`/`cs` are the element strides of rows
/// and columns (may be negative or zero).
///
/// # Safety
/// Every element `src + i * rs + p * cs` for `i < rows`, `p < depth` must be readable, and `dst`
/// must hold at least `MR * depth` elements.
pub(super) unsafe fn pack_a_panel<F: KernelFloat, const MR: usize>(
    src: *const F,
    rs: isize,
    cs: isize,
    rows: usize,
    depth: usize,
    dst: &mut [F],
) {
    let dst = &mut dst[..MR * depth];
    // SAFETY (all reads below): in-bounds by this function's contract; writes go through
    // bounds-checked slice indexing or to indices proven below `MR * depth`.
    unsafe {
        if rows == MR && rs == 1 {
            // Column-major A: the MR rows of one depth are contiguous.
            for p in 0..depth {
                let from = src.offset(p as isize * cs);
                for i in 0..MR {
                    *dst.get_unchecked_mut(p * MR + i) = *from.add(i);
                }
            }
        } else if cs == 1 {
            // Row-major A: each row is contiguous along depth; scatter into the panel.
            for i in 0..rows {
                let from = src.offset(i as isize * rs);
                for p in 0..depth {
                    *dst.get_unchecked_mut(p * MR + i) = *from.add(p);
                }
            }
            if rows < MR {
                for p in 0..depth {
                    for i in rows..MR {
                        *dst.get_unchecked_mut(p * MR + i) = F::ZERO;
                    }
                }
            }
        } else {
            for p in 0..depth {
                for i in 0..rows {
                    *dst.get_unchecked_mut(p * MR + i) =
                        *src.offset(i as isize * rs + p as isize * cs);
                }
                for i in rows..MR {
                    *dst.get_unchecked_mut(p * MR + i) = F::ZERO;
                }
            }
        }
    }
}

/// Packs `depth` rows by `cols <= NR` columns of B into `dst` (`NR * depth` elements).
///
/// `src` points at element (first depth, first column).
///
/// # Safety
/// Every element `src + p * rs + j * cs` for `p < depth`, `j < cols` must be readable, and `dst`
/// must hold at least `NR * depth` elements.
pub(super) unsafe fn pack_b_panel<F: KernelFloat, const NR: usize>(
    src: *const F,
    rs: isize,
    cs: isize,
    cols: usize,
    depth: usize,
    dst: &mut [F],
) {
    let dst = &mut dst[..NR * depth];
    // SAFETY (all reads below): in-bounds by this function's contract; destination indices are
    // below `NR * depth`.
    unsafe {
        if cols == NR && cs == 1 {
            // Row-major B: the NR columns of one depth are contiguous.
            for p in 0..depth {
                let from = src.offset(p as isize * rs);
                for j in 0..NR {
                    *dst.get_unchecked_mut(p * NR + j) = *from.add(j);
                }
            }
        } else if rs == 1 {
            // Column-major B: each column is contiguous along depth; scatter into the panel.
            for j in 0..cols {
                let from = src.offset(j as isize * cs);
                for p in 0..depth {
                    *dst.get_unchecked_mut(p * NR + j) = *from.add(p);
                }
            }
            if cols < NR {
                for p in 0..depth {
                    for j in cols..NR {
                        *dst.get_unchecked_mut(p * NR + j) = F::ZERO;
                    }
                }
            }
        } else {
            for p in 0..depth {
                for j in 0..cols {
                    *dst.get_unchecked_mut(p * NR + j) =
                        *src.offset(p as isize * rs + j as isize * cs);
                }
                for j in cols..NR {
                    *dst.get_unchecked_mut(p * NR + j) = F::ZERO;
                }
            }
        }
    }
}
