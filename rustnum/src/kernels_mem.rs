//! Memory kernels and the shared parallel crossover table (elements).
use crate::{Array, Element};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
// Parallel crossovers (elements), measured on this machine with
// `examples/memory_probe.rs thresholds` run with and without RAYON_NUM_THREADS=1:
// parallel wins from the first size listed; below it the serial loop is as fast or faster.
/// Transcendental / costly unary ufuncs (tanh, exp, ...): parallel wins from ~32k elements.
#[cfg(feature = "rayon")]
pub(crate) const UNARY: usize = 32_768;
/// Cheap memory-bound element-wise work (scalar arithmetic, negation, fill): ~2M elements.
#[cfg(feature = "rayon")]
pub(crate) const CHEAP: usize = 2_097_152;
/// Two-array element-wise work with equal shapes: serial wins up to ~1M, parallel from ~2M.
#[cfg(feature = "rayon")]
pub(crate) const BINARY: usize = 2_097_152;
/// Broadcasting binary work (row/column/general): parallel wins from ~1M elements.
#[cfg(feature = "rayon")]
pub(crate) const BROADCAST: usize = 1_048_576;
/// Strided copy / transpose: parallel wins from ~256k elements.
#[cfg(feature = "rayon")]
pub(crate) const COPY: usize = 262_144;
/// Axis reductions that stream whole rows into the output (axis 0 / middle axes): ~256k.
#[cfg(feature = "rayon")]
pub(crate) const REDUCE_ROWS: usize = 262_144;
/// Axis reductions with one independent lane per output (last axis, strided lanes): ~2M.
#[cfg(feature = "rayon")]
pub(crate) const REDUCE_LANES: usize = 2_097_152;
/// Full reductions to a scalar (sum, min, max, prod): parallel wins from ~32k elements.
#[cfg(feature = "rayon")]
pub(crate) const REDUCE_ALL: usize = 32_768;
pub(crate) fn fits(source: &[usize], target: &[usize]) -> bool {
    source.len() <= target.len()
        && source
            .iter()
            .rev()
            .zip(target.iter().rev())
            .all(|(&s, &t)| s == 1 || s == t)
}
fn row<T: Element>(a: &Array<T>, shape: &[usize], mut index: usize) -> (usize, isize) {
    let mut off = a.offset as isize;
    let pad = shape.len() - a.ndim();
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        let c = index % shape[axis];
        index /= shape[axis];
        if axis >= pad && a.shape[axis - pad] != 1 {
            off += c as isize * a.strides[axis - pad];
        }
    }
    let stride = if a.ndim() == 0 || *a.shape.last().unwrap() == 1 {
        0
    } else {
        *a.strides.last().unwrap()
    };
    (off as usize, stride)
}
pub(crate) fn zip_row<T: Element, U: Element, R: Element>(
    out: &mut [R],
    a: &Array<T>,
    b: &Array<U>,
    shape: &[usize],
    r: usize,
    f: &impl Fn(T, U) -> R,
) {
    let (ao, ast) = row(a, shape, r);
    let (bo, bst) = row(b, shape, r);
    let len = out.len();
    match (ast, bst) {
        (1, 1) => {
            for ((o, &x), &y) in out
                .iter_mut()
                .zip(&a.data[ao..ao + len])
                .zip(&b.data[bo..bo + len])
            {
                *o = f(x, y);
            }
        }
        (1, 0) => {
            let y = b.data[bo];
            let len = out.len();
            for (o, &x) in out.iter_mut().zip(&a.data[ao..ao + len]) {
                *o = f(x, y);
            }
        }
        (0, 1) => {
            let x = a.data[ao];
            let len = out.len();
            for (o, &y) in out.iter_mut().zip(&b.data[bo..bo + len]) {
                *o = f(x, y);
            }
        }
        _ => {
            for (j, o) in out.iter_mut().enumerate() {
                *o = f(
                    a.data[(ao as isize + j as isize * ast) as usize],
                    b.data[(bo as isize + j as isize * bst) as usize],
                );
            }
        }
    }
}
pub(crate) fn assign_row<T: Element>(
    out: &mut [T],
    b: &Array<T>,
    shape: &[usize],
    r: usize,
    f: &impl Fn(T, T) -> T,
) {
    let (base, step) = row(b, shape, r);
    if step == 0 {
        let y = b.data[base];
        for x in out {
            *x = f(*x, y);
        }
    } else if step == 1 {
        for (x, &y) in out.iter_mut().zip(&b.data[base..]) {
            *x = f(*x, y);
        }
    } else {
        for (i, x) in out.iter_mut().enumerate() {
            *x = f(*x, b.data[(base as isize + i as isize * step) as usize]);
        }
    }
}
pub(crate) fn copy<T: Element>(a: &Array<T>) -> Vec<T> {
    if let Some(s) = a.as_slice() {
        return crate::pool::collect(s.len(), s.iter().copied());
    }
    let n = a.size();
    let mut out = crate::pool::output(n);
    let d = a.ndim();
    // Output row slabs are disjoint. Within each slab, 32x32 tiles traverse the
    // source's fastest axis innermost, including negative strides and n-d batches.
    if d >= 2 && a.strides[d - 2].unsigned_abs() == 1 {
        let rows = a.shape[d - 2];
        let cols = a.shape[d - 1];
        if rows == 0 || cols == 0 {
            return out;
        }
        let matrix = rows * cols;
        let fill = |batch: usize, dst: &mut [T]| {
            let base = a.flat_offset(batch * matrix);
            let slab = |rblock: usize, dst: &mut [T]| {
                let r0 = rblock * 32;
                let height = dst.len() / cols;
                for c0 in (0..cols).step_by(32) {
                    for c in c0..(c0 + 32).min(cols) {
                        for r in 0..height {
                            let src = base as isize
                                + (r0 + r) as isize * a.strides[d - 2]
                                + c as isize * a.strides[d - 1];
                            dst[r * cols + c] = a.data[src as usize];
                        }
                    }
                }
            };
            #[cfg(feature = "rayon")]
            if n >= COPY {
                dst.par_chunks_mut(32 * cols)
                    .enumerate()
                    .for_each(|(i, s)| slab(i, s));
                return;
            }
            dst.chunks_mut(32 * cols)
                .enumerate()
                .for_each(|(i, s)| slab(i, s));
        };
        out.chunks_mut(matrix)
            .enumerate()
            .for_each(|(i, s)| fill(i, s));
    } else {
        // Reorder traversal within each bounded output block around the smallest
        // absolute source stride; no coordinate Vec is allocated per element.
        // Length-1 and zero-stride (broadcast) axes are poor inner loops.
        let inner = (0..d)
            .filter(|&i| a.shape[i] > 1)
            .min_by_key(|&i| match a.strides[i].unsigned_abs() {
                0 => usize::MAX,
                s => s,
            })
            .unwrap_or(d.saturating_sub(1));
        let width = if d == 0 { 1 } else { a.shape[inner] };
        let output_step = crate::layout::size(&a.shape[inner + 1..]);
        for line in 0..n / width.max(1) {
            let before = line / output_step;
            let after = line % output_step;
            let start = before * width * output_step + after;
            let base = a.flat_offset(start) as isize;
            for j in 0..width {
                out[start + j * output_step] =
                    a.data[(base + j as isize * a.strides[inner]) as usize];
            }
        }
    }
    out
}
