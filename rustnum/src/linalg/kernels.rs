//! Slice-level numerical kernels used by the future `Array` linalg adapters.
//!
//! All matrices are row-major logical matrices.  GEMM accepts element strides,
//! so a transposed `Array` view can eventually call it without materialising a
//! copy.  Other routines take contiguous row-major inputs and return owned
//! row-major results.

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::Num;
use crate::element::{add, mul};

/// The deliberately small floating-point surface needed by these kernels.
pub trait KernelFloat:
    Copy
    + Default
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const EPSILON: Self;
    const INFINITY: Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn max_of(self, other: Self) -> Self;
    fn is_finite(self) -> bool;
    fn sin_cos(self) -> (Self, Self);
    fn atan2(self, other: Self) -> Self;
    fn from_usize(value: usize) -> Self;
}

macro_rules! impl_kernel_float {
    ($type:ty) => {
        impl KernelFloat for $type {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const EPSILON: Self = <$type>::EPSILON;
            const INFINITY: Self = <$type>::INFINITY;

            fn abs(self) -> Self {
                self.abs()
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn ln(self) -> Self {
                self.ln()
            }
            fn max_of(self, other: Self) -> Self {
                self.max(other)
            }
            fn is_finite(self) -> bool {
                self.is_finite()
            }
            fn sin_cos(self) -> (Self, Self) {
                self.sin_cos()
            }
            fn atan2(self, other: Self) -> Self {
                self.atan2(other)
            }
            fn from_usize(value: usize) -> Self {
                value as Self
            }
        }
    };
}

impl_kernel_float!(f32);
impl_kernel_float!(f64);

/// Failure returned by a numerical kernel before it is mapped to `rustnum::Error`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KernelError {
    Singular,
    NotPositiveDefinite,
    Shape,
    /// An iterative eigen/singular-value solver did not converge (e.g. NaN input).
    NoConvergence,
}

/// Norm orders accepted by `norm_axis`, `vector_norm`, `matrix_norm` and `cond_ord`, like the
/// `ord` argument of `np.linalg.norm`.
///
/// Vectors accept every order except `Nuc`; matrices accept every order except `Zero`.
/// `Fro` and `Two` coincide for vectors.
///
/// ```
/// use rustnum as np;
/// use np::linalg::NormOrd;
/// let x = np::array![3.0, -4.0];
/// assert_eq!(np::linalg::vector_norm(&x, NormOrd::One), 7.0);
/// assert_eq!(np::linalg::vector_norm(&x, NormOrd::Inf), 4.0);
/// let a = np::array![[1.0, -2.0], [3.0, -4.0]];
/// assert_eq!(np::linalg::matrix_norm(&a, NormOrd::Inf), 7.0);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormOrd {
    /// Frobenius norm (`'fro'`).
    Fro,
    /// `ord=2`: Euclidean norm of a vector, largest singular value of a matrix.
    Two,
    /// `ord=1`: sum of magnitudes, or maximum absolute column sum.
    One,
    /// `ord=inf`: largest magnitude, or maximum absolute row sum.
    Inf,
    /// `ord=-1`: harmonic-type norm of a vector, or minimum absolute column sum.
    NegOne,
    /// `ord=-2`: vector `(sum |x|^-2)^(-1/2)`, or smallest singular value of a matrix.
    NegTwo,
    /// `ord=-inf`: smallest magnitude, or minimum absolute row sum.
    NegInf,
    /// `ord=0` (vectors only): number of non-zero entries.
    Zero,
    /// `ord='nuc'` (matrices only): sum of singular values.
    Nuc,
}

fn matrix_fits<T>(
    data: &[T],
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
) -> bool {
    if rows == 0 || cols == 0 {
        return true;
    }
    let row_end = (rows - 1) as i128 * row_stride as i128;
    let col_end = (cols - 1) as i128 * col_stride as i128;
    let low = row_end.min(0) + col_end.min(0);
    let high = row_end.max(0) + col_end.max(0);
    low >= 0 && high < data.len() as i128
}

fn assert_matrix_fits<T>(
    data: &[T],
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
) {
    assert!(
        matrix_fits(data, rows, cols, row_stride, col_stride),
        "matrix strides are outside the supplied slice"
    );
}

/// True when no two distinct `(row, col)` positions map to the same element offset.
///
/// A broadcast (zero) stride or a layout whose strides overlap would alias.  The common
/// layouts (row- or column-major, possibly with padding) are accepted in O(1); only
/// unusual interleaved strides pay for an O(m n log(m n)) sort.
fn output_is_non_aliasing(rows: usize, cols: usize, row_stride: isize, col_stride: isize) -> bool {
    if rows <= 1 && cols <= 1 {
        return true;
    }
    if rows <= 1 {
        return col_stride != 0;
    }
    if cols <= 1 {
        return row_stride != 0;
    }
    let (row_step, col_step) = (row_stride.unsigned_abs(), col_stride.unsigned_abs());
    if row_step == 0 || col_step == 0 {
        return false;
    }
    // The faster-varying axis must fit entirely inside one step of the slower axis.
    if col_step <= row_step && col_step * cols <= row_step {
        return true;
    }
    if row_step <= col_step && row_step * rows <= col_step {
        return true;
    }
    let mut offsets = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            offsets.push(row as isize * row_stride + col as isize * col_stride);
        }
    }
    offsets.sort_unstable();
    offsets.windows(2).all(|pair| pair[0] != pair[1])
}

/// Generic, allocation-owning matrix product for element types without a BLAS path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm<T>(
    m: usize,
    n: usize,
    k: usize,
    a: &[T],
    a_row_stride: isize,
    a_col_stride: isize,
    b: &[T],
    b_row_stride: isize,
    b_col_stride: isize,
) -> Vec<T>
where
    T: Num,
{
    assert_matrix_fits(a, m, k, a_row_stride, a_col_stride);
    assert_matrix_fits(b, k, n, b_row_stride, b_col_stride);
    let mut result = vec![T::ZERO; m * n];
    // i-k-j keeps the output row hot in cache and also avoids an allocation per
    // element for strided Array views.
    for i in 0..m {
        for p in 0..k {
            let left = a[(i as isize * a_row_stride + p as isize * a_col_stride) as usize];
            for j in 0..n {
                let right = b[(p as isize * b_row_stride + j as isize * b_col_stride) as usize];
                let index = i * n + j;
                result[index] = add(result[index], mul(left, right));
            }
        }
    }
    result
}

/// Stride-aware f32 GEMM: `C = alpha * A * B + beta * C`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_f32(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    a_row_stride: isize,
    a_col_stride: isize,
    b: &[f32],
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f32,
    c: &mut [f32],
    c_row_stride: isize,
    c_col_stride: isize,
) {
    assert_matrix_fits(a, m, k, a_row_stride, a_col_stride);
    assert_matrix_fits(b, k, n, b_row_stride, b_col_stride);
    assert_matrix_fits(c, m, n, c_row_stride, c_col_stride);
    assert!(
        output_is_non_aliasing(m, n, c_row_stride, c_col_stride),
        "GEMM output strides must not alias"
    );
    if m == 0 || n == 0 {
        return;
    }
    // SAFETY: validation above proves every logical A/B/C element lies in its
    // corresponding slice. C uses caller-provided non-aliasing Array strides;
    // the Array adapter only supplies writable, non-broadcast output layouts.
    unsafe {
        tiled_gemm(
            matrixmultiply::sgemm,
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            a_row_stride,
            a_col_stride,
            b.as_ptr(),
            b_row_stride,
            b_col_stride,
            beta,
            c.as_mut_ptr(),
            c_row_stride,
            c_col_stride,
        );
    }
}

/// A raw matrixmultiply entry point (`sgemm` or `dgemm`).
type RawGemm<T> = unsafe fn(
    usize,
    usize,
    usize,
    T,
    *const T,
    isize,
    isize,
    *const T,
    isize,
    isize,
    T,
    *mut T,
    isize,
    isize,
);

/// Multiply-adds (m·n·k) below which a GEMM runs as one call.
#[cfg(feature = "rayon")]
const PARALLEL_GEMM_MIN_WORK: usize = 1 << 21;

/// Side of the square output tiles handed to rayon. Each tile is one
/// single-threaded matrixmultiply call over the full K dimension.
#[cfg(feature = "rayon")]
const GEMM_TILE: usize = 256;

/// A pointer shared by the tasks of one tiled GEMM.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy)]
struct SharedPtr<T>(*mut T);

// SAFETY: tasks only read A and B, and each task writes a disjoint tile of C
// (see `tiled_gemm`), so sharing these pointers across threads cannot race.
#[cfg(feature = "rayon")]
unsafe impl<T> Send for SharedPtr<T> {}
#[cfg(feature = "rayon")]
unsafe impl<T> Sync for SharedPtr<T> {}

#[cfg(feature = "rayon")]
impl<T> SharedPtr<T> {
    // A method (not field access) so closures capture the Send wrapper itself.
    fn get(self) -> *mut T {
        self.0
    }
}

/// Runs `raw` over the whole product, splitting large outputs into tiles that
/// rayon spreads over every core.
///
/// matrixmultiply's own threading stops at 4 threads. Independent output tiles
/// let work stealing use all cores and hand more tiles to fast cores on hybrid
/// CPUs.
///
/// # Safety
/// The pointers and strides must describe a valid A (m×k), B (k×n) and a
/// non-aliasing C (m×n), as checked by `gemm_f32` / `gemm_f64`.
#[allow(clippy::too_many_arguments)]
unsafe fn tiled_gemm<T: Copy>(
    raw: RawGemm<T>,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
) {
    #[cfg(feature = "rayon")]
    if m.saturating_mul(n).saturating_mul(k.max(1)) >= PARALLEL_GEMM_MIN_WORK
        && rayon::current_num_threads() > 1
        && (m > GEMM_TILE || n > GEMM_TILE)
    {
        use rayon::prelude::*;
        let tiles: Vec<(usize, usize)> = (0..m)
            .step_by(GEMM_TILE)
            .flat_map(|i| (0..n).step_by(GEMM_TILE).map(move |j| (i, j)))
            .collect();
        let (a, b, c) = (
            SharedPtr(a.cast_mut()),
            SharedPtr(b.cast_mut()),
            SharedPtr(c),
        );
        let (alpha, beta) = (SendCopy(alpha), SendCopy(beta));
        tiles.into_par_iter().for_each(move |(i0, j0)| {
            let rows = GEMM_TILE.min(m - i0);
            let cols = GEMM_TILE.min(n - j0);
            // SAFETY: (i0, j0) starts a tile inside C, the tiles partition C so
            // no two tasks write the same element, and A rows / B columns are
            // only read. wrapping_offset keeps the address math defined even
            // when k == 0 and A or B is empty (matrixmultiply then reads none).
            unsafe {
                raw(
                    rows,
                    k,
                    cols,
                    alpha.get(),
                    a.get().wrapping_offset(i0 as isize * rsa),
                    rsa,
                    csa,
                    b.get().wrapping_offset(j0 as isize * csb),
                    rsb,
                    csb,
                    beta.get(),
                    c.get()
                        .wrapping_offset(i0 as isize * rsc + j0 as isize * csc),
                    rsc,
                    csc,
                );
            }
        });
        return;
    }
    // SAFETY: forwarded from the caller's contract.
    unsafe { raw(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) }
}

/// A scalar copied into every tile task.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy)]
struct SendCopy<T>(T);

// SAFETY: only instantiated with f32 and f64, which are plain values.
#[cfg(feature = "rayon")]
unsafe impl<T> Send for SendCopy<T> {}
#[cfg(feature = "rayon")]
unsafe impl<T> Sync for SendCopy<T> {}

#[cfg(feature = "rayon")]
impl<T: Copy> SendCopy<T> {
    fn get(self) -> T {
        self.0
    }
}

/// Stride-aware f64 GEMM: `C = alpha * A * B + beta * C`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_f64(
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    a_row_stride: isize,
    a_col_stride: isize,
    b: &[f64],
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: &mut [f64],
    c_row_stride: isize,
    c_col_stride: isize,
) {
    assert_matrix_fits(a, m, k, a_row_stride, a_col_stride);
    assert_matrix_fits(b, k, n, b_row_stride, b_col_stride);
    assert_matrix_fits(c, m, n, c_row_stride, c_col_stride);
    assert!(
        output_is_non_aliasing(m, n, c_row_stride, c_col_stride),
        "GEMM output strides must not alias"
    );
    if m == 0 || n == 0 {
        return;
    }
    // SAFETY: see `gemm_f32`; matrixmultiply supports arbitrary valid input
    // strides and the destination supplied by the adapter never aliases itself.
    unsafe {
        tiled_gemm(
            matrixmultiply::dgemm,
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            a_row_stride,
            a_col_stride,
            b.as_ptr(),
            b_row_stride,
            b_col_stride,
            beta,
            c.as_mut_ptr(),
            c_row_stride,
            c_col_stride,
        );
    }
}

/// LU factorisation with partial pivoting of a square row-major matrix.
#[derive(Clone, Debug)]
pub(crate) struct Lu<T> {
    lu: Vec<T>,
    pivots: Vec<usize>,
    swaps: usize,
}

impl<T: KernelFloat> Lu<T> {
    /// Factors `a`, retaining row-swap data needed by solve and determinant.
    pub(crate) fn factor(a: &[T], n: usize) -> Result<Self, KernelError> {
        if a.len() != n * n {
            return Err(KernelError::Shape);
        }
        let mut lu = a.to_vec();
        let mut pivots = (0..n).collect::<Vec<_>>();
        let mut swaps = 0;
        for col in 0..n {
            let mut pivot = col;
            let mut largest = lu[col * n + col].abs();
            for row in col + 1..n {
                let candidate = lu[row * n + col].abs();
                if candidate > largest {
                    largest = candidate;
                    pivot = row;
                }
            }
            if largest == T::ZERO {
                return Err(KernelError::Singular);
            }
            pivots[col] = pivot;
            if pivot != col {
                for j in 0..n {
                    lu.swap(col * n + j, pivot * n + j);
                }
                swaps += 1;
            }
            let diagonal = lu[col * n + col];
            for row in col + 1..n {
                let multiplier = lu[row * n + col] / diagonal;
                lu[row * n + col] = multiplier;
                for j in col + 1..n {
                    lu[row * n + j] = lu[row * n + j] - multiplier * lu[col * n + j];
                }
            }
        }
        Ok(Self { lu, pivots, swaps })
    }

    /// Returns the determinant implied by this factorisation.
    pub(crate) fn det(&self) -> T {
        let n = self.pivots.len();
        let mut determinant = if self.swaps.is_multiple_of(2) {
            T::ONE
        } else {
            -T::ONE
        };
        for i in 0..n {
            determinant = determinant * self.lu[i * n + i];
        }
        determinant
    }

    /// Solves `AX=B`, where B has `rhs_cols` contiguous columns.
    pub(crate) fn solve(&self, rhs: &[T], rhs_cols: usize) -> Result<Vec<T>, KernelError> {
        let n = self.pivots.len();
        if rhs.len() != n * rhs_cols {
            return Err(KernelError::Shape);
        }
        let mut x = rhs.to_vec();
        for col in 0..n {
            let pivot = self.pivots[col];
            if pivot != col {
                for j in 0..rhs_cols {
                    x.swap(col * rhs_cols + j, pivot * rhs_cols + j);
                }
            }
        }
        for row in 0..n {
            for j in 0..rhs_cols {
                let mut value = x[row * rhs_cols + j];
                for k in 0..row {
                    value = value - self.lu[row * n + k] * x[k * rhs_cols + j];
                }
                x[row * rhs_cols + j] = value;
            }
        }
        for row in (0..n).rev() {
            for j in 0..rhs_cols {
                let mut value = x[row * rhs_cols + j];
                for k in row + 1..n {
                    value = value - self.lu[row * n + k] * x[k * rhs_cols + j];
                }
                x[row * rhs_cols + j] = value / self.lu[row * n + row];
            }
        }
        Ok(x)
    }
}

/// Computes a matrix inverse through LU factorisation.
pub(crate) fn inv<T: KernelFloat>(a: &[T], n: usize) -> Result<Vec<T>, KernelError> {
    let lu = Lu::factor(a, n)?;
    let mut identity = vec![T::ZERO; n * n];
    for i in 0..n {
        identity[i * n + i] = T::ONE;
    }
    lu.solve(&identity, n)
}

/// Computes the lower Cholesky factor of a positive-definite square matrix.
pub(crate) fn cholesky<T: KernelFloat>(a: &[T], n: usize) -> Result<Vec<T>, KernelError> {
    if a.len() != n * n {
        return Err(KernelError::Shape);
    }
    let mut lower = vec![T::ZERO; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum = sum - lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if sum <= T::ZERO {
                    return Err(KernelError::NotPositiveDefinite);
                }
                lower[i * n + j] = sum.sqrt();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    Ok(lower)
}

/// Reduced Householder QR factorisation, returning `(Q, R, reduced_columns)`.
pub(crate) fn qr<T: KernelFloat>(
    a: &[T],
    rows: usize,
    cols: usize,
) -> Result<(Vec<T>, Vec<T>, usize), KernelError> {
    if a.len() != rows * cols {
        return Err(KernelError::Shape);
    }
    let reduced = rows.min(cols);
    let mut r = a.to_vec();
    let mut q_full = vec![T::ZERO; rows * rows];
    for i in 0..rows {
        q_full[i * rows + i] = T::ONE;
    }
    for k in 0..reduced {
        // Like LAPACK's dlarfg: when everything below the diagonal is already zero the
        // reflector is the identity (this also covers zero columns and the last column of a
        // square matrix), so R keeps its sign and Q is untouched.
        let mut scale = T::ZERO;
        let mut below = T::ZERO;
        for row in k..rows {
            let magnitude = r[row * cols + k].abs();
            if magnitude > scale {
                scale = magnitude;
            }
            if row > k && magnitude > below {
                below = magnitude;
            }
        }
        if below == T::ZERO {
            continue;
        }
        // Scaled 2-norm so tiny or huge columns neither underflow nor overflow.
        let mut norm_sq = T::ZERO;
        for row in k..rows {
            let scaled = r[row * cols + k] / scale;
            norm_sq = norm_sq + scaled * scaled;
        }
        let norm = norm_sq.sqrt() * scale;
        let leading = r[k * cols + k];
        let signed_norm = if leading >= T::ZERO { norm } else { -norm };
        let mut v = vec![T::ZERO; rows - k];
        v[0] = leading + signed_norm;
        for row in k + 1..rows {
            v[row - k] = r[row * cols + k];
        }
        let beta = T::ONE / (signed_norm * v[0]);
        for col in k..cols {
            let mut dot = T::ZERO;
            for row in k..rows {
                dot = dot + v[row - k] * r[row * cols + col];
            }
            for row in k..rows {
                r[row * cols + col] = r[row * cols + col] - beta * v[row - k] * dot;
            }
        }
        for row in 0..rows {
            let mut dot = T::ZERO;
            for col in k..rows {
                dot = dot + q_full[row * rows + col] * v[col - k];
            }
            for col in k..rows {
                q_full[row * rows + col] = q_full[row * rows + col] - beta * dot * v[col - k];
            }
        }
    }
    let mut q = vec![T::ZERO; rows * reduced];
    for row in 0..rows {
        for col in 0..reduced {
            q[row * reduced + col] = q_full[row * rows + col];
        }
    }
    let mut reduced_r = vec![T::ZERO; reduced * cols];
    for row in 0..reduced {
        reduced_r[row * cols..(row + 1) * cols].copy_from_slice(&r[row * cols..(row + 1) * cols]);
    }
    Ok((q, reduced_r, reduced))
}

/// Minimum-norm least-squares solution `pinv(A) B` (what NumPy's SVD-based `lstsq` returns).
fn lstsq_min_norm<T: KernelFloat + Num>(
    a: &[T],
    rows: usize,
    cols: usize,
    b: &[T],
    rhs_cols: usize,
) -> Result<Vec<T>, KernelError> {
    let inverse = pinv(a, rows, cols)?;
    Ok(gemm(
        cols,
        rhs_cols,
        rows,
        &inverse,
        rows as isize,
        1,
        b,
        rhs_cols as isize,
        1,
    ))
}

/// Least-squares solution, like `np.linalg.lstsq(a, b, rcond=None)[0]`.
///
/// Full-column-rank tall systems use Householder QR; under-determined or rank-deficient systems
/// return the minimum-norm solution through the SVD, exactly as NumPy does.
pub(crate) fn lstsq<T: KernelFloat + Num>(
    a: &[T],
    rows: usize,
    cols: usize,
    b: &[T],
    rhs_cols: usize,
) -> Result<Vec<T>, KernelError> {
    if b.len() != rows * rhs_cols {
        return Err(KernelError::Shape);
    }
    if rows < cols {
        return lstsq_min_norm(a, rows, cols, b, rhs_cols);
    }
    let (q, r, _) = qr(a, rows, cols)?;
    // QR without pivoting is only trusted when R is comfortably non-singular; otherwise NumPy's
    // answer is the minimum-norm solution, which the SVD route below reproduces.
    let mut largest = <T as KernelFloat>::ZERO;
    let mut smallest = T::INFINITY;
    for i in 0..cols {
        let d = KernelFloat::abs(r[i * cols + i]);
        if d > largest {
            largest = d;
        }
        if d < smallest {
            smallest = d;
        }
    }
    let hundred = <T as KernelFloat>::from_usize(100);
    if cols > 0 && smallest <= largest * T::EPSILON * <T as KernelFloat>::from_usize(rows) * hundred
    {
        return lstsq_min_norm(a, rows, cols, b, rhs_cols);
    }
    let mut y = vec![<T as KernelFloat>::ZERO; cols * rhs_cols];
    for i in 0..cols {
        for j in 0..rhs_cols {
            for row in 0..rows {
                y[i * rhs_cols + j] =
                    y[i * rhs_cols + j] + q[row * cols + i] * b[row * rhs_cols + j];
            }
        }
    }
    for row in (0..cols).rev() {
        for j in 0..rhs_cols {
            let mut value = y[row * rhs_cols + j];
            for k in row + 1..cols {
                value = value - r[row * cols + k] * y[k * rhs_cols + j];
            }
            y[row * rhs_cols + j] = value / r[row * cols + row];
        }
    }
    Ok(y)
}

fn largest_component<T: KernelFloat>(column: &[T]) -> T {
    let mut selected = T::ZERO;
    let mut magnitude = T::ZERO;
    for &value in column {
        if value.abs() > magnitude {
            magnitude = value.abs();
            selected = value;
        }
    }
    selected
}

/// Cyclic Jacobi eigensolver for a real symmetric matrix.
pub(crate) fn eigh<T: KernelFloat>(a: &[T], n: usize) -> Result<(Vec<T>, Vec<T>), KernelError> {
    if a.len() != n * n {
        return Err(KernelError::Shape);
    }
    if !a.iter().all(|value| value.is_finite()) {
        return Err(KernelError::NoConvergence);
    }
    let mut work = a.to_vec();
    let mut vectors = vec![T::ZERO; n * n];
    for i in 0..n {
        vectors[i * n + i] = T::ONE;
    }
    // Convergence is relative to the size of the matrix, so 1e-20 * A and 1e20 * A behave like A.
    let mut scale = T::ZERO;
    for &value in a {
        if value.abs() > scale {
            scale = value.abs();
        }
    }
    let threshold = T::EPSILON * scale;
    let max_sweeps = (n.max(1) * n.max(1) * 32).max(32);
    for _ in 0..max_sweeps {
        let mut largest = T::ZERO;
        for p in 0..n {
            for q in p + 1..n {
                if work[p * n + q].abs() > largest {
                    largest = work[p * n + q].abs();
                }
            }
        }
        if largest <= threshold {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let off = work[p * n + q];
                if off.abs() <= threshold {
                    continue;
                }
                let half = T::ONE / (T::ONE + T::ONE);
                let (sin, cos) =
                    ((off + off).atan2(work[q * n + q] - work[p * n + p]) * half).sin_cos();
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let pk = work[p * n + k];
                    let qk = work[q * n + k];
                    work[p * n + k] = cos * pk - sin * qk;
                    work[k * n + p] = work[p * n + k];
                    work[q * n + k] = sin * pk + cos * qk;
                    work[k * n + q] = work[q * n + k];
                }
                let pp = work[p * n + p];
                let qq = work[q * n + q];
                work[p * n + p] = cos * cos * pp - (sin + sin) * cos * off + sin * sin * qq;
                work[q * n + q] = sin * sin * pp + (sin + sin) * cos * off + cos * cos * qq;
                work[p * n + q] = T::ZERO;
                work[q * n + p] = T::ZERO;
                for row in 0..n {
                    let vp = vectors[row * n + p];
                    let vq = vectors[row * n + q];
                    vectors[row * n + p] = cos * vp - sin * vq;
                    vectors[row * n + q] = sin * vp + cos * vq;
                }
            }
        }
    }
    let mut ordering = (0..n).collect::<Vec<_>>();
    ordering.sort_by(|&i, &j| {
        work[i * n + i]
            .partial_cmp(&work[j * n + j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut values = vec![T::ZERO; n];
    let mut sorted_vectors = vec![T::ZERO; n * n];
    for (new_col, &old_col) in ordering.iter().enumerate() {
        values[new_col] = work[old_col * n + old_col];
        for row in 0..n {
            sorted_vectors[row * n + new_col] = vectors[row * n + old_col];
        }
        let sign = largest_component(
            &(0..n)
                .map(|row| sorted_vectors[row * n + new_col])
                .collect::<Vec<_>>(),
        );
        if sign < T::ZERO {
            for row in 0..n {
                sorted_vectors[row * n + new_col] = -sorted_vectors[row * n + new_col];
            }
        }
    }
    Ok((values, sorted_vectors))
}

/// One-sided Jacobi SVD returning reduced `(U, singular_values, Vt, rank_dim)`.
type Svd<T> = (Vec<T>, Vec<T>, Vec<T>, usize);

pub(crate) fn svd<T: KernelFloat>(
    a: &[T],
    rows: usize,
    cols: usize,
) -> Result<Svd<T>, KernelError> {
    if a.len() != rows * cols {
        return Err(KernelError::Shape);
    }
    if !a.iter().all(|value| value.is_finite()) {
        return Err(KernelError::NoConvergence);
    }
    let reduced = rows.min(cols);
    let mut columns = a.to_vec();
    let mut right = vec![T::ZERO; cols * cols];
    for i in 0..cols {
        right[i * cols + i] = T::ONE;
    }
    let max_sweeps = (cols.max(1) * cols.max(1) * 64).max(64);
    for _ in 0..max_sweeps {
        let mut changed = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let (mut alpha, mut beta, mut gamma) = (T::ZERO, T::ZERO, T::ZERO);
                for row in 0..rows {
                    let left = columns[row * cols + p];
                    let right_value = columns[row * cols + q];
                    alpha = alpha + left * left;
                    beta = beta + right_value * right_value;
                    gamma = gamma + left * right_value;
                }
                if alpha == T::ZERO
                    || beta == T::ZERO
                    || gamma.abs() <= T::EPSILON * alpha.sqrt() * beta.sqrt()
                {
                    continue;
                }
                let tau = (beta - alpha) / (gamma + gamma);
                let t = if tau >= T::ZERO {
                    T::ONE / (tau + (T::ONE + tau * tau).sqrt())
                } else {
                    -T::ONE / (-tau + (T::ONE + tau * tau).sqrt())
                };
                let cosine = T::ONE / (T::ONE + t * t).sqrt();
                let sine = cosine * t;
                for row in 0..rows {
                    let left = columns[row * cols + p];
                    let right_value = columns[row * cols + q];
                    columns[row * cols + p] = cosine * left - sine * right_value;
                    columns[row * cols + q] = sine * left + cosine * right_value;
                }
                for row in 0..cols {
                    let left = right[row * cols + p];
                    let right_value = right[row * cols + q];
                    right[row * cols + p] = cosine * left - sine * right_value;
                    right[row * cols + q] = sine * left + cosine * right_value;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut singular_all = vec![T::ZERO; cols];
    for col in 0..cols {
        for row in 0..rows {
            singular_all[col] =
                singular_all[col] + columns[row * cols + col] * columns[row * cols + col];
        }
        singular_all[col] = singular_all[col].sqrt();
    }
    let mut ordering = (0..cols).collect::<Vec<_>>();
    ordering.sort_by(|&i, &j| {
        singular_all[j]
            .partial_cmp(&singular_all[i])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut u = vec![T::ZERO; rows * reduced];
    let mut s = vec![T::ZERO; reduced];
    let mut vt = vec![T::ZERO; reduced * cols];
    // Singular values at or below NumPy's rank tolerance have no meaningful direction: their U
    // columns are completed to an orthonormal basis below instead of dividing rounding noise.
    let largest_singular = ordering.first().map_or(T::ZERO, |&i| singular_all[i]);
    let tolerance = T::EPSILON * T::from_usize(rows.max(cols)) * largest_singular;
    let mut deficient = Vec::new();
    for new_col in 0..reduced {
        let old_col = ordering[new_col];
        s[new_col] = singular_all[old_col];
        if s[new_col] > tolerance {
            for row in 0..rows {
                u[row * reduced + new_col] = columns[row * cols + old_col] / s[new_col];
            }
        } else {
            deficient.push(new_col);
        }
        for col in 0..cols {
            vt[new_col * cols + col] = right[col * cols + old_col];
        }
    }
    complete_orthonormal_columns(&mut u, rows, reduced, &deficient);
    for new_col in 0..reduced {
        let sign = largest_component(
            &(0..rows)
                .map(|row| u[row * reduced + new_col])
                .collect::<Vec<_>>(),
        );
        if sign < T::ZERO {
            for row in 0..rows {
                u[row * reduced + new_col] = -u[row * reduced + new_col];
            }
            for col in 0..cols {
                vt[new_col * cols + col] = -vt[new_col * cols + col];
            }
        }
    }
    Ok((u, s, vt, reduced))
}

/// Fills the listed (all-zero) columns of the `rows x cols` matrix `u` so that every column is
/// unit length and orthogonal to the others (Gram-Schmidt against the standard basis).
fn complete_orthonormal_columns<T: KernelFloat>(
    u: &mut [T],
    rows: usize,
    cols: usize,
    missing: &[usize],
) {
    for &target in missing {
        let mut best_residual: Vec<T> = Vec::new();
        let mut best_norm = T::ZERO;
        for axis in 0..rows {
            let mut candidate = vec![T::ZERO; rows];
            candidate[axis] = T::ONE;
            // Two passes keep the result orthogonal to working precision.
            for _ in 0..2 {
                for other in 0..cols {
                    if other == target || (missing.contains(&other) && other > target) {
                        continue;
                    }
                    let mut dot = T::ZERO;
                    for row in 0..rows {
                        dot = dot + candidate[row] * u[row * cols + other];
                    }
                    for row in 0..rows {
                        candidate[row] = candidate[row] - dot * u[row * cols + other];
                    }
                }
            }
            let length = norm(&candidate);
            if length > best_norm {
                best_norm = length;
                best_residual = candidate;
            }
        }
        if best_norm > T::ZERO {
            for row in 0..rows {
                u[row * cols + target] = best_residual[row] / best_norm;
            }
        }
    }
}

/// Moore-Penrose pseudoinverse computed from the reduced SVD.
pub(crate) fn pinv<T: KernelFloat>(
    a: &[T],
    rows: usize,
    cols: usize,
) -> Result<Vec<T>, KernelError> {
    let (u, singular, vt, reduced) = svd(a, rows, cols)?;
    let leading = singular.first().copied().unwrap_or(T::ZERO);
    let tolerance = T::EPSILON * T::from_usize(rows.max(cols)) * leading;
    let mut inverse = vec![T::ZERO; cols * rows];
    for component in 0..reduced {
        if singular[component] <= tolerance {
            continue;
        }
        for row in 0..cols {
            for col in 0..rows {
                inverse[row * rows + col] = inverse[row * rows + col]
                    + vt[component * cols + row] * u[col * reduced + component]
                        / singular[component];
            }
        }
    }
    Ok(inverse)
}

/// Numerical rank using NumPy's default SVD tolerance.
pub(crate) fn matrix_rank<T: KernelFloat>(
    a: &[T],
    rows: usize,
    cols: usize,
) -> Result<usize, KernelError> {
    let (_, singular, _, _) = svd(a, rows, cols)?;
    let largest = singular.first().copied().unwrap_or(T::ZERO);
    let tolerance = T::EPSILON * T::from_usize(rows.max(cols)) * largest;
    Ok(singular
        .into_iter()
        .filter(|value| *value > tolerance)
        .count())
}

/// Euclidean norm of a contiguous vector or Frobenius norm of flattened data.
pub(crate) fn norm<T: KernelFloat>(data: &[T]) -> T {
    data.iter()
        .fold(T::ZERO, |sum, &value| sum + value * value)
        .sqrt()
}

/// Vector norm of contiguous data for every order except [`NormOrd::Nuc`], like `np.linalg.norm(x, ord)`.
pub(crate) fn vector_norm<T: KernelFloat>(data: &[T], order: NormOrd) -> Result<T, KernelError> {
    match order {
        NormOrd::Fro | NormOrd::Two => Ok(norm(data)),
        NormOrd::One => Ok(data.iter().fold(T::ZERO, |sum, &value| sum + value.abs())),
        NormOrd::Inf => Ok(data.iter().fold(T::ZERO, |best, &value| {
            if value.abs() > best {
                value.abs()
            } else {
                best
            }
        })),
        NormOrd::NegInf => {
            let mut values = data.iter().map(|value| value.abs());
            let first = values.next().ok_or(KernelError::Shape)?;
            Ok(values.fold(first, |best, value| if value < best { value } else { best }))
        }
        NormOrd::Zero => Ok(T::from_usize(
            data.iter().filter(|&&value| value != T::ZERO).count(),
        )),
        NormOrd::NegOne => Ok(T::ONE
            / data
                .iter()
                .fold(T::ZERO, |sum, &value| sum + T::ONE / value.abs())),
        NormOrd::NegTwo => Ok(T::ONE
            / data
                .iter()
                .fold(T::ZERO, |sum, &value| sum + T::ONE / (value * value))
                .sqrt()),
        NormOrd::Nuc => Err(KernelError::Shape),
    }
}

/// Matrix norm for every order except [`NormOrd::Zero`], like `np.linalg.norm(A, ord)`.
pub(crate) fn matrix_norm<T: KernelFloat>(
    a: &[T],
    rows: usize,
    cols: usize,
    order: NormOrd,
) -> Result<T, KernelError> {
    if a.len() != rows * cols {
        return Err(KernelError::Shape);
    }
    let sums = |by_column: bool| -> Vec<T> {
        let (outer, inner) = if by_column {
            (cols, rows)
        } else {
            (rows, cols)
        };
        (0..outer)
            .map(|o| {
                (0..inner).fold(T::ZERO, |sum, i| {
                    let index = if by_column {
                        i * cols + o
                    } else {
                        o * cols + i
                    };
                    sum + a[index].abs()
                })
            })
            .collect()
    };
    let largest = |values: &[T]| {
        values
            .iter()
            .fold(T::ZERO, |best, &v| if v > best { v } else { best })
    };
    let smallest = |values: &[T]| -> Result<T, KernelError> {
        let (&first, rest) = values.split_first().ok_or(KernelError::Shape)?;
        Ok(rest
            .iter()
            .fold(first, |best, &v| if v < best { v } else { best }))
    };
    match order {
        NormOrd::Fro => Ok(norm(a)),
        NormOrd::Two => Ok(svd(a, rows, cols)?.1.first().copied().unwrap_or(T::ZERO)),
        NormOrd::NegTwo => {
            let (_, s, _, _) = svd(a, rows, cols)?;
            s.last().copied().ok_or(KernelError::Shape)
        }
        NormOrd::Nuc => Ok(svd(a, rows, cols)?
            .1
            .into_iter()
            .fold(T::ZERO, |sum, value| sum + value)),
        NormOrd::One => Ok(largest(&sums(true))),
        NormOrd::NegOne => smallest(&sums(true)),
        NormOrd::Inf => Ok(largest(&sums(false))),
        NormOrd::NegInf => smallest(&sums(false)),
        NormOrd::Zero => Err(KernelError::Shape),
    }
}

/// Vector norm along one axis of a contiguous two-dimensional matrix.
///
/// `axis=0` reduces over rows (one result per column), `axis=1` over columns.
pub(crate) fn norm_axis<T: KernelFloat>(
    a: &[T],
    rows: usize,
    cols: usize,
    axis: usize,
    order: NormOrd,
) -> Result<Vec<T>, KernelError> {
    if a.len() != rows * cols || axis > 1 {
        return Err(KernelError::Shape);
    }
    let (len, count) = if axis == 0 {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let mut slice = vec![T::ZERO; count];
    let mut output = Vec::with_capacity(len);
    for out in 0..len {
        for (index, slot) in slice.iter_mut().enumerate() {
            *slot = if axis == 0 {
                a[index * cols + out]
            } else {
                a[out * cols + index]
            };
        }
        output.push(vector_norm(&slice, order)?);
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Additional kernels: general eigenproblem, slogdet, cond, multi_dot, tensordot, ...
// ---------------------------------------------------------------------------

/// `(eigenvalue_re, eigenvalue_im, eigenvector_re, eigenvector_im)` of [`eig`].
///
/// Eigenvectors are the columns of the row-major `n x n` buffers: column `j` belongs to
/// eigenvalue `j`.
pub(crate) type Eig<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);

/// Reduces `h` to upper Hessenberg form with Householder reflections (EISPACK `orthes`),
/// accumulating the orthogonal similarity transform in `v`.
fn hessenberg<T: KernelFloat>(h: &mut [T], v: &mut [T], n: usize) {
    let high = n - 1;
    let mut ort = vec![T::ZERO; n];
    for m in 1..high {
        let mut scale = T::ZERO;
        for i in m..=high {
            scale = scale + h[i * n + m - 1].abs();
        }
        if scale == T::ZERO {
            continue;
        }
        let mut sum = T::ZERO;
        for i in (m..=high).rev() {
            ort[i] = h[i * n + m - 1] / scale;
            sum = sum + ort[i] * ort[i];
        }
        let mut g = sum.sqrt();
        if ort[m] > T::ZERO {
            g = -g;
        }
        sum = sum - ort[m] * g;
        ort[m] = ort[m] - g;
        // H = (I - u u' / sum) H (I - u u' / sum)
        for j in m..n {
            let mut f = T::ZERO;
            for i in (m..=high).rev() {
                f = f + ort[i] * h[i * n + j];
            }
            f = f / sum;
            for i in m..=high {
                h[i * n + j] = h[i * n + j] - f * ort[i];
            }
        }
        for i in 0..=high {
            let mut f = T::ZERO;
            for j in (m..=high).rev() {
                f = f + ort[j] * h[i * n + j];
            }
            f = f / sum;
            for j in m..=high {
                h[i * n + j] = h[i * n + j] - f * ort[j];
            }
        }
        ort[m] = scale * ort[m];
        h[m * n + m - 1] = scale * g;
    }
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = if i == j { T::ONE } else { T::ZERO };
        }
    }
    for m in (1..high).rev() {
        if h[m * n + m - 1] == T::ZERO {
            continue;
        }
        for i in m + 1..=high {
            ort[i] = h[i * n + m - 1];
        }
        for j in m..=high {
            let mut g = T::ZERO;
            for i in m..=high {
                g = g + ort[i] * v[i * n + j];
            }
            // Double division avoids possible underflow.
            g = (g / ort[m]) / h[m * n + m - 1];
            for i in m..=high {
                v[i * n + j] = v[i * n + j] + g * ort[i];
            }
        }
    }
    // Entries below the subdiagonal only held the Householder vectors; clear them.
    for i in 2..n {
        for j in 0..i - 1 {
            h[i * n + j] = T::ZERO;
        }
    }
}

/// Complex division `(xr + i xi) / (yr + i yi)`.
fn complex_div<T: KernelFloat>(xr: T, xi: T, yr: T, yi: T) -> (T, T) {
    if yr.abs() > yi.abs() {
        let r = yi / yr;
        let d = yr + r * yi;
        ((xr + r * xi) / d, (xi - r * xr) / d)
    } else {
        let r = yr / yi;
        let d = yi + r * yr;
        ((r * xr + xi) / d, (r * xi - xr) / d)
    }
}

/// Francis double-shift QR on the Hessenberg matrix `h` followed by back-substitution for the
/// eigenvectors of the quasi-triangular form (EISPACK `hqr2`, as in JAMA).
///
/// On return `d`/`e` hold the real/imaginary parts of the eigenvalues, and `v` holds the real
/// eigenvectors: for a complex pair `(j, j + 1)` with `e[j] > 0`, column `j` is the real part and
/// column `j + 1` the imaginary part of the eigenvector for `d[j] + i e[j]`.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
fn francis_qr<T: KernelFloat>(
    h: &mut [T],
    v: &mut [T],
    d: &mut [T],
    e: &mut [T],
    nn: usize,
) -> Result<(), KernelError> {
    macro_rules! hm {
        ($i:expr, $j:expr) => {
            h[($i) as usize * nn + ($j) as usize]
        };
    }
    macro_rules! vm {
        ($i:expr, $j:expr) => {
            v[($i) as usize * nn + ($j) as usize]
        };
    }
    macro_rules! dd {
        ($i:expr) => {
            d[($i) as usize]
        };
    }
    macro_rules! ee {
        ($i:expr) => {
            e[($i) as usize]
        };
    }
    let zero = T::ZERO;
    let one = T::ONE;
    let two = one + one;
    let eps = T::EPSILON;
    let size = nn as isize;
    let low = 0_isize;
    let high = size - 1;
    let mut n = size - 1;
    let mut exshift = zero;
    let (mut p, mut q, mut r, mut s, mut z) = (zero, zero, zero, zero, zero);
    let (mut t, mut w, mut x, mut y);

    let mut norm = zero;
    for i in 0..size {
        for j in (i - 1).max(0)..size {
            norm = norm + hm!(i, j).abs();
        }
    }

    let mut iter = 0_usize;
    while n >= low {
        // Look for a single small sub-diagonal element.
        let mut l = n;
        while l > low {
            s = hm!(l - 1, l - 1).abs() + hm!(l, l).abs();
            if s == zero {
                s = norm;
            }
            if hm!(l, l - 1).abs() <= eps * s {
                break;
            }
            l -= 1;
        }

        if l == n {
            // One root found.
            hm!(n, n) = hm!(n, n) + exshift;
            dd!(n) = hm!(n, n);
            ee!(n) = zero;
            n -= 1;
            iter = 0;
        } else if l == n - 1 {
            // Two roots found.
            w = hm!(n, n - 1) * hm!(n - 1, n);
            p = (hm!(n - 1, n - 1) - hm!(n, n)) / two;
            q = p * p + w;
            z = q.abs().sqrt();
            hm!(n, n) = hm!(n, n) + exshift;
            hm!(n - 1, n - 1) = hm!(n - 1, n - 1) + exshift;
            x = hm!(n, n);
            if q >= zero {
                // Real pair.
                z = if p >= zero { p + z } else { p - z };
                dd!(n - 1) = x + z;
                dd!(n) = dd!(n - 1);
                if z != zero {
                    dd!(n) = x - w / z;
                }
                ee!(n - 1) = zero;
                ee!(n) = zero;
                x = hm!(n, n - 1);
                s = x.abs() + z.abs();
                p = x / s;
                q = z / s;
                r = (p * p + q * q).sqrt();
                p = p / r;
                q = q / r;
                for j in n - 1..size {
                    z = hm!(n - 1, j);
                    hm!(n - 1, j) = q * z + p * hm!(n, j);
                    hm!(n, j) = q * hm!(n, j) - p * z;
                }
                for i in 0..=n {
                    z = hm!(i, n - 1);
                    hm!(i, n - 1) = q * z + p * hm!(i, n);
                    hm!(i, n) = q * hm!(i, n) - p * z;
                }
                for i in low..=high {
                    z = vm!(i, n - 1);
                    vm!(i, n - 1) = q * z + p * vm!(i, n);
                    vm!(i, n) = q * vm!(i, n) - p * z;
                }
            } else {
                // Complex pair.
                dd!(n - 1) = x + p;
                dd!(n) = x + p;
                ee!(n - 1) = z;
                ee!(n) = -z;
            }
            n -= 2;
            iter = 0;
        } else {
            // No convergence yet: form shift.
            x = hm!(n, n);
            y = zero;
            w = zero;
            if l < n {
                y = hm!(n - 1, n - 1);
                w = hm!(n, n - 1) * hm!(n - 1, n);
            }
            if iter == 10 {
                // Wilkinson's original ad hoc shift.
                exshift = exshift + x;
                for i in low..=n {
                    hm!(i, i) = hm!(i, i) - x;
                }
                s = hm!(n, n - 1).abs() + hm!(n - 1, n - 2).abs();
                x = T::from_usize(3) / T::from_usize(4) * s;
                y = x;
                w = -(T::from_usize(7) / T::from_usize(16)) * s * s;
            }
            if iter == 30 {
                // MATLAB's ad hoc shift.
                s = (y - x) / two;
                s = s * s + w;
                if s > zero {
                    s = s.sqrt();
                    if y < x {
                        s = -s;
                    }
                    s = x - w / ((y - x) / two + s);
                    for i in low..=n {
                        hm!(i, i) = hm!(i, i) - s;
                    }
                    exshift = exshift + s;
                    x = T::from_usize(964) / T::from_usize(1000);
                    y = x;
                    w = x;
                }
            }
            iter += 1;
            if iter > 300 {
                return Err(KernelError::NoConvergence);
            }

            // Look for two consecutive small sub-diagonal elements.
            let mut m = n - 2;
            while m >= l {
                z = hm!(m, m);
                r = x - z;
                s = y - z;
                p = (r * s - w) / hm!(m + 1, m) + hm!(m, m + 1);
                q = hm!(m + 1, m + 1) - z - r - s;
                r = hm!(m + 2, m + 1);
                s = p.abs() + q.abs() + r.abs();
                p = p / s;
                q = q / s;
                r = r / s;
                if m == l {
                    break;
                }
                if hm!(m, m - 1).abs() * (q.abs() + r.abs())
                    < eps
                        * (p.abs() * (hm!(m - 1, m - 1).abs() + z.abs() + hm!(m + 1, m + 1).abs()))
                {
                    break;
                }
                m -= 1;
            }
            for i in m + 2..=n {
                hm!(i, i - 2) = zero;
                if i > m + 2 {
                    hm!(i, i - 3) = zero;
                }
            }

            // Double QR step involving rows l:n and columns m:n.
            for k in m..n {
                let not_last = k != n - 1;
                if k != m {
                    p = hm!(k, k - 1);
                    q = hm!(k + 1, k - 1);
                    r = if not_last { hm!(k + 2, k - 1) } else { zero };
                    x = p.abs() + q.abs() + r.abs();
                    if x == zero {
                        continue;
                    }
                    p = p / x;
                    q = q / x;
                    r = r / x;
                }
                s = (p * p + q * q + r * r).sqrt();
                if p < zero {
                    s = -s;
                }
                if s != zero {
                    if k != m {
                        hm!(k, k - 1) = -s * x;
                    } else if l != m {
                        hm!(k, k - 1) = -hm!(k, k - 1);
                    }
                    p = p + s;
                    x = p / s;
                    y = q / s;
                    z = r / s;
                    q = q / p;
                    r = r / p;
                    for j in k..size {
                        p = hm!(k, j) + q * hm!(k + 1, j);
                        if not_last {
                            p = p + r * hm!(k + 2, j);
                            hm!(k + 2, j) = hm!(k + 2, j) - p * z;
                        }
                        hm!(k, j) = hm!(k, j) - p * x;
                        hm!(k + 1, j) = hm!(k + 1, j) - p * y;
                    }
                    for i in 0..=n.min(k + 3) {
                        p = x * hm!(i, k) + y * hm!(i, k + 1);
                        if not_last {
                            p = p + z * hm!(i, k + 2);
                            hm!(i, k + 2) = hm!(i, k + 2) - p * r;
                        }
                        hm!(i, k) = hm!(i, k) - p;
                        hm!(i, k + 1) = hm!(i, k + 1) - p * q;
                    }
                    for i in low..=high {
                        p = x * vm!(i, k) + y * vm!(i, k + 1);
                        if not_last {
                            p = p + z * vm!(i, k + 2);
                            vm!(i, k + 2) = vm!(i, k + 2) - p * r;
                        }
                        vm!(i, k) = vm!(i, k) - p;
                        vm!(i, k + 1) = vm!(i, k + 1) - p * q;
                    }
                }
            }
        }
    }

    // Back-substitute to find vectors of the upper quasi-triangular form.
    if norm == zero {
        return Ok(());
    }
    n = size - 1;
    while n >= 0 {
        p = dd!(n);
        q = ee!(n);
        if q == zero {
            // Real vector.
            let mut l = n;
            hm!(n, n) = one;
            let mut i = n - 1;
            while i >= 0 {
                w = hm!(i, i) - p;
                r = zero;
                for j in l..=n {
                    r = r + hm!(i, j) * hm!(j, n);
                }
                if ee!(i) < zero {
                    z = w;
                    s = r;
                } else {
                    l = i;
                    if ee!(i) == zero {
                        hm!(i, n) = if w == zero { -r / (eps * norm) } else { -r / w };
                    } else {
                        // Solve real equations.
                        x = hm!(i, i + 1);
                        y = hm!(i + 1, i);
                        q = (dd!(i) - p) * (dd!(i) - p) + ee!(i) * ee!(i);
                        t = (x * s - z * r) / q;
                        hm!(i, n) = t;
                        hm!(i + 1, n) = if x.abs() > z.abs() {
                            (-r - w * t) / x
                        } else {
                            (-s - y * t) / z
                        };
                    }
                    // Overflow control.
                    t = hm!(i, n).abs();
                    if (eps * t) * t > one {
                        for j in i..=n {
                            hm!(j, n) = hm!(j, n) / t;
                        }
                    }
                }
                i -= 1;
            }
        } else if q < zero {
            // Complex vector (processed once per pair, at its second member).
            let mut l = n - 1;
            if hm!(n, n - 1).abs() > hm!(n - 1, n).abs() {
                hm!(n - 1, n - 1) = q / hm!(n, n - 1);
                hm!(n - 1, n) = -(hm!(n, n) - p) / hm!(n, n - 1);
            } else {
                let (re, im) = complex_div(zero, -hm!(n - 1, n), hm!(n - 1, n - 1) - p, q);
                hm!(n - 1, n - 1) = re;
                hm!(n - 1, n) = im;
            }
            hm!(n, n - 1) = zero;
            hm!(n, n) = one;
            let mut i = n - 2;
            while i >= 0 {
                let (mut ra, mut sa) = (zero, zero);
                for j in l..=n {
                    ra = ra + hm!(i, j) * hm!(j, n - 1);
                    sa = sa + hm!(i, j) * hm!(j, n);
                }
                w = hm!(i, i) - p;
                if ee!(i) < zero {
                    z = w;
                    r = ra;
                    s = sa;
                } else {
                    l = i;
                    if ee!(i) == zero {
                        let (re, im) = complex_div(-ra, -sa, w, q);
                        hm!(i, n - 1) = re;
                        hm!(i, n) = im;
                    } else {
                        // Solve complex equations.
                        x = hm!(i, i + 1);
                        y = hm!(i + 1, i);
                        let mut vr = (dd!(i) - p) * (dd!(i) - p) + ee!(i) * ee!(i) - q * q;
                        let vi = (dd!(i) - p) * two * q;
                        if vr == zero && vi == zero {
                            vr = eps * norm * (w.abs() + q.abs() + x.abs() + y.abs() + z.abs());
                        }
                        let (re, im) =
                            complex_div(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
                        hm!(i, n - 1) = re;
                        hm!(i, n) = im;
                        if x.abs() > z.abs() + q.abs() {
                            hm!(i + 1, n - 1) = (-ra - w * hm!(i, n - 1) + q * hm!(i, n)) / x;
                            hm!(i + 1, n) = (-sa - w * hm!(i, n) - q * hm!(i, n - 1)) / x;
                        } else {
                            let (re, im) =
                                complex_div(-r - y * hm!(i, n - 1), -s - y * hm!(i, n), z, q);
                            hm!(i + 1, n - 1) = re;
                            hm!(i + 1, n) = im;
                        }
                    }
                    // Overflow control.
                    t = hm!(i, n - 1).abs().max_of(hm!(i, n).abs());
                    if (eps * t) * t > one {
                        for j in i..=n {
                            hm!(j, n - 1) = hm!(j, n - 1) / t;
                            hm!(j, n) = hm!(j, n) / t;
                        }
                    }
                }
                i -= 1;
            }
        }
        n -= 1;
    }

    // Back transformation to get the eigenvectors of the original matrix.
    for j in (low..size).rev() {
        for i in low..=high {
            z = zero;
            for k in low..=j.min(high) {
                z = z + vm!(i, k) * hm!(k, j);
            }
            vm!(i, j) = z;
        }
    }
    Ok(())
}

/// Eigenvalues and right eigenvectors of a general real square matrix, like `np.linalg.eig`.
///
/// The matrix is reduced to Hessenberg form and iterated with Francis' double-shift QR; the
/// eigenvectors come from back-substitution on the quasi-triangular Schur form.  Eigenvalues are
/// returned in the order they appear on the diagonal of the Schur form (complex conjugate pairs
/// adjacent, positive imaginary part first).  Each eigenvector has unit 2-norm and its component
/// of largest magnitude is real and positive.  No balancing is applied.
pub(crate) fn eig<T: KernelFloat>(a: &[T], n: usize) -> Result<Eig<T>, KernelError> {
    if a.len() != n * n {
        return Err(KernelError::Shape);
    }
    if !a.iter().all(|value| value.is_finite()) {
        return Err(KernelError::NoConvergence);
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    let mut h = a.to_vec();
    let mut v = vec![T::ZERO; n * n];
    let mut re = vec![T::ZERO; n];
    let mut im = vec![T::ZERO; n];
    hessenberg(&mut h, &mut v, n);
    francis_qr(&mut h, &mut v, &mut re, &mut im, n)?;

    let mut vec_re = vec![T::ZERO; n * n];
    let mut vec_im = vec![T::ZERO; n * n];
    let mut j = 0;
    while j < n {
        let pair = im[j] != T::ZERO && j + 1 < n;
        for row in 0..n {
            vec_re[row * n + j] = v[row * n + j];
            if pair {
                vec_im[row * n + j] = v[row * n + j + 1];
                vec_re[row * n + j + 1] = v[row * n + j];
                vec_im[row * n + j + 1] = -v[row * n + j + 1];
            }
        }
        j += if pair { 2 } else { 1 };
    }
    for col in 0..n {
        normalize_eigenvector(&mut vec_re, &mut vec_im, n, col);
    }
    Ok((re, im, vec_re, vec_im))
}

/// Scales column `col` to unit 2-norm and rotates its phase so the largest component is real and positive.
fn normalize_eigenvector<T: KernelFloat>(re: &mut [T], im: &mut [T], n: usize, col: usize) {
    let mut largest = 0;
    let mut largest_magnitude = T::ZERO;
    for row in 0..n {
        let (x, y) = (re[row * n + col], im[row * n + col]);
        let magnitude = x * x + y * y;
        if magnitude > largest_magnitude {
            largest_magnitude = magnitude;
            largest = row;
        }
    }
    if largest_magnitude == T::ZERO {
        return;
    }
    // Multiply by conj(v_largest) / |v_largest| so that v_largest becomes |v_largest|.
    let (pr, pi) = (re[largest * n + col], im[largest * n + col]);
    let length = largest_magnitude.sqrt();
    let (cr, ci) = (pr / length, -pi / length);
    let mut total = T::ZERO;
    for row in 0..n {
        let (x, y) = (re[row * n + col], im[row * n + col]);
        let (nx, ny) = (x * cr - y * ci, x * ci + y * cr);
        re[row * n + col] = nx;
        im[row * n + col] = ny;
        total = total + nx * nx + ny * ny;
    }
    im[largest * n + col] = T::ZERO;
    let length = total.sqrt();
    for row in 0..n {
        re[row * n + col] = re[row * n + col] / length;
        im[row * n + col] = im[row * n + col] / length;
    }
}

/// Eigenvalues `(re, im)` of a general real square matrix, like `np.linalg.eigvals`.
pub(crate) fn eigvals<T: KernelFloat>(a: &[T], n: usize) -> Result<(Vec<T>, Vec<T>), KernelError> {
    let (re, im, _, _) = eig(a, n)?;
    Ok((re, im))
}

/// Ascending eigenvalues of a real symmetric matrix, like `np.linalg.eigvalsh`.
pub(crate) fn eigvalsh<T: KernelFloat>(a: &[T], n: usize) -> Result<Vec<T>, KernelError> {
    Ok(eigh(a, n)?.0)
}

/// Sign and natural log of the absolute determinant, like `np.linalg.slogdet`.
///
/// An exactly singular matrix gives `(0, -inf)`.
pub(crate) fn slogdet<T: KernelFloat>(a: &[T], n: usize) -> Result<(T, T), KernelError> {
    let lu = match Lu::factor(a, n) {
        Ok(lu) => lu,
        Err(KernelError::Singular) => return Ok((T::ZERO, -T::INFINITY)),
        Err(error) => return Err(error),
    };
    let mut sign = if lu.swaps.is_multiple_of(2) {
        T::ONE
    } else {
        -T::ONE
    };
    let mut log_abs = T::ZERO;
    for i in 0..n {
        let diagonal = lu.lu[i * n + i];
        if diagonal < T::ZERO {
            sign = -sign;
        }
        log_abs = log_abs + diagonal.abs().ln();
    }
    Ok((sign, log_abs))
}

/// 2-norm condition number `s_max / s_min`, like `np.linalg.cond(a)`; `inf` for a singular matrix.
pub(crate) fn cond<T: KernelFloat>(a: &[T], rows: usize, cols: usize) -> Result<T, KernelError> {
    let (_, s, _, _) = svd(a, rows, cols)?;
    let (Some(&largest), Some(&smallest)) = (s.first(), s.last()) else {
        return Err(KernelError::Shape);
    };
    Ok(if smallest == T::ZERO {
        T::INFINITY
    } else {
        largest / smallest
    })
}

/// Condition number in the norm `order`, like `np.linalg.cond(a, p)`.
///
/// `Two` and `NegTwo` use the SVD (any shape); the other orders need a square matrix and are
/// `norm(a) * norm(inv(a))`, with `inf` when `a` is singular.
pub(crate) fn cond_ord<T: KernelFloat>(
    a: &[T],
    rows: usize,
    cols: usize,
    order: NormOrd,
) -> Result<T, KernelError> {
    match order {
        NormOrd::Two => cond(a, rows, cols),
        NormOrd::NegTwo => Ok(T::ONE / cond(a, rows, cols)?),
        NormOrd::Zero => Err(KernelError::Shape),
        _ => {
            if rows != cols {
                return Err(KernelError::Shape);
            }
            let forward = matrix_norm(a, rows, cols, order)?;
            match inv(a, rows) {
                Ok(inverse) => Ok(forward * matrix_norm(&inverse, rows, cols, order)?),
                Err(KernelError::Singular) => Ok(T::INFINITY),
                Err(error) => Err(error),
            }
        }
    }
}

/// Dynamic-programming matrix-chain ordering for matrices with dimensions
/// `dims[i] x dims[i + 1]`.
///
/// Returns the minimal number of scalar multiplications and the split table:
/// `splits[i * count + j]` is the index `k` such that the optimal product of matrices `i..=j` is
/// `(i..=k) @ (k + 1..=j)`.
pub(crate) fn matrix_chain_order(dims: &[usize]) -> (u128, Vec<usize>) {
    let count = dims.len().saturating_sub(1);
    let mut cost = vec![0_u128; count * count];
    let mut splits = vec![0_usize; count * count];
    for length in 2..=count {
        for i in 0..=count - length {
            let j = i + length - 1;
            cost[i * count + j] = u128::MAX;
            for k in i..j {
                let candidate = cost[i * count + k]
                    + cost[(k + 1) * count + j]
                    + dims[i] as u128 * dims[k + 1] as u128 * dims[j + 1] as u128;
                if candidate < cost[i * count + j] {
                    cost[i * count + j] = candidate;
                    splits[i * count + j] = k;
                }
            }
        }
    }
    let total = if count == 0 { 0 } else { cost[count - 1] };
    (total, splits)
}

/// Product of a chain of row-major matrices `(data, rows, cols)` using the cheapest
/// parenthesisation, like `np.linalg.multi_dot`.  Returns `(data, rows, cols)` of the product.
pub(crate) fn multi_dot<T>(
    matrices: &[(&[T], usize, usize)],
) -> Result<(Vec<T>, usize, usize), KernelError>
where
    T: Num,
{
    fn product<T>(matrices: &[(&[T], usize, usize)], splits: &[usize], i: usize, j: usize) -> Vec<T>
    where
        T: Num,
    {
        if i == j {
            return matrices[i].0.to_vec();
        }
        let k = splits[i * matrices.len() + j];
        let left = product(matrices, splits, i, k);
        let right = product(matrices, splits, k + 1, j);
        let (m, inner, n) = (matrices[i].1, matrices[k].2, matrices[j].2);
        gemm(m, n, inner, &left, inner as isize, 1, &right, n as isize, 1)
    }
    if matrices.len() < 2 {
        return Err(KernelError::Shape);
    }
    for (index, &(data, rows, cols)) in matrices.iter().enumerate() {
        if data.len() != rows * cols || (index > 0 && matrices[index - 1].2 != rows) {
            return Err(KernelError::Shape);
        }
    }
    let mut dims = vec![matrices[0].1];
    dims.extend(matrices.iter().map(|&(_, _, cols)| cols));
    let (_, splits) = matrix_chain_order(&dims);
    let result = product(matrices, &splits, 0, matrices.len() - 1);
    Ok((result, matrices[0].1, matrices[matrices.len() - 1].2))
}

/// Permutes the axes of a C-order buffer (`out.shape[i] = shape[perm[i]]`), like
/// `np.transpose(a, perm)` followed by a copy to C order.
pub(crate) fn permute_axes<T: Copy>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
) -> Result<Vec<T>, KernelError> {
    let ndim = shape.len();
    let mut seen = vec![false; ndim];
    if perm.len() != ndim
        || data.len() != shape.iter().product::<usize>()
        || perm
            .iter()
            .any(|&axis| axis >= ndim || core::mem::replace(&mut seen[axis], true))
    {
        return Err(KernelError::Shape);
    }
    let mut source_strides = vec![1_usize; ndim];
    for axis in (0..ndim.saturating_sub(1)).rev() {
        source_strides[axis] = source_strides[axis + 1] * shape[axis + 1];
    }
    let out_shape: Vec<usize> = perm.iter().map(|&axis| shape[axis]).collect();
    let out_strides: Vec<usize> = perm.iter().map(|&axis| source_strides[axis]).collect();
    let total = data.len();
    let mut out = Vec::with_capacity(total);
    if total == 0 {
        return Ok(out);
    }
    let mut index = vec![0_usize; ndim];
    let mut offset = 0;
    for _ in 0..total {
        out.push(data[offset]);
        // Odometer step, last axis fastest.
        for axis in (0..ndim).rev() {
            index[axis] += 1;
            offset += out_strides[axis];
            if index[axis] < out_shape[axis] {
                break;
            }
            offset -= out_strides[axis] * out_shape[axis];
            index[axis] = 0;
        }
    }
    Ok(out)
}

/// Tensor contraction over the paired axes `axes_a[i]` of `a` and `axes_b[i]` of `b`, like
/// `np.tensordot(a, b, axes=(axes_a, axes_b))`.
///
/// Both operands are permuted so the contracted axes are contiguous, viewed as matrices and
/// multiplied with [`gemm`].  The result shape is the free axes of `a` followed by those of `b`.
pub(crate) fn tensordot<T>(
    a: &[T],
    a_shape: &[usize],
    b: &[T],
    b_shape: &[usize],
    axes_a: &[usize],
    axes_b: &[usize],
) -> Result<(Vec<T>, Vec<usize>), KernelError>
where
    T: Num,
{
    if axes_a.len() != axes_b.len()
        || a.len() != a_shape.iter().product::<usize>()
        || b.len() != b_shape.iter().product::<usize>()
    {
        return Err(KernelError::Shape);
    }
    let mut used_a = vec![false; a_shape.len()];
    let mut used_b = vec![false; b_shape.len()];
    for (&ia, &ib) in axes_a.iter().zip(axes_b) {
        if ia >= a_shape.len()
            || ib >= b_shape.len()
            || a_shape[ia] != b_shape[ib]
            || core::mem::replace(&mut used_a[ia], true)
            || core::mem::replace(&mut used_b[ib], true)
        {
            return Err(KernelError::Shape);
        }
    }
    let free_a: Vec<usize> = (0..a_shape.len()).filter(|&axis| !used_a[axis]).collect();
    let free_b: Vec<usize> = (0..b_shape.len()).filter(|&axis| !used_b[axis]).collect();
    let m: usize = free_a.iter().map(|&axis| a_shape[axis]).product();
    let n: usize = free_b.iter().map(|&axis| b_shape[axis]).product();
    let k: usize = axes_a.iter().map(|&axis| a_shape[axis]).product();
    let perm_a: Vec<usize> = free_a.iter().chain(axes_a).copied().collect();
    let perm_b: Vec<usize> = axes_b.iter().chain(&free_b).copied().collect();
    let is_identity = |perm: &[usize]| perm.iter().enumerate().all(|(i, &axis)| i == axis);
    let a_owned;
    let a_matrix: &[T] = if is_identity(&perm_a) {
        a
    } else {
        a_owned = permute_axes(a, a_shape, &perm_a)?;
        &a_owned
    };
    let b_owned;
    let b_matrix: &[T] = if is_identity(&perm_b) {
        b
    } else {
        b_owned = permute_axes(b, b_shape, &perm_b)?;
        &b_owned
    };
    let result = gemm(m, n, k, a_matrix, k as isize, 1, b_matrix, n as isize, 1);
    let shape = free_a
        .iter()
        .map(|&axis| a_shape[axis])
        .chain(free_b.iter().map(|&axis| b_shape[axis]))
        .collect();
    Ok((result, shape))
}

/// Dot product of the flattened operands, like `np.vdot(a, b)` for real data.
pub(crate) fn vdot<T>(a: &[T], b: &[T]) -> Result<T, KernelError>
where
    T: Num,
{
    if a.len() != b.len() {
        return Err(KernelError::Shape);
    }
    Ok(a.iter()
        .zip(b)
        .fold(T::ZERO, |sum, (&x, &y)| add(sum, mul(x, y))))
}

/// Integer power of a square matrix, like `np.linalg.matrix_power(a, power)`.
///
/// `power == 0` gives the identity (even for singular input); negative powers invert `a` first
/// and fail with [`KernelError::Singular`] when it has no inverse.
pub(crate) fn matrix_power<T: KernelFloat + Num>(
    a: &[T],
    n: usize,
    power: i64,
) -> Result<Vec<T>, KernelError> {
    if a.len() != n * n {
        return Err(KernelError::Shape);
    }
    let mut result = vec![<T as KernelFloat>::ZERO; n * n];
    for i in 0..n {
        result[i * n + i] = <T as KernelFloat>::ONE;
    }
    if power == 0 {
        return Ok(result);
    }
    let mut base = if power < 0 { inv(a, n)? } else { a.to_vec() };
    let mut exponent = power.unsigned_abs();
    let stride = n as isize;
    loop {
        if exponent & 1 == 1 {
            result = gemm(n, n, n, &result, stride, 1, &base, stride, 1);
        }
        exponent >>= 1;
        if exponent == 0 {
            return Ok(result);
        }
        base = gemm(n, n, n, &base, stride, 1, &base, stride, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!(
            (left - right).abs() <= 1e-10 * right.abs().max(1.0),
            "{left} != {right}"
        );
    }
    fn product(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        gemm(m, n, k, a, k as isize, 1, b, n as isize, 1)
    }

    #[test]
    fn gemm_handles_transposed_strides() {
        let a = [1.0_f64, 3.0, 2.0, 4.0]; // logical [[1, 2], [3, 4]] with strides [1, 2]
        let b = [5.0_f64, 6.0, 7.0, 8.0];
        assert_eq!(
            gemm(2, 2, 2, &a, 1, 2, &b, 2, 1),
            vec![19.0, 22.0, 43.0, 50.0]
        );
        let mut c = [0.0; 4];
        gemm_f64(2, 2, 2, 1.0, &a, 1, 2, &b, 2, 1, 0.0, &mut c, 2, 1);
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn lu_inverse_and_solve_have_ground_truth_residuals() {
        let a = [4.0, 7.0, 2.0, 6.0];
        let inverse = inv(&a, 2).unwrap();
        let identity = product(&a, &inverse, 2, 2, 2);
        close(identity[0], 1.0);
        close(identity[1], 0.0);
        close(identity[2], 0.0);
        close(identity[3], 1.0);
        close(Lu::factor(&a, 2).unwrap().det(), 10.0);
        let solution = Lu::factor(&a, 2).unwrap().solve(&[1.0, 0.0], 1).unwrap();
        let residual = product(&a, &solution, 2, 2, 1);
        close(residual[0], 1.0);
        close(residual[1], 0.0);
        assert!(matches!(
            Lu::factor(&[1.0, 2.0, 2.0, 4.0], 2),
            Err(KernelError::Singular)
        ));
    }

    #[test]
    fn cholesky_reconstructs_input() {
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        let reconstructed = product(&l, &[l[0], l[2], l[1], l[3]], 2, 2, 2);
        for (actual, expected) in reconstructed.iter().zip(a) {
            close(*actual, expected);
        }
    }

    #[test]
    fn qr_is_orthonormal_and_reconstructs() {
        let a = [12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0];
        let (q, r, p) = qr(&a, 3, 3).unwrap();
        assert_eq!(p, 3);
        let qt = [q[0], q[3], q[6], q[1], q[4], q[7], q[2], q[5], q[8]];
        let identity = product(&qt, &q, 3, 3, 3);
        for i in 0..3 {
            for j in 0..3 {
                close(identity[i * 3 + j], if i == j { 1.0 } else { 0.0 });
            }
        }
        let reconstructed = product(&q, &r, 3, 3, 3);
        for (actual, expected) in reconstructed.iter().zip(a) {
            close(*actual, expected);
        }
    }

    #[test]
    fn eigh_satisfies_eigen_equation() {
        let a = [2.0, 1.0, 1.0, 2.0];
        let (values, vectors) = eigh(&a, 2).unwrap();
        assert!(values[0] <= values[1]);
        for col in 0..2 {
            for row in 0..2 {
                close(
                    a[row * 2] * vectors[col] + a[row * 2 + 1] * vectors[2 + col],
                    values[col] * vectors[row * 2 + col],
                );
            }
        }
    }

    #[test]
    fn svd_reconstructs_and_rank_is_correct() {
        let a = [3.0, 1.0, 1.0, 3.0];
        let (u, s, vt, p) = svd(&a, 2, 2).unwrap();
        assert_eq!(p, 2);
        assert!(s[0] >= s[1]);
        let scaled_u = [u[0] * s[0], u[1] * s[1], u[2] * s[0], u[3] * s[1]];
        let reconstructed = product(&scaled_u, &vt, 2, 2, 2);
        for (actual, expected) in reconstructed.iter().zip(a) {
            close(*actual, expected);
        }
        assert_eq!(matrix_rank(&[1.0, 2.0, 2.0, 4.0], 2, 2).unwrap(), 1);
    }

    #[test]
    fn least_squares_pseudoinverse_and_norms_match_hand_values() {
        let a = [1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let x = lstsq(&a, 3, 2, &[1.0, 2.0, 2.0], 1).unwrap();
        let fitted = product(&a, &x, 3, 2, 1);
        // The least-squares residual is orthogonal to each column of A.
        close(fitted[0] + fitted[1] + fitted[2], 5.0);
        close(fitted[0] + 2.0 * fitted[1] + 3.0 * fitted[2], 11.0);
        let inverse = pinv(&[1.0, 2.0, 2.0, 4.0], 2, 2).unwrap();
        let reconstructed = product(&[1.0, 2.0, 2.0, 4.0], &inverse, 2, 2, 2);
        let projection = product(&reconstructed, &[1.0, 2.0, 2.0, 4.0], 2, 2, 2);
        for (actual, expected) in projection.iter().zip([1.0, 2.0, 2.0, 4.0]) {
            close(*actual, expected);
        }
        close(
            matrix_norm(&[1.0, -2.0, 3.0, -4.0], 2, 2, NormOrd::One).unwrap(),
            6.0,
        );
        assert_eq!(
            norm_axis(&[3.0, 4.0], 1, 2, 1, NormOrd::Two).unwrap(),
            vec![5.0]
        );
    }
}
