//! Slice-level numerical kernels used by the future `Array` linalg adapters.
//!
//! All matrices are row-major logical matrices.  GEMM accepts element strides,
//! so a transposed `Array` view can eventually call it without materialising a
//! copy.  Other routines take contiguous row-major inputs and return owned
//! row-major results.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The deliberately small floating-point surface needed by these kernels.
pub(crate) trait KernelFloat:
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

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
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

            fn abs(self) -> Self {
                self.abs()
            }
            fn sqrt(self) -> Self {
                self.sqrt()
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
}

/// Matrix norm choices supported by [`matrix_norm`] and [`norm_axis`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NormOrd {
    Fro,
    Two,
    One,
    Inf,
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

fn output_is_non_aliasing(rows: usize, cols: usize, row_stride: isize, col_stride: isize) -> bool {
    for first_row in 0..rows {
        for first_col in 0..cols {
            let first = first_row as isize * row_stride + first_col as isize * col_stride;
            for second_row in first_row..rows {
                let start_col = if second_row == first_row {
                    first_col + 1
                } else {
                    0
                };
                for second_col in start_col..cols {
                    if first == second_row as isize * row_stride + second_col as isize * col_stride
                    {
                        return false;
                    }
                }
            }
        }
    }
    true
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
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    assert_matrix_fits(a, m, k, a_row_stride, a_col_stride);
    assert_matrix_fits(b, k, n, b_row_stride, b_col_stride);
    let mut result = vec![T::default(); m * n];
    // i-k-j keeps the output row hot in cache and also avoids an allocation per
    // element for strided Array views.
    for i in 0..m {
        for p in 0..k {
            let left = a[(i as isize * a_row_stride + p as isize * a_col_stride) as usize];
            for j in 0..n {
                let right = b[(p as isize * b_row_stride + j as isize * b_col_stride) as usize];
                let index = i * n + j;
                result[index] = result[index] + left * right;
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
        matrixmultiply::sgemm(
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
        matrixmultiply::dgemm(
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
        let mut norm_sq = T::ZERO;
        for row in k..rows {
            norm_sq = norm_sq + r[row * cols + k] * r[row * cols + k];
        }
        let norm = norm_sq.sqrt();
        if norm <= T::EPSILON {
            continue;
        }
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

/// Least-squares solution using the reduced QR factorisation for tall systems.
pub(crate) fn lstsq<T: KernelFloat>(
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
        let inverse = pinv(a, rows, cols)?;
        return Ok(gemm(
            cols,
            rhs_cols,
            rows,
            &inverse,
            rows as isize,
            1,
            b,
            rhs_cols as isize,
            1,
        ));
    }
    let (q, r, _) = qr(a, rows, cols)?;
    let mut y = vec![T::ZERO; cols * rhs_cols];
    for i in 0..cols {
        for j in 0..rhs_cols {
            for row in 0..rows {
                y[i * rhs_cols + j] =
                    y[i * rhs_cols + j] + q[row * cols + i] * b[row * rhs_cols + j];
            }
        }
    }
    for row in (0..cols).rev() {
        if r[row * cols + row].abs() <= T::EPSILON {
            return Err(KernelError::Singular);
        }
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
    let mut work = a.to_vec();
    let mut vectors = vec![T::ZERO; n * n];
    for i in 0..n {
        vectors[i * n + i] = T::ONE;
    }
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
        let mut diagonal = T::ZERO;
        for i in 0..n {
            diagonal = diagonal + work[i * n + i].abs();
        }
        if largest <= T::EPSILON * (diagonal + T::ONE) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let off = work[p * n + q];
                if off.abs() <= T::EPSILON {
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
                    || gamma.abs() <= T::EPSILON * (alpha * beta).sqrt()
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
    for new_col in 0..reduced {
        let old_col = ordering[new_col];
        s[new_col] = singular_all[old_col];
        if s[new_col] > T::EPSILON {
            for row in 0..rows {
                u[row * reduced + new_col] = columns[row * cols + old_col] / s[new_col];
            }
        }
        for col in 0..cols {
            vt[new_col * cols + col] = right[col * cols + old_col];
        }
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

/// Matrix norm: Frobenius, spectral (2), maximum column sum (1), or maximum row sum (∞).
pub(crate) fn matrix_norm<T: KernelFloat>(
    a: &[T],
    rows: usize,
    cols: usize,
    order: NormOrd,
) -> Result<T, KernelError> {
    if a.len() != rows * cols {
        return Err(KernelError::Shape);
    }
    match order {
        NormOrd::Fro => Ok(norm(a)),
        NormOrd::Two => Ok(svd(a, rows, cols)?.1.first().copied().unwrap_or(T::ZERO)),
        NormOrd::One => {
            let mut best = T::ZERO;
            for col in 0..cols {
                let mut sum = T::ZERO;
                for row in 0..rows {
                    sum = sum + a[row * cols + col].abs();
                }
                if sum > best {
                    best = sum;
                }
            }
            Ok(best)
        }
        NormOrd::Inf => {
            let mut best = T::ZERO;
            for row in 0..rows {
                let mut sum = T::ZERO;
                for col in 0..cols {
                    sum = sum + a[row * cols + col].abs();
                }
                if sum > best {
                    best = sum;
                }
            }
            Ok(best)
        }
    }
}

/// Norm along one axis of a contiguous two-dimensional matrix.
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
    let len = if axis == 0 { cols } else { rows };
    let mut output = vec![T::ZERO; len];
    for out in 0..len {
        let mut sum = T::ZERO;
        let mut max = T::ZERO;
        let count = if axis == 0 { rows } else { cols };
        for index in 0..count {
            let value = if axis == 0 {
                a[index * cols + out].abs()
            } else {
                a[out * cols + index].abs()
            };
            sum = sum + value;
            if value > max {
                max = value;
            }
        }
        output[out] = match order {
            NormOrd::Fro | NormOrd::Two => {
                let mut squares = T::ZERO;
                for index in 0..count {
                    let value = if axis == 0 {
                        a[index * cols + out]
                    } else {
                        a[out * cols + index]
                    };
                    squares = squares + value * value;
                }
                squares.sqrt()
            }
            NormOrd::One => sum,
            NormOrd::Inf => max,
        };
    }
    Ok(output)
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
