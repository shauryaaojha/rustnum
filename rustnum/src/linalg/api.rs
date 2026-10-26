//! Array-facing linear algebra: `Array::matmul`, `Array::dot`, `Array::trace` and the free
//! functions of `np::linalg`.
//!
//! `f32` and `f64` products call the packed GEMM engine with the real strides of the operands, so
//! transposed and sliced views are multiplied without copying. Other element types use a
//! generic loop on contiguous copies. Decompositions work on `f32` and `f64` matrices
//! (the `LinalgFloat` bound); v0.1 has no complex dtype, so complex results (`eig`, `eigvals`)
//! are returned as separate real and imaginary arrays.

use super::kernels::{self, KernelError, KernelFloat, NormOrd};
use crate::broadcast::broadcast_shapes;
use crate::element::{add, mul};
use crate::layout::advance;
use crate::{Array, Element, Error, Float, Num, Result};
use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::sync::Arc;

/// Floating-point element types accepted by the decompositions in `np::linalg`: `f32` and `f64`.
///
/// This is a sealed marker: it is implemented for every `Float` that also has a numerical
/// kernel, which today means exactly `f32` and `f64`.
///
/// ```
/// use rustnum as np;
/// fn det_twice<F: np::linalg::LinalgFloat>(a: &np::Array<F>) -> (F, F) {
///     (np::linalg::det(a), np::linalg::det(a))
/// }
/// let (x, y) = det_twice(&np::array![[2.0f32, 0.0], [0.0, 3.0]]);
/// assert_eq!((x, y), (6.0, 6.0));
/// ```
pub trait LinalgFloat: Float + KernelFloat {}
impl<T: Float + KernelFloat> LinalgFloat for T {}

const EIGEN_FAILED: &str = "Eigenvalues did not converge";
const SVD_FAILED: &str = "SVD did not converge";

fn shape_error(message: impl Into<String>) -> Error {
    Error::Shape(message.into())
}
/// Maps a kernel failure; `no_convergence` is NumPy's wording for the solver involved.
fn kernel_error(error: KernelError, no_convergence: &str) -> Error {
    match error {
        KernelError::Singular => Error::Singular,
        KernelError::NotPositiveDefinite => Error::NotPositiveDefinite,
        KernelError::NoConvergence => Error::NoConvergence(no_convergence.into()),
        KernelError::Shape => shape_error("Last 2 dimensions of the array must be square"),
    }
}
fn eigen_error(error: KernelError) -> Error {
    kernel_error(error, EIGEN_FAILED)
}
fn svd_error(error: KernelError) -> Error {
    kernel_error(error, SVD_FAILED)
}
/// Unwraps a kernel result of a function whose NumPy counterpart raises (panics here).
fn or_panic<T>(result: std::result::Result<T, KernelError>, no_convergence: &str) -> T {
    result.unwrap_or_else(|error| panic!("{}", kernel_error(error, no_convergence)))
}
fn fail<T>(result: Result<T>) -> T {
    result.unwrap_or_else(|error| panic!("{error}"))
}
/// Contiguous row-major elements, borrowed when possible.
fn data<T: Element>(a: &Array<T>) -> Cow<'_, [T]> {
    match a.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(a.to_vec()),
    }
}
fn matrix_shape<T: Element>(a: &Array<T>) -> Result<(usize, usize)> {
    if a.ndim() != 2 {
        return Err(shape_error(format!(
            "{}-dimensional array given. Array must be two-dimensional",
            a.ndim()
        )));
    }
    Ok((a.shape()[0], a.shape()[1]))
}
fn square_shape<T: Element>(a: &Array<T>) -> Result<usize> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(shape_error("Last 2 dimensions of the array must be square"));
    }
    Ok(rows)
}

/// Strides of the two core (matrix) axes of `a`; a 1-d operand is promoted to a row vector when
/// `left`, to a column vector otherwise.
fn core_strides<T: Element>(a: &Array<T>, left: bool) -> (isize, isize) {
    let strides = a.strides();
    match (a.ndim(), left) {
        (1, true) => (0, strides[0]),
        (1, false) => (strides[0], 0),
        (n, _) => (strides[n - 2], strides[n - 1]),
    }
}

/// A matmul operand ready for the stride-aware kernels.
struct Operand<T> {
    array: Array<T>,
    /// Strides of the broadcast batch axes (0 where the operand is broadcast).
    batch: Vec<isize>,
    row: isize,
    col: isize,
}

fn operand<T: Num>(
    a: &Array<T>,
    left: bool,
    batch_shape: &[usize],
    strided_ok: bool,
) -> Operand<T> {
    // Native packing reads arbitrary signed strides; generic types use contiguous memory.
    let array = if strided_ok {
        a.clone()
    } else {
        a.to_contiguous()
    };
    let (row, col) = core_strides(&array, left);
    let core = if array.ndim() == 1 { 1 } else { 2 };
    let own_batch = array.ndim() - core;
    let mut batch = vec![0; batch_shape.len()];
    for axis in 0..own_batch {
        if array.shape()[axis] != 1 {
            batch[batch_shape.len() - own_batch + axis] = array.strides()[axis];
        }
    }
    Operand {
        array,
        batch,
        row,
        col,
    }
}

/// Calls `f(batch_index, left_start, right_start)` for every batch of the broadcast product.
fn for_each_batch<T: Element>(
    batch_shape: &[usize],
    left: &Operand<T>,
    right: &Operand<T>,
    mut f: impl FnMut(usize, usize, usize),
) {
    let batches: usize = batch_shape.iter().product();
    let mut coordinate = vec![0; batch_shape.len()];
    let (_, left_offset, _) = left.array.raw_parts();
    let (_, right_offset, _) = right.array.raw_parts();
    for index in 0..batches {
        let mut a_start = left_offset as isize;
        let mut b_start = right_offset as isize;
        for (axis, &c) in coordinate.iter().enumerate() {
            a_start += c as isize * left.batch[axis];
            b_start += c as isize * right.batch[axis];
        }
        f(index, a_start as usize, b_start as usize);
        advance(&mut coordinate, batch_shape);
    }
}

/// Generic strided matrix product with wrapping arithmetic: `c = a @ b`, `c` row-major `m x n`.
fn gemm_generic<T: Num>(
    (m, k, n): (usize, usize, usize),
    a: &[T],
    a_strides: (isize, isize),
    b: &[T],
    b_strides: (isize, isize),
    c: &mut [T],
) {
    for i in 0..m {
        for p in 0..k {
            let left = a[(i as isize * a_strides.0 + p as isize * a_strides.1) as usize];
            let row = &mut c[i * n..(i + 1) * n];
            for (j, slot) in row.iter_mut().enumerate() {
                let right = b[(p as isize * b_strides.0 + j as isize * b_strides.1) as usize];
                *slot = add(*slot, mul(left, right));
            }
        }
    }
}

/// Borrows the storage of `a` as `&[U]` when `T` is `U`.
fn storage_as<T: Element, U: Element>(a: &Array<T>) -> Option<&[U]> {
    (&a.data as &dyn Any)
        .downcast_ref::<Arc<Vec<U>>>()
        .map(|data| data.as_slice())
}

impl<T: Num> Array<T> {
    /// Matrix product, like `np.matmul(a, b)` or `a @ b`.
    ///
    /// Follows NumPy: two 1-d arrays give a 0-d dot product; a 1-d operand is promoted to a
    /// row (left) or column (right) vector and removed from the result; with more than two
    /// dimensions the leading "batch" axes broadcast. `f32` and `f64` operands go straight to
    /// the packed GEMM engine using the real strides of the views (no copy for transposed or sliced
    /// operands); other types use a generic loop with wrapping integer arithmetic.
    ///
    /// Panics when the inner dimensions differ, an operand is 0-d, or the batch shapes cannot
    /// be broadcast.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::array![[1, 2], [3, 4]];
    /// assert_eq!(a.matmul(&a).to_vec(), [7, 10, 15, 22]);
    /// // views are multiplied without copying
    /// let f = np::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// assert_eq!(f.matmul(&f.t()).to_vec(), [14.0, 32.0, 32.0, 77.0]);
    /// // 1-d promotion and batch broadcasting
    /// let v = np::array![1.0, 1.0, 1.0];
    /// assert_eq!(f.matmul(&v).to_vec(), [6.0, 15.0]);
    /// let batch = np::ones::<f64>([2, 3, 4]).matmul(&np::ones::<f64>([4, 5]));
    /// assert_eq!(batch.shape(), [2, 3, 5]);
    /// ```
    pub fn matmul(&self, other: &Array<T>) -> Array<T> {
        let mut out = Array::default();
        self.matmul_write(other, &mut out, true, T::ONE, None);
        out
    }

    /// Matrix product into an existing output, like `np.matmul(a, b, out=out)`.
    ///
    /// Supports the same vector promotion and batch broadcasting as `matmul`.
    /// Panics unless `out` is C-contiguous and has exactly the result shape.
    /// Reuses uniquely owned output storage; shared storage is detached on write.
    /// GEMM packing still uses temporary workspace, but no new result is allocated.
    ///
    /// ```
    /// use rustnum as np;
    /// let a = np::array![[1.0, 2.0], [3.0, 4.0]];
    /// let mut out = np::zeros::<f64>([2, 2]);
    /// a.matmul_into(&a, &mut out);
    /// assert_eq!(out.to_vec(), [7.0, 10.0, 15.0, 22.0]);
    /// ```
    pub fn matmul_into(&self, other: &Array<T>, out: &mut Array<T>) {
        self.matmul_write(other, out, false, T::ONE, None);
    }

    fn matmul_write(
        &self,
        other: &Array<T>,
        output: &mut Array<T>,
        allocate: bool,
        alpha: T,
        map: Option<fn(T) -> T>,
    ) {
        for (index, array) in [self, other].into_iter().enumerate() {
            assert!(
                array.ndim() > 0,
                "matmul: Input operand {index} does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)"
            );
        }
        let (left_vector, right_vector) = (self.ndim() == 1, other.ndim() == 1);
        let (m, k) = if left_vector {
            (1, self.shape()[0])
        } else {
            (self.shape()[self.ndim() - 2], self.shape()[self.ndim() - 1])
        };
        let (other_k, n) = if right_vector {
            (other.shape()[0], 1)
        } else {
            (
                other.shape()[other.ndim() - 2],
                other.shape()[other.ndim() - 1],
            )
        };
        assert!(
            k == other_k,
            "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size {other_k} is different from {k})"
        );
        let left_batch = &self.shape()[..self.ndim() - if left_vector { 1 } else { 2 }];
        let right_batch = &other.shape()[..other.ndim() - if right_vector { 1 } else { 2 }];
        let batch_shape = broadcast_shapes(left_batch, right_batch).unwrap_or_else(|| {
            panic!(
                "operands could not be broadcast together with remapped shapes [original->remapped]: {:?} {:?}",
                self.shape(),
                other.shape()
            )
        });
        let mut output_shape = batch_shape.clone();
        if !left_vector {
            output_shape.push(m);
        }
        if !right_vector {
            output_shape.push(n);
        }

        if allocate {
            *output = Array::from_vec(vec![T::ZERO; output_shape.iter().product()], &output_shape);
        } else {
            assert_eq!(
                output.shape(),
                output_shape,
                "matmul: output operand has a mismatch in its core dimension"
            );
            assert!(
                output.is_contiguous(),
                "matmul: output array is not C-contiguous"
            );
        }
        let native =
            TypeId::of::<T>() == TypeId::of::<f64>() || TypeId::of::<T>() == TypeId::of::<f32>();
        let left = operand(self, true, &batch_shape, native);
        let right = operand(other, false, &batch_shape, native);
        let block = m * n;
        let total = batch_shape.iter().product::<usize>() * block;
        if total == 0 {
            return;
        }
        let (a_strides, b_strides) = ((left.row, left.col), (right.row, right.col));
        let offset = output.offset;
        let out = Arc::make_mut(&mut output.data);

        macro_rules! native_product {
            ($float:ty) => {
                if let (Some(a), Some(b)) = (
                    storage_as::<T, $float>(&left.array),
                    storage_as::<T, $float>(&right.array),
                ) {
                    let out = (out as &mut dyn Any)
                        .downcast_mut::<Vec<$float>>()
                        .expect("same type");
                    let alpha = *(&alpha as &dyn Any)
                        .downcast_ref::<$float>()
                        .expect("same type");
                    let map = *(&map as &dyn Any)
                        .downcast_ref::<Option<fn($float) -> $float>>()
                        .expect("same type");
                    for_each_batch(&batch_shape, &left, &right, |index, a_start, b_start| {
                        super::packed_gemm::run(
                            (m, k, n),
                            alpha,
                            super::packed_gemm::Matrix {
                                data: a,
                                offset: a_start,
                                row: a_strides.0,
                                col: a_strides.1,
                            },
                            super::packed_gemm::Matrix {
                                data: b,
                                offset: b_start,
                                row: b_strides.0,
                                col: b_strides.1,
                            },
                            0.0,
                            &mut out[offset + index * block..offset + (index + 1) * block],
                            0,
                            (n as isize, 1),
                            map,
                        );
                    });
                    return;
                }
            };
        }
        native_product!(f64);
        native_product!(f32);
        let out = &mut out[offset..offset + total];
        out.fill(T::ZERO);
        if k != 0 {
            let (a, _, _) = left.array.raw_parts();
            let (b, _, _) = right.array.raw_parts();
            for_each_batch(&batch_shape, &left, &right, |index, a_start, b_start| {
                gemm_generic(
                    (m, k, n),
                    &a[a_start..],
                    a_strides,
                    &b[b_start..],
                    b_strides,
                    &mut out[index * block..(index + 1) * block],
                );
            });
        }
    }

    /// Dot product, like `np.dot(a, b)` for arrays of at most two dimensions.
    ///
    /// A 0-d operand multiplies element-wise; 1-d with 1-d is the inner product (0-d result);
    /// otherwise this is `matmul`. Panics for arrays with more than two dimensions or
    /// misaligned shapes.
    ///
    /// ```
    /// use rustnum as np;
    /// let v = np::array![1, 2, 3];
    /// assert_eq!(v.dot(&v).item(), 14);
    /// let m = np::array![[1, 2], [3, 4]];
    /// assert_eq!(m.dot(&np::array![1, 1]).to_vec(), [3, 7]);
    /// assert_eq!(m.dot(&np::Array::scalar(2)).to_vec(), [2, 4, 6, 8]);
    /// ```
    pub fn dot(&self, other: &Array<T>) -> Array<T> {
        assert!(
            self.ndim() <= 2 && other.ndim() <= 2,
            "dot only supports arrays with at most 2 dimensions"
        );
        if self.ndim() == 0 || other.ndim() == 0 {
            return self.zip_map(other.clone(), mul);
        }
        self.matmul(other)
    }

    /// Sum along the main diagonal of a 2-d array, like `np.trace(a)`.
    ///
    /// Panics unless the array is 2-d. Integer sums wrap on overflow.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::array![[1, 2, 3], [4, 5, 6]].trace(), 6);
    /// assert_eq!(np::array![[1.5, 0.0], [0.0, 2.5]].t().trace(), 4.0);
    /// ```
    pub fn trace(&self) -> T {
        assert!(
            self.ndim() == 2,
            "diag requires an array of at least two dimensions"
        );
        let (storage, offset, strides) = self.raw_parts();
        let diagonal = strides[0] + strides[1];
        (0..self.shape()[0].min(self.shape()[1])).fold(T::ZERO, |sum, i| {
            add(
                sum,
                storage[(offset as isize + i as isize * diagonal) as usize],
            )
        })
    }
}

/// Scaled matrix product plus an addend, like `scipy.linalg.blas.dgemm` / `sgemm`.
///
/// Returns `alpha * a @ b + beta * c`. Inputs must be two-dimensional and aligned;
/// `c` must have the exact product shape. All inputs may be strided views.
/// Zero alpha ignores A/B values; zero beta ignores C values (including NaNs).
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1.0, 2.0], [3.0, 4.0]];
/// let c = np::ones::<f64>([2, 2]);
/// assert_eq!(np::linalg::gemm(2.0, &a, &a, 3.0, &c).to_vec(), [17.0, 23.0, 33.0, 47.0]);
/// ```
pub fn gemm<F: LinalgFloat>(
    alpha: F,
    a: &Array<F>,
    b: &Array<F>,
    beta: F,
    c: &Array<F>,
) -> Array<F> {
    let (m, k) = fail(matrix_shape(a));
    let (bk, n) = fail(matrix_shape(b));
    assert_eq!(
        k, bk,
        "matmul: Input operand 1 has a mismatch in its core dimension 0"
    );
    assert_eq!(
        c.shape(),
        [m, n],
        "gemm: output operand has a mismatch in its core dimension"
    );
    let mut out = if beta == <F as Num>::ZERO {
        vec![<F as Num>::ZERO; m * n]
    } else {
        c.to_vec()
    };
    macro_rules! typed {
        ($float:ty) => {
            if let (Some(ad), Some(bd)) = (storage_as::<F, $float>(a), storage_as::<F, $float>(b)) {
                let scalar = |x: &F| *(x as &dyn Any).downcast_ref::<$float>().expect("same type");
                super::packed_gemm::run(
                    (m, k, n),
                    scalar(&alpha),
                    super::packed_gemm::Matrix {
                        data: ad,
                        offset: a.offset,
                        row: a.strides()[0],
                        col: a.strides()[1],
                    },
                    super::packed_gemm::Matrix {
                        data: bd,
                        offset: b.offset,
                        row: b.strides()[0],
                        col: b.strides()[1],
                    },
                    scalar(&beta),
                    (&mut out as &mut dyn Any)
                        .downcast_mut::<Vec<$float>>()
                        .expect("same type"),
                    0,
                    (n as isize, 1),
                    None,
                );
                return Array::from_vec(out, &[m, n]);
            }
        };
    }
    typed!(f64);
    typed!(f32);
    unreachable!("LinalgFloat is sealed to f32 and f64")
}

/// Fused scaled matrix product and map, like `f(alpha * np.matmul(a, b))`.
///
/// The function is applied once per output element after the complete reduction,
/// while its finished tile is in cache. Supports vector promotion and broadcast
/// batches, and accepts a non-capturing function pointer for f32 or f64.
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1.0f64, 2.0]];
/// let y = np::linalg::matmul_map(&a, &a.t(), 0.2, f64::tanh);
/// assert!((y.item() - 1.0f64.tanh()).abs() < 1e-12);
/// ```
pub fn matmul_map<F: LinalgFloat>(a: &Array<F>, b: &Array<F>, alpha: F, f: fn(F) -> F) -> Array<F> {
    let mut out = Array::default();
    a.matmul_write(b, &mut out, true, alpha, Some(f));
    out
}

/// Outer product of two arrays (flattened first), like `np.outer(a, b)`.
///
/// ```
/// use rustnum as np;
/// let o = np::linalg::outer(&np::array![1, 2], &np::array![3, 4, 5]);
/// assert_eq!(o.shape(), [2, 3]);
/// assert_eq!(o.to_vec(), [3, 4, 5, 6, 8, 10]);
/// ```
pub fn outer<T: Num>(a: &Array<T>, b: &Array<T>) -> Array<T> {
    let (left, right) = (data(a), data(b));
    let output = left
        .iter()
        .flat_map(|&x| right.iter().map(move |&y| mul(x, y)))
        .collect();
    Array::from_vec(output, &[left.len(), right.len()])
}

/// Inner product of two 1-d arrays, like `np.inner(a, b)`.
///
/// Panics unless both are 1-d with the same length.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::linalg::inner(&np::array![1, 2, 3], &np::array![4, 5, 6]), 32);
/// ```
pub fn inner<T: Num>(a: &Array<T>, b: &Array<T>) -> T {
    assert!(
        a.ndim() == 1 && b.ndim() == 1,
        "inner: expected two 1-dimensional arrays"
    );
    assert!(
        a.size() == b.size(),
        "shapes {:?} and {:?} not aligned: {} (dim 0) != {} (dim 0)",
        a.shape(),
        b.shape(),
        a.size(),
        b.size()
    );
    data(a)
        .iter()
        .zip(data(b).iter())
        .fold(T::ZERO, |sum, (&x, &y)| add(sum, mul(x, y)))
}

/// Determinant of a square matrix, like `np.linalg.det(a)`.
///
/// Panics unless `a` is a square matrix. A singular matrix gives `0`.
///
/// ```
/// use rustnum as np;
/// let d = np::linalg::det(&np::array![[1.0f64, 2.0], [3.0, 4.0]]);
/// assert!((d + 2.0).abs() < 1e-12);
/// assert_eq!(np::linalg::det(&np::array![[1.0, 2.0], [2.0, 4.0]]), 0.0);
/// ```
pub fn det<F: LinalgFloat>(a: &Array<F>) -> F {
    let n = fail(square_shape(a));
    match kernels::Lu::factor(&data(a), n) {
        Ok(factor) => factor.det(),
        Err(_) => <F as Num>::ZERO,
    }
}

/// Inverse of a nonsingular square matrix, like `np.linalg.inv(a)`.
///
/// Returns `Err(Error::Singular)` for a singular matrix and `Err(Error::Shape)` for input that
/// is not a square matrix.
///
/// ```
/// use rustnum as np;
/// let inv = np::linalg::inv(&np::array![[2.0, 0.0], [0.0, 4.0]]).unwrap();
/// assert_eq!(inv.to_vec(), [0.5, 0.0, 0.0, 0.25]);
/// assert!(matches!(np::linalg::inv(&np::zeros::<f64>([2, 2])), Err(np::Error::Singular)));
/// ```
pub fn inv<F: LinalgFloat>(a: &Array<F>) -> Result<Array<F>> {
    let n = square_shape(a)?;
    kernels::inv(&data(a), n)
        .map(|inverse| Array::from_vec(inverse, &[n, n]))
        .map_err(eigen_error)
}

/// Solution `x` of `a @ x = b`, like `np.linalg.solve(a, b)`.
///
/// `b` may be a vector or a matrix of right-hand sides. Returns `Err(Error::Singular)` when `a`
/// is singular and `Err(Error::Shape)` for misaligned shapes.
///
/// ```
/// use rustnum as np;
/// let a = np::array![[3.0f64, 1.0], [1.0, 2.0]];
/// let x = np::linalg::solve(&a, &np::array![9.0, 8.0]).unwrap();
/// assert!((x.to_vec()[0] - 2.0).abs() < 1e-12 && (x.to_vec()[1] - 3.0).abs() < 1e-12);
/// ```
pub fn solve<F: LinalgFloat>(a: &Array<F>, b: &Array<F>) -> Result<Array<F>> {
    let n = square_shape(a)?;
    if b.ndim() != 1 && b.ndim() != 2 {
        return Err(shape_error(
            "solve: Input operand 1 must have one or two dimensions",
        ));
    }
    if b.shape()[0] != n {
        return Err(shape_error(
            "solve: Input operand 1 has a mismatch in its core dimension 0",
        ));
    }
    let rhs_cols = if b.ndim() == 1 { 1 } else { b.shape()[1] };
    let solution = kernels::Lu::factor(&data(a), n)
        .and_then(|factor| factor.solve(&data(b), rhs_cols))
        .map_err(eigen_error)?;
    let shape: &[usize] = if b.ndim() == 1 { &[n] } else { &[n, rhs_cols] };
    Ok(Array::from_vec(solution, shape))
}

/// Least-squares solution of `a @ x = b`, like `np.linalg.lstsq(a, b)[0]`.
///
/// Rank-deficient and under-determined systems give the minimum-norm solution.
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1.0f64, 0.0], [1.0, 1.0], [1.0, 2.0]];
/// let x = np::linalg::lstsq(&a, &np::array![1.0, 3.0, 5.0]).unwrap();
/// assert!((x.to_vec()[0] - 1.0).abs() < 1e-10 && (x.to_vec()[1] - 2.0).abs() < 1e-10);
/// ```
pub fn lstsq<F: LinalgFloat>(a: &Array<F>, b: &Array<F>) -> Result<Array<F>> {
    let (rows, cols) = matrix_shape(a)?;
    if b.ndim() != 1 && b.ndim() != 2 {
        return Err(shape_error("expected a vector or matrix right-hand side"));
    }
    if b.shape()[0] != rows {
        return Err(shape_error("Incompatible dimensions"));
    }
    let rhs_cols = if b.ndim() == 1 { 1 } else { b.shape()[1] };
    let solution = kernels::lstsq(&data(a), rows, cols, &data(b), rhs_cols).map_err(svd_error)?;
    let shape: &[usize] = if b.ndim() == 1 {
        &[cols]
    } else {
        &[cols, rhs_cols]
    };
    Ok(Array::from_vec(solution, shape))
}

/// Moore-Penrose pseudoinverse, like `np.linalg.pinv(a)`.
///
/// Panics for non-2-d input or non-finite entries (`"SVD did not converge"`).
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1.0f64, 0.0], [0.0, 2.0], [0.0, 0.0]];
/// let p = np::linalg::pinv(&a);
/// assert_eq!(p.shape(), [2, 3]);
/// assert!((p.to_vec()[4] - 0.5).abs() < 1e-12);
/// ```
pub fn pinv<F: LinalgFloat>(a: &Array<F>) -> Array<F> {
    let (rows, cols) = fail(matrix_shape(a));
    Array::from_vec(
        or_panic(kernels::pinv(&data(a), rows, cols), SVD_FAILED),
        &[cols, rows],
    )
}

/// Frobenius norm of a matrix or 2-norm of a vector, like `np.linalg.norm(a)`.
///
/// Any shape is flattened first.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::linalg::norm(&np::array![3.0, 4.0]), 5.0);
/// assert_eq!(np::linalg::norm(&np::array![[3.0, 0.0], [0.0, 4.0]]), 5.0);
/// ```
pub fn norm<F: LinalgFloat>(a: &Array<F>) -> F {
    kernels::norm(&data(a))
}

/// Vector norm along one axis, like `np.linalg.norm(a, ord=ord, axis=axis)`.
///
/// Works for any dimensionality: the result drops `axis`. Negative axes count from the end.
/// `NormOrd::Nuc` is invalid here and panics with `"Invalid norm order for vectors"`.
///
/// ```
/// use rustnum as np;
/// use np::linalg::NormOrd;
/// let a = np::array![[3.0, 4.0], [6.0, 8.0]];
/// assert_eq!(np::linalg::norm_axis(&a, NormOrd::Two, 1).to_vec(), [5.0, 10.0]);
/// assert_eq!(np::linalg::norm_axis(&a, NormOrd::One, 0).to_vec(), [9.0, 12.0]);
/// ```
pub fn norm_axis<F: LinalgFloat>(a: &Array<F>, ord: NormOrd, axis: isize) -> Array<F> {
    let ndim = a.ndim() as isize;
    assert!(
        (-ndim..ndim).contains(&axis),
        "axis {axis} is out of bounds for array of dimension {ndim}"
    );
    let axis = if axis < 0 { axis + ndim } else { axis } as usize;
    let len = a.shape()[axis];
    let mut shape = a.shape().to_vec();
    shape.remove(axis);
    let count: usize = shape.iter().product();
    let moved = a.moveaxis(axis as isize, -1);
    let norms =
        kernels::norm_axis(&data(&moved), count, len, 1, ord).unwrap_or_else(|error| match error {
            KernelError::NoConvergence => panic!("{SVD_FAILED}"),
            _ => panic!("Invalid norm order for vectors"),
        });
    Array::from_vec(norms, &shape)
}

/// Lower Cholesky factor `L` with `a = L @ L.T`, like `np.linalg.cholesky(a)`.
///
/// Returns `Err(Error::NotPositiveDefinite)` when `a` is not positive definite.
///
/// ```
/// use rustnum as np;
/// let l = np::linalg::cholesky(&np::array![[4.0f64, 2.0], [2.0, 3.0]]).unwrap();
/// assert!((l.to_vec()[0] - 2.0).abs() < 1e-12);
/// let bad = np::linalg::cholesky(&np::array![[1.0, 2.0], [2.0, 1.0]]);
/// assert!(matches!(bad, Err(np::Error::NotPositiveDefinite)));
/// ```
pub fn cholesky<F: LinalgFloat>(a: &Array<F>) -> Result<Array<F>> {
    let n = square_shape(a)?;
    kernels::cholesky(&data(a), n)
        .map(|lower| Array::from_vec(lower, &[n, n]))
        .map_err(eigen_error)
}

/// Reduced QR factorisation, like `np.linalg.qr(a, mode="reduced")`.
///
/// Returns `(q, r)` with `q` of shape `[m, min(m, n)]`, `r` of shape `[min(m, n), n]` and
/// `a = q @ r`. Panics for non-2-d input.
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let (q, r) = np::linalg::qr(&a);
/// assert_eq!((q.shape(), r.shape()), (&[3, 2][..], &[2, 2][..]));
/// assert!(q.matmul(&r).allclose(a, 1e-12, 1e-12));
/// ```
pub fn qr<F: LinalgFloat>(a: &Array<F>) -> (Array<F>, Array<F>) {
    let (rows, cols) = fail(matrix_shape(a));
    let (q, r, reduced) = or_panic(kernels::qr(&data(a), rows, cols), EIGEN_FAILED);
    (
        Array::from_vec(q, &[rows, reduced]),
        Array::from_vec(r, &[reduced, cols]),
    )
}

/// Eigendecomposition of a symmetric matrix, like `np.linalg.eigh(a)`.
///
/// Returns `(values, vectors)` with ascending eigenvalues and eigenvectors as the columns of
/// `vectors`. The whole matrix is read and assumed symmetric. Panics for non-square input
/// and for non-finite entries (`"Eigenvalues did not converge"`).
///
/// ```
/// use rustnum as np;
/// let a = np::array![[2.0f64, 1.0], [1.0, 2.0]];
/// let (w, v) = np::linalg::eigh(&a);
/// assert!((w.to_vec()[0] - 1.0).abs() < 1e-12 && (w.to_vec()[1] - 3.0).abs() < 1e-12);
/// // a = v diag(w) v.T
/// let restored = (&v * &w).matmul(&v.t());
/// assert!(restored.allclose(a, 1e-12, 1e-12));
/// ```
pub fn eigh<F: LinalgFloat>(a: &Array<F>) -> (Array<F>, Array<F>) {
    let n = fail(square_shape(a));
    let (values, vectors) = or_panic(kernels::eigh(&data(a), n), EIGEN_FAILED);
    (
        Array::from_vec(values, &[n]),
        Array::from_vec(vectors, &[n, n]),
    )
}

/// Reduced singular value decomposition, like `np.linalg.svd(a, full_matrices=False)`.
///
/// Returns `(u, s, vt)` with `s` descending, `u` of shape `[m, k]`, `vt` of shape `[k, n]`
/// (`k = min(m, n)`) and `a = u @ diag(s) @ vt`. Panics for non-2-d input and for non-finite
/// entries (`"SVD did not converge"`).
///
/// ```
/// use rustnum as np;
/// let a = np::array![[3.0f64, 0.0], [0.0, -2.0]];
/// let (u, s, vt) = np::linalg::svd(&a);
/// assert!((s.to_vec()[0] - 3.0).abs() < 1e-12 && (s.to_vec()[1] - 2.0).abs() < 1e-12);
/// // a = u diag(s) vt
/// let restored = (&u * &s).matmul(&vt);
/// assert!(restored.allclose(a, 1e-12, 1e-12));
/// ```
pub fn svd<F: LinalgFloat>(a: &Array<F>) -> (Array<F>, Array<F>, Array<F>) {
    let (rows, cols) = fail(matrix_shape(a));
    let (u, values, vt, reduced) = or_panic(kernels::svd(&data(a), rows, cols), SVD_FAILED);
    (
        Array::from_vec(u, &[rows, reduced]),
        Array::from_vec(values, &[reduced]),
        Array::from_vec(vt, &[reduced, cols]),
    )
}

/// Numerical rank, like `np.linalg.matrix_rank(a)`.
///
/// Counts singular values above `s_max * max(m, n) * eps`.
///
/// ```
/// use rustnum as np;
/// assert_eq!(np::linalg::matrix_rank(&np::array![[1.0, 2.0], [2.0, 4.0]]), 1);
/// assert_eq!(np::linalg::matrix_rank(&np::eye::<f64>(3)), 3);
/// ```
pub fn matrix_rank<F: LinalgFloat>(a: &Array<F>) -> usize {
    let (rows, cols) = fail(matrix_shape(a));
    or_panic(kernels::matrix_rank(&data(a), rows, cols), SVD_FAILED)
}

/// Integer power of a square matrix, like `np.linalg.matrix_power(a, n)`.
///
/// `n == 0` gives the identity; a negative `n` inverts `a` first and returns
/// `Err(Error::Singular)` when `a` has no inverse.
///
/// ```
/// use rustnum as np;
/// let fib = np::array![[1.0, 1.0], [1.0, 0.0]];
/// assert_eq!(np::linalg::matrix_power(&fib, 10).unwrap().to_vec(), [89.0, 55.0, 55.0, 34.0]);
/// let half = np::linalg::matrix_power(&np::array![[2.0, 0.0], [0.0, 2.0]], -1).unwrap();
/// assert_eq!(half.to_vec(), [0.5, 0.0, 0.0, 0.5]);
/// ```
pub fn matrix_power<F: LinalgFloat>(a: &Array<F>, n: i32) -> Result<Array<F>> {
    let size = square_shape(a)?;
    kernels::matrix_power(&data(a), size, i64::from(n))
        .map(|power| Array::from_vec(power, &[size, size]))
        .map_err(eigen_error)
}

/// Eigenvalues and right eigenvectors of a general square matrix, like `np.linalg.eig(a)`.
///
/// Returns `(w_re, w_im, v_re, v_im)`: the eigenvalues `w_re + i w_im` (shape `[n]`) and the
/// eigenvectors `v_re + i v_im` as the columns of two `[n, n]` arrays. Real eigenvalues have
/// `w_im == 0`; complex conjugate pairs are adjacent, positive imaginary part first. Every
/// eigenvector has unit 2-norm and its largest component is real and positive. Eigenvalue
/// order follows the Schur form and may differ from NumPy's.
///
/// Returns `Err(Error::NoConvergence)` for non-finite input.
///
/// ```
/// use rustnum as np;
/// let rotation = np::array![[0.0, -1.0], [1.0, 0.0]];
/// let (re, im, _, _) = np::linalg::eig(&rotation).unwrap();
/// assert_eq!((re.to_vec(), im.to_vec()), (vec![0.0, 0.0], vec![1.0, -1.0]));
/// ```
#[allow(clippy::type_complexity)]
pub fn eig<F: LinalgFloat>(a: &Array<F>) -> Result<(Array<F>, Array<F>, Array<F>, Array<F>)> {
    let n = square_shape(a)?;
    kernels::eig(&data(a), n)
        .map(|(w_re, w_im, v_re, v_im)| {
            (
                Array::from_vec(w_re, &[n]),
                Array::from_vec(w_im, &[n]),
                Array::from_vec(v_re, &[n, n]),
                Array::from_vec(v_im, &[n, n]),
            )
        })
        .map_err(eigen_error)
}

/// Eigenvalues of a general square matrix as `(re, im)`, like `np.linalg.eigvals(a)`.
///
/// Returns `Err(Error::NoConvergence)` for non-finite input.
///
/// ```
/// use rustnum as np;
/// let (re, im) = np::linalg::eigvals(&np::array![[2.0, 0.0], [0.0, 3.0]]).unwrap();
/// assert_eq!((re.to_vec(), im.to_vec()), (vec![2.0, 3.0], vec![0.0, 0.0]));
/// ```
pub fn eigvals<F: LinalgFloat>(a: &Array<F>) -> Result<(Array<F>, Array<F>)> {
    let n = square_shape(a)?;
    kernels::eigvals(&data(a), n)
        .map(|(re, im)| (Array::from_vec(re, &[n]), Array::from_vec(im, &[n])))
        .map_err(eigen_error)
}

/// Ascending eigenvalues of a symmetric matrix, like `np.linalg.eigvalsh(a)`.
///
/// ```
/// use rustnum as np;
/// let w = np::linalg::eigvalsh(&np::array![[2.0f64, 1.0], [1.0, 2.0]]);
/// assert!((w.to_vec()[0] - 1.0).abs() < 1e-12 && (w.to_vec()[1] - 3.0).abs() < 1e-12);
/// ```
pub fn eigvalsh<F: LinalgFloat>(a: &Array<F>) -> Array<F> {
    let n = fail(square_shape(a));
    Array::from_vec(or_panic(kernels::eigvalsh(&data(a), n), EIGEN_FAILED), &[n])
}

/// Sign and natural log of the absolute determinant, like `np.linalg.slogdet(a)`.
///
/// A singular matrix gives `(0, -inf)`.
///
/// ```
/// use rustnum as np;
/// let (sign, logabsdet) = np::linalg::slogdet(&np::array![[1.0, 2.0], [3.0, 4.0]]);
/// assert_eq!(sign, -1.0);
/// assert!((logabsdet - 2.0_f64.ln()).abs() < 1e-12);
/// ```
pub fn slogdet<F: LinalgFloat>(a: &Array<F>) -> (F, F) {
    let n = fail(square_shape(a));
    or_panic(kernels::slogdet(&data(a), n), EIGEN_FAILED)
}

/// 2-norm condition number `s_max / s_min`, like `np.linalg.cond(a)`.
///
/// ```
/// use rustnum as np;
/// assert!((np::linalg::cond(&np::eye::<f64>(3)) - 1.0).abs() < 1e-12);
/// ```
pub fn cond<F: LinalgFloat>(a: &Array<F>) -> F {
    let (rows, cols) = fail(matrix_shape(a));
    or_panic(kernels::cond(&data(a), rows, cols), SVD_FAILED)
}

/// Condition number in another norm, like `np.linalg.cond(a, p)` for `p` in
/// `{fro, 1, 2, inf, -1, -2, -inf, nuc}`; `inf` for a singular matrix.
///
/// ```
/// use rustnum as np;
/// use np::linalg::NormOrd;
/// let a = np::array![[1.0f64, 2.0], [3.0, 4.0]];
/// let c = np::linalg::cond_ord(&a, NormOrd::One);
/// assert!((c - 21.0).abs() < 1e-10);
/// ```
pub fn cond_ord<F: LinalgFloat>(a: &Array<F>, ord: NormOrd) -> F {
    let (rows, cols) = fail(matrix_shape(a));
    kernels::cond_ord(&data(a), rows, cols, ord).unwrap_or_else(|error| match error {
        KernelError::NoConvergence => panic!("{SVD_FAILED}"),
        _ => panic!("Invalid norm order for matrices"),
    })
}

/// Vector norm of any order except `Nuc`, like `np.linalg.vector_norm(x, ord=ord)`.
///
/// The input is flattened first, as in NumPy.
///
/// ```
/// use rustnum as np;
/// use np::linalg::NormOrd;
/// let x = np::array![3.0, -4.0];
/// assert_eq!(np::linalg::vector_norm(&x, NormOrd::Two), 5.0);
/// assert_eq!(np::linalg::vector_norm(&x, NormOrd::Inf), 4.0);
/// ```
pub fn vector_norm<F: LinalgFloat>(a: &Array<F>, ord: NormOrd) -> F {
    kernels::vector_norm(&data(a), ord).unwrap_or_else(|_| panic!("Invalid norm order for vectors"))
}

/// Matrix norm of any order except `Zero`, like `np.linalg.matrix_norm(a, ord=ord)`.
///
/// ```
/// use rustnum as np;
/// use np::linalg::NormOrd;
/// let a = np::array![[1.0, -2.0], [3.0, -4.0]];
/// assert_eq!(np::linalg::matrix_norm(&a, NormOrd::One), 6.0);
/// assert_eq!(np::linalg::matrix_norm(&a, NormOrd::Inf), 7.0);
/// ```
pub fn matrix_norm<F: LinalgFloat>(a: &Array<F>, ord: NormOrd) -> F {
    let (rows, cols) = fail(matrix_shape(a));
    kernels::matrix_norm(&data(a), rows, cols, ord).unwrap_or_else(|error| match error {
        KernelError::NoConvergence => panic!("{SVD_FAILED}"),
        _ => panic!("Invalid norm order for matrices"),
    })
}

/// Product of several matrices with the cheapest parenthesisation, like `np.linalg.multi_dot`.
///
/// The first and last arrays may be 1-d (treated as a row / column vector and squeezed from
/// the result); all others must be 2-d. Panics with fewer than two arrays or misaligned shapes.
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1, 2], [3, 4]];
/// let b = np::array![[0, 1], [1, 0]];
/// let c = np::array![[2, 0], [0, 2]];
/// assert_eq!(np::linalg::multi_dot(&[&a, &b, &c]).to_vec(), [4, 2, 8, 6]);
/// ```
pub fn multi_dot<T: Num>(arrays: &[&Array<T>]) -> Array<T> {
    assert!(arrays.len() >= 2, "Expecting at least two arrays.");
    let last = arrays.len() - 1;
    let buffers: Vec<Cow<'_, [T]>> = arrays.iter().map(|array| data(array)).collect();
    let mut dims = Vec::with_capacity(arrays.len());
    for (index, array) in arrays.iter().enumerate() {
        dims.push(match array.ndim() {
            1 if index == 0 => (1, array.shape()[0]),
            1 if index == last => (array.shape()[0], 1),
            2 => (array.shape()[0], array.shape()[1]),
            _ => {
                panic!("multi_dot: the first and last arguments may be 1-D, all others must be 2-D")
            }
        });
    }
    let chain: Vec<(&[T], usize, usize)> = buffers
        .iter()
        .zip(&dims)
        .map(|(buffer, &(rows, cols))| (&buffer[..], rows, cols))
        .collect();
    let (product, rows, cols) =
        kernels::multi_dot(&chain).unwrap_or_else(|_| panic!("multi_dot: shapes are not aligned"));
    let mut shape = Vec::new();
    if arrays[0].ndim() == 2 {
        shape.push(rows);
    }
    if arrays[last].ndim() == 2 {
        shape.push(cols);
    }
    Array::from_vec(product, &shape)
}

fn normalize_axes(axes: &[isize], ndim: usize) -> Vec<usize> {
    axes.iter()
        .map(|&axis| {
            let size = ndim as isize;
            assert!(
                (-size..size).contains(&axis),
                "tensordot: axis {axis} is out of bounds for array of dimension {ndim}"
            );
            (if axis < 0 { axis + size } else { axis }) as usize
        })
        .collect()
}

/// Tensor contraction over paired axes, like `np.tensordot(a, b, axes=(axes_a, axes_b))`.
///
/// The result shape is the free axes of `a` followed by the free axes of `b`; contracting
/// every axis gives a 0-d array. Panics with `shape-mismatch for sum` when the paired axes
/// differ in length, repeat, or have different sizes.
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1, 2], [3, 4]];
/// let product = np::linalg::tensordot(&a, &a, &[1], &[0]);
/// assert_eq!(product.to_vec(), [7, 10, 15, 22]);
/// ```
pub fn tensordot<T: Num>(
    a: &Array<T>,
    b: &Array<T>,
    axes_a: &[isize],
    axes_b: &[isize],
) -> Array<T> {
    let (product, shape) = kernels::tensordot(
        &data(a),
        a.shape(),
        &data(b),
        b.shape(),
        &normalize_axes(axes_a, a.ndim()),
        &normalize_axes(axes_b, b.ndim()),
    )
    .unwrap_or_else(|_| panic!("shape-mismatch for sum"));
    Array::from_vec(product, &shape)
}

/// `np.tensordot(a, b, axes=n)`: contract the last `n` axes of `a` with the first `n` of `b`.
///
/// Not a separate NumPy function; it is the integer form of NumPy's `axes` argument.
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1, 2], [3, 4]];
/// assert_eq!(np::linalg::tensordot_n(&a, &a, 2).item(), 30);
/// ```
pub fn tensordot_n<T: Num>(a: &Array<T>, b: &Array<T>, n: usize) -> Array<T> {
    assert!(
        n <= a.ndim() && n <= b.ndim(),
        "tensordot: cannot contract {n} axes of arrays with {} and {} dimensions",
        a.ndim(),
        b.ndim()
    );
    let axes_a: Vec<isize> = (a.ndim() - n..a.ndim()).map(|axis| axis as isize).collect();
    let axes_b: Vec<isize> = (0..n).map(|axis| axis as isize).collect();
    tensordot(a, b, &axes_a, &axes_b)
}

/// Dot product of the flattened arrays, like `np.vdot(a, b)` (no conjugation: real data only).
///
/// ```
/// use rustnum as np;
/// let a = np::array![[1, 2], [3, 4]];
/// assert_eq!(np::linalg::vdot(&a, &a), 30);
/// ```
pub fn vdot<T: Num>(a: &Array<T>, b: &Array<T>) -> T {
    kernels::vdot(&data(a), &data(b)).unwrap_or_else(|_| panic!("vectors have different lengths"))
}
