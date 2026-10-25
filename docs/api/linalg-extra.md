# rustnum v0.1 linalg extras: addendum to API.md section 11

Additional `np::linalg` functions and behaviour changes. Same conventions as `docs/API.md`: names follow
NumPy, misuse panics with NumPy-style wording, bad data returns `rustnum::Result`. Wrappers live in
`rustnum/src/linalg/api.rs.pending` (not yet compiled); the numerical code is in `rustnum/src/linalg/kernels.rs`
(slice-level, no `Array`). `F: Float` means `f32` or `f64`; `T: Num` also admits integers.

v0.1 has **no complex dtype**. Functions whose NumPy result is complex return the real and imaginary parts as two
arrays of the same shape: `(re, im)`. Real inputs only.

## Norm orders
```rust
pub enum NormOrd { Fro, Two, One, Inf, NegOne, NegTwo, NegInf, Zero, Nuc }
```
`norm_axis` (section 11) now accepts every order below. Meaning follows `np.linalg.norm(x, ord)`:

| `NormOrd` | vector | matrix |
|---|---|---|
| `Fro`, `Two` | 2-norm (`Fro` is accepted for vectors as a synonym) | `Fro`: Frobenius; `Two`: largest singular value |
| `One` | sum of magnitudes | maximum absolute column sum |
| `Inf` | largest magnitude | maximum absolute row sum |
| `NegOne` | `1 / sum(1/abs(x))` | minimum absolute column sum |
| `NegTwo` | `1 / sqrt(sum(1/x^2))` | smallest singular value |
| `NegInf` | smallest magnitude (empty input panics) | minimum absolute row sum |
| `Zero` | number of non-zero entries | invalid |
| `Nuc` | invalid | sum of singular values |

An invalid combination panics: `Invalid norm order for vectors` / `Invalid norm order for matrices`.

`vector_norm(&a, ord) -> F` flattens `a` (`np.linalg.vector_norm`). `matrix_norm(&a2d, ord) -> F` (`np.linalg.matrix_norm`).

## Eigenproblems
`eig(&a) -> Result<(Array<F>, Array<F>, Array<F>, Array<F>)>` returns `(w_re, w_im, v_re, v_im)`.
- `a` is a real square matrix. `w_re`, `w_im` have shape `[n]`; `v_re`, `v_im` have shape `[n, n]`; column `j` of
  `v_re + i v_im` is the right eigenvector of `w_re[j] + i w_im[j]`, i.e. `a @ v[:, j] = w[j] * v[:, j]`.
- Method: Householder reduction to Hessenberg form, Francis double-shift QR for the real Schur form, then
  back-substitution for the eigenvectors (EISPACK `orthes` + `hqr2`). No balancing step.
- Real eigenvalues have `w_im == 0` exactly. Complex eigenvalues come in conjugate pairs stored in adjacent positions,
  **positive imaginary part first**, and the partner's eigenvector is the exact conjugate.
- **Order** is the order of the diagonal of the Schur form. NumPy (LAPACK) uses the same idea but its shifts and
  balancing differ, so do not rely on the order matching NumPy; compare sorted or matched eigenvalues.
- **Normalisation:** each eigenvector has unit 2-norm and its component of largest magnitude is real and positive
  (the same sign rule as `eigh` and `svd`). NumPy also normalises to unit norm; it does not fix the sign.
- Defective matrices (for example `[[2, 1], [0, 2]]`) return the single eigenvector direction twice, as NumPy does.
- Non-finite input returns an error (below). `n == 0` returns empty arrays.

`eigvals(&a) -> Result<(Array<F>, Array<F>)>` is `(w_re, w_im)` of `eig`.
`eigvalsh(&a) -> Array<F>` is the ascending eigenvalues of a symmetric matrix (the first output of `eigh`). Like `eigh` it
reads the whole matrix and assumes it is symmetric (NumPy reads only the lower triangle).

## Determinant and conditioning
- `slogdet(&a) -> (F, F)`: `(sign, logabsdet)` from the LU factorisation; `det = sign * exp(logabsdet)`.
  An exactly singular matrix gives `(0, -inf)`, a 0x0 matrix `(1, 0)`. Survives determinants that overflow `det`.
- `cond(&a) -> F`: 2-norm condition number `s_max / s_min` from the SVD, any 2-d shape; `inf` when `s_min == 0`.
  An exactly rank-deficient matrix often yields about `1e16` rather than `inf`, exactly as NumPy does.
- `cond_ord(&a, ord: NormOrd) -> F`: `np.linalg.cond(a, p)`. `Two`/`NegTwo` use the SVD (any shape); `Fro`, `One`, `Inf`,
  `NegOne`, `NegInf`, `Nuc` need a square matrix and compute `norm(a) * norm(inv(a))`, returning `inf` for a singular matrix.

## Products
- `multi_dot(&[&a, &b, ...]) -> Array<T>` (`T: Num`): `np.linalg.multi_dot`. Chooses the cheapest parenthesisation by
  dynamic programming (the matrix-chain algorithm, cost = number of scalar multiplications) and multiplies with the
  same kernels as `matmul`. The first and last array may be 1-d (row / column vector, squeezed from the result); the
  others must be 2-d. Panics: `Expecting at least two arrays.`; shape misalignment panics.
- `tensordot(&a, &b, axes_a: &[isize], axes_b: &[isize]) -> Array<T>` (`T: Num`): `np.tensordot(a, b, axes=(axes_a, axes_b))`.
  Negative axes count from the end. Both operands are permuted so the contracted axes are contiguous, reshaped to
  matrices and multiplied with one GEMM. Result shape: free axes of `a`, then free axes of `b`; contracting everything
  gives a 0-d array; an empty axis list gives the outer product. Panics `shape-mismatch for sum` when the lists differ
  in length, repeat an axis, or pair axes of different size.
- `tensordot_n(&a, &b, n: usize) -> Array<T>`: the integer form `axes=n` (last `n` axes of `a` with first `n` of `b`).
  Not a NumPy function name; it exists because Rust has no overloaded argument.
- `vdot(&a, &b) -> T` (`T: Num`): `np.vdot`, dot product of the flattened arrays. Real data, so no conjugation.
  Sizes must agree (`vectors have different lengths`); the shapes need not.

## `matrix_power` for negative exponents
`matrix_power(&a, n: i32) -> Result<Array<F>>` (section 11) now accepts `n < 0`: it inverts `a` once and raises the
inverse to `|n|` by repeated squaring. `Err(Error::Singular)` if `a` has no inverse. `n == 0` is the identity even for
a singular `a` (NumPy does the same).

## Behaviour changes in existing kernels
These were found by the NumPy parity tests and fixed in `kernels.rs`:
- `lstsq` returns the **minimum-norm** solution for rank-deficient and under-determined systems like NumPy. It used to
  return a meaningless solution of size about 1e14 for a rank-deficient tall matrix. Full-rank tall systems still use
  Householder QR.
- `qr` follows LAPACK's reflector convention: a column whose sub-diagonal is already zero is left alone, so the last
  diagonal entry of a square `R` keeps its sign and `qr(upper_triangular)` returns `Q = I`. Q and R now match NumPy
  entry for entry. Tiny and huge matrices no longer underflow or overflow in the norm.
- `svd`: columns of `U` belonging to numerically zero singular values (at or below `eps * max(m, n) * s_max`) are
  completed to an orthonormal basis instead of being left zero, so `U` and `Vt` always have orthonormal columns/rows.
- `eigh`, `svd`: convergence thresholds are relative to the size of the matrix. A `1e-20 * A` used to be declared
  "already diagonal".
- `gemm_f32` / `gemm_f64`: the output-aliasing check compared every pair of elements (`O((m n)^2)`); it is now O(1) for
  ordinary layouts, so a 1000 x 1000 product no longer hangs.
- `eigh`, `svd`, `eig` return an error on NaN/infinity instead of silently returning garbage.

## Errors
`Error::Singular`, `Error::NotPositiveDefinite` and `Error::Shape(String)` as in section 11. Non-convergence (NaN or
infinite entries, or a QR iteration exceeding 300 steps per eigenvalue) is reported as
`Error::Shape("Eigenvalues did not converge ...")` until the core `Error` gains a dedicated variant. Functions whose
section 11 signature is not a `Result` (`eigh`, `svd`, `pinv`, `matrix_rank`, `eigvalsh`, `cond`, `slogdet`) panic with
that message instead.

## Accuracy against NumPy 2.3.2 (float64)
Measured by `rustnum/src/linalg/tests.rs`; entries are max abs error divided by `max(1, max|expected|)` on small
well-conditioned inputs.

| function | error | function | error |
|---|---|---|---|
| `gemm`, `inv`, `solve`, `cholesky`, `qr` | 2e-16 to 4e-16 | `eigh` values (5x5, 6x6) | 9e-16 |
| `svd` values (4x3, 7x4) | 3e-16 | `pinv`, `lstsq` | 4e-16 to 6e-16 |
| `eig` eigenvalues, 6x6 / 30x30 | 4e-15 / 9e-15 | `matrix_power(A, -7)` | 1e-16 |
| `slogdet`, `cond`, `multi_dot`, `tensordot` | 2e-16 to 7e-16 | companion matrix of (x-1)...(x-6) | 7e-13 |

Hilbert 8x8 (condition number 1.5e10): `inv` has relative error 1e-8 against the exact rational inverse, `det` 1e-8,
`cond` 1e-7, and the solution of `H x = H 1` is off by 4e-7. That is the expected `cond * eps` behaviour, not a defect.
