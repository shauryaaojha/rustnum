# rustnum API contract (v0.1 core; v0.2 additions are sections 22 onward)

This is the binding contract for the final v0.1 code. Names and behavior follow NumPy unless this document states otherwise; misuse panics with NumPy-style wording, while I/O, parsing, singular matrices, non-positive-definite matrices, and solver non-convergence return `rustnum::Result` where the signature permits it.

Conventions: `T: Element`, `N: Num`, `F: Float`, `LF: linalg::LinalgFloat`, `a` and `b` are arrays, shapes implement `AsRef<[usize]>` where shown, and an `axis: isize` accepts negative indexing.

## 0. Crate surface

- Users write `use rustnum as np;`; core items are re-exported once at the crate root.
- Public modules are `np::fft`, `np::io`, `np::linalg`, `np::poly`, `np::memory`, and `np::random`; implementation modules remain private.
- Root macros are `np::array!` for literals through three dimensions and `np::s!` for slice arguments.
- Public APIs contain no explicit lifetime parameters and no public `unsafe` functions.
- Integer addition, subtraction, multiplication, powers, reductions, and linear-algebra products wrap on overflow like fixed-width NumPy dtypes.

## 1. Element types and errors

- `Element: Copy + Clone + Send + Sync + Debug + Default + PartialEq + PartialOrd + 'static` exposes `DTYPE`, `to_f64`, and `from_f64` (plus defaulted `to_complex_parts`/`from_complex_parts` for `astype` and the `.npy` hooks `npy_descr`/`npy_itemsize`/`npy_write`/`npy_read`); implementations are `f32`, `f64`, `i32`, `i64`, `u8`, `usize`, `bool`, and (v0.2) `Complex<f32>` (`complex64`) and `Complex<f64>` (`complex128`).
- `Num: Element + Add + Sub + Mul + Div + Rem` exposes `ZERO`, `ONE`, `MIN_VALUE`, `MAX_VALUE`, `abs`, and `from_usize`; implementations are `f32`, `f64`, `i32`, `i64`, `u8`, `usize`, and `Complex<F>`.
- `Float: Num + Neg` exposes floating constants and elementary operations; the sealed implementations are `f32` and `f64`.
- `Error` variants are `Shape(String)`, `DType { expected: String, found: String }`, `Singular`, `NotPositiveDefinite`, `Io(std::io::Error)`, `Parse(String)`, `Format(String)`, and `NoConvergence(String)`.
- `type Result<T> = std::result::Result<T, Error>`.

## 2. `Array<T>`

- `Array<T>` is a runtime-dimensional, row-major-by-default buffer with shape, element strides, offset, and copy-on-write `Arc<Vec<T>>` value semantics.
- Constructors are `Array::from_vec(data: Vec<T>, shape: &[usize])`, `Array::from_shape_fn(shape: impl AsRef<[usize]>, f: impl FnMut(&[usize]) -> T)`, and `Array::scalar(value: T)`.
- Metadata methods are `shape() -> &[usize]`, `ndim() -> usize`, `size() -> usize`, `len() -> usize`, `is_empty() -> bool`, `strides() -> &[isize]`, `dtype() -> &'static str`, and `is_contiguous() -> bool`.
- Element methods are `get(index: &[usize]) -> T`, `set(index: &[usize], value: T)`, `item() -> T`, `to_vec() -> Vec<T>`, `as_slice() -> Option<&[T]>`, and `iter() -> impl ExactSizeIterator<Item = T>`.
- Conversion methods are `to_contiguous()`, `astype<U: Element>() -> Array<U>`, `map<U: Element>(f) -> Array<U>`, `mapv_inplace(f)`, `zip_map<U: Element, R: Element>(other: impl IntoArray<U>, f) -> Array<R>`, `fill(value)`, and `copy()`.
- Traits are `Clone`, `Default`, `Display`, `Debug`, `PartialEq`, `From<Vec<T>>`, `FromIterator<T>`, const-array and slice indexing, mutable indexing, and iteration over `&Array<T>` in C order.
- A shape `[]` is a 0-d array containing one value; `len()` panics for a 0-d array and `item()` panics unless `size() == 1`.

## 3. Array-like values and broadcasting

- `IntoArray<T> { fn into_array(self) -> Array<T>; }` is implemented for `T`, `Array<T>`, `&Array<T>`, `Vec<T>`, `&[T]`, and `[T; N]`.
- `broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>>` returns the NumPy broadcast shape.
- `broadcast_arrays(a: &Array<T>, b: &Array<U>) -> (Array<T>, Array<U>)` returns zero-stride broadcast views.
- `a.broadcast_to(target: &[usize]) -> Array<T>` returns a view and panics for an incompatible target.

## 4. Creation functions at the root

- `zeros<T: Element>(shape: impl AsRef<[usize]>) -> Array<T>` and `full<T: Element>(shape: impl AsRef<[usize]>, value: T) -> Array<T>`.
- `ones<T: Num>(shape: impl AsRef<[usize]>) -> Array<T>`.
- `zeros_like<T: Element>(&a)`, `full_like<T: Element>(&a, value)`, and `ones_like<T: Num>(&a)` preserve the shape.
- `arange<T: Num>(start: T, stop: T, step: T) -> Array<T>` is stop-exclusive and rejects zero step.
- `linspace<F: Float>(start: F, stop: F, num: usize)` and `logspace<F: Float>(start: F, stop: F, num: usize)` include the endpoint.
- `eye<T: Num>(n)`, `identity<T: Num>(n)`, and `tri<T: Num>(n)` return square arrays.
- `diag<T: Element>(&a)` maps 1-d to a diagonal matrix and 2-d to its main diagonal.
- `tril<T: Num>(&a)` and `triu<T: Num>(&a)` retain the lower and upper triangles.
- `meshgrid<T: Element>(&x, &y) -> (Array<T>, Array<T>)` uses NumPy's `indexing="xy"` convention.

## 5. Indexing and slicing

- `SliceArg` variants are `Index(isize)`, `Range { start: Option<isize>, end: Option<isize>, step: isize }`, `Inclusive { start: Option<isize>, end: isize, step: isize }`, and `NewAxis`; `NEW_AXIS` is `SliceArg::NewAxis`.
- `s![...]` supports indices, Rust ranges, negative indices, `;step`, and `NEW_AXIS`.
- `slice(args: &[SliceArg]) -> Array<T>` returns a view and `slice_assign(args, values: impl IntoArray<T>)` broadcasts into the selected region.
- `index_axis(axis, i)`, `row(i)`, and `col(i)` drop the selected axis where applicable.
- `mask(&Array<bool>) -> Array<T>` flattens selected values and `mask_assign(&mask, value)` assigns them.
- `take(indices: &[usize], axis: isize) -> Array<T>` implements `np.take`.
- `compress(mask: &Array<bool>, axis: isize) -> Array<T>` implements `np.compress`.

## 6. Operators and comparisons

- `+`, `-`, `*`, `/`, and `%` support owned/borrowed arrays and scalar operands, broadcast, and preserve the element dtype.
- Unary `-` is available for signed integers and floats; `+=`, `-=`, `*=`, `/=`, and `%=` accept arrays, references, and scalars with RHS broadcasting.
- `eq_elem`, `ne_elem`, `lt`, `le`, `gt`, and `ge` accept `impl IntoArray<T>` and return `Array<bool>`.
- `array_equal(other: impl IntoArray<T>) -> bool` requires equal shape and values.
- `allclose<F: Float>(other, rtol, atol) -> bool` and `isclose<F: Float>(a, b, rtol, atol) -> Array<bool>` use `abs(a-b) <= atol + rtol*abs(b)`.
- Boolean arrays support `&`, `|`, `^`, and `!`; root functions are `logical_and`, `logical_or`, `logical_xor`, and `logical_not`.

## 7. Element-wise math

- `Array<F: Float>` methods are `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round`, `round_decimals(i32)`, `trunc`, `reciprocal`, `powf(F)`, `powi(i32)`, `is_nan`, `is_inf`, `is_finite`, `sigmoid`, and `relu`.
- `Array<N: Num>` methods are `abs`, `square`, `clip(lo, hi)`, and `sign`.
- Root broadcasting functions are `maximum<N>`, `minimum<N>`, `power<N>`, `arctan2<F>`, `hypot<F>`, and `where_<T>(&condition, x, y)`.
- Additional `Array<F>` methods are `sinc`, `i0`, `deg2rad`, `rad2deg`, `rint`, `fix`, `fabs`, `positive`, `negative`, `signbit`, `spacing`, `modf`, `frexp`, `nan_to_num`, `nan_to_num_with`, `unwrap`, and `unwrap_with`.
- Root floating functions are `logaddexp`, `logaddexp2`, `fmod`, `remainder`, `floor_divide`, `divmod`, `copysign`, `heaviside`, `nextafter`, `ldexp`, and `float_power`; arguments are `impl IntoArray`, and `float_power` returns `Array<f64>`.
- `angle<F: Float>(a: impl IntoArray<F>)` and `angle_deg<F: Float>(a: impl IntoArray<F>)` are root free functions returning `Array<F>`.
- `ediff1d<N: Num>(a: impl IntoArray<N>)` and `ediff1d_with<N: Num>(a, to_end: &[N], to_begin: &[N])` are root free functions and flatten first.
- `gcd(a: impl IntoArray<i64>, b: impl IntoArray<i64>)` and `lcm(...)` return `Array<i64>`.
- `hanning(m)`, `hamming(m)`, `blackman(m)`, `bartlett(m)`, and `kaiser(m, beta)` return `Array<f64>`.

## 8. Reductions and statistics methods

- `Array<T: Element>` methods are `min`, `max`, `argmin`, `argmax`, `min_axis`, `max_axis`, `argmin_axis`, `argmax_axis`, `any`, `all`, `any_axis`, `all_axis`, and `count_nonzero`.
- `Array<N: Num>` methods are `sum`, `prod`, `ptp`, `sum_axis`, `prod_axis`, `cumsum`, `cumprod`, `cumsum_axis`, and `cumprod_axis`; integer accumulation wraps.
- `Array<F: Float>` methods are `mean`, `var`, `std`, `var_ddof`, `std_ddof`, `mean_axis`, `var_axis`, `std_axis`, `var_ddof_axis(axis, ddof)`, `std_ddof_axis(axis, ddof)`, `median`, `percentile`, and `median_axis`.
- `Array<F: Float>` NaN methods are `nansum`, `nanmean`, `nanmin`, `nanmax`, `nansum_axis`, `nanmean_axis`, `nanmin_axis`, and `nanmax_axis`.
- Empty `min`, `max`, `argmin`, and `argmax` panic; empty/all-NaN floating statistics return NaN where NumPy would warn.

## 9. Shape manipulation and sets

- View/copy methods are `reshape(&[usize])`, `reshape_infer(&[isize])`, `ravel`, `flatten`, `t`, `transpose(&[usize])`, `permute_axes(&[usize])`, `swapaxes`, `moveaxis`, `squeeze`, `squeeze_axis`, and `expand_dims`.
- Reordering methods are `flip`, `flip_all`, `roll`, `repeat`, `tile`, `sort`, `sort_axis`, `argsort`, and `argsort_axis`; NaNs sort last.
- Root combining functions are `concatenate(&[&Array<T>], axis)`, `stack`, `vstack`, `hstack`, `split`, and `array_split`.
- Root index/set functions are `unique`, `unique_counts`, `nonzero`, `argwhere`, `searchsorted`, and constant-mode `pad(a, width, value)`.
- `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`, and `isin` flatten inputs; set outputs are sorted and unique except NumPy-compatible NaN edge cases.

## 10. Statistics, signal, and numeric free functions

- `Bins` is `Count(usize)` or `Edges(Vec<f64>)` and implements `From<usize>`, `From<Vec<f64>>`, `From<&[f64]>`, and `From<&Array<f64>>`.
- `ConvolveMode` is `Full`, `Same`, or `Valid`; its default is `Full`.
- `Spacing` is `Uniform(f64)` or `Coords(Vec<f64>)` and implements `From<f64>`, `From<Vec<f64>>`, and `From<&Array<f64>>`.
- `cov<T: Element>(m: &Array<T>, bias: bool, ddof: Option<usize>) -> Array<f64>` treats rows as variables.
- `corrcoef<T: Element>(x: &Array<T>) -> Array<f64>` returns the row-wise correlation matrix.
- `histogram<T: Element>(a, bins: impl Into<Bins>, range, density, weights) -> (Array<f64>, Array<f64>)` returns counts and edges; string bin rules are unsupported.
- `histogram_bin_edges<T: Element>(a, bins: impl Into<Bins>, range) -> Array<f64>` uses the same bin selection.
- `bincount(x: &Array<i64>, weights: Option<&Array<f64>>, minlength: usize) -> Array<f64>` rejects negative inputs.
- `digitize<T: Element, B: Element>(x, bins, right) -> Array<usize>` accepts increasing or decreasing bins.
- `average<T: Element>(a, weights) -> f64` and `average_axis<T: Element>(a, weights, axis) -> Array<f64>` reject a zero weight sum.
- `quantile<T: Element>(a, q) -> f64` and `quantile_axis<T: Element>(a, q, axis) -> Array<f64>` support scalar `q` and linear interpolation.
- `nanquantile<T: Element>(a, q) -> f64` and `nanquantile_axis<T: Element>(a, q, axis) -> Array<f64>` ignore NaNs.
- `convolve<N: Num>(a, v, mode)` and `correlate<N: Num>(a, v, mode)` are 1-d and return `Array<N>`.
- `diff<N: Num>(a, n, axis) -> Array<N>` preserves dtype and wraps integer subtraction.
- `gradient<F: Float>(f, dx) -> Vec<Array<F>>` returns one gradient per axis.
- `gradient_axis<F: Float>(f, spacing: impl Into<Spacing>, axis, edge_order) -> Array<F>` accepts edge order 1 or 2.
- `interp(x: &Array<f64>, xp: &Array<f64>, fp: &Array<f64>, left, right) -> Array<f64>` is one-dimensional linear interpolation.
- `trapezoid<F: Float>(y, spacing: impl Into<Spacing>, axis) -> Array<F>` removes the integration axis.
- `cross<N: Num>(a, b) -> Array<N>` uses last axes of length 2 or 3 and broadcasts leading axes.
- `kron<N: Num>(a, b) -> Array<N>` implements the n-dimensional Kronecker product.

## 11. Linear algebra (`np::linalg`)

- `LinalgFloat: Float` is the sealed bound for `f32` and `f64` decomposition kernels.
- `NormOrd` variants are `Fro`, `Two`, `One`, `Inf`, `NegOne`, `NegTwo`, `NegInf`, `Zero`, and `Nuc`; `Nuc` is matrix-only and `Zero` is vector-only.
- `Array<N: Num>::matmul(&b) -> Array<N>` implements vector promotion and broadcast batches; f32/f64 use the packed AVX2/FMA GEMM with a portable fallback, and generic integer products wrap.
- `Array<N: Num>::matmul_into(&b, &mut out)` reuses a contiguous output with the exact product shape (shared storage detaches on write); packing uses temporary workspace.
- `linalg::gemm<F: LinalgFloat>(alpha, &a, &b, beta, &c) -> Array<F>` computes `alpha * a @ b + beta * c` on 2-d arrays, like SciPy BLAS `dgemm`/`sgemm`.
- `linalg::matmul_map<F: LinalgFloat>(&a, &b, alpha, f: fn(F) -> F) -> Array<F>` fuses `f(alpha * a @ b)` in the GEMM epilogue, including vectors and broadcast batches.
- `Array<N: Num>::dot(&b) -> Array<N>` supports operands of at most two dimensions and scalar multiplication.
- `Array<N: Num>::trace() -> N` requires a 2-d array and wraps integer accumulation.
- `outer<T: Num>(a: &Array<T>, b: &Array<T>) -> Array<T>` flattens both inputs.
- `inner<T: Num>(a: &Array<T>, b: &Array<T>) -> T` requires equal-length vectors.
- `det<F: LinalgFloat>(a: &Array<F>) -> F` returns zero for a singular matrix.
- `inv<F: LinalgFloat>(a: &Array<F>) -> Result<Array<F>>` returns `Singular` or `Shape` errors.
- `solve<F: LinalgFloat>(a: &Array<F>, b: &Array<F>) -> Result<Array<F>>` accepts vector or matrix right-hand sides.
- `lstsq<F: LinalgFloat>(a: &Array<F>, b: &Array<F>) -> Result<Array<F>>` returns the minimum-norm solution.
- `pinv<F: LinalgFloat>(a: &Array<F>) -> Array<F>` panics with `SVD did not converge` on non-finite input.
- `norm<F: LinalgFloat>(a: &Array<F>) -> F` is the flattened Euclidean/Frobenius norm.
- `norm_axis<F: LinalgFloat>(a: &Array<F>, ord: NormOrd, axis: isize) -> Array<F>` computes vector norms along one axis.
- `vector_norm<F: LinalgFloat>(a: &Array<F>, ord: NormOrd) -> F` flattens its input.
- `matrix_norm<F: LinalgFloat>(a: &Array<F>, ord: NormOrd) -> F` requires a 2-d input.
- `cholesky<F: LinalgFloat>(a: &Array<F>) -> Result<Array<F>>` returns a lower factor or `NotPositiveDefinite`.
- `qr<F: LinalgFloat>(a: &Array<F>) -> (Array<F>, Array<F>)` returns the reduced factorization.
- `eigh<F: LinalgFloat>(a: &Array<F>) -> (Array<F>, Array<F>)` returns ascending symmetric eigenvalues and column eigenvectors.
- `svd<F: LinalgFloat>(a: &Array<F>) -> (Array<F>, Array<F>, Array<F>)` returns reduced `(u, s, vt)`.
- `matrix_rank<F: LinalgFloat>(a: &Array<F>) -> usize` uses NumPy's default SVD tolerance.
- `matrix_power<F: LinalgFloat>(a: &Array<F>, n: i32) -> Result<Array<F>>` supports negative powers through one inverse.
- `eig<F: LinalgFloat>(a: &Array<F>) -> Result<(Array<F>, Array<F>, Array<F>, Array<F>)>` returns `(w_re, w_im, v_re, v_im)`.
- `eigvals<F: LinalgFloat>(a: &Array<F>) -> Result<(Array<F>, Array<F>)>` returns `(w_re, w_im)`.
- `eigvalsh<F: LinalgFloat>(a: &Array<F>) -> Array<F>` returns ascending symmetric eigenvalues.
- `slogdet<F: LinalgFloat>(a: &Array<F>) -> (F, F)` returns sign and log absolute determinant.
- `cond<F: LinalgFloat>(a: &Array<F>) -> F` returns the 2-norm condition number.
- `cond_ord<F: LinalgFloat>(a: &Array<F>, ord: NormOrd) -> F` selects another supported norm.
- `multi_dot<T: Num>(arrays: &[&Array<T>]) -> Array<T>` chooses the cheapest matrix-chain parenthesization and wraps integer arithmetic.
- `tensordot<T: Num>(a: &Array<T>, b: &Array<T>, axes_a: &[isize], axes_b: &[isize]) -> Array<T>` wraps integer arithmetic.
- `tensordot_n<T: Num>(a: &Array<T>, b: &Array<T>, n: usize) -> Array<T>` implements NumPy's integer `axes=n` form.
- `vdot<T: Num>(a: &Array<T>, b: &Array<T>) -> T` flattens both real arrays, does no conjugation, and wraps integer arithmetic.
- Non-finite eigensolver input maps to `NoConvergence("Eigenvalues did not converge")`; SVD-based failures use `NoConvergence("SVD did not converge")` or the same panic text for infallible signatures.

## 12. Random (`np::random::Rng`)

- `Rng::seed(seed: u64)` and `Rng::from_entropy()` construct the xoshiro256** generator; seeded streams are platform-stable but intentionally differ from NumPy PCG64.
- `random(&mut self, shape: &[usize])`, `uniform(low, high, shape)`, `normal(mean, std, shape)`, and `standard_normal(shape)` return `Array<f64>`.
- `integers(low: i64, high: i64, shape: &[usize]) -> Array<i64>` uses an exclusive high bound.
- `choice<T: Element>(&mut self, a, n, replace) -> Array<T>`, `shuffle<T: Element>(&mut self, a: &mut Array<T>)`, and `permutation(n) -> Array<usize>` provide sampling and permutations.
- `binomial(n: u64, p: f64, shape: &[usize]) -> Array<i64>` and `poisson(lam: f64, shape: &[usize]) -> Array<i64>` return signed integer arrays.
- `exponential(scale, shape)`, `beta(a, b, shape)`, and `gamma(shape_param, scale, shape)` return `Array<f64>`.

## 13. I/O (`np::io`)

- `save_npy<T: Element>(path: impl AsRef<Path>, a: &Array<T>) -> Result<()>` and `load_npy<T: Element>(path) -> Result<Array<T>>` support NPY v1/v2, C/Fortran order, and exact dtype checking.
- `to_npy_bytes<T: Element>(&a) -> Vec<u8>` and `from_npy_bytes<T: Element>(bytes: &[u8]) -> Result<Array<T>>` provide the in-memory NPY codec.
- `save_csv<T>(path: impl AsRef<Path>, a: &Array<T>, delimiter: char) -> Result<()> where T: Element + Display` requires a 2-d array.
- `load_csv<T>(path: impl AsRef<Path>, delimiter: char, skip_header: usize) -> Result<Array<T>> where T: Element + FromStr, <T as FromStr>::Err: Display` implements the supported `np.loadtxt` subset.
- `savez<T: Element>(path: impl AsRef<Path>, arrays: &[(&str, &Array<T>)]) -> Result<()>` writes named uncompressed NPY members in a ZIP archive.
- `savez_compressed<T: Element>(path: impl AsRef<Path>, arrays: &[(&str, &Array<T>)]) -> Result<()>` DEFLATE-compresses members when smaller.
- `load_npz<T: Element>(path: impl AsRef<Path>) -> Result<Vec<(String, Array<T>)>>` loads every member in archive order and requires one dtype.
- `NpzFile::open(path) -> Result<NpzFile>`, `keys(&self) -> Vec<String>`, and `get<T: Element>(&self, name: &str) -> Result<Array<T>>` provide lazy mixed-dtype member access.
- NPZ names must be non-empty and unique; readers verify ZIP structure, compression, sizes, and CRC-32 and report corruption as `Error::Format`.

## 14. FFT (`np::fft`)

- v0.2 follows NumPy: every transform takes a real or a complex array (trait `FftInput`, implemented for `Array<F>` and `Array<Complex<F>>`) and returns a complex array; transforms use NumPy's default backward normalization and operate on the last axis (the last two for the `2` forms). `f64` input gives `complex128`, `f32` gives `complex64`.
- `fft<A: FftInput>(&a)` and `ifft<A: FftInput>(&a)` return `Array<Complex<A::Real>>`.
- `rfft<F: Float>(&a) -> Array<Complex<F>>` returns `n/2 + 1` bins and `irfft<A: FftInput>(&a, n: Option<usize>) -> Array<A::Real>`; `None` means NumPy's default `2 * (m - 1)`.
- `fft2<A: FftInput>(&a)` and `ifft2<A: FftInput>(&a)` transform the last two axes.
- `fftfreq<F: Float>(n, d)` and `rfftfreq<F: Float>(n, d)` return `Array<F>`.
- `fftshift<T: Element>(&a)` and `ifftshift<T: Element>(&a)` shift every axis and return `Array<T>`; they work on complex arrays.
- Length zero panics; radix-2 and Bluestein kernels support every positive length without external dependencies.
- Breaking change from v0.1: the `(re, im)` pair signatures are gone.

## 15. Polynomials (`np::poly`)

- Polynomial coefficients are 1-d, highest power first; all polynomial arguments are `impl IntoArray<F>` with `F: Float`.
- `polyval<F: Float>(p: impl IntoArray<F>, x: impl IntoArray<F>) -> Array<F>` evaluates by Horner's method and preserves `x`'s shape.
- `polyfit<F: Float>(x: impl IntoArray<F>, y: impl IntoArray<F>, deg: usize) -> Array<F>` returns `deg + 1` least-squares coefficients; weights, covariance, full output, and 2-d `y` are unsupported.
- `polyadd<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F>` aligns constant coefficients.
- `polysub<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F>` aligns constant coefficients.
- `polymul<F: Float>(a: impl IntoArray<F>, b: impl IntoArray<F>) -> Array<F>` convolves coefficients.
- `polydiv<F: Float>(u: impl IntoArray<F>, v: impl IntoArray<F>) -> (Array<F>, Array<F>)` returns quotient and remainder.
- `polyder<F: Float>(p: impl IntoArray<F>, m: usize) -> Array<F>` differentiates `m` times.
- `polyint<F: Float>(p: impl IntoArray<F>, m: usize, k: &[F]) -> Array<F>` integrates `m` times.
- `roots<F: Float>(p: impl IntoArray<F>) -> (Array<F>, Array<F>)` returns real and imaginary root arrays; root order is not a stable contract.

## 16. Einstein summation

- `einsum<N: Num>(subscripts: &str, operands: &[&Array<N>]) -> Array<N>` is re-exported at the root.
- Supported forms include explicit/implicit output, repeated labels, any operand count, scalar operands, labeled broadcasting, and right-aligned ellipses.
- Unsupported forms are `optimize=`, `einsum_path`, `out=`, `dtype=`, and NumPy's interleaved sublist syntax.
- The implementation is a direct nested contraction and wraps fixed-width integer multiply/add operations.

## 17. Scalar complex numbers

- `Complex<F> { pub re: F, pub im: F }` is a scalar type for `F = f32 | f64`; since v0.2 it is also an `Element` and a `Num` (section 22).
- Constructors are `Complex::new`, `from_real`, `from_polar`, and `i`.
- Methods are `conj`, `norm`, `norm_sqr`, `arg`, `is_nan`, `is_infinite`, `is_finite`, `exp`, `ln`, `sqrt`, `powc`, `powf`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, and `cmp_numpy`.
- Operators are complex/complex and complex/real `+`, `-`, `*`, `/`, unary `-`, and their assignment forms; branch cuts follow signed zero.

## 18. Printing and performance

- `Display` uses NumPy-style array layout and `Debug` uses `array(..., dtype=...)`; large arrays are summarized.
- Contiguous element-wise operations and reductions use slice fast paths; the default `rayon` feature parallelizes sufficiently large inputs.
- Results with and without `rayon` are identical for integer operations and agree within floating rounding for floating reductions.
- f32/f64 matrix multiplication uses rustnum's packed GEMM (AVX2/FMA or scalar fallback); numerical kernels are pure Rust, with optional rayon for parallelism.

## 19. Deliberate v0.1 limits

- Not supported as array dtypes (v0.1 list, see sections 22 onward for what v0.2 added): complex, float16/bfloat16, structured/record, object, datetime, masked, and string values.
- Not supported: memory mapping, fancy multi-axis advanced indexing, FFT axis/pad/crop options, polynomial classes, and the omitted keyword-rich statistics options documented above.
- Views have copy-on-write value semantics and never mutate a parent array; use explicit assignment methods to mutate selected data.

## 20. In-place and consuming operations

- Same-dtype, single-output unary element-wise methods have `<name>_inplace(&mut self)` and `into_<name>(self) -> Self` variants: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round`, `trunc`, `reciprocal`, `abs`, `square`, `sign`, `sigmoid`, `relu`, `sinc`, `i0`, `deg2rad`, `rad2deg`, `rint`, `fix`, `fabs`, `positive`, `negative`, `spacing`, and `nan_to_num`.
- Parameterized variants preserve the original arguments: `round_decimals_inplace(d)` / `into_round_decimals(d)`, `powf_inplace(p)` / `into_powf(p)`, `powi_inplace(p)` / `into_powi(p)`, `clip_inplace(lo, hi)` / `into_clip(lo, hi)`, and `nan_to_num_with_inplace(nan, posinf, neginf)` / `into_nan_to_num_with(nan, posinf, neginf)`.
- These express NumPy's `np.<ufunc>(a, out=a)` with copy-on-write value semantics. Shared inputs detach before mutation; unique contiguous arrays retain their data pointer. Sigmoid is the stable logistic formula; ReLU corresponds to `np.maximum(a, 0, out=a)`.
- Type-changing predicates (`is_nan`, `is_inf`, `is_finite`, `signbit`), tuple results (`modf`, `frexp`), and the non-element-wise `unwrap` do not have same-buffer variants.
- `(a.matmul(&b) / scale).into_tanh()` allocates no storage or metadata after matmul when its result is unique and contiguous. Owned arithmetic and unary negation reuse matching unique contiguous buffers; two-owned-array operators may reuse either side without reversing noncommutative operations.
- `astype::<T>(&self)` shares same-dtype storage using copy-on-write; `into_astype::<T>(self)` also retains the metadata without allocation. Different dtypes allocate a converted result. This is the Rust value-semantic equivalent of `np.asarray(a, dtype=T)`.
- `mapv_inplace(f)` retains its existing serial callback contract, including non-`Sync` captures. The added `mapv_inplace_parallel(f: impl Fn(T) -> T + Send + Sync)` uses the same contiguous parallel path as built-in in-place ufuncs. Requiring `Sync` on the existing method would break source compatibility.
- `IntoArray::scalar_value(&self) -> Option<T>` defaults to `None`; primitive scalar implementations return `Some(value)` so compound assignment avoids allocating a scalar array. Custom implementations must return `Some` only when `into_array` represents that scalar.

## 21. Memory cache (`np::memory`)

- `pool_stats() -> PoolStats` reports `cached_bytes`, `limit_bytes`, `cached_buffers`, `hits`, and `misses`.
- `clear_pool()` drops all cached buffers to the system allocator; whether released pages immediately leave the process working set depends on the OS allocator.
- `set_pool_limit(bytes)` sets the hard cache cap directly; zero disables it, and lowering it below current use clears the cache. The initial cap is 25% of the default 2 GiB memory budget (512 MiB). `RUSTNUM_POOL_BYTES` overrides the cap in bytes, read once on first use.
- The private cache uses thread-safe `(TypeId, exact capacity)` buckets for buffers of at least 1 MiB. A buffer exceeding remaining capacity is dropped immediately. Cached payload capacities count against the cap; small bookkeeping allocations do not.
- Final unique array owners return buffers to the cache. Views and clones never release live shared storage. An iterator can outlive its array; if it is the final Arc owner, that storage goes directly to the allocator when the iterator ends.
- Creation, element-wise output, and copying kernels obtain cached initialized buffers. Kernels overwrite every output before it is observable; no uninitialized reads or unsafe pool code are used. `from_vec` adopts the supplied Vec directly, so producers outside these kernels (including the existing GEMM implementation) do not acquire pooled buffers.

## 22. Complex arrays (v0.2)

- `Complex<f32>` is NumPy `complex64` and `Complex<f64>` is `complex128`; `Array<Complex<F>>` is an ordinary array (`DTYPE` is `"complex64"`/`"complex128"`). Elements are ordered lexicographically on `(re, im)`, so `sort`, `argsort`, `min`, `max`, `argmin`, `argmax` and comparisons follow NumPy.
- `Complex<F>` implements `Element` and `Num`, so every `Num` routine works: operators with broadcasting, `sum`, `prod`, `cumsum`, `cumprod`, `sum_axis`, `matmul`, `dot`, `trace`, `outer`, `einsum`, `square`, `power` (complex exponents). `%` panics, as NumPy raises.
- Mixed operands: `Array<Complex<F>>` with `Array<F>` (either side), with a real scalar `F` (either side, concrete `f32`/`f64` on the left), and with a `Complex<F>` scalar. The real operand is promoted to `x + 0j`.
- Methods on `Array<Complex<F>>`: `real()`, `imag()`, `conj()`, `angle()`, `angle_deg()`, `exp`, `log`, `log2`, `log10`, `sqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `is_nan`, `is_inf`, `is_finite`, `mean`, `mean_axis`, and `vdot(&other)` (conjugates the first argument; `linalg::vdot` is generic and does not).
- `absolute<A: Absolute>(&a) -> Array<A::Output>` is `np.absolute`/`np.abs`: real output for complex input. (`Array::abs()` on complex data comes from `Num::abs` and returns `|z| + 0j` as a complex array.)
- `astype` between `Complex<f32>`, `Complex<f64>` and real types keeps both parts between complex dtypes, gives `x + 0j` from real and drops the imaginary part towards real.
- Display follows NumPy: `[1.+2.j 3.-4.j]`, `Debug` is `array([...], dtype=complex128)`.
- I/O: `.npy`/`.npz` `<c8` and `<c16` read and write through the usual `np::io` functions.
- Not done: complex `np::linalg` (`solve`, `inv`, `eig` with complex input), complex `std`/`var`, arcsin/arccos/arctan on complex arrays, `np::poly::roots` still returns `(re, im)` pairs.
