# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Integer `/` and `%` (operators, `/=`, `%=`, scalar-left and `ma`) no longer panic: a zero divisor gives 0 and `MIN / -1` wraps, like NumPy. Masked integer division masks zero divisors.
- `svd`, `qr`, `lstsq`, `pinv`, `matrix_rank` and `cond` scale the input first when its entries are near overflow or underflow (as LAPACK does), so matrices of size ~1e±300 give NumPy's answer instead of inf, NaN or 0.
- `convolve`, `correlate`, `diff`, `cross`, `kron` and `einsum` wrap on integer overflow instead of panicking in debug builds.
- Structured-array sorting puts NaN fields last field by field, so `partial_cmp` on `record!` types never returns `None`.
- `np::ma` unary functions keep the original data under masked or out-of-domain positions.
- `chebroots` / `chebcompanion` of a degree-1 series returned half the root.
- `np::power` on `f16` / `bf16` panicked or overflowed (it used the integer path); it now computes in f32.
- Slices with a huge step (`a[::2**63-1]`, accepted by NumPy) no longer panic with "slice step is too large".
- `.npy` headers with a malformed shape return `Error::Format` instead of panicking.
- `nanmedian`, `nanpercentile` and `nanvar` no longer copy contiguous input twice; the buffer pool skips buffers with spare capacity.

### Changed
- `sort` drops its single-threaded radix path (and its `unsafe`): measured slower than the comparison sort at every size, and never reached with rayon on.

### Added
- `tests/fuzz_io.rs`: 35,500 mutated `.npy`, `.npz` and text files must return `Ok` or `Err`, never panic.
- `tests/audit_fixes.rs`: overflow, empty-input, NaN, scaling and edge-case probes checked against NumPy 2.3.

## [0.2.0] - 2026-11-02

### Added
- `numpy.random`: every remaining `Generator` distribution (binomial, poisson, gamma, beta, chisquare, dirichlet, multinomial, multivariate_normal, hypergeometric, vonmises, wald, zipf and the rest).
- `np::polynomial`: `Polynomial`, `Chebyshev`, `Legendre`, `Laguerre`, `Hermite`, `HermiteE` classes and their module functions (`*fit`, `*val`, `*der`, `*int`, `*roots`, `*fromroots`, conversions).
- Remaining nan functions: `nanvar`, `nanstd`, `nanmedian`, `nanpercentile`, `nanargmax`/`nanargmin`, `nanprod`, `nancumsum`, `nancumprod`.
- Sorting and selection: `lexsort`, `partition`, `argpartition`, `flatnonzero`, `select`, `piecewise`, `extract`, `place`, `copyto`, `trim_zeros`.
- Index helpers: `tril_indices`/`triu_indices` (and `_from`), `diag_indices` (and `_from`), `fill_diagonal`, `ndindex`, `ndenumerate`, `mask_indices`.
- Shape and creation: `rot90`, `dstack`, `column_stack`, `row_stack`, `atleast_1d/2d/3d`, `geomspace`, `vander`, `fromiter`, `frombuffer`, `around`, `real_if_close`, `isposinf`, `isneginf`.
- Stride tricks: `sliding_window_view` and bounds-checked `as_strided`; writes copy overlapping views first.
- Bitwise ops, shifts, `packbits`, `unpackbits`, `bitwise_count` and integer array operators.
- `vectorize`, `apply_along_axis`, `apply_over_axes`.
- `io::loadtxt`, `io::savetxt`, `io::genfromtxt`.
- `set_printoptions`, `get_printoptions` and scoped print options honoured by `Display` and `Debug`.
- `histogram2d`, `histogramdd`, `einsum_path`, `linalg::tensorinv`, `linalg::tensorsolve`.
- Complex dtype: `Complex<f32>` (`complex64`) and `Complex<f64>` (`complex128`) are array elements (`Element` + `Num`). Operators with broadcasting and mixed real/complex operands, `real`, `imag`, `conj`, `angle`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `power`, `sum`/`mean`/`prod`, `matmul`/`dot`/`vdot`, `astype` to and from real, NumPy-style display, `np::absolute`, and `.npy`/`.npz` `<c8`/`<c16` read and write.
- `f16` (IEEE binary16, `float16`) and `bf16` (bfloat16) array elements: `Element`, `Num` and `Float` implementations that compute through `f32`, round-to-nearest-even conversions for NaN/inf/subnormals, F16C-accelerated bulk `astype` between `f32` and `f16`, NumPy-style display and `.npy` `<f2` read/write (bfloat16 has no NumPy dtype and no `.npy` encoding).
- Advanced (integer-array) indexing: `s!` accepts index arrays (`a.slice(&s![[0, 2], [1, 3]])`, `slice_assign` for the assign form) with NumPy's broadcasting and axis-placement rules, plus `ix_`, `take_along_axis`, `put_along_axis`, `put`, `choose`, `ravel_multi_index` and `unravel_index`.
- Masked arrays: public module `np::ma` with `MaskedArray<T>`, `masked_array`, `masked_where`, `masked_invalid`, `masked_equal`, `masked_greater`/`less` (and `_equal`), `masked_inside`/`outside`, `masked_values`, `filled`, `compressed`, `getmask`, `getdata`, `count`, mask-propagating arithmetic (division by zero and out-of-domain `sqrt`/`log` are masked), masked-aware reductions (`sum`, `mean`, `std`, `var`, `min`, `max`, `argmin`, `argmax` and `_axis` forms) and NumPy-style display with `--`.
- datetime64 / timedelta64: `np::DateTime64<U>` and `np::TimeDelta64<U>` array elements with NumPy units (`np::datetime::{Y, M, W, D, h, m, s, ms, us, ns}`), `NaT`, ISO 8601 `parse`, calendar-aware unit `astype`, arithmetic (`dt - dt`, `dt +- td`, `td * int`, `td / td`, `%`, `floor_div`), comparisons, `sort`/`argsort` with `NaT` last, `arange`, `datetime_as_string`, `isnat`, NumPy-style display, and `.npy`/`.npz` `<M8[unit]` / `<m8[unit]` read and write.
- Structured dtypes: the `rustnum::record!` declarative macro defines a record struct as an array element (nested records, complex, f16 and datetime fields), with `np::Record` (`FIELDS`, `descr()`), `Array::field` / `set_field` (`a['x']`), `argsort_order` / `sort_order` (`order=`), NumPy-style display and `.npy`/`.npz` read and write with a structured `descr`.
- Python bindings: new workspace crate `rustnum-py` (PyO3 0.30, `abi3-py39`; not a default member) exposing an `ndarray` class over float64/float32/int64/bool with creation functions, operators, indexing, reductions, `matmul`, `linalg`, `random` and zero-copy `numpy.asarray` through `__array_interface__`.

### Changed (breaking)
- `SliceArg` has a new variant `Fancy`; exhaustive matches on it need an arm.
- `np::fft` follows NumPy's signatures: `fft`, `ifft`, `fft2`, `ifft2` take one real or complex array and return an `Array<Complex<F>>`, `rfft` returns `Array<Complex<F>>`, and `irfft(&a, n: Option<usize>)` takes a complex array (`None` is NumPy's default length). The v0.1 `(re, im)` pair signatures are removed.

### Changed
- `linalg`: blocked LU, blocked Cholesky and compact-WY QR, Householder tridiagonalisation with implicit-QL `eigh`, and Golub-Kahan SVD, all through the packed GEMM; null-space completion is O(n^3).
- SIMD: AVX2 f64/f32 kernels for `expm1`, `log2`, `log10`, `log1p`, `tan` (max 4 ulp measured); NEON math and GEMM microkernels are compile-checked for aarch64 only.
- `repr` (`Debug`) of arrays wraps rows one column earlier and puts `dtype=` on its own line when the line passes `linewidth`, matching `numpy.array_repr`; datetime and structured dtypes print their NumPy `dtype=` text.
- `random`: replaced Box-Muller normal and exponential variate generation with NumPy-parity 256-strip Ziggurat method.
- `random`: added deterministic chunked parallel filling for large outputs (>= 65536 elements) behind the `rayon` feature using xoshiro256** `jump()` streams.
- `random`: the variate stream for normal and exponential distributions has changed due to switching to the Ziggurat method; seed-42 regression values are updated accordingly.
- `manip`: accelerated `sort`, `argsort`, and `unique` with single-pass NaN partitioning, LSD radix sort for `f64`/`f32`/`i64`/`i32` lanes >= 65536, and Rayon parallel sorting for large lanes.
- `simd_math`: added private AVX2+FMA transcendental SIMD math kernels (`exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sin`, `cos`, `tan`, `tanh`, `sigmoid`) with runtime CPU feature dispatch and scalar fallback.

## [0.1.0] - 2026-10-26

First release of `rustnum`: NumPy-style n-dimensional arrays in pure Rust.

### Added
- `Array<T>`: one n-dimensional array type with copy-on-write value semantics, strides, O(1)
  transpose, slicing and broadcasting, for `f32`, `f64`, `i32`, `i64`, `u8`, `usize` and `bool`.
- Creation: `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `tri` and friends; the
  `array!` and `s!` macros.
- Operators and comparisons with broadcasting and scalar arguments; NumPy-style `Display`.
- Element-wise functions (`exp`, `log`, `sqrt`, trigonometric and hyperbolic functions, `clip`,
  `floor`, `round` with NumPy's half-to-even, `maximum`, `minimum`, ...).
- Reductions over the whole array or one axis: `sum`, `mean`, `std`, `var`, `min`, `max`,
  `argmax`, `argmin`, `cumsum`, `median`, `percentile`.
- Manipulation: `reshape`, `transpose`, `swapaxes`, `concatenate`, `stack`, `split`, `flip`,
  `roll`, `repeat`, `tile`, `sort`, `argsort`, `unique`, `where_`, `astype`.
- `linalg`: `matmul` (via `matrixmultiply`), `dot`, `det`, `inv`, `solve`, `norm`, `cholesky`,
  `qr`, `eigh`, `svd`, `pinv`, `matrix_rank`.
- `random`: `Rng` (xoshiro256**) with `random`, `normal`, `uniform`, `integers`, `choice`,
  `permutation` and `shuffle`.
- `io`: `.npy` read and write (format 1.0 and 2.0, C and Fortran order) and CSV; both
  interoperate with NumPy.
- `.npz` read and write (`savez`, `savez_compressed`, `load_npz`, lazy `NpzFile`) with a
  built-in ZIP, DEFLATE and CRC-32 implementation; no dependencies, zip64 supported.
- Optional `rayon` feature (on by default) for parallel element-wise ops and reductions.
- NumPy parity test suite driven by fixtures from `tools/gen_fixtures.py`.
- Continuous integration on Linux, macOS and Windows; dual MIT / Apache-2.0 licence.

[Unreleased]: https://github.com/shauryaaojha/rustnum/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shauryaaojha/rustnum/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shauryaaojha/rustnum/releases/tag/v0.1.0
