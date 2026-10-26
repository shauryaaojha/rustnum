# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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

[Unreleased]: https://github.com/shauryaaojha/rustnum/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shauryaaojha/rustnum/releases/tag/v0.1.0
