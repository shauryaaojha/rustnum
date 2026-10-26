# rustnum

NumPy-style n-dimensional arrays for Rust. One array type, NumPy names, pure Rust.

`rustnum` is the first library of the *ML in Rust* project: an attempt to rebuild the Python
ML stack in Rust, one library at a time. If you know NumPy, you should be able to guess the name
of most functions on the first try. If you are an AI agent, start with [`llms.txt`](llms.txt).

## Quick start

```rust
use rustnum as np;

fn main() {
    let a = np::arange(0.0, 6.0, 1.0).reshape(&[2, 3]); // np.arange(6.0).reshape(2, 3)
    let b = np::array![10.0, 20.0, 30.0]; // np.array([10.0, 20.0, 30.0])
    let c = &a + &b; // broadcasting: shape [2, 3] + shape [3]
    println!("{c}");
    println!("sum = {}", c.sum());
    println!("column means = {}", c.mean_axis(0)); // np.mean(c, axis=0)
    println!("second column = {}", c.slice(&np::s![.., 1])); // c[:, 1]
    println!("a @ a.T =\n{}", a.matmul(&a.t()));
}
```

This exact code is tested in `rustnum/tests/readme_quickstart.rs`, so it cannot go out of date.
More programs are in `rustnum/examples/`: `quickstart`, `linear_regression`, `pca`, `kmeans`,
`image_ops`. Run one with:

```bash
cargo run --release -p rustnum --example pca
```

## Install

The crate is not on crates.io yet. Use it from a path or from git:

```toml
[dependencies]
rustnum = { path = "rustnum" }
# or: rustnum = { git = "https://github.com/shauryaaojha/rustnum" }
```

Then write `use rustnum as np;` and call `np::zeros`, `np::linalg::inv`, and so on.
There are no system libraries to install and no C or Fortran compiler is needed.

Features:

- `rayon` (on by default): large element-wise operations and reductions use all CPU cores.
  Turn it off with `default-features = false`. Results are the same either way.

## What you get

- **One array type**, `Array<T>`, for 0-d, 1-d and n-d data. No lifetimes and no `Array2` / `ArrayD` split.
- **Element types:** `f32`, `f64`, `i32`, `i64`, `u8`, `usize`, `bool`.
- **Creation:** `zeros`, `ones`, `full`, `arange`, `linspace`, `logspace`, `eye`, `diag`, `tri`, `meshgrid`,
  and the `array!` macro.
- **Indexing:** slices with the `s!` macro (negative indices and steps), boolean masks, `take`, row and column access.
- **Broadcasting** for every operator and binary function. Scalars work anywhere an array does.
- **Operators:** `+ - * / %` (also `+=` and friends), comparisons, logical `& | ^ !`.
- **Math:** `exp`, `log`, `sqrt`, trig, `clip`, `maximum`, `where_`, and more.
- **Reductions:** `sum`, `mean`, `std`, `min`, `argmax`, `cumsum`, `median`, `percentile`, each with an `_axis` form.
- **Shape manipulation:** `reshape`, `transpose`, `concatenate`, `stack`, `split`, `flip`, `roll`, `sort`, `unique`.
- **Linear algebra:** `matmul` (BLAS-speed GEMM through `matrixmultiply`), `solve`, `inv`, `det`, `lstsq`,
  `cholesky`, `qr`, `eigh`, `svd`, `norm`.
- **Random numbers:** `np::random::Rng` with `random`, `normal`, `uniform`, `integers`, `choice`, `shuffle`, and more.
- **File I/O:** `.npy` files (compatible with NumPy both ways) and CSV.
- **NumPy-style printing:** `println!("{a}")` prints `[[1. 2.]\n [3. 4.]]`.

The full NumPy to rustnum table is in [`docs/NUMPY_MAPPING.md`](docs/NUMPY_MAPPING.md).

## NumPy vs rustnum

| Topic | NumPy | rustnum |
|---|---|---|
| Element type | Chosen at run time (`dtype`) | Chosen at compile time: `Array<f64>`, `Array<i64>`, `Array<bool>` |
| Dimensions | Run time | Run time (one type for 1-d, 2-d, n-d) |
| Views | A view writes through to its parent | **Copy-on-write.** Slices share memory until someone writes; the writer gets its own copy. Use `slice_assign` to write into part of an array |
| Errors | Exceptions | Bugs (shape mismatch, bad index) **panic** with NumPy's wording. Bad data (files, singular matrix) returns `np::Result` |
| Keyword arguments | `sum(axis=1)` | No kwargs: `a.sum()` and `a.sum_axis(1)`. Negative axes work |
| Scalars | `a + 1`, `np.maximum(a, 0)` | `&a + 1.0`, `np::maximum(&a, 0.0)`. Float literals need the `.0` |
| Random | `default_rng(42)` (PCG64) | `Rng::seed(42)` (xoshiro256**). Same style, different numbers |
| GEMM | OpenBLAS / MKL | `matrixmultiply` (pure Rust) |
| File format | `.npy`, `.npz` | `.npy` and CSV (no `.npz`) |

Not in v0.1: a complex dtype (`Complex<F>` exists as a scalar type; `np::fft` returns `(re, im)` pairs),
`float16`, structured and datetime dtypes, masked arrays, fancy indexing with index arrays on several axes.

## Benchmarks

Median times in milliseconds, lower is better. Windows 11, 18 hardware threads, Rust 1.99.1 (`--release`,
`rayon` on), NumPy 2.3.2 with OpenBLAS 0.3.30. rustnum: `cargo bench -p rustnum --bench core`.
NumPy: `python tools/bench_numpy.py`. Both were run on an idle machine.

| Benchmark | NumPy | rustnum | Faster |
|---|---:|---:|---|
| add (1e6 f64, contiguous) | 3.04 | 2.67 | rustnum 1.1x |
| add broadcast ([1000,1000] + [1000]) | 3.04 | 4.13 | NumPy 1.4x |
| sum (1e7 f64) | 7.05 | 1.84 | rustnum 3.8x |
| mean_axis(0) [1000,1000] | 0.42 | 4.30 | NumPy 10x |
| transpose().to_contiguous() [1000,1000] | 3.26 | 6.50 | NumPy 2.0x |
| exp (1e6 f64) | 5.76 | 3.25 | rustnum 1.8x |
| sort (1e6 f64) | 12.02 | 32.29 | NumPy 2.7x |
| inv (200x200) | 0.97 * | 6.12 | NumPy 6.3x |
| svd (100x100) | 6.87 | 113.04 | NumPy 16x |
| matmul (256x256 f64) | 0.69 | 0.66 | about equal |
| matmul (512x512 f64) | 2.68 | 5.16 | NumPy 1.9x |
| matmul (1024x1024 f64) | 17.43 | 35.81 | NumPy 2.1x |

\* NumPy `inv` measured with `OPENBLAS_NUM_THREADS=1`. With default threading it took 85–102 ms on this machine
(an OpenBLAS threading problem on Windows), which would make rustnum look 17x faster; that comparison is not fair.

Where rustnum is slower, and why:
- `mean_axis(0)`, `transpose().to_contiguous()`: reductions and copies across a strided axis walk memory in the
  wrong order. Cache-blocked kernels are the next optimization.
- `svd`, `inv`: pure-Rust one-sided Jacobi SVD and unblocked LU against LAPACK's blocked algorithms.
- `sort`: single-threaded comparison sort with NaN handling, against NumPy's SIMD sort.
- Large `matmul`: `matrixmultiply` is about half the speed of OpenBLAS at 512+.

## Tests

```bash
cargo test                       # unit tests, doctests, property tests
cargo test --no-default-features # same, without rayon
python tools/gen_fixtures.py     # writes NumPy reference outputs to rustnum/tests/fixtures/
cargo test                       # now the parity tests also run, comparing against NumPy
```

`gen_fixtures.py` needs Python with NumPy installed. Without the fixture files the parity tests
cannot run. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the test layers fit together.

## Documentation

- [`docs/NUMPY_MAPPING.md`](docs/NUMPY_MAPPING.md): NumPy call on the left, rustnum call on the right.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): how arrays are stored, how to add a function.
- [`docs/API.md`](docs/API.md): the API contract.
- [`docs/PLAN.md`](docs/PLAN.md): design and research.
- [`llms.txt`](llms.txt): short API cheat sheet for LLMs.
- [`AGENTS.md`](AGENTS.md): rules for AI agents and contributors.

## Roadmap

- **rustnum v0.2:** `einsum`, `.npz`, fancy indexing, `fft`, `polynomial`, complex numbers.
- **rustnum v0.3:** SIMD kernels, optional BLAS backend, `f16` / `bf16`.
- **rustnum v0.4:** Python bindings (PyO3).
- **Next libraries:** dataframes (Polars glue), plotting, classical ML, autograd and neural nets.

## License

MIT OR Apache-2.0.
