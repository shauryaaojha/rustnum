# rustnum: NumPy for Rust, plan and design

`rustnum` is the first library of the *ML in Rust* project: a NumPy-style n-dimensional
array library that the later libraries (dataframes, plotting, classical ML, neural nets)
will build on.

```rust
use rustnum as np;

let x = np::linspace(0.0, 1.0, 5);          // like np.linspace(0, 1, 5)
let w = np::random::Rng::seed(42).normal(0.0, 1.0, &[5, 3]);
let y = x.reshape(&[1, 5]).matmul(&w).tanh(); // like np.tanh(x.reshape(1, 5) @ w)
println!("{y}");
```

## 1. Research summary

| Finding | Source | What we take from it |
|---|---|---|
| NumPy broadcasting is a stride trick: broadcast axes get stride 0, so no data is copied | NumPy C code explanations | Same design. Broadcasting is free |
| An array is a buffer + shape + strides + offset. Transpose, slicing and reshape-of-contiguous are O(1) views | NumPy internals | Same layout |
| `ndarray` (Rust, 137M downloads) is fast and complete, but its API leans on lifetimes (`ArrayView<'a>`), many type aliases (`Array2`, `ArrayViewMut3`, `IxDyn`) and many traits. Linear algebra, stats and random are separate crates | crates.io, docs.rs | Our main differentiator is one array type, no lifetimes in the public API, everything in one crate |
| Newer "drop-in NumPy" crates (`ferray`, `numrs2`, `numrst`) exist but have very low adoption (< 1k downloads/90 days) | crates.io (Oct 2026) | The space is open. Nobody has won "NumPy ergonomics in Rust" yet |
| Rust API Guidelines (C-CASE, C-CONV, C-COMMON-TRAITS, doc examples on every item) | rust-lang.github.io/api-guidelines | Followed as our checklist |
| Microsoft Pragmatic Rust Guidelines: M-DESIGN-FOR-AI, M-SINGLE-ITEM-PATH, M-PANIC-ON-BUG, M-TAUTOLOGICAL-TESTS, M-RUST-SHAPED | microsoft.github.io/rust-guidelines | APIs that are easy for humans are easy for AI. One path per item. Bugs panic, bad input returns `Result`. Tests assert ground truth (NumPy itself) |
| `matrixmultiply` (158M downloads) is a pure-Rust, BLAS-speed GEMM that accepts arbitrary strides | crates.io | Used for f32/f64 `matmul`. No C/Fortran BLAS needed |
| `rayon` (620M downloads) gives data parallelism with almost no code change | crates.io | Large element-wise ops and reductions run on all cores |

## 2. NumPy vs rustnum

| Topic | NumPy (Python) | rustnum (what we build) | Why |
|---|---|---|---|
| Element type | Decided at runtime (`dtype`) | Decided at compile time: `Array<f64>`, `Array<i64>`, `Array<bool>` | Type errors are caught by the compiler, and there is no per-element dispatch cost |
| Number of dimensions | Runtime | Runtime (`shape: Vec<usize>`) | Same flexibility as NumPy, one type for 1D/2D/nD (no `Array2` vs `ArrayD` split) |
| Views | Views write through to the parent (`b = a[0]; b[0] = 9` changes `a`) | **Value semantics with copy-on-write.** Views share memory until someone writes, then the writer gets its own copy | No aliasing bugs, no lifetimes, safe across threads. Deliberate difference |
| Copy vs view confusion | Easy to get wrong | Cannot affect correctness, only memory use | Same reason |
| Errors | Exceptions at runtime | Programming bugs (shape mismatch, index out of range) **panic** with NumPy's exact error wording. Bad data (file I/O, singular matrix) returns `Result` | Follows M-PANIC-ON-BUG. NumPy wording means humans and AI already know what it means |
| Keyword args (`axis=1`, `keepdims=True`) | Yes | No kwargs in Rust: `a.sum()` vs `a.sum_axis(1)`; negative axes work (`-1`) | Rust-shaped, but still predictable from the NumPy name |
| Scalars in operations | `a + 1`, `np.maximum(a, 0)` | Same: `&a + 1.0`, `np::maximum(&a, 0.0)`, `a.gt(0.5)` | Any argument that takes an array also takes a scalar |
| Parallelism | Single-threaded per op (GIL, mostly) | Large ops use all cores via `rayon` (feature, on by default) | Free speedup |
| GEMM | OpenBLAS / MKL (C, Fortran) | `matrixmultiply` (pure Rust) | Builds anywhere with just `cargo build` |
| Random | `np.random.default_rng(42)` (PCG64) | `np::random::Rng::seed(42)` (xoshiro256**) | Same API style. Numbers differ from NumPy's stream, by design |
| File format | `.npy` / `.npz` | Reads and writes `.npy` (100% compatible), CSV | Exchange arrays with Python both ways |
| Install | `pip install numpy` | `cargo add rustnum` | Pure Rust, no system libraries |

### Out of scope for v0.1 (honest list)
Complex numbers, `float16`, structured/record dtypes, object arrays, datetime dtypes,
masked arrays, `np.fft`, polynomials, `.npz`, memory-mapped files, string arrays,
`einsum`, fancy multi-axis advanced indexing. These are later phases (section 7).

## 3. Design principles

### 3.1 Developer ease
1. **Looks like NumPy.** `use rustnum as np;` then `np::zeros`, `np::arange`, `np::linspace`,
   `np::concatenate`, `np::linalg::inv`. A NumPy user should guess the name right the first time.
2. **One array type.** `Array<T>`. No views with lifetimes, no dimension type parameters.
3. **Cheap clones.** `Array` is a handle (`Arc` buffer + shape + strides + offset). Cloning, slicing,
   transposing and broadcasting never copy data.
4. **Scalars anywhere.** Through the `IntoArray<T>` trait, a function that takes an array also
   takes `T`, `&Array<T>`, `Array<T>`, `Vec<T>` or `&[T]`.
5. **Literal macro.** `array![[1.0, 2.0], [3.0, 4.0]]` and slicing with `s![1..3, .., -1]`.
6. **NumPy-style printing.** `println!("{a}")` prints `[[1. 2.]\n [3. 4.]]`.
7. **Pure Rust.** No BLAS, no C compiler, no system packages. `cargo build` just works on Windows,
   macOS and Linux.

### 3.2 AI ease (agents writing code with rustnum)
1. **NumPy names and NumPy error messages.** Models have seen billions of lines of NumPy. If our
   names, argument order and panic wording match, generated code is right more often and
   errors are self-explanatory.
2. **A mapping table** (`docs/NUMPY_MAPPING.md`): every supported NumPy call, its rustnum
   equivalent, and known differences. An agent can translate Python to Rust line by line.
3. **`llms.txt` and `AGENTS.md`** at the repo root: a compact API summary and the rules for
   contributing (where code goes, how to test).
4. **One path per public item** (M-SINGLE-ITEM-PATH). Core items live at the crate root; only
   `linalg`, `random` and `io` are sub-modules, exactly like NumPy.
5. **Every public function has a doc example that runs** (doctest). The examples are the spec.
6. **Strong types, few generics in signatures.** No lifetimes or exotic trait bounds in the public
   API, so compiler errors stay short and readable.
7. **Tests check NumPy itself** (M-TAUTOLOGICAL-TESTS). Python generates expected outputs with
   real NumPy; Rust tests compare against them.

## 4. How a library like this is made (anatomy)

```
ML in RUST/
├── Cargo.toml            workspace (later: rustframe, rustlearn, rustnet...)
├── AGENTS.md             rules for AI agents and contributors
├── llms.txt              compact API summary for LLMs
├── docs/                 PLAN.md, NUMPY_MAPPING.md, ARCHITECTURE.md
└── rustnum/
    ├── Cargo.toml        metadata, features (rayon default), deps
    ├── src/
    │   ├── lib.rs        crate docs, re-exports (the public surface)
    │   ├── element.rs    Element / Num / Float traits (the "dtype" system)
    │   ├── array.rs      Array<T>: storage, shape, strides, offset, indexing
    │   ├── layout.rs     strides, contiguity checks, element iteration
    │   ├── broadcast.rs  broadcasting rules
    │   ├── creation.rs   zeros, ones, arange, linspace, eye, ...
    │   ├── slicing.rs    s! macro, slice, negative indices, steps
    │   ├── ops.rs        + - * / % operators, comparisons, logic
    │   ├── ufunc.rs      exp, log, sqrt, sin, clip, maximum, ...
    │   ├── reduce.rs     sum, mean, std, argmax, cumsum along axes
    │   ├── manip.rs      reshape, transpose, concatenate, stack, split, flip, sort, unique, where
    │   ├── display.rs    NumPy-style printing
    │   ├── linalg.rs     matmul, dot, inv, det, solve, qr, eigh, svd, ...
    │   ├── random.rs     Rng: random, normal, uniform, integers, choice, shuffle
    │   └── io.rs         .npy and CSV load/save
    ├── tests/            integration + NumPy parity tests
    ├── benches/          criterion benchmarks
    └── examples/         runnable examples (linear regression, ...)
└── tools/
    ├── gen_fixtures.py   generates expected outputs with real NumPy
    └── bench_numpy.py    the same benchmarks in NumPy, for comparison
```

Things to keep in mind when building a library (not an app):
- **The public API is a promise.** Everything `pub` is something we must keep working. Keep internals `pub(crate)`.
- **Semantic versioning.** 0.x while the API is still moving.
- **Docs are part of the product.** Crate-level docs, module docs, an example on every item.
- **Errors vs panics are part of the API** (section 2).
- **Features must be additive** (M-FEATURES-ADDITIVE). `rayon` on or off gives the same results.
- **Common traits** (C-COMMON-TRAITS): `Clone`, `Debug`, `Display`, `PartialEq`, `Default`, `From`, `FromIterator`, `Index`/`IndexMut`, `IntoIterator`, `Send`, `Sync`.
- **CI gates:** `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`, `cargo doc`.

## 5. Core architecture

```rust
pub struct Array<T> {
    data: Arc<Vec<T>>,   // shared buffer, copy-on-write via Arc::make_mut
    shape: Vec<usize>,   // [rows, cols, ...]; [] means a 0-d scalar array
    strides: Vec<isize>, // in elements, not bytes; 0 for broadcast axes; negative for flips
    offset: usize,       // index of element [0, 0, ...] inside data
}
```
- **Element traits:** `Element` (all types incl. `bool`), `Num: Element` (`f32 f64 i32 i64 u8 usize`), `Float: Num` (`f32 f64`). Math like `exp` exists only on `Array<F: Float>`, so `int_array.exp()` is a compile error, not a runtime surprise.
- **Fast path:** when an array is C-contiguous, operations work on the raw slice (and in parallel above a size threshold). Strided arrays go through a general index iterator.
- **Writes:** `IndexMut`, `set`, `+=` call `Arc::make_mut`, so a shared buffer is copied once before the first write.

## 6. Build phases (today)

| Phase | Who | Scope | Exit criteria |
|---|---|---|---|
| 0. Plan + scaffold | Claude | This plan, workspace, module skeleton, `AGENTS.md`, git | `cargo build` passes on the empty skeleton |
| 1. Core | codex | element, array, layout, broadcast, creation, slicing, ops, ufunc, reduce, manip, display, `astype` | `cargo test` green, every pub fn has a doctest |
| 2a. Linear algebra + speed | codex (worktree) | `linalg`, rayon fast paths, criterion benches | Tests green, results match NumPy within 1e-10 (f64) |
| 2b. Random + I/O + parity | agy (worktree) | `random`, `io` (.npy, CSV), `tools/gen_fixtures.py`, parity tests | `.npy` round-trips with real NumPy, parity tests green |
| 3. Docs | agy | `README.md`, `docs/NUMPY_MAPPING.md`, `docs/ARCHITECTURE.md`, `llms.txt`, examples | Every mapping row compiles; examples run |
| 4. Verify | Claude | Merge, full test suite, clippy, doc build, benchmarks vs NumPy | All gates green, benchmark table in README |

## 7. Later phases (not today)
- v0.2: `einsum`, `.npz`, fancy indexing with index arrays on several axes, `fft`, `polynomial`, complex numbers.
- v0.3: SIMD kernels (`wide`), optional BLAS backend feature, `f16`/`bf16`.
- v0.4: Python bindings via PyO3 (`import rustnum`) so Python users can try it.
- Then the next libraries on top: dataframe glue (Polars), plotting (`seaborn`-style), classical ML, autograd.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Copy-on-write surprises NumPy users who expect views to write through | Documented as the first row of the differences table; `slice_assign` gives the explicit in-place write |
| Generic strided iteration is slow | Contiguous fast paths for every hot op; benchmarks catch regressions |
| Hand-written SVD/eig less accurate than LAPACK | Jacobi methods (numerically stable), parity tests vs NumPy with tolerances |
| AI agents write tautological tests | Parity tests against NumPy are the source of truth |
