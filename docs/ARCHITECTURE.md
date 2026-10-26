# rustnum architecture

How the library works inside, and how to add to it. Read `docs/API.md` first for the public contract and
`AGENTS.md` for the rules. This file is for people (and agents) who change `rustnum/src/`.

## 1. Module map

| File | What it holds |
|---|---|
| `element.rs` | `Element`, `Num`, `Float` traits: the "dtype" system |
| `array.rs`, `layout.rs` | `Array<T>`, strides, contiguity checks, element iteration |
| `broadcast.rs` | `IntoArray`, broadcasting rules |
| `creation.rs` | `zeros`, `arange`, `linspace`, `eye`, `array!` |
| `slicing.rs` | `s!`, `slice`, `slice_assign`, masks, `take` |
| `ops.rs` | Operators and comparisons |
| `ufunc.rs` | Element-wise math (`exp`, `sqrt`, `clip`, `maximum`, ...) |
| `reduce.rs` | `sum`, `mean`, `argmax`, `cumsum`, and their `_axis` forms |
| `manip.rs` | `reshape`, `transpose`, `concatenate`, `sort`, `unique`, ... |
| `display.rs` | NumPy-style `Display` and `Debug` |
| `linalg.rs`, `random.rs`, `io.rs` | Public modules `np::linalg`, `np::random`, `np::io` |

Core items are defined in private modules and re-exported once in `lib.rs`. Only `linalg`, `random`
and `io` are public modules. Do not add a second path to an item.

## 2. Storage: buffer, shape, strides, offset

```rust
pub struct Array<T> {
    data: Arc<Vec<T>>,   // shared buffer
    shape: Vec<usize>,   // [] means 0-d (one value)
    strides: Vec<isize>, // in elements, not bytes; 0 = broadcast axis, negative = flipped
    offset: usize,       // position of element [0, 0, ...] inside data
}
```

The element at index `[i, j]` lives at `data[offset + i * strides[0] + j * strides[1]]`.

A 2 x 3 array created by `arange(0.0, 6.0, 1.0).reshape(&[2, 3])`:

```
data    = [0, 1, 2, 3, 4, 5]
shape   = [2, 3]
strides = [3, 1]          row step = 3 elements, column step = 1 element
offset  = 0

          col 0  col 1  col 2
  row 0 [   0      1      2  ]      data index = 0*3 + j
  row 1 [   3      4      5  ]      data index = 1*3 + j
```

Other arrays over the same `data` buffer, with no copying:

```
a.t()                shape [3, 2]  strides [1, 3]  offset 0     (swap the strides)
a.slice(s![.., 1])   shape [2]     strides [3]     offset 1     (start at column 1, drop axis)
a.slice(s![..;-1])   shape [2, 3]  strides [-3, 1] offset 3     (start at last row, walk backwards)
a.flip(1)            shape [2, 3]  strides [3, -1] offset 2
```

Because every one of these is only new `shape`, `strides` and `offset` values, slicing, transposing, flipping and
broadcasting are O(1). `is_contiguous()` is true when the strides are exactly the C-order strides for the shape
(and the offset plus size fit the buffer in order). Only then is `as_slice()` `Some`.

A 0-d array has `shape = []` and `strides = []`; its one value is at `data[offset]`.

## 3. Copy-on-write

`Array` is a handle: cloning bumps the `Arc` count and copies `shape` and `strides` (a few numbers), not the data.
Every method that writes (`set`, `IndexMut`, `fill`, `mapv_inplace`, `slice_assign`, `mask_assign`, `+=`, `shuffle`)
first calls `Arc::make_mut(&mut self.data)`:

- If this handle is the only owner, `make_mut` returns the buffer and the write is in place.
- If other handles share the buffer, `make_mut` clones the whole `Vec<T>` first, so the writer gets its own copy
  and the other arrays are untouched.

Two details to keep in mind when you write mutating code:

1. After `make_mut` clones the buffer, the clone keeps the same layout, so the existing `shape`, `strides` and `offset`
   stay valid. (A clone of a small slice of a huge buffer copies the whole buffer. That is accepted in v0.1.)
2. A view must never write into its parent. If you add a mutating method, call `make_mut` before touching any element.
   A test that mutates a clone and checks the original is mandatory.

This is the main documented difference from NumPy (views write through there).

## 4. Broadcasting is zero strides

To broadcast an array to a new shape, align shapes from the right. For each axis, the sizes must be equal or the
array's size must be 1. A size-1 axis (or a missing leading axis) gets stride 0, so every index along it reads the
same element.

```
a: shape [2, 3] strides [3, 1]          b: shape [3] strides [1]

b.broadcast_to([2, 3]):
   shape   [2, 3]
   strides [0, 1]      <- new leading axis, stride 0
   b[i, j] = data[offset + i*0 + j*1] = b[j]
```

`broadcast_shapes(a, b)` returns `None` when the shapes are incompatible. Operators and functions that need
it panic with NumPy's wording: `operands could not be broadcast together with shapes [2, 3] [4]`.

A binary operation first computes the result shape, then broadcasts both inputs to it (as views), then runs the
usual loops. No input data is copied.

## 5. Iteration: contiguous fast path and odometer

All element-wise operations and reductions have two code paths:

1. **Contiguous fast path.** If every input is contiguous with the same shape, work directly on
   `&[T]` slices: `out[i] = f(a[i], b[i])`. This is a simple loop the compiler can vectorise. Above the rayon
   threshold (next section), the slices are processed in parallel chunks.
2. **General strided path.** Otherwise walk the index space with an *odometer*: keep one running index per axis
   and one running data position per input. Step the last axis by its stride; when it wraps, reset it and carry
   into the next axis, adjusting each position by the stride difference. Nothing is allocated per element, and
   there is no per-element `Vec<usize>` index.

```
shape [2, 3], strides [3, 1]:    positions: 0 1 2 | 3 4 5
shape [3, 2], strides [1, 3] (transpose):  0 3 | 1 4 | 2 5      (carry happens every 2 steps)
```

Reductions along an axis use the same walker over the *other* axes, and fold along the reduced axis. `sum` uses
pairwise summation in both the serial and parallel paths, so `f32` sums stay accurate and the `rayon` feature
does not change results beyond float rounding.

## 6. Parallelism: the rayon threshold

With the `rayon` feature (on by default), contiguous element-wise operations and reductions on arrays with
`size >= 1 << 15` (32,768 elements) run in parallel. Smaller arrays stay serial because thread hand-off costs more than the
work. The strided path is serial.

Rules for contributors:

- Integer results must be identical with and without the feature.
- Float results must be within rounding (same pairwise sum structure).
- Gate parallel code with `#[cfg(feature = "rayon")]` and keep a serial version that compiles without it.
  `cargo test --no-default-features` must pass.

## 7. Matrix multiplication dispatch

`a.matmul(&b)` follows NumPy's rules:

| Left | Right | Result |
|---|---|---|
| 1-d `[k]` | 1-d `[k]` | 0-d (dot product) |
| 2-d `[m, k]` | 2-d `[k, n]` | 2-d `[m, n]` |
| 1-d `[k]` | 2-d `[k, n]` | 1-d `[n]` (a 1 is prepended, then removed) |
| 2-d `[m, k]` | 1-d `[k]` | 1-d `[m]` |
| n-d | n-d | batched: leading dims broadcast, last two dims multiply |

Dispatch for one 2-d product:

1. Check the inner dimensions agree, else panic with a NumPy-style message.
2. If the element type is `f32` or `f64`, call rustnum's packed GEMM (`linalg/gemm/`, BLIS-style: cache blocking, AVX2+FMA micro-kernels, rayon over
   blocks). It takes a signed row stride and column stride for each operand and packs through them, so
   transposed and flipped views are passed as-is with no copy.
3. Otherwise (`i32`, `i64`, `u8`, `usize`) use a plain triple loop with a cache-friendly order (`i, k, j`).
4. For batched input, broadcast the batch dimensions, then run steps 1 to 3 once per batch item.

Linear-algebra routines in `np::linalg` (`solve`, `inv`, `qr`, `eigh`, `svd`) are written in plain Rust and
work on contiguous copies of their input.

## 8. How to add a new element-wise function (ufunc)

Example: add `a.cosh_minus_one()`. Work through this list in order.

1. **Decide the name.** Use NumPy's name. If NumPy has none, say in the doc comment that it is an extra
   (like `sigmoid`). Check `docs/API.md`; the name must not already exist under another spelling.
2. **Decide the bound.** Float-only math goes on `impl<F: Float> Array<F>`. Functions that make sense for integers
   go on `impl<T: Num> Array<T>`. If you need a scalar function that does not exist yet, add it to the `Float` trait
   in `element.rs` and implement it for `f32` and `f64`.
3. **Write the method in `ufunc.rs`**, next to similar functions. Use the shared helper that applies a closure
   to every element (the one `exp` and `sqrt` use), so you get the contiguous fast path, the strided path and the
   rayon path for free. Do not write your own loop.
4. **Binary function?** Make it a free function taking `impl IntoArray<T>` for both arguments and re-export it
   once in `lib.rs`. It must broadcast.
5. **Doc comment.** First line is one sentence that names the NumPy call: `/// Element-wise cosh(x) - 1, like np.cosh(a) - 1.`
   Add a runnable example with a hand-computed expected value.
6. **Re-export** in `lib.rs` if it is a free function. Methods need no re-export.
7. **Unit tests** in the same file: a small known case, a non-contiguous input (for example `a.t()`), a 0-d input,
   an empty input, and NaN handling if relevant.
8. **Parity test.** Add the function to `tools/gen_fixtures.py` (generate input and `np.<fn>` output with real NumPy),
   run `python tools/gen_fixtures.py`, and add a comparison in the parity test file.
9. **Docs.** Add a row to `docs/NUMPY_MAPPING.md` and a line to `llms.txt`. If it is a new public item, add it to `docs/API.md`
   (ask the maintainer first; the API contract is binding).
10. **Gates:**
    ```bash
    cargo fmt --all
    cargo clippy --all-targets -- -D warnings
    cargo test --all-features
    cargo test --no-default-features
    cargo doc --no-deps
    ```

## 9. How to add a new reduction

Example: add `a.mean_abs()` and `a.mean_abs_axis(axis)`.

1. **Name and bound.** NumPy name; `Float` if it divides or takes roots, `Num` otherwise.
2. **Whole-array form** in `reduce.rs`: returns a plain value (`T`, `usize` or `bool`). Use the contiguous slice
   fast path (and pairwise summation if you add floats). Use the shared strided walker for non-contiguous input.
3. **Axis form** `_axis(axis: isize)`: normalise the axis (negative counts from the end, out of range panics with
   `axis 3 is out of bounds for array of dimension 2`), then fold along that axis. The result shape is the input shape
   with that axis removed. Cumulative operations (`cumsum_axis`) keep the shape instead.
4. **Empty input.** Decide what NumPy does. If the reduction has no identity (like `min`), panic with NumPy's message.
5. **`ddof`-style options** become a separate method (`var_ddof`), never a flag argument.
6. **Parallel path.** Add it behind `#[cfg(feature = "rayon")]` above the `1 << 15` threshold, with the same
   pairwise structure as the serial path.
7. **Doc comment and example.** First line names the NumPy equivalent. Show a 2-d example for the axis form.
8. **Tests.** Whole-array, each axis of a 3-d array, negative axis, empty, non-contiguous input, `f32` and `f64`.
   Parity test against NumPy via `tools/gen_fixtures.py`.
9. **Docs.** Rows in `docs/NUMPY_MAPPING.md` and `llms.txt`.
10. Run the gates from section 8.

## 10. Testing layers

| Layer | Where | What it checks |
|---|---|---|
| Doctests | `///` examples on every public item | The documented example runs and prints what it claims. These are also the spec for AI agents |
| Unit tests | `#[cfg(test)]` in each `src/*.rs` | Hand-computed values, panics (`#[should_panic(expected = "...")]`), edge cases (empty, 0-d, strided) |
| Property tests | `proptest` in `rustnum/tests/` | Invariants: `a.t().t() == a`, `reshape` then `ravel` is identity, `sum` with and without rayon agree, broadcasting is symmetric |
| NumPy parity | `rustnum/tests/` + `tools/gen_fixtures.py` | Real NumPy generates inputs and expected outputs into `rustnum/tests/fixtures/` (`.npy`). Rust loads them and compares with a tolerance (1e-10 for f64). This is the ground truth |
| Example and README tests | `rustnum/tests/readme_quickstart.rs`, `rustnum/examples/` | The README quick start and the example programs keep compiling and running |

Rules (from `AGENTS.md`): tests assert ground truth, not a re-implementation of the function under test. If you
cannot get an expected value from NumPy, compute it by hand and write it in the test.
Run the parity tests only after `python tools/gen_fixtures.py` has created the fixture files (they are not committed).
