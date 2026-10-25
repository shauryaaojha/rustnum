# rustnum v0.1 API contract

This is the binding spec for every module. Names follow NumPy. When this file and NumPy
disagree, this file wins; when this file is silent, do what NumPy does.

Convention used below: `a`, `b` are `Array<T>`; `shape` is `&[usize]`; `axis` is `isize`
(negative counts from the end, `-1` = last axis). Panics use NumPy's wording where one
exists, e.g. `operands could not be broadcast together with shapes [2, 3] [4]`.

## 0. Crate surface
- Users write `use rustnum as np;`. Core items are at the crate root (`np::Array`, `np::zeros`).
- Public modules: `np::linalg`, `np::random`, `np::io`. Everything else is a private module
  re-exported from `lib.rs` exactly once (one path per item).
- Macros at the root: `np::array!`, `np::s!`.
- `np::Error` (enum) + `np::Result<T>` for fallible operations (I/O, parsing, linalg on
  singular / non-positive-definite input). Everything else panics on misuse.
- No lifetimes in any public signature. No `unsafe` unless measured necessary and commented.

## 1. Element types (`element.rs`)
```rust
pub trait Element: Copy + Clone + Send + Sync + Debug + Default + PartialEq + PartialOrd + 'static {
    const DTYPE: &'static str;            // "float64", "float32", "int64", "int32", "uint8", "bool", "uint64"(usize)
    fn to_f64(self) -> f64;               // bool -> 0.0/1.0
    fn from_f64(v: f64) -> Self;          // truncating cast like NumPy astype (bool: v != 0)
}
pub trait Num: Element + Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> + Rem<Output=Self> {
    const ZERO: Self; const ONE: Self; const MIN_VALUE: Self; const MAX_VALUE: Self;
    fn abs(self) -> Self; fn from_usize(n: usize) -> Self;
}
pub trait Float: Num + Neg<Output=Self> {
    const NAN: Self; const INFINITY: Self; const NEG_INFINITY: Self; const EPSILON: Self; const PI: Self; const E: Self;
    // exp ln log2 log10 sqrt cbrt sin cos tan asin acos atan sinh cosh tanh exp_m1 ln_1p
    // floor ceil round trunc powf powi is_nan is_infinite is_finite atan2 hypot max min (NaN-propagating where NumPy does)
}
```
Implementations: `Element` for `f32 f64 i32 i64 u8 usize bool`; `Num` for `f32 f64 i32 i64 u8 usize`;
`Float` for `f32 f64`. Signed-only helpers may use a private `Signed` trait.

## 2. The array (`array.rs`, `layout.rs`)
```rust
pub struct Array<T> { data: Arc<Vec<T>>, shape: Vec<usize>, strides: Vec<isize>, offset: usize }
```
- Strides are in elements. C-order (row-major) by default. `[]` shape = 0-d array (holds one value).
- Value semantics with copy-on-write: every mutating method calls `Arc::make_mut` first. A view
  never writes into another array. (Documented difference from NumPy.)
- Traits: `Clone` (cheap), `Debug` (`array([[1., 2.]], dtype=float64)`), `Display` (NumPy print),
  `PartialEq` (same shape and all elements equal), `Default` (empty 1-d), `From<Vec<T>>` (1-d),
  `From<T>`? NO (ambiguous) — use `Array::scalar(v)`, `FromIterator<T>` (1-d), `Index<[usize; N]>` and
  `IndexMut<[usize; N]>` for any const N, `Index<&[usize]>`/`IndexMut<&[usize]>`, `IntoIterator for &Array<T>`
  (yields `T` in C order), `Send + Sync`.

Methods (all on `Array<T: Element>` unless noted):
| Method | NumPy | Notes |
|---|---|---|
| `Array::from_vec(data: Vec<T>, shape: &[usize]) -> Array<T>` | `np.array(...).reshape` | panics if `len != product(shape)` |
| `Array::from_shape_fn(shape, f: impl FnMut(&[usize]) -> T)` | `np.fromfunction` | |
| `Array::scalar(v: T)` | `np.array(v)` | 0-d |
| `shape() -> &[usize]`, `ndim() -> usize`, `size() -> usize`, `len() -> usize` (first axis, panics on 0-d), `is_empty()` | `.shape .ndim .size len()` | |
| `strides() -> &[isize]`, `dtype() -> &'static str` | `.strides .dtype` | strides in elements |
| `is_contiguous() -> bool` | `.flags.c_contiguous` | |
| `get(index: &[usize]) -> T`, `set(index: &[usize], v: T)` | `a[i, j]`, `a[i, j] = v` | panics: `index 5 is out of bounds for axis 0 with size 3` |
| `item() -> T` | `.item()` | panics unless size == 1 |
| `to_vec() -> Vec<T>` | `.ravel().tolist()` | C order |
| `as_slice() -> Option<&[T]>` | | `Some` only when contiguous |
| `iter() -> impl Iterator<Item = T> + '_` | `.flat` | C order |
| `to_contiguous() -> Array<T>` | `np.ascontiguousarray` | no copy if already contiguous |
| `astype<U: Element>() -> Array<U>` | `.astype` | via to_f64/from_f64, except identical types |
| `map<U: Element>(f: impl Fn(T) -> U) -> Array<U>` | `np.vectorize` | |
| `mapv_inplace(f: impl Fn(T) -> T)` | | |
| `zip_map<U, R>(other: impl IntoArray<U>, f: impl Fn(T, U) -> R) -> Array<R>` | binary ufunc | broadcasts |
| `fill(v: T)` | `.fill` | |
| `copy() -> Array<T>` | `.copy()` | deep, contiguous |

## 3. Scalars anywhere (`broadcast.rs`)
```rust
pub trait IntoArray<T: Element> { fn into_array(self) -> Array<T>; }
```
Implemented for `T` (0-d), `Array<T>`, `&Array<T>` (cheap clone), `Vec<T>` (1-d), `&[T]` (1-d), `[T; N]` (1-d).
Every function/method below that takes "array-like" takes `impl IntoArray<T>`.

Broadcasting (`broadcast.rs`): `broadcast_shapes(&[usize], &[usize]) -> Option<Vec<usize>>` (pub),
`a.broadcast_to(shape) -> Array<T>` (view, zero strides; panics on incompatible),
`np::broadcast_arrays(&a, &b) -> (Array<T>, Array<U>)`.

## 4. Creation (`creation.rs`) — free functions at the root
`zeros::<T>(shape)`, `ones::<T>(shape)`, `full(shape, v)`, `zeros_like(&a)`, `ones_like(&a)`, `full_like(&a, v)`,
`arange(start: T, stop: T, step: T) -> Array<T>` (T: Num; panics on step 0; float length = ceil((stop-start)/step)),
`linspace(start: F, stop: F, num: usize) -> Array<F>` (endpoint included), `logspace(start, stop, num)` (base 10),
`eye::<T>(n)`, `identity::<T>(n)`, `diag(v: &Array<T>) -> Array<T>` (1-d → 2-d diagonal matrix, 2-d → 1-d diagonal),
`tri::<T>(n)`, `tril(&a)`, `triu(&a)`, `meshgrid(&x, &y) -> (Array<T>, Array<T>)` (indexing='xy').
`array!` macro: `array![1, 2, 3]`, `array![[1.0, 2.0], [3.0, 4.0]]`, 3-d nesting too. Panics on ragged input.

## 5. Indexing and slicing (`slicing.rs`)
- `s![...]` builds `Vec<SliceArg>`: `a..b`, `a..`, `..b`, `..`, `a..=b`, a single index (removes the axis),
  steps with `;` (`s![..;2, 1..;-1]` -- NumPy `[::2, 1::-1]`), negative values count from the end, `NewAxis` via `np::NEW_AXIS`.
  Integer literals must work without suffixes: `s![1..3, .., -1]`.
- `a.slice(args: &[SliceArg]) -> Array<T>` (view, O(1)), `a.slice_assign(args, values: impl IntoArray<T>)` (in place, broadcasts).
- Axis helpers: `a.index_axis(axis, i) -> Array<T>` (drops axis), `a.row(i)`, `a.col(j)` (2-d).
- Boolean mask: `a.mask(&m: &Array<bool>) -> Array<T>` (1-d result, like `a[m]`; m must match shape),
  `a.mask_assign(&m, v: T)` (`a[m] = v`).
- Integer take: `a.take(indices: &[usize], axis) -> Array<T>` (`np.take`).

## 6. Operators (`ops.rs`)
- `+ - * / %` for all four combinations `Array op Array`, `&Array op &Array`, `Array op &Array`, `&Array op Array`,
  plus `Array op T`, `&Array op T`, and `T op Array`, `T op &Array` for each concrete `Num` type (macro).
  All broadcast. Integer division by zero panics like Rust; floats follow IEEE.
- Unary `-` for signed/float types. `+= -= *= /= %=` with `impl IntoArray<T>`-like RHS (`Array`, `&Array`, `T`), in place, RHS broadcasts to LHS shape.
- Comparisons return `Array<bool>`: `a.eq_elem(b) a.ne_elem(b) a.lt(b) a.le(b) a.gt(b) a.ge(b)`, `b: impl IntoArray<T>`.
- Logic on `Array<bool>`: `& | ^ !` operators, `np::logical_and/or/xor/not`.
- `a.allclose(b, rtol, atol) -> bool`, `np::isclose(a, b, rtol, atol) -> Array<bool>` (NumPy formula, Float only), `a.array_equal(b)`.

## 7. Element-wise math (`ufunc.rs`) — methods on `Array<F: Float>`, return a new array
`exp exp2 expm1 log log2 log10 log1p sqrt cbrt square sin cos tan arcsin arccos arctan sinh cosh tanh
floor ceil round trunc abs sign reciprocal powf(p) powi(n) clip(lo, hi) is_nan is_inf is_finite (-> Array<bool>)
sigmoid relu` (sigmoid/relu are extras, documented as non-NumPy).
For `Num`: `abs`, `square`, `clip`, `sign` also on integers.
Binary free functions (broadcasting): `np::maximum np::minimum np::power np::arctan2 np::hypot`, `np::where_(cond: &Array<bool>, x, y)`
(`where` is a Rust keyword, hence `where_`). NaN handling: `maximum/minimum` propagate NaN like NumPy.

## 8. Reductions (`reduce.rs`)
Whole array → scalar: `sum prod min max mean var std argmin argmax (-> usize, flat index) any all (bool) cumsum cumprod (-> 1-d flat)
count_nonzero median percentile(q) ptp`. `mean var std median percentile` need `Float`. `var/std` have `ddof: usize` variants
`var_ddof(ddof)`, `std_ddof(ddof)` (default 0 like NumPy). Empty `min/max/argmin/argmax` panic: `zero-size array to reduction operation minimum which has no identity`.
Along an axis → array with that axis removed: `sum_axis(axis) prod_axis mean_axis var_axis std_axis min_axis max_axis
argmin_axis argmax_axis (-> Array<usize>) any_axis all_axis cumsum_axis cumprod_axis (keeps shape) median_axis`.
Keep dims: `a.sum_axis(1).expand_dims(1)` (documented pattern; no keepdims flag).
`nan` variants: `nansum nanmean nanmax nanmin` (Float).

## 9. Shape manipulation (`manip.rs`)
Methods: `reshape(shape: &[isize]-like)` — accepts `&[usize]`; plus `reshape_infer(&[isize])` allowing one `-1`;
view when contiguous, copy otherwise; panic `cannot reshape array of size 6 into shape [4, 2]`.
`ravel flatten t() (reverse axes, = .T) transpose(axes: &[usize]) permute_axes(=transpose) swapaxes(a, b) moveaxis(src, dst)
squeeze squeeze_axis(axis) expand_dims(axis) flip(axis) flip_all() roll(shift: isize, axis) repeat(n, axis) tile(reps: &[usize])
sort() (along last axis, floats NaN last) sort_axis(axis) argsort() (last axis -> Array<usize>) argsort_axis(axis)`.
Free functions: `np::concatenate(&[&Array<T>], axis)`, `np::stack(&[&Array<T>], axis)`, `np::vstack`, `np::hstack`,
`np::split(&a, sections: usize, axis) -> Vec<Array<T>>` (equal split, panics otherwise), `np::array_split`,
`np::unique(&a) -> Array<T>` (sorted 1-d), `np::unique_counts(&a) -> (Array<T>, Array<usize>)`,
`np::nonzero(&a) -> Vec<Array<usize>>`, `np::argwhere(&a) -> Array<usize>`, `np::searchsorted(&sorted, v) -> usize`,
`np::clip` NOT here (method only), `np::pad(&a, width: usize, value: T)` (constant mode).

## 10. Printing (`display.rs`)
`Display` matches NumPy layout: `[1 2 3]`, `[1. 2.5 3.]`, `[[1. 2.]\n [3. 4.]]`, bools as `[ True False]`, 0-d as the bare value.
Floats: NumPy's default precision 8, trailing zeros trimmed, aligned columns, summarised with `...` when size > 1000
(first/last 3 per axis). `Debug`: `array([[1., 2.],\n       [3., 4.]], dtype=float64)` style (exact spacing best effort).

## 11. Linear algebra (`linalg.rs`) — `np::linalg::*`, plus `matmul`/`dot` as methods
Methods on `Array<T: Num>`: `a.matmul(&b)` (`@`): 1-d·1-d → 0-d, 2-d @ 2-d, 1-d @ 2-d, 2-d @ 1-d, batched n-d (broadcast batch dims);
`a.dot(&b)` (NumPy `dot` rules for ≤ 2-d), `a.trace()`. f32/f64 use `matrixmultiply::{sgemm,dgemm}` with strides; other types a generic loop.
Free in `np::linalg` (`F: Float`): `outer(&a,&b)`, `inner`, `det(&a) -> F`, `inv(&a) -> Result<Array<F>>`, `solve(&a, &b) -> Result<Array<F>>`,
`lstsq(&a, &b) -> Result<Array<F>>` (solution only), `pinv(&a)`, `norm(&a) -> F` (Frobenius / 2-norm vector), `norm_axis(&a, ord, axis)`,
`cholesky(&a) -> Result<Array<F>>` (lower), `qr(&a) -> (Array<F>, Array<F>)` (reduced, Householder),
`eigh(&a) -> (Array<F>, Array<F>)` (symmetric; ascending eigenvalues, eigenvectors as columns; Jacobi),
`svd(&a) -> (Array<F>, Array<F>, Array<F>)` (reduced: U, S descending, Vt; one-sided Jacobi), `matrix_rank(&a) -> usize`, `matrix_power(&a, n)`.
Errors: `Error::Singular`, `Error::NotPositiveDefinite`, `Error::Shape(String)` for linalg input checks.

## 12. Random (`random.rs`) — `np::random::Rng`
```rust
let mut rng = np::random::Rng::seed(42);   // np.random.default_rng(42)
let mut rng = np::random::Rng::from_entropy();
```
Generator: xoshiro256** seeded via SplitMix64 (no deps). Methods: `random(shape) -> Array<f64>` [0,1),
`uniform(low, high, shape)`, `normal(mean, std, shape)` (Box-Muller or ziggurat), `standard_normal(shape)`,
`integers(low: i64, high: i64, shape) -> Array<i64>` (high exclusive, unbiased), `choice(&a, n, replace: bool)`,
`shuffle(&mut a)` (first axis), `permutation(n) -> Array<usize>`, `binomial(n, p, shape)`, `poisson(lam, shape)`,
`exponential(scale, shape)`, `beta`/`gamma` optional. Same seed → same numbers on every platform (tested).

## 13. I/O (`io.rs`) — `np::io::*`
`save_npy(path, &a) -> Result<()>`, `load_npy::<T>(path) -> Result<Array<T>>` (format v1.0/2.0, little-endian,
C and Fortran order, dtype must match T or return `Error::DType`), `save_csv(path, &a2d, delimiter) -> Result<()>`,
`load_csv::<T>(path, delimiter, skip_header: usize) -> Result<Array<T>>` (`np.loadtxt`). `to_npy_bytes`/`from_npy_bytes` for in-memory.

## 14. Performance rules
- Every element-wise op and reduction has a contiguous fast path on slices.
- With feature `rayon` (default), contiguous ops on arrays with `size >= 1 << 15` run in parallel. Results must be
  identical with and without the feature for integer ops and within float rounding for float sums (use pairwise summation
  in both paths so `sum` of f32 is accurate like NumPy).
- No allocation per element; no `Vec<usize>` index allocation per element in strided iteration (use an odometer).
