# rustnum API addendum: statistics, signal, sets, einsum

Same rules as `docs/API.md`: names follow NumPy, misuse panics with NumPy's wording, one path per
item (everything below is re-exported at the crate root, `np::cov`, `np::einsum`, ...). Code lives in
`rustnum/src/extra/` (pure slice kernels, tested against NumPy) and `extra/api.rs` (the `Array`
wrappers).

Conventions specific to this addendum:
- Main array arguments are `&Array<T>` (like `np::unique(&a)`), so `T` is inferred without turbofish.
- Statistics functions return `f64` / `Array<f64>` for any element type (NumPy: float64 for ints).
  `T` is preserved by `convolve`, `correlate`, `diff`, `cross`, `kron`, `einsum` and the set functions.
- Options that are keyword arguments in NumPy are plain trailing parameters (`Option<_>` for "None").
  Axis is `isize`, negative counts from the end. Whole-array results are scalars (`f64`), per-axis results
  are arrays with that axis removed, with the `_axis` suffix (API.md section 8).
- NaN sorts last everywhere; NaN != NaN.

## 15. Statistics (`extra/stats_kernels.rs`)
```rust
pub enum Bins { Count(usize), Edges(Vec<f64>) }   // From<usize>, From<Vec<f64>>, From<&[f64]>, From<&Array<f64>>
```
| Function | NumPy | Notes |
|---|---|---|
| `cov(m: &Array<T>, bias: bool, ddof: Option<usize>) -> Array<f64>` | `np.cov(m, bias=, ddof=)` | rows are variables; 2-d `[nvars, nobs]` -> `[nvars, nvars]`, 1-d -> 0-d. `ddof` default 1 (0 if `bias`), explicit `ddof` wins. `nobs - ddof <= 0` gives inf/NaN. No `y`, `rowvar`, weights. |
| `corrcoef(x: &Array<T>) -> Array<f64>` | `np.corrcoef(x)` | clipped to `[-1, 1]`, constant row -> NaN |
| `histogram(a, bins: impl Into<Bins>, range: Option<(f64, f64)>, density: bool, weights: Option<&Array<f64>>) -> (Array<f64>, Array<f64>)` | `np.histogram` | `(counts, edges)`. Last bin closed on the right; NaN / out-of-range ignored. Counts are `f64`. No string bin rules (`"auto"`, ...). Empty data -> range `(0, 1)`; `first == last` -> `+-0.5`. Equal-width path replicates NumPy's index correction, so values on edges land in the same bin as NumPy |
| `histogram_bin_edges(a, bins: impl Into<Bins>, range) -> Array<f64>` | `np.histogram_bin_edges` | |
| `bincount(x: &Array<i64>, weights: Option<&Array<f64>>, minlength: usize) -> Array<f64>` | `np.bincount` | negative input panics; counts are `f64` |
| `digitize(x: &Array<T>, bins: &Array<B>, right: bool) -> Array<usize>` | `np.digitize` | increasing or decreasing `bins` (all-equal counts as increasing); NaN -> `bins.len()`; output has `x`'s shape |
| `average(a, weights: Option<&Array<f64>>) -> f64` | `np.average` | weights same size as `a`; zero sum panics `Weights sum to zero, can't be normalized`. No `returned` |
| `average_axis(a, weights, axis) -> Array<f64>` | `np.average(axis=)` | weights: shape of `a` or 1-d of length `shape[axis]` |
| `quantile(a, q: f64) -> f64`, `quantile_axis(a, q, axis) -> Array<f64>` | `np.quantile` | method `"linear"` only; `q` scalar in `[0, 1]`; NaN in the data -> NaN; reproduces NumPy's `inf` behaviour (`quantile([1, inf], .5)` is NaN); empty input panics |
| `nanquantile(a, q) -> f64`, `nanquantile_axis(a, q, axis)` | `np.nanquantile` | NaNs dropped; all-NaN / empty -> NaN |

## 16. Signal and numeric (`extra/signal_kernels.rs`)
```rust
pub enum ConvolveMode { Full, Same, Valid }       // Default = Full
pub enum Spacing { Uniform(f64), Coords(Vec<f64>) } // From<f64>, From<Vec<f64>>, From<&Array<f64>>
```
| Function | NumPy | Notes |
|---|---|---|
| `convolve(a: &Array<T>, v: &Array<T>, mode: ConvolveMode) -> Array<T>` | `np.convolve` | 1-d, `T: Num`. Longer `v` is swapped with `a` like NumPy; `Same` is centred with offset `(m - 1) / 2` of `Full`; empty input panics `a cannot be empty` |
| `correlate(a, v, mode) -> Array<T>` | `np.correlate` | `c[k] = sum a[n+k] v[n]`; longer `v` -> swap and reverse (NumPy semantics). No default mode (NumPy's default is `"valid"`) |
| `diff(a, n: usize, axis: isize) -> Array<T>` | `np.diff` | axis shrinks by `n` to at most 0; `n == 0` copies. No `prepend` / `append`. Unsigned types underflow-panic in debug |
| `gradient(f: &Array<F>, dx: F) -> Vec<Array<F>>` | `np.gradient(f, dx)` | one array per axis (a 1-element `Vec` for 1-d). Central differences inside, first-order at the edges |
| `gradient_axis(f, spacing: impl Into<Spacing>, axis, edge_order: usize) -> Array<F>` | `np.gradient(f, x, axis=, edge_order=)` | `edge_order` 1 or 2; coordinate spacing uses NumPy's non-uniform formulas; needs `edge_order + 1` samples |
| `interp(x, xp, fp: &Array<f64>, left: Option<f64>, right: Option<f64>) -> Array<f64>` | `np.interp` | below `xp[0]` -> `left` (default `fp[0]`), above the last -> `right` (default `fp[-1]`); NaN -> NaN (except single-point `xp`, as NumPy); `xp` increasing is not checked; no `period` |
| `trapezoid(y: &Array<F>, spacing: impl Into<Spacing>, axis) -> Array<F>` | `np.trapezoid` | axis removed (0-d for 1-d `y`, use `.item()`); < 2 samples -> 0 |
| `cross(a, b: &Array<T>) -> Array<T>` | `np.cross` | last axis 2 or 3, leading axes broadcast; two 2-vectors give the z component only (axis disappears); no `axisa/b/c` |
| `kron(a, b) -> Array<T>` | `np.kron` | n-d, shorter shape padded with leading 1s |

`outer` is `np::linalg::outer` (not here).

## 17. Sets (`extra/set_kernels.rs`)
All take `&Array<T: Element>`, flatten their inputs, and return sorted 1-d arrays without duplicates
(except `isin`). No `assume_unique`, `return_indices`, `invert`.

| Function | NumPy | NaN behaviour (same as NumPy) |
|---|---|---|
| `intersect1d(a, b) -> Array<T>` | `np.intersect1d` | never contains NaN |
| `union1d(a, b)` | `np.union1d` | one NaN if any input has NaN |
| `setdiff1d(a, b)` | `np.setdiff1d` | a NaN of `a` is kept (never "in" `b`) |
| `setxor1d(a, b)` | `np.setxor1d` | a NaN present in both inputs appears twice |
| `isin(element, test_elements) -> Array<bool>` | `np.isin` | shape of `element`; NaN is never found |

## 18. einsum (`extra/einsum_kernels.rs`)
`einsum<T: Num>(subscripts: &str, operands: &[&Array<T>]) -> Array<T>` (`np.einsum`).

Supported:
- explicit output (`"ij,jk->ik"`, `"i,i->"`, `"ij->ji"`) and implicit output (`"ij,jk"`: labels that occur
  exactly once, in ASCII order, so `"ji"` is a transpose);
- repeated labels in one operand: diagonal and trace (`"ii->i"`, `"ii"`, `"bii->b"`, `"ijj->i"`);
- summed labels, any number of operands (tested with 1 to 3), batch dimensions (`"bij,bjk->bik"`);
- `...` ellipsis, right-aligned and broadcast like NumPy (`"...ij,...jk->...ik"`, `"i...->...i"`); an
  ellipsis axis needs `...` in the explicit output (NumPy's rule);
- size-1 broadcasting of labelled dimensions, spaces, upper and lower case letters, scalar operands (`",->"`);
- strided operands in the kernel (`Operand { data, shape, strides, offset }`); the `Array` wrapper passes
  contiguous data.

Not supported: `optimize=` / `einsum_path`, `out=`, `dtype=`, the interleaved sublist form
`einsum(a, [0, 1], b, [1, 2], [0, 2])`. Cost is the product of all label sizes times the operand count (a plain
nested loop, no pairwise contraction ordering), so a long chain such as `"ab,bc,cd,de->ae"` is slower than
chained `matmul`s.

Panics (NumPy wording): `invalid subscript 'x' in einstein sum subscripts string, subscripts must be letters`,
`more operands provided to einstein sum function than specified in the subscripts string`,
`einstein sum subscripts string contains too many subscripts for operand 0`,
`operand has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions.`,
`einstein sum subscripts string included output subscript 'z' which never appeared in an input`,
`einstein sum subscripts string includes output subscript 'i' multiple times`,
`dimensions in operand 0 for collapsing index 'i' don't match (3 != 4)`,
`operands could not be broadcast together with remapped shapes [original->remapped]` (short form).

## 19. Not done (honest list)
String bin rules for `histogram`; array-valued `q` and non-linear methods for `quantile`; `cov` with `y`,
`rowvar`, `fweights`, `aweights`; `average(returned=True)`; `gradient` with a list of per-axis coordinate arrays;
`interp(period=)`; `diff(prepend=, append=)`; `cross(axisa=, axisb=, axisc=)`; complex inputs.
