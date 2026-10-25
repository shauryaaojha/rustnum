Continue Phase 1. Your summary says the foundation compiles, but most of docs/API.md §0–§10 and §14 is still missing.
The job is not done until EVERY item below exists, is tested and documented. Do not stop early; work through the list
in order and keep running the acceptance gates as you go.

## Spec clarifications (these are decisions, apply them)
1. `abs`, `square`, `clip(lo, hi)`, `sign` are defined ONCE on `Array<T: Num>` (not again on Float). Unsigned `abs` = identity.
2. `s!` must accept literals AND runtime expressions of type usize / isize / i32 / i64: `s![i..i + 5, .., j]`, `s![..;2]`,
   `s![1..;-1]`, `s![-3..]`, `s![.., np::NEW_AXIS]` (NEW_AXIS inserts a length-1 axis).
3. `reshape(&[usize])` and `reshape_infer(&[isize])` (one -1 allowed).
4. Both arguments of `where_`, `maximum`, `minimum`, `power`, `arctan2`, `hypot` and every comparison take `impl IntoArray<T>`.
   `matmul`/`dot` (not yours) will take `&Array<T>`. `vstack`/`hstack`/`concatenate`/`stack` take `&[&Array<T>]`.
   `array_split(&a, sections: usize, axis: isize)`, `searchsorted(&sorted, v: T) -> usize` (side=left),
   `pad(&a, width: usize, value: T)`. Every axis parameter is `isize` (negative allowed), including swapaxes/moveaxis.
5. `percentile(q: F)` with q in 0..=100 (linear interpolation, NumPy default); `quantile` is another agent's job.
6. `is_empty()` on 0-d is false. `get` returns `T` and panics when out of bounds.
7. Add `a.compress(&mask_1d: &Array<bool>, axis: isize)` (np.compress) so rows can be selected with a 1-d mask.
8. Array methods are `is_nan`, `is_inf`, `is_finite` (-> Array<bool>).
9. Add axis forms: `nansum_axis nanmean_axis nanmax_axis nanmin_axis var_ddof_axis std_ddof_axis`.
10. `round()` must round half to EVEN like NumPy (`f64::round_ties_even`), and add `round_decimals(d: i32)` (np.round(a, d)).
11. `T op Array` for each concrete Num type, and `&a * 2.0` / `2.0 * &a` must both compile for `Array<f64>`, `&a * 2.0f32` for f32.

## Checklist (each must exist, with a doc example that runs)
- §2 every method and trait in the table (from_shape_fn, item, to_contiguous, astype, map, mapv_inplace, zip_map,
  fill, copy, Index/IndexMut with [usize; N] and &[usize], FromIterator, IntoIterator for &Array, Debug/Display, PartialEq, Default).
- §3 IntoArray for T, Array, &Array, Vec, &[T], [T; N]; broadcast_shapes, broadcast_to, broadcast_arrays.
- §4 all creation fns incl. logspace, identity, diag (both directions), tri, tril, triu, meshgrid; array! for 1/2/3-d with ragged panic.
- §5 slice, slice_assign (broadcast), index_axis, row, col, mask, mask_assign, take, compress.
- §6 all operator combos via macros, compound ops, comparisons, bool & | ^ !, logical_*, allclose, isclose, array_equal.
- §7 every listed ufunc + sigmoid/relu + maximum/minimum/power/arctan2/hypot/where_ with NaN propagation.
- §8 every reduction, axis form, ddof form, nan form, median, percentile, ptp, count_nonzero, cumsum/cumprod (flat and axis),
  NumPy panic wording for empty min/max/argmin/argmax. Pairwise summation for float sum/mean.
- §9 every manipulation method and free fn listed (reshape view vs copy, ravel, flatten, t, transpose, permute_axes, swapaxes,
  moveaxis, squeeze, squeeze_axis, expand_dims, flip, flip_all, roll, repeat, tile, sort (NaN last), sort_axis, argsort (stable),
  argsort_axis, concatenate, stack, vstack, hstack, split, array_split, unique, unique_counts, nonzero, argwhere, searchsorted, pad).
- §10 NumPy-exact Display for ints, floats (precision 8, trimmed zeros, aligned, `1.` style), bools (`[ True False]`), 0-d, n-d with
  blank lines between 2-d blocks for 3-d, summarisation with `...` above 1000 elements. Debug as `array(..., dtype=float64)`.
- §14 rayon fast paths for contiguous element-wise ops and sum/prod/min/max at size >= 1<<15 (feature-gated, results identical
  for ints), allocation-free odometer for strided iteration.

## Tests (minimum bar)
At least 150 tests total (unit + integration + doctests), including: every panic message, every reduction on each axis
of a [2,3,4] array with values from np.arange(24) (hand-write NumPy's results), slicing with negative steps, COW isolation,
reshape of transposed arrays, Display strings copied from real NumPy output (you may run `python -c "import numpy as np; ..."`
to get exact expected strings), broadcasting shape errors, and at least 5 proptest properties.

## Acceptance (all must pass at the end)
cargo fmt --all; cargo clippy --all-targets --all-features -- -D warnings; cargo test --all-features;
cargo test -p rustnum --no-default-features; cargo doc --no-deps

Same "Don't" rules as before (no linalg/random/io, no new deps, no git commit/reset/checkout).
Finish with: per-section checklist status (done / not done), exact test counts per command, deviations.
