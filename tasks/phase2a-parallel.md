# PARALLEL MODE — read this first, it overrides the "Read first" assumption below
You are working in C:/Users/shaur/Documents/mlrs-linalg (a git worktree, branch `linalg`). Pass all cargo commands there.
Phase 1 (the core Array type) is being written RIGHT NOW by another agent in a different worktree, so in your
worktree the core modules are still stubs. Work in two steps:

STEP A (do now, must compile and pass tests here):
- Create rustnum/src/linalg/ as a module directory: rustnum/src/linalg/mod.rs (replace linalg.rs) with a private
  `pub(crate) mod kernels;` (rustnum/src/linalg/kernels.rs). kernels.rs is pure slice-level code that does NOT use Array:
  row-major `&[f64]`/`&[f32]` + dims (+ strides where useful). Implement and unit-test there: gemm wrapper over
  matrixmultiply (f32, f64, arbitrary row/col strides), generic gemm for any `T: Copy + Add + Mul + Default`,
  LU with partial pivoting (det, solve, inv), Cholesky, Householder QR (reduced), lstsq via QR, cyclic Jacobi eigh,
  one-sided Jacobi SVD (+ sign rule), pinv, matrix_rank, norms. Make the kernels generic over a small private float
  trait implemented for f32 and f64 so they can be wired to `rustnum::Float` later.
  Tests: ground-truth identities from the section "Tests" below, on raw slices.
- rustnum/benches/core.rs: write the criterion benchmarks against the PUBLIC API described in docs/API.md, but keep
  the file compiling now: put the Array-based benches behind `#[cfg(any())]`-free approach is NOT possible, so write
  them into rustnum/benches/core.rs.pending (not compiled) and keep core.rs a kernel-only bench (gemm 256/512/1024,
  inv 200, svd 100) that compiles now.
- tools/bench_numpy.py: complete (independent of Rust).
- `cargo test -p rustnum` and `cargo clippy -p rustnum --all-targets -- -D warnings` must pass in your worktree.

STEP B (wrappers, written now but NOT compiled yet):
- Write rustnum/src/linalg/api.rs.pending containing the full public `np::linalg` functions and the `matmul`/`dot`/`trace`
  methods on Array exactly per docs/API.md §11, calling your kernels. It is NOT declared in mod.rs yet. Use only the
  Array API listed in docs/API.md §2–§3 (from_vec, shape, ndim, size, strides, as_slice, is_contiguous, to_contiguous,
  to_vec, broadcast_to, IntoArray, Error, Result). The orchestrator will merge the core and ask you to wire it.

Finish with the summary described at the bottom.

You are working in a git worktree of the rustnum repo (path given by --cd). Rust stable
(x86_64-pc-windows-gnu) is installed; cargo is at C:/Users/shaur/.cargo/bin/cargo.exe.

# Task: Phase 2a of rustnum — linear algebra + performance + benchmarks

## Read first
1. AGENTS.md, docs/API.md (§11 and §14 are your job), docs/PLAN.md.
2. rustnum/src/lib.rs and the core modules (array.rs, layout.rs, broadcast.rs, element.rs, ops.rs, reduce.rs,
   manip.rs) — Phase 1 is finished. Reuse its internals (iterators, contiguous fast paths, IntoArray, Error).

## Deliver
1. rustnum/src/linalg.rs — everything in API.md §11:
   - Methods on Array: `matmul` (1-d/2-d/batched n-d with broadcast batch dims), `dot`, `trace`.
     f32/f64 dispatch to `matrixmultiply::sgemm/dgemm` passing real strides (no copy for transposed views);
     other Num types use a cache-friendly generic i-k-j loop. Dispatch without specialization: e.g. a private
     `Num` hook or `TypeId` check + safe transmute-free casting through `as_slice` of a contiguous copy.
   - `np::linalg`: outer, inner, det (LU with partial pivoting), inv, solve (LU), lstsq (QR), pinv (SVD),
     norm / norm_axis (ord: Frobenius/2, 1, inf for vectors), cholesky, qr (Householder, reduced),
     eigh (cyclic Jacobi, ascending, eigenvectors as columns, deterministic sign: largest-|component| positive),
     svd (one-sided Jacobi, reduced, S descending, same sign rule on U columns), matrix_rank (SVD, NumPy tolerance),
     matrix_power.
   - Errors: Error::Singular, Error::NotPositiveDefinite, Error::Shape(..) as specified.
2. Performance (§14): make sure the rayon paths exist for large contiguous element-wise ops and reductions;
   do not regress correctness. Parallel matmul is matrixmultiply's job (enable its "threading" feature only
   through our `rayon` feature: `rayon = ["dep:rayon", "matrixmultiply/threading"]` in rustnum/Cargo.toml).
3. rustnum/benches/core.rs — criterion benchmarks: add (1e6 f64, contiguous), add with broadcasting
   ([1000,1000] + [1000]), sum (1e7 f64), mean_axis(0) on [1000,1000], transpose().to_contiguous() [1000,1000],
   matmul 256/512/1024 f64, exp on 1e6 f64, sort 1e6 f64, inv 200x200, svd 100x100.
4. tools/bench_numpy.py — the same benchmarks with NumPy + timeit (median of N), printing a markdown table
   with the same names, so results can be compared side by side.
5. Tests in linalg.rs and rustnum/tests/linalg.rs: identities that are ground truth
   (A·inv(A) ≈ I, Q orthonormal and QR = A, U·diag(S)·Vt = A, eigh: A·v = λ·v, cholesky L·Lᵀ = A,
   solve residual ≈ 0, det of known matrices, hand-computed small matmul results, batched matmul shapes,
   singular matrix → Err(Singular)). f64 tolerance 1e-10 relative to scale.
6. Doc comment + runnable example on every public item.

## Acceptance (all must pass)
```
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test -p rustnum --no-default-features
cargo bench -p rustnum --no-run
```

## Don't
- Do not touch random.rs, io.rs, tools/gen_fixtures.py or anything under tests/parity* — another agent owns them.
- Do not change public signatures of Phase 1 items. Fix a Phase 1 bug only if it blocks you; list it in the summary.
- No new dependencies other than enabling matrixmultiply's `threading` feature. No git commit/reset/checkout.

## Finish
Concise summary: files written, acceptance command results (test counts), any deviations from API.md, open doubts.
