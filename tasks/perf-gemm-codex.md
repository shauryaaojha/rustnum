You are working in C:/Users/shaur/Documents/mlrs-perf-gemm (git worktree of the rustnum repo, branch `perf-gemm`).
cargo: C:/Users/shaur/.cargo/bin/cargo.exe (GNU toolchain, Rust 1.99). Python 3.13 + NumPy 2.3.2 (OpenBLAS 0.3.30): `python`.
Keep `.cargo/config.toml` (jobs = 2): the laptop has 16 GB RAM and two other agents build in parallel.
Never run more than one cargo command at a time. No git commit/reset/checkout. Do not spawn subagents.

# Task: rustnum's own GEMM (performance plan phase P1)

Read first: AGENTS.md, docs/rustnum/PERF_PLAN.md (sections 1–4; you own G1 and the P1 phase),
rustnum/src/linalg/kernels.rs (`gemm_f32`, `gemm_f64`, `tiled_gemm`, `gemm_generic`), rustnum/src/linalg/api.rs
(`matmul`, `dot`, batched dispatch, `raw_parts`), rustnum/tests/linalg.rs, rustnum/benches/core.rs.

Target machine: Intel Core Ultra 5 125H (hybrid: 4 P-cores with HT, 8 E-cores, 2 LP E-cores; AVX2 + FMA, no AVX-512;
P-core L1d 48 KB, L2 2 MB, shared L3 18 MB). Today large f64 matmul = matrixmultiply single-threaded per 256×256 tile,
spread over rayon: ~140 GF/s at 8192², ≈ OpenBLAS.

## Deliver
1. New module rustnum/src/linalg/gemm/ (mod.rs + submodules) implementing BLIS-style GEMM for f64 and f32:
   - packing of A into MR-row panels and B into NR-column panels (KC deep), handling any input strides (transposed and
     sliced views, negative strides) without materialising a contiguous copy first;
   - cache blocking MC/KC/NC chosen for the machine above, kept in one table (`blocking.rs`) with a comment per value;
   - micro-kernels with `std::arch` AVX2+FMA intrinsics: f64 6×8 (12 ymm accumulators), f32 6×16; masked/edge handling
     for partial tiles; a portable scalar fallback; runtime dispatch with `is_x86_feature_detected!` cached in a static;
     an aarch64 NEON kernel if time allows (otherwise scalar fallback there);
   - parallelism as rayon tasks over (NC block × MC block) pairs with packed B shared per NC block, so work stealing gives
     fast cores more blocks; small problems stay single-threaded (threshold measured, not guessed);
   - an epilogue: `C = alpha·A·B + beta·C`, then an optional element-wise `fn(f64) -> f64` / `fn(f32) -> f32` applied
     to each finished C tile while it is in cache.
   Every `unsafe` block has a `// SAFETY:` comment.
2. Replace the matrixmultiply calls behind `gemm_f32` / `gemm_f64` with the new GEMM; remove the `matrixmultiply`
   dependency from rustnum/Cargo.toml (and `tiled_gemm` if it becomes unused). The generic integer path stays.
3. Public API additions (SciPy BLAS naming), documented with runnable doctests:
   - `np::linalg::gemm(alpha, &a, &b, beta, &c) -> Array<F>` (like `scipy.linalg.blas.dgemm`),
   - `a.matmul_into(&b, &mut out)` writing into an existing contiguous output of the right shape (no allocation),
   - `np::linalg::matmul_map(&a, &b, alpha, f)` returning `f(alpha · a @ b)` computed in the GEMM epilogue
     (this is the fused path for `tanh(a @ b / scale)`).
4. Tests (keep every existing test passing): random shapes incl. 1×N, N×1, k = 0, primes (257, 509, 1031), transposed and
   sliced views for A, B and C, alpha/beta combinations, f32 and f64, epilogue correctness, comparison against the naive
   loop already in rustnum/tests/linalg.rs; a proptest over shapes and strides; NumPy parity must stay 93/93
   (`python tools/gen_fixtures.py` then `cargo test --test parity -- --nocapture`).
5. Benchmarks: add to rustnum/benches/core.rs `matmul (2048x2048 f64)`, `matmul (4096x4096 f64)`, f32 rows for 1024 and
   4096, `matmul_map tanh (2048x2048 f64)`; add the same names to tools/bench_numpy.py. You may run short criterion
   checks for your own tuning (other agents may be using the CPU, so treat numbers as rough); do NOT run the
   heavy_matmul example (the owner runs it).

## Acceptance (all must pass)
cargo fmt --all; cargo clippy --all-targets --all-features -- -D warnings; cargo test --all-features;
cargo test -p rustnum --no-default-features; cargo doc --no-deps; cargo bench -p rustnum --no-run

## Don't
Do not edit array.rs, layout.rs, ops.rs, ufunc.rs, reduce.rs, broadcast.rs, manip.rs, random/, or any non-GEMM part of
linalg/kernels.rs (other agents own them). No new dependencies.

## Finish
Concise summary: files changed, design (blocking values, kernels, parallel scheme), test counts per gate, the rough
GF/s you measured at 1024/2048/4096 for f64 and f32 single-thread and all-threads, open doubts.
