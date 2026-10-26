You are working in C:/Users/shaur/Documents/mlrs-perf-misc (git worktree of the rustnum repo, branch `perf-misc`).
cargo: C:/Users/shaur/.cargo/bin/cargo.exe (GNU toolchain, Rust 1.99). Python 3.13 + NumPy 2.3.2: `python`.
Keep `.cargo/config.toml` (jobs = 2): the laptop has 16 GB RAM and two other agents build in parallel.
Never run more than one cargo command at a time. No git commit/reset/checkout. Do not spawn subagents.

# Task: SIMD math, fast random numbers and fast sorting for rustnum

Read first: AGENTS.md, docs/rustnum/PERF_PLAN.md (you own G5, G6 and G10), rustnum/src/ufunc.rs (read only),
rustnum/src/random/engine.rs and api.rs, rustnum/src/manip.rs (sort, argsort, unique), rustnum/src/element.rs.
Target CPU: Intel Core Ultra 5 125H, AVX2 + FMA, no AVX-512.

## Deliver
1. **SIMD transcendental kernels** in a new private module rustnum/src/simd_math.rs (declare `mod simd_math;` in
   lib.rs, only that line). Slice-to-slice functions `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sin`,
   `cos`, `tan`, `tanh`, `sigmoid`, for f64 and f32: `fn exp_f64(src: &[f64], dst: &mut [f64])` and an in-place form.
   AVX2+FMA implementations with `std::arch` (range reduction + minimax polynomials, correct handling of NaN, ±inf,
   ±0, subnormals and overflow), runtime dispatch cached in a static, scalar fallback. Accuracy: ≤ 1 ulp for exp/log
   families, ≤ 2 ulp for trig/tanh, measured against the `f64` std functions on 10⁷ random inputs over the full range
   plus edge cases; put the measured max ulp in each function's doc comment. Every `unsafe` block has `// SAFETY:`.
   Do NOT edit ufunc.rs (another agent is changing it); the orchestrator wires your kernels into the ufuncs after the
   merge. Write a short wiring note at the top of simd_math.rs.
2. **Random numbers:** replace Box-Muller in random/engine.rs with the ziggurat method for normal and exponential
   (NumPy uses ziggurat); keep `Rng` public API unchanged. Add parallel deterministic filling for large outputs
   (≥ 1<<16 elements): split the output into fixed-size chunks (independent of thread count), give chunk i the stream
   `seed` advanced by i `jump()`s of xoshiro256**, fill chunks in parallel with rayon (behind the `rayon` feature; the
   serial path must produce the identical sequence). Update the seed-42 regression values deliberately and document
   the stream change in CHANGELOG.md. Keep the statistical tests (mean/std/KS-style checks) passing.
   Exit: `standard_normal(&[4096, 4096])` ≤ 290 ms single-threaded and ≤ 60 ms with all threads.
3. **Sorting (manip.rs sort, argsort, unique only):** move NaNs to the end in one pass, then sort the rest with
   `sort_unstable_by(total_cmp)` (stable sort for `argsort` to keep NumPy's stable order), parallel (`par_sort`) for
   large lanes behind the `rayon` feature, LSD radix sort (bit-flipped keys) for f64/f32/i64/i32 lanes ≥ 1<<16, and
   argsort via (key, index) pairs. NaN handling and order must match NumPy exactly (parity).
   Exit: `sort (1e6 f64)` ≤ 12 ms.
4. Tests: every existing test passes; NumPy parity stays 93/93 (`python tools/gen_fixtures.py`, then
   `cargo test --test parity -- --nocapture`); ulp tests for every SIMD function (both dispatch paths: force the scalar
   path in a test via a crate-private switch); sort/argsort equality against the old implementation on random data with
   NaN/±0/±inf; parallel RNG == serial RNG for the same seed.
5. Benchmarks: add `exp (1e6 f64) simd kernel`, `tanh (1e6 f64) simd kernel`, `standard_normal (4096x4096)`,
   `sort (1e7 f64)` to rustnum/benches/core.rs and the matching names to tools/bench_numpy.py. Short criterion checks
   are fine; other agents may be using the CPU, so treat numbers as rough. Do NOT run the heavy_matmul example.

## Acceptance (all must pass)
cargo fmt --all; cargo clippy --all-targets --all-features -- -D warnings; cargo test --all-features;
cargo test -p rustnum --no-default-features; cargo doc --no-deps; cargo bench -p rustnum --no-run

## Don't
Do not edit array.rs, layout.rs, ops.rs, ufunc.rs, reduce.rs, broadcast.rs, rustnum/src/linalg/, or manip.rs outside
sort/argsort/unique. No new dependencies. Do not change public signatures.

## Finish
Concise summary: files changed, max ulp per function, rough timings vs the old code, test counts per gate, deviations.
