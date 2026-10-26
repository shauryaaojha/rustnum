You are working in C:/Users/shaur/Documents/mlrs-perf-mem (git worktree of the rustnum repo, branch `perf-mem`).
cargo: C:/Users/shaur/.cargo/bin/cargo.exe (GNU toolchain, Rust 1.99). Python 3.13 + NumPy 2.3.2: `python`.
Keep `.cargo/config.toml` (jobs = 2): the laptop has 16 GB RAM and two other agents build in parallel.
Never run more than one cargo command at a time. No git commit/reset/checkout. Do not spawn subagents.

# Task: memory management and memory-bound kernels for rustnum

Read first: AGENTS.md, docs/rustnum/PERF_PLAN.md (you own G2, G3, G4, G11, G12), docs/API.md, rustnum/src/array.rs,
layout.rs, ops.rs, ufunc.rs, reduce.rs, broadcast.rs, element.rs.

The owner's workload: `a = tanh((a @ b) / scale); b = tanh((b @ a) / scale)` at 8192² f64 (512 MB per matrix).
Today every op allocates a fresh 512 MB buffer, even when its input is an owned temporary, and some paths copy twice.

## Deliver
1. **Buffer reuse for owned temporaries.** When an operator or method consumes an `Array` whose storage is uniquely owned
   (`Arc::get_mut` succeeds), contiguous, and the same shape and element type as the result, write the result into that
   buffer instead of allocating: `Array op T`, `Array op &Array`, `Array op Array` (reuse whichever side matches),
   unary `-`, compound assignment (already in place: verify it never copies when unique), `astype` to the same type.
2. **In-place element-wise API** (NumPy's `out=` idea, Rust-shaped): a consuming variant for every unary ufunc and
   `sigmoid`/`relu`, named `<name>_inplace(&mut self)` for in-place and `into_<name>(self) -> Self` for the consuming
   chainable form (document both in docs/API.md under a new "In-place and consuming operations" section, with the NumPy
   `out=` equivalent). `mapv_inplace` must use the same parallel contiguous fast path as the allocating ufuncs.
   After this, `(a.matmul(&b) / scale).into_tanh()` must perform zero allocations after the matmul.
3. **Buffer pool** (rustnum/src/pool.rs, private): a size-bucketed cache of large freed buffers (≥ 1 MiB) so that
   repeated alloc/free of 512 MB arrays reuses memory and avoids page-fault and zeroing cost. Requirements: per element
   type (TypeId) buckets, thread-safe, a hard cap on cached bytes (default 25% of a configurable limit, default 2 GiB,
   override with `RUSTNUM_POOL_BYTES`; 0 disables), returns memory to the OS when the cap would be exceeded, and
   `np::memory::{pool_stats, clear_pool, set_pool_limit}` as a small public module (document it in docs/API.md).
   Buffers that come from the pool must still be fully initialised before any read (no uninitialised reads, no UB).
   Measure whether it helps on 4096² and 8192²-sized alloc/free loops; if it does not help, keep it disabled by default
   and say so.
4. **Allocation audit.** Remove avoidable allocations and double copies: `copy()` (to_vec + from_vec), `to_contiguous`,
   `broadcast_to` materialisation, per-lane allocations in reductions, `zip_map` intermediates, temporary `Vec<usize>`
   index buffers. Add rustnum/tests/alloc.rs with a counting `#[global_allocator]` that asserts allocation counts and
   bytes for: owned-op reuse, `into_tanh` chain, compound assignment, contiguous reductions (0 allocations beyond the
   result), and `heavy_matmul`'s step at a small size (one allocation for the matmul result, none after).
5. **Memory-bound kernels:**
   - G2 axis reductions: for C-order input, reduce over axis 0 by accumulating whole rows into the output (contiguous,
     auto-vectorised), last axis by parallel rows, middle axes by blocking; pairwise/blocked summation for float accuracy
     (keep NumPy parity). Exit: `mean_axis(0)` [1000,1000] ≤ 0.45 ms.
   - G3 strided copy: cache-blocked tiled transpose (2-d and the last two axes of n-d), parallel over tiles; a general
     strided copy whose inner loop runs along the axis with the smallest stride. Exit: `transpose().to_contiguous()`
     [1000,1000] ≤ 3.2 ms.
   - G4 broadcasting: specialised inner loops for (contiguous, contiguous), (contiguous, scalar), (row broadcast) and
     (column broadcast), outer loop parallel. Exit: `[1000,1000] + [1000]` ≤ 3.0 ms.
   - G12: per-op parallel thresholds measured on this machine and kept in one table.
6. Tests: every existing test passes; NumPy parity stays 93/93 (`python tools/gen_fixtures.py`, then
   `cargo test --test parity -- --nocapture`); new tests for each reuse path (result equals the allocating path, and the
   input buffer pointer is reused when unique, not when shared), pool limits, and the strided/broadcast kernels on
   transposed, sliced, negative-stride and 0-d/empty inputs.
7. Benchmarks: keep names in rustnum/benches/core.rs; add `into_tanh (1e6 f64)`, `heavy step (1024)`
   (`(a.matmul(&b) / s).into_tanh()`), `alloc-free loop (4096^2 f64)`. Short criterion checks are fine; other agents may
   be using the CPU, so treat numbers as rough. Do NOT run the heavy_matmul example (the owner runs it).

## Acceptance (all must pass)
cargo fmt --all; cargo clippy --all-targets --all-features -- -D warnings; cargo test --all-features;
cargo test -p rustnum --no-default-features; cargo doc --no-deps; cargo bench -p rustnum --no-run

## Don't
Do not edit rustnum/src/linalg/ (another agent rewrites GEMM there), manip.rs sorting code, random/, or the SIMD
transcendental math (another agent adds rustnum/src/simd_math.rs; you may leave a clearly marked call site comment where
ufuncs will use it). No new dependencies. Do not change existing public signatures; only add.

## Finish
Concise summary: files changed, what each reuse/pool path does, allocation-count results from tests/alloc.rs, pool
measurements, rough timings for G2/G3/G4, test counts per gate, deviations.
