# PARALLEL MODE — read this first, it overrides the "Read first" assumption below
You are working in C:/Users/shaur/Documents/mlrs-randio (a git worktree, branch `random-io`). Run all cargo commands there.
Phase 1 (the core Array type) is being written RIGHT NOW by another agent in a different worktree, so in your
worktree the core modules are still stubs. Work in two steps:

STEP A (do now, must compile and pass tests here):
- random: rustnum/src/random/mod.rs (replace random.rs) with a private `pub(crate) mod engine;`
  (rustnum/src/random/engine.rs): SplitMix64, xoshiro256**, next_u64/next_f64, unbiased bounded integers, normal
  (Box-Muller or ziggurat), exponential, binomial, poisson, gamma, beta, Fisher-Yates shuffle on `&mut [T]`,
  permutation, choice indices with/without replacement — all producing plain `Vec`s, no Array. Full tests as listed below.
- io: rustnum/src/io/mod.rs (replace io.rs) with a private `pub(crate) mod npy;` and `pub(crate) mod csv;`:
  .npy header parse/write (v1.0/2.0, dtypes, fortran_order, shape), encode/decode of raw little-endian element bytes to
  Vec<f64|f32|i64|i32|u8|bool|usize>, CSV parse/format of 1-d/2-d numeric data — all on `(Vec<T>, Vec<usize> shape)`, no Array.
  Errors: use a private error enum now; it will be mapped to `rustnum::Error` when wiring.
  Tests: header round-trips, NumPy-written bytes (generate small .npy files with `python -c` inside the test and skip if
  python/numpy is missing), Fortran order, 0-d, v2.0 headers, CSV edge cases.
- tools/gen_fixtures.py: complete (independent of Rust). Run it.
- `cargo test -p rustnum` and `cargo clippy -p rustnum --all-targets -- -D warnings` must pass in your worktree.

STEP B (written now but NOT compiled yet):
- rustnum/src/random/api.rs.pending: the public `Rng` per docs/API.md §12 calling engine.rs and returning Array.
- rustnum/src/io/api.rs.pending: save_npy/load_npy/to_npy_bytes/from_npy_bytes/save_csv/load_csv per §13.
- rustnum/tests/parity.rs.pending and rustnum/tests/npy_python.rs.pending: the parity runner and NumPy round-trip tests
  as described below, written against the public API in docs/API.md (§2–§10, §12, §13).
  Use only the Array API from docs/API.md. The orchestrator will merge the core and ask you to wire these files.

Finish with the summary described at the bottom.

You are working in a git worktree of the rustnum repo (the directory given by --add-dir).
Rust stable (x86_64-pc-windows-gnu) is installed: cargo is C:/Users/shaur/.cargo/bin/cargo.exe.
Python 3.13 with NumPy 2.3 is available as `python`.

# Task: Phase 2b of rustnum — random, I/O, and NumPy parity tests

## Read first
1. AGENTS.md, docs/API.md (§12 random, §13 io are your job), docs/PLAN.md.
2. rustnum/src/lib.rs and core modules (array.rs, element.rs, broadcast.rs, creation.rs, reduce.rs, manip.rs, ufunc.rs).
   Phase 1 (core array) is finished; build on its public API and the crate `Error`/`Result`.

## Deliver
1. rustnum/src/random.rs — `np::random::Rng` exactly as API.md §12. xoshiro256** seeded by SplitMix64, no new
   dependencies. `from_entropy` may use std time + address entropy. Unbiased `integers` (Lemire or rejection).
   Tests: same seed → same sequence (hard-code the first 5 u64 outputs of seed 42 as a regression test after
   checking them against the reference xoshiro256** algorithm), statistical sanity (mean/std of 1e5 normals
   within 0.02 of the target, uniform in range, integers cover all values, permutation is a permutation,
   shuffle keeps the multiset, choice without replacement has no duplicates).
2. rustnum/src/io.rs — API.md §13: .npy v1.0/v2.0 read + write (v1.0 written, v2.0 when header > 65535),
   dtypes `<f8 <f4 <i8 <i4 |u1 |b1 <u8` mapping to f64 f32 i64 i32 u8 bool usize, C and Fortran order on read,
   0-d arrays, `to_npy_bytes`/`from_npy_bytes`, CSV save/load (`np.savetxt`/`np.loadtxt` behaviour, 2-d or 1-d).
   Errors: Error::DType on mismatch, Error::Format for bad headers, Error::Parse for CSV.
3. tools/gen_fixtures.py — generates NumPy reference outputs into rustnum/tests/fixtures/ as .npy files plus
   rustnum/tests/fixtures/manifest.json listing each case: {name, op, inputs: [files], params, output: file, dtype, rtol, atol}.
   Cover at least 60 cases across: creation (arange, linspace, eye, tri), broadcasting arithmetic (+ - * / % with
   shapes like (3,1)+(1,4), (2,3,4)*(4,)), ufuncs (exp, log, sqrt, tanh, sin, clip, floor, round — note NumPy rounds
   half to even), reductions (sum/mean/std/var/min/max/argmax/cumsum on whole array and each axis of a 3-d array,
   median, percentile), manipulation (reshape, transpose, swapaxes, concatenate, stack, flip, roll, repeat, tile,
   sort, argsort, unique, where), slicing (negative steps), comparisons, astype, and linalg (matmul 2-d and batched,
   dot, det, inv, solve, norm, cholesky, qr reconstruct, eigh eigenvalues, svd singular values, pinv, matrix_rank).
   Use a fixed seed. Fixture .npy files are gitignored; the test must skip with a clear message if they are missing.
4. rustnum/tests/parity.rs — loads manifest.json (write a tiny JSON reader in the test or use a simple line
   format instead of JSON if easier; no new dependencies), runs each case through rustnum, compares with
   allclose (or exact equality for ints/bools), and prints a summary "parity: N/N cases passed". It must fail
   listing every mismatching case. The runner dispatches on `op` with one `match`; linalg ops do NOT exist in your
   worktree yet (another agent is writing them in parallel), so do not call any linalg function: unknown ops are
   reported as "skipped: op not wired" and do not fail the test. Keep the dispatch easy to extend (one match arm per op)
   so the linalg arms can be added after the merge. Generate the linalg fixtures anyway.
5. Round-trip test with real NumPy: tests/npy_python.rs writes .npy from Rust, calls `python -c` to load it with
   NumPy and assert values; and the reverse. Skip gracefully if `python` or NumPy is unavailable.
6. Doc comment + runnable example on every public item.

## Acceptance (all must pass)
```
python tools/gen_fixtures.py
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test -p rustnum --no-default-features
```

## Don't
- Do not touch linalg.rs, benches/, tools/bench_numpy.py — another agent owns them.
- Do not change Phase 1 public signatures; if you find a Phase 1 bug, fix it minimally and list it in your summary.
- No new crate dependencies. No git commit/reset/checkout.

## Finish
End with a concise summary: files written, results of each acceptance command (test counts, parity N/N),
deviations from API.md, open doubts.
