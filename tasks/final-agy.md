You are working in C:/Users/shaur/Documents/mlrs-final (git worktree of the rustnum repo, branch `final`).
cargo: C:/Users/shaur/.cargo/bin/cargo.exe (GNU toolchain). Python 3.13 + NumPy 2.3: `python`.
`.cargo/config.toml` limits builds to 3 jobs: keep it, the laptop is low on RAM. Never run several cargo commands at once.

rustnum (a NumPy-style array library) is implemented and wired: core, random, io (.npy/CSV/npz), fft, poly, stats/signal/
sets/einsum (1235 tests pass). Linear algebra (`np::linalg`, `matmul`, `dot`) is being wired by another agent RIGHT NOW in a
different worktree: in your worktree `linalg` is still kernels only, so do not call any linalg API yet.
Read: AGENTS.md, docs/API.md, rustnum/src/lib.rs (the real public API), docs/NUMPY_MAPPING.md.

Deliver:
1. Parity runner: rename rustnum/tests/parity.rs.pending -> rustnum/tests/parity.rs and make it compile against the REAL API.
   Run `python tools/gen_fixtures.py` first (writes rustnum/tests/fixtures/*.npy + manifest.json). Every non-linalg case in the
   manifest must be wired to one match arm and PASS; linalg cases print "skipped: op not wired" for now (keep one clearly
   marked match block for them). If a case fails because rustnum differs from NumPy, decide: rustnum bug -> fix it in
   rustnum/src (minimal, with a regression test); fixture bug -> fix tools/gen_fixtures.py. List every such fix.
   The test prints "parity: N/M cases passed, K skipped".
2. Examples: rename rustnum/examples/{quickstart,kmeans,image_ops}.rs.pending -> .rs and make them compile and run
   (`cargo run --release -p rustnum --example <name>`). They were written against docs/API.md before the code existed, so
   fix them to the real API; keep each top doc comment with the NumPy equivalent. Leave linear_regression.rs.pending and
   pca.rs.pending as they are (they need linalg).
3. README test: rename rustnum/tests/readme_quickstart.rs.pending -> .rs and make it pass; if the README quick start does
   not compile against the real API, fix README.md so both agree.
4. Check every code block in docs/NUMPY_MAPPING.md and llms.txt against the real public API (names, argument types,
   return types). Fix the docs where they are wrong. Do not change public API to match docs.
Do NOT touch rustnum/src/linalg/, rustnum/src/misc/, benches/. No git commit/reset/checkout. No new dependencies.
Do not spawn subagents. Gates: cargo fmt --all; cargo clippy --all-targets --all-features -- -D warnings;
cargo test --all-features; cargo doc --no-deps.
End with a concise summary: parity N/M, fixes made in src (with reason), docs fixes, gate results.
