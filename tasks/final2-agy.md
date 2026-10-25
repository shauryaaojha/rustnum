You are working in C:/Users/shaur/Documents/mlrs-final (git worktree of the rustnum repo, branch `final`).
cargo: C:/Users/shaur/.cargo/bin/cargo.exe (GNU toolchain). Python 3.13 + NumPy 2.3: `python`.
Keep `.cargo/config.toml` (jobs = 3; low RAM). Never run several cargo commands at once. Do not spawn subagents.

rustnum (NumPy-style arrays in Rust) is fully implemented and wired, including np::linalg. 1394 tests pass and the NumPy
parity test rustnum/tests/parity.rs passes 93/93. Read AGENTS.md, rustnum/src/lib.rs (the real public API),
rustnum/src/linalg/api.rs, README.md, docs/NUMPY_MAPPING.md, llms.txt.

Deliver:
1. rustnum/tests/parity.rs: fix all compiler warnings (unused functions/fields/variables) so clippy -D warnings passes.
   Do not change what the cases check. It must still print "parity: 93/93 cases passed, 0 skipped".
2. Examples: rename rustnum/examples/quickstart.rs.pending, kmeans.rs.pending, image_ops.rs.pending to .rs and make each
   compile and run: `cargo run --release -p rustnum --example <name>`. They were written against an older spec; fix them
   to the REAL API (read signatures in rustnum/src). Keep each top doc comment showing the NumPy equivalent.
   kmeans must recover 3 centroids within 0.3 of the true blob centers (assert). image_ops writes its .npy to std::env::temp_dir().
   Do NOT touch linear_regression.rs.pending / pca.rs.pending (another agent owns them).
3. README test: rename rustnum/tests/readme_quickstart.rs.pending to .rs and make it pass. If the README quick start does not
   compile against the real API, fix README.md so README and test agree exactly.
4. Check every code snippet and signature in docs/NUMPY_MAPPING.md and llms.txt against the REAL public API (names,
   argument types, return types, which functions are methods vs free functions, `np::linalg` / `np::random` / `np::io` /
   `np::fft` / `np::poly` paths). Fix the docs where wrong. Add rows for public functions that are missing from the mapping
   (fft, poly, stats/sets/einsum, misc math, windows, npz). Never change the public API to match the docs.
No git commit/reset/checkout. No new dependencies. Do not edit rustnum/src except to fix a real bug you find (list it).
Gates: cargo fmt --all; cargo clippy --all-targets --all-features -- -D warnings; cargo test --all-features; cargo doc --no-deps;
and each of the three examples runs.
End with a concise summary: files changed, gate results with test counts, example outputs, docs fixes.
