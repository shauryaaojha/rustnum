You are working in C:/Users/shaur/Documents/mlrs-cx (git worktree of the rustnum repo, branch `codex-final`).
cargo: C:/Users/shaur/.cargo/bin/cargo.exe (GNU toolchain). Python 3.13 + NumPy 2.3: `python`.
Keep `.cargo/config.toml` (jobs = 3; the laptop is low on RAM). Never run several cargo commands at once.

rustnum (NumPy-style arrays in Rust) is fully implemented and wired; 1394 tests pass. Read AGENTS.md, rustnum/src/lib.rs,
rustnum/src/linalg/mod.rs + api.rs (public linalg), docs/API.md, docs/api/*.md.

Deliver:
1. Examples needing linalg: rename rustnum/examples/linear_regression.rs.pending and pca.rs.pending to .rs, make them
   compile and run with `cargo run --release -p rustnum --example <name>` against the REAL API, keep their top doc comments
   (NumPy equivalent). linear_regression: normal equations via np::linalg::solve AND gradient descent must agree within 1e-3
   (assert it). pca: explained variance ratio of the first component of strongly correlated 2-D data must exceed 0.85 (assert).
2. Integer overflow: rustnum/src/linalg/kernels.rs `gemm` (generic) uses plain + and *, so multi_dot / tensordot / vdot /
   matrix_power panic on integer overflow in debug builds while matmul wraps. Make them all wrap like NumPy (use the core's
   wrapping add/mul helpers that matmul already uses). Add regression tests with i32/i64 overflow.
3. docs/API.md: update it to describe the FINAL code. Merge docs/api/fft-poly.md, extra.md, misc.md, npz.md, linalg-extra.md
   into it as numbered sections; fix every signature that drifted (e.g. LinalgFloat bound, Error variants incl. NoConvergence,
   binomial/poisson return Array<i64>, savez/load_npz in np::io, angle/ediff1d free functions, Bins/ConvolveMode/Spacing).
   Keep it a contract document: one line per item. Then delete the merged docs/api/*.md files.
4. Update docs/PLAN.md section 2 "Out of scope for v0.1" and section 6 (phases) to reflect what was actually delivered
   (fft, polynomial, einsum, npz are now IN scope and done; complex dtype still out — Complex<F> exists as a scalar type).
Do NOT touch rustnum/tests/parity*, README.md, docs/NUMPY_MAPPING.md, llms.txt, quickstart/kmeans/image_ops examples
(another agent owns those). No git commit/reset/checkout. No new dependencies.
Gates: cargo fmt --all; cargo clippy --all-targets --all-features -- -D warnings; cargo test --all-features;
cargo test -p rustnum --no-default-features; cargo doc --no-deps.
End with a concise summary: files changed, gate results with test counts, example outputs.
