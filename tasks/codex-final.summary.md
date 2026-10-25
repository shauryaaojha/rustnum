Implemented all requested changes.

- Enabled [linear_regression.rs](C:/Users/shaur/Documents/mlrs-cx/rustnum/examples/linear_regression.rs) and [pca.rs](C:/Users/shaur/Documents/mlrs-cx/rustnum/examples/pca.rs).
- Made generic linalg GEMM and `vdot` use wrapping add/multiply helpers; added i32/i64 overflow regressions.
- Added the benchmark-local wrapping shim required by its direct kernel inclusion.
- Consolidated the five API addenda into [docs/API.md](C:/Users/shaur/Documents/mlrs-cx/docs/API.md) and deleted them.
- Updated [docs/PLAN.md](C:/Users/shaur/Documents/mlrs-cx/docs/PLAN.md) to reflect the delivered v0.1 scope.
- Preserved `.cargo/config.toml`; excluded files were untouched.

Examples:

- Linear regression: maximum solution difference `5.55e-16`; MSE `0.01057`.
- PCA: explained-variance ratio `[0.92411723, 0.07588277]`.

Gates:

- `cargo fmt --all` — passed
- Clippy, all targets/features with warnings denied — passed
- All features — 1,395 passed, 1 ignored
- No default features — 1,395 passed, 1 ignored
- `cargo doc --no-deps` — passed
- `git diff --check` — clean