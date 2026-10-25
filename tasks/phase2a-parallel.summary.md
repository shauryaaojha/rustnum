Completed Step A and staged Step B.

Files added/updated:

- [linalg kernels](/C:/Users/shaur/Documents/mlrs-linalg/rustnum/src/linalg/kernels.rs) — stride-aware f32/f64 GEMM, generic GEMM, LU, inverse/solve/determinant, Cholesky, QR, least squares, Jacobi eigh/SVD, pinv, rank, norms, and 7 raw-slice identity tests.
- [linalg module](/C:/Users/shaur/Documents/mlrs-linalg/rustnum/src/linalg/mod.rs) — private kernel registration.
- [pending Array API adapters](/C:/Users/shaur/Documents/mlrs-linalg/rustnum/src/linalg/api.rs.pending)
- [kernel benchmarks](/C:/Users/shaur/Documents/mlrs-linalg/rustnum/benches/core.rs) and [pending public-API benchmarks](/C:/Users/shaur/Documents/mlrs-linalg/rustnum/benches/core.rs.pending)
- [NumPy comparison benchmark](/C:/Users/shaur/Documents/mlrs-linalg/tools/bench_numpy.py)
- `Cargo.toml` / lockfile: `rayon` now enables `matrixmultiply/threading`.

Validation passed:

- `cargo fmt --all`
- `cargo clippy -p rustnum --all-targets --all-features -- -D warnings`
- `cargo test -p rustnum` — 7 passed
- `cargo test --all-features` — 7 passed
- `cargo test -p rustnum --no-default-features` — 7 passed
- `cargo bench -p rustnum --no-run`
- `cargo doc --no-deps`

Deviation by design: Array-facing APIs remain pending/unregistered, per Step B. Direct no-copy GEMM for transposed Array views needs a core accessor for the backing buffer/offset; the documented current public Array surface only exposes contiguous slices, so the pending adapters presently use logical vectors for those cases. No Phase-1 rayon operation changes were possible here because those modules remain stubs.