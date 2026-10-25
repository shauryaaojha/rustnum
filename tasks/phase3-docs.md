You are working in C:/Users/shaur/Documents/ML in RUST (Cargo workspace, crate `rustnum/`).
Rust stable (x86_64-pc-windows-gnu): cargo is C:/Users/shaur/.cargo/bin/cargo.exe. Python 3.13 + NumPy 2.3 as `python`.

# Task: Phase 3 — documentation and examples for rustnum

rustnum is finished and tested (core, linalg, random, io). Your job is to make it easy for humans AND AI agents to use.
Write in clear, simple English. No marketing language, no emojis.

## Read first
AGENTS.md, docs/PLAN.md, docs/API.md, rustnum/src/lib.rs, then skim each module in rustnum/src/ for the real
public signatures. The CODE is the source of truth: if docs/API.md and the code disagree, document the code and
list the difference in your summary.

## Deliver
1. README.md (repo root): what rustnum is, a 15-line quick start (`use rustnum as np;`), install (`cargo add rustnum`
   from git path for now: `rustnum = { path = "rustnum" }`), feature list, the NumPy vs rustnum differences table
   (shortened from PLAN.md §2), a "Benchmarks" section with a placeholder line `<!-- BENCH TABLE -->` (the orchestrator
   fills it), test instructions (`cargo test`, `python tools/gen_fixtures.py` then parity tests), project roadmap
   (next libraries), license line (MIT OR Apache-2.0).
2. docs/NUMPY_MAPPING.md: one table per area (creation, attributes, indexing/slicing, arithmetic, ufuncs, reductions,
   manipulation, linalg, random, io). Columns: NumPy | rustnum | Notes. Cover EVERY public function/method of rustnum.
   Add a "Common gotchas" section (copy-on-write views, no kwargs → `_axis` methods, `where_`, `reshape_infer` for -1,
   float literals need `.0` (`np::arange(0.0, 5.0, 1.0)`), `&a + &b` borrows vs `a + b` moves, scalar-on-left support,
   random streams differ from NumPy).
3. docs/ARCHITECTURE.md: storage/strides/offset diagram (ASCII), copy-on-write, broadcasting via zero strides,
   iteration (contiguous fast path vs odometer), rayon threshold, matmul dispatch, how to add a new ufunc / reduction
   (step-by-step checklist for contributors and agents), testing layers (unit, doctest, proptest, NumPy parity).
4. llms.txt (repo root): a compact (≤ 250 lines) API cheat sheet for LLMs following the llms.txt convention
   (title line, one-paragraph summary, then sections of `signature — one-line meaning`). Include the 10 most common
   NumPy → rustnum translations as code pairs.
5. rustnum/examples/: runnable examples, each with a top doc comment saying what NumPy code it mirrors:
   - quickstart.rs (creation, arithmetic, broadcasting, slicing, reductions, printing)
   - linear_regression.rs (generate data with np::random, fit with normal equations via linalg::solve AND with
     gradient descent; print both weights; they must agree within 1e-3)
   - pca.rs (random 2-D correlated data → center → covariance → eigh → project; print explained variance ratio)
   - kmeans.rs (3 Gaussian blobs, k-means with broadcasting distance matrix, print centroids)
   - image_ops.rs (treat a [64,64] array as an image: normalize, flip, transpose, 3x3 box blur with slicing,
     save as .npy)
   Every example must run: `cargo run --release -p rustnum --example <name>`.
6. Check that every code block in README.md and NUMPY_MAPPING.md is correct against the real API: put the README
   quick start into rustnum/tests/readme_quickstart.rs (a test that runs the same code) so it cannot rot.

## Acceptance (all must pass)
```
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
for e in quickstart linear_regression pca kmeans image_ops; do cargo run --release -p rustnum --example $e; done
```

## Don't
- Do not change any code in rustnum/src/ (docs only). If you find a bug, describe it in your summary instead.
- Do not touch docs/PLAN.md or docs/API.md. No git commit/reset/checkout. No new dependencies.

## Finish
Concise summary: files written, acceptance results, API.md vs code differences found, bugs found.
