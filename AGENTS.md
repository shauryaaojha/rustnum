# AGENTS.md

Rules for AI agents and human contributors working on this repo.

## What this repo is
*ML in Rust*: rebuilding the Python ML stack in Rust, one library at a time. The first library
is `rustnum/` (NumPy). Design: `docs/PLAN.md`. Binding API spec: `docs/API.md`.

## Rules
1. **Follow `docs/API.md`.** Names, argument order and panic wording follow NumPy. Do not invent
   new public names; if something is missing from the spec, copy NumPy's name.
2. **One path per public item.** Core items are defined in private modules and re-exported once in
   `rustnum/src/lib.rs`. Only `linalg`, `random`, `io` are public modules.
3. **No lifetimes, no `unsafe`** in public APIs. `unsafe` inside is allowed only with a `// SAFETY:` comment
   and a benchmark that justifies it.
4. **Panics vs errors.** Misuse (shape mismatch, bad index, bad axis) panics with a NumPy-style message.
   Bad data (files, parsing, singular matrices) returns `rustnum::Result`.
5. **Every public item gets a doc comment with a runnable example.** The first line is one sentence
   and names the NumPy equivalent, e.g. `/// Sum of all elements, like np.sum(a).`
6. **Tests assert ground truth.** Prefer NumPy parity fixtures (`tools/gen_fixtures.py` →
   `rustnum/tests/fixtures/`) or hand-computed values over re-implementing the function in the test.
7. **Gates before you finish:**
   ```bash
   cargo fmt --all
   cargo clippy --all-targets -- -D warnings
   cargo test --all-features
   cargo test --no-default-features
   cargo doc --no-deps
   ```
8. Keep dependencies minimal: `matrixmultiply`, optional `rayon`. Dev: `proptest`, `criterion`. Ask before adding more.

## Layout
`rustnum/src/<module>.rs` per area (see `docs/PLAN.md` section 4). Integration tests in `rustnum/tests/`,
benchmarks in `rustnum/benches/`, examples in `rustnum/examples/`, Python helpers in `tools/`.
