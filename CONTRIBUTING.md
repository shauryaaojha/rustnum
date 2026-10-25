# Contributing

Thanks for helping. `AGENTS.md` holds the rules for both humans and AI agents; this page is the
short version of how to get started.

## Build and test

You need a stable Rust toolchain. Python 3 with NumPy is optional but recommended: tests that
compare against real NumPy skip themselves when it is missing.

```bash
cargo build
cargo test --all-features            # default features plus rayon
cargo test --no-default-features     # results must not depend on rayon
```

Before you open a pull request, run the same gates CI runs:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo doc --no-deps
```

## Parity fixtures

Tests assert ground truth, and the ground truth is NumPy itself. `tools/gen_fixtures.py` runs
real NumPy with a fixed seed and writes expected outputs into `rustnum/tests/fixtures/`
(`.npy` files plus `manifest.json`). The fixtures are not committed; generate them locally:

```bash
python -m pip install numpy
python tools/gen_fixtures.py
cargo test -p rustnum --test parity -- --nocapture
```

To cover a new function, add a case to `tools/gen_fixtures.py` and one match arm in
`rustnum/tests/parity.rs`. Do not re-implement the function inside its own test.

## The rules from AGENTS.md

1. Follow `docs/API.md`: names, argument order and panic wording follow NumPy. If something is
   missing from the spec, copy NumPy's name; do not invent new ones.
2. One path per public item: define it in a private module, re-export it once in
   `rustnum/src/lib.rs`. Only `linalg`, `random` and `io` are public modules.
3. No lifetimes in public APIs. No `unsafe` unless a `// SAFETY:` comment and a benchmark
   justify it.
4. Misuse (shape mismatch, bad index, bad axis) panics with a NumPy-style message. Bad data
   (files, parsing, singular matrices) returns `rustnum::Result`.
5. Every public item has a doc comment with a runnable example. The first line is one sentence
   and names the NumPy equivalent, for example `/// Sum of all elements, like np.sum(a).`
6. Tests check ground truth: NumPy fixtures or hand-computed values.
7. Keep dependencies minimal (`matrixmultiply`, optional `rayon`; dev: `proptest`, `criterion`).
   Ask before adding anything else. The `.npz` layer is dependency-free on purpose.

## Commits and pull requests

Keep changes focused, explain the why in the description, and make sure CI is green. By
contributing you agree that your work is licensed under the MIT and Apache-2.0 licenses of this
repository.
