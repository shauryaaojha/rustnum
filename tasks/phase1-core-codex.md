You are working in C:/Users/shaur/Documents/ML in RUST (a Cargo workspace). Rust stable with
the x86_64-pc-windows-gnu toolchain is installed; cargo is at C:/Users/shaur/.cargo/bin/cargo.exe
(add C:/Users/shaur/.cargo/bin to PATH if `cargo` is not found).

# Task: Phase 1 of rustnum — the core array library

## Read first (mandatory, in this order)
1. AGENTS.md — repo rules.
2. docs/API.md — the binding API contract. Sections 0–10 and 14 are YOUR job.
3. docs/PLAN.md — design rationale (sections 2, 3, 5).
4. rustnum/Cargo.toml, rustnum/src/lib.rs — the existing skeleton (every module file is a stub).

## Deliver
Implement, in the existing files under rustnum/src/:
1. element.rs — Element / Num / Float traits and impls (API.md §1).
2. array.rs + layout.rs — `Array<T>` with Arc copy-on-write storage, shape/strides/offset, all methods and trait impls
   of §2, an allocation-free odometer iterator for strided arrays, contiguous fast paths.
3. broadcast.rs — `IntoArray<T>` and broadcasting (§3).
4. creation.rs — §4, including the `array!` macro (exported at crate root with #[macro_export]).
5. slicing.rs — §5, including the `s!` macro, `SliceArg`, `NEW_AXIS`, steps, negative indices. `s![1..3, .., -1]` and
   `s![..;2, 1..;-1]` must compile with unsuffixed integer literals.
6. ops.rs — §6 (operators via macros for all owned/borrowed combinations and scalar-on-left for each concrete Num type,
   compound assignment, comparisons, bool logic, allclose/isclose/array_equal).
7. ufunc.rs — §7.
8. reduce.rs — §8. Use pairwise summation for float sums (accuracy like NumPy).
9. manip.rs — §9.
10. display.rs — §10 (NumPy-style Display and Debug).
11. lib.rs — crate-level docs (a short quick-start that runs as a doctest, using `use rustnum as np;`), the `Error` enum
    (variants: Shape(String), DType { expected: String, found: String }, Singular, NotPositiveDefinite, Io(std::io::Error),
    Parse(String), Format(String)) implementing Display + std::error::Error + From<std::io::Error>, `pub type Result<T>`,
    and the re-exports (one path per item). Leave `pub mod io; pub mod linalg; pub mod random;` declared but keep those
    three files as compiling stubs with only a module doc comment — another agent implements them.
12. Rayon: behind the `rayon` feature, element-wise ops and sum/prod/min/max on contiguous arrays with size >= 1<<15
    run in parallel (§14). The crate must also compile and pass tests with --no-default-features.
13. Tests: unit tests in each module (#[cfg(test)]) plus rustnum/tests/core.rs (integration) covering broadcasting,
    slicing (incl. negative steps), views + copy-on-write (mutating a slice does not change the original), reshape
    of non-contiguous arrays, reductions along every axis of a 3-d array, display output strings, panic messages
    (#[should_panic(expected = "...")]). Use hand-computed expected values (what NumPy prints), not re-implementations.
    Add a couple of proptest properties (e.g. transpose twice = identity; broadcast add commutative; reshape round-trip).
14. Every public item has a doc comment whose first line names the NumPy equivalent, with a runnable example.

## Acceptance criteria (run them all; all must pass)
```
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test -p rustnum --no-default-features
cargo doc --no-deps
```

## Don't
- Do not implement linalg.rs, random.rs, io.rs (stubs only). `matmul`/`dot` are NOT yours.
- Do not add dependencies beyond those already in rustnum/Cargo.toml.
- Do not change docs/API.md or docs/PLAN.md. If something in the spec is impossible in Rust, choose the closest
  design, and list it in your final summary.
- Do not use lifetimes in public signatures, do not expose ndarray-like view types.
- Do not run git commands that commit, reset or checkout.

## Finish
End with a concise summary: files written, the exact results of each acceptance command (test counts),
any spec deviations and why, and anything you were unsure about.
