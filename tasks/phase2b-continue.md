You are continuing Phase 2b in C:/Users/shaur/Documents/mlrs-randio (worktree, branch random-io). The previous run stopped on a
quota error. Current state: rustnum/src/random/engine.rs (646 lines), rustnum/src/io/npy.rs (659 lines), rustnum/src/io/mod.rs,
rustnum/src/random/mod.rs and the four .pending files exist. MISSING: rustnum/src/io/csv.rs (mod.rs declares it, so the crate
does not compile), tools/gen_fixtures.py, and the tests for engine.rs/npy.rs may be incomplete.
Re-read tasks/phase2b-parallel.md for the full brief, then:
1. Write rustnum/src/io/csv.rs (parse/format 1-d/2-d numeric data on (Vec<T>, shape), np.loadtxt/np.savetxt behaviour,
   delimiter: char, skip_header: usize, comments starting with '#', blank lines skipped, Parse errors with line numbers) + tests.
2. Make sure engine.rs and npy.rs have the tests the brief asks for (seed-42 regression values checked against the reference
   xoshiro256** algorithm, statistical sanity checks, npy header round-trips, Fortran order, 0-d, v2.0, files written by real
   NumPy via `python -c`, skipped if python/numpy is missing).
3. Write tools/gen_fixtures.py exactly as the brief specifies (>= 60 cases, manifest.json) and run it.
4. Fix the unused-import warning. Gates: cargo test -p rustnum; cargo clippy -p rustnum --all-targets -- -D warnings.
Do not touch anything outside this worktree. No git commit/reset/checkout. Do not spawn subagents; write the files yourself.
End with a concise summary: files written, test counts, fixture case count.
