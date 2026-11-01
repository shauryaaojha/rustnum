# rustnum performance plan (v0.1 → H3 "fast")

Goal: every benchmark at least as fast as NumPy 2.3 + OpenBLAS on the same machine, and clearly faster on
workloads like `examples/heavy_matmul.rs`. Measured 2026-10-26 on the dev laptop.

## 1. Machine (this shapes the design)

**Intel Core Ultra 5 125H**, a hybrid CPU: 4 P-cores (8 threads) + 8 E-cores + 2 low-power E-cores = 14 cores /
18 threads, AVX2 + FMA, **no AVX-512**, laptop power limits. Two consequences:
- Work split into equal static chunks waits on the slowest core (an LP E-core). OpenBLAS splits statically, so it
  scales poorly here (8 → 18 threads only gives 76 → 105 GF/s). Fine-grained work stealing (rayon) handles mixed
  cores better. This is where we can beat OpenBLAS on modern laptops (Intel hybrid, Apple M-series, ARM big.LITTLE).
- Kernels must be AVX2/FMA (256-bit) first, with runtime dispatch; AVX-512 and NEON come second.
- Timings drift with power mode and heat: benchmark plugged in, "Best performance" mode, 3 runs, report the median.

## 2. What `heavy_matmul` actually measures

Both runs took ~29 s for 12 matmuls of 4096² plus 12 divisions and 12 `tanh` (user run). The tie hides two opposite gaps:

| Step (one 4096² step) | NumPy 18 thr | rustnum 18 thr | Who wins |
|---|---:|---:|---|
| `a @ b` | 1.31 s (105 GF/s) | 5.30 s (26 GF/s) | NumPy 4x |
| `/ scale` | 117 ms | 13–18 ms | rustnum 7x |
| `tanh` | 649 ms (single-threaded) | 63 ms (rayon) | rustnum 10x |
| `standard_normal` (setup) | 290 ms | 482–639 ms | NumPy 2x |

Thread scaling of the matmul alone:

| Threads | NumPy/OpenBLAS | rustnum/matrixmultiply |
|---:|---:|---:|
| 1 | 17 GF/s | 13 GF/s |
| 4 | 43 GF/s | 27 GF/s |
| 8 | 76 GF/s | 30 GF/s |
| 18 | 105 GF/s | 26 GF/s |

**Root cause #1:** `matrixmultiply` hard-caps threading at 4 (`const MAX_THREADS: usize = 4` in
`matrixmultiply-0.3.11/src/threading.rs`). rustnum uses 4 of 18 threads for matmul, and its single-thread kernel is
also ~25% behind OpenBLAS.

Projection: with matmul at NumPy speed, heavy_matmul drops from ~29 s to ~17 s (NumPy stays at ~29 s because its
`tanh` is single-threaded): **~1.7x faster than NumPy**. With our own kernel at 140 GF/s plus fusion: **~13 s, ~2.2x**.

## 3. Every gap found, with its cause

| # | Op | Gap vs NumPy (README bench) | Cause in the code | Fix |
|---|---|---|---|---|
| G1 | matmul f64/f32 | 2.1x at 1024², 4x at 4096² | `matrixmultiply` capped at 4 threads; generic micro-kernel | P0 quick win, then own GEMM (P1) |
| G2 | `mean_axis(0)`, all `*_axis` reductions | 10x | `reduce.rs:40` `lanes()` reduces one strided lane per output element: for axis 0 every element read jumps a full row (cache miss each read), single-threaded | Axis-0 on C-order: accumulate whole rows into the output buffer (contiguous, vectorisable), parallel over column blocks; last-axis: parallel over rows |
| G3 | `transpose().to_contiguous()`, any strided copy | 2x | `array.rs:364` `copy()` → `to_vec()` → odometer `iter().collect()`: element-by-element gather, single-threaded | Cache-blocked tiled transpose (e.g. 32×32 tiles), parallel over tiles; generic strided copy with a contiguous inner loop |
| G4 | broadcast binary ops (`[1000,1000] + [1000]`) | 1.4x | `array.rs:380` `zip_map` falls back to two odometer iterators when either side is not contiguous (a broadcast view never is) | Inner-loop kernels: (contiguous, contiguous), (contiguous, scalar), (row-broadcast) with the outer loop parallel |
| G5 | `sort` | 2.7x | `manip.rs` `sort_by(compare)` per lane, comparison sort with NaN check in every compare, single-threaded | Partition NaNs out first, then `par_sort_unstable_by(total_cmp)`; LSD radix sort for f64/i64 (bit-flip trick) on large inputs |
| G6 | `exp`, `log`, `tanh`… single-thread | 1.5x | scalar `libm` per element | SIMD (AVX2) polynomial kernels with ≤1 ulp error like NumPy's, via `std::arch` + runtime dispatch; already parallel above 1<<15 |
| G7 | `inv`, `solve`, `det` | 6.3x | unblocked LU in `linalg/kernels.rs` | Blocked right-looking LU with partial pivoting; trailing update through our GEMM; parallel panel factorisation |
| G8 | `svd` | 16x | one-sided Jacobi | Householder bidiagonalisation (blocked) + divide-and-conquer SVD; keep Jacobi for small or high-accuracy cases |
| G9 | `eigh`, `qr`, `cholesky`, `lstsq` | not benchmarked | cyclic Jacobi eigh, unblocked QR/Cholesky | Tridiagonalisation + divide-and-conquer eigh; blocked Householder QR (WY form); blocked Cholesky |
| G10 | `standard_normal`, `normal` | 2x (single thread both) | `random/engine.rs:117` Box-Muller: `ln`, `sqrt`, `cos`, `sin` per pair | Ziggurat (NumPy uses it), then parallel fill: split the output into chunks, each with a `jump()`-ed xoshiro stream so results stay deterministic for a given seed and thread count is irrelevant |
| G11 | chained ops `(a @ b / s).tanh()` | 3 full passes, 3 allocations of 128 MB | every op allocates a new array, even when the input is an owned, unique temporary | (a) owned ops reuse the buffer when `Arc::strong_count == 1`; (b) `np::linalg::gemm(alpha, a, b, beta, c)` (SciPy BLAS name) folds the scale into the GEMM; (c) `mapv_inplace` / in-place ufuncs |
| G12 | thread pool on hybrid CPUs | affects every parallel op | rayon defaults to 18 threads incl. LP E-cores; one size threshold (1<<15) for all ops | Per-op thresholds tuned by benchmark; chunk sizes small enough for stealing; optional `RUSTNUM_NUM_THREADS` |

## 4. Phases

Each phase ends with: gates green, NumPy parity 93/93 still passing, benchmark table re-run on an idle, plugged-in
machine, numbers recorded in `docs/rustnum/BENCH.md`.

### P0. Quick wins (1–2 days)
1. **Parallel matmul over row panels.** Split C into row panels sized for the P-core L2, run one single-threaded
   `matrixmultiply` call per panel in a rayon task (`MATMUL_NUM_THREADS`-free: call with threading off). Work stealing
   puts more panels on P-cores. Batched matmul parallelises over batches first.
   Exit: matmul 4096² ≥ 90 GF/s at 18 threads; heavy_matmul ≤ 19 s.
2. **Owned-buffer reuse** for `Array op scalar`, `Array op &Array` and unary ufuncs on owned unique arrays (G11a).
   Exit: no allocation in `(a.matmul(&b) / s).tanh()` after the matmul (checked with a counting allocator in a test).
3. **Axis reductions rewrite** (G2). Exit: `mean_axis(0)` [1000,1000] ≤ 0.45 ms (NumPy 0.42).
4. **Blocked transpose and strided copy** (G3). Exit: ≤ NumPy's 3.26 ms.
5. **Broadcast inner-loop kernels** (G4). Exit: ≤ NumPy's 3.04 ms.

### P1. Own GEMM (3–5 days)
BLIS-style f64 and f32 GEMM in `rustnum/src/linalg/gemm/`:
- Packing of A (MC×KC) and B (KC×NC) panels; cache blocking tuned for P-core L1 48 KB / L2 2 MB / L3 18 MB,
  with an autotune table per CPU family.
- AVX2+FMA micro-kernels: f64 6×8 (12 ymm accumulators), f32 6×16; edge kernels; scalar fallback; NEON later.
- Two-level parallelism (jc over NC blocks, ic over MC blocks) as rayon tasks, so slow cores take fewer blocks.
- Epilogue hook: `C = alpha·A·B + beta·C`, then an optional element-wise function applied while the tile is in
  cache (enables a fused `matmul → scale → tanh` without extra passes).
- Public addition: `np::linalg::gemm(alpha, &a, &b, beta, &c)` (SciPy `linalg.blas.dgemm` name).
- Removes the `matrixmultiply` dependency. The kernels move to `rust-ml-kernels` when that crate exists.
Exit: single-thread ≥ OpenBLAS (17 GF/s here); 18 threads ≥ 130 GF/s at 4096² (OpenBLAS 105); 256² not slower than
today; f32 ≥ 2x f64. heavy_matmul ≤ 14 s.

### P2. SIMD math and RNG (2–3 days)
- AVX2 kernels for exp, log, log1p, expm1, sin, cos, tanh, sigmoid, sqrt-based ops, with ulp tests against `f64`
  reference on 10⁷ random inputs and NumPy parity (G6).
- Ziggurat normal and exponential; parallel deterministic fills with xoshiro `jump()` (G10).
Exit: single-thread exp ≤ NumPy's 5.76 ms; `standard_normal(4096²)` ≤ 290 ms single-thread and ≤ 60 ms parallel.

### P3. Blocked LAPACK-class kernels (4–6 days)
Blocked LU (inv, solve, det), blocked Cholesky, blocked Householder QR, bidiagonalisation + divide-and-conquer SVD,
tridiagonalisation + divide-and-conquer eigh; all trailing updates through P1's GEMM (G7–G9).
Exit: inv 200² ≤ 1.0 ms (NumPy 1-thread 0.97), SVD 100² ≤ 7 ms (NumPy 6.87); add benchmarks for 1000² inv/svd/eigh/qr
and match NumPy within 1.5x; all linalg parity and the 150 kernel tests still pass.

### P4. Sorting and the rest (1–2 days)
Parallel sort with NaN partition, radix sort for large f64/i64, `argsort` via (key, index) radix; parallel
`unique`, `cumsum` (two-pass scan), `argmax`/`argmin` (G5, G12).
Exit: sort 1e6 f64 ≤ NumPy's 12.0 ms; per-op thread thresholds recorded in `docs/rustnum/BENCH.md`.

## 5. Benchmark additions (so every fix is measured)
- `heavy_matmul` as a criterion bench at 2048² (CI-sized) plus the 4096² example for manual runs.
- Thread sweeps (1, 4, 8, 18) for matmul, sum, exp, sort, using `RAYON_NUM_THREADS` and
  `OPENBLAS_NUM_THREADS` on the NumPy side.
- Sizes 64, 256, 1024, 4096 for matmul and inv; f32 rows.
- Every NumPy row states its thread count; anomalies (like OpenBLAS `inv` at 85–102 ms with default threads on this
  CPU) are footnoted, never used as wins.

## 6. Agent split (parallel worktrees)
| Phase | Agent |
|---|---|
| P0.1 parallel panels + P1 own GEMM | codex (high), one worktree: they touch the same files |
| P0.2–P0.5 reductions, copy, broadcast, buffer reuse | agy (Gemini 3.8 High) |
| P2 SIMD math + RNG | Claude subagent (Sonnet, high) |
| P3 blocked LAPACK | Claude subagent (Sonnet, high), starts when P1 GEMM lands |
| P4 sorting | agy |
| Benchmarks, verification, merges | orchestrator |

## 7. Status (2026-10-26)
| Gap | Status | Where |
|---|---|---|
| G1 matmul | Done: own BLIS-style GEMM (AVX2/FMA 6x8 f64, 6x16 f32), rayon tasks over cache blocks, fused epilogue; `gemm`, `matmul_into`, `matmul_map`; matrixmultiply removed | `linalg/gemm/` |
| G2 axis reductions, G3 strided copy, G4 broadcast, G12 thresholds | Done, thresholds measured | `kernels_mem.rs`, `reduce.rs` |
| G11 allocations | Done: owned-buffer reuse, `*_inplace` / `into_*`, buffer pool (`np::memory`, default cap 512 MiB, `RUSTNUM_POOL_BYTES`), `tests/alloc.rs` | `pool.rs`, `memory.rs`, `ops.rs`, `ufunc.rs` |
| G6 SIMD math | Wired for f64 exp/exp2/log/sin/cos/tanh/sigmoid and f32 exp; measured 1–3 ulp; expm1/log2/log10/log1p/tan and most f32 still scalar | `simd_math.rs`, `ufunc.rs` |
| G10 RNG | Done: ziggurat, deterministic parallel fill (seed-only, thread-count independent) | `random/engine.rs` |
| G5 sort | Done: NaN partition, parallel and radix sort | `manip.rs` |
| G7–G9 blocked LU/QR/Cholesky, bidiag/tridiag SVD/eigh | Done: blocked LU (panel 32/64), blocked Cholesky, compact-WY QR, Householder tridiagonalisation + implicit QL (`tql2`) eigh, bidiagonalisation + Golub–Kahan SVD (no rows x rows buffer), O(n³) null-space completion; trailing updates and back-transformation through the packed GEMM. Measured under load vs NumPy: svd 100² 3.9 vs 8.1 ms, eigh 500² 160 vs 132 ms, qr 1000² 159 vs 253 ms, cholesky 1000² 32 vs 22 ms, solve 1000² 31 vs 30 ms. Open: svd 500² 391 vs 121 ms (Givens updates are scalar; needs D&C or blocked rotations) | `linalg/kernels.rs` |

Verification notes: the agent reports for G6/G10 were wrong in several places (an AVX-512 instruction on the AVX2 path,
false accuracy figures, a non-compiling test); all were fixed and re-measured by the orchestrator before merging.
