# rust-ml-stack roadmap: the whole ML and AI stack in Rust, zero to hero

**Goal.** Rebuild the Python ML/AI ecosystem in Rust, every layer, one crate at a time, each crate taken from zero
(an empty folder) to hero (the fastest, best-documented option for its job, usable from Rust and Python).

**Working name.** `rust-ml-stack` (umbrella crate) and `rust-ml-*` (member crates). Checked on crates.io on
2026-10-26: all 63 `rust-ml-*` names are free, and so are `rust-ml-stack`, `rust-ml-models`, `rust-ml-kit` and
`rust-ml-suite`. `rust-ml` itself is taken (v0.1.5, last release April 2025), so it cannot be the umbrella.
Users write `use rust_ml_stack as ml;`. `rustnum` keeps its name (renaming to `rust-ml-num` is the owner's call).

**Count.** 65 crates in 8 tiers, built in 6 waves. `rustnum` is done (v0.1).

Related: research and evidence in `docs/research/rust-native-ml-stack.html`; the per-crate method in the
`rust-ml-library` skill (`~/.claude/skills/rust-ml-library/`); rustnum's plan and contract in `docs/PLAN.md` and
`docs/API.md` (the template every crate follows).

---

## 1. Rules for every crate

### 1.1 What we build and what we depend on
- **We build** every ML, data-science and AI library: anything that has a Python counterpart in the ML stack.
  This includes the tensor engine, autodiff, GPU kernel language, dataframes, tokenizers and the LLM server.
- **We depend on** general-purpose Rust infrastructure only:
  `rayon`, `wgpu` (GPU API), `half`, `bytemuck`, `serde`/`serde_json`, `syn`/`quote`/`proc-macro2`, `pyo3`,
  `tokio`/`axum` (servers), image/audio codecs (`png`, `zune-jpeg`, `symphonia`), `criterion`, `proptest`.
  Adding anything else needs an entry in `docs/DEPENDENCIES.md` with the reason.
- **Boundary (not Rust, not rewritten):** GPU drivers and their user-mode APIs (CUDA driver, ROCm/HIP, Metal,
  Vulkan, DirectX), and rustc's LLVM backend. CUDA is reached through the driver API over FFI; that call is the line.
- `matrixmultiply` (used by rustnum today) is replaced by our own GEMM from `rust-ml-kernels` in Wave 1.

### 1.2 Zero-to-hero levels (a crate's status is always one of these)
| Level | Name | Done when |
|---|---|---|
| H0 | Contract | `docs/<crate>/PLAN.md` (research table, Python vs Rust table, scope) and `API.md` merged |
| H1 | Works | Core API implemented, gates green, parity fixtures from the Python library pass |
| H2 | Complete | Every item in the v0.1 contract exists with a runnable doctest; examples run; mapping doc + `llms.txt` section |
| H3 | Fast | Benchmarks vs the Python library on an idle machine, equal thread counts; every row within 1.5x or faster |
| H4 | Hero | Python bindings in `rust-ml-py`, published to crates.io and PyPI, used by at least two other crates in the workspace, faster than the Python library on its headline benchmark |

### 1.3 The method (same as rustnum)
Research → PLAN.md + API.md → scaffold → core first → other areas kernel-first in parallel worktrees → wiring →
parity tests against the real Python library → idle-machine benchmarks → docs. Full playbook: the
`rust-ml-library` skill. Agents: agy first, then codex, then Claude subagents; `.cargo/config.toml` with `jobs = 3`.

### 1.4 Gates (every crate, every merge)
`cargo fmt --check` · `cargo clippy --all-targets --all-features -D warnings` · `cargo test --all-features` ·
`cargo test --no-default-features` · `cargo doc --no-deps` · parity M/M · every example runs.

### 1.5 Repo layout
One Cargo workspace (this repo). `rustnum/` stays at the root; every new crate goes in `crates/<name>/`.
Shared Python fixture generators in `tools/<crate>/`. One `docs/<crate>/` folder per crate.

---

## 2. The crates

Size: S (days), M (about rustnum v0.1: ~1–2 days with the agent fleet), L (2–4x M), XL (a project of its own).
"Parity source" is the Python library whose outputs the tests compare against. "Headline benchmark" is the H3/H4 target.

### Tier 0: scientific foundations (NumPy, SciPy, pandas, matplotlib, NetworkX)
| # | Crate | Python counterpart | v0.1 scope | Depends on | Parity source | Headline benchmark | Size |
|---|---|---|---|---|---|---|---|
| 1 | `rustnum` ✅ H2 | NumPy | done; next: own GEMM, blocked LU/QR/SVD, SIMD ufuncs, parallel sort | rust-ml-kernels (CPU) | NumPy | README table within 1.5x | M |
| 2 | `rust-ml-sparse` | scipy.sparse | CSR/CSC/COO/DIA, sparse × dense, sparse × sparse, CG/GMRES/BiCGSTAB, sparse LU, reorderings | rustnum | SciPy | SpMV, sparse solve | L |
| 3 | `rust-ml-special` | scipy.special | gamma, beta, erf, Bessel, zeta, hypergeometric, digamma, logsumexp, softmax | rustnum | SciPy | vectorised erf/gammaln | M |
| 4 | `rust-ml-stats` | scipy.stats | 100+ distributions (pdf/cdf/ppf/rvs), hypothesis tests, KDE, correlation | rust-ml-special | SciPy | rvs and cdf throughput | L |
| 5 | `rust-ml-optimize` | scipy.optimize | BFGS, L-BFGS-B, Nelder-Mead, trust-region, least_squares, root finding, LP (simplex, interior point), QP | rustnum, rust-ml-sparse | SciPy | L-BFGS on Rosenbrock 1e4 | L |
| 6 | `rust-ml-signal` | scipy.signal | filters (IIR/FIR design, filtfilt), spectral (welch, spectrogram), resample, windows, convolution | rustnum | SciPy | filtfilt, welch | M |
| 7 | `rust-ml-ode` | scipy.integrate + interpolate | RK45, DOP853, BDF, Radau, quad, splines, RBF interpolation | rustnum | SciPy | stiff ODE solve | M |
| 8 | `rust-ml-frame` | pandas | DataFrame/Series, dtypes incl. nullable and categorical, groupby, merge/join, pivot, resample, rolling, CSV/JSON/Parquet IO, Arrow-compatible memory | rustnum | pandas | groupby-agg, join, CSV read | XL |
| 9 | `rust-ml-plot` | matplotlib | figure/axes model, line/scatter/bar/hist/image, SVG and PNG output, text layout | rustnum | visual goldens | render 1e6-point scatter | L |
| 10 | `rust-ml-viz` | seaborn | distribution, relational, categorical, matrix (heatmap, clustermap), regression plots, themes | rust-ml-plot, rust-ml-frame, rust-ml-stats | seaborn goldens | pairplot on 1e5 rows | M |
| 11 | `rust-ml-graph` | NetworkX | graph types, traversal, shortest paths, centrality, community detection, flows, generators | rustnum, rust-ml-sparse | NetworkX | PageRank 1e6 edges | L |

### Tier 1: compute engine (the PyTorch/JAX core)
| # | Crate | Python counterpart | v0.1 scope | Depends on | Parity source | Headline benchmark | Size |
|---|---|---|---|---|---|---|---|
| 12 | `rust-ml-kernels` | Triton, CUDA C++ | kernel DSL as a proc-macro: one Rust kernel → CPU SIMD, WGSL (wgpu) and PTX (CUDA); tiling, shared memory, tensor-core MMA; autotuning; own GEMM | `syn`, `wgpu` | NumPy/PyTorch results | GEMM vs OpenBLAS (CPU) and cuBLAS (GPU) | XL |
| 13 | `rust-ml-gpu` | CUDA runtime, torch.cuda | device runtimes: wgpu (Vulkan, Metal, DX12, WebGPU), CUDA driver API, ROCm/HIP; caching allocator, streams, events, multi-device | rust-ml-kernels | n/a | allocator and launch overhead | L |
| 14 | `rust-ml-tensor` | torch.Tensor | device-agnostic tensor, dtypes f64/f32/f16/bf16/int8/fp8/bool, strided views, broadcasting, lazy op graph, CPU and GPU ops | rust-ml-gpu, rustnum | PyTorch | element-wise, reductions, matmul | XL |
| 15 | `rust-ml-autograd` | torch.autograd, jax.grad | reverse and forward mode, higher-order grads, checkpointing, custom backward, `grad`/`vjp`/`jvp` | rust-ml-tensor | PyTorch gradients | backward pass of an MLP | L |
| 16 | `rust-ml-compile` | torch.compile, jax.jit, XLA | graph capture, fusion, memory planning, autotuned kernels, `jit` and `vmap` transforms | rust-ml-tensor, rust-ml-kernels | eager results | fused transformer block | XL |
| 17 | `rust-ml-dist` | NCCL, torch.distributed | collectives (all-reduce, all-gather, reduce-scatter, broadcast) over GPU peer-to-peer, TCP and RDMA; process groups | rust-ml-gpu | NCCL bandwidth numbers | all-reduce bus bandwidth | XL |

### Tier 2: deep-learning framework
| # | Crate | Python counterpart | v0.1 scope | Depends on | Parity source | Headline benchmark | Size |
|---|---|---|---|---|---|---|---|
| 18 | `rust-ml-nn` | torch.nn | Module system, Linear, Conv1-3d, norms, attention, RNN/LSTM/GRU, embeddings, dropout, losses, init | rust-ml-autograd | PyTorch | ResNet-50 forward/backward | L |
| 19 | `rust-ml-optim` | torch.optim | SGD, Adam, AdamW, Lion, Adafactor, LR schedulers, gradient clipping, EMA | rust-ml-autograd | PyTorch | optimizer step on 1e8 params | S |
| 20 | `rust-ml-data` | torch.utils.data | Dataset, DataLoader, samplers, sharding, multi-threaded prefetch to GPU, augmentation pipeline | rust-ml-tensor, rust-ml-frame | PyTorch | images/s from disk | M |
| 21 | `rust-ml-train` | Lightning, Accelerate | training loop, mixed precision, gradient accumulation, checkpoints, callbacks, logging | rust-ml-nn, rust-ml-optim, rust-ml-data | n/a | tokens/s on GPT-2 small | M |
| 22 | `rust-ml-quant` | bitsandbytes, torch.ao | PTQ, QAT, int8/int4/fp8, GPTQ, AWQ, pruning, distillation | rust-ml-nn | PyTorch/bitsandbytes | int4 matmul throughput | L |
| 23 | `rust-ml-onnx` | onnx, onnxruntime | ONNX import and export, graph optimisation, run via rust-ml-tensor | rust-ml-nn | onnxruntime | ResNet-50 inference | L |

### Tier 3: classical ML (scikit-learn, XGBoost, statsmodels, Optuna)
| # | Crate | Python counterpart | v0.1 scope | Depends on | Parity source | Headline benchmark | Size |
|---|---|---|---|---|---|---|---|
| 24 | `rust-ml-learn` | sklearn base | Estimator/Transformer/Predictor traits, preprocessing, pipelines, metrics, CV, grid/random search | rustnum, rust-ml-frame | scikit-learn | pipeline fit on 1e6 rows | M |
| 25 | `rust-ml-linear` | sklearn.linear_model | OLS, ridge, lasso, elastic net, logistic, GLMs, SGD models | rust-ml-learn, rust-ml-optimize | scikit-learn | logistic on 1e6×100 | M |
| 26 | `rust-ml-tree` | sklearn.tree/ensemble | CART, random forest, extra trees, isolation forest | rust-ml-learn | scikit-learn | RF fit 1e6 rows | M |
| 27 | `rust-ml-boost` | XGBoost, LightGBM, CatBoost | histogram GBM on CPU and GPU, categorical support, early stopping, SHAP values | rust-ml-tree, rust-ml-kernels | XGBoost/LightGBM | training time at matched AUC | XL |
| 28 | `rust-ml-neighbors` | sklearn.neighbors | kNN, radius neighbours, KD/ball tree, exact and approximate search | rust-ml-learn | scikit-learn | kNN query 1e6×128 | M |
| 29 | `rust-ml-svm` | sklearn.svm, libsvm | SVC, SVR, linear SVM, kernels, one-class | rust-ml-learn | scikit-learn | SVC fit 1e5 rows | M |
| 30 | `rust-ml-cluster` | sklearn.cluster | k-means(++), mini-batch, DBSCAN, HDBSCAN, GMM, agglomerative, spectral | rust-ml-learn | scikit-learn | k-means 1e7 points | M |
| 31 | `rust-ml-decomp` | sklearn.decomposition/manifold, umap | PCA, randomized SVD, NMF, ICA, LDA, t-SNE, UMAP | rust-ml-learn, rust-ml-sparse | scikit-learn, umap-learn | UMAP 1e6 points | L |
| 32 | `rust-ml-prob` | PyMC, NumPyro, Stan | probabilistic programs, NUTS/HMC, variational inference, diagnostics | rust-ml-autograd, rust-ml-stats | PyMC | effective samples/s | XL |
| 33 | `rust-ml-statsmodels` | statsmodels, lifelines | OLS/GLM inference tables, ANOVA, mixed models, survival (Kaplan-Meier, Cox) | rust-ml-stats, rust-ml-linear | statsmodels | GLM fit with summary | L |
| 34 | `rust-ml-timeseries` | statsforecast, sktime, Prophet | ARIMA, ETS, Theta, Prophet-style, cross-validation for series | rust-ml-statsmodels | statsforecast | AutoARIMA on 1e4 series | L |
| 35 | `rust-ml-recsys` | implicit, LightFM | ALS, BPR, factorization machines, two-tower retrieval | rust-ml-sparse, rust-ml-nn | implicit | ALS on 1e8 interactions | M |
| 36 | `rust-ml-explain` | SHAP, LIME | TreeSHAP, KernelSHAP, LIME, permutation importance, partial dependence | rust-ml-learn, rust-ml-boost | shap | TreeSHAP on 1e5 rows | M |
| 37 | `rust-ml-tune` | Optuna | TPE, CMA-ES, random/grid, pruning, parallel trials, persistence | rustnum | Optuna | trials/s, best value reached | M |
| 38 | `rust-ml-auto` | AutoGluon, FLAML | `Classifier::auto` / `Regressor::auto` / `Forecaster::auto` across classical and deep models, ensembling | all of tier 3, rust-ml-nn | AutoGluon | score at fixed time budget | L |

### Tier 4: domain deep learning
| # | Crate | Python counterpart | v0.1 scope | Depends on | Parity source | Headline benchmark | Size |
|---|---|---|---|---|---|---|---|
| 39 | `rust-ml-image` | OpenCV, Pillow | decode/encode, colour spaces, resize, filters, morphology, geometric transforms, contours | rustnum, codecs | OpenCV | resize and blur 4K | L |
| 40 | `rust-ml-vision` | torchvision, timm, ultralytics | transforms, ResNet/ConvNeXt/ViT, detection (YOLO-family, DETR), segmentation (U-Net, SAM-style) | rust-ml-nn, rust-ml-image | torchvision | images/s train and infer | XL |
| 41 | `rust-ml-audio` | torchaudio, librosa | STFT, mel, MFCC, resampling, Whisper-style ASR, TTS models | rust-ml-signal, rust-ml-nn | librosa, torchaudio | mel spectrogram throughput | L |
| 42 | `rust-ml-gnn` | PyG, DGL | message passing, GCN/GAT/GraphSAGE, sampling, batching | rust-ml-nn, rust-ml-sparse, rust-ml-graph | PyG | epoch time on ogbn-arxiv | L |
| 43 | `rust-ml-diffusion` | diffusers | schedulers, UNet, DiT, VAE, text-to-image pipeline, training | rust-ml-nn, rust-ml-transformers | diffusers | images/s at 512² | L |
| 44 | `rust-ml-sciml` | torchdiffeq, DeepXDE | neural ODEs (adjoint), PINNs, operator learning | rust-ml-ode, rust-ml-autograd | torchdiffeq | adjoint solve time | M |
| 45 | `rust-ml-gym` | Gymnasium, PettingZoo | environment API, classic control, vectorised envs, wrappers | rustnum | Gymnasium | env steps/s | M |
| 46 | `rust-ml-rl` | Stable-Baselines3, CleanRL | PPO, DQN, SAC, TD3, A2C, replay buffers | rust-ml-nn, rust-ml-gym | SB3 | reward vs wall time | L |

### Tier 5: transformers and LLMs
| # | Crate | Python counterpart | v0.1 scope | Depends on | Parity source | Headline benchmark | Size |
|---|---|---|---|---|---|---|---|
| 47 | `rust-ml-formats` | safetensors, gguf, onnx files | safetensors, GGUF, npy/npz, checkpoint sharding, HF Hub download | rustnum | reference files | load 70B shards | S |
| 48 | `rust-ml-tokenize` | HF tokenizers, tiktoken | BPE, WordPiece, Unigram, byte-level, `tokenizer.json` and tiktoken compatibility, training | none | HF tokenizers | GB/s encode | M |
| 49 | `rust-ml-attention` | FlashAttention, xformers | FlashAttention fwd/bwd, paged attention, RoPE/ALiBi, GQA/MQA, KV-cache kernels | rust-ml-kernels | FlashAttention | TFLOP/s per GPU | L |
| 50 | `rust-ml-transformers` | HF transformers | model zoo: Llama, Qwen, Gemma, Mistral, Phi, BERT, T5, ViT, CLIP, Whisper; generation, sampling | rust-ml-nn, rust-ml-attention, rust-ml-tokenize, rust-ml-formats | HF transformers logits | tokens/s generation | XL |
| 51 | `rust-ml-serve` | vLLM, SGLang, TGI | continuous batching, paged KV cache, prefix caching, speculative decoding, structured output, OpenAI-compatible API | rust-ml-transformers | vLLM | throughput at fixed latency | XL |
| 52 | `rust-ml-peft` | peft | LoRA, QLoRA, DoRA, adapters, prefix tuning, merging | rust-ml-nn, rust-ml-quant | peft | fine-tune memory and speed | M |
| 53 | `rust-ml-align` | TRL, OpenRLHF | SFT, DPO, ORPO, PPO-RLHF, GRPO, reward models | rust-ml-train, rust-ml-peft | TRL | samples/s | L |
| 54 | `rust-ml-parallel` | DeepSpeed, FSDP, Megatron | FSDP/ZeRO 1-3, tensor and pipeline parallel, sequence parallel, MoE expert parallel | rust-ml-dist, rust-ml-train | DeepSpeed | MFU on 8 GPUs | XL |
| 55 | `rust-ml-embed` | sentence-transformers | embedding models, pooling, contrastive training, rerankers | rust-ml-transformers | sentence-transformers | sentences/s | M |
| 56 | `rust-ml-eval` | lm-eval-harness | benchmark tasks (MMLU, GSM8K, HumanEval…), metrics, harness | rust-ml-transformers | lm-eval-harness | same scores | M |

### Tier 6: AI applications
| # | Crate | Python counterpart | v0.1 scope | Depends on | Parity source | Headline benchmark | Size |
|---|---|---|---|---|---|---|---|
| 57 | `rust-ml-vector` | FAISS, hnswlib | HNSW, IVF, PQ/OPQ, flat, GPU search, filtering | rust-ml-kernels | FAISS recall | QPS at recall 0.95 | L |
| 58 | `rust-ml-rag` | LlamaIndex | loaders, chunking, hybrid retrieval, reranking, pipelines | rust-ml-vector, rust-ml-embed | LlamaIndex | end-to-end latency | M |
| 59 | `rust-ml-agents` | LangGraph, smolagents | agent loop, tools, MCP client/server, memory, multi-agent graphs | rust-ml-serve | n/a | task success, latency | M |
| 60 | `rust-ml-privacy` | Opacus, Flower | DP-SGD, privacy accounting, federated averaging | rust-ml-train | Opacus | DP-SGD overhead | M |

### Tier 7: tooling and delivery
| # | Crate | Python counterpart | v0.1 scope | Depends on | Parity source | Headline benchmark | Size |
|---|---|---|---|---|---|---|---|
| 61 | `rust-ml-deploy` | BentoML, ONNX export | versioned model packages, native/WASM/`no_std` runtimes, HTTP server generation | rust-ml-tensor, rust-ml-learn | n/a | cold start, binary size | M |
| 62 | `rust-ml-py` | (bindings) | PyO3 wheels for every H3 crate, NumPy/pandas zero-copy interop | all | Python test suites | call overhead | L |
| 63 | `rust-ml-track` | MLflow, W&B | runs, metrics, artifacts, model registry, web UI, drift monitoring | rust-ml-frame, axum | n/a | log rate | M |
| 64 | `rust-ml-notebook` | Jupyter kernel | Jupyter kernel with rich display (tables, plots, tensors) | rust-ml-plot | n/a | cell latency | M |
| 65 | `rust-ml-stack` | (umbrella) | one dependency with features (`sci`, `frame`, `learn`, `deep`, `gpu`, `llm`, `py`) | all | n/a | n/a | S |

---

## 3. Build order: six waves

Crates inside a wave are independent of each other and run in parallel worktrees; a wave starts when the crates it
depends on reach H1. "Critical path" marks the crates that block the most others.

| Wave | Crates | Critical path |
|---|---|---|
| 1. Foundations | rustnum (H3 work), rust-ml-kernels, rust-ml-gpu, rust-ml-special, rust-ml-sparse, rust-ml-formats, rust-ml-tokenize | **rust-ml-kernels → rust-ml-gpu** (all GPU work waits on them) |
| 2. Core engines | rust-ml-tensor, rust-ml-stats, rust-ml-optimize, rust-ml-signal, rust-ml-ode, rust-ml-frame, rust-ml-plot, rust-ml-graph, rust-ml-learn | **rust-ml-tensor** (all deep learning waits on it) |
| 3. Frameworks | rust-ml-autograd, rust-ml-nn, rust-ml-optim, rust-ml-data, rust-ml-linear, rust-ml-tree, rust-ml-neighbors, rust-ml-svm, rust-ml-cluster, rust-ml-decomp, rust-ml-viz, rust-ml-image, rust-ml-gym, rust-ml-tune | **rust-ml-autograd** |
| 4. Systems | rust-ml-compile, rust-ml-train, rust-ml-quant, rust-ml-onnx, rust-ml-boost, rust-ml-statsmodels, rust-ml-prob, rust-ml-recsys, rust-ml-explain, rust-ml-attention, rust-ml-vector, rust-ml-deploy | **rust-ml-attention**, **rust-ml-boost** |
| 5. Models and apps | rust-ml-transformers, rust-ml-vision, rust-ml-audio, rust-ml-gnn, rust-ml-sciml, rust-ml-rl, rust-ml-timeseries, rust-ml-auto, rust-ml-embed, rust-ml-rag, rust-ml-py, rust-ml-track, rust-ml-notebook | **rust-ml-transformers** |
| 6. Scale | rust-ml-dist, rust-ml-parallel, rust-ml-serve, rust-ml-peft, rust-ml-align, rust-ml-diffusion, rust-ml-eval, rust-ml-agents, rust-ml-privacy, rust-ml-stack | **rust-ml-dist → rust-ml-parallel** |

Every crate keeps climbing H1 → H4 after its wave; waves describe when a crate can start, not when it stops.

---

## 4. Per-crate checklist (copy into `docs/<crate>/STATUS.md`)
```
[ ] H0  research table (prior art in Rust + Python internals), PLAN.md, API.md (one line per item, decisions made)
[ ] H0  name, crate folder under crates/, workspace member, .cargo jobs=3
[ ] H1  core implemented by agent; verified by orchestrator (gates + audit_api.sh)
[ ] H1  tools/<crate>/gen_fixtures.py + tests/parity.rs, M/M passing
[ ] H2  every API item + doctest; examples; docs/<crate>/MAPPING.md; llms.txt section
[ ] H3  benches/ + tools/<crate>/bench_python.py; idle-machine table in docs/<crate>/BENCH.md
[ ] H4  rust-ml-py bindings, crates.io + PyPI release, used by two workspace crates
```

---

## 5. What happens first (Wave 1 kickoff)

| Crate | First step | Agent |
|---|---|---|
| rust-ml-kernels | Research CubeCL, Triton, cutile-rs designs; PLAN.md + API.md for the kernel DSL; CPU SIMD backend and WGSL backend first, PTX second | agy (Gemini 3.8 High) for scaffold, codex for the proc-macro |
| rust-ml-gpu | wgpu runtime + allocator + launch path; CUDA driver API FFI second | codex |
| rustnum H3 | own GEMM (via rust-ml-kernels CPU), blocked LU/QR, one-sided Jacobi → divide-and-conquer SVD, SIMD exp/log, parallel sort | agy |
| rust-ml-special | scipy.special parity fixtures, Cephes-quality algorithms | Claude subagent |
| rust-ml-sparse | CSR/CSC/COO + SpMV + iterative solvers | Claude subagent |
| rust-ml-formats | safetensors + GGUF + HF Hub download | Claude subagent |
| rust-ml-tokenize | BPE/WordPiece/Unigram + `tokenizer.json` parity with HF tokenizers | agy |

## 6. Decisions the owner makes
1. Project name: keep `rust-ml-stack` / `rust-ml-*` or choose another.
2. GitHub remote and whether to publish to crates.io/PyPI at H4 (publishing reserves the names).
3. Whether `rustnum` is renamed to `rust-ml-num` for consistency (default: keep `rustnum`).
