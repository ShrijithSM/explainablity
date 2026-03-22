# Chronoscope Developer Log

## 2026-03-06

### Major Updates
- **Fixed 4-bit Loading Error**: Resolved `ValueError: .to is not supported` by disabling quantization for the 0.5B model. For small models, `bitsandbytes` + `accelerate` conflicts are avoided by loading in full/half precision (still fits easily in 6GB VRAM).
- **Exp2 Live Dashboard**: Ported the real-time visualization logic from `exp4` to `exp2` (`exp2_live_heatmap.py`). 
    - Added a live-updating `rich.Table` heatmap.
    - Added an interactive loop to allow multiple analysis runs without restarting the script.
    - Added `-p` / `--prompt` CLI support to collect input before initializing the heavy TUI.
- **Random Patching Noise**: Set the default `patching_noise` in `config.py` to `gaussian` per user request ("keep it random").
- **Terminal Optimization**: Addressed terminal "spam" by keeping a single persistent process for live experiments and using `Stop-Process` to clean up stale GPU/Python instances.

### Technical Decisions
- **Quantization Policy**: For models < 1B parameters, Chronoscope defaults to FP16/BF16 on 6GB+ cards to maximize stability. Quantization is reserved for 7B+ models or ultra-low VRAM scenarios.
- **UI Flow**: Prompt collection is moved *before* model loading. This minimizes the "frozen" state of the terminal during startup.
- **Divergence Metric**: Using normalized L2 distance for the heatmap, with intensity-based color mapping (White → Blue → Green → Yellow → Red).

## 2026-03-10

### Major Updates
- **Eager Graph Engine**: Implemented `chronoscope/graph.py` as a lightweight, immediate-execution workflow runner. This replaces the need for compiled graphs (like LangGraph) while maintaining modularity.
- **NEXUS Structural Layer**: Integrated advanced graph theory concepts into the interpretability pipeline:
    - **Hypergraph Extraction**: Added `extract_hyperedges` to `analyzer.py` to identify latent motifs in the residual stream.
    - **Isomorphic Mapping**: Added `detect_isomorphic_clusters` to map motifs across different layers to abstract structural principles.
- **Exp5 Eager NEXUS**: Created a new experiment (`exp5_eager_nexus.py`) that demonstrates the full eager graph pipeline with structural mapping.
- **Report Synthesis**: Updated `synthesizer.py` to include a dedicated section for structural motifs and isomorphic clusters.

### Technical Decisions
- **Eager vs. Compiled**: Opted for a custom eager engine in `explainablity` to allow for high-frequency feedback and manual control over the interpretability trace as it generates.
- **Principle Mapping**: Used a token-overlap and latent-similarity hybrid for isomorphic matching across layers.

### Ongoing Tasks
- [ ] Add graceful error handling for out-of-memory (OOM) during long generation traces.
- [ ] Implement incremental SVD update if trajectory length exceeds `max_cache_size`.
- [ ] Integrate a small local LLM for automatic labeling of hyperedge "principles".

## 2026-03-14

### Major Updates
- **Gaps A–E: Full Statistical Rigor Pipeline** — `analyzer.py` grew from ~400 lines to **1791 lines**, implementing the five remaining analytical gaps:
    - **Gap A (Stationarity & Cointegration):** Per-head ADF testing (`_test_per_head_stationarity`), selective first-differencing (`_apply_selective_differencing`), Johansen cointegration test (`_check_cointegration`), and automatic VECM fallback (`_fit_vecm`). Joint ADF+KPSS stationarity test added to `observer.py` with a 4-diagnosis interpretation table (stationary / unit_root / fractional / ambiguous).
    - **Gap B (Statistical Significance):** Granger F-test p-value matrix (`_granger_pvalue_matrix`), Benjamini-Hochberg FDR correction (`_apply_fdr_correction`), phase-scramble bootstrap surrogates (`_bootstrap_surrogate_pvalues`), conditional transfer entropy (`_conditional_transfer_entropy`), and Partial Directed Coherence (`_partial_directed_coherence`).
    - **Gap C (Perturbation / Intervention):** Migrated from manual hook-based ablation to **pyvene `IntervenableModel`** (`_make_ablation_model`). Added mean-ablation cache across reference prompts (`_build_mean_ablation_cache`), activation patching with KL restoration scoring (`_activation_patch_experiment`), and direct vs. total effect mediation analysis (`_direct_vs_total_effect`).
    - **Gap D (Thinking Time Axis):** CoT step segmenter (`cot_segmenter.py`) with heuristic boundary detection (explicit markers + sentence boundaries), character→token mapping, and per-step entropy aggregation. Arc-length intrinsic time (`compute_intrinsic_time`) with discrete curvature and high-curvature flagging. Topological phase clock via Euler characteristic spike detection. HMM phase discovery (optional, gated behind `use_hmm_phase_discovery`).
    - **Gap E (Signal Quality):** Multi-metric head computation in `interceptor.py` — Shannon entropy, Rényi-2 entropy, max attention, effective rank, sink fraction, variance, kurtosis, top-3/5 mass, argmax position norm, attention spread std, and Gini concentration. Attention sink auto-detection and removal with re-normalization. Scalar vs. 12-dim vector feature modes.
- **DashboardBridge** — New module `dashboard_bridge.py` (573 lines) implementing a live streaming bridge between the Python analysis runtime and a browser dashboard. Supports WebSocket (primary), file polling, and JS injection transports. Pushes per-token frames, VAR/FDR influence frames, perturbation results, HMM phase frames, TDA topology frames, signal quality snapshots, and composite score frames. Includes a built-in HTTP server for serving the dashboard HTML.
- **Exp6 (Head Interference)** — New experiment `exp6_head_interference.py` (25.9KB) tracking localized attention head interference patterns across layers, with cross-layer aggregation pushed through the dashboard bridge.

### Technical Decisions
- **pyvene over manual hooks for Gap C:** `pyvene.IntervenableModel` provides type-safe zero/mean/gaussian/vanilla interventions and `CollectIntervention` for building mean-ablation caches. Manual `_make_ablation_hook` is now skipped in tests (`@pytest.mark.skip`).
- **Config-gated features:** Expensive analyses (bootstrap surrogates, transfer entropy, PDC, HMM, activation patching) are gated behind boolean config flags defaulting to `False`, keeping the default pipeline fast.
- **12-feature head vector mode:** When `head_feature_mode='vector'`, each head produces a 12-dimensional feature vector per timestep instead of a scalar, enabling richer downstream VAR analysis on attention dynamics.

## 2026-03-19

### Major Updates
- **Test Suite** — Created `chronoscope/tests/` with three focused test modules:
    - `test_stationarity.py`: Tests for per-head ADF (`_test_per_head_stationarity`), selective differencing (`_apply_selective_differencing`), and joint ADF+KPSS (`joint_stationarity_test`) on synthetic stationary/non-stationary/mixed signals. Environment-guarded against the known `deprecated_kwarg()` statsmodels bug.
    - `test_significance.py`: Tests Granger p-value matrix shape, true-causal detection on synthetic VAR(1) data, and BH-FDR correction behavior (marginal rejection, strong-signal survival, sorted pairs).
    - `test_perturbation.py`: Legacy ablation hook tests (now skipped for pyvene migration). Live tests for `CoTSegmenter` — basic segmentation, step coverage, and aggregation shape.
- **Context Document** — Created `context.md` as a 15-minute orientation guide covering project identity, architecture, all 5 analysis pillars (SVD, FFT/ACF/ADF, TDA, Causal/VAR, NEXUS), experiment inventory, and stability/reporting.

### Technical Decisions
- **Statsmodels guard pattern:** `_statsmodels_broken` flag at test-module level tries a probe `adfuller()` call and skips the entire class if a `TypeError` is raised. This handles the known incompatibility without masking real failures.

## 2026-03-22

### Major Updates
- **PyTorch API Integration** — Replaced NumPy/SciPy implementations with PyTorch equivalents in performance-critical paths:
    - `observer.py` → `compute_intrinsic_time`: Arc-length steps, cumulative intrinsic time, and discrete curvature now use `torch.linalg.norm`, `torch.cumsum`, and `torch.as_tensor` for GPU-friendly computation. Falls back to `.cpu().numpy()` only at the return boundary.
    - `analyzer.py` → `_test_per_head_stationarity`: Column-mean centering via `torch.mean` before passing to statsmodels ADF.
    - `analyzer.py` → `_apply_selective_differencing`: `torch.diff` + `torch.where` with broadcast mask replaces the manual NumPy loop.
    - `analyzer.py` → `_bootstrap_surrogate_pvalues`: Batch phase-scramble surrogates generated with `torch.fft.rfft` → `torch.polar` → `torch.fft.irfft` in a single `[S, T, H]` tensor. VAR coefficient norms computed via `torch.linalg.norm`.
    - `interceptor.py` → `_compute_head_metrics`: Effective rank now uses `torch.linalg.svdvals` on the full `[H, T, T]` attention matrix. Shannon entropy via `torch.special.entr`, Rényi-2 via `torch.log`, variance/kurtosis via `torch.var` and moment calculations — all fused on-device before a single `.cpu().numpy()` transfer.
- **Integration Hub** — `integration_hub/` directory with a full-stack structure (backend/, frontend/, configs/, docs/, scripts/, shared/). Includes `chronoscope_v2.html` (124KB), a comprehensive live WebSocket dashboard for real-time interpretability visualization.
- **Package Init Updated** — `chronoscope/__init__.py` now exports `DashboardBridge` as a first-class public API alongside the existing `ChronoscopeConfig`, `load_model`, `ChronoscopeInterceptor`, `SignalObserver`, `CausalAnalyzer`, and `ReportSynthesizer`.

### Technical Decisions
- **PyTorch boundary strategy:** Tensor operations stay on-device as long as possible. `.cpu().numpy()` conversions happen only at function return boundaries or when feeding into statsmodels/scipy APIs that require NumPy. This maximizes fused kernel benefits while maintaining API compatibility.
- **`torch.polar` for surrogate generation:** Using `torch.polar(abs, angle)` to construct complex tensors for phase-scramble surrogates avoids manual `cos/sin` + complex construction, leveraging PyTorch's optimized complex arithmetic.

### Ongoing Tasks
- [x] ~~Implement Gaps A–E statistical rigor pipeline~~ ✅
- [x] ~~Add formal test suite for core analysis methods~~ ✅
- [x] ~~Migrate performance-critical NumPy paths to PyTorch~~ ✅
- [x] ~~Build live dashboard bridge with WebSocket transport~~ ✅
- [ ] Add graceful error handling for OOM during long generation traces.
- [ ] Implement incremental SVD update if trajectory length exceeds `max_cache_size`.
- [ ] Integrate a small local LLM for automatic labeling of hyperedge "principles".
- [ ] Run full regression test suite post-PyTorch migration and fix any remaining failures.
- [ ] Profile PyTorch vs. NumPy paths end-to-end to quantify speedup on representative workloads.
- [ ] Add windowed TDA performance benchmarks (ripser is the current bottleneck).
