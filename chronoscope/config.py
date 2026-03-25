"""
Configuration for Chronoscope experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ChronoscopeConfig:
    """All experiment parameters in one place."""

    # --- Model ---
    model_name = "gpt2"  # Change to "Qwen/Qwen2.5-0.5B" when downloaded
    n_heads: int = 12  # GPT-2 has 12 heads; Qwen2.5-0.5B has 14
    hidden_dim: int = 768  # GPT-2: 768; Qwen2.5-0.5B: 896
    total_tokens: int = 100
    target_layer: int = 11  # GPT-2 has 12 layers (0-11); Qwen has 24 (0-23)
    local_model_snapshot_path: Optional[str] = None
    use_airllm: bool = False # Direct loading is now primary (resolves disk space issues)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_4bit: bool = False  # Only enable for 7B+ models on low VRAM
    torch_dtype: str = "float16"

    airllm_vram_headroom_gb: float = 1.0  # GB reserved for activations/buffers

    # --- Interceptor ---
    # Layer name substrings to hook into. These target the output of each
    # decoder layer, which is the residual stream.
    target_layers: List[str] = field(
        default_factory=lambda: ["layers.", "h."]  # Matches Qwen (layers.) and GPT-2 (h.)
    )
    max_cache_size: int = 1000  # Number of tokens to retain in CPU RAM during generation

    # Whether to derive per-head attention metrics (e.g., entropy) as an
    # additional multivariate time series for head–head interaction analysis.
    capture_attentions: bool = True
    head_metric: str = "entropy"  # currently: "entropy" over attention weights
    head_var_max_lag: int = 3     # maximum lag for VAR-based head interactions

    # --- Observer (Classical TS) ---
    svd_components: int = 8  # Reduce hidden_dim → 8 principal components
    fft_top_k: int = 5  # Report top-K dominant frequencies
    acf_max_lag: int = 20  # Max lag for autocorrelation

    # --- Analyzer (Causal) ---
    patching_noise: str = "gaussian"  # "zero", "mean", or "gaussian"
    dtw_radius: int = 5  # Sakoe-Chiba band radius for DTW speedup
    # Sliding-window TDA on the compressed trajectory for more local topology.
    tda_window_size: int = 10        # reduced from 40 for finer anomaly detection (Exp4)
    tda_distance_threshold: float = 2.0  # ripser distance threshold (Exp4)
    tda_window_stride: int = 20
    tda_enable_windowed: bool = True
    optimize_sweep: bool = True
    
    # ── Analysis modes (Exp3 → Exp6) ───────────────────────────────────────
    analyse_generated_only: bool = True   # exclude prompt tokens from VAR/TDA
    prompt_token_count: int = 0           # set at runtime from tokenizer

    # ── Exp6 layer/head selection ──────────────────────────────────────────
    exp6_top_k_layers: int = 3
    exp6_top_k_heads: int = 3

    # --- Synthesizer ---
    report_dir: str = "reports"
    save_plots: bool = True

    # --- Experiment ---
    max_new_tokens: int = 50
    dashboard_transport: str = "websocket"
    dashboard_ws_port: int = 8765
    dashboard_http_port: int = 8766
    dashboard_html_path: str = "integration_hub/frontend/public/chronoscope_live.html"
    dashboard_stat_every: int = 5             # signal quality cadence (tokens)
    dashboard_tda_every: int = 1              # TDA fires on spike (not cadenced)
    clean_prompt: str = (
        "All cats are animals. Whiskers is a cat. Therefore, Whiskers is a"
    )
    corrupted_prompt: Optional[str] = None  # Auto-generated if None

    # ── Gap 0: Sweep (Exp2, gated) ────────────────────────────────────────
    run_causal_sweep: bool = False            # gate for optional deep-dive
    causal_sweep_n_pairs: int = 30            # top-k pairs to perturb in sweep

    # ── Gap A: Stationarity & Cointegration ────────────────────────────────
    var_lag_selection: str = 'aic'            # 'aic' | 'bic' | 'fixed'
    var_max_lags: int = 5                     # ceiling for IC selection
    run_johansen_cointegration: bool = True   # VECM if cointegrated
    joint_stationarity_test: bool = True      # run both ADF + KPSS

    # ── Gap B: Statistical Significance ────────────────────────────────────
    granger_ftest: bool = True                # Granger F-test p-value matrix
    fdr_alpha: float = 0.05                   # BH-FDR target
    enable_bootstrap_surrogates: bool = False # expensive (500 VAR fits)
    n_bootstrap_surrogates: int = 500
    use_transfer_entropy: bool = False        # amber — model-free TE
    compute_pdc: bool = False                 # partial directed coherence

    # ── Gap C: Perturbation / Intervention ─────────────────────────────────
    perturbation_mode: str = 'zero'           # 'zero' | 'mean' | 'gaussian'
    reference_prompts: List[str] = field(default_factory=lambda: [
        "The capital of France is Paris.",
        "Solve for x: 2x + 3 = 7.",
        "Once upon a time in a distant land,",
        "The mitochondria is the powerhouse of the cell.",
        "import numpy as np",
        "Water boils at 100 degrees Celsius.",
        "The French Revolution began in 1789.",
        "To be or not to be, that is the question.",
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, a prime number is divisible only by 1 and itself.",
    ])
    run_activation_patching: bool = False
    activation_patch_clean: str = ""
    activation_patch_corrupted: str = ""

    # ── Gap D: Thinking Time Axis ──────────────────────────────────────────
    use_cot_time_axis: bool = False           # segment tokens into CoT steps
    cot_prompt_prefix: str = "Let's think step by step."
    use_topological_phase_clock: bool = True  # Euler characteristic phases
    euler_spike_threshold_std: float = 2.0
    use_arc_length_resampling: bool = False   # equal cognitive effort
    arc_length_n_points: int = 80
    use_hmm_phase_discovery: bool = False     # amber — requires hmmlearn
    hmm_n_states: int = 4

    # ── Gap E: Signal Quality ──────────────────────────────────────────────
    head_metric_type: str = 'shannon_entropy' # 'shannon_entropy' | 'renyi_entropy_2' | 'effective_rank'
    head_feature_mode: str = 'scalar'         # 'scalar' | 'vector' (5-dim)
    remove_attention_sink: bool = True        # remove BOS sink artifact
    attention_sink_positions: Optional[List[int]] = None   # None = auto-detect
    compute_ov_metric: bool = False           # requires v_proj hook

    def get_torch_dtype(self):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)

    # ── NEXUS: Local LLM for hyperedge principle labeling ─────────────────
    use_local_llm_labeling: bool = False
    # Transport: "ollama" uses the Ollama HTTP API (localhost:11434).
    #            "transformers" uses a HuggingFace pipeline on-device.
    local_llm_transport: str = "ollama"       # "ollama" | "transformers"
    local_llm_model: str = "qwen2.5-coder:3b" # Ollama model tag or HF repo id
    local_llm_max_tokens: int = 60            # max tokens for label generation
    local_llm_timeout: float = 10.0           # seconds per labeling request
