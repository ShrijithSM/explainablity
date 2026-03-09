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
    model_name: str = "Qwen/Qwen2.5-0.5B" # Changed to 7B
    use_airllm: bool = False # Direct loading is now primary (resolves disk space issues)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_4bit: bool = False  # Disabled for 0.5B models (avoids .to() conflicts)
    torch_dtype: str = "float16"

    airllm_vram_headroom_gb: float = 1.0  # GB reserved for activations/buffers

    # --- Interceptor ---
    # Layer name substrings to hook into. These target the output of each
    # decoder layer, which is the residual stream.
    target_layers: List[str] = field(
        default_factory=lambda: ["layers."]  # Matches all decoder layers
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
    tda_window_size: int = 40
    tda_window_stride: int = 20
    tda_enable_windowed: bool = True

    # --- Synthesizer ---
    report_dir: str = "reports"
    save_plots: bool = True

    # --- Experiment ---
    max_new_tokens: int = 50
    clean_prompt: str = (
        "All cats are animals. Whiskers is a cat. Therefore, Whiskers is a"
    )
    corrupted_prompt: Optional[str] = None  # Auto-generated if None

    def get_torch_dtype(self):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)
