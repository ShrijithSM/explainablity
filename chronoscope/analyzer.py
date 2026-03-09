"""
The Analyzer: Causal Inference Engine.

Implements causal patching sweeps, DTW trajectory comparison,
topological data analysis (persistent homology), and composite
validity scoring.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.progress import Progress

from .config import ChronoscopeConfig
from .interceptor import ChronoscopeInterceptor
from .observer import SignalObserver

console = Console()


class CausalAnalyzer:
    """
    Performs causal interventions on the model's reasoning trace and
    analyzes the resulting trajectory divergence using DTW, TDA, and SVD.
    """

    def __init__(
        self,
        interceptor: ChronoscopeInterceptor,
        observer: SignalObserver,
        config: ChronoscopeConfig,
    ):
        self.interceptor = interceptor
        self.observer = observer
        self.config = config
        # Lazy-loaded statsmodels VAR class for head interaction analysis.
        self._var_cls = None

    # ------------------------------------------------------------------ #
    #  Causal Patching Sweep
    # ------------------------------------------------------------------ #

    def causal_patching_sweep(
        self,
        prompt: str,
        layer_names: Optional[List[str]] = None,
        token_range: Optional[range] = None,
    ) -> Dict:
        """
        Sweep across layers × tokens. For each (layer, token), patch the
        activation and measure the output divergence.

        Returns a causal heatmap: [n_layers, n_tokens] of divergence scores.
        """
        # Get clean trajectory first
        console.print("[cyan]Capturing clean trajectory...[/]")
        clean_traj, clean_text = self.interceptor.capture_generation(prompt)

        if not clean_traj:
            raise RuntimeError("No activations captured. Check target_layers config.")

        if layer_names is None:
            layer_names = sorted(clean_traj.keys())

        # Get token count from the first layer's trajectory
        first_layer = list(clean_traj.values())[0]
        n_tokens = first_layer.shape[0]

        if token_range is None:
            # Only patch input tokens (not generated ones — those don't exist yet)
            inputs = self.interceptor.tokenizer(prompt, return_tensors="pt")
            n_input_tokens = inputs["input_ids"].shape[1]
            token_range = range(n_input_tokens)

        n_layers = len(layer_names)
        n_tok = len(token_range)
        heatmap = np.zeros((n_layers, n_tok))

        # Compress clean trajectory for comparison
        clean_ref_layer = clean_traj[layer_names[-1]]  # Use last layer
        clean_compressed, _, _ = self.observer.svd_compress(clean_ref_layer)

        console.print(
            f"[cyan]Running patching sweep: {n_layers} layers × {n_tok} tokens[/]"
        )

        with Progress() as progress:
            task = progress.add_task("Patching...", total=n_layers * n_tok)

            for li, layer_name in enumerate(layer_names):
                for ti, token_idx in enumerate(token_range):
                    try:
                        patched_traj, patched_text = self.interceptor.patch(
                            prompt,
                            target_layer_name=layer_name,
                            token_indices=[token_idx],
                        )

                        # Compare last-layer trajectories
                        if layer_names[-1] in patched_traj:
                            patched_ref = patched_traj[layer_names[-1]]
                            divergence = self._trajectory_divergence(
                                clean_ref_layer, patched_ref
                            )
                            heatmap[li, ti] = divergence
                        else:
                            heatmap[li, ti] = 0.0

                    except Exception as e:
                        heatmap[li, ti] = -1.0  # Mark errors

                    progress.advance(task)

        return {
            "heatmap": heatmap,
            "layer_names": layer_names,
            "token_indices": list(token_range),
            "clean_text": clean_text,
            "token_labels": self._get_token_labels(prompt, token_range),
        }

    def _get_token_labels(self, prompt: str, token_range: range) -> List[str]:
        """Get human-readable token labels for the heatmap."""
        tokens = self.interceptor.tokenizer.tokenize(prompt)
        labels = []
        for idx in token_range:
            if idx < len(tokens):
                labels.append(f"{idx}:{tokens[idx]}")
            else:
                labels.append(f"{idx}:<gen>")
        return labels

    # ------------------------------------------------------------------ #
    #  Trajectory Divergence (L2 + Cosine)
    # ------------------------------------------------------------------ #

    def _trajectory_divergence(
        self, clean: torch.Tensor, patched: torch.Tensor
    ) -> float:
        """
        Quick divergence metric between two trajectories.
        Uses mean L2 distance across shared tokens.
        """
        min_len = min(clean.shape[0], patched.shape[0])
        c = clean[:min_len].float()
        p = patched[:min_len].float()

        # Normalized L2 distance
        l2 = torch.norm(c - p, dim=-1).mean().item()
        return l2

    # ------------------------------------------------------------------ #
    #  Dynamic Time Warping (DTW)
    # ------------------------------------------------------------------ #

    def dtw_divergence(
        self, clean_compressed: np.ndarray, patched_compressed: np.ndarray
    ) -> Dict:
        """
        Compute DTW distance between clean and patched trajectories.
        This handles sequences of different lengths (from divergent generation).
        """
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        distance, path = fastdtw(
            clean_compressed,
            patched_compressed,
            radius=self.config.dtw_radius,
            dist=euclidean,
        )

        # Normalized by path length
        normalized = distance / len(path) if path else 0.0

        return {
            "dtw_distance": float(distance),
            "dtw_normalized": float(normalized),
            "path_length": len(path),
            "clean_length": len(clean_compressed),
            "patched_length": len(patched_compressed),
        }

    # ------------------------------------------------------------------ #
    #  Topological Data Analysis (Persistent Homology)
    # ------------------------------------------------------------------ #

    def topological_analysis(
        self, compressed: np.ndarray, max_dim: int = 1
    ) -> Dict:
        """
        Compute persistent homology on the SVD-compressed trajectory.

        A valid reasoning trace should form a smooth manifold (few topological
        features). Hallucinated jumps create holes (high Betti numbers).

        Args:
            compressed: [Tokens, n_components] trajectory.
            max_dim: Maximum homological dimension (0=components, 1=loops).

        Returns:
            Dict with persistence diagrams and Betti numbers.
        """
        try:
            import ripser

            result = ripser.ripser(compressed, maxdim=max_dim, thresh=np.inf)
            diagrams = result["dgms"]

            betti_numbers = {}
            persistence_stats = {}

            for dim, dgm in enumerate(diagrams):
                # Filter out infinite death times
                finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm

                betti_numbers[f"betti_{dim}"] = len(finite)

                if len(finite) > 0:
                    lifetimes = finite[:, 1] - finite[:, 0]
                    persistence_stats[f"dim_{dim}"] = {
                        "count": len(finite),
                        "mean_lifetime": float(np.mean(lifetimes)),
                        "max_lifetime": float(np.max(lifetimes)),
                        "total_persistence": float(np.sum(lifetimes)),
                    }
                else:
                    persistence_stats[f"dim_{dim}"] = {
                        "count": 0,
                        "mean_lifetime": 0.0,
                        "max_lifetime": 0.0,
                        "total_persistence": 0.0,
                    }

            return {
                "diagrams": diagrams,
                "betti_numbers": betti_numbers,
                "persistence_stats": persistence_stats,
            }

        except ImportError:
            console.print(
                "[yellow]ripser not installed. Skipping TDA. "
                "Install with: pip install ripser[/]"
            )
            return {"error": "ripser not installed", "betti_numbers": {}}

    def topological_analysis_windowed(
        self, compressed: np.ndarray, max_dim: int = 1
    ) -> Dict:
        """
        Run persistent homology on sliding windows of the compressed
        trajectory to capture local topological changes along token-time.
        """
        try:
            import ripser
        except ImportError:
            console.print(
                "[yellow]ripser not installed. Skipping windowed TDA. "
                "Install with: pip install ripser[/]"
            )
            return {"error": "ripser not installed", "windows": []}

        if not getattr(self.config, "tda_enable_windowed", False):
            return {"windows": []}

        n_tokens = compressed.shape[0]
        window_size = min(self.config.tda_window_size, n_tokens)
        stride = max(1, self.config.tda_window_stride)

        windows = []
        for start in range(0, max(1, n_tokens - window_size + 1), stride):
            end = min(start + window_size, n_tokens)
            segment = compressed[start:end]
            if segment.shape[0] < 3:
                continue

            result = ripser.ripser(segment, maxdim=max_dim, thresh=np.inf)
            diagrams = result["dgms"]

            betti_numbers = {}
            for dim, dgm in enumerate(diagrams):
                finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
                betti_numbers[f"betti_{dim}"] = len(finite)

            windows.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "diagrams": diagrams,
                    "betti_numbers": betti_numbers,
                }
            )

        return {"windows": windows}

    # ------------------------------------------------------------------ #
    #  Composite Validity Score
    # ------------------------------------------------------------------ #

    def compute_validity_score(
        self,
        dtw_result: Dict,
        spectral_result: Dict,
        tda_result: Dict,
        stationarity_result: Dict,
    ) -> Dict:
        """
        Compute a composite causal validity score.

        High score = the reasoning trace is causally grounded.
        Low score = potential hallucination or memorization.

        Components:
            1. DTW Sensitivity (higher = premise matters causally)
            2. Spectral Coherence (checking behavior present)
            3. Topological Smoothness (low Betti = smooth reasoning)
            4. Non-Stationarity (model actively reasoning, not static)
        """
        scores = {}

        # 1. DTW Sensitivity: Normalized DTW distance (0-1 via sigmoid)
        dtw_norm = dtw_result.get("dtw_normalized", 0.0)
        scores["dtw_sensitivity"] = float(1.0 / (1.0 + np.exp(-dtw_norm)))

        # 2. Spectral Coherence: Presence of dominant frequencies
        agg_power = spectral_result.get("aggregate_power", None)
        if agg_power is not None and len(agg_power) > 1:
            # Ratio of top frequency to total power (peakiness)
            peak_ratio = float(np.max(agg_power[1:]) / (np.sum(agg_power[1:]) + 1e-9))
            scores["spectral_coherence"] = peak_ratio
        else:
            scores["spectral_coherence"] = 0.0

        # 3. Topological Smoothness: Inverse of total Betti numbers
        betti = tda_result.get("betti_numbers", {})
        total_betti = sum(betti.values()) if betti else 0
        scores["topological_smoothness"] = float(1.0 / (1.0 + total_betti))

        # 4. Non-Stationarity: p-value from ADF (low p = stationary = less reasoning)
        p_val = stationarity_result.get("p_value", 0.5)
        if p_val is not None:
            scores["active_reasoning"] = float(
                1.0 - (1.0 / (1.0 + np.exp(5 * (p_val - 0.05))))
            )
        else:
            scores["active_reasoning"] = 0.5

        # Composite: Weighted average
        weights = {
            "dtw_sensitivity": 0.35,
            "spectral_coherence": 0.20,
            "topological_smoothness": 0.25,
            "active_reasoning": 0.20,
        }

        composite = sum(scores[k] * weights[k] for k in weights)
        scores["composite_validity"] = float(composite)

        # Verdict
        if composite > 0.65:
            scores["verdict"] = "CAUSALLY GROUNDED"
        elif composite > 0.40:
            scores["verdict"] = "PARTIALLY GROUNDED (review required)"
        else:
            scores["verdict"] = "LIKELY HALLUCINATION"

        return scores

    # ------------------------------------------------------------------ #
    #  Attention Head Interaction Analysis (Time-Series on Heads)
    # ------------------------------------------------------------------ #

    def _ensure_var_cls(self):
        """Lazy import of statsmodels VAR to avoid hard dependency at import time."""
        if self._var_cls is not None:
            return
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
        except ImportError:
            console.print(
                "[yellow]statsmodels not installed. "
                "Install with: pip install statsmodels[/]"
            )
            self._var_cls = None
            return
        self._var_cls = VAR

    def head_interaction_analysis(
        self,
        prompt: str,
        layer_name: Optional[str] = None,
        max_lag: Optional[int] = None,
    ) -> Dict:
        """
        Treat per-head attention metrics as a multivariate time series and
        estimate directed interactions between heads via a VAR model.

        Returns:
            Dict with:
                - 'layer_name': analyzed layer
                - 'series': np.ndarray [T, H] of head metric time series
                - 'influence_matrix': np.ndarray [H, H] aggregated |coefficients|
                  across lags (row=head source, col=head target)
                - 'var_max_lag': used maximum lag
        """
        self._ensure_var_cls()
        if self._var_cls is None:
            return {"error": "statsmodels VAR unavailable"}

        # Capture a fresh generation so that head metrics are populated.
        trajectory, generated_text = self.interceptor.capture_generation(prompt)
        if not trajectory:
            return {"error": "no activations captured"}

        # Default to last decoder layer if none specified.
        if layer_name is None:
            layer_names = sorted(trajectory.keys())
            layer_name = layer_names[-1]

        metric_series = self.interceptor.get_head_metric_series(layer_name)
        if metric_series.numel() == 0:
            return {"error": "no head metrics captured", "layer_name": layer_name}

        # metric_series: [T, H]
        series_np = metric_series.numpy()
        T, H = series_np.shape

        lag = max_lag or self.config.head_var_max_lag
        # Require enough timesteps for a stable VAR fit.
        if T <= 2 * lag + 1 or H < 2:
            return {
                "error": "insufficient data for VAR",
                "layer_name": layer_name,
                "series": series_np,
            }

        try:
            model = self._var_cls(series_np)
            res = model.fit(maxlags=lag, ic=None)
        except Exception as e:
            return {"error": str(e), "layer_name": layer_name}

        # res.coefs has shape [lag, H, H]; aggregate absolute weights across lags.
        coefs = getattr(res, "coefs", None)
        if coefs is None or coefs.size == 0:
            return {
                "error": "VAR returned no coefficients",
                "layer_name": layer_name,
                "series": series_np,
            }

        influence = np.abs(coefs).sum(axis=0)  # [H, H]

        return {
            "layer_name": layer_name,
            "generated_text": generated_text,
            "series": series_np,
            "influence_matrix": influence,
            "var_max_lag": int(lag),
        }
