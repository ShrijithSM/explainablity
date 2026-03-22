"""
The Analyzer: Causal Inference Engine.

Implements causal patching sweeps, DTW trajectory comparison,
topological data analysis (persistent homology), composite
validity scoring, and rigorous time-series statistical testing
(Gaps A–E).
"""

import numpy as np
import os
import torch
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from rich.console import Console
from rich.progress import Progress
import uuid

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
        # Bridge-facing cached outputs.
        self._last_influence_matrix = None
        self._last_fdr_result = None
        self._last_pdc = None
        self._selected_lag = None
        self._last_joint_stationarity = None
        self._last_coint = None
        self._vecm_used = False
        self._last_pert_results = None
        self._last_mediation_results = None
        self._last_hmm_result = None

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

    def stochastic_patching_sweep(
        self,
        prompt: str,
        layer_names: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> Dict:
        """
        Optimized sweep: Only patches 'salient' tokens where the trajectory
        shows high activity/energy, skipping static regions.
        """
        # 1. Capture clean trace
        console.print("[cyan]Capturing clean trajectory...[/]")
        clean_traj, clean_text = self.interceptor.capture_generation(prompt)
        
        if not clean_traj:
            raise RuntimeError("No activations captured.")

        if layer_names is None:
            # For speed, we might only want to patch the 'deepest' and 'middle' layers
            all_layers = sorted(clean_traj.keys())
            if len(all_layers) > 3:
                mid = len(all_layers) // 2
                from .models import get_deepest_layer
                deepest = get_deepest_layer(all_layers)
                layer_names = sorted(list(set([all_layers[mid], deepest])))
            else:
                layer_names = all_layers

        # 2. Salience Scan: Find tokens with high 'update energy'
        # We use the deepest layer as the proxy for semantic activity
        ref_layer = clean_traj[layer_names[-1]]
        # Compute L2 norm of the difference between consecutive tokens
        deltas = torch.norm(ref_layer[1:] - ref_layer[:-1], dim=-1).detach().cpu().numpy()
        
        # Identify top-K indices with highest deltas
        # (Offset by 1 because deltas are between tokens)
        salient_indices = np.argsort(deltas)[::-1][:top_k]
        salient_indices = sorted(list(set(salient_indices + 1))) 
        
        console.print(f"  [green]Salience scan identified {len(salient_indices)} interest points out of {ref_layer.shape[0]} tokens.[/]")

        # 3. Patch only the salient tokens
        n_layers = len(layer_names)
        n_tok = len(salient_indices)
        
        # Sparse result heatmap
        res_heatmap = np.zeros((n_layers, n_tok))
        
        with Progress() as progress:
            task = progress.add_task("Optimized Patching...", total=n_layers * n_tok)
            for li, layer_name in enumerate(layer_names):
                for ti, token_idx in enumerate(salient_indices):
                    try:
                        patched_traj, _ = self.interceptor.patch(
                            prompt,
                            target_layer_name=layer_name,
                            token_indices=[int(token_idx)],
                        )
                        if layer_names[-1] in patched_traj:
                            val = self._trajectory_divergence(ref_layer, patched_traj[layer_names[-1]])
                            res_heatmap[li, ti] = val
                    except:
                        pass
                    progress.advance(task)

        # 4. Reconstruct full heatmap shape for the report
        full_tok_count = ref_layer.shape[0]
        full_heatmap = np.zeros((n_layers, full_tok_count))
        for li in range(n_layers):
            for ti, token_idx in enumerate(salient_indices):
                full_heatmap[li, token_idx] = res_heatmap[li, ti]

        token_labels = self.interceptor.get_token_labels(prompt, clean_text)

        return {
            "heatmap": full_heatmap,
            "layer_names": layer_names,
            "token_indices": salient_indices,
            "clean_text": clean_text,
            "token_labels": token_labels,
        }

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

    # ================================================================== #
    #  GAP A: Stationarity & Cointegration Methods
    # ================================================================== #

    def _test_per_head_stationarity(self, metric_series: np.ndarray) -> dict:
        """
        A.1 — Run ADF test on each of the H head entropy series independently.

        Args:
            metric_series: np.ndarray of shape [T, H]

        Returns:
            dict with 'p_values', 'is_stationary', 'needs_diff', 'diff_mask'
        """
        from statsmodels.tsa.stattools import adfuller
        import torch

        series_t  = torch.as_tensor(metric_series, dtype=torch.float64)   # [T, H]
        col_means = torch.mean(series_t, dim=0, keepdim=True)             # [1, H]
        centered  = (series_t - col_means).numpy()                        # [T, H] zero-mean

        T, H = centered.shape
        p_values = np.zeros(H)
        for h in range(H):
            series = centered[:, h]
            if np.std(series) < 1e-9:
                p_values[h] = 1.0  # constant series → not stationary in a useful sense
                continue
            try:
                adf_result = adfuller(series, maxlag=None, autolag='AIC')
                p_values[h] = adf_result[1]
            except Exception:
                p_values[h] = 1.0

        is_stationary = p_values < 0.05
        diff_mask = ~is_stationary

        return {
            'p_values': p_values,
            'is_stationary': is_stationary,
            'needs_diff': bool(diff_mask.any()),
            'diff_mask': diff_mask,
        }

    def _apply_selective_differencing(
        self,
        metric_series: np.ndarray,
        diff_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        A.2 — First-difference only the heads flagged as non-stationary.
        Trim one row from ALL heads so the matrix stays rectangular.

        Returns:
            differenced_series: np.ndarray [T-1, H]
            diff_mask: passed through for downstream use
        """
        import torch
        
        series_t = torch.as_tensor(metric_series, dtype=torch.float64)   # [T, H]
        mask     = torch.as_tensor(diff_mask, dtype=torch.bool)           # [H]

        differenced = torch.diff(series_t, n=1, dim=0)                    # [T-1, H]
        trimmed     = series_t[1:, :]                                      # [T-1, H]

        result = torch.where(mask.unsqueeze(0), differenced, trimmed)     # [T-1, H]
        out    = result.numpy()

        return out, diff_mask

    def _check_cointegration(
        self,
        series_np: np.ndarray,
        max_lags: int = 3
    ) -> dict:
        """
        A.3 — Run Johansen cointegration test on the [T, H] head entropy matrix.

        Returns:
            dict with 'cointegrated', 'n_coint_vectors', 'trace_stats', 'crit_values_95'
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        result = coint_johansen(series_np, det_order=0, k_ar_diff=max(max_lags - 1, 1))
        crit_95 = result.cvt[:, 1]
        trace_stats = result.lr1
        rank = int(np.sum(trace_stats > crit_95))

        return {
            'cointegrated': rank > 0,
            'n_coint_vectors': rank,
            'trace_stats': trace_stats,
            'crit_values_95': crit_95,
        }

    def _fit_vecm(
        self,
        series_np: np.ndarray,
        rank: int,
        max_lags: int = 3
    ) -> object:
        """
        A.3 — Fit a VECM when cointegration is detected.
        Returns fitted VECMResults object.
        """
        from statsmodels.tsa.vector_ar.vecm import VECM

        model = VECM(series_np, k_ar_diff=max(max_lags - 1, 1), coint_rank=rank, deterministic='ci')
        return model.fit()

    # ================================================================== #
    #  GAP B: Statistical Significance Methods
    # ================================================================== #

    def _granger_pvalue_matrix(
        self,
        var_result,
        series_np: np.ndarray,
        max_lag: int
    ) -> np.ndarray:
        """
        B.1 — Compute Granger causality F-test p-value for every directed pair (j→i).

        Returns:
            pval_matrix: np.ndarray [H, H] — pval_matrix[i, j] = p-value for j→i
                         diagonal is NaN
        """
        H = series_np.shape[1]
        pval_matrix = np.full((H, H), np.nan)

        for i in range(H):
            for j in range(H):
                if i == j:
                    continue
                try:
                    test = var_result.test_causality(
                        caused=i,
                        causing=j,
                        kind='f'
                    )
                    pval_matrix[i, j] = test.pvalue
                except Exception:
                    pval_matrix[i, j] = 1.0

        return pval_matrix

    def _apply_fdr_correction(
        self,
        pval_matrix: np.ndarray,
        alpha: float = 0.05
    ) -> dict:
        """
        B.2 — Apply Benjamini-Hochberg FDR correction to the p-value matrix.

        Returns:
            dict with 'reject_matrix', 'pval_corrected', 'n_significant', 'significant_pairs'
        """
        from statsmodels.stats.multitest import multipletests

        H = pval_matrix.shape[0]

        flat_pvals = []
        flat_indices = []
        for i in range(H):
            for j in range(H):
                if i != j and not np.isnan(pval_matrix[i, j]):
                    flat_pvals.append(pval_matrix[i, j])
                    flat_indices.append((i, j))

        if not flat_pvals:
            return {
                'reject_matrix': np.zeros((H, H), dtype=bool),
                'pval_corrected': np.full((H, H), np.nan),
                'n_significant': 0,
                'significant_pairs': [],
            }

        reject, pvals_corrected, _, _ = multipletests(
            flat_pvals, alpha=alpha, method='fdr_bh'
        )

        reject_matrix = np.zeros((H, H), dtype=bool)
        corrected_matrix = np.full((H, H), np.nan)

        for k, (i, j) in enumerate(flat_indices):
            reject_matrix[i, j] = reject[k]
            corrected_matrix[i, j] = pvals_corrected[k]

        significant_pairs = [
            (j, i, corrected_matrix[i, j])
            for (i, j) in flat_indices
            if reject_matrix[i, j]
        ]
        significant_pairs.sort(key=lambda x: x[2])

        return {
            'reject_matrix': reject_matrix,
            'pval_corrected': corrected_matrix,
            'n_significant': int(reject_matrix.sum()),
            'significant_pairs': significant_pairs,
        }

    def _bootstrap_surrogate_pvalues(
        self,
        series_np: np.ndarray,
        observed_influence: np.ndarray,
        n_surrogates: int = 500,
        lag: int = 3
    ) -> np.ndarray:
        """
        B.3 — Phase-scramble each head series independently to break temporal
        structure. Refit VAR on each surrogate. Return empirical p-value matrix.

        Returns:
            empirical_pval: np.ndarray [H, H]
        """
        from statsmodels.tsa.vector_ar.var_model import VAR
        import torch
        import torch.fft
        import torch.linalg

        T, H = series_np.shape
        surrogate_scores = np.zeros((n_surrogates, H, H))

        series_t  = torch.as_tensor(series_np, dtype=torch.float64)         # [T, H]
        spectrum  = torch.fft.rfft(series_t, dim=0)                         # [F, H] complex
        magnitude = spectrum.abs()                                          # [F, H] real

        F = spectrum.shape[0]
        rand_phases = torch.rand(n_surrogates, F, H, dtype=torch.float64) * 2 * torch.pi
        rand_phases[:, 0, :] = 0.0   # preserve DC component

        scrambled = magnitude.unsqueeze(0) * torch.polar(
            torch.ones(n_surrogates, F, H, dtype=torch.float64),
            rand_phases
        )                                                                    # [S, F, H]

        # Inverse FFT for all surrogates in one call
        surrogates = torch.fft.irfft(scrambled, n=T, dim=1)                 # [S, T, H]
        surrogates_np = surrogates.numpy()

        for s in range(n_surrogates):
            surrogate = surrogates_np[s]
            try:
                surr_model = VAR(surrogate)
                surr_result = surr_model.fit(maxlags=lag, ic='aic')
                
                coefs_t = torch.as_tensor(surr_result.coefs, dtype=torch.float64)
                surrogate_scores[s] = torch.linalg.norm(coefs_t, ord=1, dim=0).numpy()
            except Exception:
                surrogate_scores[s] = observed_influence

        empirical_pval = np.mean(
            surrogate_scores >= observed_influence[None, :, :], axis=0
        )
        return empirical_pval

    def _conditional_transfer_entropy(
        self,
        series_np: np.ndarray,
        k: int = 3,
        discretize_bins: int = 8
    ) -> np.ndarray:
        """
        B.4 — Compute conditional transfer entropy TE(j→i) for all pairs.
        TE(j→i) = I(X_i(t+1) ; X_j^{1..k} | X_i^{1..k})

        Returns:
            te_matrix: np.ndarray [H, H] — te_matrix[i,j] = TE(j→i) in nats
        """
        T, H = series_np.shape

        def discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
            bins = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
            return np.digitize(x, bins) - 1

        disc = np.stack(
            [discretize(series_np[:, h], discretize_bins) for h in range(H)],
            axis=1
        )

        te_matrix = np.zeros((H, H))

        for i in range(H):
            for j in range(H):
                if i == j:
                    continue

                counts = {}
                for t in range(k, T - 1):
                    x_i_future = int(disc[t + 1, i])
                    x_i_past = tuple(disc[t - k:t, i].tolist())
                    x_j_past = tuple(disc[t - k:t, j].tolist())
                    key = (x_i_future, x_i_past, x_j_past)
                    counts[key] = counts.get(key, 0) + 1

                total = sum(counts.values())
                if total == 0:
                    continue

                te = 0.0
                for (xif, xip, xjp), cnt in counts.items():
                    p_joint = cnt / total
                    p_xip_xjp = sum(
                        v for (a, b, c), v in counts.items() if b == xip and c == xjp
                    ) / total
                    p_xif_xip = sum(
                        v for (a, b, c), v in counts.items() if a == xif and b == xip
                    ) / total
                    p_xip = sum(
                        v for (a, b, c), v in counts.items() if b == xip
                    ) / total

                    if p_joint > 0 and p_xip_xjp > 0 and p_xif_xip > 0 and p_xip > 0:
                        te += p_joint * np.log(
                            (p_joint * p_xip) / (p_xip_xjp * p_xif_xip)
                        )

                te_matrix[i, j] = max(te, 0.0)

        return te_matrix

    def _partial_directed_coherence(
        self,
        var_result,
        freqs: np.ndarray = None
    ) -> dict:
        """
        B.5 — Compute Partial Directed Coherence (PDC) from fitted VAR coefficients.
        PDC(j→i, f) = |A(f)[i,j]|² / Σ_k |A(f)[k,j]|²

        Returns:
            dict with 'pdc' [F,H,H], 'freqs', 'pdc_low', 'pdc_high', 'dominant_freq_per_pair'
        """
        if freqs is None:
            freqs = np.linspace(0, 0.5, 128)

        coefs = var_result.coefs
        p, H, _ = coefs.shape

        pdc = np.zeros((len(freqs), H, H))

        for fi, f in enumerate(freqs):
            A_f = np.eye(H, dtype=complex)
            for lag_idx in range(p):
                A_f -= coefs[lag_idx] * np.exp(-2j * np.pi * f * (lag_idx + 1))

            A_abs_sq = np.abs(A_f) ** 2
            col_sums = A_abs_sq.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            pdc[fi] = A_abs_sq / col_sums

        low_mask = freqs <= 0.1
        high_mask = freqs >= 0.4

        pdc_low = pdc[low_mask].mean(axis=0) if low_mask.any() else np.zeros((H, H))
        pdc_high = pdc[high_mask].mean(axis=0) if high_mask.any() else np.zeros((H, H))

        return {
            'pdc': pdc,
            'freqs': freqs,
            'pdc_low': pdc_low,
            'pdc_high': pdc_high,
            'dominant_freq_per_pair': np.argmax(pdc, axis=0) / len(freqs) * 0.5,
        }

    # ================================================================== #
    #  GAP C: Perturbation / Intervention Methods
    # ================================================================== #

    def _make_ablation_model(
        self,
        layer_idx: int,
        h_source: int,
        mode: str = 'zero',
        mean_activation: torch.Tensor = None
    ):
        """
        C.1 — Return a pyvene IntervenableModel that ablates head h_source.
        """
        import pyvene as pv
        
        if mode == 'zero':
            config = pv.IntervenableConfig([{
                "layer": layer_idx,
                "component": "attention_output",
                "intervention_type": pv.ZeroIntervention,
                "source_representation": None,
            }])
        elif mode == 'mean':
            config = pv.IntervenableConfig([{
                "layer": layer_idx,
                "component": "attention_output",
                "intervention_type": pv.ConstantSourceIntervention,
                "source_representation": mean_activation,
            }])
        elif mode == 'gaussian':
            config = pv.IntervenableConfig([{
                "layer": layer_idx,
                "component": "attention_output",
                "intervention_type": pv.NoiseIntervention,
            }])
        else:
            raise ValueError(f"Unknown mode {mode}")
            
        return pv.IntervenableModel(config, self.interceptor.model)

    def _activation_patch_experiment(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layer_idx: int,
        h_source: int
    ) -> dict:
        """
        C.3 — Activation patching using pyvene (VanillaIntervention).
        Patches head h_source from clean_prompt into corrupted_prompt run.
        """
        import pyvene as pv
        import torch.nn.functional as F

        patch_config = pv.IntervenableConfig([{
            "layer": layer_idx,
            "component": "attention_output",
            "intervention_type": pv.VanillaIntervention,
        }])
        patcher = pv.IntervenableModel(patch_config, self.interceptor.model)

        clean_inputs = self.interceptor.tokenizer(clean_prompt, return_tensors='pt').to(self.interceptor.device)
        corr_inputs = self.interceptor.tokenizer(corrupted_prompt, return_tensors='pt').to(self.interceptor.device)

        _, patched_output = patcher(
            base=corr_inputs,
            sources=[clean_inputs],
            unit_locations={"base": h_source, "source": h_source}
        )
        
        with torch.no_grad():
            clean_logits = self.interceptor.model(**clean_inputs).logits[:, -1, :]
            corrupt_logits = self.interceptor.model(**corr_inputs).logits[:, -1, :]
            
        patched_logits = patched_output.logits[:, -1, :]

        kl_baseline = float(F.kl_div(corrupt_logits.log_softmax(-1), clean_logits.softmax(-1), reduction='sum'))
        kl_patched = float(F.kl_div(patched_logits.log_softmax(-1), clean_logits.softmax(-1), reduction='sum'))

        restoration = 1.0 - (kl_patched / kl_baseline) if kl_baseline > 0 else 0.0

        return {
            'patch_effect': float(F.kl_div(
                corrupt_logits.log_softmax(-1),
                patched_logits.softmax(-1),
                reduction='sum'
            )),
            'restoration': float(np.clip(restoration, 0.0, 1.0)),
            'kl_baseline': float(kl_baseline),
            'kl_patched': float(kl_patched),
        }

    def _build_mean_ablation_cache(
        self,
        reference_prompts: List[str],
        layer_idx: int,
        h_source: int
    ) -> torch.Tensor:
        """
        C.2 — Run the model on reference_prompts, collect head h_source activations
        using pyvene CollectIntervention.
        """
        import pyvene as pv
        
        collect_config = pv.IntervenableConfig([{
            "layer": layer_idx,
            "component": "attention_output",
            "intervention_type": pv.CollectIntervention,
        }])
        collector = pv.IntervenableModel(collect_config, self.interceptor.model)

        collected = []
        for prompt in reference_prompts:
            inputs = self.interceptor.tokenizer(prompt, return_tensors="pt").to(self.interceptor.device)
            with torch.no_grad():
                _, collected_output = collector(
                    base=inputs, 
                    sources=None,
                    unit_locations={"base": h_source}
                )
            collected.append(collected_output.collected_activations[0])

        if not collected:
            return torch.empty(0)

        # Handle varying sequence lengths by padding before stacking
        max_len = max(t.shape[1] for t in collected)
        padded = []
        for t in collected:
            pad_size = max_len - t.shape[1]
            if pad_size > 0:
                last_val = t[:, -1:, :].expand(-1, pad_size, -1)
                t = torch.cat([t, last_val], dim=1)
            padded.append(t)

        mean_activation = torch.stack(padded).mean(dim=0)
        return mean_activation

    def _direct_vs_total_effect(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layer_idx: int,
        source_head: int,
        target_head: int,
        mediator_heads: List[int]
    ) -> dict:
        """
        C.4 — Estimate direct effect of source_head on target_head,
        controlling for mediation through mediator_heads.
        Uses composed pv.VanillaInterventions.
        """
        import pyvene as pv
        import torch.nn.functional as F

        def _run_with_patches(heads_to_patch: List[int]) -> torch.Tensor:
            if not heads_to_patch:
                with torch.no_grad():
                    return self.interceptor.model(**self.interceptor.tokenizer(corrupted_prompt, return_tensors="pt").to(self.interceptor.device)).logits[:, -1]
                    
            configs = [{
                "layer": layer_idx,
                "component": "attention_output",
                "intervention_type": pv.VanillaIntervention,
            } for _ in heads_to_patch]

            model_patched = pv.IntervenableModel(pv.IntervenableConfig(configs), self.interceptor.model)
            
            clean_inputs = self.interceptor.tokenizer(clean_prompt, return_tensors="pt").to(self.interceptor.device)
            corr_inputs = self.interceptor.tokenizer(corrupted_prompt, return_tensors="pt").to(self.interceptor.device)
            
            locations = [{"base": h, "source": h} for h in heads_to_patch]
            
            with torch.no_grad():
                _, out = model_patched(
                    base=corr_inputs,
                    sources=[clean_inputs] * len(heads_to_patch),
                    unit_locations=locations
                )
            return out.logits[:, -1]

        with torch.no_grad():
            baseline_logits = self.interceptor.model(**self.interceptor.tokenizer(corrupted_prompt, return_tensors="pt").to(self.interceptor.device)).logits[:, -1]

        total_logits = _run_with_patches([source_head])
        indirect_logits = _run_with_patches([source_head] + mediator_heads)

        total_effect = float(F.kl_div(baseline_logits.log_softmax(-1), total_logits.softmax(-1), reduction='sum'))
        indirect_effect = float(F.kl_div(baseline_logits.log_softmax(-1), indirect_logits.softmax(-1), reduction='sum'))

        direct_effect = max(total_effect - indirect_effect, 0.0)
        mediation_ratio = indirect_effect / total_effect if total_effect > 0 else 0.0

        return {
            'total_effect':    total_effect,
            'indirect_effect': indirect_effect,
            'direct_effect':   direct_effect,
            'mediation_ratio': float(np.clip(mediation_ratio, 0.0, 1.0)),
            'source_head':     source_head,
            'target_head':     target_head,
            'mediator_heads':  mediator_heads,
        }

    # ================================================================== #
    #  GAP D: Thinking Time Axis Methods
    # ================================================================== #

    def _segment_by_topological_phases(
        self,
        euler_characteristic_series: np.ndarray,
        spike_threshold_std: float = 2.0
    ) -> List[Tuple[int, int]]:
        """
        D.2 — Use Euler characteristic spikes as phase boundaries.
        Returns token-index spans of each phase.
        """
        mean_ec = euler_characteristic_series.mean()
        std_ec = euler_characteristic_series.std()
        if std_ec < 1e-9:
            return [(0, len(euler_characteristic_series))]

        threshold = mean_ec + spike_threshold_std * std_ec
        spike_positions = np.where(np.abs(euler_characteristic_series) > threshold)[0]

        boundaries = [0] + spike_positions.tolist() + [len(euler_characteristic_series)]
        boundaries = sorted(set(boundaries))

        phases = []
        for start_tok, end_tok in zip(boundaries[:-1], boundaries[1:]):
            if end_tok - start_tok >= 3:
                phases.append((int(start_tok), int(end_tok)))

        return phases

    def _aggregate_entropy_by_phase(
        self,
        metric_series_np: np.ndarray,
        phases: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        D.2 — Aggregate head entropy series by topological phase.
        Returns [n_phases, H] array.
        """
        H = metric_series_np.shape[1]
        phase_entropy = np.zeros((len(phases), H))

        for pi, (start, end) in enumerate(phases):
            phase_entropy[pi] = metric_series_np[start:end, :].mean(axis=0)

        return phase_entropy

    def _resample_at_equal_arc_length(
        self,
        metric_series_np: np.ndarray,
        hidden_states: np.ndarray,
        n_resampled_points: int = 80
    ) -> np.ndarray:
        """
        D.4 — Resample head entropy series from equal token-spacing to
        equal arc-length spacing (equal cognitive effort per step).

        Returns:
            resampled: np.ndarray [n_resampled_points, H]
        """
        from scipy.interpolate import interp1d

        T = hidden_states.shape[0]

        diff = np.diff(hidden_states, axis=0)
        arc_steps = np.linalg.norm(diff, axis=1)
        cumulative_arc = np.concatenate([[0.0], np.cumsum(arc_steps)])

        total_arc = cumulative_arc[-1]
        if total_arc < 1e-9:
            return metric_series_np[:n_resampled_points]

        query_arcs = np.linspace(0, total_arc, n_resampled_points)
        resampled = np.zeros((n_resampled_points, metric_series_np.shape[1]))

        for h in range(metric_series_np.shape[1]):
            f = interp1d(
                cumulative_arc,
                metric_series_np[:, h],
                kind='linear',
                fill_value='extrapolate'
            )
            resampled[:, h] = f(query_arcs)

        return resampled

    def _discover_phases_hmm(
        self,
        metric_series_np: np.ndarray,
        n_states: int = 4,
        n_iter: int = 200
    ) -> dict:
        """
        D.5 — Fit a Gaussian HMM to discover latent cognitive phases.

        Returns:
            dict with 'state_sequence', 'phase_spans', 'transition_matrix',
            'state_means', 'bic', 'per_head_dominant_state'
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            return {'error': 'hmmlearn not installed'}

        model = GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=n_iter,
            random_state=42
        )
        model.fit(metric_series_np)

        state_sequence = model.predict(metric_series_np)

        phase_spans = []
        start = 0
        for t in range(1, len(state_sequence)):
            if state_sequence[t] != state_sequence[t - 1]:
                phase_spans.append((int(state_sequence[t - 1]), start, t))
                start = t
        phase_spans.append((int(state_sequence[-1]), start, len(state_sequence)))

        log_likelihood = model.score(metric_series_np)
        T, H = metric_series_np.shape
        n_params = n_states * (n_states - 1) + n_states * H * 2 + n_states
        bic = -2 * log_likelihood + n_params * np.log(T)

        per_head_dominant = np.argmax(model.means_, axis=0)

        hmm_result = {
            'state_sequence': state_sequence,
            'phase_spans': phase_spans,
            'transition_matrix': model.transmat_,
            'state_means': model.means_,
            'bic': float(bic),
            'per_head_dominant_state': per_head_dominant,
            'n_states_used': n_states,
        }
        self._last_hmm_result = hmm_result
        return hmm_result

    # ================================================================== #
    #  Refactored head_interaction_analysis — Rigorous Pipeline
    # ================================================================== #

    def head_interaction_analysis(
        self,
        prompt: str,
        layer_name: Optional[str] = None,
        max_lag: Optional[int] = None,
    ) -> Dict:
        """
        Treat per-head attention metrics as a multivariate time series and
        estimate directed interactions between heads via a rigorous pipeline:

        Pipeline order:
            1. Sanitize & filter constant heads
            2. Gap A.1: Per-head ADF stationarity test
            3. Gap A.5: Joint ADF+KPSS (if config.joint_stationarity_test)
            4. Gap A.3: Johansen cointegration → VECM or selective differencing
            5. Gap A.4: AIC-based lag selection
            6. VAR or VECM fit
            7. Gap B.1: Granger F-test p-value matrix
            8. Gap B.2: BH-FDR correction → mask influence matrix
            9. Gap B.5: PDC (if config.compute_pdc)

        Returns:
            Dict with backward-compatible keys plus new statistical outputs.
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
            # Fallback for backends where generate() doesn't surface attention
            # weights to hooks or generation outputs.
            self.interceptor.capture_head_metrics_from_forward(prompt)
            metric_series = self.interceptor.get_head_metric_series(layer_name)

        if metric_series.numel() == 0:
            return {"error": "no head metrics captured", "layer_name": layer_name}

        # metric_series can be scalar [T, H] or vector [T, H, F].
        series_np = metric_series.numpy()
        if series_np.ndim == 3:
            t, h, f = series_np.shape
            series_np = series_np.reshape(t, h * f)

        # Optional Exp3/Exp6 behaviour: exclude prompt tokens from causality.
        prompt_len = int(getattr(self.config, "prompt_token_count", 0) or 0)
        if getattr(self.config, "analyse_generated_only", True) and prompt_len > 0 and series_np.shape[0] > (prompt_len + 5):
            series_np = series_np[prompt_len:]

        # ── 1. Sanitize ────────────────────────────────────────────────────
        series_np = np.nan_to_num(series_np, nan=0.0, posinf=0.0, neginf=0.0)

        active_indices = np.where(series_np.std(axis=0) > 1e-6)[0]
        if len(active_indices) < 2:
            return {
                "error": "insufficient diversity in head dynamics",
                "layer_name": layer_name,
                "active_heads_count": len(active_indices)
            }

        filtered_series = series_np[:, active_indices]
        T, H_active = filtered_series.shape

        # Keep VAR identifiable by limiting equations relative to token count.
        # Conservative cap to keep small-sample VAR stable.
        max_equations = max(2, min(12, T // 4))
        if H_active > max_equations:
            col_var = filtered_series.var(axis=0)
            top_cols = np.argsort(col_var)[::-1][:max_equations]
            top_cols = np.sort(top_cols)
            filtered_series = filtered_series[:, top_cols]
            active_indices = active_indices[top_cols]
            H_active = filtered_series.shape[1]

        lag = max_lag or self.config.var_max_lags
        if T <= 2 * lag + 1:
            return {
                "error": "insufficient timesteps for VAR",
                "layer_name": layer_name,
                "tokens": T,
                "required": 2 * lag + 1
            }

        # ── Result accumulator ─────────────────────────────────────────────
        result = {
            "layer_name": layer_name,
            "generated_text": generated_text,
            "series": series_np,
            "active_heads": active_indices.tolist(),
        }
        self._last_fdr_result = None
        self._last_pdc = None
        self._selected_lag = None
        self._last_joint_stationarity = None
        self._last_coint = None
        self._vecm_used = False
        result["feature_decomposition"] = self.observer.decompose_feature_space(series_np)

        # ── 2. Gap A.1: Per-head ADF stationarity ─────────────────────────
        stationarity_report = self._test_per_head_stationarity(filtered_series)
        result['stationarity'] = stationarity_report

        # ── 3. Gap A.5: Joint ADF+KPSS ────────────────────────────────────
        if self.config.joint_stationarity_test:
            joint_report = self.observer.joint_stationarity_test(filtered_series)
            result['joint_stationarity'] = joint_report
            self._last_joint_stationarity = joint_report

        # ── 4-6. Cointegration/VECM or VAR ────────────────────────────────
        USE_VECM = False
        series_for_var = filtered_series

        if stationarity_report['needs_diff']:
            # Check for cointegration before blindly differencing
            if self.config.run_johansen_cointegration and H_active >= 2:
                try:
                    coint_report = self._check_cointegration(filtered_series, max_lags=lag)
                    result['cointegration'] = coint_report
                    self._last_coint = coint_report

                    if coint_report['cointegrated']:
                        USE_VECM = True
                        self._vecm_used = True
                        vecm_result = self._fit_vecm(
                            filtered_series,
                            rank=coint_report['n_coint_vectors'],
                            max_lags=lag
                        )
                        H = H_active
                        gamma = vecm_result.gamma
                        # statsmodels VECM gamma is typically shaped:
                        #   [H, H * (k_ar - 1)]
                        # but older/newer variants may expose [H * (k_ar - 1), H].
                        if gamma.ndim != 2:
                            raise ValueError(f"unexpected VECM gamma rank: {gamma.ndim}")

                        if gamma.shape[0] == H:
                            if gamma.shape[1] % H != 0:
                                raise ValueError(
                                    f"invalid VECM gamma shape {gamma.shape} for H={H}"
                                )
                            n_lags = gamma.shape[1] // H
                            coef_tensor = gamma.reshape(H, n_lags, H).transpose(1, 0, 2)
                        elif gamma.shape[1] == H:
                            if gamma.shape[0] % H != 0:
                                raise ValueError(
                                    f"invalid VECM gamma shape {gamma.shape} for H={H}"
                                )
                            n_lags = gamma.shape[0] // H
                            coef_tensor = gamma.reshape(n_lags, H, H)
                        else:
                            raise ValueError(
                                f"unsupported VECM gamma layout: {gamma.shape}, H={H}"
                            )

                        influence_active = np.abs(coef_tensor).sum(axis=0)
                        result['model_type'] = 'VECM'
                        result['vecm_coint_rank'] = coint_report['n_coint_vectors']
                except Exception as e:
                    console.print(f"[yellow]Cointegration/VECM failed: {e}. Falling back to VAR.[/]")
                    USE_VECM = False

            if not USE_VECM:
                # Apply selective differencing (Gap A.2)
                series_for_var, diff_mask = self._apply_selective_differencing(
                    filtered_series, stationarity_report['diff_mask']
                )
                result['differenced'] = True
                result['diff_mask'] = diff_mask.tolist()

        # ── Standard VAR path ──────────────────────────────────────────────
        var_result = None
        if not USE_VECM:
            try:
                model = self._var_cls(series_for_var)
                # Gap A.4: AIC-based lag selection
                ic_method = self.config.var_lag_selection if self.config.var_lag_selection != 'fixed' else None
                admissible_lag = max(1, (T - 2) // max(H_active, 2))
                fit_max_lag = min(lag, 5, admissible_lag)

                res = None
                for trial_lag in range(fit_max_lag, 0, -1):
                    try:
                        res = model.fit(maxlags=trial_lag, ic=ic_method)
                        break
                    except Exception:
                        continue

                if res is None:
                    raise RuntimeError(
                        f"unable to fit VAR (T={T}, dims={H_active}, max_lag={fit_max_lag})"
                    )
                var_result = res
            except Exception as e:
                return {**result, "error": f"VAR fit failed: {str(e)}"}

            coefs = getattr(res, "coefs", None)
            if coefs is None or coefs.size == 0:
                return {**result, "error": "VAR returned no coefficients"}

            influence_active = np.abs(coefs).sum(axis=0)
            result['model_type'] = 'VAR'
            result['selected_lag'] = int(res.k_ar)
            self._selected_lag = int(res.k_ar)

        # ── Re-map to full H×H matrix ─────────────────────────────────────
        _, H_total = series_np.shape
        full_influence = np.zeros((H_total, H_total))
        for i, idx_i in enumerate(active_indices):
            for j, idx_j in enumerate(active_indices):
                full_influence[idx_i, idx_j] = influence_active[i, j]

        result['influence_matrix'] = full_influence
        self._last_influence_matrix = full_influence
        result['var_max_lag'] = int(lag)

        # ── 7. Gap B.1: Granger F-test ────────────────────────────────────
        if self.config.granger_ftest and var_result is not None:
            try:
                pval_matrix = self._granger_pvalue_matrix(
                    var_result, series_for_var, result.get('selected_lag', lag)
                )
                result['granger_pval_matrix'] = pval_matrix

                # ── 8. Gap B.2: BH-FDR correction ─────────────────────────
                fdr_result = self._apply_fdr_correction(pval_matrix, alpha=self.config.fdr_alpha)
                result['fdr_result'] = fdr_result
                self._last_fdr_result = fdr_result

                # Mask influence: zero out non-significant pairs
                masked_influence_active = influence_active.copy()
                masked_influence_active[~fdr_result['reject_matrix']] = 0.0

                masked_full = np.zeros((H_total, H_total))
                for i, idx_i in enumerate(active_indices):
                    for j, idx_j in enumerate(active_indices):
                        masked_full[idx_i, idx_j] = masked_influence_active[i, j]

                result['masked_influence'] = masked_full
                result['n_significant_pairs'] = fdr_result['n_significant']
                result['significant_pairs'] = fdr_result['significant_pairs']

            except Exception as e:
                console.print(f"[yellow]Granger/FDR failed: {e}[/]")

        # ── 9. Gap B.5: PDC ───────────────────────────────────────────────
        if self.config.compute_pdc and var_result is not None:
            try:
                pdc_result = self._partial_directed_coherence(var_result)
                result['pdc'] = pdc_result
                self._last_pdc = pdc_result
            except Exception as e:
                console.print(f"[yellow]PDC failed: {e}[/]")

        # ── Optional: Bootstrap surrogates ─────────────────────────────────
        if self.config.enable_bootstrap_surrogates and var_result is not None:
            try:
                bootstrap_pvals = self._bootstrap_surrogate_pvalues(
                    series_for_var, influence_active,
                    n_surrogates=self.config.n_bootstrap_surrogates,
                    lag=result.get('selected_lag', lag)
                )
                result['bootstrap_pvals'] = bootstrap_pvals
            except Exception as e:
                console.print(f"[yellow]Bootstrap surrogates failed: {e}[/]")

        return result

    def _build_induction_direction(
        self,
        pos_prompts: List[str],
        neg_prompts: List[str],
        layer_idx: int
    ) -> np.ndarray:
        """
        E.3 — Build functional direction using steering-vectors.
        """
        from steering_vectors import train_steering_vector
        
        target_model = self.interceptor.model.model if hasattr(self.interceptor.model, 'model') else self.interceptor.model
        
        sv = train_steering_vector(
            target_model,
            self.interceptor.tokenizer,
            training_data=list(zip(pos_prompts, neg_prompts)),
            layers=[layer_idx],
            read_token_index=-1,
        )
        direction = sv.layer_activations[layer_idx].detach().cpu().numpy()
        return direction / np.linalg.norm(direction)

    def _project_onto_functional_directions(
        self,
        head_output_series: np.ndarray,
        function_directions: dict,
    ) -> dict:
        """
        E.4 — Project head_output_series [T, H, head_dim] onto each direction.
        """
        projections = {}
        for func_name, direction in function_directions.items():
            proj = np.einsum('thd,d->th', head_output_series, direction)
            projections[func_name] = proj
        return projections

    def interventional_head_causality(
        self,
        prompt: str,
        layer_name: str,
        source_head_index: int,
    ) -> Dict[str, Any]:
        """
        Causal Intervention on Attention Heads:
        1. Capture clean head-metric time series.
        2. Capture patched head-metric time series (source_head knocked out).
        3. Measure divergence of other heads as a proxy for directed causal influence.
        """
        console.print(f"[cyan]Intervening on Head {source_head_index} in {layer_name}...[/]")
        
        # 1. Clean run
        clean_traj, clean_text = self.interceptor.capture_generation(prompt)
        clean_series = self.interceptor.get_head_metric_series(layer_name)
        
        # 2. Patched run
        patched_traj, patched_text = self.interceptor.patch_attention_heads(
            prompt, layer_name, [source_head_index], ablation_type="zero"
        )
        patched_series = self.interceptor.get_head_metric_series(layer_name)
        
        if clean_series.numel() == 0 or patched_series.numel() == 0:
            return {"error": "metrics not captured"}
            
        # Align lengths
        min_len = min(clean_series.shape[0], patched_series.shape[0])
        clean_series = clean_series[:min_len]
        patched_series = patched_series[:min_len]
        
        # Measure L2 divergence per head
        # divergence: [n_heads]
        diff = (clean_series - patched_series).pow(2)
        causal_influence = diff.mean(dim=0).cpu().numpy()
        
        # Output comparison
        text_divergence = 0.0
        if clean_text != patched_text:
            # Simple divergence metric (normalized edit distance proxy)
            text_divergence = 1.0 - (len(os.path.commonprefix([clean_text, patched_text])) / max(len(clean_text), 1))

        out = {
            "source_head": source_head_index,
            "layer": layer_name,
            "causal_influence_vector": causal_influence,
            "global_text_divergence": text_divergence,
            "clean_text": clean_text,
            "patched_text": patched_text
        }
        self._last_pert_results = [
            {
                "head": int(source_head_index),
                "target": int(np.argmax(causal_influence)) if causal_influence.size else -1,
                "mode": "zero",
                "delta_entropy": float(np.mean(causal_influence)) if causal_influence.size else 0.0,
                "restoration": 0.0,
                "kl_patch": 0.0,
                "confirmed": False,
            }
        ]
        return out

    def interventional_head_causality_multilayer(
        self,
        prompt: str,
        layer_names: List[str],
        source_head_indices: Union[int, List[int]],
    ) -> Dict[str, Any]:
        """
        Simultaneously ablate the same source head across multiple layers and
        return per-layer and aggregate divergence statistics.
        """
        if not layer_names:
            return {"error": "no layers specified"}

        if isinstance(source_head_indices, int):
            source_heads = [source_head_indices]
        else:
            source_heads = [int(h) for h in source_head_indices]

        console.print(
            f"[cyan]Intervening on Heads {source_heads} across layers: "
            f"{', '.join(layer_names)}[/]"
        )

        # 1) Clean run and capture head metrics for requested layers.
        _, clean_text = self.interceptor.capture_generation(prompt)
        clean_series_by_layer = {
            ln: self.interceptor.get_head_metric_series(ln) for ln in layer_names
        }

        # 2) One patched run with all target layers ablated simultaneously.
        layer_to_heads = {ln: source_heads for ln in layer_names}
        _, patched_text = self.interceptor.patch_attention_heads_multi(
            prompt,
            layer_to_heads=layer_to_heads,
            ablation_type="zero",
        )
        patched_series_by_layer = {
            ln: self.interceptor.get_head_metric_series(ln) for ln in layer_names
        }

        per_layer = {}
        per_head_accum = {int(h): [] for h in source_heads}
        aggregate = []
        for ln in layer_names:
            clean_series = clean_series_by_layer.get(ln, torch.empty(0))
            patched_series = patched_series_by_layer.get(ln, torch.empty(0))
            if clean_series.numel() == 0 or patched_series.numel() == 0:
                per_layer[ln] = {"error": "metrics not captured"}
                continue

            min_len = min(clean_series.shape[0], patched_series.shape[0])
            clean_series = clean_series[:min_len]
            patched_series = patched_series[:min_len]

            diff = (clean_series - patched_series).pow(2)
            causal_influence = diff.mean(dim=0).cpu().numpy()

            # Build per-head diagnostics for Exp6 dashboard perturbation panel.
            # Supports scalar [H] and vector [H, F] head metric modes.
            influence_head = np.asarray(causal_influence, dtype=float)
            if influence_head.ndim == 2:
                influence_head = influence_head.mean(axis=1)
            elif influence_head.ndim == 1:
                n_heads = int(getattr(self.config, "n_heads", 0) or 0)
                if n_heads > 0 and influence_head.size > n_heads and influence_head.size % n_heads == 0:
                    influence_head = influence_head.reshape(n_heads, -1).mean(axis=1)

            per_layer[ln] = {
                "causal_influence_vector": causal_influence,
                "mean_influence": float(causal_influence.mean()),
            }
            aggregate.append(float(causal_influence.mean()))

            if influence_head.ndim == 1 and influence_head.size > 0:
                target_head = int(np.argmax(influence_head))
                for h in source_heads:
                    h_i = int(h)
                    src_idx = min(max(h_i, 0), influence_head.size - 1)
                    per_head_accum[h_i].append(
                        {
                            "head": h_i,
                            "target_head": target_head,
                            "target_layer": ln,
                            "delta_entropy": float(influence_head[src_idx]),
                            "restoration": 0.0,
                            "kl_patch": 0.0,
                        }
                    )

        text_divergence = 0.0
        if clean_text != patched_text:
            text_divergence = 1.0 - (
                len(os.path.commonprefix([clean_text, patched_text])) / max(len(clean_text), 1)
            )

        per_head_results = []
        for h_i, rows in per_head_accum.items():
            if not rows:
                continue
            per_head_results.append(
                {
                    "head": int(h_i),
                    "target_head": int(rows[-1].get("target_head", h_i)),
                    "target_layer": rows[-1].get("target_layer"),
                    "delta_entropy": float(np.mean([r.get("delta_entropy", 0.0) for r in rows])),
                    "restoration": float(np.mean([r.get("restoration", 0.0) for r in rows])),
                    "kl_patch": float(np.mean([r.get("kl_patch", 0.0) for r in rows])),
                }
            )

        return {
            "source_heads": source_heads,
            "layers": layer_names,
            "per_layer": per_layer,
            "per_head_results": per_head_results,
            "aggregate_mean_influence": float(np.mean(aggregate)) if aggregate else 0.0,
            "global_text_divergence": text_divergence,
            "clean_text": clean_text,
            "patched_text": patched_text,
        }

    # ------------------------------------------------------------------ #
    #  NEXUS Layer: Hypergraph & Isomorphic Mapping
    # ------------------------------------------------------------------ #

    def extract_hyperedges(
        self,
        trajectory: torch.Tensor,
        token_labels: List[str],
        layer_name: str,
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Identify latent motifs (Hyperedges) in the residual stream.
        Groups tokens that exhibit similar manifold behavior (cosine similarity
        in latent space).
        """
        if trajectory.dim() == 3: # [Batch, Seq, Hidden]
            trajectory = trajectory.squeeze(0)
            
        n_tokens = trajectory.shape[0]
        if n_tokens < 2:
            return []

        # Simple adjacency based on cosine similarity
        norm_traj = torch.nn.functional.normalize(trajectory.float(), p=2, dim=-1)
        sim_matrix = torch.mm(norm_traj, norm_traj.t()).cpu().numpy()
        
        hyperedges = []
        visited = set()
        
        for i in range(n_tokens):
            if i in visited:
                continue
            
            # Find tokens highly similar to i
            cluster_indices = np.where(sim_matrix[i] > threshold)[0]
            if len(cluster_indices) > 2: # Minimal motif size
                member_labels = [token_labels[idx] for idx in cluster_indices if idx < len(token_labels)]
                
                # Mock high-level principle labeling (normally would call a small LLM)
                principle = f"Motif @ {layer_name} ({member_labels[0]}...)"
                
                hyperedges.append({
                    "hyperedge_id": f"he-{uuid.uuid4().hex[:8]}",
                    "principle": principle,
                    "tokens": member_labels,
                    "layer": layer_name,
                    "weight": float(np.mean(sim_matrix[i, cluster_indices]))
                })
                visited.update(cluster_indices)
        
        return hyperedges

    def detect_isomorphic_clusters(
        self,
        hyperedges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map Hyperedges to abstract principles (Isomorphic Mapping).
        Finds motifs from different layers/token-ranges that represent
        the same underlying structural principle.
        """
        if len(hyperedges) < 2:
            return []
            
        clusters = []
        # In a real implementation, we would embed the hyperedge 'principle' 
        # or token-sets and use cosine similarity.
        # For this eager demo, we cluster by token-overlap.
        
        for i, h1 in enumerate(hyperedges):
            for j, h2 in enumerate(hyperedges):
                if i >= j:
                    continue
                
                s1 = set(h1["tokens"])
                s2 = set(h2["tokens"])
                overlap = len(s1.intersection(s2)) / (len(s1 | s2) + 1e-9)
                
                if overlap > 0.4: # Lower threshold for cross-layer isomorphism
                    clusters.append({
                        "cluster_id": f"iso-{uuid.uuid4().hex[:8]}",
                        "shared_principle": f"Structural Isomorphism ({h1['principle']})",
                        "hyperedge_ids": [h1["hyperedge_id"], h2["hyperedge_id"]],
                        "similarity": float(overlap)
                    })
        
        return clusters
