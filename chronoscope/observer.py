"""
The Observer: Classical Time Series Signal Processing.

Applies FFT, ACF/PACF, and STL decomposition to the residual stream
trajectory captured by the Interceptor.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf, pacf, adfuller
import torch

from .config import ChronoscopeConfig


class SignalObserver:
    """
    Treats the SVD-compressed residual stream as a classical time series
    and extracts trend, seasonality, noise, and frequency components.
    """

    def __init__(self, config: ChronoscopeConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    #  SVD Compression (High-Dim → Tractable Time Series)
    # ------------------------------------------------------------------ #

    def svd_compress(
        self, trajectory: torch.Tensor, n_components: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduce [Tokens, HiddenDim] → [Tokens, n_components] via SVD.

        Returns:
            compressed: np.ndarray [Tokens, n_components]
            singular_values: np.ndarray [n_components]
            components: np.ndarray [n_components, HiddenDim] (principal directions)
        """
        n = n_components or self.config.svd_components
        X = trajectory.numpy() if isinstance(trajectory, torch.Tensor) else trajectory

        # Center the data (remove mean per dimension)
        X_centered = X - X.mean(axis=0)

        # Full SVD (economy mode)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Project onto top-n components
        compressed = U[:, :n] * S[:n]  # [Tokens, n]
        
        # Explicitly free large intermediate matrices
        extracted_s = S[:n]
        extracted_vt = Vt[:n, :]
        del U, S, Vt
        
        return compressed, extracted_s, extracted_vt

    # ------------------------------------------------------------------ #
    #  Spectral Analysis (FFT)
    # ------------------------------------------------------------------ #

    def spectral_analysis(
        self, compressed: np.ndarray
    ) -> Dict:
        """
        Run FFT on each SVD component to find dominant frequencies.

        Args:
            compressed: [Tokens, n_components] — SVD-reduced trajectory.

        Returns:
            Dict with 'frequencies', 'power_spectrum', 'dominant_periods' per component.
        """
        n_tokens, n_comp = compressed.shape
        results = {
            "per_component": [],
            "aggregate_power": None,
        }

        aggregate_power = None

        for c in range(n_comp):
            signal = compressed[:, c]

            # Compute periodogram (power spectral density)
            freqs, power = periodogram(signal, fs=1.0)  # fs=1 token/step

            # Find top-K dominant frequencies
            top_k = min(self.config.fft_top_k, len(freqs) - 1)
            top_indices = np.argsort(power[1:])[-top_k:] + 1  # Skip DC component
            dominant_freqs = freqs[top_indices]
            dominant_periods = np.where(
                dominant_freqs > 0, 1.0 / dominant_freqs, np.inf
            )

            results["per_component"].append(
                {
                    "component_idx": c,
                    "frequencies": freqs,
                    "power": power,
                    "dominant_frequencies": dominant_freqs,
                    "dominant_periods": dominant_periods,
                }
            )

            if aggregate_power is None:
                aggregate_power = power.copy()
            else:
                aggregate_power += power

        results["aggregate_power"] = aggregate_power
        results["aggregate_freqs"] = freqs

        return results

    # ------------------------------------------------------------------ #
    #  Autocorrelation Analysis (ACF / PACF)
    # ------------------------------------------------------------------ #

    def autocorrelation_analysis(
        self, compressed: np.ndarray
    ) -> Dict:
        """
        Compute ACF and PACF on the first SVD component.
        Maps to the Transformer's attention look-back patterns.

        Returns:
            Dict with 'acf_values', 'pacf_values', 'significant_lags'.
        """
        # Use the first (most important) principal component
        signal = compressed[:, 0]
        max_lag = min(self.config.acf_max_lag, len(signal) // 2 - 1)

        if max_lag < 2:
            return {
                "acf_values": np.array([1.0]),
                "pacf_values": np.array([1.0]),
                "significant_lags": [],
                "warning": "Sequence too short for meaningful ACF.",
            }

        acf_vals = acf(signal, nlags=max_lag, fft=True)
        pacf_vals = pacf(signal, nlags=max_lag, method="ywm")

        # Significant lags: |ACF| > 2/sqrt(N) (95% confidence band)
        threshold = 2.0 / np.sqrt(len(signal))
        significant = np.where(np.abs(acf_vals[1:]) > threshold)[0] + 1

        return {
            "acf_values": acf_vals,
            "pacf_values": pacf_vals,
            "significant_lags": significant.tolist(),
            "confidence_threshold": threshold,
        }

    # ------------------------------------------------------------------ #
    #  Stationarity Test (Augmented Dickey-Fuller)
    # ------------------------------------------------------------------ #

    def stationarity_test(self, compressed: np.ndarray) -> Dict:
        """
        Run ADF test on the first SVD component.
        Non-stationary = model is actively changing its belief during reasoning.

        Returns:
            Dict with 'adf_statistic', 'p_value', 'is_stationary'.
        """
        signal = compressed[:, 0]

        if len(signal) < 10:
            return {
                "adf_statistic": None,
                "p_value": None,
                "is_stationary": None,
                "warning": "Sequence too short for ADF test.",
            }

        try:
            adf_stat, p_value, used_lag, nobs, critical_values, _ = adfuller(
                signal, autolag="AIC"
            )
            return {
                "adf_statistic": float(adf_stat),
                "p_value": float(p_value),
                "is_stationary": p_value < 0.05,
                "critical_values": {k: float(v) for k, v in critical_values.items()},
                "used_lag": int(used_lag),
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------ #
    #  Additive Decomposition (Trend + Seasonal + Residual)
    # ------------------------------------------------------------------ #

    def decompose(
        self, compressed: np.ndarray, period: int = None
    ) -> Dict:
        """
        Additive decomposition: Y_t = T_t + S_t + R_t
        Uses a simple moving average for trend extraction.

        Args:
            compressed: [Tokens, n_components]
            period: Seasonal period (auto-detected from FFT if None).

        Returns:
            Dict with 'trend', 'seasonal', 'residual' arrays.
        """
        signal = compressed[:, 0]
        n = len(signal)

        # Auto-detect period from FFT if not supplied
        if period is None:
            spectral = self.spectral_analysis(compressed)
            dom_periods = spectral["per_component"][0]["dominant_periods"]
            valid = dom_periods[np.isfinite(dom_periods)]
            period = int(np.round(valid[0])) if len(valid) > 0 else max(2, n // 4)

        period = max(2, min(period, n // 2))

        # Moving average for trend
        if period >= n:
            trend = np.full(n, signal.mean())
        else:
            kernel = np.ones(period) / period
            trend = np.convolve(signal, kernel, mode="same")

        # Detrend → extract seasonal
        detrended = signal - trend

        # Average over each position in the period to get seasonal template
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal[indices] = detrended[indices].mean()

        # Residual
        residual = signal - trend - seasonal

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "detected_period": period,
            "original": signal,
        }

    # ------------------------------------------------------------------ #
    #  Incremental Parallel Analysis (Phase 3)
    # ------------------------------------------------------------------ #

    def incremental_analysis(
        self, trajectory: torch.Tensor, window_size: int = 10, distance_threshold: float = 0.5
    ) -> Dict:
        """
        Run real-time topology and non-stationarity checks on the token stream.
        Designed to run natively on the CPU in parallel with AirLLM's background GPU generation.

        Args:
            trajectory: [Tokens, HiddenDim] tensor from the Interceptor.
            window_size: Number of recent tokens to consider for the rolling stats.
            distance_threshold: Threshold to form an edge in the Euler Characteristic graph.
        
        Returns:
            Dict containing rolling variance, Euler Characteristic, and anomaly flags.
        """
        if trajectory.shape[0] < 2:
            return {"rolling_variance": 0.0, "euler_characteristic": 1, "topological_anomaly_detected": False}

        # Analyze only the recent sliding window for fast O(1) tracking
        recent_window = trajectory[-window_size:]
        n_tokens = recent_window.shape[0]
        
        # 1. Non-Stationarity: Local Variance
        # We compute the variance across the token dimension (how much the hidden state shifts step-to-step)
        # and take the mean across the hidden dimensions.
        rolling_var = torch.var(recent_window, dim=0).mean().item()

        # 2. Topological Proxy: Euler Characteristic (\u03C7 = V - E)
        # Fast proxy for Persistent Homology 
        recent_np = recent_window.float().numpy() if isinstance(recent_window, torch.Tensor) else recent_window
        
        # Calculate pairwise L2 distances
        diffs = recent_np[:, np.newaxis, :] - recent_np[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=-1)
        
        # Normalize distances to [0, 1] relative to the window
        max_dist = distances.max()
        if max_dist > 0:
            distances = distances / max_dist
            
        # Build unweighted graph adjacency matrix via threshold
        adjacency = (distances < distance_threshold).astype(int)
        
        # V = number of vertices (tokens in window)
        V = n_tokens
        # E = number of edges (undirected, ignore self-loops)
        E = (np.sum(adjacency) - V) // 2 
        
        chi = V - E
        
        # Anomaly Detection: 
        # A sudden spike or drop in Euler Characteristic implies a structural break in the reasoning manifold
        # Store history in the object to compare against the previous step
        if not hasattr(self, '_prev_chi'):
            self._prev_chi = chi
            self._prev_var = rolling_var
            return {"rolling_variance": rolling_var, "euler_characteristic": int(chi), "topological_anomaly_detected": False}
            
        chi_delta = abs(chi - self._prev_chi)
        var_delta = rolling_var / (self._prev_var + 1e-9)

        anomaly = False
        # If the Euler characteristic shifts abruptly (> 2 means graph suddenly connected or shattered)
        # OR local variance explodes by >50%, we flag an anomaly.
        if chi_delta >= 2 or var_delta > 1.5:
            anomaly = True
            
        self._prev_chi = chi
        self._prev_var = rolling_var
        
        return {
            "rolling_variance": rolling_var,
            "euler_characteristic": int(chi),
            "topological_anomaly_detected": anomaly,
            "diagnostics": f"Chi: {int(chi)} (Delta: {int(chi_delta)}), VarRatio: {var_delta:.2f}"
        }

    # ------------------------------------------------------------------ #
    #  Full Analysis Pipeline
    # ------------------------------------------------------------------ #

    def full_analysis(
        self, trajectory: torch.Tensor
    ) -> Dict:
        """
        Run the complete classical time series analysis pipeline.

        Args:
            trajectory: [Tokens, HiddenDim] tensor from the Interceptor.

        Returns:
            Dict containing all analysis results.
        """
        # Step 1: SVD compress
        compressed, singular_values, components = self.svd_compress(trajectory)

        # Step 2: Spectral analysis
        spectral = self.spectral_analysis(compressed)

        # Step 3: Autocorrelation
        autocorr = self.autocorrelation_analysis(compressed)

        # Step 4: Stationarity
        stationarity = self.stationarity_test(compressed)

        # Step 5: Decomposition
        decomposition = self.decompose(compressed)

        return {
            "compressed_trajectory": compressed,
            "singular_values": singular_values,
            "principal_components": components,
            "spectral": spectral,
            "autocorrelation": autocorr,
            "stationarity": stationarity,
            "decomposition": decomposition,
            "meta": {
                "n_tokens": compressed.shape[0],
                "n_svd_components": compressed.shape[1],
                "original_dim": trajectory.shape[-1]
                if isinstance(trajectory, torch.Tensor)
                else trajectory.shape[-1],
            },
        }
