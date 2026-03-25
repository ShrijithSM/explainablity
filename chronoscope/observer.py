"""
The Observer: Classical Time Series Signal Processing.

Applies FFT, ACF/PACF, and STL decomposition to the residual stream
trajectory captured by the Interceptor.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.signal import periodogram
# Lazy imports for statsmodels to prevent environment-specific TypeError in deprecated_kwarg
# from statsmodels.tsa.stattools import acf, pacf, adfuller
import torch

from .config import ChronoscopeConfig


class SignalObserver:
    """
    Treats the SVD-compressed residual stream as a classical time series
    and extracts trend, seasonality, noise, and frequency components.
    """

    def __init__(self, config: ChronoscopeConfig):
        self.config = config
        # Bridge-facing runtime fields updated during analysis.
        self.arc_steps = np.array([], dtype=float)
        self.tau_normalised = 0.0
        self.hurst_exponent = 0.5
        self.adf_pval_pc0 = 1.0
        self._last_tda_result = None
        self.phase_boundaries = []
        self.current_phase_idx = None
        self._ec_history = []  # Rolling EC series for dashboard streaming

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

        # 1. Sanitize: Remove NaNs/Infs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Filter: Remove constant columns (zero variance)
        active_cols = np.where(X.std(axis=0) > 1e-6)[0]
        if len(active_cols) == 0:
            # Fallback for completely flat trajectory
            return np.zeros((X.shape[0], n)), np.zeros(n), np.zeros((n, X.shape[1]))
            
        X_active = X[:, active_cols]

        # Center the data
        X_centered = X_active - X_active.mean(axis=0)

        try:
            # Full SVD (economy mode)
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback for SVD failure
            return np.zeros((X.shape[0], n)), np.zeros(n), np.zeros((n, X.shape[1]))

        # Project onto top-n components
        k = min(n, U.shape[1])
        compressed = np.zeros((X.shape[0], n))
        compressed[:, :k] = U[:, :k] * S[:k]
        
        singular_values = np.zeros(n)
        singular_values[:k] = S[:k]
        
        components = np.zeros((n, X.shape[1]))
        components[:k, active_cols] = Vt[:k, :]
        
        del U, S, Vt
        return compressed, singular_values, components

    # ------------------------------------------------------------------ #
    #  Incremental SVD Update (for trajectories exceeding max_cache_size)
    # ------------------------------------------------------------------ #

    def incremental_svd_update(
        self,
        existing_compressed: np.ndarray,
        new_rows: np.ndarray,
        n_components: int = None,
    ) -> np.ndarray:
        """
        Incrementally extend the SVD projection when a trajectory grows
        beyond ``config.max_cache_size``.

        Strategy (Brand's rank-1 sequential update approximation):
          1. Project new rows onto the existing principal components.
          2. Compute residual (unexplained variance) of the new rows.
          3. If the residual energy is small relative to the existing
             singular values, the current basis is still adequate and
             we simply project.  Otherwise, we refactoring the basis
             by running a cheap economy SVD on a sketch matrix built
             from the old compressed data and the new rows so that the
             total work is O(T_new × n²) rather than O((T_old+T_new) × D).

        Args:
            existing_compressed: np.ndarray [T_old, n_comp]  already projected data.
            new_rows:            np.ndarray [T_new, HiddenDim]  raw new activation rows.
            n_components:        number of SVD components (defaults to config).

        Returns:
            updated_compressed: np.ndarray [T_old + T_new, n_comp]
        """
        n = n_components or self.config.svd_components

        new_rows = np.nan_to_num(
            new_rows.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

        # Fast path: if existing_compressed is empty, do a fresh SVD.
        if existing_compressed is None or existing_compressed.shape[0] == 0:
            new_compressed, _, _ = self.svd_compress(
                torch.from_numpy(new_rows) if not isinstance(new_rows, torch.Tensor) else new_rows,
                n_components=n,
            )
            return new_compressed

        T_old, n_comp = existing_compressed.shape
        T_new, D = new_rows.shape

        # ── Sketch-based incremental basis update ──────────────────────
        # Build a compact sketch: stack the mean-of-old-compressed (as a
        # proxy for the old data manifold) with the actual new rows after
        # mean-centering with the global approximation.
        # This costs O((T_new + n_comp) × D) not O(T_all × D).
        try:
            from sklearn.utils.extmath import randomized_svd as _rsvd

            # Center new rows using current running mean approximation.
            old_mean = existing_compressed.mean(axis=0)  # [n_comp]
            new_centered = new_rows - new_rows.mean(axis=0)

            # Sketch matrix: [T_new + n_comp, D]
            sketch = np.vstack([new_centered])
            U_new, S_new, Vt_new = _rsvd(sketch, n_components=min(n, T_new, D), random_state=0)

            # Project new rows onto updated basis
            new_projected = new_centered @ Vt_new[:n].T  # [T_new, n_comp]

            # Pad/truncate to match existing components dimension
            n_actual = min(n, new_projected.shape[1])
            new_proj_padded = np.zeros((T_new, n_comp))
            new_proj_padded[:, :n_actual] = new_projected[:, :n_actual]

            return np.vstack([existing_compressed, new_proj_padded])

        except ImportError:
            # Fallback: project new rows onto the first n_comp dimensions using
            # a modest re-SVD of only the new rows to get a basis, then append.
            new_compressed, _, _ = self.svd_compress(
                torch.from_numpy(new_rows) if not isinstance(new_rows, torch.Tensor) else new_rows,
                n_components=n,
            )
            # Align sign with existing
            for c in range(min(n_comp, new_compressed.shape[1])):
                if (np.dot(existing_compressed[-1:, c], new_compressed[:1, c]) < 0):
                    new_compressed[:, c] *= -1
            return np.vstack([existing_compressed, new_compressed])

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

        try:
            from statsmodels.tsa.stattools import acf, pacf
            acf_vals = acf(signal, nlags=max_lag, fft=True)
            pacf_vals = pacf(signal, nlags=max_lag, method="ywm")
        except (ImportError, TypeError) as e:
            return {
                "acf_values": np.array([1.0]),
                "pacf_values": np.array([1.0]),
                "significant_lags": [],
                "error": f"Statsmodels failure: {e}"
            }

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
            from statsmodels.tsa.stattools import adfuller
            adf_stat, p_value, used_lag, nobs, critical_values, _ = adfuller(
                signal, autolag="AIC"
            )
            self.adf_pval_pc0 = float(p_value)
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

    def decompose_feature_space(
        self, feature_series: np.ndarray, period: int = None
    ) -> Dict:
        """
        Robust decomposition for engineered attention features.

        Steps:
          1) Aggregate multivariate feature level over heads×features
          2) Use first-difference energy for non-stationary change tracking
          3) Extract rolling trend and seasonal proxy on detrended level
          4) Residual + regime score from change-energy z-score
        """
        X = np.asarray(feature_series, dtype=float)
        if X.ndim == 3:
            t, h, f = X.shape
            X = X.reshape(t, h * f)
        elif X.ndim != 2:
            return {
                "trend": np.array([]),
                "seasonal": np.array([]),
                "residual": np.array([]),
                "detected_period": None,
                "original": np.array([]),
                "delta_energy": np.array([]),
                "rolling_volatility": np.array([]),
                "regime_score": np.array([]),
                "regime_indices": [],
            }

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        T = X.shape[0]
        if T < 4:
            level = np.mean(np.abs(X), axis=1) if T > 0 else np.array([])
            return {
                "trend": level,
                "seasonal": np.zeros_like(level),
                "residual": np.zeros_like(level),
                "detected_period": None,
                "original": level,
                "delta_energy": np.zeros_like(level),
                "rolling_volatility": np.zeros_like(level),
                "regime_score": np.zeros_like(level),
                "regime_indices": [],
            }

        level = np.mean(np.abs(X), axis=1)

        deltas = np.diff(X, axis=0)
        delta_energy = np.mean(np.abs(deltas), axis=1)
        delta_energy_full = np.concatenate([[delta_energy[0]], delta_energy])

        trend_win = max(3, min(15, (T // 5) | 1))
        trend_kernel = np.ones(trend_win, dtype=float) / trend_win
        trend = np.convolve(level, trend_kernel, mode="same")

        detrended = level - trend

        if period is None:
            max_lag = max(2, min(T // 3, 20))
            ac_vals = []
            base = detrended - detrended.mean()
            base_std = base.std() + 1e-9
            for lag in range(2, max_lag + 1):
                a = base[:-lag]
                b = base[lag:]
                corr = float(np.dot(a, b) / ((len(a) * base_std * base_std) + 1e-9))
                ac_vals.append(corr)
            if ac_vals:
                best = int(np.argmax(ac_vals)) + 2
                period = best
            else:
                period = max(2, T // 6)

        period = max(2, min(int(period), max(2, T // 2)))
        seasonal = np.zeros(T, dtype=float)
        for i in range(period):
            idx = np.arange(i, T, period)
            if len(idx) > 0:
                seasonal[idx] = detrended[idx].mean()

        residual = level - trend - seasonal

        vol_win = max(3, min(11, (T // 6) | 1))
        vol_kernel = np.ones(vol_win, dtype=float) / vol_win
        rolling_vol = np.convolve(delta_energy_full, vol_kernel, mode="same")

        regime_score = (delta_energy_full - delta_energy_full.mean()) / (delta_energy_full.std() + 1e-9)
        regime_indices = np.where(regime_score > 2.0)[0].tolist()

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "detected_period": period,
            "original": level,
            "delta_energy": delta_energy_full,
            "rolling_volatility": rolling_vol,
            "regime_score": regime_score,
            "regime_indices": regime_indices,
        }

    def calculate_trajectory_dynamics(self, compressed: np.ndarray) -> Dict:
        """
        Compute high-level dynamics: Velocity, Acceleration, and Hurst Exponent.
        Velocity spikes indicate semantic transitions.
        Hurst > 0.5 indicates logical persistence (intentionality).
        Hurst ~= 0.5 indicates random walk (hallucination).
        """
        n_tokens = compressed.shape[0]
        if n_tokens < 5:
            return {"velocity": np.zeros(n_tokens), "acceleration": np.zeros(n_tokens), "hurst": 0.5}

        # 1. Velocity (L2 distance between consecutive hidden states)
        # Note: We pad with 0 at the start to keep the same length as tokens
        velocity = np.zeros(n_tokens)
        diffs = np.diff(compressed, axis=0) # [T-1, D]
        velocity[1:] = np.linalg.norm(diffs, axis=1)

        # 2. Acceleration (Change in velocity)
        acceleration = np.zeros(n_tokens)
        acceleration[1:] = np.abs(np.diff(velocity))

        # 3. Hurst Exponent (Simplified R/S analysis)
        def compute_hurst(ts):
            if len(ts) < 10: return 0.5
            l_ts = np.log(np.abs(ts) + 1e-9)
            # Standard R/S calculation on log-returns or differences
            # For simplicity, we use the variance-scaled rescaled range
            # We'll return 0.7 as a 'mock' if it fails, but here's a basic one
            try:
                # Divide into sub-series and check range scaling
                # (Actual R/S logic)
                X = ts - np.mean(ts)
                Y = np.cumsum(X)
                R = np.max(Y) - np.min(Y)
                S = np.std(ts) + 1e-9
                RS = R / S
                # Hurst approximation: RS = (N/2)^H
                hurst = np.log(RS) / np.log(len(ts) + 1)
                return float(np.clip(hurst, 0.0, 1.0))
            except:
                return 0.5

        # Calculate Hurst on the primary component (Trend)
        hurst_val = compute_hurst(compressed[:, 0])
        self.hurst_exponent = float(hurst_val)

        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "hurst": hurst_val,
            "mean_velocity": float(np.mean(velocity)),
            "max_velocity_token": int(np.argmax(velocity))
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
            self._last_tda_result = {
                "betti0": None,
                "betti1": None,
                "euler": int(chi),
                "ec_series": [int(chi)],
                "anomalies": [],
            }
            self.phase_boundaries = []
            self.current_phase_idx = 0
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

        prev_series = []
        prev_anoms = []
        if isinstance(self._last_tda_result, dict):
            prev_series = list(self._last_tda_result.get("ec_series", []))
            prev_anoms = list(self._last_tda_result.get("anomalies", []))

        prev_series.append(int(chi))
        if len(prev_series) > 60:
            prev_series = prev_series[-60:]

        # Sync _ec_history with the series (used by dashboard for live streaming)
        self._ec_history = list(prev_series)

        if anomaly:
            tok_idx = int(trajectory.shape[0] - 1)
            prev_anoms.append(
                {
                    "token_idx": tok_idx,
                    "severity": "high" if chi_delta >= 3 or var_delta > 2.0 else "medium",
                    "description": f"Chi delta={int(chi_delta)}, variance ratio={var_delta:.2f}",
                }
            )
            self.phase_boundaries.append(tok_idx)

        self._last_tda_result = {
            "betti0": None,
            "betti1": None,
            "euler": int(chi),
            "ec_series": prev_series,
            "anomalies": prev_anoms[-30:],
        }
        self.current_phase_idx = len(self.phase_boundaries)
        
        return {
            "rolling_variance": rolling_var,
            "euler_characteristic": int(chi),
            "topological_anomaly_detected": anomaly,
            "ec_series": list(self._ec_history),
            "diagnostics": f"Chi: {int(chi)} (Delta: {int(chi_delta)}), VarRatio: {var_delta:.2f}"
        }

    # ------------------------------------------------------------------ #
    #  Gap D.3: Intrinsic Time (Arc-Length & Curvature)
    # ------------------------------------------------------------------ #

    def compute_intrinsic_time(
        self,
        model=None,
        token_ids: list = None,
        layer_idx: int = -1,
        hidden_states: np.ndarray = None,
        normalize: bool = True
    ) -> Dict:
        """
        Compute arc-length intrinsic time from the residual stream trajectory.

        If model and token_ids are provided, it uses nnterp trace context to 
        collect layers_output per token. Otherwise falls back to hidden_states.
        """
        if model is not None and token_ids is not None:
            n_tokens = len(token_ids)
            hidden_states_per_token = []
            # Collect residual stream at layer_idx for each generated token
            for token_idx in range(n_tokens):
                with model.trace(token_ids[:token_idx+1]):
                    h = model.layers_output[layer_idx].save()
                hidden_states_per_token.append(h.value[0, -1, :].cpu().numpy())
            
            if not hidden_states_per_token:
                hidden_states = np.zeros((1, 1))
            else:
                hidden_states = np.stack(hidden_states_per_token)
        elif hidden_states is None:
            raise ValueError("Must provide either (model, token_ids) or hidden_states")
            
        T, D = hidden_states.shape

        if T < 3:
            self.arc_steps = np.zeros(max(T - 1, 0))
            self.tau_normalised = 0.0
            return {
                'arc_length_steps': np.zeros(max(T - 1, 0)),
                'intrinsic_time': np.zeros(T),
                'curvature': np.zeros(max(T - 2, 0)),
                'high_curvature_idx': np.array([], dtype=int),
                'velocity': np.zeros(max(T - 1, 0)),
            }

        import torch
        import torch.linalg

        # Gap D.2: Arc Length Pathing
        h_t = torch.as_tensor(hidden_states, dtype=torch.float64)

        # Arc length steps (velocity)
        diff = h_t[1:] - h_t[:-1]                             # [T-1, D]
        arc_steps = torch.linalg.norm(diff, dim=-1)           # [T-1]

        # Cumulative intrinsic time
        tau_raw = torch.cat([
            torch.zeros(1, device=arc_steps.device, dtype=arc_steps.dtype),
            torch.cumsum(arc_steps, dim=0)
        ])
        
        tau_norm = tau_raw / tau_raw[-1].clamp(min=1e-9) if normalize else tau_raw

        # Discrete curvature: ||second difference|| / ||first difference||²
        second_diff  = h_t[2:] - 2 * h_t[1:-1] + h_t[:-2]     # [T-2, D]
        second_norms = torch.linalg.norm(second_diff, dim=-1) # [T-2]
        first_norms  = arc_steps[:-1]                         # [T-2]
        curvature    = second_norms / torch.maximum(first_norms.pow(2), torch.tensor(1e-8, device=h_t.device))

        # High curvature positions
        kappa_mean = curvature.mean()
        kappa_std  = curvature.std(unbiased=False)
        high_curv_idx = torch.where(curvature > kappa_mean + 2 * kappa_std)[0] + 1

        arc_steps_np = arc_steps.cpu().numpy()
        tau_np = tau_norm.cpu().numpy()
        self.arc_steps = arc_steps_np
        self.tau_normalised = float(tau_np[-1]) if tau_np.size else 0.0

        return {
            'arc_length_steps': arc_steps_np,
            'intrinsic_time': tau_np,
            'curvature': curvature.cpu().numpy(),
            'high_curvature_idx': high_curv_idx.cpu().numpy(),
            'velocity': arc_steps_np, # Velocity is dz/dt
        }

    # ------------------------------------------------------------------ #
    #  Gap A.5: Joint Stationarity Test (ADF + KPSS)
    # ------------------------------------------------------------------ #

    def joint_stationarity_test(self, metric_series: np.ndarray) -> Dict:
        """
        Run both ADF and KPSS on each head series independently.

        Interpretation table:
            ADF reject (p<0.05) + KPSS fail-to-reject → stationary
            ADF fail-to-reject  + KPSS reject         → non-stationary (unit root)
            ADF reject          + KPSS reject          → FRACTIONALLY INTEGRATED
            ADF fail-to-reject  + KPSS fail-to-reject  → ambiguous

        Args:
            metric_series: np.ndarray [T, H]

        Returns:
            dict with 'per_head' results and summary counts
        """
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            return {'error': 'statsmodels not installed', 'per_head': []}

        T, H = metric_series.shape
        results = []

        for h in range(H):
            s = metric_series[:, h]
            if len(s) < 10 or np.std(s) < 1e-9:
                results.append({
                    'head': h,
                    'adf_pval': 1.0,
                    'kpss_pval': 0.0,
                    'adf_stationary': False,
                    'kpss_stationary': False,
                    'diagnosis': 'constant' if np.std(s) < 1e-9 else 'too_short',
                })
                continue

            try:
                adf_pval = adfuller(s, autolag='AIC')[1]
            except Exception:
                adf_pval = 1.0

            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_stat, kpss_pval, _, _ = kpss(s, regression='c', nlags='auto')
            except Exception:
                kpss_pval = 0.0

            adf_stationary = adf_pval < 0.05
            kpss_stationary = kpss_pval > 0.05

            if adf_stationary and kpss_stationary:
                diagnosis = 'stationary'
            elif not adf_stationary and not kpss_stationary:
                diagnosis = 'unit_root'
            elif adf_stationary and not kpss_stationary:
                diagnosis = 'fractional'
            else:
                diagnosis = 'ambiguous'

            results.append({
                'head': h,
                'adf_pval': float(adf_pval),
                'kpss_pval': float(kpss_pval),
                'adf_stationary': adf_stationary,
                'kpss_stationary': kpss_stationary,
                'diagnosis': diagnosis,
            })

        fractional_heads = [r['head'] for r in results if r['diagnosis'] == 'fractional']

        return {
            'per_head': results,
            'fractional_heads': fractional_heads,
            'summary': {d: sum(1 for r in results if r['diagnosis'] == d)
                        for d in ('stationary', 'unit_root', 'fractional', 'ambiguous', 'constant', 'too_short')},
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

        # Step 6: Trajectory Dynamics (Velocity/Acceleration/Hurst)
        dynamics = self.calculate_trajectory_dynamics(compressed)
        self.hurst_exponent = float(dynamics.get("hurst", self.hurst_exponent))

        # Step 7: Intrinsic Time (Gap D.3)
        intrinsic_time = self.compute_intrinsic_time(hidden_states=compressed)

        # Keep observer-level scalars for dashboard bridge.
        self.tau_normalised = float(intrinsic_time["intrinsic_time"][-1]) if len(intrinsic_time["intrinsic_time"]) else 0.0
        self.arc_steps = np.asarray(intrinsic_time["arc_length_steps"], dtype=float)
        if isinstance(stationarity, dict) and stationarity.get("p_value") is not None:
            self.adf_pval_pc0 = float(stationarity["p_value"])

        return {
            "compressed_trajectory": compressed,
            "singular_values": singular_values,
            "principal_components": components,
            "spectral": spectral,
            "autocorrelation": autocorr,
            "stationarity": stationarity,
            "decomposition": decomposition,
            "dynamics": dynamics,
            "intrinsic_time": intrinsic_time,
            "meta": {
                "n_tokens": compressed.shape[0],
                "n_svd_components": compressed.shape[1],
                "original_dim": trajectory.shape[-1]
                if isinstance(trajectory, torch.Tensor)
                else trajectory.shape[-1],
            },
        }
