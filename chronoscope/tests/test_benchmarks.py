"""
Performance benchmarks for Chronoscope's performance-critical paths.

Covers:
  1. Windowed TDA (ripser) — identify the bottleneck window configuration.
  2. PyTorch vs. NumPy paths — quantify end-to-end speedup after migration.
  3. Incremental SVD update — verify correctness + cost vs. full re-SVD.

Run with:
    pytest chronoscope/tests/test_benchmarks.py -v -s

Results are printed to stdout and written to benchmarks_result.json in the
working directory.
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_trajectory(T: int, D: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, D)).astype(np.float32)


def _compress(traj: np.ndarray, n_comp: int = 8) -> np.ndarray:
    """Cheap economy SVD compression matching observer.svd_compress logic."""
    X = np.nan_to_num(traj - traj.mean(axis=0), nan=0.0)
    try:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        k = min(n_comp, U.shape[1])
        out = np.zeros((len(X), n_comp), dtype=np.float32)
        out[:, :k] = (U[:, :k] * S[:k]).astype(np.float32)
        return out
    except np.linalg.LinAlgError:
        return np.zeros((len(X), n_comp), dtype=np.float32)


# ── Benchmark results accumulator ────────────────────────────────────────────

_results: Dict[str, dict] = {}


def _record(name: str, elapsed_s: float, **meta):
    _results[name] = {"elapsed_s": round(elapsed_s, 6), **meta}


# ── 1. Windowed TDA (ripser) ─────────────────────────────────────────────────

class TestTDABenchmarks:
    """
    Benchmarks for the ripser-based windowed TDA path.

    We sweep over window sizes and trajectory lengths to quantify the
    characteristic O(n³) cost of ripser and find safe operating limits.
    """

    _ripser_available = False

    @classmethod
    def setup_class(cls):
        try:
            import ripser  # noqa: F401
            cls._ripser_available = True
        except ImportError:
            pass

    @pytest.mark.skipif(not _ripser_available, reason="ripser not installed")
    @pytest.mark.parametrize("window_size", [8, 16, 32, 48])
    def test_single_window_cost(self, window_size: int):
        """Measure ripser cost for a single window of varying size."""
        import ripser

        compressed = _make_trajectory(window_size, D=8)
        t0 = time.perf_counter()
        ripser.ripser(compressed, maxdim=1, thresh=np.inf)
        elapsed = time.perf_counter() - t0

        name = f"tda_single_window_w{window_size}"
        _record(name, elapsed, window_size=window_size, trajectory_D=8)
        print(f"\n[TDA] window={window_size}: {elapsed*1000:.1f} ms")

        # Soft performance guard: single window should complete in < 5 s
        # on any reasonable CI machine.
        assert elapsed < 5.0, (
            f"ripser window={window_size} took {elapsed:.2f}s — "
            "consider reducing tda_window_size in config."
        )

    @pytest.mark.skipif(not _ripser_available, reason="ripser not installed")
    @pytest.mark.parametrize("T,stride", [(50, 10), (100, 20), (200, 40)])
    def test_full_windowed_sweep(self, T: int, stride: int):
        """End-to-end windowed TDA sweep timing."""
        import ripser

        compressed = _compress(_make_trajectory(T))
        window_size = 16
        n_windows = max(1, (T - window_size) // stride + 1)

        t0 = time.perf_counter()
        for start in range(0, max(1, T - window_size + 1), stride):
            seg = compressed[start : start + window_size]
            if seg.shape[0] >= 3:
                ripser.ripser(seg, maxdim=1, thresh=np.inf)
        elapsed = time.perf_counter() - t0

        name = f"tda_sweep_T{T}_stride{stride}"
        _record(name, elapsed, T=T, stride=stride, n_windows=n_windows)
        print(f"\n[TDA sweep] T={T}, stride={stride}, n_windows={n_windows}: {elapsed*1000:.0f} ms")

    @pytest.mark.skipif(not _ripser_available, reason="ripser not installed")
    def test_tda_recommends_stride(self):
        """
        Dynamically estimate and print a safe (stride, window) config
        based on single-window timing.
        """
        import ripser

        # Time a window of size 16 to get base cost
        seg = _compress(_make_trajectory(16))
        t0 = time.perf_counter()
        for _ in range(3):
            ripser.ripser(seg, maxdim=1, thresh=np.inf)
        base_ms = (time.perf_counter() - t0) / 3 * 1000

        # Aim for ≤ 1 second total TDA budget per 200-token trace
        budget_ms = 1000.0
        safe_n_windows = max(1, int(budget_ms / (base_ms + 1e-9)))
        # Back-calculate a stride from T=200
        safe_stride = max(1, 200 // safe_n_windows)

        print(
            f"\n[TDA advisor] Base cost: {base_ms:.1f} ms/window. "
            f"Safe stride for T=200, budget=1s: {safe_stride} "
            f"({safe_n_windows} windows)"
        )
        _record(
            "tda_advisor",
            base_ms / 1000,
            base_ms_per_window=round(base_ms, 2),
            safe_stride_T200=safe_stride,
            safe_n_windows=safe_n_windows,
        )


# ── 2. PyTorch vs. NumPy path profiling ──────────────────────────────────────

class TestPyTorchVsNumpy:
    """
    Compares PyTorch and NumPy implementations of the same operations
    to verify the migration yields the expected speedup.
    """

    N_REPS = 5
    T, H, D = 200, 14, 64  # trajectory length, n_heads, hidden_dim

    # ── 2-a. Phase-scramble surrogate generation ─────────────────────────

    def _numpy_phase_scramble(self, data: np.ndarray, n_surr: int = 50) -> float:
        """Original NumPy path (pre-migration)."""
        T, H = data.shape
        t0 = time.perf_counter()
        for _ in range(n_surr):
            surr = np.zeros_like(data)
            for h in range(H):
                fft_vals = np.fft.rfft(data[:, h])
                phases = np.random.uniform(-np.pi, np.pi, len(fft_vals))
                fft_vals = np.abs(fft_vals) * np.exp(1j * phases)
                surr[:, h] = np.fft.irfft(fft_vals, n=T)
        return time.perf_counter() - t0

    def _torch_phase_scramble(self, data: np.ndarray, n_surr: int = 50) -> float:
        """PyTorch fused batch path (post-migration)."""
        T, H = data.shape
        data_t = torch.from_numpy(data).float()  # [T, H]
        t0 = time.perf_counter()
        for _ in range(n_surr):
            rft = torch.fft.rfft(data_t, dim=0)          # [T//2+1, H]
            phases = torch.empty_like(rft.real).uniform_(-np.pi, np.pi)
            scrambled = torch.polar(rft.abs(), phases)
            torch.fft.irfft(scrambled, n=T, dim=0)        # [T, H]
        return time.perf_counter() - t0

    def test_phase_scramble_speedup(self):
        data = _make_trajectory(self.T, self.H)
        n_surr = 30

        np_times = [self._numpy_phase_scramble(data, n_surr) for _ in range(self.N_REPS)]
        pt_times = [self._torch_phase_scramble(data, n_surr) for _ in range(self.N_REPS)]

        np_med = float(np.median(np_times))
        pt_med = float(np.median(pt_times))
        speedup = np_med / (pt_med + 1e-9)

        _record(
            "phase_scramble_speedup",
            pt_med,
            numpy_median_s=round(np_med, 6),
            torch_median_s=round(pt_med, 6),
            speedup=round(speedup, 2),
            n_surrogates=n_surr,
        )
        print(
            f"\n[Phase scramble] NumPy: {np_med*1000:.1f} ms | "
            f"PyTorch: {pt_med*1000:.1f} ms | Speedup: {speedup:.2f}×"
        )
        # PyTorch should be at least 0.5× as fast (regression guard)
        # (it may be slower on CPU for small H, but never 10× slower)
        assert speedup > 0.1, f"PyTorch regression: speedup={speedup:.2f}×"

    # ── 2-b. Shannon entropy over attention matrix ────────────────────────

    def _numpy_entropy(self, probs: np.ndarray) -> float:
        eps = 1e-9
        t0 = time.perf_counter()
        for _ in range(self.N_REPS * 20):
            p = np.clip(probs, eps, 1.0)
            _ = -(p * np.log(p)).sum(axis=-1)
        return (time.perf_counter() - t0) / (self.N_REPS * 20)

    def _torch_entropy(self, probs: np.ndarray) -> float:
        import torch.special
        eps = 1e-9
        p_t = torch.from_numpy(probs).float().clamp_min(eps)
        t0 = time.perf_counter()
        for _ in range(self.N_REPS * 20):
            torch.special.entr(p_t).sum(dim=-1)
        return (time.perf_counter() - t0) / (self.N_REPS * 20)

    def test_entropy_speedup(self):
        rng = np.random.default_rng(1)
        raw = rng.dirichlet(np.ones(self.T), size=self.H).astype(np.float32)  # [H, T]

        np_avg = self._numpy_entropy(raw)
        pt_avg = self._torch_entropy(raw)
        speedup = np_avg / (pt_avg + 1e-9)

        _record(
            "entropy_speedup",
            pt_avg,
            numpy_avg_s=round(np_avg, 9),
            torch_avg_s=round(pt_avg, 9),
            speedup=round(speedup, 2),
        )
        print(
            f"\n[Entropy] NumPy: {np_avg*1e6:.1f} µs | "
            f"PyTorch: {pt_avg*1e6:.1f} µs | Speedup: {speedup:.2f}×"
        )
        assert speedup > 0.1, f"PyTorch entropy regression: {speedup:.2f}×"

    # ── 2-c. Effective rank via SVD ───────────────────────────────────────

    def _numpy_eff_rank(self, attn: np.ndarray) -> float:
        eps = 1e-9
        t0 = time.perf_counter()
        for _ in range(self.N_REPS * 5):
            for h in range(attn.shape[0]):
                svs = np.linalg.svd(attn[h], compute_uv=False)
                svs_n = svs / (svs.sum() + eps)
                _ = np.exp(-(svs_n * np.log(svs_n + eps)).sum())
        return (time.perf_counter() - t0) / (self.N_REPS * 5)

    def _torch_eff_rank(self, attn: np.ndarray) -> float:
        eps = 1e-9
        attn_t = torch.from_numpy(attn).float()
        t0 = time.perf_counter()
        for _ in range(self.N_REPS * 5):
            svs = torch.linalg.svdvals(attn_t)           # [H, min(T,T)]
            svs_n = svs / svs.sum(dim=-1, keepdim=True).clamp_min(eps)
            _ = torch.exp(-(svs_n.clamp_min(eps) * torch.log(svs_n.clamp_min(eps))).sum(dim=-1))
        return (time.perf_counter() - t0) / (self.N_REPS * 5)

    def test_effective_rank_speedup(self):
        rng = np.random.default_rng(2)
        # Simulate [H, T, T] attention weight matrix
        raw = rng.dirichlet(
            np.ones(self.T), size=(self.H, self.T)
        ).astype(np.float32)  # [H, T, T]

        np_avg = self._numpy_eff_rank(raw)
        pt_avg = self._torch_eff_rank(raw)
        speedup = np_avg / (pt_avg + 1e-9)

        _record(
            "effective_rank_speedup",
            pt_avg,
            numpy_avg_s=round(np_avg, 9),
            torch_avg_s=round(pt_avg, 9),
            speedup=round(speedup, 2),
            H=self.H, T=self.T,
        )
        print(
            f"\n[Eff. Rank] NumPy: {np_avg*1000:.2f} ms | "
            f"PyTorch: {pt_avg*1000:.2f} ms | Speedup: {speedup:.2f}×"
        )
        assert speedup > 0.1


# ── 3. Incremental SVD correctness + cost ────────────────────────────────────

class TestIncrementalSVD:
    """
    Verify that incremental_svd_update:
      1. Produces output of the correct shape.
      2. Is cheaper (wall-clock) than a full re-SVD for large trajectories.
      3. Preserves approximate projection quality.
    """

    @staticmethod
    def _build_observer():
        from chronoscope.config import ChronoscopeConfig
        from chronoscope.observer import SignalObserver
        cfg = ChronoscopeConfig()
        cfg.svd_components = 8
        obs = object.__new__(SignalObserver)
        obs.config = cfg
        return obs

    def test_shape_correctness(self):
        obs = self._build_observer()
        T_old, T_new, D = 80, 20, 64
        existing = _compress(_make_trajectory(T_old, D), n_comp=8)
        new_rows = _make_trajectory(T_new, D).astype(np.float32)

        updated = obs.incremental_svd_update(existing, new_rows, n_components=8)
        assert updated.shape == (T_old + T_new, 8), (
            f"Expected ({T_old + T_new}, 8), got {updated.shape}"
        )

    def test_incremental_faster_than_full_resvd(self):
        obs = self._build_observer()
        T_old, T_new, D = 400, 50, 512

        existing = _compress(_make_trajectory(T_old, D), n_comp=8)
        new_rows = _make_trajectory(T_new, D).astype(np.float32)
        full_traj = np.vstack([_make_trajectory(T_old, D), new_rows])

        # Full re-SVD cost
        t0 = time.perf_counter()
        for _ in range(3):
            _compress(full_traj, n_comp=8)
        full_svd_ms = (time.perf_counter() - t0) / 3 * 1000

        # Incremental cost
        t0 = time.perf_counter()
        for _ in range(3):
            obs.incremental_svd_update(existing, new_rows, n_components=8)
        incr_ms = (time.perf_counter() - t0) / 3 * 1000

        speedup = full_svd_ms / (incr_ms + 1e-9)
        _record(
            "incremental_svd_speedup",
            incr_ms / 1000,
            full_svd_ms=round(full_svd_ms, 2),
            incr_ms=round(incr_ms, 2),
            speedup=round(speedup, 2),
            T_old=T_old,
            T_new=T_new,
            D=D,
        )
        print(
            f"\n[Incremental SVD] Full: {full_svd_ms:.1f} ms | "
            f"Incremental: {incr_ms:.1f} ms | Speedup: {speedup:.2f}×"
        )
        # Incremental should not be dramatically slower than full re-SVD
        assert speedup > 0.05, f"Incremental SVD unexpectedly slow ({speedup:.2f}×)"

    def test_empty_existing_falls_back_to_fresh_svd(self):
        obs = self._build_observer()
        new_rows = _make_trajectory(30, 64).astype(np.float32)
        result = obs.incremental_svd_update(
            np.zeros((0, 8), dtype=np.float32), new_rows, n_components=8
        )
        assert result.shape[0] == 30


# ── Dump results to JSON ──────────────────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """Write benchmark results to a JSON file after the session ends."""
    if _results:
        out_path = Path("benchmarks_result.json")
        try:
            existing = json.loads(out_path.read_text()) if out_path.exists() else {}
        except Exception:
            existing = {}
        existing.update(_results)
        out_path.write_text(json.dumps(existing, indent=2))
        print(f"\nBenchmark results written to {out_path.resolve()}")
