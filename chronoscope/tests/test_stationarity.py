"""
Unit tests for Gap A stationarity methods.
Tests _test_per_head_stationarity, _apply_selective_differencing,
and joint_stationarity_test on synthetic signals.

NOTE: Some tests may fail due to a known `deprecated_kwarg()` bug in
certain statsmodels versions (0.14.x + pandas compat). These are
environment-specific and not code bugs.
"""

import numpy as np
import pytest

# Guard against broken statsmodels installations
_statsmodels_broken = False
try:
    from statsmodels.tsa.stattools import adfuller
    adfuller(np.random.randn(50), autolag='AIC')
except TypeError:
    _statsmodels_broken = True

_sm_skip = pytest.mark.skipif(
    _statsmodels_broken,
    reason="statsmodels deprecated_kwarg() bug in this environment"
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_stationary_series(T=100, H=4, seed=42):
    """Generate T×H iid Gaussian noise — guaranteed stationary."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (T, H))


def _make_nonstationary_series(T=100, H=4, seed=42):
    """Generate T×H integrated random walk — guaranteed non-stationary."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, (T, H)), axis=0)


def _make_mixed_series(T=100, H=4, seed=42):
    """First 2 heads stationary, last 2 heads non-stationary."""
    stat = _make_stationary_series(T, 2, seed)
    nonstat = _make_nonstationary_series(T, 2, seed + 1)
    return np.hstack([stat, nonstat])


# ── A.1: Per-head ADF ─────────────────────────────────────────────────────

@_sm_skip
class TestPerHeadStationarity:

    @staticmethod
    def _get_analyzer():
        """Build a minimal CausalAnalyzer without loading a model."""
        from chronoscope.config import ChronoscopeConfig
        cfg = ChronoscopeConfig()
        # Import the class — we only need the _test_per_head_stationarity method
        from chronoscope.analyzer import CausalAnalyzer
        # Create a bare analyzer (interceptor/observer are unused for this test)
        analyzer = object.__new__(CausalAnalyzer)
        analyzer.config = cfg
        return analyzer

    def test_stationary_all_pass(self):
        analyzer = self._get_analyzer()
        series = _make_stationary_series()
        result = analyzer._test_per_head_stationarity(series)
        assert result['p_values'].shape == (4,)
        # All heads should be stationary (p < 0.05)
        assert result['is_stationary'].all(), f"Expected all stationary, got p_values={result['p_values']}"
        assert not result['needs_diff']

    def test_nonstationary_all_flagged(self):
        analyzer = self._get_analyzer()
        series = _make_nonstationary_series()
        result = analyzer._test_per_head_stationarity(series)
        # All heads should be non-stationary (p > 0.05)
        assert not result['is_stationary'].any(), f"Expected none stationary, got p_values={result['p_values']}"
        assert result['needs_diff']
        assert result['diff_mask'].all()

    def test_mixed_correct_mask(self):
        analyzer = self._get_analyzer()
        series = _make_mixed_series(T=200)
        result = analyzer._test_per_head_stationarity(series)
        # First 2 heads should be stationary, last 2 non-stationary
        assert result['is_stationary'][0] and result['is_stationary'][1]
        assert result['diff_mask'][2] and result['diff_mask'][3]


# ── A.2: Selective Differencing ────────────────────────────────────────────

class TestSelectiveDifferencing:

    @staticmethod
    def _get_analyzer():
        from chronoscope.config import ChronoscopeConfig
        from chronoscope.analyzer import CausalAnalyzer
        analyzer = object.__new__(CausalAnalyzer)
        analyzer.config = ChronoscopeConfig()
        return analyzer

    def test_output_shape(self):
        analyzer = self._get_analyzer()
        series = _make_mixed_series(T=50, H=4)
        diff_mask = np.array([False, False, True, True])
        out, _ = analyzer._apply_selective_differencing(series, diff_mask)
        assert out.shape == (49, 4)  # T-1 rows

    def test_undifferenced_heads_trimmed_only(self):
        analyzer = self._get_analyzer()
        series = _make_stationary_series(T=20, H=2)
        diff_mask = np.array([False, False])
        out, _ = analyzer._apply_selective_differencing(series, diff_mask)
        # Undifferenced heads should be trimmed first row only
        np.testing.assert_array_almost_equal(out[:, 0], series[1:, 0])

    def test_differenced_heads_are_diff(self):
        analyzer = self._get_analyzer()
        series = _make_nonstationary_series(T=20, H=2)
        diff_mask = np.array([True, True])
        out, _ = analyzer._apply_selective_differencing(series, diff_mask)
        expected = np.diff(series[:, 0])
        np.testing.assert_array_almost_equal(out[:, 0], expected)


# ── A.5: Joint ADF+KPSS ───────────────────────────────────────────────────

@_sm_skip
class TestJointStationarity:

    @staticmethod
    def _get_observer():
        from chronoscope.config import ChronoscopeConfig
        from chronoscope.observer import SignalObserver
        return SignalObserver(ChronoscopeConfig())

    def test_diagnoses_stationary(self):
        observer = self._get_observer()
        series = _make_stationary_series(T=100, H=2)
        result = observer.joint_stationarity_test(series)
        for r in result['per_head']:
            assert r['diagnosis'] in ('stationary', 'ambiguous'), f"Head {r['head']} got {r['diagnosis']}"

    def test_diagnoses_unit_root(self):
        observer = self._get_observer()
        series = _make_nonstationary_series(T=100, H=2)
        result = observer.joint_stationarity_test(series)
        for r in result['per_head']:
            assert r['diagnosis'] in ('unit_root', 'fractional'), f"Head {r['head']} got {r['diagnosis']}"

    def test_summary_keys(self):
        observer = self._get_observer()
        series = _make_mixed_series()
        result = observer.joint_stationarity_test(series)
        assert 'per_head' in result
        assert 'fractional_heads' in result
        assert 'summary' in result
        assert set(result['summary'].keys()) == {'stationary', 'unit_root', 'fractional', 'ambiguous', 'constant', 'too_short'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
