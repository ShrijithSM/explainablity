"""
Unit tests for Gap B statistical significance methods.
Tests _granger_pvalue_matrix and _apply_fdr_correction on synthetic VAR data.
"""

import numpy as np
import pytest


def _make_causal_var_data(T=200, H=4, seed=42):
    """
    Generate a synthetic VAR(1) dataset where head 0 → head 1 is causal
    with coefficient 0.8, and all other interactions are zero.
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((T, H))
    data[0] = rng.normal(0, 1, H)

    for t in range(1, T):
        noise = rng.normal(0, 0.3, H)
        data[t] = noise
        data[t, 1] += 0.8 * data[t - 1, 0]  # head0 → head1

    return data


class TestGrangerPvalueMatrix:

    @staticmethod
    def _fit_var(data, lag=1):
        from statsmodels.tsa.vector_ar.var_model import VAR
        model = VAR(data)
        return model.fit(maxlags=lag)

    @staticmethod
    def _get_analyzer():
        from chronoscope.config import ChronoscopeConfig
        from chronoscope.analyzer import CausalAnalyzer
        analyzer = object.__new__(CausalAnalyzer)
        analyzer.config = ChronoscopeConfig()
        return analyzer

    def test_shape_and_diagonal(self):
        analyzer = self._get_analyzer()
        data = _make_causal_var_data()
        res = self._fit_var(data)
        pvals = analyzer._granger_pvalue_matrix(res, data, max_lag=1)
        assert pvals.shape == (4, 4)
        # Diagonal should be NaN
        for i in range(4):
            assert np.isnan(pvals[i, i])

    def test_detects_true_causality(self):
        analyzer = self._get_analyzer()
        data = _make_causal_var_data(T=300)
        res = self._fit_var(data)
        pvals = analyzer._granger_pvalue_matrix(res, data, max_lag=1)
        # head0 → head1: pvals[1, 0] should be very small
        assert pvals[1, 0] < 0.05, f"Expected significant, got p={pvals[1, 0]}"

    def test_non_causal_pairs_large_pval(self):
        analyzer = self._get_analyzer()
        data = _make_causal_var_data(T=300)
        res = self._fit_var(data)
        pvals = analyzer._granger_pvalue_matrix(res, data, max_lag=1)
        # head2 → head3 should NOT be significant
        assert pvals[3, 2] > 0.05 or np.isnan(pvals[3, 2])


class TestFDRCorrection:

    @staticmethod
    def _get_analyzer():
        from chronoscope.config import ChronoscopeConfig
        from chronoscope.analyzer import CausalAnalyzer
        analyzer = object.__new__(CausalAnalyzer)
        analyzer.config = ChronoscopeConfig()
        return analyzer

    def test_fewer_rejections_than_raw(self):
        analyzer = self._get_analyzer()
        # Create a p-value matrix with some marginal values
        pvals = np.full((4, 4), np.nan)
        for i in range(4):
            for j in range(4):
                if i != j:
                    pvals[i, j] = 0.04  # All marginal

        fdr = analyzer._apply_fdr_correction(pvals, alpha=0.05)
        # BH correction on 12 marginal-p tests should reject fewer
        raw_sig = np.sum(pvals[~np.isnan(pvals)] < 0.05)
        assert fdr['n_significant'] <= raw_sig

    def test_strong_signal_survives_fdr(self):
        analyzer = self._get_analyzer()
        pvals = np.full((4, 4), np.nan)
        for i in range(4):
            for j in range(4):
                if i != j:
                    pvals[i, j] = 0.5  # All non-significant

        # Inject one very strong pair
        pvals[1, 0] = 0.001

        fdr = analyzer._apply_fdr_correction(pvals, alpha=0.05)
        assert fdr['reject_matrix'][1, 0] == True, "Strong signal should survive FDR"
        assert fdr['n_significant'] >= 1

    def test_significant_pairs_sorted(self):
        analyzer = self._get_analyzer()
        pvals = np.full((3, 3), np.nan)
        pvals[0, 1] = 0.001
        pvals[1, 0] = 0.002
        pvals[0, 2] = 0.5
        pvals[2, 0] = 0.5
        pvals[1, 2] = 0.5
        pvals[2, 1] = 0.5

        fdr = analyzer._apply_fdr_correction(pvals, alpha=0.05)
        if len(fdr['significant_pairs']) >= 2:
            # Should be sorted by corrected p-value ascending
            p1 = fdr['significant_pairs'][0][2]
            p2 = fdr['significant_pairs'][1][2]
            assert p1 <= p2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
