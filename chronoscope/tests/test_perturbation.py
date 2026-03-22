"""
Unit tests for Gap C perturbation methods.
Tests _make_ablation_hook with zero/mean modes on synthetic tensors.
"""

import numpy as np
import torch
import pytest


@pytest.mark.skip(reason="Legacy manual hooks replaced by pyvene.IntervenableModel")
class TestAblationHook:

    @staticmethod
    def _get_analyzer():
        from chronoscope.config import ChronoscopeConfig
        from chronoscope.analyzer import CausalAnalyzer
        analyzer = object.__new__(CausalAnalyzer)
        analyzer.config = ChronoscopeConfig()
        return analyzer

    def _simulate_hook(self, mode, h_source=1, n_heads=4, head_dim=16, seq_len=10):
        """Run the ablation hook on a synthetic tensor and return output."""
        analyzer = self._get_analyzer()
        hook = analyzer._make_ablation_hook(h_source, head_dim, mode=mode)

        hidden_dim = n_heads * head_dim
        hidden = torch.randn(1, seq_len, hidden_dim)
        output = (hidden, torch.zeros(1))  # mock tuple output

        result = hook(None, None, output)
        return result[0], hidden, h_source, head_dim

    def test_zero_mode_sets_slice_to_zero(self):
        out, original, h, hd = self._simulate_hook('zero')
        start, end = h * hd, (h + 1) * hd
        assert torch.allclose(out[:, :, start:end], torch.zeros_like(out[:, :, start:end]))

    def test_zero_mode_preserves_other_heads(self):
        out, original, h, hd = self._simulate_hook('zero')
        # Head 0 should be unchanged
        np.testing.assert_array_almost_equal(
            out[:, :, :hd].numpy(),
            original[:, :, :hd].numpy(),
            decimal=5
        )

    def test_mean_mode_preserves_scale(self):
        out, original, h, hd = self._simulate_hook('mean')
        start, end = h * hd, (h + 1) * hd
        # Mean-ablated head should have same overall magnitude (roughly)
        orig_norm = original[:, :, start:end].norm().item()
        out_norm = out[:, :, start:end].norm().item()
        # Should be within 5x — mean reduces variance but preserves scale
        assert out_norm > 0.0, "Mean ablation should not zero out"
        assert out_norm < orig_norm * 5, "Mean ablation should not explode scale"

    def test_mean_mode_is_constant_across_sequence(self):
        out, original, h, hd = self._simulate_hook('mean', seq_len=20)
        start, end = h * hd, (h + 1) * hd
        slice_out = out[0, :, start:end]  # [seq_len, head_dim]
        # Mean mode should produce the same vector at every position
        for t in range(1, slice_out.shape[0]):
            assert torch.allclose(slice_out[0], slice_out[t], atol=1e-6)

    def test_gaussian_mode_changes_values(self):
        out, original, h, hd = self._simulate_hook('gaussian')
        start, end = h * hd, (h + 1) * hd
        # Gaussian mode should change the values (extremely unlikely to be identical)
        assert not torch.allclose(out[:, :, start:end], original[:, :, start:end])


class TestCoTSegmenter:

    def test_basic_segmentation(self):
        from chronoscope.cot_segmenter import segment_cot_by_text
        text = "First, I compute the sum. Then, I check the result. Therefore, the answer is 42."
        tokens = list(text)  # character-level tokens for simplicity
        steps = segment_cot_by_text(text, tokens)
        assert len(steps) >= 2, f"Expected >=2 steps, got {len(steps)}"

    def test_step_coverage(self):
        from chronoscope.cot_segmenter import segment_cot_by_text
        text = "Step 1: Read the problem. Step 2: Identify variables. Step 3: Solve."
        tokens = text.split()
        steps = segment_cot_by_text(text, tokens)
        # Should detect at least the "Step N:" markers
        assert len(steps) >= 2

    def test_aggregation_shape(self):
        from chronoscope.cot_segmenter import segment_cot_by_text, aggregate_entropy_by_step, ReasoningStep
        steps = [
            ReasoningStep(0, 0, 5, "step one", 5),
            ReasoningStep(1, 5, 10, "step two", 5),
        ]
        entropy = np.random.randn(10, 4)
        agg = aggregate_entropy_by_step(entropy, steps, agg='mean')
        assert agg.shape == (2, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
