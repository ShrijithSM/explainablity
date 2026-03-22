"""
The Interceptor: PyTorch Hook-Based Activation Capture Engine.

Attaches to any HuggingFace causal LM and captures the residual stream
(hidden states) at specified layers during forward pass and generation.
Treats captured states as a multivariate time series over token-time.

Gap E upgrades:
  - Multi-metric head summary (Shannon, Rényi-2, max attention, effective rank, sink fraction)
  - Attention sink removal (auto-detect or explicit positions)
  - OV circuit value-weighted metric (requires v_proj hook)
"""

import gc
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from rich.console import Console

from .config import ChronoscopeConfig

console = Console()


class ChronoscopeInterceptor:
    """
    Non-invasive neural oscilloscope for extracting latent time-series
    representations from Transformer decoder layers.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: ChronoscopeConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device

        # Storage for captured activations
        # Key: layer_name, Value: list of tensors (one per token during generation)
        self._activations: Dict[str, List[torch.Tensor]] = {}
        # Storage for per-head summary metrics derived from attention weights.
        # Each entry is a list of tensors shaped [Batch, n_heads] for each
        # generation step (treated as a multivariate time series over heads).
        self._head_metrics: Dict[str, List[torch.Tensor]] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self._register_hooks()

    def _infer_model_device(self, model_obj: nn.Module) -> str:
        """
        Infer a safe device for model inputs.

        If model parameters are on "meta" (common with dispatch/offload wrappers),
        keep inputs on CPU and let the generation stack move tensors as needed.
        """
        try:
            param_device = next(model_obj.parameters()).device.type
            if param_device == "meta":
                return "cpu"
            return param_device
        except (StopIteration, AttributeError, TypeError):
            return self.device

    # ------------------------------------------------------------------ #
    #  Hook Management
    # ------------------------------------------------------------------ #

    def _make_hook(self, name: str):
        """Creates a closure that captures the output of a decoder layer."""

        def hook_fn(module, input, output):
            # Decoder layers typically return a tuple:
            #   (hidden_states, present_key_value, attention_weights_optional)
            # We want the hidden_states tensor: [Batch, SeqLen, HiddenDim]
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if isinstance(hidden, torch.Tensor):
                # Detach + CPU to avoid GPU memory blowup during long traces
                self._activations.setdefault(name, []).append(
                    hidden.detach().cpu().float()
                )

        return hook_fn

    # ------------------------------------------------------------------ #
    #  Gap E: Multi-Metric Head Computation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _remove_attention_sink(
        attn_weights_np: np.ndarray,
        sink_positions: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Remove the attention sink from each head's attention distribution
        and re-normalize the remaining weights.

        Sink positions are typically [0] (BOS token) in Qwen/Llama models.
        If sink_positions is None, auto-detect as positions where mean weight
        across all heads exceeds 3× the uniform expectation (1/T).

        Args:
            attn_weights_np: np.ndarray [H, T]
            sink_positions:  list of token indices to treat as sinks

        Returns:
            cleaned: np.ndarray [H, T] — sink positions zeroed, renormalized
        """
        cleaned = np.nan_to_num(
            attn_weights_np.astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        H, T = cleaned.shape

        if sink_positions is None:
            mean_weights = cleaned.mean(axis=0)  # [T]
            uniform_expected = 1.0 / max(T, 1)
            sink_positions = list(np.where(mean_weights > 3 * uniform_expected)[0])

        if not sink_positions:
            return cleaned

        cleaned[:, sink_positions] = 0.0

        row_sums = cleaned.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
        cleaned = cleaned / row_sums
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

        return cleaned

    def _compute_head_metrics(
        self,
        attn_weights: torch.Tensor,
    ) -> dict:
        """
        Compute multiple per-head attention summary statistics for the most
        recent token position.

        Metrics:
            shannon_entropy:  H = -Σ p log p
            renyi_entropy_2:  H₂ = -log(Σ p²)
            max_attention:    max(pᵢ)
            effective_rank:   exp(H_shannon) / T
            sink_fraction:    fraction of weight on token 0

        Returns:
            dict mapping metric_name → np.ndarray [H]
        """
        import torch
        import torch.special
        import torch.linalg

        eps = 1e-9
        last = attn_weights[:, :, -1, :]   # [B, H, T]
        probs = last.clamp_min(eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        B, H_heads, T_ctx = probs.shape

        # ── Effective rank via PyTorch SVD ──────────────────────────────
        attn_batch = attn_weights[0].float()                      # [H, T, T]
        svs_batch  = torch.linalg.svdvals(attn_batch)             # [H, min(T,T)]
        svs_norm   = svs_batch / svs_batch.sum(dim=-1, keepdim=True).clamp_min(eps)
        svs_clamp  = svs_norm.clamp_min(eps)
        eff_rank_t = torch.exp(-(svs_clamp * torch.log(svs_clamp)).sum(dim=-1)) # [H]

        # ── PyTorch Moments & Entropy calculations ──────────────────────
        shannon_t = torch.special.entr(probs).sum(dim=-1)         # [B, H]
        renyi_2_t = -torch.log(probs.pow(2).sum(dim=-1) + eps)    # [B, H]
        max_attn_t = probs.max(dim=-1)[0]                         # [B, H]
        sink_frac_t = probs[:, :, 0]                              # [B, H]

        attn_var_t = torch.var(probs, dim=-1, correction=0)       # [B, H]
        
        mu = probs.mean(dim=-1, keepdim=True)                     # [B, H, 1]
        deviations = probs - mu
        m2 = (deviations.pow(2)).mean(dim=-1)
        m4 = (deviations.pow(4)).mean(dim=-1)
        kurtosis_t = m4 / m2.pow(2).clamp(min=eps) - 3.0          # [B, H]

        p_np = probs[0].detach().float().cpu().numpy().astype(np.float64)

        # ── Optional Numpy attention sink removal (legacy fallback) ─────
        if self.config.remove_attention_sink:
            p_np = self._remove_attention_sink(
                p_np,
                sink_positions=self.config.attention_sink_positions
            )
            # Recompute entropy on cleaned attention with robust numerics.
            p_np = np.nan_to_num(p_np, nan=0.0, posinf=0.0, neginf=0.0)
            log_p = np.log(np.clip(p_np, eps, 1.0))
            shannon_np = -np.where(p_np > 0, p_np * log_p, 0.0).sum(axis=1)
            renyi_2_np = -np.log((p_np ** 2).sum(axis=1) + eps)
            max_attn_np = p_np.max(axis=1)
            sink_frac_np = p_np[:, 0]
        else:
            shannon_np = shannon_t[0].detach().cpu().numpy()
            renyi_2_np = renyi_2_t[0].detach().cpu().numpy()
            max_attn_np = max_attn_t[0].detach().cpu().numpy()
            sink_frac_np = sink_frac_t[0].detach().cpu().numpy()

        eff_rank_np = eff_rank_t.detach().cpu().numpy()
        attn_var_np = attn_var_t[0].detach().cpu().numpy()
        kurtosis_np = kurtosis_t[0].detach().cpu().numpy()

        # Rich feature engineering for long-CoT time-series analysis.
        p_sorted = np.sort(p_np, axis=1)
        top3_mass_np = p_sorted[:, -min(3, p_sorted.shape[1]):].sum(axis=1)
        top5_mass_np = p_sorted[:, -min(5, p_sorted.shape[1]):].sum(axis=1)

        positions = np.arange(p_np.shape[1], dtype=np.float64)
        mean_pos = (p_np * positions[None, :]).sum(axis=1)
        var_pos = (p_np * (positions[None, :] - mean_pos[:, None]) ** 2).sum(axis=1)
        spread_std_np = np.sqrt(np.clip(var_pos, 0.0, None))
        argmax_pos_norm_np = np.argmax(p_np, axis=1) / max(1, p_np.shape[1] - 1)

        # Gini-like concentration on sorted attention weights.
        n = p_sorted.shape[1]
        gini_num = (np.arange(1, n + 1)[None, :] * p_sorted).sum(axis=1)
        gini_np = (2.0 * gini_num / max(n, 1)) - (n + 1) / max(n, 1)
        gini_np = np.clip(gini_np, 0.0, 1.0)

        return {
            'shannon_entropy': shannon_np,
            'renyi_entropy_2': renyi_2_np,
            'max_attention': max_attn_np,
            'effective_rank': eff_rank_np,
            'sink_fraction': sink_frac_np,
            'variance': attn_var_np,
            'kurtosis': kurtosis_np,
            'top3_mass': top3_mass_np,
            'top5_mass': top5_mass_np,
            'argmax_pos_norm': argmax_pos_norm_np,
            'spread_std': spread_std_np,
            'gini': gini_np,
        }

    def _compute_ov_weighted_metric(
        self,
        prompt: str,
        layer_idx: int
    ) -> dict:
        """
        Gap E.3 — OV circuit capture via nnterp trace context.
        Captures model.attentions_output[N] without needing a dual-hook pattern.
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with self.model.trace(inputs.input_ids):
            attn_out = self.model.attentions_output[layer_idx].save()
            
        val = attn_out.value.detach().cpu().numpy()
        return {
            'ov_metric': val
        }

    def _make_attention_hook(self, name: str):
        """
        Creates a closure that derives per-head metrics from self-attention
        modules (e.g., *.self_attn). Now computes the full multi-metric
        dictionary (Gap E.1) instead of just Shannon entropy.
        """

        def hook_fn(module, input, output):
            attn = None
            if isinstance(output, tuple) and len(output) >= 2:
                candidate = output[1]
                if isinstance(candidate, torch.Tensor):
                    attn = candidate
            elif isinstance(output, torch.Tensor):
                return

            if (
                not self.config.capture_attentions
                or attn is None
                or not isinstance(attn, torch.Tensor)
            ):
                return

            try:
                if attn.dim() == 4:
                    # Full attn weights: [B, H, tgt_len, src_len]
                    pass  # use full tensor for multi-metric
                elif attn.dim() == 3:
                    attn = attn.unsqueeze(1)  # [B, 1, T, T]
                else:
                    return

                metrics = self._compute_head_metrics(attn)  # dict of [H] arrays
                layer_key = name.replace(".self_attn", "")
                self._append_head_metrics(layer_key, metrics)
            except Exception:
                # Head metrics are best-effort; never break the forward pass.
                return

        return hook_fn

    def _append_head_metrics(self, layer_key: str, metrics: dict):
        """Store per-head metrics in scalar or vector mode."""
        if self.config.head_feature_mode == 'vector':
            feature_vector = np.stack([
                metrics['shannon_entropy'],
                metrics['renyi_entropy_2'],
                metrics['max_attention'],
                metrics['effective_rank'],
                metrics['sink_fraction'],
                metrics['variance'],
                metrics['kurtosis'],
                metrics['top3_mass'],
                metrics['top5_mass'],
                metrics['argmax_pos_norm'],
                metrics['spread_std'],
                metrics['gini'],
            ], axis=1)  # [H, 12]
            self._head_metrics.setdefault(layer_key, []).append(
                torch.from_numpy(feature_vector).float()
            )
            return

        metric_key = self.config.head_metric_type
        if metric_key not in metrics:
            metric_key = 'shannon_entropy'
        scalar_vals = metrics[metric_key]  # [H]
        self._head_metrics.setdefault(layer_key, []).append(
            torch.from_numpy(scalar_vals).unsqueeze(0).float()  # [1, H]
        )

    def _populate_head_metrics_from_generation_attentions(self, outputs):
        """
        Fallback for generation backends where forward hooks do not expose
        attention weights. Uses returned generate() attentions instead.
        """
        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return

        try:
            # Expected shape: tuple[timestep] -> tuple[layer] -> tensor [B, H, T, S]
            for step_attn in attentions:
                if not isinstance(step_attn, (tuple, list)):
                    continue

                for layer_idx, attn in enumerate(step_attn):
                    if not isinstance(attn, torch.Tensor):
                        continue

                    if attn.dim() == 3:
                        attn = attn.unsqueeze(1)
                    elif attn.dim() != 4:
                        continue

                    metrics = self._compute_head_metrics(attn)
                    self._append_head_metrics(f"layers.{layer_idx}", metrics)
        except Exception:
            # Best-effort fallback only.
            return

    def _register_hooks(self):
        """Walk model tree and attach hooks to layers matching target patterns."""
        attached = 0
        target_module = self.model.model if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Module) else self.model
        for name, module in target_module.named_modules():
            if any(target in name for target in self.config.target_layers):
                # Only hook the top-level decoder layers (e.g., model.layers.0)
                parts = name.split(".")
                if len(parts) >= 2 and parts[-1].isdigit():
                    handle = module.register_forward_hook(self._make_hook(name))
                    self._hooks.append(handle)
                    attached += 1

            if (
                self.config.capture_attentions
                and name.endswith("self_attn")
            ):
                attn_handle = module.register_forward_hook(
                    self._make_attention_hook(name)
                )
                self._hooks.append(attn_handle)

        console.print(f"[bold green]Interceptor attached to {attached} layers.[/]")

    def _resolve_generation_model(self):
        """Return the concrete model object to call for generation/forward."""
        generation_model = self.model
        if hasattr(generation_model, "local_model") and callable(getattr(generation_model, "generate", None)):
            generation_model = self.model
        elif hasattr(generation_model, "_model") and callable(getattr(generation_model, "generate", None)):
            generation_model = self.model
        elif hasattr(self.model, "local_model"):
            generation_model = self.model.local_model
        elif hasattr(self.model, "_model"):
            generation_model = self.model._model
        return generation_model

    def _clear(self):
        """Reset activation buffers."""
        self._activations.clear()
        self._head_metrics.clear()

    # ------------------------------------------------------------------ #
    #  Capture: Single Forward Pass
    # ------------------------------------------------------------------ #

    def capture(self, prompt: str) -> Dict[str, torch.Tensor]:
        self._clear()
        model_device = self._infer_model_device(self.model)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)

        with torch.no_grad():
            self.model(
                **inputs,
                output_attentions=bool(self.config.capture_attentions),
                return_dict=True,
            )

        result = {}
        for name, tensors in self._activations.items():
            if tensors:
                result[name] = tensors[0]

        return result

    # ------------------------------------------------------------------ #
    #  Capture: Autoregressive Generation
    # ------------------------------------------------------------------ #

    def capture_generation_stream(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ):
        self._clear()
        max_tokens = max_new_tokens or self.config.max_new_tokens

        generation_model = self._resolve_generation_model()

        model_device = self._infer_model_device(generation_model)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=False,
            output_attentions=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        outputs = generation_model.generate(**generation_kwargs)

        if self.config.capture_attentions and not self._head_metrics:
            self._populate_head_metrics_from_generation_attentions(outputs)

        trajectory = {}
        # Keep all captured layers so downstream analysis can recurse over
        # the full depth rather than only the deepest layer.
        for name, tensors in list(self._activations.items()):
            if not tensors:
                continue
            stacked = torch.cat([t.squeeze(0) for t in tensors], dim=0)
            max_cache = getattr(self.config, 'max_cache_size', 512)
            if stacked.shape[0] > max_cache:
                stacked = stacked[-max_cache:]
                
            self._activations[name] = [stacked.unsqueeze(0)]
            trajectory[name] = stacked
        
        gc.collect()
        
        new_text = self.tokenizer.decode(outputs.sequences[0, inputs["input_ids"].shape[1]:])
        yield new_text, trajectory

    def capture_head_metrics_from_forward(self, prompt: str):
        """
        Populate head metrics from a direct forward pass with
        output_attentions=True. Used when generate() does not expose
        attention tensors for hooks/fallback extraction.
        """
        self._head_metrics.clear()
        generation_model = self._resolve_generation_model()
        model_device = self._infer_model_device(generation_model)
        if model_device == "cpu" and torch.cuda.is_available() and self.config.device.startswith("cuda"):
            # Some accelerated/quantized models report meta/cpu parameters while
            # executing on CUDA through dispatch; forcing CUDA avoids false CPU
            # placement and silent fallback failure.
            model_device = self.config.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)

        try:
            with torch.no_grad():
                outputs = generation_model(
                    **inputs,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
        except Exception as e:
            console.print(f"[yellow]capture_head_metrics_from_forward failed: {e}[/]")
            return

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return

        try:
            for layer_idx, attn in enumerate(attentions):
                if not isinstance(attn, torch.Tensor):
                    continue
                if attn.dim() == 3:
                    attn = attn.unsqueeze(1)
                elif attn.dim() != 4:
                    continue

                metrics = self._compute_head_metrics(attn)
                self._append_head_metrics(f"layers.{layer_idx}", metrics)
        except Exception as e:
            console.print(f"[yellow]head metric extraction from forward attentions failed: {e}[/]")
            return

    def capture_generation(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        full_text = ""
        final_traj = {}
        for token, traj in self.capture_generation_stream(prompt, max_new_tokens):
            full_text += token
            final_traj = traj
        return final_traj, full_text

    def get_token_labels(self, prompt: str, generated_text: str = "") -> List[str]:
        """Convert prompt and generated text into token labels for plotting."""
        p_tokens = self.tokenizer.tokenize(prompt)
        g_tokens = self.tokenizer.tokenize(generated_text) if generated_text else []
        full = p_tokens + g_tokens
        return [f"{i}:{t}" for i, t in enumerate(full)]

    # ------------------------------------------------------------------ #
    #  Head Metric Accessors
    # ------------------------------------------------------------------ #

    def get_head_metric_series(self, layer_name: str) -> torch.Tensor:
        """
        Return the head metric time series for a given layer.

        In scalar mode (config.head_feature_mode == 'scalar'):
            Each timestep produces [1, H] → stacked to [T, 1, H] → returned as [T, H]

        In vector mode (config.head_feature_mode == 'vector'):
            Each timestep produces [H, F] → stacked to [T, H, F] → returned as [T, H, F]
            Downstream can reshape to [T, H*F] for VAR if needed.
        """
        metrics = self._head_metrics.get(layer_name, [])
        if not metrics:
            return torch.empty(0)

        stacked = torch.stack(metrics, dim=0)  # [T, ...]

        if self.config.head_feature_mode == 'vector':
            # stacked shape: [T, H, 5] — pass through
            return stacked
        else:
            # Scalar mode: stacked shape is [T, 1, H] — squeeze batch dim
            if stacked.dim() == 3 and stacked.shape[1] == 1:
                return stacked[:, 0, :]  # [T, H]
            elif stacked.dim() == 2:
                return stacked  # already [T, H]
            else:
                # Fallback: best effort — take first slice of dim=1
                return stacked[:, 0, :] if stacked.dim() >= 3 else stacked

    def debug_head_metric_summary(self) -> Dict[str, int]:
        """Return count of captured metric timesteps per layer key."""
        return {k: len(v) for k, v in self._head_metrics.items()}

    # ------------------------------------------------------------------ #
    #  Causal Interventions (Patching)
    # ------------------------------------------------------------------ #

    def patch(
        self,
        prompt: str,
        target_layer_name: str,
        token_indices: List[int],
        noise_type: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        noise = noise_type or self.config.patching_noise

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            for idx in token_indices:
                if idx < hidden.shape[1]:
                    if noise == "zero":
                        hidden[:, idx, :] = 0.0
                    elif noise == "mean":
                        hidden[:, idx, :] = hidden.mean(dim=1)
                    elif noise == "gaussian":
                        hidden[:, idx, :] = torch.randn_like(hidden[:, idx, :])

            if rest is not None:
                return (hidden,) + rest
            return hidden

        patch_handle = None
        target_root = self.model.model if hasattr(self.model, 'model') else self.model
        for name, module in target_root.named_modules():
            if name == target_layer_name:
                patch_handle = module.register_forward_hook(patch_hook)
                break

        if patch_handle is None:
            raise ValueError(f"Layer '{target_layer_name}' not found.")

        try:
            trajectory, text = self.capture_generation(prompt)
        finally:
            patch_handle.remove()

        return trajectory, text

    def patch_attention_heads(
        self,
        prompt: str,
        target_layer_name: str,
        head_indices: List[int],
        ablation_type: str = "zero",
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """Ablate specific attention heads by hooking self_attn."""

        def head_patch_hook(module, input, output):
            if isinstance(output, tuple):
                attn_out = output[0]
                rest = output[1:]
            else:
                attn_out = output
                rest = None

            b, s, d = attn_out.shape
            n_heads = getattr(self.model.config, "num_attention_heads", 1)
            head_dim = d // n_heads

            reshaped = attn_out.view(b, s, n_heads, head_dim)
            for h_idx in head_indices:
                if h_idx < n_heads:
                    if ablation_type == "zero":
                        reshaped[:, :, h_idx, :] = 0.0
                    elif ablation_type == "gaussian":
                        reshaped[:, :, h_idx, :] = torch.randn_like(reshaped[:, :, h_idx, :])

            attn_out = reshaped.reshape(b, s, d)
            if rest is not None:
                return (attn_out,) + rest
            return attn_out

        handle = None
        attn_module_name = f"{target_layer_name}.self_attn"
        target_root = self.model.model if hasattr(self.model, 'model') else self.model
        for name, module in target_root.named_modules():
            if name == attn_module_name:
                handle = module.register_forward_hook(head_patch_hook)
                break

        if handle is None:
            raise ValueError(f"Attention module '{attn_module_name}' not found.")

        try:
            trajectory, text = self.capture_generation(prompt)
        finally:
            handle.remove()

        return trajectory, text

    def patch_attention_heads_multi(
        self,
        prompt: str,
        layer_to_heads: Dict[str, List[int]],
        ablation_type: str = "zero",
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """Ablate attention heads across multiple layers in a single run."""

        handles: List[torch.utils.hooks.RemovableHook] = []
        target_root = self.model.model if hasattr(self.model, 'model') else self.model

        def make_head_patch_hook(head_indices: List[int]):
            def head_patch_hook(module, input, output):
                if isinstance(output, tuple):
                    attn_out = output[0]
                    rest = output[1:]
                else:
                    attn_out = output
                    rest = None

                b, s, d = attn_out.shape
                n_heads = getattr(self.model.config, "num_attention_heads", 1)
                head_dim = d // n_heads

                reshaped = attn_out.view(b, s, n_heads, head_dim)
                for h_idx in head_indices:
                    if h_idx < n_heads:
                        if ablation_type == "zero":
                            reshaped[:, :, h_idx, :] = 0.0
                        elif ablation_type == "gaussian":
                            reshaped[:, :, h_idx, :] = torch.randn_like(reshaped[:, :, h_idx, :])

                attn_out = reshaped.reshape(b, s, d)
                if rest is not None:
                    return (attn_out,) + rest
                return attn_out

            return head_patch_hook

        for layer_name, head_indices in layer_to_heads.items():
            attn_module_name = f"{layer_name}.self_attn"
            matched = False
            for name, module in target_root.named_modules():
                if name == attn_module_name:
                    handles.append(module.register_forward_hook(make_head_patch_hook(head_indices)))
                    matched = True
                    break

            if not matched:
                for h in handles:
                    h.remove()
                raise ValueError(f"Attention module '{attn_module_name}' not found.")

        try:
            trajectory, text = self.capture_generation(prompt)
        finally:
            for h in handles:
                h.remove()

        return trajectory, text

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations.clear()
        self._head_metrics.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        console.print("[dim]Interceptor hooks removed + memory released.[/]")
