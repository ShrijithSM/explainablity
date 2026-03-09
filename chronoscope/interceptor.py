"""
The Interceptor: PyTorch Hook-Based Activation Capture Engine.

Attaches to any HuggingFace causal LM and captures the residual stream
(hidden states) at specified layers during forward pass and generation.
Treats captured states as a multivariate time series over token-time.
"""

import gc
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

    def _make_attention_hook(self, name: str):
        """
        Creates a closure that derives per-head metrics from self-attention
        modules (e.g., *.self_attn). This does not store hidden states, only
        compact head-level summaries suitable for time-series analysis.
        """

        def hook_fn(module, input, output):
            attn = None
            # Many HF attention modules return (attn_output, attn_weights, ...)
            if isinstance(output, tuple) and len(output) >= 2:
                candidate = output[1]
                if isinstance(candidate, torch.Tensor):
                    attn = candidate
            elif isinstance(output, torch.Tensor):
                # Some configs may return only attn_output; nothing to do.
                return

            if (
                not self.config.capture_attentions
                or attn is None
                or not isinstance(attn, torch.Tensor)
            ):
                return

            try:
                # Expected shape: [Batch, n_heads, tgt_len, src_len]
                if attn.dim() == 4:
                    last = attn[:, :, -1, :]  # [B, H, S] for last target token
                elif attn.dim() == 3:
                    # Some implementations: [Batch, tgt_len, src_len] (single head)
                    last = attn.unsqueeze(1)[:, :, -1, :]  # [B, 1, S]
                else:
                    return

                if self.config.head_metric == "entropy":
                    eps = 1e-9
                    probs = last.clamp_min(eps)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    entropy = -(probs * probs.log()).sum(dim=-1)  # [B, H]
                    self._head_metrics.setdefault(name, []).append(
                        entropy.detach().cpu().float()
                    )
            except Exception:
                # Head metrics are best-effort; never break the forward pass.
                return

        return hook_fn

    def _register_hooks(self):
        """Walk model tree and attach hooks to layers matching target patterns."""
        attached = 0
        target_module = self.model.model if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Module) else self.model
        for name, module in target_module.named_modules():
            if any(target in name for target in self.config.target_layers):
                # Only hook the top-level decoder layers (e.g., model.layers.0)
                # Avoid hooking sub-modules like layers.0.self_attn.q_proj
                parts = name.split(".")
                # Check if this is a direct decoder layer (pattern: *.layers.N)
                if len(parts) >= 2 and parts[-1].isdigit():
                    handle = module.register_forward_hook(self._make_hook(name))
                    self._hooks.append(handle)
                    attached += 1

            # Additionally, when enabled, attach lightweight hooks to
            # self-attention modules to derive per-head metrics.
            if (
                self.config.capture_attentions
                and name.endswith("self_attn")
            ):
                attn_handle = module.register_forward_hook(
                    self._make_attention_hook(name)
                )
                self._hooks.append(attn_handle)

        console.print(f"[bold green]Interceptor attached to {attached} layers.[/]")

    def _clear(self):
        """Reset activation buffers."""
        self._activations.clear()
        self._head_metrics.clear()

    # ------------------------------------------------------------------ #
    #  Capture: Single Forward Pass
    # ------------------------------------------------------------------ #

    def capture(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Run a single forward pass on the prompt.
        Returns dict of {layer_name: Tensor[1, SeqLen, HiddenDim]}.
        This captures the residual stream for every token in the INPUT.
        """
        self._clear()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self.model(**inputs)

        # Each layer should have exactly 1 tensor (single forward pass)
        result = {}
        for name, tensors in self._activations.items():
            result[name] = tensors[0]  # [1, SeqLen, HiddenDim]

        return result

    # ------------------------------------------------------------------ #
    #  Capture: Autoregressive Generation (Time Series over Token-Time)
    # ------------------------------------------------------------------ #

    def capture_generation_stream(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ):
        """
        Run autoregressive generation in a background thread and yield tokens as they arrive.
        Allows the main thread to process tokens (and their hidden states) concurrently,
        preventing AirLLM's sequential layer loading from head-of-line blocking the CPU.

        Yields:
            (new_text_token: str, current_trajectory: Dict[str, Tensor])
        """
        import threading
        from transformers import TextIteratorStreamer

        self._clear()
        max_tokens = max_new_tokens or self.config.max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for reproducibility
            return_dict_in_generate=True,
            output_hidden_states=False,  # We use hooks instead
            output_attentions=True,      # Ensure attention weights are computed
            streamer=streamer,
        )

        # Launch AirLLM's blocking generate() in a background thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # GC counter — run full gc.collect() every N yields to avoid overhead
        _gc_counter = 0

        # Iterate over tokens natively as soon as they are generated
        for new_text in streamer:
            trajectory = {}
            
            # --- AGGRESSIVE GARBAGE COLLECTION ---
            # Only keep the LAST layer's activations (the one used for analysis).
            # All other layers are intermediate computation waste.
            layer_names = list(self._activations.keys())
            if len(layer_names) > 1:
                target_layer = sorted(layer_names)[-1]  # Keep only deepest layer
                for name in layer_names:
                    if name != target_layer:
                        # Explicitly delete intermediate layer tensors
                        del self._activations[name]
            
            for name, tensors in list(self._activations.items()):
                if not tensors:
                    continue
                # Concatenate all captured timesteps into a single trajectory
                stacked = torch.cat([t.squeeze(0) for t in tensors], dim=0)
                
                # Truncate to a rolling window to prevent infinite CPU RAM accumulation
                max_cache = getattr(self.config, 'max_cache_size', 512)
                if stacked.shape[0] > max_cache:
                    stacked = stacked[-max_cache:]
                    
                # Overwrite backing list with the compacted tensor to free list pointers
                # This collapses N small tensors → 1 tensor, freeing N-1 objects
                old_tensors = self._activations[name]
                self._activations[name] = [stacked.unsqueeze(0)]
                del old_tensors  # Explicitly release the old list
                
                trajectory[name] = stacked
            
            # Periodic full GC pass to reclaim fragmented memory
            _gc_counter += 1
            if _gc_counter % 5 == 0:
                gc.collect()
            
            yield new_text, trajectory

        thread.join()
        
        # Post-generation cleanup
        gc.collect()

    def capture_generation(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """Backward-compatible blocking version."""
        full_text = ""
        final_traj = {}
        for token, traj in self.capture_generation_stream(prompt, max_new_tokens):
            full_text += token
            final_traj = traj
            
        return final_traj, full_text

    # ------------------------------------------------------------------ #
    #  Head Metric Accessors (Attention as Time Series over Heads)
    # ------------------------------------------------------------------ #

    def get_head_metric_series(
        self, layer_name: str
    ) -> torch.Tensor:
        """
        Return the per-head metric time series for a given layer as a tensor
        of shape [Time, n_heads].

        Notes:
            - Assumes batch size 1 (Chronoscope's standard setting).
            - Time corresponds to successive generation steps in which the
              layer's hook observed attention weights.
        """
        metrics = self._head_metrics.get(layer_name, [])
        if not metrics:
            return torch.empty(0)

        # metrics: list of [B, H]; stack over time and drop batch dimension
        stacked = torch.stack(metrics, dim=0)  # [T, B, H]
        if stacked.dim() == 3 and stacked.shape[1] == 1:
            return stacked[:, 0, :]
        return stacked

    # ------------------------------------------------------------------ #
    #  Activation Patching (Causal Intervention)
    # ------------------------------------------------------------------ #

    def patch(
        self,
        prompt: str,
        target_layer_name: str,
        token_indices: List[int],
        noise_type: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Run generation with activation patching: at the specified layer,
        replace the hidden state at specific token positions with noise.

        This is the causal intervention — if the patched token was causally
        necessary, the output should diverge dramatically.

        Args:
            prompt: Input text.
            target_layer_name: Full layer name to patch (e.g., "model.layers.4").
            token_indices: Which token positions to ablate.
            noise_type: "zero", "mean", or "gaussian". Defaults to config.

        Returns:
            patched_trajectory, generated_text
        """
        noise = noise_type or self.config.patching_noise

        # Create a patching hook
        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # Apply noise at specified token positions
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

        # Temporarily replace the hook for the target layer
        patch_handle = None
        target_module = self.model.model if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Module) else self.model
        for name, module in target_module.named_modules():
            if name == target_layer_name:
                patch_handle = module.register_forward_hook(patch_hook)
                break

        if patch_handle is None:
            raise ValueError(
                f"Layer '{target_layer_name}' not found. "
                f"Use list_hookable_layers() to find valid names."
            )

        # Run generation with the patching hook active
        try:
            trajectory, text = self.capture_generation(prompt)
        finally:
            patch_handle.remove()

        return trajectory, text

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """Remove all hooks and force garbage collection."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations.clear()
        
        # Force Python GC + CUDA cache flush
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        console.print("[dim]Interceptor hooks removed + memory released.[/]")
