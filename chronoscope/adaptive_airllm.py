"""
AdaptiveAirLLM — Auto-scaling layer batching for AirLLM.

Instead of loading 1 layer at a time (vanilla AirLLM), this wrapper:
1. Probes available VRAM at runtime
2. Measures per-layer memory cost by loading a single shard
3. Computes optimal `layers_per_chunk` to maximise GPU metal
4. Overrides the forward pass to load/execute/unload in chunks
5. Has OOM-safe fallback: if a chunk OOMs, halves chunk size and retries

Works with ANY model (Qwen, LLaMA, Mistral, etc.) and ANY GPU.
"""

import gc
import time
import torch
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from transformers.modeling_outputs import CausalLMOutputWithPast

from rich.console import Console

console = Console()

# ── VRAM helpers ─────────────────────────────────────────────────────────

def get_free_vram_bytes(device_id: int = 0) -> int:
    """Return free VRAM in bytes for the given CUDA device."""
    props = torch.cuda.get_device_properties(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    return props.total_memory - reserved


def get_total_vram_bytes(device_id: int = 0) -> int:
    props = torch.cuda.get_device_properties(device_id)
    return props.total_memory


# ── Adaptive wrapper ────────────────────────────────────────────────────

class AdaptiveAirLLM:
    """
    Wraps an AirLLM model and replaces its forward() with a chunked version
    that loads N layers simultaneously based on available VRAM.
    
    OOM-safe: if loading a chunk causes OOM, automatically halves chunk
    size and retries. This makes it safe on any hardware.
    """

    def __init__(self, base_model, vram_headroom_gb: float = 1.0):
        self._base = base_model
        self._headroom_bytes = int(vram_headroom_gb * (1024 ** 3))

        # Detect device id
        dev_str = getattr(base_model, "running_device", "cuda:0")
        self._device_id = int(dev_str.split(":")[-1]) if ":" in dev_str else 0

        # Compute layer budget
        self._per_layer_bytes = self._estimate_layer_size()
        
        # Account for AirLLM's init_model() overhead (~1.5GB for meta model + buffers)
        # The headroom must cover: activations + attention masks + init_model buffers
        init_overhead = int(1.5 * (1024 ** 3))  # ~1.5GB for model skeleton
        total_overhead = self._headroom_bytes + init_overhead
        
        self._layers_per_chunk = self._compute_chunk_size(total_overhead)

        total_vram = get_total_vram_bytes(self._device_id)
        n_layers = len(self._base.layer_names)
        n_chunks = max(1, -(-n_layers // self._layers_per_chunk))

        console.print(
            f"\n[bold magenta]━━━ AdaptiveAirLLM ━━━[/]\n"
            f"  GPU:               {torch.cuda.get_device_name(self._device_id)}\n"
            f"  VRAM:              {total_vram / 1024**3:.2f} GB\n"
            f"  Headroom:          {total_overhead / 1024**3:.2f} GB "
            f"(user: {self._headroom_bytes / 1024**3:.1f} + init: {init_overhead / 1024**3:.1f})\n"
            f"  Per-layer size:    {self._per_layer_bytes / 1024**2:.1f} MB\n"
            f"  Total layers:      {n_layers}\n"
            f"  [bold green]Layers per chunk:  {self._layers_per_chunk}[/]\n"
            f"  Chunks needed:     {n_chunks}\n"
            f"  Speedup vs 1-at-a-time: ~{n_layers / max(1, n_chunks):.1f}x I/O reduction\n"
        )

        # Monkey-patch the base model's forward
        self._base._original_forward = self._base.forward
        self._base.forward = self._chunked_forward
        self._base.__call__ = lambda *a, **kw: self._chunked_forward(*a, **kw)

    # ── Transparent delegation ──────────────────────────────────────────

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __call__(self, *args, **kwargs):
        return self._chunked_forward(*args, **kwargs)

    # ── Auto-scaling logic ──────────────────────────────────────────────

    def _estimate_layer_size(self) -> int:
        """
        Estimate GPU memory cost of one decoder layer by loading it,
        measuring, and unloading. Model-agnostic.
        """
        from airllm.utils import load_layer, clean_memory

        if len(self._base.layer_names) < 3:
            return 100 * 1024 * 1024

        test_layer_name = self._base.layer_names[1]

        torch.cuda.reset_peak_memory_stats(self._device_id)
        before = torch.cuda.memory_allocated(self._device_id)

        state_dict = load_layer(self._base.checkpoint_path, test_layer_name)
        self._base.move_layer_to_device(state_dict)

        after = torch.cuda.memory_allocated(self._device_id)
        layer_bytes = after - before

        # Unload
        self._base.layers[1].to("meta")
        clean_memory()

        if layer_bytes <= 0:
            layer_bytes = torch.cuda.max_memory_allocated(self._device_id) - before
        if layer_bytes <= 0:
            layer_bytes = 100 * 1024 * 1024

        return layer_bytes

    def _compute_chunk_size(self, total_overhead_bytes: int) -> int:
        """
        Compute layers per chunk. Auto-scales to ANY model × ANY GPU.
        """
        free = get_free_vram_bytes(self._device_id)
        usable = free - total_overhead_bytes

        if usable <= 0 or self._per_layer_bytes <= 0:
            return 1

        n = max(1, int(usable // self._per_layer_bytes))
        n = min(n, len(self._base.layer_names))

        return n

    # ── Chunked forward pass ────────────────────────────────────────────

    def _chunked_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Drop-in replacement for AirLLMBaseModel.forward() that loads
        layers in chunks. OOM-safe: halves chunk size on failure.
        """
        from airllm.utils import clean_memory

        try:
            from transformers.cache_utils import Cache, DynamicCache
            use_cache = False
        except ImportError:
            pass

        base = self._base

        # AirLLM REQUIRES reinit each forward — layers are on 'meta' after use.
        # This is inherent to its layer-streaming design.
        del base.model
        clean_memory()
        base.init_model()
        
        # Token progress counter
        if not hasattr(self, '_token_count'):
            self._token_count = 0
        self._token_count += 1
        if self._token_count % 5 == 1 or self._token_count <= 3:
            console.print(f"  [cyan]generating token {self._token_count}...[/]")

        batch = [ids.to(base.running_device).unsqueeze(0) for ids in input_ids]

        # Create attention mask and position ids
        attn_mask = torch.ones(base.max_seq_len, base.max_seq_len)
        attn_mask = attn_mask.triu(diagonal=1)[None, None, ...] == 0
        attn_mask = attn_mask.to(base.running_device)
        pos_ids = torch.arange(base.max_seq_len, dtype=torch.long, device=base.running_device)[None, :]

        all_hidden_states = [] if output_hidden_states else None

        total_layers = len(base.layer_names)
        chunk_size = self._layers_per_chunk

        with torch.inference_mode():
            i = 0  # current layer index
            while i < total_layers:
                chunk_end = min(i + chunk_size, total_layers)
                
                try:
                    # ── LOAD chunk to GPU ────────────────────────────
                    loaded_indices = []
                    for idx in range(i, chunk_end):
                        state_dict = base.load_layer_to_cpu(base.layer_names[idx])
                        base.move_layer_to_device(state_dict)
                        loaded_indices.append(idx)

                    console.print(
                        f"  [dim]chunk [{i}-{chunk_end-1}] "
                        f"({chunk_end - i} layers loaded)[/]"
                    )

                    # ── EXECUTE chunk sequentially ───────────────────
                    for idx in loaded_indices:
                        layer_name = base.layer_names[idx]
                        layer = base.layers[idx]

                        for j, seq in enumerate(batch):
                            if layer_name == base.layer_names_dict['embed']:
                                batch[j] = layer(seq)
                            elif layer_name == base.layer_names_dict['norm']:
                                batch[j] = base.run_norm(layer, seq)
                            elif layer_name == base.layer_names_dict['lm_head']:
                                batch[j] = base.run_lm_head(layer, seq)
                            else:
                                len_seq = base.get_sequence_len(seq)
                                pos_embed_args = base.get_pos_emb_args(0, len_seq)
                                attention_mask_args = base.get_attention_mask_args(attn_mask, 0, len_seq)
                                position_ids_args = base.get_position_ids_args(pos_ids, 0, len_seq)

                                kwargs = {
                                    'use_cache': False,
                                    'attention_mask': attn_mask[:, :, -len_seq:, -len_seq:],
                                    **pos_embed_args,
                                    **attention_mask_args,
                                    **position_ids_args,
                                }
                                new_seq = layer(seq, **kwargs)[0]
                                batch[j] = new_seq

                        if output_hidden_states and layer_name not in (
                            base.layer_names_dict['lm_head'],
                        ):
                            all_hidden_states.append(torch.cat(batch, 0))

                    # ── UNLOAD chunk ─────────────────────────────────
                    for idx in loaded_indices:
                        base.layers[idx].to("meta")
                    clean_memory()
                    torch.cuda.empty_cache()  # Flush CUDA allocator fragmentation

                    # Advance past this chunk
                    i = chunk_end

                except torch.cuda.OutOfMemoryError:
                    # ── OOM FALLBACK: halve chunk size and retry ─────
                    console.print(
                        f"  [bold red]OOM at chunk [{i}-{chunk_end-1}]![/] "
                        f"Halving chunk size: {chunk_size} → {max(1, chunk_size // 2)}"
                    )
                    # Unload whatever we managed to load
                    for idx in loaded_indices:
                        try:
                            base.layers[idx].to("meta")
                        except Exception:
                            pass
                    clean_memory()
                    torch.cuda.empty_cache()

                    chunk_size = max(1, chunk_size // 2)
                    # Don't advance i — retry with smaller chunk

        # Post-forward cleanup
        gc.collect()
        torch.cuda.empty_cache()

        logits = torch.cat(batch, 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[:-2] if len(all_hidden_states) > 2 else all_hidden_states

        if not return_dict:
            return tuple(
                v for v in [
                    logits,
                    None,
                    tuple(all_hidden_states) if all_hidden_states else None,
                    None,
                ] if v is not None
            )

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
            attentions=None,
        )
