"""
Model loading utilities for local HuggingFace causal LMs.
Supports Qwen, DeepSeek, LLaMA, Mistral — any AutoModelForCausalLM.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from .config import ChronoscopeConfig

console = Console()


def load_model(config: ChronoscopeConfig):
    """
    Load a HuggingFace causal LM and tokenizer.

    Returns:
        (model, tokenizer) tuple.
    """
    console.print(
        f"[bold cyan]Loading model:[/] {config.model_name} "
        f"on {config.device} ({config.torch_dtype})"
    )

    kwargs = {
        "torch_dtype": config.get_torch_dtype(),
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if config.use_airllm:
        from airllm import AutoModel
        from .adaptive_airllm import AdaptiveAirLLM
        console.print("[yellow]Loading model via AirLLM...[/]")
        
        # AirLLM is extremely slow without compression on 7B+ models
        compression_arg = '4bit' if config.load_in_4bit else None
        
        base_model = AutoModel.from_pretrained(
            config.model_name,
            compression=compression_arg
        )
        
        # Wrap with adaptive layer batching for auto-scaled VRAM usage
        model = AdaptiveAirLLM(
            base_model,
            vram_headroom_gb=config.airllm_vram_headroom_gb
        )
    else:
        # Optional 4-bit quantization for very tight VRAM
        if config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                
                # NF4/fp4 loading with double quantization for maximum memory savings
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=config.get_torch_dtype(),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    # Force modules onto the specified device to prevent 'accelerate' from moving them later
                    llm_int8_enable_fp32_cpu_offload=False,
                )
                console.print("[yellow]4-bit NF4 quantization enabled (with double-quant).[/]")
            except ImportError:
                console.print(
                    "[red]bitsandbytes not installed. Loading without quantization.[/]"
                )

        model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
        
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )

    # Ensure pad token exists (many models lack one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not config.use_airllm:
        model.eval()
        console.print(
            f"[bold green]Model loaded.[/] "
            f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
    else:
        console.print(f"[bold green]Model loaded via AirLLM.[/]")

    return model, tokenizer


def list_hookable_layers(model, max_display: int = 60):
    """
    Walk model.named_modules() and display all hookable layers.
    Use this to identify the correct target_layers for your model architecture.
    """
    table = Table(title="Hookable Layers", show_lines=False)
    table.add_column("Index", style="dim", width=5)
    table.add_column("Layer Name", style="cyan")
    table.add_column("Type", style="green")

    layers = []
    # Handle AirLLM wrappers which keep the actual module in 'model'
    import torch.nn as nn
    target_module = model.model if hasattr(model, 'model') and isinstance(model.model, nn.Module) else model
    
    for i, (name, module) in enumerate(target_module.named_modules()):
        if name:  # Skip root
            layers.append((name, type(module).__name__))

    for i, (name, type_name) in enumerate(layers[:max_display]):
        table.add_row(str(i), name, type_name)

    if len(layers) > max_display:
        table.add_row("...", f"({len(layers) - max_display} more)", "...")

    console.print(table)
    console.print(f"\n[bold]Total hookable modules:[/] {len(layers)}")

    return layers
