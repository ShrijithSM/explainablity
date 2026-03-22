"""
Experiment 3: Chain-of-Thought Topological Mapping

Phase 3 of the Chronoscope roadmap.
Goal: Force the model into a multi-step algorithmic state tracking task using
Zero-Shot CoT ("Let's think step by step"). We want to observe if generating
the intermediate reasoning steps forces the residual stream manifold to connect
into a smooth topological shape (Beta_0 approaches 1) and become non-stationary.
"""

import sys
import os
import gc
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from chronoscope.config import ChronoscopeConfig
from chronoscope.models import load_model, list_hookable_layers
from chronoscope.interceptor import ChronoscopeInterceptor
from chronoscope.observer import SignalObserver
from chronoscope.analyzer import CausalAnalyzer
from chronoscope.synthesizer import ReportSynthesizer

console = Console()


def _build_prompt(config: ChronoscopeConfig, prompt_override: str | None = None) -> str:
    """Build a run prompt without injecting CoT directives."""
    base_prompt = (prompt_override or config.clean_prompt or "").strip()
    if not base_prompt:
        raise ValueError("Prompt is empty. Provide --prompt or set config.clean_prompt.")
    return base_prompt


def run(config: ChronoscopeConfig = None, prompt: str | None = None):
    """Execute Experiment 3: Chain-of-Thought Topology."""

    if config is None:
        config = ChronoscopeConfig()

    console.rule("[bold blue]Chronoscope — Experiment 3: Chain-of-Thought Topology[/]")

    # ── Step 1: Generalized Prompt ──────────────────────────────────────
    default_prompt = (
        "Box A has 5 apples. Box B has 2 apples. "
        "I move 2 apples from Box A to Box B. "
        "Then I move 1 apple from Box B to Box A. "
        "How many apples are in Box A?"
    )
    if not getattr(config, "clean_prompt", None):
        config.clean_prompt = default_prompt

    prompt = _build_prompt(config, prompt_override=prompt)
    config.clean_prompt = prompt
    config.max_new_tokens = 60  # Give it enough room to explain its steps

    # ── Step 2: Load Model ──────────────────────────────────────────────
    console.print("\n[bold]Step 2:[/] Loading model...")
    model, tokenizer = load_model(config)
    layers = list_hookable_layers(model, max_display=60)

    # ── Step 3: Initialize Components ───────────────────────────────────
    console.print("\n[bold]Step 3:[/] Initializing components...")
    interceptor = ChronoscopeInterceptor(model, tokenizer, config)
    observer = SignalObserver(config)
    analyzer = CausalAnalyzer(interceptor, observer, config)
    synthesizer = ReportSynthesizer(config)

    # ── Step 4: Capture Full Trajectory ─────────────────────────────────
    console.print(f"\n[bold]Step 4:[/] Capturing generation trajectory...")
    console.print(f"  Prompt: [italic]{prompt}[/]")

    console.print(f"  Generated: [italic]", end="")
    sys.stdout.flush()

    # Pre-compute prompt length for slicing (needed during streaming)
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]
    del inputs  # Free tokenizer output immediately

    clean_text = ""
    clean_traj = {}
    
    # ── MULTI-CORE PROCESSING ───────────────────────────────────────────
    # AirLLM's sequential layer loading now blocks only the background thread.
    # The main thread (Chronoscope core) receives tokens & trajectories instantly.
    for new_token, current_traj in interceptor.capture_generation_stream(prompt):
        # Print tokens instantly
        sys.stdout.write(new_token)
        sys.stdout.flush()
        
        clean_text += new_token
        clean_traj = current_traj
        
        # --- TRUE PARALLEL CPU ANALYSIS ---
        # AirLLM's sequential layer loading now blocks only the background thread.
        # The main thread uses this "free" CPU time for real-time anomaly tracking!
        from chronoscope.models import get_deepest_layer
        target_layer = get_deepest_layer(current_traj.keys()) if current_traj else None
        if target_layer and len(current_traj[target_layer]) > prompt_len:
            gen_only_traj = current_traj[target_layer][prompt_len:]
            
            # Compute topological EC & non-stationarity
            live_stats = observer.incremental_analysis(gen_only_traj)
            
            # Live Semantic Shift Detection via Topological Break
            if live_stats.get('topological_anomaly_detected'):
                console.print(f"\n  [red]>> Topological Anomaly detected at: '{new_token}'[/] ({live_stats['diagnostics']})")
    
    console.print("\n")

    # --- POST-GENERATION CLEANUP ---
    if not clean_traj:
        raise RuntimeError("No trajectory captured from generation stream.")
         
    from chronoscope.models import get_deepest_layer
    target_layer = get_deepest_layer(clean_traj.keys())
    traj_tensor = clean_traj[target_layer]
    
    # ── GC: Free streaming trajectory dict (we extracted the specific layer we need) ──
    del clean_traj
    gc.collect()

    gen_tensor = traj_tensor[prompt_len:]
    if gen_tensor.shape[0] < 5:
        raise RuntimeError(
            f"Generated trajectory too short for analysis: {gen_tensor.shape[0]} tokens (need >= 5)."
        )
    console.print(f"  Analyzing purely the {gen_tensor.shape[0]} generated tokens.")
    analysis_tensor = gen_tensor

    # ── Step 5: Time Series Analysis ────────────────────────────────────
    console.print(f"\n[bold]Step 5:[/] Running sequence analysis on thought trace...")
    observer_results = observer.full_analysis(analysis_tensor)

    # ── GC: Release raw tensors now that observer has processed them ────
    # analysis_tensor is usually a slice of traj_tensor, so deleting both
    del gen_tensor, analysis_tensor, traj_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta = observer_results["meta"]
    console.print(f"  Tokens: {meta['n_tokens']}, SVD components: {meta['n_svd_components']}")

    stat = observer_results["stationarity"]
    if stat.get("p_value") is not None:
        label = "Stationary" if stat["is_stationary"] else "Non-Stationary"
        console.print(f"  ADF: p={stat['p_value']:.6f} → {label} (Expected Non-Stationary)")

    # ── Step 6: Topological Analysis ────────────────────────────────────
    console.print(f"\n[bold]Step 6:[/] Extracting Topological Shape (Persistent Homology)...")
    clean_compressed = observer_results["compressed_trajectory"]
    tda_results = analyzer.topological_analysis(clean_compressed)
    betti = tda_results.get("betti_numbers", {})
    console.print(f"  Betti numbers: {betti}")

    # ── Step 7: Localized Causal Patching ───────────────────────────────
    # Patch token index 4, which is roughly where "5" (apples) sits in the prompt
    patch_idx = min(4, prompt_len - 1)
    console.print(f"\n[bold]Step 7:[/] Causal patching at token idx {patch_idx} ('5' apples)...")
    
    patched_traj, patched_text = interceptor.patch(
        prompt,
        target_layer_name=target_layer,
        token_indices=[patch_idx],
    )
    console.print(f"  Patched generated: [italic]{patched_text}[/]")

    if target_layer not in patched_traj:
        raise RuntimeError(f"Patched trajectory missing required layer: {target_layer}")

    p_tensor = patched_traj[target_layer]
    p_gen_tensor = p_tensor[prompt_len:] if p_tensor.shape[0] > prompt_len else p_tensor
    patched_compressed, _, _ = observer.svd_compress(p_gen_tensor)
    dtw_results = analyzer.dtw_divergence(clean_compressed, patched_compressed)
    console.print(f"  DTW distance: {dtw_results['dtw_distance']:.4f}")

    # ── Step 8: Validity Score ──────────────────────────────────────────
    console.print(f"\n[bold]Step 8:[/] Computing validity...")
    validity = analyzer.compute_validity_score(
        dtw_result=dtw_results,
        spectral_result=observer_results["spectral"],
        tda_result=tda_results,
        stationarity_result=observer_results["stationarity"],
    )

    for k, v in validity.items():
        if isinstance(v, float):
            console.print(f"  {k}: {v:.4f}")
        else:
            console.print(f"  [bold]{k}: {v}[/]")

    # ── Step 9: Report Generation ───────────────────────────────────────
    console.print(f"\n[bold]Step 9:[/] Generating topological report...")
    
    # Use only real metadata gathered during this run.
    patching_results = {
        "clean_text": clean_text,
        "layer_names": [target_layer],
        "token_labels": interceptor.get_token_labels(prompt, clean_text),
    }

    report_path = synthesizer.generate_report(
        prompt=prompt,
        generated_text=clean_text,
        observer_results=observer_results,
        patching_results=patching_results,
        dtw_results=dtw_results,
        tda_results=tda_results,
        validity_scores=validity,
        experiment_name="exp3_chain_of_thought",
    )

    console.rule("[bold green]Experiment 3 Complete[/]")
    console.print(f"Report: {report_path}")

    interceptor.cleanup()
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3: Chain-of-Thought Topological Mapping"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt to analyze (no CoT prefix is injected).",
    )
    args = parser.parse_args()

    cfg = ChronoscopeConfig()
    run(cfg, prompt=args.prompt)
