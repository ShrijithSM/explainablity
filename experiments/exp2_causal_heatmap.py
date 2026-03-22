"""
Experiment 2: Full Causal Patching Heatmap

Phase 2 of the Chronoscope roadmap.
Goal: Produce a dense [Layers × Tokens] causal heatmap showing exactly
which layer-token combinations are causally critical for the model's output.

Pipeline:
  1. Load model + initialize components
  2. Sweep: patch every (layer, token) pair → measure divergence
  3. Produce heatmap + all Phase 1 metrics on the clean trace
  4. Generate enriched causal report
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from chronoscope.config import ChronoscopeConfig
from chronoscope.models import load_model, list_hookable_layers
from chronoscope.interceptor import ChronoscopeInterceptor
from chronoscope.observer import SignalObserver
from chronoscope.analyzer import CausalAnalyzer
from chronoscope.synthesizer import ReportSynthesizer

console = Console()


def run(config: ChronoscopeConfig = None):
    """Execute Experiment 2: Full Causal Patching Heatmap."""

    if config is None:
        config = ChronoscopeConfig()

    console.rule("[bold blue]Chronoscope — Experiment 2: Causal Patching Heatmap[/]")

    # ── Step 1: Load Model ──────────────────────────────────────────────
    console.print("\n[bold]Step 1:[/] Loading model...")
    model, tokenizer = load_model(config)
    layers = list_hookable_layers(model, max_display=15)

    # ── Step 2: Initialize Components ───────────────────────────────────
    console.print("\n[bold]Step 2:[/] Initializing components...")
    interceptor = ChronoscopeInterceptor(model, tokenizer, config)
    observer = SignalObserver(config)
    analyzer = CausalAnalyzer(interceptor, observer, config)
    synthesizer = ReportSynthesizer(config)

    prompt = config.clean_prompt

    if getattr(config, "optimize_sweep", False):
        console.print("  [yellow]Optimization active: Using stochastic salience-based sweep.[/]")
        patching_results = analyzer.stochastic_patching_sweep(prompt, top_k=30)
    else:
        console.print("  [blue]Optimization disabled: Running FULL exhaustive sweep. This will take time.[/]")
        patching_results = analyzer.causal_patching_sweep(prompt)

    heatmap = patching_results["heatmap"]
    layer_names = patching_results["layer_names"]
    token_labels = patching_results["token_labels"]

    console.print(f"  Heatmap shape: {heatmap.shape}")
    console.print(f"  Max divergence: {heatmap.max():.4f}")
    console.print(f"  Min divergence: {heatmap.min():.4f}")
    console.print(f"  Mean divergence: {heatmap.mean():.4f}")

    # Find top causal (layer, token) pairs
    flat_idx = heatmap.flatten().argsort()[::-1][:5]
    console.print("\n  [bold]Top 5 causal (layer, token) pairs:[/]")
    for rank, idx in enumerate(flat_idx):
        li = idx // heatmap.shape[1]
        ti = idx % heatmap.shape[1]
        lbl = token_labels[ti] if ti < len(token_labels) else f"t{ti}"
        console.print(
            f"    {rank+1}. Layer {layer_names[li]} @ token {lbl}: "
            f"divergence={heatmap[li, ti]:.4f}"
        )

    # ── Step 4: Classical Time Series Analysis on Clean Trace ───────────
    console.print(f"\n[bold]Step 4:[/] Running time series analysis on clean trace...")

    # Get clean trajectory from the sweep's internal capture
    from chronoscope.models import get_deepest_layer
    clean_traj, clean_text = interceptor.capture_generation(prompt)
    target_layer = get_deepest_layer(clean_traj.keys())
    traj_tensor = clean_traj[target_layer]
    observer_results = observer.full_analysis(traj_tensor)

    meta = observer_results["meta"]
    console.print(f"  Tokens: {meta['n_tokens']}, SVD components: {meta['n_svd_components']}")

    stat = observer_results["stationarity"]
    if stat.get("p_value") is not None:
        label = "Stationary" if stat["is_stationary"] else "Non-Stationary"
        console.print(f"  ADF: p={stat['p_value']:.6f} → {label}")

    decomp = observer_results["decomposition"]
    console.print(f"  Seasonal period: {decomp['detected_period']} tokens")

    # ── Step 5: DTW on Best Patching Pair ───────────────────────────────
    console.print(f"\n[bold]Step 5:[/] DTW divergence for most causal patch...")

    # Patch the single most causal (layer, token) pair
    best_li = flat_idx[0] // heatmap.shape[1]
    best_ti = flat_idx[0] % heatmap.shape[1]
    best_layer = layer_names[best_li]
    best_token = best_ti

    console.print(f"  Patching layer={best_layer}, token={best_token}")

    patched_traj, patched_text = interceptor.patch(
        prompt,
        target_layer_name=best_layer,
        token_indices=[best_token],
    )

    clean_compressed = observer_results["compressed_trajectory"]
    if target_layer in patched_traj:
        patched_compressed, _, _ = observer.svd_compress(patched_traj[target_layer])
        dtw_results = analyzer.dtw_divergence(clean_compressed, patched_compressed)
        console.print(f"  DTW distance: {dtw_results['dtw_distance']:.4f}")
        console.print(f"  Normalized: {dtw_results['dtw_normalized']:.4f}")
    else:
        dtw_results = {"dtw_distance": 0.0, "dtw_normalized": 0.0, "path_length": 0}

    # ── Step 6: TDA ─────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 6:[/] Topological analysis...")
    tda_results = analyzer.topological_analysis(clean_compressed)
    betti = tda_results.get("betti_numbers", {})
    console.print(f"  Betti numbers: {betti}")

    # ── Step 7: Validity Score ──────────────────────────────────────────
    console.print(f"\n[bold]Step 7:[/] Validity score...")
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

    # ── Step 8: Generate Report ─────────────────────────────────────────
    console.print(f"\n[bold]Step 8:[/] Generating causal report with heatmap...")

    report_path = synthesizer.generate_report(
        prompt=prompt,
        generated_text=clean_text,
        observer_results=observer_results,
        patching_results=patching_results,
        dtw_results=dtw_results,
        tda_results=tda_results,
        validity_scores=validity,
        experiment_name="exp2_causal_heatmap",
    )

    console.rule("[bold green]Experiment 2 Complete[/]")
    console.print(f"Report: {report_path}")

    # ── Step 9: Interpretive Footnote ──────────────────────────────────
    synthesizer.append_interpretive_footnote(
        model, tokenizer, report_path, observer_results, token_labels
    )

    interceptor.cleanup()
    return report_path


if __name__ == "__main__":
    run()
