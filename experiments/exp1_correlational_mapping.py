"""
Experiment 1: Correlational Mapping

Phase 1 of the Chronoscope roadmap.
Goal: Prove that the Transformer learns time series concepts (trend, season,
autocorrelation) inside its reasoning trace, even before proving causality.

Pipeline:
  1. Load a local HuggingFace model (Qwen2.5-0.5B)
  2. Run a reasoning prompt
  3. Capture residual stream trajectory across all decoder layers
  4. SVD compress → FFT → ACF → ADF → STL Decomposition
  5. Run a single-layer causal patching experiment
  6. Compute DTW divergence + TDA
  7. Generate a full mathematical causal report
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rich.console import Console
from chronoscope.config import ChronoscopeConfig
from chronoscope.models import load_model, list_hookable_layers
from chronoscope.interceptor import ChronoscopeInterceptor
from chronoscope.observer import SignalObserver
from chronoscope.analyzer import CausalAnalyzer
from chronoscope.synthesizer import ReportSynthesizer

console = Console()


def _append_interpretive_footnote(
    model, tokenizer, report_path: str, config: ChronoscopeConfig
) -> None:
    """
    After the main report is written, run a final query to the SAME LLM,
    asking it to interpret the report and append a short human-readable
    footnote.
    """
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_md = f.read()
    except Exception as e:
        console.print(f"[yellow]Could not read report for interpretation: {e}[/]")
        return

    # Truncate very long reports to avoid context blowup
    max_chars = 8000
    if len(report_md) > max_chars:
        report_md = report_md[-max_chars:]

    system_prompt = (
        "You are an interpretability assistant. Read the Chronoscope causal "
        "validity report below and write a short human-readable footnote that "
        "summarizes what the metrics and plots imply about the model's reasoning. "
        "Focus on: (1) how causally grounded the reasoning is, (2) whether the "
        "trajectory looks smooth vs noisy, and (3) any caveats. "
        "Keep it to 3–5 concise bullet points.\n\n"
    )

    full_prompt = system_prompt + "REPORT START:\n" + report_md + "\n\nFootnote:"

    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    footnote_header = "\n\n## 7. Interpretive Footnote\n"
    footnote_text = footnote_header + generated.strip() + "\n"

    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(footnote_text)
        console.print("[green]Appended interpretive footnote to report.[/]")
    except Exception as e:
        console.print(f"[yellow]Failed to append footnote: {e}[/]")


def run(config: ChronoscopeConfig = None):
    """Execute Experiment 1: Correlational Mapping."""

    if config is None:
        config = ChronoscopeConfig()

    console.rule("[bold blue]Chronoscope — Experiment 1: Correlational Mapping[/]")

    # ── Step 1: Load Model ──────────────────────────────────────────────
    console.print("\n[bold]Step 1:[/] Loading model...")
    model, tokenizer = load_model(config)

    # Show available layers for hooking
    console.print("\n[bold]Hookable layers:[/]")
    layers = list_hookable_layers(model, max_display=30)

    # ── Step 2: Initialize Components ───────────────────────────────────
    console.print("\n[bold]Step 2:[/] Initializing Chronoscope components...")
    interceptor = ChronoscopeInterceptor(model, tokenizer, config)
    observer = SignalObserver(config)
    analyzer = CausalAnalyzer(interceptor, observer, config)
    synthesizer = ReportSynthesizer(config)

    # ── Step 3: Capture Clean Trajectory ────────────────────────────────
    prompt = config.clean_prompt
    console.print(f"\n[bold]Step 3:[/] Capturing reasoning trajectory...")
    console.print(f"  Prompt: [italic]{prompt}[/]")

    trajectory, generated_text = interceptor.capture_generation(prompt)

    console.print(f"  Generated: [italic]{generated_text}[/]")
    console.print(f"  Layers captured: {len(trajectory)}")

    if not trajectory:
        console.print("[red]ERROR: No activations captured! Aborting.[/]")
        return

    # Use the LAST decoder layer for primary analysis
    layer_names = sorted(trajectory.keys())
    target_layer = layer_names[-1]
    traj_tensor = trajectory[target_layer]
    console.print(
        f"  Primary analysis layer: {target_layer} "
        f"→ shape: {list(traj_tensor.shape)}"
    )

    # ── Step 4: Classical Time Series Analysis ──────────────────────────
    console.print(f"\n[bold]Step 4:[/] Running classical time series analysis...")
    observer_results = observer.full_analysis(traj_tensor)

    meta = observer_results["meta"]
    console.print(f"  Tokens: {meta['n_tokens']}, SVD components: {meta['n_svd_components']}")

    sv = observer_results["singular_values"]
    console.print(f"  Top singular values: {sv[:3].round(3)}")

    stat = observer_results["stationarity"]
    if stat.get("p_value") is not None:
        label = "Stationary" if stat["is_stationary"] else "Non-Stationary"
        console.print(f"  ADF test: p={stat['p_value']:.6f} → {label}")

    acf_data = observer_results["autocorrelation"]
    sig_lags = acf_data.get("significant_lags", [])
    console.print(f"  Significant ACF lags: {sig_lags}")

    decomp = observer_results["decomposition"]
    console.print(f"  Detected seasonal period: {decomp['detected_period']} tokens")

    # ── Step 4b: Attention Head Interaction Analysis (Heads as Time Series) ──
    if getattr(config, "capture_attentions", False):
        console.print(
            f"\n[bold]Step 4b:[/] Analyzing attention head interactions "
            f"via VAR on per-head time series..."
        )
        head_results = analyzer.head_interaction_analysis(
            prompt, layer_name=target_layer
        )
        if "error" in head_results:
            console.print(f"  [yellow]Head analysis skipped:[/] {head_results['error']}")
        else:
            series = head_results["series"]
            influence = head_results["influence_matrix"]
            H = influence.shape[0]
            console.print(f"  Head series shape: T={series.shape[0]}, H={series.shape[1]}")
            # Report the strongest directed influences between heads
            flat_idx = influence.flatten().argsort()[::-1][: min(5, H * H)]
            console.print("  Top head→head influence pairs (source→target):")
            for rank, idx in enumerate(flat_idx):
                src = idx // H
                tgt = idx % H
                val = influence[src, tgt]
                console.print(
                    f"    {rank+1}. h{src} → h{tgt}: influence={val:.4f}"
                )

    # ── Step 5: Causal Patching (Single Layer) ──────────────────────────
    console.print(f"\n[bold]Step 5:[/] Running causal patching on middle layer...")

    # Patch the middle layer at the first few tokens (premise tokens)
    mid_layer = layer_names[len(layer_names) // 2]
    inputs = tokenizer(prompt, return_tensors="pt")
    n_input = inputs["input_ids"].shape[1]
    premise_tokens = list(range(min(3, n_input)))  # First 3 tokens

    console.print(f"  Patching layer: {mid_layer}")
    console.print(f"  Patching token indices: {premise_tokens}")

    patched_traj, patched_text = interceptor.patch(
        prompt,
        target_layer_name=mid_layer,
        token_indices=premise_tokens,
    )

    console.print(f"  Patched output: [italic]{patched_text}[/]")

    # ── Step 6: DTW Divergence ──────────────────────────────────────────
    console.print(f"\n[bold]Step 6:[/] Computing DTW trajectory divergence...")

    clean_compressed = observer_results["compressed_trajectory"]

    if target_layer in patched_traj:
        patched_compressed, _, _ = observer.svd_compress(patched_traj[target_layer])
        dtw_results = analyzer.dtw_divergence(clean_compressed, patched_compressed)
        console.print(f"  DTW distance: {dtw_results['dtw_distance']:.4f}")
        console.print(f"  Normalized: {dtw_results['dtw_normalized']:.4f}")
    else:
        dtw_results = {"dtw_distance": 0.0, "dtw_normalized": 0.0, "path_length": 0}
        console.print("  [yellow]Warning: Patched trajectory missing target layer.[/]")

    # ── Step 7: Topological Data Analysis ───────────────────────────────
    console.print(f"\n[bold]Step 7:[/] Running persistent homology (TDA)...")
    tda_results = analyzer.topological_analysis(clean_compressed)
    betti = tda_results.get("betti_numbers", {})
    console.print(f"  Betti numbers: {betti}")

    # Optional: sliding-window TDA to surface local topological changes
    if getattr(config, "tda_enable_windowed", False):
        console.print(
            f"\n[bold]Step 7b:[/] Sliding-window TDA over the trajectory..."
        )
        windowed_tda = analyzer.topological_analysis_windowed(clean_compressed)
        windows = windowed_tda.get("windows", [])
        console.print(f"  Windows analyzed: {len(windows)}")
        # Print a brief summary of β1 per window
        for i, w in enumerate(windows[:5]):
            b = w.get("betti_numbers", {})
            console.print(
                f"    Window {i} (tokens {w['start']}–{w['end']}): "
                f"β0={b.get('betti_0', 0)}, β1={b.get('betti_1', 0)}"
            )
        # Attach to tda_results so the report generator can plot them.
        tda_results["windowed"] = windowed_tda

    # ── Step 8: Validity Score ──────────────────────────────────────────
    console.print(f"\n[bold]Step 8:[/] Computing composite validity score...")
    validity = analyzer.compute_validity_score(
        dtw_result=dtw_results,
        spectral_result=observer_results["spectral"],
        tda_result=tda_results,
        stationarity_result=observer_results["stationarity"],
    )

    console.print(f"  DTW Sensitivity: {validity['dtw_sensitivity']:.4f}")
    console.print(f"  Spectral Coherence: {validity['spectral_coherence']:.4f}")
    console.print(f"  Topological Smoothness: {validity['topological_smoothness']:.4f}")
    console.print(f"  Active Reasoning: {validity['active_reasoning']:.4f}")
    console.print(f"  [bold]Composite: {validity['composite_validity']:.4f}[/]")
    console.print(f"  [bold]Verdict: {validity['verdict']}[/]")

    # ── Step 9: Generate Report ─────────────────────────────────────────
    console.print(f"\n[bold]Step 9:[/] Generating causal report...")

    # Build a minimal patching result dict for the report
    patching_results = {
        "heatmap": None,  # Full sweep skipped in Phase 1 for speed
        "clean_text": generated_text,
        "token_labels": [],
        "layer_names": layer_names,
    }

    report_path = synthesizer.generate_report(
        prompt=prompt,
        generated_text=generated_text,
        observer_results=observer_results,
        patching_results=patching_results,
        dtw_results=dtw_results,
        tda_results=tda_results,
        validity_scores=validity,
        experiment_name="exp1_correlational",
    )

    console.rule("[bold green]Experiment 1 Complete[/]")
    console.print(f"Report: {report_path}")

    # ── Step 10: Interpretive Footnote via the same LLM ──────────────────
    console.print(
        "\n[bold]Step 10:[/] Asking the model to add an interpretive footnote..."
    )
    _append_interpretive_footnote(model, tokenizer, report_path, config)

    # Cleanup
    interceptor.cleanup()

    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chronoscope Experiment 1: Correlational Mapping on a prompt."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to analyze instead of the default config.clean_prompt.",
    )
    args = parser.parse_args()

    cfg = ChronoscopeConfig()
    if args.prompt:
        cfg.clean_prompt = args.prompt

    run(cfg)
