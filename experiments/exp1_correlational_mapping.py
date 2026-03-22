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
from chronoscope.dashboard_bridge import DashboardBridge

console = Console()




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

    # ── Dashboard Bridge (Integration Guide §7) ────────────────────────
    bridge = None
    try:
        bridge = DashboardBridge(
            transport=getattr(config, "dashboard_transport", "websocket"),
            ws_port=int(getattr(config, "dashboard_ws_port", 8765)),
        ).start()
        dashboard_url = bridge.serve_dashboard(
            getattr(config, "dashboard_html_path", "integration_hub/frontend/public/chronoscope_live.html"),
            port=int(getattr(config, "dashboard_http_port", 8766)),
        )
        console.print(f"  [bold green]Dashboard:[/] {dashboard_url}")
        bridge.push_log("ok", f"Exp1 starting · model={config.model_name}")
    except Exception as e:
        bridge = None
        console.print(f"  [yellow]Dashboard bridge disabled: {e}[/]")

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
    from chronoscope.models import get_deepest_layer
    # Sort layers numerically to ensure correct mid-layer selection later
    def get_index(name):
        parts = name.split(".")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return -1
    layer_names = sorted(trajectory.keys(), key=get_index)
    target_layer = get_deepest_layer(layer_names)
    traj_tensor = trajectory[target_layer]
    console.print(
        f"  Primary analysis layer: {target_layer} "
        f"→ shape: {list(traj_tensor.shape)}"
    )

    # ── Step 4: Classical Time Series Analysis ──────────────────────────
    console.print(f"\n[bold]Step 4:[/] Running classical time series analysis...")
    observer_results = observer.full_analysis(traj_tensor)

    # Push per-token entropy frames to dashboard
    if bridge:
        metric_series = interceptor.get_head_metric_series(target_layer)
        n_tokens = traj_tensor.shape[0] if hasattr(traj_tensor, "shape") else 0
        for t in range(n_tokens):
            bridge.push_token_frame(
                token_idx=t,
                interceptor=interceptor,
                observer=observer,
                config=config,
            )
        bridge.push_signal_quality_frame(interceptor, config)
        bridge.push_log("ok", f"Streamed {n_tokens} token frames to dashboard")

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

    # Push TDA frame to dashboard
    if bridge:
        bridge.push_tda_frame(
            tda_result={
                "betti0": int(betti.get("betti_0", 0)),
                "betti1": int(betti.get("betti_1", 0)),
                "euler": int(betti.get("betti_0", 0) - betti.get("betti_1", 0)),
                "ec_series": [],
                "anomalies": [],
            },
            current_token=meta["n_tokens"] - 1,
        )

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
    token_labels = interceptor.get_token_labels(prompt, generated_text)

    patching_results = {
        "heatmap": None,  # Full sweep skipped in Phase 1 for speed
        "clean_text": generated_text,
        "token_labels": token_labels,
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

    synthesizer.append_interpretive_footnote(
        model, tokenizer, report_path, observer_results, token_labels
    )

    # Push composite score to dashboard
    if bridge:
        composite_frame = {
            "score": int(round(validity["composite_validity"] * 100)),
            "dtw_sensitivity": validity.get("dtw_sensitivity"),
            "spectral_coherence": validity.get("spectral_coherence"),
            "topo_smoothness": validity.get("topological_smoothness"),
            "active_reasoning": validity.get("active_reasoning"),
            "fdr_sig_pairs": 0,
            "te_score": None,
            "verdict": (
                "STRONG REASONING" if validity["composite_validity"] >= 0.7 else
                "MODERATE REASONING" if validity["composite_validity"] >= 0.4 else
                "HALLUCINATION RISK"
            ),
            "n_heads": int(getattr(config, "n_heads", 14)),
        }
        interpretation = {
            "source": "Exp1 Correlational",
            "text": f"Validity={validity['composite_validity']:.3f}, Verdict={validity['verdict']}",
        }
        bridge.push_score_frame(composite_frame, interpretation)
        bridge.push_log("ok", f"Exp1 complete · report={os.path.basename(report_path)}")
        bridge.stop()

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
