"""
Experiment 6: Attention Interaction Manifolds.
==============================================

This experiment converts per-token multi-head attention patterns into 
multivariate time series and uses Vector Autoregression (VAR) to characterize
directed interaction patterns between attention heads.

It also performs interventional "Attention Knockouts" (patching) to quantify
the causal influence of high-centrality heads on the global reasoning trajectory.
"""

import asyncio
import os
import sys
import numpy as np

# Ensure local package imports work when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from chronoscope import (
    ChronoscopeConfig, 
    load_model, 
    ChronoscopeInterceptor, 
    SignalObserver, 
    CausalAnalyzer, 
    ReportSynthesizer
)
from chronoscope.dashboard_bridge import DashboardBridge
from chronoscope.graph import EagerGraph, make_initial_state, InterpretabilityState

console = Console()


def _layer_index(name: str) -> int:
    parts = name.split(".")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return -1


def _select_topk_layers(layer_names, k: int):
    if not layer_names:
        return []
    ordered = sorted(layer_names, key=_layer_index)
    k = max(1, min(k, len(ordered)))
    if k == 1:
        return [ordered[-1]]

    idx = np.linspace(0, len(ordered) - 1, num=k, dtype=int)
    return list(dict.fromkeys([ordered[i] for i in idx]))

# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

async def capture_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Capture activations with LIVE streaming analysis and anomaly detection."""
    interceptor = config.shared_components["interceptor"]
    observer = config.shared_components["observer"]
    analyzer = config.shared_components["analyzer"]
    bridge = config.shared_components.get("dashboard_bridge")
    
    prompt = state["prompt"]
    console.print(f"[bold cyan]Node: Capture[/] - Prompt: [dim]{prompt[:50]}...[/]")
    console.print(f"  Generated: [italic]", end="")
    import sys
    sys.stdout.flush()
    
    # STRICT: never substitute synthetic data
    if interceptor is None:
        if bridge:
            bridge.push_log("err", "capture_node: interceptor is None — aborting")
        raise RuntimeError("Interceptor not initialized")
    
    # Pre-compute prompt length for slicing
    from transformers import AutoTokenizer
    try:
        # Try to use tokenizer from interceptor if available
        if hasattr(interceptor, 'tokenizer'):
            tok = interceptor.tokenizer
        else:
            tok = AutoTokenizer.from_pretrained(config.model_name)
    except Exception:
        tok = None
    
    if tok:
        inputs = tok(prompt, return_tensors="pt")
        prompt_len = int(inputs["input_ids"].shape[1])
    else:
        prompt_len = len(prompt.split())  # Fallback to word count
    
    # Store for generated-only analysis in other nodes
    config.prompt_token_count = prompt_len
    if not hasattr(config, 'shared_components'):
        config.shared_components = {}
    config.shared_components["prompt_len"] = prompt_len
    
    clean_text = ""
    clean_traj = {}
    token_idx = 0
    
    # ────── LIVE STREAMING LOOP (from Exp3) ──────────────────────────────
    for new_token, current_traj in interceptor.capture_generation_stream(prompt):
        sys.stdout.write(new_token)
        sys.stdout.flush()
        clean_text += new_token
        clean_traj = current_traj
        
        # Get deepest layer for analysis
        from chronoscope.models import get_deepest_layer
        target_layer = get_deepest_layer(current_traj.keys()) if current_traj else None
        
        if target_layer and len(current_traj[target_layer]) > prompt_len:
            gen_only = current_traj[target_layer][prompt_len:]
            
            # Live TDA (Exp4 config with window_size and distance_threshold)
            live_stats = observer.incremental_analysis(
                gen_only,
                window_size=getattr(config, "tda_window_size", 10),
                distance_threshold=getattr(config, "tda_distance_threshold", 2.0),
            )
            
            # Anomaly push (E3-K2)
            if live_stats.get("topological_anomaly_detected"):
                console.print(f"\n  [red]>> Topological Anomaly at: '{new_token}'[/]")
                if bridge:
                    bridge.push_tda_frame(
                        tda_result={
                            "betti0": live_stats.get("betti0"),
                            "betti1": live_stats.get("betti1"),
                            "euler": live_stats.get("euler_characteristic"),
                            "ec_series": live_stats.get("ec_series", []),
                            "anomalies": [{
                                "token_idx": token_idx,
                                "severity": "high",
                                "description": live_stats.get("diagnostics", "EC spike"),
                            }],
                        },
                        current_token=token_idx,
                    )
                    bridge.push_log(
                        "tda",
                        f"D.2: EC spike t={token_idx} · χ={live_stats.get('euler_characteristic',0)} "
                        f"· var={live_stats.get('rolling_variance',0):.3f}",
                    )
            
            # Entropy row push — LIVE, one per token (E3-K1)
            metric_series = interceptor.get_head_metric_series(target_layer)
            if metric_series is not None and getattr(metric_series, "numel", lambda: 0)() > 0:
                if token_idx < metric_series.shape[0]:
                    row = metric_series[token_idx]
                    if getattr(row, "ndim", 1) > 1:
                        row = row[:, 0]
                    if bridge:
                        bridge.push_token_frame(
                            token_idx=token_idx,
                            interceptor=interceptor,
                            observer=observer,
                            config=config,
                            entropy_row_override=row,
                        )
        
        # Cadenced signal quality updates (E2L-K1)
        STAT_EVERY = getattr(config, "dashboard_stat_every", 5)
        if token_idx % STAT_EVERY == 0 and bridge:
            bridge.push_signal_quality_frame(interceptor, config)
        
        token_idx += 1
    
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    # ────── POST-GENERATION (existing Exp6 logic) ───────────────────────
    if not clean_traj:
        raise RuntimeError("No trajectory captured from generation stream.")
    
    from chronoscope.models import get_deepest_layer
    target_layer = get_deepest_layer(clean_traj.keys())
    traj_tensor = clean_traj[target_layer]
    
    # Generated-only slice (E3-K3)
    gen_tensor = traj_tensor[prompt_len:]
    if gen_tensor.shape[0] < 5:
        raise RuntimeError(
            f"Generated trajectory too short for analysis: {gen_tensor.shape[0]} tokens (need >= 5)."
        )
    
    # Run full post-generation analysis on generated-only segment.
    observer_res = observer.full_analysis(gen_tensor)
    tda_res = analyzer.topological_analysis(observer_res["compressed_trajectory"])
    
    # Get labels
    labels = interceptor.get_token_labels(prompt, clean_text)
    
    if bridge is not None:
        bridge.push_signal_quality_frame(interceptor, config)
    
    return {
        "trajectory": clean_traj,
        "generated_text": clean_text,
        "observer_results": observer_res,
        "tda_results": tda_res,
        "token_labels": labels,
    }

async def causality_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Estimate directed head interaction via VAR."""
    analyzer = config.shared_components["analyzer"]
    bridge = config.shared_components.get("dashboard_bridge")

    top_k_layers = int(getattr(config, "exp6_top_k_layers", 3))
    target_layers = _select_topk_layers(list(state["trajectory"].keys()), top_k_layers)
    console.print(
        f"[bold cyan]Node: Causality Analysis[/] - Top-{len(target_layers)} layers: {target_layers}"
    )

    per_layer = {}
    successful = []
    for layer_name in target_layers:
        res = analyzer.head_interaction_analysis(state["prompt"], layer_name=layer_name)
        per_layer[layer_name] = res

        if "error" in res:
            console.print(f"[yellow]Layer {layer_name} causality skipped:[/] {res['error']}")
            if bridge:
                bridge.push_log("err", f"VAR failed on {layer_name}: {res['error']}")
            continue

        matrix = res.get("masked_influence")
        if matrix is None:
            matrix = res.get("influence_matrix")
        if matrix is None:
            continue

        score = float(np.mean(np.abs(np.asarray(matrix, dtype=float))))
        successful.append((layer_name, score, res))

    if not successful:
        if bridge:
            bridge.push_log("err", "No valid VAR result across selected layers")
        return {
            "head_interactions": {
                "error": "no valid causality result across selected layers",
                "selected_layers": target_layers,
                "per_layer": per_layer,
            }
        }

    total = sum(s for _, s, _ in successful)
    if total <= 1e-12:
        weights = {ln: 1.0 / len(successful) for ln, _, _ in successful}
    else:
        weights = {ln: s / total for ln, s, _ in successful}

    base_shape = None
    for _, _, res in successful:
        m = res.get("masked_influence")
        if m is None:
            m = res.get("influence_matrix")
        if m is not None:
            base_shape = np.asarray(m).shape
            break

    agg_influence = np.zeros(base_shape, dtype=float)
    agg_masked = np.zeros(base_shape, dtype=float)
    for layer_name, _, res in successful:
        w = weights[layer_name]
        inf = res.get("influence_matrix")
        msk = res.get("masked_influence")
        if inf is not None:
            agg_influence += w * np.asarray(inf, dtype=float)
        if msk is not None:
            agg_masked += w * np.asarray(msk, dtype=float)

    best_layer, _, best_res = max(successful, key=lambda x: x[1])
    aggregated = dict(best_res)
    aggregated["layer_name"] = best_layer
    aggregated["selected_layers"] = [ln for ln, _, _ in successful]
    aggregated["layer_weights"] = {k: float(v) for k, v in weights.items()}
    aggregated["per_layer"] = per_layer
    aggregated["influence_matrix"] = agg_influence
    if np.any(agg_masked):
        aggregated["masked_influence"] = agg_masked

    if bridge is not None:
        # Push pre-built dict directly (new push_var_frame signature handles it)
        bridge.push_var_frame(aggregated)
        bridge.push_log("ok", f"VAR/FDR complete on {best_layer} · {len(successful)} layers aggregated")

    return {"head_interactions": aggregated}

async def sweep_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Optional deep-dive causal sweep (Exp2, gated behind config flag)."""
    # Gate: only run if configured
    if not getattr(config, "run_causal_sweep", False):
        return {}  # Skip entirely if not configured

    analyzer = config.shared_components["analyzer"]
    bridge = config.shared_components.get("dashboard_bridge")
    interceptor = config.shared_components.get("interceptor")
    observer = config.shared_components.get("observer")
    
    console.print("[bold yellow]Node: Causal Sweep (OPTIONAL - deep-dive mode)[/]")

    if getattr(config, "optimize_sweep", True):
        sweep_results = analyzer.stochastic_patching_sweep(
            state["prompt"],
            top_k=getattr(config, "causal_sweep_n_pairs", 30)
        )
    else:
        sweep_results = analyzer.causal_patching_sweep(state["prompt"])

    if bridge:
        if "error" not in sweep_results:
            heatmap = sweep_results.get("heatmap")
            max_div = float(heatmap.max()) if heatmap is not None else 0.0
            bridge.push_log("pert", f"Causal sweep complete · max_div={max_div:.4f}")
        else:
            bridge.push_log("err", f"Causal sweep failed: {sweep_results['error']}")

    return {"sweep_results": sweep_results}

async def intervention_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Causal Intervention (Head Knockout)."""
    analyzer = config.shared_components["analyzer"]
    bridge = config.shared_components.get("dashboard_bridge")
    
    interaction = state.get("head_interactions", {})
    if "influence_matrix" not in interaction:
        return {}
        
    # Pick top-k heads with highest OUT-GOING influence
    influence = interaction["influence_matrix"]
    head_scores = influence.sum(axis=1)
    top_k = int(getattr(config, "exp6_top_k_heads", 3))
    top_k = max(1, min(top_k, head_scores.shape[0]))
    source_heads = np.argsort(head_scores)[::-1][:top_k].astype(int).tolist()
    if interaction.get("selected_layers"):
        target_layers = interaction["selected_layers"]
    else:
        all_layers = sorted(state.get("trajectory", {}).keys(), key=_layer_index)
        target_layers = _select_topk_layers(all_layers, int(getattr(config, "exp6_top_k_layers", 3)))
        if not target_layers:
            target_layers = [interaction["layer_name"]]

    console.print(
        f"[bold magenta]Node: Intervention[/] - Knocking out Heads {source_heads} "
        f"across layers {target_layers}"
    )
    causal_impact = analyzer.interventional_head_causality_multilayer(
        state["prompt"],
        target_layers,
        source_heads,
    )
    
    if "error" in causal_impact:
        console.print(f"[bold red]Intervention error: {causal_impact['error']}[/]")
        if bridge:
            bridge.push_log("err", f"Intervention failed: {causal_impact['error']}")
    else:
        # Extract per-head data from causal_impact (A-3 fix)
        per_head = causal_impact.get("per_head_results", [])
        if per_head:
            # Use per-head measurements, not aggregate
            pert_results = [
                {
                    "head": int(r.get("head", h)),
                    "target": int(r.get("target_head", -1)),
                    "mode": "zero",
                    "delta_entropy": float(r.get("delta_entropy", 0.0)),
                    "restoration": float(r.get("restoration", 0.0)),
                    "kl_patch": float(r.get("kl_patch", 0.0)),
                    "confirmed": float(r.get("restoration", 0.0)) > 0.5,
                }
                for i, r in enumerate(per_head)
            ]
        else:
            # Fallback: use aggregate approach if per_head not implemented
            aggregate = float(causal_impact.get("aggregate_mean_influence", 0.0))
            pert_results = [
                {
                    "head": int(h),
                    "target": -1,
                    "mode": "zero",
                    "delta_entropy": aggregate,
                    "restoration": 0.0,
                    "kl_patch": 0.0,
                    "confirmed": aggregate > 0.05,
                }
                for h in source_heads
            ]
        
        if bridge:
            bridge.push_perturbation_frame(pert_results, None)
            bridge.push_log("pert", f"Intervention complete for heads {source_heads} · {len(pert_results)} per-head results")
    
    return {"head_intervention_results": causal_impact}

async def report_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Synthesize results with interaction heatmaps."""
    synthesizer = config.shared_components["synthesizer"]
    bridge = config.shared_components.get("dashboard_bridge")
    
    console.print("[bold cyan]Node: Report Synthesis[/]")

    interaction = state.get("head_interactions", {})
    intervention = state.get("head_intervention_results", {})
    tda = state.get("tda_results", {})

    influence = interaction.get("masked_influence")
    if influence is None:
        influence = interaction.get("influence_matrix")

    if influence is not None:
        influence_arr = np.asarray(influence, dtype=float)
        causal_coherence = float(np.tanh(np.mean(np.abs(influence_arr))))
    else:
        causal_coherence = 0.0

    if "aggregate_mean_influence" in intervention:
        intervention_effect = float(np.tanh(float(intervention["aggregate_mean_influence"])))
    elif "causal_influence_vector" in intervention:
        civ = np.asarray(intervention["causal_influence_vector"], dtype=float)
        intervention_effect = float(np.tanh(np.mean(np.abs(civ))))
    else:
        intervention_effect = 0.0

    text_divergence = float(intervention.get("global_text_divergence", 0.0))
    text_divergence = max(0.0, min(1.0, text_divergence))

    betti = tda.get("betti_numbers", {}) if isinstance(tda, dict) else {}
    betti_1 = float(betti.get("betti_1", 0.0))
    topological_smoothness = 1.0 / (1.0 + betti_1)

    composite_validity = (
        0.45 * causal_coherence
        + 0.30 * intervention_effect
        + 0.15 * topological_smoothness
        + 0.10 * text_divergence
    )

    if composite_validity >= 0.7:
        verdict = "GROUNDED"
    elif composite_validity >= 0.4:
        verdict = "PARTIALLY GROUNDED (review required)"
    else:
        verdict = "INCONCLUSIVE"

    scores = {
        "composite_validity": float(composite_validity),
        "verdict": verdict,
        "causal_coherence": float(causal_coherence),
        "intervention_effect": float(intervention_effect),
        "topological_smoothness": float(topological_smoothness),
        "text_divergence": float(text_divergence),
    }

    if bridge is not None:
        tda_state = {
            "betti0": int(betti.get("betti_0", 0)),
            "betti1": int(betti.get("betti_1", 0)),
            "euler": int(betti.get("betti_0", 0) - betti.get("betti_1", 0)),
            "ec_series": [],
            "anomalies": [],
        }
        bridge.push_tda_frame(
            tda_result=tda_state,
            current_token=max(0, int(getattr(config, "max_new_tokens", 1)) - 1),
            phase_boundaries=getattr(config.shared_components["observer"], "phase_boundaries", []),
            current_phase_idx=getattr(config.shared_components["observer"], "current_phase_idx", None),
        )

        composite_frame = {
            "score": int(round(composite_validity * 100)),
            "dtw_sensitivity": None,
            "spectral_coherence": None,
            "topo_smoothness": float(topological_smoothness),
            "active_reasoning": float(causal_coherence),
            "fdr_sig_pairs": int(interaction.get("n_significant_pairs", 0)),
            "te_score": None,
            "verdict": (
                "STRONG REASONING" if composite_validity >= 0.7 else
                "MODERATE REASONING" if composite_validity >= 0.4 else
                "HALLUCINATION RISK"
            ),
            "n_heads": int(getattr(config, "n_heads", 14)),
        }
        
        # Try to populate dtw_sensitivity and spectral_coherence from official scorer (E1-K1)
        analyzer = config.shared_components.get("analyzer")
        if analyzer and hasattr(analyzer, "compute_validity_score"):
            try:
                obs_results = state.get("observer_results", {})
                official_validity = analyzer.compute_validity_score(
                    dtw_result=state.get("dtw_results", {}),
                    spectral_result=obs_results.get("spectral", {}),
                    tda_result=state.get("tda_results", {}),
                    stationarity_result=obs_results.get("stationarity", {}),
                )
                # Fill None slots with official values
                if official_validity.get("dtw_sensitivity") is not None:
                    composite_frame["dtw_sensitivity"] = official_validity.get("dtw_sensitivity")
                if official_validity.get("spectral_coherence") is not None:
                    composite_frame["spectral_coherence"] = official_validity.get("spectral_coherence")
            except Exception:
                pass  # If official scorer fails, leave as None
        
        interpretation = {
            "source": "Exp6 composite",
            "text": f"Validity={composite_validity:.3f}, intervention={intervention_effect:.3f}, topology={topological_smoothness:.3f}",
        }
        bridge.push_score_frame(composite_frame, interpretation)
        bridge.push_log("ok", "Composite score pushed to dashboard")
    
    report_path = synthesizer.generate_report(
        prompt=state["prompt"],
        generated_text=state["generated_text"],
        observer_results=state.get("observer_results", {}),
        patching_results={"token_labels": state.get("token_labels", [])},
        dtw_results={},
        tda_results=state.get("tda_results", {}),
        validity_scores=scores,
        head_interaction_results=state["head_interactions"],
        experiment_name="exp6_head_interaction"
    )
    
    return {"report_path": report_path}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run(config: ChronoscopeConfig, prompt: str = None):
    console.rule("[bold magenta]Experiment 6: Attention Interaction Manifolds[/]")
    
    # Enable attention capture (set mode from config, default to vector if not set)
    config.capture_attentions = True
    # Don't force mode — respect config setting (default is 'scalar' from config.py)
    # Users can pass --head-feature-mode vector if they want per-head feature vectors
    
    model, tokenizer = load_model(config)
    interceptor = ChronoscopeInterceptor(model, tokenizer, config)
    observer = SignalObserver(config)
    analyzer = CausalAnalyzer(interceptor, observer, config)
    synthesizer = ReportSynthesizer(config)

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
        console.print(f"[bold green]Dashboard:[/] {dashboard_url}")
        bridge.push_log("ok", f"Exp6 starting · model={config.model_name}")
    except Exception as e:
        bridge = None
        console.print(f"[yellow]Dashboard bridge disabled: {e}[/]")
    
    config.shared_components = {
        "interceptor": interceptor,
        "observer": observer,
        "analyzer": analyzer,
        "synthesizer": synthesizer,
        "dashboard_bridge": bridge,
    }
    
    # Eager Graph
    graph = EagerGraph()
    graph.add_node("capture", capture_node)
    graph.add_node("causality", causality_node)
    graph.add_node("sweep", sweep_node)  # Optional node (gated by config)
    graph.add_node("intervention", intervention_node)
    graph.add_node("report", report_node)
    
    graph.add_edge("capture", "causality")
    graph.add_edge("causality", "sweep")
    graph.add_edge("sweep", "intervention")
    graph.add_edge("intervention", "report")
    
    run_prompt = prompt or "Explain the directed acyclic graph in time-series causality."
    initial_state = make_initial_state(prompt=run_prompt)
    
    final_state = await graph.ainvoke(initial_state, config=config)
    
    if final_state.get("error"):
        console.print(f"[bold red]Graph failed:[/] {final_state['error']}")
    else:
        console.rule("[bold green]Success[/]")
        console.print(f"Report: [yellow]{final_state['report_path']}[/]")
    
    interceptor.cleanup()
    if bridge is not None:
        bridge.push_log("ok", "Exp6 complete")
        bridge.stop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 6: Attention Interaction Manifolds"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Custom prompt for attention-head causality analysis.",
    )
    parser.add_argument(
        "--top-k-layers",
        type=int,
        default=3,
        help="Number of layers to analyze/intervene across (depth-spaced).",
    )
    parser.add_argument(
        "--top-k-heads",
        type=int,
        default=3,
        help="Number of top influential heads to ablate during intervention.",
    )
    args = parser.parse_args()

    cfg = ChronoscopeConfig()
    cfg.exp6_top_k_layers = args.top_k_layers
    cfg.exp6_top_k_heads = args.top_k_heads
    asyncio.run(run(cfg, prompt=args.prompt))
