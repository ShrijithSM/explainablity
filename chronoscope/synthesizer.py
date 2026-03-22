"""
The Synthesizer: Causal Report Generator.

Produces a structured Markdown report with embedded visualizations
from the Observer and Analyzer outputs.
"""

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from datetime import datetime
from typing import Dict, Optional, Any, List
from rich.console import Console

matplotlib.use("Agg")
from .config import ChronoscopeConfig

console = Console()

class ReportSynthesizer:
    def __init__(self, config: ChronoscopeConfig):
        self.config = config

    def generate_report(
        self,
        prompt: str,
        generated_text: str,
        observer_results: Dict,
        patching_results: Dict,
        dtw_results: Dict,
        tda_results: Dict,
        validity_scores: Dict,
        hypergraph_results: Optional[Dict] = None,
        head_interaction_results: Optional[Dict] = None,
        experiment_name: str = "experiment",
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_subdir = os.path.join(
            self.config.report_dir, f"{experiment_name}_{timestamp}"
        )
        os.makedirs(report_subdir, exist_ok=True)
        plot_dir = os.path.join(report_subdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plot_paths = self._export_plots(
            observer_results,
            patching_results,
            dtw_results,
            tda_results,
            head_interaction_results,
            plot_dir,
        )

        md_content = self._build_markdown(
            prompt,
            generated_text,
            observer_results,
            validity_scores,
            plot_paths,
            hypergraph_results,
            head_interaction_results,
        )

        report_path = os.path.join(report_subdir, "causal_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return report_path

    def _export_plots(self, observer, patching, dtw, tda, head_interactions, plot_dir):
        paths = {}
        feature_decomp = None
        if head_interactions:
            feature_decomp = head_interactions.get("feature_decomposition")

        # Signal Decomp
        if feature_decomp:
            self._plot_feature_decomposition(
                feature_decomp,
                os.path.join(plot_dir, "decomposition.png"),
            )
            paths["decomposition"] = "plots/decomposition.png"
        elif "decomposition" in observer:
            self._plot_decomposition(observer["decomposition"], os.path.join(plot_dir, "decomposition.png"))
            paths["decomposition"] = "plots/decomposition.png"

        # Spectral
        if "spectral" in observer:
            self._plot_spectral(observer["spectral"], os.path.join(plot_dir, "spectral.png"))
            paths["spectral"] = "plots/spectral.png"

        # TDA Persistence (Ripser returns 'diagrams')
        if tda and "diagrams" in tda:
            # Flatten diagrams for simpler plotting in synthesizer
            flat_persistence = []
            for dim, dgm in enumerate(tda["diagrams"]):
                for point in dgm:
                    if np.isfinite(point[1]):
                        flat_persistence.append([dim, point[0], point[1]])
            
            if flat_persistence:
                self._plot_persistence(flat_persistence, os.path.join(plot_dir, "persistence.png"))
                paths["persistence"] = "plots/persistence.png"

        # Interactive 3D Trajectory (PC0, PC1, PC2)
        if (layer_traj := observer.get("compressed_trajectory")) is not None:
            if layer_traj.ndim == 2 and layer_traj.shape[1] >= 3:
                html_path = os.path.join(os.path.dirname(plot_dir), "trajectory_3d.html")
                token_labels = patching.get("token_labels") if patching else None
                self._export_interactive_3d(layer_traj, html_path, token_labels=token_labels)
                paths["trajectory_3d"] = "trajectory_3d.html"

        # Head Interactions
        if head_interactions:
            layer = head_interactions.get("layer_name", "last")
            if "fdr_masked_influence" in head_interactions:
                self._plot_head_influence(
                    head_interactions["fdr_masked_influence"],
                    f"{layer} (FDR Masked α={getattr(self.config, 'fdr_alpha', 0.05)})",
                    os.path.join(plot_dir, "head_influence.png"),
                )
                paths["head_influence"] = "plots/head_influence.png"
            elif "influence_matrix" in head_interactions:
                self._plot_head_influence(
                    head_interactions["influence_matrix"],
                    layer,
                    os.path.join(plot_dir, "head_influence.png"),
                )
                paths["head_influence"] = "plots/head_influence.png"
            
            if "pdc" in head_interactions and "pdc" in head_interactions["pdc"]:
                pdc_avg = head_interactions["pdc"]["pdc"].mean(axis=0)
                self._plot_head_influence(
                    pdc_avg,
                    f"{layer} (Mean Partial Directed Coherence)",
                    os.path.join(plot_dir, "pdc_heatmap.png"),
                )
                paths["pdc_heatmap"] = "plots/pdc_heatmap.png"

            if "series" in head_interactions:
                self._plot_head_nonstationary_diagnostics(
                    head_interactions["series"],
                    os.path.join(plot_dir, "head_nonstationary.png"),
                )
                paths["head_nonstationary"] = "plots/head_nonstationary.png"

        # Trajectory Dynamics
        if dynamics := observer.get("dynamics"):
            self._plot_dynamics(dynamics, os.path.join(plot_dir, "dynamics.png"))
            paths["dynamics"] = "plots/dynamics.png"

        return paths

    def _plot_head_influence(self, matrix, layer, path):
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Influence Strength (VAR |coefficients|)")
        plt.title(f"Attention Head Interactions — {layer}\n(Directed Causal Influence)")
        plt.xlabel("Target Head")
        plt.ylabel("Source Head")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _plot_head_nonstationary_diagnostics(self, series, path):
        """Plot non-stationary diagnostics from engineered head features."""
        arr = np.asarray(series, dtype=float)
        if arr.ndim == 3:
            t, h, f = arr.shape
            arr = arr.reshape(t, h * f)
        elif arr.ndim != 2:
            return

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        T = arr.shape[0]
        if T < 3:
            return

        # Per-token aggregate level and first-difference energy.
        level = np.mean(np.abs(arr), axis=1)
        delta = np.diff(arr, axis=0)
        delta_energy = np.mean(np.abs(delta), axis=1)

        win = max(3, min(11, T // 6))
        kernel = np.ones(win) / win
        rolling = np.convolve(delta_energy, kernel, mode="same")

        z = (delta_energy - delta_energy.mean()) / (delta_energy.std() + 1e-9)
        regime_idx = np.where(z > 2.0)[0]

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

        axes[0].plot(level, color="tab:blue")
        axes[0].set_title("Attention Feature Level (|x_t| mean over heads×features)")
        axes[0].set_ylabel("Level")
        axes[0].grid(alpha=0.25)

        axes[1].plot(delta_energy, color="tab:orange", alpha=0.8, label="|Δx_t| mean")
        axes[1].plot(rolling, color="tab:red", linewidth=2.0, label=f"Rolling mean (w={win})")
        axes[1].set_title("Non-Stationary Change Energy")
        axes[1].set_ylabel("Change")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

        axes[2].plot(z, color="tab:green", label="Change z-score")
        axes[2].axhline(2.0, color="tab:red", linestyle="--", linewidth=1.2, label="Regime threshold")
        if regime_idx.size > 0:
            axes[2].scatter(regime_idx, z[regime_idx], color="black", s=16, zorder=5)
        axes[2].set_title("Regime Shift Detector")
        axes[2].set_xlabel("Token Position")
        axes[2].set_ylabel("z")
        axes[2].legend()
        axes[2].grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _plot_decomposition(self, decomp, path):
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(decomp["original"], color="black")
        axes[0].set_title("Original PC0 Signal")
        axes[1].plot(decomp["trend"], color="blue")
        axes[1].set_title("Trend Component")
        axes[2].plot(decomp["seasonal"], color="green")
        axes[2].set_title("Seasonal Component")
        axes[3].plot(decomp["residual"], color="red")
        axes[3].set_title("Residual Noise")
        plt.tight_layout()
        plt.savefig(path)
        plt.savefig(path)
        plt.close()

    def _plot_feature_decomposition(self, decomp, path):
        """Feature-space decomposition panel for non-stationary attention series."""
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(decomp["original"], color="black")
        axes[0].set_title("Feature-Space Level (|x_t| mean over heads×features)")

        axes[1].plot(decomp["trend"], color="blue")
        axes[1].set_title("Robust Trend (rolling mean on feature level)")

        axes[2].plot(decomp["seasonal"], color="green")
        axes[2].set_title("Seasonal Proxy (phase-average on detrended level)")

        axes[3].plot(decomp["residual"], color="red", alpha=0.75, label="Residual")
        if "rolling_volatility" in decomp:
            axes[3].plot(decomp["rolling_volatility"], color="purple", alpha=0.75, label="Rolling volatility")
        axes[3].set_title("Residual + Volatility (non-stationary remainder)")
        axes[3].legend()

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _plot_dynamics(self, dynamics, path):
        """Plot Reasoning Velocity and Acceleration."""
        t = np.arange(len(dynamics["velocity"]))
        plt.figure(figsize=(10, 6))
        
        plt.plot(t, dynamics["velocity"], label="Velocity (Semantic Shift)", color="blue", alpha=0.8)
        plt.fill_between(t, dynamics["velocity"], color="blue", alpha=0.1)
        
        plt.plot(t, dynamics["acceleration"], label="Acceleration (Cognitive Stress)", color="red", alpha=0.6, linestyle="--")
        
        # Mark max velocity token
        max_idx = dynamics["max_velocity_token"]
        plt.scatter([max_idx], [dynamics["velocity"][max_idx]], color="black", zorder=5)
        plt.annotate(f"Transition Shift", (max_idx, dynamics["velocity"][max_idx]), xytext=(5, 5), textcoords="offset points")

        plt.title("Chronoscope Reasoning Dynamics")
        plt.xlabel("Token Position")
        plt.ylabel("Latent Speed")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _plot_spectral(self, spectral, path):
        plt.figure(figsize=(10, 6))
        plt.semilogy(spectral["aggregate_freqs"], spectral["aggregate_power"], color="magenta")
        plt.title("Aggregate Power Spectrum (All PC Components)")
        plt.xlabel("Frequency (cycles/token)")
        plt.ylabel("Power")
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(path)
        plt.close()

    def _plot_persistence(self, persistence, path):
        plt.figure(figsize=(8, 8))
        if len(persistence) > 0:
            p = np.array(persistence)
            plt.scatter(p[:, 1], p[:, 2], c=p[:, 0], cmap="Set1")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.title("Persistence Diagram (H0 and H1)")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.savefig(path)
        plt.close()

    def _export_interactive_3d(self, trajectory, path, token_labels=None):
        """Generate a standalone HTML file with an interactive 3D plotly graph."""
        try:
            import plotly.graph_objects as go
            
            n_tokens = trajectory.shape[0]
            # PC0, PC1, PC2
            x = trajectory[:, 0]
            y = trajectory[:, 1]
            z = trajectory[:, 2]
            
            # Default labels if none provided
            if token_labels is None:
                token_labels = [f"Token {i}" for i in range(n_tokens)]
                
            fig = go.Figure()
            
            # 1. Global Trajectory (Translucent reference)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                text=token_labels,
                hoverinfo='text',
                marker=dict(
                    size=3,
                    color=np.arange(n_tokens),
                    colorscale='Viridis',
                    opacity=0.3
                ),
                line=dict(
                    color='rgba(100, 100, 255, 0.4)',
                    width=2
                ),
                name='Global Trajectory'
            ))
            
            # 2. Windowed Local Trajectories
            if getattr(self.config, "tda_enable_windowed", False):
                window_size = getattr(self.config, "tda_window_size", 20)
                stride = getattr(self.config, "tda_window_stride", 10)
                
                for start in range(0, max(1, n_tokens - window_size + 1), stride):
                    end = min(start + window_size, n_tokens)
                    if (end - start) < 3: continue
                    
                    window_x = x[start:end]
                    window_y = y[start:end]
                    window_z = z[start:end]
                    window_labels = token_labels[start:end]
                    
                    fig.add_trace(go.Scatter3d(
                        x=window_x, y=window_y, z=window_z,
                        mode='lines+markers',
                        text=window_labels,
                        hoverinfo='text',
                        marker=dict(
                            size=4,
                            color=np.arange(len(window_x)),
                            colorscale='Plasma',
                            opacity=0.9
                        ),
                        line=dict(width=3),
                        name=f"Window {start}-{end-1}",
                        visible='legendonly' if start > 0 else True # Only show first window by default
                    ))
            
            fig.update_layout(
                title="Chronoscope Interactive 3D Reasoning Trajectory (PC0\u2013PC2)",
                scene=dict(
                    xaxis_title='PC0 (Trend/Main)',
                    yaxis_title='PC1 (Secondary)',
                    zaxis_title='PC2 (Tertiary)',
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            fig.write_html(path)
            console.print(f"  [green]Interactive 3D trajectory exported to {os.path.basename(path)}[/]")
        except Exception as e:
            console.print(f"[yellow]Failed to export interactive 3D: {e}[/]")

    def _build_markdown(self, prompt, text, observer, validity, plot_paths, hypergraph, head_interactions):
        # Extract final verdict and score
        final_verdict = validity.get("verdict", "INCONCLUSIVE")
        final_score = validity.get("composite_validity", 0.0)

        lines = [
            "# Chronoscope Causal Validity Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model:** {self.config.model_name}",
            "---",
            f"## Verdict: **{final_verdict}**",
            f"**Composite Validity Score:** {final_score:.4f}",
            "",
            "## 1. Input & Model Output",
            f"**Prompt:**\n```\n{prompt}\n```",
            f"**Generated:**\n```\n{text}\n```",
            "",
            "## 2. Observer Trace",
            f"- Tokens: {observer.get('meta', {}).get('n_tokens')}",
            f"- Seasonal Period: {observer.get('decomposition', {}).get('detected_period')} tokens",
            f"- **Logical Persistence (Hurst): {observer.get('dynamics', {}).get('hurst', 0.5):.3f}**",
            "",
            f"![Dynamics]({plot_paths.get('dynamics', '#')})",
            f"![Decomposition]({plot_paths.get('decomposition', '#')})",
            f"![Spectral]({plot_paths.get('spectral', '#')})",
        "",
        "## 5. Topological Data Analysis",
        f"![Persistence]({plot_paths.get('persistence', '#')})",
        "",
        "For an interactive 3D view of the reasoning trajectory (PC0\u2013PC2), open **[trajectory_3d.html](trajectory_3d.html)** in a browser.",
        "",
        ]

        if hypergraph:
            lines.append("## 6. Structural Layer: Abstract Reasoning Structures")
            hyperedges = hypergraph.get("hyperedges", [])
            if hyperedges:
                lines.append("| ID | Principle | Tokens | Weight |")
                lines.append("|----|-----------|--------|--------|")
                for he in hyperedges:
                    tokens = ", ".join(he['tokens'][:5])
                    lines.append(f"| `{he['hyperedge_id']}` | {he['principle']} | {tokens} | {he['weight']:.4f} |")
                lines.append("")

        if head_interactions:
            lines.append("## 7. Attention Interaction Layer: Directed Head Causality")
            
            if "stationarity" in head_interactions:
                lines.append("### Stationarity Diagnostics (Gap A)")
                stationarity = head_interactions["stationarity"]
                lines.append(f"- **Joint Stationarity Checked:** {head_interactions.get('joint_stationarity') is not None}")
                if "diff_mask" in stationarity:
                    diff_mask = np.asarray(stationarity.get("diff_mask", []), dtype=bool)
                    n_nonstat = int(diff_mask.sum()) if diff_mask.size > 0 else 0
                    lines.append(f"- **Non-stationary channels detected:** {n_nonstat}/{diff_mask.size if diff_mask.size > 0 else 0}")
                if "needs_diff" in stationarity:
                    lines.append(f"- **Selective differencing applied:** {stationarity.get('needs_diff', False)}")
                if "vecm_rank" in head_interactions:
                    lines.append(f"- **Johansen Cointegration Rank:** {head_interactions['vecm_rank']}")
                lines.append("")
                
            lines.append("### Granger Causality & PDC (Gap B)")
            if "fdr_masked_influence" in head_interactions:
                lines.append("Directed influence between attention heads estimated via VAR and filtered with BH-FDR thresholding to remove false discoveries.")
            else:
                lines.append("Directed influence between attention heads estimated via VAR on per-head metrics.")
                
            if "head_influence" in plot_paths:
                lines.append(f"![Head Influence]({plot_paths['head_influence']})")
                
            if "pdc_heatmap" in plot_paths:
                lines.append("*(Frequency-domain Partial Directed Coherence averaged across bands)*")
                lines.append(f"![PDC Heatmap]({plot_paths['pdc_heatmap']})")

            if "head_nonstationary" in plot_paths:
                lines.append("### Non-Stationary Feature Diagnostics")
                lines.append(
                    "Engineered attention features are tracked as a multivariate time series; "
                    "the panel below highlights change energy and regime-shift tokens."
                )
                lines.append(f"![Head Non-Stationary Diagnostics]({plot_paths['head_nonstationary']})")
                
            lines.append("")

        lines.append("## 8. Validity Score Breakdown")
        for m, score in validity.items():
            if m not in ["composite_validity", "verdict"]:
                lines.append(f"- **{m}**: {score:.4f}")
        lines.append(f"### **FINAL SCORE: {final_score:.4f}**")
        return "\n".join(lines)
        
    def append_interpretive_footnote(
        self, 
        model, 
        tokenizer, 
        report_path: str, 
        observer_results: Dict, 
        token_labels: List[str]
    ) -> None:
        """
        Ask the model to interpret its own Chronoscope report, 
        anchoring the explanation on 'Kinetic Spikes' (Velocity peaks).
        """
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_md = f.read()
        except:
            return

        # 1. Identify the Kinetic Anchor (where logic 'shifted' gears)
        dynamics = observer_results.get("dynamics", {})
        max_v_idx = dynamics.get("max_velocity_token", 0)
        anchor_token = "unknown"
        if token_labels and max_v_idx < len(token_labels):
            anchor_token = token_labels[max_v_idx]

        # 2. Build the Interpretive Prompt
        system_prompt = (
            "You are a Mechanistic Interpretability assistant. Review the following Chronoscope report "
            "and provide a short (3-4 bullet) interpretive footnote.\n"
            f"CRITICAL FOCUS: The 'Reasoning Velocity' spiked at token index {max_v_idx} ('{anchor_token}').\n"
            "Explain what this 'kinetic shift' implies about the reasoning manifold's transition at this point.\n\n"
        )
        
        # Truncate and prep prompt
        report_subset = report_md[-6000:] # Focus on the tail/metrics
        full_prompt = f"{system_prompt}REPORT SUBSET:\n{report_subset}\n\nInterpretive Footnote:"

        # 3. Generate Footnote
        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # 4. Append to File
        footnote_header = "\n\n---\n## 9. Kinetic-Anchored Interpretation\n"
        final_text = f"{footnote_header}**Kinetic Anchor:** Token {max_v_idx} (`{anchor_token}`)\n\n{generated.strip()}\n"

        try:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(final_text)
            console.print(f"[green]Kinetic-anchored footnote appended to {os.path.basename(report_path)}[/]")
        except Exception as e:
            console.print(f"[yellow]Failed to append footnote: {e}[/]")
