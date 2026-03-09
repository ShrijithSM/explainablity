"""
The Synthesizer: Causal Report Generator.

Produces a structured Markdown report with embedded visualizations
from the Observer and Analyzer outputs.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  - needed to enable 3D plots
from datetime import datetime
from typing import Dict
from rich.console import Console

from .config import ChronoscopeConfig

console = Console()


class ReportSynthesizer:
    """
    Generates a comprehensive mathematical causal validity report
    with embedded plots.
    """

    def __init__(self, config: ChronoscopeConfig):
        self.config = config
        self.report_dir = config.report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_report(
        self,
        prompt: str,
        generated_text: str,
        observer_results: Dict,
        patching_results: Dict,
        dtw_results: Dict,
        tda_results: Dict,
        validity_scores: Dict,
        experiment_name: str = "experiment",
    ) -> str:
        """
        Generate a full Markdown causal report with plots.

        Returns the path to the generated report.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{experiment_name}_{timestamp}"
        report_subdir = os.path.join(self.report_dir, report_name)
        os.makedirs(report_subdir, exist_ok=True)

        # Generate all plots
        plot_paths = {}
        if self.config.save_plots:
            plot_paths = self._generate_plots(
                observer_results, patching_results, tda_results, report_subdir
            )

        # Build Markdown report
        md = self._build_markdown(
            prompt,
            generated_text,
            observer_results,
            patching_results,
            dtw_results,
            tda_results,
            validity_scores,
            plot_paths,
        )

        report_path = os.path.join(report_subdir, "causal_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)

        console.print(f"[bold green]Report saved: {report_path}[/]")
        return report_path

    # ------------------------------------------------------------------ #
    #  Markdown Builder
    # ------------------------------------------------------------------ #

    def _build_markdown(
        self,
        prompt,
        generated_text,
        observer,
        patching,
        dtw,
        tda,
        validity,
        plot_paths,
    ) -> str:
        lines = []

        # Header
        lines.append("# Chronoscope Causal Validity Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Model:** {self.config.model_name}")
        lines.append("")

        # Verdict Banner
        verdict = validity.get("verdict", "UNKNOWN")
        composite = validity.get("composite_validity", 0.0)
        lines.append("---")
        lines.append(f"## Verdict: **{verdict}**")
        lines.append(f"**Composite Validity Score:** {composite:.4f}")
        lines.append("")

        # Prompt & Output
        lines.append("## 1. Input & Model Output")
        lines.append(f"\n**Prompt:**\n```\n{prompt}\n```")
        lines.append(f"\n**Generated:**\n```\n{generated_text}\n```")
        lines.append("")

        # Signal Decomposition
        lines.append("## 2. Time Series Decomposition (Observer)")
        meta = observer.get("meta", {})
        lines.append(f"- **Token count:** {meta.get('n_tokens', '?')}")
        lines.append(f"- **Original hidden dim:** {meta.get('original_dim', '?')}")
        lines.append(f"- **SVD components:** {meta.get('n_svd_components', '?')}")
        lines.append("")

        sv = observer.get("singular_values", np.array([]))
        if len(sv) > 0:
            lines.append("**Singular Values (variance captured):**")
            total = sv.sum()
            for i, s in enumerate(sv):
                pct = (s / total * 100) if total > 0 else 0
                lines.append(f"  - PC{i}: {s:.4f} ({pct:.1f}%)")
            lines.append("")

        # Stationarity
        stat = observer.get("stationarity", {})
        if stat.get("adf_statistic") is not None:
            lines.append("**ADF Stationarity Test:**")
            lines.append(f"  - Statistic: {stat['adf_statistic']:.4f}")
            lines.append(f"  - p-value: {stat['p_value']:.6f}")
            is_stat = stat.get("is_stationary", None)
            lines.append(
                f"  - {'Stationary' if is_stat else 'Non-Stationary'} "
                f"(model {'NOT actively' if is_stat else 'actively'} changing belief)"
            )
            lines.append("")

        # Autocorrelation
        acf_data = observer.get("autocorrelation", {})
        sig_lags = acf_data.get("significant_lags", [])
        if sig_lags:
            lines.append(f"**Significant ACF Lags:** {sig_lags}")
            lines.append(
                "*(These correspond to token distances where the model "
                "re-attends to prior context.)*"
            )
            lines.append("")

        # Decomposition
        decomp = observer.get("decomposition", {})
        if "detected_period" in decomp:
            lines.append(
                f"**Detected Seasonal Period:** {decomp['detected_period']} tokens"
            )
            lines.append("")

        # Plots for Observer
        if "decomposition_plot" in plot_paths:
            lines.append(f"![Signal Decomposition]({plot_paths['decomposition_plot']})")
            lines.append("")
        if "spectral_plot" in plot_paths:
            lines.append(f"![Spectral Analysis]({plot_paths['spectral_plot']})")
            lines.append("")

        # Causal Patching
        lines.append("## 3. Causal Patching Analysis")
        if patching:
            lines.append(f"**Clean output:** `{patching.get('clean_text', '?')}`")
            lines.append("")

        if "heatmap_plot" in plot_paths:
            lines.append(f"![Causal Heatmap]({plot_paths['heatmap_plot']})")
            lines.append("")

        # DTW
        lines.append("## 4. Dynamic Time Warping (Trajectory Divergence)")
        if dtw:
            lines.append(f"- **DTW Distance:** {dtw.get('dtw_distance', 0):.4f}")
            lines.append(f"- **Normalized:** {dtw.get('dtw_normalized', 0):.4f}")
            lines.append(f"- **Path Length:** {dtw.get('path_length', 0)}")
            lines.append("")

        # TDA
        lines.append("## 5. Topological Data Analysis (Persistent Homology)")
        betti = tda.get("betti_numbers", {})
        if betti:
            lines.append("**Betti Numbers:**")
            for k, v in betti.items():
                dim_label = k.replace("betti_", "")
                meaning = {
                    "0": "connected components",
                    "1": "loops/holes",
                }.get(dim_label, "")
                lines.append(f"  - β{dim_label} = {v} ({meaning})")
            lines.append("")

        if "persistence_plot" in plot_paths:
            lines.append(f"![Persistence Diagram]({plot_paths['persistence_plot']})")
            lines.append("")

        # Sliding-window TDA plots (if available)
        window_keys = sorted(
            [k for k in plot_paths.keys() if k.startswith("persistence_window_")]
        )
        if window_keys:
            lines.append(
                "Local topology over the trajectory (sliding windows of tokens):"
            )
            lines.append("")
            for key in window_keys:
                lines.append(f"![{key}]({plot_paths[key]})")
                lines.append("")

        # Optional interactive 3D trajectory link
        if "trajectory_3d_html" in plot_paths:
            lines.append(
                "For an interactive 3D view of the reasoning trajectory "
                f"(PC0–PC2), open `{plot_paths['trajectory_3d_html']}` in a browser."
            )
            lines.append("")

        # Sliding-window 3D trajectory plots (if available)
        traj_window_keys = sorted(
            [k for k in plot_paths.keys() if k.startswith("trajectory_3d_window_")]
        )
        if traj_window_keys:
            lines.append("Local 3D trajectories over sliding windows of tokens:")
            lines.append("")
            for key in traj_window_keys:
                lines.append(f"![{key}]({plot_paths[key]})")
                lines.append("")

        # Validity Scores
        lines.append("## 6. Validity Score Breakdown")
        lines.append("| Metric | Score | Weight |")
        lines.append("|--------|-------|--------|")
        weights = {
            "dtw_sensitivity": 0.35,
            "spectral_coherence": 0.20,
            "topological_smoothness": 0.25,
            "active_reasoning": 0.20,
        }
        for metric, weight in weights.items():
            score = validity.get(metric, 0.0)
            lines.append(f"| {metric} | {score:.4f} | {weight} |")
        lines.append(
            f"| **COMPOSITE** | **{validity.get('composite_validity', 0):.4f}** | **1.0** |"
        )
        lines.append("")

        # Footer
        lines.append("---")
        lines.append(
            "*Report generated by Chronoscope v0.1.0 — "
            "Glass-Box Observability Engine for LLM Reasoning Traces*"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Plot Generation
    # ------------------------------------------------------------------ #

    def _generate_plots(self, observer, patching, tda, output_dir) -> Dict[str, str]:
        paths = {}

        # 1. Signal Decomposition Plot
        decomp = observer.get("decomposition", {})
        if "original" in decomp:
            fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            fig.suptitle("Time Series Decomposition of Residual Stream (PC-0)", fontsize=14)

            axes[0].plot(decomp["original"], color="#2196F3", linewidth=1)
            axes[0].set_ylabel("Original")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(decomp["trend"], color="#FF9800", linewidth=1.5)
            axes[1].set_ylabel("Trend (Tₜ)")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(decomp["seasonal"], color="#4CAF50", linewidth=1)
            axes[2].set_ylabel("Seasonal (Sₜ)")
            axes[2].grid(True, alpha=0.3)

            axes[3].plot(decomp["residual"], color="#F44336", linewidth=1, alpha=0.7)
            axes[3].set_ylabel("Residual (Rₜ)")
            axes[3].set_xlabel("Token Position (t)")
            axes[3].grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(output_dir, "decomposition.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            paths["decomposition_plot"] = "decomposition.png"

        # 1b. 3D Trajectory Plot (Topological Geometry in 3D PC Space)
        compressed = observer.get("compressed_trajectory", None)
        if compressed is not None and compressed.shape[1] >= 3:
            try:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                xs = compressed[:, 0]
                ys = compressed[:, 1]
                zs = compressed[:, 2]
                t = np.arange(len(xs))
                p = ax.scatter(xs, ys, zs, c=t, cmap="viridis", s=20, alpha=0.8)
                ax.set_title("Reasoning Trajectory in 3D (PC0–PC2)")
                ax.set_xlabel("PC0")
                ax.set_ylabel("PC1")
                ax.set_zlabel("PC2")
                fig.colorbar(p, ax=ax, label="Token index")
                plt.tight_layout()
                path = os.path.join(output_dir, "trajectory_3d.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                paths["trajectory_3d_plot"] = "trajectory_3d.png"
            except Exception:
                plt.close()

            # Optional interactive Plotly version (HTML)
            try:
                import plotly.graph_objs as go  # type: ignore

                traces = []

                # Global trajectory trace
                xs = compressed[:, 0]
                ys = compressed[:, 1]
                zs = compressed[:, 2]
                t = np.arange(len(xs))

                traces.append(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines+markers",
                        name="Global trajectory",
                        marker=dict(
                            size=3,
                            color=t,
                            colorscale="Viridis",
                            opacity=0.6,
                        ),
                        line=dict(color="rgba(100, 100, 255, 0.6)", width=2),
                    )
                )

                # Windowed trajectories as additional traces (composite view)
                n_tokens = compressed.shape[0]
                window_size = min(
                    getattr(self.config, "tda_window_size", n_tokens), n_tokens
                )
                stride = max(
                    1, getattr(self.config, "tda_window_stride", window_size)
                )

                window_index = 0
                for start in range(0, max(1, n_tokens - window_size + 1), stride):
                    end = min(start + window_size, n_tokens)
                    segment = compressed[start:end]
                    if segment.shape[0] < 3:
                        continue
                    xs_w = segment[:, 0]
                    ys_w = segment[:, 1]
                    zs_w = segment[:, 2]
                    traces.append(
                        go.Scatter3d(
                            x=xs_w,
                            y=ys_w,
                            z=zs_w,
                            mode="lines+markers",
                            name=f"Window {window_index} ({start}-{end})",
                            marker=dict(
                                size=4,
                                color=np.linspace(0, 1, len(xs_w)),
                                colorscale="Plasma",
                                opacity=0.9,
                            ),
                            line=dict(width=3),
                        )
                    )
                    window_index += 1

                layout = go.Layout(
                    title="Interactive 3D Reasoning Trajectory (Global + Windows)",
                    scene=dict(
                        xaxis_title="PC0",
                        yaxis_title="PC1",
                        zaxis_title="PC2",
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    showlegend=True,
                )

                fig_html = go.Figure(data=traces, layout=layout)
                html_path = os.path.join(output_dir, "trajectory_3d.html")
                fig_html.write_html(html_path, include_plotlyjs="cdn")
                paths["trajectory_3d_html"] = "trajectory_3d.html"
            except Exception:
                # Plotly is optional; ignore if unavailable.
                pass

            # 1c. Sliding-window 3D trajectories (aligned with windowed TDA)
            n_tokens = compressed.shape[0]
            window_size = min(
                getattr(self.config, "tda_window_size", n_tokens), n_tokens
            )
            stride = max(1, getattr(self.config, "tda_window_stride", window_size))

            window_index = 0
            for start in range(0, max(1, n_tokens - window_size + 1), stride):
                end = min(start + window_size, n_tokens)
                segment = compressed[start:end]
                if segment.shape[0] < 3:
                    continue
                try:
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection="3d")
                    xs = segment[:, 0]
                    ys = segment[:, 1]
                    zs = segment[:, 2]
                    t_local = np.arange(len(xs))
                    p = ax.scatter(
                        xs, ys, zs, c=t_local, cmap="plasma", s=20, alpha=0.85
                    )
                    ax.set_title(
                        f"3D Trajectory Window {window_index} "
                        f"(tokens {start}–{end})"
                    )
                    ax.set_xlabel("PC0")
                    ax.set_ylabel("PC1")
                    ax.set_zlabel("PC2")
                    fig.colorbar(p, ax=ax, label="Token index (window-local)")
                    plt.tight_layout()
                    filename = f"trajectory_3d_window_{window_index}.png"
                    path = os.path.join(output_dir, filename)
                    plt.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close()
                    paths[f"trajectory_3d_window_{window_index}"] = filename
                    window_index += 1
                except Exception:
                    plt.close()

        # 2. Spectral Analysis Plot
        spectral = observer.get("spectral", {})
        agg_power = spectral.get("aggregate_power", None)
        agg_freqs = spectral.get("aggregate_freqs", None)
        if agg_power is not None and agg_freqs is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(agg_freqs[1:], agg_power[1:], color="#9C27B0", linewidth=1.2)
            ax.set_title("Aggregate Power Spectral Density (FFT)")
            ax.set_xlabel("Frequency (cycles/token)")
            ax.set_ylabel("Power (log scale)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = os.path.join(output_dir, "spectral.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            paths["spectral_plot"] = "spectral.png"

        # 3. Causal Heatmap
        if patching and patching.get("heatmap") is not None:
            heatmap = patching["heatmap"]
            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
            ax.set_title("Causal Patching Heatmap (Divergence)")
            ax.set_ylabel("Layer")
            ax.set_xlabel("Token Position")

            # Token labels
            token_labels = patching.get("token_labels", [])
            if token_labels and len(token_labels) <= 30:
                ax.set_xticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=7)

            # Layer labels
            layer_names = patching.get("layer_names", [])
            short_names = [n.split(".")[-1] if "." in n else n for n in layer_names]
            if len(short_names) <= 30:
                ax.set_yticks(range(len(short_names)))
                ax.set_yticklabels(short_names, fontsize=7)

            plt.colorbar(im, ax=ax, label="L2 Divergence")
            plt.tight_layout()
            path = os.path.join(output_dir, "causal_heatmap.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            paths["heatmap_plot"] = "causal_heatmap.png"

        # 4. Persistence Diagram (global)
        diagrams = tda.get("diagrams", None)
        if diagrams is not None:
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                colors = ["#2196F3", "#FF5722"]
                for dim, dgm in enumerate(diagrams):
                    finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
                    if len(finite) > 0:
                        ax.scatter(
                            finite[:, 0],
                            finite[:, 1],
                            label=f"H{dim}",
                            alpha=0.6,
                            s=30,
                            color=colors[dim % len(colors)],
                        )

                # Diagonal line (birth = death)
                lims = ax.get_xlim()
                ax.plot(lims, lims, "--", color="gray", alpha=0.5)
                ax.set_title("Persistence Diagram")
                ax.set_xlabel("Birth")
                ax.set_ylabel("Death")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                path = os.path.join(output_dir, "persistence.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                paths["persistence_plot"] = "persistence.png"
            except Exception:
                pass

        # 4b. Sliding-window persistence diagrams (local topology)
        windowed = tda.get("windowed", {})
        windows = windowed.get("windows", [])
        for i, win in enumerate(windows):
            diagrams_w = win.get("diagrams", None)
            if diagrams_w is None:
                continue
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                colors = ["#2196F3", "#FF5722"]
                for dim, dgm in enumerate(diagrams_w):
                    finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
                    if len(finite) > 0:
                        ax.scatter(
                            finite[:, 0],
                            finite[:, 1],
                            label=f"H{dim}",
                            alpha=0.6,
                            s=30,
                            color=colors[dim % len(colors)],
                        )

                lims = ax.get_xlim()
                ax.plot(lims, lims, "--", color="gray", alpha=0.5)
                ax.set_title(
                    f"Persistence Diagram (tokens {win.get('start', 0)}–{win.get('end', 0)})"
                )
                ax.set_xlabel("Birth")
                ax.set_ylabel("Death")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                filename = f"persistence_window_{i}.png"
                path = os.path.join(output_dir, filename)
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                paths[f"persistence_window_{i}"] = filename
            except Exception:
                continue

        return paths
