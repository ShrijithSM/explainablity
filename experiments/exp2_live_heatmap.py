"""
Experiment 2 (Live): Full Causal Patching Heatmap with Live Dashboard

This script ports the Experiment 4 live dashboard to Experiment 2,
displaying a real-time causal heatmap as the model sweeps through
layer-token pairs.
"""

import sys
import os
import time
import numpy as np
from datetime import datetime
import torch

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.console import Group
from rich.align import Align
from rich.progress import Progress, BarColumn, TextColumn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronoscope.config import ChronoscopeConfig
from chronoscope.models import load_model
from chronoscope.interceptor import ChronoscopeInterceptor
from chronoscope.observer import SignalObserver
from chronoscope.analyzer import CausalAnalyzer
from chronoscope.synthesizer import ReportSynthesizer

def create_heatmap_table(heatmap, layer_names, token_labels):
    """Render a colored rich table representing the causal heatmap."""
    table = Table(title="[bold cyan]Live Causal Heatmap[/]", show_header=True, header_style="bold magenta", border_style="dim")
    
    # Header column for layers
    table.add_column("Layer", style="dim", width=12)
    
    # Add columns for tokens
    for lbl in token_labels:
        table.add_column(lbl, justify="center")
        
    v_max = heatmap.max() if heatmap.max() > 0 else 1.0
    
    for i, layer_name in enumerate(layer_names):
        row = [layer_name]
        for j in range(len(token_labels)):
            val = heatmap[i, j]
            # Simple color mapping: higher divergence = redder
            if val == 0:
                color = "white"
            elif val == -1:
                color = "red" # Error
            else:
                intensity = int((val / v_max) * 4) + 1 # 1-5
                colors = ["blue", "cyan", "green", "yellow", "red"]
                color = colors[min(intensity-1, 4)]
                
            row.append(f"[{color}]{val:.2f}[/]")
        table.add_row(*row)
        
    return table

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, help="Initial prompt for analysis")
    args = parser.parse_args()
    
    config = ChronoscopeConfig()
    config.max_new_tokens = 20 # Keep it short for the sweep
    
    # --- EARLY PROMPT COLLECTION ---
    # We ask for the prompt before initializing the heavy TUI or loading the model.
    # This addresses the user's request to "let the prompt be fed to terminal first".
    prompt = args.prompt
    if not prompt:
        try:
            from rich.console import Console
            _temp_console = Console()
            prompt = _temp_console.input("[bold yellow]Enter initial prompt for Causal Heatmap (or 'QUIT'): [/]")
            if not prompt or prompt.upper() == "QUIT":
                return
        except (KeyboardInterrupt, EOFError):
            return

    # Pre-configure terminal objects
    log_messages = []
    
    # Setup Layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=8)
    )
    layout["main"].split_row(
        Layout(name="heatmap_pane", ratio=2),
        Layout(name="stats_pane", ratio=1)
    )
    
    header = Panel(
        Align.center("[bold cyan]Chronoscope Phase 2 | Live Causal Patching Heatmap[/]", vertical="middle"),
        style="white on blue"
    )
    layout["header"].update(header)
    
    # Initial renders
    layout["heatmap_pane"].update(Panel("Initializing...", title="[yellow]Causal Heatmap[/]", border_style="yellow"))
    layout["stats_pane"].update(Panel("Waiting for data...", title="[magenta]System Stats[/]", border_style="magenta"))
    layout["footer"].update(Panel("Booting model...", title="[green]System Logs[/]", border_style="green"))

    with Live(layout, refresh_per_second=2, screen=False) as live:
        
        def add_log(msg, style="green"):
            ts = datetime.now().strftime("%H:%M:%S")
            log_messages.append(f"[{ts}] {msg}")
            if len(log_messages) > 6:
                log_messages.pop(0)
            log_text = Text.from_markup("\n".join(f"[{style}]{m}[/{style}]" for m in log_messages))
            layout["footer"].update(Panel(log_text, title="[green]System Logs[/]", border_style="green"))

        add_log(f"Loading {config.model_name} on GPU...")
        
        try:
            # Loading model (returns a tuple of (model, tokenizer))
            model, tokenizer = load_model(config)
            add_log("Model loaded successfully.")
            
            interceptor = ChronoscopeInterceptor(model, tokenizer, config)
            observer = SignalObserver(config)
            analyzer = CausalAnalyzer(interceptor, observer, config)
            
            add_log(f"Starting sweep for prompt: '{prompt[:30]}...'")
            
            # --- Incremental Patching Sweep ---
            # We bypass the standard analyzer.causal_patching_sweep to update the UI live
            
            # Get clean trajectory first
            add_log("Capturing clean trajectory...")
            clean_traj, clean_text = interceptor.capture_generation(prompt)
            layer_names = sorted(clean_traj.keys())
            
            # Filter for a subset of layers if too many, to fit on screen
            if len(layer_names) > 8:
                indices = np.linspace(0, len(layer_names)-1, 8, dtype=int)
                layer_names = [layer_names[i] for i in indices]
            
            inputs = tokenizer(prompt, return_tensors="pt")
            n_input_tokens = inputs["input_ids"].shape[1]
            token_range = range(n_input_tokens)
            token_labels = []
            tokens = tokenizer.tokenize(prompt)
            for idx in token_range:
                token_labels.append(tokens[idx].replace(' ', '_'))
                
            n_layers = len(layer_names)
            n_tok = len(token_range)
            heatmap = np.zeros((n_layers, n_tok))
            
            clean_ref_layer = clean_traj[layer_names[-1]]
            
            add_log(f"Sweeping {n_layers} layers x {n_tok} tokens...")
            
            for li, layer_name in enumerate(layer_names):
                for ti, token_idx in enumerate(token_range):
                    add_log(f"Patching {layer_name} @ token '{token_labels[ti]}'", style="cyan")
                    
                    try:
                        patched_traj, _ = interceptor.patch(
                            prompt,
                            target_layer_name=layer_name,
                            token_indices=[token_idx],
                        )
                        
                        if layer_names[-1] in patched_traj:
                            patched_ref = patched_traj[layer_names[-1]]
                            divergence = analyzer._trajectory_divergence(clean_ref_layer, patched_ref)
                            heatmap[li, ti] = divergence
                        
                    except Exception as e:
                        heatmap[li, ti] = -1.0
                        add_log(f"Error patching: {str(e)[:50]}", style="red")
                    
                    # Update View
                    layout["heatmap_pane"].update(Panel(create_heatmap_table(heatmap, layer_names, token_labels), padding=(1, 1)))
                    
                    # Update Stats
                    stats_text = Group(
                        Text("\n-- Current Maximum --", style="bold yellow"),
                        Text(f"Layer: {layer_name}"),
                        Text(f"Token: {token_labels[ti]}"),
                        Text(f"Max Div: {heatmap.max():.4f}"),
                        Text("\n-- Progress --", style="bold blue"),
                        Text(f"Completed: {((li * n_tok + ti + 1) / (n_layers * n_tok) * 100):.1f}%")
                    )
                    layout["stats_pane"].update(Panel(stats_text, title="[magenta]Sweep Stats[/]", border_style="magenta"))

            add_log("Sweep complete. Finalizing analysis...", style="bold green")
            
            # 1. Classical Time Series Analysis on Clean Trace
            add_log("Running TS analysis on clean trace...")
            target_layer = sorted(clean_traj.keys())[-1]
            traj_tensor = clean_traj[target_layer]
            observer_results = observer.full_analysis(traj_tensor)
            clean_compressed = observer_results["compressed_trajectory"]

            # 2. TDA
            add_log("Running topological analysis...")
            tda_results = analyzer.topological_analysis(clean_compressed)

            # 3. DTW on Most Causal Patch
            add_log("Computing DTW for max-divergence patch...")
            flat_idx = heatmap.flatten().argsort()[::-1][0]
            best_li = flat_idx // heatmap.shape[1]
            best_ti = flat_idx % heatmap.shape[1]
            best_layer = layer_names[best_li]
            best_token = token_range[best_ti]
            
            patched_traj, patched_text = interceptor.patch(prompt, best_layer, [best_token])
            if target_layer in patched_traj:
                patched_compressed, _, _ = observer.svd_compress(patched_traj[target_layer])
                dtw_results = analyzer.dtw_divergence(clean_compressed, patched_compressed)
            else:
                dtw_results = {"dtw_distance": 0.0, "dtw_normalized": 0.0, "path_length": 0}

            # 4. Validity Score
            add_log("Computing validity scores...")
            validity = analyzer.compute_validity_score(
                dtw_result=dtw_results,
                spectral_result=observer_results["spectral"],
                tda_result=tda_results,
                stationarity_result=observer_results["stationarity"],
            )

            # 5. Generate Report
            add_log("Generating final report...", style="bold yellow")
            synthesizer = ReportSynthesizer(config)
            patching_results = {
                "heatmap": heatmap,
                "layer_names": layer_names,
                "token_indices": list(token_range),
                "clean_text": clean_text,
                "token_labels": token_labels,
            }
            
            report_path = synthesizer.generate_report(
                prompt=prompt,
                generated_text=clean_text,
                observer_results=observer_results,
                patching_results=patching_results,
                dtw_results=dtw_results,
                tda_results=tda_results,
                validity_scores=validity,
                experiment_name="exp2_live_heatmap",
            )
            
            add_log(f"REPORT SAVED: {os.path.basename(report_path)}", style="bold green")
            time.sleep(10)

        except Exception as e:
            add_log(f"CRITICAL ERROR: {str(e)}", style="bold red")
            return

        # --- Interactive Loop ---
        add_log("EXPERT MODE: Interactive loop active.", style="bold yellow")
        add_log("Waiting for prompt from stdin... (Type 'QUIT' to exit)", style="white")
        
        while True:
            # Check for input without blocking the rich display too much
            # But here we are using standard input, which is blocking for this thread.
            # In a real agentic setup, the agent sends a line to stdin.
            try:
                # Update UI to show we are idle
                layout["stats_pane"].update(Panel(Text("\nIDLE\nReady for next prompt.", justify="center"), title="[magenta]Status[/]", border_style="magenta"))
                
                line = sys.stdin.readline()
                if not line:
                    break
                
                prompt = line.strip()
                if not prompt:
                    continue
                if prompt.upper() == "QUIT":
                    add_log("Exiting...")
                    break
                
                add_log(f"New Prompt: '{prompt[:30]}...'", style="bold cyan")
                
                # Reset heatmap
                heatmap[:] = 0
                
                # --- Capture Clean ---
                add_log("Capturing clean trajectory...")
                clean_traj, clean_text = interceptor.capture_generation(prompt)
                clean_ref_layer = clean_traj[layer_names[-1]]
                
                # --- Sweep ---
                for li, layer_name in enumerate(layer_names):
                    for ti, token_idx in enumerate(token_range):
                        try:
                            patched_traj, _ = interceptor.patch(prompt, layer_name, [token_idx])
                            if layer_names[-1] in patched_traj:
                                patched_ref = patched_traj[layer_names[-1]]
                                divergence = analyzer._trajectory_divergence(clean_ref_layer, patched_ref)
                                heatmap[li, ti] = divergence
                        except Exception:
                            heatmap[li, ti] = -1.0
                        
                        layout["heatmap_pane"].update(Panel(create_heatmap_table(heatmap, layer_names, token_labels), padding=(1, 1)))
                        stats_text = Group(
                            Text("\n-- Current Maximum --", style="bold yellow"),
                            Text(f"Layer: {layer_name}"),
                            Text(f"Token: {token_labels[ti]}"),
                            Text(f"Max Div: {heatmap.max():.4f}"),
                            Text("\n-- Progress --", style="bold blue"),
                            Text(f"Completed: {((li * n_tok + ti + 1) / (n_layers * n_tok) * 100):.1f}%")
                        )
                        layout["stats_pane"].update(Panel(stats_text, title="[magenta]Sweep Stats[/]", border_style="magenta"))

                add_log("Sweep complete.", style="bold green")
                
            except EOFError:
                break
            except Exception as e:
                add_log(f"Loop Error: {str(e)}", style="red")

if __name__ == "__main__":
    main()
