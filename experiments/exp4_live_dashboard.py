"""
Experiment 4: Live Terminal Dashboard

This script demonstrates true parallel processing by streaming tokens and
computing Topological Euler Characteristic on the CPU while the GPU generates.
Renders to a real-time 'rich' terminal UI instead of a static markdown file.
"""

import sys
import os
import time
from datetime import datetime
import torch

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.align import Align

# Add parent directory to path so we can import chronoscope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from chronoscope.config import ChronoscopeConfig
from chronoscope.models import load_model
from chronoscope.interceptor import ChronoscopeInterceptor
from chronoscope.observer import SignalObserver

def generate_sparkline(values, max_points=30):
    """Generate a unicode sparkline for a list of values."""
    if not values:
        return ""
    values = values[-max_points:]
    v_min, v_max = min(values), max(values)
    if v_max == v_min:
        return "▃" * len(values)
    
    # 8 levels of unicode blocks
    bars = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    sparkline = ""
    for v in values:
        idx = int((v - v_min) / (v_max - v_min + 1e-9) * 7)
        sparkline += bars[idx]
    return sparkline

def main():
    config = ChronoscopeConfig()
    config.max_new_tokens = 75
    
    # Pre-configure terminal objects
    text_content = Text()
    chi_history = []
    variance_history = []
    log_messages = []
    
    # Setup Layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=8)
    )
    layout["main"].split_row(
        Layout(name="text_pane", ratio=2),
        Layout(name="telemetry_pane", ratio=1)
    )
    
    header = Panel(
        Align.center("[bold cyan]Chronoscope Live | Parallel GPU Generation & CPU Topological Analysis[/]", vertical="middle"),
        style="white on blue"
    )
    layout["header"].update(header)
    
    # Initial renders
    layout["text_pane"].update(Panel(text_content, title="[yellow]Streaming Generation[/]", border_style="yellow"))
    layout["telemetry_pane"].update(Panel("Waiting for data...", title="[magenta]Live Topology[/]", border_style="magenta"))
    layout["footer"].update(Panel("Booting model...", title="[green]System Logs[/]", border_style="green"))

    # Use lower refresh rate and disable alternate screen buffer to stop flickering on Windows
    with Live(layout, refresh_per_second=4, screen=False) as live:
        
        def add_log(msg, style="green"):
            ts = datetime.now().strftime("%H:%M:%S")
            log_messages.append(f"[{ts}] {msg}")
            if len(log_messages) > 6:
                log_messages.pop(0)
            log_text = Text.from_markup("\n".join(f"[{style}]{m}[/{style}]" for m in log_messages))
            layout["footer"].update(Panel(log_text, title="[green]System Logs[/]", border_style="green"))

        add_log("Loading base model (Qwen2.5-7B) on GPU...")
        
        # Load Model (returns a tuple of (model, tokenizer))
        model, tokenizer = load_model(config)
        add_log("Base model loaded. Initializing Chronoscope...")
        
        interceptor = ChronoscopeInterceptor(model, tokenizer, config)
        observer = SignalObserver(config)
        
        prompt = "There are 5 birds on a tree. A hunter shoots 1. How many birds are left on the tree? Let's think step by step:"
        text_content.append(prompt + " ", style="dim")
        add_log("Started concurrent streaming generation.")
        
        prompt_len: int = len(tokenizer.encode(prompt))
        
        for new_token, current_traj in interceptor.capture_generation_stream(prompt):
            # 1. LIVE CPU ANALYSIS
            topological_anomaly = False
            chi = 0
            var = 0.0
            
            # Use the available layer (accommodating aggressive GC in interceptor.py)
            from chronoscope.models import get_deepest_layer
            target_layer = get_deepest_layer(current_traj.keys())
            if target_layer:
                if len(current_traj[target_layer]) > prompt_len:
                    gen_only_traj = current_traj[target_layer][prompt_len:]
                    
                    # Compute rolling Euclidean topological graphs (Euler Characteristic)
                    live_stats = observer.incremental_analysis(gen_only_traj, window_size=10, distance_threshold=2.0)
                    
                    topological_anomaly = live_stats.get('topological_anomaly_detected', False)
                    chi = live_stats.get('euler_characteristic', 0)
                    var = live_stats.get('rolling_variance', 0.0)
                    
                    chi_history.append(chi)
                    variance_history.append(var)
            
            # 2. UPDATE VIEW STATE
            if topological_anomaly:
                text_content.append(new_token, style="bold red on yellow")
                add_log(f"ALERT: Topological semantic shift at token: '{new_token}' (χ changed violently)", style="bold red")
            else:
                text_content.append(new_token, style="white")
                
            # Update Text Pane
            layout["text_pane"].update(Panel(text_content, title="[yellow]Streaming Generation[/]", border_style="yellow"))
            
            # Update Telemetry Pane
            spark = generate_sparkline(chi_history, max_points=25)
            telemetry_text = Group(
                Text("\n-- Euler Characteristic (χ) --", style="bold magenta"),
                Text(f"Current χ: {chi}"),
                Text(f"Graph: [{spark}]", style="magenta"),
                Text("\n-- Local Non-Stationarity  --", style="bold cyan"),
                Text(f"Rolling Variance: {var:.4f}")
            )
            layout["telemetry_pane"].update(Panel(telemetry_text, title="[magenta]Live Topology[/]", border_style="magenta"))

            # Rich Live handles the screen redraw!
        
        add_log("Generation completed.", style="bold green")
        # Give user time to read the final screen before it exits
        time.sleep(10)

if __name__ == "__main__":
    main()
