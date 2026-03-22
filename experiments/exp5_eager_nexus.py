"""
Experiment 5: Chronoscope Eager Graph Analysis.
==============================================

This experiment uses an EAGER GRAPH instead of a compiled execution plan.
It integrates structural analysis (Motifs) and abstract principle
mapping (Structural Isomorphism) into the Chronoscope interpretability pipeline.

Nodes:
1. CaptureNode: Extracts hidden states and text.
2. SignalNode: Performs SVD and time-series decomposition.
3. TdaNode: Performs Persistent Homology (Topological analysis).
4. StructuralNode: Extracts structural motifs (Hyperedges).
5. IsomorphismNode: Maps motifs across different layers.
6. ReportNode: Synthesizes the final markdown report.
"""

import asyncio
from rich.console import Console
from chronoscope import (
    ChronoscopeConfig, 
    load_model, 
    ChronoscopeInterceptor, 
    SignalObserver, 
    CausalAnalyzer, 
    ReportSynthesizer
)
from chronoscope.graph import EagerGraph, make_initial_state, InterpretabilityState

console = Console()

# ---------------------------------------------------------------------------
# Graph Nodes (Eager implementation)
# ---------------------------------------------------------------------------

async def capture_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Capture the neural trace from the model."""
    interceptor = config.shared_components["interceptor"]
    
    console.print(f"[bold cyan]Node: Capture[/] - Prompt: [dim]{state['prompt'][:50]}...[/]")
    trajectory, text = interceptor.capture_generation(state["prompt"])
    
    # Get token labels for the full reasoning trace (prompt + generated)
    labels = interceptor.get_token_labels(state["prompt"], text)
    
    return {
        "trajectory": trajectory,
        "generated_text": text,
        "token_labels": labels
    }

async def signal_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Perform signal decomposition and SVD."""
    observer = config.shared_components["observer"]
    
    from chronoscope.models import get_deepest_layer
    layer_name = get_deepest_layer(state["trajectory"].keys())
    raw_tensor = state["trajectory"][layer_name]
    
    console.print(f"[bold cyan]Node: Signal Analysis[/] - Layer: {layer_name}")
    results = observer.full_analysis(raw_tensor)
    
    return {"observer_results": results}

async def tda_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Topological Data Analysis."""
    analyzer = config.shared_components["analyzer"]
    
    compressed = state["observer_results"].get("compressed_trajectory")
    if compressed is None:
        return {}
        
    console.print("[bold cyan]Node: TDA[/] - Persistent Homology")
    results = analyzer.topological_analysis(compressed)
    return {"tda_results": results}

async def structural_motif_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Extract structural motifs (Hyperedges)."""
    analyzer = config.shared_components["analyzer"]
    
    from chronoscope.models import get_deepest_layer
    layer_name = get_deepest_layer(state["trajectory"].keys())
    raw_tensor = state["trajectory"][layer_name]
    
    console.print("[bold cyan]Node: Structural[/] - Extracting motifs")
    hyperedges = analyzer.extract_hyperedges(
        raw_tensor, 
        state["token_labels"], 
        layer_name
    )
    
    return {"hyperedges": hyperedges}

async def isomorphism_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Map motifs to abstract principles."""
    analyzer = config.shared_components["analyzer"]
    
    console.print("[bold cyan]Node: Isomorphic Mapping[/] - Principle clustering")
    clusters = analyzer.detect_isomorphic_clusters(state["hyperedges"])
    
    return {"isomorphic_clusters": clusters}

async def report_node(state: InterpretabilityState, config: ChronoscopeConfig):
    """Generate final report."""
    synthesizer = config.shared_components["synthesizer"]
    
    console.print("[bold cyan]Node: Report Synthesis[/]")
    
    # Format validity scores (mocked for this demo node)
    analyzer = config.shared_components["analyzer"]
    scores = analyzer.compute_validity_score(
        {}, # No DTW in this quick graph
        state["observer_results"].get("spectral", {}),
        state["tda_results"],
        state["observer_results"].get("stationarity", {})
    )
    
    report_path = synthesizer.generate_report(
        prompt=state["prompt"],
        generated_text=state["generated_text"],
        observer_results=state["observer_results"],
        patching_results={"token_labels": state["token_labels"]},
        dtw_results={},
        tda_results=state["tda_results"],
        validity_scores=scores,
        hypergraph_results={
            "hyperedges": state["hyperedges"],
            "isomorphic_clusters": state["isomorphic_clusters"]
        },
        experiment_name="exp5_eager_chronoscope"
    )
    
    return {"report_path": report_path, "validity_scores": scores}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run(config: ChronoscopeConfig):
    """Initialize components and execute the eager graph."""
    
    console.rule("[bold magenta]Experiment 5: Chronoscope Eager Graph[/]")
    
    # 1. Setup components
    model, tokenizer = load_model(config)
    interceptor = ChronoscopeInterceptor(model, tokenizer, config)
    observer = SignalObserver(config)
    analyzer = CausalAnalyzer(interceptor, observer, config)
    synthesizer = ReportSynthesizer(config)
    
    # Store in config for node access
    config.shared_components = {
        "interceptor": interceptor,
        "observer": observer,
        "analyzer": analyzer,
        "synthesizer": synthesizer
    }
    
    # 2. Build Eager Graph
    graph = EagerGraph()
    graph.add_node("capture", capture_node)
    graph.add_node("signal", signal_node)
    graph.add_node("tda", tda_node)
    graph.add_node("structural", structural_motif_node)
    graph.add_node("isomorphism", isomorphism_node)
    graph.add_node("report", report_node)
    
    # Linear execution flow
    graph.add_edge("capture", "signal")
    graph.add_edge("signal", "tda")
    graph.add_edge("tda", "structural")
    graph.add_edge("structural", "isomorphism")
    graph.add_edge("isomorphism", "report")
    
    # 3. Execute
    initial_state = make_initial_state(
        prompt="Explain why topological data analysis is useful for transformer models."
    )
    
    final_state = await graph.ainvoke(initial_state, config=config)
    
    if final_state["error"]:
        console.print(f"[bold red]Graph failed:[/] {final_state['error']}")
    else:
        console.rule("[bold green]Success[/]")
        console.print(f"Report generated at: [yellow]{final_state['report_path']}[/]")
    
    # Cleanup
    interceptor.cleanup()
