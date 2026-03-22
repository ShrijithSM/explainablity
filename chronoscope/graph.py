"""
Chronoscope Eager Graph Engine.
===============================

Implements a step-by-step (eager) workflow runner for interpretability
experiments. This allows complex multi-node pipelines (Capture -> TDA -> 
Structural -> Isomorphism) without the overhead or opacity of a 
compiled execution graph.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union, Callable
from typing_extensions import TypedDict
import torch
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interpretability Research State
# ---------------------------------------------------------------------------

class Hyperedge(TypedDict):
    """A latent motif grouping multiple tokens/neurons under one principle."""
    hyperedge_id: str
    principle: str
    tokens: List[str]
    layer: str
    weight: float

class IsomorphicCluster(TypedDict):
    """Shared abstract patterns across different layers or prompts."""
    cluster_id: str
    shared_principle: str
    hyperedge_ids: List[str]
    similarity: float

class InterpretabilityState(TypedDict):
    """Shared state for the eager interpretability graph."""
    # Inputs
    prompt: str
    config_overrides: Dict[str, Any]
    
    # Trace data
    trajectory: Dict[str, torch.Tensor]  # layer -> tensor
    generated_text: str
    token_labels: List[str]
    
    # Analysis results
    observer_results: Dict[str, Any]
    tda_results: Dict[str, Any]
    dtw_results: Dict[str, Any]
    validity_scores: Dict[str, Any]
    
    # Structural Layer
    hyperedges: List[Hyperedge]
    isomorphic_clusters: List[IsomorphicCluster]
    
    # Attention Interaction Layer
    head_interactions: Dict[str, Any]
    head_intervention_results: Dict[str, Any]
    
    # Outputs
    report_path: Optional[str]
    current_node: str
    error: Optional[str]

# ---------------------------------------------------------------------------
# Eager Graph Runner
# ---------------------------------------------------------------------------

class EagerGraph:
    """
    A lightweight, immediate-execution workflow engine.
    Nodes are async functions that take (state, config) and return a partial state.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {} # current -> next
        self.conditional_edges: Dict[str, Callable] = {}

    def add_node(self, name: str, func: Callable):
        self.nodes[name] = func

    def add_edge(self, start_node: str, end_node: str):
        self.edges[start_node] = end_node

    def add_conditional_edge(self, start_node: str, router_func: Callable):
        self.conditional_edges[start_node] = router_func

    async def ainvoke(self, initial_state: InterpretabilityState, config: Any) -> InterpretabilityState:
        """Execute the graph eagerly until 'END' is reached."""
        state = initial_state
        current_node = state.get("current_node", "START")
        
        if current_node == "START":
            # Default to first added node
            current_node = list(self.nodes.keys())[0]

        logger.info(f"Starting EAGER interpretation graph at '{current_node}'")
        
        while current_node and current_node != "END":
            state["current_node"] = current_node
            logger.info(f"Executing Node: [ {current_node} ]")
            
            node_func = self.nodes.get(current_node)
            if not node_func:
                state["error"] = f"Node {current_node} not found."
                break
                
            try:
                # Execute the node
                import inspect
                if inspect.iscoroutinefunction(node_func):
                    result = await node_func(state, config=config)
                else:
                    result = node_func(state, config=config)
                    
                if result:
                    state.update(result)
                
                # Determine next node
                if current_node in self.conditional_edges:
                    router = self.conditional_edges[current_node]
                    if inspect.iscoroutinefunction(router):
                        current_node = await router(state)
                    else:
                        current_node = router(state)
                else:
                    current_node = self.edges.get(current_node, "END")
                    
            except Exception as e:
                logger.error(f"Error in node {current_node}: {e}", exc_info=True)
                state["error"] = str(e)
                break
                
        logger.info("Eager graph execution finished.")
        return state

def make_initial_state(prompt: str) -> InterpretabilityState:
    """Helper to create a fresh state."""
    return {
        "prompt": prompt,
        "config_overrides": {},
        "trajectory": {},
        "generated_text": "",
        "token_labels": [],
        "observer_results": {},
        "tda_results": {},
        "dtw_results": {},
        "validity_scores": {},
        "hyperedges": [],
        "isomorphic_clusters": [],
        "head_interactions": {},
        "head_intervention_results": {},
        "report_path": None,
        "current_node": "START",
        "error": None
    }
