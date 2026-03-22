# Chronoscope: Context & Progress

## 🎯 What we are trying to do
The **Chronoscope** project aims to build a **Glass-Box Observability framework for LLM Reasoning**. Rather than treating Large Language Models as black boxes, we aim to interpret their internal cognitive processes by applying classical **Time Series Signal Processing** and **Topological Data Analysis (TDA)** to their hidden states (residual stream trajectories). 

By analyzing the path the model takes through its latent space during generation, we want to objectively quantify reasoning dynamics:
- Track how the model constructs logic (e.g., in Chain-of-Thought prompting).
- Detect structural breaks, hallucinations, or random walks vs. intentional logical persistence.
- Identify latent motifs, hyperedges, and isomorphic clusters across transformer layers.

## 🚀 What we have achieved till now

### 1. Core Architecture
- **Interceptor & Observer:** Established a robust pipeline (`interceptor.py`, `observer.py`, `analyzer.py`) to hook into HuggingFace model layers during generation, capture the residual stream, and perform state-tracking.
- **SVD Compression:** We successfully compress the high-dimensional hidden state trajectories into tractable components for real-time analysis.

### 2. Time Series & Dynamics Processing
We treat the reasoning trace as a time-varying signal and have implemented:
- **Spectral Analysis (FFT):** To identify dominant frequencies and periodic behavior (seasonality) in the reasoning trace.
- **Autocorrelation (ACF/PACF):** To map to the transformer's attention look-back patterns.
- **Stationarity Testing (Augmented Dickey-Fuller):** Non-stationary traces indicate the model is actively changing its belief (actually reasoning) as opposed to being stationary.
- **Trajectory Dynamics:** Metrics for measuring Velocity (speed of semantic transitions), Acceleration (cognitive stress), and the Hurst Exponent (logical persistence vs. random walk/hallucination).
- **Vector Autoregression (VAR):** For attention head interaction analysis to see which heads influence each other over time.

### 3. Topological Data Analysis (TDA) & Structural Mapping
- **Persistent Homology:** Extracting topological shapes (Betti numbers) from the reasoning manifold.
- **Euler Characteristic & Real-Time Tracking:** Using sliding windows to natively calculate variance and Euler characteristics in parallel (CPU) while the GPU generates tokens. Sudden spikes flag "Topological Anomalies" (semantic shifts).
- **Eager Graph Engine (NEXUS):** Implemented an immediate-execution workflow runner (`chronoscope/graph.py`) avoiding heavy compiled graphs. Features hypergraph extraction and isomorphic mapping to detect latent motifs across layers.

### 4. Experiments & Tooling
Developed a suite of distinct testing experiments:
- **Exp1 (Correlational Mapping):** Proves models build time-series concepts (trend, season, ACF) before demonstrating strict causality. Performs DTW divergence and single-layer patching.
- **Exp2 (Causal Heatmap & Live Dashboard):** Built a persistent TUI dashboard (`exp2_live_heatmap.py`) with rich table heatmaps that update in real-time, preventing terminal "spam".
- **Exp3 (Chain-of-Thought Topology):** Analyzes forced multi-step reasoning (`Let's think step by step`) to observe topological manifold connectivity and non-stationary trace behavior.
- **Exp5 (Eager NEXUS):** Demonstrates the full eager graph pipeline with structural mapping.
- **Exp6 (Head Interference):** Tracks localized attention head interference.

### 5. Stability & Reporting
- Optimized loading policies (e.g. gracefully handling 4-bit loading, preferring FP16/BF16 on <1B models to prevent quantization conflicts).
- Built synthesis pipelines (`synthesizer.py`) that bundle all analysis into rich, interpretable artifacts including mathematical markdown reports (`causal_report.md`) and rich 3D dynamic trajectory visualizations (`trajectory_3d.html`).
- Created a **Composite Validity Score** (accounting for DTW sensitivity, spectral coherence, topological smoothness, and active reasoning) to formally output a verdict (e.g., LIKELY HALLUCINATION vs. STRONG REASONING).
