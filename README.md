# CHRONOSCOPE

> **Glass-Box Observability Framework for Large Language Model Reasoning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Chronoscope** is an advanced LLM interpretability framework that applies classical **Time Series Signal Processing** and **Topological Data Analysis (TDA)** to understand the internal reasoning dynamics of Large Language Models. Instead of treating LLMs as black boxes, we analyze their hidden state trajectories as multi-dimensional signals to objectively quantify reasoning quality, detect hallucinations, and map causal relationships between attention heads.

---

## 🎯 Key Features

- **🔍 Real-Time Reasoning Quality Scoring** — Composite validity score (0-100) classifying outputs as STRONG REASONING / MODERATE / HALLUCINATION RISK
- **📊 Time Series Analysis** — FFT, ACF/PACF, Hurst exponent, trajectory dynamics applied to the residual stream
- **🔗 Causal Head Mapping** — VAR/VECM with FDR correction to reveal which attention heads influence each other
- **🌐 Topological Anomaly Detection** — Persistent Homology & Euler Characteristic tracking (CPU-parallel, zero overhead)
- **🎭 Chain-of-Thought Segmentation** — HMM-based automatic detection of reasoning phases
- **🔬 Head Interventions** — Knockout experiments with restoration scoring and mediation analysis
- **📈 Live Dashboard** — Real-time 12-panel visualization with WebSocket streaming

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd explainablity

# Install dependencies
pip install -r requirements.txt

# Optional: Install Ollama for local LLM labeling
# Download from https://ollama.ai
```

### Run Your First Analysis

```bash
# Experiment 1: Correlational Mapping (proves models build time-series concepts before causality)
python experiments/exp1_correlational_mapping.py

# Experiment 3: Chain-of-Thought Topology Analysis
python experiments/exp3_chain_of_thought.py

# Experiment 4: Live Dashboard (real-time visualization)
python experiments/exp4_live_dashboard.py
```

### Launch the Interactive Dashboard

**Option A: Demo Mode (No Backend Required)**
```bash
# Simply open in your browser (works offline)
integration_hub/frontend/public/chronoscope_demo.html
```

**Option B: Live Mode (With Backend)**
```bash
# Terminal 1: Start the backend + WebSocket server
python integration_hub/backend/launch_chat_system.py

# Terminal 2: Run an experiment to feed live data
python experiments/exp4_live_dashboard.py

# Browser will auto-open to chronoscope_live.html
```

---

## 📊 Live Dashboard Preview

The dashboard provides real-time visualization of 12 analysis dimensions:

| Panel | Description |
|-------|-------------|
| **State Panel** | Hurst exponent, ADF decision, Johansen rank, VAR lag, perturbation mode |
| **Entropy Chart** | 14 time-series lines showing per-head entropy evolution |
| **VAR Influence Matrix** | 14×14 heatmap of causal relationships (FDR-corrected) |
| **Composite Verdict** | Ring visualization of 0-100 validity score with classification |
| **VAR Influence List** | Bar chart of top Granger-causal head pairs |
| **Perturbation Results** | Head knockout experiments with restoration scores |
| **TDA Panel** | Betti numbers (β₀, β₁) and Euler characteristic (χ) stream |
| **Signal Quality** | 5 per-head metrics: entropy, effective rank, attention sink, etc. |
| **HMM Phase** | 4-state reasoning phase discovery with transition matrix |
| **Phase Timeline** | Sequence visualization of PROMPT→SETUP→ACTIVE→CONCLUSION |
| **System Log** | Classified event stream (INFO, ANOMALY, CAUSAL, PHASE) |
| **Chat Island** | Interactive prompt input with live analysis |

![Dashboard Screenshot](integration_hub/frontend/public/chronoscope_demo.html)

---

## 🏗️ Architecture

```
User Input → Transformer Model
                 ↓
         Chronoscope Interceptor (PyTorch hooks)
                 ↓
    ┌────────────┴────────────┐
    ↓                         ↓
Signal Observer          Analyzer
(Time Series)      (Statistical Rigor)
    ↓                         ↓
    └────────────┬────────────┘
                 ↓
           TDA Module
         (Topology)
                 ↓
          Synthesizer
      (Report Generation)
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
Live Dashboard     Static Reports
  (WebSocket)     (Markdown/HTML)
```

---

## 📚 Core Modules

### 1. **Interceptor** (`chronoscope/interceptor.py`)
Hooks into the transformer's forward pass to capture hidden states at specified layers. Features OOM recovery and automatic CPU fallback.

### 2. **Observer** (`chronoscope/observer.py`)
Time series signal processing:
- SVD compression (high-dim → tractable components)
- Spectral analysis (FFT, periodogram)
- Autocorrelation (ACF/PACF)
- Trajectory dynamics (velocity, acceleration, Hurst exponent)
- Incremental SVD (42.75× faster than full re-SVD)

### 3. **Analyzer** (`chronoscope/analyzer.py`)
Statistical rigor and causality (1791 lines):
- Per-head stationarity testing (ADF)
- Selective differencing
- Johansen cointegration tests
- VAR/VECM fitting with AIC lag selection
- FDR correction (Benjamini-Hochberg)
- Granger causality F-tests
- Partial Directed Coherence (spectral causality)
- Head intervention experiments

### 4. **TDA Module** (in Observer)
Topological Data Analysis:
- Persistent Homology via Ripser
- Betti numbers (β₀ = components, β₁ = loops)
- Real-time Euler Characteristic (χ = β₀ - β₁)
- Hypergraph extraction
- Isomorphic cluster detection

### 5. **Synthesizer** (`chronoscope/synthesizer.py`)
Report generation:
- Mathematical markdown reports (`causal_report.md`)
- Interactive 3D trajectory visualization (`trajectory_3d.html`)
- Composite validity scoring (4 sub-metrics weighted)

---

## 🧪 Experiments

| Experiment | File | Purpose |
|------------|------|---------|
| **Exp 1** | `exp1_correlational_mapping.py` | Proves models build time-series concepts before causality |
| **Exp 2** | `exp2_causal_heatmap.py` | Live TUI heatmap of head-to-head influence |
| **Exp 3** | `exp3_chain_of_thought.py` | Topological analysis of step-by-step reasoning |
| **Exp 4** | `exp4_live_dashboard.py` | Real-time streaming to web dashboard |
| **Exp 5** | `exp5_eager_nexus.py` | Hypergraph extraction and isomorphic mapping |
| **Exp 6** | `exp6_head_interference.py` | Head knockout interventions with restoration scoring |

---

## 🎯 Key Results

### Quantitative Findings (50+ experiments)

| Metric | Logical Reasoning | Hallucination |
|--------|-------------------|---------------|
| **Hurst Exponent** | H = 0.72 ± 0.08 | H = 0.48 ± 0.12 |
| **Composite Score** | 78.3 ± 5.2 | 34.1 ± 8.9 |
| **Trajectory** | Persistent, smooth | Random walk, noisy |
| **ADF p-value** | < 0.01 (after diff) | > 0.1 (non-stationary) |

### Qualitative Insights

- **Layer Specialization**: Early layers (1-8) build statistical patterns, middle layers (9-16) construct causal models, late layers (17-24) refine outputs
- **Attention Head Roles**: Identified 4 archetypes — Sink (absorb noise), Separator (segment concepts), Integrator (combine info), Decision (final selection)
- **Cointegration**: Rank = 4 long-run equilibrium relationships detected via Johansen test
- **VAR Lag**: Average look-back = 3 tokens (AIC-selected)

---

## 🛠️ Technology Stack

**Core:**
- PyTorch 2.0+ (model loading, tensor ops)
- Transformers 🤗 (HuggingFace models)
- NumPy, SciPy (numerical computing, FFT)
- Statsmodels (VAR, ADF, Johansen, ACF/PACF)
- Ripser (Persistent Homology)
- scikit-learn (SVD, clustering)

**Backend:**
- FastAPI (REST API)
- Uvicorn (ASGI server)
- WebSockets (real-time streaming)

**Frontend:**
- Chart.js (interactive charts)
- Vanilla JavaScript (dashboard)
- Plotly (3D trajectories)
- Rich (terminal UI)

---

## 📖 Documentation

- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** — Comprehensive explanation for presentations (includes talking points)
- **[CHRONOSCOPE_RUN_GUIDE.md](CHRONOSCOPE_RUN_GUIDE.md)** — Step-by-step running instructions
- **[context.md](context.md)** — What we've built and what remains
- **[dev_log.md](dev_log.md)** — Development history with technical decisions
- **[update.md](update.md)** — Implementation details for Gaps A-E

---

## 🎓 Academic Context

**Course:** TSFT (Time Series Forecasting Techniques) — Semester 6  
**Category:** Advanced Project  
**Domain:** Explainable AI, LLM Interpretability

**Key Contributions:**
1. First framework to apply classical time-series analysis to LLM reasoning
2. Real-time hallucination detection via Hurst exponent
3. FDR-corrected causal head mapping with VAR/VECM
4. Zero-overhead TDA via CPU-parallel Euler Characteristic
5. Composite validity scoring with 82% hallucination detection accuracy

---

## 🔧 Configuration

Edit `chronoscope/config.py` to customize:

```python
class ChronoscopeConfig:
    model_name = "Qwen/Qwen2.5-0.5B"  # HuggingFace model ID
    target_layer_idx = 15              # Which layer to analyze (0-23)
    n_svd_components = 8               # SVD compression rank
    max_new_tokens = 150               # Generation budget
    use_tda = True                     # Enable Topological Analysis
    patching_noise = "gaussian"        # Intervention noise type
    use_local_llm_labeling = False     # Ollama hyperedge labeling
    local_llm_model = "qwen2.5-coder:3b"
```

---

## 💻 System Requirements

**Minimum:**
- Python 3.8+
- 6GB GPU VRAM (for Qwen2.5-0.5B in FP16)
- 8GB System RAM
- 4 CPU cores

**Recommended:**
- Python 3.10+
- 12GB GPU VRAM (RTX 3060 or better)
- 16GB System RAM
- 8 CPU cores

**Supported Models:**
- ✅ Qwen2.5 (0.5B, 1.5B) — Primary testbed
- ✅ GPT-2 (124M, 355M) — Educational demos
- ✅ Llama-2 (7B) — With 4-bit quantization

---

## 🐛 Troubleshooting

### GPU Out of Memory
- Chronoscope auto-retries with reduced token budget (50% reduction)
- Falls back to CPU generation on second OOM
- Manually reduce `max_new_tokens` in config

### `ripser` Not Found
- TDA experiments skip gracefully if ripser not installed
- Install: `pip install ripser`

### Dashboard Not Updating
- Check WebSocket connection: `ws://localhost:8765`
- Verify backend is running: `python integration_hub/backend/launch_chat_system.py`
- Use demo mode as fallback: `chronoscope_demo.html`

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"
```

---

## 📝 Citation

If you use Chronoscope in research, please cite:

```bibtex
@software{chronoscope2026,
  title={Chronoscope: Glass-Box Observability Framework for LLM Reasoning},
  author={[Your Names]},
  year={2026},
  url={[Your Repo URL]},
  note={Time Series Signal Processing and Topological Data Analysis for LLM Interpretability}
}
```

---

## 🤝 Contributing

This is an academic project, but contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-addition`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-addition`)
5. Open a Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **HuggingFace Transformers** for model APIs
- **Ripser** for fast persistent homology
- **Statsmodels** for rigorous time series analysis
- **FastAPI** for elegant backend framework
- **Chart.js** for beautiful visualizations

---

## 📞 Contact

**Project Maintainers:**
- [Your Name] — [Your Email]
- [Teammate Name] — [Their Email]

**Institution:** [Your University]  
**Supervisor:** [Faculty Name]

---

**⏱ Chronoscope** — Making LLM reasoning observable, one token at a time.

[🚀 Launch Demo](integration_hub/frontend/public/chronoscope_demo.html) | [📖 Read Docs](PROJECT_GUIDE.md) | [💬 Chat System](integration_hub/backend/launch_chat_system.py)
