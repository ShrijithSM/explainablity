# CHRONOSCOPE — QUICK START GUIDE

> **Get up and running in 5 minutes**

---

## 🚀 INSTALLATION

### 1. Prerequisites
```bash
# Python 3.8 or higher
python --version

# Git (to clone repo)
git --version

# GPU recommended but not required
# CUDA 11.8+ for GPU support
```

### 2. Clone Repository
```bash
# Clone (replace with your actual repo URL)
git clone <your-repo-url>
cd explainablity
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print('Transformers: OK')"
python -c "import statsmodels; print('Statsmodels: OK')"
```

**Optional TDA Support:**
```bash
pip install ripser
```

**Optional Local LLM Labeling:**
Download Ollama from https://ollama.ai and run:
```bash
ollama pull qwen2.5-coder:3b
```

---

## ⚡ FASTEST WAY TO SEE CHRONOSCOPE IN ACTION

### Option 1: Live Dashboard Demo (No Backend)

**Simplest — Just open in browser:**
```
Open file: integration_hub/frontend/public/chronoscope_demo.html
```

✅ **No installation required**  
✅ **No backend needed**  
✅ **Shows simulated data streaming**  
✅ **All 12 panels working**  
✅ **Perfect for presentations/demos**

---

### Option 2: Run An Experiment (With Real Model)

```bash
# Experiment 1: Causal head mapping
python experiments/exp1_correlational_mapping.py

# What you'll see:
# - Model loads
# - Generates text
# - VAR analysis runs
# - Influence heatmap created
# - Report saved to reports/
```

**Expected output:**
```
Loading model: Qwen/Qwen2.5-0.5B...
[OK] Model loaded on cuda
Prompt: The stock price follows...
Token generation: ████████████ 100%
─────────────────────────────────────
Composite Validity Score: 78/100
Verdict: STRONG REASONING
Hurst Exponent: 0.72 (persistent)
VAR Lag Order: 3 tokens
Significant Pairs: 12/196 (FDR α=0.05)
─────────────────────────────────────
Report saved: reports/causal_report.md
```

---

### Option 3: Live Chat + Dashboard (Full Stack)

```bash
# Start backend server + dashboard
python integration_hub/backend/launch_chat_system.py

# What happens:
# 1. Backend starts on port 8000
# 2. WebSocket starts on port 8765
# 3. Dashboard opens automatically in browser
# 4. You can chat and see real-time analysis
```

**Dashboard auto-opens at:**
```
http://localhost:8766
```

**Chat UI available at:**
```
http://localhost:8000
```

**Test the API with curl:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain why the sky is blue", "max_tokens": 100}'
```

---

## 📊 WHAT TO EXPECT

### First Run
- **Model download:** ~500MB (Qwen2.5-0.5B) — happens once
- **Load time:** 10-30 seconds depending on GPU
- **Generation speed:** 20-50 tokens/sec on GPU, 2-5 on CPU
- **Analysis overhead:** ~0ms during generation, 2-5 sec post-generation

### Dashboard Panels (12 total)
1. **State Panel** — Real-time metrics (Hurst, ADF, Johansen)
2. **Entropy Chart** — 14 lines showing per-head entropy
3. **VAR Matrix** — 14×14 causal influence heatmap
4. **Composite Verdict** — 0-100 score with ring visualization
5. **VAR Influence List** — Bar chart of top causal pairs
6. **Perturbation Results** — Head knockout experiments
7. **TDA Panel** — Betti numbers and Euler characteristic
8. **Signal Quality** — 5 per-head metrics
9. **HMM Phase** — 4-state reasoning phase timeline
10. **Phase Timeline** — Visual sequence of phases
11. **System Log** — Classified event stream
12. **Chat Island** — Interactive prompt input

---

## 🧪 TEST EACH EXPERIMENT

### Experiment 1: Correlational Mapping
```bash
python experiments/exp1_correlational_mapping.py
```
**Tests:** VAR analysis, FDR correction, Granger causality  
**Duration:** ~2 minutes  
**Output:** `reports/causal_report.md`

---

### Experiment 2: Causal Heatmap (TUI)
```bash
python experiments/exp2_live_heatmap.py
```
**Tests:** Live terminal dashboard with Rich library  
**Duration:** Interactive (press Ctrl+C to exit)  
**Output:** Terminal UI with real-time heatmap

---

### Experiment 3: Chain-of-Thought Topology
```bash
python experiments/exp3_chain_of_thought.py
```
**Tests:** HMM phase detection, TDA, Betti numbers  
**Duration:** ~3 minutes  
**Output:** `reports/cot_analysis.md`

---

### Experiment 4: Live Dashboard Feed
```bash
# Terminal 1: Start backend
python integration_hub/backend/launch_chat_system.py

# Terminal 2: Run experiment
python experiments/exp4_live_dashboard.py
```
**Tests:** WebSocket streaming, real-time dashboard updates  
**Duration:** ~2 minutes  
**Output:** Live dashboard updates

---

### Experiment 5: Eager NEXUS
```bash
python experiments/exp5_eager_nexus.py
```
**Tests:** Hypergraph extraction, isomorphic clusters  
**Duration:** ~3 minutes  
**Output:** `reports/nexus_report.md`

---

### Experiment 6: Head Interference
```bash
python experiments/exp6_head_interference.py
```
**Tests:** Head knockout, restoration scoring, mediation  
**Duration:** ~5 minutes (slowest, does interventions)  
**Output:** `reports/intervention_report.md`

---

## 🔧 CONFIGURATION

### Quick Config Changes
Edit `chronoscope/config.py`:

```python
class ChronoscopeConfig:
    # Model
    model_name = "Qwen/Qwen2.5-0.5B"  # Change to any HuggingFace model
    
    # Analysis
    target_layer_idx = 15              # Which layer to analyze (0-23)
    n_svd_components = 8               # SVD compression rank
    
    # Generation
    max_new_tokens = 150               # Token budget
    temperature = 0.7                  # Sampling temperature
    
    # Features
    use_tda = True                     # Enable topological analysis
    capture_attentions = True          # Enable head entropy tracking
    
    # Dashboard
    dashboard_transport = "websocket"  # "websocket" or "file"
    dashboard_ws_port = 8765           # WebSocket port
    dashboard_http_port = 8766         # Dashboard HTTP port
```

---

## 🐛 TROUBLESHOOTING

### Issue: GPU Out of Memory
**Solution 1:** Reduce token budget
```python
config.max_new_tokens = 50  # Instead of 150
```

**Solution 2:** Use CPU (automatic fallback)
```python
config.device = "cpu"
```

**Solution 3:** Smaller model
```python
config.model_name = "gpt2"  # 124M params vs 500M
```

---

### Issue: Dashboard Not Updating
**Check 1:** Backend running?
```bash
# Should see: Uvicorn running on http://127.0.0.1:8000
ps aux | grep launch_chat_system
```

**Check 2:** WebSocket connected?
Open browser console (F12) and look for:
```
WebSocket connected to ws://localhost:8765
```

**Check 3:** Use demo mode
```
Open: integration_hub/frontend/public/chronoscope_demo.html
```

---

### Issue: Import Errors
**Reinstall dependencies:**
```bash
pip install -r requirements.txt --force-reinstall --no-cache-dir
```

**Check specific imports:**
```bash
python -c "from chronoscope.config import ChronoscopeConfig; print('OK')"
python -c "from chronoscope.models import load_model; print('OK')"
python -c "from chronoscope.interceptor import ChronoscopeInterceptor; print('OK')"
```

---

### Issue: Model Download Slow
**Use local cache:**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/your/cache
python experiments/exp1_correlational_mapping.py
```

**Pre-download model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
```

---

## 📝 VERIFY INSTALLATION

### Run Test Suite
```bash
# Run all tests
pytest chronoscope/tests/ -v

# Run specific test categories
pytest chronoscope/tests/test_benchmarks.py -v
pytest chronoscope/tests/test_perturbation.py -v
pytest chronoscope/tests/test_significance.py -v
```

**Expected output:**
```
======================== test session starts =========================
collected 24 items

test_benchmarks.py::TestTDABenchmarks::test_ripser_window_sweep PASSED
test_benchmarks.py::TestPyTorchVsNumpy::test_entropy_speed PASSED
test_benchmarks.py::TestIncrementalSVD::test_shape_correctness PASSED
...
====================== 24 passed, 13 skipped ======================
```

---

## 🎯 RECOMMENDED FIRST STEPS

### For Understanding the Project:
1. Read [PROJECT_GUIDE.md](PROJECT_GUIDE.md) — Complete explanation
2. Open `chronoscope_demo.html` — See dashboard
3. Read [CHRONOSCOPE_RUN_GUIDE.md](CHRONOSCOPE_RUN_GUIDE.md) — Detailed instructions

### For Running Experiments:
1. Run Experiment 1 (`exp1_correlational_mapping.py`)
2. Open the generated report (`reports/causal_report.md`)
3. Run Experiment 4 with dashboard (`exp4_live_dashboard.py`)

### For Presentations:
1. Open `index.html` in browser — Landing page
2. Click "Launch Live Demo" — Opens `chronoscope_demo.html`
3. Review talking points in [PROJECT_GUIDE.md](PROJECT_GUIDE.md#presentation-talking-points)

---

## 📊 EXAMPLE SESSION

```bash
# Step 1: Verify installation
python -c "import torch, transformers, statsmodels; print('All dependencies OK')"

# Step 2: Run first experiment
python experiments/exp1_correlational_mapping.py

# Step 3: View results
cat reports/causal_report.md

# Step 4: Launch dashboard
python integration_hub/backend/launch_chat_system.py

# Step 5: In browser, interact with chat
# POST prompt: "What is the capital of France?"
# Watch dashboard update in real-time

# Step 6: Review dashboard panels
# - Entropy chart shows per-head dynamics
# - VAR matrix shows causal relationships
# - Composite score shows reasoning quality
```

---

## 🎓 FOR PRESENTATIONS

### Before Demo:
1. Pre-load model: `python -c "from chronoscope.models import load_model; load_model()"`
2. Open `index.html` in browser
3. Open `chronoscope_demo.html` in another tab
4. Have [PROJECT_GUIDE.md](PROJECT_GUIDE.md) open for talking points

### During Demo:
1. Show landing page (`index.html`)
2. Explain project concept (Glass-Box Observability)
3. Click "Launch Live Demo"
4. Explain each dashboard panel
5. Run a live experiment if time permits

### Backup Plan:
- Use `chronoscope_demo.html` (no backend needed)
- Show pre-generated reports in `reports/`
- Screenshots of successful runs

---

## ⏱️ TIME ESTIMATES

| Task | Time (First Run) | Time (Subsequent) |
|------|------------------|-------------------|
| Install dependencies | 5-10 min | — |
| Model download | 2-5 min | — (cached) |
| Experiment 1 | 2 min | 1 min |
| Experiment 3 | 3 min | 2 min |
| Experiment 6 | 5 min | 4 min |
| Launch chat system | 30 sec | 20 sec |
| Dashboard demo | Instant | Instant |

---

## 🎯 SUCCESS CHECKLIST

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model loads without errors
- [ ] Experiment 1 runs and generates report
- [ ] Dashboard demo opens (`chronoscope_demo.html`)
- [ ] Live chat system starts (`launch_chat_system.py`)
- [ ] WebSocket connects (check browser console)
- [ ] Dashboard updates in real-time
- [ ] All documentation reviewed

---

## 📞 NEED HELP?

**Common Issues:**
- GPU OOM → Reduce `max_new_tokens` or use CPU
- Import errors → Reinstall dependencies
- Dashboard not updating → Use demo mode
- Model download slow → Use local cache

**Documentation:**
- **Technical Details:** [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- **Run Instructions:** [CHRONOSCOPE_RUN_GUIDE.md](CHRONOSCOPE_RUN_GUIDE.md)
- **Development Log:** [dev_log.md](dev_log.md)
- **API Reference:** [context.md](context.md)

---

**🎉 You're ready to explore Chronoscope!**

Start with: `python experiments/exp1_correlational_mapping.py`
