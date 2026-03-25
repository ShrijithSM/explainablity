# 🚨 NETWORK ERROR FIX

## Problem
You're seeing: `OSError: We couldn't connect to 'https://huggingface.co'`

This happens because:
1. The model isn't downloaded yet
2. The code is configured for offline mode (for stability)

---

## ✅ SOLUTION 1: Download Model First (Recommended)

```bash
# Download the model (requires internet)
python download_model.py

# This will take 2-5 minutes
# Once complete, run your experiment:
python experiments/exp1_correlational_mapping.py
```

---

## ✅ SOLUTION 2: Use GPT-2 (Already Cached)

If GPT-2 is already on your system, use it instead:

```bash
# Download GPT-2 (much smaller - ~500MB)
python download_model.py --model gpt2

# Or edit the config to use GPT-2:
```

**Edit `chronoscope/config.py`:**
```python
class ChronoscopeConfig:
    model_name = "gpt2"  # Change from "Qwen/Qwen2.5-0.5B"
```

Then run:
```bash
python experiments/exp1_correlational_mapping.py
```

---

## ✅ SOLUTION 3: Disable Offline Mode (Temporary)

**Edit `chronoscope/models.py` (lines 31-32):**

Change:
```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
```

To:
```python
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
```

**And on line 39, change:**
```python
"local_files_only": True,
```

To:
```python
"local_files_only": False,
```

Then run:
```bash
python experiments/exp1_correlational_mapping.py
```

The model will download automatically on first run.

---

## ✅ SOLUTION 4: Check Internet Connection

```bash
# Test HuggingFace connection
curl https://huggingface.co

# If this fails, check:
# 1. Internet connection
# 2. Firewall settings
# 3. Proxy configuration
```

---

## 📊 ALTERNATIVE: Use Demo Mode (No Model Needed)

For presentations, you don't need to run experiments!

**Just open the demo:**
```
Open file: integration_hub/frontend/public/chronoscope_demo.html
```

This shows a fully working dashboard with simulated data - **perfect for presentations!**

---

## ⚡ QUICK RECOMMENDATION

**For your presentation tomorrow:**

1. **Don't run experiments** - use the demo HTML instead
2. **Open `index.html`** - show the landing page
3. **Click "Launch Demo"** - opens the working dashboard
4. **Follow PRESENTATION_GUIDE.md** script

You don't need the backend or experiments working to present successfully!

---

## 🔧 IF YOU REALLY WANT EXPERIMENTS TO WORK

**Run this now:**
```bash
# 1. Download a small model
python download_model.py --model gpt2

# 2. Edit config.py to use GPT-2
# Change line: model_name = "gpt2"

# 3. Run experiment
python experiments/exp1_correlational_mapping.py
```

**Expected time:** 10-15 minutes total

---

## ❓ WHICH SOLUTION TO CHOOSE?

| Solution | Time | Effort | For Presentation? |
|----------|------|--------|-------------------|
| **Demo HTML** | 0 min | None | ✅ **BEST** - No setup needed |
| **GPT-2** | 10 min | Low | ✅ Good - Shows real experiments |
| **Download Qwen** | 15 min | Low | ⚠️ Optional - Bigger model |
| **Disable offline** | 2 min | Medium | ⚠️ May cause issues later |

---

## 💡 MY RECOMMENDATION

**For tomorrow's presentation:**
1. **Use `chronoscope_demo.html`** - It works perfectly!
2. **Don't stress about downloads** - The demo is impressive enough
3. **Focus on understanding** - Read PRESENTATION_GUIDE.md instead

**After the presentation:**
1. Download GPT-2 for practice
2. Run experiments to see them work
3. Explore the full system

---

## ✅ VERIFY DEMO WORKS

```bash
# Just open this file in Chrome/Edge/Firefox:
integration_hub/frontend/public/chronoscope_demo.html

# You should see:
# - 12 panels
# - Animated charts
# - Data streaming
# - All features working
```

---

**Need more help? Read QUICK_START.md Troubleshooting section!**
