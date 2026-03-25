# Chronoscope — How to Run

---

## Option A — Demo Website (no backend, open directly)

The file `integration_hub/frontend/public/chronoscope_demo.html` is fully self-contained.  
It generates synthetic data internally and requires no Python, no server, and no internet (only Google Fonts and Chart.js CDN on first load).

**Steps:**

1. Open `integration_hub/frontend/public/chronoscope_demo.html` in any modern browser  
   (Chrome, Edge, Firefox — right-click → Open with)
2. The dashboard starts immediately in **DEMO MODE**, streaming simulated frames at ~700 ms/tick
3. All panels animate: entropy chart, influence matrix, perturbation, TDA, HMM phases, composite score
4. Press `?` (or `/`) to open the built-in Guide overlay
5. Press `Esc` to close the guide

> The cyan banner at the bottom reads **DEMO MODE — Simulated data stream active**.  
> No WebSocket connection is attempted.

---

## Option B — Live Dashboard (real LLM inference)

This connects the actual Chronoscope Python pipeline to the dashboard.

### 1. Install dependencies

```bash
cd d:\Jain\3rd Year\Sem6\TSFT\explainablity
pip install -r requirements.txt
# WebSocket transport
pip install websockets
```

### 2. Start the WebSocket bridge server

```bash
python -c "
import asyncio, websockets, json, pathlib
FRAMES_FILE = '/tmp/chronoscope_frame.json'

async def handler(ws):
    import time
    print(f'Client connected: {ws.remote_address}')
    while True:
        if pathlib.Path(FRAMES_FILE).exists():
            frame = json.loads(pathlib.Path(FRAMES_FILE).read_text())
            await ws.send(json.dumps(frame))
        await asyncio.sleep(0.3)

asyncio.run(websockets.serve(handler, 'localhost', 8765))
"
```

Or use the existing bridge:

```bash
python integration_hub/backend/launch_chat_system.py
```

### 3. Run an experiment that feeds live data

```bash
# Experiment 1 — correlational mapping
python experiments/exp1_correlational_mapping.py

# Experiment 4 — live dashboard feed
python experiments/exp4_live_dashboard.py
```

Each experiment writes frames to `/tmp/chronoscope_frame.json`,  
which the WebSocket bridge picks up and forwards to the browser.

### 4. Open the live dashboard

Open `integration_hub/frontend/public/chronoscope_live.html` in your browser.

The dashboard connects to `ws://localhost:8765` automatically.  
The bottom-left dots (`HOOKS / VAR / FDR / HMM`) turn cyan/green as each analysis stage completes.

---

## Option C — Direct API push (Python script)

You can push frames directly without WebSocket by opening the HTML from a local HTTP server:

```bash
# In the project root
python -m http.server 8080
# Then open http://localhost:8080/integration_hub/frontend/public/chronoscope_live.html
```

Then from Python:

```python
import json, pathlib
frame = { ... }   # dict matching ChronoscopeBridge frame schema
pathlib.Path('/tmp/chronoscope_frame.json').write_text(json.dumps(frame))
```

The dashboard polls `/tmp/chronoscope_frame.json` every 300 ms as a fallback.

---

## Chat Island

The floating **CHAT ISLAND** panel (bottom-left button) sends prompts to `http://127.0.0.1:8000/chat`.  
Start the chat server to enable it:

```bash
python integration_hub/backend/chat_server.py
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `?` or `/` | Open Guide overlay |
| `Esc` | Close Guide overlay |

---

## File Locations

| File | Purpose |
|------|---------|
| `integration_hub/frontend/public/chronoscope_demo.html` | **Self-contained demo** — open directly in browser |
| `integration_hub/frontend/public/chronoscope_live.html` | **Live dashboard** — needs WebSocket backend |
| `integration_hub/backend/launch_chat_system.py` | Start full backend (WebSocket + chat API) |
| `experiments/exp4_live_dashboard.py` | Live experiment feeding data to dashboard |
| `run_experiment.py` | General experiment runner |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Dashboard shows blank panels | WebSocket not running — use demo file or start backend |
| Chart.js/fonts not loading | No internet — download CDN assets or use a local copy |
| `conn-banner` stays amber (live file) | WebSocket failed — check `ws://localhost:8765` is listening |
| `DEMO MODE` banner stays in live file | You opened `chronoscope_demo.html` instead of `chronoscope_live.html` |
