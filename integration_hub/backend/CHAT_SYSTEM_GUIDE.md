# Chronoscope Chat System — Setup & Usage Guide

## Overview

This is a **standalone live chat system** that directly interacts with language models and streams real-time Chronoscope analysis signals to a dashboard.

Unlike experiments, this system:
- ✓ Runs independently (no experiment scripts needed)
- ✓ Accepts **chat prompts** through a UI
- ✓ Streams **tokens in real-time** to the dashboard
- ✓ Broadcasts **live signals** (entropy, velocity, hurst exponent, etc.)
- ✓ Shows **chain-of-thought** token-by-token

## Architecture

```
User Chat Input
        ↓
FastAPI Backend Server (http://127.0.0.1:8000)
        ↓
    /chat endpoint
        ↓
LLM Generation (token-by-token)
        ↓
Chronoscope Analysis
  - Interceptor (capture activations)
  - Observer (time-series signals)
  - DashboardBridge (serialize to frames)
        ↓
WebSocket (ws://127.0.0.1:8000/ws/dashboard)
        ↓
Browser Dashboard + Chat UI
  - Chat messages (left)
  - Live entropy panel (top-right)
  - Velocity panel (top-right)
  - Signal meters (far-right)
  - System log (bottom)
```

## Installation

### 1. Install Backend Dependencies

```bash
cd c:\dev\explainablity
pip install -r integration_hub/backend/requirements_chat.txt
```

Key packages installed:
- `fastapi` — web framework
- `uvicorn` — ASGI server
- `websockets` — real-time streaming

### 2. Verify Model & Config

The system uses `ChronoscopeConfig` from `chronoscope/config.py`:
- Model: `Qwen/Qwen2.5-0.5B` (or your pinned local path)
- Device: CUDA if available, else CPU
- Target layer: 23
- Heads: 14

You can edit `chronoscope/config.py` if needed, but defaults work great.

## Running the System

### Option A: Launcher Script (Recommended)

```bash
cd c:\dev\explainablity
python integration_hub/backend/launch_chat_system.py
```

This will:
1. Start the backend server
2. Wait for initialization
3. Automatically open the chat UI in your browser
4. Keep the server running

**Stop it:** Press `Ctrl+C` in the terminal

### Option B: Manual Backend + Frontend

**Terminal 1: Start backend**
```bash
cd c:\dev\explainablity
python -m uvicorn integration_hub.backend.chat_server:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2: Open frontend**
```
file:///c:/dev/explainablity/integration_hub/frontend/chronoscope_chat.html
```

Then manually open that path in your browser.

## Using the Chat System

### Chat Interface

1. **Type your prompt** in the input box (bottom-left)
2. **Press Enter** or click **Send**
3. Watch the **live response stream in real-time**

Example prompts:
- "What is 2 plus 2?"
- "Explain machine learning"
- "Let's think step by step: All cats are animals. Whiskers is a cat. What is Whiskers?"

### Dashboard Panels

**Left: Chat History**
- User messages appear in cyan
- Model responses in green
- System messages in amber

**Top-Center: Head Entropy**
- Real-time per-head attention entropy
- 14 entropy values (one per attention head)
- Green bars show signal strength

**Top-Right: Velocity**
- Arc step magnitude
- Shows how quickly signals change
- Spinner animation during generation

**Far-Right: Signal Meters**
- **Tokens**: Current token index
- **Layer**: Target analysis layer
- **Heads**: Number of attention heads
- **Hurst**: Hurst exponent (α) — long-range dependence
- **Tau**: Normalized timescale
- **Mode**: CoT (chain-of-thought) or Direct

**Bottom: System Log**
- Connection status
- Token generation progress
- Errors and diagnostics

## API Endpoints

### POST /chat

Send a prompt and stream response with live analysis.

**Request:**
```json
{
  "prompt": "What is 2+2?",
  "max_tokens": 200,
  "temperature": 0.7,
  "use_cot": false
}
```

**Response:**
```json
{
  "status": "success",
  "prompt": "What is 2+2?",
  "response": "2 plus 2 equals 4.",
  "tokens_generated": 8,
  "use_cot": false
}
```

### GET /health

Check server status.

```
$ curl http://127.0.0.1:8000/health

{"status": "ok", "model_loaded": true, "connected_clients": 1}
```

### GET /config

Get current system configuration.

```
$ curl http://127.0.0.1:8000/config

{
  "model_name": "Qwen/Qwen2.5-0.5B",
  "device": "cuda",
  "n_heads": 14,
  "hidden_dim": 896,
  "target_layer": 23,
  "torch_dtype": "float16"
}
```

### WebSocket /ws/dashboard

Real-time signal streaming. The browser automatically connects to this.

**Frame format** (sent per token):
```json
{
  "token_idx": 1,
  "token_text": " 2",
  "entropy_row": [0.87, 0.92, 0.65, ...],
  "hurst": 0.715,
  "tau": 2.3,
  "velocity": 0.18,
  "layer_idx": 23,
  "n_heads": 14
}
```

## Troubleshooting

### "Connection refused" or "WebSocket failed to connect"

**Problem:** Backend server not running or not responding  
**Solution:** 
- Check that backend started successfully
- Try restarting with `launch_chat_system.py`
- Check terminal for error messages

### "Model not loaded" error

**Problem:** Model initialization failed  
**Solution:**
- Check available disk space (models are ~2GB)
- Verify HuggingFace cache is accessible
- Check `chronoscope/config.py` model path

### Slow response or timeouts

**Problem:** Generation taking too long  
**Solution:**
- Reduce `max_tokens` (default 200)
- Use CPU mode if CUDA memory is full
- Try shorter prompts

### Dashboard panels showing spinners

**Problem:** No data streaming  
**Solution:**
- Make sure you've sent at least one chat prompt
- Check browser console for errors (F12)
- Verify WebSocket connection in "Network" tab

## Customization

### Change Model

Edit `chronoscope/config.py`:
```python
model_name: str = "Qwen/Qwen2.5-1B"  # Change here
```

### Change Target Layer

Edit `chronoscope/config.py`:
```python
target_layer: int = 12  # Analyze layer 12 instead of 23
```

### Adjust Temperature/Sampling

In the chat UI, you'd need to modify the JavaScript:
```javascript
body: JSON.stringify({
  prompt: prompt,
  temperature: 0.5,  // Lower = more focused
  max_tokens: 300    // Higher = longer responses
})
```

### Enable Chain-of-Thought Mode

Modify chat request:
```javascript
use_cot: true  // Streams intermediate reasoning steps
```

## Files

```
integration_hub/
├── backend/
│   ├── chat_server.py              ← Main FastAPI backend
│   ├── launch_chat_system.py        ← Launcher script
│   └── requirements_chat.txt        ← Dependencies
└── frontend/
    └── chronoscope_chat.html        ← Chat + dashboard UI
```

## Performance Notes

- **First run:** Model loading takes 30-60 seconds
- **Token generation:** ~200-500ms per token on GPU
- **Signal streaming:** Real-time (<100ms latency)
- **Memory:** ~4-6GB for Qwen 0.5B on VRAM (adjust with `load_in_4bit`)

## Example Session

```
Terminal:
$ python integration_hub/backend/launch_chat_system.py

[•] Starting backend server...
[✓] Backend server started (PID: 12345)
[•] Waiting for server to initialize (10s)...
[✓] Frontend opened in browser

Browser:
User: "What are the three laws of thermodynamics?"
[Tokens streaming... entropy updating... velocity bars animating...]
Assistant: "The three laws of thermodynamics are:\n1. Energy is conserved...\n2. Entropy always increases...\n3. ..."

System Log:
[12:34:56] Prompt: What are the three laws of thermodynamics?
[12:34:57] ✓ WebSocket connected
[12:35:02] Generated 87 tokens
[12:35:02] Hurst: 0.718 | Tau: 2.1 | Velocity: 0.24
```

## Next Steps

- **Extend with custom analysis:** Add new signals to `Observer` and stream them
- **Add memory:** Store conversation history for multi-turn chats
- **Scale to larger models:** Swap `Qwen2.5-0.5B` for `7B` or `14B`
- **Integrate with dashboard:** Customize the visualization panels

Enjoy exploring your model's chain-of-thought in real-time! 🚀
