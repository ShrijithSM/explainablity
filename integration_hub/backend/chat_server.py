"""
Chronoscope Online Chat Server — Live model interaction with real-time analysis.

Architecture:
  :8000  — FastAPI chat API (POST /chat) + serves chat UI
  :8765  — DashboardBridge WebSocket (chronoscope_live.html connects here)
  :8766  — HTTP server for chronoscope_live.html dashboard

Flow:
  User prompt → POST /chat → model generates token-by-token →
  interceptor captures activations → observer analyses live →
  DashboardBridge pushes frames to dashboard → panels stream in real time

This is the ONLINE mode (always-on server, interactive chat).
Exp6 is OFFLINE mode (script, fixed prompt, exit).
"""

import asyncio
import json
import sys
import os
import webbrowser
from typing import Optional, Set
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Thread

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from chronoscope.config import ChronoscopeConfig
from chronoscope.models import load_model, get_deepest_layer
from chronoscope.interceptor import ChronoscopeInterceptor
from chronoscope.observer import SignalObserver
from chronoscope.analyzer import CausalAnalyzer
from chronoscope.dashboard_bridge import DashboardBridge
from transformers import TextIteratorStreamer


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt text")
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    run_var: bool = Field(default=True, description="Run VAR head interaction analysis post-generation")
    run_intervention: bool = Field(default=False, description="Run head knockout intervention (slower)")


from pydantic import BaseModel, Field, model_validator

def _sanitize_numpy(obj):
    """Recursively strip NumPy types to native Python types for clean JSON serialization."""
    if isinstance(obj, (np.bool_, np.generic)) and hasattr(obj, 'item'):
        # handles np.bool_, np.float32, np.int64, etc.
        item = obj.item()
        # Ensure we return native bool/int/float, since numpy scalar .item() does this
        return item
    if isinstance(obj, np.ndarray):
        return [_sanitize_numpy(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    return obj

class ChatResponse(BaseModel):
    status: str
    prompt: str = ""
    response: str = ""
    tokens_generated: int = 0
    analysis_summary: dict = {}
    message: str = ""

    @model_validator(mode='after')
    def sanitize_types(self):
        self.analysis_summary = _sanitize_numpy(self.analysis_summary)
        return self


# ============================================================================
# GLOBAL STATE
# ============================================================================

_model = None
_tokenizer = None
_config: Optional[ChronoscopeConfig] = None
_interceptor: Optional[ChronoscopeInterceptor] = None
_observer: Optional[SignalObserver] = None
_analyzer = None
_bridge: Optional[DashboardBridge] = None
_generation_lock = asyncio.Lock()


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_system():
    """Load model + analysis stack + start DashboardBridge on :8765."""
    global _model, _tokenizer, _config, _interceptor, _observer, _analyzer, _bridge

    print("=" * 60)
    print("CHRONOSCOPE ONLINE — Live Chat + Analysis Dashboard")
    print("=" * 60)

    # ── Config ────────────────────────────────────────────────────────
    _config = ChronoscopeConfig()
    print(f"  Model:  {_config.model_name}")
    print(f"  Device: {_config.device}")

    # ── Load model ────────────────────────────────────────────────────
    _model, _tokenizer = load_model(_config)
    print("[OK] Model loaded")

    # ── Analysis components ───────────────────────────────────────────
    _interceptor = ChronoscopeInterceptor(_model, _tokenizer, _config)
    _observer = SignalObserver(_config)
    _analyzer = CausalAnalyzer(_interceptor, _observer, _config)
    print("[OK] Interceptor + Observer + Analyzer ready")

    # ── DashboardBridge on :8765 (what chronoscope_live.html expects) ──
    try:
        _bridge = DashboardBridge(
            transport=getattr(_config, "dashboard_transport", "websocket"),
            ws_port=int(getattr(_config, "dashboard_ws_port", 8765)),
        ).start()

        dashboard_url = _bridge.serve_dashboard(
            getattr(_config, "dashboard_html_path",
                    "integration_hub/frontend/public/chronoscope_live.html"),
            port=int(getattr(_config, "dashboard_http_port", 8766)),
        )
        print(f"[OK] Dashboard: {dashboard_url}")
        print(f"[OK] Dashboard WS: ws://localhost:{_config.dashboard_ws_port}")
        _bridge.push_log("ok", f"Online server starting · model={_config.model_name}")

        # Auto-open dashboard in browser
        webbrowser.open(dashboard_url)
    except Exception as e:
        print(f"[!] Dashboard bridge failed: {e}")
        _bridge = None

    print(f"\n[OK] Chat API ready at http://127.0.0.1:8000/chat")
    print(f"     POST a JSON body: {{\"prompt\": \"your question\"}}\n")


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_system()
    yield
    if _interceptor:
        _interceptor.cleanup()
    if _bridge:
        _bridge.push_log("ok", "Server shutting down")
        _bridge.stop()


app = FastAPI(title="Chronoscope Online", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# CHAT ENDPOINT — Online analysis while model reasons
# ============================================================================

def run_post_analysis_background(target_layer, current_traj, prompt_len, token_count, prompt, run_var, run_intervention):
    """
    Runs the heavy STATSMODELS and CPU-bound topological analysis asynchronously.
    Updates the dashboard via the bridge once finished.
    """
    global _observer, _analyzer, _bridge, _config
    
    print("[*] Background: Running post-generation analysis...")
    if _bridge:
        _bridge.push_log("ok", f"Generation complete ({token_count} tokens). Running deep analysis...")

    try:
        if target_layer and target_layer in current_traj:
            traj_tensor = current_traj[target_layer]
            gen_tensor = traj_tensor[prompt_len:] if traj_tensor.shape[0] > prompt_len else traj_tensor

            if gen_tensor.shape[0] >= 5:
                # Full observer analysis
                observer_results = _observer.full_analysis(gen_tensor)

                print(f"  Hurst={_observer.hurst_exponent:.3f}, "
                      f"tau={_observer.tau_normalised:.3f}, "
                      f"ADF p={_observer.adf_pval_pc0:.4f}")

                if _bridge:
                    _bridge.push_token_frame(token_count, _interceptor, _observer, _config)

                # TDA
                compressed = observer_results["compressed_trajectory"]
                tda_results = _analyzer.topological_analysis(compressed)
                betti = tda_results.get("betti_numbers", {})

                if _bridge:
                    _bridge.push_tda_frame(
                        tda_result={
                            "betti0": int(betti.get("betti_0", 0)),
                            "betti1": int(betti.get("betti_1", 0)),
                            "euler": int(betti.get("betti_0", 0) - betti.get("betti_1", 0)),
                            "ec_series": _observer._ec_history,
                            "anomalies": [],
                        },
                        current_token=token_count,
                        phase_boundaries=_observer.phase_boundaries,
                        current_phase_idx=_observer.current_phase_idx,
                    )

                # VAR head interaction analysis (if enabled)
                if run_var and getattr(_config, "capture_attentions", False):
                    print("[*] Background: Running VAR head interaction analysis (multi-layer)...")
                    head_result = _analyzer.head_interaction_analysis(
                        prompt  # No layer_name → analyzer auto-selects mid + deepest layers
                    )
                    if "error" not in head_result:
                        if _bridge:
                            _bridge.push_var_frame(head_result)
                            _bridge.push_log("ok", "VAR analysis complete")

                        # ── Live Intervention (Knockout Probe) ──
                        if run_intervention:
                            print("[*] Background: Running live intervention...")
                            if _bridge:
                                _bridge.push_log("pert", "Running live head interventions...")
                            
                            fdr_res = head_result.get("fdr_result", {})
                            sig_pairs = fdr_res.get("significant_pairs", [])
                            if sig_pairs:
                                source_heads = list(set([j for j, i, p in sig_pairs]))
                                
                                all_layers = sorted(current_traj.keys())
                                causal_impact = _analyzer.interventional_head_causality_multilayer(
                                    prompt, all_layers, source_heads
                                )
                                per_head = causal_impact.get("per_head_results", [])
                                
                                if per_head and _bridge:
                                    pert_results = [
                                        {
                                            "head":          int(r["head"]),
                                            "target":        int(r.get("target_head", -1)),
                                            "mode":          "zero",
                                            "delta_entropy": float(r.get("delta_entropy", 0.0)),
                                            "restoration":   float(r.get("restoration", 0.0)),
                                            "kl_patch":      float(r.get("kl_patch", 0.0)),
                                            "confirmed":     float(r.get("restoration", 0.0)) > 0.5,
                                        }
                                        for r in per_head
                                    ]
                                    _bridge.push_perturbation_frame(pert_results, None)
                                    _bridge.push_log("ok", f"Interventions complete on {len(pert_results)} heads")
                            else:
                                if _bridge:
                                    _bridge.push_log("pert", "No significant VAR pairs found to intervene on.")
                    else:
                        if _bridge:
                            _bridge.push_log("err", f"VAR: {head_result['error']}")

                # Validity score
                validity = _analyzer.compute_validity_score(
                    dtw_result={},
                    spectral_result=observer_results.get("spectral", {}),
                    tda_result=tda_results,
                    stationarity_result=observer_results.get("stationarity", {}),
                )

                composite_score = int(round(validity.get("composite_validity", 0) * 100))
                verdict = (
                    "STRONG REASONING" if composite_score > 70 else
                    "MODERATE REASONING" if composite_score > 50 else
                    "HALLUCINATION RISK"
                )

                if _bridge:
                    def _sf(v):
                        try: return float(v) if v is not None else None
                        except (TypeError, ValueError): return None

                    _bridge.push_score_frame(
                        {
                            "score": composite_score,
                            "verdict": verdict,
                            "dtw_sensitivity": _sf(validity.get("dtw_sensitivity")),
                            "spectral_coherence": _sf(validity.get("spectral_validity")),
                            "topo_smoothness": _sf(validity.get("tda_validity")),
                            "active_reasoning": _sf(validity.get("active_reasoning")),
                            "fdr_sig_pairs": 0,
                            "te_score": None,
                        },
                        {
                            "source": "Background Analysis",
                            "text": f"Score={composite_score}/100 · {verdict}",
                        },
                    )
    except Exception as e:
        print(f"[!] Background analysis error: {e}")
        import traceback
        traceback.print_exc()
        if _bridge:
            _bridge.push_log("err", f"Deep analysis failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Accepts a prompt, generates token-by-token, and streams live
    Chronoscope analysis to the dashboard as the model reasons.
    """
    global _model, _tokenizer, _config, _interceptor, _observer, _analyzer, _bridge

    if not _model or not _tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if _generation_lock.locked():
        raise HTTPException(status_code=429, detail="Generation in progress — wait for current request")

    async with _generation_lock:
        prompt = request.prompt.strip()
        max_tokens = request.max_tokens
        temperature = request.temperature

        print(f"\n[CHAT] Prompt: {prompt[:80]}...")
        if _bridge:
            _bridge.push_log("ok", f"New prompt: {prompt[:50]}...")

        # ── Tokenize + measure prompt length ─────────────────────────
        inputs = _tokenizer(prompt, return_tensors="pt").to(_config.device)
        prompt_len = inputs.input_ids.shape[1]

        # ── Clear interceptor state ──────────────────────────────────
        _interceptor._clear()
        _observer.__init__(_config)  # Reset observer for fresh analysis

        # ── Stream generation with live analysis ─────────────────────
        full_response = ""
        token_count = 0
        analysis_summary = {}

        try:
            # Use capture_generation_stream for live token-by-token analysis
            # This is the same pattern as exp6 capture_node (E3-K1)
            print("[*] Streaming: ", end="", flush=True)

            for new_token, current_traj in _interceptor.capture_generation_stream(
                prompt, max_new_tokens=max_tokens
            ):
                sys.stdout.write(new_token)
                sys.stdout.flush()
                full_response += new_token
                token_count += 1

                # ── Live analysis per token ──────────────────────────
                target_layer = get_deepest_layer(current_traj.keys()) if current_traj else None

                if target_layer and target_layer in current_traj:
                    traj = current_traj[target_layer]
                    gen_only = traj[prompt_len:] if traj.shape[0] > prompt_len else traj

                    if gen_only.shape[0] >= 2:
                        # Live TDA + EC tracking (E4-K1)
                        live_stats = _observer.incremental_analysis(
                            gen_only,
                            window_size=getattr(_config, "tda_window_size", 10),
                            distance_threshold=getattr(_config, "tda_distance_threshold", 2.0),
                        )

                        # Push TDA anomaly if detected (E3-K2)
                        if live_stats.get("topological_anomaly_detected") and _bridge:
                            _bridge.push_tda_frame(
                                tda_result={
                                    "betti0": live_stats.get("betti0"),
                                    "betti1": live_stats.get("betti1"),
                                    "euler": live_stats.get("euler_characteristic"),
                                    "ec_series": live_stats.get("ec_series", []),
                                    "anomalies": [{
                                        "token_idx": token_count,
                                        "severity": "high",
                                        "description": live_stats.get("diagnostics", "EC spike"),
                                    }],
                                },
                                current_token=token_count,
                            )
                            _bridge.push_log("tda",
                                f"D.2: EC spike t={token_count} · "
                                f"χ={live_stats.get('euler_characteristic', 0)}")

                    # Push per-token entropy frame to dashboard (E3-K1)
                    metric_series = _interceptor.get_head_metric_series(target_layer)
                    if metric_series is not None and getattr(metric_series, "numel", lambda: 0)() > 0:
                        row = metric_series[-1]
                        if hasattr(row, "ndim") and row.ndim > 1:
                            row = row[:, 0]
                        if _bridge:
                            _bridge.push_token_frame(
                                token_idx=token_count,
                                interceptor=_interceptor,
                                observer=_observer,
                                config=_config,
                                entropy_row_override=row,
                            )

                # Cadenced signal quality (E2L-K1)
                stat_every = getattr(_config, "dashboard_stat_every", 5)
                if token_count % stat_every == 0 and _bridge:
                    _bridge.push_signal_quality_frame(_interceptor, _config)

                # Yield control for async responsiveness
                await asyncio.sleep(0)

            print(f"\n[OK] Generated {token_count} tokens")

        except Exception as e:
            print(f"\n[!] Generation error: {e}")
            import traceback
            traceback.print_exc()
            if _bridge:
                _bridge.push_log("err", f"Generation error: {str(e)[:80]}")
            return ChatResponse(
                status="error",
                message=str(e),
                response=full_response,
                tokens_generated=token_count,
            )

        # ── Schedule post-generation deep analysis in the background ──
        print("[*] Scheduling post-generation analysis...")
        target_layer = get_deepest_layer(current_traj.keys()) if current_traj else None
        
        # Schedule it
        background_tasks.add_task(
            run_post_analysis_background,
            target_layer=target_layer,
            current_traj=current_traj,
            prompt_len=prompt_len,
            token_count=token_count,
            prompt=prompt,
            run_var=request.run_var,
            run_intervention=request.run_intervention
        )
        
        # Quick summary for instant response
        analysis_summary = {
            "status": "Analysis computing in background...",
        }

        return ChatResponse(
            status="success",
            prompt=prompt,
            response=full_response,
            tokens_generated=token_count,
            analysis_summary=analysis_summary,
        )


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "bridge_active": _bridge is not None,
    }


@app.get("/config")
async def get_config():
    if not _config:
        raise HTTPException(status_code=503, detail="Not initialized")
    return {
        "model_name": _config.model_name,
        "device": _config.device,
        "n_heads": _config.n_heads,
        "hidden_dim": _config.hidden_dim,
        "target_layer": _config.target_layer,
    }


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Serve a minimal chat interface that POSTs to /chat."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Chronoscope Chat</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:#0a0a0f; color:#e0e0e0; height:100vh; display:flex; flex-direction:column; }
  header { background:linear-gradient(135deg,#1a1a2e,#16213e); padding:16px 24px; border-bottom:1px solid #ffffff15; }
  header h1 { font-size:1.2em; background:linear-gradient(90deg,#00d4ff,#00ff88); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  header p { font-size:0.8em; color:#888; margin-top:4px; }
  #messages { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:80%; padding:12px 16px; border-radius:12px; font-size:0.95em; line-height:1.5; }
  .msg.user { align-self:flex-end; background:#1e3a5f; border:1px solid #2a5a8f; }
  .msg.assistant { align-self:flex-start; background:#1a1a2e; border:1px solid #ffffff15; }
  .msg .meta { font-size:0.75em; color:#666; margin-top:6px; }
  .msg .verdict { font-weight:bold; padding:2px 8px; border-radius:4px; font-size:0.8em; }
  .verdict.strong { background:#00ff8830; color:#00ff88; }
  .verdict.moderate { background:#ffb30030; color:#ffb300; }
  .verdict.risk { background:#ff3d5030; color:#ff3d50; }
  #input-bar { display:flex; gap:10px; padding:16px 24px; background:#0d0d15; border-top:1px solid #ffffff10; }
  #prompt { flex:1; background:#1a1a2e; border:1px solid #ffffff20; color:#fff; padding:12px 16px; border-radius:8px; font-size:1em; outline:none; }
  #prompt:focus { border-color:#00d4ff; }
  #send { background:linear-gradient(135deg,#00d4ff,#00a8cc); color:#000; font-weight:bold; border:none; padding:12px 24px; border-radius:8px; cursor:pointer; }
  #send:disabled { opacity:0.4; cursor:not-allowed; }
  .spinner { display:inline-block; width:16px; height:16px; border:2px solid #00d4ff40; border-top-color:#00d4ff; border-radius:50%; animation:spin .6s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  a { color:#00d4ff; }
</style>
</head>
<body>
<header>
  <h1>⟐ Chronoscope Online</h1>
  <p>Live chat with real-time analysis → <a href="http://localhost:8766" target="_blank">Open Dashboard</a></p>
</header>
<div id="messages">
  <div class="msg assistant">Ready. Type a prompt and watch the <a href="http://localhost:8766" target="_blank">live dashboard</a> as the model reasons.</div>
</div>
<div id="input-bar">
  <input id="prompt" type="text" placeholder="Ask something..." autofocus />
  <button id="send" onclick="sendChat()">Send</button>
</div>
<script>
const msgs = document.getElementById('messages');
const inp = document.getElementById('prompt');
const btn = document.getElementById('send');

inp.addEventListener('keydown', e => { if(e.key==='Enter' && !btn.disabled) sendChat(); });

async function sendChat() {
  const prompt = inp.value.trim();
  if(!prompt) return;
  inp.value = '';
  btn.disabled = true;

  // User message
  const um = document.createElement('div');
  um.className = 'msg user';
  um.textContent = prompt;
  msgs.appendChild(um);

  // Thinking indicator
  const tm = document.createElement('div');
  tm.className = 'msg assistant';
  tm.innerHTML = '<span class="spinner"></span> Generating + analyzing...';
  msgs.appendChild(tm);
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const res = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt, max_tokens:150, run_var:true})
    });
    const data = await res.json();

    let verdictHtml = '';
    if(data.analysis_summary && data.analysis_summary.verdict) {
      const v = data.analysis_summary.verdict;
      const cls = v.includes('STRONG') ? 'strong' : v.includes('MODERATE') ? 'moderate' : 'risk';
      verdictHtml = '<br><span class="verdict '+cls+'">'+v+' ('+data.analysis_summary.composite_score+'/100)</span>';
    }

    let metaHtml = '';
    if(data.analysis_summary) {
      const s = data.analysis_summary;
      const parts = [];
      if(s.hurst !== undefined) parts.push('H='+s.hurst.toFixed(3));
      if(s.tau !== undefined) parts.push('τ='+s.tau.toFixed(3));
      if(s.is_stationary !== undefined) parts.push(s.is_stationary ? 'Stationary' : 'Non-stationary');
      if(parts.length) metaHtml = '<div class="meta">'+parts.join(' · ')+'</div>';
    }

    tm.innerHTML = data.response + verdictHtml + metaHtml;
  } catch(err) {
    tm.innerHTML = '<span style="color:#ff3d50">Error: '+err.message+'</span>';
  }
  btn.disabled = false;
  msgs.scrollTop = msgs.scrollHeight;
}
</script>
</body>
</html>"""


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
