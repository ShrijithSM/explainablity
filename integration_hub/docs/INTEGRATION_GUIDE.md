# Chronoscope — UI Integration Guide
## Attaching `chronoscope_live.html` to the Python codebase

**Document type:** Coding-agent implementation brief  
**Target file:** `chronoscope_live.html` — the live, mock-free dashboard  
**Python codebase:** `chronoscope/` — `interceptor.py`, `observer.py`, `analyzer.py`, `synthesizer.py`, `config.py`, `cot_segmenter.py`  
**Single integration point:** `window.ChronoscopeBridge.push(frame)` — one JSON object per token step  
**Transport options:** WebSocket (default) · file poll · direct JS injection  

---

## 0. How the dashboard consumes data

The dashboard has **no internal data generation**. Every panel stays in a waiting-spinner state until it receives a real value through the bridge. The entire pipeline is:

```
Python codebase → DashboardBridge.push(frame) → JSON over transport → ChronoscopeBridge.push(frame) → processFrame(f) → individual render functions
```

`processFrame` guards every field with `if(f.field != null)` — partial frames work. You do not need to populate every field on every token step. Send what you have; the dashboard renders what arrives and holds everything else in its waiting state.

---

## 1. Install the transport dependency

```bash
pip install websockets>=12.0   # for WebSocket transport (recommended)
# OR — no extra deps needed for file-poll or webview injection
```

---

## 2. Create `chronoscope/dashboard_bridge.py`

This is the **only new file** the codebase needs. It is the single class that receives data from all four existing modules and serialises it into frames for the dashboard.

```python
# chronoscope/dashboard_bridge.py
"""
DashboardBridge — serialises Chronoscope analysis results into JSON frames
and pushes them to chronoscope_live.html via WebSocket, file poll, or JS injection.

Usage (from synthesizer.py or main experiment loop):

    from chronoscope.dashboard_bridge import DashboardBridge
    bridge = DashboardBridge(transport='websocket')   # or 'file' or 'inject'
    bridge.start()

    # On each token:
    bridge.push_token_frame(token_idx, interceptor, observer, config)

    # After VAR:
    bridge.push_var_frame(analyzer_result, config)

    # After perturbation:
    bridge.push_perturbation_frame(pert_results, mediation_results)

    # After HMM:
    bridge.push_hmm_frame(hmm_result, config)

    # After TDA:
    bridge.push_tda_frame(tda_result)

    # Final score:
    bridge.push_score_frame(composite)
"""

from __future__ import annotations
import json
import asyncio
import threading
import time
import pathlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

# ── colour palette for HMM phases ────────────────────────────────────────────
_PHASE_COLOURS = ['#00d4ff', '#00ff88', '#ffb300', '#aa66ff',
                  '#ff3d50', '#00ffcc', '#ff8844', '#44aaff']


def _phase_colour(idx: int) -> str:
    return _PHASE_COLOURS[idx % len(_PHASE_COLOURS)]


# ── JSON serialiser — handles numpy arrays and NaN ────────────────────────────
class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Replace NaN with None so JSON.parse doesn't choke
            return [None if (isinstance(v, float) and np.isnan(v)) else v
                    for v in obj.tolist()]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, cls=_Encoder, allow_nan=False)


# ── Main bridge class ─────────────────────────────────────────────────────────
class DashboardBridge:
    """
    Serialises Chronoscope analysis output into frames consumed by
    chronoscope_live.html.

    Args:
        transport:  'websocket' | 'file' | 'inject'
        ws_host:    WebSocket host (default 'localhost')
        ws_port:    WebSocket port (default 8765)
        poll_path:  Path for file-poll transport (default '/tmp/chronoscope_frame.json')
        inject_fn:  Callable(js_string) for webview injection transport
    """

    def __init__(
        self,
        transport:  str = 'websocket',
        ws_host:    str = 'localhost',
        ws_port:    int = 8765,
        poll_path:  str = '/tmp/chronoscope_frame.json',
        inject_fn=None,
    ):
        self.transport  = transport
        self.ws_host    = ws_host
        self.ws_port    = ws_port
        self.poll_path  = pathlib.Path(poll_path)
        self.inject_fn  = inject_fn
        self._ws_clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._last_frame: dict = {}   # cumulative — each push merges into this

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> 'DashboardBridge':
        """Start transport. Call once before any push_*() calls."""
        if self.transport == 'websocket':
            self._start_websocket_server()
        return self

    def stop(self):
        """Cleanly shut down the WebSocket server if running."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Public push methods — one per analysis stage ───────────────────────────

    def push_token_frame(
        self,
        token_idx:   int,
        interceptor,          # chronoscope.interceptor.Interceptor instance
        observer,             # chronoscope.observer.Observer instance
        config,               # chronoscope.config.Config instance
        log_events: list[dict] | None = None,
    ):
        """
        Called once per generated token.
        Populates: identity, entropy_row, hurst, tau, velocity, adf_pval_pc0,
                   cot_time_axis, head_metric, sink_removed, pert_mode, feat_mode.

        Source mappings:
            entropy_row   ← interceptor.get_head_metric_series(layer_name)[-1]
            hurst         ← observer.hurst_exponent
            tau           ← observer.tau_normalised
            velocity      ← observer.arc_steps[-1]
            adf_pval_pc0  ← observer.adf_pval_pc0  (updated each token if cheap)
        """
        layer_name = f'model.layers.{config.target_layer}.self_attn'

        # ── entropy_row: shape [H] for current token step ─────────────────────
        metric_series = interceptor.get_head_metric_series(layer_name)
        # metric_series shape: [T, H] — take last row
        entropy_row = metric_series[-1].tolist() if metric_series is not None and len(metric_series) > 0 else None

        # ── arc-length intrinsic time from observer ────────────────────────────
        tau      = float(observer.tau_normalised)      if hasattr(observer, 'tau_normalised')      else None
        velocity = float(observer.arc_steps[-1])       if hasattr(observer, 'arc_steps') and len(observer.arc_steps) > 0 else None
        hurst    = float(observer.hurst_exponent)      if hasattr(observer, 'hurst_exponent')      else None
        adf_pc0  = float(observer.adf_pval_pc0)        if hasattr(observer, 'adf_pval_pc0')        else None

        # ── cot time axis ──────────────────────────────────────────────────────
        cot_axis   = 'step-level (D.1)' if getattr(config, 'use_cot_time_axis', False) else 'token-level'
        cot_steps  = getattr(observer, 'n_cot_steps', None)

        frame: dict = {
            # ── identity (only needed on first frame, then cached by dashboard) ─
            'model_name':   getattr(config, 'model_name', 'Qwen/Qwen2.5-0.5B'),
            'layer_idx':    getattr(config, 'target_layer', 23),
            'n_heads':      getattr(config, 'n_heads', 14),
            'hidden_dim':   getattr(config, 'hidden_dim', 896),
            'total_tokens': getattr(config, 'total_tokens', 100),
            'head_metric':  getattr(config, 'head_metric', 'shannon_entropy'),
            'feat_mode':    getattr(config, 'head_feature_mode', 'scalar'),
            'sink_removed': getattr(config, 'remove_attention_sink', True),
            'pert_mode':    getattr(config, 'perturbation_mode', 'zero'),
            # ── per-token data ─────────────────────────────────────────────────
            'token_idx':    token_idx,
            'entropy_row':  entropy_row,
            'hurst':        hurst,
            'tau':          tau,
            'velocity':     velocity,
            'adf_pval_pc0': adf_pc0,
            'cot_time_axis': cot_axis,
            'cot_n_steps':  cot_steps,
            'log_events':   log_events or [],
        }
        self._send(frame)

    def push_var_frame(
        self,
        analyzer,            # chronoscope.analyzer.Analyzer instance
        config,
    ):
        """
        Called once after VAR (or VECM) is fitted on the full [T, H] series.
        Populates: stationarity, influence_matrix, fdr_reject, pval_matrix,
                   sig_pairs, significant_pairs, pdc_low, pdc_high,
                   selected_lag, coint_rank, vecm_used.

        Source mappings:
            stat_per_head    ← analyzer._last_joint_stationarity['per_head']
            influence_matrix ← analyzer._last_influence_matrix   shape [H, H]
            fdr_reject       ← analyzer._last_fdr_result['reject_matrix']
            pval_matrix      ← analyzer._last_fdr_result['pval_corrected']
            sig_pairs        ← analyzer._last_fdr_result['n_significant']
            significant_pairs← analyzer._last_fdr_result['significant_pairs']
            pdc_low          ← analyzer._last_pdc['pdc_low']   shape [H, H] → mean over target dim
            pdc_high         ← analyzer._last_pdc['pdc_high']  shape [H, H] → mean over target dim
            selected_lag     ← analyzer._selected_lag
            coint_rank       ← analyzer._last_coint['n_coint_vectors'] if exists
            vecm_used        ← analyzer._vecm_used
        """
        # ── stationarity ──────────────────────────────────────────────────────
        joint_stat  = getattr(analyzer, '_last_joint_stationarity', None)
        stat_per_head = None
        if joint_stat and 'per_head' in joint_stat:
            stat_per_head = [
                {
                    'h':          r['head'],
                    'adf_pval':   float(r['adf_pval']),
                    'kpss_pval':  float(r['kpss_pval']),
                    'diagnosis':  r['diagnosis'],
                }
                for r in joint_stat['per_head']
            ]

        # ── VAR/VECM outputs ──────────────────────────────────────────────────
        inf_mat  = getattr(analyzer, '_last_influence_matrix', None)
        fdr_res  = getattr(analyzer, '_last_fdr_result', None)
        pdc_res  = getattr(analyzer, '_last_pdc', None)
        lag      = getattr(analyzer, '_selected_lag', None)
        coint    = getattr(analyzer, '_last_coint', {})
        vecm_used= getattr(analyzer, '_vecm_used', False)

        influence_matrix = inf_mat.tolist() if inf_mat is not None else None
        fdr_reject       = fdr_res['reject_matrix'].tolist()  if fdr_res else None
        pval_matrix_raw  = fdr_res['pval_corrected']          if fdr_res else None
        pval_matrix      = pval_matrix_raw.tolist()           if pval_matrix_raw is not None else None
        sig_pairs_count  = int(fdr_res['n_significant'])      if fdr_res else None

        # significant_pairs: list of {source, target, score, pval, fdr_sig}
        sig_pairs_list = None
        if fdr_res and 'significant_pairs' in fdr_res:
            sig_pairs_list = [
                {
                    'source':  int(j),
                    'target':  int(i),
                    'score':   float(inf_mat[i, j]) if inf_mat is not None else 0.0,
                    'pval':    float(pval) if pval is not None else None,
                    'fdr_sig': True,
                }
                for j, i, pval in (fdr_res['significant_pairs'] or [])
            ]

        # pdc: [H, H] matrices — collapse to [H] by averaging over target axis
        # giving "how much does each source head contribute on average"
        pdc_low  = None
        pdc_high = None
        if pdc_res:
            if 'pdc_low'  in pdc_res and pdc_res['pdc_low']  is not None:
                pdc_low  = pdc_res['pdc_low'].mean(axis=0).tolist()   # [H] avg over targets
            if 'pdc_high' in pdc_res and pdc_res['pdc_high'] is not None:
                pdc_high = pdc_res['pdc_high'].mean(axis=0).tolist()

        frame: dict = {
            'stat_per_head':     stat_per_head,
            'coint_rank':        int(coint.get('n_coint_vectors', 0)),
            'vecm_used':         bool(vecm_used),
            'selected_lag':      int(lag) if lag is not None else None,
            'influence_matrix':  influence_matrix,
            'fdr_reject':        fdr_reject,
            'pval_matrix':       pval_matrix,
            'sig_pairs':         sig_pairs_count,
            'significant_pairs': sig_pairs_list,
            'pdc_low':           pdc_low,
            'pdc_high':          pdc_high,
            'log_events': [
                {'type': 'gap', 'msg': f'Gap A: Johansen rank={coint.get("n_coint_vectors",0)} · {"VECM" if vecm_used else "VAR"} path · lag={lag}'},
                {'type': 'sig', 'msg': f'Gap B: BH-FDR → {sig_pairs_count} significant pairs / {influence_matrix and len(influence_matrix)**2 - len(influence_matrix)} tested'},
            ],
        }
        self._send(frame)

    def push_perturbation_frame(
        self,
        pert_results: list[dict],
        mediation_results: list[dict] | None = None,
    ):
        """
        Called after perturbation experiments complete (C.1/C.2/C.3/C.4).

        Each pert_result dict must contain:
            head          int    — ablated head index
            target        int    — target head being measured
            mode          str    — 'zero' | 'mean' | 'gaussian'
            delta_entropy float  — entropy change in target after ablation
            restoration   float  — activation patching restoration score [0,1]
            kl_patch      float  — KL divergence of patched vs corrupted
            confirmed     bool   — whether restoration > 0.5

        Each mediation_result dict must contain:
            source         int
            target         int
            mediator       int
            total_effect   float
            direct_effect  float
            indirect_effect float
            ratio          float   — indirect / total
        """
        frame: dict = {
            'perturbation_results': [
                {
                    'head':          int(p['head']),
                    'target':        int(p.get('target', -1)),
                    'mode':          str(p.get('mode', 'zero')),
                    'delta_entropy': float(p.get('delta_entropy', 0.0)),
                    'restoration':   float(p.get('restoration', 0.0)),
                    'kl_patch':      float(p.get('kl_patch', 0.0)),
                    'confirmed':     bool(p.get('confirmed', False)),
                }
                for p in (pert_results or [])
            ],
            'mediation_results': [
                {
                    'source':          int(m['source']),
                    'target':          int(m['target']),
                    'mediator':        int(m['mediator']),
                    'total_effect':    float(m.get('total_effect', 0.0)),
                    'direct_effect':   float(m.get('direct_effect', 0.0)),
                    'indirect_effect': float(m.get('indirect_effect', 0.0)),
                    'ratio':           float(m.get('ratio', 0.0)),
                }
                for m in (mediation_results or [])
            ] if mediation_results else None,
            'log_events': [
                {'type': 'pert', 'msg': f'C.1/C.2: {len(pert_results or [])} ablation results received'},
            ],
        }
        self._send(frame)

    def push_hmm_frame(
        self,
        hmm_result: dict,
        config,
    ):
        """
        Called after HMM phase discovery completes (D.4).

        hmm_result keys (from analyzer._discover_phases_hmm()):
            state_seq      np.ndarray [T]      — Viterbi decoded state per token
            n_states       int                 — number of states fitted
            means          np.ndarray [S, H]   — mean entropy per state per head
            bic            float
            log_likelihood float
            trans_matrix   np.ndarray [S, S]

        Phase colours are automatically assigned from the built-in palette.
        """
        state_seq   = hmm_result.get('state_seq')       # [T] int array
        n_states    = int(hmm_result.get('n_states', getattr(config, 'hmm_n_states', 4)))
        means       = hmm_result.get('means')            # [S, H]
        trans       = hmm_result.get('trans_matrix')     # [S, S]
        bic         = hmm_result.get('bic')
        ll          = hmm_result.get('log_likelihood')

        # Build per-state metadata — label, colour, token range, mean entropy
        hmm_states = []
        if state_seq is not None:
            seq_np = np.asarray(state_seq)
            for s in range(n_states):
                mask = seq_np == s
                tokens_in_state = np.where(mask)[0]
                token_range = (
                    f'{tokens_in_state[0]}-{tokens_in_state[-1]}'
                    if len(tokens_in_state) > 0 else '–'
                )
                mean_h = float(means[s].mean()) if means is not None else 0.0
                hmm_states.append({
                    'id':           s,
                    'label':        _state_label(s, mean_h),
                    'color':        _phase_colour(s),
                    'token_range':  token_range,
                    'mean_entropy': round(mean_h, 3),
                })

        frame: dict = {
            'hmm_n_states':  n_states,
            'hmm_states':    hmm_states,
            'hmm_state_seq': state_seq.tolist() if state_seq is not None else None,
            'hmm_trans':     trans.tolist()     if trans is not None else None,
            'hmm_bic':       float(bic)         if bic is not None else None,
            'hmm_ll':        float(ll)          if ll is not None else None,
            'log_events': [
                {'type': 'tda', 'msg': f'D.4: HMM fitted · n_states={n_states} · BIC={bic:.1f if bic else "–"}'},
            ],
        }
        self._send(frame)

    def push_tda_frame(
        self,
        tda_result: dict,
        current_token: int,
        phase_boundaries: list[int] | None = None,
        current_phase_idx: int | None = None,
    ):
        """
        Called after each TDA sliding-window computation.

        tda_result keys (from observer.py TDA computation):
            betti0         int     — β₀ connected components
            betti1         int     — β₁ loops
            euler          int     — χ = β₀ − β₁
            ec_series      list[float]   — last 60 EC values (sliding window)
            anomalies      list[dict]    — { token_idx, severity, description }
        """
        frame: dict = {
            'betti0':            tda_result.get('betti0'),
            'betti1':            tda_result.get('betti1'),
            'euler':             tda_result.get('euler'),
            'ec_series':         tda_result.get('ec_series'),
            'topo_anomalies':    tda_result.get('anomalies'),
            'phase_boundaries':  phase_boundaries,
            'current_phase_idx': current_phase_idx,
        }

        # Log anomaly if spike detected at current token
        anomalies = tda_result.get('anomalies') or []
        new_anom  = [a for a in anomalies if a.get('token_idx') == current_token]
        log_evs   = []
        for a in new_anom:
            log_evs.append({
                'type': 'tda',
                'msg':  f'D.2: EC spike t={current_token} · {a["severity"]} · {a["description"]}',
            })
        if log_evs:
            frame['log_events'] = log_evs

        self._send(frame)

    def push_signal_quality_frame(
        self,
        interceptor,
        config,
    ):
        """
        Called after each token to push Gap E per-head signal quality metrics.

        Reads from interceptor._head_metrics which stores [T, H] arrays.
        Computes latest snapshot for each head.
        """
        layer_name = f'model.layers.{config.target_layer}.self_attn'
        sq_data = []

        # interceptor._head_metrics[name] → list of [H] tensors per token
        metrics = getattr(interceptor, '_head_metrics', {}).get(layer_name, [])
        if metrics:
            last = metrics[-1]  # most recent token, shape [H] or [H, 5]
            H    = getattr(config, 'n_heads', 14)

            # Depending on head_feature_mode: scalar ([H]) or vector ([H, 5])
            feat_mode = getattr(config, 'head_feature_mode', 'scalar')
            for h in range(H):
                if feat_mode == 'vector' and hasattr(last, 'shape') and last.ndim == 2:
                    # vector mode: columns are [shannon, renyi, effrank, sink, maxattn]
                    row = last[h] if hasattr(last, '__getitem__') else [0]*5
                    sq_data.append({
                        'h':        h,
                        'shannon':  float(row[0]),
                        'renyi':    float(row[1]),
                        'effrank':  float(row[2]),
                        'sink':     float(row[3]),
                        'maxattn':  float(row[4]),
                        'ov_norm':  None,   # populated separately by push_ov_frame if available
                    })
                else:
                    # scalar mode: last is [H] entropy values
                    val = float(last[h]) if hasattr(last, '__getitem__') else 0.0
                    sq_data.append({
                        'h':        h,
                        'shannon':  val,
                        'renyi':    None,
                        'effrank':  None,
                        'sink':     None,
                        'maxattn':  None,
                        'ov_norm':  None,
                    })

        if sq_data:
            self._send({'signal_quality': sq_data})

    def push_score_frame(
        self,
        composite: dict,
        interpretation: dict | None = None,
    ):
        """
        Called after synthesizer computes the composite validity score.

        composite keys:
            score              float  — 0–100
            dtw_sensitivity    float  — 0–1
            spectral_coherence float  — 0–1
            topo_smoothness    float  — 0–1
            active_reasoning   float  — 0–1
            fdr_sig_pairs      int    — count (converted to 0–1 internally)
            te_score           float  — 0–1 or None
            verdict            str    — 'STRONG REASONING' | 'MODERATE REASONING' | 'HALLUCINATION RISK'

        interpretation (optional):
            source  str  — e.g. 'Gap B — FDR correction'
            text    str  — HTML-safe explanation string (may contain <span> tags)
        """
        n_heads = composite.get('n_heads', 14)
        total_pairs = n_heads * (n_heads - 1)
        fdr_pairs   = composite.get('fdr_sig_pairs', 0)

        frame: dict = {
            'composite_score':    round(float(composite.get('score', 0))),
            'dtw_sensitivity':    _safe_float(composite.get('dtw_sensitivity')),
            'spectral_coherence': _safe_float(composite.get('spectral_coherence')),
            'topo_smoothness':    _safe_float(composite.get('topo_smoothness')),
            'active_reasoning':   _safe_float(composite.get('active_reasoning')),
            'te_score':           _safe_float(composite.get('te_score')),
            'sig_pairs':          int(fdr_pairs),
            'verdict':            composite.get('verdict', 'MODERATE REASONING'),
            'log_events': [
                {
                    'type': 'ok',
                    'msg':  f'Composite score: {round(composite.get("score",0))} → {composite.get("verdict","–")}',
                },
            ],
        }
        if interpretation:
            frame['interpretation'] = interpretation
        self._send(frame)

    def push_log(self, type_: str, msg: str):
        """Send a standalone log event (does not merge with last frame)."""
        self._send({'log_events': [{'type': type_, 'msg': msg}]})

    # ── Transport internals ────────────────────────────────────────────────────

    def _send(self, partial_frame: dict):
        """Merge partial frame into cumulative state and dispatch."""
        self._last_frame.update(partial_frame)
        payload = _dumps(partial_frame)

        if self.transport == 'websocket':
            self._ws_broadcast(payload)
        elif self.transport == 'file':
            self._write_file(partial_frame)
        elif self.transport == 'inject':
            if self.inject_fn:
                self.inject_fn(f'window.ChronoscopeBridge.push({payload})')

    def _write_file(self, frame: dict):
        frame['_mtime'] = time.time()
        self.poll_path.parent.mkdir(parents=True, exist_ok=True)
        self.poll_path.write_text(_dumps(frame))

    def _ws_broadcast(self, payload: str):
        if not self._loop or not self._ws_clients:
            return
        async def _do():
            dead = set()
            for ws in list(self._ws_clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            self._ws_clients -= dead
        asyncio.run_coroutine_threadsafe(_do(), self._loop)

    def _start_websocket_server(self):
        """Launch the WebSocket server in a background daemon thread."""
        import websockets

        async def _handler(ws):
            self._ws_clients.add(ws)
            # On fresh connection send the full cumulative frame so the
            # dashboard can render current state immediately
            if self._last_frame:
                try:
                    await ws.send(_dumps(self._last_frame))
                except Exception:
                    pass
            try:
                await ws.wait_closed()
            finally:
                self._ws_clients.discard(ws)

        async def _server():
            async with websockets.serve(_handler, self.ws_host, self.ws_port):
                await asyncio.Future()   # run forever

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(_server())

        self._thread = threading.Thread(target=_run, daemon=True, name='chronoscope-ws')
        self._thread.start()
        # Give the server a moment to bind
        time.sleep(0.1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f   # NaN check without math import
    except (TypeError, ValueError):
        return None


def _state_label(state_id: int, mean_entropy: float) -> str:
    """
    Derive a human-readable phase label from state index and mean entropy.
    Mirrors the semantic labels used in the dashboard's HMM panel.
    """
    labels = {0: 'PROMPT·PROC', 1: 'SETUP', 2: 'ACTIVE·CALC', 3: 'CONCLUSION'}
    if state_id in labels:
        return labels[state_id]
    return f'PHASE·{state_id}' + ('·HI' if mean_entropy > 2.5 else '·LO')
```

---

## 3. Add `n_heads`, `hidden_dim`, `model_name` to `config.py`

The dashboard header reads these from the first frame. Add them as config fields:

```python
# chronoscope/config.py — ADD these fields (already have target_layer, etc.)

model_name:   str = 'Qwen/Qwen2.5-0.5B'
n_heads:      int = 14        # model.num_heads
hidden_dim:   int = 896       # model.hidden_size
total_tokens: int = 100       # expected generation length
target_layer: int = 23        # layer index being analysed (deepest)
```

---

## 4. Add `_last_influence_matrix` and `_last_fdr_result` to `analyzer.py`

The bridge reads these attributes from the analyzer instance. Add assignments at the points where they are computed:

```python
# chronoscope/analyzer.py — inside head_interaction_analysis()

# After building influence matrix (VAR or VECM path):
self._last_influence_matrix = influence   # np.ndarray [H, H]
self._vecm_used             = USE_VECM

# After FDR correction (B.2):
fdr_result                  = self._apply_fdr_correction(pval_matrix)
self._last_fdr_result       = fdr_result   # dict with reject_matrix, pval_corrected, etc.

# After PDC computation (B.5):
self._last_pdc              = self._partial_directed_coherence(var_result)

# After AIC lag selection (A.4):
self._selected_lag          = result.k_ar

# After joint stationarity test (A.5):
self._last_joint_stationarity = self._joint_stationarity_test(series_np)

# After Johansen test (A.3):
self._last_coint             = coint_report   # dict with n_coint_vectors
```

---

## 5. Add `tau_normalised`, `arc_steps`, `hurst_exponent`, `adf_pval_pc0` to `observer.py`

The bridge reads these as attributes. The computation already exists in `compute_intrinsic_time()` — just store the results:

```python
# chronoscope/observer.py — inside compute_intrinsic_time() and after ADF

# After arc-length computation:
self.arc_steps       = arc_steps_np        # np.ndarray [T-1]
self.tau_normalised  = float(tau_norm[-1]) # scalar, current normalised arc-length

# After Hurst computation (already computed in existing code):
self.hurst_exponent  = float(hurst)

# After ADF on PC0 (already computed):
self.adf_pval_pc0    = float(adf_result[1])
```

---

## 6. Wire the bridge into `synthesizer.py`

Add the bridge as an optional dependency — the synthesizer already orchestrates all analysis stages, making it the natural place to call all `push_*` methods.

```python
# chronoscope/synthesizer.py

from chronoscope.dashboard_bridge import DashboardBridge

class Synthesizer:
    def __init__(self, config, interceptor, observer, analyzer,
                 dashboard: DashboardBridge | None = None):
        self.config      = config
        self.interceptor = interceptor
        self.observer    = observer
        self.analyzer    = analyzer
        self.dashboard   = dashboard   # None = dashboard disabled

    # ── Called inside the token generation loop ───────────────────────────────

    def on_token(self, token_idx: int):
        """
        Call this once per generated token from the experiment loop.
        Pushes the token frame and signal quality to the dashboard.
        """
        if self.dashboard is None:
            return

        self.dashboard.push_token_frame(
            token_idx   = token_idx,
            interceptor = self.interceptor,
            observer    = self.observer,
            config      = self.config,
            log_events  = [{'type': 'ok', 'msg': f'Token {token_idx} processed · layer {self.config.target_layer}'}],
        )
        # Signal quality pushed every token (E·SIGNAL tab)
        self.dashboard.push_signal_quality_frame(self.interceptor, self.config)

    # ── Called after generation completes ────────────────────────────────────

    def on_analysis_complete(self):
        """
        Call after all tokens have been generated and VAR analysis is done.
        Pushes VAR, perturbation, HMM, TDA, and composite score frames.
        """
        if self.dashboard is None:
            return

        # VAR + stationarity + PDC (Gap A + B)
        self.dashboard.push_var_frame(self.analyzer, self.config)
        self.dashboard.push_log('ok', 'VAR analysis complete — pushing to dashboard')

        # Perturbation (Gap C) — if results are available
        pert_results    = getattr(self.analyzer, '_last_pert_results', None)
        mediation       = getattr(self.analyzer, '_last_mediation_results', None)
        if pert_results:
            self.dashboard.push_perturbation_frame(pert_results, mediation)

        # HMM (Gap D.4) — if configured and fitted
        hmm_result = getattr(self.analyzer, '_last_hmm_result', None)
        if hmm_result and getattr(self.config, 'use_hmm_phase_discovery', False):
            self.dashboard.push_hmm_frame(hmm_result, self.config)

        # TDA (Gap D.2) — push accumulated TDA state
        tda_result = getattr(self.observer, '_last_tda_result', None)
        if tda_result:
            phase_boundaries  = getattr(self.observer, 'phase_boundaries', None)
            current_phase_idx = getattr(self.observer, 'current_phase_idx', None)
            self.dashboard.push_tda_frame(
                tda_result        = tda_result,
                current_token     = getattr(self.config, 'total_tokens', 100) - 1,
                phase_boundaries  = phase_boundaries,
                current_phase_idx = current_phase_idx,
            )

        # Composite score
        composite = self._compute_composite_score()
        interp    = self._generate_interpretation(composite)
        self.dashboard.push_score_frame(composite, interp)

    def _compute_composite_score(self) -> dict:
        """
        Build the composite score dict from all analysis results.
        Replace stub values with real computations from your synthesizer.
        """
        fdr_res  = getattr(self.analyzer, '_last_fdr_result', {})
        n_heads  = self.config.n_heads
        n_pairs  = n_heads * (n_heads - 1)
        n_sig    = int(fdr_res.get('n_significant', 0)) if fdr_res else 0

        # Active reasoning: derived from ADF non-stationarity
        adf_pval = getattr(self.observer, 'adf_pval_pc0', 0.5)
        active   = min(float(adf_pval) * 1.5, 1.0)   # higher p = more non-stationary = more active

        # Topo smoothness: inverse of EC variance
        ec_series = getattr(self.observer, '_last_tda_result', {}).get('ec_series', [])
        topo_smooth = 0.5
        if ec_series and len(ec_series) > 1:
            ec_var = float(np.var(ec_series))
            topo_smooth = float(np.clip(1.0 / (1.0 + ec_var * 0.1), 0, 1))

        # FDR sig pairs normalised to [0,1]
        fdr_norm  = min(n_sig / max(n_pairs * 0.05, 1), 1.0)

        # Hurst contribution
        hurst = getattr(self.observer, 'hurst_exponent', 0.5)
        hurst_score = float(np.clip((hurst - 0.5) * 2, 0, 1))

        # Composite: weighted average (weights sum to 1)
        score = int(round((
            0.30 * hurst_score   +
            0.20 * active        +
            0.20 * topo_smooth   +
            0.15 * fdr_norm      +
            0.15 * 0.65          # placeholder for DTW + spectral until implemented
        ) * 100))

        verdict = ('STRONG REASONING'   if score > 70 else
                   'MODERATE REASONING' if score > 50 else
                   'HALLUCINATION RISK')

        return {
            'score':              score,
            'dtw_sensitivity':    None,   # implement when DTW is ready
            'spectral_coherence': None,
            'topo_smoothness':    topo_smooth,
            'active_reasoning':   active,
            'fdr_sig_pairs':      n_sig,
            'te_score':           None,
            'verdict':            verdict,
            'n_heads':            n_heads,
        }

    def _generate_interpretation(self, composite: dict) -> dict:
        """
        Pick the most informative interpretation for the current analysis state.
        Returns a {source, text} dict for the dashboard's interpretation panel.
        """
        score   = composite.get('score', 0)
        verdict = composite.get('verdict', '')
        n_sig   = composite.get('fdr_sig_pairs', 0)
        hurst   = getattr(self.observer, 'hurst_exponent', 0.5)
        coint   = getattr(self.analyzer, '_last_coint', {})
        rank    = coint.get('n_coint_vectors', 0)

        if rank > 0:
            return {
                'source': 'Gap A — Johansen',
                'text':   f'Cointegration rank={rank} — VECM fitted. '
                          f'Long-run equilibrium preserved in the head entropy series. '
                          f'Short-run γ coefficients form the influence matrix.',
            }
        if n_sig == 0:
            return {
                'source': 'Gap B — FDR correction',
                'text':   'BH-FDR correction at α=0.05 eliminated all pairs. '
                          'No Granger-causal links are statistically significant. '
                          'Consider running more tokens or checking for unit-root heads.',
            }
        if hurst < 0.5:
            return {
                'source': 'Gap A — Hurst exponent',
                'text':   f'H={hurst:.3f} — anti-persistent (mean-reverting). '
                          f'The residual stream is oscillating rather than trending. '
                          f'This is associated with hallucination or formulaic generation.',
            }
        return {
            'source': f'Score: {score}/100',
            'text':   f'{verdict}. {n_sig} FDR-significant causal links detected. '
                      f'Hurst H={hurst:.3f} — {"persistent logical chain" if hurst > 0.65 else "mild persistence"}.',
        }
```

---

## 7. Attach the bridge in experiment entry points

Add three lines to each experiment runner that should display the dashboard:

```python
# chronoscope/experiments/exp6.py  (and any other exp*.py)

from chronoscope.dashboard_bridge import DashboardBridge

def run_exp6(config, model, tokenizer):
    # ── existing setup ──────────────────────────────────────────────────────
    interceptor = Interceptor(model, tokenizer, config)
    observer    = Observer(config)
    analyzer    = Analyzer(config)

    # ── ADD: create and start bridge ─────────────────────────────────────────
    bridge = DashboardBridge(transport='websocket', ws_port=8765)
    bridge.start()

    synthesizer = Synthesizer(config, interceptor, observer, analyzer,
                              dashboard=bridge)   # pass bridge here

    bridge.push_log('ok', f'Exp6 starting · model={config.model_name}')

    # ── existing generation loop ─────────────────────────────────────────────
    inputs = tokenizer(config.prompt, return_tensors='pt').to(config.device)
    with torch.no_grad():
        for token_idx in range(config.total_tokens):
            # ... existing per-token logic ...
            observer.update(hidden_states)
            synthesizer.on_token(token_idx)      # ADD: push per-token frame

    # ── existing post-generation analysis ────────────────────────────────────
    analyzer.head_interaction_analysis(interceptor)
    # ... existing perturbation + TDA calls ...

    synthesizer.on_analysis_complete()            # ADD: push all post-generation frames

    bridge.push_log('ok', 'Exp6 complete — dashboard updated')
    bridge.stop()
```

---

## 8. Handle the `chronoscope:metric_change` event (optional but recommended)

When the user switches metric tabs (Shannon → Rényi → Effective Rank etc.) in the dashboard, the browser fires:

```javascript
window.dispatchEvent(new CustomEvent('chronoscope:metric_change', {detail:{metric:'renyi'}}))
```

If you are using a webview (Qt, Electron), intercept this and change `config.head_metric` so the next frames use the selected metric:

```python
# In your webview event handler (Qt example):
def on_js_event(self, event_type, detail):
    if event_type == 'chronoscope:metric_change':
        new_metric = detail.get('metric')
        metric_map = {
            'shannon':  'shannon_entropy',
            'renyi':    'renyi_entropy_2',
            'effrank':  'effective_rank',
            'sink':     'sink_fraction',
            'maxattn':  'max_attention',
        }
        if new_metric in metric_map:
            self.config.head_metric = metric_map[new_metric]
            # interceptor will use new metric on next token hook fire
```

For WebSocket transport this is a future enhancement — the WebSocket is currently one-directional (Python → dashboard). To support back-channel metric switching, extend the WebSocket handler to accept incoming messages:

```python
# chronoscope/dashboard_bridge.py — extend _handler() for back-channel:

async def _handler(ws):
    self._ws_clients.add(ws)
    if self._last_frame:
        await ws.send(_dumps(self._last_frame))
    try:
        async for message in ws:
            try:
                cmd = json.loads(message)
                if cmd.get('type') == 'metric_change' and self._on_metric_change:
                    self._on_metric_change(cmd.get('metric'))
            except Exception:
                pass
    finally:
        self._ws_clients.discard(ws)
```

---

## 9. Serve the dashboard HTML

The dashboard HTML must be served over HTTP (not `file://`) for WebSocket to work reliably across browsers. Use Python's built-in server in a background thread:

```python
# chronoscope/dashboard_bridge.py — add to DashboardBridge.start()

import http.server
import threading
import pathlib

def _serve_dashboard(html_path: str = 'chronoscope_live.html', port: int = 8766):
    """
    Serve the dashboard HTML over HTTP on localhost:8766.
    Open http://localhost:8766 in any browser to view the live dashboard.
    """
    html = pathlib.Path(html_path).read_text()
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        def log_message(self, fmt, *args): pass   # silence access log

    server = http.server.HTTPServer(('localhost', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f'[CHRONOSCOPE] Dashboard: http://localhost:{port}')
    return server
```

Call in your experiment runner:

```python
bridge = DashboardBridge(transport='websocket')
bridge.start()
_serve_dashboard('chronoscope_live.html', port=8766)
# → Open http://localhost:8766 in browser
# → Dashboard auto-connects to ws://localhost:8765
```

---

## 10. Full integration checklist

```
[ ] 1.  pip install websockets>=12.0

[ ] 2.  Create chronoscope/dashboard_bridge.py  (full code in Section 2)

[ ] 3.  Add to config.py:
          model_name, n_heads, hidden_dim, total_tokens, target_layer

[ ] 4.  Add to analyzer.py after each computation:
          self._last_influence_matrix
          self._last_fdr_result
          self._last_pdc
          self._selected_lag
          self._last_joint_stationarity
          self._last_coint
          self._vecm_used
          self._last_pert_results       (after Gap C)
          self._last_mediation_results  (after Gap C.4)
          self._last_hmm_result         (after Gap D.4)

[ ] 5.  Add to observer.py:
          self.arc_steps
          self.tau_normalised
          self.hurst_exponent
          self.adf_pval_pc0
          self._last_tda_result         (dict with betti0, betti1, euler, ec_series, anomalies)
          self.phase_boundaries         (list of EC-spike token indices)
          self.current_phase_idx        (current HMM phase)

[ ] 6.  Modify synthesizer.py:
          __init__: accept dashboard: DashboardBridge | None = None
          on_token(token_idx): push token frame + signal quality
          on_analysis_complete(): push VAR, pert, HMM, TDA, score frames

[ ] 7.  In each experiment file:
          bridge = DashboardBridge(transport='websocket')
          bridge.start()
          synthesizer = Synthesizer(..., dashboard=bridge)
          synthesizer.on_token(t)          inside generation loop
          synthesizer.on_analysis_complete() after analysis

[ ] 8.  Serve dashboard HTML (Section 9) and open http://localhost:8766

[ ] 9.  Verify: connection banner dismisses when first frame arrives

[ ] 10. Verify: header shows model_name and correct n_heads

[ ] 11. Verify: entropy chart streams live as tokens are generated

[ ] 12. Verify: after VAR, influence matrix cells appear (non-waiting state)

[ ] 13. Verify: perturbation results appear in Gap C panel

[ ] 14. Verify: HMM state timeline renders in D.4·HMM tab

[ ] 15. Verify: composite score ring fills and verdict matches synthesizer

[ ] 16. Run test: bridge.push_log('ok', 'integration test') → appears in log strip

[ ] 17. REGRESSION: run exp1 and exp3 with dashboard=None (default) and confirm
          no behaviour change — bridge is completely optional
```

---

## 11. Frame emission schedule

The table below shows which `push_*` call emits each dashboard field and when in the pipeline it fires:

```
Token 0 → T-1 (each token):
  push_token_frame()          → model_name, n_heads, layer_idx, token_idx,
                                 entropy_row, hurst, tau, velocity, adf_pval_pc0,
                                 cot_time_axis, head_metric, sink_removed, pert_mode
  push_signal_quality_frame() → signal_quality[{h, shannon, renyi, effrank, sink, maxattn}]

After generation + TDA per token:
  push_tda_frame()            → betti0, betti1, euler, ec_series, topo_anomalies,
                                 current_phase_idx, phase_boundaries

After all T tokens (on_analysis_complete):
  push_var_frame()            → stat_per_head, coint_rank, vecm_used, selected_lag,
                                 influence_matrix, fdr_reject, pval_matrix,
                                 sig_pairs, significant_pairs, pdc_low, pdc_high
  push_perturbation_frame()   → perturbation_results, mediation_results
  push_hmm_frame()            → hmm_n_states, hmm_states, hmm_state_seq,
                                 hmm_trans, hmm_bic, hmm_ll
  push_score_frame()          → composite_score, dtw_sensitivity, spectral_coherence,
                                 topo_smoothness, active_reasoning, te_score,
                                 sig_pairs, verdict, interpretation
```

---

## 12. Dashboard field-to-source reference

Complete lookup table: every field in the JSON frame, which Python attribute provides it, and which file to modify.

| Frame field | Python source | File to modify |
|---|---|---|
| `model_name` | `config.model_name` | `config.py` |
| `layer_idx` | `config.target_layer` | `config.py` |
| `n_heads` | `config.n_heads` | `config.py` |
| `hidden_dim` | `config.hidden_dim` | `config.py` |
| `token_idx` | loop variable | experiment file |
| `head_metric` | `config.head_metric` | `config.py` |
| `feat_mode` | `config.head_feature_mode` | `config.py` |
| `sink_removed` | `config.remove_attention_sink` | `config.py` |
| `pert_mode` | `config.perturbation_mode` | `config.py` |
| `entropy_row` | `interceptor.get_head_metric_series(layer)[-1]` | `interceptor.py` |
| `hurst` | `observer.hurst_exponent` | `observer.py` |
| `tau` | `observer.tau_normalised` | `observer.py` |
| `velocity` | `observer.arc_steps[-1]` | `observer.py` |
| `adf_pval_pc0` | `observer.adf_pval_pc0` | `observer.py` |
| `stat_per_head` | `analyzer._last_joint_stationarity['per_head']` | `analyzer.py` |
| `coint_rank` | `analyzer._last_coint['n_coint_vectors']` | `analyzer.py` |
| `vecm_used` | `analyzer._vecm_used` | `analyzer.py` |
| `selected_lag` | `analyzer._selected_lag` | `analyzer.py` |
| `influence_matrix` | `analyzer._last_influence_matrix` | `analyzer.py` |
| `fdr_reject` | `analyzer._last_fdr_result['reject_matrix']` | `analyzer.py` |
| `pval_matrix` | `analyzer._last_fdr_result['pval_corrected']` | `analyzer.py` |
| `sig_pairs` | `analyzer._last_fdr_result['n_significant']` | `analyzer.py` |
| `significant_pairs` | `analyzer._last_fdr_result['significant_pairs']` | `analyzer.py` |
| `pdc_low` | `analyzer._last_pdc['pdc_low'].mean(axis=0)` | `analyzer.py` |
| `pdc_high` | `analyzer._last_pdc['pdc_high'].mean(axis=0)` | `analyzer.py` |
| `perturbation_results` | `analyzer._last_pert_results` | `analyzer.py` |
| `mediation_results` | `analyzer._last_mediation_results` | `analyzer.py` |
| `hmm_state_seq` | `analyzer._last_hmm_result['state_seq']` | `analyzer.py` |
| `hmm_states` | computed in `push_hmm_frame()` from means | `dashboard_bridge.py` |
| `hmm_trans` | `analyzer._last_hmm_result['trans_matrix']` | `analyzer.py` |
| `hmm_bic` | `analyzer._last_hmm_result['bic']` | `analyzer.py` |
| `betti0` | `observer._last_tda_result['betti0']` | `observer.py` |
| `betti1` | `observer._last_tda_result['betti1']` | `observer.py` |
| `ec_series` | `observer._last_tda_result['ec_series']` | `observer.py` |
| `topo_anomalies` | `observer._last_tda_result['anomalies']` | `observer.py` |
| `current_phase_idx` | `observer.current_phase_idx` | `observer.py` |
| `signal_quality` | `interceptor._head_metrics[layer][-1]` | `interceptor.py` |
| `composite_score` | `synthesizer._compute_composite_score()` | `synthesizer.py` |
| `verdict` | `synthesizer._compute_composite_score()` | `synthesizer.py` |
| `dtw_sensitivity` | pending implementation | `synthesizer.py` |
| `spectral_coherence` | pending implementation | `synthesizer.py` |
| `topo_smoothness` | derived from `ec_series` variance | `synthesizer.py` |
| `active_reasoning` | derived from `adf_pval_pc0` | `synthesizer.py` |
| `interpretation` | `synthesizer._generate_interpretation()` | `synthesizer.py` |
| `log_events` | inline in each `push_*` call | `dashboard_bridge.py` |

---

## 13. Files modified summary

```
NEW     chronoscope/dashboard_bridge.py   — full bridge class (Section 2)
MODIFY  chronoscope/config.py             — add model_name, n_heads, hidden_dim, total_tokens, target_layer
MODIFY  chronoscope/analyzer.py           — add _last_* attribute assignments after each computation
MODIFY  chronoscope/observer.py           — add tau_normalised, arc_steps, hurst_exponent, adf_pval_pc0, _last_tda_result
MODIFY  chronoscope/synthesizer.py        — add on_token(), on_analysis_complete(), accept dashboard arg
MODIFY  chronoscope/experiments/exp*.py   — instantiate bridge, call synthesizer.on_token() and on_analysis_complete()
NO CHANGE  chronoscope/interceptor.py     — get_head_metric_series() already returns the needed data
NO CHANGE  chronoscope/cot_segmenter.py  — results flow through observer.n_cot_steps
NO CHANGE  chronoscope_live.html         — dashboard is complete; do not modify
```

---

*Dashboard entry point: `chronoscope_live.html` · Bridge entry point: `window.ChronoscopeBridge.push(frame)` · WebSocket default: `ws://localhost:8765` · HTTP server default: `http://localhost:8766`*
