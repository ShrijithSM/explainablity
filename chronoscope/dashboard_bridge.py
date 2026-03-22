"""
Dashboard bridge for Chronoscope live UI.

This module pushes real analysis frames from Python to the browser dashboard
through WebSocket (default), file polling, or JS injection.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import time
from typing import Any, Callable, Optional

import numpy as np


_PHASE_COLOURS = [
    "#00d4ff",
    "#00ff88",
    "#ffb300",
    "#aa66ff",
    "#ff3d50",
    "#00ffcc",
    "#ff8844",
    "#44aaff",
]


def _phase_colour(idx: int) -> str:
    return _PHASE_COLOURS[idx % len(_PHASE_COLOURS)]


class _Encoder(json.JSONEncoder):
    """JSON encoder that handles numpy and torch-like objects safely."""

    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            arr = obj.tolist()
            if isinstance(arr, list):
                return _replace_nan_in_list(arr)
            return arr
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if np.isnan(val) else val
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "tolist"):
            # torch.Tensor support without importing torch globally
            return _replace_nan_in_list(obj.detach().cpu().tolist())
        return super().default(obj)


def _replace_nan_in_list(value: Any) -> Any:
    if isinstance(value, list):
        return [_replace_nan_in_list(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None


def _state_label(state_id: int, mean_entropy: float) -> str:
    labels = {
        0: "PROMPT·PROC",
        1: "SETUP",
        2: "ACTIVE·CALC",
        3: "CONCLUSION",
    }
    if state_id in labels:
        return labels[state_id]
    return f"PHASE·{state_id}" + ("·HI" if mean_entropy > 2.5 else "·LO")


class DashboardBridge:
    """Bridge Chronoscope runtime outputs into dashboard frames."""

    def __init__(
        self,
        transport: str = "websocket",
        ws_host: str = "localhost",
        ws_port: int = 8765,
        poll_path: str = "./reports/chronoscope_frame.json",
        inject_fn: Optional[Callable[[str], None]] = None,
    ):
        self.transport = transport
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.poll_path = pathlib.Path(poll_path)
        self.inject_fn = inject_fn

        self._ws_clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._last_frame: dict[str, Any] = {}

        self._http_server = None
        self._http_thread: Optional[threading.Thread] = None

    def start(self) -> "DashboardBridge":
        if self.transport == "websocket":
            self._start_websocket_server()
        return self

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._http_server:
            try:
                self._http_server.shutdown()
            except Exception:
                pass
            self._http_server = None

    def serve_dashboard(self, html_path: str, port: int = 8766) -> str:
        """Serve the dashboard HTML over localhost and return its URL."""
        import http.server

        html = pathlib.Path(html_path).read_text(encoding="utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

            def log_message(self, fmt, *args):
                return

        self._http_server = http.server.HTTPServer(("localhost", port), Handler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True,
            name="chronoscope-http",
        )
        self._http_thread.start()
        return f"http://localhost:{port}"

    def push_token_frame(self, token_idx: int, interceptor, observer, config, log_events=None, entropy_row_override=None):
        layer_idx = getattr(config, "target_layer", None)
        preferred_layer = f"layers.{layer_idx}" if layer_idx is not None else None

        entropy_row = entropy_row_override
        if preferred_layer:
            metric_series = interceptor.get_head_metric_series(preferred_layer)
            if entropy_row is None and metric_series is not None and getattr(metric_series, "numel", lambda: 0)() > 0:
                entropy_row = metric_series[-1]
        if entropy_row is None:
            summary = interceptor.debug_head_metric_summary()
            if summary:
                fallback_layer = sorted(summary.keys())[-1]
                metric_series = interceptor.get_head_metric_series(fallback_layer)
                if entropy_row is None and metric_series is not None and getattr(metric_series, "numel", lambda: 0)() > 0:
                    entropy_row = metric_series[-1]

        if entropy_row is not None and hasattr(entropy_row, "ndim") and entropy_row.ndim > 1:
            # Vector mode [H, F] -> default to first feature (Shannon-like stream)
            entropy_row = entropy_row[:, 0]

        velocity = None
        arc_steps = getattr(observer, "arc_steps", None)
        if arc_steps is not None and len(arc_steps) > 0:
            velocity = float(arc_steps[-1])

        frame = {
            "model_name": getattr(config, "model_name", "Qwen/Qwen2.5-0.5B"),
            "layer_idx": getattr(config, "target_layer", 0),
            "n_heads": getattr(config, "n_heads", None),
            "hidden_dim": getattr(config, "hidden_dim", None),
            "total_tokens": getattr(config, "total_tokens", getattr(config, "max_new_tokens", None)),
            "head_metric": getattr(config, "head_metric_type", "shannon_entropy"),
            "feat_mode": getattr(config, "head_feature_mode", "scalar"),
            "sink_removed": getattr(config, "remove_attention_sink", True),
            "pert_mode": getattr(config, "perturbation_mode", "zero"),
            "token_idx": int(token_idx),
            "entropy_row": entropy_row,
            "hurst": _safe_float(getattr(observer, "hurst_exponent", None)),
            "tau": _safe_float(getattr(observer, "tau_normalised", None)),
            "velocity": velocity,
            "adf_pval_pc0": _safe_float(getattr(observer, "adf_pval_pc0", None)),
            "cot_time_axis": "step-level (D.1)" if getattr(config, "use_cot_time_axis", False) else "token-level",
            "cot_n_steps": getattr(observer, "n_cot_steps", None),
            "log_events": log_events or [],
        }
        self._send(frame)

    def push_var_frame(self, analyzer_or_dict, config=None):
        """Push VAR/FDR influence frame to dashboard.
        
        Accepts either:
        - analyzer instance + config (standard path from analyzer.head_interaction_analysis)
        - pre-built dict from Exp6 (Exp6 aggregates across layers before pushing)
        """
        if isinstance(analyzer_or_dict, dict):
            # Exp6 pre-built dict path
            frame = self._build_var_frame_from_dict(analyzer_or_dict)
        else:
            # Standard analyzer instance path
            head_result = analyzer_or_dict
            frame = self._build_var_frame_from_analyzer(head_result)
        self._send(frame)

    def _build_var_frame_from_analyzer(self, head_result: dict):
        """Extract VAR frame from analyzer result dict (standard path)."""
        stationarity = head_result.get("joint_stationarity", {})
        per_head = stationarity.get("per_head") if isinstance(stationarity, dict) else None

        frame = {
            "stat_per_head": per_head,
            "coint_rank": int(head_result.get("cointegration", {}).get("n_coint_vectors", 0))
            if isinstance(head_result.get("cointegration"), dict)
            else 0,
            "vecm_used": head_result.get("model_type") == "VECM",
            "selected_lag": head_result.get("selected_lag"),
            "influence_matrix": head_result.get("influence_matrix"),
            "fdr_reject": (head_result.get("fdr_result") or {}).get("reject_matrix"),
            "pval_matrix": (head_result.get("fdr_result") or {}).get("pval_corrected"),
            "sig_pairs": (head_result.get("fdr_result") or {}).get("n_significant"),
            "significant_pairs": self._convert_pairs(
                head_result.get("significant_pairs"),
                head_result.get("influence_matrix"),
            ),
            "pdc_low": ((head_result.get("pdc") or {}).get("pdc_low") if isinstance(head_result.get("pdc"), dict) else None),
            "pdc_high": ((head_result.get("pdc") or {}).get("pdc_high") if isinstance(head_result.get("pdc"), dict) else None),
        }
        return frame

    def _build_var_frame_from_dict(self, aggregated: dict):
        """Extract VAR frame from Exp6 pre-built dict (Exp6 aggregation path)."""
        influence = aggregated.get("influence_matrix")
        if influence is None:
            influence = aggregated.get("masked_influence")
            
        fdr_reject = aggregated.get("fdr_reject_matrix")
        if fdr_reject is None:
            fdr_reject = (aggregated.get("fdr_result") or {}).get("reject_matrix")
            
        pval_mat = aggregated.get("pval_matrix")
        if pval_mat is None:
            pval_mat = (aggregated.get("fdr_result") or {}).get("pval_corrected")
        stat_per_head = None
        
        # Extract stationarity per head if available
        stationarity = aggregated.get("joint_stationarity", {})
        if isinstance(stationarity, dict):
            stat_per_head = stationarity.get("per_head")

        # Build significant pairs from influence + FDR mask
        sig_pairs_list = None
        if influence is not None and fdr_reject is not None:
            H = len(influence)
            sig_pairs_list = []
            for i in range(H):
                for j in range(H):
                    if i != j and fdr_reject[i][j]:
                        sig_pairs_list.append({
                            "source": j,
                            "target": i,
                            "score": float(influence[i][j]),
                            "pval": float(pval_mat[i][j]) if pval_mat is not None else None,
                            "fdr_sig": True,
                        })
            sig_pairs_list.sort(key=lambda x: x["score"], reverse=True)

        n_sig = int(len(sig_pairs_list)) if sig_pairs_list else aggregated.get("n_significant_pairs", 0)

        frame = {
            "stat_per_head": stat_per_head,
            "coint_rank": aggregated.get("coint_rank", 0),
            "vecm_used": aggregated.get("vecm_used", False),
            "selected_lag": aggregated.get("selected_lag"),
            "influence_matrix": influence,
            "fdr_reject": fdr_reject,
            "pval_matrix": pval_mat,
            "sig_pairs": int(n_sig),
            "significant_pairs": sig_pairs_list,
            "pdc_low": aggregated.get("pdc_low"),
            "pdc_high": aggregated.get("pdc_high"),
        }
        return frame

    def push_perturbation_frame(self, pert_results: list[dict], mediation_results: list[dict] | None = None):
        frame = {
            "perturbation_results": [
                {
                    "head": int(p.get("head", -1)),
                    "target": int(p.get("target", -1)),
                    "mode": str(p.get("mode", "zero")),
                    "delta_entropy": float(p.get("delta_entropy", 0.0)),
                    "restoration": float(p.get("restoration", 0.0)),
                    "kl_patch": float(p.get("kl_patch", 0.0)),
                    "confirmed": bool(p.get("confirmed", False)),
                }
                for p in (pert_results or [])
            ],
            "mediation_results": mediation_results if mediation_results else None,
            "log_events": [
                {
                    "type": "pert",
                    "msg": f"C.1/C.2: {len(pert_results or [])} ablation results received",
                }
            ],
        }
        self._send(frame)

    def push_hmm_frame(self, hmm_result: dict, config):
        if not hmm_result:
            return

        state_seq = hmm_result.get("state_sequence")
        n_states = int(hmm_result.get("n_states_used", getattr(config, "hmm_n_states", 4)))
        means = hmm_result.get("state_means")
        trans = hmm_result.get("transition_matrix")
        bic = hmm_result.get("bic")

        hmm_states = []
        if state_seq is not None:
            seq_np = np.asarray(state_seq)
            for s in range(n_states):
                mask = seq_np == s
                idx = np.where(mask)[0]
                token_range = f"{idx[0]}-{idx[-1]}" if len(idx) > 0 else "-"
                mean_h = float(np.asarray(means)[s].mean()) if means is not None else 0.0
                hmm_states.append(
                    {
                        "id": s,
                        "label": _state_label(s, mean_h),
                        "color": _phase_colour(s),
                        "token_range": token_range,
                        "mean_entropy": round(mean_h, 3),
                    }
                )

        log_msg = "D.4: HMM fitted"
        if bic is not None:
            try:
                log_msg = f"D.4: HMM fitted · n_states={n_states} · BIC={float(bic):.1f}"
            except Exception:
                log_msg = f"D.4: HMM fitted · n_states={n_states}"

        self._send(
            {
                "hmm_n_states": n_states,
                "hmm_states": hmm_states,
                "hmm_state_seq": state_seq,
                "hmm_trans": trans,
                "hmm_bic": _safe_float(bic),
                "hmm_ll": _safe_float(hmm_result.get("log_likelihood")),
                "log_events": [{"type": "tda", "msg": log_msg}],
            }
        )

    def push_tda_frame(self, tda_result: dict, current_token: int, phase_boundaries=None, current_phase_idx=None):
        if not tda_result:
            return

        frame = {
            "betti0": tda_result.get("betti0"),
            "betti1": tda_result.get("betti1"),
            "euler": tda_result.get("euler"),
            "ec_series": tda_result.get("ec_series"),
            "topo_anomalies": tda_result.get("anomalies"),
            "phase_boundaries": phase_boundaries,
            "current_phase_idx": current_phase_idx,
        }

        anomalies = tda_result.get("anomalies") or []
        current = [a for a in anomalies if a.get("token_idx") == current_token]
        if current:
            frame["log_events"] = [
                {
                    "type": "tda",
                    "msg": f"D.2: EC spike t={current_token} · {a.get('severity', 'info')} · {a.get('description', '')}",
                }
                for a in current
            ]

        self._send(frame)

    def push_signal_quality_frame(self, interceptor, config):
        layer_idx = getattr(config, "target_layer", None)
        key = f"layers.{layer_idx}" if layer_idx is not None else None

        metrics = None
        if key:
            metrics = interceptor._head_metrics.get(key, [])
        if not metrics and interceptor._head_metrics:
            metrics = interceptor._head_metrics.get(sorted(interceptor._head_metrics.keys())[-1], [])

        if not metrics:
            return

        last = metrics[-1]
        # Handle both scalars and vector feature modes
        if hasattr(last, "shape"):
            if last.ndim == 2:
                H = min(int(getattr(config, "n_heads", last.shape[0])), last.shape[0])
            else:
                H = min(int(getattr(config, "n_heads", len(last))), len(last))
        else:
            H = int(getattr(config, "n_heads", 1))
        
        feat_mode = getattr(config, "head_feature_mode", "scalar")

        sq_data = []
        for h in range(H):
            try:
                if feat_mode == "vector" and hasattr(last, "ndim") and last.ndim == 2:
                    if h < last.shape[0]:
                        row = last[h]
                        sq_data.append(
                            {
                                "h": h,
                                "shannon": _safe_float(row[0]),
                                "renyi": _safe_float(row[1]),
                                "effrank": _safe_float(row[3]) if len(row) > 3 else None,
                                "sink": _safe_float(row[4]) if len(row) > 4 else None,
                                "maxattn": _safe_float(row[2]) if len(row) > 2 else None,
                                "ov_norm": None,
                            }
                        )
                else:
                    # Scalar mode — access single value per head
                    if hasattr(last, "__getitem__") and h < (last.shape[0] if hasattr(last, "shape") else len(last)):
                        val = _safe_float(last[h])
                    else:
                        val = None
                    sq_data.append(
                        {
                            "h": h,
                            "shannon": val,
                            "renyi": None,
                            "effrank": None,
                            "sink": None,
                            "maxattn": None,
                            "ov_norm": None,
                        }
                    )
            except (IndexError, TypeError):
                # Fallback for mismatched dimensions
                sq_data.append(
                    {
                        "h": h,
                        "shannon": None,
                        "renyi": None,
                        "effrank": None,
                        "sink": None,
                        "maxattn": None,
                        "ov_norm": None,
                    }
                )

        self._send({"signal_quality": sq_data})

    def push_score_frame(self, composite: dict, interpretation: dict | None = None):
        frame = {
            "composite_score": round(float(composite.get("score", 0))),
            "dtw_sensitivity": _safe_float(composite.get("dtw_sensitivity")),
            "spectral_coherence": _safe_float(composite.get("spectral_coherence")),
            "topo_smoothness": _safe_float(composite.get("topo_smoothness")),
            "active_reasoning": _safe_float(composite.get("active_reasoning")),
            "te_score": _safe_float(composite.get("te_score")),
            "sig_pairs": int(composite.get("fdr_sig_pairs", 0)),
            "verdict": composite.get("verdict", "MODERATE REASONING"),
            "log_events": [
                {
                    "type": "ok",
                    "msg": f"Composite score: {round(float(composite.get('score', 0)))} -> {composite.get('verdict', '-')}",
                }
            ],
        }
        if interpretation:
            frame["interpretation"] = interpretation
        self._send(frame)

    def push_log(self, type_: str, msg: str):
        self._send({"log_events": [{"type": type_, "msg": msg}]})

    def _convert_pairs(self, pairs, influence_matrix):
        if not pairs:
            return None
        inf = np.asarray(influence_matrix) if influence_matrix is not None else None
        out = []
        for item in pairs:
            if len(item) < 3:
                continue
            source, target, pval = int(item[0]), int(item[1]), item[2]
            score = float(inf[target, source]) if inf is not None else 0.0
            out.append(
                {
                    "source": source,
                    "target": target,
                    "score": score,
                    "pval": _safe_float(pval),
                    "fdr_sig": True,
                }
            )
        return out

    def _send(self, partial_frame: dict):
        self._last_frame.update(partial_frame)
        payload = json.dumps(partial_frame, cls=_Encoder, allow_nan=False)

        if self.transport == "websocket":
            self._ws_broadcast(payload)
        elif self.transport == "file":
            self._write_file(partial_frame)
        elif self.transport == "inject" and self.inject_fn:
            self.inject_fn(f"window.ChronoscopeBridge.push({payload})")

    def _write_file(self, frame: dict):
        frame = dict(frame)
        frame["_mtime"] = time.time()
        self.poll_path.parent.mkdir(parents=True, exist_ok=True)
        self.poll_path.write_text(json.dumps(frame, cls=_Encoder, allow_nan=False), encoding="utf-8")

    def _ws_broadcast(self, payload: str):
        if not self._loop or not self._ws_clients:
            return

        async def _do_send():
            dead = set()
            for ws in list(self._ws_clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            self._ws_clients -= dead

        asyncio.run_coroutine_threadsafe(_do_send(), self._loop)

    def _start_websocket_server(self):
        import websockets

        async def _handler(ws):
            self._ws_clients.add(ws)
            if self._last_frame:
                try:
                    await ws.send(json.dumps(self._last_frame, cls=_Encoder, allow_nan=False))
                except Exception:
                    pass
            try:
                await ws.wait_closed()
            finally:
                self._ws_clients.discard(ws)

        async def _server():
            async with websockets.serve(_handler, self.ws_host, self.ws_port):
                await asyncio.Future()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(_server())

        self._thread = threading.Thread(target=_run, daemon=True, name="chronoscope-ws")
        self._thread.start()
        time.sleep(0.1)
