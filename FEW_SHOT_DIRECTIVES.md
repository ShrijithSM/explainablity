# Chronoscope — Few-Shot Directive Behaviour Examples
## PyTorch API steering guide for the coding agent

**PyTorch API reference:** https://docs.pytorch.org/docs/main/pytorch-api.html  
**Scope:** Gaps A–E · files: `analyzer.py`, `interceptor.py`, `observer.py`  
**Pattern:** Each directive is a (TASK → WRONG → RIGHT → WHY) quad.  
The agent must match the RIGHT pattern. The WRONG pattern is shown so the agent  
recognises it in existing code and replaces it.

---

## How to read these directives

Every example follows this structure:

```
DIRECTIVE   — what the agent is being asked to do
WRONG       — the pattern found in existing stub code (do not emit this)
RIGHT       — the exact PyTorch API the agent must use instead
WHY         — the reason: correctness, GPU locality, or API availability
VERIFY      — the assertion the agent must pass before committing the change
```

The agent must read all five fields. Matching WRONG output is a failure.

---

## Gap A — Stationarity preprocessing before VAR

### Directive A-1 · First-differencing with `torch.diff`

```
DIRECTIVE
  In analyzer.py::_apply_selective_differencing(), replace the Python-loop
  per-head differencing with a single batched torch operation that stays on GPU
  before the .numpy() handoff to statsmodels.

WRONG
  # Loop over heads — slow, leaves tensor on CPU, wastes two allocations
  out = np.empty((T - 1, H), dtype=np.float64)
  for h in range(H):
      if diff_mask[h]:
          out[:, h] = np.diff(metric_series[:, h])
      else:
          out[:, h] = metric_series[1:, h]

RIGHT
  import torch
  # torch.diff(input, n=1, dim=0) — docs: https://docs.pytorch.org/docs/main/torch.html
  # Computes the n-th forward difference along the given dimension.
  # diff_mask shape: [H] bool tensor  |  series_t shape: [T, H] float tensor

  series_t = torch.as_tensor(metric_series, dtype=torch.float64)   # [T, H]
  mask     = torch.as_tensor(diff_mask, dtype=torch.bool)           # [H]

  differenced = torch.diff(series_t, n=1, dim=0)                    # [T-1, H]
  trimmed     = series_t[1:, :]                                      # [T-1, H]  no-op heads

  # Combine: use diff for flagged heads, trimmed for others
  # torch.where(condition, x, y) broadcasts over the H dimension
  result = torch.where(mask.unsqueeze(0), differenced, trimmed)     # [T-1, H]
  out    = result.numpy()                                             # handoff to statsmodels

WHY
  torch.diff is a single fused kernel call. The loop allocates H intermediate
  numpy arrays and forces a Python-level dispatch per head. On GPU the loop
  requires H device→host copies; torch.where stays on device until .numpy().
  API source: https://docs.pytorch.org/docs/main/generated/torch.diff.html

VERIFY
  assert out.shape == (metric_series.shape[0] - 1, metric_series.shape[1])
  assert out.dtype == np.float64
  # For a head in diff_mask, check: out[:, h] == np.diff(metric_series[:, h])
  # For a head not in diff_mask, check: out[:, h] == metric_series[1:, h]
```

---

### Directive A-2 · Per-head mean removal with `torch.mean`

```
DIRECTIVE
  In analyzer.py::_test_per_head_stationarity(), centre each head series
  before ADF to remove level effects. Use torch.mean, not np.mean in a loop.

WRONG
  centered = metric_series.copy()
  for h in range(H):
      centered[:, h] -= metric_series[:, h].mean()

RIGHT
  import torch
  # torch.mean(input, dim, keepdim) — docs: https://docs.pytorch.org/docs/main/torch.html
  series_t  = torch.as_tensor(metric_series, dtype=torch.float64)   # [T, H]
  col_means = torch.mean(series_t, dim=0, keepdim=True)             # [1, H]
  centered  = (series_t - col_means).numpy()                        # [T, H] zero-mean

WHY
  Single BLAS-dispatched call vs H Python iterations. keepdim=True enables
  broadcast subtraction without reshape. Works identically on CPU and CUDA.
  API source: https://docs.pytorch.org/docs/main/generated/torch.mean.html

VERIFY
  assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-9)
  assert centered.shape == metric_series.shape
```

---

### Directive A-3 · Variance computation for KPSS preprocessing

```
DIRECTIVE
  In analyzer.py::_joint_stationarity_test(), compute the long-run variance
  estimate for each head using torch.var before passing scalars to kpss().

WRONG
  variances = [s.var() for s in metric_series.T]   # Python list, numpy .var()

RIGHT
  import torch
  # torch.var(input, dim, correction) — docs: https://docs.pytorch.org/docs/main/torch.html
  # correction=1 gives Bessel-corrected (unbiased) sample variance
  series_t = torch.as_tensor(metric_series, dtype=torch.float64)     # [T, H]
  variances = torch.var(series_t, dim=0, correction=1)               # [H]
  # Use variances.tolist() when passing individual floats to statsmodels KPSS
  var_list  = variances.tolist()                                      # Python list[float]

WHY
  torch.var with dim=0 computes all H variances in one vectorised pass.
  correction=1 matches numpy's default ddof=1.
  API source: https://docs.pytorch.org/docs/main/generated/torch.var.html

VERIFY
  import numpy as np
  assert np.allclose(variances.numpy(), metric_series.var(axis=0, ddof=1), atol=1e-8)
```

---

## Gap B — Statistical significance

### Directive B-1 · Power spectrum via `torch.fft.rfft`

```
DIRECTIVE
  In analyzer.py::_partial_directed_coherence() spectral preprocessing step,
  compute the one-sided power spectral density for each head using torch.fft,
  not scipy.signal.periodogram in a loop.

WRONG
  from scipy.signal import periodogram
  spectra = []
  for h in range(H):
      _, psd = periodogram(series_np[:, h])
      spectra.append(psd)
  spectra = np.stack(spectra)   # [H, F]

RIGHT
  import torch
  import torch.fft
  # torch.fft.rfft(input, n, dim) — docs: https://docs.pytorch.org/docs/main/fft.html
  # Returns complex spectrum of real-valued input, shape [..., n//2 + 1]

  series_t = torch.as_tensor(series_np, dtype=torch.float64)   # [T, H]
  # rfft along time axis (dim=0) — output shape: [T//2+1, H]
  spectrum  = torch.fft.rfft(series_t, dim=0)                  # [F, H] complex
  psd       = torch.abs(spectrum).pow(2)                        # [F, H] power
  # Normalise by T to get one-sided PSD consistent with scipy convention
  psd       = psd / series_t.shape[0]
  psd_np    = psd.numpy()                                       # [F, H] for downstream

WHY
  torch.fft.rfft dispatches to cuFFT on GPU and MKL-FFT on CPU.
  The scipy loop forces T×H Python-level ops and CPU-only execution.
  API source: https://docs.pytorch.org/docs/main/fft.html

VERIFY
  from scipy.signal import periodogram
  _, psd_ref = periodogram(series_np[:, 0])
  # Shapes match (allow for normalisation factor difference)
  assert psd_np.shape[0] == len(psd_ref)
  assert psd_np.shape[1] == H
```

---

### Directive B-2 · Influence matrix aggregation via `torch.linalg.norm`

```
DIRECTIVE
  In analyzer.py, after extracting VAR coefficients result.coefs [p, H, H],
  aggregate the absolute coefficient norms using torch.linalg instead of
  np.abs().sum().

WRONG
  coefs    = result.coefs                         # numpy [p, H, H]
  influence = np.abs(coefs).sum(axis=0)           # [H, H] — loses sign structure

RIGHT
  import torch
  import torch.linalg
  # torch.linalg.norm(A, ord, dim) — docs: https://docs.pytorch.org/docs/main/linalg.html
  # For ord=1 (sum of abs values) along the lag axis (dim=0), this is equivalent
  # to summing |A_l[i,j]| across all lags — the Chronoscope influence score.

  coefs_t   = torch.as_tensor(result.coefs, dtype=torch.float64)   # [p, H, H]
  # Nuclear norm alternative: use ord=None for Frobenius, ord=1 for column sum
  # For Chronoscope influence score we want: influence[i,j] = Σ_l |A_l[i,j]|
  # which is the L1 norm over the lag dimension
  influence = torch.linalg.norm(coefs_t, ord=1, dim=0)             # [H, H]
  influence_np = influence.numpy()

  # Alternatively, for exact Σ|A_l[i,j]|:
  influence_exact = coefs_t.abs().sum(dim=0).numpy()                # [H, H]

WHY
  torch.linalg.norm is the authoritative linear algebra API in PyTorch ≥1.9,
  replacing the deprecated torch.norm. It maps to LAPACK on CPU and cuBLAS on GPU.
  API source: https://docs.pytorch.org/docs/main/linalg.html

VERIFY
  import numpy as np
  assert np.allclose(influence_np, np.abs(result.coefs).sum(axis=0), atol=1e-10)
  assert influence_np.shape == (H, H)
```

---

### Directive B-3 · Surrogate phase scrambling via `torch.fft.rfft` + `torch.angle`

```
DIRECTIVE
  In analyzer.py::_bootstrap_surrogate_pvalues(), replace the numpy-based
  phase scramble loop with a fully vectorised torch.fft round-trip.

WRONG
  # Per-head, per-surrogate numpy loop — O(n_surrogates × H) Python dispatches
  for h in range(H):
      fft_vals = np.fft.rfft(series_np[:, h])
      random_phases = rng.uniform(0, 2 * np.pi, len(fft_vals))
      random_phases[0] = 0.0
      fft_scrambled = np.abs(fft_vals) * np.exp(1j * random_phases)
      surrogate[:, h] = np.fft.irfft(fft_scrambled, n=T)

RIGHT
  import torch
  import torch.fft
  # torch.fft.rfft, torch.fft.irfft — docs: https://docs.pytorch.org/docs/main/fft.html
  # torch.angle(input) — returns element-wise phase angle of complex tensor
  # API source: https://docs.pytorch.org/docs/main/generated/torch.angle.html

  series_t  = torch.as_tensor(series_np, dtype=torch.float64)         # [T, H]
  spectrum  = torch.fft.rfft(series_t, dim=0)                         # [F, H] complex
  magnitude = spectrum.abs()                                           # [F, H] real

  # Generate random phases for all surrogates at once: [n_surrogates, F, H]
  F = spectrum.shape[0]
  rand_phases = torch.rand(n_surrogates, F, H, dtype=torch.float64) * 2 * torch.pi
  rand_phases[:, 0, :] = 0.0   # preserve DC component (mean level)

  # Reconstruct scrambled spectra: magnitude broadcast × complex unit phasors
  # [1,F,H] * exp(i * [S,F,H]) = [S,F,H] complex
  scrambled = magnitude.unsqueeze(0) * torch.polar(
      torch.ones(n_surrogates, F, H, dtype=torch.float64),
      rand_phases
  )                                                                    # [S, F, H]

  # Inverse FFT for all surrogates in one call
  surrogates = torch.fft.irfft(scrambled, n=T, dim=1)                 # [S, T, H]
  # surrogates[s] is the s-th scrambled version of the [T, H] series

WHY
  One vectorised rfft + irfft over [S, F, H] replaces n_surrogates×H serial calls.
  torch.polar(abs, angle) is the canonical way to construct complex tensors from
  polar form. API sources:
    torch.fft:   https://docs.pytorch.org/docs/main/fft.html
    torch.polar: https://docs.pytorch.org/docs/main/generated/torch.polar.html
    torch.angle: https://docs.pytorch.org/docs/main/generated/torch.angle.html

VERIFY
  assert surrogates.shape == (n_surrogates, T, H)
  # Phase scrambling preserves magnitude spectrum
  orig_mag = torch.fft.rfft(series_t, dim=0).abs()
  surr_mag = torch.fft.rfft(surrogates[0], dim=0).abs()
  assert torch.allclose(orig_mag, surr_mag, atol=1e-6)
```

---

## Gap C — Perturbation

### Directive C-1 · Hook registration with `register_forward_hook` (interceptor.py baseline)

```
DIRECTIVE
  In interceptor.py::_attach_hooks(), register lightweight non-modifying hooks
  for data CAPTURE only. This is the CAPTURE path — distinct from Gap C
  intervention hooks which must use pyvene. Never mix capture hooks with
  intervention logic in the same hook function.

WRONG
  # Intervention logic mixed into capture hook — wrong separation of concerns
  def hook_fn(module, input, output):
      attn = output[1]
      hidden = output[0].clone()
      hidden[:, :, start:end] = 0.0           # ← intervention, not capture
      entropy = compute_entropy(attn)
      self._head_metrics[name].append(entropy)
      return (hidden,) + output[1:]

RIGHT
  import torch
  # module.register_forward_hook(hook) — docs: https://docs.pytorch.org/docs/main/nn.html
  # hook signature: hook(module, input, output) → None or modified output
  # For CAPTURE only, return None (do not return a modified tensor).

  def _make_capture_hook(self, name: str) -> callable:
      """
      Pure capture hook — records entropy, does NOT modify the forward pass.
      Return value must be None so PyTorch does not swap the module output.
      """
      def hook_fn(module, input, output):
          # output[1] is attention weights [B, H, T, T] when output_attentions=True
          attn = output[1]                              # [B, H, T, T]
          last = attn[:, :, -1, :]                      # most recent token: [B, H, T]

          # torch.clamp_min prevents log(0); in-place clamp_ also valid here
          eps   = 1e-9
          probs = torch.clamp(last, min=eps)
          probs = probs / probs.sum(dim=-1, keepdim=True)   # normalise

          # Shannon entropy: H = -Σ p log p
          # torch.log dispatches to vectorised log kernel
          entropy = -(probs * torch.log(probs)).sum(dim=-1) # [B, H]

          self._head_metrics[name].append(entropy.detach().cpu())
          # CRITICAL: return None — do not return a modified output tensor
          return None

      handle = module.register_forward_hook(hook_fn)
      self._hooks[name] = handle
      return hook_fn

WHY
  Returning None from a register_forward_hook preserves the original output
  tensor. Returning a modified tensor replaces the module output and is ONLY
  valid for intervention hooks (Gap C, managed by pyvene). Mixing both
  responsibilities in one hook creates silent correctness bugs where the
  intervention fires during capture-only runs.
  API source: https://docs.pytorch.org/docs/main/nn.html#torch.nn.Module.register_forward_hook

VERIFY
  # After attaching hook, run model forward pass and confirm:
  assert all(h.detach().cpu().shape == (1, N_HEADS) for h in self._head_metrics[name])
  # Confirm model output logits are identical with and without hook attached
```

---

### Directive C-2 · Entropy delta measurement using `torch.nn.functional.kl_div`

```
DIRECTIVE
  In analyzer.py::_activation_patch_experiment(), compute the KL divergence
  between output distributions using torch.nn.functional.kl_div, not a manual
  formula. This is the canonical PyTorch way and avoids numerical instability.

WRONG
  # Manual KL — numerically unstable and not GPU-fused
  kl = (clean_probs * torch.log(clean_probs / corrupt_probs)).sum()

RIGHT
  import torch.nn.functional as F
  # F.kl_div(input, target, reduction, log_target) — docs:
  # https://docs.pytorch.org/docs/main/nn.functional.html
  # IMPORTANT: input must be LOG-probabilities, target must be probabilities
  # (or both log-probabilities when log_target=True)

  clean_logits   = ...   # [1, vocab_size]
  corrupt_logits = ...   # [1, vocab_size]
  patched_logits = ...   # [1, vocab_size]

  # Compute softmax probabilities and log-softmax in one fused op
  clean_lp   = F.log_softmax(clean_logits,   dim=-1)   # log P_clean
  corrupt_lp = F.log_softmax(corrupt_logits, dim=-1)   # log P_corrupt
  patched_lp = F.log_softmax(patched_logits, dim=-1)   # log P_patched

  clean_p    = clean_lp.exp()   # P_clean (target)

  # KL(clean || corrupt): how far corrupt is from clean — the gap to bridge
  kl_baseline = F.kl_div(corrupt_lp, clean_p, reduction='sum', log_target=False)

  # KL(clean || patched): how far patched is from clean after intervention
  kl_patched  = F.kl_div(patched_lp,  clean_p, reduction='sum', log_target=False)

  # Restoration: fraction of the clean-corrupt gap recovered by patching
  restoration = float((1.0 - kl_patched / kl_baseline).clamp(0.0, 1.0))

WHY
  F.kl_div uses a numerically stable fused log+subtract kernel. The manual
  formula suffers from log(0) and catastrophic cancellation when probabilities
  are near zero. reduction='sum' is correct here (not 'batchmean') because
  we have a single [1, vocab] distribution, not a batch.
  API source: https://docs.pytorch.org/docs/main/nn.functional.html#torch.nn.functional.kl_div

VERIFY
  assert 0.0 <= restoration <= 1.0
  # If patched == clean, restoration should be 1.0
  assert float(F.kl_div(clean_lp, clean_p, reduction='sum')) < 1e-6
```

---

## Gap D — Thinking time

### Directive D-1 · Per-layer timing with `torch.cuda.Event`

```
DIRECTIVE
  In observer.py::compute_intrinsic_time(), measure real per-token GPU wall
  time for each layer using torch.cuda.Event, not Python time.time() which
  includes Python overhead and is not synchronised to CUDA streams.

WRONG
  import time
  start = time.time()
  with torch.no_grad():
      output = model(**inputs)
  elapsed = time.time() - start   # measures Python + GPU + synchronisation

RIGHT
  import torch
  # torch.cuda.Event — docs: https://docs.pytorch.org/docs/main/cuda.html
  # event.record() marks a point on the CUDA stream
  # torch.cuda.synchronize() waits for all kernels to complete
  # event.elapsed_time(end_event) returns milliseconds between two events

  layer_times = {}   # layer_name → elapsed_ms

  def _make_timing_hook(layer_name: str):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event   = torch.cuda.Event(enable_timing=True)

      def pre_hook(module, input):
          # register_forward_pre_hook fires before the module forward
          start_event.record()

      def post_hook(module, input, output):
          # register_forward_hook fires after the module forward
          end_event.record()
          # Do NOT call synchronize here — it would stall the GPU pipeline.
          # Instead, accumulate events and sync once after generation completes.
          layer_times[layer_name] = (start_event, end_event)
          return None   # capture only

      return pre_hook, post_hook

  # After generation loop, synchronise once and read all elapsed times
  torch.cuda.synchronize()
  elapsed = {
      name: s.elapsed_time(e)   # milliseconds, float
      for name, (s, e) in layer_times.items()
  }

WHY
  torch.cuda.Event.elapsed_time() measures GPU-side time between CUDA stream
  markers — immune to Python GIL, OS scheduling, and PCIe transfer overhead.
  time.time() around a model forward includes all of these and is unreliable
  for sub-millisecond layer timing. Deferring synchronize() avoids pipeline stalls.
  API source: https://docs.pytorch.org/docs/main/cuda.html#torch.cuda.Event

VERIFY
  torch.cuda.synchronize()
  assert all(v > 0.0 for v in elapsed.values()), "All layer times must be positive"
  assert all(v < 1000.0 for v in elapsed.values()), "No layer should take >1s"
```

---

### Directive D-2 · Arc-length intrinsic time τ via `torch.linalg.norm`

```
DIRECTIVE
  In observer.py::compute_intrinsic_time(), compute the arc-length steps
  ds(t) = ||h(t) - h(t-1)||_2 using torch.linalg.norm, not np.linalg.norm.
  The hidden states tensor is already on GPU; keep it there for the norm.

WRONG
  trajectory_np = hidden_states.cpu().numpy()   # [T, D] — forces host transfer
  arc_steps = np.linalg.norm(np.diff(trajectory_np, axis=0), axis=1)  # [T-1]

RIGHT
  import torch
  import torch.linalg
  # torch.linalg.norm(A, ord, dim) — docs: https://docs.pytorch.org/docs/main/linalg.html
  # For Euclidean (L2) vector norm along last dim, use ord=None (default) or ord=2

  # hidden_states: [T, D] tensor, may be on GPU
  diff_vectors = hidden_states[1:] - hidden_states[:-1]          # [T-1, D]
  arc_steps    = torch.linalg.norm(diff_vectors, dim=-1)         # [T-1] L2 norm per step
  tau_raw      = torch.cat([
      torch.zeros(1, device=arc_steps.device, dtype=arc_steps.dtype),
      torch.cumsum(arc_steps, dim=0)
  ])                                                               # [T] cumulative arc-length
  tau_norm     = tau_raw / tau_raw[-1].clamp(min=1e-9)           # [T] normalised [0, 1]
  # .cpu().numpy() only when writing to synthesizer report
  arc_steps_np = arc_steps.cpu().numpy()
  tau_np       = tau_norm.cpu().numpy()

WHY
  Keeping the computation on GPU avoids a [T×D] PCIe transfer. torch.linalg.norm
  dispatches to cuBLAS batched norm kernels. torch.cumsum is O(T) and fused.
  torch.linalg supersedes the deprecated torch.norm as of PyTorch 1.9.
  API source: https://docs.pytorch.org/docs/main/linalg.html

VERIFY
  import numpy as np
  manual = np.linalg.norm(np.diff(hidden_states.cpu().numpy(), axis=0), axis=1)
  assert np.allclose(arc_steps_np, manual, atol=1e-5)
  assert tau_np[0] == 0.0
  assert abs(tau_np[-1] - 1.0) < 1e-6
```

---

### Directive D-3 · Module execution graph tracing with `torch.fx`

```
DIRECTIVE
  In observer.py, when config.use_cot_time_axis is True, use torch.fx to
  extract the static execution graph of the model and count operations per
  transformer layer — gives a proxy for "computational depth" per step.

WRONG
  # No static graph analysis — only dynamic hooks, misses compile-time structure
  n_ops = len(list(model.modules()))   # counts modules not operations

RIGHT
  import torch.fx
  # torch.fx.symbolic_trace — docs: https://docs.pytorch.org/docs/main/fx.html
  # Traces a model symbolically to produce a GraphModule with an operation graph.
  # node.op values: 'call_function', 'call_method', 'call_module', 'placeholder', 'output'

  try:
      traced = torch.fx.symbolic_trace(model)
      # Count arithmetic ops per transformer layer as a complexity proxy
      layer_op_counts = {}
      for node in traced.graph.nodes:
          if node.op == 'call_module':
              module_name = node.target   # e.g. 'model.layers.23.self_attn'
              layer_op_counts[module_name] = layer_op_counts.get(module_name, 0) + 1
          elif node.op == 'call_function':
              # Count torch.* function calls (matmul, softmax, etc.)
              parent = getattr(node.target, '__module__', '')
              if 'torch' in parent:
                  layer_op_counts['__global__'] = layer_op_counts.get('__global__', 0) + 1
  except torch.fx.proxy.TraceError:
      # Some models (with dynamic control flow) are not symbolically traceable
      # Fall back to module counting
      layer_op_counts = {name: 1 for name, _ in model.named_modules()}

WHY
  torch.fx.symbolic_trace produces a static computation graph at the IR level,
  exposing every operator call — more informative than hook-based counting which
  only fires at module boundaries. The TraceError fallback handles models with
  data-dependent control flow (like Qwen's dynamic attention masking).
  API source: https://docs.pytorch.org/docs/main/fx.html

VERIFY
  assert isinstance(layer_op_counts, dict)
  assert len(layer_op_counts) > 0
```

---

## Gap E — Signal quality

### Directive E-1 · Variance and kurtosis via `torch.var` + `torch.special`

```
DIRECTIVE
  In analyzer.py, extend _compute_head_metrics() to return variance and excess
  kurtosis of the attention distribution alongside entropy. Use torch.var and
  a custom kurtosis formula built from torch.special.gammaln for the
  log-normalised version, or the direct moment formula.

WRONG
  from scipy.stats import kurtosis
  kurt_vals = [kurtosis(attn_weights[0, h, -1, :].cpu().numpy()) for h in range(H)]

RIGHT
  import torch
  import torch.special
  # torch.var(input, dim, correction) — docs: https://docs.pytorch.org/docs/main/torch.html
  # torch.special module — docs: https://docs.pytorch.org/docs/main/special.html
  # For kurtosis we use the 4th standardised central moment directly:
  #   kurtosis = E[(X - μ)^4] / (E[(X - μ)^2])^2  - 3  (excess kurtosis)

  def compute_distribution_moments(
      attn_weights: torch.Tensor   # [B, H, T, T]
  ) -> dict:
      """
      Compute variance and excess kurtosis of each head's attention distribution
      at the most recent token position. Fully vectorised, no Python loops.
      """
      eps  = 1e-9
      last = attn_weights[:, :, -1, :]                    # [B, H, T]
      probs = torch.clamp(last, min=eps)
      probs = probs / probs.sum(dim=-1, keepdim=True)     # [B, H, T] normalised

      # --- Variance of the attention distribution (E[p^2] - E[p]^2) ---
      # E[p] for a probability distribution = 1/T (if uniform), varies otherwise
      # Here: variance across token positions within each head
      # torch.var(correction=0) for population variance, correction=1 for sample
      attn_var = torch.var(probs, dim=-1, correction=0)   # [B, H]

      # --- Excess kurtosis via 4th central moment ---
      # mu = mean probability value across positions
      mu       = probs.mean(dim=-1, keepdim=True)         # [B, H, 1]
      deviations = probs - mu                             # [B, H, T] centred
      # 2nd moment (variance, population)
      m2       = (deviations.pow(2)).mean(dim=-1)         # [B, H]
      # 4th moment
      m4       = (deviations.pow(4)).mean(dim=-1)         # [B, H]
      # Excess kurtosis: m4 / m2^2 - 3
      kurtosis = m4 / m2.pow(2).clamp(min=eps) - 3.0     # [B, H]

      # --- Rényi entropy α=2 via torch.special.entr for comparison ---
      # torch.special.entr(x) = -x * log(x) elementwise
      # API source: https://docs.pytorch.org/docs/main/special.html
      # Shannon entropy = sum(entr(p)) = -Σ p log p
      shannon_from_entr = torch.special.entr(probs).sum(dim=-1)   # [B, H]

      return {
          'variance':  attn_var[0].detach().cpu().numpy(),   # [H]
          'kurtosis':  kurtosis[0].detach().cpu().numpy(),   # [H]
          'shannon':   shannon_from_entr[0].detach().cpu().numpy(),  # [H]
      }

WHY
  torch.var and moment formulae are fully GPU-batched. scipy.stats.kurtosis
  requires a CPU numpy array, forcing a device transfer per head per token step.
  torch.special.entr is the canonical elementwise -x*log(x) op used in
  information theory computations.
  API sources:
    torch.var:          https://docs.pytorch.org/docs/main/generated/torch.var.html
    torch.special.entr: https://docs.pytorch.org/docs/main/special.html
    torch.special:      https://docs.pytorch.org/docs/main/special.html

VERIFY
  from scipy.stats import kurtosis as scipy_kurt
  probs_np = probs[0, 0, :].cpu().numpy()
  assert abs(kurtosis[0, 0].item() - scipy_kurt(probs_np)) < 0.01
  assert attn_var.shape == (1, H)
  assert kurtosis.shape == (1, H)
```

---

### Directive E-2 · Effective rank via `torch.linalg.matrix_rank` + singular values

```
DIRECTIVE
  In analyzer.py::_compute_head_metrics(), compute the effective rank of the
  attention matrix using torch.linalg.svdvals — a richer measure than
  exp(H_shannon)/T because it reflects the actual low-rank structure of the
  attention pattern.

WRONG
  # Scalar effective rank from Shannon entropy — misses low-rank structure
  eff_rank = np.exp(shannon_entropy) / T_context

RIGHT
  import torch
  import torch.linalg
  # torch.linalg.svdvals(A) — docs: https://docs.pytorch.org/docs/main/linalg.html
  # Returns singular values of A in descending order. Shape [*, min(m,n)].
  # Effective rank (Roy & Vetterli 2007): exp(H_normalised_singular_spectrum)

  def compute_effective_rank(
      attn_weights: torch.Tensor,   # [B, H, T, T] full attention matrix
      h_idx: int
  ) -> float:
      """
      Compute the effective rank of the attention matrix for head h_idx
      at the current context window. Uses singular value entropy.
      """
      A   = attn_weights[0, h_idx, :, :]          # [T, T] for this head
      svs = torch.linalg.svdvals(A)                # [min(T,T)] singular values
      svs = svs / svs.sum().clamp(min=1e-9)        # normalise to probability
      svs = svs.clamp(min=1e-9)                    # avoid log(0)
      # Effective rank = exp(Shannon entropy of singular value spectrum)
      eff_rank = torch.exp(-(svs * torch.log(svs)).sum())
      return float(eff_rank.item())

  # Vectorised over all heads at once
  # attn_matrix: [B, H, T, T] — compute svdvals for all heads in one call
  # torch.linalg.svdvals supports batched input
  attn_batch = attn_weights[0]                              # [H, T, T]
  svs_batch  = torch.linalg.svdvals(attn_batch)            # [H, min(T,T)]
  svs_norm   = svs_batch / svs_batch.sum(dim=-1, keepdim=True).clamp(min=1e-9)
  svs_clamp  = svs_norm.clamp(min=1e-9)
  eff_ranks  = torch.exp(-(svs_clamp * torch.log(svs_clamp)).sum(dim=-1))   # [H]

WHY
  torch.linalg.svdvals is the PyTorch ≥1.9 canonical SVD API, dispatching to
  LAPACK dgesdd on CPU and cuSOLVER on GPU. The batched [H, T, T] call avoids
  H separate SVD invocations. Effective rank from the singular spectrum captures
  low-rank structure (sharp attention = rank-1) that scalar entropy cannot.
  API source: https://docs.pytorch.org/docs/main/linalg.html

VERIFY
  # Uniform attention matrix (softmax of zeros) should have rank ≈ T
  uniform = torch.ones(1, 1, 8, 8) / 8.0
  er = compute_effective_rank(uniform, 0)
  assert abs(er - 8.0) < 0.1, f"Uniform attention should have eff_rank ≈ T, got {er}"
  # Rank-1 (all weight on one token) should have eff_rank ≈ 1
  rank1 = torch.zeros(1, 1, 8, 8); rank1[0, 0, :, 0] = 1.0
  er1 = compute_effective_rank(rank1, 0)
  assert er1 < 1.5, f"Rank-1 attention should have eff_rank ≈ 1, got {er1}"
```

---

### Directive E-3 · OV circuit output norm via `torch.einsum` + `torch.linalg.norm`

```
DIRECTIVE
  In interceptor.py::_compute_ov_weighted_metric(), compute the value-weighted
  attention output (OV circuit contribution) using torch.einsum, which expresses
  the batched matrix-vector product cleanly without reshaping.

WRONG
  # Manual reshape + bmm — loses readability and requires contiguous memory
  weighted_v = torch.bmm(
      attn_last.squeeze(2),          # [B*H, 1, T]
      value_states.view(B*H, T, d)   # [B*H, T, d]
  ).squeeze(1)                       # [B*H, d] — loses H dimension semantics

RIGHT
  import torch
  # torch.einsum — docs: https://docs.pytorch.org/docs/main/torch.html
  # torch.linalg.norm — docs: https://docs.pytorch.org/docs/main/linalg.html

  # attn_weights: [B, H, T, T]
  # value_states: [B, H, T, head_dim]
  # We want: output[b, h, d] = Σ_t attn[b, h, last_token, t] * V[b, h, t, d]

  attn_last   = attn_weights[:, :, -1, :]          # [B, H, T] — last token row
  # einsum: 'bht,bhtd->bhd' — contract over T (token) dimension
  weighted_v  = torch.einsum('bht,bhtd->bhd',
                              attn_last, value_states)    # [B, H, head_dim]

  # L2 norm over head_dim gives the "size" of what the head writes
  # torch.linalg.norm with dim=-1 is the Euclidean norm per head
  ov_norms    = torch.linalg.norm(weighted_v, dim=-1)    # [B, H]
  ov_norms_np = ov_norms[0].detach().cpu().numpy()       # [H]

WHY
  torch.einsum compiles to a single cuBLAS batched gemv call via the einops
  backend. The bmm + reshape approach requires two extra contiguous() calls
  and obscures the mathematical intent. torch.linalg.norm(dim=-1) is the
  canonical per-vector norm call.
  API sources:
    torch.einsum:       https://docs.pytorch.org/docs/main/generated/torch.einsum.html
    torch.linalg.norm:  https://docs.pytorch.org/docs/main/linalg.html

VERIFY
  # Confirm output shape
  assert ov_norms_np.shape == (H,)
  # OV norms should be non-negative
  assert (ov_norms_np >= 0).all()
  # Manual check for head 0:
  manual = (attn_last[0, 0, :].unsqueeze(-1) * value_states[0, 0]).sum(dim=0).norm()
  assert abs(ov_norms[0, 0].item() - manual.item()) < 1e-4
```

---

## PyTorch API quick-reference for this project

Sourced from: https://docs.pytorch.org/docs/main/pytorch-api.html

| Use case | Module | Key functions used in Chronoscope |
|---|---|---|
| Differencing | `torch` | `torch.diff`, `torch.cumsum` |
| Statistics | `torch` | `torch.mean`, `torch.var`, `torch.std` |
| Moments | `torch` | `.pow(2)`, `.pow(4)`, `.mean(dim)` |
| Information theory | `torch.special` | `torch.special.entr` |
| Spectral analysis | `torch.fft` | `torch.fft.rfft`, `torch.fft.irfft` |
| Complex tensors | `torch` | `torch.polar`, `torch.angle`, `torch.abs` |
| Linear algebra | `torch.linalg` | `torch.linalg.norm`, `torch.linalg.svdvals` |
| Hook-based capture | `torch.nn` | `module.register_forward_hook` |
| GPU timing | `torch.cuda` | `torch.cuda.Event`, `.record()`, `.elapsed_time()` |
| Static graph tracing | `torch.fx` | `torch.fx.symbolic_trace` |
| Distribution divergence | `torch.nn.functional` | `F.kl_div`, `F.log_softmax`, `F.softmax` |
| Contraction / matmul | `torch` | `torch.einsum` |
| Conditional select | `torch` | `torch.where` |

---

## Global DO-NOTs across all directives

```
DO NOT use scipy.stats for anything computable with torch.special or torch moments.
DO NOT use np.fft.* — use torch.fft.rfft / irfft instead.
DO NOT use np.linalg.norm on GPU tensors — use torch.linalg.norm, then .cpu().numpy().
DO NOT call torch.cuda.synchronize() inside a hook — it stalls the GPU pipeline.
DO NOT return a modified tensor from a capture-only hook — return None.
DO NOT use the deprecated torch.norm() — use torch.linalg.norm() (PyTorch ≥1.9).
DO NOT use torch.autograd.grad for scalar statistics — it is for gradient computation only.
```

---

*PyTorch API reference: https://docs.pytorch.org/docs/main/pytorch-api.html*  
*Key modules: torch · torch.fft · torch.linalg · torch.special · torch.cuda · torch.fx · torch.nn.functional*
