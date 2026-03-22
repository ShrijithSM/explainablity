# Chronoscope — Implementation Update: Gap Closure (Green → Amber Nodes)

**Document type:** Coding-agent implementation brief  
**Scope:** Gaps A, B, C, D, E — all teal (immediate) and amber (design-required) solutions  
**Do not implement:** Purple (research-grade) or coral (paradigm-shift) nodes  
**Base codebase:** `chronoscope/` — interceptor.py, observer.py, analyzer.py, synthesizer.py  
**Target model:** Qwen/Qwen2.5-0.5B (14 heads, 24 layers, hidden_dim=896)

---

## Table of Contents

1. [Gap A — Stationarity handling before VAR](#gap-a)
2. [Gap B — Statistical significance of influence scores](#gap-b)
3. [Gap C — Perturbation / intervention design](#gap-c)
4. [Gap D — Thinking time axis operationalization](#gap-d)
5. [Gap E — Signal quality beyond entropy](#gap-e)
6. [New module layout](#new-module-layout)
7. [Dependency additions](#dependency-additions)
8. [Integration checklist](#integration-checklist)

---

## Gap A — Stationarity handling before VAR {#gap-a}

### A.1 — Per-head ADF testing (teal)

**Problem:** The existing code runs ADF on the aggregate PC0 of the residual stream, not on each of the 14 individual head entropy series. VAR stationarity requirements apply per-series.

**Location to modify:** `chronoscope/analyzer.py` — `head_interaction_analysis()` method.

**Implementation:**

```python
# chronoscope/analyzer.py
# ADD: import at top of file
from statsmodels.tsa.stattools import adfuller

def _test_per_head_stationarity(self, metric_series: np.ndarray) -> dict:
    """
    Run ADF test on each of the 14 head entropy series independently.

    Args:
        metric_series: np.ndarray of shape [T, H] — T tokens, H heads

    Returns:
        dict with keys:
            'p_values':       np.ndarray shape [H]   — ADF p-value per head
            'is_stationary':  np.ndarray shape [H]   — bool, True if p < 0.05
            'needs_diff':     bool                   — True if ANY head is non-stationary
            'diff_mask':      np.ndarray shape [H]   — which heads need differencing
    """
    T, H = metric_series.shape
    p_values = np.zeros(H)
    for h in range(H):
        series = metric_series[:, h]
        # maxlag=None lets statsmodels use AIC selection
        adf_result = adfuller(series, maxlag=None, autolag='AIC')
        p_values[h] = adf_result[1]  # index 1 is the p-value

    is_stationary = p_values < 0.05
    diff_mask = ~is_stationary  # heads that need differencing

    return {
        'p_values':      p_values,
        'is_stationary': is_stationary,
        'needs_diff':    bool(diff_mask.any()),
        'diff_mask':     diff_mask,
    }
```

---

### A.2 — Selective first-differencing (teal)

**Problem:** Non-stationary heads are fed raw into VAR. All differencing should be selective (only the non-stationary heads), not applied globally.

**Implementation:**

```python
# chronoscope/analyzer.py
# ADD inside head_interaction_analysis() BEFORE the VAR fit block

def _apply_selective_differencing(
    self,
    metric_series: np.ndarray,
    diff_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    First-difference only the heads flagged as non-stationary.
    Trim one row from ALL heads so the matrix stays rectangular.

    Args:
        metric_series: np.ndarray [T, H]
        diff_mask:     np.ndarray [H] bool — True = difference this head

    Returns:
        differenced_series: np.ndarray [T-1, H]
        diff_mask:          np.ndarray [H] — passed through for downstream use
    """
    T, H = metric_series.shape
    out = np.empty((T - 1, H), dtype=np.float64)

    for h in range(H):
        if diff_mask[h]:
            out[:, h] = np.diff(metric_series[:, h])   # s(t) - s(t-1)
        else:
            out[:, h] = metric_series[1:, h]           # trim first row only

    return out, diff_mask
```

**Integration point:** In `head_interaction_analysis()`, replace the existing direct VAR fit block with:

```python
# chronoscope/analyzer.py — head_interaction_analysis() body

metric_series = self.interceptor.get_head_metric_series(layer_name)
series_np = metric_series.numpy()  # shape [T, 14]

# A.1 — per-head stationarity
stationarity_report = self._test_per_head_stationarity(series_np)

# A.2 — selective differencing
if stationarity_report['needs_diff']:
    series_for_var, diff_mask = self._apply_selective_differencing(
        series_np, stationarity_report['diff_mask']
    )
else:
    series_for_var = series_np
    diff_mask = stationarity_report['diff_mask']

# store on self for downstream reporting
self._last_stationarity = stationarity_report
self._last_diff_mask = diff_mask
```

---

### A.3 — Johansen cointegration test → VECM if needed (amber)

**Problem:** If multiple heads share a common stochastic trend (cointegration), first-differencing destroys the long-run equilibrium information. A VECM is more appropriate.

**New dependency:** `statsmodels` already present; uses `statsmodels.tsa.vector_ar.vecm.VECM`.

**Implementation:**

```python
# chronoscope/analyzer.py

from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

def _check_cointegration(
    self,
    series_np: np.ndarray,
    max_lags: int = 3
) -> dict:
    """
    Run Johansen cointegration test on the [T, H] head entropy matrix.

    Args:
        series_np: np.ndarray [T, H]
        max_lags:  lag order for the test (should match VAR lag order)

    Returns:
        dict with keys:
            'cointegrated':    bool — True if at least one cointegrating vector found
            'n_coint_vectors': int  — number of cointegrating relationships (rank r)
            'trace_stats':     np.ndarray — trace statistics per rank
            'crit_values_95':  np.ndarray — 95% critical values per rank
    """
    # det_order=0: no trend in cointegrating relation (appropriate for entropy)
    # k_ar_diff: number of lagged differences = max_lags - 1
    result = coint_johansen(series_np, det_order=0, k_ar_diff=max_lags - 1)

    # Johansen trace test: reject H0(rank <= r) if trace_stat > crit_95
    crit_95 = result.cvt[:, 1]        # column 1 = 95% critical value
    trace_stats = result.lr1           # trace statistics
    rank = int(np.sum(trace_stats > crit_95))

    return {
        'cointegrated':    rank > 0,
        'n_coint_vectors': rank,
        'trace_stats':     trace_stats,
        'crit_values_95':  crit_95,
    }


def _fit_vecm(
    self,
    series_np: np.ndarray,
    rank: int,
    max_lags: int = 3
) -> object:
    """
    Fit a VECM when cointegration is detected.
    Returns fitted VECM result; calling code extracts the influence matrix.

    Args:
        series_np: np.ndarray [T, H] — raw (undifferenced) series
        rank:      cointegration rank from Johansen test
        max_lags:  lag order

    Returns:
        statsmodels VECMResults object
    """
    model = VECM(series_np, k_ar_diff=max_lags - 1, coint_rank=rank, deterministic='ci')
    return model.fit()
```

**Decision logic to add inside `head_interaction_analysis()`:**

```python
# chronoscope/analyzer.py — inside head_interaction_analysis()
# Run AFTER stationarity check, BEFORE VAR fit

USE_VECM = False

if stationarity_report['needs_diff']:
    coint_report = self._check_cointegration(series_np, max_lags=lag)
    self._last_coint = coint_report

    if coint_report['cointegrated']:
        # Use VECM — preserve long-run relationships
        USE_VECM = True
        vecm_result = self._fit_vecm(series_np, rank=coint_report['n_coint_vectors'], max_lags=lag)
        # VECM short-run coefficients are in vecm_result.gamma
        # shape: [k_ar_diff * H, H] — reshape to [lags, H, H]
        H = series_np.shape[1]
        gamma = vecm_result.gamma  # [k_ar_diff * H, H]
        n_lags = gamma.shape[0] // H
        coef_tensor = gamma.reshape(n_lags, H, H)
        influence = np.abs(coef_tensor).sum(axis=0)  # [H, H]
    else:
        # Cointegration absent — safe to difference and use VAR
        series_for_var, diff_mask = self._apply_selective_differencing(
            series_np, stationarity_report['diff_mask']
        )

if not USE_VECM:
    # Standard VAR path on (possibly differenced) series
    from statsmodels.tsa.vector_ar.var_model import VAR
    model = VAR(series_for_var)
    # Use AIC lag selection — do not hard-code lag=3
    result = model.fit(maxlags=lag, ic='aic')
    influence = np.abs(result.coefs).sum(axis=0)  # [H, H]
    self._last_var_result = result
```

---

### A.4 — AIC-based lag selection (teal, minor fix)

**Problem:** Lag `p=3` is hardcoded without justification. Replace with information-criterion selection.

**Change in `head_interaction_analysis()`:**

```python
# BEFORE (hardcoded):
result = model.fit(maxlags=3, ic=None)

# AFTER (AIC selection, capped at 5 to avoid overfitting on T=100):
result = model.fit(maxlags=min(lag, 5), ic='aic')
# result.k_ar now holds the selected lag order — log it
self._selected_lag = result.k_ar
```

---

### A.5 — KPSS + ADF joint stationarity reporting (amber)

**Problem:** ADF alone cannot distinguish between a unit root and a near-unit-root with high persistence. Running KPSS alongside provides a second opinion. Conflicting results (ADF non-stationary, KPSS also non-stationary) = likely fractional integration.

**New dependency:** None — `statsmodels.tsa.stattools.kpss` is already in statsmodels.

**Implementation:**

```python
# chronoscope/analyzer.py

from statsmodels.tsa.stattools import kpss

def _joint_stationarity_test(self, metric_series: np.ndarray) -> dict:
    """
    Run both ADF and KPSS on each head series.
    Interpret joint results to flag fractional integration.

    Interpretation table:
        ADF reject (p<0.05) + KPSS fail-to-reject → stationary
        ADF fail-to-reject  + KPSS reject         → non-stationary (unit root)
        ADF reject          + KPSS reject          → FRACTIONALLY INTEGRATED — flag for Hurst analysis
        ADF fail-to-reject  + KPSS fail-to-reject  → ambiguous — use Hurst exponent

    Args:
        metric_series: np.ndarray [T, H]

    Returns:
        dict per head with diagnosis string and raw statistics
    """
    T, H = metric_series.shape
    results = []

    for h in range(H):
        s = metric_series[:, h]

        adf_pval = adfuller(s, autolag='AIC')[1]
        # KPSS: nlags='auto', regression='c' (constant, no trend)
        kpss_stat, kpss_pval, _, _ = kpss(s, regression='c', nlags='auto')

        adf_stationary  = adf_pval < 0.05    # ADF rejects unit root
        kpss_stationary = kpss_pval > 0.05   # KPSS fails to reject stationarity

        if adf_stationary and kpss_stationary:
            diagnosis = 'stationary'
        elif not adf_stationary and not kpss_stationary:
            diagnosis = 'unit_root'
        elif adf_stationary and not kpss_stationary:
            diagnosis = 'fractional'   # both tests contradict — long memory
        else:
            diagnosis = 'ambiguous'

        results.append({
            'head':           h,
            'adf_pval':       float(adf_pval),
            'kpss_pval':      float(kpss_pval),
            'adf_stationary': adf_stationary,
            'kpss_stationary':kpss_stationary,
            'diagnosis':      diagnosis,
        })

    # heads flagged as fractional should use Hurst analysis (already in observer.py)
    # rather than differencing
    fractional_heads = [r['head'] for r in results if r['diagnosis'] == 'fractional']

    return {
        'per_head':         results,
        'fractional_heads': fractional_heads,
        'summary': {d: sum(1 for r in results if r['diagnosis']==d)
                    for d in ('stationary','unit_root','fractional','ambiguous')},
    }
```

**Integration:** Call `_joint_stationarity_test()` and store result on `self._last_joint_stationarity`. Log `fractional_heads` to the synthesizer report. Do NOT difference fractional heads — note them in the report for Hurst-based analysis.

---

## Gap B — Statistical significance of influence scores {#gap-b}

### B.1 — Granger F-test per directed pair (teal)

**Problem:** The current influence matrix is a heuristic sum of `|Aₗ[i][j]|`. No p-values are attached. Any ranking of influence pairs is unverified.

**Implementation:**

```python
# chronoscope/analyzer.py

def _granger_pvalue_matrix(
    self,
    var_result,
    series_np: np.ndarray,
    max_lag: int
) -> np.ndarray:
    """
    Compute Granger causality F-test p-value for every directed pair (j→i).

    Uses statsmodels VAR.test_causality(caused=i, causing=j).

    Args:
        var_result:  fitted statsmodels VARResults object
        series_np:   np.ndarray [T, H] — the series used to fit var_result
        max_lag:     lag order used

    Returns:
        pval_matrix: np.ndarray [H, H] — pval_matrix[i, j] = p-value for j→i
                     diagonal is NaN (no self-causality test)
    """
    H = series_np.shape[1]
    pval_matrix = np.full((H, H), np.nan)

    for i in range(H):
        for j in range(H):
            if i == j:
                continue
            try:
                test = var_result.test_causality(
                    caused=i,
                    causing=j,
                    kind='f'     # F-test, not chi2
                )
                pval_matrix[i, j] = test.pvalue
            except Exception:
                # Can happen when series have zero variance after differencing
                pval_matrix[i, j] = 1.0

    return pval_matrix
```

---

### B.2 — Benjamini-Hochberg FDR correction (teal)

**Problem:** 14×14 = 196 tests without multiple-comparison correction inflates false positives severely. Bonferroni is too conservative for correlated tests. BH-FDR is appropriate.

**Implementation:**

```python
# chronoscope/analyzer.py

from statsmodels.stats.multitest import multipletests

def _apply_fdr_correction(
    self,
    pval_matrix: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Apply Benjamini-Hochberg FDR correction to the flattened p-value matrix.

    Args:
        pval_matrix: np.ndarray [H, H] with NaN on diagonal
        alpha:       target FDR level (default 0.05)

    Returns:
        dict with keys:
            'reject_matrix':    np.ndarray [H, H] bool — True = significant after FDR
            'pval_corrected':   np.ndarray [H, H] — BH-adjusted p-values
            'n_significant':    int
            'significant_pairs': list of (j, i, corrected_pval) tuples sorted by pval
    """
    H = pval_matrix.shape[0]

    # Flatten, tracking indices of non-NaN entries
    flat_pvals = []
    flat_indices = []
    for i in range(H):
        for j in range(H):
            if i != j and not np.isnan(pval_matrix[i, j]):
                flat_pvals.append(pval_matrix[i, j])
                flat_indices.append((i, j))

    reject, pvals_corrected, _, _ = multipletests(
        flat_pvals, alpha=alpha, method='fdr_bh'
    )

    # Reconstruct matrices
    reject_matrix   = np.zeros((H, H), dtype=bool)
    corrected_matrix = np.full((H, H), np.nan)

    for k, (i, j) in enumerate(flat_indices):
        reject_matrix[i, j]    = reject[k]
        corrected_matrix[i, j] = pvals_corrected[k]

    significant_pairs = [
        (j, i, corrected_matrix[i, j])
        for (i, j) in flat_indices
        if reject_matrix[i, j]
    ]
    significant_pairs.sort(key=lambda x: x[2])  # sort by corrected p-value ascending

    return {
        'reject_matrix':     reject_matrix,
        'pval_corrected':    corrected_matrix,
        'n_significant':     int(reject_matrix.sum()),
        'significant_pairs': significant_pairs,   # (source_j, target_i, pval)
    }
```

**Integration inside `head_interaction_analysis()`:**

```python
# After fitting VAR and computing influence matrix:
pval_matrix    = self._granger_pvalue_matrix(result, series_for_var, self._selected_lag)
fdr_result     = self._apply_fdr_correction(pval_matrix, alpha=0.05)

# Mask influence matrix: zero out non-significant pairs
masked_influence = influence.copy()
masked_influence[~fdr_result['reject_matrix']] = 0.0

self._last_pval_matrix   = pval_matrix
self._last_fdr_result    = fdr_result
self._masked_influence   = masked_influence

# Replace the old influence ranking with the masked version
# Only significant pairs appear in the top-N report
```

---

### B.3 — Bootstrap surrogate null distribution (teal)

**Problem:** Granger F-test assumes Gaussian errors. For non-Gaussian entropy signals, an empirical null via phase-scrambled surrogates is more reliable.

**Implementation:**

```python
# chronoscope/analyzer.py

def _bootstrap_surrogate_pvalues(
    self,
    series_np: np.ndarray,
    observed_influence: np.ndarray,
    n_surrogates: int = 500,
    lag: int = 3
) -> np.ndarray:
    """
    Phase-scramble each head series independently to break temporal structure
    while preserving marginal distribution. Refit VAR on each surrogate.
    Return empirical p-value matrix: fraction of surrogates >= observed score.

    IMPORTANT: This is computationally expensive (500 VAR fits).
    Only call when n_tokens >= 100 and n_surrogates can be lowered to 200
    for quick runs by passing n_surrogates=200.

    Args:
        series_np:          np.ndarray [T, H] — original series
        observed_influence: np.ndarray [H, H] — influence matrix from real data
        n_surrogates:       number of phase-scrambled resamples
        lag:                VAR lag order to use in surrogates

    Returns:
        empirical_pval: np.ndarray [H, H] — fraction of surrogates >= observed
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    T, H = series_np.shape
    surrogate_scores = np.zeros((n_surrogates, H, H))

    rng = np.random.default_rng(seed=42)

    for s in range(n_surrogates):
        surrogate = np.empty_like(series_np)
        for h in range(H):
            # Phase scramble: FFT → randomize phases → IFFT
            fft_vals = np.fft.rfft(series_np[:, h])
            random_phases = rng.uniform(0, 2 * np.pi, len(fft_vals))
            # Preserve phase at DC (index 0) and Nyquist (if even T)
            random_phases[0] = 0.0
            fft_scrambled = np.abs(fft_vals) * np.exp(1j * random_phases)
            surrogate[:, h] = np.fft.irfft(fft_scrambled, n=T)

        try:
            surr_model = VAR(surrogate)
            surr_result = surr_model.fit(maxlags=lag, ic='aic')
            surrogate_scores[s] = np.abs(surr_result.coefs).sum(axis=0)
        except Exception:
            # If VAR fails on a surrogate (rare), fill with observed so it
            # doesn't artificially lower the p-value
            surrogate_scores[s] = observed_influence

    # empirical p-value[i,j] = fraction of surrogates where score >= observed
    empirical_pval = np.mean(surrogate_scores >= observed_influence[None, :, :], axis=0)
    return empirical_pval
```

**Integration note:** Run bootstrap only when `config.enable_bootstrap_surrogates = True` (add to config.py). Default to False — it adds ~30s per analysis at 500 surrogates on CPU.

```python
# chronoscope/config.py — ADD field:
enable_bootstrap_surrogates: bool = False
n_bootstrap_surrogates: int = 500
```

---

### B.4 — Conditional transfer entropy (amber)

**Problem:** Granger causality is transfer entropy under Gaussian assumption. For non-Gaussian head signals, model-free transfer entropy is more accurate and does not require stationarity.

**New dependency:** `pip install pyinform`  (or implement via KSG estimator)

**Implementation:**

```python
# chronoscope/analyzer.py  — NEW method

def _conditional_transfer_entropy(
    self,
    series_np: np.ndarray,
    k: int = 3,
    discretize_bins: int = 8
) -> np.ndarray:
    """
    Compute conditional transfer entropy TE(j→i) for all pairs.
    TE(j→i) = I(X_i(t+1) ; X_j^{1..k} | X_i^{1..k})

    Uses histogram-based estimation after uniform discretization.
    For continuous signals, this is an approximation — increase
    discretize_bins to 16 or use KSG estimator for better accuracy.

    Args:
        series_np:       np.ndarray [T, H]
        k:               history length (embedding dimension), default 3
        discretize_bins: number of bins for uniform discretization

    Returns:
        te_matrix: np.ndarray [H, H] — te_matrix[i,j] = TE(j→i) in nats
                   higher = j's past has more information about i's future
                   beyond i's own past
    """
    T, H = series_np.shape

    # Discretize each head signal into bins
    def discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
        bins = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
        return np.digitize(x, bins) - 1  # 0-indexed

    disc = np.stack(
        [discretize(series_np[:, h], discretize_bins) for h in range(H)],
        axis=1
    )  # [T, H]

    te_matrix = np.zeros((H, H))

    for i in range(H):
        for j in range(H):
            if i == j:
                continue

            # Build (k+1)-gram histories
            # x_i_future: X_i(t+1),  x_i_past: X_i(t-k..t),  x_j_past: X_j(t-k..t)
            counts = {}
            for t in range(k, T - 1):
                x_i_future = int(disc[t + 1, i])
                x_i_past   = tuple(disc[t - k:t, i].tolist())
                x_j_past   = tuple(disc[t - k:t, j].tolist())
                key = (x_i_future, x_i_past, x_j_past)
                counts[key] = counts.get(key, 0) + 1

            total = sum(counts.values())
            if total == 0:
                continue

            # TE = H(X_i_future | X_i_past) - H(X_i_future | X_i_past, X_j_past)
            # Computed directly from joint counts via chain rule
            te = 0.0
            for (xif, xip, xjp), cnt in counts.items():
                # p(xif, xip, xjp)
                p_joint = cnt / total
                # marginals via summing
                p_xip_xjp = sum(
                    v for (a, b, c), v in counts.items() if b == xip and c == xjp
                ) / total
                p_xif_xip = sum(
                    v for (a, b, c), v in counts.items() if a == xif and b == xip
                ) / total
                p_xip = sum(
                    v for (a, b, c), v in counts.items() if b == xip
                ) / total

                if p_joint > 0 and p_xip_xjp > 0 and p_xif_xip > 0 and p_xip > 0:
                    te += p_joint * np.log(
                        (p_joint * p_xip) / (p_xip_xjp * p_xif_xip)
                    )

            te_matrix[i, j] = max(te, 0.0)  # clip numerical negatives

    return te_matrix
```

**Integration note:** Add `config.use_transfer_entropy: bool = False`. When True, compute `te_matrix` alongside `influence` and store on `self._last_te_matrix`. Include both in the synthesizer report under separate sections. TE matrix is a richer signal but slower (~10s for 14×14 at T=100).

---

### B.5 — Partial directed coherence — frequency-domain Granger (amber)

**Problem:** Influence may be frequency-specific. The period-2 token cycle found in FFT may carry different causality from lower-frequency trends. PDC decomposes Granger causality by frequency.

**Implementation:**

```python
# chronoscope/analyzer.py — NEW method

def _partial_directed_coherence(
    self,
    var_result,
    freqs: np.ndarray = None
) -> dict:
    """
    Compute Partial Directed Coherence (PDC) from fitted VAR coefficients.
    PDC(j→i, f) indicates directed influence from j to i at frequency f.

    Formula:
        A(f) = I - Σ_l A_l * exp(-2πi f l)    (spectral matrix)
        PDC(j→i, f) = |A(f)[i,j]|² / Σ_k |A(f)[k,j]|²

    Args:
        var_result: fitted statsmodels VARResults
        freqs:      np.ndarray of normalized frequencies in [0, 0.5]
                    default: 128 points from 0 to 0.5

    Returns:
        dict with keys:
            'pdc':   np.ndarray [F, H, H] — PDC(f, i, j) = j→i influence at freq f
            'freqs': np.ndarray [F]
    """
    if freqs is None:
        freqs = np.linspace(0, 0.5, 128)

    coefs = var_result.coefs   # shape [p, H, H] — A_l matrices
    p, H, _ = coefs.shape

    pdc = np.zeros((len(freqs), H, H))

    for fi, f in enumerate(freqs):
        # Build spectral matrix A(f) = I - Σ_l A_l exp(-2πi f l)
        A_f = np.eye(H, dtype=complex)
        for l in range(p):
            A_f -= coefs[l] * np.exp(-2j * np.pi * f * (l + 1))

        # PDC[i,j] = |A_f[i,j]|² / Σ_k |A_f[k,j]|²
        A_abs_sq = np.abs(A_f) ** 2
        col_sums = A_abs_sq.sum(axis=0, keepdims=True)  # [1, H]
        col_sums[col_sums == 0] = 1.0  # avoid div by zero
        pdc[fi] = A_abs_sq / col_sums

    # Integrate PDC over frequency bands
    # band: [0, 0.1] = very low freq (slow trends)
    # band: [0.4, 0.5] = high freq including period-2 cycle (f=0.5)
    low_mask  = freqs <= 0.1
    high_mask = freqs >= 0.4

    pdc_low  = pdc[low_mask].mean(axis=0)   # [H, H]
    pdc_high = pdc[high_mask].mean(axis=0)  # [H, H]

    return {
        'pdc':       pdc,         # [F, H, H]
        'freqs':     freqs,
        'pdc_low':   pdc_low,     # average PDC in low-freq band
        'pdc_high':  pdc_high,    # average PDC in high-freq band
        'dominant_freq_per_pair': np.argmax(pdc, axis=0) / len(freqs) * 0.5,  # [H, H]
    }
```

**Integration note:** Store `pdc_result` on `self._last_pdc`. Add PDC heatmap (pdc_low and pdc_high side by side) to synthesizer output as `pdc_heatmap.png`.

---

## Gap C — Perturbation / intervention design {#gap-c}

### C.1 — Zero-ablation (teal, replaces Gaussian noise)

**Problem:** The current perturbation replaces head output with `torch.randn_like(hidden)` — this injects random energy into the residual stream rather than removing the head's contribution cleanly.

**Location:** Exp6 patching hook, and any perturbation code in `analyzer.py`.

**Implementation:**

```python
# chronoscope/analyzer.py  (or wherever the patch hook is defined)

import torch

def _make_ablation_hook(
    self,
    h_source: int,
    head_dim: int,
    mode: str = 'zero'         # 'zero' | 'mean' | 'gaussian' (legacy)
) -> callable:
    """
    Return a forward hook that ablates head h_source from the concatenated
    head output tensor in the residual stream.

    Supported modes:
        'zero':     replace head slice with zeros (clean null — recommended)
        'mean':     replace with per-position mean activation (neutral baseline)
        'gaussian': legacy — adds noise, NOT recommended for causal inference

    Args:
        h_source:  index of the head to ablate (0-indexed)
        head_dim:  dimension of each head's output (hidden_dim // n_heads)
        mode:      ablation strategy

    Returns:
        hook: callable suitable for module.register_forward_hook()
    """
    start = h_source * head_dim
    end   = (h_source + 1) * head_dim

    def hook(module, input, output):
        hidden = output[0].clone()   # [batch, seq_len, hidden_dim]

        if mode == 'zero':
            hidden[:, :, start:end] = 0.0

        elif mode == 'mean':
            # compute mean over sequence position for this head slice
            mean_val = hidden[:, :, start:end].mean(dim=1, keepdim=True)
            hidden[:, :, start:end] = mean_val.expand_as(hidden[:, :, start:end])

        elif mode == 'gaussian':
            # Legacy — kept for backward compatibility only
            hidden[:, :, start:end] = torch.randn_like(hidden[:, :, start:end])

        return (hidden,) + output[1:]

    return hook
```

**Configuration change:**

```python
# chronoscope/config.py — ADD field:
perturbation_mode: str = 'zero'   # 'zero' | 'mean' | 'gaussian'
```

---

### C.2 — Mean ablation with reference corpus (teal)

**Problem:** Zero-ablation sets a head to zero which is outside the model's training distribution. Mean ablation sets it to the average activation across a reference set — a "generic" head rather than an absent one.

**Implementation:**

```python
# chronoscope/analyzer.py

def _build_mean_ablation_cache(
    self,
    reference_prompts: list[str],
    layer_name: str,
    h_source: int,
    head_dim: int
) -> torch.Tensor:
    """
    Run the model on reference_prompts, collect head h_source activations,
    return per-position mean tensor for use in mean-ablation hook.

    Args:
        reference_prompts: list of plain text strings (5-20 prompts recommended)
        layer_name:        e.g. 'model.layers.23.self_attn'
        h_source:          head index to cache mean for
        head_dim:          hidden_dim // n_heads

    Returns:
        mean_activation: torch.Tensor [1, max_seq_len, head_dim]
                         padded with last value if sequences differ in length
    """
    start = h_source * head_dim
    end   = (h_source + 1) * head_dim
    collected = []

    def collect_hook(module, input, output):
        hidden = output[0]   # [1, seq_len, hidden_dim]
        collected.append(hidden[:, :, start:end].detach().cpu())

    handle = self._attach_hook_by_name(layer_name, collect_hook)

    for prompt in reference_prompts:
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            self.model(**inputs)

    handle.remove()

    # Pad all collected tensors to the max sequence length
    max_len = max(t.shape[1] for t in collected)
    padded = []
    for t in collected:
        pad_size = max_len - t.shape[1]
        if pad_size > 0:
            last_val = t[:, -1:, :].expand(-1, pad_size, -1)
            t = torch.cat([t, last_val], dim=1)
        padded.append(t)

    mean_activation = torch.stack(padded, dim=0).mean(dim=0)  # [1, max_len, head_dim]
    return mean_activation
```

**Integration note:** Cache is built once before perturbation experiments. Add `config.reference_prompts: list[str]` with 10 diverse default prompts covering factual recall, arithmetic, and prose generation.

---

### C.3 — Activation patching / path patching (teal)

**Problem:** Current perturbation runs generation from scratch with the head ablated. Activation patching is more precise: run a clean forward pass and a corrupted forward pass, then patch activations from one into the other at a specific site.

**Implementation:**

```python
# chronoscope/analyzer.py

def _activation_patch_experiment(
    self,
    clean_prompt: str,
    corrupted_prompt: str,
    layer_name: str,
    h_source: int,
    head_dim: int
) -> dict:
    """
    Activation patching: run CLEAN prompt, capture head h_source activations.
    Run CORRUPTED prompt normally. At the target layer, patch in the CLEAN
    head activations. Measure change in output logits vs corrupted baseline.

    This answers: "how much does head j's specific activation pattern
    (from the clean run) change the model's output on the corrupted run?"

    Args:
        clean_prompt:     prompt where head j has the behavior you want to study
        corrupted_prompt: prompt where that behavior is absent or different
        layer_name:       attention module to patch
        h_source:         head index to patch
        head_dim:         hidden_dim // n_heads

    Returns:
        dict with keys:
            'clean_logits':     torch.Tensor — output logits on clean prompt
            'corrupted_logits': torch.Tensor — output logits on corrupted prompt
            'patched_logits':   torch.Tensor — output logits after patching clean→corrupted
            'patch_effect':     float — KL(patched || corrupted) — how much patch moved output
            'restoration':      float — fraction of clean-corrupted gap restored by patch
                                         1.0 = full restoration, 0.0 = patch had no effect
    """
    start = h_source * head_dim
    end   = (h_source + 1) * head_dim

    # Step 1: clean forward pass, capture head activations
    clean_head_acts = {}

    def capture_clean_hook(module, input, output):
        clean_head_acts['hidden'] = output[0][:, :, start:end].detach()

    handle_clean = self._attach_hook_by_name(layer_name, capture_clean_hook)
    clean_inputs  = self.tokenizer(clean_prompt,     return_tensors='pt').to(self.device)
    corr_inputs   = self.tokenizer(corrupted_prompt, return_tensors='pt').to(self.device)

    with torch.no_grad():
        clean_out = self.model(**clean_inputs)
    clean_logits = clean_out.logits[:, -1, :]
    handle_clean.remove()

    # Step 2: corrupted forward pass (no hooks) — baseline
    with torch.no_grad():
        corr_out = self.model(**corr_inputs)
    corrupted_logits = corr_out.logits[:, -1, :]

    # Step 3: patched forward pass — patch clean activations into corrupted run
    def patch_hook(module, input, output):
        hidden = output[0].clone()
        # Align sequence lengths: patch as many positions as possible
        patch_len = min(hidden.shape[1], clean_head_acts['hidden'].shape[1])
        hidden[:, :patch_len, start:end] = clean_head_acts['hidden'][:, :patch_len, :]
        return (hidden,) + output[1:]

    handle_patch = self._attach_hook_by_name(layer_name, patch_hook)
    with torch.no_grad():
        patched_out = self.model(**corr_inputs)
    patched_logits = patched_out.logits[:, -1, :]
    handle_patch.remove()

    # Compute patch effect as KL divergence
    import torch.nn.functional as F
    clean_probs    = F.softmax(clean_logits,    dim=-1)
    corr_probs     = F.softmax(corrupted_logits, dim=-1)
    patched_probs  = F.softmax(patched_logits,   dim=-1)

    kl_baseline = F.kl_div(corr_probs.log(),    clean_probs,   reduction='sum').item()
    kl_patched  = F.kl_div(patched_probs.log(),  clean_probs,  reduction='sum').item()

    # Restoration: how much of the gap between corrupted and clean is closed
    restoration = 1.0 - (kl_patched / kl_baseline) if kl_baseline > 0 else 0.0

    return {
        'clean_logits':     clean_logits,
        'corrupted_logits': corrupted_logits,
        'patched_logits':   patched_logits,
        'patch_effect':     float(F.kl_div(
            corrupted_logits.log_softmax(-1),
            patched_logits.softmax(-1),
            reduction='sum'
        ).item()),
        'restoration':      float(np.clip(restoration, 0.0, 1.0)),
        'kl_baseline':      float(kl_baseline),
        'kl_patched':       float(kl_patched),
    }
```

---

### C.4 — Direct vs total effect decomposition (amber)

**Problem:** Head j may influence head i directly, OR via intermediary head k. The current perturbation cannot distinguish these paths. Decomposing direct vs total effect requires patching subsets of heads.

**Implementation:**

```python
# chronoscope/analyzer.py

def _direct_vs_total_effect(
    self,
    clean_prompt: str,
    corrupted_prompt: str,
    layer_name: str,
    source_head: int,
    target_head: int,
    mediator_heads: list[int],
    head_dim: int
) -> dict:
    """
    Estimate direct effect of source_head on target_head,
    controlling for mediation through mediator_heads.

    Method:
        Total effect    = patch(source) only, measure target entropy change
        Indirect effect = patch(source + mediators), measure target entropy change
        Direct effect   = Total - Indirect

    Args:
        source_head:    head j (the putative cause)
        target_head:    head i (the putative effect)
        mediator_heads: list of intermediate heads k to control for
        head_dim:       hidden_dim // n_heads

    Returns:
        dict with keys:
            'total_effect':    float — entropy change in target from patching source alone
            'indirect_effect': float — entropy change with source+mediators patched
            'direct_effect':   float — total - indirect
            'mediation_ratio': float — indirect / total (fraction mediated)
    """
    def _run_with_patches(heads_to_patch: list[int]) -> float:
        """
        Run corrupted prompt with listed heads patched from clean run.
        Returns entropy of target_head in the patched run.
        """
        target_start = target_head * head_dim
        target_end   = (target_head + 1) * head_dim

        # Capture clean activations for all heads to patch
        clean_acts = {}

        def capture(module, input, output):
            for h in heads_to_patch:
                s, e = h * head_dim, (h + 1) * head_dim
                clean_acts[h] = output[0][:, :, s:e].detach()

        handle_c = self._attach_hook_by_name(layer_name, capture)
        inp = self.tokenizer(clean_prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            self.model(**inp)
        handle_c.remove()

        # Run corrupted with patches applied, capture target entropy
        target_entropies = []

        def patch_and_capture(module, input, output):
            hidden = output[0].clone()
            for h in heads_to_patch:
                s, e = h * head_dim, (h + 1) * head_dim
                patch_len = min(hidden.shape[1], clean_acts[h].shape[1])
                hidden[:, :patch_len, s:e] = clean_acts[h][:, :patch_len, :]
            # Capture target head entropy from this pass
            target_slice = hidden[:, :, target_start:target_end]
            # Approximate entropy from activation norm variance as proxy
            target_entropies.append(target_slice.norm(dim=-1).mean().item())
            return (hidden,) + output[1:]

        handle_p = self._attach_hook_by_name(layer_name, patch_and_capture)
        corr_inp = self.tokenizer(corrupted_prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            self.model(**corr_inp)
        handle_p.remove()

        return float(np.mean(target_entropies)) if target_entropies else 0.0

    # Baseline: corrupted run, no patches
    baseline_target = _run_with_patches([])  # no patches = corrupted baseline

    # Total effect: patch source only
    total_patched    = _run_with_patches([source_head])
    total_effect     = abs(total_patched - baseline_target)

    # Indirect effect: patch source + mediators
    indirect_patched  = _run_with_patches([source_head] + mediator_heads)
    indirect_effect   = abs(indirect_patched - baseline_target)

    direct_effect     = max(total_effect - indirect_effect, 0.0)
    mediation_ratio   = (indirect_effect / total_effect) if total_effect > 0 else 0.0

    return {
        'total_effect':    total_effect,
        'indirect_effect': indirect_effect,
        'direct_effect':   direct_effect,
        'mediation_ratio': float(np.clip(mediation_ratio, 0.0, 1.0)),
        'source_head':     source_head,
        'target_head':     target_head,
        'mediator_heads':  mediator_heads,
    }
```

---

## Gap D — Thinking time axis {#gap-d}

### D.1 — CoT step segmentation (teal)

**Problem:** VAR is fitted on raw token index as time axis. For reasoning analysis, the meaningful time unit is a reasoning step, not a token.

**New module:** `chronoscope/cot_segmenter.py`

```python
# chronoscope/cot_segmenter.py — NEW FILE

import re
import numpy as np
from dataclasses import dataclass

@dataclass
class ReasoningStep:
    step_index:   int
    token_start:  int
    token_end:    int          # exclusive
    text:         str
    n_tokens:     int


def segment_cot_by_text(
    generated_text: str,
    generated_tokens: list[str],
) -> list[ReasoningStep]:
    """
    Segment generated text into reasoning steps using heuristic boundary detection.

    Boundaries are detected at:
        1. Explicit step markers: 'Step 1:', 'First,', 'Next,', 'Therefore,', etc.
        2. Sentence-ending punctuation followed by whitespace + capital letter
        3. Newlines that precede non-empty content

    Args:
        generated_text:   full generated string
        generated_tokens: list of decoded tokens (length == number of generated tokens)

    Returns:
        list of ReasoningStep — each covering a span of tokens
    """
    # Boundary marker patterns
    explicit_markers = re.compile(
        r'(?:^|\n)(?:Step\s+\d+[:.]\s*|'
        r'First[,.]?\s+|Second[,.]?\s+|Third[,.]?\s+|'
        r'Next[,.]?\s+|Then[,.]?\s+|Finally[,.]?\s+|'
        r'Therefore[,.]?\s+|Thus[,.]?\s+|'
        r'In conclusion[,.]?\s+|'
        r'So[,.]?\s+)',
        re.IGNORECASE | re.MULTILINE
    )

    sentence_boundary = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Find character positions of all boundaries
    boundary_chars = set()
    boundary_chars.add(0)

    for m in explicit_markers.finditer(generated_text):
        boundary_chars.add(m.start())

    for m in sentence_boundary.finditer(generated_text):
        boundary_chars.add(m.start())

    boundary_chars.add(len(generated_text))
    boundaries = sorted(boundary_chars)

    # Map character positions to token positions
    # Build cumulative character offset per token
    char_offsets = []
    cumulative = 0
    for tok in generated_tokens:
        char_offsets.append(cumulative)
        cumulative += len(tok)
    char_offsets.append(cumulative)

    def char_to_token(char_pos: int) -> int:
        """Return the token index that contains char_pos."""
        for ti, co in enumerate(char_offsets[:-1]):
            if char_offsets[ti + 1] > char_pos:
                return ti
        return len(generated_tokens) - 1

    steps = []
    for si, (start_char, end_char) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        text_segment = generated_text[start_char:end_char].strip()
        if not text_segment:
            continue
        tok_start = char_to_token(start_char)
        tok_end   = min(char_to_token(end_char), len(generated_tokens))
        if tok_end <= tok_start:
            continue
        steps.append(ReasoningStep(
            step_index  = len(steps),
            token_start = tok_start,
            token_end   = tok_end,
            text        = text_segment,
            n_tokens    = tok_end - tok_start,
        ))

    return steps


def aggregate_entropy_by_step(
    head_entropy_series: np.ndarray,
    steps: list[ReasoningStep],
    agg: str = 'mean'
) -> np.ndarray:
    """
    Aggregate per-token head entropy into per-step entropy.

    Args:
        head_entropy_series: np.ndarray [T, H]
        steps:               list of ReasoningStep
        agg:                 aggregation method — 'mean' | 'max' | 'last'

    Returns:
        step_entropy: np.ndarray [S, H] — S = number of steps, H = heads
    """
    S = len(steps)
    H = head_entropy_series.shape[1]
    step_entropy = np.zeros((S, H))

    for si, step in enumerate(steps):
        start = max(step.token_start, 0)
        end   = min(step.token_end, head_entropy_series.shape[0])
        if end <= start:
            step_entropy[si] = step_entropy[si - 1] if si > 0 else 0.0
            continue
        slice_ = head_entropy_series[start:end, :]
        if agg == 'mean':
            step_entropy[si] = slice_.mean(axis=0)
        elif agg == 'max':
            step_entropy[si] = slice_.max(axis=0)
        elif agg == 'last':
            step_entropy[si] = slice_[-1, :]

    return step_entropy
```

**Integration in `analyzer.py`:**

```python
# chronoscope/analyzer.py — ADD inside head_interaction_analysis()
# when config.use_cot_time_axis == True:

from chronoscope.cot_segmenter import segment_cot_by_text, aggregate_entropy_by_step

steps = segment_cot_by_text(
    generated_text   = self._last_generated_text,
    generated_tokens = self._last_generated_tokens
)

if len(steps) >= 5:  # need at least 5 steps for VAR to be meaningful
    step_entropy = aggregate_entropy_by_step(metric_series_np, steps, agg='mean')
    # Rerun entire A-gap pipeline on step_entropy instead of metric_series_np
    # VAR on step_entropy: time axis = reasoning step, not token
    self._last_step_entropy  = step_entropy
    self._last_steps         = steps
```

**Config addition:**

```python
# chronoscope/config.py
use_cot_time_axis:   bool = False   # if True, segment tokens into CoT steps before VAR
cot_prompt_prefix:   str  = "Let's think step by step."
```

---

### D.2 — Topological phase boundaries as clock ticks (teal)

**Problem:** Chronoscope already computes Euler characteristic spikes (topological anomalies). These spikes are natural phase boundaries. Use them as the time axis instead of tokens.

**Location:** `chronoscope/synthesizer.py` and `chronoscope/analyzer.py`

**Implementation:**

```python
# chronoscope/analyzer.py — NEW method

def _segment_by_topological_phases(
    self,
    euler_characteristic_series: np.ndarray,
    spike_threshold_std: float = 2.0
) -> list[tuple[int, int]]:
    """
    Use Euler characteristic spikes (already computed by Chronoscope TDA pipeline)
    as phase boundaries. Returns token-index spans of each phase.

    Args:
        euler_characteristic_series: np.ndarray [T] — EC at each token step
        spike_threshold_std:         number of std deviations above mean to flag as spike

    Returns:
        phases: list of (start_token, end_token) tuples — exclusive end
    """
    mean_ec  = euler_characteristic_series.mean()
    std_ec   = euler_characteristic_series.std()
    threshold = mean_ec + spike_threshold_std * std_ec

    # Spike positions = token indices where |EC| > threshold
    spike_positions = np.where(np.abs(euler_characteristic_series) > threshold)[0]

    # Build phase spans between consecutive spikes
    boundaries = [0] + spike_positions.tolist() + [len(euler_characteristic_series)]
    boundaries = sorted(set(boundaries))

    phases = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start >= 3:   # minimum 3 tokens per phase to avoid degenerate VAR
            phases.append((int(start), int(end)))

    return phases


def _aggregate_entropy_by_phase(
    self,
    metric_series_np: np.ndarray,
    phases: list[tuple[int, int]]
) -> np.ndarray:
    """
    Aggregate head entropy series by topological phase.
    Returns [n_phases, H] array — one entropy value per phase per head.
    """
    H = metric_series_np.shape[1]
    phase_entropy = np.zeros((len(phases), H))

    for pi, (start, end) in enumerate(phases):
        phase_entropy[pi] = metric_series_np[start:end, :].mean(axis=0)

    return phase_entropy
```

**Integration note:** When `euler_characteristic_series` is available from the TDA pipeline (it should be in `self.observer._euler_series` or equivalent), extract phases and store `self._last_phases`. Run a second VAR on `phase_entropy` and store as `self._phase_level_var`. Report both token-level and phase-level influence matrices.

---

### D.3 — Trajectory curvature as intrinsic time (teal)

**Problem:** Token index is coordinate time — external and uniform. Arc length along the residual stream is intrinsic time — it measures how much the model's representation is genuinely changing per step.

**Location:** `chronoscope/observer.py`

**Implementation:**

```python
# chronoscope/observer.py — ADD method

def compute_intrinsic_time(
    self,
    hidden_states: np.ndarray,   # [T, hidden_dim] — residual stream trajectory
    normalize: bool = True
) -> dict:
    """
    Compute arc-length intrinsic time from the residual stream trajectory.

    Arc length at step t:
        ds(t) = ||h(t) - h(t-1)||_2

    Cumulative intrinsic time:
        tau(t) = Σ_{s=1}^{t} ds(s)

    Curvature at step t (discrete approximation):
        kappa(t) = ||h(t+1) - 2*h(t) + h(t-1)||_2 / ||h(t) - h(t-1)||_2^2
        (numerically stabilized — denominator clamped to 1e-8)

    Args:
        hidden_states: np.ndarray [T, D]
        normalize:     if True, scale tau to [0, 1]

    Returns:
        dict with keys:
            'arc_length_steps':  np.ndarray [T-1] — ds(t) per step
            'intrinsic_time':    np.ndarray [T]   — tau(t), cumulative arc length
            'curvature':         np.ndarray [T-2] — kappa(t) per interior step
            'high_curvature_idx':np.ndarray       — token indices with kappa > 2*std
            'velocity':          np.ndarray [T-1] — alias for arc_length_steps
    """
    T, D = hidden_states.shape

    # Arc length steps (velocity)
    diff = np.diff(hidden_states, axis=0)              # [T-1, D]
    arc_steps = np.linalg.norm(diff, axis=1)           # [T-1]

    # Cumulative intrinsic time
    tau = np.concatenate([[0.0], np.cumsum(arc_steps)])  # [T]
    if normalize and tau[-1] > 0:
        tau = tau / tau[-1]

    # Discrete curvature: second derivative / first derivative squared
    second_diff = np.diff(hidden_states, n=2, axis=0)  # [T-2, D]
    second_norms = np.linalg.norm(second_diff, axis=1)  # [T-2]
    first_norms  = arc_steps[:-1]                        # [T-2]
    curvature = second_norms / np.maximum(first_norms ** 2, 1e-8)

    # High curvature positions
    kappa_mean = curvature.mean()
    kappa_std  = curvature.std()
    high_curv_idx = np.where(curvature > kappa_mean + 2 * kappa_std)[0] + 1  # offset by 1

    return {
        'arc_length_steps':   arc_steps,
        'intrinsic_time':     tau,
        'curvature':          curvature,
        'high_curvature_idx': high_curv_idx,
        'velocity':           arc_steps,
    }
```

**Integration:** Call `compute_intrinsic_time()` in `observer.py` after collecting hidden states. Store on `self._intrinsic_time`. Pass `high_curvature_idx` to the phase segmentation as an alternative set of phase boundaries (in addition to Euler characteristic spikes).

---

### D.4 — Velocity-gated resampling (amber)

**Problem:** Feeding VAR a series where some tokens represent 100x more cognitive work than others introduces heteroskedasticity. Resampling at equal arc-length intervals (equal cognitive effort) produces a more homogeneous series for VAR.

**Implementation:**

```python
# chronoscope/analyzer.py — NEW method

def _resample_at_equal_arc_length(
    self,
    metric_series_np: np.ndarray,
    hidden_states: np.ndarray,
    n_resampled_points: int = 80
) -> np.ndarray:
    """
    Resample the head entropy series from equal token-spacing
    to equal arc-length spacing (equal cognitive effort per step).

    This removes heteroskedasticity caused by fast (high curvature) vs
    slow (low curvature) processing phases having unequal representation.

    Args:
        metric_series_np:    np.ndarray [T, H] — original head entropy series
        hidden_states:       np.ndarray [T, D] — residual stream for arc-length
        n_resampled_points:  number of equal-arc-length samples to produce
                             (should be < T; default 80 for T=100)

    Returns:
        resampled: np.ndarray [n_resampled_points, H]
    """
    from scipy.interpolate import interp1d

    T = hidden_states.shape[0]

    # Compute cumulative arc length (unnormalized)
    diff = np.diff(hidden_states, axis=0)           # [T-1, D]
    arc_steps = np.linalg.norm(diff, axis=1)        # [T-1]
    cumulative_arc = np.concatenate([[0.0], np.cumsum(arc_steps)])  # [T]

    # Define equal-arc-length query points
    total_arc   = cumulative_arc[-1]
    query_arcs  = np.linspace(0, total_arc, n_resampled_points)

    # Interpolate each head's entropy at query arc positions
    resampled = np.zeros((n_resampled_points, metric_series_np.shape[1]))

    for h in range(metric_series_np.shape[1]):
        # interp1d with 'linear' is sufficient; 'cubic' is smoother but risky at boundaries
        f = interp1d(
            cumulative_arc,
            metric_series_np[:, h],
            kind='linear',
            fill_value='extrapolate'
        )
        resampled[:, h] = f(query_arcs)

    return resampled
```

**Config addition:**

```python
# chronoscope/config.py
use_arc_length_resampling: bool = False  # resample to equal cognitive effort before VAR
arc_length_n_points:       int  = 80
```

---

### D.5 — Hidden Markov phase discovery (amber)

**Problem:** Reasoning phases are not known in advance and may not align with text boundaries. An HMM can discover latent cognitive phases directly from the entropy signal.

**New dependency:** `pip install hmmlearn`

**Implementation:**

```python
# chronoscope/analyzer.py — NEW method

def _discover_phases_hmm(
    self,
    metric_series_np: np.ndarray,
    n_states: int = 4,
    n_iter: int = 200
) -> dict:
    """
    Fit a Gaussian HMM to the [T, H] head entropy series to discover
    latent cognitive phases. Each hidden state corresponds to a different
    pattern of head coordination.

    Args:
        metric_series_np: np.ndarray [T, H]
        n_states:         number of HMM states to fit (default 4)
                          tune via BIC: try 2-6, pick the elbow
        n_iter:           EM iterations

    Returns:
        dict with keys:
            'state_sequence':    np.ndarray [T] — Viterbi-decoded state per token
            'phase_spans':       list of (state, start, end) tuples
            'transition_matrix': np.ndarray [n_states, n_states]
            'state_means':       np.ndarray [n_states, H] — mean entropy per state per head
            'bic':               float — Bayesian information criterion (lower = better)
            'per_head_dominant_state': np.ndarray [H] — which state most activates each head
    """
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states,
        covariance_type='diag',    # diagonal covariance is faster and avoids overfitting
        n_iter=n_iter,
        random_state=42
    )
    model.fit(metric_series_np)

    state_sequence = model.predict(metric_series_np)   # [T]

    # Build phase spans: consecutive runs of the same state
    phase_spans = []
    start = 0
    for t in range(1, len(state_sequence)):
        if state_sequence[t] != state_sequence[t - 1]:
            phase_spans.append((int(state_sequence[t - 1]), start, t))
            start = t
    phase_spans.append((int(state_sequence[-1]), start, len(state_sequence)))

    # BIC: -2 * log_likelihood + n_params * log(T)
    log_likelihood = model.score(metric_series_np)
    T, H = metric_series_np.shape
    # n_params: transition probs + means + variances + initial probs
    n_params = n_states * (n_states - 1) + n_states * H * 2 + n_states
    bic = -2 * log_likelihood + n_params * np.log(T)

    # Per-head: which state has the highest mean entropy for that head
    per_head_dominant = np.argmax(model.means_, axis=0)  # [H]

    return {
        'state_sequence':          state_sequence,
        'phase_spans':             phase_spans,
        'transition_matrix':       model.transmat_,
        'state_means':             model.means_,
        'bic':                     float(bic),
        'per_head_dominant_state': per_head_dominant,
        'n_states_used':           n_states,
    }
```

**Integration note:** After running HMM phase discovery, run a separate VAR per discovered phase. Log phase-level influence matrices under `self._phase_var_results` as a dict keyed by state index. This enables "does head causality change between cognitive phases?" to be answered directly.

---

## Gap E — Signal quality {#gap-e}

### E.1 — Multiple entropy variants (teal)

**Problem:** Shannon entropy is the only metric tracked. Rényi entropy and effective rank capture different aspects of attention concentration.

**Location:** `chronoscope/interceptor.py` — the `hook_fn` entropy calculation.

**Implementation:**

```python
# chronoscope/interceptor.py — expand hook_fn to compute multiple metrics

def _compute_head_metrics(
    self,
    attn_weights: torch.Tensor,   # [Batch, H, T, T]
) -> dict:
    """
    Compute multiple per-head attention summary statistics for the most
    recent token position.

    Metrics computed:
        shannon_entropy:  H = -Σ p log p                (current default)
        renyi_entropy_2:  H_2 = -log(Σ p²)             (collision entropy, less sensitive to small weights)
        max_attention:    max(p_i)                       (sharpness — high = very focused)
        effective_rank:   exp(H_shannon) / T             (normalized — 1.0 = uniform, 1/T = single token)
        sink_fraction:    fraction of weight on token 0  (attention sink diagnostic)

    Args:
        attn_weights: attention weight tensor [B, H, T, T]

    Returns:
        dict mapping metric_name → np.ndarray [H]
    """
    eps = 1e-9
    last = attn_weights[:, :, -1, :]    # most recent token: [B, H, T]
    probs = last.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    p_np = probs[0].detach().cpu().numpy()   # [H, T]
    H_heads, T_ctx = p_np.shape

    # Shannon entropy
    log_p = np.log(p_np + eps)
    shannon = -(p_np * log_p).sum(axis=1)    # [H]

    # Rényi entropy α=2
    renyi_2 = -np.log((p_np ** 2).sum(axis=1) + eps)  # [H]

    # Max attention (peak weight)
    max_attn = p_np.max(axis=1)              # [H]

    # Effective rank: exp(H) / T_ctx (normalized)
    eff_rank = np.exp(shannon) / max(T_ctx, 1)  # [H]

    # Sink fraction: how much weight is on position 0
    sink_frac = p_np[:, 0]                   # [H]

    return {
        'shannon_entropy': shannon,
        'renyi_entropy_2': renyi_2,
        'max_attention':   max_attn,
        'effective_rank':  eff_rank,
        'sink_fraction':   sink_frac,
    }
```

**Storage change:** Replace `self._head_metrics[name].append(entropy)` with `self._head_metrics[name].append(metrics_dict)`. Downstream code must be updated to select metric by name.

**Config addition:**

```python
# chronoscope/config.py
head_metric: str = 'shannon_entropy'   # 'shannon_entropy' | 'renyi_entropy_2' | 'effective_rank'
```

---

### E.2 — Attention sink separation (teal)

**Problem:** Qwen2.5 (like most modern transformers) concentrates excess attention on the BOS/first token as a "sink." This artificially deflates entropy and creates correlated structure across all heads that has nothing to do with semantic processing.

**Implementation:**

```python
# chronoscope/interceptor.py — ADD inside _compute_head_metrics()
# or as a preprocessing step on the [T, H] series

def _remove_attention_sink(
    self,
    attn_weights_np: np.ndarray,   # [H, T] for the current token
    sink_positions: list[int] = None
) -> np.ndarray:
    """
    Remove the attention sink from each head's attention distribution
    and re-normalize the remaining weights.

    Sink positions are typically [0] (BOS token) in Qwen/Llama models.
    If sink_positions is None, auto-detect as positions where mean weight
    across all heads exceeds 3x the uniform expectation (1/T).

    Args:
        attn_weights_np: np.ndarray [H, T]
        sink_positions:  list of token indices to treat as sinks

    Returns:
        cleaned: np.ndarray [H, T] — sink positions zeroed, renormalized
    """
    H, T = attn_weights_np.shape

    if sink_positions is None:
        # Auto-detect: positions with mean cross-head weight > 3/T
        mean_weights = attn_weights_np.mean(axis=0)   # [T]
        uniform_expected = 1.0 / T
        sink_positions = list(np.where(mean_weights > 3 * uniform_expected)[0])

    if not sink_positions:
        return attn_weights_np   # no sinks detected

    cleaned = attn_weights_np.copy()
    cleaned[:, sink_positions] = 0.0

    # Renormalize — avoid div by zero for heads that attended only to sinks
    row_sums = cleaned.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
    cleaned = cleaned / row_sums

    return cleaned
```

**Config addition:**

```python
# chronoscope/config.py
remove_attention_sink:     bool      = True
attention_sink_positions:  list[int] = None   # None = auto-detect per forward pass
```

---

### E.3 — OV circuit value-weighted attention (teal)

**Problem:** Entropy measures where the head looks (Q×K pattern). What actually flows through the residual stream is Q×K × V — the value-weighted output. Two heads with identical entropy can have completely different effects on the residual stream.

**Implementation:**

```python
# chronoscope/interceptor.py — NEW method

def _compute_ov_weighted_metric(
    self,
    attn_weights: torch.Tensor,    # [B, H, T, T]
    value_states: torch.Tensor,    # [B, H, T, head_dim]
) -> np.ndarray:
    """
    Compute the L2 norm of the value-weighted attention output per head
    at the most recent token position. This captures WHAT flows, not WHERE
    the head looks.

    OV_output[h] = ||Σ_t attn[h, last, t] * V[h, t, :]||_2

    Args:
        attn_weights: [B, H, T, T]
        value_states: [B, H, T, head_dim]

    Returns:
        ov_norms: np.ndarray [H] — output norm per head at last token position
    """
    # attn_weights[:, :, -1, :] = attention from last token to all past: [B, H, T]
    # value_states[:, :, :, :]  = values for all past positions:         [B, H, T, d]

    # Weighted sum: [B, H, 1, T] × [B, H, T, d] → [B, H, 1, d]
    attn_last = attn_weights[:, :, -1:, :]              # [B, H, 1, T]
    weighted_v = torch.matmul(attn_last, value_states)  # [B, H, 1, d]
    weighted_v = weighted_v.squeeze(2)                   # [B, H, d]

    ov_norms = weighted_v.norm(dim=-1)   # [B, H]
    return ov_norms[0].detach().cpu().numpy()  # [H]
```

**Hook modification:** The hook currently receives `output[1]` (attention weights). To access value states, the hook needs to intercept the full attention module output or use `output_attentions=True` alongside registering a hook on the `v_proj` layer. The simplest approach for Qwen2.5:

```python
# In interceptor.py setup — register BOTH an attention hook AND a v_proj hook
# per layer. Store v_proj output in self._value_cache[layer_name] temporarily,
# then consume it in the attention hook to compute OV metric.

def _attach_ov_hooks(self, layer_name: str):
    v_proj_name = layer_name.replace('self_attn', 'self_attn.v_proj')
    # Register v_proj output hook to cache value states
    # Register self_attn hook to combine with attn_weights
    # This requires two hooks per layer — see existing hook registration pattern
    pass  # implement following the existing self._hooks dict pattern
```

---

### E.4 — Functional decomposition per head (amber)

**Problem:** Individual heads perform specific functions (induction, copying, inhibition, positional). Tracking aggregate entropy collapses all these functions into one number. Projecting head output onto known functional directions yields a richer, interpretable time series.

**Implementation:**

```python
# chronoscope/analyzer.py — NEW method

def _project_onto_functional_directions(
    self,
    head_output_series: np.ndarray,   # [T, H, head_dim]
    function_directions: dict,         # name → np.ndarray [head_dim]
) -> dict:
    """
    Project each head's output at each token onto a set of known functional
    direction vectors in head_dim space.

    Functional directions can be obtained by:
        - Training linear probes for known behaviors (induction, copying)
        - Taking SVD of head outputs on known-function datasets
        - Using difference-in-means between two contrasting prompt sets

    Args:
        head_output_series:  np.ndarray [T, H, head_dim]
        function_directions: dict mapping function name → unit vector [head_dim]

    Returns:
        projections: dict mapping function name → np.ndarray [T, H]
                     each entry is the scalar projection (dot product) per
                     token per head onto that functional direction
    """
    projections = {}
    for func_name, direction in function_directions.items():
        direction_unit = direction / (np.linalg.norm(direction) + 1e-9)
        # projection[t, h] = dot(head_output[t, h, :], direction_unit)
        proj = np.einsum('thd,d->th', head_output_series, direction_unit)  # [T, H]
        projections[func_name] = proj

    return projections


def _build_induction_direction(
    self,
    induction_prompts: list[str],
    anti_induction_prompts: list[str],
    layer_name: str
) -> np.ndarray:
    """
    Estimate the 'induction' functional direction in head_dim space by
    taking the mean-difference vector between head outputs on induction
    vs anti-induction prompts.

    Induction prompt example: 'A B A B A B' — repeating pattern
    Anti-induction prompt:    'A B C D E F' — non-repeating

    Returns:
        direction: np.ndarray [head_dim] — unit vector pointing toward induction behavior
    """
    induction_outputs    = self._collect_head_outputs(induction_prompts,    layer_name)
    anti_induction_outputs = self._collect_head_outputs(anti_induction_prompts, layer_name)

    # Mean over tokens and prompts
    mean_induction     = induction_outputs.mean(axis=(0, 1))      # [H, head_dim]
    mean_anti_induction = anti_induction_outputs.mean(axis=(0, 1)) # [H, head_dim]

    # Aggregate over all heads (or compute per-head direction)
    direction = (mean_induction - mean_anti_induction).mean(axis=0)  # [head_dim]
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    return direction
```

---

### E.5 — Vector signal per head (amber)

**Problem:** Each head currently produces 1 scalar (entropy) per token. A richer 5-dimensional feature vector enables VAR to capture multi-faceted interactions.

**Implementation:**

```python
# chronoscope/interceptor.py — replace scalar entropy storage with vector storage

# In hook_fn, replace single entropy append with multi-feature vector:
def hook_fn(module, input, output):
    attn = output[1]   # [B, H, T, T]

    metrics = self._compute_head_metrics(attn)   # dict of [H] arrays

    # Stack into feature vector: [H, 5]
    feature_vector = np.stack([
        metrics['shannon_entropy'],
        metrics['renyi_entropy_2'],
        metrics['max_attention'],
        metrics['effective_rank'],
        metrics['sink_fraction'],
    ], axis=1)  # [H, 5]

    self._head_metrics[name].append(feature_vector)
```

**Downstream:** `get_head_metric_series()` now returns `[T, H, 5]` when using vector mode. VAR is then run on a `[T, H*5]` = `[T, 70]` reshaped matrix. With T=100, this is borderline for VAR stability — use `maxlags=1` for the 70-dimensional case, or run 5 separate univariate VAR fits (one per feature type) and intersect the significant pairs.

**Config addition:**

```python
# chronoscope/config.py
head_feature_mode: str = 'scalar'   # 'scalar' | 'vector'
# scalar: [T, H]   — compatible with current pipeline
# vector: [T, H*5] — richer but needs larger T and smaller lag
```

---

## New module layout {#new-module-layout}

```
chronoscope/
├── interceptor.py        MODIFIED — multi-metric hook, OV hooks, sink separation
├── observer.py           MODIFIED — intrinsic time, joint stationarity
├── analyzer.py           MODIFIED — all Gap A/B/C methods added
├── synthesizer.py        MODIFIED — render new metrics, PDC heatmap, phase report
├── config.py             MODIFIED — all new config flags listed below
├── cot_segmenter.py      NEW      — CoT step boundary detection (Gap D.1)
└── tests/
    ├── test_stationarity.py   NEW — unit tests for A.1, A.2, A.5
    ├── test_significance.py   NEW — unit tests for B.1, B.2
    └── test_perturbation.py   NEW — unit tests for C.1, C.3
```

---

## Dependency additions {#dependency-additions}

Add to `requirements.txt` or `pyproject.toml`:

```
# Gap A
statsmodels>=0.14.0    # already present — VECM uses same package

# Gap B
# statsmodels already covers multipletests and Granger F-test

# Gap D (HMM phase discovery — amber)
hmmlearn>=0.3.0

# Gap D (arc-length resampling)
scipy>=1.11.0           # already likely present for FFT

# Gap E (no new deps — all numpy/torch)
```

---

## Config additions summary {#config-additions}

All additions to `chronoscope/config.py`:

```python
# ── Gap A ──────────────────────────────────────────────────────────────────
var_lag_selection:           str       = 'aic'         # 'aic' | 'bic' | 'fixed'
var_max_lags:                int       = 5             # ceiling for IC selection
run_johansen_cointegration:  bool      = True
joint_stationarity_test:     bool      = True          # run both ADF + KPSS

# ── Gap B ──────────────────────────────────────────────────────────────────
granger_ftest:               bool      = True
fdr_alpha:                   float     = 0.05
enable_bootstrap_surrogates: bool      = False
n_bootstrap_surrogates:      int       = 500
use_transfer_entropy:        bool      = False         # amber — slow

# ── Gap C ──────────────────────────────────────────────────────────────────
perturbation_mode:           str       = 'zero'        # 'zero' | 'mean' | 'gaussian'
reference_prompts:           list[str] = [             # for mean ablation
    "The capital of France is Paris.",
    "Solve for x: 2x + 3 = 7.",
    "Once upon a time in a distant land,",
    "The mitochondria is the powerhouse of the cell.",
    "import numpy as np",
    "Water boils at 100 degrees Celsius.",
    "The French Revolution began in 1789.",
    "To be or not to be, that is the question.",
    "The quick brown fox jumps over the lazy dog.",
    "In mathematics, a prime number is divisible only by 1 and itself.",
]
run_activation_patching:     bool      = False         # requires clean+corrupted prompt pair
activation_patch_clean:      str       = ""
activation_patch_corrupted:  str       = ""

# ── Gap D ──────────────────────────────────────────────────────────────────
use_cot_time_axis:           bool      = False
cot_prompt_prefix:           str       = "Let's think step by step."
use_topological_phase_clock: bool      = True          # uses existing EC series
euler_spike_threshold_std:   float     = 2.0
use_arc_length_resampling:   bool      = False
arc_length_n_points:         int       = 80
use_hmm_phase_discovery:     bool      = False         # amber — requires hmmlearn
hmm_n_states:                int       = 4

# ── Gap E ──────────────────────────────────────────────────────────────────
head_metric:                 str       = 'shannon_entropy'
head_feature_mode:           str       = 'scalar'      # 'scalar' | 'vector'
remove_attention_sink:       bool      = True
attention_sink_positions:    list[int] = None           # None = auto-detect
compute_ov_metric:           bool      = False          # requires v_proj hook
```

---

## Integration checklist {#integration-checklist}

The following steps must be completed in order by the coding agent:

```
[ ] 1.  Add all config fields to config.py with their default values as listed above
[ ] 2.  Install hmmlearn (amber dep) — add to requirements.txt
[ ] 3.  Modify interceptor.py:
        - Replace single entropy computation with _compute_head_metrics()
        - Add _remove_attention_sink() as preprocessing in hook_fn
        - Add _compute_ov_weighted_metric() (conditional on config.compute_ov_metric)
        - Update get_head_metric_series() to select by config.head_metric
[ ] 4.  Modify observer.py:
        - Add compute_intrinsic_time() method
        - Call it after collecting hidden states; store as self._intrinsic_time
        - Add _joint_stationarity_test() (moved from analyzer or duplicated)
[ ] 5.  Create chronoscope/cot_segmenter.py (new file, full implementation above)
[ ] 6.  Modify analyzer.py — add in this order:
        a. Imports: adfuller, kpss, multipletests, coint_johansen, VECM
        b. _test_per_head_stationarity()
        c. _apply_selective_differencing()
        d. _check_cointegration()
        e. _fit_vecm()
        f. _joint_stationarity_test()
        g. _granger_pvalue_matrix()
        h. _apply_fdr_correction()
        i. _bootstrap_surrogate_pvalues()       (conditional on config flag)
        j. _conditional_transfer_entropy()       (conditional on config flag)
        k. _partial_directed_coherence()
        l. _make_ablation_hook()                 (replaces existing hook)
        m. _build_mean_ablation_cache()
        n. _activation_patch_experiment()
        o. _direct_vs_total_effect()
        p. _segment_by_topological_phases()
        q. _aggregate_entropy_by_phase()
        r. _resample_at_equal_arc_length()
        s. _discover_phases_hmm()
        t. _project_onto_functional_directions()
        u. Update head_interaction_analysis() control flow (A.4 lag selection
           → A.1 per-head ADF → A.5 KPSS → A.3 Johansen → A.2 differencing
           → A.3 VECM or VAR → B.1 Granger → B.2 FDR → B.5 PDC)
[ ] 7.  Modify synthesizer.py:
        - Add stationarity per-head table to causal_report.md
        - Add FDR-corrected significance table (replace raw influence ranking)
        - Add PDC heatmap (low-freq vs high-freq bands side by side)
        - Add intrinsic time plot (arc-length curve over tokens)
        - Add phase timeline (horizontal bar showing HMM states or EC phases)
        - Add new metrics section if head_feature_mode == 'vector'
[ ] 8.  Write tests/test_stationarity.py:
        - Test _test_per_head_stationarity() with known stationary/non-stationary signals
        - Test _apply_selective_differencing() shape invariants
        - Test _joint_stationarity_test() diagnosis labels
[ ] 9.  Write tests/test_significance.py:
        - Test _granger_pvalue_matrix() returns [H,H] with NaN diagonal
        - Test _apply_fdr_correction() rejects fewer pairs than raw p<0.05
[ ] 10. Write tests/test_perturbation.py:
        - Test _make_ablation_hook() zero mode sets target slice to zero
        - Test _make_ablation_hook() mean mode preserves tensor scale
        - Test _activation_patch_experiment() returns restoration in [0,1]
[ ] 11. Run existing experiments (exp1, exp3, exp6) with all defaults ON to verify
        no regression in existing behavior. Expected: influence matrix should change
        (fewer spurious links after FDR correction), synthesis report should be richer.
[ ] 12. Run exp6 with perturbation_mode='zero' and compare entropy change measurements
        against the legacy Gaussian results. Document the delta in comments.
```

---

*End of update.md — all implementations above are at teal (immediate) or amber (design-required) depth only. Purple and coral nodes are explicitly out of scope for this agent run.*
