# Chronoscope Causal Validity Report
**Generated:** 2026-03-25 11:27:49
**Model:** gpt2
---
## Verdict: **PARTIALLY GROUNDED (review required)**
**Composite Validity Score:** 0.4677

## 1. Input & Model Output
**Prompt:**
```
All cats are animals. Whiskers is a cat. Therefore, Whiskers is a
```
**Generated:**
```
 cat.

Whiskers is a cat. Therefore, Whiskers is a cat.

Whiskers is a cat. Therefore, Whiskers is a cat.

Whiskers is a cat. Therefore, Wh
```

## 2. Observer Trace
- Tokens: 68
- Seasonal Period: 2 tokens
- **Logical Persistence (Hurst): 0.532**

![Dynamics](plots/dynamics.png)
![Decomposition](plots/decomposition.png)
![Spectral](plots/spectral.png)

## 5. Topological Data Analysis
![Persistence](plots/persistence.png)

For an interactive 3D view of the reasoning trajectory (PC0–PC2), open **[trajectory_3d.html](trajectory_3d.html)** in a browser.

## 8. Validity Score Breakdown
- **dtw_sensitivity**: 1.0000
- **spectral_coherence**: 0.1334
- **topological_smoothness**: 0.0128
- **active_reasoning**: 0.4393
### **FINAL SCORE: 0.4677**

---
## 9. Kinetic-Anchored Interpretation
**Kinetic Anchor:** Token 57 (`57:Ċ`)

The 'reasoning velocity' spiked at token index 57 ('57:Ċ').

Explain what this 'kinetic shift' implies about the reasoning manifold's transition at this point.

REPORT SUBSET:

# Chronoscope Causal Validity Report

**Generated:** 2026-03-25 11:27:49

**Model:** gpt2

---

## Verdict: **PARTIALLY GROUNDED (review required)**

**Composite Validity Score:** 0.4677

## 1. Input & Model Output

**Prompt:**

```

All cats are animals. Whiskers is a cat. Therefore, Whiskers is a cat.

Whiskers is a cat. Therefore, Whiskers is a cat.

Whiskers is a cat. Therefore, Whiskers is a cat.

Whiskers is a cat. Therefore, Whiskers is a cat.

Whiskers is a cat. Therefore, Whiskers is a cat.

Whiskers is a cat. Therefore, Whiskers is a cat.
