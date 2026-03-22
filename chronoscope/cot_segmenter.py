"""
CoT Step Segmenter — Gap D.1

Segments generated text into reasoning steps using heuristic boundary
detection, then maps character boundaries back to token indices.

This module is fully standalone and has no runtime dependencies on the
rest of Chronoscope — it takes raw strings/token lists and returns
structured step objects.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class ReasoningStep:
    step_index: int
    token_start: int
    token_end: int       # exclusive
    text: str
    n_tokens: int


# ── Boundary detection patterns ───────────────────────────────────────────

_EXPLICIT_MARKERS = re.compile(
    r'(?:^|\n)(?:Step\s+\d+[:.]\s*|'
    r'First[,.]?\s+|Second[,.]?\s+|Third[,.]?\s+|'
    r'Next[,.]?\s+|Then[,.]?\s+|Finally[,.]?\s+|'
    r'Therefore[,.]?\s+|Thus[,.]?\s+|'
    r'In conclusion[,.]?\s+|'
    r'So[,.]?\s+)',
    re.IGNORECASE | re.MULTILINE
)

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def segment_cot_by_text(
    generated_text: str,
    generated_tokens: List[str],
) -> List[ReasoningStep]:
    """
    Segment generated text into reasoning steps using heuristic boundary
    detection.

    Boundaries are detected at:
        1. Explicit step markers: 'Step 1:', 'First,', 'Next,', 'Therefore,', etc.
        2. Sentence-ending punctuation followed by whitespace + capital letter
        3. Newlines preceding non-empty content

    Args:
        generated_text:   full generated string
        generated_tokens: list of decoded tokens (length == number of generated tokens)

    Returns:
        list of ReasoningStep — each covering a span of tokens
    """
    # ── Collect character-level boundary positions ──────────────────────────
    boundary_chars = {0}

    for m in _EXPLICIT_MARKERS.finditer(generated_text):
        boundary_chars.add(m.start())

    for m in _SENTENCE_BOUNDARY.finditer(generated_text):
        boundary_chars.add(m.start())

    boundary_chars.add(len(generated_text))
    boundaries = sorted(boundary_chars)

    # ── Map character offsets → token indices ───────────────────────────────
    char_offsets = []
    cumulative = 0
    for tok in generated_tokens:
        char_offsets.append(cumulative)
        cumulative += len(tok)
    char_offsets.append(cumulative)

    def char_to_token(char_pos: int) -> int:
        """Return the token index that contains char_pos."""
        for ti in range(len(char_offsets) - 1):
            if char_offsets[ti + 1] > char_pos:
                return ti
        return len(generated_tokens) - 1

    # ── Build structured step list ─────────────────────────────────────────
    steps: List[ReasoningStep] = []
    for start_char, end_char in zip(boundaries[:-1], boundaries[1:]):
        text_segment = generated_text[start_char:end_char].strip()
        if not text_segment:
            continue

        tok_start = char_to_token(start_char)
        tok_end = min(char_to_token(end_char), len(generated_tokens))
        if tok_end <= tok_start:
            continue

        steps.append(ReasoningStep(
            step_index=len(steps),
            token_start=tok_start,
            token_end=tok_end,
            text=text_segment,
            n_tokens=tok_end - tok_start,
        ))

    return steps


def aggregate_entropy_by_step(
    head_entropy_series: np.ndarray,
    steps: List[ReasoningStep],
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
        end = min(step.token_end, head_entropy_series.shape[0])
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
