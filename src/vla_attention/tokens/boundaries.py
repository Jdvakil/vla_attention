"""Token-boundary parser.

The central question of this project is: "When an action token is being
predicted, how much of its attention budget goes to each modality?" Answering
that requires knowing exactly which positions in the sequence correspond to
which modality.

For MolmoAct the canonical order (per-frame) is:

    [<bos>] [vision_patches] [depth_tokens] [language_tokens] [action_tokens]

For OpenVLA the order is:

    [<bos>] [vision_patches] [language_tokens] [action_tokens]

We expose a single ``build_boundaries`` helper that produces a
``TokenBoundaries`` object containing non-overlapping ``ModalitySpan``s.
Spans are half-open ``[start, end)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

Modality = Literal["bos", "visual", "depth", "language", "action"]


@dataclass(frozen=True)
class ModalitySpan:
    modality: Modality
    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start

    def indices(self) -> np.ndarray:
        return np.arange(self.start, self.end, dtype=np.int64)

    def __contains__(self, idx: int) -> bool:
        return self.start <= idx < self.end


@dataclass
class TokenBoundaries:
    spans: list[ModalitySpan]

    @property
    def total_length(self) -> int:
        return max(s.end for s in self.spans) if self.spans else 0

    def span(self, modality: Modality) -> ModalitySpan:
        for s in self.spans:
            if s.modality == modality:
                return s
        raise KeyError(f"No span for modality={modality!r}")

    def has(self, modality: Modality) -> bool:
        return any(s.modality == modality for s in self.spans)

    def modality_of(self, idx: int) -> Modality:
        for s in self.spans:
            if idx in s:
                return s.modality
        raise IndexError(f"Index {idx} outside any span (seq_len={self.total_length})")

    def mask(self, modalities: Iterable[Modality]) -> np.ndarray:
        """Boolean mask over [total_length] selecting the listed modalities."""
        want = set(modalities)
        m = np.zeros(self.total_length, dtype=bool)
        for s in self.spans:
            if s.modality in want:
                m[s.start : s.end] = True
        return m

    def indices(self, modality: Modality) -> np.ndarray:
        return self.span(modality).indices()

    # ---- integrity checks -------------------------------------------------

    def validate(self) -> None:
        prev_end = 0
        for s in self.spans:
            if s.start < prev_end:
                raise ValueError(f"Span {s} overlaps previous end {prev_end}")
            if s.end < s.start:
                raise ValueError(f"Span {s} has end<start")
            prev_end = s.end


def build_boundaries(
    *,
    family: str,
    n_visual: int,
    n_depth: int,
    n_language: int,
    n_action: int,
    prepend_bos: bool = True,
) -> TokenBoundaries:
    """Build a ``TokenBoundaries`` for the given family + per-modality token
    counts. Only two families are handled: ``molmoact`` and ``openvla``.

    Raises ``ValueError`` for an unknown family or negative counts.
    """
    if min(n_visual, n_depth, n_language, n_action) < 0:
        raise ValueError("Token counts must be non-negative")

    spans: list[ModalitySpan] = []
    cursor = 0

    if prepend_bos:
        spans.append(ModalitySpan("bos", 0, 1))
        cursor = 1

    def add(mod: Modality, n: int) -> None:
        nonlocal cursor
        if n == 0:
            return
        spans.append(ModalitySpan(mod, cursor, cursor + n))
        cursor += n

    if family == "molmoact":
        add("visual", n_visual)
        add("depth", n_depth)
        add("language", n_language)
        add("action", n_action)
    elif family == "openvla":
        add("visual", n_visual)
        add("language", n_language)
        add("action", n_action)
    else:
        raise ValueError(f"Unknown family={family!r}")

    tb = TokenBoundaries(spans=spans)
    tb.validate()
    return tb
