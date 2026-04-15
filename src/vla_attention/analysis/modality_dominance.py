"""Experiment 1: layer-wise modality-dominance map.

For each of the N_layers, compute the mean attention mass that action tokens
allocate to each modality, aggregated across all heads, all steps, and all
rollouts. The result is a (N_layers, 3) heatmap.

Optionally break this out by task category (language_dominant vs.
vision_dominant) to see whether the attention-routing pattern shifts with
task demands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import SyntheticRollout


@dataclass
class ModalityDominanceResult:
    # Core result: per-layer, per-modality mean attention mass.
    layer_modality: np.ndarray                   # (n_layers, 3) [visual, lang, action_prev]

    # Standard deviation across rollouts (for error bars / significance).
    layer_modality_std: np.ndarray               # (n_layers, 3)

    # Optional per-category breakdowns.
    by_category: dict[str, np.ndarray]           # category -> (n_layers, 3)
    by_category_std: dict[str, np.ndarray]

    # Per-head breakdown used downstream by the head-taxonomy module.
    per_layer_per_head: np.ndarray               # (n_layers, n_heads, 3)

    modality_names: tuple[str, ...] = ("visual", "language", "action_prev")

    @property
    def n_layers(self) -> int:
        return int(self.layer_modality.shape[0])

    @property
    def n_heads(self) -> int:
        return int(self.per_layer_per_head.shape[1])


def compute_modality_dominance(
    rollouts: list[SyntheticRollout],
) -> ModalityDominanceResult:
    """Aggregate modality mass across rollouts / steps / heads."""
    if not rollouts:
        raise ValueError("compute_modality_dominance: no rollouts provided")

    # Stack every step's (L, H, 3) into a single (N_steps, L, H, 3) array.
    per_step = np.stack(
        [step.modality_mass for r in rollouts for step in r.steps], axis=0,
    )
    # Overall: mean over steps and heads.
    mean_over_heads = per_step.mean(axis=2)                    # (N, L, 3)
    layer_modality = mean_over_heads.mean(axis=0)              # (L, 3)
    layer_modality_std = mean_over_heads.std(axis=0)           # (L, 3)

    # Per-layer-per-head (used by head taxonomy).
    per_layer_per_head = per_step.mean(axis=0)                 # (L, H, 3)

    # Per-category breakdown.
    by_cat: dict[str, np.ndarray] = {}
    by_cat_std: dict[str, np.ndarray] = {}
    categories = sorted({r.category for r in rollouts})
    for cat in categories:
        idx_steps = []
        cursor = 0
        for r in rollouts:
            for _ in r.steps:
                if r.category == cat:
                    idx_steps.append(cursor)
                cursor += 1
        if not idx_steps:
            continue
        sub = per_step[idx_steps].mean(axis=2)                 # (n, L, 3)
        by_cat[cat] = sub.mean(axis=0)
        by_cat_std[cat] = sub.std(axis=0)

    return ModalityDominanceResult(
        layer_modality=layer_modality,
        layer_modality_std=layer_modality_std,
        by_category=by_cat,
        by_category_std=by_cat_std,
        per_layer_per_head=per_layer_per_head,
    )
