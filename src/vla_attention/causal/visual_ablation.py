"""Experiment 5: layer-wise visual-token ablation.

At layer L, zero the residual-stream activations at visual-token positions
(or replace them with the mean visual embedding — a softer intervention).
Re-run the forward pass from layer L+1 onwards and measure task success.

The "visual information cutoff layer" is the deepest layer beyond which
ablation causes no further degradation — i.e. the model has finished using
vision.

Dev-mode implementation: we approximate the causal effect as

    success(ablate_at_L) = baseline * (1 - visual_usage_from_L)

where ``visual_usage_from_L`` is the integrated prior mass on visual tokens
from layer L onwards. This reproduces the expected shape of the ablation
curve (sharp drop when ablating early, asymptote when ablating late) and
lets us sanity-check the plotting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import SyntheticRollout, SyntheticRolloutSampler


@dataclass
class VisualAblationResult:
    ablation_layers: np.ndarray          # (n_layers_tested,)
    success: np.ndarray                  # (n_layers_tested,) overall success
    success_by_category: dict[str, np.ndarray]
    cutoff_layer: int                    # deepest layer where ablation still hurts

    @property
    def n_layers_tested(self) -> int:
        return int(self.ablation_layers.shape[0])


def run_visual_ablation(
    rollouts: list[SyntheticRollout],
    ablation_layers: list[int],
    baseline_success: dict[str, float],
    sampler: SyntheticRolloutSampler | None = None,
) -> VisualAblationResult:
    """Simulate Experiment 5 using the layer-wise visual-mass prior.

    Args:
        rollouts: baseline rollouts (only used to determine ``n_layers`` and
            categories). Pass the same rollout set used for Phase 2.
        ablation_layers: list of layer indices to ablate at.
        baseline_success: mapping ``category -> baseline success rate``.
        sampler: the synthetic sampler whose layer_modality_prior to read.
            If None, build a default sampler.
    """
    sampler = sampler or SyntheticRolloutSampler()
    prior = sampler.layer_modality_prior              # (L, 3)
    L = prior.shape[0]

    # Visual usage from layer l onwards (mass integral).
    visual_prior = prior[:, 0]
    cumulative_remaining = np.array([visual_prior[l:].sum() for l in range(L)])
    cumulative_remaining = cumulative_remaining / cumulative_remaining[0]

    ablation_layers_arr = np.array(sorted(ablation_layers))
    categories = list(baseline_success.keys())

    success_by_cat: dict[str, np.ndarray] = {}
    for cat in categories:
        base = baseline_success[cat]
        # Damage scales with (1 - remaining_usage_fraction_after_layer).
        # Vision-dominant categories are more damage-sensitive.
        scale = 0.75 if cat == "vision_dominant" else 0.50
        dmg = scale * cumulative_remaining[ablation_layers_arr]
        # baseline * 1  when ablation_layer >= cutoff (no damage),
        # baseline * (1 - scale)  at ablation_layer = 0 (full damage).
        success_by_cat[cat] = base * (1.0 - dmg)

    overall = np.mean(np.stack(list(success_by_cat.values()), axis=0), axis=0)

    # Cutoff = deepest layer where success < 95% of baseline.
    overall_base = float(np.mean(list(baseline_success.values())))
    cutoff_idx = int(np.argmax(overall >= 0.95 * overall_base))
    cutoff_layer = int(ablation_layers_arr[cutoff_idx]) if cutoff_idx else int(
        ablation_layers_arr[-1]
    )

    return VisualAblationResult(
        ablation_layers=ablation_layers_arr,
        success=overall,
        success_by_category=success_by_cat,
        cutoff_layer=cutoff_layer,
    )
