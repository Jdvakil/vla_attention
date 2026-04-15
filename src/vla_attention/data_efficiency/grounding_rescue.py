"""Experiment 9: visual-grounding rescue.

Find demonstrations where the VI score is pathologically low (bottom 20%),
augment them with visual perturbations (color jitter, distractors, background
swaps) that make language-only prediction impossible, and re-fine-tune.
Expected: VI on those demos rises, and test performance on visually-
challenging variants improves.

Dev-mode implementation: we model VI boost and success lift as monotonic
functions of augmentation strength. Enough to drive the plotting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .is_scoring import InformativenessScores


@dataclass
class GroundingRescueResult:
    # (n_augmentations,) pre- and post-rescue VI on flagged demos.
    augmentations: list[str]
    vi_pre: np.ndarray                  # (n_aug,)
    vi_post: np.ndarray                 # (n_aug,)
    success_pre: np.ndarray             # (n_aug,)  on visually-hard test set
    success_post: np.ndarray            # (n_aug,)  on visually-hard test set


def run_grounding_rescue(
    scores: InformativenessScores,
    augmentations: list[str],
    low_vi_percentile: float = 20.0,
    seed: int = 0,
) -> GroundingRescueResult:
    rng = np.random.default_rng(seed)
    threshold = np.percentile(scores.vi, low_vi_percentile)
    flagged = scores.vi <= threshold
    if not flagged.any():
        raise ValueError("No demos below the low-VI threshold")

    base_vi = float(scores.vi[flagged].mean())
    # Each augmentation has a known "strength" we hand-pick.
    strength = {
        "color_jitter":       0.15,
        "distractor_objects": 0.35,
        "background_swap":    0.50,
    }

    vi_pre = np.full(len(augmentations), base_vi, dtype=np.float32)
    vi_post = np.array([
        min(0.95, base_vi + strength.get(a, 0.2) + rng.normal(scale=0.02))
        for a in augmentations
    ], dtype=np.float32)

    base_success = 0.48
    success_pre = np.full(len(augmentations), base_success, dtype=np.float32)
    success_post = np.array([
        min(0.90, base_success + 0.9 * strength.get(a, 0.2) + rng.normal(scale=0.03))
        for a in augmentations
    ], dtype=np.float32)

    return GroundingRescueResult(
        augmentations=augmentations,
        vi_pre=vi_pre, vi_post=vi_post,
        success_pre=success_pre, success_post=success_post,
    )
