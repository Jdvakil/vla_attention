"""Experiment 7: activation patching for modality attribution.

Collect paired success/failure rollouts on the same task. For each layer
and token-position group (visual / language / action), patch the residual
stream from success into failure and measure the recovery probability.

Dev-mode implementation: we model recovery as a function of which layer is
patched and which modality-position group is patched. Visual patching
recovers most at layers where visual prior is high; language patching
recovers most at layers where language prior is high. The output matches
the (n_layers, n_positions) heatmap the paper figure expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import SyntheticRolloutSampler


@dataclass
class ActivationPatchingResult:
    # (n_layers, 3) recovery probability for [visual, language, action_prev].
    recovery: np.ndarray
    modality_names: tuple[str, ...] = ("visual", "language", "action_prev")


def run_activation_patching(
    sampler: SyntheticRolloutSampler | None = None,
    n_tasks: int = 10,
    noise: float = 0.05,
) -> ActivationPatchingResult:
    sampler = sampler or SyntheticRolloutSampler()
    prior = sampler.layer_modality_prior              # (L, 3)
    L = prior.shape[0]

    rng = np.random.default_rng(sampler.cfg.rng_seed + 99)

    # Recovery peaks where each modality's prior mass is highest.
    # Normalise so each modality's peak ~0.9, floor ~0.05.
    recovery = np.zeros_like(prior, dtype=np.float32)
    for m in range(3):
        col = prior[:, m]
        col_norm = (col - col.min()) / max(col.max() - col.min(), 1e-6)
        recovery[:, m] = 0.05 + 0.85 * col_norm

    # Average across synthetic tasks = add task-level noise.
    recovery = recovery[None, :, :] + noise * rng.standard_normal((n_tasks, L, 3))
    recovery = recovery.mean(axis=0)
    recovery = np.clip(recovery, 0.0, 1.0)

    return ActivationPatchingResult(recovery=recovery)
