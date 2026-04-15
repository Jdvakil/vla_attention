"""Experiment 4: logit lens on action tokens.

At each of the N_layers, project the residual stream at the last position
through the action head and measure the L1 distance between the layer-l
intermediate prediction and the final (layer-N) prediction. Plot the curve.
Sharp drops = "phase-transition" layers where action-relevant computation
completes.

In dev mode the synthetic generator produces the per-layer-per-DoF error
directly; ``compute_logit_lens`` averages it across rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import SyntheticRollout


@dataclass
class LogitLensResult:
    error: np.ndarray                   # (n_layers, 7)
    error_std: np.ndarray               # (n_layers, 7)
    by_category: dict[str, np.ndarray]

    @property
    def n_layers(self) -> int:
        return int(self.error.shape[0])


def compute_logit_lens(rollouts: list[SyntheticRollout]) -> LogitLensResult:
    per_step = np.stack(
        [step.logit_lens_error for r in rollouts for step in r.steps], axis=0,
    )
    error = per_step.mean(axis=0)
    error_std = per_step.std(axis=0)

    by_cat: dict[str, np.ndarray] = {}
    categories = sorted({r.category for r in rollouts})
    for cat in categories:
        idx = []
        cursor = 0
        for r in rollouts:
            for _ in r.steps:
                if r.category == cat:
                    idx.append(cursor)
                cursor += 1
        if idx:
            by_cat[cat] = per_step[idx].mean(axis=0)

    return LogitLensResult(
        error=error, error_std=error_std, by_category=by_cat,
    )
