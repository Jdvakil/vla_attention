"""Experiment 3: attention rollout across the full VLA pipeline.

Standard attention rollout (Abnar & Zuidema, 2020) extended to cross the MLP
projector. For a given action-token prediction, we compute a distribution
over the 256/729 visual tokens and then project them back to the SigLIP
image patch grid for visualisation.

In dev mode, we synthesise the rollout by composing per-layer heatmaps from
the synthetic generator. In production, the rollout composer would consume
real attention matrices from the hook manager.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import SyntheticRollout


@dataclass
class AttentionRolloutResult:
    # (n_examples, grid, grid) heatmap over image patches, per example.
    heatmaps: np.ndarray
    # Per-example metadata (task, step, dof, etc.).
    meta: list[dict]
    grid_side: int = 27


def compute_attention_rollout(
    rollouts: list[SyntheticRollout],
    n_examples: int = 20,
    dof: int | None = None,
) -> AttentionRolloutResult:
    """Compose per-layer spatial profiles into a single rollout heatmap.

    We follow the Abnar & Zuidema formulation: the rollout is an iterative
    composition ``R_l = (A_l + I) R_{l-1}`` starting from the identity and
    adding the residual path at each layer. In the synthetic case our
    ``spatial_profile`` tensor is already a normalised heatmap per layer;
    we multiply (element-wise) across layers, re-normalise, and take the
    head-mean as the final map.
    """
    picked = _pick_examples_with_profile(rollouts, n_examples)
    if not picked:
        raise ValueError("No rollouts have spatial_profile. Re-run with "
                         "save_spatial_profile=True on at least some rollouts.")

    heatmaps: list[np.ndarray] = []
    meta: list[dict] = []
    grid = picked[0][1].spatial_profile.shape[-1]

    for rollout, step in picked:
        prof = step.spatial_profile                     # (L, H, G, G)
        # Head-mean then iterative multiply across layers.
        per_layer = prof.mean(axis=1)                    # (L, G, G)
        rollout_map = np.ones((grid, grid), dtype=np.float32)
        for layer_map in per_layer:
            rollout_map = rollout_map * (layer_map + 0.1 / (grid * grid))
            rollout_map = rollout_map / rollout_map.sum()
        heatmaps.append(rollout_map)
        meta.append({
            "task": rollout.task, "category": rollout.category,
            "rollout_id": rollout.rollout_id, "step": step.meta.get("step", 0),
            "success": rollout.success, "dof": dof,
        })

    return AttentionRolloutResult(
        heatmaps=np.stack(heatmaps, axis=0),
        meta=meta,
        grid_side=grid,
    )


def _pick_examples_with_profile(
    rollouts: list[SyntheticRollout], n: int,
) -> list[tuple[SyntheticRollout, object]]:
    out: list[tuple[SyntheticRollout, object]] = []
    for r in rollouts:
        for s in r.steps:
            if s.spatial_profile is not None:
                out.append((r, s))
                break   # one step per rollout is enough for viz
        if len(out) >= n:
            break
    return out
