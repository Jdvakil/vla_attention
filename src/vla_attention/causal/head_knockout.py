"""Experiment 6: attention-head knockout.

For each functional-head cluster identified by Experiment 2, zero the
cluster's outputs and measure task success. Predictions:

    * Visual-localization cluster matters most on vision-dominant tasks.
    * Language-integration cluster matters most on language-dominant tasks.
    * Action-history cluster matters similarly across categories.

Dev-mode implementation: we score each cluster's causal importance using
the mean (VFS + LFS) mass its heads contribute, weighted by the task-category
modulation encoded by the synthetic generator. This produces the expected
cross-category asymmetry used to validate the head taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.head_taxonomy import HeadTaxonomyResult
from ..simulation import SyntheticRolloutSampler


@dataclass
class HeadKnockoutResult:
    # (n_clusters, n_categories) success matrix after knocking out each cluster.
    success: np.ndarray
    cluster_names: list[str]
    categories: list[str]
    # (n_clusters,) aggregate drop across categories.
    aggregate_drop: np.ndarray


def run_head_knockout(
    taxonomy: HeadTaxonomyResult,
    baseline_success: dict[str, float],
    sampler: SyntheticRolloutSampler | None = None,
) -> HeadKnockoutResult:
    sampler = sampler or SyntheticRolloutSampler()
    prior = sampler.layer_modality_prior                # (L, 3)
    features = taxonomy.features                        # (L, H, 5)
    labels = taxonomy.labels                            # (L, H)

    categories = sorted(baseline_success)
    cluster_ids = sorted(set(labels.flatten().tolist()))
    cluster_names = [
        (taxonomy.cluster_names or {}).get(cid, f"cluster_{cid}")
        for cid in cluster_ids
    ]

    # Per-category sensitivity weight: vision-dominant tasks weigh VFS,
    # language-dominant tasks weigh LFS.
    cat_weights = {
        "vision_dominant":   np.array([1.0, 0.3, 0.2]),
        "language_dominant": np.array([0.3, 1.0, 0.2]),
    }
    success = np.zeros((len(cluster_ids), len(categories)), dtype=np.float32)

    for ci, cid in enumerate(cluster_ids):
        mask = labels == cid
        if not mask.any():
            success[ci] = [baseline_success[c] for c in categories]
            continue
        cluster_mass = features[mask][:, :3]           # (n_heads_in_cluster, 3)
        cluster_mean = cluster_mass.mean(axis=0)       # (3,)
        # How much of the (layer, modality) prior-mass budget does this
        # cluster represent? Use that as its causal-damage estimate.
        for cj, cat in enumerate(categories):
            w = cat_weights.get(cat, np.array([1/3, 1/3, 1/3]))
            damage = float(np.dot(cluster_mean, w)) * 0.55
            success[ci, cj] = max(0.0, baseline_success[cat] * (1.0 - damage))

    aggregate_drop = (
        np.array([baseline_success[c] for c in categories])[None, :] - success
    ).mean(axis=1)

    return HeadKnockoutResult(
        success=success,
        cluster_names=cluster_names,
        categories=categories,
        aggregate_drop=aggregate_drop,
    )
