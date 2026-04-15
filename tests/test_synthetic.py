"""Smoke tests for the synthetic attention generator + analysis pipeline."""

from __future__ import annotations

import numpy as np

from vla_attention.analysis import (
    cluster_heads,
    compute_head_features,
    compute_logit_lens,
    compute_modality_dominance,
)
from vla_attention.simulation import SyntheticConfig, SyntheticRolloutSampler


def test_synthetic_step_shapes():
    s = SyntheticRolloutSampler(SyntheticConfig(n_layers=8, n_heads=4))
    r = s.sample_rollout("libero_spatial_0", "language_dominant", 0,
                         n_steps=5, save_spatial_profile=True)
    assert r.n_steps == 5
    step = r.steps[0]
    assert step.modality_mass.shape == (8, 4, 3)
    np.testing.assert_allclose(step.modality_mass.sum(axis=-1),
                               np.ones((8, 4)), atol=1e-5)
    assert step.spatial_profile is not None
    assert step.spatial_profile.shape == (8, 4, 27, 27)
    assert step.logit_lens_error.shape == (8, 7)


def test_modality_dominance_pipeline():
    s = SyntheticRolloutSampler(SyntheticConfig(n_layers=6, n_heads=4))
    rolls = s.sample_task("libero_object_0", "vision_dominant", 6)
    res = compute_modality_dominance(rolls)
    assert res.layer_modality.shape == (6, 3)
    np.testing.assert_allclose(res.layer_modality.sum(axis=-1),
                               np.ones(6), atol=1e-5)


def test_head_taxonomy_recovers_structure():
    """The synthetic generator assigns 5 ground-truth head types. Clustering
    with k=5 should produce roughly the same partition (silhouette > 0.1)."""
    s = SyntheticRolloutSampler(SyntheticConfig(n_layers=6, n_heads=6))
    # Need rollouts from multiple categories so consistency feature is useful.
    rolls = (
        s.sample_task("t1", "vision_dominant", 4)
        + s.sample_task("t2", "language_dominant", 4)
    )
    feat = compute_head_features(rolls)
    tax = cluster_heads(feat, k_range=(5,))
    assert tax.k == 5
    assert tax.silhouette > 0.1
    assert tax.labels.shape == (6, 6)


def test_logit_lens_decreases_monotonically():
    s = SyntheticRolloutSampler(SyntheticConfig(n_layers=12, n_heads=4))
    rolls = s.sample_task("t", "language_dominant", 10)
    lens = compute_logit_lens(rolls)
    curve = lens.error.mean(axis=1)
    # Not strictly monotonic because of noise, but the overall trend should be
    # decreasing.
    assert curve[0] > curve[-1]
    assert curve[0] - curve[-1] > 0.3
