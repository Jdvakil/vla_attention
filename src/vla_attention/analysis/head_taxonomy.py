"""Experiment 2: functional taxonomy of attention heads.

For every (layer, head) compute:

    VFS: visual focus score   = mean action->visual mass
    LFS: language focus score = mean action->language mass
    AHS: action-history score = mean action->action_prev mass
    SE : spatial entropy      = H(attention over visual tokens)
    XC : cross-task consistency = 1 - std(mass across task categories)

Then k-means over the feature vector, pick k by silhouette. The resulting
label assignment corresponds to the functional-head clusters reported in
the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..simulation import SyntheticRollout


FEATURE_NAMES = ("VFS", "LFS", "AHS", "SpatialEntropy", "TaskConsistency")


@dataclass
class HeadTaxonomyResult:
    features: np.ndarray                # (L, H, n_features)
    feature_names: tuple[str, ...] = FEATURE_NAMES
    # Clustering:
    labels: np.ndarray | None = None    # (L, H) int cluster id
    centers: np.ndarray | None = None   # (k, n_features)
    silhouette: float | None = None
    k: int | None = None
    # Human-assigned cluster names (produced by ``label_clusters_heuristic``).
    cluster_names: dict[int, str] | None = None

    @property
    def feature_matrix(self) -> np.ndarray:
        L, H, F = self.features.shape
        return self.features.reshape(L * H, F)


def compute_head_features(rollouts: list[SyntheticRollout]) -> np.ndarray:
    """Return (L, H, 5) feature matrix for head-taxonomy clustering."""
    if not rollouts:
        raise ValueError("compute_head_features: no rollouts")

    # Mass per step: (N_steps, L, H, 3)
    per_step = np.stack(
        [step.modality_mass for r in rollouts for step in r.steps], axis=0,
    )
    # Entropy per step: (N_steps, L, H)
    entropy_per_step = np.stack(
        [step.spatial_entropy for r in rollouts for step in r.steps], axis=0,
    )

    mean_mass = per_step.mean(axis=0)                    # (L, H, 3)
    mean_entropy = entropy_per_step.mean(axis=0)         # (L, H)

    # Cross-task consistency: how much does modality mass vary across task
    # categories? Low variance = "always does the same thing".
    cat_masses = []
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
        cat_masses.append(per_step[idx_steps].mean(axis=0))  # (L, H, 3)
    if len(cat_masses) > 1:
        cat_masses_arr = np.stack(cat_masses, axis=0)        # (C, L, H, 3)
        consistency = 1.0 - cat_masses_arr.std(axis=0).mean(axis=-1)  # (L, H)
    else:
        consistency = np.ones_like(mean_entropy)

    features = np.stack([
        mean_mass[..., 0],                 # VFS
        mean_mass[..., 1],                 # LFS
        mean_mass[..., 2],                 # AHS
        -mean_entropy,                     # negate so "peaky" is high
        consistency,                       # higher = more consistent
    ], axis=-1).astype(np.float32)
    return features


def cluster_heads(
    features: np.ndarray,
    k_range: tuple[int, ...] = (5, 6, 7),
    random_state: int = 0,
) -> HeadTaxonomyResult:
    """K-means over the flat (L*H, F) feature matrix, pick k by silhouette."""
    L, H, F = features.shape
    flat = features.reshape(L * H, F)
    scaler = StandardScaler().fit(flat)
    X = scaler.transform(flat)

    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        score = silhouette_score(X, km.labels_) if k > 1 else -1.0
        if best is None or score > best["score"]:
            best = {
                "score": float(score), "k": k, "km": km,
                "labels": km.labels_.reshape(L, H),
                "centers": scaler.inverse_transform(km.cluster_centers_),
            }
    assert best is not None

    names = label_clusters_heuristic(best["centers"])
    return HeadTaxonomyResult(
        features=features,
        labels=best["labels"],
        centers=best["centers"],
        silhouette=best["score"],
        k=best["k"],
        cluster_names=names,
    )


def label_clusters_heuristic(centers: np.ndarray) -> dict[int, str]:
    """Assign a human-readable name to each cluster based on centroid values.

    Centers shape: (k, F) with F = (VFS, LFS, AHS, -Entropy, Consistency).

    When two clusters would otherwise receive the same label, we disambiguate
    them by comparing peakiness (more-peaky = "sharp", less-peaky = "broad").
    """
    # First pass: propose a base name for each cluster.
    proposals: list[tuple[int, str, np.ndarray]] = []
    for i, c in enumerate(centers):
        vfs, lfs, ahs, peakiness, _consistency = c
        if ahs > vfs and ahs > lfs and ahs > 0.45:
            base = "action_history"
        elif vfs > lfs and vfs > ahs:
            base = "visual_localization"
        elif lfs > vfs and lfs > ahs:
            base = "language_integration"
        elif abs(vfs - lfs) < 0.12 and vfs > 0.25 and lfs > 0.25:
            base = "cross_modal_bridge"
        else:
            base = "generic"
        proposals.append((i, base, c))

    # Second pass: disambiguate duplicates using peakiness + dominant dim.
    by_base: dict[str, list[tuple[int, np.ndarray]]] = {}
    for i, base, c in proposals:
        by_base.setdefault(base, []).append((i, c))

    names: dict[int, str] = {}
    for base, group in by_base.items():
        if len(group) == 1:
            i, _ = group[0]
            names[i] = base
            continue
        # Sort by peakiness (higher = sharp, lower = broad).
        sorted_group = sorted(group, key=lambda ic: -ic[1][3])
        qualifiers = ["sharp", "broad", "mixed", "auxiliary", "fallback"]
        for rank, (i, _c) in enumerate(sorted_group):
            qual = qualifiers[rank] if rank < len(qualifiers) else f"aux{rank}"
            names[i] = f"{base}_{qual}"
    return names
