"""Figure 3: head-taxonomy scatter + per-cluster VFS/LFS averages."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.head_taxonomy import HeadTaxonomyResult
from .style import HEAD_TYPE_COLORS


def plot_head_taxonomy(
    result: HeadTaxonomyResult,
    out_path: str | Path,
    title: str = "Attention head taxonomy",
) -> None:
    labels = result.labels
    features = result.features
    L, H, _ = features.shape

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.suptitle(
        f"{title}  ·  k={result.k}  ·  silhouette={result.silhouette:.3f}",
        fontsize=12, y=1.03,
    )

    cluster_ids = sorted(set(labels.flatten().tolist()))
    name_by_id = result.cluster_names or {cid: f"cluster {cid}" for cid in cluster_ids}

    # Panel 1: layer-head grid colored by cluster id.
    ax = axes[0]
    def _color_for_name(name: str) -> str:
        # Exact match first, then prefix match (so "visual_localization_sharp"
        # still maps to the visual-localization colour).
        if name in HEAD_TYPE_COLORS:
            return HEAD_TYPE_COLORS[name]
        for base, col in HEAD_TYPE_COLORS.items():
            if name.startswith(base + "_"):
                return col
        return "#444"

    cid_to_color = {cid: _color_for_name(name_by_id[cid]) for cid in cluster_ids}
    color_grid = np.empty((L, H, 3), dtype=np.float32)
    for cid in cluster_ids:
        rgba = plt.matplotlib.colors.to_rgb(cid_to_color[cid])
        color_grid[labels == cid] = rgba
    ax.imshow(color_grid, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer")
    ax.set_title("Cluster assignment (L × H)")
    ax.grid(False)

    # Panel 2: per-cluster mean (VFS, LFS, AHS).
    ax = axes[1]
    x = np.arange(len(cluster_ids))
    width = 0.27
    vfs = np.array([features[labels == cid][:, 0].mean() for cid in cluster_ids])
    lfs = np.array([features[labels == cid][:, 1].mean() for cid in cluster_ids])
    ahs = np.array([features[labels == cid][:, 2].mean() for cid in cluster_ids])
    ax.bar(x - width, vfs, width, label="VFS", color="#2E86AB")
    ax.bar(x,         lfs, width, label="LFS", color="#C73E1D")
    ax.bar(x + width, ahs, width, label="AHS", color="#6B8E23")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [name_by_id[cid].replace("_", "\n") for cid in cluster_ids],
        fontsize=8, rotation=0,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Per-cluster modality mass")
    ax.legend(ncol=3, loc="upper right", fontsize=8)

    # Panel 3: VFS vs LFS scatter colored by cluster.
    ax = axes[2]
    for cid in cluster_ids:
        mask = labels == cid
        ax.scatter(
            features[mask][:, 0], features[mask][:, 1],
            c=cid_to_color[cid], alpha=0.6, s=18,
            label=name_by_id[cid],
        )
    ax.set_xlabel("Visual focus (VFS)")
    ax.set_ylabel("Language focus (LFS)")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Heads in VFS-LFS space")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
