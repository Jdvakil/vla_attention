"""Figure 6: head-knockout ablation table (heatmap of success rates)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..causal.head_knockout import HeadKnockoutResult


def plot_head_knockout(
    result: HeadKnockoutResult,
    baseline_success: dict[str, float],
    out_path: str | Path,
    title: str = "Head-cluster knockout",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    mat = result.success                                   # (clusters, cats)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(result.categories)))
    ax.set_xticklabels([c.replace("_", "\n") for c in result.categories])
    ax.set_yticks(range(len(result.cluster_names)))
    ax.set_yticklabels([n.replace("_", " ") for n in result.cluster_names])

    base_line = np.array([baseline_success[c] for c in result.categories])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            delta = mat[i, j] - base_line[j]
            ax.text(j, i, f"{mat[i, j]:.2f}\n(Δ{delta:+.2f})",
                    ha="center", va="center", fontsize=8.5,
                    color="black" if mat[i, j] > 0.45 else "white")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Success rate")
    ax.grid(False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
