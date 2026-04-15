"""Figure 4: attention rollout heatmaps across task examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.attention_rollout import AttentionRolloutResult


def plot_attention_rollout(
    result: AttentionRolloutResult,
    out_path: str | Path,
    n_rows: int = 2,
    n_cols: int = 4,
    title: str = "Attention rollout over image patches",
) -> None:
    n = min(len(result.meta), n_rows * n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows + 0.4),
    )
    fig.suptitle(title, fontsize=13, y=1.02)

    axes_flat = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes_flat[i]
        heat = result.heatmaps[i]
        # Use a perceptually-uniform cmap and mark the peak.
        ax.imshow(heat, cmap="magma")
        py, px = np.unravel_index(np.argmax(heat), heat.shape)
        ax.scatter([px], [py], s=90, marker="o",
                   facecolors="none", edgecolors="#55ff88", linewidth=1.6)
        meta = result.meta[i]
        ax.set_title(
            f"{meta['task']}\nstep={meta['step']}  "
            f"{'success' if meta['success'] else 'failure'}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
