"""Figure 2: modality-dominance heatmap.

Two rows: overall, then a row per task category. Each row: a (3, n_layers)
heatmap of attention mass on visual / language / action_prev modalities.
Below each heatmap, stacked-bar strip showing how the three modalities
decompose the total at every layer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.modality_dominance import ModalityDominanceResult
from .style import MODALITY_COLORS


def plot_modality_heatmap(
    result: ModalityDominanceResult,
    out_path: str | Path,
    title: str = "Cross-modal attention dominance by depth",
) -> None:
    modalities = ("visual", "language", "action_prev")
    n_layers = result.n_layers

    rows = [("Overall", result.layer_modality)]
    for cat, arr in sorted(result.by_category.items()):
        rows.append((cat.replace("_", " ").title(), arr))

    fig, axes = plt.subplots(
        nrows=len(rows), ncols=2, figsize=(11, 2.2 * len(rows) + 1.2),
        gridspec_kw={"width_ratios": [3.3, 1.0]},
    )
    if len(rows) == 1:
        axes = axes[None, :]
    fig.suptitle(title, fontsize=13, y=1.01)

    for i, (name, arr) in enumerate(rows):
        ax_heat, ax_bar = axes[i]

        # Heatmap: modalities on rows, layers on columns.
        mat = arr.T                   # (3, n_layers)
        im = ax_heat.imshow(
            mat, aspect="auto", cmap="viridis",
            vmin=0.0, vmax=min(1.0, float(mat.max()) * 1.1),
        )
        ax_heat.set_yticks(range(3))
        ax_heat.set_yticklabels([m.replace("_", " ") for m in modalities])
        ax_heat.set_xticks(np.linspace(0, n_layers - 1, 6).astype(int))
        ax_heat.set_xlabel("Layer")
        ax_heat.set_title(name, loc="left")
        ax_heat.grid(False)
        fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, label="Mass")

        # Right: stacked bar-chart (modality composition) averaged across
        # layers, showing the overall budget split.
        mass = arr.mean(axis=0)
        colors = [MODALITY_COLORS[m] for m in modalities]
        ax_bar.bar(range(3), mass, color=colors, edgecolor="white")
        ax_bar.set_xticks(range(3))
        ax_bar.set_xticklabels(
            [m.replace("_", "\n") for m in modalities], fontsize=9,
        )
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title("Layer-avg", loc="left")
        ax_bar.set_ylabel("Mean mass")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
