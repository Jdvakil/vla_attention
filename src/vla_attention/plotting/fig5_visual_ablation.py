"""Figure 5: task success vs ablation layer, per task category."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..causal.visual_ablation import VisualAblationResult
from .style import MODALITY_COLORS


CATEGORY_COLORS = {
    "vision_dominant":   MODALITY_COLORS["visual"],
    "language_dominant": MODALITY_COLORS["language"],
    "mixed":             "#7E52A0",
}


def plot_visual_ablation(
    result: VisualAblationResult,
    baseline_success: dict[str, float],
    out_path: str | Path,
    title: str = "Visual-token ablation",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = result.ablation_layers

    for cat, ys in sorted(result.success_by_category.items()):
        color = CATEGORY_COLORS.get(cat, "#444")
        ax.plot(xs, ys, "-o", color=color,
                label=f"{cat.replace('_', ' ')}  (base={baseline_success[cat]:.2f})")
        ax.axhline(baseline_success[cat], color=color, alpha=0.3, linestyle=":")

    ax.axvline(result.cutoff_layer, color="#222", alpha=0.5, linestyle="--",
               label=f"cutoff ≈ layer {result.cutoff_layer}")

    ax.set_xlabel("Ablation layer")
    ax.set_ylabel("Task success rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
