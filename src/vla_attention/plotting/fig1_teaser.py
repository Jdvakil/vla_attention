"""Figure 1 (teaser): VLA architecture sketch + modality dominance + data
efficiency comparison, all in one multi-panel figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from ..analysis.modality_dominance import ModalityDominanceResult
from ..data_efficiency.is_scoring import PruningCurveResult
from .style import METHOD_COLORS


def plot_teaser(
    modality: ModalityDominanceResult,
    pruning: PruningCurveResult,
    out_path: str | Path,
    title: str = "Where does attention go in VLAs?",
) -> None:
    fig = plt.figure(figsize=(15.0, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.5, 1.0], wspace=0.32)
    fig.suptitle(title, fontsize=14, y=1.02)

    # ---- panel A: architecture sketch ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("MolmoAct / OpenVLA pipeline", loc="left")

    _box(ax, 0.3, 8.1, 3.4, 1.4, "RGB image\n(224×224)",   "#E7F1FA")
    _box(ax, 0.3, 6.3, 3.4, 1.4, "SigLIP ViT\n→ 729 vtoks", "#BED7EA")
    _box(ax, 0.3, 4.5, 3.4, 1.4, "MLP projector",          "#9DBFD6")
    _box(ax, 0.3, 2.7, 3.4, 1.4, "Language prompt",        "#FCE1DC")
    _box(ax, 8.2, 4.6, 5.2, 3.6,
         "Llama / OLMo backbone\n(28L × 16H)\neager attention\n[hooks attached]",
         "#DEE7C4")
    _box(ax, 8.2, 1.8, 5.2, 1.6, "Action head\n(7-DoF bins)", "#C4D18A")

    # Arrows from the left column (image / vision / projector / language)
    # into the backbone. Terminate in the empty space to the left of the box.
    for ys in (8.8, 7.0, 5.2, 3.4):
        _arrow(ax, 3.8, ys, 8.1, 6.4)
    _arrow(ax, 10.8, 4.6, 10.8, 3.4)

    # ---- panel B: modality dominance heatmap ----
    ax = fig.add_subplot(gs[1])
    mat = modality.layer_modality.T                    # (3, L)
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=min(1.0, float(mat.max()) * 1.1))
    ax.set_yticks(range(3))
    ax.set_yticklabels(["visual", "language", "action-prev"])
    ax.set_xlabel("Layer")
    ax.set_title("Action-token attention by layer", loc="left")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # ---- panel C: data-efficiency ----
    ax = fig.add_subplot(gs[2])
    xs = pruning.fractions * 100
    for i, name in enumerate(pruning.method_names):
        color = METHOD_COLORS.get(name, "#444")
        ax.plot(xs, pruning.success[i], "-o", color=color,
                label=name.replace("_", " "), linewidth=1.8)
    ax.set_xlabel("Training data retained (%)")
    ax.set_ylabel("Mean success")
    ax.set_title("Data pruning", loc="left")
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="lower left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _box(ax, x, y, w, h, text, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.22",
        linewidth=1.0, edgecolor="#333", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->,head_length=6,head_width=5",
        color="#333", linewidth=1.0, mutation_scale=10,
    ))
