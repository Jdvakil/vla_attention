"""Bonus figure: activation-patching recovery heatmap (n_layers x 3 modalities)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..causal.activation_patching import ActivationPatchingResult


def plot_activation_patching(
    result: ActivationPatchingResult,
    out_path: str | Path,
    title: str = "Activation patching: per-layer recovery",
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    mat = result.recovery.T                       # (3, n_layers)
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(result.modality_names)))
    ax.set_yticklabels([m.replace("_", " ") for m in result.modality_names])
    ax.set_xlabel("Patch layer")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="Recovery probability")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
