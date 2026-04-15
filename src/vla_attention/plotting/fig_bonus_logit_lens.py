"""Bonus figure: logit-lens error curve across layers + per DoF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.logit_lens import LogitLensResult


DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def plot_logit_lens(
    result: LogitLensResult,
    out_path: str | Path,
    title: str = "Logit-lens on action tokens",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    xs = np.arange(result.n_layers)
    for d, name in enumerate(DOF_NAMES):
        ax.plot(xs, result.error[:, d], label=name)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Prediction error (L1 to final)")
    ax.set_title("Per-DoF crystallisation curves")
    ax.legend(ncol=3, fontsize=8, loc="upper right")

    ax = axes[1]
    mean_curve = result.error.mean(axis=1)
    std_curve = result.error.std(axis=1)
    ax.plot(xs, mean_curve, color="#2E86AB", linewidth=2.4, label="mean over DoF")
    ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.15, color="#2E86AB")
    # Detect a "phase-transition" layer: the steepest descent point.
    deltas = -np.diff(mean_curve)
    pt = int(np.argmax(deltas))
    ax.axvline(pt, color="#C73E1D", linestyle="--", alpha=0.7,
               label=f"phase-transition ≈ layer {pt}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean error")
    ax.set_title("Aggregate prediction-error curve")
    ax.legend()

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
