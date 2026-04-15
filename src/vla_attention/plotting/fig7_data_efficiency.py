"""Figure 7: task success vs % of training data used, by pruning method."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..data_efficiency.is_scoring import PruningCurveResult
from .style import METHOD_COLORS


def plot_data_efficiency(
    result: PruningCurveResult,
    out_path: str | Path,
    title: str = "Attention-guided data pruning",
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xs = result.fractions * 100

    for i, name in enumerate(result.method_names):
        color = METHOD_COLORS.get(name, "#444")
        ax.plot(xs, result.success[i], "-o", color=color,
                label=name.replace("_", " "), linewidth=2.0)

    # Auto y-range around the data, leaving room for the legend.
    y_all = result.success
    y_lo = float(y_all.min()) - 0.04
    y_hi = float(y_all.max()) + 0.04
    ax.set_xlabel("Training data retained (%)")
    ax.set_ylabel("LIBERO mean success rate")
    ax.set_title(title)
    ax.set_ylim(max(0.0, y_lo), min(1.0, y_hi))
    ax.invert_xaxis()
    # Reference line for full-data baseline.
    ax.axhline(float(y_all[:, 0].mean()), color="#666", linestyle=":",
               linewidth=1.0, alpha=0.6, label="full-data baseline")
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
