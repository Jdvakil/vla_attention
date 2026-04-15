"""Shared matplotlib styling. Call ``apply_style()`` once at script start."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


MODALITY_COLORS = {
    "visual":      "#2E86AB",
    "language":    "#C73E1D",
    "action_prev": "#6B8E23",
    "depth":       "#7E52A0",
}

HEAD_TYPE_COLORS = {
    "visual_localization":  "#2E86AB",
    "language_integration": "#C73E1D",
    "cross_modal_bridge":   "#E8A33D",
    "action_history":       "#6B8E23",
    "generic":              "#888888",
}

METHOD_COLORS = {
    "attention_is": "#2E86AB",
    "random":       "#888888",
    "scizor_proxy": "#C73E1D",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    # Make sure we always use a non-interactive backend when running as a
    # batch script.
    plt.switch_backend("Agg")
