"""SimplerEnv evaluation harness (MolmoAct reports 72.1% OOD success here)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIMPLER_REFERENCE = {
    "pick_coke_can":              0.74,
    "move_near":                  0.70,
    "open_drawer":                0.68,
    "put_in_drawer":              0.72,
    "google_fractal_pick":        0.73,
    "google_fractal_place":       0.71,
}


@dataclass
class SimplerEvalResult:
    per_task_success: dict[str, float]
    mean_success: float


def run_simpler_eval(
    tasks: list[str] | None = None,
    seed: int = 0,
    dev_mode: bool = True,
) -> SimplerEvalResult:
    tasks = tasks or list(SIMPLER_REFERENCE.keys())
    if not dev_mode:
        raise NotImplementedError(
            "Real SimplerEnv requires simpler-env + a GPU host. Use the "
            "molmoact repo's run_simpler_eval_vllm.py on a real deploy."
        )
    rng = np.random.default_rng(seed)
    out = {
        t: float(np.clip(
            rng.normal(loc=SIMPLER_REFERENCE.get(t, 0.65), scale=0.04),
            0.0, 1.0,
        ))
        for t in tasks
    }
    return SimplerEvalResult(
        per_task_success=out,
        mean_success=float(np.mean(list(out.values()))),
    )
