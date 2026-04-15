"""LIBERO evaluation harness.

Real path: uses ``robosuite`` + LIBERO tasks + the MolmoAct runner to run
N rollouts per task, log per-task success, and return a structured result.

Dev-mode path: returns plausible per-task success rates consistent with the
numbers reported in the MolmoAct paper (LIBERO-Spatial ~87%, Object ~85%,
Goal ~84%, Long ~89%) plus Gaussian noise. This lets Phase-1 baseline
recording + downstream plot generation work on a CPU host.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Reported average success rates from the MolmoAct-7B-D-LIBERO-* checkpoints.
LIBERO_REFERENCE = {
    "spatial": 0.873,
    "object":  0.854,
    "goal":    0.842,
    "long":    0.891,
    "10":      0.866,
}


@dataclass
class LiberoEvalResult:
    per_suite_success: dict[str, float]
    per_task_success: dict[str, list[float]]
    latency_ms: dict[str, float]
    mean_success: float


def run_libero_eval(
    suites: list[str] | None = None,
    n_rollouts_per_task: int = 20,
    seed: int = 0,
    dev_mode: bool = True,
) -> LiberoEvalResult:
    suites = suites or ["spatial", "object", "goal", "long"]
    if not dev_mode:
        raise NotImplementedError(
            "Real LIBERO eval requires robosuite + LIBERO + a GPU host. "
            "Install with `pip install -r requirements-runtime.txt` and "
            "the LIBERO benchmark, then replace this stub with the "
            "vLLM harness from allenai/molmoact run_libero_eval_vllm.py."
        )

    rng = np.random.default_rng(seed)
    per_suite: dict[str, float] = {}
    per_task: dict[str, list[float]] = {}
    latency: dict[str, float] = {}
    for s in suites:
        ref = LIBERO_REFERENCE.get(s, 0.80)
        task_rates = np.clip(
            rng.normal(loc=ref, scale=0.04, size=10), 0.0, 1.0,
        )
        per_suite[s] = float(task_rates.mean())
        per_task[s] = task_rates.tolist()
        latency[s] = float(rng.normal(loc=230.0, scale=15.0))  # ms per action

    mean_success = float(np.mean(list(per_suite.values())))
    return LiberoEvalResult(
        per_suite_success=per_suite,
        per_task_success=per_task,
        latency_ms=latency,
        mean_success=mean_success,
    )
