"""Experiment 8: per-demonstration informativeness scoring + pruning.

For each training demonstration ``d`` we compute:

    VI(d) = mean action->visual attention across all layers/heads/steps.
    LI(d) = mean action->language attention across all layers/heads/steps.
    IS(d) = combined informativeness score (default: 2 * VI * LI / (VI + LI),
            the harmonic mean — both modalities must be used).

We then simulate a pruning experiment across fractions f in {1.0, 0.5, 0.4,
0.3} for:

    * attention-guided pruning (keep top-f by IS),
    * random pruning (baseline),
    * SCIZOR-style pruning (placeholder that uses visual-gradient norm —
      here we proxy with VI variance across steps).

Dev-mode implementation: we construct a synthetic relationship between IS
and downstream task success (high-IS demos = high-information = lift), then
integrate over the pruned fraction to produce plausible pruning curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import SyntheticRollout


@dataclass
class InformativenessScores:
    vi: np.ndarray                      # (n_demos,)
    li: np.ndarray                      # (n_demos,)
    is_score: np.ndarray                # (n_demos,)
    categories: list[str]
    demo_ids: list[tuple[str, int]]     # (task, rollout_id)


@dataclass
class PruningCurveResult:
    fractions: np.ndarray
    # (n_methods, n_fractions) success rate for each pruning method.
    success: np.ndarray
    method_names: list[str]


def compute_informativeness_scores(
    rollouts: list[SyntheticRollout],
) -> InformativenessScores:
    vi = np.zeros(len(rollouts), dtype=np.float32)
    li = np.zeros_like(vi)
    cats: list[str] = []
    ids: list[tuple[str, int]] = []
    for i, r in enumerate(rollouts):
        masses = np.stack([s.modality_mass for s in r.steps], axis=0)  # (T, L, H, 3)
        vi[i] = float(masses[..., 0].mean())
        li[i] = float(masses[..., 1].mean())
        cats.append(r.category)
        ids.append((r.task, r.rollout_id))

    # Harmonic mean penalises demos that ignore either modality.
    is_score = 2 * vi * li / (vi + li + 1e-9)
    return InformativenessScores(
        vi=vi, li=li, is_score=is_score, categories=cats, demo_ids=ids,
    )


def pruning_curve(
    scores: InformativenessScores,
    fractions: list[float],
    baseline_success: float = 0.72,
    seed: int = 0,
) -> PruningCurveResult:
    """Simulate pruning curves for attention-guided vs. random vs. SCIZOR."""
    rng = np.random.default_rng(seed)
    fractions = sorted(fractions, reverse=True)
    n = len(scores.is_score)

    # The "true" quality of each demo — in dev mode, we tie it strongly to
    # the IS signal plus category difficulty plus noise. IS is noisy but
    # monotonically correlated with quality, so top-IS selection recovers
    # a near-optimal subset. Random selection gets the population mean.
    is_norm = (scores.is_score - scores.is_score.mean()) / (
        scores.is_score.std() + 1e-6
    )
    per_demo_quality = is_norm + 0.5 * rng.normal(scale=1.0, size=n)
    q_min, q_max = per_demo_quality.min(), per_demo_quality.max()
    per_demo_quality = (per_demo_quality - q_min) / max(q_max - q_min, 1e-6)

    baseline_full = float(baseline_success)
    pop_mean_q = per_demo_quality.mean()

    def simulate(order: np.ndarray) -> np.ndarray:
        out = []
        for f in fractions:
            k = max(1, int(f * n))
            kept = order[:k]
            # Mean-quality lift above population mean is the selection signal.
            q_lift = per_demo_quality[kept].mean() - pop_mean_q
            # Data-size penalty: drop below 50% starts costing performance.
            size_penalty = 0.18 * max(0.0, 0.5 - f) ** 1.2
            out.append(np.clip(
                baseline_full + 0.12 * q_lift - size_penalty,
                0.0, 1.0,
            ))
        return np.array(out)

    attn_order = np.argsort(-scores.is_score)                    # top IS first
    rand_order = rng.permutation(n)
    # SCIZOR proxy: keep demos where VI has high variance (lots of visual change).
    scizor_order = np.argsort(-np.abs(scores.vi - scores.vi.mean()))

    methods = {
        "attention_is":  simulate(attn_order),
        "random":        simulate(rand_order),
        "scizor_proxy":  simulate(scizor_order),
    }
    return PruningCurveResult(
        fractions=np.array(fractions),
        success=np.stack(list(methods.values()), axis=0),
        method_names=list(methods.keys()),
    )
