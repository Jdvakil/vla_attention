"""Generate realistic synthetic attention + rollout data.

The analysis pipeline doesn't actually need full ``(seq, seq)`` attention
matrices — those would be 10s of GBs across a full dataset. The things the
analysis *does* need are:

    * Per-layer-per-head mass on each modality (for modality-dominance
      heatmaps and head taxonomy).
    * Per-layer-per-head spatial entropy over visual tokens (for head
      taxonomy clustering).
    * A low-res (grid_side, grid_side) visual-attention heatmap (for
      attention-rollout visualisation).
    * A per-step success label + modality prior (for causal patching).

We produce a ``StepSummary`` with exactly those quantities. The generator
also supports materialising a full ``(seq, seq)`` attention matrix on demand
for a handful of rollouts — enough for rollout visualisations without ever
loading the full dataset into memory.

Statistical priors used by the generator (all from the VLM/VLA
interpretability literature):

    P1. Layer-wise modality dominance has a sigmoid cross-over — vision
        dominates early, language dominates late.
    P2. ~30% of heads are peaky (low spatial entropy) visual-localisation
        heads that attend to a small ROI on the image grid.
    P3. ~25% of heads are broad language-integration heads.
    P4. The logit-lens error curve from action-head projections decreases
        monotonically with a phase-transition near 2/3 depth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..tokens import TokenBoundaries, build_boundaries


# ---------------------------------------------------------------------------
# head types (int codes so we can store as a small array)
# ---------------------------------------------------------------------------

HEAD_TYPE_VISUAL = 0
HEAD_TYPE_LANGUAGE = 1
HEAD_TYPE_BRIDGE = 2
HEAD_TYPE_ACTION_HISTORY = 3
HEAD_TYPE_GENERIC = 4

HEAD_TYPE_NAMES = {
    HEAD_TYPE_VISUAL: "visual_localization",
    HEAD_TYPE_LANGUAGE: "language_integration",
    HEAD_TYPE_BRIDGE: "cross_modal_bridge",
    HEAD_TYPE_ACTION_HISTORY: "action_history",
    HEAD_TYPE_GENERIC: "generic",
}

MODALITIES = ("visual", "language", "action_prev")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    """All tunable knobs of the synthetic generator."""

    # architecture ---------------------------------------------------------
    n_layers: int = 28
    n_heads: int = 16
    n_visual: int = 729
    n_depth: int = 128
    n_language: int = 32
    n_action: int = 7
    family: str = "molmoact"

    # modality-dominance curve (P1) ---------------------------------------
    visual_peak_layer: float = 4.0
    language_rise_layer: float = 14.0
    crossover_sharpness: float = 2.5

    # head typology (P2/P3) ------------------------------------------------
    frac_visual_heads: float = 0.30
    frac_language_heads: float = 0.25
    frac_bridge_heads: float = 0.20
    frac_action_history_heads: float = 0.15
    # remainder -> generic

    # per-task modulation --------------------------------------------------
    task_modulation_strength: float = 0.15

    # noise ----------------------------------------------------------------
    noise_scale: float = 0.08
    rng_seed: int = 42

    # visual ROI -----------------------------------------------------------
    grid_side: int = 27  # 27x27 = 729 tokens, matches MolmoAct

    # failure injection ----------------------------------------------------
    failure_noise_scale: float = 0.35


# ---------------------------------------------------------------------------
# output types
# ---------------------------------------------------------------------------


@dataclass
class StepSummary:
    """Aggregated attention stats for one action-token prediction step.

    Attributes:
        modality_mass: ``(n_layers, n_heads, 3)`` — mass allocated by each
            (layer, head) to each modality when the action token queries. The
            3 channels are [visual, language, action_prev].
        spatial_entropy: ``(n_layers, n_heads)`` — entropy of the attention
            distribution over visual tokens (bits). Lower = peakier head.
        spatial_profile: ``(n_layers, n_heads, grid, grid)`` or None —
            low-res 2D attention heatmap over visual tokens; stored only when
            ``save_spatial_profile=True`` because it's relatively chunky.
        logit_lens_error: ``(n_layers, 7)`` — L1 distance between the layer-l
            intermediate action prediction and the final prediction, per DoF.
        meta: per-step metadata (task, step, dof being predicted, etc.).
    """

    modality_mass: np.ndarray                  # (L, H, 3)
    spatial_entropy: np.ndarray                # (L, H)
    spatial_profile: np.ndarray | None          # (L, H, grid, grid)
    logit_lens_error: np.ndarray                # (L, 7)
    meta: dict = field(default_factory=dict)


@dataclass
class SyntheticRollout:
    task: str
    category: str
    rollout_id: int
    success: bool
    steps: list[StepSummary] = field(default_factory=list)
    boundaries: TokenBoundaries | None = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------


class SyntheticRolloutSampler:
    """Deterministic synthetic-rollout source.

    Head identities (which head is "visual localization", which is
    "language integration", etc.) are assigned once at construction time
    from the RNG seed, and then held fixed for the life of the sampler.
    That's the property head-taxonomy clustering exploits to recover a
    consistent assignment across rollouts.
    """

    def __init__(self, cfg: SyntheticConfig | None = None):
        self.cfg = cfg or SyntheticConfig()
        self._rng = np.random.default_rng(self.cfg.rng_seed)
        self._boundaries = build_boundaries(
            family=self.cfg.family,
            n_visual=self.cfg.n_visual,
            n_depth=self.cfg.n_depth,
            n_language=self.cfg.n_language,
            n_action=self.cfg.n_action,
        )
        self._head_types = self._assign_head_types()
        self._layer_modality_prior = self._layer_modality_curve()

    # ---- public read-only props ------------------------------------------

    @property
    def boundaries(self) -> TokenBoundaries:
        return self._boundaries

    @property
    def head_types(self) -> np.ndarray:
        return self._head_types

    @property
    def layer_modality_prior(self) -> np.ndarray:
        return self._layer_modality_prior

    # ---- rollout sampling -------------------------------------------------

    def sample_rollout(
        self,
        task: str,
        category: str,
        rollout_id: int,
        n_steps: int | None = None,
        success: bool | None = None,
        save_spatial_profile: bool = False,
    ) -> SyntheticRollout:
        rng = self._derived_rng(task, rollout_id)
        n_steps = n_steps or int(rng.integers(20, 40))
        if success is None:
            success = bool(rng.random() > 0.2)  # 80% success baseline

        steps = [
            self._sample_step(
                rng=rng,
                category=category,
                success=success,
                task=task,
                step=i,
                save_spatial_profile=save_spatial_profile,
            )
            for i in range(n_steps)
        ]

        return SyntheticRollout(
            task=task, category=category, rollout_id=rollout_id,
            success=success, steps=steps, boundaries=self._boundaries,
        )

    def sample_task(
        self, task: str, category: str, n_rollouts: int,
        save_spatial_profile: bool = False,
    ) -> list[SyntheticRollout]:
        return [
            self.sample_rollout(
                task=task, category=category, rollout_id=i,
                save_spatial_profile=save_spatial_profile,
            )
            for i in range(n_rollouts)
        ]

    # ---- per-step sampling ------------------------------------------------

    def _sample_step(
        self,
        *,
        rng: np.random.Generator,
        category: str,
        success: bool,
        task: str,
        step: int,
        save_spatial_profile: bool,
    ) -> StepSummary:
        cfg = self.cfg
        L, H = cfg.n_layers, cfg.n_heads

        # Per-layer prior over (visual, language, action_prev), modulated by
        # the task category.
        layer_prior = self._modulate_prior(category)          # (L, 3)

        # Build the modality-mass tensor (L, H, 3) in a vectorised way.
        modality = np.broadcast_to(layer_prior[:, None, :], (L, H, 3)).copy()

        # Head-type overrides — each type has a canonical mass vector.
        type_masses = np.array([
            [0.75, 0.15, 0.10],   # visual
            [0.15, 0.70, 0.15],   # language
            [0.40, 0.40, 0.20],   # bridge
            [0.10, 0.20, 0.70],   # action history
        ])
        for t in range(4):
            mask = self._head_types == t
            modality[mask] = type_masses[t]

        # Per-task + noise modulation, then renormalise.
        noise = cfg.noise_scale * rng.standard_normal(modality.shape)
        modality = np.clip(modality + noise, 1e-3, None)
        if not success:
            uniform = np.full(3, 1 / 3)
            modality = 0.7 * modality + 0.3 * uniform
        modality = modality / modality.sum(axis=-1, keepdims=True)

        # Spatial entropy per head. Visual-localisation heads are peaky (low
        # entropy), everything else is ~uniform over visual tokens.
        max_entropy = np.log2(cfg.grid_side ** 2)
        entropy = np.full((L, H), max_entropy * 0.9, dtype=np.float32)
        entropy[self._head_types == HEAD_TYPE_VISUAL] = max_entropy * 0.35
        entropy += cfg.noise_scale * rng.standard_normal(entropy.shape).astype(np.float32)
        entropy = np.clip(entropy, 0.05, max_entropy)

        # Optional low-res spatial heatmap (for visualization / rollout).
        profile = None
        if save_spatial_profile:
            profile = self._sample_spatial_profile(rng)

        # Logit-lens error per layer per DoF. Monotonic decrease with a
        # phase-transition around 2/3 of depth.
        depth = np.arange(L, dtype=np.float32) / max(L - 1, 1)
        base_curve = 1.0 / (1 + np.exp(5 * (depth - 0.6)))   # (L,)
        # DoF-specific: gripper (idx 6) crystallises earlier than position.
        dof_offsets = np.array([0, 0, 0, -0.05, -0.05, -0.05, -0.15])
        lens_err = base_curve[:, None] + dof_offsets[None, :]
        lens_err += cfg.noise_scale * rng.standard_normal(lens_err.shape)
        lens_err = np.clip(lens_err, 0.01, 1.5)

        meta = {
            "task": task, "category": category, "step": step,
            "success": bool(success),
        }
        return StepSummary(
            modality_mass=modality.astype(np.float32),
            spatial_entropy=entropy.astype(np.float32),
            spatial_profile=(profile.astype(np.float32) if profile is not None else None),
            logit_lens_error=lens_err.astype(np.float32),
            meta=meta,
        )

    # ---- spatial heatmap --------------------------------------------------

    def _sample_spatial_profile(self, rng: np.random.Generator) -> np.ndarray:
        """Produce a low-res (L, H, grid, grid) visual-attention heatmap.

        Visual-localization heads have a peaky ROI; everything else is
        near-uniform + a tiny amount of structured noise.
        """
        cfg = self.cfg
        L, H = cfg.n_layers, cfg.n_heads
        G = cfg.grid_side
        out = np.full((L, H, G, G), 1.0 / (G * G), dtype=np.float32)

        visual_positions = np.argwhere(self._head_types == HEAD_TYPE_VISUAL)
        for (l, h) in visual_positions:
            cy = int(rng.integers(3, G - 3))
            cx = int(rng.integers(3, G - 3))
            yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
            sigma = rng.uniform(1.5, 3.0)
            gauss = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
            gauss = gauss / gauss.sum()
            out[l, h] = 0.1 / (G * G) + 0.9 * gauss

        # Small structured noise for non-visual heads.
        noise = rng.normal(scale=0.01, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 1e-6, None)
        out = out / out.sum(axis=(-1, -2), keepdims=True)
        return out

    # ---- priors -----------------------------------------------------------

    def _layer_modality_curve(self) -> np.ndarray:
        L = self.cfg.n_layers
        depth = np.arange(L, dtype=np.float32)
        k = self.cfg.crossover_sharpness
        visual = _sigmoid(-(depth - self.cfg.visual_peak_layer) / k) + 0.15
        language = _sigmoid((depth - self.cfg.language_rise_layer) / k) + 0.15
        action_prev = 0.25 + 0.15 * _sigmoid((depth - L * 0.5) / k)
        stacked = np.stack([visual, language, action_prev], axis=1)
        return stacked / stacked.sum(axis=1, keepdims=True)

    def _modulate_prior(self, category: str) -> np.ndarray:
        prior = self._layer_modality_prior.copy()
        s = self.cfg.task_modulation_strength
        if category == "language_dominant":
            prior[:, 1] *= (1 + s)
            prior[:, 0] *= (1 - s / 2)
        elif category == "vision_dominant":
            prior[:, 0] *= (1 + s)
            prior[:, 1] *= (1 - s / 2)
        return prior / prior.sum(axis=1, keepdims=True)

    def _assign_head_types(self) -> np.ndarray:
        cfg = self.cfg
        L, H = cfg.n_layers, cfg.n_heads
        total = L * H
        counts = {
            HEAD_TYPE_VISUAL: int(total * cfg.frac_visual_heads),
            HEAD_TYPE_LANGUAGE: int(total * cfg.frac_language_heads),
            HEAD_TYPE_BRIDGE: int(total * cfg.frac_bridge_heads),
            HEAD_TYPE_ACTION_HISTORY: int(total * cfg.frac_action_history_heads),
        }
        counts[HEAD_TYPE_GENERIC] = total - sum(counts.values())
        labels = np.concatenate([np.full(n, t) for t, n in counts.items()])
        self._rng.shuffle(labels)
        labels = labels.reshape(L, H)

        # Bias: visual heads cluster in early layers, language in late layers.
        for l in range(L):
            p_visual_first = 1.0 - (l / max(L - 1, 1))
            for h in range(H):
                if labels[l, h] == HEAD_TYPE_LANGUAGE and self._rng.random() < p_visual_first * 0.5:
                    labels[l, h] = HEAD_TYPE_VISUAL
                elif labels[l, h] == HEAD_TYPE_VISUAL and self._rng.random() < (1 - p_visual_first) * 0.5:
                    labels[l, h] = HEAD_TYPE_LANGUAGE
        return labels.astype(np.int32)

    # ---- rng helpers ------------------------------------------------------

    def _derived_rng(self, task: str, rollout_id: int) -> np.random.Generator:
        key = f"{self.cfg.rng_seed}::{task}::{rollout_id}"
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
        return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# one-call dataset generator used by scripts/generate_synthetic_data.py
# ---------------------------------------------------------------------------


def generate_dataset(
    task_list: Iterable[tuple[str, str]],
    rollouts_per_task: int,
    cfg: SyntheticConfig | None = None,
    save_spatial_profile_every: int = 0,
) -> list[SyntheticRollout]:
    """Generate a full multi-task synthetic dataset.

    Args:
        task_list: iterable of ``(task_name, category)`` pairs.
        rollouts_per_task: how many rollouts per task.
        save_spatial_profile_every: if >0, attach the low-res visual-attention
            heatmap to every N-th rollout (for visualisation). 0 = never.
    """
    sampler = SyntheticRolloutSampler(cfg)
    out: list[SyntheticRollout] = []
    for task, category in task_list:
        for i in range(rollouts_per_task):
            save = (save_spatial_profile_every > 0
                    and i % save_spatial_profile_every == 0)
            out.append(sampler.sample_rollout(
                task=task, category=category, rollout_id=i,
                save_spatial_profile=save,
            ))
    return out
