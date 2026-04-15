"""End-to-end orchestrator.

Runs Phase 1 (baseline), Phase 2 (diagnostic analysis), Phase 3 (causal
validation), and Phase 4 (data-efficiency intervention) using either the
real MolmoAct runtime (``--mode real``) or the synthetic generator
(``--mode dev``, the default).

Outputs:

    results/                -- pickled numpy results per experiment.
    figures/                -- PNGs for all 7 paper figures + bonuses.
    results/run_summary.json -- single-file report that ``CLAUDE.md``
                               should link to.

Run with:

    PYTHONPATH=src python scripts/run_full_pipeline.py --mode dev
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from vla_attention import load_configs
from vla_attention.analysis import (
    compute_attention_rollout,
    compute_head_features,
    compute_logit_lens,
    compute_modality_dominance,
    cluster_heads,
)
from vla_attention.causal import (
    run_activation_patching,
    run_head_knockout,
    run_visual_ablation,
)
from vla_attention.data_efficiency import (
    compute_informativeness_scores,
    pruning_curve,
    run_grounding_rescue,
)
from vla_attention.evaluation import run_libero_eval, run_simpler_eval
from vla_attention.plotting import (
    apply_style,
    plot_activation_patching,
    plot_attention_rollout,
    plot_data_efficiency,
    plot_head_knockout,
    plot_head_taxonomy,
    plot_logit_lens,
    plot_modality_heatmap,
    plot_teaser,
    plot_visual_ablation,
)
from vla_attention.simulation import (
    SyntheticConfig,
    SyntheticRolloutSampler,
    generate_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dev", "real"], default="dev")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollouts_per_task", type=int, default=20)
    p.add_argument("--tasks_per_category", type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------


def build_task_list(tasks_per_category: int) -> list[tuple[str, str]]:
    lang = [(f"libero_spatial_{i}", "language_dominant")
            for i in range(tasks_per_category)]
    lang += [(f"libero_goal_{i}", "language_dominant")
             for i in range(tasks_per_category)]
    vis = [(f"libero_object_{i}", "vision_dominant")
           for i in range(tasks_per_category)]
    vis += [(f"libero_long_{i}", "vision_dominant")
            for i in range(tasks_per_category)]
    return lang + vis


def generate_data(args: argparse.Namespace) -> tuple[list, SyntheticRolloutSampler]:
    mcfg, _ = load_configs()
    cfg = SyntheticConfig(
        n_layers=mcfg.arch.n_layers,
        n_heads=mcfg.arch.n_heads,
        n_visual=mcfg.arch.n_visual_tokens,
        n_depth=mcfg.arch.n_depth_tokens,
        n_language=mcfg.arch.n_language_tokens,
        n_action=mcfg.arch.n_action_tokens,
        family=mcfg.family,
        rng_seed=args.seed,
    )
    task_list = build_task_list(args.tasks_per_category)
    print(f"[data] generating {len(task_list)} tasks × "
          f"{args.rollouts_per_task} rollouts ({args.mode})")
    if args.mode == "real":
        raise NotImplementedError(
            "Real-model path not wired in scripts/run_full_pipeline.py yet. "
            "See src/vla_attention/models/molmoact_wrapper.py for the runtime "
            "shell that replaces generate_dataset(). On a GPU host, substitute "
            "the real runner + hook manager here."
        )
    rollouts = generate_dataset(
        task_list=task_list,
        rollouts_per_task=args.rollouts_per_task,
        cfg=cfg,
        save_spatial_profile_every=10,
    )
    sampler = SyntheticRolloutSampler(cfg)
    return rollouts, sampler


# ---------------------------------------------------------------------------
# serialisation helpers
# ---------------------------------------------------------------------------


def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if is_dataclass(obj):
        return {k: _to_serialisable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    apply_style()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    summary: dict = {"mode": args.mode, "seed": args.seed}

    # ------ Phase 1: baseline ------
    print("\n=== Phase 1: baselines ===")
    libero = run_libero_eval(dev_mode=(args.mode == "dev"), seed=args.seed)
    simpler = run_simpler_eval(dev_mode=(args.mode == "dev"), seed=args.seed)
    summary["phase1"] = {
        "libero_mean_success": libero.mean_success,
        "libero_per_suite": libero.per_suite_success,
        "simpler_mean_success": simpler.mean_success,
    }
    dump(libero, RESULTS / "phase1_libero.pkl")
    dump(simpler, RESULTS / "phase1_simpler.pkl")
    print(f"  LIBERO mean success: {libero.mean_success:.3f}")
    print(f"  SimplerEnv mean success: {simpler.mean_success:.3f}")

    # ------ data generation ------
    rollouts, sampler = generate_data(args)
    baseline_success = {
        "language_dominant": (libero.per_suite_success["spatial"]
                              + libero.per_suite_success["goal"]) / 2,
        "vision_dominant":   (libero.per_suite_success["object"]
                              + libero.per_suite_success["long"]) / 2,
    }

    # ------ Phase 2: diagnostics ------
    print("\n=== Phase 2: diagnostic analysis ===")
    modality = compute_modality_dominance(rollouts)
    dump(modality, RESULTS / "phase2_modality.pkl")
    plot_modality_heatmap(modality, FIGURES / "fig2_modality_heatmap.png")
    print("  Figure 2 ✓")

    features = compute_head_features(rollouts)
    taxonomy = cluster_heads(features, k_range=(5, 6, 7))
    dump(taxonomy, RESULTS / "phase2_head_taxonomy.pkl")
    plot_head_taxonomy(taxonomy, FIGURES / "fig3_head_taxonomy.png")
    print(f"  Figure 3 ✓ (k={taxonomy.k}, silhouette={taxonomy.silhouette:.3f})")

    rollout_res = compute_attention_rollout(rollouts, n_examples=8)
    dump(rollout_res, RESULTS / "phase2_attention_rollout.pkl")
    plot_attention_rollout(rollout_res, FIGURES / "fig4_attention_rollout.png")
    print("  Figure 4 ✓")

    lens = compute_logit_lens(rollouts)
    dump(lens, RESULTS / "phase2_logit_lens.pkl")
    plot_logit_lens(lens, FIGURES / "fig_bonus_logit_lens.png")
    print("  Bonus logit-lens figure ✓")

    summary["phase2"] = {
        "modality_peak_layer_visual": int(np.argmax(modality.layer_modality[:, 0])),
        "modality_peak_layer_language": int(np.argmax(modality.layer_modality[:, 1])),
        "taxonomy_k": taxonomy.k,
        "taxonomy_silhouette": taxonomy.silhouette,
        "cluster_names": taxonomy.cluster_names,
    }

    # ------ Phase 3: causal ------
    print("\n=== Phase 3: causal validation ===")
    ablation = run_visual_ablation(
        rollouts=rollouts,
        ablation_layers=[2, 4, 8, 12, 16, 20, 24, sampler.cfg.n_layers - 1],
        baseline_success=baseline_success,
        sampler=sampler,
    )
    dump(ablation, RESULTS / "phase3_visual_ablation.pkl")
    plot_visual_ablation(ablation, baseline_success, FIGURES / "fig5_visual_ablation.png")
    print(f"  Figure 5 ✓ (visual-information cutoff ≈ layer {ablation.cutoff_layer})")

    knockout = run_head_knockout(taxonomy, baseline_success, sampler)
    dump(knockout, RESULTS / "phase3_head_knockout.pkl")
    plot_head_knockout(knockout, baseline_success, FIGURES / "fig6_head_knockout.png")
    print("  Figure 6 ✓")

    patching = run_activation_patching(sampler)
    dump(patching, RESULTS / "phase3_activation_patching.pkl")
    plot_activation_patching(patching, FIGURES / "fig_bonus_activation_patching.png")
    print("  Bonus activation-patching figure ✓")

    summary["phase3"] = {
        "cutoff_layer": int(ablation.cutoff_layer),
        "knockout_aggregate_drop": dict(zip(
            knockout.cluster_names, knockout.aggregate_drop.tolist(),
        )),
    }

    # ------ Phase 4: data efficiency ------
    print("\n=== Phase 4: data efficiency ===")
    scores = compute_informativeness_scores(rollouts)
    pruning = pruning_curve(
        scores, fractions=[1.0, 0.7, 0.5, 0.4, 0.3],
        baseline_success=libero.mean_success, seed=args.seed,
    )
    dump(scores, RESULTS / "phase4_is_scores.pkl")
    dump(pruning, RESULTS / "phase4_pruning.pkl")
    plot_data_efficiency(pruning, FIGURES / "fig7_data_efficiency.png")
    print("  Figure 7 ✓")

    rescue = run_grounding_rescue(
        scores,
        augmentations=["color_jitter", "distractor_objects", "background_swap"],
        low_vi_percentile=20.0, seed=args.seed,
    )
    dump(rescue, RESULTS / "phase4_grounding_rescue.pkl")
    summary["phase4"] = {
        "pruning_methods": pruning.method_names,
        "pruning_fractions": pruning.fractions.tolist(),
        "pruning_success": pruning.success.tolist(),
        "rescue_vi_delta": (rescue.vi_post - rescue.vi_pre).mean().item(),
        "rescue_success_delta": (rescue.success_post - rescue.success_pre).mean().item(),
    }

    # ------ Figure 1 (teaser) ------
    plot_teaser(modality, pruning, FIGURES / "fig1_teaser.png")
    print("  Figure 1 (teaser) ✓")

    # ------ summary ------
    with (RESULTS / "run_summary.json").open("w") as fh:
        json.dump(_to_serialisable(summary), fh, indent=2)
    print(f"\nSummary saved to {RESULTS / 'run_summary.json'}")
    print(f"Figures:  ls {FIGURES}")


if __name__ == "__main__":
    main()
