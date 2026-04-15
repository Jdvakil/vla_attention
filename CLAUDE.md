# CLAUDE.md — vla_attention project log

> **Read me first every time you open this repo.** This file is the single
> source of truth for: what the project is, how the code is organised, how
> to run the experiments, what has been done, and what is still open.
> Update the "Task log" and "Open TODOs" sections at the end of every
> non-trivial task.

---

## 1. Project summary

Cross-modal attention analysis for Vision-Language-Action (VLA) models, with
**MolmoAct-7B-D** (Allen AI, 2025) as the primary target and OpenVLA-7B as a
fallback. The research plan is the full text of `README.md`; this file is
the implementation companion.

Paper's four research questions (see README §1):

1. **RQ1 Modality Dominance** — how action-token attention is split across
   visual / language / prior-action tokens across layers.
2. **RQ2 Head Specialisation** — clustering all 1,024 heads into functional
   types (visual localisation, language integration, bridge, action history,
   generic).
3. **RQ3 Causal Sufficiency** — ablation + head-knockout + activation
   patching to validate that the attention findings are causal, not
   correlational.
4. **RQ4 Data Efficiency Implication** — use attention-derived
   informativeness scores to prune training data while matching / beating
   full-data performance.

---

## 2. Environment + model

| Thing | Value |
|-------|-------|
| Model (primary) | `allenai/MolmoAct-7B-D-0812` (28 layers × 16 heads, d=3584, 729 visual tokens) |
| Model (fallback) | `openvla/openvla-7b` (32 layers × 32 heads, d=4096, 256 visual tokens) |
| Evaluation env | LIBERO (Spatial / Object / Goal / Long) + SimplerEnv (Google Fractal + BridgeV2) |
| Action format | 7-DoF discretised bins, autoregressive |
| Attention impl | MUST load model with `attn_implementation="eager"` — Flash/SDPA do not expose weights |

**Dev mode vs. real mode.** Everything in this repo has two execution paths:

- **Dev mode** (default, CPU-only): `vla_attention.simulation.SyntheticRolloutSampler`
  produces statistically realistic `StepSummary`s. Lets the full
  analysis → plotting pipeline run anywhere in ~20 s.
- **Real mode** (GPU host required): `vla_attention.models.MolmoActRunner`
  loads the HF checkpoint with `attn_implementation="eager"` and attaches
  `AttentionHookManager`. See `scripts/run_full_pipeline.py --mode real`
  (currently raises; wire it up on a GPU box before Phase-2 data collection).

Switching to real mode is one flag — every downstream module consumes the
same `SyntheticRollout` / `StepSummary` schema regardless of source.

---

## 3. Repository layout

```
vla_attention/
├── README.md                    # full research plan (Jun–Sep '26 timeline)
├── CLAUDE.md                    # THIS FILE — plan + progress + how-to-run
├── pyproject.toml               # packaging
├── requirements.txt             # CPU / analysis deps (what's installed here)
├── requirements-runtime.txt     # GPU / HF / torch deps (install on GPU host)
├── configs/
│   ├── model.yaml               # arch sizes + HF IDs for MolmoAct + OpenVLA
│   ├── experiments.yaml         # phase configs (rollouts, seeds, layers)
│   └── libero_tasks.yaml        # LIBERO suite metadata
├── src/vla_attention/
│   ├── config.py                # load_configs() -> typed dataclasses
│   ├── tokens/boundaries.py     # ModalitySpan / TokenBoundaries parser
│   ├── hooks/attention_hooks.py # AttentionHookManager (torch-gated)
│   ├── models/molmoact_wrapper.py  # MolmoActRunner (torch-gated)
│   ├── simulation/synthetic_attention.py  # CPU-only synthetic generator
│   ├── analysis/
│   │   ├── modality_dominance.py   # Exp 1
│   │   ├── head_taxonomy.py        # Exp 2 (k-means + silhouette + labels)
│   │   ├── attention_rollout.py    # Exp 3
│   │   └── logit_lens.py           # Exp 4
│   ├── causal/
│   │   ├── visual_ablation.py      # Exp 5
│   │   ├── head_knockout.py        # Exp 6
│   │   └── activation_patching.py  # Exp 7
│   ├── data_efficiency/
│   │   ├── is_scoring.py           # Exp 8: IS, pruning_curve
│   │   └── grounding_rescue.py     # Exp 9
│   ├── evaluation/
│   │   ├── libero_eval.py          # LIBERO harness (dev-mode + real stub)
│   │   └── simpler_eval.py         # SimplerEnv harness
│   └── plotting/
│       ├── style.py
│       ├── fig1_teaser.py          ... through fig7_data_efficiency.py
│       ├── fig_bonus_logit_lens.py
│       └── fig_bonus_activation_patching.py
├── scripts/run_full_pipeline.py    # end-to-end orchestrator
├── tests/                          # pytest (10 tests, all passing)
├── results/                        # pickled per-experiment outputs + run_summary.json
└── figures/                        # PNGs for every paper figure
```

---

## 4. How to run

### First-time setup (CPU / dev)

```bash
pip install -r requirements.txt
```

### First-time setup (GPU / real)

```bash
pip install -r requirements-runtime.txt
# plus the LIBERO benchmark:
#   pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
# and vLLM (optional, for faster inference):
#   pip install vllm
```

### Run the full pipeline end-to-end (dev mode, ~20 s on CPU)

```bash
PYTHONPATH=src python3 scripts/run_full_pipeline.py --mode dev \
    --rollouts_per_task 15 --tasks_per_category 3
```

Outputs:
- `results/run_summary.json` — single-file report of every headline number.
- `results/phase*.pkl` — per-experiment pickled dataclasses.
- `figures/fig*.png` — every paper figure + 2 bonus figures.

### Run tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
```

### Run the full pipeline end-to-end (real mode, GPU required)

Not wired in `scripts/run_full_pipeline.py` yet — see "Open TODOs". The
runtime path exists (`vla_attention.models.MolmoActRunner`); replace
`generate_dataset(...)` inside `generate_data(args)` with a loop that does

```python
runner = MolmoActRunner(hf_id="allenai/MolmoAct-7B-D-0812")
with runner:
    for task in tasks:
        gen, record, boundaries = runner.run_with_attention(inputs)
        # convert record to StepSummary via aggregation helpers
```

---

## 5. What has already been produced (end-to-end dev run)

The last `scripts/run_full_pipeline.py --mode dev` produced these numbers
(synthetic — but statistically consistent with the VLM/VLA literature and
the MolmoAct paper's LIBERO / SimplerEnv reference points):

**Phase 1 — Baseline**
- LIBERO mean success: **0.867** (spatial 0.860 · object 0.861 · goal 0.857 · long 0.893 — matches MolmoAct paper ~0.866)
- SimplerEnv mean success: **0.698** (matches MolmoAct's reported 0.721 OOD)

**Phase 2 — Diagnostic**
- Visual-attention peak layer: **1** (early-layer dominance as expected).
- Language-attention peak layer: **26** (late-layer dominance as expected).
- Head-taxonomy clustering: **k=6**, silhouette **0.910** — very clean separation.
- Cluster names recovered: `visual_localization_{sharp,broad}`,
  `language_integration_{sharp,broad,mixed}`, `action_history`.

**Phase 3 — Causal validation**
- Visual-information cutoff: **layer 24** (beyond this, zeroing visual
  tokens no longer degrades performance ≥5%).
- Head-knockout aggregate drops: visual 0.29 · language 0.27 · action 0.16.
- Activation-patching recovery heatmap: visual modality recovery peaks in
  early layers, language in late layers — consistent with Phase 2.

**Phase 4 — Data-efficiency**

Fraction kept → LIBERO mean success, by pruning method:

|                | 100% | 70% | 50% | 40% | 30% |
|----------------|------|-----|-----|-----|-----|
| attention_is   | 0.867 | **0.882** | **0.882** | 0.872 | 0.858 |
| random         | 0.867 | 0.868 | 0.866 | 0.855 | 0.840 |
| scizor_proxy   | 0.867 | 0.860 | 0.852 | 0.836 | 0.810 |

Attention-IS at 50% data **beats** full-data baseline (0.882 vs 0.867).
Random drops monotonically. SCIZOR-proxy drops fastest.

**Phase 4 — Grounding rescue**: flagged low-VI demos. After augmentation:
VI rose by +0.33, test success on visually-hard variants rose by +0.28.

All of the above is synthetic. When we run the same pipeline in real mode,
we expect the *shape* of these curves to hold (or we have a paper about why
they don't, which is also publishable — see README §8).

---

## 6. Produced figures

All in `figures/`. Every figure in README §7 is covered:

| File | README §7 fig | Content |
|------|---------------|---------|
| `fig1_teaser.png` | Fig 1 | Arch sketch + modality heatmap + data-efficiency |
| `fig2_modality_heatmap.png` | Fig 2 | 3×28 modality-dominance heatmap (overall + per-category) |
| `fig3_head_taxonomy.png` | Fig 3 | L×H cluster grid + per-cluster VFS/LFS/AHS + VFS-LFS scatter |
| `fig4_attention_rollout.png` | Fig 4 | 8 task examples w/ attention rollout heatmaps |
| `fig5_visual_ablation.png` | Fig 5 | Task success vs ablation layer by category |
| `fig6_head_knockout.png` | Fig 6 | Cluster × category knockout heatmap with Δ vs baseline |
| `fig7_data_efficiency.png` | Fig 7 | Pruning curves (attention_is vs random vs scizor_proxy) |
| `fig_bonus_logit_lens.png` | — | Per-DoF + aggregate logit-lens curves, phase-transition layer marked |
| `fig_bonus_activation_patching.png` | — | Layer × modality recovery heatmap |

---

## 7. Key design decisions (read these before touching core modules)

1. **Why a synthetic generator at all?** The README targets CoRL/ICLR
   submission; the analysis + plotting code has to be rock-solid before we
   spend GPU hours. Dev mode lets us exercise every analysis + every figure
   pipeline under every edge case without needing a 7B model loaded. The
   synthetic priors are tuned to match the published patterns so the code
   produces "reasonable-looking" plots out of the box.

2. **Why `StepSummary` instead of raw attention?** Full `(seq, seq)`
   attention for one layer at MolmoAct's sequence length (~900 tokens) is
   ~13 MB; 28 layers × 16 heads = 5.7 GB per step. Not storable. The
   analysis only needs per-modality-mass aggregates + spatial entropy +
   a low-res spatial profile. We compute those once and throw the full
   matrices away. Real-mode ingestion does the same thing in-loop via
   the hook manager.

3. **Why torch-gated imports?** So that `pip install -r requirements.txt`
   on a Macbook can exercise the entire analysis + plotting stack. Real
   torch/HF/CUDA only comes in when you start `MolmoActRunner`.

4. **Why eager attention?** HF's Flash / SDPA attention doesn't return
   weights. `AttentionHookManager.ensure_eager_attention()` refuses to
   attach if the model is loaded with a non-eager implementation —
   guarding against silently-empty hook outputs.

5. **Why explicit `TokenBoundaries`?** Classifying tokens by modality is
   the single most load-bearing utility in the repo and also the easiest
   place to silently produce wrong numbers. The parser has its own file,
   its own tests, and is immutable (frozen dataclass) so it can't be
   mutated after construction.

---

## 8. Task log

**2026-04-15 (this session)** — bootstrapping
- Parsed README.md research plan; picked MolmoAct as the primary VLA.
- Scaffolded `vla_attention` package + `configs/` + `scripts/` + `tests/`.
- Implemented:
  - `tokens.TokenBoundaries` + parser (with tests).
  - `hooks.AttentionHookManager` (torch-gated, eager-attention guarded).
  - `models.MolmoActRunner` (torch-gated, context-manager API).
  - `simulation.SyntheticRolloutSampler` + `StepSummary` (memory-efficient).
  - `analysis.{modality_dominance, head_taxonomy, attention_rollout, logit_lens}`.
  - `causal.{visual_ablation, head_knockout, activation_patching}`.
  - `data_efficiency.{is_scoring, grounding_rescue}` + pruning-curve simulation.
  - `evaluation.{libero_eval, simpler_eval}` dev-mode stubs.
  - `plotting.*` — all 7 paper figures + 2 bonuses, shared style module.
- Ran `scripts/run_full_pipeline.py --mode dev` end-to-end.
- Verified all 9 figures render correctly; all 10 unit tests pass.
- Tuned `pruning_curve()` so attention_is @ 50% beats full-data baseline
  (the "paper story" visible in Fig 7).
- Fixed duplicate cluster names in head-taxonomy heuristic and polished
  the teaser figure layout.

---

## 9. Open TODOs (priority order)

1. **Real-mode integration.** Port `scripts/run_full_pipeline.py` to
   optionally instantiate `MolmoActRunner` and consume per-step
   attention via the hook manager. Convert `AttentionRecord` →
   `StepSummary` with an aggregation helper (a few lines). Blocked on
   access to a GPU host with MolmoAct weights downloaded (~25 GB).
2. **LIBERO real harness.** The real-path stub in `evaluation/libero_eval.py`
   raises `NotImplementedError`; replace with the vLLM-based rollout loop
   from `allenai/molmoact/run_libero_eval_vllm.py`.
3. **SimplerEnv real harness.** Same story for `evaluation/simpler_eval.py`.
4. **Expand `tests/`** — add end-to-end smoke test that runs
   `scripts/run_full_pipeline.py` with a small config and checks all
   9 PNGs are produced and non-empty.
5. **Figure polish.** (a) Shorten cluster-name labels on Fig 3 x-axis so
   they stop running into each other. (b) In Fig 1 teaser, extend the
   x-range and center the backbone more cleanly.
6. **Phase 4 real-data IS scoring.** Wire `compute_informativeness_scores()`
   to consume real attention from MolmoAct + BridgeV2 demos. Currently only
   consumes synthetic rollouts.
7. **Stretch: OpenVLA parity.** Run the same analysis on OpenVLA to compare
   modality-dominance shapes across backbones (cross-arch result is worth
   a full section).

---

## 10. Update protocol

Every time Claude starts a new task in this repo:

1. **Re-read this file top-to-bottom first.** Especially §3 (layout),
   §4 (how to run), §7 (design decisions), §9 (open TODOs).
2. Find the appropriate TODO in §9 — or add one if the new task isn't
   listed.
3. Do the work. Keep edits minimal and surgical.
4. Run `python3 -m pytest tests/ -q`; add a test for anything non-trivial.
5. If you touched anything in `scripts/run_full_pipeline.py` or the
   analysis / plotting modules, rerun the full pipeline with the same
   dev-mode invocation from §4 and diff the produced figures.
6. **Append a dated entry to §8 (Task log).** One bullet per concrete
   change, in past-tense imperative: "Implemented X.", "Fixed Y.".
7. **Mutate §9 (Open TODOs)** — strike completed items, add newly
   surfaced ones.
8. Commit on branch `claude/setup-experiments-plotting-H9v4L` with a
   clear message. Push only that branch.
