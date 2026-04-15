# VLA Cross-Modal Attention Analysis: Detailed Project Plan
### *Where Does Attention Go in Vision-Language-Action Models?*
**June – September 2026 | Target Venue: CoRL 2026 / ICLR 2027**

---

## 1. Problem Statement and Research Questions

Modern VLAs like OpenVLA are trained on massive heterogeneous datasets and are expected to integrate three modalities — vision, language, and action — into coherent robot behavior. But we have almost no understanding of *how* this integration actually happens inside the network, and specifically *which modality drives action generation at each layer and timestep*.

This project answers four precise research questions:

**RQ1 — Modality Dominance:** When an action token is being predicted, how much of its attention budget is allocated to visual tokens vs. language tokens vs. prior action tokens, and how does this distribution change across layers?

**RQ2 — Head Specialization:** Do individual attention heads specialize — are there identifiable "visual grounding heads," "language-routing heads," and "action-integration heads"? Can we build a functional taxonomy of all 1,024 heads in OpenVLA?

**RQ3 — Causal Sufficiency:** Are the identified high-attention pathways *causally* responsible for action prediction, or are they correlational? Can we destroy task performance by ablating specific head clusters?

**RQ4 — Data Efficiency Implication:** Can attention-derived signals (per-demonstration informativeness scores) guide training data selection to achieve equivalent or better task performance with substantially less data?

---

## 2. Model Architecture: What You Are Actually Working With

Understanding the precise architecture of OpenVLA is essential before writing a single line of analysis code. OpenVLA is a **7.5B parameter** autoregressive transformer with three distinct computational stages that the attention analysis must cross:

```
[RGB Image] → SigLIP Vision Encoder (ViT-SO400M) → MLP Projector → [Visual Tokens: 256 tokens]
[Language Instruction] → Tokenizer → [Language Tokens: ~20-50 tokens]
[Visual + Language Tokens] → Prismatic-7B (Llama-2 backbone, 32 layers, 32 heads, d_model=4096)
                          → Action Tokens (7 DoF, discretized into 256 bins, output as language tokens)
```

**Critical architectural details that affect your analysis:**

- The vision encoder (SigLIP) has its own internal attention, separate from the Llama backbone. This is a distinct modality processing stage you should analyze separately from cross-modal attention in Llama.
- The MLP projector maps visual embeddings from SigLIP dimension to Llama's 4096-dimensional space. This is a bottleneck — you should check whether visual information survives this projection with probe experiments.
- Action tokens are predicted autoregressively, one per DoF, all attending to the full context. This means each action token prediction has a potentially different attention distribution.
- OpenVLA uses causal (unidirectional) attention in the Llama backbone, which constrains what you can measure: visual tokens attend to each other and language tokens attend forward, but action tokens attend to everything.

**What this means for your experiments:** You have three distinct attention boundaries to analyze: (1) within the SigLIP encoder, (2) at the MLP projector (information bottleneck), and (3) within the 32 Llama layers during action generation. Most existing VLA interpretability work only looks at (3). Examining (1) and (2) is a genuine novelty angle.

---

## 3. Experimental Design

The experiments are organized into four phases. Phases 1-2 are the diagnostic contribution; Phases 3-4 are the causal validation and intervention that make the paper publishable.

---

### Phase 1: Infrastructure Setup (Weeks 1–2)

#### 1.1 Codebase Setup
Clone the OpenVLA repository and instrument it for attention extraction. PyTorch's `register_forward_hook` is the correct tool — do NOT modify the model weights or forward pass, only attach read hooks.

```python
# Pseudocode for attention hook registration
attention_maps = {}  # layer_idx -> (batch, heads, seq, seq)

def make_hook(layer_idx):
    def hook(module, input, output):
        # output is (attn_output, attn_weights) for Llama attention
        attention_maps[layer_idx] = output[1].detach().cpu()
    return hook

for layer_idx, layer in enumerate(model.language_model.model.layers):
    layer.self_attn.register_forward_hook(make_hook(layer_idx))
```

You will also need the token boundary indices: where visual tokens start and end in the sequence, where language tokens start and end, and where action tokens are being generated. Write a utility that parses the input sequence and returns these indices — this is trivial but critical to get right.

#### 1.2 Evaluation Environment
Use **LIBERO** (Lifelong Learning Benchmark for Robot Environments) as your primary simulation testbed. Reasons:
- 130 tasks, well-structured by linguistic complexity and visual scene variation
- Already benchmarked against OpenVLA in prior work (direct comparability)
- Fast simulation — can run thousands of rollouts on local GPUs
- Has language-conditioned and visually-conditioned task variants you can compare

Also set up **SIMPLER** (Simulated Manipulation Primitive Library for Evaluation Reproducibility) as a secondary benchmark. It simulates BridgeV2 and Google Fractal environments and is specifically designed for evaluating OpenVLA-scale models.

If you have physical robot access at CU Boulder, plan for 20-30 hours of real-robot evaluation in Weeks 11-12 on a small task set (pick-and-place, drawer manipulation). Real-robot results in robotics papers significantly increase acceptance probability.

#### 1.3 Baseline Recording
Before any analysis, run OpenVLA on the full LIBERO suite and record:
- Task success rates per task category
- Mean inference latency
- GPU memory profile during inference

This is your baseline against which all intervention results will be compared.

---

### Phase 2: Cross-Modal Attention Analysis (Weeks 3–7)

This is the diagnostic core of the paper.

#### Experiment 1: Layer-wise Modality Dominance Map

**What:** For each of the 32 Llama layers, compute the mean attention weight from each action token to each modality group (visual, language, action) across a large set of rollouts.

**How:** 
- Collect rollouts on 50 diverse LIBERO tasks (mix of language-dominant and vision-dominant tasks, defined by task structure)
- For each rollout: extract attention tensors from all 32 layers at each action token prediction step
- For layer $l$, head $h$, action token $t$: compute $\alpha_{visual}^{l,h,t} = \sum_{i \in \mathcal{V}} A_{t,i}^{l,h}$ where $\mathcal{V}$ is the set of visual token indices
- Repeat for language ($\mathcal{L}$) and previous action tokens ($\mathcal{A}$)
- Average across heads, tasks, and time steps to get a single number per layer per modality

**Output:** A 3×32 heatmap (modality × layer) showing where attention flows at each depth. This is your primary diagnostic figure. The expected pattern (based on VLM literature) is visual dominance in layers 1-5, mixed in 6-15, language dominance in 16-32. **If VLAs deviate from this pattern, that is the finding.**

**Critical variant:** Do this separately for (a) language-dominant tasks (e.g., "put the red block in the bin" where scene has only one block), and (b) vision-dominant tasks (e.g., "pick up the object" where the target requires visual discrimination). If the modality dominance map shifts between these conditions, the model is context-sensitive. If it does not shift, the model may be using a fixed information routing strategy regardless of task difficulty. Either finding is publishable.

**Compute:** ~200 rollouts × 50 steps × 32 layers × attention tensors. On a single 4090, this takes approximately 4-6 hours to collect. Analysis is CPU-side and trivial.

#### Experiment 2: Attention Head Taxonomy

**What:** Classify all 1,024 attention heads in OpenVLA into functional categories by their role in action generation.

**How:**
- For each head (layer $l$, head $h$), compute a feature vector:
  - Visual Focus Score: $VFS^{l,h} = \mathbb{E}[\alpha_{visual}^{l,h,t}]$ (mean attention to visual tokens from action tokens)
  - Language Focus Score: $LFS^{l,h} = \mathbb{E}[\alpha_{lang}^{l,h,t}]$
  - Spatial Precision: entropy of spatial attention within visual tokens (low entropy = attends to specific image regions, high = diffuse)
  - Cross-task Consistency: variance of the head's attention pattern across tasks (low = always does the same thing, high = task-specific)
- Apply k-means clustering (k=5-7) to these feature vectors
- Manually label resulting clusters based on their dominant patterns

**Expected clusters (based on VLM literature):**
- *Visual Localization Heads*: High VFS, low entropy spatial attention, attend to specific image regions
- *Language Integration Heads*: High LFS, low spatial precision
- *Cross-modal Bridging Heads*: Mix of VFS and LFS, bridge the two modalities
- *Action History Heads*: High attention to prior action tokens
- *Semantic Summary Heads*: Attend to noun tokens and object-reference tokens specifically

**Output:** A layer-head scatter plot colored by cluster membership. This figure has never been produced for a VLA and is inherently publishable as a characterization contribution.

**Extra analysis:** Does cluster membership correlate with layer depth? The VLM literature suggests visual heads cluster in early-to-middle layers. Does this hold for VLAs where action prediction changes the objective?

#### Experiment 3: Attention Rollout Across the Full Pipeline

**What:** Standard attention rollout (Abnar & Zuidema, 2020) within the Llama backbone, extended to trace attention through the MLP projector back to original image patches.

**Why this is hard and novel:** The MLP projector is not an attention module — it's a linear map. Extending attention rollout across this boundary requires treating the projector as an identity mapping for rollout purposes (this is an approximation, but it's the same assumption made by Vision Transformer rollout papers). The alternative is to use gradient-based saliency at the projector level and multiply it with the Llama attention rollout.

**How:**
- For each action token prediction: compute attention rollout within Llama (multiply attention matrices across layers as in Abnar & Zuidema)
- This gives a distribution over the 256 visual tokens for each action token
- Map these 256 visual token attentions back to the SigLIP image patches (14×14 grid for a 224×224 image)
- Visualize as an overlay on the original RGB image: which image regions did the action token most attend to?

**Output:** Attention rollout maps that show, for a given task like "pick up the red cup," whether the action token is attending to the red cup, to the language instruction, or to neither. These are beautiful visualization figures that are essential for a robotics paper because reviewers and practitioners understand them intuitively.

**Ablation:** Compare attention rollout maps across the 7 DoF action dimensions. Does predicting the x-position of the end effector attend to different visual regions than predicting the gripper state? If so, there is internal specialization at the action-dimension level.

#### Experiment 4: Logit Lens on Action Tokens

**What:** At each of the 32 layers, project the residual stream of the last token position (where the next action token will be predicted) through the action head and measure how "correct" the early-layer representation already is.

**How:**
- At layer $l$, compute $\hat{a}^l = \text{ActionHead}(\text{LayerNorm}(\text{residual}^l))$
- Compute the L1/L2 distance from this intermediate prediction to the final action $\hat{a}^{32}$
- Plot this "prediction error" across layers

**Expected finding:** If action information crystallizes gradually, the error curve decreases smoothly. If there is a "phase transition" layer where error drops sharply, that layer is performing the key action-relevant computation. This would be an important architectural finding.

**Variant:** Do this separately for each of the 7 action dimensions. Does the gripper state (binary) crystallize earlier than the end-effector position (continuous)?

---

### Phase 3: Causal Validation (Weeks 6–9, overlapping with Phase 2)

This phase is **mandatory** — without it, reviewers will correctly dismiss the attention findings as potentially correlational.

#### Experiment 5: Layer-wise Visual Token Ablation

**What:** Zero out all visual token activations at different layers and measure the resulting task success rate degradation.

**How:**
- For layer $l \in \{2, 4, 8, 12, 16, 20, 24, 28, 32\}$:
  - Intercept visual token representations at the output of layer $l$'s attention
  - Set them to zero (or to the mean visual embedding, which is a less extreme intervention)
  - Run the rest of the forward pass normally
  - Collect 100 rollouts per task category and measure success rate

**Expected finding:** If visual information is transferred to language tokens in early layers (as the VLM literature suggests), ablating vision in deep layers should have minimal effect. The "visual information cutoff layer" is the layer where zeroing visual tokens stops hurting performance. This is a causally meaningful finding: it tells you where the model has finished using vision.

**Variant:** Instead of zeroing visual tokens globally, zero only specific image regions (by masking patches). Can you disable task performance by masking only the relevant object? This tests spatial specificity.

#### Experiment 6: Attention Head Knockout

**What:** For each identified head cluster from Experiment 2, zero out that cluster's contribution and measure performance.

**How:**
- For a head cluster $C$ (e.g., all "Visual Localization Heads"):
  - Zero the output of heads in $C$ (set their attention scores to zero before softmax, or set output to zero post-projection)
  - Run inference and measure task success

This experiment determines which head clusters are *causally necessary* for task performance. The prediction is: Visual Localization Heads will matter more for vision-dominant tasks; Language Integration Heads will matter more for language-dominant tasks.

**Output:** A table showing success rate for each head cluster ablation across task categories. This is one of the strongest causal validation figures you can produce.

#### Experiment 7: Activation Patching for Modality Attribution

**What:** Collect paired rollouts (successful vs. failed, on the same task with the same instruction but different visual scenes). Patch activations from success to failure at each layer and head to determine what causally restores performance.

**How:**
- For a given task, collect 20 success rollouts and 20 failure rollouts
- For each layer $l$, take the residual stream activations at position $p$ from success rollouts and inject them into failure rollouts at position $p$
- Measure: does performance recover toward success levels?
- Compare patching at *visual token positions* vs. *language token positions* vs. *action token positions*

**Expected finding:** If patching visual tokens at specific layers recovers performance, those layers are where vision causally matters. If patching language tokens recovers performance, those layers are where language grounding occurs.

This experiment is inspired by Meng et al.'s causal tracing (ROME) and Buurmeijer et al.'s work on VLAs, but applied specifically to the modality attribution question.

---

### Phase 4: Attention-Guided Data Efficiency Intervention (Weeks 9–13)

This is the paper's primary contribution that makes it novel beyond what already exists.

#### Experiment 8: Per-Demonstration Attention Informativeness Scoring

**What:** Score each training demonstration in a held-out dataset by how "informatively" the model attends to each modality, and use these scores to prune the dataset.

**Motivation:** A demonstration where the model barely attends to visual tokens during action prediction is a demonstration where the model is essentially predicting actions from language alone — a potential shortcut that doesn't generalize. Such demonstrations either contain redundant language-predictable actions (bad training signal) or reveal a failure mode to correct.

**How:**
- Take a finetuning dataset (e.g., LIBERO's 130-task set or a subset of Open X-Embodiment, specifically BridgeV2 + DROID)
- Fine-tune OpenVLA for one "warm-up" epoch with hooks attached
- For each demonstration $d$, compute:
  - **Visual Informativeness (VI):** Mean attention from action tokens to visual tokens across all layers and timesteps in that demonstration
  - **Language Informativeness (LI):** Mean attention from action tokens to language tokens
  - **Informativeness Score (IS):** A combined metric — you need to decide the right formulation. A candidate: IS = VI × LI (both high means the demonstration requires integrating both modalities = most informative). Another: penalize demonstrations where VI is very low (language shortcuts) or very high with LI very low (vision-only with poor language grounding)
- Rank demonstrations by IS; take top K%

**Dataset pruning experiment:**
- Train full: fine-tune on 100% of data → baseline performance
- Train pruned (attention): fine-tune on top 50%, 40%, 30% by IS → measure performance
- Train pruned (random): fine-tune on random 50%, 40%, 30% → ablation to show the metric matters
- Compare against SCIZOR (self-supervised curation baseline)

**Expected result:** Attention-guided pruning to 50% data matches or exceeds full-data performance. The mechanism is that low-IS demonstrations are either redundant or actively teach language shortcuts that hurt visual generalization.

**Critical variant — modality reweighting:** Instead of hard pruning, use IS scores as sample weights in the loss function. Demonstrations with high IS contribute more to the gradient update. This is a softer intervention that should show monotonic improvement as you increase the IS weight.

#### Experiment 9: Visual Grounding Rescue

**What:** Identify demonstrations where VI is pathologically low (language shortcuts) and augment them with visual perturbations to force visual attention.

**How:**
- Flag bottom 20% of demonstrations by VI score
- For these demonstrations: apply visual augmentations — color jitter, distractor objects in the scene, changed background — to make language prediction impossible
- Fine-tune on augmented dataset and measure: does VI increase on these tasks? Does test performance on visually challenging variants improve?

**Why this is interesting:** This goes beyond just selecting data — it uses attention analysis to *fix* problematic training data rather than discard it. If it works, it suggests that attention analysis can identify *where* in the dataset language shortcuts are learned.

---

## 4. Compute Budget and Resource Plan

### Local (4× RTX 4090, 24GB each = 96GB total VRAM)

| Task | Memory Requirement | Estimated Time |
|------|-------------------|----------------|
| OpenVLA inference (fp16) | ~15GB (1× 4090) | — |
| Attention extraction (inference + hooks) | ~20GB (1× 4090) | 6-8 hrs per experiment set |
| LoRA fine-tuning (rank 16) | ~30GB (2× 4090) | 8-12 hrs per run |
| Full LIBERO evaluation suite | ~15GB | 4-6 hrs |
| SIMPLER evaluation | ~15GB | 3-5 hrs |

Most analysis experiments (Phases 1-3) run on a single 4090. Phases 1-3 are entirely local.

### Cloud Compute (Phase 4 — Intervention)

Full fine-tuning of OpenVLA (not LoRA) for the data pruning experiments requires more memory. Recommended: **Lambda Labs or RunPod** — much cheaper than AWS/GCP for raw GPU hours.

| Task | Hardware | Estimated Hours | Est. Cost |
|------|----------|-----------------|-----------|
| Warm-up epoch for IS scoring (full dataset) | 4× A100 80GB | 8 hrs | ~$80 |
| Fine-tune runs (5 configurations × 3 seeds) | 8× A100 80GB | 20 hrs each = 100 hrs total | ~$1,200 |
| Experiment 9 augmented fine-tuning | 8× A100 80GB | 40 hrs | ~$480 |
| Buffer for reruns | — | — | ~$400 |
| **Total estimated cloud cost** | | | **~$2,200** |

This is entirely manageable and fundable through a single NSF/NASA/DOD grant seed, or even out of pocket for a PhD student.

### Data Access
- **LIBERO**: Free, open-source, well-documented. Download immediately.
- **SIMPLER**: Free, open-source. Install alongside LIBERO.
- **Open X-Embodiment (BridgeV2 subset)**: Free, hosted on Google Cloud Storage. ~350GB for BridgeV2 alone.
- **OpenVLA pre-trained weights**: Available on HuggingFace (`openvla/openvla-7b`).

---

## 5. Week-by-Week Timeline

### June: Infrastructure, Analysis Setup

**Week 1 (June 1-7): Environment Setup**
- Clone OpenVLA, verify reproduction of paper's reported LIBERO numbers (do not skip this — it catches environmental issues early)
- Implement attention hook infrastructure and verify it doesn't change model outputs
- Write token boundary parser (visual/language/action token index extraction)
- Install LIBERO and SIMPLER; run 10 baseline rollouts to verify setup

**Week 2 (June 8-14): Baseline Data Collection**
- Run full LIBERO suite evaluation with OpenVLA → record baseline success rates per task
- Run 200 rollouts with attention hooks active across 20 diverse tasks → verify hook data is sensible (sanity checks: attention sums to 1 per row, shapes are correct)
- Write analysis utility functions: attention aggregation, visualization, modality index computation
- Set up WANDB logging for experiment tracking

**Week 3 (June 15-21): Experiment 1 — Modality Dominance Map**
- Collect full attention data across 50 tasks × 100 rollouts each
- Implement layer-wise aggregation pipeline
- Produce first version of the modality dominance heatmap
- **Checkpoint:** Do you see the expected pattern (visual dominates early, language dominates late)? If not, investigate why before proceeding

**Week 4 (June 22-28): Experiment 2 — Head Taxonomy**
- Compute VFS, LFS, spatial entropy, cross-task consistency for all 1,024 heads
- Run k-means with k=5,6,7 and choose best k by silhouette score
- Manually label clusters; sanity check by visualizing 10 random heads per cluster
- Produce layer-head scatter plot with cluster labels

### July: Deep Analysis and Causal Validation

**Week 5 (July 1-7): Experiment 3 — Attention Rollout**
- Implement attention rollout within Llama
- Implement projector-crossing rollout extension
- Produce per-task attention rollout visualizations (aim for 20 compelling examples)
- Identify any tasks where rollout maps are clearly wrong/uninformative → these are cases where attention ≠ information flow

**Week 6 (July 8-14): Experiment 4 — Logit Lens**
- Implement intermediate action prediction at each layer
- Compute layer-wise prediction error curves across all 7 DoF
- Look for phase transition layers; compare across task types
- Write up Phase 2 findings in a draft "Analysis" section

**Week 7 (July 15-21): Experiment 5 — Visual Token Ablation**
- Implement layer-wise ablation in the forward hook infrastructure
- Run ablation at 9 layer checkpoints × 4 task categories × 100 rollouts = ~3,600 rollouts
- Estimate wall clock time: ~10 hours on a 4090 with batch size 1
- Identify "visual information cutoff layer" from the performance degradation curve

**Week 8 (July 22-28): Experiment 6 — Head Knockout**
- Implement head zeroing (by cluster from Experiment 2)
- Run knockout for each cluster × 4 task categories × 100 rollouts = ~2,000 rollouts
- Produce head knockout ablation table
- **Decision point:** If the head knockout table is compelling (large performance drops for specific clusters), this becomes a primary result. If it's weak, it becomes a supplementary result.

**Week 9 (July 29 - Aug 4): Experiment 7 — Activation Patching**
- Collect 20 success and 20 failure rollouts per task (aim for 10 tasks = 400 rollouts total)
- Implement activation patching pipeline
- Run patching across layers and token positions
- Produce patching heatmap figure

### August: Intervention and Integration

**Week 10 (Aug 5-11): Experiment 8 Setup — IS Scoring**
- Download BridgeV2 subset of Open X-Embodiment (or use LIBERO's training set for faster iteration)
- Implement IS scoring pipeline: warm-up fine-tuning epoch → extract per-demo attention → compute IS
- Run warm-up epoch on cloud compute
- Compute IS scores for all demonstrations; analyze distribution (what does a low-IS vs. high-IS demo look like qualitatively?)

**Week 11 (Aug 12-18): Experiment 8 — Pruning Runs**
- Launch fine-tuning runs: 100%, 50%, 40%, 30% pruned (attention-guided) × 3 seeds each = 12 runs
- Launch baseline pruning runs: 50%, 40%, 30% random × 3 seeds each = 9 runs
- These all run in parallel on cloud compute over ~2-3 days
- Evaluate all checkpoints on LIBERO; compile results table

**Week 12 (Aug 19-25): Experiment 9 — Visual Grounding Rescue + Real Robot**
- Implement visual augmentation pipeline for low-VI demonstrations
- Launch augmented fine-tuning run (3 seeds) on cloud
- **Real robot experiments (if access available):** Run 5-10 tasks on physical robot to validate that attention-guided pruning generalizes beyond simulation. Even 50 real-robot trials is significant for a robotics paper.
- Evaluate Experiment 9 results

**Week 13 (Aug 26 - Sep 1): Analysis Integration**
- Compile all results into a unified narrative
- Identify which experiments are strong (go in main paper) and which are weak (supplementary or dropped)
- Run any needed additional ablations or control experiments
- Write the Experiments and Results sections

### September: Writing and Submission

**Week 14 (Sep 2-8): Full Draft**
- Write Introduction (tell the story: motivation → gap → contributions → results)
- Write Related Work (VLM interpretability, VLA data efficiency, mechanistic interpretability methods)
- Write Method section (clean description of your analytical framework and intervention)
- Polish all figures: each figure should be understandable without reading the caption

**Week 15 (Sep 9-15): Revision**
- Share draft with advisor for feedback
- Address most critical issues
- Write Abstract (do this after the paper, not before — the abstract should summarize what you actually found)
- Double-check all numbers, figures, and citations

**Week 16 (Sep 16-22): Final Polish**
- Proofread
- Ensure all experiments have correct statistical reporting (confidence intervals, number of seeds, number of rollouts)
- Write Limitations section honestly — reviewers respect papers that acknowledge their limits
- Final formatting pass

**Week 17 (Sep 23-29): Submission**
- Upload to arXiv immediately (priority-stamps your contribution date)
- Submit to target venue
- Tag codebase with the arXiv version hash

---

## 6. Paper Structure

The paper tells a single coherent story: *we systematically analyze attention in VLAs, discover how modalities are integrated, causally validate these findings, and exploit them to reduce training data requirements.*

### Title (options — refine once you know your main finding)
- *"Where Does Attention Go? Cross-Modal Attention Dynamics in Vision-Language-Action Models"*
- *"Attention-Informed Data Efficiency in Vision-Language-Action Models"*
- *"Visual Shortcuts in VLAs: Diagnosing and Correcting Modality Imbalance via Attention Analysis"*

### Section Outline

**Abstract (250 words):** State the problem (VLAs are black boxes; we don't know how they integrate modalities), the method (systematic attention analysis + causal validation on OpenVLA), the diagnostic finding (your main result from the heatmap/taxonomy), and the practical contribution (X% less training data with attention-guided pruning).

**1. Introduction (1 page):**
- Motivation: VLAs require enormous data and we don't know which parts of that data are actually useful
- Gap: existing interpretability papers analyze representations (probing) or control features; no one has done cross-modal attention flow analysis
- Contributions: (1) first modality dominance map for VLAs, (2) functional head taxonomy, (3) causal validation via ablation, (4) attention-guided data pruning achieving X% data reduction
- Teaser figure: your modality dominance heatmap alongside a task success comparison table

**2. Background (0.5 pages):** OpenVLA architecture, attention mechanisms in transformers, brief summary of prior VLA interpretability work (Häon et al., Lu et al., DeepVision-VLA)

**3. Cross-Modal Attention Analysis (1.5 pages):**
- Attention extraction methodology
- Modality dominance map (Exp 1)
- Head taxonomy (Exp 2)
- Attention rollout visualization (Exp 3)
- Logit lens results (Exp 4)
- Key finding stated clearly as a theorem-style claim: *"Visual attention in OpenVLA is primarily utilized in layers X-Y; beyond layer Y, action tokens receive less than Z% attention from visual tokens regardless of task visual complexity."*

**4. Causal Validation (1 page):**
- Visual ablation study (Exp 5)
- Head knockout (Exp 6)
- Activation patching (Exp 7)
- Key claim: *"The identified visual processing window (layers X-Y) is causally necessary for task performance; ablating visual tokens beyond layer Y incurs less than Z% performance drop."*

**5. Attention-Guided Data Efficiency (1 page):**
- Informativeness scoring methodology (define IS formally)
- Data pruning results table: accuracy vs. dataset size, comparing attention-guided vs. random vs. SCIZOR
- Visual grounding rescue results (Exp 9)
- Key claim: *"Attention-guided pruning to X% of data matches full-dataset performance, outperforming random pruning by Y%."*

**6. Experiments (1.5 pages):**
- Detailed experimental setup
- LIBERO task breakdown table
- All ablation variants
- Real robot results (if available)
- Statistical reporting: mean ± std over 3 seeds, 100 rollouts per condition

**7. Discussion and Limitations (0.5 pages):**
- What your findings mean for VLA architecture design
- Limitations: only tested on OpenVLA (Llama-2 backbone); results may differ for diffusion-based VLAs; attention is not always a faithful representation of information flow
- Future work: extending to π0, applying during pretraining not just fine-tuning

**8. Conclusion (0.25 pages)**

---

## 7. Key Figures You Need to Produce

These are the paper's essential figures. Plan each one early.

| Figure | Description | Produced From |
|--------|-------------|---------------|
| Fig 1 | Teaser: VLA architecture + modality dominance heatmap + task success comparison | Experiments 1 + 8 |
| Fig 2 | Modality dominance heatmap (32 layers × 3 modalities), separate for language-dominant vs. vision-dominant tasks | Experiment 1 |
| Fig 3 | Head taxonomy scatter plot (layer × head colored by cluster) + representative attention patterns per cluster | Experiment 2 |
| Fig 4 | Attention rollout visualizations: 4-6 task examples showing which image regions action tokens attend to | Experiment 3 |
| Fig 5 | Visual ablation curve: task success vs. ablation layer, by task category | Experiment 5 |
| Fig 6 | Head knockout ablation table | Experiment 6 |
| Fig 7 | Data efficiency: task success vs. % of training data used, comparing attention-guided, random, and SCIZOR pruning | Experiment 8 |

---

## 8. Risks and Mitigation Strategies

**Risk 1: Attention weights are not faithful to information flow.**
Mitigation: This is a real risk and must be addressed in the paper. Run the causal validation experiments (Phase 3) and use them to corroborate attention findings. In the limitations section, explicitly cite Jain & Wallace (2019) on attention faithfulness and explain that your causal validation provides additional evidence beyond raw attention weights.

**Risk 2: Modality dominance map shows no interesting pattern (attention is uniform across layers).**
Mitigation: Unlikely given what DeepVision-VLA already found, but if it happens: (a) check your token boundary indexing — a common bug is misclassifying tokens; (b) the IS scoring intervention still works even if the layer-wise pattern is flat; (c) the non-result is still publishable if accompanied by a strong intervention result.

**Risk 3: Attention-guided pruning doesn't outperform random pruning.**
Mitigation: Have backup interventions ready — modality reweighting (softer than pruning) is more robust to noise. Also try task-level rather than demonstration-level IS scoring, which has less variance. If the pruning result is weak, the paper can still be published on the strength of the diagnostic contribution alone (though the acceptance bar is higher).

**Risk 4: Experiments take much longer than estimated.**
Mitigation: The timeline has been padded, but Phase 4 is the most time-constrained. Prioritize Experiments 8 (pruning) and drop Experiment 9 (visual grounding rescue) if needed — Experiment 9 is interesting but not load-bearing for the main claims.

**Risk 5: Too much work for one paper; reviewers say it's two papers.**
Mitigation: This is a real concern with a diagnostic + intervention paper. Frame it as two inseparable contributions: the diagnostic motivates the intervention (if you didn't do the analysis, you wouldn't know what to prune), and the intervention validates the analysis (if attention-guided pruning works, it confirms the attention patterns are meaningful). This bidirectional validation structure is a strength, not a weakness.

---

## 9. What Would Guarantee Rejection and How to Avoid It

**Rejection cause 1: No causal validation.**
Reviewers in 2026 are familiar with the attention faithfulness debate. An attention analysis paper without knockout/patching experiments is viewed as incomplete. Do not skip Phase 3.

**Rejection cause 2: Only one task environment.**
Single-environment papers at CoRL and RSS face strong skepticism. Test on at least LIBERO + SIMPLER, and ideally LIBERO + SIMPLER + real robot.

**Rejection cause 3: Comparison only to random baselines.**
In the intervention experiments, you must compare to the strongest existing data curation baselines: SCIZOR, DataMIL, and Re-Mix. Beating random pruning is not publishable. Beating SCIZOR is.

**Rejection cause 4: Claiming "first" when it isn't.**
Be precise about novelty. Häon et al. (2025) and Molinari et al. (2025) already do mechanistic analysis of VLAs. Your novelty is specifically *cross-modal attention flow quantification* and *attention-guided data efficiency* — not "first mechanistic interpretability paper on VLAs." Misrepresenting novelty is a fatal flaw.

**Rejection cause 5: Weak statistical reporting.**
Always report: mean ± std over at least 3 seeds, number of rollouts per condition, success rate not just relative improvement. Robotics reviewers are sensitive to statistical rigor because real-robot experiments are noisy.

---

## 10. Stretch Goals (If Ahead of Schedule)

If Phases 1-3 move fast, these would significantly strengthen the paper:

**Stretch 1:** Repeat the full attention analysis on **OpenVLA-OFT** (the fine-tuned variant). Compare modality dominance maps before and after fine-tuning — does fine-tuning increase or decrease visual dependency? This directly answers whether fine-tuning creates language shortcuts.

**Stretch 2:** Run a subset of experiments on **π0** (diffusion-based VLA) and show that the findings qualitatively differ — diffusion VLAs may maintain visual attention throughout all layers because they don't use autoregressive action generation. The cross-architecture comparison is highly valuable for the field.

**Stretch 3:** Implement a lightweight version of the IS scoring that runs online during training (not requiring a warm-up epoch) — this would make the method practical for large-scale pretraining, which is a significant impact multiplier.

---

*This plan is aggressive but achievable for a motivated PhD student with a clear advisor relationship and access to the described compute. The critical path is Phases 2-3 (Weeks 3-9) — everything downstream depends on getting clean attention analysis results. Do not move to Phase 4 until Phase 3 produces compelling causal validation.*
