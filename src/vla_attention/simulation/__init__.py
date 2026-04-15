"""Synthetic attention generator.

Every analysis / plotting routine in this repo operates on numpy
``AttentionRecord`` objects. The synthetic generator produces realistic
records whose statistical properties match what has been reported in the
VLM / VLA interpretability literature:

* Early layers: visually-dominated attention from action tokens.
* Mid layers: mixed modality attention.
* Late layers: language + action-history dominated attention.
* A subset of heads is "visual localization" (peaky spatial attention over a
  small ROI in the 27x27 image grid). A subset is "language integration"
  (broad attention over nouns). Head membership is consistent across
  rollouts, which is the property head-taxonomy clustering exploits.

This lets the analysis pipeline be exercised end-to-end on a CPU without
needing the 7B weights or a LIBERO install. When GPU + MolmoAct are
available, swap ``SyntheticRolloutSampler`` for the real inference path.
"""

from .synthetic_attention import (  # noqa: F401
    SyntheticConfig,
    SyntheticRolloutSampler,
    SyntheticRollout,
    generate_dataset,
)
