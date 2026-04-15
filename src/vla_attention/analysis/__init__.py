"""Phase-2 diagnostic analyses.

    exp1_modality_dominance -- per-layer, per-modality attention mass.
    exp2_head_taxonomy      -- k-means over (VFS, LFS, entropy, consistency).
    exp3_attention_rollout  -- rollout + cross-projector overlay.
    exp4_logit_lens         -- per-layer action prediction error.

Each submodule takes a list of ``SyntheticRollout`` (or the real-model
analogue) and returns numpy results ready to feed into ``vla_attention.plotting``.
"""

from .modality_dominance import (  # noqa: F401
    ModalityDominanceResult,
    compute_modality_dominance,
)
from .head_taxonomy import (  # noqa: F401
    HeadTaxonomyResult,
    compute_head_features,
    cluster_heads,
)
from .attention_rollout import AttentionRolloutResult, compute_attention_rollout  # noqa: F401
from .logit_lens import LogitLensResult, compute_logit_lens  # noqa: F401
