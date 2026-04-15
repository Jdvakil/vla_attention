"""Phase-3 causal validation of the diagnostic attention findings.

    exp5_visual_ablation     -- zero visual tokens at layer L, measure success.
    exp6_head_knockout       -- zero whole head clusters, measure success.
    exp7_activation_patching -- patch success-run activations into failure runs.

On a GPU host these modules register hooks that mutate layer outputs; in dev
mode they consume ``SyntheticRollout``s and simulate the causal effect using
the head-type ground truth that the synthetic generator exposes.
"""

from .visual_ablation import VisualAblationResult, run_visual_ablation  # noqa: F401
from .head_knockout import HeadKnockoutResult, run_head_knockout  # noqa: F401
from .activation_patching import (  # noqa: F401
    ActivationPatchingResult,
    run_activation_patching,
)
