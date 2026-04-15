"""Phase-4 data-efficiency intervention.

    exp8_is_scoring_and_pruning -- per-demo informativeness score + pruning.
    exp9_grounding_rescue       -- augment low-VI demos to force visual attention.
"""

from .is_scoring import (  # noqa: F401
    InformativenessScores,
    compute_informativeness_scores,
    pruning_curve,
)
from .grounding_rescue import GroundingRescueResult, run_grounding_rescue  # noqa: F401
