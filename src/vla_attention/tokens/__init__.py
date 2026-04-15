"""Token-boundary parsing.

Given a raw input sequence for a VLA forward pass, identify the index ranges
occupied by each modality (visual, depth, language, action). All attention
analysis downstream operates on these spans, so getting them right is the
single most load-bearing utility in the codebase.
"""

from .boundaries import ModalitySpan, TokenBoundaries, build_boundaries  # noqa: F401
