"""Evaluation harnesses for LIBERO + SimplerEnv.

These modules are torch/simulator-gated. On a CPU-only dev host they return
deterministic mock success rates so the baseline-recording pipeline can run
end-to-end.
"""

from .libero_eval import LiberoEvalResult, run_libero_eval  # noqa: F401
from .simpler_eval import SimplerEvalResult, run_simpler_eval  # noqa: F401
