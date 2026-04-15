"""vla_attention: cross-modal attention analysis for Vision-Language-Action models.

Top-level package. Public sub-modules:

    vla_attention.config          -- load YAML configs + dataclass models.
    vla_attention.tokens          -- token-boundary parsing (visual / language / action).
    vla_attention.hooks           -- attention-extraction hooks for live MolmoAct/OpenVLA.
    vla_attention.models          -- MolmoAct / OpenVLA runtime wrappers (torch/HF-gated).
    vla_attention.simulation      -- synthetic attention generator for CPU-only dev.
    vla_attention.analysis        -- Phase 2 diagnostics (modality, heads, rollout, logit lens).
    vla_attention.causal          -- Phase 3 causal validation.
    vla_attention.data_efficiency -- Phase 4 informativeness scoring + pruning.
    vla_attention.plotting        -- paper figure generators.
    vla_attention.evaluation      -- LIBERO / SimplerEnv evaluation stubs.
"""

from .config import ArchConfig, ExperimentsConfig, ModelConfig, load_configs  # noqa: F401
from .tokens import ModalitySpan, TokenBoundaries  # noqa: F401

__version__ = "0.1.0"
