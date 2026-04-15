"""Runtime wrappers around MolmoAct / OpenVLA.

Imports here are torch-gated. On a CPU-only dev host importing
``vla_attention.models`` will still succeed but calling ``load_molmoact``
will raise a ``RuntimeError`` pointing you at ``requirements-runtime.txt``.
"""

from .molmoact_wrapper import (  # noqa: F401
    MolmoActRunner,
    load_molmoact,
    torch_available,
)
