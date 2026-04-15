"""Attention extraction hooks for live MolmoAct / OpenVLA.

Separated from ``vla_attention.models`` so the hook logic can be unit-tested
on plain nn.Modules without loading the 7B backbone.
"""

from .attention_hooks import AttentionHookManager, AttentionRecord  # noqa: F401
