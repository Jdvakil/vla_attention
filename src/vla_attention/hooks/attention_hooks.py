"""Register forward hooks that capture attention weights from every Llama /
Molmo decoder layer, without mutating the model's forward pass.

This file is intentionally torch-lazy: it only imports ``torch`` inside
functions that need it, so importing ``vla_attention.hooks`` on a CPU-only
box without a torch install still works for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class AttentionRecord:
    """One forward-pass worth of captured attention.

    Attributes:
        attentions: ``(n_layers, n_heads, seq_len, seq_len)`` float tensor on
            CPU, stored as numpy. seq_len is the full sequence length of the
            forward pass, including BOS + visual + language + action spans.
        seq_len: convenience alias for ``attentions.shape[-1]``.
        meta: arbitrary per-record metadata (task name, rollout index, step,
            which action DoF is being predicted, etc.).
    """

    attentions: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return int(self.attentions.shape[-1])

    @property
    def n_layers(self) -> int:
        return int(self.attentions.shape[0])

    @property
    def n_heads(self) -> int:
        return int(self.attentions.shape[1])


class AttentionHookManager:
    """Register + tear down attention hooks on a transformer backbone.

    Usage (on a GPU host with torch available):

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "allenai/MolmoAct-7B-D-0812", trust_remote_code=True,
        ... ).eval().cuda()
        >>> hm = AttentionHookManager(model)
        >>> hm.attach()
        >>> _ = model(**inputs)                       # runs normally
        >>> rec = hm.collect()                        # numpy tensor
        >>> hm.detach()

    We do NOT modify the model. We only add forward hooks on each
    ``self_attn`` module. The model MUST be instantiated with
    ``attn_implementation="eager"`` because flash/SDPA attention does not
    expose weights. ``ensure_eager_attention`` enforces this.
    """

    def __init__(self, model: Any):
        self.model = model
        self._handles: list[Any] = []
        self._per_layer: dict[int, np.ndarray] = {}
        self._layer_modules: list[Any] = []

    # ---- public API -------------------------------------------------------

    def ensure_eager_attention(self) -> None:
        """Force ``eager`` attention so weights are returned. Raises if the
        active implementation silently drops attention weights."""
        cfg = getattr(self.model, "config", None)
        if cfg is not None and getattr(cfg, "_attn_implementation", None) not in (
            None, "eager",
        ):
            raise RuntimeError(
                "Model is using "
                f"{cfg._attn_implementation!r} attention, which does not return "
                "weights. Load the model with attn_implementation='eager'."
            )

    def attach(self, layer_path: str = "auto") -> None:
        """Attach forward hooks on every decoder layer's ``self_attn`` module.

        ``layer_path`` selects the backbone layer list. ``"auto"`` tries the
        usual suspects for MolmoAct and OpenVLA.
        """
        self.ensure_eager_attention()
        self._layer_modules = _resolve_decoder_layers(self.model, layer_path)
        if not self._layer_modules:
            raise RuntimeError("Could not locate decoder layers on model")

        for idx, layer in enumerate(self._layer_modules):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            if attn is None:
                raise RuntimeError(f"Layer {idx} has no self_attn / attention module")
            handle = attn.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._per_layer.clear()

    def reset(self) -> None:
        self._per_layer.clear()

    def collect(self, meta: dict[str, Any] | None = None) -> AttentionRecord:
        """Assemble the per-layer captures into a single ``AttentionRecord``.

        Raises if not every layer produced weights in the most recent forward.
        """
        if not self._per_layer:
            raise RuntimeError("No attention captured; did you call attach()?")

        n_layers = len(self._layer_modules)
        missing = [i for i in range(n_layers) if i not in self._per_layer]
        if missing:
            raise RuntimeError(
                f"Missing attention for layers {missing}. Model likely used "
                "non-eager attention; reload with attn_implementation='eager'."
            )

        stacked = np.stack([self._per_layer[i] for i in range(n_layers)], axis=0)
        rec = AttentionRecord(attentions=stacked, meta=dict(meta or {}))
        self.reset()
        return rec

    # ---- internals --------------------------------------------------------

    def _make_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, outputs: Any) -> None:
            # HF Llama/Molmo attention returns (attn_out, attn_weights, past_kv)
            # when output_attentions=True. We defensively unpack.
            weights = _extract_attn_weights(outputs)
            if weights is None:
                return
            # weights: (batch, heads, seq, seq) torch tensor.
            self._per_layer[layer_idx] = (
                weights.detach().to("cpu", copy=False).float().numpy()[0]
            )
        return hook


def _extract_attn_weights(outputs: Any) -> Any | None:
    """Best-effort extraction of the attention-weights tensor from a layer
    output. Returns None if not present."""
    if outputs is None:
        return None
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if item is None:
                continue
            if _looks_like_attn_weights(item):
                return item
        return None
    if _looks_like_attn_weights(outputs):
        return outputs
    return None


def _looks_like_attn_weights(x: Any) -> bool:
    shape = getattr(x, "shape", None)
    if shape is None:
        return False
    # attn_weights is 4-D (B, H, Tq, Tk) and symmetric in last two dims during
    # a full-context forward pass. We relax the symmetry requirement because
    # autoregressive decoding may produce Tq=1.
    return len(shape) == 4


def _resolve_decoder_layers(model: Any, layer_path: str) -> list[Any]:
    """Find the list of decoder layers on the model. Tries a bunch of common
    attribute paths used by MolmoAct / Molmo / Llama variants."""
    if layer_path != "auto":
        return list(_dotted_get(model, layer_path))

    candidates = [
        "language_model.model.layers",   # OpenVLA (Prismatic / Llama-2)
        "model.model.layers",            # Vanilla HF Llama
        "model.transformer.blocks",      # MolmoAct / OLMo-based backbone
        "model.language_model.layers",   # Some MolmoAct revisions
        "transformer.blocks",
        "model.layers",
    ]
    for p in candidates:
        try:
            layers = _dotted_get(model, p)
            if _is_module_list(layers):
                return list(layers)
        except AttributeError:
            continue
    return []


def _dotted_get(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _is_module_list(x: Any) -> bool:
    return hasattr(x, "__len__") and hasattr(x, "__iter__") and len(x) > 0
