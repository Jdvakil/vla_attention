"""Thin runtime wrapper around a HuggingFace MolmoAct checkpoint.

The entire module is written so that *importing* it on a CPU-only host
succeeds. Only ``load_molmoact`` and ``MolmoActRunner.__init__`` require
torch + transformers + a GPU. The point of this separation is to let the
analysis/plotting/tests run anywhere, while keeping the runtime path
available for the GPU host that will actually execute experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..hooks import AttentionHookManager, AttentionRecord
from ..tokens import TokenBoundaries, build_boundaries


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _require_torch() -> None:
    if not torch_available():
        raise RuntimeError(
            "This path requires torch + transformers. Install with:\n"
            "    pip install -r requirements-runtime.txt\n"
            "Or stay in dev_mode (see configs/experiments.yaml) which uses "
            "the synthetic attention generator instead."
        )


@dataclass
class MolmoActInputs:
    """A minimal, typed bundle of what MolmoAct needs for a forward pass.

    We keep this decoupled from the HF processor output so the analysis code
    can construct fake inputs without loading the real processor.
    """
    image_paths: list[str]
    instruction: str
    unnorm_key: str = "libero_spatial"


class MolmoActRunner:
    """Hold a loaded MolmoAct checkpoint + an attached AttentionHookManager.

    Use it as a context manager so hooks are always detached cleanly:

        with MolmoActRunner("allenai/MolmoAct-7B-D-0812") as runner:
            record = runner.run_with_attention(inputs)
            # record.attentions: (n_layers, n_heads, seq, seq) numpy array
    """

    def __init__(
        self,
        hf_id: str = "allenai/MolmoAct-7B-D-0812",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        _require_torch()
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        torch_dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation="eager",  # required to expose attn weights
        ).eval()
        self.device = device
        self.hook_manager = AttentionHookManager(self.model)

    # ---- context-manager plumbing ----------------------------------------

    def __enter__(self) -> "MolmoActRunner":
        self.hook_manager.attach()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.hook_manager.detach()

    # ---- inference --------------------------------------------------------

    def run_with_attention(
        self,
        inputs: MolmoActInputs,
        *,
        max_new_tokens: int = 64,
    ) -> tuple[Any, AttentionRecord, TokenBoundaries]:
        """Run a forward pass and return (generation, attention_record, boundaries)."""
        _require_torch()
        import torch

        batch = self.processor.process(
            images=[_load_image(p) for p in inputs.image_paths],
            text=inputs.instruction,
        )
        batch = {k: v.to(self.device).unsqueeze(0) for k, v in batch.items()}

        self.hook_manager.reset()
        with torch.inference_mode():
            out = self.model.generate_from_batch(
                batch,
                {"max_new_tokens": max_new_tokens, "do_sample": False},
                tokenizer=self.processor.tokenizer,
            )

        record = self.hook_manager.collect(
            meta={"instruction": inputs.instruction, "unnorm_key": inputs.unnorm_key},
        )
        boundaries = self._infer_boundaries_from_processor(batch, inputs)
        return out, record, boundaries

    # ---- token-boundary introspection ------------------------------------

    def _infer_boundaries_from_processor(
        self,
        batch: dict[str, Any],
        inputs: MolmoActInputs,
    ) -> TokenBoundaries:
        """Reconstruct the visual/depth/language/action spans from processor
        output. MolmoAct exposes ``image_token_type_ids`` on the batch which
        tags each position; we use it when available, otherwise we fall back
        to the static config counts.
        """
        # Prefer the processor-provided segmentation if present.
        token_types = batch.get("image_token_type_ids")
        if token_types is not None:
            return _boundaries_from_token_types(token_types[0].cpu().numpy())

        # Fallback: use declared arch defaults.
        from ..config import load_configs
        mcfg, _ = load_configs()
        return build_boundaries(
            family=mcfg.family,
            n_visual=mcfg.arch.n_visual_tokens,
            n_depth=mcfg.arch.n_depth_tokens,
            n_language=mcfg.arch.n_language_tokens,
            n_action=mcfg.arch.n_action_tokens,
        )


def load_molmoact(
    hf_id: str = "allenai/MolmoAct-7B-D-0812",
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> MolmoActRunner:
    return MolmoActRunner(hf_id=hf_id, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_image(path: str | Path):
    from PIL import Image  # lazy import
    return Image.open(path).convert("RGB")


def _boundaries_from_token_types(tok_types: np.ndarray) -> TokenBoundaries:
    """Turn a (seq_len,) int array tagging each token's modality into spans.

    Convention used in MolmoAct's HF processor output:
        0 = text / language
        1 = image / visual
        2 = depth
        3 = action (for fine-tuned LIBERO checkpoints)
    BOS is implicit at position 0.
    """
    from ..tokens import ModalitySpan, TokenBoundaries

    tag_to_mod = {0: "language", 1: "visual", 2: "depth", 3: "action"}
    spans: list[ModalitySpan] = [ModalitySpan("bos", 0, 1)]

    cur_tag = None
    start = 1
    for i, t in enumerate(tok_types[1:], start=1):
        tag = int(t)
        if tag != cur_tag:
            if cur_tag is not None:
                spans.append(ModalitySpan(tag_to_mod[cur_tag], start, i))
            cur_tag = tag
            start = i
    if cur_tag is not None:
        spans.append(ModalitySpan(tag_to_mod[cur_tag], start, len(tok_types)))

    tb = TokenBoundaries(spans=spans)
    tb.validate()
    return tb
