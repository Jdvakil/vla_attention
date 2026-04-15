"""Config loading. Reads `configs/*.yaml` into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"


@dataclass
class ArchConfig:
    n_layers: int
    n_heads: int
    d_model: int
    n_visual_tokens: int
    n_depth_tokens: int
    n_language_tokens: int
    n_action_tokens: int

    @property
    def n_heads_total(self) -> int:
        return self.n_layers * self.n_heads


@dataclass
class ModelConfig:
    family: str
    hf_id: str
    libero_checkpoints: dict[str, str]
    arch: ArchConfig


@dataclass
class ExperimentsConfig:
    raw: dict[str, Any] = field(default_factory=dict)

    def phase(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    @property
    def dev_mode(self) -> bool:
        return bool(self.raw.get("dev_mode", True))

    @property
    def synthetic_seed(self) -> int:
        return int(self.raw.get("synthetic_seed", 0))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def load_configs(
    model_path: Path | None = None,
    exp_path: Path | None = None,
) -> tuple[ModelConfig, ExperimentsConfig]:
    """Load `model.yaml` + `experiments.yaml` and return typed configs."""
    model_path = model_path or (_CONFIG_DIR / "model.yaml")
    exp_path = exp_path or (_CONFIG_DIR / "experiments.yaml")

    m = _load_yaml(model_path)
    e = _load_yaml(exp_path)

    family = m["model"]["family"]
    arch_block = "arch" if family == "molmoact" else "openvla_arch"
    arch = ArchConfig(**m[arch_block])

    mcfg = ModelConfig(
        family=family,
        hf_id=m["model"]["hf_id"],
        libero_checkpoints=m["model"].get("libero_checkpoints", {}),
        arch=arch,
    )
    return mcfg, ExperimentsConfig(raw=e)


def repo_root() -> Path:
    return _REPO_ROOT
