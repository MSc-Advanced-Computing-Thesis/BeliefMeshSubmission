"""Typed config loading: config/base.yaml + config/stages/<stage>.yaml overrides.

Stage override files list only the keys that differ from base.yaml; they are
deep-merged on top. Unknown keys fail loudly (dataclass constructors reject
unexpected fields) rather than being silently ignored -- a typo in a yaml key
should be an error, not a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    lr: float
    epochs: int
    optimizer: str


@dataclass
class FusionConfig:
    grid_size: int
    grid_fallback: int


@dataclass
class ConsensusConfig:
    rho: float | None


@dataclass
class DataConfig:
    digit: int
    image_size: int


@dataclass
class Config:
    seed: int
    eval_holdout_fraction: float
    data: DataConfig
    model: ModelConfig
    fusion: FusionConfig
    consensus: ConsensusConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Return base with override's keys applied on top, recursing into nested dicts."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    stage_override: str | Path | None = None,
    base_path: str | Path = "config/base.yaml",
) -> Config:
    """Load config/base.yaml, apply a stage override file on top if given, return a validated Config.

    Paths are resolved relative to the current working directory, which is
    expected to be the project root (where experiments are run from).
    """
    with open(base_path) as f:
        raw = yaml.safe_load(f)

    if stage_override is not None:
        with open(stage_override) as f:
            override = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, override)

    return Config(
        seed=raw["seed"],
        eval_holdout_fraction=raw["eval_holdout_fraction"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        fusion=FusionConfig(**raw["fusion"]),
        consensus=ConsensusConfig(**raw["consensus"]),
    )
