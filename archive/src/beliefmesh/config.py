"""Typed config loading: config/base.yaml + config/stages/<stage>.yaml overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


def load_config(stage_override: str | Path | None = None, base_path: str | Path = "config/base.yaml") -> Config:
    """Load config/base.yaml, apply a stage override file on top if given, return a validated Config."""
    raise NotImplementedError
