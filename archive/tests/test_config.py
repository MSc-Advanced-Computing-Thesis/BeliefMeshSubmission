"""Tests for beliefmesh.config.load_config: base loading, stage override merging,
and loud failure on unknown keys."""

from __future__ import annotations

import pytest

from beliefmesh.config import load_config

BASE = """\
seed: 42
eval_holdout_fraction: 0.2
data:
  digit: 7
  image_size: 28
model:
  lr: 3.0e-4
  epochs: 30
  batch_size: 32
  optimizer: adam
fusion:
  grid_size: 360
  grid_fallback: 180
consensus:
  rho: null
"""


@pytest.fixture
def base_yaml(tmp_path):
    p = tmp_path / "base.yaml"
    p.write_text(BASE)
    return p


def test_loads_base(base_yaml):
    cfg = load_config(base_path=base_yaml)
    assert cfg.seed == 42
    assert cfg.eval_holdout_fraction == 0.2
    assert cfg.data.digit == 7
    assert cfg.model.lr == 3.0e-4
    assert cfg.model.epochs == 30
    assert cfg.fusion.grid_size == 360
    assert cfg.consensus.rho is None


def test_real_base_yaml_loads():
    # the actual config/base.yaml in the repo must parse into a valid Config
    # (run from the project root, which pytest does).
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.fusion.grid_size == 360


def test_stage_override_merges_only_listed_keys(base_yaml, tmp_path):
    override = tmp_path / "stage.yaml"
    override.write_text("fusion:\n  grid_size: 180\nconsensus:\n  rho: 0.1\n")
    cfg = load_config(stage_override=override, base_path=base_yaml)
    assert cfg.fusion.grid_size == 180        # overridden
    assert cfg.fusion.grid_fallback == 180    # untouched sibling key survives
    assert cfg.consensus.rho == 0.1           # overridden
    assert cfg.seed == 42                     # untouched top-level key survives


def test_unknown_key_fails_loudly(base_yaml, tmp_path):
    override = tmp_path / "stage.yaml"
    override.write_text("fusion:\n  grid_sizee: 180\n")  # typo'd key
    with pytest.raises(TypeError):
        load_config(stage_override=override, base_path=base_yaml)


def test_empty_override_is_a_noop(base_yaml, tmp_path):
    override = tmp_path / "empty.yaml"
    override.write_text("")
    cfg = load_config(stage_override=override, base_path=base_yaml)
    assert cfg == load_config(base_path=base_yaml)
