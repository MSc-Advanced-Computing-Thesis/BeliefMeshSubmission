# Stage 4a: Scalar weighting strategies. Experiment Specification Sec 8.
#
# Six strategies aggregating A's and C's collapsed (prediction, certainty)
# pairs: naive, certainty-weighted, squared, gated, softmax, winner-only.
# certainty = 1/(1+uncertainty), uncertainty clamped at 10, thresholds as in
# the prior repo (gated threshold 0.3, softmax temperature 1.0).
#
# PORTING NOTE: the prior repo's gated strategy had a dead-code fallback (the
# torch.where conditions tested an already-replaced tensor), so when BOTH
# anchors fell below threshold the target silently became 0. The spec
# describes the intended behaviour as falling back to naive averaging; the
# spec wins (this file implements the intended fallback).
#
# Run from the project root: python experiments/stage4_weighting_and_fusion/run_4a_scalar_weighting.py

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage4_weighting_and_fusion.harness import run_ramp_experiment

from beliefmesh.config import load_config
from beliefmesh.data.digits import get_digit7_splits
from beliefmesh.models.evidential import predictive_uncertainty

CERTAINTY_THRESHOLD = 0.3
SOFTMAX_TEMPERATURE = 1.0


def _certs(a_belief, c_belief):
    _, nu_a, alpha_a, beta_a = a_belief
    _, nu_c, alpha_c, beta_c = c_belief
    unc_a = predictive_uncertainty(nu_a, alpha_a, beta_a).clamp(max=10.0)
    unc_c = predictive_uncertainty(nu_c, alpha_c, beta_c).clamp(max=10.0)
    return 1 / (1 + unc_a), 1 / (1 + unc_c)


def naive(a, c):
    return (a[0] + c[0]) / 2


def certainty(a, c):
    ca, cc = _certs(a, c)
    return (ca * a[0] + cc * c[0]) / (ca + cc)


def squared(a, c):
    ca, cc = _certs(a, c)
    ca, cc = ca ** 2, cc ** 2
    return (ca * a[0] + cc * c[0]) / (ca + cc)


def gated(a, c):
    ca, cc = _certs(a, c)
    a_active = (ca > CERTAINTY_THRESHOLD).float()
    c_active = (cc > CERTAINTY_THRESHOLD).float()
    both_below = (a_active + c_active) == 0
    # intended behaviour per spec: fall back to naive when both below threshold
    a_active = torch.where(both_below, torch.ones_like(a_active), a_active)
    c_active = torch.where(both_below, torch.ones_like(c_active), c_active)
    return (a_active * a[0] + c_active * c[0]) / (a_active + c_active)


def softmax(a, c):
    ca, cc = _certs(a, c)
    w = torch.softmax(torch.stack([ca, cc]) / SOFTMAX_TEMPERATURE, dim=0)
    return w[0] * a[0] + w[1] * c[0]


def winner(a, c):
    ca, cc = _certs(a, c)
    return torch.where(ca > cc, a[0], c[0])


def main():
    cfg = load_config()
    seed = int(os.environ.get("EXP_SEED", cfg.seed))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx, eval_idx = get_digit7_splits(eval_fraction=cfg.eval_holdout_fraction, seed=cfg.seed)

    run_dir = Path("runs/stage4/4a_scalar_weighting")
    if seed != cfg.seed:
        run_dir = run_dir.parent / f"{run_dir.name}_seed{seed}"

    variants = {"naive": naive, "certainty": certainty, "squared": squared,
                "gated": gated, "softmax": softmax, "winner": winner}
    _, high_means = run_ramp_experiment(
        cfg, "4a_scalar_weighting", variants, run_dir,
        train_idx, eval_idx, device,
        extra_manifest={"gated_threshold": CERTAINTY_THRESHOLD,
                        "softmax_temperature": SOFTMAX_TEMPERATURE,
                        "gated_note": "intended naive fallback implemented; old repo's was dead code"})

    ranking = sorted(high_means.items(), key=lambda kv: kv[1])
    print("\nHigh-strength (>=0.6) mean MSE ranking:")
    for name, mse in ranking:
        print(f"  {name}: {mse:.4f}")


if __name__ == "__main__":
    main()
