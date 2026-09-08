# Section 5.1 baseline characterisation: per-sample NIG dump.
#
# Evaluation only. Loads an existing pretrained Stage 0 checkpoint (3-channel
# EvidentialCNN, NOT the 5-channel CoordConv mesh variant) and writes ONE ROW
# PER SAMPLE to CSV. No summary statistic is computed here -- every derived
# quantity comes from the CSV in analyse.py, so R1a/R1b/R1c/R1d share one
# code path.
#
# Reproducibility. RotatedDigitDataset re-samples the rotation angle on every
# __getitem__ from the GLOBAL `random` module, so angles are not a property of
# the split. seed_angles() reseeds `random` immediately before each pass; with
# shuffle=False and num_workers=0 that pins an identical angle draw at every
# sweep point, making sweeps exactly comparable. Absolute MSE still floats
# relative to runs/stage0/*/manifest.yaml, which averaged a different draw.
#
# Nothing in this file imports from experiments/_common/evaluation.py: those
# helpers average per batch inside the loop and clamp uncertainty at 10.0,
# both of which would destroy the per-sample record. They are on reported
# code paths and are left untouched.

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


from beliefmesh.data.digits import RotatedDigitDataset, get_digit7_splits
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.models.evidential_cnn import EvidentialCNN
from beliefmesh.models.variants import WIDTH_VARIANTS

SPLIT_SEED = 42          # config/base.yaml seed -- pins the 80/20 partition
ANGLE_SEED = 42          # pins the rotation draw within a pass
EVAL_FRACTION = 0.2
BATCH_SIZE = 32
CKPT = CHECKPOINTS

COLUMNS = ["sample_index", "true_angle_deg", "true_angle_norm", "gamma",
           "circular_error_norm", "circular_error_deg", "nu", "alpha", "beta",
           "epistemic", "aleatoric", "total", "scale", "alpha_floor_bound"]


def load_model(variant: str, device: torch.device) -> EvidentialCNN:
    """3-channel EvidentialCNN with the variant's pretrained weights."""
    model = EvidentialCNN(in_channels=3, widths=WIDTH_VARIANTS[variant]).to(device)
    model.load_state_dict(torch.load(CKPT[variant], map_location=device))
    model.eval()
    return model


def held_out_indices() -> np.ndarray:
    _, eval_idx = get_digit7_splits(eval_fraction=EVAL_FRACTION, seed=SPLIT_SEED)
    return eval_idx


def seed_angles(pass_index: int = 0):
    """Reseed the global RNG the dataset draws its rotation from. pass_index
    varies the draw for multi-pass variance estimates; pass 0 is the canonical
    pinned draw every sweep point reuses."""
    random.seed(ANGLE_SEED + pass_index)


def dump_pass(model, eval_idx, filter_type, strength, device, pass_index=0,
              target_offset_deg=0.0):
    """One full pass over the held-out partition -> list of per-sample dicts.

    target_offset_deg (R1c): added to the TRUE ANGLE only. The image is
    untouched -- the model sees exactly the same input and the answer moves
    underneath it. Wrapped to [-1, 1) on the normalised scale, the same
    period-2 convention circular_diff uses.
    """
    dataset = RotatedDigitDataset(filter_type=filter_type, filter_strength=strength,
                                  subset_indices=eval_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    seed_angles(pass_index)
    rows = []
    cursor = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if target_offset_deg:
                targets = targets + target_offset_deg / 180.0
                targets = ((targets + 1.0) % 2.0) - 1.0

            gamma, nu, alpha, beta = model(images)
            err = circular_diff(gamma, targets)

            # NIG uncertainty decomposition (models/evidential.py). alpha is
            # softplus(x)+1 so alpha>1 in exact arithmetic, but softplus
            # underflows to 0 in float32 below ~-17, giving alpha==1.0 and an
            # undefined scale. Recorded, never clamped: alpha_floor_bound
            # counts the bindings so analyse.py can report the rate.
            am1 = alpha - 1.0
            floor_bound = (am1 <= 0).to(torch.int64)
            epistemic = beta / (nu * am1)
            aleatoric = beta / am1
            total = beta * (1.0 + nu) / (nu * am1)
            scale = torch.sqrt(beta * (1.0 + nu) / (nu * alpha))

            n = gamma.shape[0]
            for i in range(n):
                t = targets[i].item()
                rows.append({
                    "sample_index": cursor + i,
                    "true_angle_deg": t * 180.0,
                    "true_angle_norm": t,
                    "gamma": gamma[i].item(),
                    "circular_error_norm": err[i].item(),
                    "circular_error_deg": err[i].item() * 180.0,
                    "nu": nu[i].item(),
                    "alpha": alpha[i].item(),
                    "beta": beta[i].item(),
                    "epistemic": epistemic[i].item(),
                    "aleatoric": aleatoric[i].item(),
                    "total": total[i].item(),
                    "scale": scale[i].item(),
                    "alpha_floor_bound": int(floor_bound[i].item()),
                })
            cursor += n
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path
