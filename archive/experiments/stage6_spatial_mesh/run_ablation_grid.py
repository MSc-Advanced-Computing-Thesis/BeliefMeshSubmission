# Overnight ablation sweep on the static-spatial world (CoordConv +
# excluded-rotation-ranges fixes active throughout -- the validated Stage 6
# architecture, see THESIS_EXPERIMENTS_SUMMARY.md Sec 6). Three parts,
# selectable via --part so they can run as independent parallel background
# processes without fighting over the same output directory:
#
#   lr      -- LR grid x {fusion, fedavg_global}, lam/rho at defaults
#   lam     -- lam grid x {fusion, fedavg_global}, lr fixed at the chosen
#              3e-5, rho at default (unused outside consensus mode)
#   lamrho  -- lam x rho cross table, consensus mode only (rho only affects
#              consensus's trust-EMA), lr fixed at 3e-5. Split into shards
#              (--shard/--nshards) so multiple processes can chew through
#              the cross product in parallel.
#
# Single seed (42) throughout -- an overnight ablation scan, not a
# statistically-replicated headline result. Same field/env/wearable-path
# construction as run_static_spatial_v2_test.py, reusing that run's saved
# field.npy so every ablation point is directly comparable to the existing
# v2 5-seed result.
#
# Run examples:
#   python -u run_ablation_grid.py --part lr
#   python -u run_ablation_grid.py --part lam
#   python -u run_ablation_grid.py --part lamrho --shard 0 --nshards 3
#   python -u run_ablation_grid.py --part lamrho --shard 1 --nshards 3
#   python -u run_ablation_grid.py --part lamrho --shard 2 --nshards 3

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
FIELD = Path("runs/stage6/offset_world/dynamic_static_spatial_v2/field.npy")
ROOT = Path("runs/stage6/offset_world/ablation")
N_WEARABLES = 3
SEED = 42
DEFAULT_LR = 3e-5
DEFAULT_LAM = 0.1
DEFAULT_RHO = 0.2
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]

LR_GRID = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
LAM_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
LAMRHO_LAM_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
LAMRHO_RHO_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99]

# Follow-up: the LR sweep (best lr=1e-4) and lam sweep (best lam=5-10) were
# each run holding the OTHER at its old default (lam=0.1 for the LR sweep,
# lr=3e-5 for the lam sweep) -- so neither confirms whether the two wins
# stack. Fusion-only, the handful of new (lr, lam) cells not already covered
# by either single-axis sweep.
JOINT_GRID = [(1e-4, 5.0), (1e-4, 10.0), (1e-4, 20.0), (3e-4, 10.0)]

# Self-consistency, redone as a tunable CAPPED weight (mesh.py's self_weight)
# instead of the earlier full-weight injection that tested 5x worse. 0.0
# reproduces the v2 5-seed headline exactly (sanity check); 1.0 checks
# whether the earlier catastrophic failure reproduces on the fixed
# architecture; the rest probe for a sweet spot below that.
SELFWEIGHT_GRID = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]


def _tag_lr(v):
    return f"{v:.0e}".replace("+", "")


def _tag_val(v):
    s = f"{v:g}"
    return s.replace(".", "p").replace("-", "m")


def build_configs(part: str, shard: int, nshards: int):
    """Returns list of dicts: mode, lr, lam, rho, tag."""
    configs = []
    if part == "lr":
        for mode in ("fusion", "fedavg_global"):
            for lr in LR_GRID:
                configs.append(dict(mode=mode, lr=lr, lam=DEFAULT_LAM, rho=DEFAULT_RHO,
                                     tag=f"lr_{mode}_lr{_tag_lr(lr)}"))
    elif part == "lam":
        for mode in ("fusion", "fedavg_global"):
            for lam in LAM_GRID:
                configs.append(dict(mode=mode, lr=DEFAULT_LR, lam=lam, rho=DEFAULT_RHO,
                                     tag=f"lam_{mode}_lam{_tag_val(lam)}"))
    elif part == "lamrho":
        for lam, rho in [(lam, rho) for lam in LAMRHO_LAM_GRID for rho in LAMRHO_RHO_GRID]:
            configs.append(dict(mode="consensus", lr=DEFAULT_LR, lam=lam, rho=rho,
                                 tag=f"lamrho_consensus_lam{_tag_val(lam)}_rho{_tag_val(rho)}"))
    elif part == "joint":
        for lr, lam in JOINT_GRID:
            configs.append(dict(mode="fusion", lr=lr, lam=lam, rho=DEFAULT_RHO,
                                 tag=f"joint_fusion_lr{_tag_lr(lr)}_lam{_tag_val(lam)}"))
    elif part == "selfweight":
        for sw in SELFWEIGHT_GRID:
            configs.append(dict(mode="fusion", lr=DEFAULT_LR, lam=DEFAULT_LAM, rho=DEFAULT_RHO,
                                 self_weight=sw,
                                 tag=f"selfweight_fusion_sw{_tag_val(sw)}"))
    else:
        raise ValueError(part)
    return configs[shard::nshards]


def main(part: str, shard: int, nshards: int):
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(FIELD)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    configs = build_configs(part, shard, nshards)
    shard_suffix = f"_shard{shard}" if nshards > 1 else ""
    summary_path = ROOT / f"summary_{part}{shard_suffix}.csv"
    print(f"=== ablation part={part} shard={shard}/{nshards} : {len(configs)} configs ===")

    rows = []
    for i, c in enumerate(configs):
        t0 = time.time()
        cfg.model.lr = c["lr"]
        run_dir = ROOT / c["tag"]
        if (run_dir / "manifest.yaml").exists():
            print(f"[{i+1}/{len(configs)}] SKIP (already done): {c['tag']}")
            continue
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        res = run_mesh_experiment(
            cfg, condition=c["tag"], run_dir=run_dir,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=c["mode"], baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Ablation ({c['tag']})",
            offset_field=field, env_seed=SEED,
            excluded_rotation_ranges=EXCLUDED_RANGES,
            lam=c["lam"], rho=c["rho"], self_weight=c.get("self_weight", 0.0),
        )
        dt = time.time() - t0
        cell_mse_steps = np.load(run_dir / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        row = dict(tag=c["tag"], mode=c["mode"], lr=c["lr"], lam=c["lam"], rho=c["rho"],
                   self_weight=c.get("self_weight", 0.0),
                   last50=res["mean_mse_last_50"], whole_run=whole_run_mse, r=r, seconds=dt)
        rows.append(row)
        print(f"[{i+1}/{len(configs)}] {c['tag']}: last50={row['last50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f} ({dt:.0f}s)")

        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"=== DONE part={part} shard={shard}/{nshards}: {len(rows)} runs, "
          f"summary -> {summary_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["lr", "lam", "lamrho", "joint", "selfweight"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    main(args.part, args.shard, args.nshards)
