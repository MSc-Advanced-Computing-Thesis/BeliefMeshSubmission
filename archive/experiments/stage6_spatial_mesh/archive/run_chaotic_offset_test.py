# First real test of build_chaotic_offset_field (run_offset_experiments.py):
# the original dynamic offset world turned out to be too spatially smooth
# (adjacent cells ~2-3 deg apart) and too temporally static (~0.06 deg/step)
# to give fusion/consensus's per-region specialisation a real advantage over
# fedavg_global's single shared model -- confirmed directly in the
# single-wearable test, where fedavg_global won despite having zero spatial
# specialisation, because the field was gentle enough that "stay fresh
# everywhere, specialise nowhere" beat "specialise, but risk staleness".
#
# This field is tuned (validated analytically via dynamic_analytic_floors
# BEFORE running anything) to have a genuinely bigger fusion-vs-global gap:
# global floor 0.0133->0.0674, per-node floor 0.0027->0.0076, ratio
# 4.97x->8.89x, plus ~40x faster temporal drift (2.4 vs 0.06 deg/step) for
# real staleness pressure. Standard 36-node/22x22 layout, otherwise
# unchanged.
#
# Also raises lr 3e-4 -> 1e-3 (~3.3x) per Christian's request ("we may have
# to jack up the learning rates") -- faster local drift needs faster local
# adaptation to actually be trackable, for both the per-node specialists
# AND fedavg_global's continuously-retrained shared model.
#
# Three arms: fusion, fedavg_global, consensus (today's tuned lam=10/rho=0.99).
# Single seed=42 -- exploratory, not a replicated result.
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_offset_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_chaotic_offset_field,
                                                         build_random_wander_path,
                                                         dynamic_analytic_floors)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_chaotic")
N_WEARABLES = 3
LR = 1e-3
LAM = 10.0
RHO = 0.99


def main(arms: list[str] = ("fusion", "fedavg_global", "consensus")):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_chaotic_offset_field(G, T)
    np.save(ROOT / "chaotic_field.npy", field)

    gf, nf = dynamic_analytic_floors(field, centres, 7, G, window=50)
    print(f"analytic floors (last-50 instantaneous): global={gf:.4f} per_node={nf:.4f} "
          f"ratio={gf/nf:.2f}x")
    print(f"field peak abs offset={np.abs(field).max():.1f} deg (ceiling 180), lr={LR}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode, **kwargs):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Chaotic offset world ({tag}, lr={LR})",
            offset_field=field, **kwargs,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}{r_str}")

    if "fusion" in arms:
        run("chaotic_fusion", "fusion")
    if "fedavg_global" in arms:
        run("chaotic_fedavg_global", "fedavg_global")
    if "consensus" in arms:
        run("chaotic_consensus_lam10_rho099", "consensus", lam=LAM, rho=RHO)

    print("\n=== CHAOTIC OFFSET WORLD RESULT ===")
    for tag, (last50, whole, r) in results.items():
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}{r_str}")
    print(f"(analytic floors: global={gf:.4f} per_node={nf:.4f})")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global", "consensus"],
                        default=["fusion", "fedavg_global", "consensus"])
    args = parser.parse_args()
    main(args.arms)
