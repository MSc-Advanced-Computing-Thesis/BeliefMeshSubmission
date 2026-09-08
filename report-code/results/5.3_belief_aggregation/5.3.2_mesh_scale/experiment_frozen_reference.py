# Aggregation-mechanism comparator (2026-08): how much of each contributor's
# belief is actually needed for peer supervision to work? Three levels of
# information retained per contributor:
#   frozen     -- zero-adaptation reference (no training at all)
#   naive      -- point estimate only (plain mean of contributor gammas)
#   certainty  -- point estimate + a scalar reliability score (certainty-
#                 weighted mean of gammas -- the prior repo's actual
#                 mechanism, Defect 1 from the original audit, reproduced
#                 here deliberately as a control -- see _aggregate's
#                 mode=="certainty" branch docstring in mesh.py)
#   nig_product -- the full (gamma, nu, alpha, beta) distribution, closed-form
#                 product-of-NIG fusion
#
# Same offset-world field/seed/wearable-policy/node-placement (7x7 stride 3,
# 36 nodes, random_wander) as run_gossip_comparator.py, so this sits on the
# same axis -- unlike the interim's naive-aggregation result, which was only
# ever run on an older colour-world configuration (pre-CoordConv-fix,
# different lr/lam), this is the first same-axis run of naive/certainty
# against the current nig_product baseline.
#
# Run: python -u experiments/stage6_spatial_mesh/run_aggregation_comparator.py --mode naive --seeds 42

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from beliefmesh.simulation.cert_mse_metrics import per_timestep_then_averaged
from beliefmesh.simulation.runner import run_mesh_experiment
from beliefmesh.simulation.offset_fields import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = ENVIRONMENT_DIR
CKPT = CHECKPOINTS["baseline"]
N_WEARABLES = 3
LR = 3e-5
DEFAULT_LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["frozen", "naive", "certainty", "nig_product"]


def run_one(mode: str, seed: int, world: str = "offset", lam: float = DEFAULT_LAM,
           track_disagreement: bool = False, root_override=None):
    cfg = load_config()
    cfg.model.lr = LR
    root = (Path(root_override) if root_override
            else Path(ART + "/5.3_belief_aggregation/5.3.2_mesh_scale/frozen"))
    if lam != DEFAULT_LAM:
        root = root.parent / f"{root.name}_lam{lam:g}"
    root.mkdir(parents=True, exist_ok=True)

    centres = np.load(ENV / "node_centres.npy")
    if world == "offset":
        grids = np.load(ENV / "environment_grids.npy")
        T, G = grids.shape[0], grids.shape[1]
        all_grids = np.full((T, G, G), 0.5)
        field = build_dynamic_offset_field(G, T)
    elif world == "colour":
        all_grids = np.load(ENV / "environment_grids.npy")
        T, G = all_grids.shape[0], all_grids.shape[1]
        field = None
    else:
        raise ValueError(world)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"agg_cmp_{mode}" if seed == SEED else f"agg_cmp_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=root / tag,
        all_grids=all_grids, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Aggregation comparator ({mode}, seed={seed}, world={world}, lam={lam})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=lam, wearable_policy=None, policy_step_size=0.4,
        track_disagreement=track_disagreement,
    )
    cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
    cell_cert_steps = np.load(root / tag / "cell_cert_steps.npy")
    T = cell_mse_steps.shape[0]
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    new_mean, new_std, n_t, _ = per_timestep_then_averaged(cell_mse_steps, cell_cert_steps, T - 50, T)
    print(f"\n### {tag} [{world}, lam={lam}]: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} "
          f"r(new)={new_mean:+.4f}+/-{new_std:.4f} (n_t={n_t})")
    dstats = res.get("disagreement_stats")
    if dstats:
        print(f"    disagreement_stats: {dstats}")
    return whole_run_mse, res['mean_mse_last_50'], new_mean, new_std


def main(mode: str, seeds: list[int], world: str = "offset", lam: float = DEFAULT_LAM,
        track_disagreement: bool = False, root_override=None):
    results = []
    for seed in seeds:
        results.append(run_one(mode, seed, world=world, lam=lam,
                               track_disagreement=track_disagreement,
                               root_override=root_override))
    if len(seeds) > 1:
        arr = np.array(results)  # (n_seeds, 4): whole_run, last50, r_mean, r_std
        print(f"\n=== {mode} {len(seeds)}-SEED SUMMARY (seeds={seeds}) ===")
        print(f"whole_run MSE: mean={arr[:,0].mean():.4f} std={arr[:,0].std():.4f} "
              f"values={list(np.round(arr[:,0],4))}")
        print(f"last50 MSE:    mean={arr[:,1].mean():.4f} std={arr[:,1].std():.4f} "
              f"values={list(np.round(arr[:,1],4))}")
        print(f"r (new, per-timestep-then-avg): mean={arr[:,2].mean():.4f} std_across_seeds={arr[:,2].std():.4f} "
              f"avg_within_run_std={arr[:,3].mean():.4f} values={list(np.round(arr[:,2],4))}")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED))
    parser.add_argument("--world", choices=["offset", "colour"], default="offset")
    parser.add_argument("--lam", type=float, default=DEFAULT_LAM)
    parser.add_argument("--track-disagreement", action="store_true")
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    args = parser.parse_args()
    main(args.mode, [int(s) for s in args.seeds.split(",")], world=args.world, lam=args.lam,
        track_disagreement=args.track_disagreement, root_override=args.root)
