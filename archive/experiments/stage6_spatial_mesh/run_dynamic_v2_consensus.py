# Consensus mode on the dynamic-world v2 (real drift + moving wake), the
# first place this session rho has ever had something to bite on -- the
# static-spatial world's lam x rho ablation found rho essentially inert
# there (temporally frozen field, no staleness for the trust-EMA to track).
#
# Two things this script does, selected by --part:
#   baseline -- single run at (lam=5, rho=0.2 -- today's defaults), same
#               field/wearable-paths/seed=42 as the fusion/fedavg_global
#               dynamic_v2 comparison, wearable_policy=random_wander (matches
#               fedavg_global's arm for a clean 3-way comparison; routing is
#               a separate axis, not conflated here).
#   rho      -- rho sweep at lam=5 fixed, mode=consensus, same world/seed,
#               to see whether rho now has a measurable effect.
#
# Reference (dynamic_v2, seed=42, lam=5, random_wander unless noted):
#   fusion:         whole_run=0.0208 r=-0.335
#   fedavg_global:  whole_run=0.0237 r=-0.118
#
# Run:
#   python -u experiments/stage6_spatial_mesh/run_dynamic_v2_consensus.py --part baseline
#   python -u experiments/stage6_spatial_mesh/run_dynamic_v2_consensus.py --part rho

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_v2_consensus")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
RHO_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99]
SEEDS = [42, 1042, 2042, 3042, 4042]


def _setup(seed: int = SEED):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]
    return cfg, uniform, centres, field, paths


def run_one(cfg, uniform, centres, field, paths, tag, rho, seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="consensus", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Dynamic v2 consensus ({tag})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, rho=rho, wearable_policy=None, policy_step_size=0.4,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res.get("certainty_mse_pearson_r")
    print(f"### {tag} (rho={rho}): last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={r:.4f}")
    return whole_run_mse, r


def main(part: str, shard: int = 0, nshards: int = 1,
         root: str | None = None, seeds: list[int] | None = None):
    global ROOT
    if root:
        ROOT = Path(root)
    seeds = seeds or SEEDS

    if part == "baseline":
        for seed in seeds:
            cfg, uniform, centres, field, paths = _setup(seed)
            run_one(cfg, uniform, centres, field, paths,
                    "consensus_baseline_rho0p2_seed%d" % seed, 0.2, seed)
        print("(reference: fusion=0.0208 r=-0.335, fedavg_global=0.0237 r=-0.118, "
              "same seed/field/wearable-paths)")
        print("=== DONE ===")
    elif part == "rho":
        my_rhos = RHO_GRID[shard::nshards]
        results = []
        for seed in seeds:
            cfg, uniform, centres, field, paths = _setup(seed)
            for rho in my_rhos:
                tag = "consensus_rho%s_seed%d" % (str(rho).replace(".", "p"), seed)
                whole_run, r = run_one(cfg, uniform, centres, field, paths,
                                       tag, rho, seed)
                results.append((rho, whole_run, r))
        print(f"\n=== RHO ABLATION shard {shard}/{nshards} (dynamic v2, lam=5, seed=42, random_wander) ===")
        import statistics as _st
        from collections import defaultdict as _dd
        byrho = _dd(list)
        for rho, whole_run, r in results:
            byrho[rho].append(whole_run)
        for rho in sorted(byrho):
            v = byrho[rho]
            sd = _st.stdev(v) if len(v) > 1 else 0.0
            print("rho=%-5s n=%d whole_run=%.5f +/- %.5f"
                  % (rho, len(v), _st.mean(v), sd))
        print("=== DONE ===")
    else:
        raise ValueError(part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["baseline", "rho"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seed list")
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    main(args.part, args.shard, args.nshards, args.root, seeds)