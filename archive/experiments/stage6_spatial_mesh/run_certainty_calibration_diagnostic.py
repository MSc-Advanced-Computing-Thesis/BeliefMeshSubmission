# Certainty-calibration decay diagnostic (2026-08), extended for the lambda
# ablation. Motivation: colour-world cert-MSE r collapses from -0.853
# (frozen) to -0.048 (trained nig_product, lam=5); offset-world decays less
# severely. Question (this pass): does the evidence-regularisation
# coefficient lam control this erosion, or does beta* merely track falling
# residual error appropriately regardless of lam? Same nig_product mechanism,
# same node placement/wearable convention/seed as the existing comparator
# runs -- NO change to fusion/training arithmetic, only additional per-step
# logging: mesh.evaluate()'s per-node (mse, certainty, hop_distance), the
# (step, nu*, beta*, applied cert) log on Mesh.applied_uncertainty_log, AND
# (new) the standard cell-level best-certainty-per-cell bookkeeping
# (cell_mse_steps/cell_cert_steps) used throughout this session's other
# comparator scripts, so whole_run/last50/r here are numerically comparable
# to every previously-reported figure using the identical methodology.
#
# Run: python -u experiments/stage6_spatial_mesh/run_certainty_calibration_diagnostic.py --world offset --lam 5.0
#      python -u experiments/stage6_spatial_mesh/run_certainty_calibration_diagnostic.py --world colour --lam 15.0

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.node.mesh import Mesh

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/certainty_calibration_diagnostic")
N_WEARABLES = 3
LR = 3e-5
DEFAULT_LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
LAST_N_STEPS = 50


def build(world: str, seed: int = SEED, lam: float = DEFAULT_LAM):
    centres = np.load(ENV / "node_centres.npy")
    if world == "offset":
        grids = np.load(ENV / "environment_grids.npy")
        T, G = grids.shape[0], grids.shape[1]
        all_grids = np.full((T, G, G), 0.5)
        offset_field = build_dynamic_offset_field(G, T)
    elif world == "colour":
        all_grids = np.load(ENV / "environment_grids.npy")
        T, G = all_grids.shape[0], all_grids.shape[1]
        offset_field = None
    else:
        raise ValueError(world)

    env = GridEnvironment(G, all_grids, offset_field=offset_field, rotation_seed=seed,
                          excluded_rotation_ranges=EXCLUDED_RANGES)
    from beliefmesh.config import load_config
    cfg = load_config()
    mesh = Mesh(centres, fov_size=7, grid_size=G, environment=env, pretrained_path=CKPT,
               lr=LR, fusion_grid=circular_grid(cfg.fusion.grid_size), mode="nig_product",
               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               sample_seed=seed, lam=lam)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]
    return mesh, paths, T, G, env


def run(world: str, seed: int = SEED, lam: float = DEFAULT_LAM):
    mesh, paths, T, G, env = build(world, seed, lam)
    n_nodes = len(mesh.nodes)

    node_mse = np.full((T, n_nodes), np.nan)
    node_cert = np.full((T, n_nodes), np.nan)
    node_hop = np.full((T, n_nodes), np.nan)
    node_cum_steps = np.zeros((T, n_nodes), dtype=np.int64)
    cell_mse_steps = np.full((T, G, G), np.nan)
    cell_cert_steps = np.full((T, G, G), np.nan)

    for step in range(T):
        positions = [paths[w][step] for w in range(N_WEARABLES)]
        mesh.run_timestep(positions, step, n_wearable_samples=1, n_train_repeats=1)
        metrics = mesh.evaluate(step)
        for node_id, m in metrics.items():
            if m["mse"] is not None:
                node_mse[step, node_id] = m["mse"]
                node_cert[step, node_id] = m["certainty"]
            if m["hop_distance"] is not None:
                node_hop[step, node_id] = m["hop_distance"]
            node_cum_steps[step, node_id] = mesh.nodes[node_id].n_samples_trained

        # standard best-certainty-per-cell bookkeeping (matches runner.py exactly)
        step_best = {}
        for node in mesh.nodes.values():
            for cell, (g, n, a, b) in node.cell_beliefs.items():
                cert = 1.0 / (1.0 + min(b / (max(n, 1e-6) * max(a - 1, 1e-6)), 10.0))
                if cell not in step_best or cert > step_best[cell][1]:
                    step_best[cell] = (g, cert)
        for cell, (pred, cert) in step_best.items():
            _, truth = env.get_cell_input(cell[0], cell[1], step)
            err = circular_diff(torch.tensor(pred), truth).item() ** 2
            cell_mse_steps[step, cell[0], cell[1]] = err
            cell_cert_steps[step, cell[0], cell[1]] = cert

        if step % 30 == 0:
            valid = ~np.isnan(node_mse[step])
            print(f"[{world} lam={lam}] step {step:03d}/{T} nodes_with_beliefs={valid.sum()}/{n_nodes} "
                  f"mean_mse={np.nanmean(node_mse[step]):.4f}", flush=True)

    applied_log = np.array(mesh.applied_uncertainty_log, dtype=np.float64)  # (n_events, 5)

    tag = f"lam{lam:g}"
    out_dir = ROOT / "lam_ablation" / world / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "diagnostic_arrays.npz",
            node_mse=node_mse, node_cert=node_cert, node_hop=node_hop,
            node_cum_steps=node_cum_steps, applied_log=applied_log,
            cell_mse_steps=cell_mse_steps, cell_cert_steps=cell_cert_steps, lam=lam)

    whole_run_mse = float(np.nanmean(cell_mse_steps))
    avg_mse_last = np.nanmean(cell_mse_steps[-LAST_N_STEPS:], axis=0)
    avg_cert_last = np.nanmean(cell_cert_steps[-LAST_N_STEPS:], axis=0)
    valid = np.isfinite(avg_mse_last) & np.isfinite(avg_cert_last)
    from scipy.stats import pearsonr
    r_val = float(pearsonr(avg_cert_last[valid], avg_mse_last[valid])[0]) if valid.sum() > 2 else float("nan")
    last50_mse = float(np.nanmean(avg_mse_last))
    print(f"[{world} lam={lam}] whole_run={whole_run_mse:.4f} last50={last50_mse:.4f} r={r_val:.4f}")
    print(f"[{world} lam={lam}] saved {out_dir / 'diagnostic_arrays.npz'} "
          f"(applied_log shape={applied_log.shape})")
    return whole_run_mse, last50_mse, r_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=["offset", "colour"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAM)
    args = parser.parse_args()
    run(args.world, args.seed, args.lam)
