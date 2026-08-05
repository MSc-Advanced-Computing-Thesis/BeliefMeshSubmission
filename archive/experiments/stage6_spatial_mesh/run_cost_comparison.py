# Per-timestep cost-share comparison (forward pass / backward+optimiser-step /
# fusion step) for fusion (grid search) vs nig_product (closed form), 2026-08.
#
# Runs DIRECTLY against Mesh.run_timestep() for a short window (not the full
# 390-step run_mesh_experiment harness) on the SAME real 36-node offset-world
# config (7x7 stride 3, seed 42, same field/wearable paths as every other
# seed-42 comparison this session) -- cost-share PERCENTAGES are a property
# of the per-call operation mix, not of run length, so a representative
# window is sufficient and does not need the full accuracy run.
#
# CPU, deliberately: torch.cuda.synchronize() around every forward/backward
# call (needed for accurate GPU wall-clock) proved pathologically slow under
# this machine's WDDM driver at the tens-of-thousands-of-calls scale a full
# run needs (stalled the GPU into a 100%-util-at-idle-power hang). Switching
# to torch.cuda.Event pairs avoided the hang but per-call Event creation
# still added enough overhead to make even a short run impractically slow.
# CPU-side time.perf_counter() needs no synchronization at all -- exact by
# construction, no instrumentation-overhead artifact to reason about.

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config
from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.node.mesh import Mesh

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
N_STEPS = 8  # representative window -- cost SHARE doesn't depend on run length


def run_one(mode: str):
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    env = GridEnvironment(G, uniform, offset_field=field, rotation_seed=SEED,
                          excluded_rotation_ranges=EXCLUDED_RANGES)
    mesh = Mesh(centres, fov_size=7, grid_size=G, environment=env,
                pretrained_path=CKPT, lr=LR, fusion_grid=circular_grid(cfg.fusion.grid_size),
                mode=mode, device=torch.device("cpu"), sample_seed=SEED, lam=LAM)

    t_wall0 = time.perf_counter()
    for step in range(N_STEPS):
        positions = [paths[w][step] for w in range(N_WEARABLES)]
        t_step0 = time.perf_counter()
        mesh.run_timestep(positions, step)
        print(f"    [{mode}] step {step} done in {time.perf_counter()-t_step0:.2f}s", flush=True)
    wall = time.perf_counter() - t_wall0

    fwd = mesh.forward_time_cumulative
    bwd = mesh.backward_time_cumulative
    fus = mesh.fusion_time_cumulative
    total_measured = fwd + bwd + fus
    print(f"\n### cost_cmp_{mode} (CPU, {N_STEPS} steps) ###")
    print(f"    forward_time_total={fwd:.4f}s  ({100*fwd/total_measured:.2f}% of measured)")
    print(f"    backward_time_total={bwd:.4f}s ({100*bwd/total_measured:.2f}% of measured)")
    print(f"    fusion_time_total={fus:.4f}s   ({100*fus/total_measured:.2f}% of measured)")
    print(f"    sum(forward+backward+fusion)={total_measured:.4f}s")
    print(f"    real wall-clock for {N_STEPS} steps={wall:.4f}s "
          f"({100*total_measured/wall:.1f}% of real wall-clock accounted for)")
    print(f"    fusion_call_count={mesh.fusion_call_count}")
    return dict(mode=mode, forward=fwd, backward=bwd, fusion=fus, wall=wall,
                fusion_calls=mesh.fusion_call_count)


if __name__ == "__main__":
    results = [run_one("fusion"), run_one("nig_product")]
    print("\n=== SUMMARY ===")
    for r in results:
        total = r['forward'] + r['backward'] + r['fusion']
        print(f"{r['mode']}: forward={r['forward']:.3f}s backward={r['backward']:.3f}s "
              f"fusion={r['fusion']:.4f}s ({100*r['fusion']/total:.3f}% of measured) "
              f"calls={r['fusion_calls']}")
    print("=== DONE ===")
