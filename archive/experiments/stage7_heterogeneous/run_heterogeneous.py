# Objective #1 (2026-08): heterogeneous device collaboration, width-only
# (depth deferred per Christian's instruction). Same dynamic world, node
# placement (7x7 FOV, stride 3, 36 nodes), wearable motion, lr/lam as every
# other dynamic_v2 / gossip-comparator script this session -- variant
# ASSIGNMENT is the only thing that changes across arms, so any MSE
# difference is attributable to architecture heterogeneity, not a confound.
#
# Arms:
#   homogeneous  -- all 36 nodes "baseline" width (32,64,128). Not rerun here:
#                   identical to gossip_cmp_fusion from run_gossip_comparator.py
#                   (same field/seed/lr/lam/policy=random_wander), whole_run
#                   MSE 0.0208 -- reused as the control.
#   random       -- each node independently assigned narrow/baseline/wide
#                   (12 each, seeded shuffle).
#   clustered    -- spatial two-region split: the 18 nodes in the first three
#                   column-bands (cy in {3,6,9}) are "narrow", the other 18
#                   (cy in {12,15,18}) are "wide". No baseline nodes in this
#                   arm -- deliberately the sharpest possible architecture
#                   contrast, per spec ("narrow-region vs wide-region").
#
# REVISION (2026-08, Christian's explicit correction): the first pass of
# random/clustered (results saved under runs/stage7/heterogeneous/het_random,
# het_clustered) let narrow/wide nodes cold-start from random init inside the
# mesh, since the baseline checkpoint's conv/fc shapes don't fit a different
# width. That confounded "architecture heterogeneity" with "some nodes were
# never pretrained at all" -- whole_run MSE blew up to 0.21-0.26 (vs 0.0208
# homogeneous) and even BASELINE-width nodes inside those meshes degraded to
# ~0.20, because fusion has no mechanism to discount beliefs coming from an
# unconverged model (see mesh.py run_timestep's "IDENTIFIED LIMITATION" note).
# Those runs are KEPT on disk as evidence for that limitation, not deleted.
#
# This version instead loads each node from its OWN independently pretrained
# checkpoint (experiments/stage0_baseline/run.py --variant {narrow,wide},
# same procedure/data/seed as the baseline checkpoint -- see
# PRETRAINED_CHECKPOINTS below) so architecture capacity is the only
# remaining variable once all three variants clear a comparable held-out MSE
# gate. New arms are tagged het_random_v2 / het_clustered_v2 to avoid
# overwriting the cold-start evidence.
#
# All arms use mode="fusion" (belief exchange) -- verified via smoke test
# that this code path needs ZERO changes for mixed architectures (a 4-float
# NIG belief carries no architecture-dependent information). That itself is
# part of objective #1's required output.
#
# A separate function (attempt_parameter_exchange) then tries mode=
# "gossip_uniform" under the "random" variant assignment and is expected to
# fail with a shape-mismatch RuntimeError once a narrow/wide model's full
# state_dict gets forced into a differently-shaped receiver -- the failure
# itself is the objective #1 deliverable for parametric exchange, so it is
# caught, the FULL traceback is written to disk, and the script continues
# rather than crashing.
#
# Run: python -u experiments/stage7_heterogeneous/run_heterogeneous.py --arm <arm>
#   where <arm> in {random, clustered, gossip_failure}

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config
from beliefmesh.node.mesh import Mesh

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
PRETRAINED_CHECKPOINTS = {
    "narrow": Path("runs/stage0/narrow/checkpoints/pretrained_digit7.pth"),
    "baseline": CKPT,
    "wide": Path("runs/stage0/wide/checkpoints/pretrained_digit7.pth"),
}
ROOT = Path("runs/stage7/heterogeneous")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
N_NODES = 36


def random_variants(seed: int = SEED) -> list[str]:
    """12 narrow / 12 baseline / 12 wide, shuffled independent of position."""
    rng = np.random.default_rng(seed)
    variants = ["narrow"] * 12 + ["baseline"] * 12 + ["wide"] * 12
    rng.shuffle(variants)
    return variants


def clustered_variants(node_centres: np.ndarray) -> list[str]:
    """Spatial two-region split by column band (node_centres[:, 1] == cy):
    the three lower column-bands are narrow, the three upper are wide."""
    cy = node_centres[:, 1]
    threshold = np.median(np.unique(cy))
    return ["narrow" if c <= threshold else "wide" for c in cy]


def _common_kwargs(cfg, tag: str, node_variants: list[str], seed: int, mode: str = "fusion",
                    checkpoint=CKPT):
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # deterministic, same field every seed
    # matches run_gossip_comparator.py / run_dynamic_v2_5seed.py: seed 42 keeps
    # the original wander paths so the single-seed result reproduces exactly.
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]
    return dict(
        cfg=cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=checkpoint,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Heterogeneous ({tag})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        node_variants=node_variants,
    ), centres


def run_one_arm(arm: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)
    centres = np.load(ENV / "node_centres.npy")

    # variant ASSIGNMENT is a fixed, designed condition (like the field) --
    # only training/sampling/wander stochasticity varies across seeds, not
    # which node got which width, matching how the field is held fixed
    # across the gossip comparator's 5-seed replication.
    if arm == "random":
        variants = random_variants()
    elif arm == "clustered":
        variants = clustered_variants(centres)
    else:
        raise ValueError(arm)

    tag = f"het_{arm}_v2" if seed == SEED else f"het_{arm}_v2_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    kwargs, _ = _common_kwargs(cfg, tag, variants, seed, mode="fusion",
                                checkpoint=PRETRAINED_CHECKPOINTS)
    res = run_mesh_experiment(**kwargs)
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res['certainty_mse_pearson_r']
    print(f"\n### {tag}: whole_run={whole_run_mse:.4f} last50={res['mean_mse_last_50']:.4f} "
          f"r={r:.4f}")
    print(f"    per-variant MSE: {res['variant_mse']}")
    return whole_run_mse, res['mean_mse_last_50'], r


def run_arm(arm: str, seeds: list[int]):
    results = [run_one_arm(arm, seed) for seed in seeds]
    if len(seeds) > 1:
        arr = np.array(results)
        print(f"\n=== het_{arm}_v2 {len(seeds)}-SEED SUMMARY (seeds={seeds}) ===")
        print(f"whole_run MSE: mean={arr[:,0].mean():.4f} std={arr[:,0].std():.4f} "
              f"values={list(np.round(arr[:,0],4))}")
        print(f"last50 MSE:    mean={arr[:,1].mean():.4f} std={arr[:,1].std():.4f} "
              f"values={list(np.round(arr[:,1],4))}")
        print(f"r:             mean={arr[:,2].mean():.4f} std={arr[:,2].std():.4f} "
              f"values={list(np.round(arr[:,2],4))}")
    print("=== DONE ===")


def attempt_parameter_exchange():
    """Objective #1's required negative result: parameter exchange (gossip)
    under mixed architectures. Expected to fail -- the failure IS the
    deliverable. Full traceback saved to ROOT/gossip_failure/error.txt."""
    cfg = load_config()
    cfg.model.lr = LR
    out_dir = ROOT / "gossip_failure"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = random_variants()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    kwargs, centres = _common_kwargs(cfg, "gossip_failure", variants, SEED, mode="gossip_uniform")
    # run_mesh_experiment builds env/mesh internally; replicate just enough of
    # its setup here to call mesh.run_timestep directly and catch the failure
    # at the first step, rather than letting run_mesh_experiment's full 390
    # -step loop obscure which step/edge actually broke.
    from beliefmesh.data.grid_environment import GridEnvironment
    from beliefmesh.fusion.grid import circular_grid

    env = GridEnvironment(kwargs["all_grids"].shape[1], kwargs["all_grids"],
                          offset_field=kwargs["offset_field"], rotation_seed=SEED,
                          excluded_rotation_ranges=EXCLUDED_RANGES)
    mesh = Mesh(centres, fov_size=7, grid_size=kwargs["all_grids"].shape[1],
                environment=env, pretrained_path=CKPT, lr=LR,
                fusion_grid=circular_grid(cfg.fusion.grid_size), mode="gossip_uniform",
                lam=LAM, node_variants=variants)
    print(f"node variants: {variants}")
    positions = [kwargs["wearable_paths"][w][0] for w in range(N_WEARABLES)]
    try:
        for step in range(5):
            mesh.run_timestep(positions, step)
        print("NO FAILURE after 5 steps (unexpected -- overlap graph may not have "
              "connected mismatched-width nodes this run; try a different seed)")
    except Exception:
        tb = traceback.format_exc()
        (out_dir / "error.txt").write_text(
            f"node_variants: {variants}\n\n{tb}")
        print(f"=== CAPTURED FAILURE (see {out_dir / 'error.txt'}) ===")
        print(tb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["random", "clustered", "gossip_failure"], required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED),
                        help="comma-separated seed list, e.g. '1042,2042,3042,4042'")
    args = parser.parse_args()
    if args.arm == "gossip_failure":
        attempt_parameter_exchange()
    else:
        run_arm(args.arm, [int(s) for s in args.seeds.split(",")])
