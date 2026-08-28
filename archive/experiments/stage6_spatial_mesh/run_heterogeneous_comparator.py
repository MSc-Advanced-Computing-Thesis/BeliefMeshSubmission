# Heterogeneous-device comparator (2026-08). The evidence-asymmetry arc
# established that fusion == naive in a HOMOGENEOUS mesh, and diagnosed why:
# gamma* = sum(nu_i gamma_i)/sum(nu_i) reduces exactly to the plain mean when
# the nu_i are equal, and identical nodes trained on near-identical data
# accumulate near-identical evidence (nu-CV ~0.03, weight-dev ~0.011, so
# fusion's answer lands ~0.19 deg from naive's against a ~33 deg RMSE).
#
# Every construction in that arc tried to induce nu asymmetry through the
# ENVIRONMENT, which is shared by all contributors to a cell and therefore
# hits them equally. This script attacks the other side: make the DEVICES
# unequal. narrow/baseline/wide width variants (models/variants.py), each
# loading its OWN independently pretrained Stage 0 checkpoint so capacity is
# not confounded with transfer damage or a cold init.
#
# Hypothesis: a narrow node genuinely knows less, the evidential head's
# regulariser (|diff|*(2 nu + alpha)) drives its nu down accordingly, nu-CV
# widens, and fusion down-weights it automatically while naive averages it
# in at full strength. If so, fusion earns its place on heterogeneous
# hardware -- which is the deployment story -- without ever needing to beat
# naive on homogeneous hardware.
#
# Matched control: the homogeneous 60 deg / no-colour-filter runs from
# run_large_offset_comparator.py (naive & nig_product both whole_run=0.0101).
# Everything here is identical to those except node_variants.
#
# Run: python -u experiments/stage6_spatial_mesh/run_heterogeneous_comparator.py --mode nig_product

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPTS = {
    "narrow":   Path("runs/stage0/narrow/checkpoints/pretrained_digit7.pth"),
    "baseline": Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth"),
    "wide":     Path("runs/stage0/wide/checkpoints/pretrained_digit7.pth"),
}
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["naive", "certainty", "nig_product"]
VARIANT_SEED = 42


def assign_variants(n_nodes: int, seed: int = VARIANT_SEED) -> list[str]:
    """Equal thirds narrow/baseline/wide, shuffled so capacity is spatially
    INTERLEAVED rather than clustered -- every fusion event then mixes
    capacities, which is the condition the hypothesis is about. A clustered
    layout would leave most events homogeneous within their neighbourhood and
    dilute the effect."""
    per = n_nodes // 3
    variants = (["narrow"] * per + ["baseline"] * per
                + ["wide"] * (n_nodes - 2 * per))
    rng = np.random.default_rng(seed)
    rng.shuffle(variants)
    return variants


def run_one(mode: str, seed: int, homogeneous: bool = False, root_override=None):
    cfg = load_config()
    cfg.model.lr = LR
    root = Path(root_override) if root_override else Path(
        "runs/stage6/offset_world/heterogeneous_comparator"
                + ("_homog" if homogeneous else ""))
    root.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)          # unused once the filter is off
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)    # default 60 deg, matched to control
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]

    # variant layout is reseeded per run seed, so replication varies the mesh
    # COMPOSITION as well as the environment/wearable realisation -- otherwise
    # all five seeds would re-test one fixed device layout. seed 42 reproduces
    # the original layout exactly (VARIANT_SEED == 42).
    variants = None if homogeneous else assign_variants(len(centres), seed=seed)
    if variants is not None:
        counts = {v: variants.count(v) for v in ("narrow", "baseline", "wide")}
        print(f"[het] node variants: {counts}")

    tag = f"het_{mode}" if seed == SEED else f"het_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=f"heterogeneous_{tag}", run_dir=root / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode,
        baseline_checkpoint=(CKPTS["baseline"] if homogeneous else CKPTS),
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Heterogeneous comparator ({mode}, seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        node_variants=variants,
        track_disagreement=(mode == "nig_product"),
        apply_colour_filter=False,
    )
    cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
    cell_cert_steps = np.load(root / tag / "cell_cert_steps.npy")
    T = cell_mse_steps.shape[0]
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    new_mean, new_std, n_t, _ = per_timestep_then_averaged(
        cell_mse_steps, cell_cert_steps, T - 50, T)
    print(f"\n### heterogeneous/{tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r(new)={new_mean:+.4f}+/-{new_std:.4f}")
    if res.get("variant_mse"):
        print(f"    variant_mse: {res['variant_mse']}")
    dstats = res.get("disagreement_stats")
    if dstats:
        print(f"    disagreement_stats: {dstats}")
    return whole_run_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    parser.add_argument("--seeds", type=str, default=str(SEED))
    parser.add_argument("--homogeneous", action="store_true",
                        help="matched control: same settings, all-baseline nodes")
    args = parser.parse_args()
    for seed in [int(s) for s in args.seeds.split(",")]:
        run_one(args.mode, seed, homogeneous=args.homogeneous, root_override=args.root)
    print("=== DONE ===")
