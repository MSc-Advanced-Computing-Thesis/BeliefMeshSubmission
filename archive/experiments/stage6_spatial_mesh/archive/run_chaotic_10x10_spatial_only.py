# 10x10 scale-up of the winning 36-node config: new-spatial/old-temporal
# field (n_keyframes=4, wide/sharp/big-amplitude regions -- floor ratio
# 9.17x at this scale, drift=0.31 deg/step, well under the ~2.4 deg/step
# that broke convergence) -- the first environment all day where fusion
# genuinely beat fedavg_global (0.0531 vs 0.0674 at 36 nodes). Christian's
# hypothesis: since a bigger grid widens the analytic floor gap further in
# fusion's favour (globally, more far-apart disagreement; per-node floor
# stays low as long as spacing >> FOV, already checked), fusion's advantage
# should be "further exacerbated" at 10x10. lam=10 per explicit request
# (though today's calibration-ratio check showed lam doesn't add real
# informativeness -- included anyway for the wider certainty-spread
# property useful elsewhere, e.g. routing policies).
#
# G=31, FOV=7, STRIDE=3 -> 100 nodes, lr=3e-4 (corrected), n_train_repeats=1,
# lam=10.0. --n-wearables CLI arg for the 6- and 3-wearable variants run in
# parallel, matching the earlier fast-temporal 10x10 comparison.
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_10x10_spatial_only.py --n-wearables 6
#      python -u experiments/stage6_spatial_mesh/run_chaotic_10x10_spatial_only.py --n-wearables 3

from __future__ import annotations

import argparse
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
from stage6_spatial_mesh.run_6abc_overlap_ablation import node_grid

from beliefmesh.config import load_config

CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
G = 31
T = 390
FOV = 7
STRIDE = 3
LR = 3e-4
LAM = 10.0
N_KEYFRAMES = 4  # winning (old) temporal pace


def main(n_wearables: int, arms: list[str]):
    cfg = load_config()
    cfg.model.lr = LR
    root = Path(f"runs/stage6/offset_world/dynamic_chaotic_10x10_spatial_only/w{n_wearables}")
    root.mkdir(parents=True, exist_ok=True)

    centres = node_grid(FOV, STRIDE, G)
    print(f"nodes: {len(centres)} (expect 100), n_wearables={n_wearables}, lr={LR}, lam={LAM}")
    uniform = np.full((T, G, G), 0.5)

    field = build_chaotic_offset_field(G, T, n_keyframes=N_KEYFRAMES)
    np.save(root / "field.npy", field)
    gf, nf = dynamic_analytic_floors(field, centres, FOV, G, window=50)
    print(f"analytic floors: global={gf:.4f} per_node={nf:.4f} ratio={gf/nf:.2f}x "
          f"peak={np.abs(field).max():.1f}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(n_wearables)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=root / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=FOV, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Chaotic 10x10 spatial-only, {n_wearables} wearables, lam={LAM} ({tag})",
            offset_field=field, lam=LAM,
        )
        cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}{r_str}")

    if "fusion" in arms:
        run("chaotic10x10so_fusion", "fusion")
    if "fedavg_global" in arms:
        run("chaotic10x10so_fedavg_global", "fedavg_global")

    print(f"\n=== CHAOTIC 10x10 SPATIAL-ONLY RESULT (n_wearables={n_wearables}) ===")
    for tag, (last50, whole, r) in results.items():
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}{r_str}")
    print("(reference, 36-node spatial-only @ lam=0.1: fusion whole_run=0.0531, "
          "fedavg_global whole_run=0.0674)")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wearables", type=int, required=True)
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global"],
                        default=["fusion", "fedavg_global"])
    args = parser.parse_args()
    main(args.n_wearables, args.arms)
