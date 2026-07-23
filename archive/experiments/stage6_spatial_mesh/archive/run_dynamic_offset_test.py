# THE decisive test. PRE-REGISTERED 2026-07-21, before running.
#
# Fixes the temporal-staticness flaw in the priority batch: the mapping now
# genuinely drifts, at different rates, starting at different times, in
# different quadrants (build_dynamic_offset_field). Revisited local knowledge
# can go actively WRONG, not just suboptimal -- the condition under which
# consensus/epistemic uncertainty carry real information and a continually-
# retrained global model must chase a moving target everywhere except its
# current anchor cell.
#
# Two arms only, both using OWN-MAP epistemic+staleness routing (the fairest
# comparison: each system routes using its own uncertainty signal):
#   fusion (belief exchange)      vs      fedavg_global (parameter exchange)
#
# CHRISTIAN'S PREDICTION (recorded verbatim before running): "our system
# largely maintain performance while fedavg suffers massively."
#
# Run: python -u experiments/stage6_spatial_mesh/run_dynamic_offset_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import git_commit
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         dynamic_analytic_floors)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/dynamic_offset")


def main():
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    path = [np.load(ENV / "wearable_path.npy")]
    centres = np.load(ENV / "node_centres.npy")

    field = build_dynamic_offset_field(G, T)
    gf, nf = dynamic_analytic_floors(field, centres, 7, G, window=50)
    base_err = 0.0
    print(f"DYNAMIC FLOORS (last-50-step mean of instantaneous floors, OMNISCIENT bound):")
    print(f"  global (any time-invariant OR perfectly-tracking-but-shared model): {gf:.4f}")
    print(f"  per-node (omniscient local, no lag): {nf:.4f}")
    print("  field: smooth RBF-interpolated background (dynamic_v2/v3 mechanism) "
          "+ triangular wake behind a fixed block, direction sweeping slowly, "
          "turbulence period 110 steps -- validated by field_preview.mp4")
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "dynamic_floors.yaml", "w") as f:
        yaml.safe_dump({"git_commit": git_commit(), "global_floor": gf, "per_node_floor": nf,
                        "prediction": "Christian: fusion largely maintains, fedavg suffers massively"}, f)

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        results[tag] = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=path, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Dynamic offset world ({tag})",
            offset_field=field, wearable_policy="epistemic_staleness",
        )
        print(f"### {tag}: {results[tag]['mean_mse_last_50']:.4f} "
              f"(r {results[tag]['certainty_mse_pearson_r']:.3f})")

    run("dynamic_fusion_epistaleness", "fusion")
    run("dynamic_global_epistaleness", "fedavg_global")

    print("\n=== DYNAMIC OFFSET TEST SUMMARY ===")
    print(f"floors -- global {gf:.4f} | per-node {nf:.4f}")
    for tag, res in results.items():
        print(f"{tag}: {res['mean_mse_last_50']:.4f} (r {res['certainty_mse_pearson_r']:.3f})")
    print("=== DYNAMIC OFFSET TEST DONE ===")


if __name__ == "__main__":
    main()
