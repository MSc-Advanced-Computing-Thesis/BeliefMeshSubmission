# Forensic finding: mesh.py's trust-tempered gradient (consensus-tempered
# loss weighting) was implemented at 15:25:21 on 2026-07-29, but EVERY lam
# sweep run from consensus_lam1.0 onward (created 15:34:23+) was launched
# AFTER that edit -- meaning every "lam effect" result from lam=1.0 up is
# actually lam-change + tempering, confounded together. Only the lam=0.1
# baseline and the rho-extremes-at-default-lam runs (both pre-15:25:21) are
# genuinely untempered.
#
# This reruns the exact consensus_lam1.0 config (lam=1.0, rho=0.2 default,
# seed=42, identical env/paths) with temper_gradient=False, to isolate
# tempering's own effect: compare directly against the already-saved
# consensus_lam1.0 result (last50=0.0359 whole_run=0.0252 r=-0.4363,
# cert mean=0.9862 std=0.0112 range=0.0693), which has tempering ON.
#
# Run: python -u experiments/stage6_spatial_mesh/run_tempering_ab.py

from __future__ import annotations

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
ROOT = Path("runs/stage6/offset_world/dynamic_lam_rho_sweep")
N_WEARABLES = 3
LAM = 1.0


def main():
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = f"consensus_lam{LAM}_untempered"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="consensus", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"tempering A/B, tempering OFF ({tag})",
        offset_field=field, lam=LAM, temper_gradient=False,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    cert_map = np.load(ROOT / tag / "avg_cert_map.npy")
    valid = cert_map[~np.isnan(cert_map)]

    print(f"\n### {tag} (tempering OFF): last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")
    print(f"certainty spread: mean={valid.mean():.4f} std={valid.std():.4f} "
          f"range={valid.max()-valid.min():.4f}")
    print("(reference, consensus_lam1.0 WITH tempering: last50=0.0359 whole_run=0.0252 "
          "r=-0.4363 cert_mean=0.9862 cert_std=0.0112 range=0.0693)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
