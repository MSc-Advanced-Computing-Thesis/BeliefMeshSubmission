# Sampled-target arms in the FULL mesh, plus the bit-identity verification for
# the two code-path changes they depend on.
#
# The body of run_one() below is a verbatim replica of
# run_aggregation_comparator.run_one's configuration -- same ENV, checkpoint,
# LR, LAM, N_WEARABLES, wearable seeding, excluded ranges, fov, repeats -- with
# exactly two additions: an output root that is never an existing directory,
# and the sample_target flag. That script is a reported code path and is not
# modified. The replica's fidelity is not asserted, it is PROVEN: --verify runs
# it with sample_target=False at seed 42 and compares cell_mse_steps against
# the stored agg_cmp_nig_product array bit-for-bit.
#
#   python -u experiments/section5_2/run_mesh_sampled.py --verify
#   python -u experiments/section5_2/run_mesh_sampled.py --mode nig_product --seeds 42,1042,...
#
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

# ── verbatim from run_aggregation_comparator.py ───────────────────────────
ENV = ENVIRONMENT_DIR
CKPT = CHECKPOINTS["baseline"]
N_WEARABLES = 3
LR = 3e-5
DEFAULT_LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]

OUT = Path(ART + "/5.3_belief_aggregation/5.3.2_mesh_scale/aggregation_arms")
import tempfile as _tf
_SCRATCH = _Path(_tf.gettempdir()) / "beliefmesh_verify"
VERIFY_OUT = _SCRATCH / "mesh_aggregation"
# --verify replays the mode-target run against a stored reference run; the
# reference (the pre-Chapter-5 aggregation comparator) is not part of this
# repository, so --verify reports the comparison only when it is present.
REFERENCE = Path(ART + "/5.3_belief_aggregation/5.3.2_mesh_scale/reference_unsampled/agg_cmp_nig_product")


def run_one(mode: str, seed: int, root: Path, sample_target: bool, tag: str,
            draws: int = 1, lam: float = DEFAULT_LAM, track_cost: bool = False):
    cfg = load_config()
    cfg.model.lr = LR
    root.mkdir(parents=True, exist_ok=True)

    centres = np.load(ENV / "node_centres.npy")
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    all_grids = np.full((T, G, G), 0.5)
    field = build_dynamic_offset_field(G, T)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=root / tag,
        all_grids=all_grids, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Sampled-target mesh ({mode}, seed={seed}, sampled={sample_target})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=lam, wearable_policy=None, policy_step_size=0.4,
        track_disagreement=(mode == "nig_product"),
        sample_target=sample_target, draws_per_target=draws,
        track_compute_cost=track_cost,
    )
    cms = np.load(root / tag / "cell_mse_steps.npy")
    ccs = np.load(root / tag / "cell_cert_steps.npy")
    whole = float(np.nanmean(cms))
    r_mean, r_sd, n_t, _ = per_timestep_then_averaged(cms, ccs, cms.shape[0] - 50, cms.shape[0])
    print(f"### {tag}: last50={res['mean_mse_last_50']:.5f} whole_run={whole:.5f} "
          f"r={r_mean:+.4f}+/-{r_sd:.4f}", flush=True)
    return whole, res["mean_mse_last_50"], r_mean


def verify():
    if not (REFERENCE / "cell_mse_steps.npy").exists():
        print("reference run %s not present; nothing to verify against" % REFERENCE)
        return True
    """Save-only / flag-off bit-identity check against the stored arm."""
    tag = "verify_nig_product_seed42"
    run_one("nig_product", SEED, VERIFY_OUT, sample_target=False, tag=tag)
    new = np.load(VERIFY_OUT / tag / "cell_mse_steps.npy")
    old = np.load(REFERENCE / "cell_mse_steps.npy")
    same_shape = new.shape == old.shape
    both_nan = np.isnan(new) == np.isnan(old)
    finite = ~np.isnan(old)
    bit_identical = bool(same_shape and both_nan.all()
                         and np.array_equal(new[finite], old[finite]))
    maxdiff = float(np.nanmax(np.abs(new - old))) if same_shape else float("nan")
    print("\n=== BIT-IDENTITY CHECK (cell_mse_steps) ===")
    print(f"  shapes            : {new.shape} vs {old.shape}  match={same_shape}")
    print(f"  NaN masks match   : {bool(both_nan.all())}")
    print(f"  finite values eq  : {bool(np.array_equal(new[finite], old[finite]))}")
    print(f"  max |difference|  : {maxdiff:.3e}")
    print(f"  BIT-IDENTICAL     : {bit_identical}")
    nig = np.load(VERIFY_OUT / tag / "cell_nig_steps.npy")
    print(f"  new cell_nig_steps: shape {nig.shape}, "
          f"finite entries {int(np.isfinite(nig).all(-1).sum())}")
    cert = np.load(VERIFY_OUT / tag / "cell_cert_steps.npy")
    m = np.isfinite(cert)
    nu, al, be = nig[..., 0][m], nig[..., 1][m], nig[..., 2][m]
    recon = 1.0 / (1.0 + np.minimum(be / (np.maximum(nu, 1e-6) * np.maximum(al - 1, 1e-6)), 10.0))
    print(f"  cert reconstructed from saved (nu,alpha,beta): "
          f"max |diff| {float(np.max(np.abs(recon - cert[m]))):.3e}")
    return bit_identical


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--mode", choices=["nig_product", "naive", "certainty",
                                       "fusion", "consensus"])
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--track-cost", action="store_true")
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM)
    ap.add_argument("--draws", type=int, default=1,
                    help="draws per target when sampling (1 = single draw)")
    ap.add_argument("--sampled", type=int, choices=[0, 1], default=1,
                    help="1 = sampled target; 0 = the unsampled 'before' arm, "
                         "re-run only to obtain cell_nig_steps/cell_hop_steps")
    a = ap.parse_args()
    if a.verify:
        ok = verify()
        sys.exit(0 if ok else 1)
    sampled = bool(a.sampled)
    suffix = "sampled" if sampled else "unsampled"
    for s in [int(x) for x in a.seeds.split(",")]:
        dtag = "" if a.draws == 1 else f"_d{a.draws}"
        ltag = "" if a.lam == DEFAULT_LAM else f"_lam{a.lam:g}"
        root = Path(a.root) if a.root else OUT
        run_one(a.mode, s, root, sample_target=sampled,
                tag=f"{a.mode}_{suffix}{dtag}{ltag}_seed{s}", draws=a.draws,
                lam=a.lam, track_cost=a.track_cost)
    print("=== DONE ===")
