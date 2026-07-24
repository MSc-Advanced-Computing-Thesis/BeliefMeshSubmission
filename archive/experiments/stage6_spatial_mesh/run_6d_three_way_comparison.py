# Stage 6D: Three-way comparison -- the central empirical result.
# Experiment Specification Sec 8 & Sec 9 step 3.
#
# NOTE on 6b/6c: the spec's "6b" and "6c" are not separate experiments -- they
# ARE the fusion arm of this same run, on the v2 and v3 environments
# respectively (frozen/naive/fusion/consensus/fedavg all share one Mesh, one
# wearable trajectory, one seed per env -- splitting fusion into its own
# directory would duplicate data and break the paired comparison). So:
#   6b = colour_world/dynamic/v2_fast_drift/fusion/
#   6c = colour_world/dynamic/v3_whiteout/fusion/
#
# Three systems on the dynamic V2 environment with identical wearable path and
# identical fresh pretrained initialisation, differing ONLY in aggregation:
#   frozen -- pretrained baseline, no training ever
#   naive  -- BFS propagation, plain mean of contributor gammas
#   fusion -- BFS propagation, corrected unweighted product-of-experts
# The naive arm is identical to the fusion arm in every respect except the
# aggregation rule, isolating fusion's contribution. Each arm's Mesh uses the
# same sample seed, so hop-0 wearable training draws are identical across arms.
#
# Prior (superseded) result: naive degraded to 0.30-0.35 (= random-guess level,
# 1/3) while the "fusion" arm -- actually certainty-weighted scalar averaging --
# stayed low. This run is the first time the comparison has actually been
# Bayesian. Direction NOT assumed (Spec Sec 8): Stage 4 validates fusion at
# N=2; many-expert behaviour at N~9 is what this measures.
#
# Run from the project root:
#   python -u experiments/stage6_spatial_mesh/run_6d_three_way_comparison.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import git_commit
from stage6_spatial_mesh.runner import run_mesh_experiment

from beliefmesh.config import load_config

BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ARMS = ["frozen", "naive", "fusion", "consensus", "fedavg", "fedavg_global"]
ARM_COLORS = {"frozen": "#999999", "naive": "#d62728", "fusion": "#2ca02c",
              "consensus": "#1f77b4", "fedavg": "#9467bd", "fedavg_global": "#8c564b"}

# Two dynamic regimes with different characters (measured, not assumed):
#   v2: fast, large-excursion red<->blue drift (mean per-step change 0.0175,
#       per-cell excursion 1.33) -- the V2-generation evolution
#   v3: slower drift (0.0056/step) but enters the WHITE-OUT regime
#       (env values up to 1.5: red -> white washout), which v2 never reaches
ENVS = {
    "v2": dict(dir=Path("experiments/stage6_spatial_mesh/environment_v2"),
               run_root=Path("runs/stage6/colour_world/dynamic/v2_fast_drift"),
               label="dynamic_v2 (fast red<->blue drift)"),
    "v3": dict(dir=Path("experiments/stage6_spatial_mesh/environment_v3"),
               run_root=Path("runs/stage6/colour_world/dynamic/v3_whiteout"),
               label="dynamic_v3_whiteout (slower drift, enters red->white regime)"),
}


def main(arms: list[str], env: str, wearables: int = 1):
    cfg = load_config()
    env_cfg = ENVS[env]
    RUN_ROOT = env_cfg["run_root"]
    all_grids = np.load(env_cfg["dir"] / "environment_grids.npy")
    node_centres = np.load(env_cfg["dir"] / "node_centres.npy")
    if wearables == 1:
        # single wearable path (V2 generation), truncated to the env's step count
        path = np.load(Path("experiments/stage6_spatial_mesh/environment_v2/wearable_path.npy"))
        wearable_paths = [path[:len(all_grids)]]
    else:
        # CONFLICT REGIME: three simultaneous wearables (native V3 paths) --
        # anchors in different condition zones at the same timestep. This is
        # the configuration where global one-model averaging should exhibit
        # the classic non-IID weight conflict, and the config the prior
        # repo's 6D actually ran. Reverses Spec Sec 10's multi-anchor descope,
        # deliberately: the tier-1 claim now requires this test.
        assert env == "v3", "3-wearable paths are native to the v3 environment"
        wearable_paths = [np.load(env_cfg["dir"] / f"wearable_path_{w}.npy")[:len(all_grids)]
                          for w in range(3)]
        RUN_ROOT = Path(str(RUN_ROOT) + "_3wearable")

    results = {}
    for arm in arms:
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        results[arm] = run_mesh_experiment(
            cfg, condition=f"6d_{env}_{'3w_' if wearables == 3 else ''}{arm}",
            run_dir=RUN_ROOT / arm,
            all_grids=all_grids, wearable_paths=wearable_paths,
            node_centres=node_centres, fov_size=7, mode=arm,
            baseline_checkpoint=BASELINE_CHECKPOINT,
            n_wearable_samples=1, n_train_repeats=10,
            title=f"Stage 6D ({arm}): dynamic environment",
            extra_manifest={"environment": env_cfg["label"],
                            "comparison": "6D three-way, Spec Sec 8"},
        )

    # ── combined comparison: every arm with saved history under this env ──
    present = [a for a in ARMS if (RUN_ROOT / a / "hop_mse_history.npy").exists()]
    if len(present) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        summary = {}
        for arm in present:
            hist = np.load(RUN_ROOT / arm / "hop_mse_history.npy",
                           allow_pickle=True).item()
            # mean over hops per step (ignoring missing hops)
            steps = len(hist[0])
            mean_series = []
            for t in range(steps):
                vals = [hist[h][t] for h in range(4) if hist[h][t] is not None]
                mean_series.append(float(np.mean(vals)) if vals else np.nan)
            kernel = np.ones(15) / 15
            padded = np.convolve(np.nan_to_num(mean_series, nan=np.nanmean(mean_series)),
                                 kernel, mode="same")
            axes[0].plot(mean_series, color=ARM_COLORS[arm], alpha=0.25, linewidth=0.8)
            axes[0].plot(padded, color=ARM_COLORS[arm], linewidth=2.0, label=arm)
            mse_map = np.load(RUN_ROOT / arm / "avg_mse_map.npy")
            summary[arm] = {"mean_mse_last_50": float(np.nanmean(mse_map)),
                            **{f"hop{h}_final": hist[h][-1] for h in range(4)}}

        axes[0].axhline(1 / 3, color="k", linewidth=0.8, linestyle=":",
                        label="random guessing (1/3)")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Mean MSE across hops")
        axes[0].set_title("Stage 6D: Three Systems in Lockstep (smoothed)")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        arms_order = present
        finals = [summary[a]["mean_mse_last_50"] for a in arms_order]
        axes[1].bar(arms_order, finals, color=[ARM_COLORS[a] for a in arms_order])
        axes[1].axhline(1 / 3, color="k", linewidth=0.8, linestyle=":")
        for i, v in enumerate(finals):
            axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        axes[1].set_ylabel(f"Mean MSE (last 50 steps)")
        axes[1].set_title("Final Comparison")
        axes[1].grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        plt.savefig(RUN_ROOT / "6d_comparison.png", dpi=150)
        print(f"Saved {RUN_ROOT / '6d_comparison.png'}")

        with open(RUN_ROOT / "comparison_summary.yaml", "w") as f:
            yaml.safe_dump({"git_commit": git_commit(), "arms": summary}, f,
                           sort_keys=False)
        print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--env", choices=list(ENVS), default="v2")
    parser.add_argument("--wearables", type=int, choices=[1, 3], default=1)
    args = parser.parse_args()
    main(args.arms, args.env, args.wearables)
