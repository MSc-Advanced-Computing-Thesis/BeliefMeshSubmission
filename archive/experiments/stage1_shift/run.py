# Stage 1: Degradation under distributional shift. Experiment Specification Sec 8.
#
# Evaluate the Stage 0 pretrained model, with no further training, across
# increasing red filter strength. Reference result: MSE degrades monotonically
# with filter strength. Purpose: establishes that adaptation under dynamic
# conditions is necessary rather than optional.
#
# Run from the project root: python experiments/stage1_shift/run.py

from __future__ import annotations

import dataclasses
import os
import random
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from beliefmesh.config import load_config
from beliefmesh.data.digits import RotatedDigitDataset, get_digit7_splits
from beliefmesh.metrics.circular import circular_mse
from beliefmesh.models.evidential import predictive_uncertainty
from beliefmesh.models.evidential_cnn import EvidentialCNN

RUN_DIR = Path("runs/stage1/red_degradation")
BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
FILTER_STRENGTHS = [round(s * 0.1, 1) for s in range(11)]  # 0.0 .. 1.0
EVAL_PASSES = 3  # angles are re-rolled per pass; averaging tightens each point


def evaluate_at_strength(model, eval_indices, strength, batch_size, device):
    loader = DataLoader(
        RotatedDigitDataset(filter_type="red", filter_strength=strength,
                            subset_indices=eval_indices),
        batch_size=batch_size, shuffle=False,
    )
    pass_mses, uncertainties = [], []
    with torch.no_grad():
        for _ in range(EVAL_PASSES):
            batch_mses = []
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                gamma, nu, alpha, beta = model(images)
                batch_mses.append(circular_mse(gamma, targets).item())
                uncertainties.extend(predictive_uncertainty(nu, alpha, beta).cpu().tolist())
            pass_mses.append(float(np.mean(batch_mses)))
    return float(np.mean(pass_mses)), float(np.std(pass_mses)), float(np.mean(uncertainties))


def main():
    cfg = load_config()
    seed = int(os.environ.get("EXP_SEED", cfg.seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_dir = RUN_DIR
    if seed != cfg.seed:
        run_dir = run_dir.parent / f"{run_dir.name}_seed{seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_indices = get_digit7_splits(eval_fraction=cfg.eval_holdout_fraction, seed=cfg.seed)

    model = EvidentialCNN().to(device)
    model.load_state_dict(torch.load(BASELINE_CHECKPOINT, map_location=device))
    model.eval()
    print(f"Loaded baseline checkpoint {BASELINE_CHECKPOINT}")

    mse_means, mse_stds, unc_means = [], [], []
    for strength in FILTER_STRENGTHS:
        mse, std, unc = evaluate_at_strength(model, eval_indices, strength,
                                             cfg.model.batch_size, device)
        mse_means.append(mse)
        mse_stds.append(std)
        unc_means.append(unc)
        print(f"Strength {strength:.1f} | MSE {mse:.4f} +/- {std:.4f} | mean uncertainty {unc:.4f}")

    increases = [b - a for a, b in zip(mse_means, mse_means[1:])]
    monotonic = all(d > 0 for d in increases)
    print(f"\nMonotonic MSE increase across all steps: {monotonic}")
    if not monotonic:
        flat = [(FILTER_STRENGTHS[i], d) for i, d in enumerate(increases) if d <= 0]
        print(f"Non-increasing steps (strength, delta): {flat}")
    print(f"Degradation: {mse_means[0]:.4f} (clean) -> {mse_means[-1]:.4f} (full red), "
          f"{mse_means[-1] / mse_means[0]:.1f}x")

    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    manifest = {
        "stage": "stage1",
        "condition": "red_degradation",
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "eval_passes": EVAL_PASSES,
        "git_commit": commit,
        "config": dataclasses.asdict(cfg),
        "results": {
            "filter_strengths": FILTER_STRENGTHS,
            "mse_means": mse_means,
            "mse_stds": mse_stds,
            "mean_uncertainties": unc_means,
            "monotonic_increase": monotonic,
        },
    }
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(FILTER_STRENGTHS, mse_means, yerr=mse_stds, marker="o", capsize=3)
    axes[0].set_xlabel("Red Filter Strength")
    axes[0].set_ylabel("Circular MSE (normalised angle)")
    axes[0].set_title("Pretrained Model MSE vs Filter Strength")
    axes[0].grid(True)
    axes[1].plot(FILTER_STRENGTHS, unc_means, marker="o", color="tab:orange")
    axes[1].set_xlabel("Red Filter Strength")
    axes[1].set_ylabel("Mean Predictive Uncertainty")
    axes[1].set_title("Uncertainty vs Filter Strength")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "stage1_results.png", dpi=150)
    print(f"Saved figure to {figures_dir / 'stage1_results.png'}")


if __name__ == "__main__":
    main()
