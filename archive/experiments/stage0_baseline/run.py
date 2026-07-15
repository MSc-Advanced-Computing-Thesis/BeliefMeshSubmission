# Stage 0: Baseline model and calibration. Experiment Specification Sec 8.
#
# Train the EvidentialCNN on unfiltered rotated digit-7 with true angle labels.
# Establishes the pretrained model used by every later stage.
#
# Two gates, not one (Spec Sec 8): held-out circular MSE (reference ~0.0106 in
# normalised angle space, ~18.5 deg mean error) establishes accuracy;
# uncertainty-error Pearson correlation (reference ~0.5649) establishes that
# the head's uncertainty is honest. Both must be consistent with the reference
# before anything downstream is trusted.
#
# Run from the project root: python experiments/stage0_baseline/run.py

from __future__ import annotations

import dataclasses
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
from beliefmesh.metrics.circular import circular_diff, circular_mse
from beliefmesh.models.evidential import nig_loss, predictive_uncertainty
from beliefmesh.models.evidential_cnn import EvidentialCNN

RUN_DIR = Path("runs/stage0/baseline")
LOG_INTERVAL = 100
EVAL_INTERVAL = 500


def evaluate(model, loader, device):
    """Held-out evaluation: mean per-batch circular MSE, plus per-sample
    uncertainties and absolute wrapped errors for the calibration correlation."""
    model.eval()
    batch_mses, uncertainties, errors = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            gamma, nu, alpha, beta = model(images)
            batch_mses.append(circular_mse(gamma, targets).item())
            uncertainties.extend(predictive_uncertainty(nu, alpha, beta).cpu().tolist())
            errors.extend(circular_diff(gamma, targets).abs().cpu().tolist())
    model.train()
    return float(np.mean(batch_mses)), uncertainties, errors


def main(train_seed: int | None = None, run_name: str = "baseline"):
    """train_seed varies weight init / angle draws only, to characterise
    training variance. The evaluation split ALWAYS uses cfg.seed (42) -- that
    is what the spec pins (Sec 2), and it must never move between runs."""
    global RUN_DIR
    RUN_DIR = Path("runs/stage0") / run_name
    cfg = load_config()
    seed = cfg.seed if train_seed is None else train_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_indices, eval_indices = get_digit7_splits(
        eval_fraction=cfg.eval_holdout_fraction, seed=cfg.seed
    )
    print(f"Split: {len(train_indices)} train / {len(eval_indices)} held-out")

    train_loader = DataLoader(
        RotatedDigitDataset(filter_type="neutral", subset_indices=train_indices),
        batch_size=cfg.model.batch_size, shuffle=True,
    )
    eval_loader = DataLoader(
        RotatedDigitDataset(filter_type="neutral", subset_indices=eval_indices),
        batch_size=cfg.model.batch_size, shuffle=False,
    )

    model = EvidentialCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)

    step = 0
    train_losses, eval_mses, eval_steps = [], [], []
    for epoch in range(cfg.model.epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimiser.zero_grad()
            gamma, nu, alpha, beta = model(images)
            loss = nig_loss(gamma, nu, alpha, beta, targets)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())
            step += 1

            if step % LOG_INTERVAL == 0:
                print(f"Epoch {epoch + 1}/{cfg.model.epochs} | Step {step} | Loss {loss.item():.4f}")
            if step % EVAL_INTERVAL == 0:
                mse, _, _ = evaluate(model, eval_loader, device)
                eval_mses.append(mse)
                eval_steps.append(step)
                print(f"  --> Held-out MSE: {mse:.4f}")

    final_mse, final_uncertainties, final_errors = evaluate(model, eval_loader, device)
    correlation = float(np.corrcoef(final_uncertainties, final_errors)[0, 1])
    mean_error_deg = float(np.sqrt(final_mse) * 180.0)

    print(f"\nFinal held-out MSE: {final_mse:.4f} (normalised) ~= {mean_error_deg:.1f} deg")
    print(f"Uncertainty-error Pearson correlation: {correlation:.4f}")
    print("Reference gate (Spec Sec 8): MSE ~0.0106 (~18.5 deg), correlation ~0.5649")

    # outputs: checkpoint, manifest, figures
    checkpoint_dir = RUN_DIR / "checkpoints"
    figures_dir = RUN_DIR / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "pretrained_digit7.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    manifest = {
        "stage": "stage0",
        "condition": run_name,
        "train_seed": seed,
        "git_commit": commit,
        "config": dataclasses.asdict(cfg),
        "results": {
            "final_holdout_mse_normalised": final_mse,
            "mean_error_degrees": mean_error_deg,
            "uncertainty_error_pearson": correlation,
        },
    }
    with open(RUN_DIR / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(train_losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("NIG Loss")
    axes[1].plot(eval_steps, eval_mses)
    axes[1].set_title("Held-out Circular MSE")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MSE (normalised angle)")
    axes[2].scatter(final_uncertainties, final_errors, alpha=0.1, s=5)
    axes[2].set_title(f"Uncertainty vs |Error| (r={correlation:.3f})")
    axes[2].set_xlabel("Predicted Uncertainty")
    axes[2].set_ylabel("Absolute Error (normalised)")
    plt.tight_layout()
    plt.savefig(figures_dir / "stage0_results.png", dpi=150)
    print(f"Saved figures to {figures_dir / 'stage0_results.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seed", type=int, default=None,
                        help="Vary training RNG only; the eval split always stays at config seed 42.")
    parser.add_argument("--run-name", type=str, default="baseline",
                        help="Subdirectory under runs/stage0/ for this run's outputs.")
    args = parser.parse_args()
    main(train_seed=args.train_seed, run_name=args.run_name)
