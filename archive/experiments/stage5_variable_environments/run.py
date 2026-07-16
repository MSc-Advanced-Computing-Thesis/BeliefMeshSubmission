# Stage 5: Variable environments. Experiment Specification Sec 8.
#
# Node A fixed on full red, Node C on full blue, both true labels. Node B's
# input is a pixel-space interpolation between full-red and full-blue images,
# drifting under three regimes across 30 epochs:
#   transition  -- linear red -> blue
#   oscillating -- |sin(pi * progress)|: red -> blue -> red (one full cycle)
#   random      -- uniform resample each epoch
# B trains on either the naive average or the corrected product-of-experts
# fusion of A's and C's predictions on B's blended inputs. No ground truth.
#
# Reference (Spec Sec 8): fusion < naive throughout linear and oscillatory
# regimes, gap widening with mismatch; at the linear midpoint fusion beats
# BOTH anchors individually (complementary partial knowledge); oscillating
# shows slight elevation in the second red period (partial forgetting);
# random still favours fusion.
#
# Run from the project root: python experiments/stage5_variable_environments/run.py

from __future__ import annotations

import dataclasses
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import git_commit

from beliefmesh.config import load_config
from beliefmesh.data.digits import RotatedDigitDataset, get_digit7_splits
from beliefmesh.data.filters import apply_colour_filter
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.fusion.product_of_experts import fuse
from beliefmesh.metrics.circular import circular_mse
from beliefmesh.node.node import Node

BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
NUM_EPOCHS = 30
INTERP_STRENGTHS = np.linspace(0, 1, 11)
MODES = ["transition", "oscillating", "random"]


def b_interp_for(epoch: int, mode: str) -> float:
    progress = epoch / (NUM_EPOCHS - 1)
    if mode == "transition":
        return progress
    if mode == "oscillating":
        # float(): np scalars crash yaml.safe_dump when the schedule is manifested
        return float(abs(np.sin(np.pi * progress)))
    return float(np.random.uniform(0, 1))


def blend(images: torch.Tensor, interp: float) -> torch.Tensor:
    red = apply_colour_filter(images, "red", 1.0)
    blue = apply_colour_filter(images, "blue", 1.0)
    return (1 - interp) * red + interp * blue


def evaluate_interpolated(model, eval_indices, interp, batch_size, device, passes=1):
    loader = DataLoader(
        RotatedDigitDataset(filter_type="neutral", subset_indices=eval_indices),
        batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    pass_mses = []
    with torch.no_grad():
        for _ in range(passes):
            batch_mses = []
            for images, targets in loader:
                blended = blend(images, interp).to(device)
                targets = targets.to(device)
                gamma, *_ = model(blended)
                batch_mses.append(circular_mse(gamma, targets).item())
            pass_mses.append(float(np.mean(batch_mses)))
    if was_training:
        model.train()
    return float(np.mean(pass_mses))


def run_mode(mode, cfg, train_indices, eval_indices, device, grid):
    print(f"\n=== Stage 5: {mode} ===")
    node_a = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_c = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_b_fusion = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_b_naive = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_frozen = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_frozen.model.eval()

    epoch_mse = {"fusion": [], "naive": []}
    interps = []
    for epoch in range(NUM_EPOCHS):
        interp = b_interp_for(epoch, mode)
        interps.append(interp)
        loader = DataLoader(
            RotatedDigitDataset(filter_type="neutral", subset_indices=train_indices),
            batch_size=cfg.model.batch_size, shuffle=True)

        for images, targets in loader:
            red = apply_colour_filter(images, "red", 1.0)
            blue = apply_colour_filter(images, "blue", 1.0)
            node_a.train_step(red, targets)
            node_c.train_step(blue, targets)

            b_images = ((1 - interp) * red + interp * blue).to(device)
            a_belief = node_a.predict(b_images)
            c_belief = node_c.predict(b_images)
            naive = (a_belief[0] + c_belief[0]) / 2
            fused, _ = fuse([a_belief, c_belief], grid)
            node_b_fusion.train_step(b_images, fused)
            node_b_naive.train_step(b_images, naive)

        epoch_mse["fusion"].append(evaluate_interpolated(
            node_b_fusion.model, eval_indices, interp, cfg.model.batch_size, device))
        epoch_mse["naive"].append(evaluate_interpolated(
            node_b_naive.model, eval_indices, interp, cfg.model.batch_size, device))
        print(f"[{mode}] Epoch {epoch + 1}/{NUM_EPOCHS} | interp {interp:.2f} | "
              f"fusion {epoch_mse['fusion'][-1]:.4f} | naive {epoch_mse['naive'][-1]:.4f}")

    # final spectrum evaluation
    models = {"A": node_a.model, "B_fusion": node_b_fusion.model,
              "B_naive": node_b_naive.model, "C": node_c.model,
              "frozen": node_frozen.model}
    spectrum = {name: [] for name in models}
    for interp in INTERP_STRENGTHS:
        for name, model in models.items():
            spectrum[name].append(evaluate_interpolated(
                model, eval_indices, float(interp), cfg.model.batch_size, device, passes=2))

    run_dir = Path("runs/stage5") / mode
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    for name, model in models.items():
        if name != "frozen":
            torch.save(model.state_dict(), run_dir / f"checkpoints/{name.lower()}.pth")

    # reference checks
    fusion_wins = sum(f <= n for f, n in zip(epoch_mse["fusion"], epoch_mse["naive"]))
    checks = {"fusion_wins_epochs": f"{fusion_wins}/{NUM_EPOCHS}"}
    if mode == "transition":
        mid = 5  # interp 0.5 is index 5 of INTERP_STRENGTHS
        checks["midpoint_fusion_beats_both_anchors"] = bool(
            spectrum["B_fusion"][mid] < min(spectrum["A"][mid], spectrum["C"][mid]))
        checks["midpoint_values"] = {
            "fusion": spectrum["B_fusion"][mid], "A": spectrum["A"][mid], "C": spectrum["C"][mid]}
    if mode == "oscillating":
        early_red = float(np.mean(epoch_mse["fusion"][:3]))
        late_red = float(np.mean(epoch_mse["fusion"][-3:]))
        checks["first_red_period_mse"] = early_red
        checks["second_red_period_mse"] = late_red
        checks["second_red_elevated"] = bool(late_red > early_red)
    if mode == "random":
        checks["mean_epoch_mse"] = {"fusion": float(np.mean(epoch_mse["fusion"])),
                                    "naive": float(np.mean(epoch_mse["naive"]))}
    print(f"[{mode}] checks: {checks}")

    manifest = {
        "stage": "stage5", "condition": mode,
        "baseline_checkpoint": str(BASELINE_CHECKPOINT), "epochs": NUM_EPOCHS,
        "git_commit": git_commit(), "config": dataclasses.asdict(cfg),
        "results": {
            "interp_schedule": interps, "epoch_mse": epoch_mse,
            "interp_strengths": [float(x) for x in INTERP_STRENGTHS],
            "final_spectrum_mse": spectrum, "checks": checks,
        },
    }
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(epoch_mse["fusion"], label="B (fusion)")
    axes[0].plot(epoch_mse["naive"], label="B (naive)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE at B's current distribution")
    axes[0].set_title(f"Stage 5 ({mode}): MSE per Epoch")
    axes[0].legend(); axes[0].grid(True)
    ax2 = axes[0].twinx()
    ax2.plot(interps, color="grey", alpha=0.35, linestyle=":")
    ax2.set_ylabel("B interp (0=red, 1=blue)", color="grey")

    for name, style in [("A", "-"), ("B_fusion", "-"), ("B_naive", "-"), ("C", "-"), ("frozen", "--")]:
        axes[1].plot(INTERP_STRENGTHS, spectrum[name], marker="o", linestyle=style, label=name)
    axes[1].set_xlabel("Filter interpolation (0=red, 1=blue)")
    axes[1].set_ylabel("Circular MSE")
    axes[1].set_title(f"Stage 5 ({mode}): Final Spectrum")
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / f"figures/stage5_{mode}.png", dpi=150)
    plt.close(fig)
    print(f"[{mode}] saved {run_dir / f'figures/stage5_{mode}.png'}")


def main(modes: list[str]):
    cfg = load_config()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_idx, eval_idx = get_digit7_splits(eval_fraction=cfg.eval_holdout_fraction, seed=cfg.seed)
    grid = circular_grid(cfg.fusion.grid_size, device=device)
    for mode in modes:
        run_mode(mode, cfg, train_idx, eval_idx, device, grid)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    main(parser.parse_args().modes)
