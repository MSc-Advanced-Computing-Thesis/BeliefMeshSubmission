# Stage 3: Naive aggregation baseline. Experiment Specification Sec 8.
#
# Node A fine-tunes on red, Node C on blue, both against true angles, with the
# filter strength ramping 0 -> 1 over 20 epochs. Node B trains exclusively on
# the naive average of A's and C's predictions on red-filtered inputs, no
# ground truth. B is evaluated on red, placing A in-distribution and C
# out-of-distribution.
#
# Reference (Spec Sec 8): C maintains elevated uncertainty (~0.035-0.038 old
# checkpoint) across strengths; B sits between A and C in MSE, as naive
# averaging predicts. Purpose: establishes the cost of averaging anchors of
# unequal reliability, and that the evidential head detects distributional
# mismatch without explicit OOD signalling.
#
# NOTE: the prior repo's stage_3.py silently ran at lr=1e-3 (Node() default);
# V2 uses the uniform config lr=3e-4. Absolute values may shift accordingly.
#
# Run from the project root: python experiments/stage3_naive_aggregation/run.py

from __future__ import annotations

import dataclasses
import os
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
from _common.evaluation import evaluate_with_uncertainty, git_commit

from beliefmesh.config import load_config
from beliefmesh.data.digits import RotatedDigitDataset, get_digit7_splits
from beliefmesh.data.filters import apply_colour_filter
from beliefmesh.node.node import Node

RUN_DIR = Path("runs/stage3/naive_average")
BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
NUM_EPOCHS = 20
EVAL_STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    cfg = load_config()
    seed = int(os.environ.get("EXP_SEED", cfg.seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = RUN_DIR
    if seed != cfg.seed:
        run_dir = run_dir.parent / f"{run_dir.name}_seed{seed}"

    train_indices, eval_indices = get_digit7_splits(
        eval_fraction=cfg.eval_holdout_fraction, seed=cfg.seed)

    nodes = {
        "A": Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device),
        "C": Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device),
        "B": Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device),
        "frozen": Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device),
    }
    nodes["frozen"].model.eval()

    def eval_all(strength, passes):
        out = {}
        for name, node in nodes.items():
            out[name] = evaluate_with_uncertainty(
                node.model, eval_indices, "red", strength,
                cfg.model.batch_size, device, passes=passes)
        return out

    epoch_losses = {"A": [], "C": [], "B": []}
    epoch_mse = {"A": [], "C": [], "B": [], "frozen": []}
    epoch_unc = {"A": [], "C": [], "B": []}
    strengths_log = []

    for epoch in range(NUM_EPOCHS):
        strength = min(epoch / (NUM_EPOCHS - 1), 1.0)
        strengths_log.append(strength)
        loader = DataLoader(
            RotatedDigitDataset(filter_type="neutral", subset_indices=train_indices),
            batch_size=cfg.model.batch_size, shuffle=True)

        sums = {"A": 0.0, "C": 0.0, "B": 0.0}
        count = 0
        for images, targets in loader:
            red = apply_colour_filter(images, "red", strength)
            blue = apply_colour_filter(images, "blue", strength)

            sums["A"] += nodes["A"].train_step(red, targets)
            sums["C"] += nodes["C"].train_step(blue, targets)

            # both anchors asked to predict on B's (red) inputs; C is OOD here
            a_gamma, *_ = nodes["A"].predict(red)
            c_gamma, *_ = nodes["C"].predict(red)
            naive_avg = (a_gamma + c_gamma) / 2
            sums["B"] += nodes["B"].train_step(red, naive_avg)
            count += 1

        for k in sums:
            epoch_losses[k].append(sums[k] / count)

        results = eval_all(strength, passes=1)
        for k in ("A", "C", "B", "frozen"):
            epoch_mse[k].append(results[k][0])
        for k in ("A", "C", "B"):
            epoch_unc[k].append(results[k][1])

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | strength {strength:.2f} | "
              f"MSE A {results['A'][0]:.4f} B {results['B'][0]:.4f} C {results['C'][0]:.4f} | "
              f"Unc A {results['A'][1]:.4f} B {results['B'][1]:.4f} C {results['C'][1]:.4f}")

    print("\nFinal evaluation across filter strengths...")
    final_mse = {k: [] for k in nodes}
    final_unc = {k: [] for k in nodes}
    for strength in EVAL_STRENGTHS:
        results = eval_all(strength, passes=2)
        for k in nodes:
            final_mse[k].append(results[k][0])
            final_unc[k].append(results[k][1])
        print(f"Strength {strength:.1f} | " + " | ".join(
            f"{k}: {final_mse[k][-1]:.4f} (unc {final_unc[k][-1]:.4f})" for k in nodes))

    # reference checks
    high = [s >= 0.4 for s in EVAL_STRENGTHS]
    b_between = all(
        min(a, c) <= b <= max(a, c)
        for a, b, c, h in zip(final_mse["A"], final_mse["B"], final_mse["C"], high) if h)
    c_unc_elevated = all(
        uc > ua for ua, uc in zip(final_unc["A"][2:], final_unc["C"][2:]))
    print(f"\nB between A and C (strengths >= 0.4): {b_between}")
    print(f"C uncertainty > A uncertainty (strengths >= 0.4): {c_unc_elevated}")
    print(f"C uncertainty range across strengths: "
          f"{min(final_unc['C']):.4f} - {max(final_unc['C']):.4f} (old ref 0.035-0.038)")

    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    for k in ("A", "B", "C"):
        torch.save(nodes[k].model.state_dict(), run_dir / f"checkpoints/node_{k.lower()}.pth")

    manifest = {
        "stage": "stage3", "condition": "naive_average",
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "epochs": NUM_EPOCHS,
        "lr_note": "V2 uniform lr=3e-4; prior repo's stage_3 ran at lr=1e-3 (Node() default)",
        "git_commit": git_commit(), "config": dataclasses.asdict(cfg),
        "results": {
            "strength_schedule": strengths_log,
            "epoch_losses": epoch_losses, "epoch_mse": epoch_mse, "epoch_uncertainty": epoch_unc,
            "eval_strengths": EVAL_STRENGTHS,
            "final_mse": final_mse, "final_uncertainty": final_unc,
            "b_between_a_and_c_high_strengths": b_between,
            "c_uncertainty_elevated_high_strengths": c_unc_elevated,
        },
    }
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0][0].plot(epoch_losses["A"], label="Node A (red, true labels)")
    axes[0][0].plot(epoch_losses["C"], label="Node C (blue, true labels)")
    axes[0][0].plot(epoch_losses["B"], label="Node B (naive avg of A+C)")
    axes[0][0].set_xlabel("Epoch (strength ramps 0 to 1)")
    axes[0][0].set_ylabel("Mean NIG Loss")
    axes[0][0].set_title("Training Loss")
    axes[0][0].legend(); axes[0][0].grid(True)

    axes[0][1].plot(epoch_mse["A"], label="A")
    axes[0][1].plot(epoch_mse["C"], label="C")
    axes[0][1].plot(epoch_mse["B"], label="B (naive avg)")
    axes[0][1].plot(epoch_mse["frozen"], label="Un-adapted")
    axes[0][1].set_xlabel("Epoch")
    axes[0][1].set_ylabel("Circular MSE (red @ current strength)")
    axes[0][1].set_title("MSE During Training")
    axes[0][1].legend(); axes[0][1].grid(True)

    axes[1][0].plot(EVAL_STRENGTHS, final_mse["A"], marker="o", label="A (red anchor)")
    axes[1][0].plot(EVAL_STRENGTHS, final_mse["B"], marker="o", label="B (naive avg)")
    axes[1][0].plot(EVAL_STRENGTHS, final_mse["C"], marker="o", label="C (blue anchor, OOD)")
    axes[1][0].plot(EVAL_STRENGTHS, final_mse["frozen"], marker="o", label="Un-adapted")
    axes[1][0].set_xlabel("Red Filter Strength")
    axes[1][0].set_ylabel("Circular MSE")
    axes[1][0].set_title("Final MSE vs Filter Strength (eval on red)")
    axes[1][0].legend(); axes[1][0].grid(True)

    axes[1][1].plot(EVAL_STRENGTHS, final_unc["A"], marker="o", label="A")
    axes[1][1].plot(EVAL_STRENGTHS, final_unc["B"], marker="o", label="B")
    axes[1][1].plot(EVAL_STRENGTHS, final_unc["C"], marker="o", label="C (expect elevated)")
    axes[1][1].plot(EVAL_STRENGTHS, final_unc["frozen"], marker="o", label="Un-adapted")
    axes[1][1].set_xlabel("Red Filter Strength")
    axes[1][1].set_ylabel("Mean Predictive Uncertainty")
    axes[1][1].set_title("Uncertainty vs Filter Strength (eval on red)")
    axes[1][1].legend(); axes[1][1].grid(True)

    plt.tight_layout()
    plt.savefig(run_dir / "figures/stage3_results.png", dpi=150)
    print(f"Saved {run_dir / 'figures/stage3_results.png'}")


if __name__ == "__main__":
    main()
