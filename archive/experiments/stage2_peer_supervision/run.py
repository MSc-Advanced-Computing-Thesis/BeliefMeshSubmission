# Stage 2: Peer supervision sufficiency. Experiment Specification Sec 8.
#
# Sequential variant (main result): at each red filter strength, Node A
# fine-tunes on true angles, then Node B trains ONLY on A's predictions
# (A's gamma as a scalar pseudo-label -- Spec Sec 5 mode-matching). Reference:
# B tracks A across all strengths and outperforms the un-adapted baseline
# throughout, with a smoother MSE curve than A's (distillation-style
# regularisation from training on a teacher's smooth predictions).
#
# Simultaneous variant: strength ramps linearly 0 -> 1 over 20 epochs while A
# and B train in the same loop. Reference: B's loss lags A's briefly in early
# epochs (A must first develop an informative signal), then closes.
#
# Run from the project root:
#   python experiments/stage2_peer_supervision/run.py            # both variants
#   python experiments/stage2_peer_supervision/run.py --variant sequential

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
from _common.evaluation import evaluate_at_strength, git_commit

from beliefmesh.config import load_config
from beliefmesh.data.digits import RotatedDigitDataset, get_digit7_splits
from beliefmesh.data.filters import apply_colour_filter
from beliefmesh.node.node import Node

BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
TRAIN_STRENGTHS = [round(s * 0.1, 1) for s in range(11)]
EVAL_STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EPOCHS_PER_STRENGTH = 5   # sequential variant, per node per strength
SIMULTANEOUS_EPOCHS = 20  # simultaneous variant


def make_train_loader(train_indices, batch_size):
    # neutral dataset; the red filter is applied per batch at the desired
    # strength so one dataset serves every strength level (as the old repo did)
    return DataLoader(
        RotatedDigitDataset(filter_type="neutral", subset_indices=train_indices),
        batch_size=batch_size, shuffle=True,
    )


def eval_nodes(nodes: dict, eval_indices, batch_size, device):
    results = {name: [] for name in nodes}
    for strength in EVAL_STRENGTHS:
        for name, node in nodes.items():
            mse = evaluate_at_strength(node.model, eval_indices, "red", strength,
                                       batch_size, device)
            results[name].append(mse)
        line = " | ".join(f"{n}: {results[n][-1]:.4f}" for n in nodes)
        print(f"Eval strength {strength:.1f} | {line}")
    return results


def run_sequential(cfg, train_indices, eval_indices, device, seed):
    print("\n=== Sequential variant ===")
    node_a = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_b = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_frozen = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_frozen.model.eval()

    for strength in TRAIN_STRENGTHS:
        print(f"Strength {strength:.1f} -- A on true labels, then B on A's predictions")
        for _ in range(EPOCHS_PER_STRENGTH):
            for images, targets in make_train_loader(train_indices, cfg.model.batch_size):
                red = apply_colour_filter(images, "red", strength)
                node_a.train_step(red, targets)
        for _ in range(EPOCHS_PER_STRENGTH):
            for images, _ in make_train_loader(train_indices, cfg.model.batch_size):
                red = apply_colour_filter(images, "red", strength)
                a_gamma, _, _, _ = node_a.predict(red)
                node_b.train_step(red, a_gamma)

    results = eval_nodes({"A": node_a, "B": node_b, "frozen": node_frozen},
                         eval_indices, cfg.model.batch_size, device)

    b_beats_frozen = all(b < f for b, f in zip(results["B"], results["frozen"]))
    curve_std_a = float(np.std(results["A"]))
    curve_std_b = float(np.std(results["B"]))
    tracking_gap = float(np.mean([abs(b - a) for a, b in zip(results["A"], results["B"])]))
    print(f"\nB beats frozen at every strength: {b_beats_frozen}")
    print(f"Mean |B - A| tracking gap: {tracking_gap:.4f}")
    print(f"MSE-curve std -- A: {curve_std_a:.4f}, B: {curve_std_b:.4f} "
          f"(reference expects B smoother)")

    run_dir = Path("runs/stage2/sequential")
    if seed != cfg.seed:
        run_dir = run_dir.parent / f"{run_dir.name}_seed{seed}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    torch.save(node_a.model.state_dict(), run_dir / "checkpoints/node_a.pth")
    torch.save(node_b.model.state_dict(), run_dir / "checkpoints/node_b.pth")

    manifest = {
        "stage": "stage2", "condition": "sequential",
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "epochs_per_strength": EPOCHS_PER_STRENGTH,
        "train_strengths": TRAIN_STRENGTHS,
        "git_commit": git_commit(), "config": dataclasses.asdict(cfg),
        "results": {
            "eval_strengths": EVAL_STRENGTHS, **results,
            "b_beats_frozen_everywhere": b_beats_frozen,
            "mean_tracking_gap_b_vs_a": tracking_gap,
            "curve_std_a": curve_std_a, "curve_std_b": curve_std_b,
        },
    }
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(EVAL_STRENGTHS, results["frozen"], marker="o", label="Un-adapted baseline")
    ax.plot(EVAL_STRENGTHS, results["A"], marker="o", label="Node A (true labels)")
    ax.plot(EVAL_STRENGTHS, results["B"], marker="o", label="Node B (peer supervised by A)")
    ax.set_xlabel("Red Filter Strength")
    ax.set_ylabel("Circular MSE (normalised angle)")
    ax.set_title("Stage 2 (sequential): Peer Supervision vs Baselines")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "figures/stage2_sequential.png", dpi=150)
    print(f"Saved {run_dir / 'figures/stage2_sequential.png'}")


def run_simultaneous(cfg, train_indices, eval_indices, device, seed):
    print("\n=== Simultaneous variant ===")
    node_a = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_b = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_frozen = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_frozen.model.eval()

    losses_a, losses_b, strengths_log = [], [], []
    for epoch in range(SIMULTANEOUS_EPOCHS):
        strength = min(epoch / (SIMULTANEOUS_EPOCHS - 1), 1.0)
        ep_a, ep_b, count = 0.0, 0.0, 0
        for images, targets in make_train_loader(train_indices, cfg.model.batch_size):
            red = apply_colour_filter(images, "red", strength)
            ep_a += node_a.train_step(red, targets)
            a_gamma, _, _, _ = node_a.predict(red)
            ep_b += node_b.train_step(red, a_gamma)
            count += 1
        losses_a.append(ep_a / count)
        losses_b.append(ep_b / count)
        strengths_log.append(strength)
        print(f"Epoch {epoch + 1}/{SIMULTANEOUS_EPOCHS} | strength {strength:.2f} | "
              f"loss A {losses_a[-1]:.4f} | loss B {losses_b[-1]:.4f}")

    results = eval_nodes({"A": node_a, "B": node_b, "frozen": node_frozen},
                         eval_indices, cfg.model.batch_size, device)

    early_lag = float(np.mean([b - a for a, b in zip(losses_a[:3], losses_b[:3])]))
    late_lag = float(np.mean([b - a for a, b in zip(losses_a[-3:], losses_b[-3:])]))
    print(f"\nB-A loss gap: first 3 epochs {early_lag:+.4f}, last 3 epochs {late_lag:+.4f} "
          f"(reference expects early lag that closes)")

    run_dir = Path("runs/stage2/simultaneous")
    if seed != cfg.seed:
        run_dir = run_dir.parent / f"{run_dir.name}_seed{seed}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    torch.save(node_a.model.state_dict(), run_dir / "checkpoints/node_a.pth")
    torch.save(node_b.model.state_dict(), run_dir / "checkpoints/node_b.pth")

    manifest = {
        "stage": "stage2", "condition": "simultaneous",
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "epochs": SIMULTANEOUS_EPOCHS,
        "git_commit": git_commit(), "config": dataclasses.asdict(cfg),
        "results": {
            "epoch_losses_a": losses_a, "epoch_losses_b": losses_b,
            "strength_schedule": strengths_log,
            "eval_strengths": EVAL_STRENGTHS, **results,
            "early_loss_lag_b_minus_a": early_lag,
            "late_loss_lag_b_minus_a": late_lag,
        },
    }
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(losses_a, label="Node A (true labels)")
    axes[0].plot(losses_b, label="Node B (peer supervised)")
    axes[0].set_xlabel("Epoch (strength ramps 0 to 1)")
    axes[0].set_ylabel("Mean NIG Loss")
    axes[0].set_title("Simultaneous Training Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(EVAL_STRENGTHS, results["frozen"], marker="o", label="Un-adapted baseline")
    axes[1].plot(EVAL_STRENGTHS, results["A"], marker="o", label="Node A")
    axes[1].plot(EVAL_STRENGTHS, results["B"], marker="o", label="Node B")
    axes[1].set_xlabel("Red Filter Strength")
    axes[1].set_ylabel("Circular MSE (normalised angle)")
    axes[1].set_title("Final Performance vs Filter Strength")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "figures/stage2_simultaneous.png", dpi=150)
    print(f"Saved {run_dir / 'figures/stage2_simultaneous.png'}")


def main(variant: str):
    cfg = load_config()
    seed = int(os.environ.get("EXP_SEED", cfg.seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_indices, eval_indices = get_digit7_splits(
        eval_fraction=cfg.eval_holdout_fraction, seed=cfg.seed)

    if variant in ("both", "sequential"):
        run_sequential(cfg, train_indices, eval_indices, device, seed)
    if variant in ("both", "simultaneous"):
        run_simultaneous(cfg, train_indices, eval_indices, device, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["both", "sequential", "simultaneous"],
                        default="both")
    main(parser.parse_args().variant)
