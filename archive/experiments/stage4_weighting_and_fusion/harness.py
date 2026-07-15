"""Shared harness for Stage 4's three sub-experiments. Spec Sec 8, Stage 4.

Common protocol (ported from the prior repo's stage_4a/4b/4c, which shared it
verbatim): red filter strength ramps 0 -> 1 over 20 epochs; Node A trains on
red with true labels, Node C on blue with true labels; each B-variant trains
on its aggregation of A's and C's full beliefs evaluated on B's red inputs.
Per-epoch MSE tracked at the current strength; final sweep across strengths.

Each variant is a target function taking the two anchors' FULL NIG tuples --
scalar strategies collapse them internally (that collapse is precisely their
defining limitation, Spec Sec 8 Stage 4 purpose), fusion consumes them whole.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import evaluate_at_strength, git_commit

from beliefmesh.data.digits import RotatedDigitDataset
from beliefmesh.data.filters import apply_colour_filter
from beliefmesh.node.node import Node

BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
NUM_EPOCHS = 20
EVAL_STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

Belief = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TargetFn = Callable[[Belief, Belief], torch.Tensor]


def run_ramp_experiment(
    cfg,
    condition: str,
    variants: dict[str, TargetFn],
    run_dir: Path,
    train_indices,
    eval_indices,
    device: torch.device,
    extra_manifest: dict | None = None,
):
    node_a = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_c = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_frozen = Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
    node_frozen.model.eval()
    b_nodes = {name: Node.from_checkpoint(BASELINE_CHECKPOINT, lr=cfg.model.lr, device=device)
               for name in variants}

    epoch_mse = {name: [] for name in variants}
    epoch_mse["A"] = []
    strengths_log = []

    for epoch in range(NUM_EPOCHS):
        strength = min(epoch / (NUM_EPOCHS - 1), 1.0)
        strengths_log.append(strength)
        loader = DataLoader(
            RotatedDigitDataset(filter_type="neutral", subset_indices=train_indices),
            batch_size=cfg.model.batch_size, shuffle=True)

        for images, targets in loader:
            red = apply_colour_filter(images, "red", strength)
            blue = apply_colour_filter(images, "blue", strength)
            node_a.train_step(red, targets)
            node_c.train_step(blue, targets)

            a_belief = node_a.predict(red)   # full tuples, uncollapsed
            c_belief = node_c.predict(red)
            for name, target_fn in variants.items():
                pseudo = target_fn(a_belief, c_belief)
                b_nodes[name].train_step(red, pseudo)

        for name in variants:
            epoch_mse[name].append(evaluate_at_strength(
                b_nodes[name].model, eval_indices, "red", strength,
                cfg.model.batch_size, device, passes=1))
        epoch_mse["A"].append(evaluate_at_strength(
            node_a.model, eval_indices, "red", strength,
            cfg.model.batch_size, device, passes=1))
        print(f"[{condition}] Epoch {epoch + 1}/{NUM_EPOCHS} | strength {strength:.2f} | " +
              " | ".join(f"{n}: {epoch_mse[n][-1]:.4f}" for n in epoch_mse))

    print(f"\n[{condition}] Final evaluation across filter strengths...")
    final_mse = {name: [] for name in list(variants) + ["A", "frozen"]}
    eval_models = {**{n: b.model for n, b in b_nodes.items()},
                   "A": node_a.model, "frozen": node_frozen.model}
    for strength in EVAL_STRENGTHS:
        for name, model in eval_models.items():
            final_mse[name].append(evaluate_at_strength(
                model, eval_indices, "red", strength,
                cfg.model.batch_size, device, passes=2))
        print(f"[{condition}] Strength {strength:.1f} | " +
              " | ".join(f"{n}: {final_mse[n][-1]:.4f}" for n in final_mse))

    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    for name, node in b_nodes.items():
        torch.save(node.model.state_dict(), run_dir / f"checkpoints/node_b_{name}.pth")
    torch.save(node_a.model.state_dict(), run_dir / "checkpoints/node_a.pth")
    torch.save(node_c.model.state_dict(), run_dir / "checkpoints/node_c.pth")

    # mean MSE over the high-strength half, where the asymmetry lives
    high = [i for i, s in enumerate(EVAL_STRENGTHS) if s >= 0.6]
    high_strength_means = {
        name: float(np.mean([final_mse[name][i] for i in high])) for name in final_mse}

    manifest = {
        "stage": "stage4", "condition": condition,
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "epochs": NUM_EPOCHS,
        "lr_note": "V2 uniform lr=3e-4; prior repo's stage_4 scripts ran at lr=1e-3 (Node() default)",
        "git_commit": git_commit(), "config": dataclasses.asdict(cfg),
        **(extra_manifest or {}),
        "results": {
            "strength_schedule": strengths_log,
            "epoch_mse": epoch_mse,
            "eval_strengths": EVAL_STRENGTHS,
            "final_mse": final_mse,
            "high_strength_mean_mse": high_strength_means,
        },
    }
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for name in variants:
        axes[0].plot(epoch_mse[name], label=name)
    axes[0].plot(epoch_mse["A"], label="A (true labels)", linestyle="--", color="k")
    axes[0].set_xlabel("Epoch (strength ramps 0 to 1)")
    axes[0].set_ylabel("Circular MSE @ current strength")
    axes[0].set_title(f"Stage 4 ({condition}): MSE During Training")
    axes[0].legend(); axes[0].grid(True)
    for name in variants:
        axes[1].plot(EVAL_STRENGTHS, final_mse[name], marker="o", label=name)
    axes[1].plot(EVAL_STRENGTHS, final_mse["A"], marker="o", linestyle="--", color="k",
                 label="A (true labels)")
    axes[1].plot(EVAL_STRENGTHS, final_mse["frozen"], marker="o", linestyle=":",
                 color="grey", label="un-adapted")
    axes[1].set_xlabel("Red Filter Strength")
    axes[1].set_ylabel("Circular MSE")
    axes[1].set_title(f"Stage 4 ({condition}): Final Performance")
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    fig_path = run_dir / f"figures/stage4_{condition}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[{condition}] Saved {fig_path}")

    return final_mse, high_strength_means
