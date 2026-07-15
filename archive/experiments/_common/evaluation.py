"""Shared evaluation helpers for stage run scripts.

All stages evaluate on the SAME seed-42 held-out split (Spec Sec 2) -- these
helpers exist so no stage script reimplements that protocol slightly
differently.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from beliefmesh.data.digits import RotatedDigitDataset
from beliefmesh.metrics.circular import circular_mse


def evaluate_at_strength(
    model: torch.nn.Module,
    eval_indices: np.ndarray,
    filter_type: str,
    strength: float,
    batch_size: int,
    device: torch.device,
    passes: int = 2,
) -> float:
    """Mean circular MSE of `model` on the held-out split under the given
    filter, averaged over `passes` passes (rotation angles re-roll per pass)."""
    loader = DataLoader(
        RotatedDigitDataset(filter_type=filter_type, filter_strength=strength,
                            subset_indices=eval_indices),
        batch_size=batch_size, shuffle=False,
    )
    was_training = model.training
    model.eval()
    pass_mses = []
    with torch.no_grad():
        for _ in range(passes):
            batch_mses = []
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                gamma, _, _, _ = model(images)
                batch_mses.append(circular_mse(gamma, targets).item())
            pass_mses.append(float(np.mean(batch_mses)))
    if was_training:
        model.train()
    return float(np.mean(pass_mses))


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
