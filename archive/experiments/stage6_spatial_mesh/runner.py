"""Central Stage 6 experiment loop. Ported from the prior repo's
experiment_runner.py, driving the corrected beliefmesh Mesh instead of the
defective SpatialNodeLearningSystem.

Records per-step: mean MSE per hop distance; per-cell MSE and certainty
(display-only certainty, Spec Sec 8 evaluation); average maps over the last
LAST_N_STEPS; per-cell certainty-vs-MSE relation (the cone, Spec Sec 8).
"""

from __future__ import annotations

import dataclasses
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import git_commit

from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.node.mesh import Mesh

WEARABLE_COLORS = ["#00FF00", "#00CC44", "#006622"]
HOP_COLORS = ["#FFD700", "#FFA500", "#FF4500", "#8B0000"]
LAST_N_STEPS = 50
TRAIL_LEN = 20


def run_mesh_experiment(
    cfg,
    condition: str,
    run_dir: Path,
    all_grids: np.ndarray,
    wearable_paths: list[np.ndarray],
    node_centres: np.ndarray,
    fov_size: int,
    mode: str,
    baseline_checkpoint: Path,
    n_wearable_samples: int = 1,
    n_train_repeats: int = 10,
    title: str = "",
    device: torch.device | None = None,
    extra_manifest: dict | None = None,
    offset_field: np.ndarray | None = None,
    env_seed: int = 42,
    wearable_policy: str | None = None,
    policy_step_size: float = 0.4,
    policy_cooldown: int = 0,
    lam: float = 0.1,
    rho: float = 0.2,
    temper_gradient: bool = True,
    excluded_rotation_ranges: list[tuple[float, float]] | None = None,
):
    """wearable_policy=None replays the given wearable_paths. 'epistemic'
    computes the single wearable's trajectory ONLINE: each step it moves
    (at policy_step_size cells/step -- the mobility capacity) toward the cell
    with the highest current epistemic uncertainty. wearable_paths[0][0] seeds
    the start position; the realised trajectory is saved for videos/analysis."""
    grid_size = all_grids.shape[1]
    total_steps = all_grids.shape[0]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = GridEnvironment(grid_size, all_grids, offset_field=offset_field,
                          rotation_seed=env_seed,
                          excluded_rotation_ranges=excluded_rotation_ranges)
    mesh = Mesh(node_centres, fov_size=fov_size, grid_size=grid_size,
                environment=env, pretrained_path=baseline_checkpoint,
                lr=cfg.model.lr, fusion_grid=circular_grid(cfg.fusion.grid_size),
                mode=mode, device=device, sample_seed=env_seed, rho=rho, lam=lam,
                temper_gradient=temper_gradient)

    coverage = np.zeros((grid_size, grid_size), dtype=int)
    for node in mesh.nodes.values():
        for r, c in node.fov_cells:
            coverage[r, c] += 1
    print(f"[{condition}] nodes={len(mesh.nodes)} coverage max={coverage.max()} "
          f"mode={mode} steps={total_steps}")

    hop_mse_history = {h: [] for h in range(4)}
    cell_mse_hist = defaultdict(list)
    cell_cert_hist = defaultdict(list)
    # per-step spatial arrays: consumed by generate_video.py
    cell_mse_steps = np.full((total_steps, grid_size, grid_size), np.nan)
    cell_cert_steps = np.full((total_steps, grid_size, grid_size), np.nan)
    n_wearables = len(wearable_paths)

    policy_rng = np.random.default_rng(env_seed)
    policy_positions = [np.array(wearable_paths[w][0], dtype=float) for w in range(n_wearables)]
    realised_paths = [[] for _ in range(n_wearables)]
    last_visit = np.full((grid_size, grid_size), -50.0)  # uniform initial staleness

    def _norm01(x):
        lo, hi = np.nanmin(x), np.nanmax(x)
        return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

    for step in range(total_steps):
        if wearable_policy in ("epistemic", "staleness", "epistemic_staleness"):
            emap = None
            if wearable_policy != "staleness":
                e = mesh.epistemic_map()
                if np.isfinite(e).any():
                    emap = np.where(np.isfinite(e), e, np.nanmax(e))
            positions = []
            # sequential per-wearable greedy assignment: each wearable sees
            # last_visit as updated by the ones before it this step, so they
            # naturally spread out instead of all chasing the same argmax cell
            for w in range(n_wearables):
                staleness = step - last_visit
                if wearable_policy == "staleness":
                    score = staleness
                elif emap is not None:
                    score = (_norm01(emap) if wearable_policy == "epistemic"
                             else _norm01(emap) + _norm01(staleness))
                else:
                    score = None
                if score is not None:
                    if policy_cooldown > 0:
                        # hard anti-camping: a persistently-high local
                        # epistemic score (e.g. a turbulent wake) can keep
                        # outscoring staleness for nearby cells forever,
                        # trapping a wearable in one region even though
                        # staleness is in the sum -- exclude recently-visited
                        # cells from the argmax outright so the policy is
                        # forced to keep moving on.
                        eligible = staleness >= policy_cooldown
                        if eligible.any():
                            score = np.where(eligible, score, -np.inf)
                    target = np.unravel_index(np.nanargmax(score), score.shape)
                    direction = np.array(target, dtype=float) - policy_positions[w]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        policy_positions[w] = policy_positions[w] + policy_step_size * direction / norm
                else:  # no signal yet: small random walk from start
                    policy_positions[w] = policy_positions[w] + policy_rng.normal(0, policy_step_size, 2)
                policy_positions[w] = np.clip(policy_positions[w], 0.1, grid_size - 1.1)
                r0, c0 = int(policy_positions[w][0]), int(policy_positions[w][1])
                last_visit[max(0, r0-1):r0+2, max(0, c0-1):c0+2] = step
                positions.append(policy_positions[w].copy())
                realised_paths[w].append(policy_positions[w].copy())
        else:
            positions = [wearable_paths[w][step] for w in range(n_wearables)]
        trained = mesh.run_timestep(positions, step,
                                    n_wearable_samples=n_wearable_samples,
                                    n_train_repeats=n_train_repeats)

        # best-certainty belief per cell across nodes, for the spatial maps
        step_best = {}
        for node in mesh.nodes.values():
            for cell, (g, n, a, b) in node.cell_beliefs.items():
                cert = 1.0 / (1.0 + min(b / (max(n, 1e-6) * max(a - 1, 1e-6)), 10.0))
                if cell not in step_best or cert > step_best[cell][1]:
                    step_best[cell] = (g, cert)
        for cell, (pred, cert) in step_best.items():
            _, truth = env.get_cell_input(cell[0], cell[1], step)
            err = circular_diff(torch.tensor(pred), truth).item() ** 2
            cell_mse_hist[cell].append(err)
            cell_cert_hist[cell].append(cert)
            cell_mse_steps[step, cell[0], cell[1]] = err
            cell_cert_steps[step, cell[0], cell[1]] = cert

        metrics = mesh.evaluate(step)
        hop_mses = {h: [] for h in range(4)}
        for m in metrics.values():
            if m["mse"] is not None and m["hop_distance"] is not None:
                hop_mses[min(m["hop_distance"], 3)].append(m["mse"])
        for h in range(4):
            hop_mse_history[h].append(float(np.mean(hop_mses[h])) if hop_mses[h] else None)

        if step % 10 == 0:
            hop_str = " | ".join(f"hop{h}: {np.mean(hop_mses[h]):.4f}"
                                 for h in range(4) if hop_mses[h])
            print(f"[{condition}] step {step:03d}/{total_steps} "
                  f"trained {len(trained)}/{len(mesh.nodes)} | {hop_str}")

    # ── aggregate maps ────────────────────────────────────────────────────
    avg_mse_map = np.full((grid_size, grid_size), np.nan)
    avg_cert_map = np.full((grid_size, grid_size), np.nan)
    for cell, hist in cell_mse_hist.items():
        avg_mse_map[cell] = np.mean(hist[-LAST_N_STEPS:])
    for cell, hist in cell_cert_hist.items():
        avg_cert_map[cell] = np.mean(hist[-LAST_N_STEPS:])

    pts = [(avg_cert_map[c], avg_mse_map[c]) for c in cell_mse_hist
           if np.isfinite(avg_mse_map[c]) and np.isfinite(avg_cert_map[c])]
    r_val = p_val = float("nan")
    if pts:
        certs_arr, mses_arr = zip(*pts)
        r_val, p_val = pearsonr(certs_arr, mses_arr)

    overall_mean_last = float(np.nanmean(avg_mse_map))
    print(f"[{condition}] mean MSE (last {LAST_N_STEPS} steps): {overall_mean_last:.4f} "
          f"| cert-MSE r={r_val:.3f}")

    # ── save ──────────────────────────────────────────────────────────────
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "hop_mse_history.npy", hop_mse_history, allow_pickle=True)
    np.save(run_dir / "avg_mse_map.npy", avg_mse_map)
    np.save(run_dir / "avg_cert_map.npy", avg_cert_map)
    np.save(run_dir / "cell_mse_steps.npy", cell_mse_steps)
    np.save(run_dir / "cell_cert_steps.npy", cell_cert_steps)
    if realised_paths[0]:
        wearable_paths = [np.array(p) for p in realised_paths]  # so figures show the real trail(s)
    # always save whatever paths were actually used (policy-driven or a
    # static replay) so generate_video.py doesn't fall back to the single
    # default environment path for multi-wearable static runs
    used_arrs = [np.asarray(wp)[:total_steps] for wp in wearable_paths]
    np.save(run_dir / "realised_wearable_path.npy", used_arrs[0])  # back-compat, single
    if n_wearables > 1:
        np.save(run_dir / "realised_wearable_paths.npy", np.stack(used_arrs))

    manifest = {
        "stage": "stage6", "condition": condition, "mode": mode,
        "baseline_checkpoint": str(baseline_checkpoint),
        "model_note": "plain EvidentialCNN (deliberate deviation from prior repo's coordinate-conditioned SpatialEvidentialCNN)",
        "fov_size": fov_size, "n_nodes": len(mesh.nodes),
        "max_coverage": int(coverage.max()),
        "n_wearables": n_wearables, "n_wearable_samples": n_wearable_samples,
        "n_train_repeats": n_train_repeats, "total_steps": total_steps,
        "env_seed": env_seed, "wearable_policy": wearable_policy or "replay",
        "policy_cooldown": policy_cooldown,
        "git_commit": git_commit(), "config": dataclasses.asdict(cfg),
        **(extra_manifest or {}),
        "results": {
            "mean_mse_last_50": overall_mean_last,
            "certainty_mse_pearson_r": float(r_val),
            "certainty_mse_pearson_p": float(p_val),
            "final_hop_means": {h: hop_mse_history[h][-1] for h in range(4)},
        },
    }
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    # ── figure ────────────────────────────────────────────────────────────
    trail_end = total_steps - 1
    trail_start = max(0, trail_end - TRAIL_LEN + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title or condition, fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for h in range(4):
        vals = [(i, v) for i, v in enumerate(hop_mse_history[h]) if v is not None]
        if vals:
            ax.plot(*zip(*vals), label=f"Hop {h}", color=HOP_COLORS[h], linewidth=1.2)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Mean MSE")
    ax.set_title("MSE by Hop Distance Over Time")
    ax.legend(); ax.grid(True, alpha=0.3)

    for ax, data, cmap, label in [
        (axes[0, 1], avg_mse_map, "RdYlGn_r", f"Avg MSE (last {LAST_N_STEPS})"),
        (axes[1, 0], avg_cert_map, "RdYlGn", f"Avg Certainty (last {LAST_N_STEPS})"),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, label=label)
        for w in range(n_wearables):
            trail = wearable_paths[w][trail_start:trail_end + 1]
            ax.plot(trail[:, 1], trail[:, 0], color=WEARABLE_COLORS[w % len(WEARABLE_COLORS)],
                    linewidth=1.5, alpha=0.85, zorder=5)
            ax.scatter(wearable_paths[w][trail_end, 1], wearable_paths[w][trail_end, 0],
                       c=WEARABLE_COLORS[w % len(WEARABLE_COLORS)], s=60, marker="s", zorder=6,
                       edgecolors="white", linewidths=0.5)
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]
    if pts:
        ax.scatter(certs_arr, mses_arr, alpha=0.5, s=20, c="steelblue", edgecolors="none")
        ax.set_xlabel("Mean Certainty"); ax.set_ylabel("Mean MSE")
        ax.set_title(f"Per-cell Certainty vs MSE (r = {r_val:.3f}, p = {p_val:.2g})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir / "figures/results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{condition}] saved {run_dir / 'figures/results.png'}")

    return manifest["results"]
