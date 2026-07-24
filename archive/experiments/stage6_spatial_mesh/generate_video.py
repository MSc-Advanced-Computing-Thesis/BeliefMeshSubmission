"""Render the Stage 6 experiment animation for any completed mesh run.

Five-panel dashboard per timestep (ported from the prior repo's video):
environment + node FOVs + wearable trail | hop-MSE chart growing over time |
rolling MSE heatmap | rolling certainty heatmap | per-cell certainty-vs-MSE
scatter (cumulative averages).

Requires the run to have saved cell_mse_steps.npy / cell_cert_steps.npy
(runner.py records these for every run from 2026-07-16 onward).

Run from the project root, e.g.:
  python experiments/stage6_spatial_mesh/generate_video.py \
      --run-dir runs/stage6/colour_world/static/overlap_density_ablation/7x7_s3 --fov 7 --static
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

DEFAULT_ENV_DIR = Path("experiments/stage6_spatial_mesh/environment_v2")
WEARABLE_COLOR = "#00FF00"
WEARABLE_COLORS = ["#00FF00", "#00CCFF", "#FF00FF", "#FFFF00"]
HOP_COLORS = ["#FFD700", "#FFA500", "#FF4500", "#8B0000"]
TRAIL_LEN = 20


def env_to_rgb(env_frame: np.ndarray) -> np.ndarray:
    """Matches apply_filter_from_env_value's colour semantics."""
    v = np.clip(env_frame, 0.0, 1.5)
    in_color = v <= 1.0
    t_white = np.clip((v - 1.0) / 0.5, 0.0, 1.0)
    r = np.where(in_color, 0.8 * v + 0.2, 1.0)
    g = np.where(in_color, 0.2, 0.2 + 0.8 * t_white)
    b = np.where(in_color, 1.0 - 0.8 * v, 0.2 + 0.8 * t_white)
    return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fov", type=int, required=True)
    parser.add_argument("--node-centres", type=Path, default=None,
                        help="npy of centres; defaults to the V2 36-node file")
    parser.add_argument("--static", action="store_true",
                        help="tile the first environment frame (6a static runs)")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--rolling", type=int, default=50)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR,
                        help="environment artifact dir (grids + node_centres + wearable_path)")
    parser.add_argument("--offset-field", type=Path, default=None,
                        help="npy of the (T,H,W) or (H,W) label-offset field; if given, "
                             "the left panel shows this instead of the colour env")
    args = parser.parse_args()

    run_dir = args.run_dir
    grids = np.load(args.env_dir / "environment_grids.npy")
    grid_size = grids.shape[1]
    total_steps = grids.shape[0]
    if args.static:
        grids = np.tile(grids[0:1], (total_steps, 1, 1))

    # realised (policy-driven) trajectory takes precedence over the replayed path
    realised_paths_file = run_dir / "realised_wearable_paths.npy"  # multi-wearable, stacked
    realised_path_file = run_dir / "realised_wearable_path.npy"    # single-wearable, back-compat
    if realised_paths_file.exists():
        wearable_paths = list(np.load(realised_paths_file))
        total_steps = min(total_steps, wearable_paths[0].shape[0])
    elif realised_path_file.exists():
        wearable_paths = [np.load(realised_path_file)]
        total_steps = min(total_steps, wearable_paths[0].shape[0])
    else:
        wearable_paths = [np.load(args.env_dir / "wearable_path.npy")]
    node_centres = np.load(args.node_centres or args.env_dir / "node_centres.npy")

    offset_field = np.load(args.offset_field) if args.offset_field else None
    if offset_field is not None and offset_field.ndim == 2:
        offset_field = np.tile(offset_field[None], (total_steps, 1, 1))
    offset_vabs = float(np.percentile(np.abs(offset_field), 99)) if offset_field is not None else 90.0

    hop_mse_history = np.load(run_dir / "hop_mse_history.npy", allow_pickle=True).item()
    cell_mse_steps = np.load(run_dir / "cell_mse_steps.npy")
    cell_cert_steps = np.load(run_dir / "cell_cert_steps.npy")
    total_steps = min(total_steps, cell_mse_steps.shape[0])
    title = args.title or run_dir.name
    out_path = run_dir / "figures" / "experiment_animation.mp4"

    # global colour scales
    mse_vals = cell_mse_steps[~np.isnan(cell_mse_steps)]
    vmax_mse = float(np.percentile(mse_vals, 95)) if len(mse_vals) else 1.0
    cert_vals = cell_cert_steps[~np.isnan(cell_cert_steps)]
    vmin_cert = float(np.percentile(cert_vals, 5)) if len(cert_vals) else 0.0
    vmax_cert = float(np.percentile(cert_vals, 95)) if len(cert_vals) else 1.0
    hop_vals = [v for h in range(4) for v in hop_mse_history[h] if v is not None]
    ylim_hop = float(np.max(hop_vals)) * 1.05 if hop_vals else 0.1

    # cumulative per-cell averages for the scatter
    mse_filled = np.where(np.isnan(cell_mse_steps), 0.0, cell_mse_steps)
    cert_filled = np.where(np.isnan(cell_cert_steps), 0.0, cell_cert_steps)
    mse_count = np.cumsum(~np.isnan(cell_mse_steps), axis=0).astype(float)
    cert_count = np.cumsum(~np.isnan(cell_cert_steps), axis=0).astype(float)
    cum_avg_mse = np.where(mse_count > 0, np.cumsum(mse_filled, axis=0) / mse_count, np.nan)
    cum_avg_cert = np.where(cert_count > 0, np.cumsum(cert_filled, axis=0) / cert_count, np.nan)

    final_mask = ~np.isnan(cum_avg_mse[-1]) & ~np.isnan(cum_avg_cert[-1])
    if final_mask.any():
        scatter_xlim = (float(np.nanmin(cum_avg_cert[-1][final_mask])),
                        float(np.nanmax(cum_avg_cert[-1][final_mask])))
        scatter_ylim = (0.0, float(np.nanmax(cum_avg_mse[-1][final_mask])) * 1.05)
    else:
        scatter_xlim, scatter_ylim = (0.0, 1.0), (0.0, 1.0)

    print(f"Rendering {total_steps} frames -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)
    half = args.fov // 2

    for t in range(total_steps):
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        fig.patch.set_facecolor("#111111")
        outer = GridSpec(1, 2, figure=fig, width_ratios=[2, 3],
                         left=0.02, right=0.98, top=0.93, bottom=0.04, wspace=0.06)
        inner = GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[0, 1],
                                        hspace=0.50, wspace=0.32)
        ax_env = fig.add_subplot(outer[0, 0])
        ax_hop = fig.add_subplot(inner[0, :])
        ax_mse = fig.add_subplot(inner[1, 0])
        ax_cert = fig.add_subplot(inner[1, 1])
        ax_scat = fig.add_subplot(inner[2, :])
        for ax in (ax_env, ax_hop, ax_mse, ax_cert, ax_scat):
            ax.set_facecolor("#1e1e1e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            ax.tick_params(colors="#aaaaaa", labelsize=7)
        fig.suptitle(f"{title}   |   Step {t:03d}/{total_steps}",
                     fontsize=12, fontweight="bold", color="white", y=0.975)

        trail_start = max(0, t - TRAIL_LEN + 1)
        trails = [wp[trail_start:t + 1] for wp in wearable_paths]

        # environment panel: offset field (mapping-conflict runs) or colour filter
        if offset_field is not None:
            im_env = ax_env.imshow(offset_field[t], cmap="RdBu_r", aspect="equal",
                                   interpolation="nearest", vmin=-offset_vabs, vmax=offset_vabs)
            if t == 0:
                cb_env = plt.colorbar(im_env, ax=ax_env, fraction=0.046, pad=0.04)
                cb_env.set_label("label offset (deg)", color="#aaaaaa", fontsize=7)
                cb_env.ax.tick_params(colors="#aaaaaa", labelsize=6)
        else:
            ax_env.imshow(env_to_rgb(grids[t]), aspect="equal", interpolation="nearest")
        for cx, cy in node_centres:
            ax_env.add_patch(plt.Rectangle((cx - half - 0.5, cy - half - 0.5),
                                           args.fov, args.fov, linewidth=0.4,
                                           edgecolor="#555555", facecolor="none"))
            ax_env.plot(cx, cy, ".", color="#888888", markersize=2.5, zorder=3)
        for wi, (trail, wp) in enumerate(zip(trails, wearable_paths)):
            color = WEARABLE_COLORS[wi % len(WEARABLE_COLORS)]
            if len(trail) > 1:
                ax_env.plot(trail[:, 1], trail[:, 0], color=color,
                            linewidth=1.5, alpha=0.75, zorder=5)
            ax_env.scatter(wp[t, 1], wp[t, 0], c=color,
                           s=70, marker="o", zorder=6, edgecolors="white", linewidths=0.7)
        ax_env.set_xlim(-0.5, grid_size - 0.5)
        ax_env.set_ylim(grid_size - 0.5, -0.5)
        ax_env.set_title("Label offset field" if offset_field is not None else "Environment (colour filter)",
                         color="white", fontsize=9, pad=4)
        ax_env.set_xticks([]); ax_env.set_yticks([])

        # hop MSE chart
        for h in range(4):
            vals = hop_mse_history[h]
            xs = [i for i in range(min(t + 1, len(vals))) if vals[i] is not None]
            if xs:
                ax_hop.plot(xs, [vals[i] for i in xs], label=f"Hop {h}",
                            color=HOP_COLORS[h], linewidth=1.5)
        ax_hop.axvline(t, color="#888888", linewidth=0.8, alpha=0.6, linestyle="--")
        ax_hop.set_xlim(0, total_steps - 1)
        ax_hop.set_ylim(0, ylim_hop)
        ax_hop.set_title("MSE by Hop Distance", color="white", fontsize=9, pad=4)
        ax_hop.legend(fontsize=7, facecolor="#333333", edgecolor="#444444",
                      labelcolor="white", loc="upper right")
        ax_hop.grid(True, alpha=0.18, color="#555555")

        # rolling heatmaps
        t0 = max(0, t - args.rolling + 1)
        for ax, data, cmap, vmin, vmax, label in (
            (ax_mse, np.nanmean(cell_mse_steps[t0:t + 1], axis=0), "RdYlGn_r",
             0, vmax_mse, f"MSE (last {args.rolling} steps)"),
            (ax_cert, np.nanmean(cell_cert_steps[t0:t + 1], axis=0), "RdYlGn",
             vmin_cert, vmax_cert, f"Certainty (last {args.rolling} steps)"),
        ):
            im = ax.imshow(data, cmap=cmap, aspect="equal",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
            for wi, (trail, wp) in enumerate(zip(trails, wearable_paths)):
                color = WEARABLE_COLORS[wi % len(WEARABLE_COLORS)]
                if len(trail) > 1:
                    ax.plot(trail[:, 1], trail[:, 0], color=color,
                            linewidth=1.2, alpha=0.8, zorder=5)
                ax.scatter(wp[t, 1], wp[t, 0], c=color,
                           s=35, marker="s", zorder=6, edgecolors="white", linewidths=0.5)
            ax.set_title(label, color="white", fontsize=8, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors="#aaaaaa", labelsize=6)

        # certainty-vs-MSE scatter
        if t > 0:
            mask = ~np.isnan(cum_avg_mse[t]) & ~np.isnan(cum_avg_cert[t])
            if mask.any():
                ax_scat.scatter(cum_avg_cert[t][mask], cum_avg_mse[t][mask],
                                alpha=0.65, s=18, c="steelblue", edgecolors="none")
        ax_scat.set_xlim(*scatter_xlim)
        ax_scat.set_ylim(*scatter_ylim)
        ax_scat.set_xlabel("Mean Certainty (cumulative)", color="#aaaaaa", fontsize=7)
        ax_scat.set_ylabel("Mean MSE (cumulative)", color="#aaaaaa", fontsize=7)
        ax_scat.set_title("Per-cell Certainty vs MSE", color="white", fontsize=9, pad=4)
        ax_scat.grid(True, alpha=0.18, color="#555555")

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        writer.append_data(np.asarray(canvas.buffer_rgba()))
        plt.close(fig)
        if t % 50 == 0:
            print(f"  frame {t + 1}/{total_steps}")

    writer.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
