"""Render proximity-sensor depth from a saved trajectory H5 as grid PNG + MP4.

Usage:
    python scripts/datagen/visualize_proximity.py PATH_TO_TRAJ.h5
    python scripts/datagen/visualize_proximity.py PATH_TO_TRAJ.h5 --traj traj_0 --out /tmp/prox_viz

Outputs in <out_dir>:
    grid.mp4   - all 29 sensors animated, 60 fps (4 substeps x T policy steps).
    grid.png   - all 29 sensors at the middle policy step, last substep.
    per_sensor/<name>.mp4  - one MP4 per sensor (only with --per-sensor).

Reads obs/proximity/link*_sensor_* tensors of shape (T, 4, 8, 8) from the H5.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROXIMITY_RE = re.compile(r"^link(\d+)_sensor_(\d+)$")


def load_proximity_data(h5_path: Path, traj_key: str) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        if traj_key not in f:
            keys = list(f.keys())
            raise SystemExit(f"trajectory key '{traj_key}' not in {h5_path} (have: {keys})")
        prox_group = f[traj_key].get("obs/proximity")
        if prox_group is None:
            raise SystemExit(
                f"{h5_path}/{traj_key} has no obs/proximity group. "
                "Was this collected with --robot skin?"
            )
        return {name: prox_group[name][...] for name in prox_group.keys()}


def sensor_layout(sensor_names: list[str]) -> tuple[list[str], int, int]:
    """Sort sensors by (link, idx) and return (sorted_names, n_rows, n_cols).

    Each link gets its own row; columns within a row are sensor indices on that link.
    """
    parsed = []
    for n in sensor_names:
        m = PROXIMITY_RE.match(n)
        if m:
            parsed.append((int(m.group(1)), int(m.group(2)), n))
    parsed.sort()
    if not parsed:
        return sensor_names, 1, len(sensor_names)
    links = sorted({p[0] for p in parsed})
    n_rows = len(links)
    n_cols = max(sum(1 for p in parsed if p[0] == link) for link in links)
    sorted_names = [p[2] for p in parsed]
    return sorted_names, n_rows, n_cols


def render_grid_frame(
    data: dict[str, np.ndarray],
    sorted_names: list[str],
    n_rows: int,
    n_cols: int,
    t: int,
    sub: int,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Render one (T, sub) tile-grid frame to an HxWx3 uint8 array."""
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 1.6, n_rows * 1.6 + 0.5),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.set_axis_off()

    by_link: dict[int, list[tuple[int, str]]] = {}
    for name in sorted_names:
        link, idx = (int(x) for x in PROXIMITY_RE.match(name).groups())
        by_link.setdefault(link, []).append((idx, name))
    links = sorted(by_link)

    for r, link in enumerate(links):
        for c, (idx, name) in enumerate(sorted(by_link[link])):
            ax = axes[r, c]
            frame = data[name][t, sub]
            ax.imshow(frame, cmap="turbo_r", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(name, fontsize=7, pad=2)

    fig.suptitle(f"step {t}/{data[sorted_names[0]].shape[0] - 1}, substep {sub}", fontsize=10)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", type=Path, help="Path to trajectories_batch_*.h5")
    parser.add_argument("--traj", default="traj_0", help="Trajectory key (default: traj_0)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: <h5_dir>/proximity_viz)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Video fps (default 60.0; matches the 60Hz proximity rate)",
    )
    parser.add_argument(
        "--per-sensor",
        action="store_true",
        help="Also write one MP4 per sensor under <out>/per_sensor/",
    )
    args = parser.parse_args()

    out_dir = args.out or args.h5_path.parent / "proximity_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_proximity_data(args.h5_path, args.traj)
    if not data:
        raise SystemExit(f"No proximity datasets found under {args.traj}/obs/proximity")

    sorted_names, n_rows, n_cols = sensor_layout(list(data.keys()))

    # Compute global depth range (ignoring zero-padded post-reset slots).
    all_data = np.stack([data[n] for n in sorted_names], axis=0)
    nonzero = all_data[all_data > 0]
    vmin = float(nonzero.min()) if nonzero.size else 0.0
    vmax = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    print(f"Loaded {len(sorted_names)} sensors, shape={all_data.shape[1:]}, depth range [{vmin:.3f}, {vmax:.3f}]m")

    T, n_sub = all_data.shape[1:3]

    # Snapshot PNG: middle policy step, last substep
    t_mid = T // 2
    snap = render_grid_frame(data, sorted_names, n_rows, n_cols, t_mid, n_sub - 1, vmin, vmax)
    imageio.imwrite(out_dir / "grid.png", snap)
    print(f"wrote {out_dir / 'grid.png'}")

    # Grid MP4: full trajectory at 60 fps (T x n_sub frames)
    grid_mp4 = out_dir / "grid.mp4"
    # yuv420p + macro_block_size=16 makes the file playable in VSCode/Quicktime/browsers.
    with imageio.get_writer(
        grid_mp4,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    ) as w:
        for t in range(T):
            for sub in range(n_sub):
                w.append_data(render_grid_frame(data, sorted_names, n_rows, n_cols, t, sub, vmin, vmax))
    print(f"wrote {grid_mp4}  ({T * n_sub} frames @ {args.fps} fps)")

    if args.per_sensor:
        per_dir = out_dir / "per_sensor"
        per_dir.mkdir(exist_ok=True)
        for name in sorted_names:
            arr = data[name]  # (T, n_sub, 8, 8)
            # Upscale 32x with nearest-neighbor and apply turbo colormap.
            cmap = matplotlib.colormaps["turbo"]
            normed = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0, 1)
            colored = (cmap(normed)[..., :3] * 255).astype(np.uint8)
            up = np.repeat(np.repeat(colored, 32, axis=2), 32, axis=3)  # (T, sub, 256, 256, 3)
            up = up.reshape(T * n_sub, *up.shape[2:])
            with imageio.get_writer(
                per_dir / f"{name}.mp4",
                fps=args.fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=16,
            ) as w:
                for frame in up:
                    w.append_data(frame)
        print(f"wrote {len(sorted_names)} per-sensor MP4s to {per_dir}")


if __name__ == "__main__":
    main()
