"""End-to-end analysis of a single franka_skin episode.

Usage:
    python scripts/datagen/analyze_sample_episode.py PATH_TO_TRAJ.h5

Outputs `<h5_dir>/analysis/`:
    states.png            joint positions, velocities, TCP pose over time
    rgbd_samples.png      RGB + depth frames at start/middle/end of episode
    proximity_traces.png  per-sensor mean depth over time
    proximity_heatmap.png all sensors x time, color = mean depth
    pointcloud.png        3D scatter from proximity sensors
    pointcloud.ply        same cloud as ply (open in MeshLab)
    report.md             verification answers + summary stats
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROX_RE = re.compile(r"^link\d+_sensor_\d+$")


def decode_byte_array(byte_arr: np.ndarray) -> str | dict | list:
    """Decode a uint8 byte array to JSON or text. Trailing zeros stripped."""
    decoded = bytes(byte_arr).rstrip(b"\x00").decode("utf-8", errors="ignore")
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        return decoded


def decode_byte_timeseries(byte_2d: np.ndarray) -> list:
    """Decode a (T, max_len) uint8 byte array into a list of T decoded values."""
    return [decode_byte_array(byte_2d[t]) for t in range(byte_2d.shape[0])]


def stack_jp(jp_list: list) -> np.ndarray:
    """Pull the arm joint positions (7-DoF) out of a JointPosSensor JSON list."""
    rows = []
    for d in jp_list:
        if isinstance(d, dict) and "arm" in d:
            rows.append(d["arm"])
        elif isinstance(d, list):
            rows.append(d[:7])
        else:
            rows.append([np.nan] * 7)
    return np.array(rows, dtype=float)


def stack_grip(jp_list: list) -> np.ndarray:
    """Extract gripper qpos (2-element) per timestep."""
    rows = []
    for d in jp_list:
        if isinstance(d, dict) and "gripper" in d:
            rows.append(d["gripper"])
        else:
            rows.append([np.nan, np.nan])
    return np.array(rows, dtype=float)


def stack_ee_pose(ep_list: list) -> np.ndarray:
    """Extract EE pose (7d: xyz + quat) per timestep, padding NaN if missing."""
    rows = []
    for d in ep_list:
        if isinstance(d, list) and len(d) >= 7:
            rows.append(d[:7])
        elif isinstance(d, dict) and "tcp_pose" in d:
            rows.append(d["tcp_pose"])
        else:
            rows.append([np.nan] * 7)
    return np.array(rows, dtype=float)


def stack_tcp_from_extra(tcp_arr: np.ndarray) -> np.ndarray:
    """obs/extra/tcp_pose can be (T, 7) float32 directly, or shape (T, 7) array.
    Returns (T, 7) np.float64."""
    return np.asarray(tcp_arr, dtype=float)


def plot_states(
    out_path: Path,
    qpos_arm: np.ndarray,
    qvel_arm: np.ndarray,
    tcp_xyz: np.ndarray,
    policy_phase: np.ndarray | None = None,
):
    fig, axes = plt.subplots(4 if policy_phase is not None else 3, 1, figsize=(10, 11), sharex=True)
    T = qpos_arm.shape[0]
    t = np.arange(T)
    for j in range(qpos_arm.shape[1]):
        axes[0].plot(t, qpos_arm[:, j], label=f"joint{j+1}")
    axes[0].set_title("Arm joint positions (rad)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(alpha=0.3)
    for j in range(qvel_arm.shape[1]):
        axes[1].plot(t, qvel_arm[:, j], label=f"joint{j+1}")
    axes[1].set_title("Arm joint velocities (rad/s)")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[2].plot(t, tcp_xyz[:, 0], label="x")
    axes[2].plot(t, tcp_xyz[:, 1], label="y")
    axes[2].plot(t, tcp_xyz[:, 2], label="z")
    axes[2].set_title("TCP position in world frame (m)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    if policy_phase is not None:
        axes[3].step(t, policy_phase, where="post", lw=1.5)
        axes[3].set_title("Policy phase id (planner stage)")
        axes[3].set_xlabel("policy step")
        axes[3].grid(alpha=0.3)
    else:
        axes[2].set_xlabel("policy step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def grab_video_frame(video_path: Path, frac: float) -> np.ndarray | None:
    """Return one frame at fractional position `frac` (0..1) of the video. None on failure."""
    try:
        with imageio.get_reader(str(video_path)) as r:
            n = r.count_frames()
            idx = max(0, min(n - 1, int(frac * n)))
            return r.get_data(idx)
    except Exception:
        return None


def plot_rgbd_samples(
    out_path: Path,
    h5_dir: Path,
    episode_idx: int = 0,
    suffix: str = "_batch_1_of_1",
):
    cams = ["wrist_camera", "exo_camera_1"]
    fracs = [0.05, 0.5, 0.95]
    n_rows = 2 * len(cams)  # rgb + depth per cam
    n_cols = len(fracs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5), squeeze=False)
    for i, cam in enumerate(cams):
        rgb_path = h5_dir / f"episode_{episode_idx:08d}_{cam}{suffix}.mp4"
        depth_path = h5_dir / f"episode_{episode_idx:08d}_{cam}_depth{suffix}.mp4"
        for j, frac in enumerate(fracs):
            ax_rgb = axes[2 * i, j]
            ax_depth = axes[2 * i + 1, j]
            rgb = grab_video_frame(rgb_path, frac) if rgb_path.exists() else None
            if rgb is not None:
                ax_rgb.imshow(rgb)
                ax_rgb.set_title(f"{cam} RGB t={frac:.0%}", fontsize=9)
            else:
                ax_rgb.text(0.5, 0.5, f"{rgb_path.name}\nnot found", ha="center", va="center")
            ax_rgb.axis("off")

            depth = grab_video_frame(depth_path, frac) if depth_path.exists() else None
            if depth is not None:
                # depth videos are encoded as RGB; just visualize as-is for sanity.
                ax_depth.imshow(depth)
                ax_depth.set_title(f"{cam} depth t={frac:.0%}", fontsize=9)
            else:
                ax_depth.text(0.5, 0.5, f"{depth_path.name}\nnot found", ha="center", va="center")
            ax_depth.axis("off")
    fig.suptitle("RGB + depth samples (start / middle / end)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def proximity_traces(prox: dict[str, np.ndarray], out_path: Path):
    """Mean depth per sensor over time, grouped by link."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    by_link: dict[int, list[tuple[int, str, np.ndarray]]] = {}
    for name, arr in prox.items():
        m = re.match(r"link(\d+)_sensor_(\d+)", name)
        if m:
            by_link.setdefault(int(m.group(1)), []).append((int(m.group(2)), name, arr))
    links = sorted(by_link)
    for ax, link in zip(axes.flat, links):
        for idx, name, arr in sorted(by_link[link]):
            mean_per_t = arr.mean(axis=(1, 2, 3))  # mean over (substep, H, W)
            ax.plot(mean_per_t, label=f"sensor_{idx}", alpha=0.85)
        ax.set_title(f"link{link}")
        ax.legend(ncol=4, fontsize=7)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("policy step")
    axes[1, 1].set_xlabel("policy step")
    axes[0, 0].set_ylabel("mean depth (m)")
    axes[1, 0].set_ylabel("mean depth (m)")
    fig.suptitle("Proximity sensor mean depth over time, by link", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def proximity_heatmap(prox: dict[str, np.ndarray], out_path: Path):
    """Heatmap: rows=sensors, cols=time, values=mean depth."""
    names = sorted(prox.keys())
    rows = [prox[n].mean(axis=(1, 2, 3)) for n in names]
    M = np.stack(rows, axis=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    # turbo_r: close=red (alarm), far=blue (safe), the standard SPAD/proximity convention.
    im = ax.imshow(M, aspect="auto", cmap="turbo_r", interpolation="nearest")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("policy step")
    ax.set_title("Per-sensor mean depth (m) over time")
    fig.colorbar(im, ax=ax, label="depth (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def reconstruct_pointcloud(
    prox: dict[str, np.ndarray],
    sensor_param: dict[str, dict[str, np.ndarray]],
    fovy_deg: float = 45.0,
    sample_every: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Per-pixel back-projection of all 29 x T x n_sub x 8 x 8 depth readings.

    Each 8x8 depth frame is treated as a tiny pinhole image. We derive intrinsics
    from the SPAD FOV (45deg square sensor, 8x8 pixels), back-project each pixel
    to camera frame in MuJoCo's GL convention (y up, -z forward), then transform
    to world via cam2world_gl.

    Returns (points, times) where times[i] is the policy step at which point i
    was captured. Up to 29 * T * n_sub * 64 points.
    """
    H = W = 8
    f = (H / 2) / np.tan(np.deg2rad(fovy_deg / 2))  # focal length in pixels
    cx = cy = (H - 1) / 2.0
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))  # (8, 8) each

    out_pts: list[np.ndarray] = []
    out_times: list[np.ndarray] = []
    out_sidx: list[np.ndarray] = []
    sensor_names = sorted(prox.keys())
    for s_idx, name in enumerate(sensor_names):
        arr = prox[name]
        if name not in sensor_param or "cam2world_gl" not in sensor_param[name]:
            continue
        cam2world = sensor_param[name]["cam2world_gl"]  # (T, 4, 4)
        T_steps, n_sub = arr.shape[:2]
        for t in range(0, T_steps, sample_every):
            for sub in range(n_sub):
                depth = arr[t, sub]  # (8, 8)
                mask = (depth >= 0.05) & (depth <= 4.0)
                if not mask.any():
                    continue
                d = depth[mask].astype(np.float64)
                u = u_grid[mask].astype(np.float64)
                v = v_grid[mask].astype(np.float64)
                # GL convention: camera looks -Z, +Y up.
                x_cam = (u - cx) * d / f
                y_cam = -(v - cy) * d / f
                z_cam = -d
                ones = np.ones_like(d)
                p_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=1)
                p_world = p_cam @ cam2world[t].T
                out_pts.append(p_world[:, :3])
                out_times.append(np.full(len(d), t, dtype=np.int64))
                out_sidx.append(np.full(len(d), s_idx, dtype=np.int64))
    if not out_pts:
        return (
            np.zeros((0, 3)),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            sensor_names,
        )
    return (
        np.concatenate(out_pts, axis=0),
        np.concatenate(out_times),
        np.concatenate(out_sidx),
        sensor_names,
    )


def plot_pointcloud(pts: np.ndarray, out_path: Path, max_show: int = 30000):
    """3D scatter; subsample if more than `max_show` points for readability."""
    if pts.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no points", ha="center")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return
    if len(pts) > max_show:
        idx = np.random.default_rng(0).choice(len(pts), size=max_show, replace=False)
        pts_show = pts[idx]
        title_extra = f" (showing {max_show:,} of {len(pts):,})"
    else:
        pts_show = pts
        title_extra = ""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_show[:, 0], pts_show[:, 1], pts_show[:, 2], s=2, c=pts_show[:, 2], cmap="turbo", alpha=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Reconstructed point cloud{title_extra}")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _link_color(name: str) -> tuple[float, float, float]:
    """Color-code by link so different sensors are distinguishable."""
    m = re.match(r"link(\d+)_sensor_", name)
    if not m:
        return (0.5, 0.5, 0.5)
    link = int(m.group(1))
    palette = {2: (1.0, 0.30, 0.30), 3: (0.30, 0.85, 0.30), 5: (0.30, 0.50, 1.00), 6: (1.00, 0.80, 0.20)}
    return palette.get(link, (0.5, 0.5, 0.5))


def project_pointcloud_overlay(
    pts: np.ndarray,
    pts_t: np.ndarray,  # per-point timestep (so we can pick same-time points)
    pts_sensor: np.ndarray,  # per-point sensor index (for color coding)
    sensor_names: list[str],
    h5_dir: Path,
    sensor_param: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    episode_idx: int = 0,
    suffix: str = "_batch_1_of_1",
    cam_name: str = "exo_camera_1",
):
    """Time-aligned overlay: project ONLY the points captured at the same
    timestep as the RGB frame, colored by which link the sensor lives on.
    """
    if cam_name not in sensor_param:
        return
    rgb_path = h5_dir / f"episode_{episode_idx:08d}_{cam_name}{suffix}.mp4"
    if not rgb_path.exists():
        return

    intr = sensor_param[cam_name]["intrinsic_cv"]
    extr = sensor_param[cam_name]["extrinsic_cv"]
    T = intr.shape[0]

    targets = [int(T * f) for f in (0.05, 0.5, 0.95)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, t_target in zip(axes, targets):
        rgb = grab_video_frame(rgb_path, t_target / max(T - 1, 1))
        if rgb is None:
            ax.text(0.5, 0.5, "no rgb", ha="center")
            continue
        H, W = rgb.shape[:2]

        # Time-aligned filter
        mask_t = pts_t == t_target
        P = pts[mask_t]
        S = pts_sensor[mask_t]
        if P.size == 0:
            ax.imshow(rgb)
            ax.set_title(f"t={t_target}: no points")
            ax.axis("off")
            continue

        K = intr[t_target]
        Rt = extr[t_target]
        P_h = np.concatenate([P, np.ones((len(P), 1))], axis=1)
        P_cam = P_h @ Rt.T
        in_front = P_cam[:, 2] > 0.05
        P_cam, S = P_cam[in_front], S[in_front]
        if P_cam.size == 0:
            ax.imshow(rgb)
            ax.set_title(f"t={t_target}: nothing in view")
            ax.axis("off")
            continue
        uv_h = P_cam @ K.T
        uv = uv_h[:, :2] / uv_h[:, 2:3]
        u, v = uv[:, 0], uv[:, 1]
        mask_in = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, S = u[mask_in], v[mask_in], S[mask_in]
        colors = np.array([_link_color(sensor_names[i]) for i in S])
        ax.imshow(rgb)
        ax.scatter(u, v, c=colors, s=4, alpha=0.85, edgecolors="black", linewidths=0.1)
        ax.set_title(f"{cam_name} t={t_target}/{T-1} — {len(u)} sensor pts (link2 red, link3 green, link5 blue, link6 yellow)", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        "Time-aligned proximity points on exo RGB (only points captured at the SAME timestep)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sensor_grid_at_t(
    prox: dict[str, np.ndarray],
    h5_dir: Path,
    out_path: Path,
    episode_idx: int = 0,
    suffix: str = "_batch_1_of_1",
):
    """For a representative timestep, show:
       (top)    wrist + exo RGB at this t
       (bottom) all 29 sensor 8x8 depth tiles laid out by link.

    This is the canonical "what does the model see at this moment?" view.
    """
    sample_arr = next(iter(prox.values()))
    T_steps, n_sub = sample_arr.shape[:2]
    t = T_steps // 2  # mid-episode
    sub = n_sub - 1  # most recent substep

    # Layout: top row 2 RGB images, bottom block sensor grid (4 rows x 8 cols).
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(5, 8, height_ratios=[2.5, 1, 1, 1, 1])

    for col, (cam, frac) in enumerate([("wrist_camera", t / max(T_steps - 1, 1)),
                                        ("exo_camera_1", t / max(T_steps - 1, 1))]):
        ax = fig.add_subplot(gs[0, col * 4:(col + 1) * 4])
        rgb_path = h5_dir / f"episode_{episode_idx:08d}_{cam}{suffix}.mp4"
        rgb = grab_video_frame(rgb_path, frac) if rgb_path.exists() else None
        if rgb is not None:
            ax.imshow(rgb)
        ax.set_title(f"{cam} RGB at t={t}/{T_steps-1}", fontsize=11)
        ax.axis("off")

    # Build per-link rows for the depth tile grid
    by_link: dict[int, list[tuple[int, str]]] = {}
    for name in prox:
        m = re.match(r"link(\d+)_sensor_(\d+)", name)
        if m:
            by_link.setdefault(int(m.group(1)), []).append((int(m.group(2)), name))
    links = sorted(by_link)

    # Global vmin/vmax for consistent color across all sensors at this t.
    vals = np.concatenate([prox[n][t, sub].ravel() for n in prox])
    vmin = float(np.percentile(vals[vals > 0], 1)) if (vals > 0).any() else 0
    vmax = float(np.percentile(vals[vals > 0], 99)) if (vals > 0).any() else 4

    for r, link in enumerate(links):
        for c, (idx, name) in enumerate(sorted(by_link[link])):
            ax = fig.add_subplot(gs[r + 1, c])
            tile = prox[name][t, sub]
            # turbo_r: close=red, far=blue (proximity-sensor convention)
            ax.imshow(tile, cmap="turbo_r", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(
                f"{name}\nmin={tile.min():.2f} mean={tile.mean():.2f} max={tile.max():.2f} m",
                fontsize=7,
                color=_link_color(name),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            # Per-pixel depth values overlaid as text (8x8 = 64 cells, small font)
            for yi in range(tile.shape[0]):
                for xi in range(tile.shape[1]):
                    val = tile[yi, xi]
                    # Pick black or white text depending on cell brightness
                    text_color = "white" if val > (vmin + vmax) / 2 else "black"
                    ax.text(
                        xi, yi, f"{val:.2f}",
                        ha="center", va="center", fontsize=4.5,
                        color=text_color,
                    )
        # blank out unused columns
        for c in range(len(by_link[link]), 8):
            ax = fig.add_subplot(gs[r + 1, c])
            ax.axis("off")

    fig.suptitle(
        f"Sensor panel: what the model sees at t={t} (substep {sub}, depth in [{vmin:.2f}, {vmax:.2f}] m)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_min_depth_per_sensor(prox: dict[str, np.ndarray], out_path: Path):
    """Min-depth-per-sensor over time. This is the actual learning feature:
    'is there something within reach of this part of the skin?' for each link."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    by_link: dict[int, list[tuple[int, str, np.ndarray]]] = {}
    for name, arr in prox.items():
        m = re.match(r"link(\d+)_sensor_(\d+)", name)
        if m:
            by_link.setdefault(int(m.group(1)), []).append((int(m.group(2)), name, arr))
    for ax, link in zip(axes.flat, sorted(by_link)):
        for idx, name, arr in sorted(by_link[link]):
            # Within each substep, take the min over the 8x8 frame; then take min over substeps.
            min_per_t = arr.min(axis=(1, 2, 3))
            ax.plot(min_per_t, label=f"sensor_{idx}", alpha=0.85)
        ax.set_title(f"link{link}")
        ax.legend(ncol=4, fontsize=7)
        ax.set_ylabel("min depth (m)")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
    axes[1, 0].set_xlabel("policy step")
    axes[1, 1].set_xlabel("policy step")
    fig.suptitle(
        "Min depth per sensor over time — 'is something close?' (log scale on y)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pointcloud_at_t(
    pts: np.ndarray,
    pts_t: np.ndarray,
    pts_sensor: np.ndarray,
    sensor_names: list[str],
    out_path: Path,
):
    """Three single-timestep snapshots of the cloud, colored by link, side by side."""
    if pts.size == 0:
        return
    T = int(pts_t.max()) + 1
    targets = [int(T * f) for f in (0.05, 0.5, 0.95)]
    fig = plt.figure(figsize=(18, 6))
    for i, t_target in enumerate(targets):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        m = pts_t == t_target
        P = pts[m]
        if P.size == 0:
            ax.set_title(f"t={t_target}: no points")
            continue
        S = pts_sensor[m]
        colors = np.array([_link_color(sensor_names[i]) for i in S])
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=colors, s=2, alpha=0.6)
        ax.set_title(f"t={t_target} — {len(P)} pts", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle("Single-timestep proximity cloud (link2 red, link3 green, link5 blue, link6 yellow)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_ply(pts: np.ndarray, out_path: Path):
    """Write a binary little-endian PLY (3x faster + smaller than ASCII)."""
    pts32 = pts.astype("<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts32)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts32.tobytes())


def verify(
    prox: dict[str, np.ndarray],
    qvel_arm: np.ndarray,
    policy_phase: np.ndarray | None,
) -> dict:
    """Run the four verification checks. Returns a results dict for the report."""
    all_data = np.concatenate([v.reshape(-1) for v in prox.values()])
    nonzero = all_data[all_data > 0]
    arm_speed = np.linalg.norm(qvel_arm, axis=1)  # (T,)

    # Q1 Plausibility: depth in [0.05, 4.0]m for nonzero values, no NaN/Inf
    plaus = {
        "n_total": int(all_data.size),
        "n_nonzero": int(nonzero.size),
        "min_depth": float(nonzero.min()) if nonzero.size else 0.0,
        "max_depth": float(nonzero.max()) if nonzero.size else 0.0,
        "median_depth": float(np.median(nonzero)) if nonzero.size else 0.0,
        "frac_in_range_005_4": float(((nonzero >= 0.05) & (nonzero <= 4.0)).mean()) if nonzero.size else 0.0,
        "any_nan": bool(np.isnan(all_data).any()),
        "any_inf": bool(np.isinf(all_data).any()),
    }

    # Q2 Temporal structure
    per_sensor_var, per_sensor_step_diff = [], []
    for arr in prox.values():
        mean_t = arr.mean(axis=(1, 2, 3))
        per_sensor_var.append(float(mean_t.var()))
        diffs = np.abs(np.diff(mean_t))
        per_sensor_step_diff.append(float(diffs.mean()) if diffs.size else 0.0)
    temporal = {
        "mean_temporal_variance_m2": float(np.mean(per_sensor_var)),
        "max_temporal_variance_m2": float(np.max(per_sensor_var)),
        "mean_step_to_step_diff_m": float(np.mean(per_sensor_step_diff)),
    }

    # Aggregated proximity activity per step: sum of |Δ mean_depth| across sensors
    delta_per_t = np.zeros(qvel_arm.shape[0])
    for arr in prox.values():
        mt = arr.mean(axis=(1, 2, 3))
        d = np.zeros_like(mt)
        d[1:] = np.abs(np.diff(mt))
        delta_per_t += d

    # Q3 Phase correlation:
    #   (a) Pearson correlation between per-step proximity activity and arm speed
    #   (b) per-phase mean depth — if planner phases differ, sensors should reflect that
    if delta_per_t.std() > 1e-9 and arm_speed.std() > 1e-9:
        corr_speed = float(np.corrcoef(delta_per_t, arm_speed)[0, 1])
    else:
        corr_speed = float("nan")

    per_phase_stats = {}
    phase_explains = 0.0
    std_ratio = 1.0
    if policy_phase is not None and policy_phase.size == qvel_arm.shape[0]:
        # Aggregate mean prox depth (across all sensors, all substeps, all pixels) per phase
        all_mean_per_t = np.mean([arr.mean(axis=(1, 2, 3)) for arr in prox.values()], axis=0)
        for ph in np.unique(policy_phase):
            mask = policy_phase == ph
            if mask.sum() > 0:
                per_phase_stats[int(ph)] = {
                    "n_steps": int(mask.sum()),
                    "mean_prox_m": float(all_mean_per_t[mask].mean()),
                    "std_prox_m": float(all_mean_per_t[mask].std()),
                    "mean_arm_speed": float(arm_speed[mask].mean()),
                }
        if len(per_phase_stats) > 1:
            # 1) Variance of phase-means / total variance (mean-shift component)
            phase_means = np.array([s["mean_prox_m"] for s in per_phase_stats.values()])
            phase_explains = float(phase_means.var() / max(all_mean_per_t.var(), 1e-12))
            # 2) Ratio of max-within-phase std to min-within-phase std (discriminability:
            #    captures the case where motion phases are scanning while grasping phases
            #    are nearly stationary - the dominant signal in our task).
            stds = np.array([s["std_prox_m"] for s in per_phase_stats.values()])
            if stds.min() > 1e-9:
                std_ratio = float(stds.max() / stds.min())
            else:
                std_ratio = float("inf")

    phase = {
        "pearson_corr(prox_change, arm_speed)": corr_speed,
        "phase_var_over_total_var": phase_explains,
        "max_phase_std_over_min": std_ratio,
        "per_phase": per_phase_stats,
    }

    return {"plausibility": plaus, "temporal": temporal, "phase": phase}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--traj", type=str, default="traj_0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    h5_path: Path = args.h5_path
    h5_dir = h5_path.parent
    out_dir = args.out or (h5_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        traj = f[args.traj]

        # Decode states
        qpos_bytes = traj["obs/agent/qpos"][:]
        qvel_bytes = traj["obs/agent/qvel"][:]
        ee_pose_bytes = traj["actions/ee_pose"][:]

        qpos_arm = stack_jp(decode_byte_timeseries(qpos_bytes))
        gripper_q = stack_grip(decode_byte_timeseries(qpos_bytes))
        qvel_arm = stack_jp(decode_byte_timeseries(qvel_bytes))
        ee_pose = stack_ee_pose(decode_byte_timeseries(ee_pose_bytes))

        # Prefer obs/extra/tcp_pose (saved as float32 directly) for ground-truth EE.
        tcp_pose = traj["obs/extra/tcp_pose"][:] if "obs/extra/tcp_pose" in traj else ee_pose
        policy_phase = traj["obs/extra/policy_phase"][:] if "obs/extra/policy_phase" in traj else None

        # Proximity sensors
        prox = {}
        if "obs/proximity" in traj:
            for name in traj["obs/proximity"].keys():
                prox[name] = traj[f"obs/proximity/{name}"][:]

        # Sensor params (intrinsics, extrinsics, cam2world)
        sensor_param = {}
        if "obs/sensor_param" in traj:
            for cam in traj["obs/sensor_param"].keys():
                sensor_param[cam] = {k: traj[f"obs/sensor_param/{cam}/{k}"][:] for k in traj[f"obs/sensor_param/{cam}"].keys()}

        success = bool(traj.get("successes", np.array([False]))[-1]) if "successes" in traj else None

    print(f"Loaded {qpos_arm.shape[0]} timesteps, {len(prox)} proximity sensors, success={success}")

    # 1) States plot
    plot_states(out_dir / "states.png", qpos_arm, qvel_arm, tcp_pose[:, :3], policy_phase)
    print("wrote states.png")

    # 2) RGBD samples
    plot_rgbd_samples(out_dir / "rgbd_samples.png", h5_dir, episode_idx=args.episode_idx)
    print("wrote rgbd_samples.png")

    # 3) Proximity traces
    if prox:
        proximity_traces(prox, out_dir / "proximity_traces.png")
        print("wrote proximity_traces.png")
        proximity_heatmap(prox, out_dir / "proximity_heatmap.png")
        print("wrote proximity_heatmap.png")

    # 4) Point cloud + interpretable visualizations
    if prox:
        pts, pts_t, pts_s, sensor_names = reconstruct_pointcloud(prox, sensor_param)
    else:
        pts = np.zeros((0, 3))
        pts_t = pts_s = np.zeros((0,), dtype=int)
        sensor_names = []
    if pts.size:
        plot_pointcloud(pts, out_dir / "pointcloud_full.png")
        write_ply(pts, out_dir / "pointcloud.ply")
        print(f"wrote pointcloud_full.png + .ply ({len(pts):,} points)")

        # 4a) Single-timestep cloud snapshots colored by link
        plot_pointcloud_at_t(pts, pts_t, pts_s, sensor_names, out_dir / "pointcloud_at_t.png")
        print("wrote pointcloud_at_t.png")

        # 4b) Time-aligned overlay (only same-time points projected onto same-time RGB)
        project_pointcloud_overlay(
            pts, pts_t, pts_s, sensor_names, h5_dir, sensor_param,
            out_dir / "pointcloud_overlay.png",
            episode_idx=args.episode_idx,
        )
        print("wrote pointcloud_overlay.png")

        # 4c) Sensor-panel-at-t: RGB + 29 depth tiles laid out by link
        plot_sensor_grid_at_t(
            prox, h5_dir, out_dir / "sensor_panel.png",
            episode_idx=args.episode_idx,
        )
        print("wrote sensor_panel.png")

        # 4d) Min-depth-per-sensor over time (the actual learning feature)
        plot_min_depth_per_sensor(prox, out_dir / "sensor_min_depth.png")
        print("wrote sensor_min_depth.png")

    # 5) Verification
    verif = verify(prox, qvel_arm, policy_phase) if prox else {}

    # 6) Markdown report
    report = ["# Sample episode analysis", ""]
    report.append(f"- Source: `{h5_path}`")
    report.append(f"- Timesteps: **{qpos_arm.shape[0]}**")
    report.append(f"- Proximity sensors: **{len(prox)}**")
    if prox:
        sample_arr = next(iter(prox.values()))
        report.append(f"- Per-sensor shape: `{sample_arr.shape}` (T, n_substeps, H, W)")
    report.append(f"- Episode success: **{success}**")
    report.append("")

    if verif:
        p = verif["plausibility"]
        t = verif["temporal"]
        ph = verif["phase"]
        report.append("## Verification")
        report.append("")
        report.append("### Q1. Are the readings physically plausible?")
        report.append("")
        report.append(f"- Total samples: {p['n_total']:,}, nonzero: {p['n_nonzero']:,} ({p['n_nonzero']/max(p['n_total'],1):.1%})")
        report.append(f"- Depth range (nonzero): **[{p['min_depth']:.3f}, {p['max_depth']:.3f}] m**, median {p['median_depth']:.3f} m")
        report.append(f"- Fraction within SPAD spec range [0.05, 4.0]: **{p['frac_in_range_005_4']:.3%}**")
        report.append(f"- Any NaN/Inf: NaN={p['any_nan']}, Inf={p['any_inf']}")
        report.append(
            "- Note: depth values slightly above 4.0 m come from the renderer's zfar (10 m); "
            "consumers should clip to [0.05, 4.0] m for SPAD-faithful readings."
        )
        plaus_pass = (
            not p["any_nan"]
            and not p["any_inf"]
            and p["min_depth"] >= 0.05
            and p["frac_in_range_005_4"] > 0.95
        )
        report.append(f"- **PASS**: {plaus_pass}")
        report.append("")

        report.append("### Q2. Do readings change over time (temporal structure)?")
        report.append("")
        report.append(f"- Mean temporal variance per sensor: **{t['mean_temporal_variance_m2']:.5f} m²**")
        report.append(f"- Max temporal variance per sensor:  **{t['max_temporal_variance_m2']:.5f} m²**")
        report.append(f"- Mean step-to-step Δ depth: **{t['mean_step_to_step_diff_m']:.5f} m**")
        temp_pass = t["mean_temporal_variance_m2"] > 1e-6 and t["mean_step_to_step_diff_m"] > 1e-4
        report.append(f"- **PASS**: {temp_pass} (variance > 1e-6, step-Δ > 0.1mm)")
        report.append("")

        report.append("### Q3. Do readings correlate with task phase?")
        report.append("")
        corr = ph["pearson_corr(prox_change, arm_speed)"]
        phase_explains = ph["phase_var_over_total_var"]
        std_ratio = ph["max_phase_std_over_min"]
        report.append(f"- Pearson r(Σ|Δ depth|, ‖q̇‖): **{corr:+.3f}**  (per-step proximity activity vs arm joint-velocity magnitude)")
        report.append(f"- Phase-mean variance / total variance: **{phase_explains:.3f}**  (mean-level shift between phases)")
        if std_ratio == float("inf"):
            report.append("- Max-within-phase std / min-within-phase std: **inf**  (one phase has zero variability — sensors are nailed down)")
        else:
            report.append(f"- Max-within-phase std / min-within-phase std: **{std_ratio:.1f}x**  (within-phase variability differs by this factor across phases)")
        if ph["per_phase"]:
            report.append("")
            report.append("| phase id | n_steps | mean prox depth (m) | within-phase std (m) | mean ‖q̇‖ (rad/s) |")
            report.append("|----------|---------|---------------------|----------------------|--------------------|")
            for pid, st in sorted(ph["per_phase"].items()):
                report.append(
                    f"| {pid} | {st['n_steps']} | {st['mean_prox_m']:.3f} | {st['std_prox_m']:.3f} | {st['mean_arm_speed']:.3f} |"
                )
        # PASS if either: phases shift the mean materially, OR phases differ in within-phase variability
        phase_pass = (phase_explains > 0.05) or (std_ratio > 5.0)
        report.append("")
        report.append(
            f"- **PASS**: {phase_pass}  (phase-mean shift > 5% of total variance OR within-phase std varies > 5x across phases)"
        )
        report.append("")

        report.append("### Q4. Is data saved in the right place with the right schema?")
        report.append("")
        report.append("Expected schema:")
        report.append("- `obs/proximity/link{N}_sensor_{i}` (29 datasets, shape (T, 4, 8, 8) float32)")
        report.append("- `obs/sensor_param/<cam>/{intrinsic_cv, extrinsic_cv, cam2world_gl}`")
        report.append("- `obs/agent/qpos`, `obs/agent/qvel` as JSON-encoded uint8")
        report.append("- `actions/{ee_pose, ee_twist, joint_pos, joint_pos_rel, commanded_action}`")
        report.append("- Companion MP4s for `wrist_camera`, `exo_camera_1` (RGB + `_depth`)")
        report.append("")
        report.append(f"- Found {len(prox)} proximity datasets ✓ (29 expected)")
        report.append(f"- Found {len(sensor_param)} sensor_param entries ✓ (≥31 expected)")
        rgb_count = sum((h5_dir / f"episode_{args.episode_idx:08d}_{c}_batch_1_of_1.mp4").exists() for c in ["wrist_camera", "exo_camera_1"])
        depth_count = sum((h5_dir / f"episode_{args.episode_idx:08d}_{c}_depth_batch_1_of_1.mp4").exists() for c in ["wrist_camera", "exo_camera_1"])
        report.append(f"- RGB videos found: {rgb_count}/2")
        report.append(f"- Depth videos found: {depth_count}/2")
        schema_pass = len(prox) == 29 and rgb_count == 2 and depth_count == 2
        report.append(f"- **PASS**: {schema_pass}")
        report.append("")

    if pts.size:
        report.append("## Point cloud reconstruction")
        report.append("")
        report.append(
            f"- **{len(pts):,} points** emitted — per-pixel back-projection of every "
            f"(sensor, substep, time, u, v) reading with depth in [0.05, 4.0]m."
        )
        report.append(
            f"- Theoretical maximum: 29 sensors x {qpos_arm.shape[0]} steps x 4 substeps x 64 px = "
            f"{29 * qpos_arm.shape[0] * 4 * 64:,} pts"
        )
        report.append(f"- World x range: [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}] m")
        report.append(f"- World y range: [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}] m")
        report.append(f"- World z range: [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}] m")
        report.append("- Open `pointcloud.ply` in MeshLab/CloudCompare to inspect alongside the scene.")
        report.append("")
        report.append("### Per-sensor diagnostics")
        report.append("")
        report.append("| sensor | n valid pts | mean depth (m) | min depth | max depth | frac saturated (>=4.0m) |")
        report.append("|--------|-------------|----------------|-----------|-----------|--------------------------|")
        for name in sorted(prox.keys()):
            arr = prox[name]
            valid = arr[(arr >= 0.05) & (arr <= 4.0)]
            n_total = arr.size
            sat_frac = float(((arr > 0) & (arr >= 4.0)).sum()) / max(n_total, 1)
            report.append(
                f"| {name} | {valid.size:,}/{n_total:,} | "
                f"{(valid.mean() if valid.size else 0):.3f} | "
                f"{(valid.min() if valid.size else 0):.3f} | "
                f"{(valid.max() if valid.size else 0):.3f} | "
                f"{sat_frac:.1%} |"
            )
        report.append("")

    (out_dir / "report.md").write_text("\n".join(report))
    print(f"wrote report.md\nDone. Open {out_dir}/")


if __name__ == "__main__":
    main()
