"""Rigorous GT test for proximity sensor data.

For one specific timestep, freshly render proximity depth at native 8x8
resolution from the same scene + robot pose, and compare to what was recorded.

Usage:
    python scripts/datagen/verify_proximity_gt.py PATH_TO_TRAJ.h5 [--t 10]

Outputs (same dir as the H5, under analysis/gt_compare_t<T>/):
    grid.png       3 panels per sensor: recorded mean, GT 8x8, |error|
    summary.md     per-sensor stats: mean error, max error, frac of bad pixels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENE_XML = "/home/jaydv/code/prox_learning/assets/scenes/ithor/FloorPlan1_physics.xml"
ROBOT_XML = "/home/jaydv/code/prox_learning/assets/robots/franka_skin/model.xml"
PREFIX = "robot_0/"
BASE_SIZE_Z = 0.58  # FrankaRobotConfig.base_size[2]


def load_recorded(h5_path: Path, t: int, traj: str = "traj_0") -> dict:
    out = {}
    with h5py.File(h5_path, "r") as f:
        qpos_bytes = f[f"{traj}/obs/agent/qpos"][t]
        out["qpos"] = json.loads(bytes(qpos_bytes).rstrip(b"\x00").decode("utf-8"))
        out["base_pose"] = f[f"{traj}/obs/extra/robot_base_pose"][t][:]
        out["proximity"] = {
            n: f[f"{traj}/obs/proximity/{n}"][t][:] for n in f[f"{traj}/obs/proximity"]
        }
        out["policy_phase"] = int(f[f"{traj}/obs/extra/policy_phase"][t]) if f"{traj}/obs/extra/policy_phase" in f else None
    return out


def build_test_model(base_pose: np.ndarray) -> mujoco.MjModel:
    """Reproduce the molmospaces scene+robot composition at the recorded base pose."""
    scene_spec = mujoco.MjSpec.from_file(SCENE_XML)
    robot_spec = mujoco.MjSpec.from_file(ROBOT_XML)

    base_body = scene_spec.worldbody.add_body(
        name=f"{PREFIX}base",
        pos=base_pose[:3].astype(float).tolist(),
        quat=base_pose[3:7].astype(float).tolist(),
        mocap=True,
    )
    # Mimic the wooden platform: a no-collision visual box (we don't need the
    # exact texture — just the right height so the robot sits at the right z).
    base_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.25, 0.25, BASE_SIZE_Z / 2],
        pos=[0.0, 0.0, BASE_SIZE_Z / 2],
        rgba=[0.4, 0.25, 0.1, 0.0],  # alpha=0 so it doesn't visually clutter
        contype=0,
        conaffinity=0,
    )
    attach_frame = base_body.add_frame(pos=[0.0, 0.0, BASE_SIZE_Z])
    fr3_link0 = robot_spec.body("fr3_link0")
    if fr3_link0 is None:
        raise RuntimeError("fr3_link0 not found in robot_spec")
    attach_frame.attach_body(fr3_link0, PREFIX, "")
    return scene_spec.compile()


def set_qpos_to_recorded(model: mujoco.MjModel, data: mujoco.MjData, qpos_dict: dict) -> None:
    arm = qpos_dict.get("arm", [])
    for i, val in enumerate(arm[:7]):
        adr = model.joint(f"{PREFIX}fr3_joint{i+1}").qposadr[0]
        data.qpos[adr] = float(val)
    # Gripper: drive the right_driver_joint; equality constraints propagate the rest.
    grip = qpos_dict.get("gripper", [])
    # Gripper qpos in the recorded data is [right, left] each in [0, 0.04] from the 2f85 model.
    # The driver joint range is [0, 0.834] rad. Open gripper -> driver_joint=0.
    # We just set whatever the saved joints are; mj_forward + equality constraints handle the rest.
    if len(grip) >= 1:
        try:
            adr = model.joint(f"{PREFIX}gripper/right_driver_joint").qposadr[0]
            data.qpos[adr] = float(grip[0]) * 30.0  # rough conversion (2cm finger -> ~0.6 rad)
        except KeyError:
            pass
    mujoco.mj_forward(model, data)


def render_gt_proximity(
    model: mujoco.MjModel, data: mujoco.MjData, sensor_names: list[str]
) -> dict[str, np.ndarray]:
    renderer = mujoco.Renderer(model, height=8, width=8)
    # Hide group 2 (skin) - sensors live inside the skin mesh volume.
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0
    gt = {}
    for name in sensor_names:
        cam_name = f"{PREFIX}{name}"
        # Verify the camera exists in the model.
        try:
            model.camera(cam_name)
        except KeyError:
            print(f"  WARN: camera {cam_name} not found, skipping")
            continue
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name, scene_option=opt)
        depth = renderer.render().copy()
        gt[name] = depth
    return gt


def plot_compare_grid(
    recorded: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    out_path: Path,
):
    """Per sensor: recorded mean / GT / |error| as 3 columns. 4 rows x 8 sensors layout."""
    by_link: dict[int, list[tuple[int, str]]] = {}
    import re

    for n in recorded:
        m = re.match(r"link(\d+)_sensor_(\d+)", n)
        if m:
            by_link.setdefault(int(m.group(1)), []).append((int(m.group(2)), n))
    for link in by_link:
        by_link[link].sort()

    n_rows = 4  # 4 links x 3 cols per link visualization
    n_cols_per_sensor = 3
    n_sensors_max = max(len(v) for v in by_link.values())
    fig, axes = plt.subplots(
        len(by_link),
        n_sensors_max * n_cols_per_sensor,
        figsize=(n_sensors_max * n_cols_per_sensor * 1.5, len(by_link) * 1.7),
        squeeze=False,
    )
    vmax = max(
        max((v.max() for v in recorded.values()), default=4.0),
        max((g.max() for g in gt.values()), default=4.0),
    )

    for r, link in enumerate(sorted(by_link)):
        for c, (idx, name) in enumerate(by_link[link]):
            rec = recorded[name].mean(axis=0)
            g = gt.get(name)
            err = np.abs(rec - g) if g is not None else None

            ax_rec = axes[r, c * n_cols_per_sensor]
            ax_rec.imshow(rec, cmap="turbo_r", vmin=0.05, vmax=4.0, interpolation="nearest")
            ax_rec.set_title(f"{name}\nrec mean", fontsize=6)
            ax_rec.set_xticks([])
            ax_rec.set_yticks([])

            ax_gt = axes[r, c * n_cols_per_sensor + 1]
            if g is not None:
                ax_gt.imshow(g, cmap="turbo_r", vmin=0.05, vmax=4.0, interpolation="nearest")
                ax_gt.set_title("GT", fontsize=6)
            else:
                ax_gt.set_title("no GT", fontsize=6)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])

            ax_err = axes[r, c * n_cols_per_sensor + 2]
            if err is not None:
                ax_err.imshow(err, cmap="hot", vmin=0.0, vmax=0.5, interpolation="nearest")
                ax_err.set_title(f"|err| max={err.max():.2f}", fontsize=6)
            ax_err.set_xticks([])
            ax_err.set_yticks([])

        # blank unused columns for shorter rows
        for c in range(len(by_link[link]), n_sensors_max):
            for cc in range(n_cols_per_sensor):
                axes[r, c * n_cols_per_sensor + cc].axis("off")

    fig.suptitle("Recorded mean (red=close) | GT 8x8 native | |error| (yellow=high)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--t", type=int, default=10)
    parser.add_argument("--traj", type=str, default="traj_0")
    args = parser.parse_args()

    out_dir = args.h5_path.parent / "analysis" / f"gt_compare_t{args.t}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading recorded state at t={args.t}...")
    rec = load_recorded(args.h5_path, args.t, args.traj)
    print(f"  qpos arm = {rec['qpos'].get('arm')}")
    print(f"  base_pose = {rec['base_pose']}")
    print(f"  policy_phase = {rec['policy_phase']}")
    print(f"  recorded {len(rec['proximity'])} proximity sensors")

    print("Building test scene + attaching franka_skin robot...")
    model = build_test_model(rec["base_pose"])
    data = mujoco.MjData(model)
    print(f"  model nbody={model.nbody}, ncam={model.ncam}, nu={model.nu}")
    set_qpos_to_recorded(model, data, rec["qpos"])

    print("Rendering GT at native 8x8 for each sensor...")
    sensor_names = sorted(rec["proximity"].keys())
    gt = render_gt_proximity(model, data, sensor_names)
    print(f"  rendered {len(gt)} GT depths")

    print("Computing per-sensor errors...")
    rows = []
    for name in sensor_names:
        rec_avg = rec["proximity"][name].mean(axis=0)  # mean over 4 substeps
        rec_clip = np.clip(rec_avg, 0.05, 4.0)
        if name in gt:
            gt_arr = np.clip(gt[name], 0.05, 4.0)
            err = np.abs(rec_clip - gt_arr)
            rows.append({
                "name": name,
                "rec_mean": float(rec_clip.mean()),
                "gt_mean": float(gt_arr.mean()),
                "mean_abs_err": float(err.mean()),
                "max_abs_err": float(err.max()),
                "frac_err_gt_10cm": float((err > 0.10).mean()),
                "frac_err_gt_30cm": float((err > 0.30).mean()),
            })
        else:
            rows.append({"name": name, "rec_mean": float(rec_clip.mean())})

    # Print summary table
    hdr = f"{'sensor':22s} {'rec mean':>9s} {'GT mean':>9s} {'mean err':>9s} {'max err':>9s} {'%>10cm':>8s} {'%>30cm':>8s}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "gt_mean" in r:
            print(
                f"{r['name']:22s} {r['rec_mean']:9.3f} {r['gt_mean']:9.3f} {r['mean_abs_err']:9.3f} "
                f"{r['max_abs_err']:9.3f} {r['frac_err_gt_10cm']*100:7.1f}% {r['frac_err_gt_30cm']*100:7.1f}%"
            )
        else:
            print(f"{r['name']:22s} {r['rec_mean']:9.3f}  (no GT)")

    # Aggregate
    full_rows = [r for r in rows if "gt_mean" in r]
    if full_rows:
        mean_err_all = np.mean([r["mean_abs_err"] for r in full_rows])
        med_err_all = np.median([r["mean_abs_err"] for r in full_rows])
        max_err_all = max(r["max_abs_err"] for r in full_rows)
        sensors_within_10cm = sum(r["frac_err_gt_10cm"] < 0.05 for r in full_rows)
        print()
        print(f"Aggregate: mean err = {mean_err_all:.3f} m, median = {med_err_all:.3f} m, max = {max_err_all:.3f} m")
        print(f"Sensors with <5% pixels above 10cm error: {sensors_within_10cm}/{len(full_rows)}")

    # Save side-by-side image
    plot_compare_grid(rec["proximity"], gt, out_dir / "grid.png")
    print(f"wrote {out_dir / 'grid.png'}")

    # Markdown summary
    md = ["# Proximity GT comparison"]
    md.append(f"- timestep: t={args.t}, policy_phase={rec['policy_phase']}")
    md.append(f"- {len(full_rows)}/{len(rows)} sensors compared")
    md.append("")
    md.append("| sensor | rec mean (m) | GT mean (m) | mean |err| (m) | max |err| (m) | frac >10cm err |")
    md.append("|--------|--------------|-------------|----------------|---------------|-----------------|")
    for r in rows:
        if "gt_mean" in r:
            md.append(
                f"| {r['name']} | {r['rec_mean']:.3f} | {r['gt_mean']:.3f} | "
                f"{r['mean_abs_err']:.3f} | {r['max_abs_err']:.3f} | {r['frac_err_gt_10cm']*100:.1f}% |"
            )
    md.append("")
    md.append(f"**Aggregate**: mean err = {mean_err_all:.3f} m, max = {max_err_all:.3f} m, sensors within 10cm tolerance (>=95% pixels): {sensors_within_10cm}/{len(full_rows)}")
    md.append("")
    md.append("Note: kitchen pickup objects (pepper shaker, etc.) are placed at runtime by the task sampler with random offsets. They are NOT in the static FloorPlan1_physics.xml. Sensors looking at placed objects will show large errors against this scene-only GT — that's expected. Sensors looking at static walls/cabinets/floor should match within ~5cm.")

    (out_dir / "summary.md").write_text("\n".join(md))
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
