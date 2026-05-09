"""Rigorous synthetic-scene proximity-sensor verification.

Two test scenes:
  EMPTY ROOM  - 4x4x3 m box, robot at center. Sweep many arm poses, render
                proximity depths at native 8x8, reconstruct world point cloud,
                compare to known wall planes.
  FLAT PLANE  - just a floor + the robot. Sweep poses; reconstructions should
                land on z=0.

For each scene the script produces:
  scene_*.png          - rendered MuJoCo scene from third-person and top-down
                         cameras (so the reader can see exactly what the robot
                         is looking at).
  pose_grid.png        - 4 representative poses x 3 panels each (rendered RGB,
                         a top-down 2D scatter of cloud points, residual hist).
  error_breakdown.png  - histograms of signed residuals to nearest wall, both
                         overall and per-wall, per-direction.
  sensor_coverage.png  - for each of the 29 sensors: bar chart of #pts that
                         landed on each wall vs robot self vs out-of-range.
  recon.html           - interactive Plotly scene: GT walls as semi-transparent
                         surfaces, robot kinematic chain as line segments,
                         scene-hit cloud colored by link, self-hit cloud grey.
  recon.ply / scene.ply - binary point clouds.
  report.md            - written analysis with numbers, breakdowns, conclusions.

A standalone helper converts any .ply -> Plotly HTML:
    python scripts/datagen/verify_synthetic_scenes.py --ply PATH.ply --html OUT.html
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import plotly.graph_objects as go

ROBOT_XML = "/home/jaydv/code/prox_learning/assets/robots/franka_skin/model.xml"
NAMESPACE = "robot_0/"
BASE_Z = 0.58

# Robot body chain we want to draw as a "skeleton" for context in the 3D viewer.
ROBOT_LINK_BODIES = [
    f"{NAMESPACE}fr3_link{i}" for i in range(8)
] + [f"{NAMESPACE}gripper/base"]


# ---------------------- scene construction ----------------------


def _write_temp(xml: str) -> str:
    fd = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    fd.write(xml)
    fd.close()
    return fd.name


def _attach_robot(scene_spec: mujoco.MjSpec, base_pos=(0, 0, 0), base_quat=(1, 0, 0, 0)):
    robot_spec = mujoco.MjSpec.from_file(ROBOT_XML)
    base = scene_spec.worldbody.add_body(
        name=f"{NAMESPACE}base", pos=list(base_pos), quat=list(base_quat), mocap=True
    )
    base.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.25, 0.25, BASE_Z / 2],
        pos=[0, 0, BASE_Z / 2],
        rgba=[0.4, 0.3, 0.2, 1],
        contype=0,
        conaffinity=0,
    )
    frame = base.add_frame(pos=[0, 0, BASE_Z])
    root = robot_spec.worldbody.first_body()
    frame.attach_body(root, NAMESPACE, "")


def build_empty_room(wall_dist: float = 4.0, ceiling_h: float = 3.0) -> mujoco.MjModel:
    half_h = ceiling_h / 2
    half_w = wall_dist + 1
    xml = f"""<mujoco model="empty_room">
        <compiler angle="radian"/>
        <option gravity="0 0 -9.8" integrator="implicitfast"/>
        <visual>
            <global offwidth="1280" offheight="960"/>
            <map znear="0.005" zfar="20"/>
            <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
            <quality shadowsize="2048"/>
        </visual>
        <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.4" rgb2="0.2 0.2 0.3" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
        </asset>
        <worldbody>
            <light pos="0 0 5" directional="true"/>
            <light pos="3 3 4" directional="false" diffuse="0.4 0.4 0.4"/>
            <geom name="floor"   type="plane" size="{half_w} {half_w} 0.05" pos="0 0 0" material="grid"/>
            <geom name="ceiling" type="plane" size="{half_w} {half_w} 0.05" pos="0 0 {ceiling_h}" zaxis="0 0 -1" rgba="0.5 0.5 0.6 1"/>
            <geom name="wall_n"  type="plane" size="{half_w} {half_h} 0.05" pos="0 {wall_dist} {half_h}" zaxis="0 -1 0" rgba="0.85 0.45 0.40 1"/>
            <geom name="wall_s"  type="plane" size="{half_w} {half_h} 0.05" pos="0 -{wall_dist} {half_h}" zaxis="0 1 0" rgba="0.40 0.80 0.45 1"/>
            <geom name="wall_e"  type="plane" size="{half_w} {half_h} 0.05" pos="{wall_dist} 0 {half_h}" zaxis="-1 0 0" rgba="0.45 0.45 0.85 1"/>
            <geom name="wall_w"  type="plane" size="{half_w} {half_h} 0.05" pos="-{wall_dist} 0 {half_h}" zaxis="1 0 0" rgba="0.85 0.85 0.45 1"/>
        </worldbody>
    </mujoco>"""
    spec = mujoco.MjSpec.from_file(_write_temp(xml))
    _attach_robot(spec)
    return spec.compile()


def build_floor_only() -> mujoco.MjModel:
    xml = """<mujoco model="floor_only">
        <compiler angle="radian"/>
        <option gravity="0 0 -9.8" integrator="implicitfast"/>
        <visual>
            <global offwidth="1280" offheight="960"/>
            <map znear="0.005" zfar="20"/>
            <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
        </visual>
        <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.4" rgb2="0.2 0.2 0.3" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
        </asset>
        <worldbody>
            <light pos="0 0 5" directional="true"/>
            <geom name="floor" type="plane" size="6 6 0.05" pos="0 0 0" material="grid"/>
        </worldbody>
    </mujoco>"""
    spec = mujoco.MjSpec.from_file(_write_temp(xml))
    _attach_robot(spec)
    return spec.compile()


# ---------------------- rendering helpers ----------------------


def proximity_camera_names(model: mujoco.MjModel) -> list[str]:
    return [
        model.camera(i).name
        for i in range(model.ncam)
        if "_sensor_" in model.camera(i).name
    ]


def render_scene_rgb(model, data, width=1280, height=720,
                     azimuth=45.0, elevation=-20.0, distance=8.0, lookat=(0, 0, 1.0)):
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat = np.array(lookat, dtype=np.float64)
    renderer.update_scene(data, cam)
    img = renderer.render().copy()
    renderer.close()
    return img


def render_all_proximity_native_8x8(model, data) -> dict[str, np.ndarray]:
    renderer = mujoco.Renderer(model, height=8, width=8)
    renderer.enable_depth_rendering()
    # Hide group 2 (skin meshes) — sensors are positioned inside the skin volume,
    # so without this they would see their own skin at ~0 distance.
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0
    out = {}
    for cam in proximity_camera_names(model):
        renderer.update_scene(data, camera=cam, scene_option=opt)
        out[cam] = renderer.render().astype(np.float32, copy=True)
    renderer.close()
    return out


def reconstruct_world_points(model, data, depths,
                              fovy: float = 45.0, d_min: float = 0.05, d_max: float = 8.0):
    """Per-pixel back-projection in MuJoCo GL convention.

    Returns (points (N,3), per-point sensor index, sensor name list).
    """
    H = W = 8
    f = (H / 2) / np.tan(np.deg2rad(fovy / 2))
    cx = cy = (H - 1) / 2.0
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    pts: list[np.ndarray] = []
    sidx: list[np.ndarray] = []
    cam_names = sorted(depths.keys())
    for s_idx, cam in enumerate(cam_names):
        depth = depths[cam]
        cam_id = model.camera(cam).id
        cam2world = np.eye(4)
        cam2world[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3)
        cam2world[:3, 3] = data.cam_xpos[cam_id]
        mask = (depth >= d_min) & (depth <= d_max)
        if not mask.any():
            continue
        d = depth[mask].astype(np.float64)
        u = u_grid[mask].astype(np.float64)
        v = v_grid[mask].astype(np.float64)
        x_c = (u - cx) * d / f
        y_c = -(v - cy) * d / f
        z_c = -d
        ones = np.ones_like(d)
        p_cam = np.stack([x_c, y_c, z_c, ones], axis=1)
        p_world = (cam2world @ p_cam.T).T
        pts.append(p_world[:, :3])
        sidx.append(np.full(len(d), s_idx))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int), cam_names
    return np.concatenate(pts), np.concatenate(sidx), cam_names


def set_arm_qpos(model, data, q7):
    for i, val in enumerate(q7):
        adr = model.joint(f"{NAMESPACE}fr3_joint{i+1}").qposadr[0]
        data.qpos[adr] = float(val)
    mujoco.mj_forward(model, data)


def link_color(name: str) -> str:
    m = re.search(r"link(\d+)_sensor_", name)
    if not m:
        return "#888888"
    palette = {2: "#e63946", 3: "#2a9d8f", 5: "#264653", 6: "#f4a261"}
    return palette.get(int(m.group(1)), "#888888")


def get_robot_skeleton(model, data):
    """Return ordered list of (body_name, world_pos) for the robot's link chain."""
    out = []
    for name in ROBOT_LINK_BODIES:
        try:
            bid = model.body(name).id
        except KeyError:
            continue
        out.append((name, data.xpos[bid].copy()))
    return out


# ---------------------- I/O ----------------------


def write_ply(pts: np.ndarray, out_path: Path):
    pts32 = pts.astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts32)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts32.tobytes())


def read_ply(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        binary = False
        n_vertex = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("format binary"):
                binary = True
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            if line == "end_header":
                break
        if binary:
            arr = np.fromfile(f, dtype="<f4", count=n_vertex * 3).reshape(-1, 3)
        else:
            arr = np.loadtxt(f, dtype=np.float32, max_rows=n_vertex)[:, :3]
    return arr


# ---------------------- Plotly composition ----------------------


def _wall_mesh(corners, color="rgba(150,150,150,0.18)", name="wall"):
    """Two-triangle Mesh3d for a quad with corners CCW."""
    x, y, z = zip(*corners)
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color,
        opacity=0.18,
        name=name,
        showlegend=True,
        hoverinfo="name",
    )


def empty_room_gt_meshes(wall_dist, ceiling_h):
    walls = []
    # floor (z=0)
    walls.append(_wall_mesh(
        [(-wall_dist, -wall_dist, 0), (wall_dist, -wall_dist, 0),
         (wall_dist, wall_dist, 0), (-wall_dist, wall_dist, 0)],
        color="rgba(120,120,140,0.30)", name="GT floor",
    ))
    # ceiling (z=ceiling_h)
    walls.append(_wall_mesh(
        [(-wall_dist, -wall_dist, ceiling_h), (wall_dist, -wall_dist, ceiling_h),
         (wall_dist, wall_dist, ceiling_h), (-wall_dist, wall_dist, ceiling_h)],
        color="rgba(160,160,180,0.20)", name="GT ceiling",
    ))
    # north wall (y=+wall_dist)
    walls.append(_wall_mesh(
        [(-wall_dist, wall_dist, 0), (wall_dist, wall_dist, 0),
         (wall_dist, wall_dist, ceiling_h), (-wall_dist, wall_dist, ceiling_h)],
        color="rgba(217,114,102,0.30)", name="GT wall_n",
    ))
    # south wall
    walls.append(_wall_mesh(
        [(-wall_dist, -wall_dist, 0), (wall_dist, -wall_dist, 0),
         (wall_dist, -wall_dist, ceiling_h), (-wall_dist, -wall_dist, ceiling_h)],
        color="rgba(102,204,115,0.30)", name="GT wall_s",
    ))
    # east wall (x=+wall_dist)
    walls.append(_wall_mesh(
        [(wall_dist, -wall_dist, 0), (wall_dist, wall_dist, 0),
         (wall_dist, wall_dist, ceiling_h), (wall_dist, -wall_dist, ceiling_h)],
        color="rgba(115,115,217,0.30)", name="GT wall_e",
    ))
    # west wall
    walls.append(_wall_mesh(
        [(-wall_dist, -wall_dist, 0), (-wall_dist, wall_dist, 0),
         (-wall_dist, wall_dist, ceiling_h), (-wall_dist, -wall_dist, ceiling_h)],
        color="rgba(217,217,115,0.30)", name="GT wall_w",
    ))
    return walls


def robot_skeleton_trace(skel, name="robot links"):
    xs, ys, zs = [], [], []
    for _, p in skel:
        xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines+markers",
        line=dict(color="black", width=6),
        marker=dict(color="black", size=4),
        name=name,
    )


def write_recon_html(
    out_path: Path,
    scene_pts: np.ndarray,
    scene_sidx: np.ndarray,
    robot_pts: np.ndarray | None,
    sensor_names: list[str],
    gt_meshes: list,
    robot_skel,
    title: str,
):
    traces: list = list(gt_meshes)

    # Group scene points by link
    link_groups: dict[int, list[int]] = {}
    for i, name in enumerate(sensor_names):
        m = re.search(r"link(\d+)", name)
        if m:
            link_groups.setdefault(int(m.group(1)), []).append(i)
    for link in sorted(link_groups):
        mask = np.isin(scene_sidx, link_groups[link])
        sel = scene_pts[mask]
        if len(sel) == 0:
            continue
        traces.append(go.Scatter3d(
            x=sel[:, 0], y=sel[:, 1], z=sel[:, 2],
            mode="markers",
            marker=dict(size=1.6, color=link_color(f"link{link}_sensor_0"), opacity=0.75),
            name=f"link{link} ({len(sel):,} pts)",
        ))

    if robot_pts is not None and len(robot_pts):
        traces.append(go.Scatter3d(
            x=robot_pts[:, 0], y=robot_pts[:, 1], z=robot_pts[:, 2],
            mode="markers",
            marker=dict(size=1.0, color="lightgray", opacity=0.35),
            name=f"self-hit ({len(robot_pts):,})",
        ))

    if robot_skel:
        traces.append(robot_skeleton_trace(robot_skel))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def ply_to_html(ply_path: Path, out_html: Path, max_points: int = 200_000):
    pts = read_ply(ply_path)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=2, color=pts[:, 2], colorscale="Turbo_r", opacity=0.7),
    )])
    fig.update_layout(title=f"{ply_path.name} ({len(pts):,} pts)",
                      scene=dict(aspectmode="data"))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"wrote {out_html}")


# ---------------------- per-pose breakdown ----------------------


def make_pose_grid_png(
    model, data, poses, out_path, wall_dist, ceiling_h,
    pose_indices, fovy=45.0,
):
    """For each chosen pose, show: rendered scene RGB | top-down 2D scatter of cloud | residual hist."""
    n = len(pose_indices)
    fig = plt.figure(figsize=(15, 4 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.4, 1.0, 1.0])
    plane_normals = np.array([[0, 0, 1], [0, 0, -1], [0, -1, 0],
                              [0, 1, 0], [-1, 0, 0], [1, 0, 0]])
    plane_offsets = np.array([0.0, -ceiling_h, -wall_dist, -wall_dist, -wall_dist, -wall_dist])

    for row, pi in enumerate(pose_indices):
        name, q = poses[pi]
        set_arm_qpos(model, data, q)
        rgb = render_scene_rgb(model, data)
        depths = render_all_proximity_native_8x8(model, data)
        pts, sidx, sensor_names = reconstruct_world_points(
            model, data, depths, d_max=wall_dist * np.sqrt(3)
        )

        ax_rgb = fig.add_subplot(gs[row, 0])
        ax_rgb.imshow(rgb)
        ax_rgb.set_title(f"pose {pi}: {name}", fontsize=9)
        ax_rgb.axis("off")

        ax_top = fig.add_subplot(gs[row, 1])
        # gt walls outline
        ax_top.plot([-wall_dist, wall_dist, wall_dist, -wall_dist, -wall_dist],
                    [-wall_dist, -wall_dist, wall_dist, wall_dist, -wall_dist],
                    color="black", lw=1)
        # cloud top-down
        if len(pts):
            for s_idx, cam in enumerate(sensor_names):
                m = sidx == s_idx
                if m.any():
                    ax_top.scatter(pts[m, 0], pts[m, 1], s=4, c=link_color(cam), alpha=0.6)
            # robot base
            ax_top.plot(0, 0, "k+", ms=10)
        ax_top.set_xlim(-wall_dist - 0.5, wall_dist + 0.5)
        ax_top.set_ylim(-wall_dist - 0.5, wall_dist + 0.5)
        ax_top.set_aspect("equal")
        ax_top.set_title(f"top-down cloud ({len(pts):,} pts)", fontsize=9)
        ax_top.set_xlabel("x")
        ax_top.set_ylabel("y")
        ax_top.grid(alpha=0.3)

        ax_hist = fig.add_subplot(gs[row, 2])
        if len(pts):
            d = pts @ plane_normals.T + plane_offsets
            nearest = np.abs(d).min(axis=1)
            ax_hist.hist(nearest * 1000, bins=60, color="#264653")
            ax_hist.axvline(50, color="red", ls="--", lw=1, label="5cm")
            ax_hist.set_xlabel("|dist to nearest wall plane| (mm)")
            ax_hist.set_ylabel("count")
            ax_hist.set_xlim(0, 1000)
            ax_hist.legend(fontsize=8)
            within = (nearest <= 0.05).mean() * 100
            ax_hist.set_title(f"residual hist; {within:.0f}% within 5cm", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------- empty-room test ----------------------


def empty_room_test(out_dir: Path, wall_dist: float = 4.0, ceiling_h: float = 3.0):
    print(f"\n=== EMPTY ROOM TEST (walls at +/-{wall_dist} m, ceiling {ceiling_h} m) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_empty_room(wall_dist=wall_dist, ceiling_h=ceiling_h)
    data = mujoco.MjData(model)
    print(f"  scene: nbody={model.nbody}, ncam={model.ncam}, "
          f"proximity cams={len(proximity_camera_names(model))}")

    poses = []
    for j1 in np.linspace(-np.pi, np.pi, 12, endpoint=False):
        for j2 in (-1.2, -0.7, -0.3):
            for j4 in (-2.3, -1.4, -0.5):
                poses.append((f"j1={j1:+.2f},j2={j2:+.1f},j4={j4:+.1f}",
                              [j1, j2, 0, j4, 0, 1.57, 0]))
    print(f"  generated {len(poses)} poses")

    # 1) Render the scene from a couple of viewpoints (with robot in home pose so the reader
    #    can see exactly what the synthetic environment looks like).
    set_arm_qpos(model, data, [0, -0.7853, 0, -2.35619, 0, 1.57079, 0])
    rgb_3p = render_scene_rgb(model, data, azimuth=45, elevation=-15, distance=9, lookat=(0, 0, 1.2))
    rgb_topdown = render_scene_rgb(model, data, azimuth=0, elevation=-89, distance=9, lookat=(0, 0, 1.0))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(rgb_3p); ax[0].set_title("Empty room — third-person view (home pose)"); ax[0].axis("off")
    ax[1].imshow(rgb_topdown); ax[1].set_title("Empty room — top-down view"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(out_dir / "scene_views.png", dpi=130); plt.close(fig)

    # 2) Sweep poses; collect points + per-pose info we need later.
    all_pts: list[np.ndarray] = []
    all_sidx: list[np.ndarray] = []
    all_pose_idx: list[np.ndarray] = []
    cam_names = None
    skel_at_home = get_robot_skeleton(model, data)
    for pi, (name, q) in enumerate(poses):
        set_arm_qpos(model, data, q)
        depths = render_all_proximity_native_8x8(model, data)
        if cam_names is None:
            cam_names = sorted(depths.keys())
        # Allow up to wall_dist * sqrt(3) so we don't pre-filter diagonal hits before classification.
        pts, sidx, _ = reconstruct_world_points(
            model, data, depths, d_max=wall_dist * np.sqrt(3)
        )
        all_pts.append(pts); all_sidx.append(sidx)
        all_pose_idx.append(np.full(len(pts), pi))

    pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    sidx = np.concatenate(all_sidx) if all_sidx else np.zeros((0,), dtype=int)
    pose_idx = np.concatenate(all_pose_idx) if all_pose_idx else np.zeros((0,), dtype=int)
    print(f"  reconstructed {len(pts):,} world-frame points across {len(poses)} poses")

    # 3) Classify points: scene-hit (near a wall plane) vs robot-self-hit (near robot links) vs other.
    plane_normals = np.array([[0, 0, 1], [0, 0, -1], [0, -1, 0],
                              [0, 1, 0], [-1, 0, 0], [1, 0, 0]])
    wall_names = ["floor", "ceiling", "wall_n", "wall_s", "wall_e", "wall_w"]
    plane_offsets = np.array([0.0, -ceiling_h, -wall_dist, -wall_dist, -wall_dist, -wall_dist])
    dist_signed = pts @ plane_normals.T + plane_offsets if len(pts) else np.zeros((0, 6))
    dist_abs = np.abs(dist_signed)
    nearest_wall_idx = dist_abs.argmin(axis=1) if len(pts) else np.zeros(0, dtype=int)
    nearest_wall_dist = dist_abs.min(axis=1) if len(pts) else np.zeros(0)

    # Self-hit detection: |residual to nearest wall| > THRESH AND inside the room means
    # the point is in mid-air near the robot (robot self-hit).
    SELF_THRESH = 0.10  # 10 cm tolerance for "on a wall plane"
    in_room = (
        (np.abs(pts[:, 0]) <= wall_dist + 0.1)
        & (np.abs(pts[:, 1]) <= wall_dist + 0.1)
        & (pts[:, 2] >= -0.1)
        & (pts[:, 2] <= ceiling_h + 0.1)
    ) if len(pts) else np.zeros(0, dtype=bool)

    on_wall = nearest_wall_dist <= SELF_THRESH
    # Self-hit: in the room but not on any wall plane (i.e., free-floating in the middle)
    is_self = in_room & ~on_wall
    is_scene = on_wall
    is_other = ~(is_self | is_scene)  # out of room, etc.
    n_total = len(pts)
    print(f"  classification (10cm threshold for 'on wall plane'):")
    print(f"    on a wall plane (scene hits): {is_scene.sum():,} ({is_scene.mean()*100 if n_total else 0:.1f}%)")
    print(f"    free-floating in room (robot self-hits): {is_self.sum():,} ({is_self.mean()*100 if n_total else 0:.1f}%)")
    print(f"    out-of-room / other: {is_other.sum():,}")

    # 4) Numerics: per-wall accuracy on points classified as wall hits, signed bias.
    print("\n  per-wall accuracy (only points within 10cm of that wall plane):")
    per_wall_stats = {}
    for w_idx, w_name in enumerate(wall_names):
        m = (nearest_wall_idx == w_idx) & is_scene
        n_w = m.sum()
        if n_w == 0:
            per_wall_stats[w_name] = None
            continue
        signed = dist_signed[m, w_idx]
        absd = dist_abs[m, w_idx]
        per_wall_stats[w_name] = dict(
            n=int(n_w),
            mean_abs_mm=float(absd.mean() * 1000),
            median_abs_mm=float(np.median(absd) * 1000),
            p99_abs_mm=float(np.percentile(absd, 99) * 1000),
            mean_signed_mm=float(signed.mean() * 1000),
            std_signed_mm=float(signed.std() * 1000),
        )
        s = per_wall_stats[w_name]
        print(f"    {w_name:8s}: n={s['n']:6,}  "
              f"mean|err|={s['mean_abs_mm']:6.2f} mm  "
              f"p99={s['p99_abs_mm']:6.2f} mm  "
              f"signed bias={s['mean_signed_mm']:+6.2f} mm  "
              f"std(signed)={s['std_signed_mm']:6.2f} mm")

    # 5) Per-sensor coverage: how many of each sensor's points hit each wall vs self vs miss.
    coverage = {n: dict(scene=0, self=0, other=0, by_wall={w: 0 for w in wall_names})
                for n in cam_names}
    for s_idx in range(len(cam_names)):
        m = sidx == s_idx
        coverage[cam_names[s_idx]]["scene"] = int((m & is_scene).sum())
        coverage[cam_names[s_idx]]["self"] = int((m & is_self).sum())
        coverage[cam_names[s_idx]]["other"] = int((m & is_other).sum())
        for w_idx, w_name in enumerate(wall_names):
            coverage[cam_names[s_idx]]["by_wall"][w_name] = int(((nearest_wall_idx == w_idx) & m & is_scene).sum())

    # 6) Plots
    # 6a. Error breakdown
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    if is_scene.sum():
        # all signed residuals stacked by wall
        for w_idx, w_name in enumerate(wall_names):
            m = (nearest_wall_idx == w_idx) & is_scene
            if m.any():
                axes[0, 0].hist(dist_signed[m, w_idx] * 1000, bins=80, alpha=0.6, label=w_name)
        axes[0, 0].set_title("Signed residual to nearest wall, by wall")
        axes[0, 0].set_xlabel("signed residual (mm)")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(nearest_wall_dist[is_scene] * 1000, bins=80, color="#264653")
        axes[0, 1].set_title(f"|residual| of {is_scene.sum():,} on-wall points")
        axes[0, 1].set_xlabel("|residual| (mm)")
        axes[0, 1].set_ylabel("count")
        axes[0, 1].grid(alpha=0.3)

        axes[0, 2].hist(nearest_wall_dist * 1000, bins=80, color="#a64942")
        axes[0, 2].set_title(f"|residual| of all {n_total:,} points (self-hits visible as long tail)")
        axes[0, 2].set_xlabel("|residual| (mm)")
        axes[0, 2].set_ylabel("count")
        axes[0, 2].set_yscale("log")
        axes[0, 2].grid(alpha=0.3)

        # Per-wall mean signed bias
        wnames_with_data = [w for w in wall_names if per_wall_stats.get(w)]
        biases = [per_wall_stats[w]["mean_signed_mm"] for w in wnames_with_data]
        axes[1, 0].bar(wnames_with_data, biases, color="#e76f51")
        axes[1, 0].axhline(0, color="black", lw=1)
        axes[1, 0].set_title("Per-wall signed bias (mm)  -- nonzero = systematic offset")
        axes[1, 0].set_ylabel("mean signed residual (mm)")
        axes[1, 0].grid(alpha=0.3, axis="y")

        # Per-wall noise (std)
        stds = [per_wall_stats[w]["std_signed_mm"] for w in wnames_with_data]
        axes[1, 1].bar(wnames_with_data, stds, color="#2a9d8f")
        axes[1, 1].set_title("Per-wall noise (std of signed residual, mm)")
        axes[1, 1].set_ylabel("std (mm)")
        axes[1, 1].grid(alpha=0.3, axis="y")

        # Per-wall counts
        counts = [per_wall_stats[w]["n"] for w in wnames_with_data]
        axes[1, 2].bar(wnames_with_data, counts, color="#264653")
        axes[1, 2].set_title("Per-wall hit counts (all poses combined)")
        axes[1, 2].set_ylabel("# points")
        axes[1, 2].grid(alpha=0.3, axis="y")

    fig.suptitle("Empty-room error breakdown", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "error_breakdown.png", dpi=130)
    plt.close(fig)

    # 6b. Sensor coverage heatmap
    cov_matrix = np.zeros((len(cam_names), len(wall_names) + 2))
    for i, name in enumerate(cam_names):
        for j, w in enumerate(wall_names):
            cov_matrix[i, j] = coverage[name]["by_wall"][w]
        cov_matrix[i, len(wall_names)] = coverage[name]["self"]
        cov_matrix[i, len(wall_names) + 1] = coverage[name]["other"]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cov_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(wall_names) + 2))
    ax.set_xticklabels(wall_names + ["self-hit", "other"], rotation=30, ha="right")
    ax.set_yticks(range(len(cam_names)))
    ax.set_yticklabels(cam_names, fontsize=7)
    ax.set_title("Per-sensor hit counts by category (over all poses)")
    fig.colorbar(im, ax=ax, label="# pts")
    fig.tight_layout()
    fig.savefig(out_dir / "sensor_coverage.png", dpi=130)
    plt.close(fig)

    # 6c. Per-pose grid (4 representative poses spread across the sweep)
    pi_picks = [0, len(poses) // 3, 2 * len(poses) // 3, len(poses) - 1]
    make_pose_grid_png(
        model, data, poses, out_dir / "pose_grid.png",
        wall_dist=wall_dist, ceiling_h=ceiling_h, pose_indices=pi_picks,
    )

    # 7) Save PLYs
    write_ply(pts[is_scene], out_dir / "scene.ply")
    write_ply(pts, out_dir / "full.ply")

    # 8) Plotly HTML with GT walls + robot skeleton
    set_arm_qpos(model, data, [0, -0.7853, 0, -2.35619, 0, 1.57079, 0])
    skel = get_robot_skeleton(model, data)
    write_recon_html(
        out_dir / "recon.html",
        scene_pts=pts[is_scene], scene_sidx=sidx[is_scene],
        robot_pts=pts[is_self], sensor_names=cam_names,
        gt_meshes=empty_room_gt_meshes(wall_dist, ceiling_h),
        robot_skel=skel,
        title=(f"Empty-room reconstruction. {is_scene.sum():,} on-wall pts (within 10cm), "
               f"{is_self.sum():,} robot self-hits. "
               f"Mean |err|={(nearest_wall_dist[is_scene].mean()*1000 if is_scene.any() else 0):.1f} mm"),
    )

    # 9) Markdown report
    md = ["# Empty-room verification", ""]
    md.append(f"- Scene: empty box, walls at +/-{wall_dist} m, ceiling at {ceiling_h} m")
    md.append(f"- Robot: franka_skin (FR3 + Robotiq + 29 SPAD-style proximity sensors)")
    md.append(f"- Poses: {len(poses)} arm configurations sweeping joint 1 (yaw, 12 angles), joint 2 (shoulder, 3 values), joint 4 (elbow, 3 values)")
    md.append(f"- Total reconstructed points: **{n_total:,}**")
    md.append("")
    md.append("## Classification")
    md.append("| category | count | %  |")
    md.append("|---|---|---|")
    md.append(f"| on a wall plane (within 10 cm) | **{is_scene.sum():,}** | {is_scene.mean()*100 if n_total else 0:.1f}% |")
    md.append(f"| robot self-hits (in-room, off-wall) | {is_self.sum():,} | {is_self.mean()*100 if n_total else 0:.1f}% |")
    md.append(f"| out-of-room / other | {is_other.sum():,} | {is_other.mean()*100 if n_total else 0:.1f}% |")
    md.append("")
    md.append("## Per-wall accuracy (on-wall points only)")
    md.append("| wall | n | mean&nbsp;\\|err\\| | p99 | signed bias | std |")
    md.append("|---|---|---|---|---|---|")
    for w in wall_names:
        s = per_wall_stats.get(w)
        if not s:
            continue
        md.append(f"| {w} | {s['n']:,} | {s['mean_abs_mm']:.2f} mm | {s['p99_abs_mm']:.2f} mm | {s['mean_signed_mm']:+.2f} mm | {s['std_signed_mm']:.2f} mm |")
    md.append("")
    md.append("- **Mean |err|** = average distance of an on-wall point to the wall plane (small = good).")
    md.append("- **Signed bias** = average of (point - plane) along the plane normal. "
              "Nonzero indicates a systematic offset of reconstruction relative to the geometry "
              "(e.g. depth values systematically too long/short).")
    md.append("- **std(signed)** = noise level (sensor + sub-pixel quantization + back-projection error).")
    md.append("")
    md.append("## Per-sensor coverage")
    md.append("See `sensor_coverage.png`. Each row = one of the 29 SPAD cameras; each column = where its returns landed (which wall, robot self, out-of-range).")
    md.append("Sensors that consistently hit the robot itself are persistent self-hit candidates and should be masked when training a downstream proximity model.")
    md.append("")
    md.append("## Per-pose breakdown")
    md.append("`pose_grid.png` shows 4 representative poses. For each: (left) the rendered MuJoCo scene from a third-person camera so you can see what the robot is looking at, (middle) a top-down 2D scatter of the reconstructed cloud (robot at the cross), (right) histogram of |dist to nearest wall|.")
    md.append("")
    md.append("## Outputs")
    md.append(f"- `scene_views.png` — third-person + top-down rendered MuJoCo scene")
    md.append(f"- `pose_grid.png` — 4 representative poses with rendered RGB + cloud + residual histogram")
    md.append(f"- `error_breakdown.png` — error histograms (signed + abs), per-wall bias/std/count bars")
    md.append(f"- `sensor_coverage.png` — per-sensor coverage heatmap")
    md.append(f"- `recon.html` — interactive Plotly with GT wall surfaces (semi-transparent), robot skeleton (black line), reconstructed cloud colored by link, robot self-hits in light gray")
    md.append(f"- `scene.ply` / `full.ply` — binary point clouds (load in MeshLab/CloudCompare)")
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"  wrote: scene_views.png, pose_grid.png, error_breakdown.png, sensor_coverage.png, recon.html, scene.ply, full.ply, report.md")
    return per_wall_stats


# ---------------------- flat-plane test ----------------------


def flat_plane_test(out_dir: Path):
    print("\n=== FLAT-PLANE TEST (single floor, varied robot poses) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_floor_only()
    data = mujoco.MjData(model)

    poses = []
    for j1 in np.linspace(-np.pi, np.pi, 12, endpoint=False):
        for j2 in (-1.2, -0.7, -0.3):
            poses.append((f"j1={j1:+.2f},j2={j2:+.1f}", [j1, j2, 0, -1.4, 0, 1.2, 0]))
    print(f"  {len(poses)} poses")

    # render scene at home for context
    set_arm_qpos(model, data, [0, -0.7853, 0, -2.35619, 0, 1.57079, 0])
    rgb_3p = render_scene_rgb(model, data, azimuth=45, elevation=-15, distance=4, lookat=(0, 0, 0.8))
    rgb_topdown = render_scene_rgb(model, data, azimuth=0, elevation=-89, distance=4, lookat=(0, 0, 0))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(rgb_3p); ax[0].set_title("Floor-only — third-person view (home pose)"); ax[0].axis("off")
    ax[1].imshow(rgb_topdown); ax[1].set_title("Floor-only — top-down view"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(out_dir / "scene_views.png", dpi=130); plt.close(fig)

    all_pts = []
    all_sidx = []
    cam_names = None
    for name, q in poses:
        set_arm_qpos(model, data, q)
        depths = render_all_proximity_native_8x8(model, data)
        if cam_names is None:
            cam_names = sorted(depths.keys())
        pts, sidx, _ = reconstruct_world_points(model, data, depths, d_max=8.0)
        all_pts.append(pts); all_sidx.append(sidx)

    pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    sidx = np.concatenate(all_sidx) if all_sidx else np.zeros((0,), dtype=int)

    # Classify: on the floor plane (|z| < 5cm) vs robot self (|z| > 5cm and within ~1.5m of base)
    on_floor = np.abs(pts[:, 2]) <= 0.05 if len(pts) else np.zeros(0, dtype=bool)
    is_self = ~on_floor & (np.linalg.norm(pts - np.array([0, 0, 0.7]), axis=1) < 1.5) if len(pts) else np.zeros(0, dtype=bool)
    is_other = ~(on_floor | is_self)
    print(f"  total: {len(pts):,}; on floor (|z|<=5cm): {on_floor.sum():,}; "
          f"robot self: {is_self.sum():,}; other: {is_other.sum():,}")

    if on_floor.any():
        signed = pts[on_floor, 2]
        abs_e = np.abs(signed)
        print(f"  floor accuracy:")
        print(f"    mean |z|: {abs_e.mean()*1000:.2f} mm  (target: ~0)")
        print(f"    p99 |z|:  {np.percentile(abs_e, 99)*1000:.2f} mm")
        print(f"    signed bias: {signed.mean()*1000:+.2f} mm  (nonzero = systematic z offset)")
        print(f"    std(signed): {signed.std()*1000:.2f} mm")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    if on_floor.any():
        axes[0].hist(pts[on_floor, 2] * 1000, bins=80, color="#264653")
        axes[0].axvline(0, color="red", lw=1)
        axes[0].set_xlabel("z (mm)  [GT = 0]")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"Signed z residual on floor pts (n={on_floor.sum():,})")
        axes[0].grid(alpha=0.3)

        axes[1].hist(np.abs(pts[on_floor, 2]) * 1000, bins=80, color="#2a9d8f")
        axes[1].set_xlabel("|z| (mm)")
        axes[1].set_ylabel("count")
        axes[1].set_title("Absolute z residual on floor pts")
        axes[1].grid(alpha=0.3)

        # Top-down scatter
        axes[2].scatter(pts[on_floor, 0], pts[on_floor, 1], s=2, c="#264653", alpha=0.4, label="on-floor")
        if is_self.any():
            axes[2].scatter(pts[is_self, 0], pts[is_self, 1], s=2, c="lightgray", alpha=0.3, label="self-hit")
        axes[2].plot(0, 0, "k+", ms=10, label="robot base")
        axes[2].set_aspect("equal")
        axes[2].set_xlabel("x (m)"); axes[2].set_ylabel("y (m)")
        axes[2].set_title("Top-down view of reconstruction")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "error_breakdown.png", dpi=130)
    plt.close(fig)

    write_ply(pts[on_floor], out_dir / "scene.ply")
    write_ply(pts, out_dir / "full.ply")

    # Plotly: floor as big translucent quad + robot skeleton + cloud
    set_arm_qpos(model, data, [0, -0.7853, 0, -2.35619, 0, 1.57079, 0])
    skel = get_robot_skeleton(model, data)
    sq = 4.0
    floor_mesh = _wall_mesh(
        [(-sq, -sq, 0), (sq, -sq, 0), (sq, sq, 0), (-sq, sq, 0)],
        color="rgba(120,120,140,0.30)", name="GT floor (z=0)",
    )

    write_recon_html(
        out_dir / "recon.html",
        scene_pts=pts[on_floor], scene_sidx=sidx[on_floor],
        robot_pts=pts[is_self], sensor_names=cam_names,
        gt_meshes=[floor_mesh], robot_skel=skel,
        title=(f"Flat-plane reconstruction. {on_floor.sum():,} on-floor pts. "
               f"Mean |z|={(np.abs(pts[on_floor, 2]).mean()*1000 if on_floor.any() else 0):.1f} mm, "
               f"signed bias={(pts[on_floor, 2].mean()*1000 if on_floor.any() else 0):+.1f} mm."),
    )

    # Markdown
    md = ["# Flat-plane verification", ""]
    md.append("- Scene: only a horizontal floor at z=0 + the robot")
    md.append(f"- Poses: {len(poses)} arm configurations (12 yaws x 3 shoulder reaches)")
    md.append(f"- Total reconstructed points: {len(pts):,}")
    md.append("")
    md.append("## Classification (5cm tolerance for 'on floor')")
    md.append(f"- on floor (|z| <= 5 cm): **{on_floor.sum():,}** ({on_floor.mean()*100 if len(pts) else 0:.1f}%)")
    md.append(f"- robot self-hit (|z|>5cm, within 1.5m of base): {is_self.sum():,}")
    md.append(f"- other: {is_other.sum():,}")
    md.append("")
    md.append("## Floor-plane accuracy (on-floor points only)")
    if on_floor.any():
        md.append(f"- Mean |z residual|: **{abs_e.mean()*1000:.2f} mm**")
        md.append(f"- p99 |z|: **{np.percentile(abs_e, 99)*1000:.2f} mm**")
        md.append(f"- Signed bias: **{signed.mean()*1000:+.2f} mm** "
                  f"(nonzero indicates a systematic offset of the reconstruction along the floor normal)")
        md.append(f"- std(signed): {signed.std()*1000:.2f} mm")
    md.append("")
    md.append("## Outputs")
    md.append("- `scene_views.png` — rendered MuJoCo scene (3rd-person + top-down)")
    md.append("- `error_breakdown.png` — signed and absolute residual histograms; top-down scatter of reconstruction")
    md.append("- `recon.html` — interactive Plotly with GT floor surface (semi-transparent), robot skeleton, on-floor pts colored by link, self-hits in light gray")
    md.append("- `scene.ply` / `full.ply` — binary point clouds")
    (out_dir / "report.md").write_text("\n".join(md))
    return abs_e if on_floor.any() else np.zeros(0)


# ---------------------- main ----------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/home/jaydv/code/prox_learning/synthetic_verify"))
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--html", type=Path, default=None)
    args = parser.parse_args()

    if args.ply:
        out_html = args.html or args.ply.with_suffix(".html")
        ply_to_html(args.ply, out_html)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    er_stats = empty_room_test(args.out / "empty_room")
    fp_resid = flat_plane_test(args.out / "flat_plane")

    # Top-level summary
    md = ["# Synthetic-scene proximity verification — summary", ""]
    md.append("## Empty room (walls 4 m, ceiling 3 m)")
    if er_stats:
        for w, s in er_stats.items():
            if not s:
                continue
            md.append(f"- **{w}**: n={s['n']:,}  mean|err|={s['mean_abs_mm']:.2f} mm  "
                      f"signed bias={s['mean_signed_mm']:+.2f} mm  std={s['std_signed_mm']:.2f} mm")
    md.append("")
    md.append("## Flat plane")
    if len(fp_resid):
        md.append(f"- mean |z|: {fp_resid.mean()*1000:.2f} mm")
        md.append(f"- p99 |z|: {np.percentile(fp_resid, 99)*1000:.2f} mm")
    md.append("")
    md.append("See `empty_room/report.md` and `flat_plane/report.md` for full breakdowns.")
    (args.out / "summary.md").write_text("\n".join(md))
    print(f"\n=== summary at {args.out / 'summary.md'} ===")


if __name__ == "__main__":
    main()
