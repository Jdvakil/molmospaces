"""``pc_occ_desk``: a home-office desk. The robot stands at the left end of the desk and reaches
behind a large monitor for a cup, then sets it on a cork coaster nearer the desk end. A desk lamp
(real Objaverse meshes) is either a wall-mounted jointed lamp on the back wall (LEFT of the arm's
path) or an adjustable desk lamp standing behind the monitor (RIGHT of the path); its shade hangs
at the height of the SKIN-COVERED wrist knuckle / forearm. No camera can see it; only the arm's
proximity skin can tell where it is.

Why this suite exists (2026-10-15, ICLR rebuttal, "occluded natural items")
---------------------------------------------------------------------------
The earlier hidden-hazard suites hid artificial-looking fixtures (basket, valve box, drain pipe,
radio) behind trim. Here the hazard is an ordinary lamp where lamps are (on the wall over the back
of the desk / on the desk behind the screen), hidden by another ordinary object: the monitor (an
AI2-THOR flat screen scaled to a 43-inch desk monitor, :data:`MONITORS`). Monitor, desk, keyboard
and decor (plant, books, clock, mug) are the same in every episode of a cell.

Camera geometry (why the robot stands at the desk END, not in front of it)
-------------------------------------------------------------------------
exo_camera_1 is robot mounted (link0 + (-1.05, -0.55, 1.30), aimed at link0 + (0.55, 0, 0.45)). A
monitor can hide something from it only if it stands between that camera and the hazard WITHOUT
standing in the arm's way. With the robot in front of the desk the camera looks along the reach,
so any occluder of the work area also blocks the arm. Parked at the desk's left end (desk on its
right, the monitor facing the "user" position behind the robot's right shoulder, i.e. facing the
camera), the arm reaches to its right-front (lane yaw :data:`LANE_YAW_DEG`) behind the monitor's
left half while the camera looks at the screen; the screen reaches to ~4 cm above the desk and to
0.68 m above it, so both lamp positions are out of view, and the arm passes 15+ cm left of the
monitor's left edge.

wrist_camera looks down along the fingers (FOV 56.7 x 87 deg), mounted 7 cm to one side of the
fingers and tilted 19 deg back toward them, so it is blind on its mount side. The hand is turned
:data:`HAND_YAW_OFF` (40 deg) about the vertical so that blind side faces the right lamp's base;
the left lamp is entirely above the camera.

Hidden hazard (``pact_intrusion_lamp_left`` / ``_right``, scored by the evaluator as ``hazard_bar``)
-----------------------------------------------------------------------------------------------
One lamp per episode, LEFT or RIGHT of the lane (coin flip, drawn independently of the cup, the
coaster, the cell and the start pose); the other side's lamp is parked outside the room. Poses
(:data:`HAZ_NOMINAL` + :data:`HAZ_JIT`, relative to the cup) come from a design search over lamp
poses that keeps fr3_link7 / gripper >= 5.5 cm and the skinned links >= 0.6 cm from the lamp on the
expert path, makes both cameras blind to it, and maximises the fixed hand tilts that hit it, at
every jitter corner. Visual: the Objaverse mesh with plain colours (textures dropped: alpha
textures can nudge occluded pixels); collision: the asset's own convex collider pieces (clearances
by face planes, :func:`hazard_distance`).

Task + expert (scripted TCP segments, hand pointing down; = ``pc_sink_blind`` motion, rotated)
---------------------------------------------------------------------------------------------
Pick the cup (``cavity_obj_0/desk_cup``, 8 cm) standing 55.5-57.5 cm along the lane and set it on
the coaster (``place_receptacle_mat``) 40.5-42 cm along it (pc_shelf_pnp success rule). Expert:
slide along the lane at the carry height, untilted, to :data:`R_DIV` -> dwell :data:`DWELL_S` (the
hand arrives at step ~35-39 and tilts at ~58-62, so the policy's step-50 chunk query sees the skin
during the pause) -> TILT the hand :data:`TILT_DEG` about the lane axis, fingers leaning toward the
lamp side (the wrist knuckle / forearm swing to the OTHER side; the only plan difference) -> slide
on above the cup -> lower -> close -> lift -> carry back to the coaster -> lower -> open -> lift ->
untilt -> back out -> start pose.

Regenerate the scene files after editing the geometry::

    python -m molmo_spaces.tasks.pc_occ_desk --write-scenes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    GripperAction,
)
from molmo_spaces.tasks.pc_shelf import SCENE_DIR, _T, _jsonable
from molmo_spaces.tasks.pc_shelf_blind import (
    CLEAR_BODIES,
    PCShelfBlindTask,
    _BlindSeg,
    blind_tool_rotation,
    robot_vertex_sets,
)
from molmo_spaces.tasks.pc_sink_blind import (
    PCSinkBlindExpert,
    PCSinkBlindExpertConfig,
    PCSinkBlindSampler,
)
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)

PC_OCC_DESK_ENV_VERSION = "pc_occ_desk_v1"
N_CELLS = 8

# --------------------------------------------------------------------------------------------
# Geometry (metres, world frame; robot base at the origin facing +x)
# --------------------------------------------------------------------------------------------
H_BASE = 0.36            # robot deck height = link0 z
Z_D = 0.74               # desk top
DESK_T = 0.03
X_F = -0.22              # desk front edge (user side, faces -x)
X_B = 0.595              # desk back edge
X_W = 0.600              # back wall inner face
Y_L = -0.27              # desk left end (next to the robot)
Y_R = -1.62              # desk right end
LANE_YAW_DEG = -50.0     # reach direction (robot frame)
R_CUP = (0.555, 0.575)   # cup centre distance from the base along the lane
R_COASTER = (0.405, 0.420)
R_DIV = 0.44             # TCP distance of the common (untilted) partial reach where the tilt starts
R_PRE = 0.30             # pre-reach TCP distance (start of the slide)
DWELL_S = 1.3
TILT_DEG = 10.0
INS_CLEAR = 0.030        # TCP above the cup top while sliding / carrying
GRASP_BELOW_TOP = 0.030
COASTER_R = 0.050
COASTER_T = 0.004
WRIST_TOP = 0.344        # skinned wrist knuckle top above the TCP, hand straight down
HAND_TOP = 0.212         # fr3_link7 / gripper top above the TCP

STAND = R.from_euler("x", 90, degrees=True)   # Objaverse / THOR assets are Y-up


def lane_frame(yaw_deg: float = LANE_YAW_DEG) -> tuple[np.ndarray, np.ndarray]:
    a = np.deg2rad(yaw_deg)
    u = np.array([np.cos(a), np.sin(a)])
    return u, np.array([-u[1], u[0]])            # along, left normal


HAND_YAW_OFF = 40.0       # hand yaw about the vertical relative to the lane (deg)


def desk_tool_rotation(tilt_deg: float = 0.0, yaw_deg: float = LANE_YAW_DEG,
                       hand_off_deg: float | None = None) -> np.ndarray:
    """Hand pointing down, fingers closing along the lane (turned ``HAND_YAW_OFF`` about the
    vertical), wrist camera on the lane's right; tilted ``tilt_deg`` about the lane axis (positive:
    approach leans LEFT, wrist swings right)."""
    off = HAND_YAW_OFF if hand_off_deg is None else hand_off_deg
    t = np.deg2rad(tilt_deg)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    return (R.from_euler("z", yaw_deg, degrees=True).as_matrix() @ Rx
            @ R.from_euler("z", off, degrees=True).as_matrix() @ blind_tool_rotation(0.0))


@dataclass(frozen=True)
class DeskCell:
    """One home-office layout family."""

    desk: tuple[float, float, float]      # desk top colour
    wall: tuple[float, float, float]
    floor: tuple[float, float, float]
    monitor: int                          # index into MONITORS
    lamp: int                             # index into LAMPS
    cup: int                              # index into CUPS
    keyboard: int                         # index into KEYBOARDS


# --------------------------------------------------------------------------------------------
# Assets (Objaverse, downloaded on demand by install_uid)
# --------------------------------------------------------------------------------------------
# monitor uid, target screen width (m)
MONITORS = (   # AI2-THOR flat screens used as a 43-inch desk monitor (screen to ~4 cm above the desk;
    ("Television_2", 0.95),   # the shorter Television_17 / _22 let the exo camera see the lamps)
    ("Television_7", 0.95),
)
# The hidden lamp of each side: an Objaverse wall-mounted jointed lamp (left, on the back wall) or
# an adjustable desk lamp (right, standing behind the monitor). kind "wall": origin = foot of the
# wall rail, +x = horizontal direction from the rail to the shade; kind "desk": origin = base centre
# on the desk, +x = direction of the head overhang.
HAZARDS = {
    1: dict(kind="wall", uid="8a4c96ff49254278ac1273d768920a6d", scale=0.8, label="wall-mounted swing-arm lamp",
            rgba=(0.20, 0.36, 0.26, 1.0)),
    -1: dict(kind="desk", uid="4b9a5c4d5e20434793388325ec18ba57", scale=1.3, label="adjustable desk lamp",
             rgba=(0.12, 0.20, 0.17, 1.0)),
}
# Nominal pose relative to the cup (design search, scripts/audit_pc_occ_desk.py checks): desk lamp
# origin at (a along the lane, t beside it), yaw relative to the lane; wall lamp rail foot at
# dy (world y) from the cup, z above the desk top, yaw relative to the lane (x follows the wall).
HAZ_NOMINAL = {
    1: dict(dy=-0.086, z=0.16, yaw=170.0),
    -1: dict(a=0.157, t=-0.180, yaw=100.0),
}
HAZ_JIT = dict(a=0.006, t=0.006, dy=0.006, z=0.006, yaw=3.0)
KEYBOARDS = ("0c0edb114d324eab9106f331244621c5",)
# static desk decor, away from the work zone: uid, target size (longest horizontal side, m) or
# None (keep), x, y, yaw (deg). x / y in the robot frame.
DECOR = (
    ("Houseplant_5", None, 0.36, -1.42, 30.0),
    ("Book_3", None, -0.10, -1.36, 80.0),
    ("Book_7", None, -0.10, -1.36, 95.0),
    ("Alarm_Clock_3", None, 0.40, -1.13, 200.0),
    ("Mug_2", None, -0.08, -1.12, 150.0),
)
CUPS = (  # radius, height, body rgba, rim rgba, label
    (0.036, 0.080, (0.93, 0.92, 0.88), (0.80, 0.78, 0.74), "white ceramic cup"),
    (0.035, 0.080, (0.20, 0.36, 0.62), (0.15, 0.27, 0.48), "blue ceramic cup"),
    (0.036, 0.080, (0.72, 0.30, 0.22), (0.58, 0.22, 0.16), "terracotta cup"),
    (0.035, 0.080, (0.30, 0.52, 0.38), (0.22, 0.40, 0.28), "green ceramic cup"),
    (0.036, 0.080, (0.90, 0.78, 0.30), (0.75, 0.62, 0.20), "yellow ceramic cup"),
    (0.035, 0.080, (0.22, 0.22, 0.24), (0.12, 0.12, 0.13), "black ceramic cup"),
    (0.036, 0.080, (0.85, 0.62, 0.68), (0.70, 0.48, 0.55), "pink ceramic cup"),
    (0.035, 0.080, (0.62, 0.64, 0.66), (0.48, 0.50, 0.52), "grey ceramic cup"),
)

PC_OCC_DESK_CELLS: tuple[DeskCell, ...] = (
    DeskCell((0.72, 0.56, 0.38), (0.90, 0.88, 0.84), (0.60, 0.48, 0.36), 0, 0, 0, 0),
    DeskCell((0.92, 0.92, 0.90), (0.78, 0.84, 0.88), (0.42, 0.34, 0.27), 1, 0, 1, 0),
    DeskCell((0.36, 0.25, 0.17), (0.93, 0.91, 0.86), (0.66, 0.60, 0.52), 0, 0, 2, 0),
    DeskCell((0.80, 0.70, 0.52), (0.86, 0.90, 0.84), (0.50, 0.40, 0.30), 1, 0, 3, 0),
    DeskCell((0.25, 0.25, 0.27), (0.92, 0.90, 0.88), (0.58, 0.46, 0.34), 0, 0, 4, 0),
    DeskCell((0.62, 0.62, 0.60), (0.88, 0.84, 0.78), (0.40, 0.32, 0.25), 1, 0, 5, 0),
    DeskCell((0.78, 0.62, 0.42), (0.84, 0.86, 0.90), (0.62, 0.52, 0.40), 0, 0, 6, 0),
    DeskCell((0.52, 0.38, 0.26), (0.94, 0.93, 0.90), (0.46, 0.38, 0.30), 1, 0, 7, 0),
)

PC_OCC_DESK_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_occ_desk_c{k}.xml") for k in range(N_CELLS)
)


def _obja_spec(uid: str, scale: float, strip_textures: bool, colliders: bool, tint=None) -> MjSpec:
    """An Objaverse/THOR asset as a spec: joints and sites removed, everything scaled by ``scale``;
    visual geoms contactless; collider pieces kept (contype 8) or removed."""
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    extra = MjSpec.from_file(str(install_uid(uid)))
    for j in list(extra.joints):
        extra.delete(j)
    for s in list(extra.sites):
        extra.delete(s)
    alpha = {mt.name: float(mt.rgba[3]) for mt in extra.materials}
    for g in list(extra.geoms):
        is_col = (g.contype != 0 or g.conaffinity != 0) or "collider" in (g.name or "").lower() or \
            "collider" in (g.meshname or "").lower()
        if is_col and not colliders:
            extra.delete(g)
            continue
        if not is_col and alpha.get(g.material, 1.0) < 0.99:
            extra.delete(g)
    if strip_textures:
        for mt in extra.materials:
            mt.textures = [""] * len(mt.textures)
            if tint is not None:
                mt.rgba = list(tint)
            mt.reflectance = 0.0
    for g in extra.geoms:
        is_col = (g.contype != 0 or g.conaffinity != 0) or "collider" in (g.meshname or "").lower()
        if is_col:
            g.contype, g.conaffinity, g.group = 8, 15, 4
            g.friction = [0.6, 0.6, 0.001]
        else:
            g.contype, g.conaffinity, g.group = 0, 0, 0
        g.pos = [float(v) * scale for v in g.pos]
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            g.size = [float(v) * scale for v in g.size]
    for msh in extra.meshes:
        msh.scale = [float(v) * scale for v in msh.scale]
    for b in extra.bodies:
        if b is not extra.worldbody:
            b.pos = [float(v) * scale for v in b.pos]
    return extra


@lru_cache(maxsize=None)
def asset_points(uid: str, scale: float) -> dict[str, np.ndarray]:
    """Visual / collider vertices of the asset standing upright (STAND), in the attach frame."""
    wrap = MjSpec()
    holder = wrap.worldbody.add_body(name="holder", quat=list(STAND.as_quat(scalar_first=True)))
    extra = _obja_spec(uid, scale, True, True)
    holder.add_frame().attach_body(extra.worldbody.bodies[0], "a/", "")
    m = wrap.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    vis, col = [], []
    for g in range(m.ngeom):
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        P = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        (col if m.geom_contype[g] else vis).append(P)
    return dict(vis=np.concatenate(vis, 0), col=np.concatenate(col, 0) if col else np.zeros((0, 3)))


@lru_cache(maxsize=None)
def hazard_template(side: int) -> dict[str, Any]:
    """Attach frame of a side's lamp (see HAZARDS) plus its visual points in that frame."""
    hz = HAZARDS[side]
    V = asset_points(hz["uid"], hz["scale"])["vis"].copy()
    z0 = float(V[:, 2].min())
    zt = float(V[:, 2].max()) - z0
    foot = V[V[:, 2] < z0 + (0.02 if hz["kind"] == "desk" else 0.03)]
    bc = (foot[:, :2].min(0) + foot[:, :2].max(0)) / 2
    if hz["kind"] == "desk":
        head = V[V[:, 2] > z0 + 0.62 * zt]
    else:
        head = V[(V[:, 2] > z0 + 0.5 * zt) & (V[:, 2] < z0 + 0.75 * zt)]
    dvec = head[:, :2].mean(0) - bc
    Rz = R.from_euler("z", -float(np.arctan2(dvec[1], dvec[0])))
    off = np.array([bc[0], bc[1], z0])
    Q = Rz.apply(V - off)
    return dict(frame_quat=(Rz * STAND).as_quat(scalar_first=True).tolist(),
                frame_pos=(-Rz.apply(off)).tolist(), height=zt, pts=Q[:: max(1, len(Q) // 1500)],
                foot_r=float(np.linalg.norm(Q[Q[:, 2] < 0.02][:, :2], axis=1).max()) if hz["kind"] == "desk" else 0.0)


@lru_cache(maxsize=None)
def monitor_template(k: int) -> dict[str, Any]:
    """Monitor frame: origin = screen front-face centre at desk level, the screen faces -x."""
    uid, width = MONITORS[k]
    V0 = asset_points(uid, 1.0)["vis"]
    ext = np.ptp(V0, axis=0)
    scale = float(width / max(ext[0], ext[1]))
    V = asset_points(uid, scale)["vis"].copy()
    z0 = float(V[:, 2].min())
    zt = float(V[:, 2].max()) - z0
    ext = np.ptp(V, axis=0)
    thin = 0 if ext[0] < ext[1] else 1            # screen normal axis
    top = V[V[:, 2] > z0 + 0.75 * zt]
    mid = V[(V[:, 2] > z0 + 0.05 * zt) & (V[:, 2] < z0 + 0.35 * zt)]
    scr = float(np.median(top[:, thin]))
    neck = float(np.median(mid[:, thin]))
    back = 1.0 if neck > scr else -1.0            # +1: stand behind the screen along +thin
    # rotation so that "back" maps to +x
    if thin == 0:
        yaw0 = 0.0 if back > 0 else np.pi
    else:
        yaw0 = -np.pi / 2 if back > 0 else np.pi / 2
    Rz = R.from_euler("z", yaw0)
    c_other = float((V[:, 1 - thin].min() + V[:, 1 - thin].max()) / 2)
    Q0 = Rz.apply(V)
    top_q = Q0[Q0[:, 2] > z0 + 0.75 * zt]
    x_front = float(top_q[:, 0].min())
    yc = float((Q0[:, 1].min() + Q0[:, 1].max()) / 2)
    off = np.array([x_front, yc, z0])
    Q = Q0 - off
    scr_pts = Q[Q[:, 2] > 0.5 * zt]
    return dict(uid=uid, scale=scale, frame_quat=(Rz * STAND).as_quat(scalar_first=True).tolist(),
                frame_pos=(-off).tolist(), height=zt, width=float(np.ptp(Q[:, 1])),
                depth=float(np.ptp(Q[:, 0])), x_back=float(Q[:, 0].max()),
                screen_bottom=float(scr_pts[:, 2].min()), c_other=c_other)


@lru_cache(maxsize=None)
def rest_template(uid: str, size: float | None) -> dict[str, Any]:
    """Generic upright asset: frame that puts the footprint centre at the origin, bottom at z=0."""
    V0 = asset_points(uid, 1.0)["vis"]
    ext = np.ptp(V0, axis=0)
    scale = 1.0 if size is None else float(size / max(ext[0], ext[1]))
    V = asset_points(uid, scale)["vis"]
    lo, hi = V.min(0), V.max(0)
    off = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return dict(scale=scale, frame_quat=STAND.as_quat(scalar_first=True).tolist(), frame_pos=(-off).tolist(),
                half=((hi - lo) / 2).tolist())


# --------------------------------------------------------------------------------------------
# Scene XML (room, desk, static decor); meshes are attached at runtime by the sampler
# --------------------------------------------------------------------------------------------
def _f(v: float) -> str:
    return f"{float(v):.4f}"


def _box(name: str, lo, hi, material: str, extra: str = "") -> str:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    return (f'    <geom name="{name}" class="furn" type="box" pos="{_f(c[0])} {_f(c[1])} {_f(c[2])}" '
            f'size="{_f(h[0])} {_f(h[1])} {_f(h[2])}" material="{material}"{extra}/>\n')


def build_scene_xml(cell_index: int) -> str:
    c = PC_OCC_DESK_CELLS[cell_index]
    g: list[str] = []
    room_l, room_r = 1.60, -2.40          # side walls (y)
    rear_x, ceil = -2.10, 2.60
    xw = X_W
    g.append(_box("wall_back", (xw, room_r - 0.06, 0), (xw + 0.06, room_l + 0.06, ceil), "mat_wall"))
    g.append(_box("wall_rear", (rear_x - 0.06, room_r - 0.06, 0), (rear_x, room_l + 0.06, ceil), "mat_wall"))
    g.append(_box("wall_left", (rear_x, room_l, 0), (xw, room_l + 0.06, ceil), "mat_wall"))
    g.append(_box("wall_right", (rear_x, room_r - 0.06, 0), (xw, room_r, ceil), "mat_wall"))
    g.append(_box("ceiling", (rear_x - 0.06, room_r - 0.06, ceil), (xw + 0.06, room_l + 0.06, ceil + 0.05), "mat_ceiling"))
    g.append(_box("baseboard_back", (xw - 0.012, room_r, 0), (xw, room_l, 0.08), "mat_trim"))
    # desk: top, left trestle (open knee space next to the robot), drawer pedestal on the right
    g.append(_box("desk_top", (X_F, Y_R, Z_D - DESK_T), (X_B, Y_L, Z_D), "mat_desk"))
    g.append(_box("desk_leg_lf", (X_F + 0.03, Y_L - 0.06, 0), (X_F + 0.07, Y_L - 0.02, Z_D - DESK_T), "mat_metal"))
    g.append(_box("desk_leg_lb", (X_B - 0.07, Y_L - 0.06, 0), (X_B - 0.03, Y_L - 0.02, Z_D - DESK_T), "mat_metal"))
    g.append(_box("desk_rail_l", (X_F + 0.03, Y_L - 0.06, 0.06), (X_B - 0.03, Y_L - 0.02, 0.10), "mat_metal"))
    g.append(_box("desk_modesty", (X_B - 0.05, Y_R + 0.45, 0.30), (X_B - 0.035, Y_L - 0.06, Z_D - DESK_T), "mat_desk_side"))
    pl, pr = Y_R + 0.42, Y_R + 0.01
    g.append(_box("pedestal", (X_F + 0.02, pr, 0.0), (X_B - 0.02, pl, Z_D - DESK_T), "mat_desk_side"))
    for i, (z0, z1) in enumerate(((0.06, 0.28), (0.29, 0.49), (0.50, Z_D - DESK_T - 0.01))):
        g.append(_box(f"drawer_{i}", (X_F + 0.012, pr + 0.01, z0), (X_F + 0.02, pl - 0.01, z1), "mat_desk"))
        g.append(_box(f"drawer_{i}_pull", (X_F - 0.005, (pr + pl) / 2 - 0.05, z1 - 0.05),
                      (X_F + 0.012, (pr + pl) / 2 + 0.05, z1 - 0.035), "mat_metal"))
    # window on the back wall to the right of the desk area, a framed picture above the desk
    wz0, wz1 = 1.05, 2.05
    g.append(_box("window_glass", (xw - 0.004, -2.10, wz0), (xw, -1.30, wz1), "mat_glass", ' contype="0" conaffinity="0"'))
    for nm, lo, hi in (("window_sill", (xw - 0.06, -2.15, wz0 - 0.03), (xw, -1.25, wz0)),
                       ("window_head", (xw - 0.03, -2.15, wz1), (xw, -1.25, wz1 + 0.05)),
                       ("window_jamb_l", (xw - 0.03, -1.30, wz0), (xw, -1.25, wz1)),
                       ("window_jamb_r", (xw - 0.03, -2.15, wz0), (xw, -2.10, wz1))):
        g.append(_box(nm, lo, hi, "mat_trim"))
    g.append(_box("picture_frame", (xw - 0.025, -0.95, 1.55), (xw, -0.35, 1.95), "mat_frame"))
    g.append(_box("picture", (xw - 0.027, -0.92, 1.58), (xw - 0.025, -0.38, 1.92), "mat_picture",
                  ' contype="0" conaffinity="0"'))
    # bookcase against the left wall, low cabinet behind the robot
    g.append(_box("bookcase", (-0.60, room_l - 0.35, 0), (0.30, room_l, 1.80), "mat_desk_side"))
    for i, z in enumerate((0.35, 0.70, 1.05, 1.40)):
        g.append(_box(f"bookcase_books_{i}", (-0.55 + 0.05 * (i % 2), room_l - 0.33, z + 0.02),
                      (0.20 - 0.08 * (i % 3), room_l - 0.08, z + 0.25), f"mat_books{i % 3}"))
    g.append(_box("sideboard", (rear_x, -1.4, 0), (rear_x + 0.45, 0.2, 0.75), "mat_desk_side"))
    body = "".join(g)
    rgb = lambda t: " ".join(_f(v) for v in t)  # noqa: E731
    return f"""<mujoco model="pc_occ_desk_c{cell_index}">
  <!-- GENERATED by molmo_spaces.tasks.pc_occ_desk.write_pc_occ_desk_scenes (cell {cell_index}).
       Home office: desk top z={Z_D:.3f} (x {X_F:.2f}..{X_B:.2f}, y {Y_R:.2f}..{Y_L:.2f}), back wall
       x={X_W:.3f}. Monitor, keyboard, lamp, cup, coaster and decor are injected at runtime. -->
  <compiler angle="radian" autolimits="true" boundmass="0" balanceinertia="true"/>
  <option impratio="10" gravity="0 0 -9.8" integrator="implicitfast" cone="elliptic" jacobian="sparse" noslip_iterations="4">
    <flag multiccd="enable" contact="enable" warmstart="enable"/>
  </option>
  <size njmax="8000" nconmax="8000"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.0005"/>
    <headlight ambient="0.40 0.40 0.40" diffuse="0.50 0.50 0.50" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture name="tex_floor" type="2d" builtin="checker" rgb1="{rgb(c.floor)}" rgb2="{rgb(np.asarray(c.floor) * 0.94)}" mark="edge" markrgb="{rgb(np.asarray(c.floor) * 0.8)}" width="512" height="512"/>
    <material name="mat_floor" texture="tex_floor" texrepeat="6 6" reflectance="0"/>
    <material name="mat_wall" rgba="{rgb(c.wall)} 1" specular="0.05" reflectance="0"/>
    <material name="mat_ceiling" rgba="0.95 0.95 0.94 1" specular="0.02" reflectance="0"/>
    <material name="mat_trim" rgba="0.96 0.96 0.95 1" specular="0.1" reflectance="0"/>
    <material name="mat_desk" rgba="{rgb(c.desk)} 1" specular="0.15" reflectance="0"/>
    <material name="mat_desk_side" rgba="{rgb(np.asarray(c.desk) * 0.85)} 1" specular="0.1" reflectance="0"/>
    <material name="mat_metal" rgba="0.20 0.20 0.22 1" specular="0.4" reflectance="0"/>
    <material name="mat_glass" rgba="0.80 0.88 0.95 1" emission="0.35" specular="0.2" reflectance="0"/>
    <material name="mat_frame" rgba="0.18 0.15 0.12 1" reflectance="0"/>
    <material name="mat_picture" rgba="0.45 0.58 0.66 1" reflectance="0"/>
    <material name="mat_books0" rgba="0.55 0.20 0.18 1" reflectance="0"/>
    <material name="mat_books1" rgba="0.20 0.35 0.55 1" reflectance="0"/>
    <material name="mat_books2" rgba="0.75 0.68 0.50 1" reflectance="0"/>
  </asset>
  <default>
    <default class="furn">
      <geom contype="8" conaffinity="15" group="0" friction="0.9 0.9 0.001" solref="0.02 1" solimp="0.998 0.998 0.001"/>
    </default>
  </default>
  <worldbody>
    <light name="l_ceiling_a" pos="-0.3 -0.5 2.50" dir="0.3 -0.1 -1" diffuse="0.62 0.62 0.60" specular="0.1 0.1 0.1" castshadow="false"/>
    <light name="l_ceiling_b" pos="-1.2 0.2 2.50" dir="0.4 -0.2 -1" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05" castshadow="false"/>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="mat_floor" contype="8" conaffinity="15"/>
{body}  </worldbody>
</mujoco>
"""


def write_pc_occ_desk_scenes() -> list[str]:
    """(Re)generate the eight cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_OCC_DESK_SCENES[k])
        p.write_text(build_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------
# Hazard: distance from robot points to the lamp's convex collider pieces
# --------------------------------------------------------------------------------------------
HAZ_BODY = "pact_intrusion_lamp"
HAZ_BODIES = {1: HAZ_BODY + "_left", -1: HAZ_BODY + "_right"}
PARK_HAZ = {1: (-3.0, -1.6, 0.3), -1: (-3.0, -0.6, 0.3)}


def hull_planes(model, g: int) -> np.ndarray:
    """Outward face planes (n, b) of a convex mesh geom in its local frame: n.x <= b inside."""
    mid = int(model.geom_dataid[g])
    v = np.asarray(model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]], float)
    h = ConvexHull(v)
    eq = h.equations          # n.x + d <= 0 inside
    return np.concatenate([eq[:, :3], -eq[:, 3:4]], 1)


def hazard_distance(data, sets: dict[str, list], hulls: list[tuple[int, np.ndarray]]) -> dict[str, float]:
    """Min distance (m, 0 when inside) of each robot body's points to the hazard's convex pieces.
    Outside a piece the max face-plane distance is used (exact when the nearest feature is a face,
    a lower bound otherwise, so clearances are conservative); inside is exact."""
    out = {}
    for key, L in sets.items():
        P = np.concatenate([data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T for g, v in L], 0)
        best = 9.0
        for h, pl in hulls:
            loc = (P - data.geom_xpos[h]) @ data.geom_xmat[h].reshape(3, 3)
            dd = (loc @ pl[:, :3].T - pl[:, 3]).max(axis=1)
            best = min(best, float(max(dd.min(), 0.0)))
        out[key] = best
    return out


def hazard_hulls(model, prefix: str = HAZ_BODY) -> list[tuple[int, np.ndarray]]:
    out = []
    for g in range(model.ngeom):
        bn = model.body(int(model.geom_bodyid[g])).name or ""
        if bn.startswith(prefix) and (model.geom_contype[g] or model.geom_conaffinity[g]) \
                and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            out.append((g, hull_planes(model, g)))
    return out


_KEEP: list = []


def add_lamp_body(spec: MjSpec, side: int):
    tpl = hazard_template(side)
    hz = HAZARDS[side]
    name = HAZ_BODIES[side]
    b = spec.worldbody.add_body(name=name, mocap=True, pos=list(PARK_HAZ[side]))
    extra = _obja_spec(hz["uid"], hz["scale"], True, True)
    b.add_frame(pos=tpl["frame_pos"], quat=tpl["frame_quat"]).attach_body(extra.worldbody.bodies[0], name + "/", "")
    _KEEP.append(extra)    # keep the source spec alive until compile
    if hz.get("rgba") is not None:   # attach resets copied material colours: tint after attaching
        for mt in spec.materials:
            if (mt.name or "").startswith(name + "/"):
                mt.rgba = list(hz["rgba"])
    return b


def lamp_pose(side: int, cup_xy, jit: dict[str, float] | None = None,
              yaw_deg: float = LANE_YAW_DEG) -> tuple[list[float], list[float]]:
    """World mocap pose of the side's lamp for a cup at ``cup_xy`` (jitter in HAZ_JIT units)."""
    j = jit or {}
    nom = HAZ_NOMINAL[side]
    u, nl = lane_frame(yaw_deg)
    yaw = np.deg2rad(yaw_deg + nom["yaw"] + j.get("yaw", 0.0))
    Rw = R.from_euler("z", yaw)
    if HAZARDS[side]["kind"] == "desk":
        xy = np.asarray(cup_xy, float) + (nom["a"] + j.get("a", 0.0)) * u + (nom["t"] + j.get("t", 0.0)) * nl
        pos = [float(xy[0]), float(xy[1]), Z_D + 0.0005]
    else:
        P = Rw.apply(hazard_template(side)["pts"])
        x = X_W - 0.003 - float(P[:, 0].max())
        pos = [x, float(cup_xy[1]) + nom["dy"] + j.get("dy", 0.0), Z_D + nom["z"] + j.get("z", 0.0)]
    return pos, Rw.as_quat(scalar_first=True).tolist()


# --------------------------------------------------------------------------------------------
# Start pose
# --------------------------------------------------------------------------------------------
def _ready_tcp() -> list[float]:
    u, _ = lane_frame()
    p = u * R_PRE
    return [float(p[0]), float(p[1]), Z_D + 0.17]


READY_TCP = tuple(_ready_tcp())
# IK of READY_TCP x desk_tool_rotation(0) (hand yaw offset 40 deg) with link0 at H_BASE
PC_OCC_DESK_READY_QPOS: list[float] = [-0.7952, -0.6642, -0.0610, -1.9709, -0.0390, 1.3079, -2.2391]


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
class PCOccDeskSampler(PCSinkBlindSampler):
    """Cup behind the monitor's left half, coaster nearer the desk end, a desk lamp left or right
    of the lane with its head over the lane edge (hidden from both cameras)."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_OCC_DESK_ENV_VERSION
    SUITE = "occ_desk"
    BASE_JIT_XY = (0.010, 0.012)
    BASE_JIT_YAW = 0.015
    CUP_DT = (-0.015, 0.015)      # cup lateral offset from the lane
    COASTER_DT = 0.015
    PLACE_TOL = 0.045
    MAX_DRAWS = 40
    MONITOR_X = 0.03              # screen front face
    MONITOR_Y = -0.78             # screen centre

    @property
    def cell(self) -> DeskCell:  # type: ignore[override]
        return PC_OCC_DESK_CELLS[self.current_house_index % N_CELLS]

    def _target_dims(self, cell_index: int) -> dict[str, Any]:
        r, h, rgba, rim, label = CUPS[PC_OCC_DESK_CELLS[cell_index].cup]
        return dict(radius=r, height=h, rgba=rgba, cap=rim, label=label)

    def _dyn_pool(self, cell_index: int) -> list[dict[str, Any]]:
        return []

    # ---------------------------------------------------------------- static meshes
    def _attach_static(self, spec: MjSpec, uid: str, scale: float, name: str, pos, quat_frame, pos_frame):
        b = spec.worldbody.add_body(name=name, pos=list(pos))
        extra = _obja_spec(uid, scale, False, True)
        _KEEP.append(extra)
        b.add_frame(pos=pos_frame, quat=quat_frame).attach_body(extra.worldbody.bodies[0], name + "/", "")
        return b

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

        PickTaskSampler.add_auxiliary_objects(self, spec)
        k = self.current_house_index % N_CELLS
        self._house_cell = k
        self._dyn = []
        c = PC_OCC_DESK_CELLS[k]
        mt = monitor_template(c.monitor)
        self._attach_static(spec, mt["uid"], mt["scale"], "desk_monitor",
                            [self.MONITOR_X, self.MONITOR_Y, Z_D], mt["frame_quat"], mt["frame_pos"])
        kt = rest_template(KEYBOARDS[c.keyboard], 0.44)
        kq = R.from_euler("z", LANE_YAW_DEG + 140.0, degrees=True)
        self._attach_static(spec, KEYBOARDS[c.keyboard], kt["scale"], "desk_keyboard",
                            [-0.13, self.MONITOR_Y + 0.02, Z_D + 0.0005],
                            (R.from_euler("z", 90, degrees=True) * STAND).as_quat(scalar_first=True).tolist(),
                            R.from_euler("z", 90, degrees=True).apply(kt["frame_pos"]).tolist())
        del kq
        for i, (uid, size, x, y, yaw) in enumerate(DECOR):
            dt = rest_template(uid, size)
            rz = R.from_euler("z", yaw, degrees=True)
            self._attach_static(spec, uid, dt["scale"], f"desk_decor{i}", [x, y, Z_D + 0.0005 + 0.03 * (uid == "Book_7")],
                                (rz * STAND).as_quat(scalar_first=True).tolist(), rz.apply(dt["frame_pos"]).tolist())
        self._add_task_bodies(spec, k)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        td = self._target_dims(cell_index)
        self._target_spec = td
        r, h = td["radius"], td["height"]
        ns = "cavity_obj_0/"
        b = spec.worldbody.add_body(name=ns + "desk_cup", pos=[-3.0, -2.6, h / 2 + 0.01])
        b.add_joint(name=ns + "desk_cup_free", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.01)
        b.add_geom(name=ns + "desk_cup_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r, h / 2, 0.0], rgba=list(td["rgba"]) + [1.0], mass=0.22,
                   contype=8, conaffinity=15, condim=4, friction=[1.0, 0.02, 0.002], priority=1)
        b.add_geom(name=ns + "desk_cup_rim", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r * 1.01, 0.005, 0.0], pos=[0.0, 0.0, h / 2 - 0.005],
                   rgba=list(td["cap"]) + [1.0], mass=0.001, contype=0, conaffinity=0)
        b.add_geom(name=ns + "desk_cup_inside", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r * 0.86, 0.0012, 0.0], pos=[0.0, 0.0, h / 2 + 0.0002],
                   rgba=[0.25, 0.17, 0.12, 1.0], mass=0.0005, contype=0, conaffinity=0)
        self._injected_obj_name = ns + "desk_cup"
        self._metadata_adder.update({
            self._injected_obj_name: {
                "category": "cup", "object_enum": "temp_object", "is_static": False,
                "boundingBox": {"x": 2 * r, "y": 2 * r, "z": h},
            }
        })
        mat = spec.worldbody.add_body(name="place_receptacle_mat", mocap=True, pos=[-3.0, -2.2, 0.01])
        mat.add_geom(name="place_receptacle_mat_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[COASTER_R, COASTER_T / 2, 0.0], rgba=[0.66, 0.50, 0.33, 1.0],
                     contype=8, conaffinity=15, friction=[0.9, 0.9, 0.001])
        for sd in (1, -1):
            add_lamp_body(spec, sd)

    def _place_robot(self, env) -> None:
        super()._place_robot(env)

    def _cell_common(self) -> dict[str, Any]:
        return dict(z_desk=Z_D, x_front=X_F, x_back=X_B, y_left=Y_L, y_right=Y_R, x_wall=X_W,
                    lane_yaw_deg=LANE_YAW_DEG, h_base=H_BASE, board_depth=X_B - X_F)

    def heights(self) -> dict[str, float]:
        h = self._target_spec["height"]
        z_hi = Z_D + h + INS_CLEAR
        return dict(z_hi=z_hi, z_g=Z_D + h - GRASP_BELOW_TOP)

    def lamp_pose(self, side: int, cup_xy, jit: dict[str, float] | None = None):
        return lamp_pose(side, cup_xy, jit)

    def _draw_episode(self, env) -> dict[str, Any] | None:
        k = self.current_house_index % N_CELLS
        td = self._target_spec
        r, h = td["radius"], td["height"]
        u, nl = lane_frame()
        rc = float(np.random.uniform(*R_CUP))
        tc = float(np.random.uniform(*self.CUP_DT))
        cup = rc * u + tc * nl
        rm = float(np.random.uniform(*R_COASTER))
        mat = rm * u + (tc + float(np.random.uniform(-self.COASTER_DT, self.COASTER_DT))) * nl
        yaw = float(np.random.uniform(0, 2 * np.pi))
        hz = self.heights()
        z_hi, z_g = hz["z_hi"], hz["z_g"]
        mat_top = Z_D + COASTER_T
        z_p = mat_top + h - GRASP_BELOW_TOP + 0.004
        side = 1 if np.random.random() < 0.5 else -1
        jit = {k: float(np.random.uniform(-v, v)) for k, v in HAZ_JIT.items()}
        tilt = TILT_DEG * side
        lane_c = tc * nl
        pre = lane_c + R_PRE * u
        div = lane_c + R_DIV * u
        P = lambda xy, z: [float(xy[0]), float(xy[1]), float(z)]  # noqa: E731
        wps = [
            ("pregrasp", P(pre, z_hi), "fast", 0.0),
            ("pregrasp", P(div, z_hi), "insert", 0.0),
            ("pregrasp", P(div, z_hi), "dwell", 0.0),
            ("pregrasp", P(div, z_hi), "tilt", tilt),
            ("pregrasp", P(cup, z_hi), "insert", tilt),
            ("grasp", P(cup, z_g), "lower", tilt),
            ("CLOSE", None, None, None),
            ("lift", P(cup, z_hi), "lift", tilt),
            ("preplace", P(mat, z_hi), "carry", tilt),
            ("place", P(mat, z_p), "lower", tilt),
            ("OPEN", None, None, None),
            ("retreat", P(mat, z_hi), "lift", tilt),
            ("retreat", P(mat, z_hi), "tilt", 0.0),
            ("retreat", P(pre, z_hi), "mid", 0.0),
            ("go_home", list(READY_TCP), "fast", 0.0),
        ]
        for tt in (tilt, -tilt):
            poses = [_T(p, desk_tool_rotation(tt if abs(t) > 0 else 0.0, LANE_YAW_DEG, HAND_YAW_OFF))
                     for (_n, p, _s, t) in wps if p is not None]
            if not self._ik_chain_ok(env, poses):
                return None
        self._mocap(env, "place_receptacle_mat", [float(mat[0]), float(mat[1]), Z_D + COASTER_T / 2 + 0.0003])
        for sd in (1, -1):
            self._mocap(env, HAZ_BODIES[sd], PARK_HAZ[sd])
        lamp_pos, lamp_q = self.lamp_pose(side, cup, jit)
        self._mocap(env, HAZ_BODIES[side], lamp_pos, lamp_q)
        quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
        self._settle_target(env, [float(cup[0]), float(cup[1]), Z_D + h / 2 + 0.002], quat)
        mujoco.mj_forward(env.current_model, env.current_data)
        side_name = "left" if side > 0 else "right"
        wrist = cup - 0.08 * u
        return dict(
            **self._cell_common(),
            pick_board_z=Z_D, place_board_z=Z_D, pick_upper=False,
            can_radius=r, can_height=h, can_label=td["label"],
            start_xyz=[float(cup[0]), float(cup[1]), Z_D + h / 2],
            goal_xyz=[float(mat[0]), float(mat[1]), mat_top + h / 2],
            mat_xyz=[float(mat[0]), float(mat[1]), Z_D + COASTER_T / 2], mat_top=mat_top,
            place_tol=self.PLACE_TOL, expert=[[n, p, s, t] for (n, p, s, t) in wps], dwell_s=DWELL_S,
            neighbours=[], n_neighbours=0, lane_t=tc, hand_off_deg=HAND_YAW_OFF,
            hidden_hazard=dict(present=True, side=side_name, y=float(lamp_pos[1]), x=float(lamp_pos[0]),
                               z_bottom=float(Z_D), body=HAZ_BODIES[side]),
            reveal_view=dict(pos=[float(wrist[0] + 0.05), float(wrist[1] - 0.45), Z_D + 0.55],
                             lookat=[float(wrist[0]), float(wrist[1]), Z_D + 0.30], fovy=70.0,
                             title="Behind the monitor (not a robot camera): desk lamp in red"),
            blind=dict(side=int(side), side_name=side_name, tilt_deg=float(tilt), lamp_pos=lamp_pos,
                       lamp_quat=lamp_q, lamp_kind=HAZARDS[side]["kind"], lamp_label=HAZARDS[side]["label"],
                       jitter=jit, z_carry=z_hi, cell=int(k)),
        )

    def _referral(self) -> str:
        return self._target_spec.get("label", "cup")

    def _task_class(self):
        return PCOccDeskTask


# --------------------------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------------------------
class PCOccDeskTask(PCShelfBlindTask):
    def _blind_setup(self, model) -> None:
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        sp = getattr(self, "scene_params", {}) or {}
        body = (sp.get("hidden_hazard") or {}).get("body", HAZ_BODY)
        self._bclr = dict(hulls=hazard_hulls(model, body),
                          sets=robot_vertex_sets(model, robot_root, keys=CLEAR_BODIES), best={}, body=body)

    def _blind_step(self, data) -> None:
        model = data.model
        if getattr(self, "_bclr", None) is None or self.episode_step_count == 0:
            self._blind_setup(model)
        if self.episode_step_count % self.CLEAR_EVERY_BLIND:
            return
        bc = self._bclr
        for key, dist in hazard_distance(data, bc["sets"], bc["hulls"]).items():
            prev = bc["best"].get(key)
            if dist < 0.4 and (prev is None or dist < prev[0]):
                bc["best"][key] = (dist, bc["body"], int(self.episode_step_count))


# --------------------------------------------------------------------------------------------
# Expert
# --------------------------------------------------------------------------------------------
class PCOccDeskExpert(PCSinkBlindExpert):
    """pc_sink_blind expert with the hand yawed to the desk lane."""

    def _compute_trajectory(self):
        sp = getattr(self.task, "scene_params", {}) or {}
        dwell = float(sp.get("dwell_s", DWELL_S))
        yaw = float(sp.get("lane_yaw_deg", LANE_YAW_DEG))
        off = float(sp.get("hand_off_deg", HAND_YAW_OFF))
        rv = self.robot_view
        mg = rv.get_gripper_movegroup_ids()[0]
        cur = rv.get_move_group(mg).leaf_frame_to_world.copy()
        prims = []
        if self.START_GRIPPER_OPEN:
            prims.append(GripperAction(rv, True, 0.0))
        segs: list = []
        holding = False
        for name, p, speed, tilt in sp.get("expert", []):
            if name in ("CLOSE", "OPEN"):
                if segs:
                    prims.append(self._seq(segs, holding))
                    segs = []
                if name == "CLOSE":
                    prims.append(GripperAction(rv, False, self.policy_config.gripper_close_duration))
                    holding = self.HOLD_AFTER_CLOSE
                else:
                    prims.append(GripperAction(rv, True, self.policy_config.gripper_open_duration))
                    holding = False
                continue
            end = _T(p, desk_tool_rotation(float(tilt), yaw, off))
            segs.append(_BlindSeg(name=name, start_pose=cur, end_pose=end,
                                  speed=float(self.SPEEDS.get(speed, 0.08)),
                                  ang_speed=self.TILT_ANG_SPEED if speed == "tilt" else 0.9,
                                  min_dur=dwell if speed == "dwell" else 0.0))
            cur = end
        if segs:
            prims.append(self._seq(segs, holding))
        return prims


class PCOccDeskExpertConfig(PCSinkBlindExpertConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCOccDeskExpert


__all__ = [
    "PC_OCC_DESK_ENV_VERSION", "PC_OCC_DESK_CELLS", "PC_OCC_DESK_SCENES", "PC_OCC_DESK_READY_QPOS",
    "PCOccDeskSampler", "PCOccDeskTask", "PCOccDeskExpert", "PCOccDeskExpertConfig",
    "write_pc_occ_desk_scenes", "build_scene_xml", "hazard_template", "lamp_pose", "monitor_template",
    "hazard_distance", "hazard_hulls", "desk_tool_rotation", "lane_frame", "H_BASE", "HAZ_BODY",
    "HAZ_BODIES", "PARK_HAZ", "READY_TCP", "N_CELLS", "Z_D",
]


if __name__ == "__main__":
    import sys

    if "--write-scenes" in sys.argv:
        for p in write_pc_occ_desk_scenes():
            print(p)
