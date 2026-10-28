"""``pc_occ_cabinet``: reach into an open kitchen base cabinet, pick a bottle from the back of the
cabinet floor and set it on a mat near the front, while a FRYING PAN stored on the cabinet's
half-depth shelf overhangs the shelf edge, left or right of the arm's lane, at the height of the
SKIN-COVERED wrist knuckle / forearm. No camera can see the pan.

Why this suite exists (2026-10-15, ICLR rebuttal, "occluded natural items")
---------------------------------------------------------------------------
The earlier hidden-hazard suites hid an artificial-looking fixture (basket, valve box, drain pipe,
radio) behind trim. Here the hazard is an ordinary household object with a real mesh (an AI2-THOR
frying pan, ``Pan_3`` scaled to a 20 cm skillet) lying where pans are stored: on the shelf of the
pots-and-pans cabinet, pushed in carelessly so its body sticks out over the shelf edge. It is hidden
by the cabinet itself: the top drawer above the door opening (and the face-frame rail under it)
blocks the robot-mounted exo camera's view of the top of the cabinet interior, and the wrist camera
looks down along the fingers. Only the arm's proximity skin can tell where the pan is.

Scene (``custom_scenes/pc_occ_cabinet_c<k>.xml``, :func:`write_pc_occ_cabinet_scenes`)
-----------------------------------------------------------------------------------
A closed kitchen. Against the far wall a run of base cabinets under a stone countertop (top
0.915-0.925 m) with a window and wall cabinets above. In the middle the POTS CABINET: face frame, a
15 cm drawer (front + pull, its box inside) above a door opening (top edge 0.735 m), both doors
standing HALF OPEN (70 deg) toward the room, cabinet floor 12 cm above the room floor, and a
HALF-DEPTH adjustable shelf at the back (front edge 37.5 cm behind the face frame, top 0.604-0.610 m,
= the carry TCP + 29 cm but >= 0.604 m). A dishwasher on one side and a drawer stack on the other
(cell dependent). On the countertop a bread box, a cutting board leaning on the backsplash, a utensil
crock and a fruit bowl (static, away from the robot). On the cabinet floor 2-4 cooking pots
(AI2-THOR meshes, mocap, re-posed per episode) stand >= 15 cm beside the lane. The robot (FR3,
40-sensor hybrid skin) stands on the LOW 15 cm mobile base of ``pc_sink_blind``; exo_camera_1 keeps
its robot-mounted pose (link0 + (-1.05, -0.55, 1.30), aimed at link0 + (0.55, 0, 0.45)).

Hidden hazard (``pact_intrusion_pan_left`` / ``_right``, scored by the evaluator as ``hazard_bar``)
------------------------------------------------------------------------------------------------
A frying pan lies flat on the half shelf, LEFT or RIGHT of the lane (50/50 per episode, drawn
independently of the bottle, the mat, the pots, the cell and the start pose). Its body overhangs
the shelf edge by 7-8 cm (rim front 29.7-30.5 cm behind the face frame, centre of mass 2.4-3.3 cm
behind the shelf edge) and its handle points outward and back (15-50 deg from the lateral axis).
The rim's inner edge is :data:`PCOccCabinetSampler.HANDLE_GAP` beside the lane centre at the depth
:data:`D_REF` the wrist knuckle reaches; the pan body spans 29-33 cm above the carry TCP (the skinned
wrist knuckle reaches 34.4 cm above the TCP; fr3_link7 / gripper top out at 21 cm). The front arc is
14-20 cm wide where the wrist can reach it, so a hand tilted toward the pan cannot swing past it
(an earlier version with only the handle sticking out let fixed tilts of +-20 deg pass outside).
Everything of the pan lies above the exo camera's "lintel shadow" line (z > lintel - 0.461 x depth,
camera 1.45 m high, 1.55 m in front of the face frame) and above the wrist camera. Collision: a
cylinder (body) and a capsule (handle) fitted to the mesh; the mesh is visual only, with plain
materials (the THOR textures carry alpha and nudged a few occluded-scene pixels by 1/255). The other
side's pan is parked outside the room.

Task + expert (scripted TCP segments, hand pointing down; = ``pc_sink_blind``)
------------------------------------------------------------------------------
Pick the bottle (``cavity_obj_0/occ_bottle``, 15.5-17 cm) standing 27.5-30 cm behind the face frame
and set it on the mat (``place_receptacle_mat``) 11.5-14 cm behind the face frame (pc_shelf_pnp
success rule). Expert: pre-insert pose in front of the opening at the carry height -> slide in
untilted until the TCP is :data:`D_DIV` inside -> dwell :data:`DWELL_S` (the skin faces the pan)
-> TILT the hand :data:`TILT_DEG` about the reach axis, fingers leaning toward the pan side, which
swings the wrist knuckle / forearm ~5.5 cm to the OTHER side (the only plan difference) -> slide on
above the bottle -> lower -> close -> lift -> carry to above the mat -> lower -> open -> lift ->
untilt (the wrist is far in front of the pan there) -> back out -> start pose.

Collection-only additions (collection configs, never the eval config): DART-style executed-action
noise, the contact-tolerant save gate of ``pc_shelf_hard`` and one skin reading per policy step.
Per-link clearance to the pan is recorded in ``obs_scene['pc_clearance']``.

Regenerate the scene files after editing the geometry::

    python -m molmo_spaces.tasks.pc_occ_cabinet --write-scenes
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
from scipy.spatial.transform import Rotation as R

from molmo_spaces.tasks.pc_shelf import SCENE_DIR, _T
from molmo_spaces.tasks.pc_shelf_blind import CLEAR_BODIES, blind_tool_rotation, robot_vertex_sets
from molmo_spaces.tasks.pc_sink_blind import (
    H_BASE,
    PC_SINK_BLIND_READY_QPOS,
    Q_ALONG_X,
    Q_ALONG_Y,
    READY_TCP,
    PCSinkBlindExpert,
    PCSinkBlindExpertConfig,
    PCSinkBlindSampler,
    PCSinkBlindTask,
    _box,
    _cyl,
    _door,
    _f,
    _quat_z_to,
    primitive_distance,
)

log = logging.getLogger(__name__)

PC_OCC_CABINET_ENV_VERSION = "pc_occ_cabinet_v1"
N_CELLS = 8

# --------------------------------------------------------------------------------------------
# Geometry (metres, world frame; robot base at the origin facing +x)
# --------------------------------------------------------------------------------------------
TOE_H = 0.10             # toe-kick height
TOE_D = 0.07             # toe-kick recess
FLOOR_T = 0.018          # cabinet floor panel
FLOOR_TOP = TOE_H + 0.02  # cabinet floor top (0.12 m)
FF_T = 0.019             # face frame thickness
STILE_W = 0.045          # face-frame stile width
RAIL_H = 0.03            # face-frame rail between the drawer and the door opening
CAB_D = 0.61             # face front -> back wall (24 in cabinet incl. face frame)
CT_T = 0.03              # countertop thickness
CT_OVER = 0.025          # countertop overhang beyond the face frame
SIDE_T = 0.018           # carcass side panels
BACK_T = 0.006           # carcass back panel
SHELF_T = 0.018          # half-depth shelf thickness
SHELF_D0 = 0.375         # half-depth shelf front edge behind the face front
DOOR_OPEN_DEG = 70.0     # doors stand half open toward the room
INS_CLEAR = 0.030        # TCP above the bottle top while sliding in / carrying
GRASP_BELOW_TOP = 0.030  # TCP below the bottle body top at the grasp
X_PRE = 0.07             # pre-insert TCP stand-off in front of the face frame
D_DIV = 0.18             # TCP depth of the common (untilted) partial insertion where the tilt starts
DWELL_S = 1.3            # pause there before tilting (pc_sink_blind timing: the step-100 query sees it)
TILT_DEG = 10.0          # expert hand tilt (deg): swings the wrist knuckle ~5.5 cm sideways
SHELF_ABOVE_TCP = 0.290  # half-shelf top (= pan underside) this far above the carry TCP ...
SHELF_TOP_MIN = 0.604    # ... but never lower: the overhanging pan underside 29.7 cm behind the face
                         # must stay above the exo camera's lintel-shadow line (0.735 - 0.461 x 0.297)
D_REF = 0.325            # depth (behind the face) at which the pan rim's inner edge sits HANDLE_GAP
                         # beside the lane: ~ the deepest point the wrist knuckle reaches at the pick
WRIST_TOP = 0.344        # skinned wrist knuckle top above the TCP, hand straight down
HAND_TOP = 0.212         # fr3_link7 / gripper top above the TCP
MAT_HALF = 0.065
MAT_HALF_T = 0.003

PAN_UID = "Pan_3"        # AI2-THOR frying pan (dark non-stick, 25 cm, 16 cm handle)
PAN_SCALE = 0.80         # -> 20 cm skillet, 13 cm handle


@dataclass(frozen=True)
class OccCell:
    """One kitchen layout family."""

    x_f: float          # face-frame front x
    z_ct: float         # countertop top
    lintel: float       # top edge of the door opening (bottom of the rail under the drawer)
    cab_w: float        # pots-cabinet width
    bottle: int         # index into BOTTLES
    dw_left: bool       # dishwasher on the left (+y) of the cabinet (else on the right)
    door: tuple[float, float, float]
    counter: tuple[float, float, float]


# Fields: x_f, z_ct, lintel, cab_w, bottle, dw_left, door rgb, counter rgb
PC_OCC_CABINET_CELLS: tuple[OccCell, ...] = (
    OccCell(0.50, 0.915, 0.735, 0.914, 0, False, (0.93, 0.93, 0.91), (0.20, 0.20, 0.22)),
    OccCell(0.49, 0.920, 0.735, 0.914, 1, True, (0.36, 0.45, 0.55), (0.88, 0.87, 0.84)),
    OccCell(0.50, 0.925, 0.735, 0.864, 2, False, (0.55, 0.40, 0.27), (0.45, 0.44, 0.42)),
    OccCell(0.49, 0.915, 0.735, 0.914, 3, True, (0.53, 0.61, 0.54), (0.93, 0.93, 0.92)),
    OccCell(0.50, 0.920, 0.735, 0.914, 4, False, (0.25, 0.27, 0.30), (0.78, 0.74, 0.68)),
    OccCell(0.49, 0.925, 0.735, 0.864, 5, True, (0.80, 0.73, 0.62), (0.15, 0.15, 0.16)),
    OccCell(0.50, 0.915, 0.735, 0.914, 6, True, (0.62, 0.20, 0.18), (0.90, 0.90, 0.88)),
    OccCell(0.49, 0.920, 0.735, 0.864, 7, False, (0.88, 0.86, 0.80), (0.32, 0.30, 0.28)),
)

BOTTLES = (  # radius, body height, body rgba, cap rgba, label (pantry bottles kept with the pans)
    (0.0330, 0.160, (0.55, 0.60, 0.15), (0.15, 0.15, 0.16), "olive oil bottle"),
    (0.0320, 0.165, (0.45, 0.12, 0.10), (0.85, 0.80, 0.70), "red wine vinegar bottle"),
    (0.0340, 0.155, (0.92, 0.85, 0.55), (0.80, 0.15, 0.10), "sunflower oil bottle"),
    (0.0325, 0.170, (0.12, 0.10, 0.09), (0.85, 0.15, 0.12), "soy sauce bottle"),
    (0.0330, 0.160, (0.85, 0.55, 0.15), (0.20, 0.20, 0.22), "maple syrup bottle"),
    (0.0320, 0.165, (0.95, 0.95, 0.92), (0.20, 0.40, 0.80), "rice vinegar bottle"),
    (0.0335, 0.158, (0.75, 0.30, 0.10), (0.95, 0.95, 0.95), "hot sauce bottle"),
    (0.0325, 0.168, (0.30, 0.45, 0.20), (0.90, 0.85, 0.30), "sesame oil bottle"),
)

PC_OCC_CABINET_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_occ_cabinet_c{k}.xml") for k in range(N_CELLS)
)


# --------------------------------------------------------------------------------------------
# The frying pan template (AI2-THOR mesh, scaled; hazard-body frame: origin = pan centre on the
# pan's underside, handle along +x)
# --------------------------------------------------------------------------------------------
STAND = R.from_euler("x", 90, degrees=True)   # THOR assets are Y-up


def _pan_spec() -> MjSpec:
    """The THOR pan as a spec: joints, sites and the THOR primitive colliders removed (the hazard
    gets its own fitted colliders), meshes and body offsets scaled by PAN_SCALE."""
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    extra = MjSpec.from_file(str(install_uid(PAN_UID)))
    for j in list(extra.joints):
        extra.delete(j)
    for s in list(extra.sites):
        extra.delete(s)
    alpha = {mt.name: float(mt.rgba[3]) for mt in extra.materials}
    for g in list(extra.geoms):
        # THOR primitive colliders, and the half-transparent "decals off" / "glass" sub-meshes (no
        # visible surface on this pan; MuJoCo's transparent pass changed a few exo pixels by 1/255
        # even with the pan fully occluded)
        if "PrimitiveCollider" in (g.name or "") or alpha.get(g.material, 1.0) < 0.99:
            extra.delete(g)
    # Plain (untextured) materials: the THOR "AlbedoTransparency" textures carry an alpha channel
    # and changed 6-186 exo pixels by 1/255 with the pan fully occluded (R2); the brushed-metal
    # textures are almost uniform at this scale, so the material colours alone keep the look.
    for mt in extra.materials:
        mt.textures = [""] * len(mt.textures)
    for g in extra.geoms:
        g.contype = 0
        g.conaffinity = 0
        g.group = 0
        g.pos = [float(v) * PAN_SCALE for v in g.pos]
    for msh in extra.meshes:
        msh.scale = [PAN_SCALE] * 3
    for b in extra.bodies:
        if b.name and b is not extra.worldbody:
            b.pos = [float(v) * PAN_SCALE for v in b.pos]
    return extra


@lru_cache(maxsize=None)
def pan_template() -> dict[str, Any]:
    """Measure the scaled pan mesh: body radius / height, handle tip, fitted handle capsule, and the
    attach frame that puts the pan centre (underside) at the origin with the handle along +x."""
    wrap = MjSpec()
    holder = wrap.worldbody.add_body(name="holder", quat=list(STAND.as_quat(scalar_first=True)))
    extra = _pan_spec()
    holder.add_frame().attach_body(extra.worldbody.bodies[0], "pan/", "")
    m = wrap.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    pts = []
    for g in range(m.ngeom):
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        pts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    P = np.concatenate(pts, 0)
    lo, hi = P.min(0), P.max(0)
    ext = hi - lo
    ax = 0 if ext[0] > ext[1] else 1                  # handle axis (the long one)
    ox = 1 - ax
    r = float(ext[ox] / 2)
    c_o = float((hi[ox] + lo[ox]) / 2)
    best = None
    for c_a, sgn in ((lo[ax] + r, 1.0), (hi[ax] - r, -1.0)):
        q = np.zeros(2)
        q[ax], q[ox] = c_a, c_o
        n = int(np.sum(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) < 0.97 * r))
        if best is None or n > best[0]:
            best = (n, c_a, sgn)
    _, c_a, sgn = best
    centre = np.zeros(3)
    centre[ax], centre[ox], centre[2] = c_a, c_o, lo[2]
    u = np.zeros(3)
    u[ax] = sgn
    yaw0 = -float(np.arctan2(u[1], u[0]))
    Rz = R.from_euler("z", yaw0)
    Q = Rz.apply(P - centre)                       # template frame: handle along +x
    x_tip = float(Q[:, 0].max())
    body = Q[np.hypot(Q[:, 0], Q[:, 1]) <= r * 1.01]
    h_body = float(body[:, 2].max())
    hp = Q[Q[:, 0] > r + 0.004]
    root_sel = hp[hp[:, 0] < r + 0.02]
    tip_sel = hp[hp[:, 0] > x_tip - 0.02]
    a = np.array([r - 0.004, float(np.mean(root_sel[:, 1])), float(np.mean(root_sel[:, 2]))])
    b = np.array([x_tip - 0.008, float(np.mean(tip_sel[:, 1])), float(np.mean(tip_sel[:, 2]))])
    # capsule radius: covers every handle vertex
    ab = b - a
    t = np.clip(((hp - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    dist = np.linalg.norm(hp - (a + t[:, None] * ab), axis=1)
    cap_r = float(min(max(dist.max(), 0.006), 0.014))
    frame_rot = Rz * STAND
    return dict(r=r, h_body=h_body, x_tip=x_tip, cap_a=a.tolist(), cap_b=b.tolist(), cap_r=cap_r,
                tip_bottom=float(tip_sel[:, 2].min()), tip_y=float(np.mean(tip_sel[:, 1])),
                frame_quat=frame_rot.as_quat(scalar_first=True).tolist(),
                frame_pos=(-Rz.apply(centre)).tolist(), n_vert=int(len(P)))


def cell_heights(c: OccCell) -> dict[str, float]:
    """Heights that follow from the cell's bottle: carry TCP height, shelf top (= pan underside)."""
    r, h = BOTTLES[c.bottle][0], BOTTLES[c.bottle][1]
    z_hi = FLOOR_TOP + h + INS_CLEAR
    shelf_top = max(z_hi + SHELF_ABOVE_TCP, SHELF_TOP_MIN)
    return dict(z_hi=z_hi, shelf_top=shelf_top, pan_bottom=shelf_top, bottle_r=r, bottle_h=h)


# --------------------------------------------------------------------------------------------
# Scene XML
# --------------------------------------------------------------------------------------------
def build_scene_xml(cell_index: int) -> str:
    """The static MJCF of one kitchen cell (room, cabinet run, pots cabinet, countertop, decor)."""
    c = PC_OCC_CABINET_CELLS[cell_index]
    hz = cell_heights(c)
    rng = np.random.default_rng(9900 + cell_index)
    xf, W = c.x_f, c.cab_w
    xw = xf + CAB_D                    # kitchen wall (inner face)
    zc_top = c.z_ct
    zc_bot = c.z_ct - CT_T             # countertop underside = carcass top
    hw = W / 2
    run_l, run_r = hw + 0.60, -hw - 0.60      # base-cabinet run (+ one 60 cm cabinet each side)
    room_l, room_r = run_l + 0.35, run_r - 0.35
    rear_x = -1.45
    ceil = 2.50
    L = c.lintel
    zs = hz["shelf_top"]
    g: list[str] = []
    # ---- room shell (every skin ray ends on a real surface)
    g.append(_box("wall_kitchen", (xw, room_r - 0.06, 0), (xw + 0.06, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_rear", (rear_x - 0.06, room_r - 0.06, 0), (rear_x, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_left", (rear_x, room_l, 0), (xw, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_right", (rear_x, room_r - 0.06, 0), (xw, room_r, ceil), material="mat_wall"))
    g.append(_box("ceiling", (rear_x - 0.06, room_r - 0.06, ceil), (xw + 0.06, room_l + 0.06, ceil + 0.05), material="mat_ceiling"))
    g.append(_box("baseboard_rear", (rear_x, room_r, 0), (rear_x + 0.015, room_l, 0.08), material="mat_trim"))
    # ---- base-cabinet run: toe kick
    g.append(_box("toekick", (xf + TOE_D, run_r, 0.0), (xw, run_l, TOE_H), material="mat_kick"))
    # pots cabinet: sides, floor, back, half-depth shelf
    g.append(_box("potscab_side_l", (xf + FF_T, hw - SIDE_T, TOE_H), (xw, hw, zc_bot), material="mat_carcass"))
    g.append(_box("potscab_side_r", (xf + FF_T, -hw, TOE_H), (xw, -hw + SIDE_T, zc_bot), material="mat_carcass"))
    g.append(_box("potscab_floor", (xf + FF_T, -hw + SIDE_T, FLOOR_TOP - FLOOR_T), (xw - BACK_T, hw - SIDE_T, FLOOR_TOP), material="mat_carcass_floor"))
    g.append(_box("potscab_back", (xw - BACK_T, -hw + SIDE_T, FLOOR_TOP), (xw, hw - SIDE_T, zc_bot), material="mat_carcass"))
    g.append(_box("potscab_shelf", (xf + SHELF_D0, -hw + SIDE_T + 0.001, zs - SHELF_T), (xw - BACK_T, hw - SIDE_T - 0.001, zs), material="mat_carcass_floor"))
    # shelf pins (flat, contactless)
    for sy in (1.0, -1.0):
        for dx in (SHELF_D0 + 0.04, CAB_D - 0.05):
            g.append(_cyl(f"potscab_shelf_pin_{'l' if sy > 0 else 'r'}_{int(dx * 100)}", (xf + dx, sy * (hw - SIDE_T - 0.004), zs - SHELF_T - 0.003),
                          0.003, 0.004, material="mat_handle", quat=Q_ALONG_Y, extra=' contype="0" conaffinity="0"'))
    # drawer box above the door opening (its bottom = the compartment ceiling)
    g.append(_box("potscab_drawer_bottom", (xf + FF_T + 0.01, -hw + SIDE_T + 0.012, L + RAIL_H), (xw - 0.04, hw - SIDE_T - 0.012, L + RAIL_H + 0.012), material="mat_carcass_floor"))
    g.append(_box("potscab_drawer_back", (xw - 0.052, -hw + SIDE_T + 0.012, L + RAIL_H + 0.012), (xw - 0.04, hw - SIDE_T - 0.012, zc_bot - 0.02), material="mat_carcass_floor"))
    # face frame: stiles, bottom rail, rail under the drawer
    g.append(_box("potscab_stile_l", (xf, hw - STILE_W, TOE_H), (xf + FF_T, hw, zc_bot), material="mat_door"))
    g.append(_box("potscab_stile_r", (xf, -hw, TOE_H), (xf + FF_T, -hw + STILE_W, zc_bot), material="mat_door"))
    g.append(_box("potscab_rail_bottom", (xf, -hw + STILE_W, TOE_H), (xf + FF_T, hw - STILE_W, FLOOR_TOP + 0.012), material="mat_door"))
    g.append(_box("potscab_rail_mid", (xf, -hw + STILE_W, L), (xf + FF_T, hw - STILE_W, L + RAIL_H), material="mat_door"))
    g.append(_box("potscab_rail_top", (xf, -hw + STILE_W, zc_bot - 0.02), (xf + FF_T, hw - STILE_W, zc_bot), material="mat_door"))
    # overlay drawer front + bar pull
    g.append(_box("potscab_drawer_front", (xf - 0.019, -hw + 0.004, L + 0.004), (xf, hw - 0.004, zc_bot - 0.004), material="mat_door"))
    g.append(_box("potscab_drawer_inset", (xf - 0.0205, -hw + 0.04, L + 0.03), (xf - 0.019, hw - 0.04, zc_bot - 0.03), material="mat_door_inset",
                  extra=' contype="0" conaffinity="0"'))
    g.append(_cyl("potscab_drawer_pull", (xf - 0.045, 0.0, (L + zc_bot) / 2 + 0.01), 0.007, 0.10, material="mat_handle", quat=Q_ALONG_Y))
    for sy in (1.0, -1.0):
        g.append(_cyl(f"potscab_drawer_pull_post_{'l' if sy > 0 else 'r'}", (xf - 0.032, sy * 0.09, (L + zc_bot) / 2 + 0.01), 0.005, 0.013,
                      material="mat_handle", quat=Q_ALONG_X))
    # doors standing half open (hinged on the stiles' outer edges)
    door_w = (W - 2 * STILE_W) / 2 + 0.012
    g += _door("potscab_door_l", xf - 0.002, hw - 0.006, door_w, FLOOR_TOP + 0.006, L + 0.012, DOOR_OPEN_DEG, +1.0, "mat_door")
    g += _door("potscab_door_r", xf - 0.002, -hw + 0.006, door_w, FLOOR_TOP + 0.006, L + 0.012, DOOR_OPEN_DEG, -1.0, "mat_door")
    # neighbouring base cabinets: drawer stack on one side, dishwasher on the other
    for side in (1.0, -1.0):
        y_in, y_out = (hw, run_l) if side > 0 else (run_r, -hw)
        g.append(_box(f"basecab_{'l' if side > 0 else 'r'}_body", (xf + FF_T, y_in, TOE_H), (xw, y_out, zc_bot), material="mat_carcass"))
        is_dw = (side > 0) == c.dw_left
        tag = "l" if side > 0 else "r"
        if is_dw:
            g.append(_box(f"dishwasher_{tag}_panel", (xf - 0.004, y_in + 0.004, TOE_H + 0.005), (xf + FF_T, y_out - 0.004, zc_bot - 0.12), material="mat_steel"))
            g.append(_box(f"dishwasher_{tag}_control", (xf - 0.004, y_in + 0.004, zc_bot - 0.115), (xf + FF_T, y_out - 0.004, zc_bot - 0.005), material="mat_steel_dark"))
            g.append(_cyl(f"dishwasher_{tag}_handle", (xf - 0.03, (y_in + y_out) / 2, zc_bot - 0.15), 0.008, 0.24, material="mat_handle", quat=Q_ALONG_Y))
        else:
            z_edges = [TOE_H + 0.005, 0.40, 0.66, zc_bot - 0.005]
            for i in range(3):
                g.append(_box(f"drawer_{tag}_{i}", (xf - 0.004, y_in + 0.004, z_edges[i] + 0.002), (xf + FF_T, y_out - 0.004, z_edges[i + 1] - 0.002), material="mat_door"))
                g.append(_cyl(f"drawer_{tag}_{i}_pull", (xf - 0.025, (y_in + y_out) / 2, z_edges[i + 1] - 0.06), 0.007, 0.08, material="mat_handle", quat=Q_ALONG_Y))
    # countertop + backsplash
    cf = xf - CT_OVER
    g.append(_box("countertop", (cf, run_r - 0.02, zc_bot), (xw, run_l + 0.02, zc_top), material="mat_counter"))
    g.append(_box("backsplash", (xw - 0.01, run_r - 0.02, zc_top), (xw, run_l + 0.02, zc_top + 0.10), material="mat_counter"))
    # window above the counter (frame + glass showing daylight)
    wz0, wz1 = zc_top + 0.22, zc_top + 1.12
    g.append(_box("window_glass", (xw - 0.004, -0.42, wz0), (xw, 0.42, wz1), material="mat_glass", extra=' contype="0" conaffinity="0"'))
    for nm, lo, hi in (("window_sill", (xw - 0.06, -0.48, wz0 - 0.03), (xw, 0.48, wz0)),
                       ("window_head", (xw - 0.03, -0.48, wz1), (xw, 0.48, wz1 + 0.05)),
                       ("window_jamb_l", (xw - 0.03, 0.42, wz0), (xw, 0.48, wz1)),
                       ("window_jamb_r", (xw - 0.03, -0.48, wz0), (xw, -0.42, wz1)),
                       ("window_mullion", (xw - 0.02, -0.015, wz0), (xw, 0.015, wz1))):
        g.append(_box(nm, lo, hi, material="mat_trim"))
    # wall cabinets left and right of the window
    for side in (1.0, -1.0):
        y_a, y_b = (0.55, run_l) if side > 0 else (run_r, -0.55)
        tag = "l" if side > 0 else "r"
        g.append(_box(f"wallcab_{tag}", (xw - 0.33, y_a, zc_top + 0.46), (xw, y_b, zc_top + 1.22), material="mat_carcass"))
        g.append(_box(f"wallcab_{tag}_door", (xw - 0.349, y_a + 0.004, zc_top + 0.464), (xw - 0.33, y_b - 0.004, zc_top + 1.216), material="mat_door"))
        g.append(_cyl(f"wallcab_{tag}_pull", (xw - 0.37, y_a + 0.05 if side > 0 else y_b - 0.05, zc_top + 0.55), 0.006, 0.05, material="mat_handle"))
    # countertop items (static, away from the robot): bread box, cutting board, utensil crock, fruit bowl
    s_items = 1.0 if rng.random() < 0.5 else -1.0
    by = s_items * (hw + 0.22)
    g.append(_box("counter_breadbox", (xw - 0.30, by - 0.20, zc_top), (xw - 0.06, by + 0.20, zc_top + 0.17), material="mat_wood"))
    g.append(_box("counter_breadbox_lid", (xw - 0.305, by - 0.195, zc_top + 0.10), (xw - 0.30, by + 0.195, zc_top + 0.165), material="mat_wood_dark",
                  extra=' contype="0" conaffinity="0"'))
    g.append(_cyl("counter_breadbox_knob", (xw - 0.312, by, zc_top + 0.13), 0.008, 0.008, material="mat_handle", quat=Q_ALONG_X))
    q_board = R.from_euler("y", -12, degrees=True).as_quat(scalar_first=True)
    g.append(f'    <geom name="counter_board" class="furn" type="box" pos="{_f(xw - 0.035)} {_f(-by * 0.55)} {_f(zc_top + 0.19)}" '
             f'size="0.008 0.17 0.19" quat="{" ".join(_f(v) for v in q_board)}" material="mat_wood"/>\n')
    cy = -by * 0.95
    g.append(_cyl("counter_crock", (xw - 0.14, cy, zc_top + 0.085), 0.06, 0.085, material="mat_ceramic"))
    for i, (dy, dz, ln) in enumerate(((-0.02, 0.0, 0.14), (0.015, 0.01, 0.16), (0.0, -0.01, 0.12))):
        q_u = R.from_euler("x", (i - 1) * 9, degrees=True).as_quat(scalar_first=True)
        g.append(f'    <geom name="counter_utensil_{i}" class="furn" type="capsule" pos="{_f(xw - 0.14)} {_f(cy + dy)} {_f(zc_top + 0.17 + ln / 2 + dz)}" '
                 f'size="0.006 {_f(ln / 2)}" quat="{" ".join(_f(v) for v in q_u)}" material="mat_wood_dark"/>\n')
    g.append(_cyl("counter_fruitbowl", (xw - 0.30, -by * 0.25, zc_top + 0.04), 0.12, 0.04, material="mat_ceramic"))
    for i, (dx, dy, col) in enumerate(((0.03, 0.03, "mat_apple"), (-0.04, 0.02, "mat_orange"), (0.0, -0.05, "mat_apple"))):
        g.append(_cyl(f"counter_fruit_{i}", (xw - 0.30 + dx, -by * 0.25 + dy, zc_top + 0.11), 0.035, 0.0, material=col, kind="sphere"))
    body = "".join(g)
    door = " ".join(_f(v) for v in c.door)
    ctr = " ".join(_f(v) for v in c.counter)
    inset = " ".join(_f(max(0.0, v - 0.06)) for v in c.door)
    return f"""<mujoco model="pc_occ_cabinet_c{cell_index}">
  <!-- GENERATED by molmo_spaces.tasks.pc_occ_cabinet.write_pc_occ_cabinet_scenes (cell {cell_index}).
       Pots cabinet: face front x={xf:.3f}, countertop top z={c.z_ct:.3f}, door opening top
       z={L:.3f}, half-depth shelf top z={zs:.4f} (front edge {SHELF_D0:.3f} behind the face).
       The bottle, the mat, the floor pots and the hidden frying pan are injected at runtime.
       Do not edit by hand. -->
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
    <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.72 0.66 0.58" rgb2="0.68 0.62 0.54" mark="edge" markrgb="0.55 0.50 0.44" width="512" height="512"/>
    <material name="mat_floor" texture="tex_floor" texrepeat="5 5" reflectance="0"/>
    <material name="mat_wall" rgba="0.90 0.88 0.84 1" specular="0.05" reflectance="0"/>
    <material name="mat_ceiling" rgba="0.95 0.95 0.94 1" specular="0.02" reflectance="0"/>
    <material name="mat_trim" rgba="0.96 0.96 0.95 1" specular="0.1" reflectance="0"/>
    <material name="mat_kick" rgba="0.16 0.16 0.17 1" reflectance="0"/>
    <material name="mat_carcass" rgba="0.90 0.88 0.84 1" specular="0.05" reflectance="0"/>
    <material name="mat_carcass_floor" rgba="0.84 0.82 0.78 1" specular="0.05" reflectance="0"/>
    <material name="mat_door" rgba="{door} 1" specular="0.12" reflectance="0"/>
    <material name="mat_door_inset" rgba="{inset} 1" specular="0.12" reflectance="0"/>
    <material name="mat_counter" rgba="{ctr} 1" specular="0.3" reflectance="0"/>
    <material name="mat_steel" rgba="0.74 0.75 0.77 1" specular="0.5" reflectance="0"/>
    <material name="mat_steel_dark" rgba="0.22 0.22 0.24 1" specular="0.3" reflectance="0"/>
    <material name="mat_handle" rgba="0.60 0.61 0.63 1" specular="0.6" reflectance="0"/>
    <material name="mat_glass" rgba="0.80 0.88 0.95 1" emission="0.35" specular="0.2" reflectance="0"/>
    <material name="mat_ceramic" rgba="0.94 0.93 0.90 1" specular="0.4" reflectance="0"/>
    <material name="mat_wood" rgba="0.70 0.52 0.33 1" specular="0.1" reflectance="0"/>
    <material name="mat_wood_dark" rgba="0.42 0.29 0.18 1" specular="0.1" reflectance="0"/>
    <material name="mat_apple" rgba="0.75 0.12 0.10 1" specular="0.3" reflectance="0"/>
    <material name="mat_orange" rgba="0.95 0.55 0.10 1" specular="0.3" reflectance="0"/>
  </asset>
  <default>
    <default class="furn">
      <geom contype="8" conaffinity="15" group="0" friction="0.9 0.9 0.001" solref="0.02 1" solimp="0.998 0.998 0.001"/>
    </default>
  </default>
  <worldbody>
    <light name="l_ceiling_a" pos="0.0 0.3 2.40" dir="0.25 -0.1 -1" diffuse="0.62 0.62 0.60" specular="0.1 0.1 0.1" castshadow="false"/>
    <light name="l_ceiling_b" pos="-0.8 -0.5 2.40" dir="0.4 0.2 -1" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 0" material="mat_floor" contype="8" conaffinity="15"/>
{body}  </worldbody>
</mujoco>
"""


def write_pc_occ_cabinet_scenes() -> list[str]:
    """(Re)generate the eight cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_OCC_CABINET_SCENES[k])
        p.write_text(build_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------
# Hidden pan bodies
# --------------------------------------------------------------------------------------------
HAZ_BODY = "pact_intrusion_pan"              # prefix of both pan bodies
HAZ_BODIES = {1: HAZ_BODY + "_left", -1: HAZ_BODY + "_right"}
PARK_HAZ = {1: (-3.0, -1.6, 0.3), -1: (-3.0, -1.0, 0.3)}


def add_pan_body(spec: MjSpec, side: int):
    """A mocap pan (visual THOR mesh + fitted cylinder / capsule colliders), parked outside."""
    tpl = pan_template()
    name = HAZ_BODIES[side]
    b = spec.worldbody.add_body(name=name, mocap=True, pos=list(PARK_HAZ[side]))
    extra = _pan_spec()
    b.add_frame(pos=tpl["frame_pos"], quat=tpl["frame_quat"]).attach_body(extra.worldbody.bodies[0], name + "/", "")
    kw = dict(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001], rgba=[0.2, 0.8, 0.2, 0.0], group=3)
    b.add_geom(name=f"{name}_col_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
               size=[tpl["r"], tpl["h_body"] / 2, 0.0], pos=[0.0, 0.0, tpl["h_body"] / 2], **kw)
    a, bb = np.asarray(tpl["cap_a"], float), np.asarray(tpl["cap_b"], float)
    mid, v = (a + bb) / 2, bb - a
    Lc = float(np.linalg.norm(v))
    b.add_geom(name=f"{name}_col_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
               size=[tpl["cap_r"], Lc / 2, 0.0], pos=mid.tolist(), quat=_quat_z_to(v / Lc), **kw)
    return b


# --------------------------------------------------------------------------------------------
# Floor cookware (AI2-THOR pots, mocap, re-posed per episode)
# --------------------------------------------------------------------------------------------
FLOOR_POTS = ("Pot_8", "Pot_2", "Pot_22", "Pot_14", "Pot_24")
POT_PARK = (-3.4, 1.8, 0.2)


def pot_body_name(i: int) -> str:
    return f"pact_clutter_pot{i}"


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
class PCOccCabinetSampler(PCSinkBlindSampler):
    """Bottle at the back of the pots-cabinet floor, mat at the front, a frying pan on the half
    shelf left or right of the lane, overhanging the shelf edge (hidden from both cameras)."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_OCC_CABINET_ENV_VERSION
    SUITE = "occ_cabinet"
    BASE_JIT_XY = (0.012, 0.015)
    BASE_JIT_YAW = 0.02
    CAN_D = (0.275, 0.300)        # bottle centre behind the face front
    CAN_Y = (-0.03, 0.03)
    MAT_D = (0.115, 0.140)
    MAT_DY = 0.02
    PLACE_TOL = 0.05
    # inner edge of the pan rim at depth D_REF beside the lane centre, per side (+1 = left / +y,
    # -1 = right / -y). Set from the fixed-tilt scan of 2026-10-15 (27 draws, cells 0/3/5, gaps
    # 3.2-3.6 / 2.2-2.6 cm): the untilted wrist cleared the rim by <= 1.8 cm (left) / 1.2 cm (right)
    # and the expert (10 deg away) by >= 3.7 / 2.9 cm, so both gaps moved ~2 cm in: every fixed tilt
    # near 0 now hits either side and the expert keeps ~1 cm.
    HANDLE_GAP = {1: (0.010, 0.014), -1: (0.004, 0.008)}
    PAN_FRONT_D = (0.297, 0.305)  # pan rim front behind the face front (overhangs the shelf edge 7-8 cm)
    HANDLE_YAW = (15.0, 50.0)     # handle direction: from the outward lateral axis toward the back (deg)
    ITEM_LANE = 0.15              # pot footprints keep this far off the lane centre
    MAX_DRAWS = 40

    @property
    def cell(self) -> OccCell:  # type: ignore[override]
        return PC_OCC_CABINET_CELLS[self.current_house_index % N_CELLS]

    def _target_dims(self, cell_index: int) -> dict[str, Any]:
        r, h, rgba, cap, label = BOTTLES[PC_OCC_CABINET_CELLS[cell_index].bottle]
        return dict(radius=r, height=h, rgba=rgba, cap=cap, label=label)

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
        from molmo_spaces.utils.lazy_loading_utils import install_uid

        PickTaskSampler.add_auxiliary_objects(self, spec)
        k = self.current_house_index % N_CELLS
        self._house_cell = k
        self._dyn = []
        for i, uid in enumerate(FLOOR_POTS):
            extra = MjSpec.from_file(str(install_uid(uid)))
            body = extra.worldbody.bodies[0]
            for j in list(extra.joints):
                extra.delete(j)
            for s in list(extra.sites):
                extra.delete(s)
            body.mocap = True
            body.quat = list(STAND.as_quat(scalar_first=True))
            frame = spec.worldbody.add_frame(pos=[POT_PARK[0], POT_PARK[1] + 0.4 * i, POT_PARK[2]])
            frame.attach_body(body, f"{pot_body_name(i)}/", "")
            self._dyn.append(dict(name=pot_body_name(i), uid=uid, kind="thor", half=(0.13, 0.13, 0.12)))
        self._pot_geom = None
        self._add_task_bodies(spec, k)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        td = self._target_dims(cell_index)
        self._target_spec = td
        r, h = td["radius"], td["height"]
        ns = "cavity_obj_0/"
        b = spec.worldbody.add_body(name=ns + "occ_bottle", pos=[-3.0, -2.6, h / 2 + 0.01])
        b.add_joint(name=ns + "occ_bottle_free", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.01)
        b.add_geom(name=ns + "occ_bottle_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r, h / 2, 0.0], rgba=list(td["rgba"]) + [1.0], mass=0.25,
                   contype=8, conaffinity=15, condim=4, friction=[1.0, 0.02, 0.002], priority=1)
        b.add_geom(name=ns + "occ_bottle_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r * 0.45, 0.006, 0.0], pos=[0.0, 0.0, h / 2 + 0.006],
                   rgba=list(td["cap"]) + [1.0], mass=0.001, contype=0, conaffinity=0)
        b.add_geom(name=ns + "occ_bottle_label", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r * 1.004, h * 0.18, 0.0], pos=[0.0, 0.0, -h * 0.08],
                   rgba=[0.96, 0.94, 0.86, 1.0], mass=0.001, contype=0, conaffinity=0)
        self._injected_obj_name = ns + "occ_bottle"
        self._metadata_adder.update({
            self._injected_obj_name: {
                "category": "bottle", "object_enum": "temp_object", "is_static": False,
                "boundingBox": {"x": 2 * r, "y": 2 * r, "z": h},
            }
        })
        mat = spec.worldbody.add_body(name="place_receptacle_mat", mocap=True, pos=[-3.0, -2.2, 0.01])
        mat.add_geom(name="place_receptacle_mat_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[MAT_HALF, MAT_HALF, MAT_HALF_T], rgba=[0.15, 0.16, 0.18, 1.0],
                     contype=8, conaffinity=15, friction=[0.9, 0.9, 0.001])
        for sd in (1, -1):
            add_pan_body(spec, sd)

    # ---------------------------------------------------------------- floor pots
    def _measure_pots(self, env) -> list[dict[str, Any]]:
        """Rest offset / footprint of every floor pot (measured once per compiled model)."""
        from molmo_spaces.tasks.pc_kitchen import _geom_points, _subtree_geoms

        m, d = env.current_model, env.current_data
        if self._pot_geom is not None and self._pot_geom[0] == id(m):
            return self._pot_geom[1]
        saved = (d.mocap_pos.copy(), d.mocap_quat.copy())
        out = []
        for i, it in enumerate(self._dyn):
            bid = None
            prefix = it["name"] + "/"
            for bb in range(m.nbody):
                nm = m.body(bb).name or ""
                if nm.startswith(prefix) and int(m.body_parentid[bb]) == 0:
                    bid = bb
                    break
            mid = int(m.body_mocapid[bid])
            d.mocap_pos[mid] = 0.0
            d.mocap_quat[mid] = STAND.as_quat(scalar_first=True)
            mujoco.mj_kinematics(m, d)
            pts = np.concatenate([_geom_points(m, d, g) for g in _subtree_geoms(m, bid)])
            out.append(dict(body=bid, mocap=mid, z0=-float(pts[:, 2].min()), lo=pts[:, :2].min(0),
                            hi=pts[:, :2].max(0), h=float(np.ptp(pts[:, 2]))))
        d.mocap_pos[:], d.mocap_quat[:] = saved
        mujoco.mj_kinematics(m, d)
        self._pot_geom = (id(m), out)
        return out

    def _park_all(self, env) -> None:
        d = env.current_data
        for i, pg in enumerate(self._measure_pots(env)):
            d.mocap_pos[pg["mocap"]] = [POT_PARK[0], POT_PARK[1] + 0.4 * i, POT_PARK[2]]
            d.mocap_quat[pg["mocap"]] = STAND.as_quat(scalar_first=True)

    def _cell_common(self) -> dict[str, Any]:
        c = self.cell
        hz = cell_heights(c)
        return dict(x_f=c.x_f, x0=c.x_f, z_ct=c.z_ct, lintel=c.lintel, cab_w=c.cab_w,
                    floor_top=FLOOR_TOP, z_carry=hz["z_hi"], shelf_top=hz["shelf_top"],
                    shelf_front=c.x_f + SHELF_D0, pan_bottom=hz["pan_bottom"],
                    board_depth=CAB_D, h_base=H_BASE)

    def pan_pose(self, c: OccCell, y_c: float, side: int, gap: float, d_front: float,
                 handle_deg: float) -> tuple[list[float], list[float], dict[str, float]]:
        """Pan body pose (mocap pos, quat): rim front ``d_front`` behind the face (the body overhangs
        the shelf edge, centre of mass on the shelf), rim inner edge ``gap`` beside the lane at depth
        D_REF (side), handle pointing outward and back by ``handle_deg`` from the lateral axis."""
        tpl = pan_template()
        hz = cell_heights(c)
        r = tpl["r"]
        d_c = d_front + r
        w = float(np.sqrt(max(0.0, r ** 2 - (d_c - D_REF) ** 2)))   # rim half-chord at D_REF
        cy = y_c + side * (gap + w)
        psi = float(np.deg2rad(handle_deg))
        yaw = float(np.arctan2(side * np.cos(psi), np.sin(psi)))     # +x = back, side*y = outward
        pos = [c.x_f + d_c, cy, hz["shelf_top"] + 0.0005]
        quat = R.from_euler("z", yaw).as_quat(scalar_first=True).tolist()
        info = dict(yaw_deg=float(np.degrees(yaw)), rim_x=c.x_f + D_REF, rim_y=y_c + side * gap,
                    centre_behind_shelf_edge=float(d_c - SHELF_D0))
        return pos, quat, info

    def _draw_episode(self, env) -> dict[str, Any] | None:
        c = self.cell
        k = self.current_house_index % N_CELLS
        td = self._target_spec
        r, h = td["radius"], td["height"]
        hz = cell_heights(c)
        dc = float(np.random.uniform(*self.CAN_D))
        yc = float(np.random.uniform(*self.CAN_Y))
        dm = float(np.random.uniform(*self.MAT_D))
        ym = yc + float(np.random.uniform(-self.MAT_DY, self.MAT_DY))
        xc, xm = c.x_f + dc, c.x_f + dm
        yaw = float(np.random.uniform(0, 2 * np.pi))
        z_b = FLOOR_TOP
        mat_top = z_b + 2 * MAT_HALF_T
        # hidden pan: side drawn independently of everything visible
        side = 1 if np.random.random() < 0.5 else -1          # +1 = left (+y), -1 = right (-y)
        gap = float(np.random.uniform(*self.HANDLE_GAP[side]))
        d_front = float(np.random.uniform(*self.PAN_FRONT_D))
        handle_deg = float(np.random.uniform(*self.HANDLE_YAW))
        pan_pos, pan_quat, pan_info = self.pan_pose(c, yc, side, gap, d_front, handle_deg)
        tilt = TILT_DEG * side       # fingers lean toward the pan side -> wrist swings away
        z_hi = hz["z_hi"]
        z_g = z_b + h - GRASP_BELOW_TOP
        z_p = mat_top + h - GRASP_BELOW_TOP + 0.004
        x_pre = c.x_f - X_PRE
        x_div = c.x_f + D_DIV
        wps = [
            ("pregrasp", [x_pre, yc, z_hi], "fast", 0.0),
            ("pregrasp", [x_div, yc, z_hi], "insert", 0.0),
            ("pregrasp", [x_div, yc, z_hi], "dwell", 0.0),
            ("pregrasp", [x_div, yc, z_hi], "tilt", tilt),
            ("pregrasp", [xc, yc, z_hi], "insert", tilt),
            ("grasp", [xc, yc, z_g], "lower", tilt),
            ("CLOSE", None, None, None),
            ("lift", [xc, yc, z_hi], "lift", tilt),
            ("preplace", [xm, ym, z_hi], "carry", tilt),
            ("place", [xm, ym, z_p], "lower", tilt),
            ("OPEN", None, None, None),
            ("retreat", [xm, ym, z_hi], "lift", tilt),
            ("retreat", [xm, ym, z_hi], "tilt", 0.0),
            ("retreat", [x_pre, ym, z_hi], "mid", 0.0),
            ("go_home", list(READY_TCP), "fast", 0.0),
        ]
        # both tilts must be feasible: the accepted (visible) layouts never depend on the side
        for tt in (tilt, -tilt):
            poses = [_T(p, blind_tool_rotation(tt if abs(t) > 0 else 0.0))
                     for (_n, p, _s, t) in wps if p is not None]
            if not self._ik_chain_ok(env, poses):
                return None
        # cooking pots on the cabinet floor, off the reach lane
        pg = self._measure_pots(env)
        out: list = []
        placed: list = [(xc - r - 0.03, yc - r - 0.03, xc + r + 0.03, yc + r + 0.03),
                        (xm - MAT_HALF - 0.02, ym - MAT_HALF - 0.02, xm + MAT_HALF + 0.02, ym + MAT_HALF + 0.02)]
        lim = c.cab_w / 2 - SIDE_T - 0.01
        avail = list(range(len(self._dyn)))
        np.random.shuffle(avail)
        n_items = int(np.random.randint(2, 5))
        for i in avail:
            if len(out) >= n_items:
                break
            for _ in range(25):
                pyaw = float(np.random.uniform(0, 2 * np.pi))
                ca, sa = np.cos(pyaw), np.sin(pyaw)
                lo, hi = pg[i]["lo"], pg[i]["hi"]
                corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
                rc = corners @ np.array([[ca, sa], [-sa, ca]])
                flo, fhi = rc.min(0), rc.max(0)
                s_ = 1.0 if np.random.random() < 0.5 else -1.0
                if s_ > 0:
                    y_lo, y_hi = yc + self.ITEM_LANE - flo[1], lim - fhi[1]
                else:
                    y_lo, y_hi = -lim - flo[1], yc - self.ITEM_LANE - fhi[1]
                x_lo, x_hi = c.x_f + FF_T + 0.02 - flo[0], c.x_f + CAB_D - BACK_T - 0.02 - fhi[0]
                if y_hi <= y_lo or x_hi <= x_lo:
                    continue
                y = float(np.random.uniform(y_lo, y_hi))
                x = float(np.random.uniform(x_lo, x_hi))
                box = (x + flo[0], y + flo[1], x + fhi[0], y + fhi[1])
                if any(box[0] < p[2] + 0.01 and p[0] < box[2] + 0.01 and box[1] < p[3] + 0.01 and p[1] < box[3] + 0.01
                       for p in placed):
                    continue
                if FLOOR_TOP + pg[i]["h"] > hz["shelf_top"] - SHELF_T - 0.01 and box[2] > c.x_f + SHELF_D0:
                    continue
                placed.append(box)
                q = (R.from_euler("z", pyaw) * STAND).as_quat(scalar_first=True)
                out.append((i, [x, y, FLOOR_TOP + pg[i]["z0"] + 0.0005], q))
                break
        self._park_all(env)
        d = env.current_data
        for i, pos, q in out:
            d.mocap_pos[pg[i]["mocap"]] = pos
            d.mocap_quat[pg[i]["mocap"]] = q
        self._mocap(env, "place_receptacle_mat", [xm, ym, z_b + MAT_HALF_T + 0.0003])
        for sd in (1, -1):
            self._mocap(env, HAZ_BODIES[sd], PARK_HAZ[sd])
        self._mocap(env, HAZ_BODIES[side], pan_pos, pan_quat)
        quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
        self._settle_target(env, [xc, yc, z_b + h / 2 + 0.002], quat)
        mujoco.mj_forward(env.current_model, env.current_data)
        side_name = "left" if side > 0 else "right"
        return dict(
            **self._cell_common(),
            pick_board_z=z_b, place_board_z=z_b, pick_upper=False,
            can_radius=r, can_height=h, can_label=td["label"],
            start_xyz=[xc, yc, z_b + h / 2], goal_xyz=[xm, ym, mat_top + h / 2],
            mat_xyz=[xm, ym, z_b + MAT_HALF_T], mat_top=mat_top, place_tol=self.PLACE_TOL,
            expert=[[n, p, s, t] for (n, p, s, t) in wps], dwell_s=DWELL_S,
            neighbours=[[self._dyn[i]["name"], self._dyn[i]["uid"], pos, [float(v) for v in q]] for i, pos, q in out],
            n_neighbours=len(out),
            # evaluator contract (eval_act_place.py writes hidden_hazard_present / _y / _side)
            hidden_hazard=dict(present=True, side=side_name, y=float(pan_info["rim_y"]), x=float(pan_info["rim_x"]),
                               z_bottom=float(hz["pan_bottom"]), body=HAZ_BODIES[side], item="frying pan"),
            # diagnostic free camera (reports/2026-10-13/video_blind.py): low in the cabinet on the
            # free side, looking up at the pan handle and the arm
            reveal_view=dict(pos=[c.x_f + 0.57, yc - side * 0.33, hz["shelf_top"] + 0.115],
                             lookat=[c.x_f + 0.27, yc + side * 0.03, hz["shelf_top"] - 0.01], fovy=78.0,
                             title="Inside the cabinet (not a robot camera): frying pan on the shelf"),
            blind=dict(side=int(side), side_name=side_name, tilt_deg=float(tilt), pan_pos=pan_pos,
                       pan_quat=pan_quat, rim_gap=gap, pan_front_depth=d_front, handle_deg=handle_deg,
                       pan_z=hz["pan_bottom"], z_carry=z_hi, cell=int(k)),
        )

    def _referral(self) -> str:
        return self._target_spec.get("label", "bottle")

    def _task_class(self):
        return PCOccCabinetTask


# --------------------------------------------------------------------------------------------
# Task: pc_sink_blind task with clearance to the pan colliders
# --------------------------------------------------------------------------------------------
class PCOccCabinetTask(PCSinkBlindTask):
    def _blind_setup(self, model) -> None:
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        hg = [g for g in range(model.ngeom)
              if (model.body(int(model.geom_bodyid[g])).name or "").startswith(HAZ_BODY)
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
        sp = getattr(self, "scene_params", {}) or {}
        body = (sp.get("hidden_hazard") or {}).get("body", HAZ_BODY)
        self._bclr = dict(hg=hg, sets=robot_vertex_sets(model, robot_root, keys=CLEAR_BODIES), best={},
                          body=body)

    def _blind_step(self, data) -> None:
        model = data.model
        if getattr(self, "_bclr", None) is None or self.episode_step_count == 0:
            self._blind_setup(model)
        if self.episode_step_count % self.CLEAR_EVERY_BLIND:
            return
        bc = self._bclr
        for key, dist in primitive_distance(data, bc["sets"], bc["hg"]).items():
            prev = bc["best"].get(key)
            if dist < 0.4 and (prev is None or dist < prev[0]):
                bc["best"][key] = (dist, bc["body"], int(self.episode_step_count))


# --------------------------------------------------------------------------------------------
# Expert (= pc_sink_blind's: per-waypoint hand tilt + dwell from scene_params)
# --------------------------------------------------------------------------------------------
class PCOccCabinetExpert(PCSinkBlindExpert):
    pass


class PCOccCabinetExpertConfig(PCSinkBlindExpertConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCOccCabinetExpert


PC_OCC_CABINET_READY_QPOS = list(PC_SINK_BLIND_READY_QPOS)

__all__ = [
    "PC_OCC_CABINET_ENV_VERSION", "PC_OCC_CABINET_CELLS", "PC_OCC_CABINET_SCENES",
    "PC_OCC_CABINET_READY_QPOS", "PCOccCabinetSampler", "PCOccCabinetTask", "PCOccCabinetExpert",
    "PCOccCabinetExpertConfig", "write_pc_occ_cabinet_scenes", "build_scene_xml", "cell_heights",
    "pan_template", "primitive_distance", "H_BASE", "HAZ_BODY", "HAZ_BODIES", "PARK_HAZ",
    "READY_TCP", "N_CELLS", "SHELF_D0", "D_DIV", "TILT_DEG",
]


if __name__ == "__main__":
    import sys

    if "--write-scenes" in sys.argv:
        for p in write_pc_occ_cabinet_scenes():
            print(p)
    if "--template" in sys.argv:
        print(pan_template())
