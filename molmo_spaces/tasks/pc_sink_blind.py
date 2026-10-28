"""``pc_sink_blind``: reach into the cabinet under a kitchen sink, pick a bottle from the back of the
cabinet floor and set it on a mat near the front, while the sink's HIDDEN drain pipe threatens the
SKIN-COVERED wrist knuckle / forearm, never the gripper.

Why this suite exists (2026-10-14, ICLR rebuttal)
--------------------------------------------------
People reaching into the cabinet under a sink cannot see the plumbing and feel their way. Here the
only thing that differs between episode configurations is the drain pipe under the sink: neither
robot camera can see it, and it hangs where only the skinned links (fr3_link5 / fr3_link6) pass.

Scene (``custom_scenes/pc_sink_blind_c<k>.xml``, :func:`write_pc_sink_blind_scenes`)
-----------------------------------------------------------------------------------
A closed kitchen. Against the far wall a run of base cabinets with a stone countertop (top
0.91-0.93 m): the SINK BASE in the middle (face frame, 15 cm false front under the countertop,
both doors swung open, cabinet floor 12 cm above the room floor), a drawer cabinet on one side and a
dishwasher on the other (cell dependent), wall cabinets left and right, a window above the sink.
The countertop holds a large single-bowl undermount stainless sink (bowl floor 21-23 cm below the
counter top) with a centred faucet; under the sink, at the back, the hot / cold supply lines and
their shut-off valves. A few cleaning products stand on the cabinet floor away from the reach lane
(mocap, re-posed per episode). The robot (FR3, 40-sensor hybrid skin) stands on a LOW mobile base
(link0 at 0.15 m, :data:`H_BASE`) in front of the sink base; exo_camera_1 keeps its robot-mounted
pose of every pc suite (link0 + (-1.05, -0.55, 1.30), aimed at link0 + (0.55, 0, 0.45)), i.e. it
watches the arm and the cabinet opening from behind / above the robot.

Hidden hazard (``pact_intrusion_drain_left`` / ``_right``, scored by the evaluator as ``hazard_bar``)
------------------------------------------------------------------------------------------------
The sink has an OFFSET drain (a common single-bowl design that frees cabinet space; a double-bowl
sink was not used because it has plumbing under BOTH bowls): the drain hole sits in the bowl floor
to the LEFT or to the RIGHT of the reach lane (50/50 per episode, drawn independently of the
bottle, the mat, the items, the cell and the start pose). Under it hang the basket strainer, a short
tailpiece, an elbow and a 1.5 in waste pipe that runs SIDEWAYS toward the cabinet side at the height
of the skinned wrist knuckle, then back along the side to the P-trap, whose trap arm enters the back
wall (two mocap bodies, one per side; the unused one is parked outside the room). The pipe's inner
edge is 3.2-3.6 cm (left) / 2.2-2.6 cm (right) beside the lane centre (the FR3 forearm leans to +y)
and 28-33 cm behind the face frame. The false front's bottom edge hides all of it from exo_camera_1
(inside the "lintel shadow"), and the wrist camera looks down along the fingers, so neither camera
ever sees it. The pipe's bottom is 30 cm above the TCP during the carry: the gripper / fr3_link7
(top 21 cm above the TCP) stay >= 8 cm below it, the skinned wrist knuckle (top 33-34 cm above
the TCP) reaches into it.

Task + expert (scripted TCP segments, hand pointing down)
--------------------------------------------------------
Pick the bottle (``cavity_obj_0/sink_bottle``) standing 27.5-30 cm behind the face frame (the back
half of the cabinet floor) and set it on the rubber mat (``place_receptacle_mat``) 11.5-14 cm behind
the face frame (pc_shelf_pnp success rule). Expert: pre-insert pose in front of the opening at the
carry height -> slide in untilted until the TCP is :data:`D_DIV` (18 cm) inside -> dwell 1.3 s
(the wrist skin faces the pipe region; the hand arrives ~step 85 and tilts ~step 108, so the
policy's step-100 chunk query sees the skin at the pause; disclosed, as pc_shelf_blind) -> TILT the
hand 10 deg about the reach axis, fingers leaning toward the drain side, which swings the wrist
knuckle / forearm ~5.5 cm to the OTHER side (the only plan difference) -> slide on above the bottle
-> lower -> close -> lift -> carry to above the mat -> lower -> open -> lift -> untilt (the wrist is
far in front of the pipe there) -> back out -> start pose.

Collection-only additions (switched on by the collection configs, never by the eval config):
DART-style executed-action noise, the contact-tolerant save gate of ``pc_shelf_hard`` and one skin
reading per policy step. Per-link clearance to the drain pipe is recorded in
``obs_scene['pc_clearance']`` (scripts/audit_arm_clearance.py reads it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    GripperAction,
)
from molmo_spaces.tasks.pc_shelf import (
    SCENE_DIR,
    PCShelfPnPSampler,
    _T,
    _jsonable,
)
from molmo_spaces.tasks.pc_shelf_blind import (
    CLEAR_BODIES,
    PCShelfBlindExpert,
    PCShelfBlindExpertConfig,
    PCShelfBlindTask,
    _BlindSeg,
    blind_tool_rotation,
    robot_vertex_sets,
)
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask  # noqa: F401
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)

PC_SINK_BLIND_ENV_VERSION = "pc_sink_blind_v1"
N_CELLS = 8

# --------------------------------------------------------------------------------------------
# Geometry (metres, world frame; robot base at the origin facing +x)
# --------------------------------------------------------------------------------------------
H_BASE = 0.15            # robot deck height = link0 z (low mobile base)
TOE_H = 0.10             # toe-kick height
TOE_D = 0.07             # toe-kick recess
FLOOR_T = 0.018          # cabinet floor panel
FLOOR_TOP = TOE_H + 0.02  # cabinet floor top (0.12 m)
FF_T = 0.019             # face frame / false front thickness
STILE_W = 0.045          # face-frame stile width
CAB_D = 0.585            # face front -> back wall
CT_T = 0.03              # countertop thickness
CT_OVER = 0.025          # countertop overhang beyond the face frame
SIDE_T = 0.018           # carcass side panels
BOWL_X0 = 0.07           # bowl inner front wall behind the face front
BOWL_L = 0.42            # bowl inner front-to-back length
SS_T = 0.003             # stainless sheet thickness (bowl walls / floor)
INS_CLEAR = 0.030        # TCP above the bottle top while sliding in / carrying
GRASP_BELOW_TOP = 0.030  # TCP below the bottle body top at the grasp
X_PRE = 0.07             # pre-insert TCP stand-off in front of the face frame
D_DIV = 0.18             # TCP depth of the common (untilted) partial insertion where the tilt starts
DWELL_S = 1.3            # pause there before tilting: the hand arrives ~step 85, the tilt starts ~step 108,
                         # so the policy's step-100 chunk query sees the skin at the pause
TILT_DEG = 10.0          # expert hand tilt (deg): swings the wrist knuckle ~5.5 cm sideways
R_PIPE = 0.024           # 1.5 in drain pipe (48 mm OD)
R_STRAINER = 0.045       # basket strainer body
PIPE_BELOW_WRIST = 0.044  # pipe bottom this far below the straight wrist-knuckle top
WRIST_TOP = 0.344        # skinned wrist knuckle top above the TCP, hand straight down
HAND_TOP = 0.212         # fr3_link7 / gripper top above the TCP
D_DRAIN = 0.300          # drain (strainer axis) behind the face front
TRAP_DIP = 0.055         # P-trap dip below the waste pipe axis
TRAP_BACK = 0.065        # P-trap U width (along x) before the trap arm
TRAP_FROM_WALL = 0.11    # P-trap inlet this far in front of the back wall
MAT_HALF = 0.065
MAT_HALF_T = 0.003


@dataclass(frozen=True)
class SinkCell:
    """One kitchen layout family."""

    x_f: float          # face-frame front x
    z_ct: float         # countertop top
    lintel: float       # bottom edge of the false front (top of the door opening)
    cab_w: float        # sink-base width
    bowl_w: float       # bowl inner width
    bottle: int         # index into BOTTLES
    dw_left: bool       # dishwasher on the left (+y) of the sink base (else on the right)
    door: tuple[float, float, float]
    counter: tuple[float, float, float]


# Fields: x_f, z_ct, lintel, cab_w, bowl_w, bottle, dw_left, door rgb, counter rgb
PC_SINK_BLIND_CELLS: tuple[SinkCell, ...] = (
    SinkCell(0.50, 0.915, 0.735, 0.914, 0.70, 0, False, (0.93, 0.93, 0.91), (0.20, 0.20, 0.22)),
    SinkCell(0.49, 0.920, 0.735, 0.914, 0.72, 1, True, (0.36, 0.45, 0.55), (0.88, 0.87, 0.84)),
    SinkCell(0.50, 0.925, 0.735, 0.864, 0.66, 2, False, (0.55, 0.40, 0.27), (0.45, 0.44, 0.42)),
    SinkCell(0.49, 0.915, 0.735, 0.914, 0.70, 3, True, (0.53, 0.61, 0.54), (0.93, 0.93, 0.92)),
    SinkCell(0.50, 0.920, 0.735, 0.914, 0.72, 4, False, (0.25, 0.27, 0.30), (0.78, 0.74, 0.68)),
    SinkCell(0.49, 0.925, 0.735, 0.864, 0.66, 5, True, (0.80, 0.73, 0.62), (0.15, 0.15, 0.16)),
    SinkCell(0.50, 0.915, 0.735, 0.914, 0.70, 6, True, (0.62, 0.20, 0.18), (0.90, 0.90, 0.88)),
    SinkCell(0.49, 0.920, 0.735, 0.864, 0.68, 7, False, (0.88, 0.86, 0.80), (0.32, 0.30, 0.28)),
)

BOTTLES = (  # radius, body height, body rgba, cap rgba, label
    (0.0330, 0.160, (0.15, 0.45, 0.85), (0.95, 0.95, 0.95), "blue dish soap bottle"),
    (0.0320, 0.165, (0.95, 0.80, 0.15), (0.20, 0.55, 0.25), "yellow cleaner bottle"),
    (0.0340, 0.155, (0.92, 0.92, 0.90), (0.85, 0.15, 0.10), "white bleach bottle"),
    (0.0325, 0.170, (0.20, 0.65, 0.35), (0.95, 0.95, 0.95), "green detergent bottle"),
    (0.0330, 0.160, (0.85, 0.25, 0.55), (0.25, 0.25, 0.28), "pink soap bottle"),
    (0.0320, 0.165, (0.55, 0.30, 0.75), (0.95, 0.95, 0.95), "purple cleaner bottle"),
    (0.0335, 0.158, (0.95, 0.50, 0.15), (0.20, 0.20, 0.22), "orange degreaser bottle"),
    (0.0325, 0.168, (0.35, 0.80, 0.85), (0.15, 0.30, 0.65), "teal glass-cleaner bottle"),
)

PC_SINK_BLIND_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_sink_blind_c{k}.xml") for k in range(N_CELLS)
)


def cell_heights(c: SinkCell) -> dict[str, float]:
    """Heights that follow from the cell's bottle: carry TCP height, pipe axis, bowl floor."""
    r, h = BOTTLES[c.bottle][0], BOTTLES[c.bottle][1]
    z_hi = FLOOR_TOP + h + INS_CLEAR
    pipe_bot = z_hi + WRIST_TOP - PIPE_BELOW_WRIST
    z_pipe = pipe_bot + R_PIPE
    z_bb = z_pipe + R_PIPE + 0.033          # bowl floor top: strainer + coupling above the elbow
    return dict(z_hi=z_hi, z_pipe=z_pipe, pipe_bottom=pipe_bot, z_bb=z_bb,
                bowl_depth=c.z_ct - z_bb, bottle_r=r, bottle_h=h)


# --------------------------------------------------------------------------------------------
# Scene XML
# --------------------------------------------------------------------------------------------
def _f(v: float) -> str:
    return f"{float(v):.4f}"


def _box(name: str, lo, hi, cls: str = "furn", material: str | None = None, rgba=None,
         extra: str = "") -> str:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    mat = f' material="{material}"' if material else ""
    col = "" if rgba is None else f' rgba="{" ".join(_f(v) for v in rgba)} 1"'
    return (f'    <geom name="{name}" class="{cls}" type="box" pos="{_f(c[0])} {_f(c[1])} {_f(c[2])}" '
            f'size="{_f(h[0])} {_f(h[1])} {_f(h[2])}"{mat}{col}{extra}/>\n')


def _cyl(name: str, pos, r: float, hh: float, cls: str = "furn", material: str | None = None,
         rgba=None, quat=None, kind: str = "cylinder", extra: str = "") -> str:
    mat = f' material="{material}"' if material else ""
    col = "" if rgba is None else f' rgba="{" ".join(_f(v) for v in rgba)} 1"'
    q = "" if quat is None else f' quat="{" ".join(_f(v) for v in quat)}"'
    size = f"{_f(r)} {_f(hh)}" if kind != "sphere" else f"{_f(r)}"
    return (f'    <geom name="{name}" class="{cls}" type="{kind}" pos="{_f(pos[0])} {_f(pos[1])} {_f(pos[2])}" '
            f'size="{size}"{q}{mat}{col}{extra}/>\n')


Q_ALONG_X = (0.70710678, 0.0, 0.70710678, 0.0)   # cylinder axis (local z) -> world x
Q_ALONG_Y = (0.70710678, 0.70710678, 0.0, 0.0)   # cylinder axis -> world y


def _door(name: str, hinge_x: float, hinge_y: float, width: float, z0: float, z1: float,
          open_deg: float, sign: float, mat: str) -> list[str]:
    """A cabinet door swung open about a vertical hinge at (hinge_x, hinge_y). sign=+1: the door's
    free edge points to +y when closed (left door hinged on the right... ), here: the door lies in
    the face plane toward -sign*y when closed and swings out toward -x."""
    a = np.deg2rad(open_deg)
    # closed: from the hinge toward the opening centre (direction -sign*y); open: rotate toward -x
    d = np.array([-np.sin(a), -sign * np.cos(a)])
    c = np.array([hinge_x, hinge_y]) + d * width / 2
    yaw = float(np.arctan2(d[1], d[0]))
    q = R.from_euler("z", yaw).as_quat(scalar_first=True)
    out = [f'    <geom name="{name}" class="furn" type="box" pos="{_f(c[0])} {_f(c[1])} {_f((z0 + z1) / 2)}" '
           f'size="{_f(width / 2)} 0.0095 {_f((z1 - z0) / 2)}" quat="{" ".join(_f(v) for v in q)}" material="{mat}"/>\n']
    # pull near the free edge on the door's outer (now inner-facing) side
    nrm = np.array([-d[1], d[0]]) * (-sign)
    pe = np.array([hinge_x, hinge_y]) + d * (width - 0.05) + nrm * 0.022
    out.append(f'    <geom name="{name}_pull" class="furn" type="capsule" pos="{_f(pe[0])} {_f(pe[1])} {_f(z1 - 0.12)}" '
               f'size="0.006 0.05" material="mat_handle"/>\n')
    return out


def build_scene_xml(cell_index: int) -> str:
    """The static MJCF of one kitchen cell (room, cabinets, countertop, sink, faucet, decor)."""
    c = PC_SINK_BLIND_CELLS[cell_index]
    hz = cell_heights(c)
    rng = np.random.default_rng(8800 + cell_index)
    xf, W = c.x_f, c.cab_w
    xw = xf + CAB_D                    # kitchen wall (inner face)
    zc_top = c.z_ct
    zc_bot = c.z_ct - CT_T             # countertop underside = carcass top
    hw = W / 2
    run_l, run_r = hw + 0.60, -hw - 0.60      # base-cabinet run (+ one 60 cm cabinet each side)
    room_l, room_r = run_l + 0.35, run_r - 0.35
    rear_x = -1.45
    ceil = 2.50
    g: list[str] = []
    # ---- room shell (every skin ray ends on a real surface)
    g.append(_box("wall_kitchen", (xw, room_r - 0.06, 0), (xw + 0.06, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_rear", (rear_x - 0.06, room_r - 0.06, 0), (rear_x, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_left", (rear_x, room_l, 0), (xw, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_right", (rear_x, room_r - 0.06, 0), (xw, room_r, ceil), material="mat_wall"))
    g.append(_box("ceiling", (rear_x - 0.06, room_r - 0.06, ceil), (xw + 0.06, room_l + 0.06, ceil + 0.05), material="mat_ceiling"))
    g.append(_box("baseboard_rear", (rear_x, room_r, 0), (rear_x + 0.015, room_l, 0.08), material="mat_trim"))
    # ---- base-cabinet run: toe kick, carcass sides
    g.append(_box("toekick", (xf + TOE_D, run_r, 0.0), (xw, run_l, TOE_H), material="mat_kick"))
    # sink base: side panels, floor, back panel (cut out above the floor for plumbing: wall shows)
    g.append(_box("sinkbase_side_l", (xf + FF_T, hw - SIDE_T, TOE_H), (xw, hw, zc_bot), material="mat_carcass"))
    g.append(_box("sinkbase_side_r", (xf + FF_T, -hw, TOE_H), (xw, -hw + SIDE_T, zc_bot), material="mat_carcass"))
    g.append(_box("sinkbase_floor", (xf + FF_T, -hw + SIDE_T, FLOOR_TOP - FLOOR_T), (xw, hw - SIDE_T, FLOOR_TOP), material="mat_carcass_floor"))
    g.append(_box("sinkbase_back_lo", (xw - 0.006, -hw + SIDE_T, FLOOR_TOP), (xw, hw - SIDE_T, FLOOR_TOP + 0.10), material="mat_carcass"))
    # face frame: stiles, bottom rail, top rail + false front
    g.append(_box("sinkbase_stile_l", (xf, hw - STILE_W, TOE_H), (xf + FF_T, hw, zc_bot), material="mat_door"))
    g.append(_box("sinkbase_stile_r", (xf, -hw, TOE_H), (xf + FF_T, -hw + STILE_W, zc_bot), material="mat_door"))
    g.append(_box("sinkbase_rail_bottom", (xf, -hw + STILE_W, TOE_H), (xf + FF_T, hw - STILE_W, FLOOR_TOP + 0.012), material="mat_door"))
    g.append(_box("sinkbase_false_front", (xf, -hw + STILE_W, c.lintel), (xf + FF_T, hw - STILE_W, zc_bot), material="mat_door"))
    # false-front trim lines (flat, contactless)
    g.append(_box("sinkbase_false_front_trim", (xf - 0.0015, -hw + STILE_W + 0.03, c.lintel + 0.03),
                  (xf, hw - STILE_W - 0.03, zc_bot - 0.03), material="mat_door_inset", extra=' contype="0" conaffinity="0"'))
    # doors swung open ~100 deg (hinged on the stiles' outer edges)
    door_w = (W - 2 * STILE_W) / 2 + 0.012
    g += _door("sinkbase_door_l", xf - 0.002, hw - 0.006, door_w, FLOOR_TOP + 0.006, c.lintel + 0.012, 100.0, +1.0, "mat_door")
    g += _door("sinkbase_door_r", xf - 0.002, -hw + 0.006, door_w, FLOOR_TOP + 0.006, c.lintel + 0.012, 100.0, -1.0, "mat_door")
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
    # countertop with the sink cutout (bowl opening) and the backsplash
    bx0, bx1 = xf + BOWL_X0, xf + BOWL_X0 + BOWL_L
    bw2 = c.bowl_w / 2
    cf = xf - CT_OVER
    g.append(_box("counter_front", (cf, run_r - 0.02, zc_bot), (bx0, run_l + 0.02, zc_top), material="mat_counter"))
    g.append(_box("counter_back", (bx1, run_r - 0.02, zc_bot), (xw, run_l + 0.02, zc_top), material="mat_counter"))
    g.append(_box("counter_left", (bx0, bw2, zc_bot), (bx1, run_l + 0.02, zc_top), material="mat_counter"))
    g.append(_box("counter_right", (bx0, run_r - 0.02, zc_bot), (bx1, -bw2, zc_top), material="mat_counter"))
    g.append(_box("backsplash", (xw - 0.01, run_r - 0.02, zc_top), (xw, run_l + 0.02, zc_top + 0.10), material="mat_counter"))
    # undermount bowl (stainless): floor + four walls, under the countertop
    z_bb = hz["z_bb"]
    g.append(_box("sink_bowl_floor", (bx0 - SS_T, -bw2 - SS_T, z_bb - SS_T), (bx1 + SS_T, bw2 + SS_T, z_bb), material="mat_sink"))
    g.append(_box("sink_bowl_wall_front", (bx0 - SS_T, -bw2 - SS_T, z_bb), (bx0, bw2 + SS_T, zc_bot), material="mat_sink"))
    g.append(_box("sink_bowl_wall_back", (bx1, -bw2 - SS_T, z_bb), (bx1 + SS_T, bw2 + SS_T, zc_bot), material="mat_sink"))
    g.append(_box("sink_bowl_wall_left", (bx0, bw2, z_bb), (bx1, bw2 + SS_T, zc_bot), material="mat_sink"))
    g.append(_box("sink_bowl_wall_right", (bx0, -bw2 - SS_T, z_bb), (bx1, -bw2, zc_bot), material="mat_sink"))
    # bowl rim lip under the countertop edge (mounting clips zone), flat
    g.append(_box("sink_bowl_flange", (bx0 - 0.02, -bw2 - 0.02, zc_bot - 0.004), (bx1 + 0.02, bw2 + 0.02, zc_bot), material="mat_sink",
                  extra=' contype="0" conaffinity="0"'))
    # faucet (centred behind the bowl) + soap dispenser
    fx = bx1 + 0.045
    g.append(_cyl("faucet_base", (fx, 0.0, zc_top + 0.02), 0.028, 0.02, material="mat_chrome"))
    g.append(_cyl("faucet_column", (fx, 0.0, zc_top + 0.20), 0.016, 0.17, material="mat_chrome"))
    g.append(_cyl("faucet_neck", (fx - 0.09, 0.0, zc_top + 0.37), 0.014, 0.09, material="mat_chrome", quat=Q_ALONG_X))
    g.append(_cyl("faucet_spout", (fx - 0.18, 0.0, zc_top + 0.33), 0.016, 0.04, material="mat_chrome"))
    g.append(_cyl("faucet_lever", (fx + 0.01, 0.045, zc_top + 0.12), 0.008, 0.035, material="mat_chrome", quat=Q_ALONG_Y))
    g.append(_cyl("soap_pump", (fx, 0.17, zc_top + 0.04), 0.014, 0.04, material="mat_chrome"))
    # window above the sink (frame + glass showing daylight)
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
    # countertop items (static, away from the robot): a dish rack and a cutting board
    side_items = 1.0 if rng.random() < 0.5 else -1.0
    ry = side_items * (bw2 + 0.25)
    g.append(_box("counter_dishrack_base", (xf + 0.10, ry - 0.16, zc_top), (xf + 0.42, ry + 0.16, zc_top + 0.02), material="mat_plastic_grey"))
    for i in range(5):
        yy = ry - 0.12 + 0.06 * i
        g.append(_cyl(f"counter_plate_{i}", (xf + 0.26, yy, zc_top + 0.12), 0.10, 0.004, material="mat_ceramic", quat=Q_ALONG_Y))
    g.append(_box("counter_board", (xf + 0.20, -ry - 0.17, zc_top), (xf + 0.46, -ry + 0.11, zc_top + 0.018), material="mat_wood"))
    # under the sink, at the back: hot / cold supply lines + shut-off valves (static)
    for nm, yy, col in (("hot", 0.035, "mat_valve_red"), ("cold", -0.035, "mat_valve_blue")):
        g.append(_cyl(f"supply_{nm}_valve", (xw - 0.035, yy, 0.42), 0.012, 0.035, material="mat_chrome", quat=Q_ALONG_X))
        g.append(_cyl(f"supply_{nm}_knob", (xw - 0.075, yy, 0.42), 0.016, 0.006, material=col, quat=Q_ALONG_X))
        g.append(_cyl(f"supply_{nm}_line", (xw - 0.05, yy, (0.43 + zc_bot) / 2 + 0.01), 0.006, (zc_bot - 0.43) / 2 - 0.01, material="mat_braid"))
    body = "".join(g)
    door = " ".join(_f(v) for v in c.door)
    ctr = " ".join(_f(v) for v in c.counter)
    inset = " ".join(_f(max(0.0, v - 0.06)) for v in c.door)
    return f"""<mujoco model="pc_sink_blind_c{cell_index}">
  <!-- GENERATED by molmo_spaces.tasks.pc_sink_blind.write_pc_sink_blind_scenes (cell {cell_index}).
       Kitchen sink base: face front x={xf:.3f}, countertop top z={c.z_ct:.3f}, door opening top
       (false-front bottom) z={c.lintel:.3f}, bowl floor z={z_bb:.3f}. Movable items, the bottle,
       the mat and the hidden drain plumbing are injected at runtime. Do not edit by hand. -->
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
    <material name="mat_sink" rgba="0.72 0.73 0.75 1" specular="0.6" reflectance="0"/>
    <material name="mat_chrome" rgba="0.80 0.81 0.83 1" specular="0.8" reflectance="0"/>
    <material name="mat_steel" rgba="0.74 0.75 0.77 1" specular="0.5" reflectance="0"/>
    <material name="mat_steel_dark" rgba="0.22 0.22 0.24 1" specular="0.3" reflectance="0"/>
    <material name="mat_handle" rgba="0.60 0.61 0.63 1" specular="0.6" reflectance="0"/>
    <material name="mat_glass" rgba="0.80 0.88 0.95 1" emission="0.35" specular="0.2" reflectance="0"/>
    <material name="mat_plastic_grey" rgba="0.55 0.56 0.58 1" specular="0.2" reflectance="0"/>
    <material name="mat_ceramic" rgba="0.94 0.93 0.90 1" specular="0.4" reflectance="0"/>
    <material name="mat_wood" rgba="0.70 0.52 0.33 1" specular="0.1" reflectance="0"/>
    <material name="mat_valve_red" rgba="0.80 0.12 0.10 1" reflectance="0"/>
    <material name="mat_valve_blue" rgba="0.12 0.25 0.75 1" reflectance="0"/>
    <material name="mat_braid" rgba="0.70 0.70 0.70 1" specular="0.4" reflectance="0"/>
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


def write_pc_sink_blind_scenes() -> list[str]:
    """(Re)generate the eight cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_SINK_BLIND_SCENES[k])
        p.write_text(build_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------
# Hidden plumbing: geometry in the drain body frame (origin = drain centre on the bowl floor top)
# --------------------------------------------------------------------------------------------
HAZ_BODY = "pact_intrusion_drain"            # prefix of both plumbing bodies
HAZ_BODIES = {1: HAZ_BODY + "_left", -1: HAZ_BODY + "_right"}
SIDE_RUN = 0.24          # sideways run of the waste pipe from the drain toward the cabinet side


def plumbing_parts(c: SinkCell, side: int) -> list[dict[str, Any]]:
    """Collision + visual primitives of one drain assembly, local to the drain centre (on the bowl
    floor top); side = +1 (drain left of the lane, pipe runs to +y) / -1 (mirror).
    Strainer -> short tailpiece -> elbow -> waste pipe SIDEWAYS (toward the cabinet side, at the
    skinned wrist height) -> elbow -> waste pipe BACK along the side -> P-trap near the back wall
    -> trap arm into the wall. kind: capsule (a, b, r) / cylinder (c, axis, r, h) / sphere (c, r)."""
    hz = cell_heights(c)
    z_bb = hz["z_bb"]
    zp = hz["z_pipe"] - z_bb                         # pipe axis (negative)
    x_wall = CAB_D - D_DRAIN                         # back wall, local x
    x_trap = x_wall - TRAP_FROM_WALL                 # P-trap inlet leg, local x
    ys = side * SIDE_RUN                             # back run, local y
    zs0, zs1 = -SS_T, -SS_T - 0.040                  # strainer body (under the bowl floor plate)
    parts = [
        # visual drain ring + basket in the bowl floor (above the floor plate, contactless)
        dict(name="flange", kind="cylinder", c=[0, 0, 0.0008], axis="z", r=0.046, h=0.0008, vis_only=True, mat="chrome"),
        dict(name="basket", kind="cylinder", c=[0, 0, 0.0012], axis="z", r=0.036, h=0.0006, vis_only=True, mat="dark"),
        dict(name="strainer", kind="cylinder", c=[0, 0, (zs0 + zs1) / 2], axis="z", r=R_STRAINER, h=(zs0 - zs1) / 2, mat="chrome"),
        dict(name="nut_tail", kind="cylinder", c=[0, 0, zs1 - 0.006], axis="z", r=0.030, h=0.008, mat="pvc_fit"),
        dict(name="tailpiece", kind="capsule", a=[0, 0, zs1], b=[0, 0, zp], r=0.020, mat="pvc"),
        dict(name="elbow_a", kind="sphere", c=[0, 0, zp], r=R_PIPE + 0.004, mat="pvc_fit"),
        dict(name="side_run", kind="capsule", a=[0, 0, zp], b=[0, ys, zp], r=R_PIPE, mat="pvc"),
        dict(name="elbow_b", kind="sphere", c=[0, ys, zp], r=R_PIPE + 0.004, mat="pvc_fit"),
        dict(name="back_run", kind="capsule", a=[0, ys, zp], b=[x_trap, ys, zp], r=R_PIPE, mat="pvc"),
        dict(name="nut_trap", kind="cylinder", c=[x_trap, ys, zp - 0.010], axis="z", r=0.031, h=0.010, mat="pvc_fit"),
        dict(name="trap_in", kind="capsule", a=[x_trap, ys, zp], b=[x_trap, ys, zp - TRAP_DIP], r=R_PIPE, mat="pvc"),
        dict(name="trap_u", kind="capsule", a=[x_trap, ys, zp - TRAP_DIP], b=[x_trap + TRAP_BACK, ys, zp - TRAP_DIP], r=R_PIPE, mat="pvc"),
        dict(name="trap_out", kind="capsule", a=[x_trap + TRAP_BACK, ys, zp - TRAP_DIP], b=[x_trap + TRAP_BACK, ys, zp - 0.012], r=R_PIPE, mat="pvc"),
        dict(name="trap_arm", kind="capsule", a=[x_trap + TRAP_BACK, ys, zp - 0.012], b=[x_wall + 0.015, ys, zp - 0.012], r=R_PIPE, mat="pvc"),
        dict(name="escutcheon", kind="cylinder", c=[x_wall - 0.002, ys, zp - 0.012], axis="x", r=0.045, h=0.002, vis_only=True, mat="chrome"),
    ]
    return parts


_HAZ_RGBA = dict(chrome=[0.78, 0.79, 0.81, 1.0], dark=[0.10, 0.10, 0.11, 1.0], pvc=[0.93, 0.93, 0.91, 1.0],
                 pvc_fit=[0.86, 0.86, 0.84, 1.0])


def _axis_quat(axis: str) -> list[float]:
    return {"z": [1.0, 0.0, 0.0, 0.0], "x": list(Q_ALONG_X), "y": list(Q_ALONG_Y)}[axis]


PARK_HAZ = {1: (-3.0, -1.6, 0.3), -1: (-3.0, -1.0, 0.3)}


def add_plumbing_body(spec: MjSpec, c: SinkCell, side: int):
    name = HAZ_BODIES[side]
    b = spec.worldbody.add_body(name=name, mocap=True, pos=list(PARK_HAZ[side]))
    for p in plumbing_parts(c, side):
        kw = dict(rgba=_HAZ_RGBA[p["mat"]])
        if p.get("vis_only"):
            kw.update(contype=0, conaffinity=0)
        else:
            kw.update(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
        if p["kind"] == "capsule":
            a, bb = np.asarray(p["a"], float), np.asarray(p["b"], float)
            mid, v = (a + bb) / 2, bb - a
            L = float(np.linalg.norm(v))
            q = _quat_z_to(v / L)
            b.add_geom(name=f"{name}_{p['name']}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       size=[p["r"], L / 2, 0.0], pos=mid.tolist(), quat=q, **kw)
        elif p["kind"] == "cylinder":
            b.add_geom(name=f"{name}_{p['name']}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[p["r"], p["h"], 0.0], pos=list(p["c"]), quat=_axis_quat(p["axis"]), **kw)
        else:
            b.add_geom(name=f"{name}_{p['name']}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[p["r"], 0.0, 0.0], pos=list(p["c"]), **kw)
    return b


def _quat_z_to(v: np.ndarray) -> list[float]:
    z = np.array([0.0, 0.0, 1.0])
    v = np.asarray(v, float) / np.linalg.norm(v)
    ax = np.cross(z, v)
    s = float(np.linalg.norm(ax))
    if s < 1e-9:
        return [1.0, 0.0, 0.0, 0.0] if v[2] > 0 else [0.0, 1.0, 0.0, 0.0]
    ang = float(np.arctan2(s, float(z @ v)))
    return R.from_rotvec(ax / s * ang).as_quat(scalar_first=True).tolist()


def primitive_distance(data, sets: dict[str, list], geoms: list[int]) -> dict[str, float]:
    """Min distance (m, 0 when touching/inside) of each robot body's collision points to the given
    hazard geoms (capsule / cylinder / sphere / box), signed-distance-function exact per point."""
    m = data.model
    out = {}
    pts = {key: np.concatenate([data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T for g, v in L], 0)
           for key, L in sets.items()}
    for key, P in pts.items():
        best = 9.0
        for h in geoms:
            C = data.geom_xpos[h]
            Rm = data.geom_xmat[h].reshape(3, 3)
            loc = (P - C) @ Rm
            sz = m.geom_size[h]
            t = int(m.geom_type[h])
            if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                zc = np.clip(loc[:, 2], -sz[1], sz[1])
                d = np.linalg.norm(loc - np.stack([np.zeros_like(zc), np.zeros_like(zc), zc], 1), axis=1) - sz[0]
            elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                q = np.stack([np.linalg.norm(loc[:, :2], axis=1) - sz[0], np.abs(loc[:, 2]) - sz[1]], 1)
                d = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
            elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                d = np.linalg.norm(loc, axis=1) - sz[0]
            else:
                q = np.abs(loc) - sz
                d = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
            best = min(best, float(max(d.min(), 0.0)))
        out[key] = best
    return out


# --------------------------------------------------------------------------------------------
# Start pose
# --------------------------------------------------------------------------------------------
READY_TCP = (0.30, 0.0, 0.62)
# IK of READY_TCP x blind_tool_rotation(0) with link0 at H_BASE (elbow-up family)
PC_SINK_BLIND_READY_QPOS: list[float] = [0.0, -0.7545, 0.0, -2.2470, 0.0, 1.4926, -1.5707]


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
ITEMS = (  # kind, radius / half extents, height, rgba, cap rgba
    ("bottle", 0.045, 0.24, (0.20, 0.55, 0.85), (0.95, 0.95, 0.95)),   # spray cleaner
    ("bottle", 0.055, 0.22, (0.95, 0.95, 0.92), (0.20, 0.40, 0.80)),   # detergent jug
    ("bottle", 0.035, 0.20, (0.90, 0.75, 0.15), (0.20, 0.20, 0.22)),
    ("bottle", 0.040, 0.18, (0.75, 0.20, 0.25), (0.95, 0.95, 0.95)),
    ("box", (0.06, 0.09), 0.14, (0.20, 0.50, 0.30), None),              # dishwasher pods
    ("box", (0.05, 0.07), 0.06, (0.95, 0.85, 0.20), None),              # sponge pack
    ("box", (0.08, 0.10), 0.05, (0.30, 0.35, 0.40), None),              # trash-bag box
    ("bucket", 0.075, 0.15, (0.25, 0.45, 0.70), None),
)


class PCSinkBlindSampler(PCShelfPnPSampler):
    """Bottle at the back of the sink-base floor, mat at the front, hidden drain left or right."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_SINK_BLIND_ENV_VERSION
    SUITE = "sink_blind"
    BASE_JIT_XY = (0.012, 0.015)
    BASE_JIT_YAW = 0.02
    CAN_D = (0.275, 0.300)        # bottle centre behind the face front
    CAN_Y = (-0.03, 0.03)
    MAT_D = (0.115, 0.140)
    MAT_DY = 0.02
    PLACE_TOL = 0.05
    # pipe inner edge beside the lane centre, per side (+1 = left / +y, -1 = right / -y). Kinematic
    # sweep 2026-10-14 (8 cells x 3 draws, skinned points in the pipe / strainer zones, cm): the arm's
    # lateral extent there is [-5.1, +5.3] untilted and shifts ~5.5 cm per 10 deg of tilt. The pipe
    # runs SIDEWAYS from the drain to the cabinet side (SIDE_RUN), so every tilt that swings the wrist
    # toward the drain side by more than the gap hits it (a straight-back pipe let +-20 deg pass
    # outside it). With these gaps every fixed tilt hits at least one side (both sides for about
    # -2.5..0 deg), and the expert's 10 deg tilt away clears its pipe by ~3-4 cm.
    PIPE_GAP = {1: (0.032, 0.036), -1: (0.022, 0.026)}
    DRAIN_DX = 0.005
    ITEM_LANE = 0.18              # items keep their footprint this far off the lane centre (low items 4 cm less)
    MAX_DRAWS = 40

    @property
    def cell(self) -> SinkCell:  # type: ignore[override]
        return PC_SINK_BLIND_CELLS[self.current_house_index % N_CELLS]

    def _target_dims(self, cell_index: int) -> dict[str, Any]:
        r, h, rgba, cap, label = BOTTLES[PC_SINK_BLIND_CELLS[cell_index].bottle]
        return dict(radius=r, height=h, rgba=rgba, cap=cap, label=label)

    def _dyn_pool(self, cell_index: int) -> list[dict[str, Any]]:
        pool = []
        for i, (kind, dims, h, rgba, cap) in enumerate(ITEMS):
            if kind == "box":
                half = (dims[0], dims[1], h / 2)
            else:
                half = (dims, dims, h / 2)
            pool.append(dict(name=f"pact_clutter_sk{i}", kind=kind, half=half, rgba=rgba, cap=cap))
        return pool

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

        PickTaskSampler.add_auxiliary_objects(self, spec)
        k = self.current_house_index % N_CELLS
        self._house_cell = k
        self._dyn = self._dyn_pool(k)
        px, py, pz = self.PARK_XYZ
        for i, it in enumerate(self._dyn):
            b = spec.worldbody.add_body(name=it["name"], mocap=True, pos=[px, py + 0.3 * i, pz + it["half"][2] + 0.001])
            rgba = list(it["rgba"]) + [1.0]
            hx, hy, hz = it["half"]
            kw = dict(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
            if it["kind"] == "box":
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], rgba=rgba, **kw)
            elif it["kind"] == "bucket":
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx, hz, 0.0], rgba=rgba, **kw)
                b.add_geom(name=it["name"] + "_rim", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx * 1.04, 0.006, 0.0],
                           pos=[0, 0, hz - 0.006], rgba=[0.2, 0.35, 0.55, 1.0], contype=0, conaffinity=0)
            else:
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx, hz, 0.0], rgba=rgba, **kw)
                b.add_geom(name=it["name"] + "_neck", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx * 0.35, 0.02, 0.0],
                           pos=[0, 0, hz + 0.02], rgba=list(it["cap"]) + [1.0], **kw)
        self._add_task_bodies(spec, k)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        td = self._target_dims(cell_index)
        self._target_spec = td
        r, h = td["radius"], td["height"]
        ns = "cavity_obj_0/"
        b = spec.worldbody.add_body(name=ns + "sink_bottle", pos=[-3.0, -2.6, h / 2 + 0.01])
        b.add_joint(name=ns + "sink_bottle_free", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.01)
        b.add_geom(name=ns + "sink_bottle_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r, h / 2, 0.0], rgba=list(td["rgba"]) + [1.0], mass=0.25,
                   contype=8, conaffinity=15, condim=4, friction=[1.0, 0.02, 0.002], priority=1)
        b.add_geom(name=ns + "sink_bottle_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r * 0.45, 0.006, 0.0], pos=[0.0, 0.0, h / 2 + 0.006],
                   rgba=list(td["cap"]) + [1.0], mass=0.001, contype=0, conaffinity=0)
        b.add_geom(name=ns + "sink_bottle_label", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r * 1.004, h * 0.18, 0.0], pos=[0.0, 0.0, -h * 0.08],
                   rgba=[0.96, 0.96, 0.94, 1.0], mass=0.001, contype=0, conaffinity=0)
        self._injected_obj_name = ns + "sink_bottle"
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
            add_plumbing_body(spec, PC_SINK_BLIND_CELLS[cell_index], sd)

    def _place_robot(self, env) -> None:
        jx, jy = self.BASE_JIT_XY
        self._cur_base_xyz = (float(np.random.uniform(-jx, jx)), float(np.random.uniform(-jy, jy)), 0.0)
        self._cur_base_yaw = float(np.random.uniform(-self.BASE_JIT_YAW, self.BASE_JIT_YAW))
        robot_view = env.current_robot.robot_view
        q = R.from_euler("z", self._cur_base_yaw).as_quat(scalar_first=True)
        robot_view.base.pose = pos_quat_to_pose_mat(np.array(self._cur_base_xyz), np.array(q))
        mujoco.mj_forward(env.current_model, env.current_data)

    def _cell_common(self) -> dict[str, Any]:
        c = self.cell
        hz = cell_heights(c)
        return dict(x_f=c.x_f, x0=c.x_f, z_ct=c.z_ct, lintel=c.lintel, cab_w=c.cab_w, bowl_w=c.bowl_w,
                    floor_top=FLOOR_TOP, z_carry=hz["z_hi"], z_pipe=hz["z_pipe"], z_bowl_floor=hz["z_bb"],
                    bowl_depth=hz["bowl_depth"], board_depth=CAB_D, h_base=H_BASE)

    def drain_origin(self, c: SinkCell, y_c: float, side: int, gap: float, dx: float = 0.0) -> list[float]:
        hz = cell_heights(c)
        y_in = y_c + side * gap
        return [c.x_f + D_DRAIN + dx, y_in + side * R_PIPE, hz["z_bb"]]

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
        # hidden drain: side drawn independently of everything visible
        side = 1 if np.random.random() < 0.5 else -1          # +1 = left (+y), -1 = right (-y)
        gap = float(np.random.uniform(*self.PIPE_GAP[side]))
        ddx = float(np.random.uniform(-self.DRAIN_DX, self.DRAIN_DX))
        drain = self.drain_origin(c, yc, side, gap, ddx)
        tilt = TILT_DEG * side       # fingers lean toward the drain side -> wrist swings away
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
        # items on the cabinet floor, off the reach lane
        out: list = []
        placed: list = [(xc, yc, r + 0.03, r + 0.03), (xm, ym, MAT_HALF + 0.02, MAT_HALF + 0.02)]
        lim = c.cab_w / 2 - SIDE_T - 0.01
        avail = list(range(len(self._dyn)))
        np.random.shuffle(avail)
        n_items = int(np.random.randint(3, 6))
        for i in avail:
            if len(out) >= n_items:
                break
            hx, hy, hzz = self._dyn[i]["half"]
            lane = self.ITEM_LANE if 2 * hzz > 0.08 else self.ITEM_LANE - 0.04
            for _ in range(25):
                s_ = 1.0 if np.random.random() < 0.5 else -1.0
                if s_ > 0:
                    y_lo, y_hi = yc + lane + hy, lim - hy
                else:
                    y_lo, y_hi = -lim + hy, yc - lane - hy
                if y_hi <= y_lo:
                    continue
                y = float(np.random.uniform(y_lo, y_hi))
                x = c.x_f + FF_T + float(np.random.uniform(0.02 + hx, CAB_D - FF_T - 0.02 - hx))
                fp = (x, y, hx, hy)
                if any(abs(x - p[0]) < hx + p[2] + 0.01 and abs(y - p[1]) < hy + p[3] + 0.01 for p in placed):
                    continue
                placed.append(fp)
                out.append((i, [x, y, z_b + hzz + 0.0005]))
                break
        self._apply_layout(env, out)
        for i, pos in out:
            q = R.from_euler("z", float(np.random.uniform(0, 2 * np.pi))).as_quat(scalar_first=True)
            self._mocap(env, self._dyn[i]["name"], pos, q)
        self._mocap(env, "place_receptacle_mat", [xm, ym, z_b + MAT_HALF_T + 0.0003])
        for sd in (1, -1):
            self._mocap(env, HAZ_BODIES[sd], PARK_HAZ[sd])
        self._mocap(env, HAZ_BODIES[side], drain)
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
            neighbours=[[self._dyn[i]["name"], self._dyn[i]["kind"], pos,
                         list(self._dyn[i]["half"])] for i, pos in out],
            n_neighbours=len(out),
            # evaluator contract (eval_act_place.py writes hidden_hazard_present / _y / _side)
            hidden_hazard=dict(present=True, side=side_name, y=float(drain[1]), x=float(drain[0]),
                               z_bottom=float(hz["pipe_bottom"]), body=HAZ_BODIES[side]),
            # diagnostic free camera (reports/2026-10-13/video_blind.py): from the back corner of
            # the cabinet on the dry side, looking at the drain pipe and the arm
            reveal_view=dict(pos=[c.x_f + 0.50, yc - side * 0.30, FLOOR_TOP + 0.30],
                             lookat=[c.x_f + 0.22, yc + side * 0.04, hz["z_pipe"] - 0.03], fovy=80.0,
                             title="Under the sink (not a robot camera): drain pipe in red"),
            blind=dict(side=int(side), side_name=side_name, tilt_deg=float(tilt), drain=drain,
                       pipe_gap=gap, pipe_z=hz["z_pipe"], z_carry=z_hi, cell=int(k)),
        )

    def _referral(self) -> str:
        return self._target_spec.get("label", "bottle")

    def _task_class(self):
        return PCSinkBlindTask


# --------------------------------------------------------------------------------------------
# Task: pc_shelf_blind (pnp success, hazard audit, gate) with clearance to the drain plumbing
# --------------------------------------------------------------------------------------------
class PCSinkBlindTask(PCShelfBlindTask):
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

    def get_obs_scene(self) -> dict[str, Any]:
        return super().get_obs_scene()

    def _maybe_log_obstacle_diag(self, success: bool) -> None:
        super()._maybe_log_obstacle_diag(success)


# --------------------------------------------------------------------------------------------
# Expert
# --------------------------------------------------------------------------------------------
class PCSinkBlindExpert(PCShelfBlindExpert):
    """pc_shelf_blind expert (per-waypoint hand tilt) with the dwell from scene_params['dwell_s'].

    No empty-gripper abort after the close (HOLD_AFTER_CLOSE = False): the Robotiq view measures the
    finger gap with mujoco.mj_geomDistance, which returns a spurious 0.0 on some queries in these
    multi-CCD scenes, and aborted 1 of 4 preflight episodes with the bottle firmly in the hand
    (2026-10-14 10:06). A real miss now simply fails the success rule at the end of the episode."""

    HOLD_AFTER_CLOSE = False

    def _compute_trajectory(self):
        sp = getattr(self.task, "scene_params", {}) or {}
        dwell = float(sp.get("dwell_s", DWELL_S))
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
            end = _T(p, blind_tool_rotation(float(tilt)))
            segs.append(_BlindSeg(name=name, start_pose=cur, end_pose=end,
                                  speed=float(self.SPEEDS.get(speed, 0.08)),
                                  ang_speed=self.TILT_ANG_SPEED if speed == "tilt" else 0.9,
                                  min_dur=dwell if speed == "dwell" else 0.0))
            cur = end
        if segs:
            prims.append(self._seq(segs, holding))
        return prims


class PCSinkBlindExpertConfig(PCShelfBlindExpertConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCSinkBlindExpert


__all__ = [
    "PC_SINK_BLIND_ENV_VERSION", "PC_SINK_BLIND_CELLS", "PC_SINK_BLIND_SCENES", "PC_SINK_BLIND_READY_QPOS",
    "PCSinkBlindSampler", "PCSinkBlindTask", "PCSinkBlindExpert", "PCSinkBlindExpertConfig",
    "write_pc_sink_blind_scenes", "build_scene_xml", "cell_heights", "plumbing_parts",
    "primitive_distance", "H_BASE", "HAZ_BODY", "HAZ_BODIES", "PARK_HAZ", "READY_TCP", "N_CELLS",
]


if __name__ == "__main__":
    import sys

    if "--write-scenes" in sys.argv:
        for p in write_pc_sink_blind_scenes():
            print(p)
