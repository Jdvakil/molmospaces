"""Realistic KITCHEN suites for the proximity-skin FR3.

Two tasks share one kitchen (eight layout cells, one compiled scene per cell):

* ``pc_kitchen_pnp``  (prehensile): take a tall cup that stands on the counter UNDER the upper wall
  cabinets, in the pocket between the backsplash, a standard-depth fridge on the left and the
  canisters / bottles that stand along the wall, and put it on a wooden tray further right on the
  same counter.
* ``pc_kitchen_push`` (non-prehensile): slide a round biscuit tin with the CLOSED gripper from the
  front of the counter back under the cabinet edge, along a lane between the fridge and the
  counter items, onto a flat silicone trivet (a contactless visible decal) that marks the goal.

Scene (world frame, metres). The FR3 stands on a 0.28 m platform (link0 at z = 0.63) in front of a
0.90 m counter run along +y; robot forward = +x:

    x = 0.26   counter front edge        x = 0.86   backsplash / kitchen wall
    z = 0.90   counter top               z_b = 0.90 + gap   upper-cabinet bottom (gap 0.43 | 0.49)
    upper-cabinet front face at 0.86 - depth (depth 0.31 | 0.35), door handles 3 cm proud of it
    fridge (0.84 deep, 0.80 wide, 1.80 tall) on the left; its right side panel at y_f (0.33 | 0.39)
    cooktop + range hood at the far right, room walls 0.1-2.6 m away on every side, ceiling 2.50

The eight cells are every combination of (fridge side y_f, cabinet gap, cabinet depth); each cell
is its own XML (``pc_kitchen_c<k>.xml``), so ``house_index % 8`` selects the geometry and the same
seed always rebuilds the same scene. Counter items are real THOR meshes and simple ceramic
canisters, all MOCAP bodies (static, re-posed per episode, z pinned to counter + their own rest
height, so nothing ever floats); candidate poses are rejection-sampled away from the expert's
nominal TCP + hand path so every demonstration stays contact-free while the skin is loaded.

Contact naming follows the PACT audit (``pact_contact_audit.py``): target root body
``cavity_obj_*`` (task contact), ``place_receptacle`` (receptacle), ``pact_clutter_*`` (clutter
hazard), all other scene geometry is ``other_environment`` (hazard).

Regenerate the scene files after editing the geometry::

    python -m molmo_spaces.tasks.pc_kitchen --write-scenes
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import PickPlannerPolicy
from molmo_spaces.tasks.cavity_pick_task_sampler import CavityPickTaskSampler
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.object_metadata import ObjectMeta

log = logging.getLogger(__name__)

SCENE_DIR = Path(__file__).resolve().parents[1] / "data_generation" / "custom_scenes"

# ------------------------------------------------------------------------------------------------
# Geometry shared by the XML generator, the samplers and the experts.
# ------------------------------------------------------------------------------------------------
COUNTER_Z = 0.90            # counter top surface
COUNTER_FRONT_X = 0.26      # front edge of the counter slab
WALL_X = 0.86               # kitchen wall face
BACKSPLASH_X = 0.852        # tile face (items standing against the wall stop here)
ROOM_BACK_X = -1.55         # wall behind the robot
ROOM_RIGHT_Y = -1.75        # wall on the robot's right
CEILING_Z = 2.50
# Robot stand: a 0.28 m platform under the standard 0.35 m pedestal puts link0 at z = 0.63 and the
# FR3 ready-pose TCP at z ~1.04, 14 cm above the counter. (On a 0.18 m platform the ready pose put
# the fingertips 2 cm above the counter and the first approach dragged the hand across it.)
PLATFORM_TOP = 0.277        # robot platform; the pedestal bottom sits 3 mm above it (no contact)
BASE_XYZ = (0.0, 0.0, 0.28) # robot base (pedestal bottom); link0 at z = 0.63
FRIDGE_FRONT_X = 0.02
FRIDGE_W = 0.80
FRIDGE_H = 1.80
COOKTOP_Y = (-1.46, -0.86)

N_CELLS = 8
_FRIDGE_Y = (0.33, 0.39)    # fridge right side panel (faces the work area)
_CAB_GAP = (0.43, 0.49)     # upper-cabinet bottom above the counter
_CAB_DEPTH = (0.31, 0.35)   # upper-cabinet depth (front face at WALL_X - depth)


def cell_params(cell: int) -> dict[str, float]:
    """Layout of one of the eight cells: every (fridge, gap, depth) combination."""
    cell = int(cell) % N_CELLS
    fy = _FRIDGE_Y[cell & 1]
    gap = _CAB_GAP[(cell >> 1) & 1]
    depth = _CAB_DEPTH[(cell >> 2) & 1]
    return {
        "cell": cell,
        "fridge_y": fy,
        "cab_gap": gap,
        "cab_depth": depth,
        "cab_bottom_z": COUNTER_Z + gap,
        "cab_front_x": WALL_X - depth,
        "left_wall_y": fy + FRIDGE_W + 0.04,
    }


def scene_path(cell: int) -> Path:
    return SCENE_DIR / f"pc_kitchen_c{int(cell) % N_CELLS}.xml"


def scene_paths(n: int) -> list[str]:
    """Scene list for ``scene_xml_paths``: index i -> cell i % 8 (wrap-around parallelism)."""
    return [str(scene_path(i % N_CELLS)) for i in range(n)]


# ------------------------------------------------------------------------------------------------
# XML generator
# ------------------------------------------------------------------------------------------------
def _box(name, lo, hi, cls="furn", material=None, extra=""):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    mat = f' material="{material}"' if material else ""
    return (
        f'    <geom name="{name}" class="{cls}" type="box" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" '
        f'size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}"{mat}{extra}/>\n'
    )


def _panels(y_lo, y_hi, width, gap=0.004):
    """Split [y_lo, y_hi] into door panels of roughly ``width`` with visible gaps."""
    n = max(1, int(round((y_hi - y_lo) / width)))
    w = (y_hi - y_lo) / n
    return [(y_lo + i * w + gap / 2, y_lo + (i + 1) * w - gap / 2) for i in range(n)]


def build_scene_xml(cell: int, params: dict[str, float] | None = None, model: str = "pc_kitchen") -> str:
    # params / model: the hard-suite generator (pc_kitchen_hard_c<k>.xml) passes its own cell
    # layout; the defaults rebuild the original pc_kitchen_c<k>.xml byte for byte.
    p = cell_params(cell) if params is None else params
    fy, zb, xf, lwy = p["fridge_y"], p["cab_bottom_z"], p["cab_front_x"], p["left_wall_y"]
    zt = zb + 0.72                       # upper-cabinet top
    yr = ROOM_RIGHT_Y
    cx_room = (ROOM_BACK_X + WALL_X) / 2
    hx_room = (WALL_X - ROOM_BACK_X) / 2 + 0.06
    cy_room = (yr + lwy) / 2
    hy_room = (lwy - yr) / 2 + 0.06
    g = []
    g.append(f'    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 0" material="mat_floor" contype="8" conaffinity="15"/>\n')
    # room shell (every skin ray terminates on a real surface well inside 3.5 m)
    g.append(_box("wall_back", (ROOM_BACK_X - 0.06, yr - 0.06, 0), (ROOM_BACK_X, lwy + 0.06, CEILING_Z), material="mat_wall"))
    g.append(_box("wall_kitchen", (WALL_X, yr - 0.06, 0), (WALL_X + 0.06, lwy + 0.06, CEILING_Z), material="mat_wall"))
    g.append(_box("wall_left", (ROOM_BACK_X, lwy, 0), (WALL_X, lwy + 0.06, CEILING_Z), material="mat_wall"))
    g.append(_box("wall_right", (ROOM_BACK_X, yr - 0.06, 0), (WALL_X, yr, CEILING_Z), material="mat_wall"))
    g.append(_box("ceiling", (ROOM_BACK_X - 0.06, yr - 0.06, CEILING_Z), (WALL_X + 0.06, lwy + 0.06, CEILING_Z + 0.05), material="mat_ceiling"))
    g.append(_box("baseboard_back", (ROOM_BACK_X, yr, 0), (ROOM_BACK_X + 0.015, lwy, 0.08), material="mat_cab_body"))
    g.append(_box("baseboard_right", (ROOM_BACK_X, yr, 0), (0.30, yr + 0.015, 0.08), material="mat_cab_body"))
    # robot platform (the pedestal stands on it with a 3 mm gap)
    g.append(_box("robot_platform", (-0.25, -0.25, 0.0), (0.25, 0.25, PLATFORM_TOP), material="mat_platform"))
    # ---------------- base cabinets + counter ----------------
    g.append(_box("base_cab_carcass", (0.30, yr, 0.10), (WALL_X, fy, 0.86), material="mat_cab_body"))
    g.append(_box("base_cab_toekick", (0.36, yr, 0.0), (WALL_X, fy, 0.10), material="mat_kick"))
    for i, (a, b) in enumerate(_panels(yr, fy, 0.50)):
        g.append(_box(f"base_door_{i}", (0.282, a, 0.11), (0.300, b, 0.70), material="mat_cab_door"))
        g.append(_box(f"base_drawer_{i}", (0.282, a, 0.716), (0.300, b, 0.852), material="mat_cab_door"))
        ym = (a + b) / 2
        g.append(_box(f"base_drawer_handle_{i}", (0.268, ym - 0.08, 0.778), (0.282, ym + 0.08, 0.790), material="mat_handle"))
        g.append(_box(f"base_door_handle_{i}", (0.268, b - 0.05, 0.56), (0.282, b - 0.038, 0.66), material="mat_handle"))
    g.append(_box("counter_top", (COUNTER_FRONT_X, yr, 0.86), (WALL_X, fy, COUNTER_Z), material="mat_counter"))
    g.append(_box("backsplash", (WALL_X - 0.008, yr, COUNTER_Z), (WALL_X, fy, zb), material="mat_tile"))
    # cooktop (proud black glass) + burner rings (flat, contactless print on the glass)
    g.append(_box("cooktop", (0.38, COOKTOP_Y[0], COUNTER_Z), (0.82, COOKTOP_Y[1], COUNTER_Z + 0.006), material="mat_cooktop"))
    for i, (bx, by, r) in enumerate(((0.50, -1.30, 0.085), (0.50, -1.02, 0.07), (0.70, -1.30, 0.07), (0.70, -1.02, 0.085))):
        g.append(f'    <geom name="burner_{i}" type="cylinder" pos="{bx} {by} {COUNTER_Z + 0.0065}" size="{r} 0.0005" material="mat_burner" contype="0" conaffinity="0" group="0"/>\n')
    # ---------------- upper cabinets ----------------
    g.append(_box("upper_cab_carcass", (xf, yr, zb), (WALL_X, fy, zt), material="mat_cab_body"))
    g.append(_box("upper_cab_soffit", (xf, yr, zt), (WALL_X, fy, CEILING_Z), material="mat_wall"))
    for i, (a, b) in enumerate(_panels(yr, fy, 0.42)):
        g.append(_box(f"upper_door_{i}", (xf - 0.018, a, zb + 0.004), (xf, b, zt - 0.004), material="mat_cab_door"))
        # bar handle near the bottom of each door, 3 cm proud of the face (a real hazard)
        g.append(_box(f"upper_door_handle_{i}", (xf - 0.030, b - 0.045, zb + 0.035), (xf - 0.018, b - 0.033, zb + 0.145), material="mat_handle"))
    g.append(_box("undercab_led", (xf + 0.02, yr, zb - 0.006), (xf + 0.045, fy, zb), material="mat_led"))
    # range hood under the cabinets above the cooktop
    g.append(_box("range_hood", (WALL_X - 0.50, COOKTOP_Y[0], zb - 0.12), (WALL_X, COOKTOP_Y[1], zb), material="mat_steel"))
    # ---------------- fridge (left flank) ----------------
    g.append(_box("fridge_body", (FRIDGE_FRONT_X, fy, 0.0), (WALL_X, fy + FRIDGE_W, FRIDGE_H), material="mat_steel"))
    g.append(_box("fridge_door_lower", (FRIDGE_FRONT_X - 0.025, fy + 0.004, 0.03), (FRIDGE_FRONT_X, fy + FRIDGE_W - 0.004, 1.185), material="mat_steel_door"))
    g.append(_box("fridge_door_upper", (FRIDGE_FRONT_X - 0.025, fy + 0.004, 1.205), (FRIDGE_FRONT_X, fy + FRIDGE_W - 0.004, FRIDGE_H - 0.01), material="mat_steel_door"))
    g.append(_box("fridge_handle_lower", (FRIDGE_FRONT_X - 0.050, fy + 0.050, 0.78), (FRIDGE_FRONT_X - 0.025, fy + 0.070, 1.10), material="mat_handle"))
    g.append(_box("fridge_handle_upper", (FRIDGE_FRONT_X - 0.050, fy + 0.050, 1.26), (FRIDGE_FRONT_X - 0.025, fy + 0.070, 1.52), material="mat_handle"))
    g.append(_box("fridge_kick", (FRIDGE_FRONT_X - 0.02, fy + 0.02, 0.0), (FRIDGE_FRONT_X, fy + FRIDGE_W - 0.02, 0.03), material="mat_kick"))
    g.append(_box("over_fridge_cab", (0.26, fy, 1.86), (WALL_X, fy + FRIDGE_W, 2.26), material="mat_cab_body"))
    g.append(_box("over_fridge_door", (0.242, fy + 0.004, 1.864), (0.26, fy + FRIDGE_W - 0.004, 2.256), material="mat_cab_door"))
    g.append(_box("over_fridge_soffit", (0.26, fy, 2.26), (WALL_X, fy + FRIDGE_W, CEILING_Z), material="mat_wall"))
    geoms = "".join(g)

    lights = (
        f'    <light name="ceil_main" pos="-0.25 {(fy + yr) / 2 + 0.3:.3f} 2.42" dir="0.35 0 -1" diffuse="0.62 0.61 0.58" specular="0.15 0.15 0.15" castshadow="true"/>\n'
        f'    <light name="ceil_fill" pos="-0.9 -0.6 2.42" dir="0.4 0.2 -1" diffuse="0.30 0.30 0.30" specular="0.05 0.05 0.05" castshadow="false"/>\n'
        f'    <light name="undercab" pos="{xf + 0.10:.3f} -0.2 {zb - 0.02:.3f}" dir="0 0 -1" diffuse="0.22 0.21 0.19" specular="0 0 0" castshadow="false" cutoff="75" exponent="2"/>\n'
    )

    movables = f"""
    <!-- ============ movable (MOCAP, re-posed per episode by the sampler; parked under the floor
         when unused). Body origins sit at the item's base centre so z = counter top. ============ -->
    <body name="place_receptacle" mocap="true" pos="-0.6 -0.6 -1.0">
      <geom name="tray_floor" class="item" type="box" pos="0 0 0.004" size="0.150 0.110 0.004" material="mat_wood"/>
      <geom name="tray_rim_l" class="item" type="box" pos="0 0.104 0.017" size="0.150 0.006 0.009" material="mat_wood"/>
      <geom name="tray_rim_r" class="item" type="box" pos="0 -0.104 0.017" size="0.150 0.006 0.009" material="mat_wood"/>
      <geom name="tray_rim_f" class="item" type="box" pos="-0.144 0 0.017" size="0.006 0.098 0.009" material="mat_wood"/>
      <geom name="tray_rim_b" class="item" type="box" pos="0.144 0 0.017" size="0.006 0.098 0.009" material="mat_wood"/>
    </body>
    <body name="push_goal_decal" mocap="true" pos="-0.6 -0.6 -1.0">
      <geom name="push_goal_decal_g" type="cylinder" pos="0 0 0.0006" size="0.078 0.0006" material="mat_decal" contype="0" conaffinity="0" group="0"/>
    </body>
    <body name="pact_clutter_can_a" mocap="true" pos="-0.6 -0.6 -1.0">
      <geom name="can_a_body" class="item" type="cylinder" pos="0 0 0.095" size="0.055 0.095" material="mat_ceramic"/>
      <geom name="can_a_lid" class="item" type="cylinder" pos="0 0 0.202" size="0.057 0.012" material="mat_lid"/>
      <geom name="can_a_knob" class="item" type="sphere" pos="0 0 0.222" size="0.014" material="mat_lid"/>
    </body>
    <body name="pact_clutter_can_b" mocap="true" pos="-0.6 -0.6 -1.0">
      <geom name="can_b_body" class="item" type="cylinder" pos="0 0 0.075" size="0.048 0.075" material="mat_ceramic"/>
      <geom name="can_b_lid" class="item" type="cylinder" pos="0 0 0.162" size="0.050 0.012" material="mat_lid"/>
      <geom name="can_b_knob" class="item" type="sphere" pos="0 0 0.182" size="0.013" material="mat_lid"/>
    </body>
    <body name="pact_clutter_can_c" mocap="true" pos="-0.6 -0.6 -1.0">
      <geom name="can_c_body" class="item" type="cylinder" pos="0 0 0.055" size="0.042 0.055" material="mat_ceramic"/>
      <geom name="can_c_lid" class="item" type="cylinder" pos="0 0 0.122" size="0.044 0.012" material="mat_lid"/>
      <geom name="can_c_knob" class="item" type="sphere" pos="0 0 0.141" size="0.012" material="mat_lid"/>
    </body>
    <body name="pact_clutter_board" mocap="true" pos="-0.6 -0.6 -1.0">
      <geom name="board_g" class="item" type="box" pos="0 0 0.20" size="0.0075 0.15 0.20" material="mat_wood_board"/>
    </body>
"""

    return f"""<mujoco model="{model}_c{p['cell']}">
  <!-- GENERATED by molmo_spaces/tasks/pc_kitchen.py (cell {p['cell']}: fridge side y={fy:.2f},
       upper-cabinet bottom z={zb:.2f} (gap {p['cab_gap']:.2f}), depth {p['cab_depth']:.2f} -> front x={xf:.2f}).
       Do not edit by hand: change build_scene_xml() and regenerate. -->
  <compiler angle="radian" autolimits="true" boundmass="0" balanceinertia="true"/>
  <option impratio="10" gravity="0 0 -9.8" integrator="implicitfast" cone="elliptic" jacobian="sparse" noslip_iterations="4">
    <flag multiccd="enable" contact="enable" warmstart="enable"/>
  </option>
  <size njmax="8000" nconmax="8000"/>
  <statistic center="0 -0.3 1.0" extent="3.0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.0005"/>
    <headlight ambient="0.36 0.36 0.36" diffuse="0.45 0.45 0.45" specular="0.08 0.08 0.08"/>
  </visual>
  <asset>
    <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.80 0.76 0.70" rgb2="0.76 0.72 0.66" mark="edge" markrgb="0.60 0.57 0.52" width="512" height="512"/>
    <material name="mat_floor" texture="tex_floor" texrepeat="6 6" reflectance="0.04"/>
    <texture name="tex_tile" type="2d" builtin="checker" rgb1="0.95 0.95 0.93" rgb2="0.92 0.92 0.90" mark="edge" markrgb="0.70 0.70 0.68" width="256" height="256"/>
    <material name="mat_tile" texture="tex_tile" texrepeat="3.3 6.6" texuniform="true" reflectance="0.08" specular="0.4"/>
    <material name="mat_wall" rgba="0.89 0.87 0.83 1" specular="0.05"/>
    <material name="mat_ceiling" rgba="0.94 0.94 0.93 1" specular="0.02"/>
    <material name="mat_counter" rgba="0.33 0.33 0.35 1" reflectance="0.10" specular="0.35"/>
    <material name="mat_cab_door" rgba="0.53 0.61 0.54 1" specular="0.15"/>
    <material name="mat_cab_body" rgba="0.92 0.91 0.88 1" specular="0.05"/>
    <material name="mat_handle" rgba="0.62 0.63 0.65 1" specular="0.6" reflectance="0.15"/>
    <material name="mat_steel" rgba="0.76 0.77 0.79 1" specular="0.5" reflectance="0.15"/>
    <material name="mat_steel_door" rgba="0.80 0.81 0.83 1" specular="0.55" reflectance="0.18"/>
    <material name="mat_kick" rgba="0.17 0.17 0.18 1"/>
    <material name="mat_platform" rgba="0.25 0.25 0.27 1" specular="0.1"/>
    <material name="mat_cooktop" rgba="0.05 0.05 0.06 1" specular="0.7" reflectance="0.25"/>
    <material name="mat_burner" rgba="0.24 0.22 0.22 1"/>
    <material name="mat_led" rgba="1.0 0.97 0.90 1" emission="0.9"/>
    <material name="mat_wood" rgba="0.66 0.48 0.31 1" specular="0.1"/>
    <material name="mat_wood_board" rgba="0.72 0.55 0.36 1" specular="0.1"/>
    <material name="mat_ceramic" rgba="0.93 0.92 0.87 1" specular="0.4"/>
    <material name="mat_lid" rgba="0.55 0.40 0.26 1" specular="0.1"/>
    <material name="mat_decal" rgba="0.66 0.15 0.12 1" specular="0.1"/>
  </asset>
  <default>
    <default class="furn">
      <geom contype="8" conaffinity="15" group="0" friction="0.9 0.9 0.001" solref="0.02 1" solimp="0.998 0.998 0.001" density="700"/>
    </default>
    <default class="item">
      <geom contype="8" conaffinity="15" group="0" friction="0.9 0.9 0.001" solref="0.02 1" solimp="0.998 0.998 0.001" density="500"/>
    </default>
  </default>
  <worldbody>
{lights}{geoms}{movables}  </worldbody>
</mujoco>
"""


def write_scenes(out_dir: Path = SCENE_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cell in range(N_CELLS):
        path = out_dir / f"pc_kitchen_c{cell}.xml"
        path.write_text(build_scene_xml(cell))
        (out_dir / f"pc_kitchen_c{cell}_metadata.json").write_text(json.dumps({"objects": {}}, indent=1) + "\n")
        paths.append(path)
    return paths


# ------------------------------------------------------------------------------------------------
# Counter items. XML items have their origin at the base centre; THOR items are measured at
# runtime from the compiled model (rest height, footprint) so they stand exactly on the counter.
# ------------------------------------------------------------------------------------------------
STAND_QUAT = R.from_euler("x", 90, degrees=True)      # THOR assets are Y-up
PARK_POS = (-0.6, -0.6, -1.0)                          # under the floor, inside the room footprint

XML_ITEMS: dict[str, dict[str, float]] = {
    # name -> footprint radius, height (base at the body origin)
    "can_a": {"r": 0.057, "h": 0.236},
    "can_b": {"r": 0.050, "h": 0.195},
    "can_c": {"r": 0.044, "h": 0.153},
}
THOR_ITEMS: dict[str, str] = {
    "oil": "Bottle_1",
    "wine": "Wine_Bottle_1",
    "soap": "Soap_Bottle_1",
    "towel": "Paper_Towel_1",
    "kettle": "Kettle_1",
    "toaster": "Toaster_1",
    "bread": "Bread_3",
    "bowl": "Bowl_5",
    "salt": "Salt_Shaker_1",
    "pepper": "Pepper_Shaker_1",
}
BOARD_LEAN = np.deg2rad(12.0)   # cutting board leaning on the backsplash
BOARD_HALF = (0.0075, 0.15, 0.20)


def clutter_body_name(item: str) -> str:
    return f"pact_clutter_{item}"


def _yaw_quat(yaw: float, stand: bool) -> np.ndarray:
    rot = R.from_euler("z", yaw)
    if stand:
        rot = rot * STAND_QUAT
    return rot.as_quat(scalar_first=True)


def _subtree_geoms(m: mujoco.MjModel, body_id: int) -> list[int]:
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        while b != 0 and b != body_id:
            b = int(m.body_parentid[b])
        if b == body_id:
            out.append(g)
    return out


def _geom_points(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> np.ndarray:
    xm = d.geom_xmat[g].reshape(3, 3)
    xp = d.geom_xpos[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return xp + v @ xm.T
    c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return xp + (c + corners * h) @ xm.T


def _aabb2_dist(pts_xy: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Horizontal distance from points to an axis-aligned rectangle (0 inside)."""
    d = np.maximum(np.maximum(lo - pts_xy, 0.0), np.maximum(pts_xy - hi, 0.0))
    return np.linalg.norm(d, axis=1)


def _densify(wps: list[np.ndarray], step: float = 0.02) -> np.ndarray:
    pts = []
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            pts.append(a + t * (b - a))
    return np.asarray(pts)


# ------------------------------------------------------------------------------------------------
# Expert geometry (shared by the samplers' keep-out test and the experts, so they always agree)
# ------------------------------------------------------------------------------------------------
HOME_TCP_EST = np.array([0.33, 0.0, 1.04])  # home TCP (FR3 ready pose on the platform)

# --- pick-and-place ---
PNP_CUP_UID = "pc_kitchen_can"    # a 33 mm-radius beverage can (primitive cylinder, real can size)
PNP_CAN_R = 0.033
PNP_CAN_H = 0.122
PNP_CUP_XY = (0.61, 0.20)         # nominal cup centre (under the cabinet, 13-19 cm from the fridge)
PNP_CUP_JIT = (0.012, 0.012)
PNP_TRAY_XY = (0.50, -0.34)       # nominal tray centre
PNP_TRAY_JIT = (0.012, 0.020)
PNP_TRAY_FLOOR = 0.008            # tray floor top above its base
PNP_GRASP_H = 0.068               # TCP height above the can bottom (just above its centre of mass)
# Approach pitched 25 deg below horizontal, toward +x: the wrist rides above the counter and in
# front of the cabinet face. (8 deg put link6/7 on the counter top during the approach.)
# The THOR tumbler first tried here (Cup_10, tapered convex-hull collision) rolled ~28 deg in the
# pads on every carry and toppled on release; a straight cylinder grips cleanly.
PNP_PITCH = np.deg2rad(25.0)
PNP_STANDOFF = 0.10               # pregrasp distance back along the approach axis
PNP_LIFT = 0.05
PNP_RETRACT_X = 0.47              # TCP x for the lateral carry (in front of the wall items)
PNP_HOVER = 0.05                  # above the place pose before the descent
PNP_RETREAT = 0.10
PNP_PLACE_TOL = 0.06              # cup centre within this of the tray centre (xy)

# --- push ---
PUSH_TIN_R = 0.065
PUSH_TIN_H = 0.080
PUSH_START_XY = (0.38, 0.20)
PUSH_START_JIT = (0.010, 0.010)
PUSH_DIST = (0.18, 0.20)          # push length range
PUSH_PITCH = np.deg2rad(35.0)     # fingers point down-forward, wrist tilted back toward the robot
PUSH_TCP_Z = 0.035                # TCP height above the counter while pushing
PUSH_FACE_DX = 0.020              # closed-finger front edge ahead of the TCP along +x (measured)
PUSH_CONTACT_R = PUSH_TIN_R - 0.002   # tin body radius at the finger height (lid is 2 mm wider)
PUSH_PRE_GAP = 0.015              # start of the stroke: face this far behind the tin
PUSH_HOVER = 0.10
PUSH_BACKOFF = 0.05
PUSH_UP = 0.12
PUSH_TOL = 0.035                  # success: tin centre within this of the goal (xy)


def pnp_approach_axis() -> np.ndarray:
    return np.array([np.cos(PNP_PITCH), 0.0, -np.sin(PNP_PITCH)])


def push_approach_axis() -> np.ndarray:
    return np.array([np.sin(PUSH_PITCH), 0.0, -np.cos(PUSH_PITCH)])


def pnp_waypoints(cup_xy, tray_xy, cup_bottom_z: float = COUNTER_Z) -> dict[str, np.ndarray]:
    a = pnp_approach_axis()
    grasp = np.array([cup_xy[0], cup_xy[1], cup_bottom_z + PNP_GRASP_H])
    lift = grasp + np.array([0.0, 0.0, PNP_LIFT])
    retract = np.array([min(PNP_RETRACT_X, grasp[0] - 0.08), grasp[1], lift[2]])
    place = np.array([tray_xy[0], tray_xy[1], COUNTER_Z + PNP_TRAY_FLOOR + PNP_GRASP_H + 0.008])
    a_xy = np.array([a[0], a[1], 0.0]) / max(np.linalg.norm(a[:2]), 1e-9)
    return {
        "pregrasp": grasp - PNP_STANDOFF * a,
        "grasp": grasp,
        "lift": lift,
        "retract": retract,
        "over_tray": place + np.array([0.0, 0.0, PNP_HOVER]),
        "place": place,
        "retreat": place - PNP_RETREAT * a_xy + np.array([0.0, 0.0, 0.03]),
    }


def push_waypoints(start_xy, goal_xy) -> dict[str, np.ndarray]:
    z = COUNTER_Z + PUSH_TCP_Z
    y = float(start_xy[1])
    face0 = float(start_xy[0]) - PUSH_CONTACT_R - PUSH_PRE_GAP
    face1 = float(goal_xy[0]) - PUSH_CONTACT_R
    down = np.array([face0 - PUSH_FACE_DX, y, z])
    end = np.array([face1 - PUSH_FACE_DX, y, z])
    back = end - np.array([PUSH_BACKOFF, 0.0, 0.0])
    return {
        "hover": down + np.array([-0.02, 0.0, PUSH_HOVER]),
        "down": down,
        "push_end": end,
        "back": back,
        "up": back + np.array([0.0, 0.0, PUSH_UP]),
    }


def _with_backbone(tcp_pts: np.ndarray, axis: np.ndarray, length: float = 0.26) -> np.ndarray:
    """TCP path plus the hand/wrist line behind it (points back along -axis)."""
    out = [tcp_pts]
    for s in np.linspace(0.05, length, 6):
        out.append(tcp_pts - s * axis[None, :])
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------
# Samplers
# ------------------------------------------------------------------------------------------------
class _PCKitchenSampler(CavityPickTaskSampler):
    """Kitchen base: custom-scene plumbing, item installation, per-episode layout + keep-out."""

    PACT_PLACE_ENVIRONMENT_VERSION = "pc_kitchen_v1"
    TASK_KIND = "base"
    REFERRAL = "object"
    POOL_SIZE = 1
    TARGET_UIDS: tuple[str, ...] = (PNP_CUP_UID,)
    BASE_XYZ = BASE_XYZ
    # slot table: (slot, candidate items, P(present), x range, y range, yaw mode)
    SLOTS: tuple = ()
    PATH_CLEAR = 0.085          # min horizontal distance, hand/TCP path -> item surface
    PATH_VCLEAR = 0.07          # path points this far above an item's top cannot hit it
    PLACE_TRIES = 30

    def __init__(self, config) -> None:
        # Shared datagen speed-up (drops the unused ObjectImagePointsSensor, lazy skin-camera
        # registry poses; RGB/skin pixels unchanged). No-op under the frozen evaluator.
        try:
            from molmo_spaces.tasks.pc_shelf import _install_fast_hooks

            _install_fast_hooks(config)
        except Exception as e:  # pragma: no cover - the hooks are an optimisation only
            log.warning(f"[PCKitchen] fast hooks unavailable: {e!r}")
        super().__init__(config)
        self._item_geom: dict[str, dict[str, Any]] = {}
        self._item_model_id = None
        self._theta: dict[str, Any] = {}
        if getattr(config, "seed", None) is None:
            # Collection workers would otherwise share one RNG stream and repeat episodes.
            self.seed_task_sampling(int.from_bytes(os.urandom(4), "little") ^ os.getpid())

    # ---- house / scene plumbing ----
    def _build_grasp_uid_pool(self, n: int) -> list[str]:
        return list(self.TARGET_UIDS)

    def _cell(self) -> int:
        return int(self.current_house_index) % N_CELLS

    def _cell_params(self, cell: int) -> dict[str, float]:
        """Layout of a cell (the hard suite overrides this with its own cell table)."""
        return cell_params(cell)

    def _obj_rest(self):
        return (PNP_CUP_XY[0], PNP_CUP_XY[1], COUNTER_Z)

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        self._inject_target(spec)
        for item, uid in THOR_ITEMS.items():
            extra = MjSpec.from_file(str(install_uid(uid)))
            body = extra.worldbody.bodies[0]
            for joint in list(extra.joints):
                extra.delete(joint)
            for site in list(extra.sites):      # THOR receptacle-trigger boxes: not real geometry
                extra.delete(site)
            body.mocap = True
            body.quat = list(STAND_QUAT.as_quat(scalar_first=True))
            frame = spec.worldbody.add_frame(pos=list(PARK_POS))
            frame.attach_body(body, f"{clutter_body_name(item)}/", "")
        self._item_model_id = None

    def _inject_target(self, spec: MjSpec) -> None:
        """Pick target: a beverage can (free body ``cavity_obj_0``), aluminium body + lid rim."""
        PickTaskSampler.add_auxiliary_objects(self, spec)
        name = "cavity_obj_0"
        ox, oy, oz = self._obj_rest()
        body = spec.worldbody.add_body(name=name, pos=[ox, oy, oz + PNP_CAN_H / 2 + 0.002])
        body.add_freejoint(name=f"{name}_jntfree")
        can = body.add_geom(name="can_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[PNP_CAN_R, PNP_CAN_H / 2 - 0.004, 0], pos=[0, 0, -0.004],
                            rgba=[0.78, 0.09, 0.10, 1.0], mass=0.33)
        rim = body.add_geom(name="can_top", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[PNP_CAN_R - 0.004, 0.004, 0], pos=[0, 0, PNP_CAN_H / 2 - 0.004],
                            rgba=[0.80, 0.81, 0.83, 1.0], mass=0.01)
        band = body.add_geom(name="can_band", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                             size=[PNP_CAN_R + 0.0004, 0.012, 0], pos=[0, 0, 0.012],
                             rgba=[0.92, 0.92, 0.90, 1.0], contype=0, conaffinity=0, mass=0.0001)
        for gg in (can, rim):
            gg.contype = 1
            gg.conaffinity = 15
            gg.condim = 4
            gg.friction = [1.0, 0.01, 0.0001]
        del band
        self._injected_obj_name = name
        self._metadata_adder.update({
            name: {
                "asset_id": None,
                "category": "can",
                "object_enum": "temp_object",
                "is_static": False,
                "boundingBox": {"x": 2 * PNP_CAN_R, "y": 2 * PNP_CAN_R, "z": PNP_CAN_H},
            }
        })

    def _thor_body(self, m, item: str) -> int:
        prefix = f"{clutter_body_name(item)}/"
        for b in range(m.nbody):
            name = m.body(b).name or ""
            if name.startswith(prefix) and int(m.body_parentid[b]) == 0:
                return b
        raise KeyError(prefix)

    def _measure_items(self, env) -> None:
        """Rest height / footprint of every movable item, measured once per compiled model."""
        m, d = env.current_model, env.current_data
        if self._item_model_id == id(m) and self._item_geom:
            return
        geom = {}
        for item, spec in XML_ITEMS.items():
            bid = m.body(clutter_body_name(item)).id
            geom[item] = {"body": bid, "stand": False, "z0": 0.0, "lo": np.array([-spec["r"], -spec["r"]]),
                          "hi": np.array([spec["r"], spec["r"]]), "h": spec["h"]}
        mocap_saved = (d.mocap_pos.copy(), d.mocap_quat.copy())
        for item in THOR_ITEMS:
            bid = self._thor_body(m, item)
            mid = int(m.body_mocapid[bid])
            d.mocap_pos[mid] = 0.0
            d.mocap_quat[mid] = _yaw_quat(0.0, True)
            mujoco.mj_kinematics(m, d)
            pts = np.concatenate([_geom_points(m, d, g) for g in _subtree_geoms(m, bid)])
            geom[item] = {"body": bid, "stand": True, "z0": -float(pts[:, 2].min()),
                          "lo": pts[:, :2].min(0), "hi": pts[:, :2].max(0),
                          "h": float(pts[:, 2].max() - pts[:, 2].min())}
        d.mocap_pos[:], d.mocap_quat[:] = mocap_saved
        mujoco.mj_kinematics(m, d)
        self._item_geom = geom
        self._item_model_id = id(m)

    def _set_mocap(self, env, body_name_or_id, pos, quat) -> None:
        m, d = env.current_model, env.current_data
        bid = body_name_or_id if isinstance(body_name_or_id, (int, np.integer)) else m.body(body_name_or_id).id
        mid = int(m.body_mocapid[bid])
        d.mocap_pos[mid] = np.asarray(pos, float)
        d.mocap_quat[mid] = np.asarray(quat, float)

    def _park_all(self, env) -> None:
        for item, g in self._item_geom.items():
            self._set_mocap(env, g["body"], PARK_POS, _yaw_quat(0.0, g["stand"]))
        self._set_mocap(env, "pact_clutter_board", PARK_POS, [1, 0, 0, 0])
        self._set_mocap(env, "place_receptacle", PARK_POS, [1, 0, 0, 0])
        self._set_mocap(env, "push_goal_decal", PARK_POS, [1, 0, 0, 0])

    def _item_footprint(self, item: str, x: float, y: float, yaw: float):
        g = self._item_geom[item]
        lo, hi = g["lo"], g["hi"]
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
        c, s = np.cos(yaw), np.sin(yaw)
        rc = corners @ np.array([[c, s], [-s, c]])
        return rc.min(0) + [x, y], rc.max(0) + [x, y]

    def _place_item(self, env, item: str, x: float, y: float, yaw: float) -> None:
        g = self._item_geom[item]
        z = COUNTER_Z + g["z0"] + 0.0005
        self._set_mocap(env, g["body"], (x, y, z), _yaw_quat(yaw, g["stand"]))

    def _place_board(self, env, y: float) -> tuple[np.ndarray, np.ndarray]:
        t, half_w, half_h = BOARD_HALF
        x = BACKSPLASH_X - (t * np.cos(BOARD_LEAN) + 2 * half_h * np.sin(BOARD_LEAN)) - 0.001
        z = COUNTER_Z + t * np.sin(BOARD_LEAN) + 0.0005
        q = R.from_euler("y", BOARD_LEAN).as_quat(scalar_first=True)
        self._set_mocap(env, "pact_clutter_board", (x, y, z), q)
        lo = np.array([x - t, y - half_w])
        hi = np.array([BACKSPLASH_X, y + half_w])
        return lo, hi

    # ---- keep-out ----
    def _footprint_ok(self, lo, hi, th) -> bool:
        cp = self._cell_params(self._cell())
        if lo[0] < COUNTER_FRONT_X + 0.01 or hi[0] > BACKSPLASH_X + 1e-6:
            return False
        if lo[1] < ROOM_RIGHT_Y + 0.01 or hi[1] > cp["fridge_y"] - 0.008:
            return False
        if hi[1] > COOKTOP_Y[0] - 0.01 and lo[1] < COOKTOP_Y[1] + 0.01:
            return False
        for plo, phi in th["_placed"]:
            if lo[0] < phi[0] + 0.012 and hi[0] > plo[0] - 0.012 and lo[1] < phi[1] + 0.012 and hi[1] > plo[1] - 0.012:
                return False
        return True

    def _path_ok(self, lo, hi, top_z, th) -> bool:
        for pts, clear in th["_keepout"]:
            sel = pts[:, 2] < top_z + self.PATH_VCLEAR
            if not np.any(sel):
                continue
            if float(_aabb2_dist(pts[sel, :2], lo, hi).min()) < clear:
                return False
        return True

    def _fill_slots(self, env, th) -> None:
        used = set()
        th["clutter"] = []
        for slot, items, p_present, xr, yr, yaw_mode in self.SLOTS:
            if np.random.random() > p_present:
                continue
            choices = [i for i in items if i not in used]
            if not choices:
                continue
            item = str(np.random.choice(choices))
            for _ in range(self.PLACE_TRIES):
                against_wall = isinstance(xr, str)
                x = BACKSPLASH_X - 0.2 if against_wall else float(np.random.uniform(*xr))
                y = float(np.random.uniform(*yr))
                if item == "board":
                    t, half_w, half_h = BOARD_HALF
                    lo = np.array([BACKSPLASH_X - 0.10, y - half_w])
                    hi = np.array([BACKSPLASH_X, y + half_w])
                    top = COUNTER_Z + 2 * half_h
                else:
                    yaw = {"any": float(np.random.uniform(0, 2 * np.pi)),
                           "wall": float(np.random.uniform(-0.15, 0.15)),
                           "side": float(np.pi / 2 + np.random.uniform(-0.15, 0.15))}[yaw_mode]
                    lo, hi = self._item_footprint(item, x, y, yaw)
                    # against-the-wall slots: slide the item back until it (almost) touches the tiles
                    if against_wall:
                        shift = (BACKSPLASH_X - float(np.random.uniform(0.004, 0.02))) - hi[0]
                        x += shift
                        lo, hi = lo + [shift, 0], hi + [shift, 0]
                    top = COUNTER_Z + self._item_geom[item]["h"]
                if not self._footprint_ok(lo, hi, th) or not self._path_ok(lo, hi, top, th):
                    continue
                if item == "board":
                    lo, hi = self._place_board(env, y)
                    rec = {"item": item, "slot": slot, "xy": [float((lo[0] + hi[0]) / 2), y], "yaw": 0.0}
                else:
                    self._place_item(env, item, x, y, yaw)
                    rec = {"item": item, "slot": slot, "xy": [x, y], "yaw": yaw}
                rec.update(lo=[float(v) for v in lo], hi=[float(v) for v in hi], top=float(top))
                th["clutter"].append(rec)
                th["_placed"].append((lo, hi))
                used.add(item)
                break
        th["n_clutter"] = len(th["clutter"])

    # ---- target placement ----
    def _target_qadr(self, env) -> int:
        m = env.current_model
        bid = m.body(self._injected_obj_name).id
        jadr = int(m.body_jntadr[bid])
        return int(m.jnt_qposadr[jadr])

    def _target_rest_offset(self, env, quat) -> float:
        """Height of the target's body origin above its lowest collision point."""
        m, d = env.current_model, env.current_data
        qadr = self._target_qadr(env)
        saved = d.qpos[qadr:qadr + 7].copy()
        d.qpos[qadr:qadr + 3] = (0.0, 0.0, 0.0)
        d.qpos[qadr + 3:qadr + 7] = quat
        mujoco.mj_kinematics(m, d)
        bid = m.body(self._injected_obj_name).id
        gs = [g for g in _subtree_geoms(m, bid) if m.geom_contype[g] or m.geom_conaffinity[g]]
        z0 = -float(np.concatenate([_geom_points(m, d, g) for g in gs])[:, 2].min())
        d.qpos[qadr:qadr + 7] = saved
        mujoco.mj_kinematics(m, d)
        return z0

    def _settle_target(self, env, x: float, y: float, quat, steps: int = 150) -> np.ndarray:
        """Stand the target on the counter and let it come to rest; every other DOF is restored."""
        m, d = env.current_model, env.current_data
        qadr = self._target_qadr(env)
        z0 = self._target_rest_offset(env, quat)
        qpos0 = d.qpos.copy()
        d.qpos[qadr:qadr + 3] = (x, y, COUNTER_Z + z0 + 0.002)
        d.qpos[qadr + 3:qadr + 7] = quat
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(steps):
            mujoco.mj_step(m, d)
        obj = d.qpos[qadr:qadr + 7].copy()
        d.qpos[:] = qpos0
        d.qpos[qadr:qadr + 7] = obj
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        return obj

    # ---- per-episode orchestration ----
    def _draw_theta(self) -> dict[str, Any]:
        raise NotImplementedError

    def _apply_theta(self, env, th) -> None:
        raise NotImplementedError

    def _finish_task(self, task, th) -> None:
        raise NotImplementedError

    def _sample_task(self, env):
        cell = self._cell()
        self._measure_items(env)
        self._cur_base_xyz = tuple(self.BASE_XYZ)
        self._cur_base_yaw = 0.0
        th = self._draw_theta()
        th.update(self._cell_params(cell))
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["task_kind"] = self.TASK_KIND
        th["house_index"] = int(self.current_house_index)
        th["_placed"] = []
        self._park_all(env)
        self._apply_theta(env, th)
        self._fill_slots(env, th)
        mujoco.mj_forward(env.current_model, env.current_data)
        self._theta = th
        self.candidate_objects = self._get_scene_objects(env)
        # PickTaskSampler._sample_task minus the referral-expression search: that path resolves
        # context objects through the annotation/HF stack and cost 14-45 s per episode here, for a
        # one-object task whose instruction is fixed anyway.
        self._select_pickup_object(env)
        task_cfg = self.config.task_config
        task_cfg.referral_expressions["pickup_obj_name"] = self.REFERRAL
        task_cfg.referral_expressions_priority["pickup_obj_name"] = [(1.0, 1.0, self.REFERRAL)]
        task = PickTask(env, self.config)
        task.scene_params = {k: v for k, v in th.items() if not k.startswith("_")}
        self._finish_task(task, th)
        log.info(
            f"[PCKitchen] {self.TASK_KIND} cell={cell} fridge_y={th['fridge_y']:.2f} "
            f"cab_bottom={th['cab_bottom_z']:.2f} cab_front={th['cab_front_x']:.2f} "
            f"n_clutter={th['n_clutter']} items={[c['item'] for c in th['clutter']]}"
        )
        return task


class PCKitchenPnPSampler(_PCKitchenSampler):
    PACT_PLACE_ENVIRONMENT_VERSION = "pc_kitchen_pnp_v1"
    TASK_KIND = "pnp"
    REFERRAL = "red can"
    TARGET_UIDS = (PNP_CUP_UID,)
    # x is either an absolute range or "wall": the item is slid back until it stands 4-20 mm in
    # front of the backsplash tiles (see _fill_slots).
    SLOTS = (
        ("behind_cup", ("oil", "wine", "towel", "can_a"), 0.9, "wall", (0.14, 0.27), "any"),
        ("right_of_cup", ("soap", "can_c", "salt", "pepper"), 0.75, (0.63, 0.70), (0.015, 0.07), "any"),
        ("wall_mid_a", ("can_b", "can_a", "oil", "can_c"), 0.85, "wall", (-0.05, 0.08), "any"),
        ("wall_mid_b", ("can_b", "can_c", "kettle", "board"), 0.8, "wall", (-0.20, -0.06), "any"),
        ("behind_tray", ("board", "kettle", "wine", "towel"), 0.8, "wall", (-0.42, -0.26), "any"),
        ("right_of_tray", ("toaster", "kettle", "bread"), 0.85, "wall", (-0.76, -0.58), "wall"),
        ("front_right", ("bowl", "bread"), 0.6, (0.40, 0.50), (-0.80, -0.64), "wall"),
        ("by_cooktop", ("salt", "pepper", "oil"), 0.6, "wall", (-0.84, -0.80), "any"),
    )

    def _draw_theta(self):
        jx, jy = PNP_CUP_JIT
        tx, ty = PNP_TRAY_JIT
        return {
            "cup_xy": [PNP_CUP_XY[0] + float(np.random.uniform(-jx, jx)),
                       PNP_CUP_XY[1] + float(np.random.uniform(-jy, jy))],
            "cup_yaw": float(np.random.uniform(0, 2 * np.pi)),
            "tray_xy": [PNP_TRAY_XY[0] + float(np.random.uniform(-tx, tx)),
                        PNP_TRAY_XY[1] + float(np.random.uniform(-ty, ty))],
            "tray_yaw": float(np.random.uniform(-0.12, 0.12)),
            "target_uid": PNP_CUP_UID,
        }

    def _apply_theta(self, env, th) -> None:
        cx, cy = th["cup_xy"]
        q = _yaw_quat(th["cup_yaw"], False)
        obj = self._settle_target(env, cx, cy, q)
        th["cup_rest_pose"] = [float(v) for v in obj]
        tx, ty = th["tray_xy"]
        self._set_mocap(env, "place_receptacle", (tx, ty, COUNTER_Z + 0.0005),
                        R.from_euler("z", th["tray_yaw"]).as_quat(scalar_first=True))
        th["_placed"] += [
            (np.array([cx - PNP_CAN_R - 0.01, cy - PNP_CAN_R - 0.01]), np.array([cx + PNP_CAN_R + 0.01, cy + PNP_CAN_R + 0.01])),
            (np.array([tx - 0.17, ty - 0.13]), np.array([tx + 0.17, ty + 0.13])),
        ]
        w = pnp_waypoints((float(obj[0]), float(obj[1])), (tx, ty))
        order = ["pregrasp", "grasp", "lift", "retract", "over_tray", "place", "retreat"]
        tcp = _densify([HOME_TCP_EST] + [w[k] for k in order])
        th["_keepout"] = [(_with_backbone(tcp, pnp_approach_axis()), self.PATH_CLEAR)]

    def _finish_task(self, task, th) -> None:
        task.__class__ = PCKitchenPnPTask
        tx, ty = th["tray_xy"]
        task.config.task_config.pickup_obj_goal_pose = [tx, ty, COUNTER_Z + PNP_TRAY_FLOOR + PNP_CAN_H / 2, 1.0, 0.0, 0.0, 0.0]
        task.place_target = (tx, ty, COUNTER_Z + PNP_TRAY_FLOOR)


class PCKitchenPushSampler(_PCKitchenSampler):
    PACT_PLACE_ENVIRONMENT_VERSION = "pc_kitchen_push_v1"
    TASK_KIND = "push"
    REFERRAL = "biscuit tin"
    TARGET_UIDS = ("pc_kitchen_tin",)
    SLOTS = (
        ("behind_goal", ("oil", "wine", "towel", "can_a"), 0.9, "wall", (0.13, 0.27), "any"),
        ("right_of_lane", ("soap", "can_c", "salt", "pepper"), 0.8, (0.47, 0.64), (-0.005, 0.035), "any"),
        ("wall_mid_a", ("can_b", "can_a", "oil", "can_c"), 0.85, "wall", (-0.06, 0.06), "any"),
        ("wall_mid_b", ("kettle", "board", "can_b"), 0.8, "wall", (-0.36, -0.12), "any"),
        ("wall_right", ("toaster", "bread", "kettle"), 0.85, "wall", (-0.74, -0.42), "wall"),
        ("front_right", ("bowl", "bread"), 0.6, (0.40, 0.50), (-0.70, -0.30), "wall"),
        ("by_cooktop", ("salt", "pepper", "oil"), 0.6, "wall", (-0.84, -0.78), "any"),
    )

    def _obj_rest(self):
        return (PUSH_START_XY[0], PUSH_START_XY[1], COUNTER_Z)

    def _inject_target(self, spec: MjSpec) -> None:
        """A round biscuit tin (lid + body), free body named cavity_obj_0."""
        PickTaskSampler.add_auxiliary_objects(self, spec)
        name = "cavity_obj_0"
        body = spec.worldbody.add_body(name=name, pos=[PUSH_START_XY[0], PUSH_START_XY[1], COUNTER_Z + PUSH_TIN_H / 2 + 0.002])
        body.add_freejoint(name=f"{name}_jntfree")
        tin = body.add_geom(name="tin_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[PUSH_TIN_R - 0.002, PUSH_TIN_H / 2 - 0.006, 0], pos=[0, 0, -0.006],
                            rgba=[0.62, 0.10, 0.12, 1.0], mass=0.30)
        lid = body.add_geom(name="tin_lid", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[PUSH_TIN_R, 0.009, 0], pos=[0, 0, PUSH_TIN_H / 2 - 0.009],
                            rgba=[0.80, 0.68, 0.30, 1.0], mass=0.05)
        for gg in (tin, lid):
            gg.contype = 1
            gg.conaffinity = 15
            gg.friction = [0.45, 0.01, 0.001]
            gg.condim = 4
            gg.solref = [0.01, 1.0]
        try:
            body.first_joint().damping = OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
        except Exception:
            pass
        self._injected_obj_name = name
        self._metadata_adder.update({
            name: {
                "asset_id": None,
                "category": "tin",
                "object_enum": "temp_object",
                "is_static": False,
                "boundingBox": {"x": 2 * PUSH_TIN_R, "y": 2 * PUSH_TIN_R, "z": PUSH_TIN_H},
            }
        })

    def _draw_theta(self):
        jx, jy = PUSH_START_JIT
        sx = PUSH_START_XY[0] + float(np.random.uniform(-jx, jx))
        sy = PUSH_START_XY[1] + float(np.random.uniform(-jy, jy))
        dist = float(np.random.uniform(*PUSH_DIST))
        return {"tin_start_xy": [sx, sy], "push_dist": dist, "goal_xy": [sx + dist, sy],
                "tin_yaw": float(np.random.uniform(0, 2 * np.pi)), "target_uid": "pc_kitchen_tin"}

    def _apply_theta(self, env, th) -> None:
        sx, sy = th["tin_start_xy"]
        gx, gy = th["goal_xy"]
        obj = self._settle_target(env, sx, sy, R.from_euler("z", th["tin_yaw"]).as_quat(scalar_first=True))
        th["tin_rest_pose"] = [float(v) for v in obj]
        self._set_mocap(env, "push_goal_decal", (gx, gy, COUNTER_Z), [1, 0, 0, 0])
        r = PUSH_TIN_R
        th["_placed"] += [
            (np.array([sx - r - 0.10, sy - r]), np.array([gx + r, sy + r])),   # tin lane + hand
        ]
        w = push_waypoints((float(obj[0]), float(obj[1])), (gx, gy))
        order = ["hover", "down", "push_end", "back", "up"]
        tcp = _densify([HOME_TCP_EST] + [w[k] for k in order])
        tin_line = _densify([np.array([obj[0], obj[1], COUNTER_Z + 0.04]), np.array([gx, gy, COUNTER_Z + 0.04])])
        th["_keepout"] = [
            (_with_backbone(tcp, push_approach_axis(), 0.30), self.PATH_CLEAR),
            (tin_line, PUSH_TIN_R + 0.03),
        ]

    def _finish_task(self, task, th) -> None:
        task.__class__ = PCKitchenPushTask
        gx, gy = th["goal_xy"]
        task.config.task_config.pickup_obj_goal_pose = [gx, gy, COUNTER_Z + PUSH_TIN_H / 2, 1.0, 0.0, 0.0, 0.0]
        task.push_goal = (gx, gy)


class PCKitchenPnPCheckSampler(PCKitchenPnPSampler):
    """Preflight: every slot filled (maximum clutter), same geometry otherwise."""

    SLOTS = tuple((s, i, 1.0, xr, yr, ym) for (s, i, _p, xr, yr, ym) in PCKitchenPnPSampler.SLOTS)


class PCKitchenPushCheckSampler(PCKitchenPushSampler):
    SLOTS = tuple((s, i, 1.0, xr, yr, ym) for (s, i, _p, xr, yr, ym) in PCKitchenPushSampler.SLOTS)


# ------------------------------------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------------------------------------
_HAZARD_SKIP = ("floor",)


class _PCKitchenTask(PickTask):
    """Adds scene_params to the obs and names every robot contact (target / receptacle / hazard)."""

    def get_obs_scene(self) -> dict[str, Any]:
        scene = super().get_obs_scene()
        scene["scene_params"] = getattr(self, "scene_params", {})
        return scene

    def _accumulate_obstacle_diag(self, data, pickup_obj) -> None:
        super()._accumulate_obstacle_diag(data, pickup_obj)
        try:
            model = data.model
            hits = getattr(self, "_contact_names", None)
            if hits is None:
                hits = self._contact_names = {}
            for c in data.contact[: data.ncon]:
                if c.dist > 0:
                    continue
                r1 = model.body(int(model.body_rootid[model.geom_bodyid[c.geom1]])).name or ""
                r2 = model.body(int(model.body_rootid[model.geom_bodyid[c.geom2]])).name or ""
                if r1.startswith("robot_0/") == r2.startswith("robot_0/"):
                    continue
                other_root = r2 if r1.startswith("robot_0/") else r1
                gid = c.geom2 if r1.startswith("robot_0/") else c.geom1
                if any(s in other_root.lower() for s in _HAZARD_SKIP):
                    continue
                if other_root.startswith("cavity_obj_"):
                    key = "target:" + other_root
                elif other_root.startswith("place_receptacle"):
                    key = "receptacle"
                else:
                    key = other_root if other_root and other_root != "world" else (
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or "unnamed")
                hits[key] = hits.get(key, 0) + 1
        except Exception as e:  # pragma: no cover - diagnostics never break a rollout
            log.debug(f"[PCKitchen] contact attribution failed: {e}")

    def _maybe_log_obstacle_diag(self, success: bool) -> None:
        already = getattr(self, "_obstacle_diag_logged", True)
        super()._maybe_log_obstacle_diag(success)
        if not already and getattr(self, "_obstacle_diag_logged", False):
            hits = getattr(self, "_contact_names", {}) or {}
            hazards = {k: v for k, v in hits.items() if not k.startswith("target:") and k != "receptacle"}
            log.info(
                "[PCKitchen] robot contacts by body: %s | HAZARD contacts: %s",
                hits or "{}", hazards or "none",
            )
            self._contact_names = {}

    def _target_body(self):
        return MlSpacesObject(data=self._env.mj_datas[0], object_name=self.config.task_config.pickup_obj_name)

    def _robot_touching_target(self) -> bool:
        data = self._env.mj_datas[0]
        model = data.model
        name = self.config.task_config.pickup_obj_name
        for c in data.contact[: data.ncon]:
            r1 = model.body(int(model.body_rootid[model.geom_bodyid[c.geom1]])).name or ""
            r2 = model.body(int(model.body_rootid[model.geom_bodyid[c.geom2]])).name or ""
            if (r1.startswith("robot_0/") and r2 == name) or (r2.startswith("robot_0/") and r1 == name):
                return True
        return False


class PCKitchenPnPTask(_PCKitchenTask):
    """Success: the cup stands upright on the tray, near its centre, and the hand has let go."""

    def get_task_description(self) -> str:
        return "Put the red can on the wooden tray."

    def judge_success(self) -> bool:
        try:
            tx, ty, tz = getattr(self, "place_target")
            data = self._env.mj_datas[0]
            model = data.model
            bid = model.body(self.config.task_config.pickup_obj_name).id
            p = np.asarray(data.xpos[bid], float)
            up = np.asarray(data.xmat[bid], float).reshape(3, 3)[:, 2]
            dxy = float(np.hypot(p[0] - tx, p[1] - ty))
            dz = float(p[2] - (tz + PNP_CAN_H / 2))
            ok = dxy <= PNP_PLACE_TOL and abs(dz) < 0.02 and up[2] > np.cos(np.deg2rad(20)) \
                and not self._robot_touching_target()
            if ok != getattr(self, "_last_verdict", None):
                self._last_verdict = ok
                log.info(f"[PCKitchen] pnp placed={ok}: cup ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                         f"dxy={dxy:.3f} dz={dz:+.3f} upright={up[2]:.3f}")
            return ok
        except Exception as e:  # pragma: no cover
            log.warning(f"[PCKitchen] pnp judge_success raised, scoring as failure: {e!r}")
            return False


class PCKitchenPushTask(_PCKitchenTask):
    """Success: the tin's centre is within PUSH_TOL of the goal and it is still flat on the counter."""

    def get_task_description(self) -> str:
        return "Push the biscuit tin onto the red trivet."

    def judge_success(self) -> bool:
        try:
            gx, gy = getattr(self, "push_goal")
            data = self._env.mj_datas[0]
            model = data.model
            bid = model.body(self.config.task_config.pickup_obj_name).id
            p = np.asarray(data.xpos[bid], float)
            up = np.asarray(data.xmat[bid], float).reshape(3, 3)[:, 2]
            dxy = float(np.hypot(p[0] - gx, p[1] - gy))
            flat = up[2] > np.cos(np.deg2rad(8)) and abs(p[2] - (COUNTER_Z + PUSH_TIN_H / 2)) < 0.012
            ok = dxy <= PUSH_TOL and flat and not self._robot_touching_target()
            if ok != getattr(self, "_last_verdict", None):
                self._last_verdict = ok
                log.info(f"[PCKitchen] push at_goal={ok}: tin ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                         f"goal ({gx:.3f},{gy:.3f}) dxy={dxy:.3f} flat={flat}")
            return ok
        except Exception as e:  # pragma: no cover
            log.warning(f"[PCKitchen] push judge_success raised, scoring as failure: {e!r}")
            return False


# ------------------------------------------------------------------------------------------------
# Experts
# ------------------------------------------------------------------------------------------------
def _frame_from_axis(approach: np.ndarray, ref_R: np.ndarray) -> np.ndarray:
    """TCP rotation with z = approach and the closing axis horizontal; of the two such frames
    (180 deg apart about the approach) take the one nearest the current hand orientation."""
    z = approach / np.linalg.norm(approach)
    y = np.cross(np.array([0.0, 0.0, 1.0]), z)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    cands = [np.column_stack([x, y, z]), np.column_stack([-x, -y, z])]
    errs = [R.from_matrix(ref_R.T @ c).magnitude() for c in cands]
    return cands[int(np.argmin(errs))]


class _PCKitchenExpertMixin:
    def _pose(self, p, Rm) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = p
        return T

    def _seq(self, segs, holding=False) -> TCPMoveSequence:
        return TCPMoveSequence(
            self.task.env.current_robot.robot_view,
            self._tcp_to_jp_fn,
            self.policy_config.move_settle_time,
            is_holding_object=holding,
            gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
            tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
            tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
            move_segments=segs,
        )

    def _start_pose(self) -> np.ndarray:
        rv = self.task.env.current_robot.robot_view
        mg = rv.get_gripper_movegroup_ids()[0]
        return rv.get_move_group(mg).leaf_frame_to_world.copy()

    def _require_ik(self, poses: dict[str, np.ndarray]) -> None:
        for name, T in poses.items():
            if not self.check_feasible_ik(T):
                raise ValueError(f"IK failed for {name} pose at {np.round(T[:3, 3], 3)}")

    def _obj_pos(self) -> np.ndarray:
        env = self.task.env
        m, d = env.current_model, env.current_data
        bid = m.body(self.config.task_config.pickup_obj_name).id
        return np.asarray(d.xpos[bid], float).copy()


class PCKitchenPnPPolicy(_PCKitchenExpertMixin, PickPlannerPolicy):
    """approach (pitched side grasp) -> grasp -> close -> lift -> retract -> carry over the tray ->
    lower -> open -> back out. One fixed strategy; every waypoint comes from pnp_waypoints()."""

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        th = getattr(self.task, "scene_params", {}) or {}
        rv = self.task.env.current_robot.robot_view
        start = self._start_pose()
        cup = self._obj_pos()
        tray_xy = th.get("tray_xy", PNP_TRAY_XY)
        cup_bottom = COUNTER_Z
        w = pnp_waypoints((cup[0], cup[1]), tray_xy, cup_bottom)
        Rg = _frame_from_axis(pnp_approach_axis(), start[:3, :3])
        P = {k: self._pose(v, Rg) for k, v in w.items()}
        self._require_ik({k: P[k] for k in ("pregrasp", "grasp", "lift", "retract", "over_tray", "place", "retreat")})
        fast, slow = self.policy_config.speed_fast, self.policy_config.speed_slow
        # NOTE: the opening command at t=0 must be a ZERO-duration primitive. A timed gripper action
        # replays robot_view.get_ctrl_dict(), and right after the sampler resets the scene that ctrl
        # is not the arm's current pose: the arm lurched ~40 cm in 0.4 s on the first preflight.
        prims = [
            GripperAction(rv, True, 0.0),
            self._seq([
                TCPMoveSegment(name="pregrasp", start_pose=start, end_pose=P["pregrasp"], speed=fast),
                TCPMoveSegment(name="grasp", start_pose=P["pregrasp"], end_pose=P["grasp"], speed=slow),
            ]),
            GripperAction(rv, False, self.policy_config.gripper_close_duration),
            self._seq([
                TCPMoveSegment(name="lift", start_pose=P["grasp"], end_pose=P["lift"], speed=slow),
                TCPMoveSegment(name="lift", start_pose=P["lift"], end_pose=P["retract"], speed=0.10),
                TCPMoveSegment(name="preplace", start_pose=P["retract"], end_pose=P["over_tray"], speed=0.17),
                TCPMoveSegment(name="place", start_pose=P["over_tray"], end_pose=P["place"], speed=0.05),
            ], holding=True),
            GripperAction(rv, True, self.policy_config.gripper_open_duration),
            self._seq([
                TCPMoveSegment(name="retreat", start_pose=P["place"], end_pose=P["retreat"], speed=0.10),
            ]),
        ]
        log.info(f"[PCKitchen] PnP plan: cup ({cup[0]:.3f},{cup[1]:.3f}) -> tray "
                 f"({tray_xy[0]:.3f},{tray_xy[1]:.3f}); grasp z={w['grasp'][2]:.3f} place z={w['place'][2]:.3f}")
        return prims


class PCKitchenPushPolicy(_PCKitchenExpertMixin, PickPlannerPolicy):
    """close the gripper -> hover behind the tin -> lower to push height -> straight push along +x
    at 5 cm/s -> back off -> lift away. One fixed strategy; waypoints from push_waypoints()."""

    PUSH_SPEED = 0.05

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        th = getattr(self.task, "scene_params", {}) or {}
        rv = self.task.env.current_robot.robot_view
        start = self._start_pose()
        tin = self._obj_pos()
        goal = th.get("goal_xy")
        w = push_waypoints((tin[0], tin[1]), goal)
        Rp = _frame_from_axis(push_approach_axis(), start[:3, :3])
        P = {k: self._pose(v, Rp) for k, v in w.items()}
        self._require_ik(P)
        fast, slow = self.policy_config.speed_fast, self.policy_config.speed_slow
        # Zero-duration close at t=0 (see PCKitchenPnPPolicy): the fingers close during the approach.
        prims = [
            GripperAction(rv, False, 0.0),
            self._seq([
                TCPMoveSegment(name="pregrasp", start_pose=start, end_pose=P["hover"], speed=fast),
                TCPMoveSegment(name="grasp", start_pose=P["hover"], end_pose=P["down"], speed=slow),
            ]),
            self._seq([
                TCPMoveSegment(name="place", start_pose=P["down"], end_pose=P["push_end"], speed=self.PUSH_SPEED),
            ]),
            self._seq([
                TCPMoveSegment(name="retreat", start_pose=P["push_end"], end_pose=P["back"], speed=slow),
                TCPMoveSegment(name="retreat", start_pose=P["back"], end_pose=P["up"], speed=fast),
            ]),
        ]
        log.info(f"[PCKitchen] Push plan: tin ({tin[0]:.3f},{tin[1]:.3f}) -> goal ({goal[0]:.3f},{goal[1]:.3f}); "
                 f"stroke x {w['down'][0]:.3f} -> {w['push_end'][0]:.3f} at z={w['down'][2]:.3f}")
        return prims


class PCKitchenPnPPolicyConfig(PickPlannerPolicyConfig):
    speed_fast: float = 0.17
    speed_slow: float = 0.05
    move_settle_time: float = 0.2
    # Let the 2F-85 finish closing on the can (~0.7 s) before the lift starts.
    gripper_close_duration: float = 0.8
    gripper_open_duration: float = 0.4
    max_retries: int = 0
    filter_colliding_grasps: bool = False
    filter_feasible_grasps: bool = False
    grasp_feasibility_batch_size: int = 8
    tcp_pos_err_threshold: float = 0.06
    tcp_rot_err_threshold: float = float(np.radians(25.0))

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCKitchenPnPPolicy


class PCKitchenPushPolicyConfig(PCKitchenPnPPolicyConfig):
    speed_fast: float = 0.15
    speed_slow: float = 0.05

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCKitchenPushPolicy


# ================================================================================================
# HARD variant: pc_kitchen_pnp_hard (2026-10-13). Same task, robot, cameras, skin, timing and expert
# as pc_kitchen_pnp; only the geometry is tighter. Everything is real, camera-visible furniture or
# counter items resting on the counter (nothing floating, no invisible geometry).
#
#   * lower + deeper upper cabinets: bottom 0.38 | 0.41 m above the counter (base 0.43 | 0.49),
#     depth 0.35 | 0.38 m (base 0.31 | 0.35): the hand grasps 10-13 cm under the carcass and the
#     wrist / forearm pass closer to the cabinet bottom and its door handles;
#   * fridge closer to the can and the upper arm: side panel at y = 0.32 | 0.35 (base 0.33 | 0.39),
#     5-8 cm from the open gripper; at y = 0.30 the upper arm (fr3_link3) rested on the fridge
#     door for ~1 s in some start poses, so 0.32 is the closest cell;
#   * canisters / bottles / shakers right next to the hand and the carried can: NEAR slots are
#     redrawn until their gap to the swept hand model (2F-85 box + wrist stub + carried can along
#     the expert's exact screw-interpolated path) is 1-3 cm; every other item keeps >= 1 cm (the
#     base suite kept 8.5 cm from the TCP line, about 2-5 cm from the hand).
#
# Collection adds DART-style action noise (robot_config.action_noise_config, executed command
# only; the recorded action stays the clean expert command) and keeps light-contact successes
# (PCKitchenPnPHardCollectTask). The eval / definition config runs with noise off and the plain
# pc_kitchen_pnp success test.
#
#     python -m molmo_spaces.tasks.pc_kitchen --write-hard-scenes
# ================================================================================================
HARD_N_CELLS = 8
_HARD_FRIDGE_Y = (0.32, 0.35)
_HARD_CAB_GAP = (0.38, 0.41)
_HARD_CAB_DEPTH = (0.35, 0.38)


def hard_cell_params(cell: int) -> dict[str, float]:
    """Layout of one of the eight hard cells: every (fridge, gap, depth) combination."""
    cell = int(cell) % HARD_N_CELLS
    fy = _HARD_FRIDGE_Y[cell & 1]
    gap = _HARD_CAB_GAP[(cell >> 1) & 1]
    depth = _HARD_CAB_DEPTH[(cell >> 2) & 1]
    return {
        "cell": cell,
        "fridge_y": fy,
        "cab_gap": gap,
        "cab_depth": depth,
        "cab_bottom_z": COUNTER_Z + gap,
        "cab_front_x": WALL_X - depth,
        "left_wall_y": fy + FRIDGE_W + 0.04,
    }


def hard_scene_path(cell: int) -> Path:
    return SCENE_DIR / f"pc_kitchen_hard_c{int(cell) % HARD_N_CELLS}.xml"


def hard_scene_paths(n: int) -> list[str]:
    """Scene list for ``scene_xml_paths``: index i -> hard cell i % 8."""
    return [str(hard_scene_path(i % HARD_N_CELLS)) for i in range(n)]


def write_hard_scenes(out_dir: Path = SCENE_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cell in range(HARD_N_CELLS):
        path = out_dir / f"pc_kitchen_hard_c{cell}.xml"
        path.write_text(build_scene_xml(cell, params=hard_cell_params(cell), model="pc_kitchen_hard"))
        (out_dir / f"pc_kitchen_hard_c{cell}_metadata.json").write_text(json.dumps({"objects": {}}, indent=1) + "\n")
        paths.append(path)
    return paths


# Swept-hand model (TCP frame: z = approach axis, y = finger closing axis), two boxes measured on the
# FR3 + 2F-85 of the hybrid-skin robot: gripper base x +-0.038, y +-0.042, z -0.158..-0.065; finger
# linkage + pads x +-0.014..0.018, y +-0.063 (open pads stay inside), z -0.097..+0.019. (A single
# x +-0.04 box put the fingertips ~1.2 cm too far forward: grazes drawn with it never touched.)
_HAND_PTS = np.array(
    [[x, y, z] for x in (-0.039, 0.0, 0.039) for y in (-0.043, -0.021, 0.0, 0.021, 0.043)
     for z in (-0.160, -0.130, -0.100, -0.066)]
    + [[x, y, z] for x in (-0.018, 0.0, 0.018) for y in (-0.066, -0.044, -0.022, 0.0, 0.022, 0.044, 0.066)
       for z in (-0.098, -0.070, -0.045, -0.020, 0.0, 0.021)]
)
_WRIST_S = (0.20, 0.26, 0.32)      # wrist / flange stub behind the gripper base (on the axis)
_WRIST_R = 0.060
# FR3 ready-pose TCP rotation on the 0.28 m platform (measured; the policy's start pose).
HOME_R_EST = np.array([[0.987, 0.141, 0.073], [0.145, -0.988, -0.053], [0.065, 0.063, -0.996]])
HARD_GAP_MIN = 0.010               # every item: >= 1 cm from the swept hand
HARD_NEAR = (0.010, 0.030)         # NEAR slots: 1-3 cm from the swept hand
HARD_FAR = (HARD_GAP_MIN, 9.0)
HARD_VMARGIN = 0.015               # hand points this far above an item's top cannot hit it
# Collection only (PCKitchenPnPHardCollectSampler): with probability P_GRAZE one carry-side NEAR
# slot (GRAZE_SLOTS) is drawn at a grazing gap instead, so the hand / carried can brushes that item
# while it passes at carry speed (light contact, kept only if it passes the light-contact rule).
# Not at the cup: the hand dwells there (slow approach, 0.8 s close), so a graze lasted > 150 frames.
# Eval / definition scenes never graze (P_GRAZE = 0).
# Gap to the NOMINAL carry leg; every other leg keeps >= 1 cm. The executed carry runs ~2-6 mm
# inside the nominal line (joint-space tracking of the lateral move; measured on saved demos:
# nominal -0.6 / -1.6 mm -> executed +1.2 / +4.0 mm), so only some grazes touch (with the action
# noise): 3-23 frames of 2 ms, < 0.5 mm deep in the first ones. This is the collection setting.
HARD_GRAZE = (-0.0025, 0.0005)
# Grazes use the ceramic canisters only: their gap is exact (cylinders); a THOR bottle's gap is to
# its footprint box, which left 1-2 cm of air (drawn grazes never touched).
GRAZE_SLOTS = ("carry_back",)     # carry_front sits nearer the approach leg than the carry leg


def _screw_poses(T0: np.ndarray, T1: np.ndarray, step: float = 0.02) -> list[np.ndarray]:
    """The TCPMoveSequence path between two poses (same twist interpolation as the expert)."""
    from molmo_spaces.utils.linalg_utils import transform_to_twist, twist_to_transform

    lin, ang = transform_to_twist(np.linalg.inv(T0) @ T1)
    n = max(2, int(np.ceil(max(np.linalg.norm(T1[:3, 3] - T0[:3, 3]), 0.25 * np.linalg.norm(ang)) / step)) + 1)
    return [T0 @ twist_to_transform(lin * t, ang * t) for t in np.linspace(0.0, 1.0, n)]


def _hand_cloud(poses: list[np.ndarray], holding: list[bool], legs: list[int]):
    """World points, radii and leg ids swept by the gripper, the wrist stub and (when held) the can."""
    pts, rad, lid = [], [], []
    for T, hold, leg in zip(poses, holding, legs):
        Rm, p = T[:3, :3], T[:3, 3]
        chunk = [p + _HAND_PTS @ Rm.T, np.array([p - s * Rm[:, 2] for s in _WRIST_S])]
        radii = [np.zeros(len(_HAND_PTS)), np.full(len(_WRIST_S), _WRIST_R)]
        if hold:
            chunk.append(np.array([[p[0], p[1], p[2] + dz] for dz in (-PNP_GRASP_H, -0.03, 0.0, 0.04)]))
            radii.append(np.full(4, PNP_CAN_R + 0.001))
        for c, r_ in zip(chunk, radii):
            pts.append(c)
            rad.append(r_)
            lid.append(np.full(len(c), leg, dtype=int))
    return np.concatenate(pts), np.concatenate(rad), np.concatenate(lid)


LEG_APPROACH, LEG_CARRY, LEG_RETREAT = 0, 1, 2


def pnp_hand_cloud(cup_xy, tray_xy, start: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swept hand of the pc_kitchen_pnp expert (PCKitchenPnPPolicy) for one layout:
    (points (N, 3), radii (N,), leg ids (N,) in LEG_APPROACH / LEG_CARRY / LEG_RETREAT)."""
    if start is None:
        start = np.eye(4)
        start[:3, :3] = HOME_R_EST
        start[:3, 3] = HOME_TCP_EST
    w = pnp_waypoints(cup_xy, tray_xy)
    Rg = _frame_from_axis(pnp_approach_axis(), start[:3, :3])

    def pose(k):
        T = np.eye(4)
        T[:3, :3] = Rg
        T[:3, 3] = w[k]
        return T

    legs = [(start, pose("pregrasp"), False, LEG_APPROACH), (pose("pregrasp"), pose("grasp"), False, LEG_APPROACH),
            (pose("grasp"), pose("lift"), True, LEG_CARRY), (pose("lift"), pose("retract"), True, LEG_CARRY),
            (pose("retract"), pose("over_tray"), True, LEG_CARRY), (pose("over_tray"), pose("place"), True, LEG_CARRY),
            (pose("place"), pose("retreat"), False, LEG_RETREAT)]
    poses, hold, ids = [], [], []
    for a, b, h, leg in legs:
        seg = _screw_poses(a, b)
        poses += seg
        hold += [h] * len(seg)
        ids += [leg] * len(seg)
    return _hand_cloud(poses, hold, ids)


class PCKitchenPnPHardSampler(PCKitchenPnPSampler):
    """pc_kitchen_pnp_hard: tighter cells (pc_kitchen_hard_c<k>.xml) + NEAR clutter slots."""

    PACT_PLACE_ENVIRONMENT_VERSION = "pc_kitchen_pnp_hard_v1"
    TASK_CLS: type | None = None       # set below (PCKitchenPnPHardTask); collection overrides it
    PLACE_TRIES = 160
    P_GRAZE = 0.0                      # eval / definition scenes: every item >= 1 cm from the hand
    GRAZE_WIN = HARD_GRAZE             # graze gap window (collection only)
    # (slot, candidate items, P(present), x range | "wall", y range, yaw mode, (gap lo, gap hi))
    SLOTS = (
        ("behind_cup", ("oil", "wine", "towel", "can_a"), 0.9, "wall", (0.13, 0.27), "any", HARD_FAR),
        ("right_of_cup", ("soap", "can_c", "salt", "pepper"), 0.95, (0.56, 0.74), (0.03, 0.12), "any", HARD_NEAR),
        ("carry_back", ("can_b", "can_c", "oil", "soap"), 0.9, (0.52, 0.68), (-0.16, 0.0), "any", HARD_NEAR),
        # short shakers only: the gripper base sweeps this strip at z ~0.98-1.1 (taller items always hit)
        ("carry_front", ("salt", "pepper"), 0.75, (0.30, 0.44), (-0.20, 0.10), "any", HARD_NEAR),
        ("wall_mid", ("can_b", "can_a", "kettle", "board"), 0.8, "wall", (-0.22, -0.02), "any", HARD_FAR),
        ("behind_tray", ("board", "kettle", "wine", "towel"), 0.8, "wall", (-0.42, -0.26), "any", HARD_FAR),
        ("right_of_tray", ("toaster", "kettle", "bread"), 0.85, "wall", (-0.76, -0.58), "wall", HARD_FAR),
        ("front_right", ("bowl", "bread"), 0.6, (0.40, 0.50), (-0.80, -0.64), "wall", HARD_FAR),
        ("by_cooktop", ("salt", "pepper", "oil"), 0.6, "wall", (-0.84, -0.80), "any", HARD_FAR),
    )

    def _cell_params(self, cell: int) -> dict[str, float]:
        return hard_cell_params(cell)

    def _apply_theta(self, env, th) -> None:
        super()._apply_theta(env, th)
        cup = th["cup_rest_pose"]
        # The arm already stands in this episode's (noisy) ready pose (randomize_scene runs before
        # _sample_task), so the first leg of the swept hand starts from the real TCP pose.
        # The base itself is (re)placed later (_select_pickup_object -> _sample_and_place_robot), so
        # carry the base-relative TCP pose over to this episode's base pose (BASE_XYZ, yaw 0).
        start = None
        try:
            rv = env.current_robot.robot_view
            tcp_w = np.asarray(rv.get_move_group(rv.get_gripper_movegroup_ids()[0]).leaf_frame_to_world, float)
            base_now = np.asarray(rv.base.pose, float)
            base_ep = np.eye(4)
            base_ep[:3, :3] = R.from_euler("z", float(self._cur_base_yaw)).as_matrix()
            base_ep[:3, 3] = np.asarray(self._cur_base_xyz, float)
            start = base_ep @ np.linalg.inv(base_now) @ tcp_w
            if not np.all(np.isfinite(start)) or np.linalg.norm(start[:3, 3] - HOME_TCP_EST) > 0.25:
                start = None
        except Exception as e:  # pragma: no cover - fall back to the nominal ready pose
            log.debug(f"[PCKitchenHard] start pose unavailable: {e!r}")
        th["start_tcp"] = None if start is None else [float(v) for v in start[:3, 3]]
        th["_hand_cloud"] = pnp_hand_cloud((float(cup[0]), float(cup[1])), th["tray_xy"], start)

    def _item_gap(self, item: str, x: float, y: float, lo, hi, top: float, th, legs=None) -> float:
        """Smallest horizontal gap (m) from the item to the swept hand (legs: restrict to these)."""
        P, r, L = th["_hand_cloud"]
        sel = P[:, 2] - r < top + HARD_VMARGIN
        if legs is not None:
            sel &= np.isin(L, legs)
        if not np.any(sel):
            return float("inf")
        if item in XML_ITEMS:       # canisters are cylinders: exact circle distance
            d = np.hypot(P[sel, 0] - x, P[sel, 1] - y) - XML_ITEMS[item]["r"]
        else:
            d = _aabb2_dist(P[sel, :2], np.asarray(lo, float), np.asarray(hi, float))
        return float((d - r[sel]).min())

    def _fill_slots(self, env, th) -> None:
        used = set()
        th["clutter"] = []
        graze_slot = None
        if self.P_GRAZE > 0 and np.random.random() < self.P_GRAZE:
            near = [s[0] for s in self.SLOTS if s[6] == HARD_NEAR and s[0] in GRAZE_SLOTS]
            graze_slot = str(np.random.choice(near)) if near else None
        th["graze_slot"] = graze_slot
        for slot, items, p_present, xr, yr, yaw_mode, (gap_lo, gap_hi) in self.SLOTS:
            if slot == graze_slot:
                p_present, (gap_lo, gap_hi) = 1.0, self.GRAZE_WIN
            if np.random.random() > p_present:
                continue
            choices = [i for i in items if i not in used]
            if slot == graze_slot:
                choices = [i for i in choices if i in XML_ITEMS] or choices
            if not choices:
                continue
            item = str(np.random.choice(choices))
            for _ in range(self.PLACE_TRIES):
                against_wall = isinstance(xr, str)
                x = BACKSPLASH_X - 0.2 if against_wall else float(np.random.uniform(*xr))
                y = float(np.random.uniform(*yr))
                yaw = 0.0
                if item == "board":
                    t, half_w, half_h = BOARD_HALF
                    lo = np.array([BACKSPLASH_X - 0.10, y - half_w])
                    hi = np.array([BACKSPLASH_X, y + half_w])
                    top = COUNTER_Z + 2 * half_h
                else:
                    yaw = {"any": float(np.random.uniform(0, 2 * np.pi)),
                           "wall": float(np.random.uniform(-0.15, 0.15)),
                           "side": float(np.pi / 2 + np.random.uniform(-0.15, 0.15))}[yaw_mode]
                    lo, hi = self._item_footprint(item, x, y, yaw)
                    if against_wall:
                        shift = (BACKSPLASH_X - float(np.random.uniform(0.004, 0.02))) - hi[0]
                        x += shift
                        lo, hi = lo + [shift, 0], hi + [shift, 0]
                    top = COUNTER_Z + self._item_geom[item]["h"]
                if not self._footprint_ok(lo, hi, th):
                    continue
                if slot == graze_slot:
                    # brush the hand / carried can on the carry leg only; clear the other legs
                    gap = self._item_gap(item, x, y, lo, hi, top, th, legs=[LEG_CARRY])
                    if not (gap_lo <= gap <= gap_hi):
                        continue
                    if self._item_gap(item, x, y, lo, hi, top, th, legs=[LEG_APPROACH, LEG_RETREAT]) < HARD_GAP_MIN:
                        continue
                else:
                    gap = self._item_gap(item, x, y, lo, hi, top, th)
                    if not (gap_lo <= gap <= gap_hi):
                        continue
                if item == "board":
                    lo, hi = self._place_board(env, y)
                    rec = {"item": item, "slot": slot, "xy": [float((lo[0] + hi[0]) / 2), y], "yaw": 0.0}
                else:
                    self._place_item(env, item, x, y, yaw)
                    rec = {"item": item, "slot": slot, "xy": [x, y], "yaw": yaw}
                rec.update(lo=[float(v) for v in lo], hi=[float(v) for v in hi], top=float(top),
                           hand_gap=float(gap), near=bool(gap_hi < 1.0), graze=bool(slot == graze_slot))
                th["clutter"].append(rec)
                th["_placed"].append((lo, hi))
                used.add(item)
                break
        th["n_clutter"] = len(th["clutter"])
        th["n_near"] = int(sum(1 for c in th["clutter"] if c.get("near")))
        gaps = [c["hand_gap"] for c in th["clutter"] if np.isfinite(c["hand_gap"])]
        th["min_hand_gap"] = float(min(gaps)) if gaps else None

    def _finish_task(self, task, th) -> None:
        super()._finish_task(task, th)
        task.__class__ = self.TASK_CLS or PCKitchenPnPHardTask


class PCKitchenPnPHardCheckSampler(PCKitchenPnPHardSampler):
    """Preflight: every slot filled (maximum clutter), same geometry otherwise."""

    SLOTS = tuple((s, i, 1.0, xr, yr, ym, g) for (s, i, _p, xr, yr, ym, g) in PCKitchenPnPHardSampler.SLOTS)


# ---- light-contact save rule (collection only) ----
LIGHT_CONTACT_MAX_FRAMES = 150     # 2 ms control frames with any arm/hazard contact (= 0.30 s)
LIGHT_CONTACT_MAX_DEPTH = 0.010    # deepest arm/hazard penetration (m)
_HAZARD_CLASSES = ("hazard_bar", "other_environment", "clutter", "mounted_fixture")


class LightContactAudit:
    """Arm (and carried can) vs hazard contact, sampled after every 2 ms control step through the
    task's ``_contact_audit_hook`` with the evaluator's classifier (pact_place_contact_audit):
    clutter / other_environment / hazard_bar / mounted_fixture are hazards; the target and the
    tray are not."""

    def __init__(self) -> None:
        self._seen: set[float] = set()
        self.frames = 0
        self.hazard_frames = 0
        self.max_depth = 0.0
        self.first_step: int | None = None
        self.bodies: dict[str, int] = {}
        self.parts: dict[str, dict[str, int]] = {}    # hazard -> {robot body / target: pair count}

    def observe(self, env, step: int) -> None:
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact, place_environment_contact_pairs

        t = round(float(env.current_data.time), 12)
        if t in self._seen:
            return
        self._seen.add(t)
        self.frames += 1
        names: set[str] = set()
        for pair in place_environment_contact_pairs(env):
            if classify_contact(pair) not in _HAZARD_CLASSES:
                continue
            self.max_depth = max(self.max_depth, max(0.0, -float(pair["distance_m"])))
            r1, r2 = pair["root1"], pair["root2"]
            if r1.startswith("robot_0/") or r1.startswith("cavity_obj_"):
                root, geom, mine = r2, pair["geom2"], pair["body1"]
            else:
                root, geom, mine = r1, pair["geom1"], pair["body2"]
            env_name = root if root and root != "world" else geom
            names.add(env_name)
            parts = self.parts.setdefault(env_name, {})
            mine = str(mine).replace("robot_0/", "")
            parts[mine] = parts.get(mine, 0) + 1
        if names:
            self.hazard_frames += 1
            if self.first_step is None:
                self.first_step = int(step)
            for n in names:
                self.bodies[n] = self.bodies.get(n, 0) + 1

    def light(self) -> bool:
        return self.hazard_frames <= LIGHT_CONTACT_MAX_FRAMES and self.max_depth <= LIGHT_CONTACT_MAX_DEPTH

    def summary(self) -> dict[str, Any]:
        return {
            "rule": f"save iff success and hazard_frames <= {LIGHT_CONTACT_MAX_FRAMES} (2 ms) "
                    f"and max_depth <= {LIGHT_CONTACT_MAX_DEPTH} m",
            "frames": int(self.frames),
            "hazard_frames": int(self.hazard_frames),
            "hazard_time_s": float(self.hazard_frames) * 0.002,
            "max_depth_m": float(self.max_depth),
            "first_step": self.first_step,
            "bodies": dict(sorted(self.bodies.items())),
            "robot_parts": {k: dict(sorted(v.items())) for k, v in sorted(self.parts.items())},
            "light_contact": bool(0 < self.hazard_frames and self.light()),
            "contact_free": bool(self.hazard_frames == 0),
        }


class PCKitchenPnPHardTask(PCKitchenPnPTask):
    """pc_kitchen_pnp_hard eval / definition task: the pc_kitchen_pnp success test, unchanged
    (no contact gate; the evaluator scores contacts with its own audit)."""


class PCKitchenPnPHardCollectTask(PCKitchenPnPHardTask):
    """Collection only: success AND light hazard contact (<= 150 x 2 ms frames, <= 1 cm deep).
    Heavy-contact successes count as failures, so they are not saved. The verdict and the
    contact stats (frames, bodies, depth) go to scene_params['hazard_contact'] in obs_scene."""

    def reset(self):
        self._light_audit = LightContactAudit()
        self._contact_audit_hook = self._light_audit
        self._light_reject_logged = False
        return super().reset()

    def judge_success(self) -> bool:
        ok = super().judge_success()
        audit = getattr(self, "_light_audit", None)
        if audit is None:
            return ok
        summ = audit.summary()
        summ["task_success"] = bool(ok)
        summ["saved"] = bool(ok and audit.light())
        params = getattr(self, "scene_params", None)
        if isinstance(params, dict):
            params["hazard_contact"] = summ
        if ok and not audit.light() and not getattr(self, "_light_reject_logged", False):
            self._light_reject_logged = True
            log.info(f"[PCKitchenHard] success rejected (heavy contact): hazard_frames={audit.hazard_frames} "
                     f"max_depth={audit.max_depth:.4f} bodies={audit.bodies} parts={audit.parts}")
        return bool(ok and audit.light())

    def _maybe_log_obstacle_diag(self, success: bool) -> None:
        already = getattr(self, "_obstacle_diag_logged", True)
        super()._maybe_log_obstacle_diag(success)
        audit = getattr(self, "_light_audit", None)
        if audit is not None and not already and getattr(self, "_obstacle_diag_logged", False):
            log.info(f"[PCKitchenHard] hazard contact (2 ms frames): {audit.hazard_frames} "
                     f"max_depth={audit.max_depth:.4f} bodies={audit.bodies or 'none'}")


PCKitchenPnPHardSampler.TASK_CLS = PCKitchenPnPHardTask


class PCKitchenPnPHardCollectSampler(PCKitchenPnPHardSampler):
    """Collection: the eval scenes, plus a grazing NEAR item in P_GRAZE of the episodes; the task
    applies the light-contact save rule."""

    TASK_CLS = PCKitchenPnPHardCollectTask
    P_GRAZE = 0.30


class PCKitchenPnPHardCollectGrazeSampler(PCKitchenPnPHardCollectSampler):
    """Collection top-up (part 4): the same scenes, grazes drawn more often and ~4 mm deeper on the
    nominal carry leg, i.e. compensating the measured inward offset of the executed carry, so most
    of them touch (light-contact share of the parts 1-3 runs came out ~10 %, target ~15 %)."""

    P_GRAZE = 0.5
    GRAZE_WIN = (-0.006, -0.002)


class PCKitchenPnPHardCollectCheckSampler(PCKitchenPnPHardCheckSampler):
    TASK_CLS = PCKitchenPnPHardCollectTask
    P_GRAZE = 0.5


class PCKitchenPnPHardGrazeProbeSampler(PCKitchenPnPHardCheckSampler):
    """Harness only: every episode grazes (to measure how heavy a graze contact is)."""

    TASK_CLS = PCKitchenPnPHardCollectTask
    P_GRAZE = 1.0


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write-scenes", action="store_true")
    ap.add_argument("--write-hard-scenes", action="store_true")
    args = ap.parse_args()
    if args.write_scenes:
        for p in write_scenes():
            print(p)
    if args.write_hard_scenes:
        for p in write_hard_scenes():
            print(p)
