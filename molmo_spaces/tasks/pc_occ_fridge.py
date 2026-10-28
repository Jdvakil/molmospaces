"""``pc_occ_fridge``: reach into an open refrigerator, pick a jar from the back of the lowest shelf
and set it on a small tray at the front of the shelf, while two wine bottles lying in a wine rack
that clips under the shelf above (left or right of the reach lane) threaten the SKIN-COVERED wrist
knuckle, never the gripper.

Why this suite exists (2026-10-15, ICLR rebuttal, "occluded natural items")
---------------------------------------------------------------------------
The earlier hidden-hazard suites hid artificial-looking obstacles (basket, valve box, drain pipe,
radio) behind trim. Here the hazard is an ordinary household object with a real mesh (the THOR
``Wine_Bottle_1`` glass bottle) stored where people store it (a chrome under-shelf wine rack, a
standard fridge accessory), and what hides it from the cameras is the fridge itself: the opaque
frosted shelf above the work level and its front trim. Everything else in the fridge (milk and
juice cartons, eggs, cheese, jars, the cartons and bottles in the door bins) is present in every
episode and drawn independently of the rack side.

Why the hazard hangs from the shelf and does not stand on it (design note, measured 2026-10-15)
---------------------------------------------------------------------------------------------
With the FR3 hand pointing down (the only way to pick a jar from the back of a fridge shelf
without sweeping the shelf), the skinned links that pass near the jar's lane are the wrist knuckle
(fr3_link6) and the forearm (fr3_link5): 0.21-0.40 m ABOVE the TCP, while link7 / the gripper
reach 0.21 m above it and +-3.7 cm sideways. A bottle STANDING on the work shelf beside the lane
is a column that reaches the gripper's height too: keeping the gripper >= 5 cm from it puts it
>= 8.7 cm from the lane, where only the forearm's +y lean (+13 cm) reaches it, so no binary
left/right choice exists (any fixed tilt between ~+8 and ~+11 deg passes between the two sides).
It would also be seen by the wrist camera, which looks down and covers +-17 cm at shelf level.
Front cartons cannot hide the knuckle band either: the exo camera looks down at ~22 deg, so a
25 cm carton hides only things lower than itself behind it, and the knuckle band is 32-51 cm above
the shelf. A hazard in the band 0.27-0.36 m above the TCP, hanging from the shelf above, is
>= 5 cm above the gripper / link7 on every path, never in the wrist camera (above the hand) and
inside the shelf's "lintel shadow" for the exo camera.

Scene (``custom_scenes/pc_occ_fridge_c<k>.xml``, :func:`write_pc_occ_fridge_scenes`)
------------------------------------------------------------------------------------
A closed kitchen: a 76 cm top-freezer refrigerator in a niche between a counter run (base
cabinets, countertop, wall cabinets, left) and a tall pantry cabinet (right). The fresh-food door
is hinged on the right and stands open 105-115 deg (three door bins with bottles and cartons).
Inside: white liner, two crisper drawers at the bottom, the work shelf (the crisper cover, top at
0.445-0.455 m), the shelf above it (opaque frosted glass with a grey front trim, underside 0.38 m
above the carry height of the TCP), a top shelf, a fridge light. Shelves start 15 cm behind the
cabinet face (deep door bins nest there when the door is closed). The upper shelves hold real-mesh
cartons, eggs and cheese and primitive jars (static per cell); 4-6 items (real-mesh milk / juice
cartons, egg box, cheese, juice box, a 2 L and a 1.5 L bottle, a yogurt tub, butter, a leftovers
box) are re-placed every episode on the work shelf off the lane (>= 13-16 cm from its centre).

Hidden hazard (``pact_intrusion_winerack_left`` / ``_right``, scored by the evaluator as ``hazard_bar``)
------------------------------------------------------------------------------------------------------
A chrome wire wine rack clipped under the shelf above holds two THOR wine bottles lying front to
back, necks toward the door, on the LEFT or the RIGHT of the reach lane (50/50 per episode, drawn
independently of the jar, the tray, the items, the cell and the start pose). The neck tips are
18 cm behind the shelf's front edge (33 cm behind the cabinet face), inside the shelf's "lintel
shadow" for the exo camera; a chrome neck rest runs under both necks (so a wrist swung far toward
the rack cannot slip between the two necks). The inner neck's lane-side surface is 3.0-3.6 cm
(left) / 2.2-2.8 cm (right) beside the lane centre; the necks hang 0.30-0.335 m above the TCP at the
carry height (the bodies, further back, down to 0.27 m): link7 / the gripper (top 0.214 m above
the TCP) stay >= 5 cm below, the skinned wrist knuckle (top 0.344 m) reaches into the necks when
the hand is at the jar.

Task + expert (scripted TCP segments, hand pointing down)
--------------------------------------------------------
Pick the jar (``cavity_obj_0/fridge_jar``) standing 33-34.5 cm behind the cabinet face (under the
bottle necks' depth) and set it on a small square plate (``place_receptacle_mat``, 10 cm) 22.5-23 cm
behind the face, just behind the shelf trim (pc_shelf_pnp success rule). Expert = the
pc_sink_blind expert: pre-insert pose in front of the fridge at the carry height -> slide in
untilted until the TCP is :data:`D_DIV` (24 cm) inside -> dwell :data:`DWELL_S` (the wrist skin
faces the bottle necks 5 cm ahead; the step-100 chunk query falls in the pause; disclosed, as
pc_shelf_blind / pc_sink_blind) -> TILT the hand :data:`TILT_DEG` about the reach axis, fingers
leaning toward the rack side, so the wrist knuckle swings ~5.5 cm to the other side -> slide on
above the jar -> lower -> close -> lift -> carry to above the plate -> lower -> open -> lift ->
untilt (the wrist is in front of the necks there) -> back out -> start pose.

Collection-only additions (switched on by the collection configs, never by the eval config):
DART-style executed-action noise, the contact-tolerant save gate of ``pc_shelf_hard`` and one skin
reading per policy step. Per-link clearance to the rack is recorded in ``obs_scene['pc_clearance']``.
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

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pc_shelf import SCENE_DIR, PCShelfPnPSampler, _T
from molmo_spaces.tasks.pc_shelf_blind import CLEAR_BODIES, blind_tool_rotation, robot_vertex_sets
from molmo_spaces.tasks.pc_sink_blind import (
    PC_SINK_BLIND_READY_QPOS,
    PCSinkBlindExpert,
    PCSinkBlindExpertConfig,
    PCSinkBlindTask,
    primitive_distance,
)
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask  # noqa: F401

log = logging.getLogger(__name__)

PC_OCC_FRIDGE_ENV_VERSION = "pc_occ_fridge_v1"
N_CELLS = 8

# --------------------------------------------------------------------------------------------
# Geometry (metres, world frame; robot base at the origin facing +x)
# --------------------------------------------------------------------------------------------
H_BASE = 0.35            # robot deck height = link0 z (the standard pc-suite mobile base)
FR_W = 0.76              # fridge outer width (30 in)
FR_WALL = 0.045          # insulated side walls
FR_IN_HW = FR_W / 2 - FR_WALL     # interior half width (0.335)
DEPTH_IN = 0.62          # cabinet face -> interior back wall
FR_BACK = 0.70           # cabinet face -> outer back
SHELF_SET = 0.15         # shelves' front edge behind the cabinet face (the deep door bins nest there)
GLASS_T = 0.008          # shelf glass
TRIM_DROP = 0.012        # front trim lip below the glass underside
TRIM_RISE = 0.012        # ... and above the glass top
TRIM_D = 0.022           # trim depth (x)
Z_FLOOR_IN = 0.10        # fridge interior floor (above the toe grille)
Z_FRESH_TOP = 1.40       # fresh-food compartment ceiling
Z_FREEZER_TOP = 1.74     # top of the fridge
INS_CLEAR = 0.030        # TCP above the jar top while sliding in / carrying
GRASP_BELOW_TOP = 0.030  # TCP below the jar body top at the grasp
X_PRE = 0.07             # pre-insert TCP stand-off in front of the cabinet face
D_DIV = 0.24             # TCP depth (behind the face) of the common untilted insertion where the tilt starts
DWELL_S = 1.3            # pause there before tilting (the step-100 chunk query falls in it; disclosed)
TILT_DEG = 10.0          # expert hand tilt (deg): swings the wrist knuckle ~5.5 cm sideways
BOARD_ABOVE_TCP = 0.380  # upper shelf glass underside above the TCP carry height
WRIST_TOP = 0.344        # skinned wrist knuckle top above the TCP, hand straight down
HAND_TOP = 0.214         # fr3_link7 / gripper top above the TCP (tilted 10 deg)
MAT_HALF = 0.050         # tray half size (a small square plate)
MAT_HALF_T = 0.004

# wine bottle (THOR Wine_Bottle_1): body r 0.0446, length 0.325, neck r 0.0155 (mesh profile)
WB_R = 0.0446
WB_L = 0.325
WB_NECK_R = 0.016
WB_NECK_L = 0.095        # neck tip -> shoulder start
WB_BODY_X0 = 0.130       # neck tip -> full-radius body
RACK_TOP_GAP = 0.015     # bottle top below the glass underside
RACK_WIRE = 0.003
RACK_PITCH = 0.0972      # bottle spacing in the rack
RACK_SETBACK = 0.180     # neck tips behind the shelf's front edge (inside the lintel shadow; keeps the
                         # forearm, which reaches x_f + 0.315 at the deepest grasp, clear of the necks)
NECK_BAR_X = 0.020       # the rack's front neck rest (a chrome tube under both necks) behind the neck tips
NECK_BAR_R = 0.005

_OBJ = Path(ASSETS_DIR) / "objects"
_WINE_DIR = _OBJ / "thor" / "Kitchen Objects" / "WineBottle" / "Prefabs" / "Wine_Bottle_1"
WINE_GLASS_OBJ = _WINE_DIR / "Wine_Bottle_1_wine_bottle_1" / "Wine_Bottle_1_wine_bottle_1_1.obj"
WINE_LABEL_OBJ = _WINE_DIR / "Wine_Bottle_1_wine_bottle_1" / "Wine_Bottle_1_wine_bottle_1_0.obj"
WINE_LABEL_PNG = _WINE_DIR / "Labels.png"


def _objaverse(uid: str) -> dict[str, str]:
    d = _OBJ / "objaverse" / uid
    return dict(obj=str(d / f"{uid}_visual.obj"), png=str(d / f"{uid}_visual_0.png"))


# Real-mesh fridge items (objaverse, Y-up, centred, mirrored x as in their own XML). half = the
# world half extents after standing them up (x, y, z).
MESH_ITEMS = {
    "milk": dict(**_objaverse("87e1ed3a6e59459cbf8cf75da77bb587"), half=(0.050, 0.0516, 0.100)),
    "juice": dict(**_objaverse("55d0a4c01b9e4a2cab8112162ea61cfb"), half=(0.0845, 0.0402, 0.100)),
    "eggs": dict(**_objaverse("302286c578e146d9a0823426e5ab5a8f"), half=(0.0562, 0.039, 0.0397)),
    "cheese": dict(**_objaverse("48d3e1ba9b4040dc9b10ff68b53a6234"), half=(0.0787, 0.0357, 0.025)),
    "juicebox": dict(**_objaverse("dd7183e276b3406c930e18e79a7b8c6c"), half=(0.0374, 0.0233, 0.075)),
}
Q_UPRIGHT = (0.70710678, 0.70710678, 0.0, 0.0)   # mesh +y -> world +z


@dataclass(frozen=True)
class FridgeCell:
    """One kitchen / fridge layout family."""

    x_f: float          # cabinet face x
    z_s: float          # work shelf (crisper cover) top
    jar: int            # index into JARS
    door_deg: float     # fresh-food door opening angle
    shell: tuple[float, float, float]   # fridge outer colour (white / stainless / black ...)
    cabinet: tuple[float, float, float]
    counter: tuple[float, float, float]
    wall: tuple[float, float, float]


# Fields: x_f, z_s, jar, door_deg, fridge shell rgb, kitchen cabinet rgb, counter rgb, wall rgb
PC_OCC_FRIDGE_CELLS: tuple[FridgeCell, ...] = (
    FridgeCell(0.430, 0.450, 0, 108.0, (0.94, 0.94, 0.93), (0.93, 0.93, 0.91), (0.20, 0.20, 0.22), (0.90, 0.88, 0.84)),
    FridgeCell(0.425, 0.445, 1, 112.0, (0.74, 0.75, 0.77), (0.36, 0.45, 0.55), (0.88, 0.87, 0.84), (0.93, 0.92, 0.88)),
    FridgeCell(0.430, 0.455, 2, 105.0, (0.16, 0.16, 0.17), (0.55, 0.40, 0.27), (0.45, 0.44, 0.42), (0.86, 0.86, 0.83)),
    FridgeCell(0.425, 0.450, 3, 115.0, (0.94, 0.94, 0.93), (0.53, 0.61, 0.54), (0.93, 0.93, 0.92), (0.91, 0.89, 0.85)),
    FridgeCell(0.430, 0.445, 4, 110.0, (0.74, 0.75, 0.77), (0.25, 0.27, 0.30), (0.78, 0.74, 0.68), (0.88, 0.90, 0.92)),
    FridgeCell(0.425, 0.455, 5, 106.0, (0.94, 0.94, 0.93), (0.80, 0.73, 0.62), (0.15, 0.15, 0.16), (0.92, 0.90, 0.86)),
    FridgeCell(0.430, 0.450, 6, 113.0, (0.16, 0.16, 0.17), (0.62, 0.20, 0.18), (0.90, 0.90, 0.88), (0.89, 0.87, 0.83)),
    FridgeCell(0.425, 0.445, 7, 109.0, (0.74, 0.75, 0.77), (0.88, 0.86, 0.80), (0.32, 0.30, 0.28), (0.90, 0.90, 0.88)),
)

JARS = (  # radius, body height, body rgba, lid rgba, label rgba, label
    (0.0330, 0.085, (0.55, 0.08, 0.12), (0.85, 0.80, 0.70), (0.95, 0.92, 0.85), "strawberry jam jar"),
    (0.0320, 0.090, (0.92, 0.92, 0.88), (0.20, 0.45, 0.80), (0.30, 0.55, 0.85), "yogurt cup"),
    (0.0335, 0.095, (0.45, 0.55, 0.20), (0.90, 0.75, 0.20), (0.95, 0.95, 0.90), "pickle jar"),
    (0.0325, 0.080, (0.85, 0.65, 0.15), (0.25, 0.25, 0.27), (0.95, 0.90, 0.80), "honey jar"),
    (0.0330, 0.088, (0.80, 0.20, 0.10), (0.95, 0.95, 0.95), (0.95, 0.85, 0.30), "salsa jar"),
    (0.0320, 0.092, (0.95, 0.85, 0.30), (0.60, 0.15, 0.15), (0.95, 0.95, 0.95), "mustard jar"),
    (0.0335, 0.082, (0.35, 0.20, 0.12), (0.85, 0.70, 0.20), (0.95, 0.92, 0.85), "chocolate spread jar"),
    (0.0325, 0.086, (0.95, 0.95, 0.92), (0.85, 0.20, 0.25), (0.95, 0.75, 0.80), "pudding cup"),
)

PC_OCC_FRIDGE_SCENES: tuple[str, ...] = tuple(str(SCENE_DIR / f"pc_occ_fridge_c{k}.xml") for k in range(N_CELLS))


def cell_heights(c: FridgeCell) -> dict[str, float]:
    """Heights that follow from the cell's jar: carry TCP height, shelf above, rack bottom."""
    h = JARS[c.jar][1]
    z_hi = c.z_s + h + INS_CLEAR
    z_ub = z_hi + BOARD_ABOVE_TCP                    # upper shelf glass underside
    axis = -RACK_TOP_GAP - WB_R                      # bottle axis below the glass underside
    rack_bot = axis - WB_R - 2 * RACK_WIRE           # lowest rack point below the glass underside
    return dict(z_hi=z_hi, z_ub=z_ub, z_ut=z_ub + GLASS_T, rack_bottom=z_ub + rack_bot,
                bottle_axis=z_ub + axis, jar_h=h, jar_r=JARS[c.jar][0],
                bottom_above_tcp=BOARD_ABOVE_TCP + rack_bot)


# --------------------------------------------------------------------------------------------
# Scene XML
# --------------------------------------------------------------------------------------------
def _f(v: float) -> str:
    return f"{float(v):.4f}"


def _box(name: str, lo, hi, material: str | None = None, rgba=None, extra: str = "", cls: str = "furn") -> str:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    mat = f' material="{material}"' if material else ""
    col = "" if rgba is None else f' rgba="{" ".join(_f(v) for v in rgba)}"'
    return (f'    <geom name="{name}" class="{cls}" type="box" pos="{_f(c[0])} {_f(c[1])} {_f(c[2])}" '
            f'size="{_f(h[0])} {_f(h[1])} {_f(h[2])}"{mat}{col}{extra}/>\n')


def _rbox(name: str, center, half, yaw: float, material: str, extra: str = "") -> str:
    q = R.from_euler("z", yaw).as_quat(scalar_first=True)
    return (f'    <geom name="{name}" class="furn" type="box" pos="{_f(center[0])} {_f(center[1])} {_f(center[2])}" '
            f'size="{_f(half[0])} {_f(half[1])} {_f(half[2])}" quat="{" ".join(_f(v) for v in q)}" material="{material}"{extra}/>\n')


def _cyl(name: str, pos, r: float, hh: float, material: str | None = None, rgba=None, quat=None,
         kind: str = "cylinder", extra: str = "") -> str:
    mat = f' material="{material}"' if material else ""
    col = "" if rgba is None else f' rgba="{" ".join(_f(v) for v in rgba)}"'
    q = "" if quat is None else f' quat="{" ".join(_f(v) for v in quat)}"'
    size = f"{_f(r)} {_f(hh)}" if kind != "sphere" else f"{_f(r)}"
    return (f'    <geom name="{name}" class="furn" type="{kind}" pos="{_f(pos[0])} {_f(pos[1])} {_f(pos[2])}" '
            f'size="{size}"{q}{mat}{col}{extra}/>\n')


Q_ALONG_X = (0.70710678, 0.0, 0.70710678, 0.0)   # cylinder axis (local z) -> world x
Q_ALONG_Y = (0.70710678, 0.70710678, 0.0, 0.0)   # cylinder axis -> world y
_NOCOL = ' contype="0" conaffinity="0"'


def _mesh_geom(name: str, kind: str, pos, yaw: float = 0.0) -> str:
    """A static real-mesh fridge item standing on a surface: pos = its bottom centre."""
    half = MESH_ITEMS[kind]["half"]
    q = (R.from_euler("z", yaw) * R.from_quat(Q_UPRIGHT, scalar_first=True)).as_quat(scalar_first=True)
    return (f'    <geom name="{name}" type="mesh" mesh="mesh_{kind}" material="mat_{kind}" '
            f'pos="{_f(pos[0])} {_f(pos[1])} {_f(pos[2] + half[2])}" quat="{" ".join(_f(v) for v in q)}"{_NOCOL}/>\n'
            + f'    <geom name="{name}_col" class="furn" type="box" pos="{_f(pos[0])} {_f(pos[1])} {_f(pos[2] + half[2])}" '
            f'size="{_f(half[0])} {_f(half[1])} {_f(half[2])}" quat="{" ".join(_f(v) for v in R.from_euler("z", yaw).as_quat(scalar_first=True))}" group="3" rgba="0 0 0 0"/>\n')


def _jar_geoms(name: str, pos, r: float, h: float, body, lid) -> str:
    """Static primitive jar / bottle: glass body, lid, label band (pos = bottom centre)."""
    x, y, z = pos
    return (_cyl(name, (x, y, z + h / 2), r, h / 2, rgba=list(body) + [1.0])
            + _cyl(name + "_lid", (x, y, z + h + 0.008), r * 1.02, 0.008, rgba=list(lid) + [1.0], extra=_NOCOL)
            + _cyl(name + "_label", (x, y, z + h * 0.45), r * 1.005, h * 0.22, rgba=[0.95, 0.94, 0.90, 1.0], extra=_NOCOL))


def _door_geoms(c: FridgeCell) -> list[str]:
    """Fresh-food door hinged on the right (-y) edge of the cabinet, opened ``door_deg``."""
    g: list[str] = []
    hx, hy = c.x_f, -FR_W / 2
    a = np.deg2rad(c.door_deg)
    # closed: from the hinge toward +y in the face plane; opening swings the free edge toward -x
    u = np.array([-np.sin(a), np.cos(a)])          # hinge -> free edge
    n = np.array([np.cos(a), np.sin(a)])           # door inner-face normal (into the fridge when closed)
    yaw = float(np.arctan2(u[1], u[0]))
    W, T = FR_W - 0.004, 0.065
    z0, z1 = 0.085, Z_FRESH_TOP + 0.03
    ctr = np.array([hx, hy]) + u * W / 2 - n * T / 2
    g.append(_rbox("fridge_door", (ctr[0], ctr[1], (z0 + z1) / 2), (W / 2, T / 2, (z1 - z0) / 2), yaw, "mat_shell"))
    # inner liner panel + gasket
    lin = np.array([hx, hy]) + u * W / 2 + n * 0.004
    g.append(_rbox("fridge_door_liner", (lin[0], lin[1], (z0 + z1) / 2), ((W - 0.07) / 2, 0.004, (z1 - z0 - 0.07) / 2), yaw, "mat_liner"))
    # handle on the outer face near the free edge
    hp = np.array([hx, hy]) + u * (W - 0.06) - n * (T + 0.035)
    g.append(_cyl("fridge_door_handle", (hp[0], hp[1], z1 - 0.40), 0.012, 0.28, material="mat_handle"))
    for k, z in enumerate((0.36, 0.72, 1.06)):
        # door bins: floor + front rail (+ transparent front), on the inner face
        bc = np.array([hx, hy]) + u * W / 2 + n * (0.008 + 0.055)
        g.append(_rbox(f"fridge_door_bin{k}_floor", (bc[0], bc[1], z), ((W - 0.12) / 2, 0.055, 0.006), yaw, "mat_bin"))
        fc = np.array([hx, hy]) + u * W / 2 + n * (0.008 + 0.108)
        g.append(_rbox(f"fridge_door_bin{k}_front", (fc[0], fc[1], z + 0.045), ((W - 0.12) / 2, 0.003, 0.045), yaw, "mat_bin_front"))
        # bin contents (static): bottles / cartons standing in the bin
        for j, (kind, along) in enumerate((("bottle", 0.16), ("carton", 0.36), ("bottle", 0.56))):
            p = np.array([hx, hy]) + u * along + n * (0.008 + 0.055)
            if kind == "bottle":
                hgt = 0.24 if k < 2 else 0.20
                g.append(_cyl(f"fridge_door_bin{k}_item{j}", (p[0], p[1], z + 0.006 + hgt / 2), 0.035, hgt / 2,
                              rgba=[(0.25, 0.45, 0.20), (0.85, 0.55, 0.15), (0.70, 0.12, 0.12)][(j + k) % 3] + (1.0,)))
                g.append(_cyl(f"fridge_door_bin{k}_item{j}_cap", (p[0], p[1], z + 0.006 + hgt + 0.012), 0.014, 0.012,
                              rgba=[0.92, 0.92, 0.92, 1.0], extra=_NOCOL))
            else:
                g.append(_mesh_geom(f"fridge_door_bin{k}_item{j}", "juice" if k != 1 else "milk", (p[0], p[1], z + 0.006),
                                    yaw=yaw))
    return g


def build_scene_xml(cell_index: int) -> str:
    """The static MJCF of one kitchen cell (room, counter run, pantry, fridge with shelves, door,
    decor). The jar, the tray, the re-posed items and the wine racks are injected at runtime."""
    c = PC_OCC_FRIDGE_CELLS[cell_index]
    hz = cell_heights(c)
    rng = np.random.default_rng(8900 + cell_index)
    xf = c.x_f
    xb = xf + FR_BACK
    xw = xb + 0.03                         # kitchen wall behind the fridge
    hw = FR_W / 2
    room_l, room_r = hw + 1.45, -hw - 1.10
    rear_x = -1.45
    ceil = 2.55
    z_s, z_ub, z_ut = c.z_s, hz["z_ub"], hz["z_ut"]
    xs = xf + SHELF_SET                    # shelves' front edge
    xin_b = xf + DEPTH_IN                  # interior back
    g: list[str] = []
    # ---- room shell (every skin ray ends on a real surface)
    g.append(_box("wall_kitchen", (xw, room_r - 0.06, 0), (xw + 0.06, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_rear", (rear_x - 0.06, room_r - 0.06, 0), (rear_x, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_left", (rear_x, room_l, 0), (xw, room_l + 0.06, ceil), material="mat_wall"))
    g.append(_box("wall_right", (rear_x, room_r - 0.06, 0), (xw, room_r, ceil), material="mat_wall"))
    g.append(_box("ceiling", (rear_x - 0.06, room_r - 0.06, ceil), (xw + 0.06, room_l + 0.06, ceil + 0.05), material="mat_ceiling"))
    g.append(_box("baseboard_rear", (rear_x, room_r, 0), (rear_x + 0.015, room_l, 0.08), material="mat_trim"))
    # ---- fridge cabinet: sides, top, back, floor, toe grille, freezer door
    g.append(_box("fridge_side_l", (xf, hw - FR_WALL, 0.0), (xb, hw, Z_FREEZER_TOP), material="mat_shell"))
    g.append(_box("fridge_side_r", (xf, -hw, 0.0), (xb, -hw + FR_WALL, Z_FREEZER_TOP), material="mat_shell"))
    g.append(_box("fridge_top", (xf, -hw, Z_FREEZER_TOP - 0.04), (xb, hw, Z_FREEZER_TOP), material="mat_shell"))
    g.append(_box("fridge_back", (xin_b, -hw + FR_WALL, 0.0), (xb, hw - FR_WALL, Z_FREEZER_TOP - 0.04), material="mat_liner"))
    g.append(_box("fridge_grille", (xf, -hw + FR_WALL, 0.0), (xf + 0.04, hw - FR_WALL, Z_FLOOR_IN - 0.01), material="mat_kick"))
    g.append(_box("fridge_floor_in", (xf + 0.04, -hw + FR_WALL, Z_FLOOR_IN - 0.03), (xin_b, hw - FR_WALL, Z_FLOOR_IN), material="mat_liner"))
    # liner side panels (inner faces, slightly lighter) with shelf rails
    for s, tag in ((1.0, "l"), (-1.0, "r")):
        yin = s * FR_IN_HW
        g.append(_box(f"fridge_liner_{tag}", (xf + 0.03, min(yin, yin + s * 0.004), Z_FLOOR_IN),
                      (xin_b, max(yin, yin + s * 0.004), Z_FRESH_TOP), material="mat_liner", extra=_NOCOL))
        for zz in np.arange(0.62, Z_FRESH_TOP - 0.05, 0.08):
            g.append(_box(f"fridge_rail_{tag}_{int(zz * 100)}", (xs + 0.04, min(yin, yin - s * 0.006), zz - 0.004),
                          (xin_b - 0.02, max(yin, yin - s * 0.006), zz + 0.004), material="mat_liner_rib", extra=_NOCOL))
    # divider between fresh food and freezer; the freezer door (closed, full width)
    g.append(_box("fridge_divider", (xf, -hw + FR_WALL, Z_FRESH_TOP), (xin_b, hw - FR_WALL, Z_FRESH_TOP + 0.06), material="mat_liner"))
    g.append(_box("freezer_door", (xf - 0.065, -hw + 0.002, Z_FRESH_TOP + 0.065), (xf, hw - 0.002, Z_FREEZER_TOP - 0.005), material="mat_shell"))
    g.append(_box("freezer_door_handle", (xf - 0.105, -hw + 0.08, Z_FRESH_TOP + 0.09), (xf - 0.085, -hw + 0.40, Z_FRESH_TOP + 0.11), material="mat_handle"))
    # face frame around the fresh-food opening (gasket seat)
    g.append(_box("fridge_face_l", (xf, hw - FR_WALL - 0.012, Z_FLOOR_IN), (xf + 0.03, hw - FR_WALL, Z_FRESH_TOP), material="mat_liner_rib"))
    g.append(_box("fridge_face_r", (xf, -hw + FR_WALL, Z_FLOOR_IN), (xf + 0.03, -hw + FR_WALL + 0.012, Z_FRESH_TOP), material="mat_liner_rib"))
    # ---- crisper drawers (two, side by side) below the work shelf
    cz0, cz1 = Z_FLOOR_IN + 0.005, z_s - 0.03
    for s, tag in ((1.0, "l"), (-1.0, "r")):
        y_a, y_b = (0.004, FR_IN_HW - 0.006) if s > 0 else (-FR_IN_HW + 0.006, -0.004)
        g.append(_box(f"crisper_{tag}_front", (xs - 0.02, y_a, cz0), (xs, y_b, cz1), material="mat_bin_front"))
        g.append(_box(f"crisper_{tag}_handle", (xs - 0.03, (y_a + y_b) / 2 - 0.08, cz1 - 0.035), (xs - 0.02, (y_a + y_b) / 2 + 0.08, cz1 - 0.02), material="mat_liner_rib"))
        g.append(_box(f"crisper_{tag}_body", (xs, y_a, cz0), (xin_b - 0.01, y_b, cz1 - 0.02), material="mat_crisper", extra=_NOCOL))
    # ---- work shelf (crisper cover): frosted glass + front trim
    g.append(_box("shelf_work_glass", (xs, -FR_IN_HW, z_s - GLASS_T), (xin_b, FR_IN_HW, z_s), material="mat_shelf"))
    g.append(_box("shelf_work_trim", (xs - 0.004, -FR_IN_HW, z_s - GLASS_T - TRIM_DROP), (xs + TRIM_D, FR_IN_HW, z_s + 0.003), material="mat_trim_strip"))
    # ---- shelf above the work level (the hazard's shelf) + top shelf
    g.append(_box("shelf_upper_glass", (xs, -FR_IN_HW, z_ub), (xin_b, FR_IN_HW, z_ut), material="mat_shelf"))
    g.append(_box("shelf_upper_trim", (xs - 0.004, -FR_IN_HW, z_ub - TRIM_DROP), (xs + TRIM_D, FR_IN_HW, z_ut + TRIM_RISE), material="mat_trim_strip"))
    z_top = z_ut + 0.27
    if z_top < Z_FRESH_TOP - 0.10:
        g.append(_box("shelf_top_glass", (xs, -FR_IN_HW, z_top - GLASS_T), (xin_b, FR_IN_HW, z_top), material="mat_shelf"))
        g.append(_box("shelf_top_trim", (xs - 0.004, -FR_IN_HW, z_top - GLASS_T - TRIM_DROP), (xs + TRIM_D, FR_IN_HW, z_top + TRIM_RISE), material="mat_trim_strip"))
    # fridge light housing (ceiling, back) - emissive, contactless
    g.append(_box("fridge_light", (xin_b - 0.12, -0.10, Z_FRESH_TOP - 0.02), (xin_b - 0.04, 0.10, Z_FRESH_TOP), material="mat_lamp", extra=_NOCOL))
    # ---- decor on the upper shelf and the top shelf (static, cell-specific)
    def place_row(z, prefix, kinds, x_lo, x_hi):
        y = -FR_IN_HW + 0.03
        for j, kind in enumerate(kinds):
            if kind in MESH_ITEMS:
                hx_, hy_, _ = MESH_ITEMS[kind]["half"]
                w = 2 * hy_
                x = float(rng.uniform(x_lo + hx_, x_hi - hx_))
                g.append(_mesh_geom(f"{prefix}{j}", kind, (x, y + hy_, z), yaw=float(rng.choice([0.0, np.pi]))))
            else:
                r_, h_ = (0.036, 0.11) if kind == "jar" else (0.034, 0.075)
                w = 2 * r_
                cols = [(0.70, 0.20, 0.15), (0.90, 0.85, 0.40), (0.35, 0.55, 0.25), (0.92, 0.92, 0.90)]
                x = float(rng.uniform(x_lo + r_, x_hi - r_))
                g.append(_jar_geoms(f"{prefix}{j}", (x, y + r_, z), r_, h_, cols[int(rng.integers(4))],
                                    (0.85, 0.85, 0.85) if kind == "cup" else (0.80, 0.65, 0.25)))
            y += w + float(rng.uniform(0.015, 0.05))
            if y > FR_IN_HW - 0.05:
                break
    upper_kinds = list(rng.permutation(["milk", "juice", "eggs", "cheese", "jar", "cup", "juicebox"]))
    place_row(z_ut, "upper_item", upper_kinds, xs + 0.03, xin_b - 0.05)
    if z_top < Z_FRESH_TOP - 0.10:
        top_kinds = list(rng.permutation(["eggs", "cheese", "jar", "jar", "cup", "juicebox"]))
        place_row(z_top, "top_item", top_kinds, xs + 0.03, xin_b - 0.05)
    # ---- fresh-food door (open) with bins and their contents
    g += _door_geoms(c)
    # ---- kitchen: counter run on the left, tall pantry on the right
    gap = 0.015
    yl0, yl1 = hw + gap, room_l
    zc = 0.92
    g.append(_box("basecab_left", (xf + 0.04, yl0, 0.10), (xw, yl1, zc - 0.03), material="mat_cabinet"))
    g.append(_box("basecab_left_kick", (xf + 0.10, yl0, 0.0), (xw, yl1, 0.10), material="mat_kick"))
    for i, (y_a, y_b) in enumerate(((yl0, yl0 + 0.45), (yl0 + 0.45, yl0 + 0.90), (yl0 + 0.90, yl1))):
        g.append(_box(f"basecab_left_door{i}", (xf + 0.02, y_a + 0.004, 0.105), (xf + 0.04, y_b - 0.004, zc - 0.20), material="mat_cabinet_door"))
        g.append(_box(f"basecab_left_drawer{i}", (xf + 0.02, y_a + 0.004, zc - 0.195), (xf + 0.04, y_b - 0.004, zc - 0.035), material="mat_cabinet_door"))
        g.append(_cyl(f"basecab_left_pull{i}", (xf + 0.005, (y_a + y_b) / 2, zc - 0.115), 0.006, 0.06, material="mat_handle", quat=Q_ALONG_Y))
    g.append(_box("counter_left", (xf - 0.005, yl0, zc - 0.03), (xw, yl1, zc), material="mat_counter"))
    g.append(_box("backsplash_left", (xw - 0.012, yl0, zc), (xw, yl1, zc + 0.50), material="mat_tile"))
    g.append(_box("wallcab_left", (xw - 0.33, yl0, zc + 0.50), (xw, yl1, zc + 1.25), material="mat_cabinet"))
    g.append(_box("wallcab_left_door", (xw - 0.348, yl0 + 0.004, zc + 0.505), (xw - 0.33, yl1 - 0.004, zc + 1.245), material="mat_cabinet_door"))
    # counter decor: a fruit bowl and a kettle-like canister (static)
    by = yl0 + 0.55
    g.append(_cyl("counter_bowl", (xf + 0.32, by, zc + 0.04), 0.12, 0.04, material="mat_ceramic"))
    for i in range(4):
        ang = i * np.pi / 2 + 0.3
        g.append(_cyl(f"counter_fruit{i}", (xf + 0.32 + 0.05 * np.cos(ang), by + 0.05 * np.sin(ang), zc + 0.09), 0.037, 0.0,
                      rgba=[(0.75, 0.10, 0.08), (0.90, 0.60, 0.10), (0.55, 0.70, 0.15), (0.80, 0.15, 0.10)][i] + (1.0,), kind="sphere"))
    g.append(_cyl("counter_canister", (xf + 0.45, yl0 + 1.05, zc + 0.10), 0.07, 0.10, material="mat_steel"))
    yr0, yr1 = room_r, -hw - gap
    g.append(_box("pantry_right", (xf + 0.02, yr0, 0.0), (xw, yr1, 2.20), material="mat_cabinet"))
    g.append(_box("pantry_right_door_lo", (xf, yr0 + 0.004, 0.10), (xf + 0.02, yr1 - 0.004, 1.30), material="mat_cabinet_door"))
    g.append(_box("pantry_right_door_hi", (xf, yr0 + 0.004, 1.31), (xf + 0.02, yr1 - 0.004, 2.18), material="mat_cabinet_door"))
    g.append(_cyl("pantry_right_pull", (xf - 0.015, yr1 - 0.06, 1.15), 0.006, 0.10, material="mat_handle"))
    # cabinet / soffit above the fridge
    g.append(_box("fridge_overcab", (xf + 0.02, -hw - gap, Z_FREEZER_TOP + 0.03), (xw, hw + gap, 2.20), material="mat_cabinet"))
    g.append(_box("fridge_overcab_door", (xf, -hw - gap + 0.004, Z_FREEZER_TOP + 0.035), (xf + 0.02, hw + gap - 0.004, 2.195), material="mat_cabinet_door"))
    body = "".join(g)
    shell = " ".join(_f(v) for v in c.shell)
    cab = " ".join(_f(v) for v in c.cabinet)
    cabd = " ".join(_f(min(1.0, v * 1.04)) for v in c.cabinet)
    ctr = " ".join(_f(v) for v in c.counter)
    wall = " ".join(_f(v) for v in c.wall)
    shell_spec = "0.6" if c.shell[0] < 0.8 else "0.3"
    mesh_assets = "".join(
        f'    <texture name="tex_{k}" type="2d" file="{v["png"]}"/>\n'
        f'    <material name="mat_{k}" texture="tex_{k}" specular="0.15" reflectance="0"/>\n'
        f'    <mesh name="mesh_{k}" file="{v["obj"]}" scale="-1 1 1"/>\n' for k, v in MESH_ITEMS.items())
    return f"""<mujoco model="pc_occ_fridge_c{cell_index}">
  <!-- GENERATED by molmo_spaces.tasks.pc_occ_fridge.write_pc_occ_fridge_scenes (cell {cell_index}).
       Top-freezer fridge, cabinet face x={xf:.3f}, work shelf top z={z_s:.3f}, upper shelf glass
       underside z={z_ub:.4f}, shelves' front edge x={xs:.3f}. The jar, the tray, the re-posed items and
       the hidden wine racks are injected at runtime. Do not edit by hand. -->
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
    <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.80 0.78 0.74" rgb2="0.74 0.72 0.68" mark="edge" markrgb="0.60 0.58 0.55" width="512" height="512"/>
    <material name="mat_floor" texture="tex_floor" texrepeat="6 6" reflectance="0"/>
    <texture name="tex_tile" type="2d" builtin="checker" rgb1="0.92 0.92 0.90" rgb2="0.90 0.90 0.88" mark="edge" markrgb="0.70 0.70 0.68" width="256" height="256"/>
    <material name="mat_tile" texture="tex_tile" texrepeat="8 3" reflectance="0"/>
    <material name="mat_wall" rgba="{wall} 1" specular="0.05" reflectance="0"/>
    <material name="mat_ceiling" rgba="0.95 0.95 0.94 1" specular="0.02" reflectance="0"/>
    <material name="mat_trim" rgba="0.96 0.96 0.95 1" specular="0.1" reflectance="0"/>
    <material name="mat_kick" rgba="0.16 0.16 0.17 1" reflectance="0"/>
    <material name="mat_shell" rgba="{shell} 1" specular="{shell_spec}" reflectance="0"/>
    <material name="mat_liner" rgba="0.95 0.95 0.95 1" specular="0.25" reflectance="0"/>
    <material name="mat_liner_rib" rgba="0.88 0.89 0.90 1" specular="0.25" reflectance="0"/>
    <material name="mat_shelf" rgba="0.80 0.86 0.88 1" specular="0.6" shininess="0.8" reflectance="0"/>
    <material name="mat_trim_strip" rgba="0.62 0.64 0.66 1" specular="0.5" reflectance="0"/>
    <material name="mat_bin" rgba="0.93 0.94 0.95 1" specular="0.3" reflectance="0"/>
    <material name="mat_bin_front" rgba="0.82 0.87 0.90 1" specular="0.5" reflectance="0"/>
    <material name="mat_crisper" rgba="0.86 0.90 0.92 1" specular="0.3" reflectance="0"/>
    <material name="mat_lamp" rgba="1 1 0.96 1" emission="0.8" reflectance="0"/>
    <material name="mat_handle" rgba="0.70 0.71 0.73 1" specular="0.7" reflectance="0"/>
    <material name="mat_steel" rgba="0.74 0.75 0.77 1" specular="0.5" reflectance="0"/>
    <material name="mat_ceramic" rgba="0.94 0.93 0.90 1" specular="0.4" reflectance="0"/>
    <material name="mat_cabinet" rgba="{cab} 1" specular="0.1" reflectance="0"/>
    <material name="mat_cabinet_door" rgba="{cabd} 1" specular="0.12" reflectance="0"/>
    <material name="mat_counter" rgba="{ctr} 1" specular="0.3" reflectance="0"/>
{mesh_assets}  </asset>
  <default>
    <default class="furn">
      <geom contype="8" conaffinity="15" group="0" friction="0.9 0.9 0.001" solref="0.02 1" solimp="0.998 0.998 0.001"/>
    </default>
  </default>
  <worldbody>
    <light name="l_ceiling_a" pos="0.0 0.3 2.45" dir="0.25 -0.1 -1" diffuse="0.60 0.60 0.58" specular="0.1 0.1 0.1" castshadow="false"/>
    <light name="l_ceiling_b" pos="-0.8 -0.5 2.45" dir="0.4 0.2 -1" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05" castshadow="false"/>
    <light name="l_fridge" pos="{_f(xin_b - 0.10)} 0 {_f(Z_FRESH_TOP - 0.05)}" dir="-0.3 0 -1" diffuse="0.30 0.30 0.28" specular="0 0 0" castshadow="false" attenuation="1 0.6 0"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 0" material="mat_floor" contype="8" conaffinity="15"/>
{body}  </worldbody>
</mujoco>
"""


def write_pc_occ_fridge_scenes() -> list[str]:
    """(Re)generate the eight cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_OCC_FRIDGE_SCENES[k])
        p.write_text(build_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------
# Hidden wine rack: geometry in the rack frame (origin = inner edge, neck tips, glass underside)
# --------------------------------------------------------------------------------------------
HAZ_BODY = "pact_intrusion_winerack"           # prefix of both rack bodies
HAZ_BODIES = {1: HAZ_BODY + "_left", -1: HAZ_BODY + "_right"}
PARK_HAZ = {1: (-3.0, -1.6, 0.3), -1: (-3.0, -1.0, 0.3)}
BOTTLE_RGBA = ((0.10, 0.22, 0.08, 0.92), (0.30, 0.14, 0.05, 0.92))   # green / brown glass


def rack_parts(side: int) -> list[dict[str, Any]]:
    """Collision + visual primitives of one rack with two bottles, local to the rack origin:
    x = 0 at the neck tips (toward the door), +x toward the back wall; y = 0 at the inner bottle
    neck's lane-side surface, growing outward (side * y); z = 0 at the glass underside. Only the
    necks / shoulders reach forward to where the wrist knuckle passes; the wire cradles and
    hangers hold the bottle bodies further back.
    kind: capsule (a, b, r) / cylinder (c, axis, r, h) / sphere (c, r) / bottle mesh (c)."""
    s = float(side)
    axis_z = -RACK_TOP_GAP - WB_R
    cradle_z = axis_z - WB_R - RACK_WIRE
    y_in = s * (WB_NECK_R - WB_R - 0.004)                      # lane side of the inner bottle body
    y_out = s * (WB_NECK_R + RACK_PITCH + WB_R + 0.004)        # outer side of the outer bottle body
    parts: list[dict[str, Any]] = []
    for i in range(2):
        yb = s * (WB_NECK_R + i * RACK_PITCH)
        parts += [
            dict(name=f"b{i}_neck", kind="capsule", a=[0.008, yb, axis_z], b=[WB_NECK_L, yb, axis_z], r=WB_NECK_R, mat="glass"),
            dict(name=f"b{i}_shoulder", kind="capsule", a=[WB_NECK_L, yb, axis_z], b=[WB_BODY_X0 - 0.005, yb, axis_z], r=0.030, mat="glass"),
            dict(name=f"b{i}_body", kind="cylinder", c=[(WB_BODY_X0 + WB_L) / 2, yb, axis_z], axis="x", r=WB_R,
                 h=(WB_L - WB_BODY_X0) / 2, mat="glass"),
            dict(name=f"b{i}_mesh", kind="bottle", c=[WB_L, yb, axis_z], i=i, vis_only=True),
        ]
    for nm, x in (("front", 0.175), ("back", 0.295)):
        parts += [
            dict(name=f"cradle_{nm}", kind="capsule", a=[x, y_in, cradle_z], b=[x, y_out, cradle_z], r=RACK_WIRE, mat="chrome"),
            dict(name=f"hanger_{nm}_in", kind="capsule", a=[x, y_in, cradle_z], b=[x, y_in, -0.002], r=RACK_WIRE, mat="chrome"),
            dict(name=f"hanger_{nm}_out", kind="capsule", a=[x, y_out, cradle_z], b=[x, y_out, -0.002], r=RACK_WIRE, mat="chrome"),
        ]
    # front neck rest: a tube under both necks (so a wrist swung far toward the rack cannot slip
    # between the two necks), carried back to the front cradle by two side rails
    zb = axis_z - WB_NECK_R - NECK_BAR_R
    y_n0, y_n1 = 0.0, s * (2 * WB_NECK_R + RACK_PITCH)
    parts += [
        dict(name="neck_rest", kind="capsule", a=[NECK_BAR_X, y_n0, zb], b=[NECK_BAR_X, y_n1, zb], r=NECK_BAR_R, mat="chrome"),
        dict(name="rail_in", kind="capsule", a=[NECK_BAR_X, y_n0, zb], b=[0.175, y_in, cradle_z], r=RACK_WIRE, mat="chrome"),
        dict(name="rail_out", kind="capsule", a=[NECK_BAR_X, y_n1, zb], b=[0.175, y_out, cradle_z], r=RACK_WIRE, mat="chrome"),
    ]
    parts += [
        dict(name="clip_in", kind="capsule", a=[0.160, y_in, -0.004], b=[0.310, y_in, -0.004], r=0.004, mat="chrome"),
        dict(name="clip_out", kind="capsule", a=[0.160, y_out, -0.004], b=[0.310, y_out, -0.004], r=0.004, mat="chrome"),
    ]
    return parts


_HAZ_RGBA = dict(chrome=[0.80, 0.81, 0.83, 1.0], glass=[0.10, 0.22, 0.08, 1.0])


def _axis_quat(axis: str) -> list[float]:
    return {"z": [1.0, 0.0, 0.0, 0.0], "x": list(Q_ALONG_X), "y": list(Q_ALONG_Y)}[axis]


def _quat_z_to(v: np.ndarray) -> list[float]:
    z = np.array([0.0, 0.0, 1.0])
    v = np.asarray(v, float) / np.linalg.norm(v)
    ax = np.cross(z, v)
    sn = float(np.linalg.norm(ax))
    if sn < 1e-9:
        return [1.0, 0.0, 0.0, 0.0] if v[2] > 0 else [0.0, 1.0, 0.0, 0.0]
    ang = float(np.arctan2(sn, float(z @ v)))
    return R.from_rotvec(ax / sn * ang).as_quat(scalar_first=True).tolist()


def _ensure_wine_assets(spec: MjSpec) -> None:
    names = {m.name for m in spec.meshes}
    if "wine_glass_mesh" in names:
        return
    me = spec.add_mesh(name="wine_glass_mesh", file=str(WINE_GLASS_OBJ))
    me2 = spec.add_mesh(name="wine_label_mesh", file=str(WINE_LABEL_OBJ))
    del me, me2
    spec.add_texture(name="wine_label_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(WINE_LABEL_PNG))
    mat = spec.add_material(name="wine_label_mat")
    tx = list(mat.textures)
    tx[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "wine_label_tex"
    mat.textures = tx
    for i, rgba in enumerate(BOTTLE_RGBA):
        gm = spec.add_material(name=f"wine_glass_mat{i}")
        gm.rgba = list(rgba)
        gm.specular = 0.8
        gm.shininess = 0.9


# mesh +y (bottle base -> neck) -> world -x; mesh x -> world y; mesh z -> world z
Q_BOTTLE_LYING = R.from_euler("z", 90.0, degrees=True).as_quat(scalar_first=True).tolist()


def add_rack_body(spec: MjSpec, side: int):
    _ensure_wine_assets(spec)
    name = HAZ_BODIES[side]
    b = spec.worldbody.add_body(name=name, mocap=True, pos=list(PARK_HAZ[side]))
    for p in rack_parts(side):
        if p["kind"] == "bottle":
            # visual mesh: the mesh base (y=0) sits at local x = WB_L, its neck tip at x = 0
            b.add_geom(name=f"{name}_{p['name']}_glass", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="wine_glass_mesh",
                       material=f"wine_glass_mat{p['i']}", pos=list(p["c"]), quat=Q_BOTTLE_LYING, contype=0, conaffinity=0,
                       group=1)
            b.add_geom(name=f"{name}_{p['name']}_label", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="wine_label_mesh",
                       material="wine_label_mat", pos=list(p["c"]), quat=Q_BOTTLE_LYING, contype=0, conaffinity=0,
                       group=1)
            continue
        rgba = _HAZ_RGBA[p["mat"]]
        # glass collision primitives are invisible (the mesh draws the bottle); chrome wires are visible
        kw = dict(rgba=rgba if p["mat"] == "chrome" else [0.0, 0.0, 0.0, 0.0],
                  contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
        if p["mat"] != "chrome":
            kw["group"] = 3
        if p["kind"] == "capsule":
            a, bb = np.asarray(p["a"], float), np.asarray(p["b"], float)
            mid, v = (a + bb) / 2, bb - a
            L = float(np.linalg.norm(v))
            b.add_geom(name=f"{name}_{p['name']}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       size=[p["r"], L / 2, 0.0], pos=mid.tolist(), quat=_quat_z_to(v / L), **kw)
        elif p["kind"] == "cylinder":
            b.add_geom(name=f"{name}_{p['name']}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[p["r"], p["h"], 0.0], pos=list(p["c"]), quat=_axis_quat(p["axis"]), **kw)
        else:
            b.add_geom(name=f"{name}_{p['name']}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[p["r"], 0.0, 0.0], pos=list(p["c"]), **kw)
    return b


# --------------------------------------------------------------------------------------------
# Start pose
# --------------------------------------------------------------------------------------------
# Same pose relative to link0 as pc_sink_blind (READY_TCP (0.30, 0, 0.62) over a 0.15 m base), so
# the sink's IK solution is reused unchanged.
READY_TCP = (0.30, 0.0, H_BASE + 0.47)
PC_OCC_FRIDGE_READY_QPOS: list[float] = list(PC_SINK_BLIND_READY_QPOS)


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
# Re-posed fridge items (mocap, drawn per episode, never depending on the rack side):
# (name, kind, half extents (x, y, z), rgba, cap rgba). kind "mesh:<k>" uses MESH_ITEMS[k].
ITEMS = (
    ("milk", "mesh:milk", MESH_ITEMS["milk"]["half"], None, None),
    ("juice", "mesh:juice", MESH_ITEMS["juice"]["half"], None, None),
    ("eggs", "mesh:eggs", MESH_ITEMS["eggs"]["half"], None, None),
    ("cheese", "mesh:cheese", MESH_ITEMS["cheese"]["half"], None, None),
    ("juicebox", "mesh:juicebox", MESH_ITEMS["juicebox"]["half"], None, None),
    ("soda", "bottle", (0.050, 0.050, 0.150), (0.55, 0.12, 0.10), (0.85, 0.10, 0.10)),     # 2 L bottle
    ("water", "bottle", (0.040, 0.040, 0.120), (0.70, 0.85, 0.95), (0.20, 0.45, 0.85)),    # 1.5 L water
    ("yogurt_tub", "jar", (0.055, 0.055, 0.045), (0.95, 0.95, 0.93), (0.25, 0.55, 0.85)),
    ("butter", "box", (0.060, 0.035, 0.025), (0.95, 0.88, 0.45), None),
    ("leftovers", "box", (0.085, 0.065, 0.040), (0.85, 0.88, 0.90), (0.30, 0.55, 0.40)),
)


class PCOccFridgeSampler(PCShelfPnPSampler):
    """Jar at the back of the lowest fridge shelf, tray at the front, hidden wine rack left or right."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_OCC_FRIDGE_ENV_VERSION
    SUITE = "occ_fridge"
    BASE_JIT_XY = (0.012, 0.015)
    BASE_JIT_YAW = 0.02
    CAN_D = (0.330, 0.345)        # jar centre behind the cabinet face
    CAN_Y = (-0.03, 0.03)
    MAT_D = (0.225, 0.232)        # tray centre behind the cabinet face (just behind the shelf trim)
    MAT_DY = 0.015
    PLACE_TOL = 0.05
    # rack inner edge beside the lane centre, per side (+1 = left / +y, -1 = right / -y). Tuned on
    # the kinematic plan scan (scripts/audit_pc_occ_fridge.py); the FR3 forearm leans to +y.
    RACK_GAP = {1: (0.030, 0.036), -1: (0.022, 0.028)}
    RACK_DX = 0.005
    ITEM_LANE = 0.16              # items keep their footprint this far off the lane centre (low items 3 cm less)
    MAX_DRAWS = 40

    @property
    def cell(self) -> FridgeCell:  # type: ignore[override]
        return PC_OCC_FRIDGE_CELLS[self.current_house_index % N_CELLS]

    def _target_dims(self, cell_index: int) -> dict[str, Any]:
        r, h, rgba, lid, lab, label = JARS[PC_OCC_FRIDGE_CELLS[cell_index].jar]
        return dict(radius=r, height=h, rgba=rgba, cap=lid, label_rgba=lab, label=label)

    def _dyn_pool(self, cell_index: int) -> list[dict[str, Any]]:
        return [dict(name=f"pact_clutter_fr{i}", item=nm, kind=kind, half=tuple(half), rgba=rgba, cap=cap)
                for i, (nm, kind, half, rgba, cap) in enumerate(ITEMS)]

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

        PickTaskSampler.add_auxiliary_objects(self, spec)
        k = self.current_house_index % N_CELLS
        self._house_cell = k
        self._dyn = self._dyn_pool(k)
        px, py, pz = self.PARK_XYZ
        have = {m.name for m in spec.meshes}
        for i, it in enumerate(self._dyn):
            b = spec.worldbody.add_body(name=it["name"], mocap=True, pos=[px, py + 0.3 * i, pz + it["half"][2] + 0.001])
            hx, hy, hz = it["half"]
            kw = dict(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
            if it["kind"].startswith("mesh:"):
                mk = it["kind"].split(":", 1)[1]
                if f"dyn_mesh_{mk}" not in have:
                    me = spec.add_mesh(name=f"dyn_mesh_{mk}", file=MESH_ITEMS[mk]["obj"])
                    me.scale = [-1.0, 1.0, 1.0]
                    spec.add_texture(name=f"dyn_tex_{mk}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=MESH_ITEMS[mk]["png"])
                    mat = spec.add_material(name=f"dyn_mat_{mk}")
                    tx = list(mat.textures)
                    tx[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = f"dyn_tex_{mk}"
                    mat.textures = tx
                    mat.specular = 0.15
                    have.add(f"dyn_mesh_{mk}")
                b.add_geom(name=it["name"] + "_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"dyn_mesh_{mk}",
                           material=f"dyn_mat_{mk}", quat=list(Q_UPRIGHT), contype=0, conaffinity=0)
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                           rgba=[0, 0, 0, 0], group=3, **kw)
            elif it["kind"] == "bottle":
                rgba = list(it["rgba"]) + [1.0]
                body_h = hz * 0.78
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx, body_h, 0.0],
                           pos=[0, 0, -hz + body_h], rgba=rgba, **kw)
                b.add_geom(name=it["name"] + "_shoulder", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                           size=[hx * 0.55, hz * 0.10, 0.0], pos=[0, 0, -hz + 2 * body_h + hz * 0.06], rgba=rgba, **kw)
                b.add_geom(name=it["name"] + "_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, 0.012, 0.0],
                           pos=[0, 0, hz - 0.012], rgba=list(it["cap"]) + [1.0], **kw)
                b.add_geom(name=it["name"] + "_label", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[hx * 1.006, hz * 0.22, 0.0], pos=[0, 0, -hz * 0.25],
                           rgba=[0.95, 0.94, 0.90, 1.0], contype=0, conaffinity=0)
            elif it["kind"] == "jar":
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx, hz, 0.0],
                           rgba=list(it["rgba"]) + [1.0], **kw)
                b.add_geom(name=it["name"] + "_lid", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx * 1.02, 0.006, 0.0],
                           pos=[0, 0, hz - 0.004], rgba=list(it["cap"]) + [1.0], contype=0, conaffinity=0)
            else:
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                           rgba=list(it["rgba"]) + [1.0], **kw)
                if it["cap"] is not None:
                    b.add_geom(name=it["name"] + "_lid", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx * 1.01, hy * 1.01, 0.005],
                               pos=[0, 0, hz - 0.004], rgba=list(it["cap"]) + [1.0], contype=0, conaffinity=0)
        self._add_task_bodies(spec, k)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        td = self._target_dims(cell_index)
        self._target_spec = td
        r, h = td["radius"], td["height"]
        ns = "cavity_obj_0/"
        b = spec.worldbody.add_body(name=ns + "fridge_jar", pos=[-3.0, -2.6, h / 2 + 0.01])
        b.add_joint(name=ns + "fridge_jar_free", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.01)
        b.add_geom(name=ns + "fridge_jar_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r, h / 2, 0.0], rgba=list(td["rgba"]) + [1.0], mass=0.25,
                   contype=8, conaffinity=15, condim=4, friction=[1.0, 0.02, 0.002], priority=1)
        b.add_geom(name=ns + "fridge_jar_lid", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r * 1.03, 0.007, 0.0], pos=[0.0, 0.0, h / 2 + 0.004],
                   rgba=list(td["cap"]) + [1.0], mass=0.001, contype=0, conaffinity=0)
        b.add_geom(name=ns + "fridge_jar_label", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r * 1.004, h * 0.22, 0.0], pos=[0.0, 0.0, -h * 0.06],
                   rgba=list(td["label_rgba"]) + [1.0], mass=0.001, contype=0, conaffinity=0)
        self._injected_obj_name = ns + "fridge_jar"
        self._metadata_adder.update({
            self._injected_obj_name: {
                "category": "jar", "object_enum": "temp_object", "is_static": False,
                "boundingBox": {"x": 2 * r, "y": 2 * r, "z": h},
            }
        })
        mat = spec.worldbody.add_body(name="place_receptacle_mat", mocap=True, pos=[-3.0, -2.2, 0.01])
        mat.add_geom(name="place_receptacle_mat_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[MAT_HALF, MAT_HALF, MAT_HALF_T], rgba=[0.93, 0.93, 0.95, 1.0],
                     contype=8, conaffinity=15, friction=[0.9, 0.9, 0.001])
        mat.add_geom(name="place_receptacle_mat_rim", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[MAT_HALF + 0.004, MAT_HALF + 0.004, 0.002], pos=[0.0, 0.0, -0.001],
                     rgba=[0.55, 0.62, 0.70, 1.0], contype=0, conaffinity=0)
        for sd in (1, -1):
            add_rack_body(spec, sd)

    def _place_robot(self, env) -> None:
        jx, jy = self.BASE_JIT_XY
        self._cur_base_xyz = (float(np.random.uniform(-jx, jx)), float(np.random.uniform(-jy, jy)), 0.0)
        self._cur_base_yaw = float(np.random.uniform(-self.BASE_JIT_YAW, self.BASE_JIT_YAW))
        robot_view = env.current_robot.robot_view
        q = R.from_euler("z", self._cur_base_yaw).as_quat(scalar_first=True)
        from molmo_spaces.utils.pose import pos_quat_to_pose_mat

        robot_view.base.pose = pos_quat_to_pose_mat(np.array(self._cur_base_xyz), np.array(q))
        mujoco.mj_forward(env.current_model, env.current_data)

    def _cell_common(self) -> dict[str, Any]:
        c = self.cell
        hz = cell_heights(c)
        return dict(x_f=c.x_f, x0=c.x_f, z_s=c.z_s, shelf_front=c.x_f + SHELF_SET, z_upper=hz["z_ub"],
                    z_carry=hz["z_hi"], board_depth=DEPTH_IN, h_base=H_BASE, door_deg=c.door_deg)

    def rack_origin(self, c: FridgeCell, y_c: float, side: int, gap: float, dx: float = 0.0) -> list[float]:
        hz = cell_heights(c)
        return [c.x_f + SHELF_SET + RACK_SETBACK + dx, y_c + side * gap, hz["z_ub"]]

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
        z_b = c.z_s
        mat_top = z_b + 2 * MAT_HALF_T
        # hidden wine rack: side drawn independently of everything visible
        side = 1 if np.random.random() < 0.5 else -1          # +1 = left (+y), -1 = right (-y)
        gap = float(np.random.uniform(*self.RACK_GAP[side]))
        rdx = float(np.random.uniform(-self.RACK_DX, self.RACK_DX))
        origin = self.rack_origin(c, yc, side, gap, rdx)
        tilt = TILT_DEG * side       # fingers lean toward the rack side -> wrist swings away
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
        # re-posed items on the work shelf: tall ones in the front zone, low ones anywhere, all off the lane
        out: list = []
        xs = c.x_f + SHELF_SET
        placed: list = [(xc, yc, r + 0.03, r + 0.03), (xm, ym, MAT_HALF + 0.02, MAT_HALF + 0.02)]
        lim = FR_IN_HW - 0.012
        avail = list(range(len(self._dyn)))
        np.random.shuffle(avail)
        n_items = int(np.random.randint(4, 7))
        for i in avail:
            if len(out) >= n_items:
                break
            hx, hy, hzz = self._dyn[i]["half"]
            tall = 2 * hzz > 0.12
            lane = self.ITEM_LANE if tall else self.ITEM_LANE - 0.03
            x_hi = xs + 0.24 if tall else c.x_f + DEPTH_IN - 0.02
            for _ in range(25):
                s_ = 1.0 if np.random.random() < 0.5 else -1.0
                if s_ > 0:
                    y_lo, y_hi = yc + lane + hy, lim - hy
                else:
                    y_lo, y_hi = -lim + hy, yc - lane - hy
                if y_hi <= y_lo or x_hi - hx <= xs + TRIM_D + 0.01 + hx:
                    continue
                y = float(np.random.uniform(y_lo, y_hi))
                x = float(np.random.uniform(xs + TRIM_D + 0.01 + hx, x_hi - hx))
                if any(abs(x - p[0]) < hx + p[2] + 0.012 and abs(y - p[1]) < hy + p[3] + 0.012 for p in placed):
                    continue
                placed.append((x, y, hx, hy))
                out.append((i, [x, y, z_b + hzz + 0.0005]))
                break
        self._apply_layout(env, out)
        for i, pos in out:
            yaw_i = float(np.random.choice([0.0, np.pi]) + np.random.uniform(-0.12, 0.12))
            q = R.from_euler("z", yaw_i).as_quat(scalar_first=True)
            self._mocap(env, self._dyn[i]["name"], pos, q)
        self._mocap(env, "place_receptacle_mat", [xm, ym, z_b + MAT_HALF_T + 0.0003])
        for sd in (1, -1):
            self._mocap(env, HAZ_BODIES[sd], PARK_HAZ[sd])
        self._mocap(env, HAZ_BODIES[side], origin)
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
            hidden_hazard=dict(present=True, side=side_name, y=float(origin[1]), x=float(origin[0]),
                               z_bottom=float(hz["rack_bottom"]), body=HAZ_BODIES[side]),
            # diagnostic free camera (reports/2026-10-13/video_blind.py): inside the fridge, low on the
            # other side, looking up at the rack and the wrist
            reveal_view=dict(pos=[c.x_f + 0.56, yc - side * 0.27, c.z_s + 0.08],
                             lookat=[c.x_f + 0.30, yc + side * 0.06, hz["z_ub"] - 0.08], fovy=80.0,
                             title="Inside the fridge (not a robot camera): wine rack in red"),
            blind=dict(side=int(side), side_name=side_name, tilt_deg=float(tilt), rack=origin,
                       rack_gap=gap, rack_bottom=float(hz["rack_bottom"]), z_carry=z_hi, cell=int(k)),
        )

    def _referral(self) -> str:
        return self._target_spec.get("label", "jar")

    def _task_class(self):
        return PCOccFridgeTask


# --------------------------------------------------------------------------------------------
# Task: pc_sink_blind task (pnp success, hazard audit, gate) with clearance to the wine rack
# --------------------------------------------------------------------------------------------
class PCOccFridgeTask(PCSinkBlindTask):
    def _blind_setup(self, model) -> None:
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        hg = [g for g in range(model.ngeom)
              if (model.body(int(model.geom_bodyid[g])).name or "").startswith(HAZ_BODY)
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
        sp = getattr(self, "scene_params", {}) or {}
        body = (sp.get("hidden_hazard") or {}).get("body", HAZ_BODY)
        self._bclr = dict(hg=hg, sets=robot_vertex_sets(model, robot_root, keys=CLEAR_BODIES), best={},
                          body=body)


class PCOccFridgeExpertConfig(PCSinkBlindExpertConfig):
    """The pc_sink_blind expert (per-waypoint tilt, scene_params dwell, no empty-gripper abort)."""

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCSinkBlindExpert


__all__ = [
    "PC_OCC_FRIDGE_ENV_VERSION", "PC_OCC_FRIDGE_CELLS", "PC_OCC_FRIDGE_SCENES", "PC_OCC_FRIDGE_READY_QPOS",
    "PCOccFridgeSampler", "PCOccFridgeTask", "PCOccFridgeExpertConfig",
    "write_pc_occ_fridge_scenes", "build_scene_xml", "cell_heights", "rack_parts", "primitive_distance",
    "H_BASE", "HAZ_BODY", "HAZ_BODIES", "PARK_HAZ", "READY_TCP", "N_CELLS",
]


if __name__ == "__main__":
    import sys

    if "--write-scenes" in sys.argv:
        for p in write_pc_occ_fridge_scenes():
            print(p)
