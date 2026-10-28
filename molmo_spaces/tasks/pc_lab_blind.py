"""``pc_lab_blind``: pick a reagent bottle / beaker from the back of a laboratory bench, under the
reagent shelf, while a service fixture mounted UNDER that shelf (hidden from both cameras)
threatens the SKIN-COVERED wrist knuckle / forearm, never the gripper.

Why this suite exists (2026-10-14, ICLR rebuttal, agent BLINDLAB)
------------------------------------------------------------------
Third camera-hidden hazard scene, after ``pc_shelf_blind`` (pantry bookcase, hidden wire basket)
and ``pc_shelf_push_blind`` (same bookcase, push). The paper's own experiment is in a laboratory;
this suite moves the hidden-hazard design onto a lab bench: the only thing that differs between
episode configurations is a real fixture that neither camera can see, placed where only the
skinned links (fr3_link5 / fr3_link6, also fr3_link4) pass.

Scene (``custom_scenes/pc_lab_blind_c<k>.xml``, :func:`write_pc_lab_blind_scenes`)
--------------------------------------------------------------------------------------
A closed lab room. In front of the robot: a laboratory bench against the far wall (black epoxy /
dark phenolic top 3 cm thick on steel or wood base cabinets with drawers, toe kick), front edge
15 cm in front of the reagent shelf, i.e. 0.23-0.25 m from the robot base. Above the back of the
bench stands a two-tier reagent shelf (a hutch: two end panels 1.2 m apart, a back panel against
the tiled wall, two 2.5 cm shelves, a bottle rail on the upper one) loaded with reagent bottles,
jars and boxes.
The lower shelf hangs 0.50-0.52 m above the bench top (sized per cell like pc_shelf_blind's tall
bay, so the skinned wrist top passes ~2.5 cm under it while the hand carries the bottle) and is
0.40 m deep. Static lab items at both ends of the bench (hot-plate stirrer, tube rack, pipette
stand, glove box, bottles), a tall storage cabinet on the robot's left, a side bench with
equipment on its right, a low cabinet behind it. Every geom is a primitive resting on a surface or
mounted on the wall / furniture. Lights cast no shadows; no material is reflective.

Eight layout cells (``house_index % 8``): the same shelf-front distance (0.38-0.40 m), bench-top
height (0.66-0.70 m) and shelf clearance recipe as ``pc_shelf_blind`` (reach / IK checked), with
eight finishes (bench top, cabinets, shelf), eight target containers and two fixture types.

Task (prehensile, pc_shelf_pnp success rule)
--------------------------------------------
Pick the container (``cavity_obj_0/lab_bottle``: amber / brown / white reagent jars, a plastic
media bottle, beakers of coloured solution; a primitive cylinder 6.2-6.7 cm wide, 11.2-13.5 cm
tall) standing at the BACK of the bench under the reagent shelf (26.5-28.5 cm behind the shelf's
front edge) and set it upright on the blue silicone spill mat (``place_receptacle_mat``) near the
bench front (mat centre 0.5-3.5 cm in front of the shelf's front edge).

Hidden hazard (``pact_intrusion_fixture``, scored by the evaluator as ``hazard_bar``)
------------------------------------------------------------------------------------
A service fixture screwed to the underside of the lower shelf: a gas / vacuum valve manifold
(aluminium block, three coloured valve handles on its front face, supply lines to the wall) in the
even cells, an under-shelf power outlet box (two duplex outlets, conduit to the wall) in the odd
cells. 15-18 cm wide, 10-12 cm deep, 5.3-5.8 cm tall, its front 20-21 cm behind the shelf's front
edge. The exo camera (robot mounted, 1.65 m high, looking down at ~17 deg) cannot see under a
shelf this high: a point d behind the shelf's front edge is hidden when it hangs less than
~0.3 x d below the underside, so the fixture's 5.8 cm fit in the 6 cm of shadow at 20 cm. The wrist
camera looks down along the hand and never sees above it. The fixture hangs on the LEFT or the
RIGHT of the bottle's lane (coin flip, drawn independently of the bottle, the mat, the clutter, the
cell and the start-pose noise). The two positions overlap the lane the way pc_shelf_push_blind's
baskets do (left: inner edge 7.8-8.6 cm left of the lane centre; right: inner edge 1.6-2.4 cm LEFT
of the lane centre, i.e. reaching over the lane 30 cm above the hand), so the 6-7 cm gap between
them is narrower than the skinned links' cross-section at any hand tilt: no fixed plan threads
between them.

Expert (scripted TCP segments, hand pointing down, fingers closing along x)
--------------------------------------------------------------------------
Start pose: hand down in front of the bench (pc_shelf_blind's ``PC_SHELF_BLIND_READY_QPOS``).
pre-insert pose in front of the shelf above the bottle's height -> slide 12 cm under the shelf,
untilted -> dwell (``DWELL_LAB_S``; the wrist skin faces the fixture from 6-11 cm here; the pause
spans the policy's step-50 chunk query) -> TILT the hand 20 deg about the finger axis (world x),
fingers leaning toward the fixture's side, which swings the wrist knuckle and forearm ~9 cm to the
OTHER side (the only plan difference) -> slide over the bottle -> lower -> close -> lift -> carry it
out from under the shelf, still tilted, to above the mat -> lower -> open -> lift -> untilt -> straight
back out to 10 cm in front of the shelf -> start pose. Tilting about the finger axis leaves the held bottle vertical.

Collection-only additions (switched on by the collection configs, never by the eval config):
DART-style executed-action noise and the contact-tolerant save gate of ``pc_shelf_hard``, and one
skin reading per policy step (as pc_shelf_push_blind; see pc_lab_blind_configs.py). Per-link
clearance to the fixture is recorded in ``obs_scene['pc_clearance']``.
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
    MAT_HALF,
    MAT_HALF_T,
    N_CELLS,
    SCENE_DIR,
    _T,
)
from molmo_spaces.tasks.pc_shelf_blind import (
    ARM_TOP_TILT,
    CEIL_CLEAR,
    D_DIV,
    GRASP_BELOW_TOP,
    INS_CLEAR,
    READY_TCP,
    TILT_DEG,
    X_PRE,
    PCShelfBlindExpert,
    PCShelfBlindExpertConfig,
    PCShelfBlindSampler,
    PCShelfBlindTask,
    _BlindSeg,
    blind_tool_rotation,
    box_distance,
    robot_vertex_sets,
)
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

log = logging.getLogger(__name__)

PC_LAB_BLIND_ENV_VERSION = "pc_lab_blind_v1"
HAZ_BODY = "pact_intrusion_fixture"
HAZ_GEOM = "pact_intrusion_fixture_g"

SHELF_D = 0.40            # reagent shelf depth, front edge -> wall (pc_shelf_blind case depth)
SHELF_T = 0.025           # shelf board thickness
BACK_T = 0.012            # hutch back panel
SHELF_IN_D = SHELF_D - BACK_T
HUTCH_W = 1.20            # clear width between the hutch end panels
END_T = 0.02
TIER2_H = 0.32            # clear height of the upper tier
BENCH_FRONT = 0.15        # bench front edge this far in front of the shelf's front edge
TOP_T = 0.03              # bench top slab
BENCH_HALF_W = 0.85
ROOM_REAR_X = -1.45
ROOM_CEIL_Z = 2.50
WALL_L_Y = 0.97
WALL_R_Y = -1.02
DWELL_LAB_S = 1.4         # pause at the divergence point (reached ~step 37, left ~step 60: spans the step-50 query)
INSERT0_SPEED = 0.11      # first, untilted slide under the shelf (pc_shelf_blind: 0.07 m/s)


@dataclass(frozen=True)
class LabCell:
    """One layout family. ``x0`` = the reagent shelf's front edge, ``b`` = bench top height."""

    x0: float
    b: float
    bay: float                # clear height bench top -> lower shelf underside
    top_rgb: tuple[float, float, float]
    cab_rgb: tuple[float, float, float]
    shelf_rgb: tuple[float, float, float]
    wall_rgb: tuple[float, float, float]
    fixture: str              # "manifold" | "outlet"

    @property
    def ceil(self) -> float:
        return self.b + self.bay

    @property
    def wall_x(self) -> float:
        return self.x0 + SHELF_D

    @property
    def bench_x(self) -> float:
        return self.x0 - BENCH_FRONT


# Target containers (one per cell): radius, height, body rgba, cap rgb (None = open beaker),
# label rgb (None = no label), liquid rgb (beakers), referral.
CONTAINERS = (
    (0.0325, 0.122, (0.52, 0.27, 0.07, 1.0), (0.08, 0.08, 0.09), (0.95, 0.95, 0.92), None, "amber reagent jar"),
    (0.0330, 0.118, (0.82, 0.88, 0.92, 0.45), None, None, (0.15, 0.35, 0.85), "beaker of blue solution"),
    (0.0310, 0.130, (0.93, 0.93, 0.90, 1.0), (0.80, 0.12, 0.10), (0.30, 0.55, 0.85), None, "white reagent bottle"),
    (0.0335, 0.112, (0.36, 0.19, 0.06, 1.0), (0.92, 0.92, 0.90), (0.98, 0.90, 0.40), None, "brown glass jar"),
    (0.0320, 0.125, (0.82, 0.88, 0.92, 0.45), None, None, (0.20, 0.70, 0.30), "beaker of green solution"),
    (0.0315, 0.135, (0.86, 0.86, 0.80, 1.0), (0.12, 0.30, 0.75), (0.95, 0.95, 0.92), None, "plastic media bottle"),
    (0.0330, 0.120, (0.52, 0.27, 0.07, 1.0), (0.95, 0.80, 0.10), (0.95, 0.95, 0.92), None, "amber bottle with a yellow cap"),
    (0.0325, 0.115, (0.82, 0.88, 0.92, 0.45), None, None, (0.95, 0.50, 0.10), "beaker of orange solution"),
)
OBJ_HS = tuple(c[1] for c in CONTAINERS)


def _bay(k: int) -> float:
    """Bench top -> shelf underside, pc_shelf_blind's recipe (skinned wrist top 2.5 cm under it)."""
    return round(OBJ_HS[k] + INS_CLEAR + ARM_TOP_TILT + CEIL_CLEAR, 3)


EPOXY = (0.07, 0.07, 0.08)
PHENOLIC = (0.17, 0.18, 0.19)
PC_LAB_BLIND_CELLS: tuple[LabCell, ...] = (
    LabCell(0.39, 0.68, _bay(0), EPOXY, (0.78, 0.79, 0.80), (0.92, 0.92, 0.90), (0.88, 0.88, 0.85), "manifold"),
    LabCell(0.40, 0.66, _bay(1), PHENOLIC, (0.92, 0.92, 0.90), (0.78, 0.64, 0.45), (0.86, 0.87, 0.88), "outlet"),
    LabCell(0.38, 0.70, _bay(2), EPOXY, (0.45, 0.52, 0.60), (0.92, 0.92, 0.90), (0.90, 0.89, 0.84), "manifold"),
    LabCell(0.39, 0.66, _bay(3), EPOXY, (0.60, 0.46, 0.30), (0.62, 0.64, 0.66), (0.87, 0.86, 0.83), "outlet"),
    LabCell(0.40, 0.68, _bay(4), PHENOLIC, (0.80, 0.81, 0.82), (0.78, 0.64, 0.45), (0.88, 0.88, 0.85), "manifold"),
    LabCell(0.38, 0.66, _bay(5), EPOXY, (0.92, 0.92, 0.90), (0.62, 0.64, 0.66), (0.86, 0.87, 0.88), "outlet"),
    LabCell(0.39, 0.70, _bay(6), EPOXY, (0.78, 0.79, 0.80), (0.92, 0.92, 0.90), (0.90, 0.89, 0.84), "manifold"),
    LabCell(0.40, 0.68, _bay(7), PHENOLIC, (0.80, 0.76, 0.66), (0.78, 0.64, 0.45), (0.87, 0.86, 0.83), "outlet"),
)

PC_LAB_BLIND_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_lab_blind_c{k}.xml") for k in range(N_CELLS)
)

# Reagent palette: (body rgba, cap rgb)
REAGENTS = (
    ((0.52, 0.27, 0.07, 1.0), (0.08, 0.08, 0.09)),   # amber glass, black cap
    ((0.36, 0.19, 0.06, 1.0), (0.92, 0.92, 0.90)),   # brown glass, white cap
    ((0.93, 0.93, 0.90, 1.0), (0.80, 0.12, 0.10)),   # white HDPE, red cap
    ((0.86, 0.86, 0.80, 1.0), (0.12, 0.30, 0.75)),   # natural HDPE, blue cap
    ((0.93, 0.93, 0.90, 1.0), (0.15, 0.55, 0.25)),   # white HDPE, green cap
    ((0.52, 0.27, 0.07, 1.0), (0.95, 0.80, 0.10)),   # amber glass, yellow cap
    ((0.80, 0.84, 0.86, 1.0), (0.12, 0.30, 0.75)),   # clear glass (frosted look), blue cap
)
BOX_RGB = ((0.92, 0.92, 0.90), (0.30, 0.45, 0.75), (0.85, 0.75, 0.30), (0.55, 0.40, 0.70), (0.70, 0.55, 0.38))


def _f(v: float) -> str:
    return f"{float(v):.4f}"


def _rgba(c) -> str:
    c = tuple(c)
    if len(c) == 3:
        c = c + (1.0,)
    return " ".join(_f(v) for v in c)


def _box(name: str, cls: str, pos, half, rgba=None) -> str:
    col = "" if rgba is None else f' rgba="{_rgba(rgba)}"'
    return (f'<geom name="{name}" class="{cls}" type="box" pos="{_f(pos[0])} {_f(pos[1])} {_f(pos[2])}" '
            f'size="{_f(half[0])} {_f(half[1])} {_f(half[2])}"{col}/>')


def _cyl(name: str, cls: str, pos, r: float, hh: float, rgba=None, euler=None) -> str:
    col = "" if rgba is None else f' rgba="{_rgba(rgba)}"'
    eu = "" if euler is None else f' euler="{_f(euler[0])} {_f(euler[1])} {_f(euler[2])}"'
    return (f'<geom name="{name}" class="{cls}" type="cylinder" pos="{_f(pos[0])} {_f(pos[1])} {_f(pos[2])}" '
            f'size="{_f(r)} {_f(hh)}"{col}{eu}/>')


def _bottle(name: str, x: float, y: float, z: float, r: float, h: float, body, cap, label=True) -> list[str]:
    """A reagent bottle / jar standing on z: body, screw cap, paper label band."""
    g = [_cyl(f"{name}_body", "decor", (x, y, z + h / 2 + 0.0004), r, h / 2, body)]
    cap_h = min(0.022, 0.18 * h)
    g.append(_cyl(f"{name}_cap", "deco_v", (x, y, z + h - cap_h / 2 + 0.0014), r * 0.86 if r > 0.03 else r * 0.92,
                  cap_h / 2, cap))
    if label:
        g.append(_cyl(f"{name}_label", "deco_v", (x, y, z + 0.42 * h), r * 1.006, 0.17 * h, (0.96, 0.96, 0.93)))
    return g


def _shelf_items(rng: np.random.Generator, tag: str, x0: float, y_lo: float, y_hi: float,
                 z: float, max_h: float, depth: float) -> list[str]:
    """Reagent bottles / jars / boxes standing in rows on a shelf top (static decor)."""
    g: list[str] = []
    y = y_lo + 0.01
    n = 0
    while y < y_hi - 0.05:
        r_ = rng.random()
        if r_ < 0.70:
            r = float(rng.uniform(0.025, 0.045))
            h = float(rng.uniform(0.09, min(0.24, max_h)))
            body, cap = REAGENTS[int(rng.integers(len(REAGENTS)))]
            x = x0 + float(rng.uniform(0.03 + r, min(depth - 0.02 - r, 0.20)))
            g += _bottle(f"lab_shelf_{tag}_b{n}", x, y + r, z, r, h, body, cap, label=rng.random() < 0.8)
            if rng.random() < 0.45 and depth > 0.25:     # a second row behind it
                r2 = float(rng.uniform(0.025, 0.04))
                h2 = float(rng.uniform(0.09, min(0.24, max_h)))
                b2, c2 = REAGENTS[int(rng.integers(len(REAGENTS)))]
                g += _bottle(f"lab_shelf_{tag}_c{n}", x0 + depth - 0.03 - r2, y + r, z, r2, h2, b2, c2)
            y += 2 * r + float(rng.uniform(0.008, 0.03))
        elif r_ < 0.88:
            w = float(rng.uniform(0.08, 0.16))
            if y + w > y_hi - 0.01:
                y += 0.03
                continue
            d = float(rng.uniform(0.10, min(0.22, depth - 0.03)))
            h = float(rng.uniform(0.06, min(0.16, max_h)))
            c = BOX_RGB[int(rng.integers(len(BOX_RGB)))]
            g.append(_box(f"lab_shelf_{tag}_x{n}", "decor", (x0 + 0.02 + d / 2, y + w / 2, z + h / 2 + 0.0004),
                          (d / 2, w / 2, h / 2), c))
            y += w + float(rng.uniform(0.01, 0.03))
        else:
            y += float(rng.uniform(0.03, 0.08))
        n += 1
    return g


def _bench_decor(rng: np.random.Generator, c: LabCell) -> list[str]:
    """Static lab equipment at both ends of the bench, clear of the |y| < 0.37 working zone."""
    g: list[str] = []
    b, x0 = c.b, c.x0
    z = b + 0.0004
    # left end: hot-plate stirrer with a beaker on it (open bench, in front of the shelf)
    hx, hy, hz = 0.10, 0.09, 0.045
    px, py = c.bench_x + 0.03 + hx, 0.52
    g.append(_box("lab_hotplate_base", "decor", (px, py, z + hz), (hx, hy, hz), (0.93, 0.93, 0.92)))
    g.append(_box("lab_hotplate_top", "decor", (px + 0.01, py, z + 2 * hz + 0.004), (0.08, 0.08, 0.004), (0.85, 0.86, 0.86)))
    for i, dy in enumerate((-0.04, 0.04)):
        g.append(_cyl(f"lab_hotplate_knob{i}", "deco_v", (px - hx - 0.006, py + dy, z + 0.03), 0.012, 0.006,
                      (0.15, 0.15, 0.16), euler=(0.0, 1.5708, 0.0)))
    g += _bottle("lab_hotplate_beaker", px + 0.01, py, z + 2 * hz + 0.008, 0.04, 0.10, (0.80, 0.84, 0.86, 1.0),
                 (0.80, 0.84, 0.86), label=False)
    # left end, under the shelf: two bottles + a sharps container
    g += _bottle("lab_left_b0", x0 + 0.24, 0.45, z, 0.04, 0.20, *REAGENTS[int(rng.integers(len(REAGENTS)))])
    g += _bottle("lab_left_b1", x0 + 0.30, 0.53, z, 0.035, 0.16, *REAGENTS[int(rng.integers(len(REAGENTS)))])
    g.append(_box("lab_sharps", "decor", (x0 + 0.12, 0.47, z + 0.09), (0.07, 0.06, 0.09), (0.85, 0.15, 0.10)))
    g.append(_box("lab_sharps_lid", "deco_v", (x0 + 0.12, 0.47, z + 0.183), (0.072, 0.062, 0.006), (0.95, 0.85, 0.20)))
    # right end: tube rack with tubes, pipette stand, glove box
    rx, ry = x0 + 0.06, -0.50
    g.append(_box("lab_rack", "decor", (rx, ry, z + 0.035), (0.05, 0.11, 0.035), (0.20, 0.45, 0.80)))
    for i in range(6):
        g.append(_cyl(f"lab_rack_tube{i}", "decor", (rx + (0.02 if i % 2 else -0.02), ry - 0.08 + 0.032 * i,
                                                     z + 0.075), 0.007, 0.045, (0.90, 0.92, 0.95, 1.0)))
    g.append(_cyl("lab_pipette_stand", "decor", (x0 + 0.26, -0.52, z + 0.20), 0.008, 0.20, (0.75, 0.76, 0.78)))
    g.append(_cyl("lab_pipette_foot", "decor", (x0 + 0.26, -0.52, z + 0.006), 0.07, 0.006, (0.30, 0.30, 0.32)))
    for i, dy in enumerate((-0.03, 0.0, 0.03)):
        g.append(_cyl(f"lab_pipette_{i}", "decor", (x0 + 0.235, -0.52 + dy, z + 0.22), 0.009, 0.10,
                      (0.92, 0.92, 0.90)))
    g.append(_box("lab_gloves", "decor", (c.bench_x + 0.08, -0.60, z + 0.045), (0.06, 0.12, 0.045), (0.45, 0.35, 0.75)))
    g += _bottle("lab_right_b0", x0 + 0.14, -0.40, z, 0.035, 0.17, *REAGENTS[int(rng.integers(len(REAGENTS)))])
    # wash bottle at the left front
    g += _bottle("lab_wash", c.bench_x + 0.06, 0.40, z, 0.035, 0.15, (0.95, 0.95, 0.95, 1.0), (0.85, 0.15, 0.10),
                 label=False)
    return g


def build_lab_scene_xml(cell_index: int) -> str:
    """The static MJCF of one layout cell (room + bench + reagent shelf + static decor)."""
    c = PC_LAB_BLIND_CELLS[cell_index]
    rng = np.random.default_rng(7300 + cell_index)
    x0, b, ceil = c.x0, c.b, c.ceil
    wall_x = c.wall_x
    xf = c.bench_x
    g: list[str] = []
    # ---- room shell
    room_cx = (ROOM_REAR_X + wall_x) / 2
    room_hx = (wall_x - ROOM_REAR_X) / 2 + 0.10
    room_cy = (WALL_L_Y + WALL_R_Y) / 2
    room_hy = (WALL_L_Y - WALL_R_Y) / 2 + 0.10
    g.append(f'<geom name="room_wall_far" class="wall" type="box" pos="{_f(wall_x + 0.05)} {_f(room_cy)} 1.25" size="0.05 {_f(room_hy)} 1.25"/>')
    g.append(f'<geom name="room_wall_rear" class="wall" type="box" pos="{_f(ROOM_REAR_X - 0.05)} {_f(room_cy)} 1.25" size="0.05 {_f(room_hy)} 1.25"/>')
    g.append(f'<geom name="room_wall_left" class="wall" type="box" pos="{_f(room_cx)} {_f(WALL_L_Y + 0.05)} 1.25" size="{_f(room_hx)} 0.05 1.25"/>')
    g.append(f'<geom name="room_wall_right" class="wall" type="box" pos="{_f(room_cx)} {_f(WALL_R_Y - 0.05)} 1.25" size="{_f(room_hx)} 0.05 1.25"/>')
    g.append(f'<geom name="room_ceiling" class="wall" type="box" pos="{_f(room_cx)} {_f(room_cy)} {_f(ROOM_CEIL_Z + 0.04)}" size="{_f(room_hx)} {_f(room_hy)} 0.04" rgba="0.93 0.93 0.92 1"/>')
    # tiled backsplash on the far wall between the bench top and the upper tier
    z_tile_top = ceil + SHELF_T + TIER2_H + SHELF_T
    g.append(f'<geom name="room_backsplash" class="tile" type="box" pos="{_f(wall_x - 0.002)} 0 {_f((b + z_tile_top) / 2)}" size="0.002 {_f(BENCH_HALF_W)} {_f((z_tile_top - b) / 2)}"/>')
    g.append(_box("room_baseboard_rear", "wall", (ROOM_REAR_X + 0.006, room_cy, 0.05), (0.006, room_hy - 0.1, 0.05), (0.30, 0.30, 0.32)))
    # ---- laboratory bench: base cabinets, toe kick, epoxy top
    cab_x0, cab_x1 = xf + 0.03, wall_x
    g.append(_box("lab_bench_cabinet", "furn", ((cab_x0 + cab_x1) / 2, 0.0, (0.10 + b - TOP_T) / 2),
                  ((cab_x1 - cab_x0) / 2, BENCH_HALF_W - 0.01, (b - TOP_T - 0.10) / 2), c.cab_rgb))
    g.append(_box("lab_bench_toekick", "furn", ((cab_x0 + 0.06 + cab_x1) / 2, 0.0, 0.05),
                  ((cab_x1 - cab_x0 - 0.06) / 2, BENCH_HALF_W - 0.02, 0.05), (0.12, 0.12, 0.13)))
    g.append(_box("lab_bench_top", "benchtop", ((xf + wall_x) / 2, 0.0, b - TOP_T / 2),
                  ((wall_x - xf) / 2, BENCH_HALF_W, TOP_T / 2), c.top_rgb))
    # drawer / door fronts with pulls on the cabinet face (decor only, 3 mm proud)
    dz = b - TOP_T - 0.02
    for i, yc in enumerate(np.linspace(-0.64, 0.64, 5)):
        g.append(_box(f"lab_bench_drawer_{i}", "deco_v", (cab_x0 - 0.0015, yc, dz - 0.07), (0.0015, 0.135, 0.065),
                      tuple(min(1.0, v * 1.04) for v in c.cab_rgb)))
        g.append(_box(f"lab_bench_door_{i}", "deco_v", (cab_x0 - 0.0015, yc, (0.12 + dz - 0.15) / 2),
                      (0.0015, 0.135, (dz - 0.15 - 0.12) / 2), tuple(min(1.0, v * 1.04) for v in c.cab_rgb)))
        g.append(_box(f"lab_bench_pull_{i}", "deco_v", (cab_x0 - 0.006, yc, dz - 0.07), (0.004, 0.05, 0.006), (0.70, 0.71, 0.73)))
        g.append(_box(f"lab_bench_knob_{i}", "deco_v", (cab_x0 - 0.006, yc + 0.10, dz - 0.20), (0.004, 0.006, 0.03), (0.70, 0.71, 0.73)))
    # ---- reagent shelf (two-tier hutch) over the back of the bench
    z_s2 = ceil + SHELF_T + TIER2_H                 # upper shelf underside
    z_top = z_s2 + SHELF_T
    xs_c = x0 + SHELF_IN_D / 2
    for side, y in (("l", HUTCH_W / 2 + END_T / 2), ("r", -(HUTCH_W / 2 + END_T / 2))):
        g.append(_box(f"lab_shelf_end_{side}", "shelf", (x0 + SHELF_D / 2, y, (b + z_top) / 2),
                      (SHELF_D / 2, END_T / 2, (z_top - b) / 2), c.shelf_rgb))
    g.append(_box("lab_shelf_back", "shelf", (wall_x - BACK_T / 2, 0.0, (b + z_top) / 2),
                  (BACK_T / 2, HUTCH_W / 2, (z_top - b) / 2), tuple(0.96 * v for v in c.shelf_rgb)))
    for nm, zu in (("lab_shelf_lower", ceil), ("lab_shelf_upper", z_s2)):
        g.append(_box(nm, "shelf", (xs_c, 0.0, zu + SHELF_T / 2), (SHELF_IN_D / 2, HUTCH_W / 2, SHELF_T / 2), c.shelf_rgb))
    # bottle rail on the upper tier (a rod 3 cm in, 6 cm up, between the end panels); the
    # lower shelf has no lip: the start pose's wrist knuckle hovers 2-6 cm above its front edge
    g.append(_cyl("lab_shelf_upper_rail", "shelf", (x0 + 0.03, 0.0, z_top + 0.06), 0.005, HUTCH_W / 2,
                  (0.70, 0.71, 0.73), euler=(1.5708, 0.0, 0.0)))
    g.append(_box("lab_shelf_cap", "shelf", (x0 + SHELF_D / 2, 0.0, z_top + 0.009),
                  (SHELF_D / 2, HUTCH_W / 2 + END_T, 0.009), c.shelf_rgb))
    # reagent bottles on both tiers (the lower tier's lip hides their bottoms)
    g += _shelf_items(rng, "t1", x0, -HUTCH_W / 2, HUTCH_W / 2, ceil + SHELF_T, TIER2_H - 0.04, SHELF_IN_D)
    g += _shelf_items(rng, "t2", x0, -HUTCH_W / 2, HUTCH_W / 2, z_top, 0.26, SHELF_IN_D)
    # ---- bench equipment
    g += _bench_decor(rng, c)
    # ---- left: tall steel storage cabinet (front face y = 0.60 facing the robot)
    lx0, lx1, ly0, ly1, lh = -0.55, xf - 0.06, 0.60, 0.90, 1.95
    g.append(_box("lab_cabinet_left", "furn", ((lx0 + lx1) / 2, (ly0 + ly1) / 2, lh / 2),
                  ((lx1 - lx0) / 2, (ly1 - ly0) / 2, lh / 2), (0.80, 0.81, 0.82)))
    for i, xm in enumerate(((lx0 + lx1) / 2 - (lx1 - lx0) / 4, (lx0 + lx1) / 2 + (lx1 - lx0) / 4)):
        g.append(_box(f"lab_cabinet_left_door{i}", "deco_v", (xm, ly0 - 0.0015, 1.0), ((lx1 - lx0) / 4 - 0.006, 0.0015, 0.88),
                      (0.84, 0.85, 0.86)))
        g.append(_box(f"lab_cabinet_left_pull{i}", "deco_v", (xm + (0.10 if i == 0 else -0.10), ly0 - 0.006, 1.05),
                      (0.006, 0.004, 0.06), (0.35, 0.35, 0.37)))
    g.append(_box("lab_cabinet_left_sign", "deco_v", ((lx0 + lx1) / 2, ly0 - 0.003, 1.55), (0.08, 0.0015, 0.06),
                  (0.95, 0.80, 0.10)))
    g.append(_box("lab_cabinet_left_box", "decor", ((lx0 + lx1) / 2, (ly0 + ly1) / 2, lh + 0.09), (0.18, 0.12, 0.09),
                  (0.66, 0.50, 0.32)))
    # ---- right: side bench (standing height) with a centrifuge and a balance
    rx0, rx1, ry0, ry1, rh = -0.55, xf - 0.04, -0.62, WALL_R_Y, 0.90
    g.append(_box("lab_sidebench_cabinet", "furn", ((rx0 + rx1) / 2, (ry0 + ry1) / 2 - 0.01, (0.10 + rh - TOP_T) / 2),
                  ((rx1 - rx0) / 2, (ry0 - ry1) / 2 - 0.02, (rh - TOP_T - 0.10) / 2), c.cab_rgb))
    g.append(_box("lab_sidebench_top", "benchtop", ((rx0 + rx1) / 2, (ry0 + ry1) / 2, rh - TOP_T / 2),
                  ((rx1 - rx0) / 2, (ry0 - ry1) / 2, TOP_T / 2), c.top_rgb))
    ccy = (ry0 + ry1) / 2 + 0.02
    g.append(_box("lab_centrifuge", "decor", (-0.30, ccy, rh + 0.13), (0.17, 0.15, 0.13), (0.90, 0.90, 0.88)))
    g.append(_cyl("lab_centrifuge_lid", "deco_v", (-0.30, ccy, rh + 0.262), 0.11, 0.003, (0.35, 0.40, 0.48)))
    g.append(_box("lab_balance", "decor", (0.02, ccy, rh + 0.04), (0.10, 0.12, 0.04), (0.88, 0.88, 0.86)))
    g.append(_box("lab_balance_shield", "decor", (0.04, ccy, rh + 0.16), (0.08, 0.10, 0.08), (0.80, 0.86, 0.90)))
    g += _bottle("lab_sidebench_b0", -0.10, ccy + 0.06, rh, 0.035, 0.18, *REAGENTS[0])
    # ---- behind the robot: low cabinet with boxes
    bx0, bx1, bh = -0.80, -0.42, 0.74
    g.append(_box("lab_rear_cabinet", "furn", ((bx0 + bx1) / 2, 0.0, bh / 2), ((bx1 - bx0) / 2, 0.40, bh / 2), c.cab_rgb))
    g.append(_box("lab_rear_cabinet_top", "benchtop", ((bx0 + bx1) / 2, 0.0, bh + 0.0125), ((bx1 - bx0) / 2 + 0.01, 0.41, 0.0125),
                  c.top_rgb))
    for i, (y, col) in enumerate(((-0.25, BOX_RGB[0]), (0.05, BOX_RGB[1]), (0.28, BOX_RGB[4]))):
        g.append(_box(f"lab_rear_box_{i}", "decor", ((bx0 + bx1) / 2 - 0.02, y, bh + 0.025 + 0.07 + 0.0004), (0.10, 0.08, 0.07), col))
    body = "\n    ".join(g)
    return f"""<mujoco model="pc_lab_blind_c{cell_index}">
  <!-- Generated by molmo_spaces.tasks.pc_lab_blind.write_pc_lab_blind_scenes (cell {cell_index}).
       Closed laboratory room: bench (top z={b:.3f}, front edge x={xf:.3f}) against the far wall
       (x={wall_x:.3f}), two-tier reagent shelf over its back (front edge x={x0:.3f}, lower shelf
       underside z={ceil:.3f}, depth {SHELF_D:.2f}). Movable items are injected at runtime. -->
  <compiler angle="radian" autolimits="true" boundmass="0" balanceinertia="true"/>
  <option impratio="10" gravity="0 0 -9.8" integrator="implicitfast" cone="elliptic" jacobian="sparse" noslip_iterations="4">
    <flag multiccd="enable" contact="enable" warmstart="enable"/>
  </option>
  <size njmax="6000" nconmax="6000"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.0005"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/>
  </visual>
  <asset>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.74 0.75 0.74" rgb2="0.68 0.69 0.68" width="300" height="300"/>
    <material name="floormat" texture="floortex" texrepeat="10 10" reflectance="0"/>
    <texture name="tiletex" type="2d" builtin="checker" rgb1="0.95 0.95 0.94" rgb2="0.90 0.91 0.90" width="200" height="200"/>
    <material name="tilemat" texture="tiletex" texrepeat="18 6" reflectance="0"/>
  </asset>
  <default>
    <default class="wall">
      <geom contype="8" conaffinity="15" group="0" friction="0.9 0.9 0.001" rgba="{_rgba(c.wall_rgb)}"/>
    </default>
    <default class="tile">
      <geom contype="8" conaffinity="15" group="0" material="tilemat"/>
    </default>
    <default class="furn">
      <geom contype="8" conaffinity="15" group="0" friction="0.6 0.6 0.001" solref="0.02 1" solimp="0.998 0.998 0.001"/>
    </default>
    <default class="benchtop">
      <geom contype="8" conaffinity="15" group="0" friction="0.7 0.7 0.001" solref="0.02 1" solimp="0.998 0.998 0.001"/>
    </default>
    <default class="shelf">
      <geom contype="8" conaffinity="15" group="0" friction="0.45 0.45 0.001" solref="0.02 1" solimp="0.998 0.998 0.001"/>
    </default>
    <default class="decor">
      <geom contype="8" conaffinity="15" group="0" friction="0.6 0.6 0.001" solref="0.02 1" solimp="0.998 0.998 0.001"/>
    </default>
    <default class="deco_v">
      <geom contype="0" conaffinity="0" group="0"/>
    </default>
  </default>
  <worldbody>
    <light name="l_ceiling_a" pos="0.2 0.4 2.40" dir="0.2 -0.1 -1" diffuse="0.75 0.75 0.72" castshadow="false"/>
    <light name="l_ceiling_b" pos="-0.6 -0.5 2.40" dir="0.4 0.2 -1" diffuse="0.45 0.45 0.45" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 0" material="floormat" contype="8" conaffinity="15"/>
    {body}
  </worldbody>
</mujoco>
"""


def write_pc_lab_blind_scenes() -> list[str]:
    """(Re)generate the eight cell XMLs and their metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_LAB_BLIND_SCENES[k])
        p.write_text(build_lab_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------
# Hidden fixture geometry (per cell, fixed; the side and the exact place are drawn per episode)
# --------------------------------------------------------------------------------------------
def fixture_dims(cell_index: int) -> dict[str, Any]:
    rng = np.random.default_rng(8800 + cell_index)
    c = PC_LAB_BLIND_CELLS[cell_index]
    return dict(kind=c.fixture,
                half=[float(rng.uniform(0.10, 0.12)) / 2, float(rng.uniform(0.15, 0.18)) / 2,
                      float(rng.uniform(0.053, 0.058)) / 2],
                label="gas / vacuum valve manifold" if c.fixture == "manifold" else "under-shelf power outlet box")


def _add_fixture(spec: MjSpec, dims: dict[str, Any]) -> None:
    """Mocap body ``pact_intrusion_fixture`` (parked outside the room until posed): one collision
    box (the fixture's envelope) plus surface details that stay within ~6 mm of it."""
    hx, hy, hz = dims["half"]
    b = spec.worldbody.add_body(name=HAZ_BODY, mocap=True, pos=[-3.0, -1.6, 0.3])
    vis = dict(contype=0, conaffinity=0, group=0)
    if dims["kind"] == "manifold":
        b.add_geom(name=HAZ_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                   rgba=[0.70, 0.72, 0.75, 1.0], contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
        # mounting plate flush with the shelf underside
        b.add_geom(name=HAZ_BODY + "_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[hx + 0.006, hy + 0.006, 0.002], pos=[0.0, 0.0, hz - 0.002],
                   rgba=[0.45, 0.46, 0.48, 1.0], **vis)
        # three valve bodies + lever handles on the front face (toward the robot)
        cols = ([0.95, 0.80, 0.10, 1.0], [0.15, 0.55, 0.25, 1.0], [0.10, 0.10, 0.11, 1.0])
        for i, dy in enumerate(np.linspace(-0.6 * hy, 0.6 * hy, 3)):
            b.add_geom(name=f"{HAZ_BODY}_valve{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.009, 0.003, 0.0], pos=[-hx - 0.003, float(dy), -0.2 * hz],
                       quat=[0.7071068, 0.0, 0.7071068, 0.0], rgba=[0.80, 0.70, 0.35, 1.0], **vis)
            b.add_geom(name=f"{HAZ_BODY}_handle{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.002, 0.004, 0.45 * hz], pos=[-hx - 0.0055, float(dy), -0.2 * hz],
                       rgba=cols[i], **vis)
        # supply lines from the back of the block to the wall
        for i, dy in enumerate((-0.4 * hy, 0.4 * hy)):
            b.add_geom(name=f"{HAZ_BODY}_line{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.006, 0.06, 0.0], pos=[hx + 0.06, float(dy), hz - 0.012],
                       quat=[0.7071068, 0.0, 0.7071068, 0.0], rgba=[0.80, 0.55, 0.25, 1.0], **vis)
        # stamped label on the front
        b.add_geom(name=HAZ_BODY + "_tag", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0008, 0.03, 0.008],
                   pos=[-hx - 0.0008, 0.0, 0.55 * hz], rgba=[0.95, 0.95, 0.92, 1.0], **vis)
    else:
        b.add_geom(name=HAZ_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                   rgba=[0.88, 0.87, 0.82, 1.0], contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
        b.add_geom(name=HAZ_BODY + "_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[hx + 0.005, hy + 0.005, 0.002], pos=[0.0, 0.0, hz - 0.002],
                   rgba=[0.60, 0.61, 0.63, 1.0], **vis)
        for i, dy in enumerate((-0.45 * hy, 0.45 * hy)):
            b.add_geom(name=f"{HAZ_BODY}_faceplate{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.0015, 0.022, 0.4 * hz + 0.004], pos=[-hx - 0.0015, float(dy), 0.0],
                       rgba=[0.97, 0.96, 0.92, 1.0], **vis)
            for j, dz in enumerate((-0.2 * hz, 0.2 * hz)):
                b.add_geom(name=f"{HAZ_BODY}_socket{i}{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[0.0006, 0.010, 0.006], pos=[-hx - 0.0032, float(dy), float(dz)],
                           rgba=[0.20, 0.20, 0.21, 1.0], **vis)
        b.add_geom(name=HAZ_BODY + "_pilot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.004, 0.0015, 0.0],
                   pos=[-hx - 0.0015, 0.0, 0.5 * hz], quat=[0.7071068, 0.0, 0.7071068, 0.0],
                   rgba=[0.85, 0.10, 0.08, 1.0], **vis)
        b.add_geom(name=HAZ_BODY + "_conduit", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.009, 0.06, 0.0], pos=[hx + 0.06, 0.0, hz - 0.012],
                   quat=[0.7071068, 0.0, 0.7071068, 0.0], rgba=[0.62, 0.63, 0.65, 1.0], **vis)


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
class PCLabBlindSampler(PCShelfBlindSampler):
    """Container at the back of the bench under the reagent shelf, mat at the bench front, hidden
    under-shelf fixture left or right of the lane."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_LAB_BLIND_ENV_VERSION
    SUITE = "lab_blind"
    N_DYN = 10
    OBJ_D = (0.265, 0.285)        # container centre behind the shelf's front edge (back of the bench)
    OBJ_Y = (-0.03, 0.03)
    MAT_D = (-0.035, -0.005)      # mat centre relative to the shelf's front edge (in front of it)
    MAT_DY = 0.03
    # fixture inner edge beside the lane centre, per side (+ = toward the fixture's side); the
    # pc_shelf_push_blind offsets (no fixed hand tilt threads between the two positions)
    FIX_GAP = {1: (0.078, 0.086), -1: (-0.024, -0.016)}
    FIX_D0 = (0.200, 0.210)       # fixture front behind the shelf's front edge
    LANE_HALF = 0.15              # clutter keeps this far off the lane centre
    BAY_Y = 0.36                  # clutter zone |y| (static equipment stands beyond)
    CLUTTER_MAX_H = 0.20          # items under the shelf
    MAX_DRAWS = 40

    @property
    def cell(self) -> LabCell:   # type: ignore[override]
        return PC_LAB_BLIND_CELLS[self.current_house_index % N_CELLS]

    def _target_dims(self, cell_index: int) -> dict[str, Any]:
        r, h, body, cap, label_rgb, liquid, label = CONTAINERS[cell_index % len(CONTAINERS)]
        return dict(radius=r, height=h, rgba=body, cap=cap, label_rgb=label_rgb, liquid=liquid, label=label)

    # ---------------------------------------------------------------- per-house mocap clutter
    def _dyn_pool(self, cell_index: int) -> list[dict[str, Any]]:
        rng = np.random.default_rng(9300 + cell_index)
        pool: list[dict[str, Any]] = []
        for i in range(self.N_DYN):
            if i < 4:       # reagent bottles
                r = float(rng.uniform(0.028, 0.042))
                h = float(rng.uniform(0.10, 0.19))
                body, cap = REAGENTS[int(rng.integers(len(REAGENTS)))]
                pool.append(dict(kind="bottle", half=(r, r, h / 2), rgba=body, cap=cap))
            elif i < 6:     # beakers with coloured solution
                r = float(rng.uniform(0.035, 0.048))
                h = float(rng.uniform(0.07, 0.12))
                liq = [(0.15, 0.35, 0.85), (0.20, 0.70, 0.30), (0.95, 0.50, 0.10), (0.75, 0.15, 0.55)][int(rng.integers(4))]
                pool.append(dict(kind="beaker", half=(r, r, h / 2), rgba=(0.82, 0.88, 0.92, 0.45), liquid=liq))
            elif i == 6:    # pipette tip box
                pool.append(dict(kind="box", half=(0.06, 0.045, 0.035), rgba=(0.20, 0.20, 0.22, 1.0),
                                 lid=BOX_RGB[1 + int(rng.integers(3))]))
            elif i == 7:    # Erlenmeyer flask (base + neck)
                pool.append(dict(kind="flask", half=(0.045, 0.045, 0.075), rgba=(0.82, 0.88, 0.92, 0.45),
                                 liquid=(0.85, 0.75, 0.15)))
            elif i == 8:    # wash bottle
                pool.append(dict(kind="wash", half=(0.035, 0.035, 0.08), rgba=(0.95, 0.95, 0.95, 1.0),
                                 cap=(0.85, 0.15, 0.10)))
            else:           # tube rack with tubes
                pool.append(dict(kind="rack", half=(0.045, 0.09, 0.055), rgba=(0.95, 0.80, 0.15, 1.0)))
        for i, it in enumerate(pool):
            it["name"] = f"pact_clutter_lab{i}"
        return pool

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        PickTaskSampler.add_auxiliary_objects(self, spec)
        k = self.current_house_index % N_CELLS
        self._house_cell = k
        self._dyn = self._dyn_pool(k)
        px, py, pz = self.PARK_XYZ
        col = dict(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
        vis = dict(contype=0, conaffinity=0)
        CYL, BOX = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX
        for i, it in enumerate(self._dyn):
            hx, hy, hz = it["half"]
            b = spec.worldbody.add_body(name=it["name"], mocap=True, pos=[px, py + 0.3 * i, pz + hz + 0.001])
            nm = it["name"]
            kind = it["kind"]
            if kind in ("bottle", "wash"):
                b.add_geom(name=nm + "_g", type=CYL, size=[hx, hz, 0.0], rgba=list(it["rgba"]), **col)
                ch = min(0.02, 0.18 * hz * 2)
                b.add_geom(name=nm + "_cap", type=CYL, size=[hx * 0.85, ch / 2, 0.0], pos=[0, 0, hz - ch / 2 + 0.0012],
                           rgba=list(it["cap"]) + [1.0], **vis)
                if kind == "bottle":
                    b.add_geom(name=nm + "_label", type=CYL, size=[hx * 1.006, 0.32 * hz, 0.0],
                               pos=[0, 0, -0.15 * hz], rgba=[0.96, 0.96, 0.93, 1.0], **vis)
                else:
                    b.add_geom(name=nm + "_nozzle", type=CYL, size=[0.003, 0.03, 0.0],
                               pos=[0.012, 0.0, hz + 0.022], quat=[0.9659, 0.0, 0.2588, 0.0],
                               rgba=[0.95, 0.95, 0.95, 1.0], **vis)
            elif kind == "beaker":
                b.add_geom(name=nm + "_g", type=CYL, size=[hx, hz, 0.0], rgba=list(it["rgba"]), **col)
                b.add_geom(name=nm + "_liquid", type=CYL, size=[hx * 0.94, 0.30 * hz, 0.0], pos=[0, 0, -0.62 * hz],
                           rgba=list(it["liquid"]) + [1.0], **vis)
            elif kind == "flask":
                b.add_geom(name=nm + "_g", type=CYL, size=[hx, 0.55 * hz, 0.0], pos=[0, 0, -0.45 * hz],
                           rgba=list(it["rgba"]), **col)
                b.add_geom(name=nm + "_neck", type=CYL, size=[0.016, 0.45 * hz, 0.0], pos=[0, 0, 0.55 * hz],
                           rgba=list(it["rgba"]), **col)
                b.add_geom(name=nm + "_liquid", type=CYL, size=[hx * 0.93, 0.30 * hz, 0.0], pos=[0, 0, -0.68 * hz],
                           rgba=list(it["liquid"]) + [1.0], **vis)
            elif kind == "box":
                b.add_geom(name=nm + "_g", type=BOX, size=[hx, hy, hz * 0.8], pos=[0, 0, -0.2 * hz],
                           rgba=list(it["rgba"]), **col)
                b.add_geom(name=nm + "_lid", type=BOX, size=[hx * 1.02, hy * 1.02, hz * 0.2], pos=[0, 0, 0.8 * hz],
                           rgba=list(it["lid"]) + [1.0], **col)
            else:   # rack
                b.add_geom(name=nm + "_g", type=BOX, size=[hx, hy, 0.45 * hz], pos=[0, 0, -0.55 * hz],
                           rgba=list(it["rgba"]), **col)
                for j in range(5):
                    b.add_geom(name=f"{nm}_tube{j}", type=CYL, size=[0.007, 0.55 * hz, 0.0],
                               pos=[0.012 if j % 2 else -0.012, -hy + 0.02 + j * (2 * hy - 0.04) / 4, 0.45 * hz],
                               rgba=[0.90, 0.92, 0.95, 1.0], **col)
        self._add_task_bodies(spec, k)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        td = self._target_dims(cell_index)
        self._target_spec = td
        r, h = td["radius"], td["height"]
        ns = "cavity_obj_0/"
        name = ns + "lab_bottle"
        b = spec.worldbody.add_body(name=name, pos=[-3.0, -2.6, h / 2 + 0.01])
        b.add_joint(name=name + "_free", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.01)
        b.add_geom(name=name + "_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0.0],
                   rgba=list(td["rgba"]), mass=0.20, contype=8, conaffinity=15, condim=4,
                   friction=[1.0, 0.02, 0.002], priority=1)
        vis = dict(contype=0, conaffinity=0, mass=0.0005)
        if td["cap"] is not None:
            b.add_geom(name=name + "_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[r * 1.012, 0.011, 0.0], pos=[0.0, 0.0, h / 2 - 0.011 + 0.0012],
                       rgba=list(td["cap"]) + [1.0], **vis)
        if td["label_rgb"] is not None:
            b.add_geom(name=name + "_label", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[r * 1.006, 0.19 * h, 0.0], pos=[0.0, 0.0, -0.08 * h],
                       rgba=[0.96, 0.96, 0.93, 1.0], **vis)
            b.add_geom(name=name + "_label_band", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[r * 1.008, 0.03 * h, 0.0], pos=[0.0, 0.0, 0.04 * h],
                       rgba=list(td["label_rgb"]) + [1.0], **vis)
        if td["liquid"] is not None:
            b.add_geom(name=name + "_liquid", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[r * 0.94, 0.30 * h, 0.0], pos=[0.0, 0.0, -0.18 * h],
                       rgba=list(td["liquid"]) + [1.0], **vis)
            b.add_geom(name=name + "_rim", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[r * 1.01, 0.003, 0.0], pos=[0.0, 0.0, h / 2 - 0.003],
                       rgba=[0.88, 0.92, 0.95, 1.0], **vis)
        self._injected_obj_name = name
        self._metadata_adder.update({
            name: {"category": "bottle", "object_enum": "temp_object", "is_static": False,
                   "boundingBox": {"x": 2 * r, "y": 2 * r, "z": h}},
        })
        mat = spec.worldbody.add_body(name="place_receptacle_mat", mocap=True, pos=[-3.0, -2.2, 0.01])
        mat.add_geom(name="place_receptacle_mat_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[MAT_HALF, MAT_HALF, MAT_HALF_T], rgba=[0.16, 0.36, 0.62, 1.0],
                     contype=8, conaffinity=15, friction=[0.9, 0.9, 0.001])
        self._fix = fixture_dims(cell_index)
        _add_fixture(spec, self._fix)

    def _cell_common(self) -> dict[str, Any]:
        c = self.cell
        # pc_shelf key names kept for the shared tools (x0 = shelf front edge, b_lo = bench top,
        # bay_lo = bench top -> lower shelf underside)
        return dict(x0=c.x0, b_lo=c.b, b_hi=c.ceil + SHELF_T, bay_lo=c.bay, bay_hi=TIER2_H,
                    bay_w=HUTCH_W, board_depth=SHELF_IN_D, bench_front_x=c.bench_x, bench_top_z=c.b,
                    shelf_front_x=c.x0, shelf_under_z=c.ceil, wall_x=c.wall_x)

    def _clutter_bay(self, x_lo: float, x_hi: float, h: float) -> dict:
        return dict(z=self.cell.b, h=h, x_lo=x_lo, x_hi=x_hi, y_lo=-self.BAY_Y, y_hi=self.BAY_Y)

    def _draw_episode(self, env) -> dict[str, Any] | None:
        c = self.cell
        k = self.current_house_index % N_CELLS
        td = self._target_spec
        r, h = td["radius"], td["height"]
        z_b = c.b
        ceil = c.ceil
        dc = float(np.random.uniform(*self.OBJ_D))
        yc = float(np.random.uniform(*self.OBJ_Y))
        dm = float(np.random.uniform(*self.MAT_D))
        ym = yc + float(np.random.uniform(-self.MAT_DY, self.MAT_DY))
        xc, xm = c.x0 + dc, c.x0 + dm
        yaw = float(np.random.uniform(0, 2 * np.pi))
        mat_top = z_b + 2 * MAT_HALF_T
        # hidden fixture: side drawn independently of everything visible
        side = 1 if np.random.random() < 0.5 else -1          # +1 = left (+y), -1 = right (-y)
        gap = float(np.random.uniform(*self.FIX_GAP[side]))
        hx, hy, hz = self._fix["half"]
        y_in = yc + side * gap
        d0 = float(np.random.uniform(*self.FIX_D0))
        fix_center = [c.x0 + d0 + hx, y_in + side * hy, ceil - hz - 0.0005]
        tilt = TILT_DEG * side       # fingers lean toward the fixture side -> wrist swings away
        z_hi = z_b + h + INS_CLEAR
        z_g = z_b + h - GRASP_BELOW_TOP
        z_p = mat_top + h - GRASP_BELOW_TOP + 0.004
        x_pre = c.x0 - X_PRE
        x_div = c.x0 + D_DIV
        wps = [
            ("pregrasp", [x_pre, yc, z_hi], "fast", 0.0),
            ("pregrasp", [x_div, yc, z_hi], "insert0", 0.0),
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
            # straight back out until the wrist is clear of the shelf's front edge, then up
            ("retreat", [x_pre - 0.03, ym, z_hi], "mid", 0.0),
            ("go_home", list(READY_TCP), "fast", 0.0),
        ]
        # both tilts must be feasible: the accepted (visible) layouts never depend on the side
        for tt in (tilt, -tilt):
            poses = [_T(p, blind_tool_rotation(tt if abs(t) > 0 else 0.0))
                     for (_n, p, _s, t) in wps if p is not None]
            if not self._ik_chain_ok(env, poses):
                return None
        # clutter: lab items on the bench away from the lane, under the shelf and in front of it
        avail = list(range(len(self._dyn)))
        out: list = []
        lane = ((c.bench_x - 0.05 + c.wall_x) / 2, (yc + ym) / 2, (c.wall_x - c.bench_x + 0.05) / 2,
                self.LANE_HALF)
        obj_fp = (xc, yc, r + 0.03, r + 0.03)
        mat_fp = (xm, ym, MAT_HALF + 0.02, MAT_HALF + 0.02)
        keep = [lane, obj_fp, mat_fp]
        bay_back = self._clutter_bay(c.x0 + 0.02, c.wall_x - 0.02, self.CLUTTER_MAX_H + 0.02)
        bay_front = self._clutter_bay(c.bench_x + 0.015, c.x0 + 0.04, 0.30)
        placed: list = []
        for _ in range(3):
            if np.random.random() < 0.75:
                self._random_item(avail, placed, keep, bay_back, out,
                                  kinds=("bottle", "beaker", "box", "flask", "wash", "rack"))
        for _ in range(2):
            if np.random.random() < 0.6:
                self._random_item(avail, placed, keep, bay_front, out,
                                  kinds=("bottle", "beaker", "box", "flask", "wash", "rack"))
        self._apply_layout(env, out)
        self._mocap(env, "place_receptacle_mat", [xm, ym, z_b + MAT_HALF_T + 0.0003])
        self._mocap(env, HAZ_BODY, fix_center)
        quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
        self._settle_target(env, [xc, yc, z_b + h / 2 + 0.002], quat)
        mujoco.mj_forward(env.current_model, env.current_data)
        side_name = "left" if side > 0 else "right"
        return dict(
            **self._cell_common(),
            pick_board_z=z_b, place_board_z=z_b, pick_upper=False,
            can_radius=r, can_height=h, can_label=td["label"], obj_label=td["label"],
            start_xyz=[xc, yc, z_b + h / 2], goal_xyz=[xm, ym, mat_top + h / 2], start_yaw=yaw,
            mat_xyz=[xm, ym, z_b + MAT_HALF_T], mat_top=mat_top, place_tol=self.PLACE_TOL,
            dwell_s=DWELL_LAB_S,
            expert=[[n, p, s, t] for (n, p, s, t) in wps],
            neighbours=[[self._dyn[i]["name"], self._dyn[i]["kind"], pos,
                         list(self._dyn[i]["half"])] for i, pos in out],
            n_neighbours=len(out),
            # evaluator contract (eval_act_place.py writes hidden_hazard_present / _y / _side)
            hidden_hazard=dict(present=True, side=side_name, y=float(fix_center[1]),
                               x=float(fix_center[0]), z_bottom=float(ceil - 2 * hz - 0.0005),
                               body=HAZ_BODY, kind=self._fix["kind"], label=self._fix["label"]),
            # diagnostic free camera for reports/2026-10-13/video_blind.py (not a robot camera):
            # under the shelf in the back right corner, looking back at the arm coming in; the same
            # pose for either fixture side
            reveal_view=dict(pos=[c.wall_x - 0.03, -0.30, ceil - 0.12],
                             lookat=[c.x0, 0.05, ceil - 0.15], fovy=80.0,
                             title="Under the reagent shelf (not a robot camera): hidden fixture in red"),
            blind=dict(side=int(side), side_name=side_name, tilt_deg=float(tilt),
                       basket_center=fix_center, basket_half=list(self._fix["half"]),
                       basket_front_depth=d0, basket_gap=gap, fixture=self._fix["kind"],
                       ceiling_z=ceil, z_carry=z_hi, cell=int(k)),
        )

    def _referral(self) -> str:
        return self._target_spec.get("label", "bottle")

    def _task_class(self):
        return PCLabBlindTask


# --------------------------------------------------------------------------------------------
# Task: pc_shelf_blind's success / hazard audit, with per-link clearance to the fixture
# --------------------------------------------------------------------------------------------
class PCLabBlindTask(PCShelfBlindTask):
    def _blind_setup(self, model) -> None:
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        self._bclr = dict(hg=model.geom(HAZ_GEOM).id, sets=robot_vertex_sets(model, robot_root), best={})

    def _blind_step(self, data) -> None:
        model = data.model
        if getattr(self, "_bclr", None) is None or self.episode_step_count == 0:
            self._blind_setup(model)
        if self.episode_step_count % self.CLEAR_EVERY_BLIND:
            return
        bc = self._bclr
        for key, dist in box_distance(data, bc["sets"], bc["hg"]).items():
            prev = bc["best"].get(key)
            if dist < 0.4 and (prev is None or dist < prev[0]):
                bc["best"][key] = (dist, HAZ_BODY, int(self.episode_step_count))

    def _maybe_log_obstacle_diag(self, success: bool) -> None:
        already = getattr(self, "_obstacle_diag_logged", True)
        super()._maybe_log_obstacle_diag(success)
        if not already and getattr(self, "_obstacle_diag_logged", False):
            sp = getattr(self, "scene_params", {}) or {}
            log.info("[PCLabBlind] fixture=%s side=%s", (sp.get("blind") or {}).get("fixture"),
                     (sp.get("blind") or {}).get("side_name"))


# --------------------------------------------------------------------------------------------
# Expert
# --------------------------------------------------------------------------------------------
class PCLabBlindExpert(PCShelfBlindExpert):
    """pc_shelf_blind's expert (per-waypoint hand tilt) with the dwell read from
    ``scene_params['dwell_s']`` and a faster first slide under the shelf (``insert0``)."""

    SPEEDS = dict(PCShelfBlindExpert.SPEEDS, insert0=INSERT0_SPEED)

    def _compute_trajectory(self):
        sp = getattr(self.task, "scene_params", {}) or {}
        dwell = float(sp.get("dwell_s", DWELL_LAB_S))
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


class PCLabBlindExpertConfig(PCShelfBlindExpertConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCLabBlindExpert


__all__ = [
    "PC_LAB_BLIND_ENV_VERSION", "PC_LAB_BLIND_CELLS", "PC_LAB_BLIND_SCENES", "PCLabBlindSampler",
    "PCLabBlindTask", "PCLabBlindExpert", "PCLabBlindExpertConfig", "write_pc_lab_blind_scenes",
    "build_lab_scene_xml", "fixture_dims", "HAZ_BODY", "HAZ_GEOM", "DWELL_LAB_S",
]
