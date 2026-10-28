"""pc_fumehood_push: NON-PREHENSILE push inside the paper's fume hood.

Task. A wide-mouth reagent jar (``cavity_obj_0/push_jar``, r=4.5 cm, h=12.5 cm) stands on the
fume-hood work surface just inside the sash. The robot closes its gripper and uses the closed
fingers as a pusher: it reaches in through the sash opening, sets the fingertips against the jar's
lower back (~2.5 cm above the surface), pushes it straight across the work surface onto a taped
goal ring 12-21 cm deeper in the hood, backs out along the same line and leaves the hood.
Success = jar centre within ``GOAL_TOL`` (3.5 cm) of the goal centre, jar upright (tilt < 15 deg)
and resting on the work surface.

Scene. ``custom_scenes/pc_fumehood_push.xml`` includes the frozen paper shell
(``pact_place_corridor_v5.xml`` -> v3: lab room walls + ceiling, bench, hood sides/back/top, frame
posts, mocap sash and jambs, outside pedestal + tray) and adds a pool of 14 real lab items built
from primitives (``pact_clutter_*`` mocap bodies whose origin is the centre of their base, so they
rest exactly on the work surface) plus a flat contactless goal decal. Per episode the sampler
  * lowers the sash rail to an OSHA-style working height (0.43-0.48 m opening),
  * closes the paper's jamb slabs in as horizontal sash panes (each spans its inner edge to the
    hood frame post) a few cm outside the arm's measured envelope at the hood mouth,
  * poses 6-9 lab items: two flank the arm where the wrist/forearm pass the mouth, two flank the
    push stroke, one big item behind the goal, 2-4 extras elsewhere in the hood -- all placed
    against a MEASURED, conservative lane-relative envelope of the expert's arm (``ARM_OCC``) so
    the demonstrations are clean while the skin sees real geometry at a few cm,
  * poses the jar and the goal decal, and pins the robot base (paper pose, x ~ 0.14).
Layout cells: ``house_index % 8`` picks the lane offset (4 values) and which side gets the tall
items (2 families). ``(seed, house_index)`` deterministically rebuilds the same scene.

Why a cylinder. A box pushed by a fingertip is orientation-unstable (any yaw error grows within a
centimetre of travel). A vertical cylinder pushed by the closed finger front is not: every contact
normal on its wall passes through its axis, so the pusher never applies a yaw torque and the jar
translates along the push line as long as the contact stays on the finger face (lateral alignment
within ~1 cm, which the servoed expert holds to < 1 mm). Low jar friction (0.35) and a contact
height of ~2.5 cm (mu*h << r) make tipping physically impossible.

Expert (``PcFumehoodPushPolicy``). One strategy for every episode:

    gripper-close  close the gripper (pusher) while holding the start configuration
    pregrasp       joint-space move to the staging pose (IK from the noise-free home, so every
                   episode enters with the same arm posture), then a straight task-space line
                   down through the sash opening to the pre-push pose 5 cm behind the jar
    grasp          slow creep until the finger face is ~1 cm into contact
    preplace       steady push along the lane (6 cm/s)
    place          final slow centimetres onto the goal, short hold
    retreat        back off along the same line to the pre-push pose
    go_home        rise back to the staging pose outside the hood

Task-space segments are tracked with an integral anti-sag servo (``_ServoLine``). Phase names are
the closed vocabulary of ``BaseObjectManipulationPlannerPolicy.get_all_phases``. Episodes are
~170-190 control steps (policy_dt 66 ms).

Contacts are scored by name: ``cavity_obj_*`` (the jar) is task contact; ``pact_clutter_*`` is
clutter; hood, sash, jambs and bench are other_environment. ``PcFumehoodPushTask`` logs contacted
bodies by name, jar tilt and (check configs) conservative min clearances per hazard group; the
collection sampler (``PcFumehoodPushCollectSampler``) saves an expert episode only if the robot
touched nothing but the jar; for a learned policy its success is the task criterion alone.
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec, mjtJoint
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from molmo_spaces.configs.policy_configs import ObjectManipulationPlannerPolicyConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
)
from molmo_spaces.tasks.cavity_pick_task_sampler import CavityPickTaskSampler
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_7d

log = logging.getLogger(__name__)

PC_FUMEHOOD_PUSH_ENVIRONMENT_VERSION = "pc_fumehood_push_v1"

# --------------------------------------------------------------------------------------------
# Paper fume-hood geometry (pact_place_corridor_v3.xml; read, never edited).
# --------------------------------------------------------------------------------------------
BENCH_Z = 0.72               # bench_top top face (0.70 + 0.02) = hood work surface
SASH_X = 0.58                # sash / jamb / frame-post plane
HOOD_FRONT_X = 0.60          # frame posts end at x = 0.60
HOOD_BACK_X = 1.348          # hood_back inner face
HOOD_INNER_Y = 0.438         # hood_side_l/r inner faces at |y| = 0.438
SASH_HALF = (0.015, 0.44, 0.025)
JAMB_HALF = (0.012, 0.18, 0.20)
PAPER_AP_W = 0.85            # paper keeps the jambs out at +-(0.425 + 0.18)

# --------------------------------------------------------------------------------------------
# Pushed container: wide-mouth HDPE reagent jar, body origin at the centre of its base.
# --------------------------------------------------------------------------------------------
JAR_NAME = "cavity_obj_0/push_jar"
JAR_R = 0.045
JAR_BODY_HALF_H = 0.050
JAR_CAP_R = 0.038
JAR_CAP_HALF_H = 0.0125
JAR_H = 2 * JAR_BODY_HALF_H + 2 * JAR_CAP_HALF_H      # 0.125
JAR_MASS = 0.30
JAR_FRICTION = (0.35, 0.005, 0.0001)

# --------------------------------------------------------------------------------------------
# Lab clutter pool (must agree with pc_fumehood_push.xml). name -> (footprint radius about the
# body origin, height, size class). Footprint radius is the circumscribed circle of the item's
# plan-view outline (boxes: half-diagonal), so a yawed item never exceeds it.
# --------------------------------------------------------------------------------------------
CLUTTER_ITEMS: dict[str, tuple[float, float, str]] = {
    "pact_clutter_solvent_bottle": (0.064, 0.298, "tall"),
    "pact_clutter_media_bottle": (0.051, 0.202, "tall"),
    "pact_clutter_grad_cylinder": (0.035, 0.292, "tall"),
    "pact_clutter_wash_bottle": (0.050, 0.205, "tall"),
    "pact_clutter_amber_bottle": (0.035, 0.154, "mid"),
    "pact_clutter_flask": (0.052, 0.190, "mid"),
    "pact_clutter_beaker": (0.045, 0.120, "mid"),
    "pact_clutter_tube_rack": (0.108, 0.152, "wide"),
    "pact_clutter_glove_box": (0.135, 0.090, "wide"),
    "pact_clutter_hotplate": (0.131, 0.202, "wide"),
    "pact_clutter_sharps_bin": (0.098, 0.222, "wide"),
    "pact_clutter_tip_box": (0.074, 0.065, "wide"),
    "pact_clutter_carboy": (0.100, 0.360, "back"),
    "pact_clutter_ring_stand": (0.120, 0.610, "back"),
}
LEGACY_PARKED_MOCAPS = (
    "pact_clutter_l0", "pact_clutter_l1", "pact_clutter_r0", "pact_clutter_r1",
    "protr_s", "protr_m", "protr_l", "pact_intrusion_left", "pact_intrusion_right",
    "pact_clutter_mount_wall_left", "pact_clutter_mount_wall_right", "pact_clutter_mount_ceiling",
)
PARK_Z = -2.0
GOAL_MARKER = "pc_push_goal_marker"

N_CELLS = 8
# Lane (push line) lateral offset per cell; cells 0-3 put the tall items on the lane's left
# (+y), cells 4-7 on its right. Lane offsets alternate sides so every family has both.
CELL_LANE_Y = (-0.10, -0.035, 0.035, 0.10, 0.10, 0.035, -0.035, -0.10)


def _cell_tall_side(cell: int) -> float:
    return 1.0 if cell < 4 else -1.0


# --------------------------------------------------------------------------------------------
# Measured arm envelope (lane-relative). 8 expert episodes (all cells, no clutter) sampled every
# control step: corners/faces of the local AABB of every robot collision geom, binned in 5 cm
# x-bands. Each entry: x-band start -> list of (z threshold, y_rel lo, y_rel hi): the arm reaches
# below that height somewhere in [lo, hi] (relative to the lane centre) within the band.
# Conservative (AABBs of the meshes are larger than the meshes).
# --------------------------------------------------------------------------------------------
OCC_X0, OCC_X1, OCC_DX = 0.55, 0.85, 0.05
ARM_OCC: dict[float, list[tuple[float, float, float]]] = {
    0.55: [(0.80, -0.07, 0.13), (0.85, -0.10, 0.20), (0.90, -0.10, 0.21), (9.0, -0.10, 0.21)],
    0.60: [(0.80, -0.07, 0.11), (0.85, -0.10, 0.20), (0.90, -0.10, 0.21), (9.0, -0.10, 0.21)],
    0.65: [(0.80, -0.07, 0.04), (0.85, -0.10, 0.05), (0.90, -0.10, 0.11), (0.95, -0.10, 0.18),
           (9.0, -0.10, 0.18)],
    0.70: [(0.80, -0.04, 0.04), (0.85, -0.04, 0.04), (0.90, -0.05, 0.06), (9.0, -0.05, 0.06)],
    0.75: [(9.0, -0.04, 0.04)],
    0.80: [(9.0, -0.02, 0.02)],
}
FRAME_POST_Y = 0.45


def _occ_interval(x: float, z_top: float):
    """Lane-relative y interval the arm occupies below height z_top in the band containing x
    (None, None if the band is free). Bands are inflated by 1 cm for the per-episode jitter."""
    if x < OCC_X0 or x >= OCC_X1:
        return None, None
    xb = round(OCC_X0 + OCC_DX * int((x - OCC_X0) // OCC_DX), 2)
    rows = ARM_OCC.get(xb)
    if not rows:
        return None, None
    # the arm reaches below z_top if any recorded threshold <= z_top...; use the first row whose
    # threshold exceeds the item top (the envelope of everything lower than the item's top)
    for zt, lo, hi in rows:
        if z_top <= zt:
            return lo - 0.01, hi + 0.01
    lo, hi = rows[-1][1], rows[-1][2]
    return lo - 0.01, hi + 0.01


class PcFumehoodPushSampler(CavityPickTaskSampler):
    """Per-episode layout for the fume-hood push (see module docstring)."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_FUMEHOOD_PUSH_ENVIRONMENT_VERSION
    POOL_SIZE = 1

    # robot base: paper corridor puts link0 at x ~ 0.14 (BASE_FWD); keep jitter small.
    BASE_X = 0.14
    BASE_JIT_XY = 0.008
    BASE_JIT_YAW = 0.015

    # jar start / goal (world). Push travel 0.12 .. 0.21 m along +x.
    JAR_X = (0.675, 0.715)
    LANE_JIT_Y = 0.012
    GOAL_X = (0.835, 0.885)
    GOAL_TOL = 0.035

    # sash bottom edge height above the work surface (realistic working heights; the robot's
    # home pose puts the wrist ~0.30 m above the bench just behind the sash plane).
    SASH_OPENING = (0.43, 0.48)

    # clutter placement against the MEASURED arm envelope (see ARM_OCC below)
    CLEAR_MARGIN = 0.015                # surface gap item <-> conservative arm envelope
    FLANK_EXTRA_GAP = (0.0, 0.025)      # flank items sit this much beyond the margin
    N_EXTRA = (2, 4)                    # extra items in the hood beyond the flank slots + back
    PLACE_ATTEMPTS = 60
    ITEM_GAP = 0.015                    # min surface gap between two items
    WALL_GAP = 0.012                    # min gap item <-> hood side/back wall
    JAR_CORRIDOR = 0.075                # lane half-width kept free for the jar (r + 3 cm)
    # horizontal-sash panes (the paper's jamb slabs) close in on the lane: inner-edge gap beyond
    # the arm envelope at the sash plane
    JAMB_GAP = (0.005, 0.045)

    CLEARANCE_DIAG = False              # check configs turn this on (clearance probes)
    REQUIRE_CLEAN = False               # collection: success also requires zero hazard contact

    def __init__(self, config) -> None:
        # Datagen speed hooks shared with the shelf suite (orchestrator, 2026-10-13): drop the
        # unused ObjectImagePointsSensor and refresh skin-camera registry poses lazily. RGB and
        # skin pixels are unchanged; ~6x faster collection. Skipped automatically when the
        # frozen evaluator drives (it installs its own copy) or with PC_SHELF_FAST_HOOKS=0.
        from molmo_spaces.tasks.pc_shelf import _install_fast_hooks

        _install_fast_hooks(config)
        super().__init__(config)

    # ------------------------------------------------------------------ pool / objects
    def _build_grasp_uid_pool(self, n: int) -> list[str]:
        # The push object is not an Objaverse asset; no grasp files are involved.
        return ["push_jar"]

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        PickTaskSampler.add_auxiliary_objects(self, spec)
        body = spec.worldbody.add_body(name=JAR_NAME, pos=[0.70, 0.0, BENCH_Z + 0.0005])
        body.add_joint(name=JAR_NAME + "_jntfree", type=mjtJoint.mjJNT_FREE, damping=0.0)
        common = dict(contype=8, conaffinity=15, condim=3, priority=1,
                      friction=list(JAR_FRICTION), solref=[0.01, 1.0],
                      solimp=[0.95, 0.99, 0.001, 0.5, 2.0])
        body.add_geom(name=JAR_NAME + "_body_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[JAR_R, JAR_BODY_HALF_H, 0], pos=[0, 0, JAR_BODY_HALF_H],
                      rgba=[0.94, 0.94, 0.92, 1.0], mass=JAR_MASS * 0.93, **common)
        body.add_geom(name=JAR_NAME + "_cap_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[JAR_CAP_R, JAR_CAP_HALF_H, 0],
                      pos=[0, 0, 2 * JAR_BODY_HALF_H + JAR_CAP_HALF_H],
                      rgba=[0.12, 0.35, 0.72, 1.0], mass=JAR_MASS * 0.07, **common)
        # printed label band + hazard diamond (visual only) so the jar's yaw is visible
        body.add_geom(name=JAR_NAME + "_label_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[JAR_R + 0.0006, 0.022, 0], pos=[0, 0, 0.055],
                      rgba=[0.97, 0.97, 0.97, 1.0], contype=0, conaffinity=0, group=1,
                      mass=0.0001)
        body.add_geom(name=JAR_NAME + "_diamond_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.0012, 0.012, 0.012], pos=[JAR_R + 0.0008, 0, 0.055],
                      quat=list(R.from_euler("x", 45, degrees=True).as_quat(scalar_first=True)),
                      rgba=[0.90, 0.20, 0.10, 1.0], contype=0, conaffinity=0, group=1,
                      mass=0.0001)
        self._injected_obj_name = JAR_NAME
        self._metadata_adder.update({
            JAR_NAME: {
                "asset_id": None,
                "category": "reagent jar",
                "object_enum": "temp_object",
                "is_static": False,
                "boundingBox": {"x": 2 * JAR_R, "y": 2 * JAR_R, "z": JAR_H},
            }
        })
        log.info(f"[PcPush] injected '{JAR_NAME}' (r={JAR_R}, h={JAR_H:.3f}, m={JAR_MASS})")

    # ------------------------------------------------------------------ theta
    def _cell(self) -> int:
        try:
            return int(self.current_house_index) % N_CELLS
        except (AttributeError, TypeError, ValueError):
            return 0

    def _draw_theta(self) -> dict[str, Any]:
        # One draw from the (seeded) global stream, mixed with the house index: the same
        # (seed, house_index) always rebuilds the same scene, while datagen workers -- which all
        # start from the same config seed -- still produce decorrelated layouts per house.
        try:
            house = int(self.current_house_index)
        except (AttributeError, TypeError, ValueError):
            house = 0
        mix = (int(np.random.randint(0, 2**31 - 1)) ^ ((house * 2654435761) % (2**31))) % (2**31)
        rng = self._rng = np.random.RandomState(mix)
        cell = self._cell()
        lane_y = float(CELL_LANE_Y[cell] + rng.uniform(-self.LANE_JIT_Y, self.LANE_JIT_Y))
        jar_x = float(rng.uniform(*self.JAR_X))
        goal_x = float(rng.uniform(*self.GOAL_X))
        th: dict[str, Any] = {
            "pact_place_environment_version": self.PACT_PLACE_ENVIRONMENT_VERSION,
            "pc_suite": "pc_fumehood_push",
            "layout_seed": int(mix),
            "house_index": int(house),
            "cell": int(cell),
            "n_cells": N_CELLS,
            "lane_y": lane_y,
            "tall_side": _cell_tall_side(cell),
            "jar_start_xy": [jar_x, lane_y],
            "jar_start_yaw": float(rng.uniform(-np.pi, np.pi)),
            "goal_xy": [goal_x, lane_y],
            "goal_tol_m": float(self.GOAL_TOL),
            "push_distance_m": float(goal_x - jar_x),
            "sash_opening_m": float(rng.uniform(*self.SASH_OPENING)),
            "base_xyz": [
                float(self.BASE_X + rng.uniform(-self.BASE_JIT_XY, self.BASE_JIT_XY)),
                float(rng.uniform(-self.BASE_JIT_XY, self.BASE_JIT_XY)),
                0.0,
            ],
            "base_yaw": float(rng.uniform(-self.BASE_JIT_YAW, self.BASE_JIT_YAW)),
            "jar": {"name": JAR_NAME, "radius_m": JAR_R, "height_m": JAR_H, "mass_kg": JAR_MASS,
                    "friction": list(JAR_FRICTION)},
        }
        lo, hi = _occ_interval(0.575, 2.0)
        m = self.CLEAR_MARGIN
        th["jamb_inner_y"] = [
            float(min(lane_y + hi + m + rng.uniform(*self.JAMB_GAP), HOOD_INNER_Y - 0.03)),
            float(max(lane_y + lo - m - rng.uniform(*self.JAMB_GAP), -HOOD_INNER_Y + 0.03)),
        ]
        th["clutter"] = self._draw_clutter(th)
        return th

    # ---------------------------------------------------------------- clutter layout
    def _inside_hood(self, x: float, y: float, r: float) -> bool:
        return (HOOD_FRONT_X + r + 0.005 <= x <= HOOD_BACK_X - r - self.WALL_GAP
                and abs(y) <= HOOD_INNER_Y - r - self.WALL_GAP)

    def _clear_of_placed(self, x, y, r, placed) -> bool:
        for (px, py, pr) in placed:
            if np.hypot(x - px, y - py) < r + pr + self.ITEM_GAP:
                return False
        return True

    def _clear_of_task(self, x, y, r, h, th) -> bool:
        """Item footprint (x, y, r) of height h is clear of the arm's swept envelope (lane-
        relative, measured), the jar corridor, the jar start pose and the goal ring."""
        lane = th["lane_y"]
        m = self.CLEAR_MARGIN
        yr = y - lane
        for xb0 in np.arange(OCC_X0, OCC_X1 - 1e-9, OCC_DX):
            if xb0 + OCC_DX < x - r - m or xb0 > x + r + m:
                continue
            lo, hi = _occ_interval(xb0 + 0.5 * OCC_DX, BENCH_Z + h + m)
            if lo is not None and (yr + r + m > lo) and (yr - r - m < hi):
                return False
        # jar corridor from the jar's start to just beyond the goal
        x0 = th["jar_start_xy"][0] - JAR_R - 0.03
        x1 = th["goal_xy"][0] + JAR_R + 0.06
        if x0 - r <= x <= x1 + r and abs(yr) - r < self.JAR_CORRIDOR:
            return False
        gx, gy = th["goal_xy"]
        if np.hypot(x - gx, y - gy) < r + 0.05 + 0.05:
            return False
        return True

    def _draw_clutter(self, th: dict[str, Any]) -> list[dict[str, Any]]:
        rng = getattr(self, "_rng", None) or np.random
        lane = th["lane_y"]
        side = th["tall_side"]
        placed: list[tuple[float, float, float]] = []
        out: list[dict[str, Any]] = []
        used: set[str] = set()

        def put(name, x, y, slot):
            r, h, _ = CLUTTER_ITEMS[name]
            placed.append((x, y, r))
            used.add(name)
            yaw = float(rng.uniform(-np.pi, np.pi))
            if name == "pact_clutter_wash_bottle":      # spout points away from the lane
                yaw = float(np.pi / 2 if y > lane else -np.pi / 2)
            out.append({"body": name, "slot": slot, "xy": [float(x), float(y)], "yaw": yaw,
                        "radius_m": float(r), "height_m": float(h)})

        def flank_y(x, r, h, sgn):
            """Closest legal lateral offset on side sgn at station x (+ a random extra gap)."""
            m = self.CLEAR_MARGIN
            need = self.JAR_CORRIDOR
            for xb0 in np.arange(OCC_X0, OCC_X1 - 1e-9, OCC_DX):
                if xb0 + OCC_DX < x - r - m or xb0 > x + r + m:
                    continue
                lo, hi = _occ_interval(xb0 + 0.5 * OCC_DX, BENCH_Z + h + m)
                if lo is None:
                    continue
                need = max(need, hi if sgn > 0 else -lo)
            return lane + sgn * (need + m + r + rng.uniform(*self.FLANK_EXTRA_GAP) + 0.002)

        tall_pool = ["pact_clutter_solvent_bottle", "pact_clutter_media_bottle",
                     "pact_clutter_grad_cylinder", "pact_clutter_wash_bottle"]
        mid_pool = ["pact_clutter_amber_bottle", "pact_clutter_flask", "pact_clutter_beaker"]
        wide_pool = ["pact_clutter_tube_rack", "pact_clutter_hotplate", "pact_clutter_sharps_bin",
                     "pact_clutter_glove_box", "pact_clutter_tip_box"]
        # flank slots hug the arm envelope: the front pair sits where the wrist/forearm pass
        # the hood mouth, the mid pair beside the push stroke
        slots = [
            ("front_a", (0.63, 0.74), side, tall_pool),
            ("front_b", (0.63, 0.74), -side, mid_pool + tall_pool),
            ("mid_a", (0.78, 0.92), side, wide_pool + mid_pool),
            ("mid_b", (0.78, 0.92), -side, wide_pool + mid_pool),
        ]
        for slot, xr, sgn, pool in slots:
            cands = [n for n in pool if n not in used]
            rng.shuffle(cands)
            done = False
            for name in cands:
                r, h, _ = CLUTTER_ITEMS[name]
                for _ in range(16):
                    x = float(rng.uniform(*xr))
                    y = float(flank_y(x, r, h, sgn))
                    if (self._inside_hood(x, y, r) and self._clear_of_task(x, y, r, h, th)
                            and self._clear_of_placed(x, y, r, placed)):
                        put(name, x, y, slot)
                        done = True
                        break
                if done:
                    break
        # one back item behind the goal (carboy or ring stand)
        back = ["pact_clutter_carboy", "pact_clutter_ring_stand"]
        rng.shuffle(back)
        for name in back[:1]:
            r, h, _ = CLUTTER_ITEMS[name]
            for _ in range(self.PLACE_ATTEMPTS):
                x = float(rng.uniform(1.05, HOOD_BACK_X - r - self.WALL_GAP))
                y = float(rng.uniform(-HOOD_INNER_Y + r + self.WALL_GAP,
                                      HOOD_INNER_Y - r - self.WALL_GAP))
                if self._clear_of_task(x, y, r, h, th) and self._clear_of_placed(x, y, r, placed):
                    put(name, x, y, "back")
                    break
        # extras anywhere in the hood that is clear of the arm, the jar and the goal
        n_extra = int(rng.randint(self.N_EXTRA[0], self.N_EXTRA[1] + 1))
        rest = [n for n in CLUTTER_ITEMS if n not in used and CLUTTER_ITEMS[n][2] != "back"]
        rng.shuffle(rest)
        for name in rest:
            if n_extra <= 0:
                break
            r, h, _ = CLUTTER_ITEMS[name]
            for _ in range(self.PLACE_ATTEMPTS):
                x = float(rng.uniform(HOOD_FRONT_X + r, HOOD_BACK_X - r))
                y = float(rng.uniform(-HOOD_INNER_Y + r, HOOD_INNER_Y - r))
                if (self._inside_hood(x, y, r) and self._clear_of_task(x, y, r, h, th)
                        and self._clear_of_placed(x, y, r, placed)):
                    put(name, x, y, "extra")
                    n_extra -= 1
                    break
        return out

    # ---------------------------------------------------------------- scene application
    def _mocap_set(self, env, body, pos, quat=None):
        m, d = env.current_model, env.current_data
        mid = int(m.body_mocapid[m.body(body).id])
        d.mocap_pos[mid] = np.asarray(pos, dtype=float)
        if quat is not None:
            d.mocap_quat[mid] = np.asarray(quat, dtype=float)

    def _apply_theta(self, env, th: dict[str, Any]) -> None:
        m, d = env.current_model, env.current_data
        # sash rail: bottom edge at work surface + opening (paper geometry, lowered)
        sash_c = [SASH_X, 0.0, BENCH_Z + th["sash_opening_m"] + SASH_HALF[2]]
        self._mocap_set(env, "sash", sash_c)
        # jamb slabs = horizontal sash panes, each spanning from its inner edge out to the hood
        # frame post (|y| = 0.45); resized per episode (model persists across episodes, so the
        # size is always set explicitly), bounding volumes updated with the size
        yl, yr = th["jamb_inner_y"]
        hl = 0.5 * (FRAME_POST_Y - yl)
        hr = 0.5 * (FRAME_POST_Y + yr)
        jl = [SASH_X, yl + hl, BENCH_Z + JAMB_HALF[2]]
        jr = [SASH_X, yr - hr, BENCH_Z + JAMB_HALF[2]]
        for gname, half in (("jamb_l_g", hl), ("jamb_r_g", hr)):
            g = m.geom(gname).id
            size = np.array([JAMB_HALF[0], half, JAMB_HALF[2]], dtype=float)
            m.geom_size[g] = size
            m.geom_rbound[g] = float(np.linalg.norm(size))
            m.geom_aabb[g, 3:] = size
        self._mocap_set(env, "jamb_l", jl)
        self._mocap_set(env, "jamb_r", jr)
        for i, name in enumerate(LEGACY_PARKED_MOCAPS):
            try:
                self._mocap_set(env, name, [0.0, 2.6 + 0.05 * i, PARK_Z])
            except Exception:
                pass
        for i, name in enumerate(CLUTTER_ITEMS):
            self._mocap_set(env, name, [0.4 + 0.3 * (i % 4), 2.6 + 0.3 * (i // 4), PARK_Z],
                            [1.0, 0.0, 0.0, 0.0])
        for spec in th["clutter"]:
            q = R.from_euler("z", spec["yaw"]).as_quat(scalar_first=True)
            self._mocap_set(env, spec["body"], [spec["xy"][0], spec["xy"][1], BENCH_Z], q)
        gx, gy = th["goal_xy"]
        self._mocap_set(env, GOAL_MARKER, [gx, gy, BENCH_Z], [1.0, 0.0, 0.0, 0.0])
        th["sash_center"] = list(map(float, sash_c))
        th["jamb_centers"] = [list(map(float, jl)), list(map(float, jr))]
        mujoco.mj_forward(m, d)

    def _settle_jar(self, env, th: dict[str, Any]) -> None:
        m, d = env.current_model, env.current_data
        bid = m.body(JAR_NAME).id
        qadr = int(m.jnt_qposadr[int(m.body_jntadr[bid])])
        qpos0 = d.qpos.copy()
        x, y = th["jar_start_xy"]
        q = R.from_euler("z", th["jar_start_yaw"]).as_quat(scalar_first=True)
        d.qpos[qadr:qadr + 3] = [x, y, BENCH_Z + 0.0008]
        d.qpos[qadr + 3:qadr + 7] = q
        d.qvel[:] = 0.0
        for _ in range(120):
            mujoco.mj_step(m, d)
        jar_q = d.qpos[qadr:qadr + 7].copy()
        d.qpos[:] = qpos0
        d.qvel[:] = 0.0
        d.qpos[qadr:qadr + 7] = jar_q
        mujoco.mj_forward(m, d)
        th["jar_settled_pose"] = [float(v) for v in jar_q]

    def _place_robot(self, env, th: dict[str, Any]) -> None:
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        jar = om.get_object_by_name(JAR_NAME)
        task_cfg.pickup_obj_name = JAR_NAME
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(jar.pose).tolist()
        robot_view = env.current_robot.robot_view
        self._cur_base_xyz = tuple(th["base_xyz"])
        self._cur_base_yaw = float(th["base_yaw"])
        base_quat = R.from_euler("z", self._cur_base_yaw).as_quat(scalar_first=True)
        robot_view.base.pose = pos_quat_to_pose_mat(
            np.array(self._cur_base_xyz, dtype=float), np.array(base_quat, dtype=float)
        )
        mujoco.mj_forward(env.current_model, env.current_data)
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
        goal = list(task_cfg.pickup_obj_start_pose)
        goal[0], goal[1] = float(th["goal_xy"][0]), float(th["goal_xy"][1])
        task_cfg.pickup_obj_goal_pose = goal
        task_cfg.referral_expressions["pickup_obj_name"] = "reagent jar"
        task_cfg.referral_expressions_priority["pickup_obj_name"] = [(1.0, 1.0, "reagent jar")]

    def _sample_task(self, env: CPUMujocoEnv):
        th = self._draw_theta()
        self._theta = th
        self._apply_theta(env, th)
        self._settle_jar(env, th)
        self.candidate_objects = self._get_scene_objects(env)
        self._place_robot(env, th)
        self.setup_cameras(env)
        task = PcFumehoodPushTask(env, self.config)
        task.scene_params = {k: v for k, v in th.items() if not k.startswith("_")}
        task.goal_tol = float(self.GOAL_TOL)
        task.clearance_diag = bool(self.CLEARANCE_DIAG)
        task.require_clean = bool(self.REQUIRE_CLEAN)
        log.info(
            f"[PcPush] cell={th['cell']} lane_y={th['lane_y']:+.3f} jar=({th['jar_start_xy'][0]:.3f},"
            f"{th['jar_start_xy'][1]:+.3f}) goal=({th['goal_xy'][0]:.3f},{th['goal_xy'][1]:+.3f}) "
            f"push={th['push_distance_m']:.3f} sash_open={th['sash_opening_m']:.3f} "
            f"clutter={[c['body'].replace('pact_clutter_', '') + '@' + c['slot'] for c in th['clutter']]}"
        )
        return task


class PcFumehoodPushCheckSampler(PcFumehoodPushSampler):
    """Preflight: same layout distribution, plus per-step geometric clearance diagnostics."""

    CLEARANCE_DIAG = True


class PcFumehoodPushCollectSampler(PcFumehoodPushSampler):
    """Collection sampler: when the SCRIPTED EXPERT is the registered policy, an episode counts
    as a success (and is therefore saved) only if the robot touched nothing but the jar. With any
    other policy (closed-loop evaluation of a learned policy) success is the task criterion
    alone, exactly as with PcFumehoodPushSampler; collisions are reported separately."""

    REQUIRE_CLEAN = True


# ============================================================================================
# Task
# ============================================================================================
_HAZARD_GROUPS = {
    "sash": ("sash_g",),
    "jamb": ("jamb_l_g", "jamb_r_g"),
    "hood": ("hood_side_l", "hood_side_r", "hood_back", "hood_top", "hood_frame_l",
             "hood_frame_r"),
    "bench": ("bench_top", "bench_body"),
}


class PcFumehoodPushTask(PickTask):
    """Success = jar centre within goal_tol of the goal, upright, resting on the work surface."""

    goal_tol: float = 0.035
    clearance_diag: bool = False
    require_clean: bool = False
    MAX_TILT_DEG = 15.0

    def get_task_description(self) -> str:
        return "Push the reagent jar onto the yellow ring"

    def get_obs_scene(self) -> dict[str, Any]:
        scene = super().get_obs_scene()
        scene["scene_params"] = getattr(self, "scene_params", {})
        scene["push_diag"] = self._push_summary()
        return scene

    # ---------------------------------------------------------------- state helpers
    def _jar_state(self, data):
        obj = MlSpacesObject(data=data, object_name=self.config.task_config.pickup_obj_name)
        p = np.asarray(obj.position, dtype=float)
        zaxis = R.from_quat(np.asarray(obj.quat, dtype=float), scalar_first=True).as_matrix()[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(zaxis[2], -1.0, 1.0))))
        return obj, p, tilt

    def _goal_xy(self) -> np.ndarray:
        return np.asarray(self.config.task_config.pickup_obj_goal_pose[:2], dtype=float)

    def _verdict(self, p: np.ndarray, tilt: float) -> tuple[bool, float]:
        dxy = float(np.linalg.norm(p[:2] - self._goal_xy()))
        on_surface = abs(float(p[2]) - BENCH_Z) < 0.015
        return bool(dxy <= self.goal_tol and tilt < self.MAX_TILT_DEG and on_surface), dxy

    # ---------------------------------------------------------------- per-step metrics
    def _reset_push_diag_if_new_episode(self) -> None:
        if self.episode_step_count == 0 or not hasattr(self, "_pd"):
            self._pd = {"max_tilt_deg": 0.0, "min_clear": {}, "min_clear_body": {},
                        "contact_steps": 0, "contacts_by_body": {}, "logged": False,
                        "jar_contact_steps": 0}
            self._diag_cache = None

    _UNIT = np.stack(np.meshgrid(*(np.linspace(-1.0, 1.0, 5),) * 3, indexing="ij"),
                     -1).reshape(-1, 3)

    def _robot_and_hazard_geoms(self):
        if getattr(self, "_diag_cache", None) is not None:
            return self._diag_cache
        model = self.env.current_model
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        robot_geoms = [g for g in range(model.ngeom)
                       if model.body_rootid[model.geom_bodyid[g]] == robot_root
                       and (model.geom_contype[g] or model.geom_conaffinity[g])
                       and model.body(int(model.geom_bodyid[g])).name != "robot_0/base"]
        groups: dict[str, list[int]] = {}
        for key, names in _HAZARD_GROUPS.items():
            ids = []
            for n in names:
                try:
                    ids.append(int(model.geom(n).id))
                except Exception:
                    pass
            groups[key] = ids
        th = getattr(self, "scene_params", {}) or {}
        for spec in th.get("clutter", []):
            b = model.body(spec["body"]).id
            groups[spec["body"]] = [g for g in range(model.ngeom)
                                    if model.geom_bodyid[g] == b
                                    and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self._diag_cache = (robot_geoms, groups)
        return self._diag_cache

    @staticmethod
    def _sdf(model, data, g: int, pts: np.ndarray) -> np.ndarray:
        """Signed distance from world points to primitive geom g (box / cylinder / capsule)."""
        Rm = data.geom_xmat[g].reshape(3, 3)
        q = (pts - data.geom_xpos[g]) @ Rm
        sz = model.geom_size[g]
        t = int(model.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            d = np.abs(q) - sz
            out = np.linalg.norm(np.maximum(d, 0.0), axis=1)
            return out + np.minimum(d.max(axis=1), 0.0)
        if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            d = np.stack([np.hypot(q[:, 0], q[:, 1]) - sz[0], np.abs(q[:, 2]) - sz[1]], axis=1)
            out = np.linalg.norm(np.maximum(d, 0.0), axis=1)
            return out + np.minimum(d.max(axis=1), 0.0)
        if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            z = np.clip(q[:, 2], -sz[1], sz[1])
            return np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2 + (q[:, 2] - z) ** 2) - sz[0]
        return np.full(len(pts), np.inf)

    def _clearance_step(self, data) -> None:
        """Min distance from the robot (points on the local AABBs of its collision geoms, so a
        conservative LOWER bound on the true clearance) to every hazard group."""
        model = self.env.current_model
        robot_geoms, groups = self._robot_and_hazard_geoms()
        pts, owner = [], []
        for g in robot_geoms:
            c = model.geom_aabb[g, :3]
            h = model.geom_aabb[g, 3:]
            w = data.geom_xpos[g] + (c + self._UNIT * h) @ data.geom_xmat[g].reshape(3, 3).T
            pts.append(w)
            owner.extend([g] * len(w))
        pts = np.concatenate(pts)
        owner = np.asarray(owner)
        for key, hz in groups.items():
            best = self._pd["min_clear"].get(key, 9.0)
            for hg in hz:
                d = self._sdf(model, data, hg, pts)
                i = int(np.argmin(d))
                if d[i] < best:
                    best = float(d[i])
                    self._pd["min_clear_body"][key] = (model.geom(int(owner[i])).name
                                                       or model.body(int(model.geom_bodyid[owner[i]])).name)
            self._pd["min_clear"][key] = float(best)

    def _contact_step(self, data) -> None:
        model = self.env.current_model
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        jar_id = model.body(JAR_NAME).id
        hit = False
        for c in data.contact[: data.ncon]:
            if c.dist > 0:
                continue
            r1 = model.body_rootid[model.geom_bodyid[c.geom1]]
            r2 = model.body_rootid[model.geom_bodyid[c.geom2]]
            if (r1 == robot_root) == (r2 == robot_root):
                continue
            other = int(r2 if r1 == robot_root else r1)
            if other == jar_id:
                self._pd["jar_contact_steps"] += 1
                continue
            g = c.geom2 if r1 == robot_root else c.geom1
            name = model.body(other).name or ""
            if not name or name == "world":
                name = model.geom(int(g)).name or "world"
            if "floor" in name.lower():
                continue
            self._pd["contacts_by_body"][name] = self._pd["contacts_by_body"].get(name, 0) + 1
            hit = True
        if hit:
            self._pd["contact_steps"] += 1

    def get_info(self) -> list[dict[str, Any]]:
        self._reset_obstacle_diag_if_new_episode()
        self._reset_push_diag_if_new_episode()
        data = self._env.mj_datas[0]
        obj, p, tilt = self._jar_state(data)
        self._pd["max_tilt_deg"] = max(self._pd["max_tilt_deg"], tilt)
        try:
            self._accumulate_obstacle_diag(data, obj)
            self._contact_step(data)
            if self.clearance_diag and self.episode_step_count % 2 == 0:
                self._clearance_step(data)
        except Exception as e:  # pragma: no cover - diagnostics never break a rollout
            log.debug(f"[PcPush] diag failed: {e!r}")
        success, dxy = self._verdict(p, tilt)
        info = {
            "position_error": dxy,
            "rotation_error": float(np.radians(tilt)),
            "success": success,
            "episode_step": self.episode_step_count,
        }
        self._maybe_log_obstacle_diag(success)
        self._maybe_log_push_diag(success, p, dxy)
        return [info]

    def _push_summary(self) -> dict[str, Any]:
        pd = getattr(self, "_pd", None) or {}
        return {
            "max_tilt_deg": float(pd.get("max_tilt_deg", 0.0)),
            "robot_env_contact_steps": int(pd.get("contact_steps", 0)),
            "robot_env_contacts_by_body": dict(pd.get("contacts_by_body", {})),
            "jar_contact_steps": int(pd.get("jar_contact_steps", 0)),
            "min_clearance_m": dict(pd.get("min_clear", {})),
        }

    def _maybe_log_push_diag(self, success: bool, p: np.ndarray, dxy: float) -> None:
        if self._pd.get("logged"):
            return
        try:
            ending = bool(self.is_timed_out().any() or self.is_terminal().any())
        except Exception:
            ending = False
        if not ending:
            return
        self._pd["logged"] = True
        mc = self._pd.get("min_clear", {})
        clutter_mc = {k: v for k, v in mc.items() if k.startswith("pact_clutter_")}
        worst = min(clutter_mc.items(), key=lambda kv: kv[1]) if clutter_mc else ("-", float("nan"))
        log.info(
            "[PushDiag] success=%s jar=(%.3f,%.3f,%.3f) d_goal=%.3f max_tilt=%.1fdeg "
            "env_contact_steps=%d contacts=%s jar_contact_steps=%d steps=%d "
            "clear: sash=%.3f jamb=%.3f hood=%.3f bench=%.3f clutter_min=%.3f(%s)",
            success, p[0], p[1], p[2], dxy, self._pd["max_tilt_deg"], self._pd["contact_steps"],
            self._pd["contacts_by_body"] or "{}", self._pd["jar_contact_steps"],
            self.episode_step_count, mc.get("sash", float("nan")), mc.get("jamb", float("nan")),
            mc.get("hood", float("nan")), mc.get("bench", float("nan")), worst[1],
            worst[0].replace("pact_clutter_", ""),
        )

    def judge_success(self) -> bool:
        data = self._env.mj_datas[0]
        _obj, p, tilt = self._jar_state(data)
        ok, dxy = self._verdict(p, tilt)
        pd = getattr(self, "_pd", None) or {}
        # Strictness applies to the SCRIPTED EXPERT only (collection save gate): a demo counts
        # only if it touched nothing but the jar. A learned policy under evaluation is always
        # scored on the task criterion alone; its collisions are reported separately.
        expert = isinstance(getattr(self, "_registered_policy", None), PcFumehoodPushPolicy)
        if ok and self.require_clean and expert:
            # policy-step contact sampling plus the expert's every-control-step contact audit
            audit = getattr(getattr(self, "_registered_policy", None), "_pact_contact_audit", None)
            audit_clean = True
            if audit is not None:
                try:
                    audit_clean = bool(audit.summary().get("collision_free", True))
                except Exception:
                    audit_clean = True
            if (pd.get("contact_steps", 0) > 0 or not audit_clean
                    or pd.get("max_tilt_deg", 0.0) >= self.MAX_TILT_DEG):
                ok = False
                log.info(f"[PcPush] judge: task success but NOT clean (contacts="
                         f"{pd.get('contacts_by_body')}, audit_clean={audit_clean}, "
                         f"max_tilt={pd.get('max_tilt_deg', 0.0):.1f})")
        if ok != getattr(self, "_last_judge", None):
            self._last_judge = ok
            log.info(f"[PcPush] judge: success={ok} d_goal={dxy:.3f} tilt={tilt:.1f} z={p[2]:.3f}")
        return ok


# ============================================================================================
# Expert
# ============================================================================================
_R0 = np.array([[0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0]])


def _push_R(pitch_deg: float) -> np.ndarray:
    p = np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(p), 0.0, np.sin(p)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(p), 0.0, np.cos(p)]])
    return Ry @ _R0


def _pose(pos, Rm) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


class _SagState:
    """Integral task-space position offset shared by consecutive servo lines."""

    def __init__(self, gain: float, max_offset: float) -> None:
        self.gain = gain
        self.max_offset = max_offset
        self.offset = np.zeros(3)


class _ServoLine(ActionPrimitive):
    """Straight task-space line (linear position, slerped orientation) at constant speed with
    integral anti-sag correction; holds the end pose for ``hold`` seconds."""

    def __init__(self, robot_view, tcp_to_jp_fn, get_tcp_fn, start_pose, end_pose, speed,
                 name, sag: _SagState, hold: float = 0.0, min_time: float = 0.25):
        self.move_time = max(float(np.linalg.norm(end_pose[:3, 3] - start_pose[:3, 3])) / speed,
                             min_time)
        super().__init__(robot_view, self.move_time + hold)
        self.tcp_to_jp_fn = tcp_to_jp_fn
        self.get_tcp_fn = get_tcp_fn
        self.start_pose = start_pose
        self.end_pose = end_pose
        self._name = name
        self.sag = sag
        self._slerp = Slerp([0.0, 1.0], R.from_matrix(np.stack([start_pose[:3, :3],
                                                                 end_pose[:3, :3]])))
        self._logged = False

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            log.info(f"Moving to {self._name}")
        done = self.elapsed_time() >= self.duration
        if done and not self._logged:
            self._logged = True
            err = self.end_pose[:3, 3] - self.get_tcp_fn()[:3, 3]
            log.info(f"[PushServo:{self._name}] end err={np.round(err, 4)} "
                     f"offset={np.round(self.sag.offset, 4)}")
        return done

    def target_pose(self) -> np.ndarray:
        t = 0.0 if self.start_time is None else min(1.0, self.elapsed_time() / self.move_time)
        T = np.eye(4)
        T[:3, :3] = self._slerp([t]).as_matrix()[0]
        T[:3, 3] = self.start_pose[:3, 3] + t * (self.end_pose[:3, 3] - self.start_pose[:3, 3])
        return T

    def get_current_action(self):
        tgt = self.target_pose()
        err = tgt[:3, 3] - self.get_tcp_fn()[:3, 3]
        self.sag.offset = np.clip(self.sag.offset + self.sag.gain * err,
                                  -self.sag.max_offset, self.sag.max_offset)
        cmd = tgt.copy()
        cmd[:3, 3] = cmd[:3, 3] + self.sag.offset
        mg = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.tcp_to_jp_fn(mg, cmd)

    def get_current_phase(self) -> str:
        return self._name


class _CloseHold(ActionPrimitive):
    """Close the gripper while holding the arm at the joint configuration measured when the
    primitive starts (the inherited GripperAction replays the stored ctrl targets, which lets the
    arm drift during the close)."""

    def __init__(self, robot_view, duration: float):
        super().__init__(robot_view, duration)
        self._q = None

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            self._q = np.asarray(self.robot_view.get_qpos_dict()["arm"], dtype=float).copy()
            mg = self.robot_view.get_gripper_movegroup_ids()[0]
            self.robot_view.get_gripper(mg).set_gripper_ctrl_open(False)
            log.info("Closing gripper (pusher)...")
        return self.elapsed_time() >= self.duration

    def get_current_action(self):
        action = self.robot_view.get_ctrl_dict()
        if self._q is not None:
            action["arm"] = self._q.copy()
        return action

    def get_current_phase(self) -> str:
        return "gripper-close"


class _JointLine(ActionPrimitive):
    """Linear joint-space move of the arm from the configuration at start to ``q_target``."""

    def __init__(self, robot_view, q_target: np.ndarray, speed_rad_s: float, name: str,
                 min_time: float = 1.0):
        self.q_target = np.asarray(q_target, dtype=float)
        self.speed = float(speed_rad_s)
        self.min_time = float(min_time)
        self._q0 = None
        self._name = name
        # duration is fixed at plan time from the CURRENT configuration (the rollout starts there)
        q_now = np.asarray(robot_view.get_qpos_dict()["arm"], dtype=float)
        dur = max(float(np.max(np.abs(self.q_target - q_now))) / self.speed, self.min_time)
        super().__init__(robot_view, dur)

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            self._q0 = np.asarray(self.robot_view.get_qpos_dict()["arm"], dtype=float).copy()
            log.info(f"Moving to {self._name} (joint space, {self.duration:.2f}s)")
        return self.elapsed_time() >= self.duration

    def get_current_action(self):
        t = 0.0 if self.start_time is None else min(1.0, self.elapsed_time() / self.duration)
        # smoothstep keeps the joint-space move free of velocity jumps at both ends
        s = t * t * (3.0 - 2.0 * t)
        q0 = self._q0 if self._q0 is not None else self.q_target
        action = self.robot_view.get_ctrl_dict()
        action["arm"] = q0 + s * (self.q_target - q0)
        return action

    def get_current_phase(self) -> str:
        return self._name


class PcFumehoodPushPolicy(BaseObjectManipulationPlannerPolicy):
    """Scripted pusher (see module docstring)."""

    PITCH_DEG = 40.0          # approach axis pitched down: fingertips lead, housing stays high
    PUSH_TCP_H = 0.034        # TCP height above the work surface while pushing
    FRONT_OFFSET = 0.025      # TCP -> effective front of the closed finger face along +x (measured
                              # 0.019 at first touch, ~0.027 at the end of the push under load)
    PRE_GAP = 0.05            # pre-push standoff behind the jar's back face
    CREEP_IN = 0.012          # creep this far past nominal contact before the steady push
    FINAL_SLOW = 0.025        # last stretch of the push at the slow speed
    STAGE_X = 0.50            # staging point just outside the sash plane, above the bench edge
    STAGE_Z = 0.93
    JOINT_SPEED = 1.0         # rad/s for the joint-space move home -> staging
    IK_SEED = None            # arm seed for the staging IK (None: the robot's noise-free home)

    V_APPROACH = 0.13
    V_CREEP = 0.045
    V_PUSH = 0.06
    V_FINAL = 0.03
    V_RETREAT = 0.14
    V_HOME = 0.16

    def _tcp_now(self) -> np.ndarray:
        mg = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.robot_view.get_move_group(mg).leaf_frame_to_world.copy()

    def _jar_xy(self) -> np.ndarray:
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        return np.asarray(obj.position, dtype=float)[:2]

    def reset(self, reset_retries: bool = True):
        from molmo_spaces.tasks.pact_contact_audit import PactContactAudit

        if reset_retries or not hasattr(self, "_pact_contact_audit"):
            self._pact_contact_audit = PactContactAudit()
            self._pact_control_step = 0
        self.task._contact_audit_hook = self._pact_contact_audit
        out = super().reset(reset_retries)
        # GraspPoseSensor (obs/extra/grasp_pose) reads target_poses["grasp"]; the base class only
        # fills target_poses from TCPMoveSequences, so register the servo-line end poses too.
        for prim in self.action_primitives:
            if isinstance(prim, _ServoLine):
                self.target_poses[prim.get_current_phase()] = prim.end_pose
        return out

    def get_action(self, observation):
        self._pact_contact_audit.observe(self.task.env, self._pact_control_step)
        action = super().get_action(observation)
        self._pact_control_step += 1
        return action

    def get_info(self):
        info = super().get_info()
        self._pact_contact_audit.observe(self.task.env, self._pact_control_step)
        summary = self._pact_contact_audit.summary()
        summary.pop("contact_frames", None)
        info["pact_contact_audit"] = summary
        return info

    def _ik_stage(self, pose: np.ndarray) -> np.ndarray | None:
        """IK for the staging pose from the noise-free home configuration, so every episode
        enters the push with the same elbow-up arm posture regardless of the start noise."""
        rv = self.robot_view
        kin = self.task.env.current_robot.kinematics
        mg = rv.get_gripper_movegroup_ids()[0]
        unlocked = [g for g in rv.move_group_ids() if g not in set(rv.get_gripper_movegroup_ids())]
        q0 = {k: np.asarray(v, dtype=float).copy() for k, v in rv.get_qpos_dict().items()}
        seed = self.IK_SEED if self.IK_SEED is not None else self.config.robot_config.init_qpos.get("arm")
        if seed is not None:
            q0["arm"] = np.asarray(seed, dtype=float).copy()
        sol = kin.ik(mg, pose, unlocked, q0, rv.base.pose)
        if sol is None:
            return None
        return np.asarray(sol["arm"], dtype=float)

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        th = getattr(self.task, "scene_params", {}) or {}
        jar_xy = self._jar_xy()
        goal_x = float(th["goal_xy"][0])
        y = float(jar_xy[1])                      # push line through the jar's actual centre
        Rp = _push_R(self.PITCH_DEG)
        z = BENCH_Z + self.PUSH_TCP_H
        x_contact = float(jar_xy[0]) - JAR_R - self.FRONT_OFFSET
        x_pre = x_contact - self.PRE_GAP
        x_creep = x_contact + self.CREEP_IN
        x_end = goal_x - JAR_R - self.FRONT_OFFSET
        x_slow = max(x_creep, x_end - self.FINAL_SLOW)

        p_stage = _pose([self.STAGE_X, y, self.STAGE_Z], Rp)
        p_pre = _pose([x_pre, y, z], Rp)
        p_creep = _pose([x_creep, y, z], Rp)
        p_slow = _pose([x_slow, y, z], Rp)
        p_end = _pose([x_end, y, z], Rp)
        # back off along the push line to the pre-push point, then retrace the approach up and
        # out to the staging point (retreating low toward the robot folds the forearm down onto
        # the bench's front edge)
        p_back = _pose([x_pre, y, z + 0.008], Rp)

        q_stage = self._ik_stage(p_stage)
        if q_stage is None:
            raise ValueError("IK failed for the push staging pose")

        sag = _SagState(gain=0.35, max_offset=0.05)
        rv = self.robot_view

        def line(a, b, v, name, hold=0.0):
            return _ServoLine(rv, self._tcp_to_jp_fn, self._tcp_now, a, b, v, name, sag, hold=hold)

        self._plan_log = dict(y=y, x_pre=x_pre, x_contact=x_contact, x_end=x_end, z=z)
        log.info(f"[PcPushExpert] lane y={y:+.3f} pre={x_pre:.3f} contact={x_contact:.3f} "
                 f"end={x_end:.3f} z={z:.3f} pitch={self.PITCH_DEG} q_stage={np.round(q_stage, 2)}")
        return [
            _CloseHold(rv, self.policy_config.gripper_close_duration),
            _JointLine(rv, q_stage, self.JOINT_SPEED, "pregrasp", min_time=1.2),
            line(p_stage, p_pre, self.V_APPROACH, "pregrasp", hold=0.2),
            line(p_pre, p_creep, self.V_CREEP, "grasp"),
            line(p_creep, p_slow, self.V_PUSH, "preplace"),
            line(p_slow, p_end, self.V_FINAL, "place", hold=0.2),
            line(p_end, p_back, self.V_RETREAT, "retreat"),
            line(p_back, p_stage, self.V_HOME, "go_home", hold=0.2),
        ]


class PcFumehoodPushPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    """Wires PcFumehoodPushPolicy; no grasp machinery, no retries after the rollout starts."""

    max_retries: int = 0
    filter_colliding_grasps: bool = False
    filter_feasible_grasps: bool = False
    max_sequential_ik_failures: int = 12
    gripper_close_duration: float = 0.5

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PcFumehoodPushPolicy


__all__ = [
    "PC_FUMEHOOD_PUSH_ENVIRONMENT_VERSION",
    "PcFumehoodPushSampler",
    "PcFumehoodPushCheckSampler",
    "PcFumehoodPushCollectSampler",
    "PcFumehoodPushTask",
    "PcFumehoodPushPolicy",
    "PcFumehoodPushPolicyConfig",
]
