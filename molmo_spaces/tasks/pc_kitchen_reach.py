"""``pc_kitchen_reach``: move a jar along the counter UNDER the kitchen wall cabinets, where the
risk sits on the SKIN-COVERED wrist knuckle and forearm, not on the gripper.

Why this suite exists (2026-10-13 contact diagnostic)
-----------------------------------------------------
In every earlier pc_* suite the hazardous contacts were made by the gripper (coupler, follower,
pad, base) or fr3_link7, parts with no skin; only one contact in ~40 scenes touched a skinned link.
Here the geometry is chosen so that the closest hazard to the arm is a surface only the skin on
links 5-6 can face: the underside of the upper cabinets (and the LED strip under their front
edge). The gripper works 5-13 cm above the counter with every counter item kept >= 6 cm away.

Scene
-----
The ``pc_kitchen`` room (counter 0.90 m, backsplash x = 0.852, fridge on the left, cooktop and
range hood on the far right, closed room, FR3 on a 0.28 m platform: link0 at z = 0.63), generated
by :func:`molmo_spaces.tasks.pc_kitchen.build_scene_xml` with this suite's eight cell layouts:
upper-cabinet bottom 0.47 / 0.49 / 0.51 / 0.53 m above the counter (real kitchens: 18-21 in),
cabinet depth 0.31 / 0.35 m (front face x = 0.55 / 0.51), fridge side panel y = 0.36 / 0.40, and
the transfer direction (jar on the fridge side or on the cooktop side). One static XML per cell:
``custom_scenes/pc_kitchen_reach_c<k>.xml`` (:func:`write_pc_kitchen_reach_scenes`).

Task
----
Pick a lidded jar (free cylinder ``cavity_obj_0/kitchen_jar``) that stands on the counter under
the cabinets and set it on a cork coaster (``place_receptacle_coaster``) 0.28-0.44 m along the
backsplash, also under the cabinets. Success = the pc_shelf_pnp rule (jar upright on the coaster,
centre within 4 cm, resting, released).

Expert (scripted TCP segments, one fixed tool orientation)
----------------------------------------------------------
The hand points down, tilted 20 deg toward +y (approach pitched TOOL_PITCH_DEG = 70 below
horizontal), fingers closing along x. ready (in front of the cabinets) -> in front of the slide start -> slide start
beside the jar (fingertips 2.4 cm from it) -> slide in along the open finger slot (+y, rising 20 deg: the jar enters between the fingers with
no motion along the approach axis) -> close -> lift 1.5 cm -> lateral carry under the cabinets
(slow) -> lower onto the coaster -> open -> slide out along the slot (-y, descending) -> straight
back out (-x) -> ready (:func:`reach_waypoints`).
With the hand pointing down, the wrist knuckle (fr3_link6) and the forearm next to it (fr3_link5)
are the arm's highest parts above the hand: up to ARM_TOP = 0.341 m above the TCP (0.325-0.341
along the lane), fr3_link7 tops out at +0.21 m, the gripper at +0.16 m. The per-cell TCP heights
are set from the cabinet height so that during the carry the wrist/forearm top is C_CARRY =
1.5 cm (worst point of the lane; up to ~3 cm elsewhere) below the LED strip under the cabinet
front edge, ~3-4.5 cm while sliding in / placing. fr3_link7 stays ~14 cm below the cabinets and
the fingertips stay >= ~5 cm above the counter (lowest at the slide start of the 0.47 m cells).
Why the 20 deg tilt: it turns the wrist-side forearm sensor (link5_back_sensor_5) and the knuckle
sensor link6_sensor_1 toward the cabinet underside, so the skin reads the clearance of the
closest arm point almost directly (design scan: sensed 1.4-3.1 cm at a true 1.7-2.1 cm). A
straight top-down hand has the same clearances but a blind knuckle top (closest skin reading
~7 cm at a true 2 cm). The exo camera (behind/above the robot) cannot see the wrist-to-cabinet
gap: the cabinet front hides it.

The in-rollout audit :meth:`PCKitchenReachTask.reach_clearance` measures the per-link clearance
of every datagen / eval episode (``obs_scene['pc_clearance']``; scripts/audit_arm_clearance.py).

Clutter (mocap, resting on the counter, never floating; ``pact_clutter_*`` = clutter hazards)
------------------------------------------------------------------------------------------------
* behind the lane, against the backsplash: 2-4 short items (spice jars, shakers, a mug; <= 11 cm
  tall) whose faces stay >= 6 cm behind the open fingers' swept box;
* beyond the cooktop-side lane end: 2-3 tall items (the pc_kitchen ceramic canisters, an oil
  bottle, a soap bottle), >= 6 cm outside the swept box of the slide-out and the forearm.

Collection recipe (configs ``FrankaSkinPCKitchenReach*``): DART-style executed-action noise and
the contact-tolerant save gate of ``pc_shelf_hard`` (success AND hazard contact <= 150 frames x
2 ms). Eval config: noise off, no gate. Contact attribution, hazard audit and success rule are
``pc_shelf_hard``'s (:class:`PCShelfPnPHardTask`).
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

from molmo_spaces.tasks import pc_kitchen as _kit
from molmo_spaces.tasks.pc_shelf import (
    SCENE_DIR,
    _jsonable,
    _PCShelfSamplerBase,
    _T,
)
from molmo_spaces.tasks.pc_shelf_hard import (
    PCShelfPnPHardExpert,
    PCShelfPnPHardExpertConfig,
    PCShelfPnPHardTask,
)
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

log = logging.getLogger(__name__)

PC_KITCHEN_REACH_ENV_VERSION = "pc_kitchen_reach_v1"
N_CELLS = 8

COUNTER_Z = _kit.COUNTER_Z          # 0.90
WALL_X = _kit.WALL_X                # 0.86
BACKSPLASH_X = _kit.BACKSPLASH_X    # 0.852
BASE_Z = 0.28                       # pedestal bottom on the kitchen platform (link0 at z = 0.63)
LED_H = 0.006                       # under-cabinet LED strip (6 mm below the cabinet bottom)

# ---- the hand: approach pitched 70 deg down, tilted 20 deg toward +y; fingers close along x ----
# With this tilt the 'back' forearm sensor next to the wrist (link5_back_sensor_5) and the wrist
# knuckle sensor (link6_sensor_1) face the cabinet underside, so the skin reads the clearance of
# the closest arm point almost directly (design scan: sensed 1.4-3.1 cm at a true 1.7-2.1 cm). A
# straight top-down hand keeps the same clearances but its knuckle top is blind (closest skin
# reading ~7 cm at a true 2 cm).
TOOL_PITCH_DEG = 70.0
TOOL_AZ_DEG = 90.0
# Measured on model_hybrid.xml with link0 at z = 0.63 (TCP-relative, this hand orientation):
ARM_TOP = 0.341       # top of the skinned wrist / forearm above the TCP under the cabinets (max
                      # over the lane; 0.325-0.341 depending on the arm configuration)
GRIP_DX = 0.077       # open fingers' half-extent along x (they close along x)
LINK5_DY_NEG = 0.207  # forearm reach toward -y (the wrist leans back toward -y)
LINK5_DY_POS = 0.176  # forearm reach toward +y
JAR_TOP_ABOVE_TCP = 0.025   # TCP 2.5 cm below the jar's top: nothing of the open hand lies inside
                            # r = 0.034 of the approach axis closer than 0.056 m to the TCP, and
                            # the tilted jar rim reaches 0.034 m
FINGER_SLOT_HALF = 0.0176   # finger assembly half-width along the slot (tool x) near the pads

# ---- expert constants ----
C_CARRY = 0.015       # top of the skinned wrist/forearm -> LED-strip underside during the carry
LIFT = 0.015          # jar lift off the counter for the carry
SLIDE = 0.085         # slide in / out along the open finger slot (tool x: -y and 20 deg down);
                      # beside the jar the tilted fingertips then pass it with a ~2.4 cm gap
FRONT_DX = 0.12       # the hand reaches the slide start from this far in front of it (along +x),
                      # so its path from / to the ready pose never sweeps through the jar
COASTER_R = 0.05
COASTER_HALF_T = 0.003
PLACE_TOL = 0.04
READY_TCP = (0.40, 0.0, 0.99)
# IK of READY_TCP x reach_tool_rotation() on the kitchen platform (seed [0,-0.3,0,-2.2,0,1.9,0]
# solved at (0.62, 0, 1.0) first, then moved to READY_TCP).
PC_KITCHEN_REACH_READY_QPOS: list[float] = [-0.1312, -0.4680, -0.0617, -2.3463, 0.3238, 1.8003, 1.2986]

CLR = 0.06            # clutter surface -> arm swept box


@dataclass(frozen=True)
class ReachCell:
    gap: float            # upper-cabinet bottom above the counter
    depth: float          # upper-cabinet depth (front face at WALL_X - depth)
    fridge_y: float       # fridge right side panel (faces the work area)
    jar_fridge_side: bool # jar starts on the fridge side (+y) and goes toward the cooktop

    @property
    def cab_bottom_z(self) -> float:
        return COUNTER_Z + self.gap

    @property
    def cab_front_x(self) -> float:
        return WALL_X - self.depth

    @property
    def z_carry(self) -> float:
        """TCP height of the carry: the skinned wrist / forearm top C_CARRY under the LED strip."""
        return self.cab_bottom_z - LED_H - ARM_TOP - C_CARRY

    @property
    def z_grasp(self) -> float:
        return self.z_carry - LIFT

    @property
    def jar_h(self) -> float:
        return round(self.z_grasp - COUNTER_Z + JAR_TOP_ABOVE_TCP, 4)

    def params(self, k: int) -> dict[str, float]:
        return {
            "cell": k, "fridge_y": self.fridge_y, "cab_gap": self.gap, "cab_depth": self.depth,
            "cab_bottom_z": self.cab_bottom_z, "cab_front_x": self.cab_front_x,
            "left_wall_y": self.fridge_y + _kit.FRIDGE_W + 0.04,
        }


PC_KITCHEN_REACH_CELLS: tuple[ReachCell, ...] = (
    ReachCell(0.47, 0.31, 0.36, True),
    ReachCell(0.49, 0.35, 0.40, False),
    ReachCell(0.51, 0.31, 0.40, True),
    ReachCell(0.53, 0.35, 0.36, False),
    ReachCell(0.47, 0.35, 0.40, False),
    ReachCell(0.49, 0.31, 0.36, True),
    ReachCell(0.51, 0.35, 0.36, False),
    ReachCell(0.53, 0.31, 0.40, True),
)

PC_KITCHEN_REACH_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_kitchen_reach_c{k}.xml") for k in range(N_CELLS)
)


def write_pc_kitchen_reach_scenes() -> list[str]:
    """(Re)generate the eight cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k, c in enumerate(PC_KITCHEN_REACH_CELLS):
        p = Path(PC_KITCHEN_REACH_SCENES[k])
        xml = _kit.build_scene_xml(k, params=c.params(k), model="pc_kitchen_reach")
        xml = xml.replace("GENERATED by molmo_spaces/tasks/pc_kitchen.py",
                          "GENERATED by molmo_spaces/tasks/pc_kitchen_reach.py "
                          "(write_pc_kitchen_reach_scenes, pc_kitchen room generator)", 1)
        p.write_text(xml)
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


def reach_tool_rotation() -> np.ndarray:
    """Tool frame: approach (z) pitched TOOL_PITCH_DEG below horizontal toward +y (20 deg off
    vertical), fingers (y) along world -x (they close along x), slot (x) = (0, -0.94, -0.34).
    Joint 7 stays ~0.8-1.9 rad over the lane (limit 3.0)."""
    p, az = np.radians(TOOL_PITCH_DEG), np.radians(TOOL_AZ_DEG)
    z = np.array([np.cos(p) * np.cos(az), np.cos(p) * np.sin(az), -np.sin(p)])
    y = np.array([-np.sin(az), np.cos(az), 0.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def slide_dir() -> np.ndarray:
    """Unit direction along the open finger slot (tool x): toward -y and 20 deg down."""
    return reach_tool_rotation()[:, 0].copy()


def reach_waypoints(c: "ReachCell", xa: float, ya: float, xb: float, yb: float):
    """Expert TCP waypoints [(phase, xyz | None, speed)] for one episode (all under the cabinets).

    Slide in along the slot (+y, rising 20 deg) so the jar enters between the open fingers with
    no motion along the approach axis; close; lift; carry along the backsplash; lower onto the
    coaster; open; slide out along the slot (-y, descending); back out to the ready pose."""
    u = slide_dir()
    zg, zc = c.z_grasp, c.z_carry
    g = np.array([xa, ya, zg])
    pre = g + SLIDE * u
    front = pre - np.array([FRONT_DX, 0.0, 0.0])
    z_place = zg + 2 * COASTER_HALF_T + 0.006
    pl = np.array([xb, yb, z_place])
    out = pl + SLIDE * u
    back = out + np.array([-FRONT_DX, 0.0, 0.01])
    return [
        ("pregrasp", front.tolist(), "fast"),
        ("pregrasp", pre.tolist(), "insert"),
        ("grasp", g.tolist(), "insert"),
        ("CLOSE", None, None),
        ("lift", [xa, ya, zc], "lift"),
        ("preplace", [xb, yb, zc], "carry"),
        ("place", pl.tolist(), "lower"),
        ("OPEN", None, None),
        ("retreat", out.tolist(), "lift"),
        ("retreat", back.tolist(), "mid"),
        ("retreat", list(READY_TCP), "fast"),
    ]


# ---------------------------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------------------------
JARS = (  # radius, body rgba, lid rgba, label
    (0.0310, (0.86, 0.52, 0.16), (0.85, 0.85, 0.86), "honey jar"),
    (0.0305, (0.62, 0.10, 0.14), (0.95, 0.95, 0.93), "jam jar"),
    (0.0315, (0.30, 0.52, 0.22), (0.20, 0.20, 0.21), "pesto jar"),
    (0.0300, (0.88, 0.80, 0.60), (0.55, 0.36, 0.20), "spice jar"),
    (0.0310, (0.78, 0.20, 0.10), (0.85, 0.80, 0.30), "sauce jar"),
    (0.0305, (0.92, 0.92, 0.90), (0.15, 0.35, 0.60), "yogurt jar"),
    (0.0315, (0.35, 0.22, 0.14), (0.80, 0.80, 0.82), "coffee jar"),
    (0.0300, (0.95, 0.70, 0.10), (0.20, 0.20, 0.22), "mustard jar"),
)
CERAMIC = (0.93, 0.92, 0.87)
LID_WOOD = (0.55, 0.40, 0.26)
SHORT_COLORS = [(0.80, 0.76, 0.66), (0.62, 0.30, 0.18), (0.20, 0.42, 0.30), (0.92, 0.92, 0.92),
                (0.25, 0.25, 0.28), (0.70, 0.60, 0.20), (0.16, 0.30, 0.55)]

# XML canisters of the pc_kitchen room (origin at the base centre): name -> (radius, height)
XML_CANISTERS = {
    "pact_clutter_can_a": (0.057, 0.236),
    "pact_clutter_can_b": (0.050, 0.195),
    "pact_clutter_can_c": (0.044, 0.153),
}
PARK_Z = -1.0          # under the floor (inside the room footprint)


class PCKitchenReachSampler(_PCShelfSamplerBase):
    """Jar under the wall cabinets -> coaster along the backsplash, top-down hand."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_KITCHEN_REACH_ENV_VERSION
    SUITE = "kitchen_reach"
    BASE_JIT_XY = (0.01, 0.015)
    BASE_JIT_YAW = 0.02
    MAX_DRAWS = 40
    LANE_X = (0.59, 0.63)
    LANE_MIN_SPAN = 0.28
    COOK_Y = (-0.22, -0.12)        # lane end on the cooktop side (slide in/out reaches 8 cm further)
    FRIDGE_MARGIN = 0.16           # fridge-side lane end <= fridge_y - this
    FRIDGE_Y_MAX = 0.20            # skin coverage of the wrist top degrades beyond +0.2
    FRIDGE_Y_MIN = 0.10
    COOKTOP_EDGE_Y = -0.80         # tall items stay on this side of the cooktop

    @property
    def cell(self) -> ReachCell:  # type: ignore[override]
        return PC_KITCHEN_REACH_CELLS[self.current_house_index % N_CELLS]

    # ---------------------------------------------------------------- items
    def _target_dims(self, cell_index: int) -> dict[str, Any]:
        r, rgba, lid, label = JARS[cell_index % len(JARS)]
        c = PC_KITCHEN_REACH_CELLS[cell_index]
        return dict(radius=r, height=c.jar_h, rgba=rgba, cap=lid, label=label)

    def _dyn_pool(self, cell_index: int) -> list[dict[str, Any]]:
        """Per-cell clutter pool: short items for the backsplash, tall ones for the lane end."""
        rng = np.random.default_rng(9500 + cell_index)
        pool: list[dict[str, Any]] = []
        for i in range(5):     # short: spice jars, shakers, a mug
            kind = ("spice", "spice", "shaker", "shaker", "mug")[i]
            if kind == "spice":
                r, h = float(rng.uniform(0.022, 0.027)), float(rng.uniform(0.08, 0.11))
            elif kind == "shaker":
                r, h = 0.02, float(rng.uniform(0.085, 0.10))
            else:
                r, h = 0.040, 0.095
            pool.append(dict(kind=kind, slot="short", r=r, h=h, half=(r, r, h / 2),
                             rgba=SHORT_COLORS[int(rng.integers(len(SHORT_COLORS)))]))
        pool.append(dict(kind="oil_bottle", slot="tall", r=0.030, h=0.285, half=(0.03, 0.03, 0.1425),
                         rgba=(0.55, 0.60, 0.20)))
        pool.append(dict(kind="soap_bottle", slot="tall", r=0.034, h=0.20, half=(0.034, 0.034, 0.10),
                         rgba=(0.20, 0.55, 0.75)))
        for i, it in enumerate(pool):
            it["name"] = f"pact_clutter_kr{i}"
        for name, (r, h) in XML_CANISTERS.items():
            pool.append(dict(kind="canister", slot="tall", r=r, h=h, half=(r, r, h / 2),
                             rgba=CERAMIC, name=name, xml=True))
        return pool

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        PickTaskSampler.add_auxiliary_objects(self, spec)
        k = self.current_house_index % N_CELLS
        self._house_cell = k
        self._dyn = self._dyn_pool(k)
        for it in self._dyn:
            if it.get("xml"):
                continue
            b = spec.worldbody.add_body(name=it["name"], mocap=True, pos=[-0.6, -0.6, PARK_Z])
            rgba = list(it["rgba"]) + [1.0]
            r, h = it["r"], it["h"]
            kw = dict(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
            if it["kind"] in ("spice", "shaker"):
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[r, h / 2 - 0.008, 0], pos=[0, 0, h / 2 - 0.008], rgba=rgba, **kw)
                cap = [0.82, 0.82, 0.84, 1.0] if it["kind"] == "shaker" else [0.15, 0.15, 0.16, 1.0]
                b.add_geom(name=it["name"] + "_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[r * 1.02, 0.008, 0], pos=[0, 0, h - 0.008], rgba=cap, **kw)
            elif it["kind"] == "mug":
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[r, h / 2, 0], pos=[0, 0, h / 2], rgba=rgba, **kw)
                b.add_geom(name=it["name"] + "_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[0.006, 0.012, 0.03], pos=[0, -(r + 0.006), h / 2], rgba=rgba, **kw)
            elif it["kind"] == "oil_bottle":
                hb = 0.20
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[r, hb / 2, 0], pos=[0, 0, hb / 2], rgba=rgba[:3] + [0.92], **kw)
                b.add_geom(name=it["name"] + "_neck", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[0.012, 0.035, 0], pos=[0, 0, hb + 0.035], rgba=rgba[:3] + [0.92], **kw)
                b.add_geom(name=it["name"] + "_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[0.014, 0.0075, 0], pos=[0, 0, h - 0.0075], rgba=[0.12, 0.12, 0.12, 1.0], **kw)
            else:  # soap bottle with a pump
                hb = 0.16
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[r, hb / 2, 0], pos=[0, 0, hb / 2], rgba=rgba, **kw)
                b.add_geom(name=it["name"] + "_pump", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[0.010, 0.02, 0], pos=[0, 0, hb + 0.02], rgba=[0.92, 0.92, 0.92, 1.0], **kw)
        self._add_task_bodies(spec, k)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        td = self._target_dims(cell_index)
        self._target_spec = td
        r, h = td["radius"], td["height"]
        ns = "cavity_obj_0/"
        b = spec.worldbody.add_body(name=ns + "kitchen_jar", pos=[-0.6, 0.6, PARK_Z + h])
        b.add_joint(name=ns + "kitchen_jar_free", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.01)
        lid_h = 0.012
        b.add_geom(name=ns + "kitchen_jar_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r, (h - lid_h) / 2, 0.0], pos=[0.0, 0.0, -lid_h / 2],
                   rgba=list(td["rgba"]) + [1.0], mass=0.25,
                   contype=8, conaffinity=15, condim=4, friction=[1.0, 0.02, 0.002], priority=1)
        b.add_geom(name=ns + "kitchen_jar_lid", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[r, lid_h / 2, 0.0], pos=[0.0, 0.0, h / 2 - lid_h / 2],
                   rgba=list(td["cap"]) + [1.0], mass=0.02,
                   contype=8, conaffinity=15, condim=4, friction=[1.0, 0.02, 0.002], priority=1)
        self._injected_obj_name = ns + "kitchen_jar"
        self._metadata_adder.update({
            self._injected_obj_name: {
                "category": "jar", "object_enum": "temp_object", "is_static": False,
                "boundingBox": {"x": 2 * r, "y": 2 * r, "z": h},
            }
        })
        mat = spec.worldbody.add_body(name="place_receptacle_coaster", mocap=True,
                                      pos=[-0.6, 0.2, PARK_Z])
        mat.add_geom(name="place_receptacle_coaster_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[COASTER_R, COASTER_HALF_T, 0.0], rgba=[0.66, 0.50, 0.32, 1.0],
                     contype=8, conaffinity=15, friction=[0.9, 0.9, 0.001])

    def _referral(self) -> str:
        return self._target_spec.get("label", "jar")

    def _task_class(self):
        return PCKitchenReachTask

    # ---------------------------------------------------------------- placement helpers
    def _park_all(self, env) -> None:
        for i, it in enumerate(self._dyn):
            self._mocap(env, it["name"], [-0.6 + 0.12 * (i % 6), -0.6 + 0.12 * (i // 6), PARK_Z])
        self._mocap(env, "place_receptacle", [-0.6, -0.9, PARK_Z])

    def _place_robot(self, env) -> None:
        jx, jy = self.BASE_JIT_XY
        self._cur_base_xyz = (float(np.random.uniform(-jx, jx)), float(np.random.uniform(-jy, jy)), BASE_Z)
        self._cur_base_yaw = float(np.random.uniform(-self.BASE_JIT_YAW, self.BASE_JIT_YAW))
        robot_view = env.current_robot.robot_view
        q = R.from_euler("z", self._cur_base_yaw).as_quat(scalar_first=True)
        from molmo_spaces.utils.pose import pos_quat_to_pose_mat

        robot_view.base.pose = pos_quat_to_pose_mat(np.array(self._cur_base_xyz), np.array(q))
        mujoco.mj_forward(env.current_model, env.current_data)

    @staticmethod
    def _disc_ok(x: float, y: float, r: float, placed: list) -> bool:
        return all(np.hypot(x - px, y - py) >= r + pr + 0.01 for (px, py, pr) in placed)

    def _cell_common(self) -> dict[str, Any]:
        c = self.cell
        return dict(cab_gap=c.gap, cab_depth=c.depth, cab_bottom_z=c.cab_bottom_z,
                    cab_front_x=c.cab_front_x, led_bottom_z=c.cab_bottom_z - LED_H,
                    fridge_y=c.fridge_y, jar_fridge_side=bool(c.jar_fridge_side),
                    z_grasp=c.z_grasp, z_carry=c.z_carry, arm_top_above_tcp=ARM_TOP,
                    arm_clear_carry_nominal=C_CARRY, arm_clear_grasp_nominal=C_CARRY + LIFT,
                    tool_pitch_deg=TOOL_PITCH_DEG, tool_az_deg=TOOL_AZ_DEG, slide=SLIDE)

    # ---------------------------------------------------------------- episode
    def _draw_episode(self, env) -> dict[str, Any] | None:
        c = self.cell
        td = self._target_spec
        r, h = td["radius"], td["height"]
        y_fr = float(np.random.uniform(self.FRIDGE_Y_MIN,
                                       min(self.FRIDGE_Y_MAX, c.fridge_y - self.FRIDGE_MARGIN)))
        y_ck = float(np.random.uniform(*self.COOK_Y))
        if y_fr - y_ck < self.LANE_MIN_SPAN:
            return None
        ya, yb = (y_fr, y_ck) if c.jar_fridge_side else (y_ck, y_fr)
        xa = float(np.random.uniform(*self.LANE_X))
        xb = float(np.random.uniform(*self.LANE_X))
        mat_top = COUNTER_Z + 2 * COASTER_HALF_T
        Rt = reach_tool_rotation()
        wps = reach_waypoints(c, xa, ya, xb, yb)
        poses = [_T(p, Rt) for (_n, p, _s) in wps if p is not None]
        if not self._ik_chain_ok(env, poses):
            return None
        # ---- clutter (mocap, resting on the counter) ----
        placed: list = [(xa, ya, r + 0.02), (xb, yb, COASTER_R + 0.01)]
        out: list = []
        x_lane = max(xa, xb)
        y_lo, y_hi = min(ya, yb), max(ya, yb)
        shorts = [i for i, it in enumerate(self._dyn) if it["slot"] == "short"]
        talls = [i for i, it in enumerate(self._dyn) if it["slot"] == "tall"]
        np.random.shuffle(shorts)
        np.random.shuffle(talls)
        n_short = int(np.random.randint(2, 5))
        for i in shorts:
            if n_short <= 0:
                break
            it = self._dyn[i]
            rr = it["r"]
            x_min = x_lane + GRIP_DX + CLR + rr     # open fingers reach +0.077 in x
            x_max = BACKSPLASH_X - rr - 0.003
            if x_max <= x_min:
                continue
            for _ in range(25):
                x = float(np.random.uniform(x_min, x_max))
                y = float(np.random.uniform(y_lo - 0.10, min(y_hi + 0.05, c.fridge_y - rr - 0.01)))
                if self._disc_ok(x, y, rr, placed):
                    placed.append((x, y, rr))
                    out.append((i, [x, y, COUNTER_Z + 0.0004]))
                    n_short -= 1
                    break
        n_tall = int(np.random.randint(2, 4))
        for i in talls:
            if n_tall <= 0:
                break
            it = self._dyn[i]
            rr = it["r"]
            # the slide in / out takes the TCP 6.1 cm beyond the lane end; the forearm reaches
            # 0.207 m further toward -y (high up, where the tall items' tops are)
            y_max = y_lo - SLIDE * 0.94 - LINK5_DY_NEG - CLR - rr
            y_min = self.COOKTOP_EDGE_Y + rr
            if y_max <= y_min:
                continue
            for _ in range(25):
                x = float(np.random.uniform(0.40 + rr, BACKSPLASH_X - rr - 0.003))
                y = float(np.random.uniform(y_min, y_max))
                if self._disc_ok(x, y, rr, placed):
                    placed.append((x, y, rr))
                    out.append((i, [x, y, COUNTER_Z + 0.0004]))
                    n_tall -= 1
                    break
        self._park_all(env)
        for i, pos in out:
            yaw = float(np.random.uniform(0, 2 * np.pi))
            self._mocap(env, self._dyn[i]["name"], pos, R.from_euler("z", yaw).as_quat(scalar_first=True))
        self._mocap(env, "place_receptacle_coaster", [xb, yb, COUNTER_Z + COASTER_HALF_T + 0.0003])
        quat = R.from_euler("z", float(np.random.uniform(0, 2 * np.pi))).as_quat(scalar_first=True)
        self._settle_target(env, [xa, ya, COUNTER_Z + h / 2 + 0.002], quat)
        mujoco.mj_forward(env.current_model, env.current_data)
        return dict(
            **self._cell_common(),
            can_radius=r, can_height=h, can_label=td["label"], jar_label=td["label"],
            start_xyz=[xa, ya, COUNTER_Z + h / 2], goal_xyz=[xb, yb, mat_top + h / 2],
            mat_xyz=[xb, yb, COUNTER_Z + COASTER_HALF_T], mat_top=mat_top, place_tol=PLACE_TOL,
            expert=[[n, p, s] for (n, p, s) in wps],
            neighbours=[[self._dyn[i]["name"], self._dyn[i]["kind"], pos,
                         list(self._dyn[i]["half"])] for i, pos in out],
            n_neighbours=len(out),
        )


# ---------------------------------------------------------------------------------------------
# Task: pc_shelf_hard contact audit + success rule, plus an exact per-link clearance audit
# ---------------------------------------------------------------------------------------------
SKINNED = ("fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6")
GRIPPER_SUB = 250       # hull vertices kept per gripper geom (arm links keep all 102-152)


class PCKitchenReachTask(PCShelfPnPHardTask):
    """Per robot body: minimum distance (m) to any hazard geom (everything except the target jar,
    the coaster, the floor and contactless decor) over the episode, every policy step, from the
    collision hulls' vertices to the exact box / cylinder / sphere surfaces. Stored in
    obs_scene['pc_clearance'] = {body: [dist, hazard_geom, step]}."""

    def _kr_setup(self, model) -> None:
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        pickup = model.body(self.config.task_config.pickup_obj_name).id
        rg, rv, owners = [], [], []
        hz = []
        for g in range(model.ngeom):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            b = int(model.geom_bodyid[g])
            root = int(model.body_rootid[b])
            bname = model.body(b).name or ""
            if root == robot_root:
                short = bname.split("/", 1)[-1] if bname.startswith("robot_0/") else bname
                if short.startswith("gripper/"):
                    key = "gripper"
                elif short.startswith("fr3_link") and short != "fr3_link0":
                    key = short
                else:
                    continue
                if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                    continue
                mid = int(model.geom_dataid[g])
                ga = int(model.mesh_graphadr[mid])
                adr = int(model.mesh_vertadr[mid])
                if ga >= 0:
                    nv = int(model.mesh_graph[ga])
                    ids = np.asarray(model.mesh_graph[ga + 2 + nv: ga + 2 + 2 * nv], dtype=int)
                    v = np.asarray(model.mesh_vert[adr + ids], dtype=float)
                else:
                    v = np.asarray(model.mesh_vert[adr: adr + int(model.mesh_vertnum[mid])], dtype=float)
                if key == "gripper" and len(v) > GRIPPER_SUB:
                    v = v[:: int(np.ceil(len(v) / GRIPPER_SUB))]
                rg.append(g)
                rv.append(v)
                owners += [key] * len(v)
                continue
            if root == pickup or int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            name = model.geom(g).name or bname
            if name.startswith("place_receptacle") or bname.startswith("place_receptacle"):
                continue
            if int(model.geom_type[g]) not in (int(mujoco.mjtGeom.mjGEOM_BOX),
                                               int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                                               int(mujoco.mjtGeom.mjGEOM_SPHERE)):
                continue
            hz.append((g, name, model.body(b).name or ""))
        self._kr = dict(rg=rg, rv=rv, owners=np.asarray(owners), hz=hz, best={}, keys=sorted(set(owners)))

    def _kr_step(self, data) -> None:
        model = data.model
        if getattr(self, "_kr", None) is None or self.episode_step_count == 0:
            self._kr_setup(model)
        kr = self._kr
        if not kr["rg"]:
            return
        P = np.concatenate([data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T
                            for g, v in zip(kr["rg"], kr["rv"])], axis=0)
        lo, hi = P.min(0) - 0.30, P.max(0) + 0.30
        own = kr["owners"]
        step = int(self.episode_step_count)
        for g, name, _bname in kr["hz"]:
            c = data.geom_xpos[g]
            rb = float(model.geom_rbound[g])
            if np.any(c + rb < lo) or np.any(c - rb > hi):
                continue
            Rm = data.geom_xmat[g].reshape(3, 3)
            loc = (P - c) @ Rm
            sz = model.geom_size[g]
            t = int(model.geom_type[g])
            if t == int(mujoco.mjtGeom.mjGEOM_BOX):
                q = np.abs(loc) - sz
                dist = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
            elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                q = np.stack([np.linalg.norm(loc[:, :2], axis=1) - sz[0], np.abs(loc[:, 2]) - sz[1]], 1)
                dist = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
            else:
                dist = np.linalg.norm(loc, axis=1) - sz[0]
            if float(dist.min()) > 0.30:
                continue
            for key in kr["keys"]:
                dm = float(dist[own == key].min())
                prev = kr["best"].get(key)
                if prev is None or dm < prev[0]:
                    kr["best"][key] = [round(dm, 4), name, step]

    def reach_clearance(self) -> dict[str, Any] | None:
        kr = getattr(self, "_kr", None)
        if kr is None:
            return None
        return {k: list(v) for k, v in sorted(kr["best"].items())}

    def _accumulate_obstacle_diag(self, data, pickup_obj) -> None:
        super()._accumulate_obstacle_diag(data, pickup_obj)
        try:
            self._kr_step(data)
        except Exception as e:  # pragma: no cover - a diagnostic must never break a rollout
            log.debug(f"[PCKitchenReach] clearance audit failed: {e!r}")

    def get_obs_scene(self) -> dict[str, Any]:
        d = super().get_obs_scene()
        cl = self.reach_clearance()
        if cl is not None:
            sp = dict(d.get("scene_params") or {})
            sp["pc_clearance"] = _jsonable(cl)
            d["scene_params"] = sp
            d["pc_clearance"] = _jsonable(cl)
        return d

    def _maybe_log_obstacle_diag(self, success: bool) -> None:
        already = getattr(self, "_obstacle_diag_logged", True)
        super()._maybe_log_obstacle_diag(success)
        if not already and getattr(self, "_obstacle_diag_logged", False):
            cl = self.reach_clearance() or {}
            sk = [v[0] for k, v in cl.items() if k in ("fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6")]
            gr = [v[0] for k, v in cl.items() if k in ("fr3_link7", "gripper")]
            log.info("[PCKitchenReach] clearance skinned(l3-6) min=%s gripper/l7 min=%s | %s",
                     f"{min(sk):.4f}" if sk else "?", f"{min(gr):.4f}" if gr else "?",
                     ", ".join(f"{k}={v[0]:.4f}({v[1]}@{v[2]})" for k, v in
                               sorted(cl.items(), key=lambda kv: kv[1][0])))


# ---------------------------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------------------------
class PCKitchenReachExpert(PCShelfPnPHardExpert):
    """pc_shelf_pnp_hard expert (hazard audit + collection gate) with the top-down hand and a
    slow lateral carry under the cabinets."""

    SPEEDS = dict(PCShelfPnPHardExpert.SPEEDS, carry=0.07)

    def _rot(self, sp) -> np.ndarray:
        return reach_tool_rotation()


class PCKitchenReachExpertConfig(PCShelfPnPHardExpertConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCKitchenReachExpert


__all__ = [
    "PC_KITCHEN_REACH_ENV_VERSION", "PC_KITCHEN_REACH_CELLS", "PC_KITCHEN_REACH_SCENES",
    "PC_KITCHEN_REACH_READY_QPOS", "PCKitchenReachSampler", "PCKitchenReachTask",
    "PCKitchenReachExpert", "PCKitchenReachExpertConfig", "write_pc_kitchen_reach_scenes",
    "reach_tool_rotation", "N_CELLS",
]


if __name__ == "__main__":
    import sys

    if "--write-scenes" in sys.argv:
        for p in write_pc_kitchen_reach_scenes():
            print(p)
