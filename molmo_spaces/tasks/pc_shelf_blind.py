"""``pc_shelf_blind``: retrieve a can from deep inside a tall pantry bay whose hidden under-shelf
basket threatens the SKIN-COVERED wrist knuckle and forearm, never the gripper.

Why this suite exists (2026-10-14, ICLR rebuttal)
--------------------------------------------------
In the earlier pc_* suites every hazard was visible to the cameras and the hazardous contacts
were made by the unskinned gripper / fr3_link7 (reports/2026-10-13/contact_parts.png), so vision
alone could plan around them. Here the scene is built so that the ONLY thing that differs between
episode configurations is a real object that neither camera can see, sitting where only the
skinned links (fr3_link5 / fr3_link6, also fr3_link4) pass.

Camera analysis (measured on the pc_shelf room, see the acceptance scripts)
-----------------------------------------------------------------------------
* exo_camera_1 is robot mounted behind/right/above the robot (world ~(-1.05, -0.54, 1.65)),
  looks at (0.55, 0, 0.80) (pitch -27 deg) and renders at fovy 45 (624x352). It sees all open
  space around the arm (both side units, the sideboard, the arm itself). It cannot see under a
  board: a point hanging h below the underside of a shelf board, d behind its front edge, is
  hidden when h < d x tan(viewing angle) (the "lintel shadow" of the board's front edge): ~0.5 d
  for a board at 1.0 m, only ~0.3 d for the 1.16-1.22 m board above this suite's tall bay.
* wrist_camera rides on the gripper and looks along the approach axis (FOV 56.7 x ~87 deg). With
  the hand pointing down (this suite, also at the start pose) it never sees anything above the
  hand.
So the underside of the board above the work bay, 20+ cm behind its front edge, is blind for
both cameras at every step (for a 5.5 cm tall object); a top-down hand puts the skinned wrist knuckle / forearm (up to
0.344 m above the TCP; fr3_link7 tops out at +0.21 m, the gripper at +0.16 m) right under it.

Scene
-----
The ``pc_shelf`` storage room (closed room, three-bay bookcase against the far wall, side
bookcase left, sideboard right, low cabinet behind; cameras, skin, timing unchanged), generated
by :func:`molmo_spaces.tasks.pc_shelf.build_scene_xml` with this suite's eight cells
(:data:`PC_SHELF_BLIND_CELLS`): the bookcase stands closer (front face x = 0.38-0.40 m, so a
top-down hand reaches 26.5-28.5 cm deep), is deeper (0.40 m case, pc_shelf 0.34 m: the open
hand's far finger stays >= 5 cm off the back panel) and its lower work bay is a TALL pantry bay (clear height
0.50-0.523 m, sized per cell so the skinned wrist top passes ~2.5 cm under the board above while
the hand carries the can over the board). Upper work bay, decor bays, finish: as pc_shelf.

Hidden hazard (``pact_intrusion_basket``, scored by the evaluator as ``hazard_bar``)
-------------------------------------------------------------------------------------
An under-shelf wire basket (0.10-0.18 m wide, 0.10-0.12 m deep, 5.3-5.8 cm tall) mounted flush
under the board above the work bay, its front 20-21 cm behind the board's front edge (inside the
exo camera's lintel shadow: at the bay ceiling height the camera looks down at ~15 deg, so the
shadow is ~0.3 x depth = 6 cm), on the LEFT or the RIGHT of the can's lane (50/50, drawn
independently of the can, the mat, the neighbours, the cell and the start-pose noise); its inner
edge sits 8.5-9.5 cm (left) / 2.5-3.5 cm (right) beside the lane centre (the FR3 forearm leans
to +y). The arm's free space under the board is ~2.5 cm, the basket hangs 5.3-5.8 cm: a
wrist/forearm that passes under it collides.

Task + expert (scripted TCP segments)
-------------------------------------
Pick the can (``cavity_obj_0/shelf_can``) standing 26.5-28.5 cm deep (at the back) of the bay's
board and set it on the coaster mat (``place_receptacle_mat``) 13-15 cm deep on the same board (pc_shelf_pnp success
rule). The hand points DOWN, fingers closing along x (into the bay). Start pose: hand down in
front of the bookcase (``PC_SHELF_BLIND_READY_QPOS``). Expert: move to a pre-insert pose in front
of the bay above the can's height -> slide 12 cm into the bay, untilted -> dwell 0.6 s (all of
this is common to every configuration; the wrist skin now faces the basket from 6-11 cm) -> TILT
the hand 20 deg about the x axis, fingers leaning toward the basket's side, which swings the
wrist knuckle and forearm ~9 cm to the OTHER side (the only plan difference) -> slide in above
the can -> lower -> close -> lift -> carry to above the mat -> lower -> open -> lift -> untilt ->
back out -> start pose. A blind (camera-only) plan must guess: no tilt passes under
both basket positions, either fixed tilt passes under one of them (see the acceptance checks).

Acceptance checks (2026-10-14, scripts/audit_pc_shelf_blind.py, 8 cells, 80 Monte-Carlo draws)
-----------------------------------------------------------------------------------------------
* never seen: exo / wrist max pixel change from removing the basket = 0 / <= 1 (of 255) at t=0
  and at every 2nd step of the expert rollouts; t=0 frames identical for basket left/right/none.
* skin sees it before the plans split (hand 12 cm in, dwell): link6_sensor_3 reads a LEFT basket
  at 6-7 cm, link6_sensor_0 / _4 a RIGHT basket at 7-11 cm, 17.5-19 cm otherwise.
* plans (kinematic replay, fixed hand tilt -20/-10/0/+10/+20 deg): skinned-link hit rate
  51/49/94/49/49 %; expert (tilt away) never hits (skinned min 1.4 cm, median 4.4 cm); the
  opposite tilt always hits; fr3_link7 + gripper >= 8.6 cm from the basket in every plan.
* expert success 8/8 (noise off) and 8/8 (collection recipe: noise on, gate on); skin valid
  99.93 %.

Collection-only additions (switched on by the collection configs, never by the eval config):
DART-style executed-action noise and the contact-tolerant save gate of ``pc_shelf_hard``.
Per-link clearance to the basket is recorded in ``obs_scene['pc_clearance']`` (body ->
[min distance m, 'pact_intrusion_basket', step]; scripts/audit_arm_clearance.py reads it).
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

from molmo_spaces.tasks import pc_shelf as _base
from molmo_spaces.tasks.pc_shelf import (
    D_INT,
    MAT_HALF,
    MAT_HALF_T,
    N_CELLS,
    SCENE_DIR,
    PCShelfPnPSampler,
    ShelfCell,
    _PCSeg,
    _T,
    _jsonable,
)
from molmo_spaces.tasks.pc_shelf_hard import (
    PCShelfPnPHardExpert,
    PCShelfPnPHardExpertConfig,
    PCShelfPnPHardTask,
)
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    GripperAction,
)

log = logging.getLogger(__name__)

PC_SHELF_BLIND_ENV_VERSION = "pc_shelf_blind_v1"

# Measured on model_hybrid.xml (link0 at z = 0.35), hand pointing down, tilt 0 / +-20 deg:
ARM_TOP_TILT = 0.333      # top of the skinned wrist knuckle / forearm above the TCP (tilted)
ARM_TOP_STRAIGHT = 0.344  # ... with the hand straight down
CAN_HS = (0.122, 0.118, 0.130, 0.112, 0.125, 0.135, 0.120, 0.115)   # PCShelfPnPSampler.CANS
INS_CLEAR = 0.030         # TCP above the can's top while sliding in / carrying (fingertips ~2 cm clear)
GRASP_BELOW_TOP = 0.025   # TCP below the can's top at the grasp
CEIL_CLEAR = 0.025        # skinned top -> board underside while carrying (tilted hand)
TILT_DEG = 20.0
X_PRE = 0.07              # pre-insert TCP stand-off in front of the bookcase face
D_DIV = 0.12              # TCP depth of the (common, untilted) partial insertion where the tilt starts
DWELL_S = 0.6             # pause there before tilting: the policy's 50-step query lands on the skin view
CASE_D_BLIND = 0.40       # deeper pantry bookcase (pc_shelf: 0.34): the open hand's far finger stays
D_INT_BLIND = CASE_D_BLIND - _base.BACK_T   # >= 5 cm off the back panel at the deepest can


def _bay_lo(k: int) -> float:
    return round(CAN_HS[k] + INS_CLEAR + ARM_TOP_TILT + CEIL_CLEAR, 3)


# Fields: x0, b_lo, bay_lo (TALL pantry bay), bay_hi, bay_w, pnp_pick_upper, push_upper, wood.
PC_SHELF_BLIND_CELLS: tuple[ShelfCell, ...] = (
    ShelfCell(0.39, 0.68, _bay_lo(0), 0.29, 0.50, False, False, (0.72, 0.58, 0.40)),  # light oak
    ShelfCell(0.40, 0.66, _bay_lo(1), 0.28, 0.48, False, False, (0.42, 0.28, 0.18)),  # walnut
    ShelfCell(0.38, 0.70, _bay_lo(2), 0.30, 0.54, False, False, (0.90, 0.89, 0.86)),  # white laminate
    ShelfCell(0.39, 0.66, _bay_lo(3), 0.28, 0.48, False, False, (0.80, 0.70, 0.52)),  # birch
    ShelfCell(0.40, 0.68, _bay_lo(4), 0.29, 0.52, False, False, (0.62, 0.62, 0.60)),  # grey laminate
    ShelfCell(0.38, 0.66, _bay_lo(5), 0.28, 0.50, False, False, (0.35, 0.24, 0.15)),  # dark oak
    ShelfCell(0.39, 0.70, _bay_lo(6), 0.30, 0.50, False, False, (0.78, 0.62, 0.42)),  # maple
    ShelfCell(0.40, 0.68, _bay_lo(7), 0.29, 0.50, False, False, (0.24, 0.22, 0.21)),  # black ash
)

PC_SHELF_BLIND_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_shelf_blind_c{k}.xml") for k in range(N_CELLS)
)


def build_blind_scene_xml(cell_index: int) -> str:
    saved = (_base.PC_SHELF_CELLS, _base.CASE_D, _base.D_INT)
    try:
        _base.PC_SHELF_CELLS = PC_SHELF_BLIND_CELLS
        _base.CASE_D, _base.D_INT = CASE_D_BLIND, D_INT_BLIND
        xml = _base.build_scene_xml(cell_index)
    finally:
        _base.PC_SHELF_CELLS, _base.CASE_D, _base.D_INT = saved
    xml = xml.replace(f'<mujoco model="pc_shelf_c{cell_index}">',
                      f'<mujoco model="pc_shelf_blind_c{cell_index}">', 1)
    return xml.replace("Generated by molmo_spaces.tasks.pc_shelf.write_pc_shelf_scenes",
                       "Generated by molmo_spaces.tasks.pc_shelf_blind.write_pc_shelf_blind_scenes", 1)


def write_pc_shelf_blind_scenes() -> list[str]:
    """(Re)generate the eight blind-cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_SHELF_BLIND_SCENES[k])
        p.write_text(build_blind_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------
# Hand orientation + start pose
# --------------------------------------------------------------------------------------------
def blind_tool_rotation(tilt_deg: float = 0.0) -> np.ndarray:
    """Hand pointing down, fingers closing along world x, wrist camera on the -y side; tilted by
    ``tilt_deg`` about world x (positive: approach leans toward +y, wrist swings toward -y)."""
    Rd = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    t = np.deg2rad(tilt_deg)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    return Rx @ Rd


READY_TCP = (0.30, 0.0, 0.95)
# IK of READY_TCP x blind_tool_rotation(0) (elbow-up family, solved from the pc_shelf ready pose)
PC_SHELF_BLIND_READY_QPOS: list[float] = [0.0, -0.6051, 0.0, -1.7863, 0.0, 1.1812, -1.5707]


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
class PCShelfBlindSampler(PCShelfPnPSampler):
    """Can deep in the tall bay, mat at the bay front, hidden basket left or right."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_SHELF_BLIND_ENV_VERSION
    SUITE = "blind"
    N_DYN = 14
    CAN_D = (0.265, 0.285)        # can centre depth behind the bookcase face (back of the bay)
    CAN_Y = (-0.03, 0.03)
    MAT_D = (0.130, 0.150)
    MAT_DY = 0.02
    # basket inner edge beside the lane centre, per side: the FR3 forearm leans toward +y, so a
    # left basket must start further out for the away-tilted wrist to clear it by >= 3.5 cm
    # (kinematic sweep 2026-10-14: away plan 3.7-5.3 cm, no-tilt and toward plans collide)
    BASKET_GAP = {1: (0.085, 0.095), -1: (0.025, 0.035)}
    BASKET_W = (0.15, 0.18)
    BASKET_D0 = (0.200, 0.210)    # basket front behind the board's front edge
    BASKET_DEPTH = (0.10, 0.12)
    BASKET_H = (0.053, 0.058)
    MAX_DRAWS = 40

    @property
    def cell(self) -> ShelfCell:
        return PC_SHELF_BLIND_CELLS[self.current_house_index % N_CELLS]

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        super()._add_task_bodies(spec, cell_index)
        # the hidden under-shelf basket (mocap, posed per episode; parked outside the room)
        b = spec.worldbody.add_body(name="pact_intrusion_basket", mocap=True, pos=[-3.0, -1.6, 0.3])
        b.add_geom(name="pact_intrusion_basket_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.07, 0.08, 0.032], rgba=[0.72, 0.73, 0.76, 1.0],
                   contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])

    def _basket_resize(self, env, half) -> None:
        m = env.current_model
        gid = m.geom("pact_intrusion_basket_g").id
        m.geom_size[gid] = np.asarray(half, dtype=float)
        # keep the broadphase bound valid for the new size
        m.geom_rbound[gid] = float(np.linalg.norm(half))
        m.geom_aabb[gid] = np.array([0.0, 0.0, 0.0, *half], dtype=float)

    def _draw_episode(self, env) -> dict[str, Any] | None:
        c = self.cell
        k = self.current_house_index % N_CELLS
        td = self._target_spec
        r, h = td["radius"], td["height"]
        z_b = c.b_lo
        ceil = c.b_lo + c.bay_lo
        dc = float(np.random.uniform(*self.CAN_D))
        yc = float(np.random.uniform(*self.CAN_Y))
        dm = float(np.random.uniform(*self.MAT_D))
        ym = yc + float(np.random.uniform(-self.MAT_DY, self.MAT_DY))
        xc, xm = c.x0 + dc, c.x0 + dm
        yaw = float(np.random.uniform(0, 2 * np.pi))
        mat_top = z_b + 2 * MAT_HALF_T
        # hidden basket: side drawn independently of everything visible
        side = 1 if np.random.random() < 0.5 else -1          # +1 = left (+y), -1 = right (-y)
        gap = float(np.random.uniform(*self.BASKET_GAP[side]))
        bw = float(np.random.uniform(*self.BASKET_W))
        y_in = yc + side * gap
        y_out = y_in + side * bw
        lim = c.bay_w / 2 - 0.008
        y_out = float(np.clip(y_out, -lim, lim))
        if abs(y_out - y_in) < 0.10:
            return None
        d0 = float(np.random.uniform(*self.BASKET_D0))
        dd = float(np.random.uniform(*self.BASKET_DEPTH))
        dd = min(dd, D_INT_BLIND - 0.008 - d0)
        bh = float(np.random.uniform(*self.BASKET_H))
        basket_center = [c.x0 + d0 + dd / 2, (y_in + y_out) / 2, ceil - bh / 2 - 0.0005]
        basket_half = [dd / 2, abs(y_out - y_in) / 2, bh / 2]
        tilt = TILT_DEG * side       # approach leans toward the basket side -> wrist swings away
        # expert waypoints (TCP world positions + hand tilt)
        z_hi = z_b + h + INS_CLEAR
        z_g = z_b + h - GRASP_BELOW_TOP
        z_p = mat_top + h - GRASP_BELOW_TOP + 0.004
        x_pre = c.x0 - X_PRE
        x_div = c.x0 + D_DIV
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
        # neighbours: a few items on the bay board away from the lane, a few in the upper bay
        avail = list(range(len(self._dyn)))
        out: list = []
        bay = self._bay(z_b, min(c.bay_lo, 0.30))
        lane = (c.x0 + D_INT / 2, (yc + ym) / 2, D_INT / 2 + 0.05, 0.14)
        can_fp = (xc, yc, r + 0.03, r + 0.03)
        mat_fp = (xm, ym, MAT_HALF + 0.02, MAT_HALF + 0.02)
        placed: list = []
        for _ in range(3):
            if np.random.random() < 0.7:
                self._random_item(avail, placed, [lane, can_fp, mat_fp], bay, out)
        bay_hi = self._bay(c.b_hi, c.bay_hi)
        placed_hi: list = []
        for _ in range(3):
            self._random_item(avail, placed_hi, [], bay_hi, out)
        self._apply_layout(env, out)
        self._mocap(env, "place_receptacle_mat", [xm, ym, z_b + MAT_HALF_T + 0.0003])
        self._basket_resize(env, basket_half)
        self._mocap(env, "pact_intrusion_basket", basket_center)
        quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
        self._settle_target(env, [xc, yc, z_b + h / 2 + 0.002], quat)
        mujoco.mj_forward(env.current_model, env.current_data)
        return dict(
            **self._cell_common(),
            pick_board_z=z_b, place_board_z=z_b, pick_upper=False,
            can_radius=r, can_height=h, can_label=td["label"],
            start_xyz=[xc, yc, z_b + h / 2], goal_xyz=[xm, ym, mat_top + h / 2],
            mat_xyz=[xm, ym, z_b + MAT_HALF_T], mat_top=mat_top, place_tol=self.PLACE_TOL,
            expert=[[n, p, s, t] for (n, p, s, t) in wps],
            neighbours=[[self._dyn[i]["name"], self._dyn[i]["kind"], pos,
                         list(self._dyn[i]["half"])] for i, pos in out],
            n_neighbours=len(out),
            # evaluator contract (eval_act_place.py writes hidden_hazard_present / _y / _side into
            # every episode record); the basket is present in every episode, its side is the draw
            hidden_hazard=dict(present=True, side="left" if side > 0 else "right",
                               y=float(basket_center[1]), x=float(basket_center[0]),
                               z_bottom=float(ceil - bh - 0.0005), body="pact_intrusion_basket"),
            blind=dict(side=int(side), side_name="left" if side > 0 else "right",
                       tilt_deg=float(tilt), basket_center=basket_center,
                       basket_half=basket_half, basket_front_depth=d0, basket_gap=gap,
                       ceiling_z=ceil, z_carry=z_hi, cell=int(k)),
        )

    def _referral(self) -> str:
        return self._target_spec.get("label", "can")

    def _task_class(self):
        return PCShelfBlindTask


# --------------------------------------------------------------------------------------------
# Task: pc_shelf_pnp_hard success + hazard audit, plus per-link clearance to the basket
# --------------------------------------------------------------------------------------------
HAZ_BODY = "pact_intrusion_basket"
CLEAR_BODIES = ("fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "gripper")


def robot_vertex_sets(model, robot_root: int, keys=CLEAR_BODIES) -> dict[str, list]:
    """Collision-geometry points of the robot bodies of interest: every mesh vertex (convex hulls,
    <= 400 per geom) or the 8 corners + centre of a primitive's bounding box."""
    out: dict[str, list] = {}
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        b = int(model.geom_bodyid[g])
        if int(model.body_rootid[b]) != robot_root:
            continue
        bn = model.body(b).name.split("/", 1)[-1]
        key = "gripper" if bn.startswith("gripper") else bn
        if key not in keys:
            continue
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[g])
            adr, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            v = np.asarray(model.mesh_vert[adr:adr + n], dtype=float)
            if len(v) > 400:
                v = v[:: int(np.ceil(len(v) / 400))]
        else:
            c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
            v = np.array([c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1)
                          for sz in (-1, 1)] + [c])
        out.setdefault(key, []).append((g, v))
    return out


def box_distance(data, sets: dict[str, list], hg: int) -> dict[str, float]:
    """Min distance (m, 0 when touching/inside) of each body's points to the box geom ``hg``."""
    C = data.geom_xpos[hg]
    Rm = data.geom_xmat[hg].reshape(3, 3)
    half = data.model.geom_size[hg]
    out = {}
    for key, L in sets.items():
        P = np.concatenate([data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T for g, v in L], 0)
        q = np.abs((P - C) @ Rm) - half
        out[key] = float(np.linalg.norm(np.maximum(q, 0.0), axis=1).min())
    return out


class PCShelfBlindTask(PCShelfPnPHardTask):
    CLEAR_EVERY_BLIND = 2

    def _blind_setup(self, model) -> None:
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        self._bclr = dict(hg=model.geom("pact_intrusion_basket_g").id,
                          sets=robot_vertex_sets(model, robot_root), best={})

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

    def blind_clearance(self) -> dict[str, Any] | None:
        bc = getattr(self, "_bclr", None)
        if bc is None:
            return None
        return {k: [round(v[0], 4), v[1], v[2]] for k, v in sorted(bc["best"].items())}

    def _accumulate_obstacle_diag(self, data, pickup_obj) -> None:
        super()._accumulate_obstacle_diag(data, pickup_obj)
        try:
            self._blind_step(data)
        except Exception as e:  # pragma: no cover - a diagnostic must never break a rollout
            log.debug(f"[PCShelfBlind] clearance diagnostic failed: {e!r}")

    def get_obs_scene(self) -> dict[str, Any]:
        d = super().get_obs_scene()
        cl = self.blind_clearance()
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
            cl = self.blind_clearance() or {}
            sp = getattr(self, "scene_params", {}) or {}
            log.info("[PCShelfBlind] basket side=%s clearance %s", (sp.get("blind") or {}).get("side_name"),
                     ", ".join(f"{k}={v[0]:.4f}@{v[2]}" for k, v in sorted(cl.items(), key=lambda kv: kv[1][0])))


# --------------------------------------------------------------------------------------------
# Expert
# --------------------------------------------------------------------------------------------
@dataclass
class _BlindSeg(_PCSeg):
    """pc_shelf segment with an optional minimum duration (a dwell holds the pose)."""

    min_dur: float = 0.0

    @property
    def duration(self) -> float:
        return float(max(_PCSeg.duration.fget(self), self.min_dur))


class PCShelfBlindExpert(PCShelfPnPHardExpert):
    """Replays ``scene_params['expert']`` = [name, tcp, speed, tilt_deg] (per-waypoint hand tilt)."""

    SPEEDS = dict(PCShelfPnPHardExpert.SPEEDS, carry=0.07, tilt=0.05)
    TILT_ANG_SPEED = 0.45

    def _compute_trajectory(self):
        sp = getattr(self.task, "scene_params", {}) or {}
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
            seg = _BlindSeg(name=name, start_pose=cur, end_pose=end,
                            speed=float(self.SPEEDS.get(speed, 0.08)),
                            ang_speed=self.TILT_ANG_SPEED if speed == "tilt" else 0.9,
                            min_dur=DWELL_S if speed == "dwell" else 0.0)
            segs.append(seg)
            cur = end
        if segs:
            prims.append(self._seq(segs, holding))
        return prims


class PCShelfBlindExpertConfig(PCShelfPnPHardExpertConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCShelfBlindExpert


__all__ = [
    "PC_SHELF_BLIND_ENV_VERSION", "PC_SHELF_BLIND_CELLS", "PC_SHELF_BLIND_SCENES",
    "PC_SHELF_BLIND_READY_QPOS", "PCShelfBlindSampler", "PCShelfBlindTask", "PCShelfBlindExpert",
    "PCShelfBlindExpertConfig", "write_pc_shelf_blind_scenes", "build_blind_scene_xml",
    "blind_tool_rotation", "READY_TCP",
]
