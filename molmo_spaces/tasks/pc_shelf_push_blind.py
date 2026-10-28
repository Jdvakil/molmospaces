"""``pc_shelf_push_blind``: the NON-PREHENSILE version of ``pc_shelf_blind`` -- push a box from
the front of a pantry bay to its back while a hidden under-shelf basket threatens the SKIN-COVERED
wrist knuckle / forearm, never the gripper.

Why this suite exists (2026-10-14, ICLR rebuttal)
--------------------------------------------------
``pc_shelf_blind`` (pick-and-place) showed the gap the paper claims: a hazard that neither camera
can see, on the links only the skin covers. The user also asked for non-prehensile tasks. Here the
robot never grasps: the fingers are closed for the whole episode and the closed hand pushes a box
along the work board, deep into the bay, under the same hidden basket.

Scene
-----
The ``pc_shelf_blind`` pantry bookcase (same room, cameras, skin, timing, case depth 0.40 m, bay
width, finish and -- per cell -- the SAME absolute height of the board above the work bay, so the
camera geometry around the hidden basket is unchanged), with the lower work board RAISED by
(can height + 1 cm) to 0.762-0.825 m: the push TCP height (board + 4 cm) equals the blind suite's
carry height, so the arm posture under the board above (skinned wrist top ~2.5 cm under it while
tilted) is the measured one. Work-bay clear height 0.398 m. Scenes
``custom_scenes/pc_shelf_push_blind_c<k>.xml`` (:func:`write_pc_shelf_push_blind_scenes`).

Hidden hazard (``pact_intrusion_basket``, scored by the evaluator as ``hazard_bar``)
-------------------------------------------------------------------------------------
As ``pc_shelf_blind`` (:class:`PCShelfBlindSampler` constants): an under-shelf wire basket
5.3-5.8 cm tall mounted flush under the board above the work bay, front 20-21 cm behind the
board's front edge (inside the exo camera's lintel shadow), LEFT or RIGHT of the push lane (50/50,
drawn independently of the box, the goal, the neighbours, the cell and the start-pose noise). The
two positions are closer together than in pc_shelf_blind: the left basket's inner edge 7.8-8.6 cm
left of the lane centre, the right basket's inner edge 1.6-2.4 cm LEFT of it (the basket reaches
over the lane, 30 cm above the hand), so that no fixed hand tilt threads between them (a -5 deg
tilt did with the pc_shelf_blind offsets).

Task + expert (scripted TCP segments, hand pointing down, fingers CLOSED)
-------------------------------------------------------------------------
Push the flat box (``cavity_obj_0/shelf_box``, 7-9 cm deep, 4-5 cm tall) from the front of the bay (front face
2.5-4.5 cm behind the bookcase face) straight back onto the label decal (``pc_goal_marker``) by
the back panel (box back face 1.2-2.0 cm off it). Success: box centre within 3.5 cm of the label
centre, upright, on the board, never lifted (< 1.5 cm), never pinched between both fingers, and
the push over (hand off the box, box at rest), so an end-at-success eval cannot stop mid-push.
Expert: pre-push pose in front of the bay at the push height -> close the fingers -> approach the
box -> push it untilted until the TCP is 12 cm inside the bay -> dwell 1.5 s (pushing paused; the
wrist skin faces the basket from 6-11 cm; the hand gets there ~80 policy steps in, so the pause
spans the policy's step-100 chunk query) -> TILT the hand 20 deg about the push axis, fingers
leaning toward the basket's side, which swings the wrist knuckle / forearm ~9 cm to the OTHER side
(the only plan difference) -> push on to the label -> back out tilted to 12 cm -> untilt -> out
-> start pose. A blind (camera-only) plan must guess the side.

Acceptance checks (2026-10-14, scripts/audit_pc_shelf_push_blind.py, 8 cells + 66 Monte-Carlo draws)
-------------------------------------------------------------------------------------------------
* never seen: exo / wrist max pixel change from removing the basket = 0 / 0 at t=0 and at the
  pause, <= 1 (of 255) along the expert rollouts.
* skin sees it at the pause (TCP 12 cm in, before the tilt): link6_sensor_3 reads a LEFT basket at
  7.3 cm, link6_sensor_0 / _4 a RIGHT basket at 8.2-8.9 / 6.5-7.6 cm, 17.8 / 19.1 cm otherwise.
* plans (kinematic replay of the expert path, every fixed hand tilt -20..+20 deg in 2.5 deg steps):
  each hits the basket (skinned link inside it) in 45-100 % of draws (side-balanced >= 50 %); the
  expert never (skinned min 1.6 cm), the opposite tilt always; fr3_link7 + gripper >= 8.5 cm from
  the basket in every plan, >= 8.7 cm in the rollouts.
* expert success 8/8 (noise off) and 8/8 (collection recipe: noise on, gate on).

Collection-only additions (switched on by the collection configs, never by the eval config):
DART-style executed-action noise, the contact-tolerant save gate of ``pc_shelf_hard`` and one skin
reading per policy step (see pc_shelf_push_blind_configs.py). Per-link clearance to the basket is
recorded in ``obs_scene['pc_clearance']`` and the push record (max lift, pinch) in
``obs_scene['pc_push']``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.tasks import pc_shelf as _base
from molmo_spaces.tasks.pc_shelf import (
    D_INT,
    N_CELLS,
    SCENE_DIR,
    PCShelfPushSampler,
    ShelfCell,
    _T,
)
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    GripperAction,
)
from molmo_spaces.tasks.pc_shelf_blind import (
    CAN_HS,
    CASE_D_BLIND,
    D_DIV,
    D_INT_BLIND,
    INS_CLEAR,
    PC_SHELF_BLIND_CELLS,
    READY_TCP,
    TILT_DEG,
    X_PRE,
    PCShelfBlindExpert,
    PCShelfBlindExpertConfig,
    PCShelfBlindSampler,
    PCShelfBlindTask,
    _BlindSeg,
    blind_tool_rotation,
)

log = logging.getLogger(__name__)

PC_SHELF_PUSH_BLIND_ENV_VERSION = "pc_shelf_push_blind_v1"

Z_PUSH = 0.040        # TCP above the work board while pushing (closed fingertips ~3 cm above it)
FING_X = 0.019        # closed fingertips: front face ahead of the TCP along +x, TCP-2..+1 cm band (measured)
OVERPUSH = 0.006      # push the box centre this far past the label centre (it stops ~4 mm short)
LIFT_MAX = 0.015      # a box raised more than this off its board = picked up, not pushed
BOX_STILL = 0.01      # m/s: success needs the box at rest (and the hand off it)
DWELL_PUSH_S = 1.5    # pause at the divergence point (reached ~step 80): spans the step-100 query


def _push_cell(c: ShelfCell, k: int) -> ShelfCell:
    """pc_shelf_blind cell k with the work board raised to (blind carry TCP height - Z_PUSH); the
    board above the work bay keeps its height (same ceiling, same basket height, same cameras)."""
    z_tcp = c.b_lo + CAN_HS[k] + INS_CLEAR
    b_lo = round(z_tcp - Z_PUSH, 3)
    ceil = c.b_lo + c.bay_lo
    return ShelfCell(c.x0, b_lo, round(ceil - b_lo, 3), c.bay_hi, c.bay_w, False, False, c.wood)


PC_SHELF_PUSH_BLIND_CELLS: tuple[ShelfCell, ...] = tuple(
    _push_cell(c, k) for k, c in enumerate(PC_SHELF_BLIND_CELLS)
)

PC_SHELF_PUSH_BLIND_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_shelf_push_blind_c{k}.xml") for k in range(N_CELLS)
)


def build_push_blind_scene_xml(cell_index: int) -> str:
    saved = (_base.PC_SHELF_CELLS, _base.CASE_D, _base.D_INT)
    try:
        _base.PC_SHELF_CELLS = PC_SHELF_PUSH_BLIND_CELLS
        _base.CASE_D, _base.D_INT = CASE_D_BLIND, D_INT_BLIND
        xml = _base.build_scene_xml(cell_index)
    finally:
        _base.PC_SHELF_CELLS, _base.CASE_D, _base.D_INT = saved
    xml = xml.replace(f'<mujoco model="pc_shelf_c{cell_index}">',
                      f'<mujoco model="pc_shelf_push_blind_c{cell_index}">', 1)
    return xml.replace(
        "Generated by molmo_spaces.tasks.pc_shelf.write_pc_shelf_scenes",
        "Generated by molmo_spaces.tasks.pc_shelf_push_blind.write_pc_shelf_push_blind_scenes", 1)


def write_pc_shelf_push_blind_scenes() -> list[str]:
    """(Re)generate the eight push-blind cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_SHELF_PUSH_BLIND_SCENES[k])
        p.write_text(build_push_blind_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
class PCShelfPushBlindSampler(PCShelfBlindSampler):
    """Box at the bay front, label by the back panel, hidden basket left or right of the lane."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_SHELF_PUSH_BLIND_ENV_VERSION
    SUITE = "push_blind"
    BOX_FRONT = (0.025, 0.045)    # box front face behind the bookcase face
    BOX_Y = (-0.03, 0.03)         # lane centre (as the can's lane in pc_shelf_blind)
    BOX_YAW = 0.04
    BACK_GAP = (0.012, 0.020)     # goal: box back face this far off the back panel
    PUSH_TOL = 0.035
    NEIGH_MAX_H = 0.16            # work-board neighbours: low boxes / jars only
    # Basket inner edge beside the lane centre, per side (+ = toward the basket's side). The two
    # positions sit closer together than in pc_shelf_blind (left 8.5-9.5, right 2.5-3.5 cm), the
    # RIGHT basket's inner edge even reaching 1.6-2.4 cm over the lane centre (it hangs 30 cm above
    # the hand, only the skinned wrist / forearm rise to it): with the blind values a fixed hand
    # tilt of -5 deg threaded between both positions (hit 4 % of scenes; kinematic scan
    # 2026-10-14 08:15). Now the gap between the two positions (6-7 cm) is narrower than the
    # skinned links' cross-section at any tilt (>= 7.8 cm), so every fixed tilt hits one of them,
    # while the expert's 20 deg tilt clears its basket laterally by >= 3.7 cm (left) / 4.6 cm (right).
    BASKET_GAP = {1: (0.078, 0.086), -1: (-0.024, -0.016)}
    LANE_HALF = 0.15              # work-board neighbours keep this far off the lane centre

    # Flat boxes (4.0-5.0 cm tall): the box top sits 0-1 cm above the push TCP, so only the closed
    # fingertips (narrow, 1.8-2.0 cm ahead of the TCP) touch it; the hand's slanted linkage above the
    # TCP (3-6 cm ahead) stays above the box, and the 20 deg tilt (pivot = TCP) barely moves the
    # contact point sideways (a taller box was pushed at its top edge by the linkage and drifted).
    BOXES = (  # half extents (x depth, y width, z height), rgba, label
        ((0.040, 0.048, 0.022), (0.85, 0.20, 0.15), "red tea box"),
        ((0.042, 0.052, 0.020), (0.20, 0.45, 0.75), "blue cookie box"),
        ((0.038, 0.055, 0.024), (0.95, 0.85, 0.30), "yellow cracker box"),
        ((0.044, 0.046, 0.021), (0.30, 0.60, 0.35), "green tea box"),
        ((0.040, 0.054, 0.023), (0.92, 0.92, 0.90), "white gift box"),
        ((0.043, 0.050, 0.020), (0.28, 0.33, 0.45), "slate box"),
        ((0.036, 0.050, 0.025), (0.95, 0.55, 0.20), "orange snack box"),
        ((0.041, 0.047, 0.022), (0.55, 0.40, 0.25), "brown box"),
    )

    @property
    def cell(self) -> ShelfCell:
        return PC_SHELF_PUSH_BLIND_CELLS[self.current_house_index % N_CELLS]

    def _target_dims(self, cell_index: int) -> dict[str, Any]:
        half, rgba, label = self.BOXES[cell_index % len(self.BOXES)]
        return dict(half=half, rgba=rgba, label=label)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        # the pc_shelf_push box + label decal (uses this class's _target_dims) ...
        PCShelfPushSampler._add_task_bodies(self, spec, cell_index)
        # ... and the hidden under-shelf basket of pc_shelf_blind (mocap, parked outside the room)
        b = spec.worldbody.add_body(name="pact_intrusion_basket", mocap=True, pos=[-3.0, -1.6, 0.3])
        b.add_geom(name="pact_intrusion_basket_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.07, 0.08, 0.032], rgba=[0.72, 0.73, 0.76, 1.0],
                   contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])

    def _draw_episode(self, env) -> dict[str, Any] | None:
        c = self.cell
        k = self.current_house_index % N_CELLS
        td = self._target_spec
        hx, hy, hz = td["half"]
        z_b = c.b_lo
        ceil = c.b_lo + c.bay_lo
        front = float(np.random.uniform(*self.BOX_FRONT))
        ys = float(np.random.uniform(*self.BOX_Y))
        yaw = float(np.random.uniform(-self.BOX_YAW, self.BOX_YAW))
        h_eff = abs(hx * np.cos(yaw)) + abs(hy * np.sin(yaw))
        xs = c.x0 + front + h_eff
        back_gap = float(np.random.uniform(*self.BACK_GAP))
        xg = c.x0 + D_INT_BLIND - back_gap - hx
        yg = ys
        # hidden basket: side drawn independently of everything visible (pc_shelf_blind recipe)
        side = 1 if np.random.random() < 0.5 else -1          # +1 = left (+y), -1 = right (-y)
        gap = float(np.random.uniform(*self.BASKET_GAP[side]))
        bw = float(np.random.uniform(*self.BASKET_W))
        y_in = ys + side * gap
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
        tilt = TILT_DEG * side       # fingers lean toward the basket side -> wrist swings away
        # expert waypoints (TCP world positions + hand tilt); fingers closed from the start
        z_p = z_b + Z_PUSH
        x_pre = c.x0 - X_PRE
        x_con = xs - h_eff - FING_X - 0.004
        x_div = c.x0 + D_DIV
        x_end = xg - hx - FING_X + OVERPUSH
        if x_div < x_con + 0.04 or x_end < x_div + 0.08:
            return None
        wps = [
            ("pregrasp", [x_pre, ys, z_p], "fast", 0.0),
            ("CLOSE", None, None, None),
            ("grasp", [x_con, ys, z_p], "approach", 0.0),
            ("place", [x_div, ys, z_p], "push", 0.0),
            ("place", [x_div, ys, z_p], "dwell", 0.0),
            ("place", [x_div, ys, z_p], "tilt", tilt),
            ("place", [x_end, yg, z_p], "push", tilt),
            ("retreat", [x_div, yg, z_p], "mid", tilt),
            ("retreat", [x_div, yg, z_p], "tilt", 0.0),
            ("retreat", [x_pre, yg, z_p], "mid", 0.0),
            ("go_home", list(READY_TCP), "fast", 0.0),
        ]
        # both tilts must be feasible: the accepted (visible) layouts never depend on the side
        for tt in (tilt, -tilt):
            poses = [_T(p, blind_tool_rotation(tt if abs(t) > 0 else 0.0))
                     for (_n, p, _s, t) in wps if p is not None]
            if not self._ik_chain_ok(env, poses):
                return None
        # neighbours: a few LOW items (boxes, jars <= NEIGH_MAX_H) on the work board away from the
        # lane (the pushing hand works 11 cm lower than pc_shelf_blind's carrying hand, so the
        # tilted forearm sweeps lower beside the lane), a few in the upper bay
        avail = list(range(len(self._dyn)))
        out: list = []
        bay = self._bay(z_b, self.NEIGH_MAX_H + 0.02)
        lane = (c.x0 + D_INT / 2, ys, D_INT / 2 + 0.05, self.LANE_HALF)
        sweep = ((xs + xg) / 2, ys, (xg - xs) / 2 + hx + 0.02, hy + 0.03)
        placed: list = []
        for _ in range(3):
            if np.random.random() < 0.7:
                self._random_item(avail, placed, [lane, sweep], bay, out)
        bay_hi = self._bay(c.b_hi, c.bay_hi)
        placed_hi: list = []
        for _ in range(3):
            self._random_item(avail, placed_hi, [], bay_hi, out)
        self._apply_layout(env, out)
        self._mocap(env, "pc_goal_marker", [xg, yg, z_b + 0.0007])
        self._basket_resize(env, basket_half)
        self._mocap(env, "pact_intrusion_basket", basket_center)
        quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
        self._settle_target(env, [xs, ys, z_b + hz + 0.002], quat)
        mujoco.mj_forward(env.current_model, env.current_data)
        d = self._cell_common()
        d["board_depth"] = D_INT_BLIND
        d.update(
            push_board_z=z_b, box_half=list(td["half"]), box_label=td["label"],
            start_xyz=[xs, ys, z_b + hz], start_yaw=yaw, goal_xyz=[xg, yg, z_b + hz],
            push_dir=[1.0, 0.0], push_dist=float(xg - xs), push_tol=self.PUSH_TOL, tcp_push_z=z_p,
            box_front_gap=front, back_gap=back_gap, dwell_s=DWELL_PUSH_S,
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
                       ceiling_z=ceil, z_push=z_p, cell=int(k)),
        )
        return d

    def _referral(self) -> str:
        return self._target_spec.get("label", "box")

    def _task_class(self):
        return PCShelfPushBlindTask


# --------------------------------------------------------------------------------------------
# Task: push success (never lifted / pinched) + the pc_shelf_blind hazard audit and clearance
# --------------------------------------------------------------------------------------------
class PCShelfPushBlindTask(PCShelfBlindTask):
    """Success: box centre within push_tol of the label centre, upright (15 deg), on the board,
    never lifted >= LIFT_MAX, never touched by BOTH fingers at once (no grasp), and now out of
    the hand and at rest (< BOX_STILL). With the collection gate set by the expert, also hazard
    contact <= gate frames (as pc_shelf_blind)."""

    def _push_setup(self, model) -> None:
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        box = model.body(self.config.task_config.pickup_obj_name).id
        side = np.zeros(model.ngeom, dtype=np.int8)     # 1 = left finger, 2 = right finger
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            if int(model.body_rootid[b]) != robot_root:
                continue
            leaf = (model.body(b).name or "").split("/")[-1].lower()
            full = (model.body(b).name or "").lower()
            if "gripper" not in full:
                continue
            if "left" in leaf:
                side[g] = 1
            elif "right" in leaf:
                side[g] = 2
        self._pb = dict(box=int(box), side=side, max_lift=0.0, pinched=False, pinch_step=None)

    def _push_step(self, data) -> None:
        model = data.model
        if getattr(self, "_pb", None) is None or self.episode_step_count == 0:
            self._push_setup(model)
        st = self._pb
        sp = getattr(self, "scene_params", {}) or {}
        z0 = float(sp.get("start_xyz", [0, 0, 0])[2])
        st["max_lift"] = max(st["max_lift"], float(data.xpos[st["box"]][2]) - z0)
        if st["pinched"]:
            return
        seen = 0
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if b1 == st["box"]:
                seen |= int(st["side"][g2])
            elif b2 == st["box"]:
                seen |= int(st["side"][g1])
        if seen == 3:
            st["pinched"] = True
            st["pinch_step"] = int(self.episode_step_count)

    def _accumulate_obstacle_diag(self, data, pickup_obj) -> None:
        super()._accumulate_obstacle_diag(data, pickup_obj)
        try:
            self._push_step(data)
        except Exception as e:  # pragma: no cover - a diagnostic must never break a rollout
            log.debug(f"[PCShelfPushBlind] push tracking failed: {e!r}")

    def push_stats(self) -> dict[str, Any] | None:
        st = getattr(self, "_pb", None)
        if st is None:
            return None
        return dict(max_lift=round(float(st["max_lift"]), 4), pinched=bool(st["pinched"]),
                    pinch_step=st["pinch_step"])

    def _push_success(self) -> tuple[bool, str]:
        sp = getattr(self, "scene_params", {}) or {}
        obj = self._obj()
        p = np.asarray(obj.position, dtype=float)
        g = np.asarray(sp["goal_xyz"], dtype=float)
        dxy = float(np.linalg.norm(p[:2] - g[:2]))
        zaxis = R.from_quat(obj.quat, scalar_first=True).as_matrix()[:, 2]
        upright = float(zaxis[2]) >= np.cos(np.deg2rad(15.0))
        on_board = abs(p[2] - g[2]) < 0.015
        st = self.push_stats() or dict(max_lift=0.0, pinched=False)
        pushed = st["max_lift"] < LIFT_MAX and not st["pinched"]
        # the push is over: hand off the box and the box at rest (as pc_shelf_pnp's "out of the
        # hand"), so an end-at-success eval cannot end mid-push, before the deepest point
        free = not self._robot_touching(obj.body_id)
        data = self._env.mj_datas[0]
        dadr = int(data.model.body_dofadr[int(obj.body_id)])
        speed = float(np.linalg.norm(data.qvel[dadr:dadr + 3])) if dadr >= 0 else 0.0
        still = speed < BOX_STILL
        ok = (dxy <= float(sp.get("push_tol", 0.035)) and upright and on_board and pushed
              and free and still)
        return ok, (f"dxy={dxy:.3f} upright={upright} dz={p[2] - g[2]:+.3f} "
                    f"max_lift={st['max_lift']:.3f} pinched={st['pinched']} free={free} "
                    f"speed={speed:.3f}")

    def _success(self) -> tuple[bool, str]:
        ok, msg = self._push_success()
        gate = getattr(self, "_pc_contact_gate", None)
        if gate is None:
            return ok, msg
        st = self.pc_contact_stats() or {}
        hf = int(st.get("hazard_frames", 0))
        return ok and hf <= int(gate), f"{msg} hazard_frames={hf} gate<={int(gate)}"

    def get_obs_scene(self) -> dict[str, Any]:
        d = super().get_obs_scene()
        ps = self.push_stats()
        if ps is not None:
            sp = dict(d.get("scene_params") or {})
            sp["pc_push"] = _base._jsonable(ps)
            d["scene_params"] = sp
            d["pc_push"] = _base._jsonable(ps)
        return d


# --------------------------------------------------------------------------------------------
# Expert
# --------------------------------------------------------------------------------------------
class PCShelfPushBlindExpert(PCShelfBlindExpert):
    """pc_shelf_blind expert (per-waypoint hand tilt, dwell) for a push, as pc_shelf_push: the
    fingers close once the hand is at the pre-push pose (a GripperAction before any TCP move
    would drift the arm) and no empty-grasp check (the fingers close on nothing). The dwell lasts
    ``scene_params['dwell_s']`` (DWELL_PUSH_S): the hand reaches the divergence point ~80 policy
    steps in, so the pause spans the policy's step-100 chunk query (pc_shelf_blind: step 50)."""

    HOLD_AFTER_CLOSE = False

    def _compute_trajectory(self):
        sp = getattr(self.task, "scene_params", {}) or {}
        dwell = float(sp.get("dwell_s", DWELL_PUSH_S))
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


class PCShelfPushBlindExpertConfig(PCShelfBlindExpertConfig):
    gripper_close_duration: float = 0.5

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCShelfPushBlindExpert


__all__ = [
    "PC_SHELF_PUSH_BLIND_ENV_VERSION", "PC_SHELF_PUSH_BLIND_CELLS", "PC_SHELF_PUSH_BLIND_SCENES",
    "PCShelfPushBlindSampler", "PCShelfPushBlindTask", "PCShelfPushBlindExpert",
    "PCShelfPushBlindExpertConfig", "write_pc_shelf_push_blind_scenes",
    "build_push_blind_scene_xml", "Z_PUSH", "FING_X",
]
