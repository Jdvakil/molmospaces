"""Parameterized enclosure-reach data generation (advisor spec — see ENCLOSURE_DATAGEN_DESIGN.md).

One scene generator (not bespoke scenes): a shelf cubby whose aperture, depth, target pose,
interior protrusion and lighting are drawn PER EPISODE and applied by re-posing mocap slabs
(no recompile). The expert is OBSERVATION-REALIZABLE: it reacts to the hidden protrusion only
once the proximity skin could detect it (detection-gated), modulates speed with clearance, and
aborts cleanly when the residual gap is infeasible. All sampled parameters + the camera-visibility
raycast label + behavior class are logged into obs_scene for stratified eval / decorrelation checks.

Mixture cells: free / hidden / visible / abort, decorrelated from clearance, depth and lighting
by construction (independent draws; cell only gates protrusion presence and intrusion).
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.tasks.cavity_pick_task_sampler import CavityPickTaskSampler
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.utils.object_metadata import ObjectMeta
from molmo_spaces.utils.synset_utils import get_valid_pickupable_obja_uids

log = logging.getLogger(__name__)

# Insertion envelope of the distal assembly (hand + wrist) perpendicular to the tube axis.
# v1 estimate (hand width measured 0.172 m; wrist dia ~0.12); validated by the contact probe.
DIST_W = 0.18
DIST_H = 0.175   # vertical envelope at the 20-deg pitched TRAVEL pose (hand rides low: bottom at
                 # z0+0.031, top at z0+0.171) — sized so the per-wall passable margin is ~c for
                 # top AND side walls, making the abort/deflect cell math wall-uniform.
SHELF_TOP_Z = 0.72          # static cubby floor (top surface of shelf_board)
TUBE_X0 = 0.58              # world x of the aperture plane (front edge of the slabs)
SLAB_LEN = 0.50             # slab half-length along x (slabs span TUBE_X0 .. TUBE_X0+1.0)
PROTR = {"protr_s": 0.0175, "protr_m": 0.025, "protr_l": 0.035}  # half cross-sections
SENSOR_RANGE = 1.0          # detection gate range (SPAD spec reaches 4 m; FOV is the limiter)
SENSOR_RANGE_DERATE = 0.85  # multipath derating near concave corners (advisor caution)
SENSOR_HALF_FOV_COS = float(np.cos(np.deg2rad(22.5)))  # spec-true half FOV (45 deg total)

# TCP orientation for in-tube travel: approach axis (tcp z) = +x world pitched 20 deg DOWN,
# fingers spread horizontal. The down-pitch tucks the gripper housing up away from the shelf
# board (measured: 6.9 cm below TCP when straight -> ~1.9 cm at 20 deg), so the hand can travel
# low enough to pinch short objects without dragging on the board.
_R0 = np.array([[0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0]])
_PITCH = np.deg2rad(20.0)
R_INSERT = np.array([[np.cos(_PITCH), 0.0, np.sin(_PITCH)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(_PITCH), 0.0, np.cos(_PITCH)]]) @ _R0


def _pose(pos, R=R_INSERT) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T




class TaskSpaceServo(ActionPrimitive):
    """Hold-and-converge on a target TCP pose: each step, integrate the measured task-space
    position error into the commanded pose (anti-sag integral action). Ends after `duration`."""

    def __init__(self, robot_view, tcp_to_jp_fn, get_tcp_fn, target_pose, duration=2.6,
                 gain=0.8, max_offset=0.14, name="servo"):
        super().__init__(robot_view, duration)
        self.tcp_to_jp_fn = tcp_to_jp_fn
        self.get_tcp_fn = get_tcp_fn
        self.target_pose = target_pose
        self.gain = gain
        self.max_offset = max_offset
        self._offset = np.zeros(3)
        self._name = name
        self._logged_end = False

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            self._offset = np.zeros(3)
        done = self.elapsed_time() >= self.duration
        if done and not self._logged_end:
            self._logged_end = True
            err = self.target_pose[:3, 3] - self.get_tcp_fn()[:3, 3]
            log.info(f"[Servo:{self._name}] end err={np.round(err, 3)} "
                     f"offset={np.round(self._offset, 3)}")
        return done

    def get_current_action(self):
        err = self.target_pose[:3, 3] - self.get_tcp_fn()[:3, 3]
        self._offset = np.clip(self._offset + self.gain * err, -self.max_offset, self.max_offset)
        cmd = self.target_pose.copy()
        cmd[:3, 3] = cmd[:3, 3] + self._offset
        mg = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.tcp_to_jp_fn(mg, cmd)

    def get_current_phase(self) -> str:
        return self._name


class EnclosureReachSampler(CavityPickTaskSampler):
    """Per-episode θ sampling + mocap slab posing + lighting + raycast visibility label."""

    MIXTURE = (("free", 0.28), ("hidden", 0.33), ("visible", 0.28), ("abort", 0.11))
    POOL_SIZE = 24
    EXCLUDED_TARGET_CATEGORIES = ("egg",)
    BASE_XYZ = (0.0, 0.0, 0.0)

    # target objects: must fit the front pinch (inter-finger max 0.08) and the corridor
    def _build_grasp_uid_pool(self, n: int) -> list[str]:
        pool = []
        for uid in get_valid_pickupable_obja_uids():
            anno = ObjectMeta.annotation(uid) or {}
            bb = anno.get("boundingBox", {})
            dims = sorted(float(bb.get(k, 0)) for k in "xyz")
            cat = str(anno.get("category", "")).lower()
            if any(x in cat or x in uid.lower() for x in self.EXCLUDED_TARGET_CATEGORIES):
                continue
            # max dim must fit the 8cm finger gap at ANY yaw (settle randomizes orientation)
            if 0.030 <= dims[0] and dims[2] <= 0.070:
                pool.append(uid)
            if len(pool) >= n:
                break
        return pool or super()._build_grasp_uid_pool(n)

    # ---------------- θ sampling ----------------
    def _draw_theta(self) -> dict[str, Any]:
        rng = np.random
        cell = rng.choice([c for c, _ in self.MIXTURE], p=[p for _, p in self.MIXTURE])
        # camera-visible parameters: drawn INDEPENDENTLY of the cell (decorrelation by construction)
        clearance = float(rng.uniform(0.01, 0.05) if rng.random() < 0.7 else rng.uniform(0.05, 0.08))
        depth = float(rng.uniform(0.20, 0.35))
        target_frac = float(rng.uniform(0.5, 0.9))
        light_scale = float(10 ** rng.uniform(np.log10(0.10), np.log10(1.2)))
        ap_w = DIST_W + clearance
        ap_h = DIST_H + clearance
        theta: dict[str, Any] = dict(
            cell=cell, clearance=clearance, depth=depth, target_frac=target_frac,
            light_scale=light_scale, ap_w=ap_w, ap_h=ap_h,
            protrusion_present=cell != "free",
            residual_margin=float("nan"),   # only defined when an obstacle is present
            protr_pos_frac=float("nan"),
        )
        if theta["protrusion_present"]:
            theta["protr_wall"] = str(rng.choice(["left", "right", "top"]))
            theta["protr_name"] = str(rng.choice(list(PROTR.keys())))
            # visible-cell protrusions sit near the aperture mouth (that's how visibility
            # physically arises); hidden/abort draws stay deep. The LOGGED raycast label is
            # the ground truth used for stratification either way.
            theta["protr_pos_frac"] = float(rng.uniform(0.05, 0.30) if cell == "visible"
                                            else rng.uniform(0.25, 0.75))
            # DECORRELATION: draw the RESIDUAL MARGIN (gap the arm has left after the obstacle)
            # independently of clearance — this is the behavior-driving hidden quantity the skin
            # measures and the expert reacts to. intrusion is then DERIVED (clearance - residual)
            # purely for mocap placement. Logging residual_margin (not intrusion) keeps the
            # hidden-vs-visible correlation matrix ~0: a wide aperture tells you nothing about
            # whether the arm must deflect or abort. (Old code drew intrusion ∝ clearance →
            # corr(intrusion,clearance)=+0.46, a visual shortcut the advisor would flag.)
            if cell == "abort":
                residual = float(rng.uniform(-0.030, -0.004))   # infeasible: obstacle past the arm
            else:
                residual = float(rng.uniform(0.006, 0.045))     # feasible deflection gap
            theta["residual_margin"] = residual
            theta["intrusion"] = float(np.clip(clearance - residual, 0.005, clearance + 0.045))
        return theta

    # ---------------- scene application ----------------
    def _mocap_set(self, env, body, pos):
        m, d = env.current_model, env.current_data
        mid = int(m.body_mocapid[m.body(body).id])
        d.mocap_pos[mid] = np.asarray(pos, dtype=float)

    @staticmethod
    def _stash_aabbs(th, boxes):
        th["obstacle_aabbs"] = [[list(map(float, c)), list(map(float, h))] for c, h in boxes]

    def _apply_theta(self, env, th: dict[str, Any]) -> None:
        m, d = env.current_model, env.current_data
        z0 = SHELF_TOP_Z
        cx = TUBE_X0 + SLAB_LEN
        self._mocap_set(env, "encl_left", [cx, th["ap_w"] / 2 + 0.02, z0 + 0.33])
        self._mocap_set(env, "encl_right", [cx, -th["ap_w"] / 2 - 0.02, z0 + 0.33])
        self._mocap_set(env, "encl_top", [cx, 0.0, z0 + th["ap_h"] + 0.02])
        self._mocap_set(env, "encl_back", [TUBE_X0 + th["depth"] + 0.02, 0.0, z0 + 0.33])
        # FRONT APERTURE FRAME: the opening is exactly the aperture — cameras off the tube
        # axis cannot see the interior (vision keeps coarse context only; fine geometry = skin)
        self._mocap_set(env, "front_top", [TUBE_X0 - 0.015, 0.0, z0 + th["ap_h"] + 0.30])
        self._mocap_set(env, "front_left", [TUBE_X0 - 0.015, th["ap_w"] / 2 + 0.18, z0 + 0.33])
        self._mocap_set(env, "front_right", [TUBE_X0 - 0.015, -th["ap_w"] / 2 - 0.18, z0 + 0.33])
        # park all protrusions, then place the chosen one
        for k, (px, py) in zip(PROTR, ((0.0, 0.8), (0.0, 1.2), (0.0, 1.6))):
            self._mocap_set(env, k, [px, py, -2.0])
        if th["protrusion_present"]:
            s = PROTR[th["protr_name"]]
            x = TUBE_X0 + th["protr_pos_frac"] * th["depth"]
            i = th["intrusion"]
            if th["protr_wall"] == "left":
                pos = [x, th["ap_w"] / 2 + 0.10 - i, z0 + th["ap_h"] * float(np.random.uniform(0.35, 0.65))]
            elif th["protr_wall"] == "right":
                pos = [x, -(th["ap_w"] / 2 + 0.10 - i), z0 + th["ap_h"] * float(np.random.uniform(0.35, 0.65))]
            else:  # top — bar hangs down; long axis vertical is approximated by same bar lying in y
                pos = [x, 0.0, z0 + th["ap_h"] + 0.10 - i]
            self._mocap_set(env, th["protr_name"], pos)
            th["protr_center"] = list(map(float, pos))
            th["protr_half"] = [s, 0.10, s]
        # LIVE obstacle list (every skin-sensable surface as posed THIS episode) — feeds the
        # expert's live speed law. Floor/board excluded: link sensors do not face down at it.
        boxes = [
            ([cx, th["ap_w"] / 2 + 0.02, z0 + 0.33], [SLAB_LEN, 0.02, 0.35]),
            ([cx, -th["ap_w"] / 2 - 0.02, z0 + 0.33], [SLAB_LEN, 0.02, 0.35]),
            ([cx, 0.0, z0 + th["ap_h"] + 0.02], [SLAB_LEN, 0.45, 0.02]),
            ([TUBE_X0 + th["depth"] + 0.02, 0.0, z0 + 0.33], [0.02, 0.45, 0.35]),
            ([TUBE_X0 - 0.015, 0.0, z0 + th["ap_h"] + 0.30], [0.015, 0.45, 0.30]),
            ([TUBE_X0 - 0.015, th["ap_w"] / 2 + 0.18, z0 + 0.33], [0.015, 0.18, 0.35]),
            ([TUBE_X0 - 0.015, -th["ap_w"] / 2 - 0.18, z0 + 0.33], [0.015, 0.18, 0.35]),
        ]
        if th["protrusion_present"]:
            boxes.append((th["protr_center"], th["protr_half"]))
        self._stash_aabbs(th, boxes)
        # lighting: scale diffuse of all lights + headlight (per-episode, log-uniform)
        if not hasattr(self, "_light_base"):
            self._light_base = m.light_diffuse.copy()
            self._headlight_base = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        m.light_diffuse[:] = self._light_base * th["light_scale"]
        m.vis.headlight.diffuse[:] = self._headlight_base[0] * th["light_scale"]
        m.vis.headlight.ambient[:] = self._headlight_base[1] * max(th["light_scale"], 0.15)
        mujoco.mj_forward(m, d)

    # target rest position from θ (used by the settle machinery)
    def _obj_rest(self):
        th = getattr(self, "_theta", None)
        if not th:
            return (TUBE_X0 + 0.25, 0.0, SHELF_TOP_Z)
        x = TUBE_X0 + max(0.12, th["target_frac"] * th["depth"] - 0.04)
        y = float(np.random.uniform(-1, 1) * (th["ap_w"] / 2 - 0.05))
        return (x, y, SHELF_TOP_Z)

    OBJ_JIT_XY = (0.015, 0.015)

    # ---------------- camera-visibility raycast label ----------------
    def _cam_visible_label(self, env, th) -> bool:
        if not th.get("protrusion_present"):
            return False
        m, d = env.current_model, env.current_data
        cams = []
        try:
            cid = m.camera("robot_0/gripper/wrist_camera").id
            cams.append(np.array(d.cam_xpos[cid]))
        except Exception:
            pass
        base = np.array(self._cur_base_xyz)
        yaw = self._cur_base_yaw
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        exo = np.array([*(base[:2] + Rz @ np.array([0.10, 0.57])), 0.35 + 0.66])
        cams.append(exo)
        center = np.array(th["protr_center"])
        half = np.array(th["protr_half"])
        pbody = m.body(th["protr_name"]).id
        targets = [center + np.array([-half[0], 0, 0]),
                   center + np.array([-half[0], half[1] * 0.6, 0]),
                   center + np.array([-half[0], -half[1] * 0.6, 0]),
                   center + np.array([-half[0], 0, half[2] * 0.6]),
                   center + np.array([-half[0], 0, -half[2] * 0.6])]
        geomid = np.zeros(1, dtype=np.int32)
        for c in cams:
            for t in targets:
                v = t - c
                dist = float(np.linalg.norm(v))
                if dist < 1e-6:
                    continue
                hit = mujoco.mj_ray(m, d, c.astype(np.float64), (v / dist).astype(np.float64),
                                    None, 1, -1, geomid)
                if hit >= 0 and geomid[0] >= 0 and int(m.geom_bodyid[geomid[0]]) == pbody:
                    return True
        return False

    # ---------------- per-episode orchestration ----------------
    def _sample_task(self, env: CPUMujocoEnv):
        # draw θ honoring the mixture cell's visibility label (bounded rejection on protrusion draw)
        th = self._draw_theta()
        self._cur_base_xyz = (float(np.random.uniform(-0.02, 0.02)),
                              float(np.random.uniform(-0.02, 0.02)), 0.0)
        self._cur_base_yaw = float(np.random.uniform(-0.05, 0.05))
        for attempt in range(20):
            self._apply_theta(env, th)
            if th["cell"] not in ("hidden", "visible"):
                break
            vis = self._cam_visible_label(env, th)
            if (th["cell"] == "visible") == vis:
                break
            # redraw ONLY protrusion placement (keeps visible params untouched -> decorrelation)
            th["protr_wall"] = str(np.random.choice(["left", "right", "top"]))
            th["protr_name"] = str(np.random.choice(list(PROTR.keys())))
            th["protr_pos_frac"] = float(np.random.uniform(0.05, 0.30) if th["cell"] == "visible"
                                         else np.random.uniform(0.25, 0.75))
        self._theta = th
        task = super()._sample_task(env)
        # PickTaskSampler hardcodes PickTask(env, config) and ignores task_config.task_cls;
        # re-class so EnclosureReachTask's judge_success (abort = success) and get_obs_scene
        # (θ + behavior_class logging) are active. Safe: subclass adds no constructor state.
        task.__class__ = EnclosureReachTask
        th["cam_visible"] = self._cam_visible_label(env, th)
        th["target_uid"] = self._uid_pool[self.current_house_index % len(self._uid_pool)]
        task.scene_params = dict(th)
        task.enclosure_info = dict(tube_x0=TUBE_X0, z0=SHELF_TOP_Z, ap_w=th["ap_w"],
                                   ap_h=th["ap_h"], depth=th["depth"])
        log.info(f"[Enclosure] cell={th['cell']} clearance={th['clearance']*100:.1f}cm "
                 f"depth={th['depth']:.2f} light={th['light_scale']:.3f} "
                 f"protr={th.get('protr_wall')}/{th.get('intrusion', 0)*100:.1f}cm "
                 f"cam_visible={th['cam_visible']}")
        return task


class EnclosureReachTask(PickTask):
    """Pick-from-enclosure task: success = lifted-and-retrieved OR clean abort (its own class)."""

    def judge_success(self) -> bool:
        policy = getattr(self, "_registered_policy", None)
        behavior = getattr(policy, "behavior_class", None)
        if behavior == "abort":
            rv = self.env.current_robot.robot_view
            mg = rv.get_gripper_movegroup_ids()[0]
            tcp_x = float(rv.get_move_group(mg).leaf_frame_to_world[0, 3])
            return tcp_x < TUBE_X0 - 0.03   # retreated cleanly outside the aperture plane
        return super().judge_success()

    def get_obs_scene(self) -> dict[str, Any]:
        d = super().get_obs_scene()
        d["scene_params"] = getattr(self, "scene_params", {})
        policy = getattr(self, "_registered_policy", None)
        d["behavior_class"] = getattr(policy, "behavior_class", "unknown")
        return d


class EnclosureExpertPolicy(BaseObjectManipulationPlannerPolicy):
    """Observation-realizable scripted expert (see module docstring).

    Behavior classes: 'free' (no event), 'deflect' (detection-gated re-route around the
    protrusion), 'abort' (detection-gated clean retreat when residual gap is infeasible).
    Speed is modulated by the corridor margin (v ∝ clearance), and after detection by the
    residual gap — both quantities the skin measures continuously.
    """

    SPEED_FAST = 0.20
    SPEED_MIN = 0.04
    GRASP_X_STANDOFF = 0.10   # pregrasp this far in front of the object
    SETTLE = 1.2

    def __init__(self, config, task) -> None:
        super().__init__(config, task)
        self.behavior_class = "free"
        self._detected = False
        self._sensor_cam_ids: list[int] | None = None

    # ---- helpers ----
    # Arm surface envelope beyond the TCP point at the pitched travel pose (m). Converts
    # TCP-path-to-AABB distance into the SKIN-surface margin the sensors actually read —
    # probe 1 regresses commanded speed against exactly this live quantity.
    ENV_LO = np.array([0.0, 0.090, 0.019])   # behind / -y / below
    ENV_HI = np.array([0.0, 0.090, 0.121])   # ahead  / +y / above
    MARGIN_OPEN = 0.10

    def _v(self, margin: float) -> float:
        # margin = LIVE surface margin (≈ c/2 centered in the tube), not the θ scalar
        return float(np.clip(self.SPEED_FAST * margin / 0.035, self.SPEED_MIN, self.SPEED_FAST))

    def _obstacle_aabbs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(np.asarray(c, dtype=float), np.asarray(h, dtype=float))
                for c, h in self._theta().get("obstacle_aabbs", [])]

    def _surf_dist(self, p, c, h) -> float:
        g = np.zeros(3)
        for k in range(3):
            lo, hi = c[k] - h[k], c[k] + h[k]
            if p[k] + self.ENV_HI[k] < lo:
                g[k] = lo - (p[k] + self.ENV_HI[k])
            elif p[k] - self.ENV_LO[k] > hi:
                g[k] = (p[k] - self.ENV_LO[k]) - hi
        return float(np.linalg.norm(g))

    def _seg_margin(self, a, b) -> float:
        """LIVE speed-law input: min predicted skin-to-obstacle gap along the TCP path a->b,
        from the episode's actual posed geometry (NOT the episode's clearance scalar)."""
        boxes = self._obstacle_aabbs()
        if not boxes:
            return self.MARGIN_OPEN
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        m = self.MARGIN_OPEN
        for t in np.linspace(0.0, 1.0, 7):
            p = a + t * (b - a)
            m = min(m, min(self._surf_dist(p, c, h) for c, h in boxes))
        return float(np.clip(m, 0.004, self.MARGIN_OPEN))

    def _tcp_now(self) -> np.ndarray:
        mg = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.robot_view.get_move_group(mg).leaf_frame_to_world.copy()

    # ---- embed transform (world<-task-local). Identity for standalone scenes; for in-house
    # scenes the sampler stamps scene_params['embed']=(bx,by,yaw). Poses sent to IK / motion
    # primitives MUST be world (check_failure compares world gripper to the segment target),
    # so build them with _P(...); proximity margins stay in the LOCAL frame (TUBE_X0 etc). ----
    def _embed_T(self) -> np.ndarray:
        e = (self._theta() or {}).get("embed")
        if not e:
            return np.eye(4)
        bx, by, yaw = e
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = np.array([bx, by, 0.0])
        return T

    def _P(self, xyz, Rmat=None) -> np.ndarray:
        """task-local position+orientation -> WORLD 4x4 pose for the motion stack."""
        return self._embed_T() @ _pose(xyz, R_INSERT if Rmat is None else Rmat)

    def _tcp_local(self) -> np.ndarray:
        return np.linalg.inv(self._embed_T()) @ self._tcp_now()

    def _seq(self, segs, holding=False) -> TCPMoveSequence:
        return TCPMoveSequence(
            self.robot_view, self._tcp_to_jp_fn, self.SETTLE,
            move_segments=segs, is_holding_object=holding,
            gripper_empty_threshold=0.001,
            tcp_pos_err_threshold=0.10, tcp_rot_err_threshold=0.8,
        )

    def _theta(self) -> dict:
        return getattr(self.task, "scene_params", {}) or {}

    def _grasp_R(self) -> np.ndarray:
        """Grasp-descent orientation. Defaults to R_INSERT (20 deg down-pitch, the travel pose);
        samplers can request a steeper pitch via scene_params['grasp_pitch_deg'] so the gripper
        points further down and the arm can descend onto a bench-height object at forward reach."""
        deg = float(self._theta().get("grasp_pitch_deg", 20.0))
        p = np.deg2rad(deg)
        Ry = np.array([[np.cos(p), 0.0, np.sin(p)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(p), 0.0, np.cos(p)]])
        return Ry @ _R0

    def _target_pos(self) -> np.ndarray:
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        return np.array(obj.position, dtype=float)

    # ---- nominal plan: no knowledge of the (hidden) protrusion ----
    def reset(self, reset_retries: bool = True):
        if reset_retries:           # new EPISODE: clear the episode-level behavior label
            self.behavior_class = "free"
        super().reset(reset_retries)

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        # NOTE: do NOT reset behavior_class here — this runs on every retry replan, and the
        # episode's label (abort/deflect) must survive retries to reach judge_success/obs_scene.
        self._detected = False
        self._logged_close = False
        th = self._theta()
        c = float(th.get("clearance", 0.04))
        z_travel = SHELF_TOP_Z + 0.050          # TCP height: object grasp height, just off the floor
        # plan in the task-LOCAL frame (object pos transformed world->local); margins use these
        tgt = (np.linalg.inv(self._embed_T()) @ np.append(self._target_pos(), 1.0))[:3]
        start = self._tcp_now()                 # actual current world pose (start orientation)
        staging_l = np.array([TUBE_X0 - 0.05, 0.0, z_travel + 0.02])
        # pinch AT the object's height (dip from travel height), gripping center on the object
        gz = float(np.clip(float(tgt[2]) + 0.005, SHELF_TOP_Z + 0.026, SHELF_TOP_Z + 0.055))
        grasp_l = np.array([float(tgt[0]) + 0.012, float(tgt[1]), gz])
        # Steeper down-pitch for the final descent (pregrasp+grasp): a near-horizontal wrist
        # (R_INSERT, 20 deg) cannot dip the TCP onto a bench-height object at forward reach
        # (IK saturates ~10 cm high). Pointing the gripper down lets the arm descend onto it.
        Rg = self._grasp_R()
        # APPROACH-AXIS descent: pregrasp sits above-behind the grasp point along the gripper's
        # pointing axis (Rg z), so the final segment slides the fingers along the direction they
        # point and the object enters the finger gap from the front. The old horizontal advance
        # at object height BULLDOZED the object (CLOSE@ logs showed it pushed 15-25 cm deeper,
        # then the fingers closed on air).
        a_axis = Rg[:, 2]
        pregrasp_l = grasp_l - 0.13 * a_axis
        # enter above object-top height (pregrasp height), forward of the aperture plane but
        # always behind the pregrasp so the advance never backtracks
        enter_l = np.array([min(TUBE_X0 + 0.06, float(pregrasp_l[0]) - 0.05),
                            float(tgt[1]), float(pregrasp_l[2])])
        staging, enter = self._P(staging_l), self._P(enter_l)
        pregrasp, grasp = self._P(pregrasp_l, Rg), self._P(grasp_l, Rg)
        self._retreat_pose = staging
        # LIVE speed law: each segment's speed comes from the measured-geometry margin along
        # THAT segment (what the skin reads there), not from the episode's clearance scalar.
        v_ins = self._v(self._seg_margin(staging_l, enter_l))
        v_adv = self._v(self._seg_margin(enter_l, pregrasp_l))
        v_gr = max(self._v(self._seg_margin(pregrasp_l, grasp_l)) * 0.6, self.SPEED_MIN)
        lift = self._P(staging_l + np.array([0, 0, 0.06]))
        return [
            GripperAction(self.robot_view, True, 0.0),
            self._seq([
                TCPMoveSegment(name="approach", start_pose=start, end_pose=staging, speed=self.SPEED_FAST),
                TCPMoveSegment(name="insert", start_pose=staging, end_pose=enter, speed=v_ins),
                TCPMoveSegment(name="advance", start_pose=enter, end_pose=pregrasp, speed=v_adv),
                TCPMoveSegment(name="grasp", start_pose=pregrasp, end_pose=grasp, speed=v_gr),
            ]),
            TaskSpaceServo(self.robot_view, self._tcp_to_jp_fn, self._tcp_now, grasp, name="grasp"),
            GripperAction(self.robot_view, False, self.policy_config.gripper_close_duration),
            self._seq([
                TCPMoveSegment(name="extract", start_pose=grasp, end_pose=staging, speed=v_adv),
                TCPMoveSegment(name="lift", start_pose=staging, end_pose=lift, speed=0.10),
            ], holding=True),
        ]

    # ---- detection gating ----
    def _sensor_poses(self):
        m = self.task.env.current_model
        d = self.task.env.current_data
        if self._sensor_cam_ids is None:
            self._sensor_cam_ids = [i for i in range(m.ncam) if "_sensor_" in m.camera(i).name]
        for cid in self._sensor_cam_ids:
            yield np.array(d.cam_xpos[cid]), d.cam_xmat[cid].reshape(3, 3)

    _AABB_SAMPLES = [(-1, 0, 0), (-1, .7, 0), (-1, -.7, 0), (-1, 0, .7), (-1, 0, -.7),
                     (0, .9, 0), (0, -.9, 0), (0, 0, -.9)]

    def _protrusion_detected(self) -> bool:
        """Spec-true FOV gate: an extended protrusion is detected when ANY sampled surface
        point falls inside some sensor's 22.5-deg half-FOV within range. Empirically this
        fires ~19 cm before contact for ~75% of blocking protrusions; the rest are genuinely
        invisible to a skin without hand coverage and are handled by the stall gate."""
        th = self._theta()
        if not th.get("protrusion_present") or "protr_center" not in th:
            return False
        center = np.array(th["protr_center"])
        half = np.array(th["protr_half"])
        rng_eff = SENSOR_RANGE * SENSOR_RANGE_DERATE
        # protr_center/half are task-LOCAL; sensors read WORLD -> transform the sampled points
        T = self._embed_T()
        pts = [(T @ np.append(center + half * np.array(s), 1.0))[:3] for s in self._AABB_SAMPLES]
        for pos, xmat in self._sensor_poses():
            fwd = -xmat[:, 2]   # MuJoCo cameras look along -z
            for pt in pts:
                v = pt - pos
                dist = float(np.linalg.norm(v))
                if dist > rng_eff or dist < 1e-9:
                    continue
                if float(np.dot(v / dist, fwd)) > SENSOR_HALF_FOV_COS:
                    return True
        return False

    def _stalled(self) -> bool:
        """Proprioceptive contact gate (student-observable: commanded-vs-actual joint/TCP gap).
        Fires when in-tube tracking error stays >4 cm for ~0.5 s during insertion phases —
        the signature of bumping a blocking obstacle the FOV gate could not see."""
        if self.action_idx >= len(self.action_primitives):
            return False
        act = self.action_primitives[self.action_idx]
        if not isinstance(act, TCPMoveSequence) or act.move_seg_idx is None:
            return False
        phase = act.get_current_phase()
        if phase not in ("insert", "advance", "deflect", "pass_protrusion"):
            self._stall_count = 0
            return False
        try:
            target = act.get_current_target_pose()
        except Exception:
            return False
        err = float(np.linalg.norm(target[:3, 3] - self._tcp_now()[:3, 3]))
        self._stall_count = getattr(self, "_stall_count", 0) + 1 if err > 0.04 else 0
        return self._stall_count >= 8

    # ---- detection-gated replanning ----
    def _replan_on_detection(self):
        th = self._theta()
        c = float(th["clearance"])
        i = float(th["intrusion"])
        # use the drawn residual when present (the independent hidden quantity); fall back to
        # the geometric difference for legacy thetas
        residual_margin = float(th.get("residual_margin", c - i))
        now = self._tcp_now()                       # WORLD (used for abort retreat start)
        if residual_margin < 0.004:   # infeasible: abort + retreat
            self._replan_abort(now, "residual gap infeasible")
            return
        # feasible: deflect away from the protrusion wall while passing it, then continue.
        # plan in the task-LOCAL frame, then map to world with _P.
        self.behavior_class = "deflect"
        now_l = self._tcp_local()[:3, 3]
        tgt = (np.linalg.inv(self._embed_T()) @ np.append(self._target_pos(), 1.0))[:3]
        z_travel = float(now_l[2])
        wall = th["protr_wall"]
        shift = i / 2 + 0.008
        dy, dz = 0.0, 0.0
        if wall == "left":
            dy = -shift
        elif wall == "right":
            dy = shift
        else:
            dz = -shift
        x_pr = float(th["protr_center"][0])
        p1_l = np.array([max(float(now_l[0]), x_pr - 0.12), float(now_l[1]) + dy, z_travel + dz])
        p2_l = np.array([x_pr + 0.10, float(now_l[1]) + dy, z_travel + dz])
        pregrasp_l = np.array([float(tgt[0]) - self.GRASP_X_STANDOFF, float(tgt[1]), z_travel])
        gz = float(np.clip(float(tgt[2]) + 0.005, SHELF_TOP_Z + 0.026, SHELF_TOP_Z + 0.055))
        grasp_l = np.array([float(tgt[0]) + 0.012, float(tgt[1]), gz])
        out_mid_l = np.array([x_pr + 0.10, float(now_l[1]) + dy, z_travel + dz])
        in_mid_l = np.array([x_pr - 0.12, float(now_l[1]) + dy, z_travel + dz])
        p1, p2, pregrasp, grasp = self._P(p1_l), self._P(p2_l), self._P(pregrasp_l), self._P(grasp_l)
        out_mid, in_mid = self._P(out_mid_l), self._P(in_mid_l)
        # LIVE speeds (margins in local): the pass margin reflects the actual residual gap
        v_defl = self._v(self._seg_margin(now_l, p1_l))
        v_pass = self._v(self._seg_margin(p1_l, p2_l))
        v_adv = self._v(self._seg_margin(p2_l, pregrasp_l))
        v_gr = max(self._v(self._seg_margin(pregrasp_l, grasp_l)) * 0.6, self.SPEED_MIN)
        lift = self._retreat_pose.copy(); lift[2, 3] += 0.06
        self.action_primitives = [
            self._seq([
                TCPMoveSegment(name="deflect", start_pose=now, end_pose=p1, speed=v_defl),
                TCPMoveSegment(name="pass_protrusion", start_pose=p1, end_pose=p2, speed=v_pass),
                TCPMoveSegment(name="advance", start_pose=p2, end_pose=pregrasp, speed=v_adv),
                TCPMoveSegment(name="grasp", start_pose=pregrasp, end_pose=grasp, speed=v_gr),
            ]),
            TaskSpaceServo(self.robot_view, self._tcp_to_jp_fn, self._tcp_now, grasp, name="grasp"),
            GripperAction(self.robot_view, False, self.policy_config.gripper_close_duration),
            self._seq([
                TCPMoveSegment(name="extract_deflect", start_pose=grasp, end_pose=out_mid, speed=v_pass),
                TCPMoveSegment(name="extract_pass", start_pose=out_mid, end_pose=in_mid, speed=v_pass),
                TCPMoveSegment(name="extract", start_pose=in_mid, end_pose=self._retreat_pose, speed=v_adv),
                TCPMoveSegment(name="lift", start_pose=self._retreat_pose, end_pose=lift, speed=0.10),
            ], holding=True),
        ]
        self.action_idx = 0
        log.info(f"[EnclosureExpert] DEFLECT around {wall} protrusion (residual {residual_margin*100:.1f}cm)")

    def _replan_abort(self, now: np.ndarray, why: str) -> None:
        self.behavior_class = "abort"
        self.action_primitives = [
            self._seq([TCPMoveSegment(name="retreat", start_pose=now,
                                      end_pose=self._retreat_pose, speed=0.06)]),
        ]
        self.action_idx = 0
        self._stall_count = 0
        log.info(f"[EnclosureExpert] ABORT: {why} — retreating")

    def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
        if not self._detected and self._tcp_local()[0, 3] > TUBE_X0 - 0.18:
            if self._protrusion_detected():
                self._detected = True
                self._replan_on_detection()
            elif self._stalled():
                self._detected = True
                self._replan_abort(self._tcp_now(), "in-tube stall (blind contact)")
        # diagnostic: object vs TCP at the moment the close starts
        if self.action_idx < len(self.action_primitives):
            act = self.action_primitives[self.action_idx]
            if isinstance(act, GripperAction) and not act.target_open and not getattr(self, "_logged_close", False):
                self._logged_close = True
                tcp = self._tcp_now()[:3, 3]
                obj = self._target_pos()
                log.info(f"[EnclosureExpert] CLOSE@ tcp={np.round(tcp,3)} obj={np.round(obj,3)} "
                         f"delta={np.round(obj-tcp,3)}")
        return super().get_action(info)

    def get_all_phases(self):
        phases = super().get_all_phases()
        for name in ("approach", "insert", "advance", "deflect", "pass_protrusion",
                     "extract", "extract_deflect", "extract_pass", "retreat"):
            if name not in phases:
                phases[name] = max(phases.values()) + 1
        return phases


# ---------------- policy config wiring ----------------
from molmo_spaces.configs.policy_configs import (  # noqa: E402
    ObjectManipulationPlannerPolicyConfig,
    PickPlannerPolicyConfig,
)


class EnclosureExpertPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    """Wires EnclosureExpertPolicy as the rollout policy."""

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = EnclosureExpertPolicy


class FumehoodSampler(EnclosureReachSampler):
    """Fumehood variant — NO camera occlusion. Glass sash (cameras see through; arm must pass
    under), jambs set opening width, VISIBLE upright obstacles inside the hood. Sensors matter
    for whole-arm clearance (sash edge above the wrist, jambs beside the forearm, obstacles by
    the elbow), not for hiding things. Pair with fumehood.xml (sash plane = TUBE_X0)."""

    MIXTURE = (("free", 0.40), ("hidden", 0.30), ("visible", 0.15), ("abort", 0.15))

    def _apply_theta(self, env, th):
        m, d = env.current_model, env.current_data
        z0 = SHELF_TOP_Z
        # sash: bottom edge = opening height (z0 + DIST_H + clearance); slab half-height 0.30
        sash_bottom = z0 + th["ap_h"]
        self._mocap_set(env, "sash", [TUBE_X0, 0.0, sash_bottom + 0.025])  # opaque sash RAIL (no glass)
        # jambs: inner edges at +-ap_w/2 (slab half-width 0.18)
        self._mocap_set(env, "jamb_l", [TUBE_X0, th["ap_w"] / 2 + 0.18, z0 + 0.20])
        self._mocap_set(env, "jamb_r", [TUBE_X0, -th["ap_w"] / 2 - 0.18, z0 + 0.20])
        # obstacles: VISIBLE upright bars standing on the bench inside the hood
        for k, (px, py) in zip(PROTR, ((0.0, 0.8), (0.0, 1.2), (0.0, 1.6))):
            self._mocap_set(env, k, [px, py, -2.0])
        if th["protrusion_present"]:
            name = th["protr_name"]
            geom_half_z = {"protr_s": 0.10, "protr_m": 0.11, "protr_l": 0.12}[name]
            s = PROTR[name]
            x = TUBE_X0 + th["protr_pos_frac"] * th["depth"]
            side = 1 if th["protr_wall"] == "left" else -1
            if th["protr_wall"] == "top":
                side = 1 if np.random.random() < 0.5 else -1
                th["protr_wall"] = "left" if side > 0 else "right"
            # bar offset from corridor center: intrusion i means the bar's inner face reaches
            # (ap_w/2 - i) from center — same residual-margin math as the enclosure
            y = side * (th["ap_w"] / 2 + s - th["intrusion"])
            pos = [x, float(y), z0 + geom_half_z]
            self._mocap_set(env, name, pos)
            th["protr_center"] = list(map(float, pos))
            th["protr_half"] = [s, s, geom_half_z]
        # LIVE obstacle list: sash rail, jambs, hood shell, obstacle bar (bench excluded)
        boxes = [
            ([TUBE_X0, 0.0, sash_bottom + 0.025], [0.015, 0.44, 0.025]),
            ([TUBE_X0, th["ap_w"] / 2 + 0.18, z0 + 0.20], [0.012, 0.18, 0.20]),
            ([TUBE_X0, -th["ap_w"] / 2 - 0.18, z0 + 0.20], [0.012, 0.18, 0.20]),
            ([0.95, 0.45, 1.12], [0.40, 0.012, 0.40]),
            ([0.95, -0.45, 1.12], [0.40, 0.012, 0.40]),
            ([1.36, 0.0, 1.12], [0.012, 0.46, 0.40]),
        ]
        if th["protrusion_present"]:
            boxes.append((th["protr_center"], th["protr_half"]))
        self._stash_aabbs(th, boxes)
        if not hasattr(self, "_light_base"):
            self._light_base = m.light_diffuse.copy()
            self._headlight_base = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        m.light_diffuse[:] = self._light_base * th["light_scale"]
        m.vis.headlight.diffuse[:] = self._headlight_base[0] * th["light_scale"]
        m.vis.headlight.ambient[:] = self._headlight_base[1] * max(th["light_scale"], 0.15)
        mujoco.mj_forward(m, d)

    def _cam_visible_label(self, env, th) -> bool:
        """Ray test that SEES THROUGH glass (alpha < 0.5 geoms are transparent to RGB)."""
        if not th.get("protrusion_present"):
            return False
        m, d = env.current_model, env.current_data
        cams = []
        try:
            cams.append(np.array(d.cam_xpos[m.camera("robot_0/gripper/wrist_camera").id]))
        except Exception:
            pass
        base = np.array(self._cur_base_xyz); yaw = self._cur_base_yaw
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        cams.append(np.array([*(base[:2] + Rz @ np.array([0.10, 0.57])), 0.35 + 0.66]))
        center = np.array(th["protr_center"]); pbody = m.body(th["protr_name"]).id
        geomid = np.zeros(1, dtype=np.int32)
        for c in cams:
            pnt = c.copy()
            for hop in range(4):   # hop through transparent geoms
                v = center - pnt; dist = float(np.linalg.norm(v))
                if dist < 1e-6:
                    break
                hit = mujoco.mj_ray(m, d, pnt.astype(np.float64), (v / dist).astype(np.float64),
                                    None, 1, -1, geomid)
                if hit < 0 or geomid[0] < 0:
                    break
                if int(m.geom_bodyid[geomid[0]]) == pbody:
                    return True
                if float(m.geom_rgba[geomid[0]][3]) < 0.5:   # glass: continue past
                    pnt = pnt + (v / dist) * (hit + 0.002)
                    continue
                break
        return False


class FumehoodExpertPolicy(EnclosureExpertPolicy):
    """Obstacles are VISIBLE here (glass + open front), so the expert may plan around them
    from the start — no hidden-geometry gating needed; speed modulation + servo + stall kept."""

    def _protrusion_detected(self) -> bool:
        return bool(self._theta().get("protrusion_present"))


class FumehoodExpertPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = FumehoodExpertPolicy


class PanelSlalomSampler(FumehoodSampler):
    """Photo-1 recreation: upright panels on a table; arm threads the gap in pair 1, then a
    second pair deeper is laterally OFFSET (the visible 'protrusion') forcing a deflection.
    Everything camera-visible; sensors carry whole-arm gap clearance.

    NO 'hidden' cell: the scene has no occluders by design (advisor: no camera occlusion),
    so the hidden/visible raycast split degenerates — obstacles are always visible."""

    MIXTURE = (("free", 0.35), ("visible", 0.45), ("abort", 0.20))

    def _draw_theta(self):
        th = super()._draw_theta()
        if th["protrusion_present"]:
            # pair-2 station: spread along the run (the mouth-hugging 'visible' prior is
            # meaningless here — everything is visible at any depth)
            th["protr_pos_frac"] = float(np.random.uniform(0.30, 0.85))
        return th

    def _cam_visible_label(self, env, th) -> bool:
        # open tabletop, no occluders: any present panel is camera-visible by construction
        return bool(th.get("protrusion_present"))

    def _apply_theta(self, env, th):
        m, d = env.current_model, env.current_data
        z0 = SHELF_TOP_Z
        pz = z0 + 0.20
        # pair 1: the entry gap = aperture width
        self._mocap_set(env, "p1l", [TUBE_X0, th["ap_w"] / 2 + 0.14, pz])
        self._mocap_set(env, "p1r", [TUBE_X0, -th["ap_w"] / 2 - 0.14, pz])
        # pair 2: deeper; one side intrudes by i (the deflection driver), other stays flush
        x2 = TUBE_X0 + max(0.12, th["protr_pos_frac"] * th["depth"]) if th["protrusion_present"] \
            else TUBE_X0 + 0.5 * th["depth"]
        i = th.get("intrusion", 0.0) if th["protrusion_present"] else 0.0
        if th.get("protr_wall") == "top":
            th["protr_wall"] = "left" if np.random.random() < 0.5 else "right"
        side = 1 if th.get("protr_wall") == "left" else -1
        yl = th["ap_w"] / 2 + 0.14 - (i if side > 0 else 0.0)
        yr = -th["ap_w"] / 2 - 0.14 + (i if side < 0 else 0.0)
        self._mocap_set(env, "p2l", [x2, yl, pz])
        self._mocap_set(env, "p2r", [x2, yr, pz])
        if th["protrusion_present"]:
            iy = (yl - 0.14) if side > 0 else (yr + 0.14)
            th["protr_center"] = [float(x2), float(iy), pz]
            th["protr_half"] = [0.015, 0.14, 0.20]
        # LIVE obstacle list: the four panels as posed this episode
        self._stash_aabbs(th, [
            ([TUBE_X0, th["ap_w"] / 2 + 0.14, pz], [0.015, 0.14, 0.20]),
            ([TUBE_X0, -th["ap_w"] / 2 - 0.14, pz], [0.015, 0.14, 0.20]),
            ([float(x2), float(yl), pz], [0.015, 0.14, 0.20]),
            ([float(x2), float(yr), pz], [0.015, 0.14, 0.20]),
        ])
        if not hasattr(self, "_light_base"):
            self._light_base = m.light_diffuse.copy()
            self._headlight_base = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        m.light_diffuse[:] = self._light_base * th["light_scale"]
        m.vis.headlight.diffuse[:] = self._headlight_base[0] * th["light_scale"]
        m.vis.headlight.ambient[:] = self._headlight_base[1] * max(th["light_scale"], 0.15)
        mujoco.mj_forward(m, d)


R_TOPDOWN = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]])
CUB_FLOOR_Z = 0.44
CUB_X = (0.42, 0.83)


class CubbyOverreachSampler(EnclosureReachSampler):
    """Photo-2 recreation: open-top cubby; arm reaches OVER the front wall, descends inside.
    Front-wall height = clearance knob; a divider sometimes narrows the target compartment.

    NO 'hidden' cell: open-top box, the divider is always camera-visible by design."""

    MIXTURE = (("free", 0.35), ("visible", 0.45), ("abort", 0.20))
    OBJ_JIT_XY = (0.04, 0.10)

    def _draw_theta(self):
        th = super()._draw_theta()
        th["wall_top"] = float(CUB_FLOOR_Z + np.random.uniform(0.16, 0.30))
        return th

    def _cam_visible_label(self, env, th) -> bool:
        # open-top cubby, no occluders: a present divider is camera-visible by construction
        return bool(th.get("protrusion_present"))

    def _apply_theta(self, env, th):
        m, d = env.current_model, env.current_data
        self._mocap_set(env, "front_wall", [0.40, 0.0, th["wall_top"] - 0.16])
        self._mocap_set(env, "divider", [0.0, 1.0, -2.0])
        if th["protrusion_present"]:
            # divider splits the box; target compartment width = ap_w analog via intrusion math
            dy = float(np.random.uniform(-0.12, 0.12))
            self._mocap_set(env, "divider", [0.62, dy, CUB_FLOOR_Z + 0.14])
            th["protr_center"] = [0.62, dy, CUB_FLOOR_Z + 0.14]
            th["protr_half"] = [0.20, 0.012, 0.14]
        # LIVE obstacle list: front wall (as posed), box shell, divider (floor excluded)
        boxes = [
            ([0.40, 0.0, th["wall_top"] - 0.16], [0.015, 0.30, 0.16]),
            ([0.85, 0.0, 0.62], [0.015, 0.30, 0.22]),
            ([0.62, 0.30, 0.62], [0.22, 0.015, 0.22]),
            ([0.62, -0.30, 0.62], [0.22, 0.015, 0.22]),
        ]
        if th["protrusion_present"]:
            boxes.append((th["protr_center"], th["protr_half"]))
        self._stash_aabbs(th, boxes)
        if not hasattr(self, "_light_base"):
            self._light_base = m.light_diffuse.copy()
            self._headlight_base = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        m.light_diffuse[:] = self._light_base * th["light_scale"]
        m.vis.headlight.diffuse[:] = self._headlight_base[0] * th["light_scale"]
        m.vis.headlight.ambient[:] = self._headlight_base[1] * max(th["light_scale"], 0.15)
        mujoco.mj_forward(m, d)

    def _obj_rest(self):
        th = getattr(self, "_theta", None)
        if not th:
            return (0.62, 0.0, CUB_FLOOR_Z)
        y = float(np.random.uniform(-0.18, 0.18))
        if th.get("protrusion_present"):
            dy = th["protr_center"][1]
            # feasible cells: snug beside the divider (skin engagement). Abort cells: keep the
            # object clear — settle against the divider face fails and silently kills the
            # episode (that's why cubby batches had zero aborts).
            res = float(th.get("residual_margin", 0.01))
            off = 0.075 if res >= 0.004 else 0.13
            y = dy + (off if np.random.random() < 0.5 else -off)
            y = float(np.clip(y, -0.22, 0.22))
        return (float(np.random.uniform(0.50, 0.74)), y, CUB_FLOOR_Z)


class CubbyExpertPolicy(EnclosureExpertPolicy):
    """Over-the-wall expert: arc over the front wall, descend to the target, pick, lift out.
    Speed modulated by the LIVE margin (wall lip / divider / box walls). Obstacles visible,
    so infeasible compartments abort at PLAN time (observation-realizable: camera sees it)."""

    # topdown pose: fingertips AT the TCP, housing extends UP; fingers spread in y
    ENV_LO = np.array([0.050, 0.090, 0.0])
    ENV_HI = np.array([0.050, 0.090, 0.155])

    def _protrusion_detected(self) -> bool:
        return False   # divider handled at plan time (visible); stall gate still active

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        self._detected = True   # nothing hidden here
        self._logged_close = False
        th = self._theta()
        c = float(th.get("clearance", 0.04))
        wall_top = float(th.get("wall_top", CUB_FLOOR_Z + 0.2))
        tgt = (np.linalg.inv(self._embed_T()) @ np.append(self._target_pos(), 1.0))[:3]  # LOCAL
        start = self._tcp_now()
        over_z = wall_top + 0.02 + 0.4 * c    # tight lip crossing: 2-5.2cm — the skin reads it
        p_pre_l = np.array([0.30, 0.0, over_z + 0.10])
        gz = float(tgt[2]) + 0.012
        # descent column: nominally over the target, but if a feasible divider stands beside it,
        # shift the column AWAY from the divider face (sensor-driven deflection) so the wrist
        # keeps clearance while still reaching the object — behavior 'deflect'.
        col_y = float(tgt[1])
        if th.get("protrusion_present"):
            residual = float(th.get("residual_margin", c - float(th.get("intrusion", 0.0))))
            div_y = float(th["protr_center"][1])
            if residual < 0.004:
                self.behavior_class = "abort"
                p_over_ab_l = np.array([0.41, float(tgt[1]), over_z])
                probe_l = np.array([0.50, float(tgt[1]), over_z])
                v_in = self._v(self._seg_margin(p_pre_l, p_over_ab_l))
                log.info("[CubbyExpert] plan-time ABORT: divider residual "
                         f"{residual * 100:.1f}cm — approach, inspect, retreat")
                return [
                    GripperAction(self.robot_view, True, 0.0),
                    self._seq([
                        TCPMoveSegment(name="approach", start_pose=start, end_pose=self._P(p_pre_l, R_TOPDOWN), speed=self.SPEED_FAST),
                        TCPMoveSegment(name="insert", start_pose=self._P(p_pre_l, R_TOPDOWN), end_pose=self._P(probe_l, R_TOPDOWN), speed=v_in),
                        TCPMoveSegment(name="retreat", start_pose=self._P(probe_l, R_TOPDOWN), end_pose=self._P(p_pre_l, R_TOPDOWN), speed=0.06),
                    ]),
                ]
            # feasible divider beside the target: deflect the descent column to the far side
            self.behavior_class = "deflect"
            away = 1.0 if col_y >= div_y else -1.0
            col_y = float(np.clip(col_y + away * 0.02, -0.22, 0.22))
        p_over_l = np.array([0.41, col_y, over_z])
        p_above_l = np.array([float(tgt[0]), col_y, over_z])
        grasp_l = np.array([float(tgt[0]), float(tgt[1]), gz])
        p_pre, p_over, p_above, grasp = (self._P(p_pre_l, R_TOPDOWN), self._P(p_over_l, R_TOPDOWN),
                                         self._P(p_above_l, R_TOPDOWN), self._P(grasp_l, R_TOPDOWN))
        self._retreat_pose = p_over
        v_ins = self._v(self._seg_margin(p_pre_l, p_over_l))
        v_adv = self._v(self._seg_margin(p_over_l, p_above_l))
        v_gr = max(self._v(self._seg_margin(p_above_l, grasp_l)) * 0.6, self.SPEED_MIN)
        return [
            GripperAction(self.robot_view, True, 0.0),
            self._seq([
                TCPMoveSegment(name="approach", start_pose=start, end_pose=p_pre, speed=self.SPEED_FAST),
                TCPMoveSegment(name="insert", start_pose=p_pre, end_pose=p_over, speed=v_ins),
                TCPMoveSegment(name="advance", start_pose=p_over, end_pose=p_above, speed=v_adv),
                TCPMoveSegment(name="grasp", start_pose=p_above, end_pose=grasp, speed=v_gr),
            ]),
            TaskSpaceServo(self.robot_view, self._tcp_to_jp_fn, self._tcp_now, grasp, name="grasp"),
            GripperAction(self.robot_view, False, self.policy_config.gripper_close_duration),
            self._seq([
                TCPMoveSegment(name="extract", start_pose=grasp, end_pose=p_above, speed=v_gr),
                TCPMoveSegment(name="lift", start_pose=p_above, end_pose=p_over, speed=v_adv),
            ], holding=True),
        ]


class CubbyExpertPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = CubbyExpertPolicy


class TourFumehoodSampler(FumehoodSampler):
    """Fumehood at the MCAP-TOUR geometry — bench top 0.585 m, mouth at x=0.35, robot on the
    floor — the workspace with PROVEN deep insertions (52 cm TCP in the variation suite). The
    task frame constants are module-level; one datagen process runs one sampler type, so we
    rebase them at construction. Pair with fumehood_tour.xml and a near-zero base_size."""

    X0 = 0.35
    Z0 = 0.585

    def __init__(self, config) -> None:
        import molmo_spaces.tasks.enclosure_reach as er
        er.TUBE_X0 = self.X0
        er.SHELF_TOP_Z = self.Z0
        super().__init__(config)


class BigFumehoodPickSampler(FumehoodSampler):
    """BIG-opening fumehood, CLEAN PICK (sweep-scale aperture, no obstacles). The standard
    FumehoodSampler mixture is mostly abort/deflect + a tight door, so almost nothing grasps.
    Here: every episode is a free clean pick, the sash/jambs open to the sweep's big-hood scale
    so the whole arm enters, and the object sits within the FR3's reliable reach so the gripper
    actually closes on it. Pairs with fumehood.xml (mocap sash/jambs posed by _apply_theta)."""

    MIXTURE = (("free", 1.0),)
    # The enclosure base is pinned at world origin (CavityPick BASE_XYZ + the per-episode
    # jitter in EnclosureReachSampler._sample_task), which leaves the object at world x~0.7-0.87
    # AT/PAST the FR3's reach: the grasp servo's integral correction saturates at its +-0.14
    # clamp and the gripper still ends ~10 cm high / off-center, closing on air (0% success).
    # Like the standard pick samplers, place the base NEAR the object instead -- nudge it forward
    # into the hood mouth so the object lands in the arm's reliable-grasp envelope.
    BASE_FWD = 0.08  # small forward nudge: base-origin gave the best grasp posture; 0.18 hurt z

    # Everyday graspable categories, GRASP-FILE-VALIDATED: PickPlannerPolicy executes annotated
    # grasp poses from each object's grasp file (handle/rim grasps for mugs that are wider than
    # the 85 mm finger gap), so the pool only needs feasible-grasp annotations + a sane size cap.
    # Mugs/cups are listed first so house 0 picks a mug.
    PICK_CATEGORIES = ("mug", "cup", "apple", "tomato", "potato", "orange", "pear",
                       "peach", "lemon", "can")
    EXCLUDED_TARGET_CATEGORIES = ("egg", "candle", "can opener", "canister")

    def _build_grasp_uid_pool(self, n: int) -> list[str]:
        from molmo_spaces.utils.grasp_sample import has_grasp_folder, has_valid_grasp_file
        mugs, rest = [], []
        for uid in get_valid_pickupable_obja_uids():
            anno = ObjectMeta.annotation(uid) or {}
            cat = str(anno.get("category", "")).lower()
            if any(x in cat for x in self.EXCLUDED_TARGET_CATEGORIES):
                continue
            if not any(c in cat for c in self.PICK_CATEGORIES):
                continue
            bb = anno.get("boundingBox", {})
            dims = sorted(float(bb.get(k, 0)) for k in "xyz")
            if dims[0] < 0.03 or dims[2] > 0.18:
                continue
            try:
                if not (has_grasp_folder(uid) and has_valid_grasp_file(uid)):
                    continue
            except Exception:
                continue
            (mugs if ("mug" in cat or "cup" in cat) else rest).append(uid)
            if len(mugs) + len(rest) >= n * 3:
                break
        pool = (mugs + rest)[:n]
        if pool:
            log.info(f"[BigFumehoodPick] grasp-validated pool ({len(pool)}, "
                     f"{len(mugs)} mug/cup): {pool}")
            return pool
        log.warning("[BigFumehoodPick] category pool EMPTY -- falling back to default pool")
        return super()._build_grasp_uid_pool(n)

    def _draw_theta(self):
        th = super()._draw_theta()
        th["protrusion_present"] = False
        th["ap_w"] = float(np.random.uniform(0.50, 0.85))   # wide jambs   (sweep big-hood scale)
        th["ap_h"] = float(np.random.uniform(0.45, 0.62))   # high sash    (big opening height)
        th["depth"] = float(np.random.uniform(0.18, 0.26))  # shallow hood -> object reachable
        th["target_frac"] = float(np.random.uniform(0.45, 0.65))
        th["clearance"] = float(np.random.uniform(0.06, 0.10))
        th["grasp_pitch_deg"] = 50.0  # steep down-pitch so the arm descends onto the object
        return th

    def _obj_rest(self):
        x, y, z = super()._obj_rest()
        th = getattr(self, "_theta", None) or {}
        # OFF-CENTERLINE placement: an object dead-ahead sits in the robot's direct approach
        # line (gripper clipped it and dropped it); offset it to one side so the planner
        # approaches at an angle. Stay clear of the jambs (ap_w/2 minus margin).
        y_hi = max(0.12, min(0.20, float(th.get("ap_w", 0.6)) / 2 - 0.08))
        y_off = float(np.random.choice([-1.0, 1.0]) * np.random.uniform(0.10, y_hi))
        # keep the object just inside the mouth so the (forward-shifted) base reaches it cleanly
        return (float(min(x, TUBE_X0 + 0.10)), y_off, z)

    def _sample_and_place_robot(self, env):
        # _sample_task has just pinned _cur_base_xyz to ~origin; shift it forward into the hood
        # so the FR3 grasps from a mid-workspace pose (un-saturates IK) instead of full extension.
        bx, by, bz = self._cur_base_xyz
        self._cur_base_xyz = (bx + self.BASE_FWD, by, bz)
        return super()._sample_and_place_robot(env)


class ObstacleFumehoodPickSampler(BigFumehoodPickSampler):
    """Big-opening fumehood pick WITH a hazard bar standing on the bench beside the
    approach corridor. Purpose: saturate the 3-15 cm proximity band (the steering band
    a safety head trains on) while the grasp-file pick machinery keeps succeeding.

    Bar placement is COUPLED to the object draw: _obj_rest puts the object on the bar's
    side of the corridor a controlled lateral gap from the bar's inner face, so the
    wrist/gripper passes the bar at ~4-12 cm on every obstacle episode instead of the
    free-pick ~50 cm. Pair with ObstacleAwarePickPlannerPolicy, which reads the bar AABB
    from scene_params and bows the approach away from it (a visible 'veer' in demos).
    Free episodes (1 - OBSTACLE_P) reproduce the clean BigFumehoodPick behavior."""

    OBSTACLE_P = 0.75            # fraction of episodes with the bar present
    BAR_FACE_Y = (0.14, 0.24)    # |y| of the bar's inner (corridor-side) face
    BAR_X_FRAC = (0.20, 0.55)    # bar depth into the hood, fraction of hood depth
    # lateral gap bar face -> object center. With the policy's GRIP_HALF=0.10 and
    # SAFE_GAP=0.08 this puts the straight-line surface clearance at 2-11 cm, so
    # roughly 2/3 of bar episodes trigger a visible deflection and the rest pass
    # close without one — both modes appear in the ACT data.
    OBJ_GAP = (0.12, 0.20)

    # Manifest override (hybrid_obstacle_independent_v2). None means legacy: the
    # runtime Bernoulli below is drawn exactly as it always was. Only the
    # manifest runner ever sets this, and only for the manifest config, so every
    # legacy obstacle config keeps its OBSTACLE_P behavior byte for byte.
    _forced_hazard_present: bool | None = None
    _manifest_row: dict | None = None

    def set_manifest_row(self, row: dict, retry_index: int = 0) -> None:
        """Pin this episode's hazard assignment to a committed manifest row.

        Hazard presence then comes only from ``row['hazard_present']`` and the
        runtime Bernoulli is bypassed, so it can no longer depend on how many
        draws the worker has already consumed.
        """
        if "hazard_present" not in row:
            raise ValueError("manifest row is missing 'hazard_present'")
        self._manifest_row = row
        self._forced_hazard_present = bool(row["hazard_present"])
        self._manifest_retry_index = int(retry_index)

    def clear_manifest_row(self) -> None:
        """Restore legacy Bernoulli behavior."""
        self._manifest_row = None
        self._forced_hazard_present = None

    def _hazard_present_for_episode(self) -> bool:
        if self._forced_hazard_present is None:
            return bool(np.random.random() < self.OBSTACLE_P)
        return self._forced_hazard_present

    def _draw_theta(self):
        th = super()._draw_theta()   # Big's clean-pick draws (protrusion forced off)
        # near-constant lighting: one-env ACT data should not fight 10x brightness swings
        th["light_scale"] = float(np.random.uniform(0.75, 1.10))
        if self._hazard_present_for_episode():
            # cell name outside hidden/visible skips _sample_task's raycast rejection
            # loop (which would redraw the coupled placement fields drawn here)
            th["cell"] = "bar"
            th["protrusion_present"] = True
            th["protr_name"] = str(np.random.choice(list(PROTR.keys())))
            th["protr_wall"] = str(np.random.choice(["left", "right"]))
            th["protr_pos_frac"] = float(np.random.uniform(*self.BAR_X_FRAC))
            face = float(np.random.uniform(*self.BAR_FACE_Y))
            th["bar_face_y"] = face
            # FumehoodSampler._apply_theta puts the inner face at side*(ap_w/2 - intrusion)
            th["intrusion"] = float(th["ap_w"] / 2 - face)
            th["residual_margin"] = float(th["clearance"] - th["intrusion"])
            th["obj_gap"] = float(np.random.uniform(*self.OBJ_GAP))
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        # hazard-orange bars: visually distinct for the cameras (and the advisor)
        m = env.current_model
        for name in PROTR:
            try:
                m.geom_rgba[m.geom(f"{name}_g").id] = (1.0, 0.45, 0.05, 1.0)
            except Exception:
                pass

    def _obj_rest(self):
        x, y, z = super()._obj_rest()
        th = getattr(self, "_theta", None) or {}
        if th.get("protrusion_present"):
            side = 1.0 if th["protr_wall"] == "left" else -1.0
            y = float(side * (th["bar_face_y"] - th["obj_gap"]))
        return (x, y, z)


class ObstacleFumehoodPickCheckSampler(ObstacleFumehoodPickSampler):
    """Preflight variant: bar present on EVERY episode."""

    OBSTACLE_P = 1.0


from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (  # noqa: E402
    PickPlannerPolicy,
)


class ObstacleAwarePickPlannerPolicy(PickPlannerPolicy):
    """PickPlannerPolicy + a deflection around the episode's bar: if the straight
    start->pregrasp TCP line passes the bar's inner face closer than the gripper
    envelope + a safety gap, the approach is rebuilt with two waypoints bracketing
    the bar, bowed away from it laterally, at a cautious passing speed. Episodes
    without a bar (or with a naturally-clearing line) keep the parent's exact plan,
    so this is a strict superset of the proven pick behavior."""

    GRIP_HALF = 0.10     # open-gripper lateral half-extent around the TCP line
    SAFE_GAP = 0.08      # surface clearance the deflection enforces beyond GRIP_HALF
    PASS_SPEED = 0.05    # m/s while alongside the bar (between the deflect waypoints)

    def __init__(self, config, task) -> None:
        super().__init__(config, task)
        self.behavior_class = "free"

    @staticmethod
    def _pose(p: np.ndarray, R: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        prims = super()._compute_trajectory()
        th = getattr(self.task, "scene_params", {}) or {}
        if not th.get("protrusion_present") or "protr_center" not in th:
            return prims
        c = np.asarray(th["protr_center"], dtype=float)
        h = np.asarray(th["protr_half"], dtype=float)
        approach = prims[1]
        seg_pre, seg_grasp = approach._move_segments[0], approach._move_segments[1]
        p0 = seg_pre.start_pose[:3, 3].copy()
        p1 = seg_pre.end_pose[:3, 3].copy()
        if abs(p1[0] - p0[0]) < 1e-6:
            return prims
        t_bar = (c[0] - p0[0]) / (p1[0] - p0[0])
        if not (0.02 < t_bar < 0.98):
            return prims                       # approach never crosses the bar's x-station
        cross = p0 + t_bar * (p1 - p0)
        side = 1.0 if c[1] >= 0.0 else -1.0
        face_y = c[1] - side * h[1]            # inner (corridor-side) face of the bar
        clear = side * (face_y - cross[1]) - self.GRIP_HALF
        need = self.SAFE_GAP - clear
        if need <= 0.0:
            return prims                       # straight line already clears the bar
        self.behavior_class = "deflect"
        ap_w = float(th.get("ap_w", 0.6))
        y_wp = float(np.clip(cross[1] - side * need, -(ap_w / 2 - 0.12), ap_w / 2 - 0.12))
        # waypoints bracketing the bar along x, bowed to y_wp, z on the original line
        x_lo, x_hi = sorted((p0[0], p1[0]))
        xa = float(np.clip(c[0] - (h[0] + 0.10), x_lo + 0.01, x_hi - 0.02))
        xb = float(np.clip(c[0] + (h[0] + 0.08), xa + 0.01, x_hi - 0.01))
        line_z = lambda x: p0[2] + (x - p0[0]) / (p1[0] - p0[0]) * (p1[2] - p0[2])  # noqa: E731
        R = seg_pre.end_pose[:3, :3]
        wp_a = self._pose(np.array([xa, y_wp, line_z(xa)]), R)
        wp_b = self._pose(np.array([xb, y_wp, line_z(xb)]), R)
        log.info(f"[ObstaclePick] DEFLECT: bar face y={face_y:+.3f}, line clear "
                 f"{clear * 100:.1f}cm -> bow {need * 100:.1f}cm to y={y_wp:+.3f}")
        robot_view = self.task.env.current_robot.robot_view
        prims[1] = TCPMoveSequence(
            robot_view,
            self._tcp_to_jp_fn,
            self.policy_config.move_settle_time,
            gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
            tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
            tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
            move_segments=[
                TCPMoveSegment(name="pregrasp", start_pose=seg_pre.start_pose,
                               end_pose=wp_a, speed=self.policy_config.speed_fast),
                TCPMoveSegment(name="pregrasp", start_pose=wp_a,
                               end_pose=wp_b, speed=self.PASS_SPEED),
                TCPMoveSegment(name="pregrasp", start_pose=wp_b,
                               end_pose=seg_pre.end_pose, speed=self.policy_config.speed_slow),
                seg_grasp,
            ],
        )
        return prims


class ObstacleAwarePickPlannerPolicyConfig(PickPlannerPolicyConfig):
    """Wires ObstacleAwarePickPlannerPolicy as the rollout policy."""

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = ObstacleAwarePickPlannerPolicy


class PactCollisionCorridorSampler(BigFumehoodPickSampler):
    """Alternating hidden-from-wrist intrusion in the wrist/link-6 passage.

    The manipulation target is sampled independently near the hood centre. A
    committed row pins whether the overhead panel enters from the left or right.
    The panels have identical matte appearance and only the active panel is posed
    in the work volume. Their geometry is sensed by the link-5/link-6 skin.
    """

    PANEL_X = 0.615
    PANEL_Z = 0.89
    PANEL_HALF = np.array([0.055, 0.240, 0.090], dtype=float)
    PANEL_INNER_FACE_Y = 0.100
    SASH_APERTURE_HEIGHT = 0.70
    TARGET_UID = "Cup_10"
    BASE_FWD = 0.14
    _pact_manifest_row: dict | None = None

    def _build_grasp_uid_pool(self, n: int) -> list[str]:
        # Cup_10 is a grasp-validated 7.0 x 7.3 cm cup, safely inside the
        # Franka's 8.5 cm finger span. The previous 10.1 cm cup slipped during
        # lift in 2/8 development rows and made task solvability, rather than
        # surface avoidance, the limiting variable.
        return [self.TARGET_UID] * int(n)

    def set_pact_manifest_row(self, row: dict) -> None:
        side = str(row.get("intrusion_side", ""))
        if side not in {"left", "right"}:
            raise ValueError(f"intrusion_side must be left or right, got {side!r}")
        self._pact_manifest_row = dict(row)

    def _draw_theta(self):
        th = super()._draw_theta()
        # Keep the opposite-side detour physically open and identical across
        # instances; task variation comes from target y, panel jitter, and side.
        th["ap_w"] = 0.85
        th["ap_h"] = self.SASH_APERTURE_HEIGHT
        row = self._pact_manifest_row
        side_name = (
            str(row["intrusion_side"])
            if row is not None
            else ("left" if np.random.random() < 0.5 else "right")
        )
        side = 1.0 if side_name == "left" else -1.0
        x_jitter = float(row.get("panel_x_jitter_m", 0.0)) if row is not None else 0.0
        face_jitter = (
            float(row.get("panel_face_jitter_m", 0.0)) if row is not None else 0.0
        )
        face = float(self.PANEL_INNER_FACE_Y + face_jitter)
        center = [
            float(self.PANEL_X + x_jitter),
            float(side * (face + self.PANEL_HALF[1])),
            float(self.PANEL_Z),
        ]
        th.update(
            {
                "cell": "pact_collision_corridor",
                "protrusion_present": True,
                "protr_wall": side_name,
                "protr_name": f"pact_intrusion_{side_name}",
                "protr_center": center,
                "protr_half": self.PANEL_HALF.tolist(),
                "pact_intrusion_side": side_name,
                "pact_panel_inner_face_y_m": face,
                "pact_environment_version": "pact_collision_corridor_v1",
                "light_scale": 1.0,
            }
        )
        return th

    def _apply_theta(self, env, th):
        # Reuse the validated fumehood shell placement, but prevent it from
        # interpreting the custom panel name as one of its legacy PROTR bars.
        active = bool(th["protrusion_present"])
        th["protrusion_present"] = False
        FumehoodSampler._apply_theta(self, env, th)
        th["protrusion_present"] = active

        for name, y in (
            ("pact_intrusion_left", 1.8),
            ("pact_intrusion_right", -1.8),
        ):
            self._mocap_set(env, name, [0.0, y, -2.0])
        self._mocap_set(env, th["protr_name"], th["protr_center"])
        th["obstacle_aabbs"].append(
            [list(map(float, th["protr_center"])), list(map(float, th["protr_half"]))]
        )
        mujoco.mj_forward(env.current_model, env.current_data)

    def _obj_rest(self):
        # Independent of intrusion side: target pixels cannot leak the route.
        return (
            float(TUBE_X0 + 0.18),
            float(np.random.uniform(-0.04, 0.04)),
            float(SHELF_TOP_Z),
        )


class PactCollisionCorridorControlSampler(PactCollisionCorridorSampler):
    """Fresh in-distribution control for the geometry-generalization study."""

    PACT_GEOMETRY_CONDITION = "C0"
    APERTURE_WIDTH = 0.85

    def _draw_theta(self):
        th = super()._draw_theta()
        th["ap_w"] = float(self.APERTURE_WIDTH)
        th["pact_geometry_condition"] = self.PACT_GEOMETRY_CONDITION
        th["pact_geometry_generalization_version"] = "v1"
        return th


class PactCollisionCorridorDeeperHigherSampler(
    PactCollisionCorridorControlSampler
):
    """C1: a panel deeper in the corridor and higher above the bench."""

    PACT_GEOMETRY_CONDITION = "C1"
    PANEL_X = 0.68
    PANEL_Z = 0.96


class PactCollisionCorridorTighterSampler(PactCollisionCorridorControlSampler):
    """C2: a farther-intruding panel with a tighter vertical aperture."""

    PACT_GEOMETRY_CONDITION = "C2"
    PANEL_INNER_FACE_Y = 0.070
    APERTURE_WIDTH = 0.70


class PactCollisionCorridorShallowerWiderSampler(
    PactCollisionCorridorControlSampler
):
    """C3: a shallower panel paired with a wider vertical aperture."""

    PACT_GEOMETRY_CONDITION = "C3"
    PANEL_X = 0.55
    APERTURE_WIDTH = 1.00


class PactCollisionCorridorPanelX058Sampler(PactCollisionCorridorControlSampler):
    """Attempt-2 envelope candidate: panel x-position 0.58 m."""

    PACT_GEOMETRY_CONDITION = "X_058"
    PANEL_X = 0.58


class PactCollisionCorridorPanelX065Sampler(PactCollisionCorridorControlSampler):
    """Attempt-2 envelope candidate: panel x-position 0.65 m."""

    PACT_GEOMETRY_CONDITION = "X_065"
    PANEL_X = 0.65


class PactCollisionCorridorPanelZ085Sampler(PactCollisionCorridorControlSampler):
    """Attempt-2 envelope candidate: panel height 0.85 m."""

    PACT_GEOMETRY_CONDITION = "Z_085"
    PANEL_Z = 0.85


class PactCollisionCorridorPanelZ093Sampler(PactCollisionCorridorControlSampler):
    """Attempt-2 envelope candidate: panel height 0.93 m."""

    PACT_GEOMETRY_CONDITION = "Z_093"
    PANEL_Z = 0.93


class PactCollisionCorridorPanelHalfY018Sampler(
    PactCollisionCorridorControlSampler
):
    """Attempt-2 envelope candidate: lateral half-extent 0.18 m."""

    PACT_GEOMETRY_CONDITION = "HALF_Y_018"
    PANEL_HALF = np.array([0.055, 0.180, 0.090], dtype=float)


class PactCollisionCorridorPanelHalfY030Sampler(
    PactCollisionCorridorControlSampler
):
    """Attempt-2 envelope candidate: lateral half-extent 0.30 m."""

    PACT_GEOMETRY_CONDITION = "HALF_Y_030"
    PANEL_HALF = np.array([0.055, 0.300, 0.090], dtype=float)


class PactCollisionCorridorAperture095Sampler(PactCollisionCorridorControlSampler):
    """Attempt-2 envelope candidate: aperture width 0.95 m."""

    PACT_GEOMETRY_CONDITION = "AP_W_095"
    APERTURE_WIDTH = 0.95

class PactCollisionCorridorPolicy(ObstacleAwarePickPlannerPolicy):
    """Privileged expert that bows away from the active overhead intrusion."""

    GRIP_HALF = 0.11
    # Remediation v2 raises the expert-only nominal clearance from 8 cm to
    # 10 cm.  The scene geometry is intentionally unchanged.
    SAFE_GAP = 0.10
    PASS_SPEED = 0.045

    def reset(self, reset_retries: bool = True):
        from molmo_spaces.tasks.pact_contact_audit import PactContactAudit

        if reset_retries or not hasattr(self, "_pact_contact_audit"):
            self._pact_contact_audit = PactContactAudit()
            self._pact_control_step = 0
        self.task._contact_audit_hook = self._pact_contact_audit
        return super().reset(reset_retries)

    def get_action(self, observation):
        self._pact_contact_audit.observe(self.task.env, self._pact_control_step)
        action = super().get_action(observation)
        self._pact_control_step += 1
        return action

    def get_info(self):
        self._pact_contact_audit.observe(self.task.env, self._pact_control_step)
        info = super().get_info()
        info["pact_contact_audit"] = self._pact_contact_audit.summary()
        return info


class PactCollisionCorridorPolicyConfig(PickPlannerPolicyConfig):
    """Wires the collision-corridor expert into normal datagen."""

    # Retrying after rollout start calls reset() from a moved robot/object
    # state, which can turn a scientific motion failure into an uncaught
    # trajectory-construction IK exception.  Initial reset/trajectory
    # construction is already retried by the manifest runner before the
    # outcome-bearing boundary.  Once a rollout starts, terminate any planner
    # failure as a normal task failure and retain that row's outcome.
    max_retries: int = 0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PactCollisionCorridorPolicy


# -------------------------------------------------------------------------------------- #
# Forked pick-and-place corridor. The original corridor sampler and expert above remain
# byte-identical at the class-source level; all stronger-task behavior is additive here.
# -------------------------------------------------------------------------------------- #
from molmo_spaces.configs.policy_configs import (  # noqa: E402
    PickAndPlacePlannerPolicyConfig,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (  # noqa: E402
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E402
    JointMoveSegment,
    JointMoveSequence,
    NoopAction,
)
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask  # noqa: E402
from molmo_spaces.utils.linalg_utils import transform_to_twist, twist_to_transform  # noqa: E402
from molmo_spaces.utils.pose import pos_quat_to_pose_mat  # noqa: E402


class PactPlaceCorridorTask(PickAndPlaceTask):
    """Upstream support-and-release success criterion plus corridor metadata."""

    def get_obs_scene(self) -> dict[str, Any]:
        scene = super().get_obs_scene()
        scene["scene_params"] = getattr(self, "scene_params", {})
        return scene


class PactPlaceCorridorSampler(PactCollisionCorridorSampler):
    """Forked corridor whose target must be placed on the outside tray."""

    PLACE_RECEPTACLE_NAME = "place_receptacle"
    PLACE_RECEPTACLE_START_POSE = [0.35, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    PLACE_TRAY_X_BOUNDS = (0.25, 0.45)
    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v1"

    def _draw_theta(self):
        th = super()._draw_theta()
        th.update(
            {
                "pact_place_environment_version": self.PACT_PLACE_ENVIRONMENT_VERSION,
                "place_receptacle_name": self.PLACE_RECEPTACLE_NAME,
                "place_receptacle_start_pose": list(self.PLACE_RECEPTACLE_START_POSE),
                "place_tray_x_bounds_m": list(self.PLACE_TRAY_X_BOUNDS),
            }
        )
        return th

    def _sample_task(self, env: CPUMujocoEnv):
        task_config = self.config.task_config
        task_config.place_receptacle_name = self.PLACE_RECEPTACLE_NAME
        task_config.place_receptacle_start_pose = list(
            self.PLACE_RECEPTACLE_START_POSE
        )
        task_config.referral_expressions.setdefault("place_name", "blue tray")
        task = super()._sample_task(env)
        task_config.referral_expressions.setdefault(
            "pickup_name",
            task_config.referral_expressions.get("pickup_obj_name", "cup"),
        )
        task.__class__ = PactPlaceCorridorTask
        task._supported_rel_poses = {}
        task.scene_params = dict(getattr(task, "scene_params", {}) or {})
        task.scene_params["task_success_criterion"] = (
            "PickAndPlaceTask.supported_released_receptacle_stable"
        )
        return task


class PactPlaceCorridorV2Sampler(PactPlaceCorridorSampler):
    """Same corridor; receptacle translated (and optionally shrunk) out of the
    arm's inbound/outbound sweep. Scene XML is ``pact_place_corridor_v2.xml``.
    Constants are filled from the A0 reachability sweep.
    """

    PLACE_RECEPTACLE_START_POSE = [0.35, 0.32, 0.0, 1.0, 0.0, 0.0, 0.0]
    PLACE_TRAY_X_BOUNDS = (0.25, 0.45)
    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v2"


class PactPlaceCorridorV3Sampler(PactPlaceCorridorV2Sampler):
    """v2 tray and corridor; four immovable shelf-clutter mocap boxes.

    Slot XY is filled from the A0 clutter sweep. Jitter is drawn per row from
    ``task_seed_u64`` and is independent of ``intrusion_side``.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v3"
    CLUTTER_BODY_NAMES = (
        "pact_clutter_l0",
        "pact_clutter_l1",
        "pact_clutter_r0",
        "pact_clutter_r1",
    )
    CLUTTER_SLOT_NOMINAL_XY = {
        "l0": (0.70, 0.34),
        "l1": (0.75, 0.34),
        "r0": (0.70, -0.34),
        "r1": (0.75, -0.34),
    }
    CLUTTER_HEIGHT_M = 0.10
    CLUTTER_HALF_X_M = 0.025
    CLUTTER_HALF_Y_M = 0.05
    CLUTTER_TOP_Z_M = 0.82
    CLUTTER_PARK_XYZ_M = (0.0, 0.0, -2.0)
    CLUTTER_POOL_BODY_NAMES = CLUTTER_BODY_NAMES

    def _clutter_slots(self, row: dict | None) -> dict[str, dict[str, Any]]:
        x_jitter = (row or {}).get("clutter_x_jitter_m") or {}
        y_jitter = (row or {}).get("clutter_y_jitter_m") or {}
        slots: dict[str, dict[str, Any]] = {}
        nominal_xyz = getattr(self, "CLUTTER_SLOT_NOMINAL", None)
        if nominal_xyz is not None:
            for slot, spec in nominal_xyz.items():
                nx, ny, nz = (float(v) for v in spec["center_m"])
                hx, hy, hz = (float(v) for v in spec["half_m"])
                jx = float(x_jitter.get(slot, 0.0))
                jy = float(y_jitter.get(slot, 0.0))
                slots[slot] = {
                    "body": f"pact_clutter_{slot}",
                    "nominal_xyz_m": [nx, ny, nz],
                    "jitter_xy_m": [jx, jy],
                    "center_m": [nx + jx, ny + jy, nz],
                    "half_m": [hx, hy, hz],
                    "support": spec.get("support"),
                    "size_name": spec.get("size_name"),
                }
            return slots
        height = float(self.CLUTTER_HEIGHT_M)
        half_x = float(self.CLUTTER_HALF_X_M)
        half_y = float(self.CLUTTER_HALF_Y_M)
        half_z = height / 2.0
        for slot, (nx, ny) in self.CLUTTER_SLOT_NOMINAL_XY.items():
            jx = float(x_jitter.get(slot, 0.0))
            jy = float(y_jitter.get(slot, 0.0))
            center = [float(nx + jx), float(ny + jy), float(SHELF_TOP_Z + half_z)]
            slots[slot] = {
                "body": f"pact_clutter_{slot}",
                "nominal_xy_m": [float(nx), float(ny)],
                "jitter_xy_m": [jx, jy],
                "center_m": center,
                "half_m": [half_x, half_y, half_z],
            }
        return slots

    def _set_clutter_geom_size(self, env, slots: dict[str, dict[str, Any]]) -> None:
        model = env.current_model
        for slot, spec in slots.items():
            geom = model.geom(f"pact_clutter_{slot}_g")
            model.geom_size[int(geom.id)] = np.asarray(spec["half_m"], dtype=float)

    def _draw_theta(self):
        th = super()._draw_theta()
        slots = self._clutter_slots(self._pact_manifest_row)
        th.update(
            {
                "pact_place_environment_version": self.PACT_PLACE_ENVIRONMENT_VERSION,
                "pact_clutter": slots,
                "pact_clutter_height_m": float(self.CLUTTER_HEIGHT_M),
                "pact_clutter_half_x_m": float(self.CLUTTER_HALF_X_M),
                "pact_clutter_half_y_m": float(self.CLUTTER_HALF_Y_M),
                "pact_clutter_top_z_m": float(self.CLUTTER_TOP_Z_M),
            }
        )
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        slots = th.get("pact_clutter") or self._clutter_slots(self._pact_manifest_row)
        posed: set[str] = set()
        for spec in slots.values():
            body = spec["body"]
            if any(
                forbidden in body
                for forbidden in ("cavity_obj_", "pact_intrusion_", "place_receptacle")
            ):
                raise ValueError(f"illegal clutter body name {body!r}")
            self._mocap_set(env, body, spec["center_m"])
            posed.add(body)
        for body in self.CLUTTER_POOL_BODY_NAMES:
            if body not in posed:
                self._mocap_set(env, body, list(self.CLUTTER_PARK_XYZ_M))
        if slots:
            self._set_clutter_geom_size(env, slots)
        mujoco.mj_forward(env.current_model, env.current_data)


PACT_PLACE_V104_ENVIRONMENT_VERSION = (
    "pact_place_corridor_v10_4_first_shot_static_pendant"
)


class PactPlaceCorridorV104Sampler(PactPlaceCorridorV3Sampler):
    """V6c sampling, verbatim, plus the V10.4 environment marker.

    A thin pass-through. It changes no clutter slot, jitter, panel behaviour,
    target/tray geometry, camera, or contact semantics; the compiled-static
    pendant lives in the scene XML, not in sampling. The marker exists so the
    single registered speed amendment and the V10.4 telemetry can be gated on
    it, and so no historical environment can reach either.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = PACT_PLACE_V104_ENVIRONMENT_VERSION

    def _draw_theta(self):
        th = super()._draw_theta()
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_v104_static_pendant"] = True
        th["pact_v104_pendant_body"] = "pact_clutter_mount_v104"
        return th


class PactPlaceCorridorV4Sampler(PactPlaceCorridorV3Sampler):
    """v3 corridor and tray; 16-body clutter pool with a 13-slot lattice.

    Lattice coordinates are the A0e admission set (clearance C = 0.030 m from
    the measured v6c swept volume). Unused pool bodies stay parked at z = -2.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v4"
    CLUTTER_POOL_BODY_NAMES = tuple(f"pact_clutter_{index:02d}" for index in range(16))
    CLUTTER_BODY_NAMES = tuple(f"pact_clutter_{index:02d}" for index in range(13))
    CLUTTER_SLOT_NOMINAL = {
        "00": {
            "center_m": (0.62, -0.385, 1.12),
            "half_m": (0.025, 0.02, 0.04),
            "support": "wall",
            "size_name": "wall_block",
        },
        "01": {
            "center_m": (0.62, -0.375, 1.385),
            "half_m": (0.04, 0.03, 0.015),
            "support": "ceiling",
            "size_name": "overhead_thin",
        },
        "02": {
            "center_m": (0.72, -0.385, 1.12),
            "half_m": (0.025, 0.02, 0.04),
            "support": "wall",
            "size_name": "wall_block",
        },
        "03": {
            "center_m": (0.72, -0.385, 1.24),
            "half_m": (0.025, 0.02, 0.04),
            "support": "wall",
            "size_name": "wall_block",
        },
        "04": {
            "center_m": (0.65, -0.385, 1.24),
            "half_m": (0.025, 0.02, 0.04),
            "support": "wall",
            "size_name": "wall_block",
        },
        "05": {
            "center_m": (0.72, 0.385, 1.12),
            "half_m": (0.025, 0.02, 0.04),
            "support": "wall",
            "size_name": "wall_block",
        },
        "06": {
            "center_m": (0.70, 0.385, 1.24),
            "half_m": (0.025, 0.02, 0.04),
            "support": "wall",
            "size_name": "wall_block",
        },
        "07": {
            "center_m": (0.72, -0.375, 1.385),
            "half_m": (0.04, 0.03, 0.015),
            "support": "ceiling",
            "size_name": "overhead_thin",
        },
        "08": {
            "center_m": (0.72, 0.385, 1.00),
            "half_m": (0.025, 0.02, 0.04),
            "support": "wall",
            "size_name": "wall_block",
        },
        "09": {
            "center_m": (0.70, -0.385, 1.00),
            "half_m": (0.025, 0.02, 0.04),
            "support": "wall",
            "size_name": "wall_block",
        },
        "10": {
            "center_m": (0.72, 0.375, 1.385),
            "half_m": (0.04, 0.03, 0.015),
            "support": "ceiling",
            "size_name": "overhead_thin",
        },
        "11": {
            "center_m": (0.62, 0.375, 1.385),
            "half_m": (0.04, 0.03, 0.015),
            "support": "ceiling",
            "size_name": "overhead_thin",
        },
        "12": {
            "center_m": (0.70, -0.38, 0.77),
            "half_m": (0.025, 0.025, 0.05),
            "support": "floor",
            "size_name": "floor_narrow",
        },
    }


class PactPlaceCorridorV5Sampler(PactPlaceCorridorV3Sampler):
    """v8 corridor with a frozen palette of Objaverse clutter.

    V3/V4 keep their mocap-box implementation unchanged.  V5 installs every
    palette UID through the same MjSpec path as the target. Prop slots remain
    free bodies; mount slots are jointless mocap bodies so overhead fixtures can
    be posed per episode without falling. Clutter is deliberately absent from
    ``obstacle_aabbs`` so the frozen expert neither replans nor changes its speed
    law in response to it.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v5"
    CLUTTER_PARK_XYZ_M = (4.0, 4.0, -2.0)
    CLUTTER_SETTLE_STEPS = 300
    CLUTTER_FREE_JOINT_DAMPING = 0.05
    CLUTTER_MAX_SETTLED_LINEAR_SPEED_M_S = 0.025
    CLUTTER_MAX_SETTLED_ANGULAR_SPEED_RAD_S = 0.25
    CLUTTER_MAX_SETTLED_XY_DRIFT_M = 0.005
    CLUTTER_CONTAINMENT_TOLERANCE_M = 1e-4
    LEGACY_CLUTTER_BODY_NAMES = PactPlaceCorridorV3Sampler.CLUTTER_BODY_NAMES

    def _palette(self) -> list[dict[str, Any]]:
        row = self._pact_manifest_row or {}
        palette = list(row.get("pact_clutter_palette") or [])
        if not palette:
            raise ValueError("v5 manifest row is missing pact_clutter_palette")
        slots = [str(item["slot"]) for item in palette]
        if len(slots) != len(set(slots)):
            raise ValueError("duplicate v5 clutter palette slot")
        if not 12 <= len(palette) <= 20:
            raise ValueError("v5 clutter palette must contain 12-20 objects")
        explicit_classes = [item.get("slot_class") for item in palette]
        if any(value is not None for value in explicit_classes):
            if any(value not in {"mount", "prop"} for value in explicit_classes):
                raise ValueError("v5 slot_class must be explicit mount or prop")
            mount_count = explicit_classes.count("mount")
            prop_count = explicit_classes.count("prop")
            if not 5 <= mount_count <= 6:
                raise ValueError("v5 palette must contain 5-6 mount slots")
            if not 12 <= prop_count <= 14:
                raise ValueError("v5 palette must contain 12-14 prop slots")
        return palette

    def _layout(self) -> dict[str, Any]:
        row = self._pact_manifest_row or {}
        layout = dict(row.get("pact_clutter_layout") or {})
        if not layout:
            raise ValueError("v5 manifest row is missing pact_clutter_layout")
        objects = list(layout.get("objects") or [])
        if not objects:
            raise ValueError("v5 clutter layout has no active objects")
        palette_by_slot = {str(item["slot"]): item for item in self._palette()}
        slots = [str(item.get("palette_slot", "")) for item in objects]
        if len(slots) != len(set(slots)):
            raise ValueError("v5 clutter layout activates a palette slot twice")
        for item, slot in zip(objects, slots):
            if slot not in palette_by_slot:
                raise ValueError(f"layout references unknown palette slot {slot!r}")
            if str(item.get("uid")) != str(palette_by_slot[slot]["uid"]):
                raise ValueError(f"layout uid does not match frozen palette slot {slot!r}")
            slot_class = str(palette_by_slot[slot].get("slot_class") or "prop")
            support = str(item.get("support") or "")
            if slot_class == "mount" and support != "overhead":
                raise ValueError(f"mount slot {slot!r} may only be used overhead")
            if slot_class == "prop" and support == "overhead":
                raise ValueError(f"prop slot {slot!r} may not be used overhead")
        return layout

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        from molmo_spaces.utils.lazy_loading_utils import install_uid

        # Inject the frozen Cup_10 target exactly as older place samplers do.
        super().add_auxiliary_objects(spec)
        self._pact_clutter_objects: list[dict[str, Any]] = []
        name_to_meta: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(self._palette()):
            uid = str(item["uid"])
            slot = str(item["slot"])
            slot_class = str(item.get("slot_class") or "prop")
            clutter_spec = MjSpec.from_file(str(install_uid(uid)))
            body = clutter_spec.worldbody.bodies[0]
            if slot_class == "mount":
                for joint in list(clutter_spec.joints):
                    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                        clutter_spec.delete(joint)
                body.mocap = True
            elif not body.first_joint():
                body.add_joint(
                    name=f"{uid}_pact_clutter_{slot}_free",
                    type=mujoco.mjtJoint.mjJNT_FREE,
                    damping=float(self.CLUTTER_FREE_JOINT_DAMPING),
                )
            namespace = f"pact_clutter_{slot}/"
            original_name = body.name
            park = np.asarray(self.CLUTTER_PARK_XYZ_M, dtype=float) + np.array(
                [0.35 * index, 0.0, 0.0]
            )
            frame = spec.worldbody.add_frame(pos=park)
            frame.attach_body(body, namespace, "")
            full_name = namespace + original_name
            if any(
                forbidden in full_name
                for forbidden in ("cavity_obj_", "pact_intrusion_", "place_receptacle")
            ):
                raise ValueError(f"illegal clutter body name {full_name!r}")
            if not full_name.startswith("pact_clutter_"):
                raise ValueError(f"clutter body lacks required prefix: {full_name!r}")
            annotation = ObjectMeta.annotation(uid) or {}
            record = {
                "slot": slot,
                "uid": uid,
                "body": full_name,
                "park_m": park.tolist(),
                "slot_class": slot_class,
            }
            self._pact_clutter_objects.append(record)
            name_to_meta[full_name] = {
                "asset_id": uid,
                "category": annotation.get("category", "object"),
                "object_enum": "temp_object",
                "is_static": slot_class == "mount",
                "boundingBox": annotation.get("boundingBox", {}),
            }
        self._metadata_adder.update(name_to_meta)
        log.info(
            "[PACT place v5] injected %d prop/mount clutter assets",
            len(self._pact_clutter_objects),
        )

    @staticmethod
    def _free_joint_addresses(model, body_name: str) -> tuple[int, int]:
        body_id = int(model.body(body_name).id)
        joint_id = int(model.body_jntadr[body_id])
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(
            mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError(f"clutter body {body_name!r} is not a free body")
        return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])

    def _set_free_pose(
        self,
        env,
        body_name: str,
        position: list[float],
        quat_wxyz: list[float],
    ) -> None:
        qadr, dadr = self._free_joint_addresses(env.current_model, body_name)
        quat = np.asarray(quat_wxyz, dtype=float)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
        env.current_data.qpos[qadr : qadr + 3] = np.asarray(position, dtype=float)
        env.current_data.qpos[qadr + 3 : qadr + 7] = quat
        env.current_data.qvel[dadr : dadr + 6] = 0.0

    def _set_mocap_pose(
        self,
        env,
        body_name: str,
        position: list[float],
        quat_wxyz: list[float],
    ) -> None:
        self._mocap_set(env, body_name, position)
        model, data = env.current_model, env.current_data
        mocap_id = int(model.body_mocapid[int(model.body(body_name).id)])
        if mocap_id < 0:
            raise ValueError(f"clutter mount {body_name!r} is not a mocap body")
        quat = np.asarray(quat_wxyz, dtype=float)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
        data.mocap_quat[mocap_id] = quat

    @staticmethod
    def _body_collision_aabb(
        model, data, body_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a body's descendant collision-geom AABB in world axes."""
        root_id = int(model.body_rootid[int(model.body(body_name).id)])
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for geom_id in range(int(model.ngeom)):
            body_id = int(model.geom_bodyid[geom_id])
            if int(model.body_rootid[body_id]) != root_id:
                continue
            if int(model.geom_contype[geom_id]) == 0 and int(
                model.geom_conaffinity[geom_id]
            ) == 0:
                continue
            # ``geom_aabb`` is expressed in the geom-local frame.  Its world
            # transform must therefore use geom_xpos/geom_xmat (important for
            # the nested, rotated child bodies in THOR assets).
            local_center = np.asarray(model.geom_aabb[geom_id, :3], dtype=float)
            local_half = np.asarray(model.geom_aabb[geom_id, 3:], dtype=float)
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            world_center = (
                np.asarray(data.geom_xpos[geom_id], dtype=float)
                + rotation @ local_center
            )
            world_half = np.abs(rotation) @ local_half
            lows.append(world_center - world_half)
            highs.append(world_center + world_half)
        if not lows:
            raise ValueError(f"active clutter body has no collision geoms: {body_name}")
        return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)

    def _draw_theta(self):
        # Call the pre-clutter implementation directly: V3/V4 remain unchanged,
        # while V5 does not try to treat free mesh bodies as scalar mocap boxes.
        th = PactPlaceCorridorV2Sampler._draw_theta(self)
        th.update(
            {
                "pact_place_environment_version": self.PACT_PLACE_ENVIRONMENT_VERSION,
                "pact_clutter_palette": self._palette(),
                "pact_clutter_layout": self._layout(),
                "pact_clutter_movable_free_bodies": True,
                "pact_clutter_mounts_are_mocap": any(
                    item.get("slot_class") == "mount" for item in self._palette()
                ),
                "pact_clutter_added_to_obstacle_aabbs": False,
            }
        )
        return th

    def _apply_theta(self, env, th):
        # Shell, intrusion, light, tray, and target behavior are the frozen V2
        # path.  In particular, clutter is never appended to obstacle_aabbs.
        PactPlaceCorridorV2Sampler._apply_theta(self, env, th)
        for body in self.LEGACY_CLUTTER_BODY_NAMES:
            self._mocap_set(env, body, [0.0, 0.0, -2.0])
        layout = dict(th["pact_clutter_layout"])
        layout_by_slot = {
            str(item["palette_slot"]): dict(item)
            for item in list(layout["objects"])
        }
        # Recheck the frozen metadata boxes against this episode's exact shell.
        # B2 admits against the minimum 0.20 m depth, while this assertion also
        # protects later manifest edits from silently intersecting a wall.
        tolerance = float(self.CLUTTER_CONTAINMENT_TOLERANCE_M)
        shell_lo = np.asarray(
            [TUBE_X0, -float(th["ap_w"]) / 2.0, SHELF_TOP_Z], dtype=float
        )
        shell_hi = np.asarray(
            [
                TUBE_X0 + float(th["depth"]),
                float(th["ap_w"]) / 2.0,
                SHELF_TOP_Z + float(th["ap_h"]),
            ],
            dtype=float,
        )
        for slot, object_layout in layout_by_slot.items():
            center = np.asarray(object_layout["center_m"], dtype=float)
            half = np.asarray(object_layout["half_m"], dtype=float)
            if np.any(center - half < shell_lo - tolerance) or np.any(
                center + half > shell_hi + tolerance
            ):
                raise ValueError(
                    "v5 clutter metadata box escapes the episode shell: "
                    f"slot={slot} bounds={(center - half).tolist(), (center + half).tolist()} "
                    f"shell={shell_lo.tolist(), shell_hi.tolist()}"
                )
        self._pact_active_clutter_names: list[str] = []
        self._pact_active_clutter_layout: dict[str, dict[str, Any]] = {}
        for item in self._pact_clutter_objects:
            slot = str(item["slot"])
            is_mount = item["slot_class"] == "mount"
            set_pose = self._set_mocap_pose if is_mount else self._set_free_pose
            if slot in layout_by_slot:
                object_layout = layout_by_slot[slot]
                desired_center = np.asarray(object_layout["center_m"], dtype=float)
                quat = list(map(float, object_layout["quat_wxyz"]))
                # Objaverse/THOR roots are not consistently located at their
                # collision-proxy centers. Measure the compiled AABB at
                # the requested orientation and solve the root translation
                # that puts the physical box at B2's frozen center.
                set_pose(env, item["body"], [0.0, 0.0, 0.0], quat)
                mujoco.mj_forward(env.current_model, env.current_data)
                local_low, local_high = self._body_collision_aabb(
                    env.current_model, env.current_data, item["body"]
                )
                position = desired_center - (local_low + local_high) / 2.0
                # Props get a one-centimetre settling drop. Mounts are
                # kinematic fixtures whose planned collision center is exact.
                if not is_mount:
                    position[2] += 0.01
                set_pose(
                    env,
                    item["body"],
                    position.tolist(),
                    quat,
                )
                self._pact_active_clutter_names.append(item["body"])
                self._pact_active_clutter_layout[item["body"]] = object_layout
            else:
                set_pose(env, item["body"], item["park_m"], [1, 0, 0, 0])
        if len(self._pact_active_clutter_names) != len(layout_by_slot):
            raise ValueError(
                "active v5 clutter body count does not match layout: "
                f"{self._pact_active_clutter_names} vs {sorted(layout_by_slot)}"
            )
        mujoco.mj_forward(env.current_model, env.current_data)

    def _settle_injected_object(self, env: CPUMujocoEnv) -> None:
        """Settle target and active clutter, restore every unrelated model DOF."""
        from scipy.spatial.transform import Rotation as R

        model, data = env.current_model, env.current_data
        qpos_before = data.qpos.copy()
        target_body = str(self._injected_obj_name)
        active = list(self._pact_active_clutter_names)
        slot_class_by_body = {
            str(item["body"]): str(item["slot_class"])
            for item in self._pact_clutter_objects
        }
        active_props = [
            name for name in active if slot_class_by_body[name] == "prop"
        ]
        active_mounts = [
            name for name in active if slot_class_by_body[name] == "mount"
        ]
        names = [target_body, *active_props]
        addresses = {
            name: self._free_joint_addresses(model, name) for name in names
        }

        # Preserve the target's frozen rest randomization.
        tqadr, tdadr = addresses[target_body]
        bx, by, floor_z = self._obj_rest()
        jx, jy = self.OBJ_JIT_XY
        data.qpos[tqadr : tqadr + 3] = [
            bx + float(np.random.uniform(-jx, jx)),
            by + float(np.random.uniform(-jy, jy)),
            floor_z + 0.12,
        ]
        yaw = float(np.random.uniform(0, 2 * np.pi))
        data.qpos[tqadr + 3 : tqadr + 7] = (
            R.from_euler("z", yaw) * R.from_euler("x", 90, degrees=True)
        ).as_quat(scalar_first=True)
        data.qvel[:] = 0.0
        for _ in range(int(self.CLUTTER_SETTLE_STEPS)):
            mujoco.mj_step(model, data)

        settled_qpos = {
            name: data.qpos[qadr : qadr + 7].copy()
            for name, (qadr, _dadr) in addresses.items()
        }
        active_roots = {
            int(model.body_rootid[int(model.body(name).id)]): name for name in active
        }
        target_root = int(model.body_rootid[int(model.body(target_body).id)])
        initial_object_contacts = []
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            left_root = int(
                model.body_rootid[int(model.geom_bodyid[int(contact.geom1)])]
            )
            right_root = int(
                model.body_rootid[int(model.geom_bodyid[int(contact.geom2)])]
            )
            left_active = active_roots.get(left_root)
            right_active = active_roots.get(right_root)
            if left_active is None and right_active is None:
                continue
            pair = {
                "geom1": model.geom(int(contact.geom1)).name or "",
                "geom2": model.geom(int(contact.geom2)).name or "",
                "active1": left_active,
                "active2": right_active,
                "distance_m": float(contact.dist),
            }
            initial_object_contacts.append(pair)
            if (
                left_active is not None
                and right_active is not None
                and left_root != right_root
            ):
                raise ValueError(f"settled clutter objects overlap: {pair}")
            if (
                (left_active is not None and right_root == target_root)
                or (right_active is not None and left_root == target_root)
            ):
                raise ValueError(f"settled clutter overlaps target: {pair}")
        settled_records = []
        for name in active_props:
            qadr, dadr = addresses[name]
            linear = float(np.linalg.norm(data.qvel[dadr : dadr + 3]))
            angular = float(np.linalg.norm(data.qvel[dadr + 3 : dadr + 6]))
            settled_low, settled_high = self._body_collision_aabb(model, data, name)
            settled_collision_center = (settled_low + settled_high) / 2.0
            planned_xy = np.asarray(
                self._pact_active_clutter_layout[name]["center_m"][:2], dtype=float
            )
            settled_xy = settled_collision_center[:2]
            xy_drift = float(np.linalg.norm(settled_xy - planned_xy))
            if xy_drift > self.CLUTTER_MAX_SETTLED_XY_DRIFT_M:
                raise ValueError(
                    f"clutter drifted during settle: {name} {xy_drift:.6f} m"
                )
            if linear > self.CLUTTER_MAX_SETTLED_LINEAR_SPEED_M_S:
                raise ValueError(
                    f"clutter did not settle: {name} linear speed {linear:.6f} m/s"
                )
            if angular > self.CLUTTER_MAX_SETTLED_ANGULAR_SPEED_RAD_S:
                raise ValueError(
                    f"clutter did not settle: {name} angular speed {angular:.6f} rad/s"
                )
            body_id = int(model.body(name).id)
            settled_records.append(
                {
                    "body": name,
                    "layout": dict(self._pact_active_clutter_layout[name]),
                    "qpos": settled_qpos[name].tolist(),
                    "position_m": np.asarray(data.xpos[body_id], dtype=float).tolist(),
                    "collision_center_m": settled_collision_center.tolist(),
                    "collision_bounds_m": [settled_low.tolist(), settled_high.tolist()],
                    "xmat": np.asarray(data.xmat[body_id], dtype=float).tolist(),
                    "linear_speed_m_s": linear,
                    "angular_speed_rad_s": angular,
                    "xy_drift_m": xy_drift,
                }
            )

        mount_records = []
        for name in active_mounts:
            low, high = self._body_collision_aabb(model, data, name)
            body_id = int(model.body(name).id)
            mount_records.append(
                {
                    "body": name,
                    "layout": dict(self._pact_active_clutter_layout[name]),
                    "position_m": np.asarray(data.xpos[body_id], dtype=float).tolist(),
                    "collision_center_m": ((low + high) / 2.0).tolist(),
                    "collision_bounds_m": [low.tolist(), high.tolist()],
                    "xmat": np.asarray(data.xmat[body_id], dtype=float).tolist(),
                    "settling_skipped": True,
                    "reason": "kinematic_mocap_overhead_fixture",
                }
            )

        # Validate the settled collision bounds, not only the metadata boxes
        # used by B2. MuJoCo stores each geom's local AABB as center/half-extents.
        workspace_bounds = self._theta.get("pact_clutter_workspace_bounds_m")
        if workspace_bounds is None:
            shell_lo = np.asarray(
                [TUBE_X0, -float(self._theta["ap_w"]) / 2.0, SHELF_TOP_Z],
                dtype=float,
            )
            shell_hi = np.asarray(
                [
                    TUBE_X0 + float(self._theta["depth"]),
                    float(self._theta["ap_w"]) / 2.0,
                    SHELF_TOP_Z + float(self._theta["ap_h"]),
                ],
                dtype=float,
            )
        else:
            shell_lo = np.asarray(workspace_bounds[0], dtype=float)
            shell_hi = np.asarray(workspace_bounds[1], dtype=float)
        tolerance = float(self.CLUTTER_CONTAINMENT_TOLERANCE_M)
        settled_bounds: dict[str, list[list[float]]] = {}
        for name in active:
            low, high = self._body_collision_aabb(model, data, name)
            settled_bounds[name] = [low.tolist(), high.tolist()]
            # Enforce the exact lateral/aperture and back-wall bounds after
            # settling.  The metadata precheck enforces the planned floor
            # bound; THOR collision proxies can be intentionally embedded in
            # a support surface, so their lower Z AABB is recorded but is not
            # a meaningful enclosure-escape test.
            if np.any(low[:2] < shell_lo[:2] - tolerance) or np.any(
                high > shell_hi + tolerance
            ):
                raise ValueError(
                    "settled v5 clutter collision proxy escapes the episode shell: "
                    f"body={name} bounds={low.tolist(), high.tolist()} "
                    f"shell={shell_lo.tolist(), shell_hi.tolist()}"
                )

        # Restore robot, parked objects, and all other state exactly; keep only
        # the target and active clutter's settled free-joint poses.
        data.qpos[:] = qpos_before
        data.qvel[:] = 0.0
        for name, values in settled_qpos.items():
            qadr, _dadr = addresses[name]
            data.qpos[qadr : qadr + 7] = values
        mujoco.mj_forward(model, data)
        th = getattr(self, "_theta", None)
        if th is not None:
            th["pact_clutter_settle"] = {
                "settle_steps": int(self.CLUTTER_SETTLE_STEPS),
                "linear_speed_threshold_m_s": float(
                    self.CLUTTER_MAX_SETTLED_LINEAR_SPEED_M_S
                ),
                "angular_speed_threshold_rad_s": float(
                    self.CLUTTER_MAX_SETTLED_ANGULAR_SPEED_RAD_S
                ),
                "xy_drift_threshold_m": float(
                    self.CLUTTER_MAX_SETTLED_XY_DRIFT_M
                ),
                "stable_at_step0": True,
                "model_nq": int(model.nq),
                "episode_shell_bounds_m": [shell_lo.tolist(), shell_hi.tolist()],
                "settled_collision_bounds_m": settled_bounds,
                "objects": settled_records,
                "mounts": mount_records,
                "dynamic_prop_static_fixture_asymmetry": (
                    "floor props settle and can topple; overhead mocap fixtures are "
                    "kinematically mounted and immovable"
                ),
                "initial_object_contacts": initial_object_contacts,
            }

    def _sample_task(self, env: CPUMujocoEnv):
        task = super()._sample_task(env)
        task.scene_params = dict(getattr(task, "scene_params", {}) or {})
        task.scene_params["pact_clutter_pool_body_names"] = [
            item["body"] for item in self._pact_clutter_objects
        ]
        task.scene_params["pact_clutter_active_body_names"] = list(
            self._pact_active_clutter_names
        )
        return task


class PactPlaceCorridorV9Sampler(PactPlaceCorridorV5Sampler):
    """V9.2's active side panel plus staggered real-object route blocker.

    The V5 injector remains the single object-installation path.  V9 only
    narrows the manifest contract and records the two vessel collision boxes
    as live obstacle boxes after the real assets have been posed.  V9.2 restores
    exactly one hidden left/right panel and keeps the bottle on the corridor
    centreline.  The panel selects the open lane; four bottle depth stations
    locally tighten that lane without closing it or leaking panel side through
    RGB-visible clutter.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_2"
    VESSEL_ROLES = frozenset({"inbound_vessel", "outbound_vessel"})
    VESSEL_CATEGORIES = frozenset(
        {
            "vase",
            "soapbottle",
            "pot",
            "candle",
            "can",
            "spray can",
            "aerosol can",
            "travel mug",
        }
    )
    DECOR_CATEGORIES = frozenset(
        {"mug", "apple", "bowl", "plate", "potato", "can", "candle"}
    )
    EXCLUDED_CATEGORIES = frozenset(
        {"cup", "teacup", "plastic cup", "ceramic cup", "clay cup"}
    )
    CLUTTER_WORKSPACE_LOW = (0.50, -0.43, SHELF_TOP_Z)
    CLUTTER_WORKSPACE_HIGH = (1.34, 0.43, 1.50)

    def _palette(self) -> list[dict[str, Any]]:
        row = self._pact_manifest_row or {}
        palette = list(row.get("pact_clutter_palette") or [])
        if not 8 <= len(palette) <= 12:
            raise ValueError("v9 palette must contain 2 vessels and 6-10 decor objects")
        slots = [str(item.get("slot", "")) for item in palette]
        if len(slots) != len(set(slots)):
            raise ValueError("duplicate v9 clutter palette slot")
        vessels = [item for item in palette if str(item.get("role")) in self.VESSEL_ROLES]
        decor = [item for item in palette if str(item.get("role")) == "decor"]
        if len(vessels) != 2 or {str(item.get("role")) for item in vessels} != self.VESSEL_ROLES:
            raise ValueError("v9 palette must contain one inbound and one outbound vessel")
        if not 6 <= len(decor) <= 10 or len(vessels) + len(decor) != len(palette):
            raise ValueError("v9 palette must contain 6-10 decor objects")
        counts: dict[str, int] = {}
        for item in palette:
            category = str(item.get("category", "object"))
            counts[category] = counts.get(category, 0) + 1
            if counts[category] > 2:
                raise ValueError(f"v9 category cap exceeded for {category!r}")
            if category in self.EXCLUDED_CATEGORIES:
                raise ValueError(f"v9 palette contains excluded cup-like category {category!r}")
            if str(item.get("slot_class") or "prop") != "prop":
                raise ValueError("v9 clutter must use movable free-body props")
            if str(item.get("support") or "shelf_standing") != "shelf_standing":
                raise ValueError("v9 palette objects must be standing free bodies")
        for item in vessels:
            if str(item.get("category", "")).lower() not in self.VESSEL_CATEGORIES:
                raise ValueError(f"v9 vessel category is not approved: {item.get('category')!r}")
            dimensions = [float(value) for value in item.get("dimensions_m", [])]
            if len(dimensions) != 3 or not 0.15 <= dimensions[2] <= 0.25:
                raise ValueError(f"v9 vessel height is outside 0.15-0.25 m: {dimensions}")
        return palette

    def _layout(self) -> dict[str, Any]:
        layout = super()._layout()
        row = self._pact_manifest_row or {}
        layout_side = layout.get("intrusion_side")
        if layout_side is not None and str(layout_side) != str(row.get("intrusion_side")):
            raise ValueError("v9 layout panel side does not match its manifest row")
        if layout.get("legacy_panel_active") and layout_side not in {"left", "right"}:
            raise ValueError("active-panel v9 layout is missing its committed panel side")
        objects = list(layout.get("objects") or [])
        palette_by_slot = {str(item["slot"]): item for item in self._palette()}
        by_role = {
            str(palette_by_slot[str(item["palette_slot"])]["role"]): item
            for item in objects
        }
        if set(by_role) != set(self.VESSEL_ROLES) | {"decor"}:
            raise ValueError("v9 layout roles do not match the frozen palette")
        if len(
            [
                item
                for item in objects
                if str(palette_by_slot[str(item["palette_slot"])]["role"])
                in self.VESSEL_ROLES
            ]
        ) != 2:
            raise ValueError("v9 layout must activate both vessel slots")
        return layout

    def _draw_theta(self):
        th = PactPlaceCorridorV5Sampler._draw_theta(self)
        panel_active = bool(th["pact_clutter_layout"].get("legacy_panel_active"))
        # Preserve the exact inherited panel record for both current and
        # historical replay rows.  V9.2 keeps it active; V9.1 rows explicitly
        # carry legacy_panel_active=false and remain reproducibly parked.
        th["pact_v9_legacy_panel"] = {
            "present": bool(th.get("protrusion_present")),
            "name": th.get("protr_name"),
            "side": th.get("protr_wall"),
            "center": th.get("protr_center"),
            "half": th.get("protr_half"),
        }
        if not panel_active:
            th["protrusion_present"] = False
            th.pop("protr_center", None)
            th.pop("protr_half", None)
        th["pact_v9_legacy_panel_active"] = panel_active
        th["pact_clutter_workspace_bounds_m"] = [
            list(self.CLUTTER_WORKSPACE_LOW),
            list(self.CLUTTER_WORKSPACE_HIGH),
        ]
        layout_by_slot = {
            str(item["palette_slot"]): item
            for item in list(th["pact_clutter_layout"]["objects"])
        }
        palette_by_slot = {str(item["slot"]): item for item in self._palette()}
        hazards = []
        for role in ("inbound_vessel", "outbound_vessel"):
            item = next(
                object_layout
                for slot, object_layout in layout_by_slot.items()
                if str(palette_by_slot[slot].get("role")) == role
            )
            hazards.append(
                {
                    "name": f"pact_vessel_{role}",
                    "role": role,
                    "slot": str(item["palette_slot"]),
                    "uid": str(item["uid"]),
                    "center": [float(value) for value in item["center_m"]],
                    "half": [float(value) for value in item["half_m"]],
                    "phase": "inbound" if role == "inbound_vessel" else "outbound",
                }
            )
        th.update(
            {
                "pact_v9_hazards": hazards,
                "pact_v9_hazard_list_source": (
                    "active_hidden_panel_plus_real_objaverse_vessels"
                    if panel_active
                    else "real_objaverse_vessels_panel_parked_historical"
                ),
                "pact_v9_vessels_added_to_obstacle_aabbs": True,
            }
        )
        return th

    def _apply_theta(self, env, th):
        # Install the shell directly, then pose exactly the panel committed by
        # the row.  This also retains replay compatibility with V9.1 rows that
        # explicitly park the panel.
        panel_present = bool(th.get("protrusion_present"))
        th["protrusion_present"] = False
        FumehoodSampler._apply_theta(self, env, th)
        th["protrusion_present"] = panel_present
        for name, y in (
            ("pact_intrusion_left", 1.8),
            ("pact_intrusion_right", -1.8),
        ):
            self._mocap_set(env, name, [0.0, y, -2.0])
        if th.get("pact_v9_legacy_panel_active"):
            self._mocap_set(env, str(th["protr_name"]), th["protr_center"])
            th.setdefault("obstacle_aabbs", []).append(
                [
                    list(map(float, th["protr_center"])),
                    list(map(float, th["protr_half"])),
                ]
            )
        for body in self.LEGACY_CLUTTER_BODY_NAMES:
            self._mocap_set(env, body, [0.0, 0.0, -2.0])
        layout_by_slot = {
            str(item["palette_slot"]): item
            for item in list(th["pact_clutter_layout"]["objects"])
        }
        tolerance = 1e-4
        shell_lo = np.asarray(self.CLUTTER_WORKSPACE_LOW, dtype=float)
        shell_hi = np.asarray(self.CLUTTER_WORKSPACE_HIGH, dtype=float)
        for slot, object_layout in layout_by_slot.items():
            center = np.asarray(object_layout["center_m"], dtype=float)
            half = np.asarray(object_layout["half_m"], dtype=float)
            if np.any(center - half < shell_lo - tolerance) or np.any(
                center + half > shell_hi + tolerance
            ):
                raise ValueError(
                    "v9 clutter metadata box escapes the physical bench workspace: "
                    f"slot={slot} bounds={(center - half).tolist(), (center + half).tolist()} "
                    f"shell={shell_lo.tolist(), shell_hi.tolist()}"
                )
        self._pact_active_clutter_names: list[str] = []
        self._pact_active_clutter_layout: dict[str, dict[str, Any]] = {}
        for item in self._pact_clutter_objects:
            slot = str(item["slot"])
            is_mount = item["slot_class"] == "mount"
            set_pose = self._set_mocap_pose if is_mount else self._set_free_pose
            if slot in layout_by_slot:
                object_layout = layout_by_slot[slot]
                desired_center = np.asarray(object_layout["center_m"], dtype=float)
                quat = list(map(float, object_layout["quat_wxyz"]))
                set_pose(env, item["body"], [0.0, 0.0, 0.0], quat)
                mujoco.mj_forward(env.current_model, env.current_data)
                local_low, local_high = self._body_collision_aabb(
                    env.current_model, env.current_data, item["body"]
                )
                position = desired_center - (local_low + local_high) / 2.0
                if not is_mount:
                    position[2] = 0.72 + 0.001 - float(local_low[2])
                set_pose(
                    env,
                    item["body"],
                    position.tolist(),
                    quat,
                )
                self._pact_active_clutter_names.append(item["body"])
                self._pact_active_clutter_layout[item["body"]] = object_layout
            else:
                set_pose(env, item["body"], item["park_m"], [1, 0, 0, 0])
        if len(self._pact_active_clutter_names) != len(layout_by_slot):
            raise ValueError(
                "active v5 clutter body count does not match layout: "
                f"{self._pact_active_clutter_names} vs {sorted(layout_by_slot)}"
            )
        body_by_slot = {
            str(item["slot"]): str(item["body"])
            for item in self._pact_clutter_objects
        }
        palette_by_slot = {str(item["slot"]): item for item in self._palette()}
        exact_hazards = []
        for hazard in list(th.get("pact_v9_hazards") or []):
            body_name = body_by_slot[str(hazard["slot"])]
            low, high = self._body_collision_aabb(
                env.current_model, env.current_data, body_name
            )
            center = (low + high) / 2.0
            half = (high - low) / 2.0
            exact_hazards.append(
                {
                    **hazard,
                    "body": body_name,
                    "center": center.tolist(),
                    "half": half.tolist(),
                    "category": str(palette_by_slot[str(hazard["slot"])]["category"]),
                }
            )
            # Both vessels remain live obstacle additions.  In V9.2 the active
            # panel was already added above; historical V9.1 rows keep it parked.
            th.setdefault("obstacle_aabbs", []).append([center.tolist(), half.tolist()])
        th["pact_v9_hazards"] = exact_hazards
        mujoco.mj_forward(env.current_model, env.current_data)


class PactPlaceCorridorV93Sampler(PactPlaceCorridorV9Sampler):
    """V9.3 two-bottle 2-D chicane with paired, side-independent jitter."""

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_3"
    VESSEL_JITTER_LIMIT_M = 0.020

    def _layout(self) -> dict[str, Any]:
        import copy

        layout = copy.deepcopy(super()._layout())
        row = self._pact_manifest_row or {}
        x_jitter = row.get("clutter_x_jitter_m") or {}
        y_jitter = row.get("clutter_y_jitter_m") or {}
        if not isinstance(x_jitter, dict) or not isinstance(y_jitter, dict):
            raise ValueError("V9.3 clutter jitter must be keyed by palette slot")
        vessel_slots = {
            str(layout["inbound_vessel_slot"]),
            str(layout["outbound_vessel_slot"]),
        }
        for item in layout["objects"]:
            slot = str(item["palette_slot"])
            jx = float(x_jitter.get(slot, 0.0))
            jy = float(y_jitter.get(slot, 0.0))
            if slot not in vessel_slots and (jx != 0.0 or jy != 0.0):
                raise ValueError(f"V9.3 jitter may only move vessel slots: {slot}")
            if abs(jx) > self.VESSEL_JITTER_LIMIT_M or abs(jy) > self.VESSEL_JITTER_LIMIT_M:
                raise ValueError(f"V9.3 vessel jitter exceeds +/-20 mm: {slot}")
            item["center_m"][0] = float(item["center_m"][0]) + jx
            item["center_m"][1] = float(item["center_m"][1]) + jy
            item["jitter_xy_m"] = [jx, jy]
        layout["applied_clutter_x_jitter_m"] = {
            str(key): float(value) for key, value in x_jitter.items()
        }
        layout["applied_clutter_y_jitter_m"] = {
            str(key): float(value) for key, value in y_jitter.items()
        }
        layout["paired_side_cell"] = row.get("paired_side_cell")
        return layout


class PactPlaceCorridorV94MountedPreviewSampler(PactPlaceCorridorV93Sampler):
    """V9.3 plus one kinematic wall beam and one ceiling-mounted box."""

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_4_mounted_preview"
    MOUNT_BODIES = {
        "wall_left": "pact_clutter_mount_wall_left",
        "wall_right": "pact_clutter_mount_wall_right",
        "ceiling": "pact_clutter_mount_ceiling",
    }
    MOUNT_GEOMS = {
        key: f"{body}_g" for key, body in MOUNT_BODIES.items()
    }

    def _mounted_fixtures(self) -> list[dict[str, Any]]:
        fixtures = [
            dict(item)
            for item in list(
                (self._pact_manifest_row or {}).get("pact_mounted_fixtures") or []
            )
        ]
        supports = [str(item.get("support")) for item in fixtures]
        if sorted(supports) not in (
            ["ceiling", "wall_left"],
            ["ceiling", "wall_right"],
        ):
            raise ValueError("V9.4 requires exactly one wall and one ceiling fixture")
        for item in fixtures:
            support = str(item["support"])
            center = np.asarray(item.get("center_m"), dtype=float)
            half = np.asarray(item.get("half_m"), dtype=float)
            if center.shape != (3,) or half.shape != (3,) or np.any(half <= 0.0):
                raise ValueError(f"invalid V9.4 mounted fixture geometry: {item}")
            if support.startswith("wall_"):
                side = 1.0 if support == "wall_left" else -1.0
                wall_face = side * float(center[1] + side * half[1])
                if abs(wall_face - 0.45) > 1e-6:
                    raise ValueError(f"V9.4 wall fixture is detached: {item}")
            else:
                if abs(float(center[2] + half[2]) - 1.515) > 1e-6:
                    raise ValueError(f"V9.4 ceiling fixture is detached: {item}")
            if not 0.58 <= center[0] - half[0] <= center[0] + half[0] <= 1.36:
                raise ValueError(f"V9.4 fixture escapes enclosure depth: {item}")
        return fixtures

    def _draw_theta(self):
        th = super()._draw_theta()
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_v94_mounted_fixtures"] = self._mounted_fixtures()
        th["pact_v94_mounted_clutter_is_kinematic"] = True
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        for support, body in self.MOUNT_BODIES.items():
            park_y = 2.2 if support == "wall_left" else -2.2 if support == "wall_right" else 0.0
            self._mocap_set(env, body, [0.0, park_y, -2.0])
        exact = []
        for item in list(th.get("pact_v94_mounted_fixtures") or []):
            support = str(item["support"])
            body = self.MOUNT_BODIES[support]
            geom = env.current_model.geom(self.MOUNT_GEOMS[support])
            half = np.asarray(item["half_m"], dtype=float)
            center = np.asarray(item["center_m"], dtype=float)
            env.current_model.geom_size[int(geom.id)] = half
            self._mocap_set(env, body, center.tolist())
            record = {
                **item,
                "name": f"pact_mounted_{support}",
                "body": body,
                "role": "ceiling_fixture" if support == "ceiling" else "wall_fixture",
                "center": center.tolist(),
                "half": half.tolist(),
                "phase": "outbound",
                "kinematic": True,
            }
            exact.append(record)
            th.setdefault("obstacle_aabbs", []).append(
                [center.tolist(), half.tolist()]
            )
        th.setdefault("pact_v9_hazards", []).extend(exact)
        th["pact_v94_active_mount_bodies"] = [item["body"] for item in exact]
        mujoco.mj_forward(env.current_model, env.current_data)


class PactPlaceCorridorV95LowWallSampler(PactPlaceCorridorV93Sampler):
    """V9.3 plus one low, rigid lateral fixture; no active ceiling obstacle."""

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_5_low_wall"
    WALL_BODIES = {
        "wall_left": "pact_clutter_mount_wall_left",
        "wall_right": "pact_clutter_mount_wall_right",
    }
    WALL_GEOMS = {key: f"{body}_g" for key, body in WALL_BODIES.items()}

    def _wall_fixture(self) -> dict[str, Any]:
        fixture = dict(
            (self._pact_manifest_row or {}).get("pact_mounted_wall_fixture") or {}
        )
        support = str(fixture.get("support") or "")
        if support not in self.WALL_BODIES:
            raise ValueError("V9.5 requires exactly one left/right wall fixture")
        center = np.asarray(fixture.get("center_m"), dtype=float)
        half = np.asarray(fixture.get("half_m"), dtype=float)
        if center.shape != (3,) or half.shape != (3,) or np.any(half <= 0.0):
            raise ValueError(f"invalid V9.5 wall fixture geometry: {fixture}")
        side = 1.0 if support == "wall_left" else -1.0
        exterior_face = side * float(center[1] + side * half[1])
        if abs(exterior_face - 0.45) > 1e-6:
            raise ValueError(f"V9.5 wall fixture is detached: {fixture}")
        low = center - half
        high = center + half
        if low[0] < 0.58 - 1e-6 or high[0] > 0.86 + 1e-6:
            raise ValueError(f"V9.5 wall fixture escapes the 0.58-0.86 m depth: {fixture}")
        if not 0.87 - 1e-6 <= low[2] <= 0.98 + 1e-6:
            raise ValueError(f"V9.5 wall fixture bottom is outside 0.87-0.98 m: {fixture}")
        if not 1.06 - 1e-6 <= high[2] <= 1.15 + 1e-6:
            raise ValueError(f"V9.5 wall fixture top is outside 1.06-1.15 m: {fixture}")
        if 1.03 - low[2] < 0.05 - 1e-6:
            raise ValueError("V9.5 fixture must extend at least 50 mm below z=1.03 m")
        return fixture

    def _draw_theta(self):
        th = super()._draw_theta()
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_v95_wall_fixture"] = self._wall_fixture()
        th["pact_v95_mounted_clutter_is_kinematic"] = True
        th["pact_v95_ceiling_fixture_active"] = False
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        self._mocap_set(env, "pact_clutter_mount_wall_left", [0.0, 2.2, -2.0])
        self._mocap_set(env, "pact_clutter_mount_wall_right", [0.0, -2.2, -2.0])
        self._mocap_set(env, "pact_clutter_mount_ceiling", [0.0, 0.0, -2.0])
        item = dict(th["pact_v95_wall_fixture"])
        support = str(item["support"])
        body = self.WALL_BODIES[support]
        center = np.asarray(item["center_m"], dtype=float)
        half = np.asarray(item["half_m"], dtype=float)
        env.current_model.geom_size[int(env.current_model.geom(self.WALL_GEOMS[support]).id)] = half
        self._mocap_set(env, body, center.tolist())
        hazard = {
            **item,
            "name": f"pact_mounted_{support}",
            "body": body,
            "role": "wall_fixture",
            "center": center.tolist(),
            "half": half.tolist(),
            "phase": "inbound_and_outbound",
            "kinematic": True,
        }
        th.setdefault("obstacle_aabbs", []).append([center.tolist(), half.tolist()])
        th.setdefault("pact_v9_hazards", []).append(hazard)
        th["pact_v95_active_mount_body"] = body
        mujoco.mj_forward(env.current_model, env.current_data)


class PactPlaceCorridorV96ClusterSampler(PactPlaceCorridorV93Sampler):
    """V9.3 with each leg's single vessel replaced by a clustered hazard.

    W1 measured the skin's resolving power: an 8x8 sensor over a 45 deg cone has
    a pixel pitch of ``0.1036 * R``, so an 0.089 m bottle clears one pixel only
    inside 0.86 m.  The V9.5 inbound bottle changed 40 of 4.85M raw values --
    below the sensor's resolving power, not a weak signal awaiting tuning.  V9.6
    therefore gives each leg a cluster of tall vessels standing shoulder to
    shoulder so the leg presents a contiguous silhouette wide enough to resolve.

    The palette is twelve prop slots: three ``inbound_cluster`` members, three
    ``outbound_cluster`` members and six RGB-only ``decor`` items.  Every asset
    is one already accepted in ``palette_v9_1.json``.  The 40-sensor suite, the
    encoder, the observation contract and the intrusion panel are untouched.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_6_cluster"
    CLUSTER_ROLES = ("inbound_cluster", "outbound_cluster")
    CLUSTER_MEMBERS_PER_LEG = (3, 4)
    PALETTE_SIZE_RANGE = (12, 16)
    MIN_CLUSTER_SPAN_M = 0.25
    MAX_CLUSTER_GAP_M = 0.04

    def _palette(self) -> list[dict[str, Any]]:
        row = self._pact_manifest_row or {}
        palette = list(row.get("pact_clutter_palette") or [])
        low_size, high_size = self.PALETTE_SIZE_RANGE
        if not low_size <= len(palette) <= high_size:
            raise ValueError(
                f"v9.6 palette size {len(palette)} outside {self.PALETTE_SIZE_RANGE}"
            )
        slots = [str(item.get("slot", "")) for item in palette]
        if len(slots) != len(set(slots)):
            raise ValueError("duplicate v9.6 clutter palette slot")
        by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in self.CLUSTER_ROLES}
        decor = []
        for item in palette:
            role = str(item.get("role"))
            if role in by_role:
                by_role[role].append(item)
            elif role == "decor":
                decor.append(item)
            else:
                raise ValueError(f"unknown v9.6 clutter role: {role!r}")
            if str(item.get("slot_class") or "prop") != "prop":
                raise ValueError("v9.6 clutter must use movable free-body props")
            if str(item.get("support") or "shelf_standing") != "shelf_standing":
                raise ValueError("v9.6 palette objects must be standing free bodies")
            if str(item.get("category", "")) in self.EXCLUDED_CATEGORIES:
                raise ValueError(f"v9.6 palette contains excluded category {item.get('category')!r}")
        for role, members in by_role.items():
            low, high = self.CLUSTER_MEMBERS_PER_LEG
            if not low <= len(members) <= high:
                raise ValueError(f"v9.6 {role} must hold {low}-{high} members")
            for item in members:
                if str(item.get("category", "")).lower() not in self.VESSEL_CATEGORIES:
                    raise ValueError(
                        f"v9.6 cluster member category is not approved: {item.get('category')!r}"
                    )
                dimensions = [float(value) for value in item.get("dimensions_m", [])]
                if len(dimensions) != 3 or not 0.15 <= dimensions[2] <= 0.25:
                    raise ValueError(
                        f"v9.6 cluster member height is outside 0.15-0.25 m: {dimensions}"
                    )
        if not 6 <= len(decor) <= 10:
            raise ValueError("v9.6 palette must contain 6-10 decor objects")
        # The V9 two-per-category cap exists to keep decor from repeating.  A
        # cluster is deliberately made of like objects, so the cap applies to
        # decor only.
        counts: dict[str, int] = {}
        for item in decor:
            category = str(item.get("category", "object"))
            counts[category] = counts.get(category, 0) + 1
            if counts[category] > 2:
                raise ValueError(f"v9.6 decor category cap exceeded for {category!r}")
        return palette

    def _cluster_members(self, layout: dict[str, Any], role: str) -> list[dict[str, Any]]:
        palette_by_slot = {str(item["slot"]): item for item in self._palette()}
        return [
            item
            for item in list(layout.get("objects") or [])
            if str(palette_by_slot[str(item["palette_slot"])].get("role")) == role
        ]

    def _layout(self) -> dict[str, Any]:
        row = self._pact_manifest_row or {}
        layout = PactPlaceCorridorV5Sampler._layout(self)
        layout_side = layout.get("intrusion_side")
        if layout_side is not None and str(layout_side) != str(row.get("intrusion_side")):
            raise ValueError("v9.6 layout panel side does not match its manifest row")
        if not layout.get("legacy_panel_active"):
            raise ValueError("v9.6 requires one active legacy side panel")
        for role in self.CLUSTER_ROLES:
            members = self._cluster_members(layout, role)
            low, high = self.CLUSTER_MEMBERS_PER_LEG
            if not low <= len(members) <= high:
                raise ValueError(f"v9.6 layout must activate {low}-{high} {role} members")
            record = dict(layout.get(role) or {})
            if float(record.get("span_along_line_m", 0.0)) < self.MIN_CLUSTER_SPAN_M - 1e-9:
                raise ValueError(
                    f"v9.6 {role} silhouette is below the {self.MIN_CLUSTER_SPAN_M} m "
                    "resolving-power floor"
                )
            if float(record.get("gap_m", 1.0)) > self.MAX_CLUSTER_GAP_M + 1e-9:
                raise ValueError(f"v9.6 {role} inter-item gap exceeds the contiguity ceiling")
        return layout

    def _draw_theta(self):
        th = PactPlaceCorridorV5Sampler._draw_theta(self)
        panel_active = bool(th["pact_clutter_layout"].get("legacy_panel_active"))
        th["pact_v9_legacy_panel"] = {
            "present": bool(th.get("protrusion_present")),
            "name": th.get("protr_name"),
            "side": th.get("protr_wall"),
            "center": th.get("protr_center"),
            "half": th.get("protr_half"),
        }
        if not panel_active:
            raise ValueError("v9.6 requires the panel to stay active")
        th["pact_v9_legacy_panel_active"] = True
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_clutter_workspace_bounds_m"] = [
            list(self.CLUTTER_WORKSPACE_LOW),
            list(self.CLUTTER_WORKSPACE_HIGH),
        ]
        layout = th["pact_clutter_layout"]
        hazards = []
        for role in self.CLUSTER_ROLES:
            for member in self._cluster_members(layout, role):
                hazards.append(
                    {
                        "name": f"pact_cluster_{role}_{member['palette_slot']}",
                        "role": role,
                        "slot": str(member["palette_slot"]),
                        "uid": str(member["uid"]),
                        "center": [float(value) for value in member["center_m"]],
                        "half": [float(value) for value in member["half_m"]],
                        "phase": "inbound" if role == "inbound_cluster" else "outbound",
                    }
                )
        th.update(
            {
                "pact_v9_hazards": hazards,
                "pact_v9_hazard_list_source": "active_hidden_panel_plus_two_objaverse_clusters",
                "pact_v9_vessels_added_to_obstacle_aabbs": True,
                "pact_v96_cluster_roles": list(self.CLUSTER_ROLES),
            }
        )
        return th

    def _apply_theta(self, env, th):
        PactPlaceCorridorV9Sampler._apply_theta(self, env, th)
        layout = th["pact_clutter_layout"]
        unions = []
        for role in self.CLUSTER_ROLES:
            posed = [item for item in th["pact_v9_hazards"] if str(item.get("role")) == role]
            if not posed:
                continue
            lows = np.array([np.array(i["center"]) - np.array(i["half"]) for i in posed])
            highs = np.array([np.array(i["center"]) + np.array(i["half"]) for i in posed])
            low, high = lows.min(axis=0), highs.max(axis=0)
            unions.append(
                {
                    "role": role,
                    "member_slots": [str(i["slot"]) for i in posed],
                    "union_center_m": ((low + high) / 2.0).tolist(),
                    "union_half_m": ((high - low) / 2.0).tolist(),
                    "union_extent_m": (high - low).tolist(),
                    "declared_span_m": float(dict(layout.get(role) or {}).get("span_along_line_m", 0.0)),
                }
            )
        th["pact_v96_cluster_unions"] = unions
        mujoco.mj_forward(env.current_model, env.current_data)


class PactPlaceCorridorV97HazardSampler(PactPlaceCorridorV96ClusterSampler):
    """V9.6 with the hazard's *width* decontracted and left to measured subtense.

    V9.6 required each leg to present a contiguous silhouette of at least
    0.25 m.  That number was a proxy for the skin's resolving power, not a
    property of it.  The sensor's requirement is angular -- a 0.120 m hazard
    clears two pixels out to R = 0.58 m, and the ranges W1 measured on the frozen
    trajectories are 0.11-0.14 m -- so a hazard narrow enough to fit the
    corridor's 0.120 m budget is still resolvable where it matters.

    This subclass therefore admits one to four members per leg and applies no
    span floor.  Everything else, including the 40-sensor suite, the encoder, the
    observation contract and the intrusion panel, is inherited untouched.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_7_subtense"
    CLUSTER_MEMBERS_PER_LEG = (1, 4)
    PALETTE_SIZE_RANGE = (8, 16)
    MIN_CLUSTER_SPAN_M = 0.0


class PactPlaceCorridorV98PendantSampler(PactPlaceCorridorV93Sampler):
    """Settled V9.5 clutter plus one symmetric, kinematic ceiling pendant.

    The pendant is deliberately independent of panel side.  It is registered
    as a real 3-D obstacle so the privileged expert's surface-distance speed
    law sees the fixture, while the production student still receives the
    unchanged observation suite.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_8_pendant"
    PENDANT_BODY = "pact_clutter_mount_ceiling"
    PENDANT_GEOM = "pact_clutter_mount_ceiling_g"

    def _pendant_fixture(self) -> dict[str, Any]:
        from pact_place_v98_pendant_contract import validate_pendant_geometry

        fixture = dict(
            (self._pact_manifest_row or {}).get("pact_mounted_ceiling_fixture") or {}
        )
        if str(fixture.get("support") or "") != "ceiling":
            raise ValueError("V9.8 requires one ceiling-mounted pendant fixture")
        validate_pendant_geometry(fixture.get("center_m"), fixture.get("half_m"))
        return fixture

    def _pendant_parked(self) -> bool:
        """Diagnostic control: run the identical scene with no active pendant.

        This exists so a V9.8 expert-screen failure can be attributed. Parking
        the fixture leaves the inherited V9.5 clutter and panel untouched, so a
        parked run isolates the pendant from the rest of the layout. It is never
        set on an admission or collection row.
        """
        return bool((self._pact_manifest_row or {}).get("pact_v98_pendant_parked"))

    def _draw_theta(self):
        th = super()._draw_theta()
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        parked = self._pendant_parked()
        th["pact_v98_pendant_parked"] = parked
        th["pact_v98_pendant_fixture"] = {} if parked else self._pendant_fixture()
        th["pact_v98_mounted_clutter_is_kinematic"] = True
        th["pact_v98_lateral_lane_cost_m"] = 0.0
        row = self._pact_manifest_row or {}
        th["pact_v98_pendant_lateral_bow"] = bool(
            row.get("pact_v98_pendant_lateral_bow")
        )
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        from pact_place_v98_pendant_contract import PENDANT_BODY, PENDANT_GEOM

        self._mocap_set(env, PENDANT_BODY, [0.0, 0.0, -2.0])
        if th.get("pact_v98_pendant_parked"):
            th["pact_v98_active_mount_body"] = None
            mujoco.mj_forward(env.current_model, env.current_data)
            return
        item = dict(th["pact_v98_pendant_fixture"])
        center = np.asarray(item["center_m"], dtype=float)
        half = np.asarray(item["half_m"], dtype=float)
        geom = env.current_model.geom(PENDANT_GEOM)
        env.current_model.geom_size[int(geom.id)] = half
        self._mocap_set(env, PENDANT_BODY, center.tolist())
        hazard = {
            **item,
            "name": "pact_mounted_ceiling_fixture",
            "body": PENDANT_BODY,
            "role": "ceiling_fixture",
            "center": center.tolist(),
            "half": half.tolist(),
            "phase": "inbound_and_outbound",
            "kinematic": True,
            "lateral_lane_cost_m": 0.0,
        }
        th.setdefault("obstacle_aabbs", []).append([center.tolist(), half.tolist()])
        th.setdefault("pact_v9_hazards", []).append(hazard)
        th["pact_v98_active_mount_body"] = PENDANT_BODY
        th["pact_v98_lateral_lane_cost_m"] = 0.0
        mujoco.mj_forward(env.current_model, env.current_data)


class PactPlaceCorridorV99PendantSampler(PactPlaceCorridorV93Sampler):
    """Settled V9.5 clutter plus one fixed, side-independent ceiling pendant.

    V9.9 does not inherit V9.8 lag, face-window, or ceiling-envelope
    validation. The pendant is identical on both panel sides and is active on
    empty inbound and loaded outbound traversal.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_9_pendant"
    PENDANT_BODY = "pact_clutter_mount_ceiling"
    PENDANT_GEOM = "pact_clutter_mount_ceiling_g"

    def _pendant_fixture(self) -> dict[str, Any]:
        from pact_place_v99_pendant_contract import validate_pendant_geometry

        fixture = dict(
            (self._pact_manifest_row or {}).get("pact_v99_pendant_fixture")
            or (self._pact_manifest_row or {}).get("pact_mounted_ceiling_fixture")
            or {}
        )
        if str(fixture.get("support") or "") != "ceiling":
            raise ValueError("V9.9 requires one ceiling-mounted pendant fixture")
        validate_pendant_geometry(fixture.get("center_m"), fixture.get("half_m"))
        return fixture

    def _pendant_parked(self) -> bool:
        return bool((self._pact_manifest_row or {}).get("pact_v99_pendant_parked"))

    def _route_params(self) -> dict[str, Any]:
        row = self._pact_manifest_row or {}
        return dict(row.get("pact_v99_route") or {})

    def _draw_theta(self):
        th = super()._draw_theta()
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        parked = self._pendant_parked()
        th["pact_v99_pendant_parked"] = parked
        th["pact_v99_pendant_fixture"] = {} if parked else self._pendant_fixture()
        th["pact_v99_route"] = {} if parked else self._route_params()
        th["pact_v99_mounted_clutter_is_kinematic"] = True
        th["pact_v98_pendant_lateral_bow"] = False
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        from pact_place_v99_pendant_contract import PENDANT_BODY, PENDANT_GEOM

        self._mocap_set(env, PENDANT_BODY, [0.0, 0.0, -2.0])
        if th.get("pact_v99_pendant_parked"):
            th["pact_v99_active_mount_body"] = None
            mujoco.mj_forward(env.current_model, env.current_data)
            return
        item = dict(th["pact_v99_pendant_fixture"])
        center = np.asarray(item["center_m"], dtype=float)
        half = np.asarray(item["half_m"], dtype=float)
        geom = env.current_model.geom(PENDANT_GEOM)
        env.current_model.geom_size[int(geom.id)] = half
        self._mocap_set(env, PENDANT_BODY, center.tolist())
        hazard = {
            **item,
            "name": "pact_mounted_ceiling_fixture",
            "body": PENDANT_BODY,
            "role": "ceiling_fixture",
            "center": center.tolist(),
            "half": half.tolist(),
            "phase": "inbound_and_outbound",
            "kinematic": True,
            "lateral_lane_cost_m": 0.0,
        }
        th.setdefault("obstacle_aabbs", []).append([center.tolist(), half.tolist()])
        th.setdefault("pact_v9_hazards", []).append(hazard)
        th["pact_v99_active_mount_body"] = PENDANT_BODY
        mujoco.mj_forward(env.current_model, env.current_data)


PACT_PLACE_V10_ENVIRONMENT_VERSION = "pact_place_corridor_v10_compound_pendant"
PACT_PLACE_V102_ENVIRONMENT_VERSION = "pact_place_corridor_v10_2_raised_pendant"
# Both V10 versions use the full-route lane primitive on the same compiled V10
# scene. V10.2 additionally unlocks the registered route-piece speed schedule.
PACT_PLACE_V105_ENVIRONMENT_VERSION = (
    "pact_place_corridor_v10_5_v95_clutter_static_pendant"
)


class PactPlaceCorridorV105Sampler(PactPlaceCorridorV93Sampler):
    """Settled fixture-free V9.5 clutter, verbatim, plus the V10.5 marker.

    Sampling behaviour is V9.3's: the same palette, layout families, vessel
    jitter, panel, target/tray, cameras, contact audit, and clutter-stability
    semantics. Household objects stay collision-enabled movable free bodies.

    The pendant is not sampled at all. It lives in the compiled scene XML that
    the manifest row selects before model construction, so there is nothing to
    pose, resize, or refresh at runtime. The marker exists so the one
    registered speed cap and the V10.5 telemetry can be gated on it, and so no
    historical environment can reach either.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = PACT_PLACE_V105_ENVIRONMENT_VERSION
    PENDANT_BODY = "pact_clutter_mount_v105"

    def _draw_theta(self):
        th = super()._draw_theta()
        row = self._pact_manifest_row or {}
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_v105_static_pendant"] = True
        th["pact_v105_pendant_body"] = self.PENDANT_BODY
        # Telemetry/expert only. These never reach a student observation.
        th["pact_v105_pose_id"] = row.get("pose_id")
        th["pact_v105_scene_sha256"] = row.get("pact_v105_scene_sha256")
        th["pact_v105_assembly_sha256"] = row.get("pact_v105_assembly_sha256")
        return th

    def sample_task(self, *args, **kwargs):
        """Refuse a scene/pose/hash mismatch before the task is created."""
        row = self._pact_manifest_row or {}
        expected = row.get("pact_v105_scene_sha256")
        if expected:
            import hashlib
            from pathlib import Path

            scene = getattr(
                getattr(self.cfg, "task_sampler_config", None), "scene_xml", None
            ) or getattr(self.cfg, "scene_xml", None)
            if scene is None:
                raise ValueError("V10.5 row binds a scene hash but no scene is set")
            observed = hashlib.sha256(Path(str(scene)).read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError(
                    f"V10.5 scene hash mismatch for pose {row.get('pose_id')!r}: "
                    f"{observed} != {expected}"
                )
        return super().sample_task(*args, **kwargs)


PACT_PLACE_V106_ENVIRONMENT_VERSION = (
    "pact_place_corridor_v10_6_v95_clutter_asymmetric_pendant"
)


PACT_PLACE_V1010_ENVIRONMENT_VERSION = "pact_place_corridor_v10_10_four_object"
# V10.10 is the V10.6 lane with four clutter slots parked, not a new lane. Every
# gate that switches on the V10.6 marker -- the speed amendment, the frame
# telemetry, the expert routing -- must therefore accept it, or V10.10 would
# silently run different speeds and emit no telemetry while looking healthy.
PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS = (
    PACT_PLACE_V106_ENVIRONMENT_VERSION,
    PACT_PLACE_V1010_ENVIRONMENT_VERSION,
)


class PactPlaceCorridorV106Sampler(PactPlaceCorridorV93Sampler):
    """V10.5's sampler behaviour with the V10.6 asymmetric marker.

    Identical to V10.5 in every sampled quantity -- settled fixture-free V9.5
    palette, layout families, vessel jitter, panel, target/tray, cameras,
    contact audit, clutter stability. The pendant is compiled into the scene
    the manifest row selects, so nothing here poses or resizes it.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = PACT_PLACE_V106_ENVIRONMENT_VERSION
    PENDANT_BODY = "pact_clutter_mount_v106"

    def _draw_theta(self):
        th = super()._draw_theta()
        row = self._pact_manifest_row or {}
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_v106_static_pendant"] = True
        th["pact_v106_pendant_body"] = self.PENDANT_BODY
        # Telemetry/expert only; never reaches a student observation.
        th["pact_v106_pose_id"] = row.get("pose_id")
        th["pact_v106_scene_sha256"] = row.get("pact_v106_scene_sha256")
        th["pact_v106_assembly_sha256"] = row.get("pact_v106_assembly_sha256")
        # The policy has no manifest row of its own, so the assembly it needs
        # for clearance telemetry has to travel through scene_params.
        th["pact_v106_x_m"] = row.get("pact_v106_x_m")
        th["pact_v106_r_neg_m"] = row.get("pact_v106_r_neg_m")
        th["pact_v106_r_pos_m"] = row.get("pact_v106_r_pos_m")
        return th

    def sample_task(self, *args, **kwargs):
        """Refuse a scene/pose/hash mismatch before the task is created."""
        row = self._pact_manifest_row or {}
        expected = row.get("pact_v106_scene_sha256")
        if expected:
            import hashlib
            from pathlib import Path

            # The datagen config carries scene_xml_paths, not scene_xml. Reading
            # the wrong attribute made this guard refuse every task instead of
            # verifying it.
            experiment_config = getattr(self, "config", None)
            sampler_config = getattr(experiment_config, "task_sampler_config", None)
            paths = list(getattr(sampler_config, "scene_xml_paths", None) or [])
            scene = paths[0] if paths else None
            if scene is None:
                raise ValueError("V10.6 row binds a scene hash but no scene is set")
            if len(set(paths)) > 1:
                raise ValueError(
                    f"V10.6 expects one scene, got {sorted(set(paths))}"
                )
            observed = hashlib.sha256(Path(str(scene)).read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError(
                    f"V10.6 scene hash mismatch for pose {row.get('pose_id')!r}: "
                    f"{observed} != {expected}"
                )
        return super().sample_task(*args, **kwargs)


class PactPlaceCorridorV1010FourObjectSampler(PactPlaceCorridorV106Sampler):
    """V10.7 environment with exactly four household objects active.

    Everything the V10.6/V10.7 lane defines is reused byte-for-byte: the three
    compiled static-pendant scenes and their hashes, routes, speeds, target
    distribution, the four layout families, two intrusion sides, three pendant
    poses, cameras, the 40-sensor proximity suite and the contact taxonomy.

    The single change is which clutter slots are live. The full eight-asset
    palette stays compiled, so observations and checkpoints remain
    shape-compatible; four slots are simply left out of the layout. The
    inherited ``_apply_theta`` already parks every compiled body whose slot is
    absent from the layout at its own ``park_m``, so parking needs no new
    mechanism and no new failure mode.

    Active   01 Soap_Bottle_30 (outbound vessel), 06 Soap_Bottle_11 (inbound
             vessel), 03 Plate_10, 04 Plate_22.
    Parked   00 Candle_2, 02 Mug_2, 05 (can), 07 Candle_1.

    Both route-bearing vessels stay active, so the corridor the task is about is
    unchanged; what is removed is decor.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v10_10_four_object"
    ACTIVE_CLUTTER_SLOTS = ("01", "03", "04", "06")
    INACTIVE_CLUTTER_SLOTS = ("00", "02", "05", "07")
    ACTIVE_CLUTTER_COUNT = 4
    EXPECTED_ACTIVE_UIDS = {
        "01": "Soap_Bottle_30", "03": "Plate_10",
        "04": "Plate_22", "06": "Soap_Bottle_11",
    }

    @classmethod
    def four_object_identity_sha256(cls, objects) -> str:
        """Which four objects are active -- stable across episodes.

        The positional hash below changes every episode because V9.3 applies
        per-episode clutter jitter, so it cannot be a row binding computed
        before sampling. Identity can.
        """
        import hashlib
        import json as _json

        payload = [
            {"palette_slot": str(o["palette_slot"]), "uid": str(o["uid"]),
             "role": str(o.get("role", ""))}
            for o in sorted(objects, key=lambda x: str(x["palette_slot"]))
        ]
        return hashlib.sha256(
            _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def four_object_layout_sha256(cls, objects) -> str:
        import hashlib
        import json as _json

        payload = [
            {
                "palette_slot": str(o["palette_slot"]), "uid": str(o["uid"]),
                "role": str(o.get("role", "")),
                "center_m": [float(v) for v in o["center_m"]],
                "quat_wxyz": [float(v) for v in o["quat_wxyz"]],
            }
            for o in sorted(objects, key=lambda x: str(x["palette_slot"]))
        ]
        return hashlib.sha256(
            _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _layout(self):
        import copy

        layout = copy.deepcopy(super()._layout())
        by_slot = {str(o["palette_slot"]): o for o in layout["objects"]}
        missing = [s for s in self.ACTIVE_CLUTTER_SLOTS if s not in by_slot]
        if missing:
            raise ValueError(f"V10.10 active slots absent from the layout: {missing}")
        for slot, uid in self.EXPECTED_ACTIVE_UIDS.items():
            if str(by_slot[slot]["uid"]) != uid:
                raise ValueError(
                    f"V10.10 slot {slot} carries {by_slot[slot]['uid']!r}, "
                    f"expected {uid!r}")
        active = [by_slot[s] for s in self.ACTIVE_CLUTTER_SLOTS]
        if len(active) != self.ACTIVE_CLUTTER_COUNT:
            raise ValueError(f"V10.10 activated {len(active)} slots, expected 4")
        parked = sorted(set(by_slot) - set(self.ACTIVE_CLUTTER_SLOTS))
        if parked != sorted(self.INACTIVE_CLUTTER_SLOTS):
            raise ValueError(f"V10.10 parked slots {parked} != "
                             f"{sorted(self.INACTIVE_CLUTTER_SLOTS)}")
        layout["objects"] = active
        layout["active_clutter_slots"] = list(self.ACTIVE_CLUTTER_SLOTS)
        layout["inactive_clutter_slots"] = list(self.INACTIVE_CLUTTER_SLOTS)
        layout["active_clutter_count"] = self.ACTIVE_CLUTTER_COUNT
        layout["active_clutter_uids"] = {
            str(o["palette_slot"]): str(o["uid"]) for o in active}
        layout["four_object_identity_sha256"] = self.four_object_identity_sha256(active)
        layout["four_object_layout_sha256"] = self.four_object_layout_sha256(active)
        layout["layout_id"] = f"{layout['layout_id']}_v1010_4obj"
        return layout

    def _draw_theta(self):
        th = super()._draw_theta()
        layout = th.get("pact_clutter_layout") or {}
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_v1010_active_clutter_slots"] = list(self.ACTIVE_CLUTTER_SLOTS)
        th["pact_v1010_inactive_clutter_slots"] = list(self.INACTIVE_CLUTTER_SLOTS)
        th["pact_v1010_active_clutter_count"] = self.ACTIVE_CLUTTER_COUNT
        th["pact_v1010_active_clutter_uids"] = layout.get("active_clutter_uids")
        th["pact_v1010_identity_sha256"] = layout.get("four_object_identity_sha256")
        th["pact_v1010_layout_sha256"] = layout.get("four_object_layout_sha256")
        th["pact_v1010_sampler_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        active = list(getattr(self, "_pact_active_clutter_names", []))
        if len(active) != self.ACTIVE_CLUTTER_COUNT:
            raise ValueError(
                f"V10.10 expected {self.ACTIVE_CLUTTER_COUNT} active clutter bodies, "
                f"got {len(active)}: {active}")


PACT_PLACE_V10_LANE_ENVIRONMENT_VERSIONS = (
    PACT_PLACE_V10_ENVIRONMENT_VERSION,
    PACT_PLACE_V102_ENVIRONMENT_VERSION,
)


class PactPlaceCorridorV10CompoundPendantSampler(PactPlaceCorridorV93Sampler):
    """Settled V9.5 clutter plus one fixed connected compound ceiling pendant."""

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v10_compound_pendant"
    PENDANT_BODY = "pact_clutter_mount_v10"

    def _pendant_assembly(self) -> dict[str, Any]:
        from pact_place_v10_geometry import build_assembly, build_lobe

        payload = dict(
            (self._pact_manifest_row or {}).get("pact_v10_pendant_assembly") or {}
        )
        if not payload:
            raise ValueError("V10 requires pact_v10_pendant_assembly")
        if payload.get("components"):
            return payload
        lobes = []
        for item in payload.get("lobes") or []:
            lobes.append(
                build_lobe(
                    center_x_m=item["center_m"][0],
                    center_y_m=item["center_m"][1],
                    center_z_m=item["center_m"][2],
                    half_x_m=item["half_m"][0],
                    half_y_m=item["half_m"][1],
                    half_z_m=item["half_m"][2],
                )
            )
        return build_assembly(lobes)

    def _pendant_parked(self) -> bool:
        return bool((self._pact_manifest_row or {}).get("pact_v10_pendant_parked"))

    def _route_params(self) -> dict[str, Any]:
        return dict((self._pact_manifest_row or {}).get("pact_v10_route") or {})

    def _draw_theta(self):
        th = super()._draw_theta()
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        parked = self._pendant_parked()
        th["pact_v10_pendant_parked"] = parked
        th["pact_v10_pendant_assembly"] = {} if parked else self._pendant_assembly()
        th["pact_v10_route"] = {} if parked else self._route_params()
        th["pact_v99_pendant_parked"] = True
        th["pact_v98_pendant_lateral_bow"] = False
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        from pact_place_v10_compound_pendant_contract import PENDANT_BODY
        from pact_place_v10_geometry import active_components
        from pact_place_v10_scene import pose_assembly_on_data

        parked = bool(th.get("pact_v10_pendant_parked"))
        assembly = None if parked else dict(th.get("pact_v10_pendant_assembly") or {})
        pose_assembly_on_data(
            env.current_model,
            env.current_data,
            assembly,
            parked=parked,
        )
        if parked or not assembly:
            th["pact_v10_active_mount_body"] = None
            mujoco.mj_forward(env.current_model, env.current_data)
            return
        for item in active_components(assembly):
            center = list(map(float, item["center_m"]))
            half = list(map(float, item["half_m"]))
            hazard = {
                **item,
                "name": f"pact_mounted_{item['name']}",
                "body": PENDANT_BODY,
                "role": "ceiling_fixture",
                "center": center,
                "half": half,
                "phase": "inbound_and_outbound",
                "kinematic": True,
                "lateral_lane_cost_m": 0.0,
            }
            th.setdefault("obstacle_aabbs", []).append([center, half])
            th.setdefault("pact_v9_hazards", []).append(hazard)
        th["pact_v10_active_mount_body"] = PENDANT_BODY
        mujoco.mj_forward(env.current_model, env.current_data)


class PactPlaceCorridorV102RaisedPendantSampler(
    PactPlaceCorridorV10CompoundPendantSampler
):
    """V10.2 raised, collision-legible pendant on the compiled V10 scene.

    Same scene, same parked legacy mounts, same V9.5 clutter families. The only
    differences are the registered raised assembly, the V10.2 environment
    marker, and the route-piece speed schedule the marker unlocks.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = PACT_PLACE_V102_ENVIRONMENT_VERSION

    def _pendant_assembly(self) -> dict[str, Any]:
        payload = dict(
            (self._pact_manifest_row or {}).get("pact_v10_pendant_assembly") or {}
        )
        if not payload or not payload.get("components"):
            raise ValueError(
                "V10.2 requires an explicit pact_v10_pendant_assembly with components"
            )
        from pact_place_v102_geometry import PROBE_LABEL_V102

        if str(payload.get("probe_label") or "") != PROBE_LABEL_V102:
            raise ValueError(
                "V10.2 refuses an assembly that is not the registered raised probe"
            )
        return payload

    def _route_params(self) -> dict[str, Any]:
        from pact_place_v102_route import route_is_v102

        route = dict((self._pact_manifest_row or {}).get("pact_v10_route") or {})
        scene_marker = {
            "pact_place_environment_version": self.PACT_PLACE_ENVIRONMENT_VERSION
        }
        if not route_is_v102(scene_marker, route):
            raise ValueError(
                "V10.2 requires the exact registered route markers and speed-schedule hash"
            )
        return route


class PactPlaceTCPMoveSequence(TCPMoveSequence):
    """Place-corridor TCP sequence with a repaired empty-gripper detector.

    Upstream ``TCPMoveSequence.check_failure`` is unchanged. This subclass is
    constructed only from ``PactPlaceCorridorPolicy._sequence``.

    Fix A: the empty-gripper check is a drop detector for transport. It is not
    armed on ``placement_descent``, whose next primitive is the scripted release.

    Fix B: during holding transport, the empty predicate must hold for
    ``EMPTY_GRIPPER_PERSIST_STEPS`` consecutive control steps. This is not a
    threshold change; ``gripper_empty_threshold`` stays at the policy default
    (0.002 m). The measured glitch is a one-step 8.5 mm → 0.00 mm sample.
    """

    EMPTY_GRIPPER_PERSIST_STEPS = 3
    EMPTY_GRIPPER_DISARMED_SEGMENTS = frozenset({"placement_descent"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._empty_gripper_streak = 0

    def reset(self) -> None:
        super().reset()
        self._empty_gripper_streak = 0

    def _current_move_segment_name(self) -> str | None:
        if not self.move_segments:
            return None
        if self.move_seg_idx is None:
            return self.move_segments[0].name
        if self.move_seg_idx < len(self.move_segments):
            return self.move_segments[self.move_seg_idx].name
        return self.move_segments[-1].name

    def _empty_gripper_sample(self) -> bool:
        if not self.is_holding_object:
            return False
        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        gripper = self.robot_view.get_gripper(gripper_mg_id)
        return (
            gripper.inter_finger_dist
            < gripper.inter_finger_dist_range[0] + self.gripper_empty_threshold
        )

    def _persistent_empty_gripper_failure(self) -> bool:
        disarmed = (
            self._current_move_segment_name() in self.EMPTY_GRIPPER_DISARMED_SEGMENTS
        )
        if disarmed or not self.is_holding_object:
            self._empty_gripper_streak = 0
            return False
        if self._empty_gripper_sample():
            self._empty_gripper_streak += 1
            if self._empty_gripper_streak >= self.EMPTY_GRIPPER_PERSIST_STEPS:
                gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
                gripper = self.robot_view.get_gripper(gripper_mg_id)
                log.info(
                    "Object is not in grasp! "
                    f"{gripper.inter_finger_dist:.05f} < "
                    f"{gripper.inter_finger_dist_range[0] + self.gripper_empty_threshold:.05f} "
                    f"for {self._empty_gripper_streak} consecutive steps"
                )
                return True
            return False
        self._empty_gripper_streak = 0
        return False

    def _tcp_tracking_failure(self) -> bool:
        if self.move_seg_idx is None:
            return False
        from scipy.spatial.transform import Rotation as R

        curr_target_pose = self.get_current_target_pose()
        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        gripper = self.robot_view.get_gripper(gripper_mg_id)
        trf = np.linalg.inv(gripper.leaf_frame_to_world) @ curr_target_pose
        pos_err = np.linalg.norm(trf[:3, 3])
        rot_err = R.from_matrix(trf[:3, :3]).magnitude()
        return bool(
            pos_err > self.tcp_pos_err_threshold
            or rot_err > self.tcp_rot_err_threshold
        )

    def check_failure(self) -> bool:
        if self._persistent_empty_gripper_failure():
            return True
        return self._tcp_tracking_failure()


class PactPlaceCorridorPolicy(PickAndPlacePlannerPolicy):
    """Obstacle-aware inbound pick and higher-clearance outbound carry/place."""

    _AABB_SAMPLES = (
        (-1.0, 0.0, 0.0),
        (-1.0, 0.7, 0.0),
        (-1.0, -0.7, 0.0),
        (-1.0, 0.0, 0.7),
        (-1.0, 0.0, -0.7),
        (0.0, 0.9, 0.0),
        (0.0, -0.9, 0.0),
        (0.0, 0.0, -0.9),
    )
    # V9 can reuse its movable route blocker in both travel directions.  The
    # empty inbound envelope is narrower than the loaded outbound envelope.
    INBOUND_ENVELOPE_HALF_Y = 0.11
    INBOUND_SAFE_GAP = 0.04
    OUTBOUND_ENVELOPE_HALF_Y = 0.15
    OUTBOUND_SAFE_GAP = 0.14
    # Four centimetres is the deliberate bottle surface gap.  V9.2 composes it
    # with the panel's larger clearance instead of treating them as opposed
    # chicane walls.
    V9_VESSEL_SAFE_GAP = 0.04
    # Mounted fixtures are rigid scene structure.  Their route envelope is
    # deliberately smaller than the loaded-vessel envelope so the preview
    # remains feasible while still producing a geometry-dependent detour.
    MOUNTED_FIXTURE_ENVELOPE_HALF_Y = 0.10
    # V9.8 ceiling pendant only. Measured wrist lag toward the centreline
    # (0.208 m on −y bows, 0.108 m on +y) plus 4 mm. Wall fixtures keep 0.10.
    CEILING_FIXTURE_ENVELOPE_HALF_Y_NEG = 0.212
    CEILING_FIXTURE_ENVELOPE_HALF_Y_POS = 0.112
    MOUNTED_FIXTURE_SAFE_GAP = 0.025
    V95_MIN_FIXTURE_BOW_M = 0.040
    V95_MIN_PLANNED_LINK_CLEARANCE_M = 0.020
    V95_BOW_SEARCH_STEP_M = 0.010
    OUTBOUND_CARRY_RAISE_M = 0.0
    OUTBOUND_PASS_SPEED = 0.045
    OUTSIDE_STAGING_X_M = TUBE_X0 - 0.10
    V9_OUTSIDE_STAGING_X_M = TUBE_X0 - 0.14
    V93_OUTSIDE_STAGING_X_M = TUBE_X0 - 0.22
    # Development probes at +/-12 mm did not improve outbound execution. Keep
    # the validated corridor expert's selected grasp unchanged.
    GRASP_WORLD_Z_OFFSET_M = 0.0
    PASS_SPEED = 0.045
    APERTURE_EDGE_RESERVE = 0.02
    RELEASE_CLEARANCE_M = 0.005  # release just above the tray, not pressed into it
    # Bound the outbound_approach twist so IK tracking stays in the carry plane.
    # Rows 4 and 17 aborted on 8.8 cm of vertical deviation across one long segment.
    OUTBOUND_APPROACH_MAX_STEP_M = 0.04
    SETTLE_WINDOW_STEPS = 25

    def __init__(self, config, task) -> None:
        super().__init__(config, task)
        self.behavior_class = "straight"
        self.inbound_deflected = False
        self.outbound_deflected = False
        self._pact_place_bow_diagnostics = self._empty_bow_diagnostics()
        self._pact_place_v99_route = {
            "inbound_pendant": self._v99_empty_route_record(),
            "outbound_pendant": self._v99_empty_route_record(),
        }
        self._pact_place_v10_route = {
            "inbound_pendant": self._v99_empty_route_record(),
            "outbound_pendant": self._v99_empty_route_record(),
        }
        self._pact_place_v102_frames: list[dict[str, Any]] = []
        self._pact_place_v104_frames: list[dict[str, Any]] = []
        self._pact_place_v104_speed_amendment: dict[str, Any] = {}
        self._pact_place_v106_frames: list[dict[str, Any]] = []
        self._pact_place_v106_speed_amendment: dict[str, Any] = {}
        self._pact_v106_pendant_geom_ids: list[int] | None = None
        self._pact_v106_probe_ids: list[int] | None = None
        self._pact_v106_boxes = None
        self._pact_v106_risk_boxes = None
        self._pact_v106_shape_cache = None
        self._pact_v106_assembly = None
        self._pact_place_v105_frames: list[dict[str, Any]] = []
        self._pact_place_v105_speed_amendment: dict[str, Any] = {}
        self._pact_v105_pendant_geom_ids: list[int] | None = None
        self._pact_v105_probe_ids: list[int] | None = None
        self._pact_v105_boxes = None
        self._pact_v105_risk_boxes = None
        self._pact_v105_shape_cache = None
        self._pact_v105_assembly = None
        self._pact_v104_pendant_geom_ids: list[int] | None = None
        self._pact_v104_probe_ids: list[int] | None = None
        self._pact_v104_boxes = None
        self._pact_v104_shape_cache = None
        self._pact_place_last_tcp_m: np.ndarray | None = None
        self._pact_place_last_sim_time_s: float | None = None
        self._pact_v102_component_geoms: list[tuple[str, int]] | None = None
        self._pact_v102_probe_ids: dict[bool, list[int]] = {}
        self._sensor_cam_ids: list[int] | None = None
        self._pact_detected_hazard_names: set[str] = set()
        self._pact_detected_hazards: list[dict[str, Any]] = []
        self._pact_maneuver_interactions: list[dict[str, Any]] = []
        self._pact_active_maneuver: str | None = None

    def _v9_enabled(self) -> bool:
        version = str(
            (getattr(self.task, "scene_params", {}) or {}).get(
                "pact_place_environment_version", ""
            )
        )
        return version in {
            "pact_place_corridor_v9",
            "pact_place_corridor_v9_2",
            "pact_place_corridor_v9_3",
            "pact_place_corridor_v9_4_mounted_preview",
            "pact_place_corridor_v9_5_low_wall",
            "pact_place_corridor_v9_8_pendant",
            "pact_place_corridor_v9_9_pendant",
            PACT_PLACE_V10_ENVIRONMENT_VERSION,
            PACT_PLACE_V102_ENVIRONMENT_VERSION,
            PACT_PLACE_V105_ENVIRONMENT_VERSION,
            *PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS,
        }

    def _environment_version(self) -> str:
        return str(
            (getattr(self.task, "scene_params", {}) or {}).get(
                "pact_place_environment_version", ""
            )
        )

    def _v99_enabled(self) -> bool:
        return self._environment_version() == "pact_place_corridor_v9_9_pendant"

    def _v10_enabled(self) -> bool:
        return self._environment_version() in PACT_PLACE_V10_LANE_ENVIRONMENT_VERSIONS

    def _v104_enabled(self) -> bool:
        return self._environment_version() == PACT_PLACE_V104_ENVIRONMENT_VERSION

    def _v105_enabled(self) -> bool:
        return self._environment_version() == PACT_PLACE_V105_ENVIRONMENT_VERSION

    def _v106_enabled(self) -> bool:
        return self._environment_version() in PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS

    def _v102_enabled(self) -> bool:
        return self._environment_version() == PACT_PLACE_V102_ENVIRONMENT_VERSION

    @classmethod
    def _ceiling_fixture_envelope_half_y(cls, waypoint_side: float) -> float:
        if float(waypoint_side) < 0.0:
            return float(cls.CEILING_FIXTURE_ENVELOPE_HALF_Y_NEG)
        return float(cls.CEILING_FIXTURE_ENVELOPE_HALF_Y_POS)

    def _mounted_fixture_bow_envelope_half_y(
        self, fixture_role: str, waypoint_side: float
    ) -> float:
        if (
            fixture_role == "ceiling_fixture"
            and self._environment_version() == "pact_place_corridor_v9_8_pendant"
        ):
            return self._ceiling_fixture_envelope_half_y(waypoint_side)
        return float(self.MOUNTED_FIXTURE_ENVELOPE_HALF_Y)

    @staticmethod
    def _mounted_fixture_roles(
        environment_version: str,
        *,
        pendant_lateral_bow: bool = False,
    ) -> tuple[str, ...]:
        """Return the mounted-fixture roles that receive a lateral bow.

        V9.4/V9.5 wall fixtures are genuine lateral obstacles and always bow.
        V9.8's ceiling pendant hangs above the TCP (bottom z=1.10 vs TCP
        z≈0.885) so the default is no lateral bow. Set
        ``pact_v98_pendant_lateral_bow`` on the row to opt into the
        TCP-only sideways detour; that path is the measured phantom.
        """
        if environment_version == "pact_place_corridor_v9_4_mounted_preview":
            return ("wall_fixture", "ceiling_fixture")
        if environment_version == "pact_place_corridor_v9_5_low_wall":
            return ("wall_fixture",)
        if (
            environment_version == "pact_place_corridor_v9_8_pendant"
            and pendant_lateral_bow
        ):
            return ("ceiling_fixture",)
        # V9.9 uses the full-route lane primitive, not the V9.8 ceiling bow.
        return ()

    def _preferred_v9_waypoint_side(self) -> float | None:
        """Return +1/-1 for the single lane left open by the active panel."""
        th = getattr(self.task, "scene_params", {}) or {}
        if th.get("pact_v9_legacy_panel_active"):
            wall = str(th.get("protr_wall") or "")
            if wall == "left":
                return -1.0
            if wall == "right":
                return 1.0
        layout = th.get("pact_clutter_layout") or {}
        direction = str(layout.get("expected_bow_direction") or "")
        if direction == "+y":
            return 1.0
        if direction == "-y":
            return -1.0
        return None

    def _embed_T(self) -> np.ndarray:
        """Return the world-from-task transform used by enclosure experts."""
        embed = (getattr(self.task, "scene_params", {}) or {}).get("embed")
        if not embed:
            return np.eye(4)
        base_x, base_y, yaw = map(float, embed)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        transform[:3, 3] = np.asarray([base_x, base_y, 0.0], dtype=float)
        return transform

    def _hazard_list(self) -> list[dict[str, Any]]:
        """Return panel and V9 vessel hazards in task-local coordinates."""
        th = getattr(self.task, "scene_params", {}) or {}
        hazards: list[dict[str, Any]] = []
        if th.get("protrusion_present") and "protr_center" in th:
            hazards.append(
                {
                    "name": str(th.get("protr_name", "pact_intrusion")),
                    "role": "panel",
                    "avoidance": "panel_bow",
                    "center": [float(value) for value in th["protr_center"]],
                    "half": [float(value) for value in th["protr_half"]],
                    "wall": str(th.get("protr_wall", "right")),
                }
            )
        for hazard in list(th.get("pact_v9_hazards") or []):
            if "center" not in hazard or "half" not in hazard:
                continue
            hazards.append(
                {
                    **hazard,
                    "name": str(hazard["name"]),
                    "center": [float(value) for value in hazard["center"]],
                    "half": [float(value) for value in hazard["half"]],
                    "avoidance": {
                        "inbound_vessel": "vessel_deflect",
                        "wall_fixture": "mounted_fixture_bow",
                        "ceiling_fixture": "mounted_fixture_bow",
                    }.get(str(hazard.get("role")), "vessel_bow"),
                }
            )
        return hazards

    def _sensor_poses(self):
        model = self.task.env.current_model
        data = self.task.env.current_data
        if self._sensor_cam_ids is None:
            self._sensor_cam_ids = [
                index
                for index in range(int(model.ncam))
                if "_sensor_" in (model.camera(index).name or "")
            ]
        for camera_id in self._sensor_cam_ids:
            yield (
                np.asarray(data.cam_xpos[camera_id], dtype=float),
                np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3),
            )

    def _protrusion_detected(
        self, allowed_roles: set[str] | None = None
    ) -> dict[str, Any] | None:
        """Return the identity of the first currently detectable hazard.

        This is the same sampled AABB surface gate as the reach expert: the
        real sensor poses, 22.5 degree half-FOV, and 0.85 m derated range are
        used.  Returning the hazard record prevents a later detection from
        accidentally reusing the panel's center and wall.
        """
        if not self._v9_enabled():
            return None
        rng_eff = SENSOR_RANGE * SENSOR_RANGE_DERATE
        transform = self._embed_T()
        for hazard in self._hazard_list():
            if hazard["name"] in self._pact_detected_hazard_names:
                continue
            if allowed_roles is not None and hazard.get("role") not in allowed_roles:
                continue
            center = np.asarray(hazard["center"], dtype=float)
            half = np.asarray(hazard["half"], dtype=float)
            points = [
                (transform @ np.append(center + half * np.asarray(sample), 1.0))[:3]
                for sample in self._AABB_SAMPLES
            ]
            for position, xmat in self._sensor_poses():
                forward = -xmat[:, 2]
                for point in points:
                    delta = point - position
                    distance = float(np.linalg.norm(delta))
                    if distance < 1e-9 or distance > rng_eff:
                        continue
                    if float(np.dot(delta / distance, forward)) > SENSOR_HALF_FOV_COS:
                        return dict(hazard)
        return None

    def _phase_for_hazard(self, hazard: dict[str, Any]) -> bool:
        try:
            phase = str(self.get_phase())
        except Exception:
            return False
        role = str(hazard.get("role"))
        if role == "inbound_vessel":
            return phase in {"approach", "insert", "advance", "pregrasp"}
        if role in {"wall_fixture", "ceiling_fixture"}:
            return phase.startswith("inbound") or phase.startswith("outbound")
        if role in {"panel", "outbound_vessel"}:
            return phase.startswith("outbound")
        return False

    def _active_maneuver_for_phase(self) -> str | None:
        try:
            phase = str(self.get_phase())
        except Exception:
            return self._pact_active_maneuver
        if phase in {"deflect", "pass_protrusion"}:
            return self._pact_active_maneuver or "vessel deflect"
        if phase.startswith("outbound") and "pass" in phase:
            if any(
                self._pact_place_bow_diagnostics.get(prefix, {}).get(
                    "accepted_bow_m", 0.0
                )
                for prefix in (
                    "outbound_wall_fixture",
                    "outbound_ceiling_fixture",
                )
            ):
                return "mounted fixture bow"
            if self._pact_place_bow_diagnostics.get("outbound_vessel", {}).get(
                "accepted_bow_m", 0.0
            ):
                return "vessel bow"
            return "panel bow"
        return self._pact_active_maneuver

    def _handle_detected_hazard(self, hazard: dict[str, Any]) -> None:
        name = str(hazard["name"])
        phase = str(self.get_phase())
        previous = self._active_maneuver_for_phase()
        if previous and hazard.get("avoidance") not in {previous, None}:
            self._pact_maneuver_interactions.append(
                {
                    "step": int(getattr(self, "_pact_place_control_step", 0)),
                    "policy_phase": phase,
                    "existing_maneuver": previous,
                    "detected_hazard": name,
                    "action": "reported_without_silent_overwrite",
                }
            )
        self._pact_detected_hazard_names.add(name)
        self._pact_detected_hazards.append(
            {
                **hazard,
                # This online expert-policy diagnostic is an AABB/cone proxy.  It
                # must never be presented as evidence from the production PACT
                # proximity tensor; raw sensor admission is performed separately.
                "detection_source": "sampled_aabb_cone_proxy_not_raw_proximity",
                "step": int(getattr(self, "_pact_place_control_step", 0)),
                "policy_phase": phase,
            }
        )
        if hazard.get("avoidance") == "vessel_deflect" and self._phase_for_hazard(hazard):
            if self.inbound_deflected:
                return
            self._replan_on_detection(hazard)
        elif hazard.get("avoidance") in {
            "panel_bow",
            "vessel_bow",
            "mounted_fixture_bow",
        }:
            self._pact_active_maneuver = {
                "panel_bow": "panel bow",
                "vessel_bow": "vessel bow",
                "mounted_fixture_bow": "mounted fixture bow",
            }[str(hazard.get("avoidance"))]

    def _inbound_deflect_segments(
        self, current: np.ndarray, target: np.ndarray, hazard: dict[str, Any]
    ) -> list[TCPMoveSegment] | None:
        center_local = np.asarray(hazard["center"], dtype=float)
        half = np.asarray(hazard["half"], dtype=float)
        center = (self._embed_T() @ np.append(center_local, 1.0))[:3]
        delta_x = float(target[0] - current[0])
        if abs(delta_x) < 1e-6:
            return None
        t_cross = float((center[0] - current[0]) / delta_x)
        if not 0.02 < t_cross < 0.98:
            return None
        cross = current + t_cross * (target - current)
        obstacle_side = 1.0 if center[1] >= 0.0 else -1.0
        inner_face_y = center[1] - obstacle_side * half[1]
        straight_clearance = (
            obstacle_side * (inner_face_y - cross[1]) - self.INBOUND_ENVELOPE_HALF_Y
        )
        required_bow = self.INBOUND_SAFE_GAP - straight_clearance
        lateral_limit = max(
            0.0,
            float((getattr(self.task, "scene_params", {}) or {}).get("ap_w", 0.85))
            / 2.0
            - self.INBOUND_ENVELOPE_HALF_Y
            - self.APERTURE_EDGE_RESERVE,
        )
        if required_bow <= 0.0 or required_bow > lateral_limit:
            return None
        travel_direction = 1.0 if delta_x > 0.0 else -1.0
        longitudinal_pad = 0.05
        before_x = center[0] - travel_direction * (half[0] + longitudinal_pad)
        after_x = center[0] + travel_direction * (half[0] + longitudinal_pad)
        t_before = float(np.clip((before_x - current[0]) / delta_x, 0.04, 0.90))
        t_after = float(np.clip((after_x - current[0]) / delta_x, t_before + 0.02, 0.96))
        before = current + t_before * (target - current)
        after = current + t_after * (target - current)
        waypoint_y = float(
            np.clip(
                cross[1] - obstacle_side * required_bow,
                -lateral_limit,
                lateral_limit,
            )
        )
        before[1] = waypoint_y
        after[1] = waypoint_y
        return [before, after]

    def _replan_on_detection(self, hazard: dict[str, Any] | str) -> None:
        """Rebuild only the remaining inbound sequence around ``hazard``."""
        if isinstance(hazard, str):
            hazard = next(
                item for item in self._hazard_list() if item["name"] == hazard
            )
        if hazard.get("avoidance") != "vessel_deflect":
            return
        if self.action_idx >= len(self.action_primitives):
            return
        current_primitive = self.action_primitives[self.action_idx]
        if not isinstance(current_primitive, TCPMoveSequence):
            return
        if len(current_primitive.move_segments) < 2:
            return
        current = self.robot_view.get_move_group(
            self.robot_view.get_gripper_movegroup_ids()[0]
        ).leaf_frame_to_world.copy()
        target_pregrasp = current_primitive.move_segments[-2].end_pose.copy()
        target_grasp = current_primitive.move_segments[-1].end_pose.copy()
        waypoints = self._inbound_deflect_segments(
            current, target_pregrasp[:3, 3].copy(), hazard
        )
        if waypoints is None:
            log.warning("[PactPlace] detected vessel has no admitted inbound detour")
            return
        rotation = target_pregrasp[:3, :3]
        waypoint_poses = []
        for point in waypoints:
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = point
            waypoint_poses.append(pose)
        replacement = self._sequence(
            [
                TCPMoveSegment(
                    name="deflect",
                    start_pose=current,
                    end_pose=waypoint_poses[0],
                    speed=self.policy_config.speed_fast,
                ),
                TCPMoveSegment(
                    name="pass_protrusion",
                    start_pose=waypoint_poses[0],
                    end_pose=waypoint_poses[1],
                    speed=self.PASS_SPEED,
                ),
                TCPMoveSegment(
                    name="advance",
                    start_pose=waypoint_poses[1],
                    end_pose=target_pregrasp,
                    speed=self.policy_config.speed_slow,
                ),
                TCPMoveSegment(
                    name="grasp",
                    start_pose=target_pregrasp,
                    end_pose=target_grasp,
                    speed=self.policy_config.speed_slow,
                ),
            ],
            holding=False,
        )
        self.action_primitives[self.action_idx] = replacement
        self.inbound_deflected = True
        self._pact_active_maneuver = "vessel deflect"
        self.behavior_class = "scripted_inbound_vessel_deflect"
        log.info("[PactPlace] DEFLECT around detected vessel %s", hazard["name"])

    @staticmethod
    def _empty_bow_record() -> dict[str, Any]:
        return {
            "planned_bow_m": 0.0,
            "accepted_bow_m": 0.0,
            "bow_fallback_taken": False,
            "straight_clearance_m": None,
            "required_clearance_m": None,
            "response_source": None,
            "waypoint_y_m": None,
            "waypoint_side": None,
        }

    @classmethod
    def _empty_bow_diagnostics(cls) -> dict[str, dict[str, Any]]:
        return {
            "inbound": cls._empty_bow_record(),
            "outbound": cls._empty_bow_record(),
        }

    def _record_bow(
        self,
        prefix: str,
        *,
        planned_bow_m: float,
        accepted_bow_m: float,
        bow_fallback_taken: bool,
        straight_clearance_m: float | None = None,
        required_clearance_m: float | None = None,
        response_source: str | None = None,
        waypoint_y_m: float | None = None,
        waypoint_side: float | None = None,
    ) -> None:
        previous = self._pact_place_bow_diagnostics.get(
            prefix, self._empty_bow_record()
        )
        take_waypoint = float(planned_bow_m) + 1e-12 >= float(
            previous.get("planned_bow_m", 0.0)
        )
        waypoint_y_out = previous.get("waypoint_y_m")
        waypoint_side_out = previous.get("waypoint_side")
        if take_waypoint and waypoint_y_m is not None:
            waypoint_y_out = float(waypoint_y_m)
            waypoint_side_out = (
                None if waypoint_side is None else float(waypoint_side)
            )
        self._pact_place_bow_diagnostics[prefix] = {
            # A compound hazard can be evaluated against several consecutive
            # segments.  Preserve the strongest admitted bow instead of letting
            # a later non-crossing segment erase it with zeros.
            "planned_bow_m": max(
                float(previous.get("planned_bow_m", 0.0)), float(planned_bow_m)
            ),
            "accepted_bow_m": max(
                float(previous.get("accepted_bow_m", 0.0)), float(accepted_bow_m)
            ),
            "bow_fallback_taken": bool(
                previous.get("bow_fallback_taken") or bow_fallback_taken
            ),
            "straight_clearance_m": (
                previous.get("straight_clearance_m")
                if straight_clearance_m is None
                else float(straight_clearance_m)
            ),
            "required_clearance_m": (
                previous.get("required_clearance_m")
                if required_clearance_m is None
                else float(required_clearance_m)
            ),
            "response_source": response_source or previous.get("response_source"),
            "waypoint_y_m": waypoint_y_out,
            "waypoint_side": waypoint_side_out,
        }

    def _get_placement_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj,
        place_receptacle,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preplace_pose, place_pose, postplace_pose = super()._get_placement_poses(
            grasp_pose_world,
            pickup_obj,
            place_receptacle,
        )
        place_pose = place_pose.copy()
        place_pose[2, 3] += self.RELEASE_CLEARANCE_M
        if not self.check_feasible_ik(place_pose):
            raise ValueError("IK failed for place pose with release clearance")
        return preplace_pose, place_pose, postplace_pose

    @staticmethod
    def _place_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        return pose

    @staticmethod
    def _route_side_at_x(segment: TCPMoveSegment, obstacle_x: float) -> float:
        """Choose the lane occupied where this segment crosses an obstacle."""
        start = segment.start_pose[:3, 3]
        end = segment.end_pose[:3, 3]
        dx = float(end[0] - start[0])
        t = 0.5 if abs(dx) < 1e-9 else float(np.clip((obstacle_x - start[0]) / dx, 0.0, 1.0))
        cross_y = float(start[1] + t * (end[1] - start[1]))
        return 1.0 if cross_y >= 0.0 else -1.0

    def _bow_segment(
        self,
        segment: TCPMoveSegment,
        *,
        prefix: str,
        envelope_half_y: float,
        safe_gap: float,
        center: np.ndarray | None = None,
        half: np.ndarray | None = None,
        preferred_waypoint_side: float | None = None,
    ) -> tuple[list[TCPMoveSegment], bool]:
        th = getattr(self.task, "scene_params", {}) or {}
        if center is None or half is None:
            if not th.get("protrusion_present") or "protr_center" not in th:
                self._record_bow(
                    prefix,
                    planned_bow_m=0.0,
                    accepted_bow_m=0.0,
                    bow_fallback_taken=False,
                )
                return [segment], False
            center = np.asarray(th["protr_center"], dtype=float)
            half = np.asarray(th["protr_half"], dtype=float)
        else:
            center = np.asarray(center, dtype=float)
            half = np.asarray(half, dtype=float)
        if center.shape != (3,) or half.shape != (3,):
            self._record_bow(
                prefix,
                planned_bow_m=0.0,
                accepted_bow_m=0.0,
                bow_fallback_taken=False,
            )
            return [segment], False
        start = segment.start_pose[:3, 3].copy()
        end = segment.end_pose[:3, 3].copy()
        delta_x = float(end[0] - start[0])
        if abs(delta_x) < 1e-6:
            self._record_bow(
                prefix,
                planned_bow_m=0.0,
                accepted_bow_m=0.0,
                bow_fallback_taken=False,
            )
            return [segment], False
        t_cross = float((center[0] - start[0]) / delta_x)
        if not 0.02 < t_cross < 0.98:
            self._record_bow(
                prefix,
                planned_bow_m=0.0,
                accepted_bow_m=0.0,
                bow_fallback_taken=False,
            )
            return [segment], False
        cross = start + t_cross * (end - start)
        if preferred_waypoint_side is None:
            obstacle_side = 1.0 if center[1] >= 0.0 else -1.0
            waypoint_side = -obstacle_side
            open_face_y = center[1] + waypoint_side * half[1]
        else:
            waypoint_side = 1.0 if preferred_waypoint_side >= 0.0 else -1.0
            open_face_y = center[1] + waypoint_side * half[1]
        straight_clearance = (
            waypoint_side * (cross[1] - open_face_y) - envelope_half_y
        )
        required_bow = safe_gap - straight_clearance
        if required_bow <= 0.0:
            self._record_bow(
                prefix,
                planned_bow_m=0.0,
                accepted_bow_m=0.0,
                bow_fallback_taken=False,
            )
            return [segment], False
        aperture_width = float(th.get("ap_w", 0.85))
        lateral_limit = max(
            0.0,
            aperture_width / 2
            - envelope_half_y
            - self.APERTURE_EDGE_RESERVE,
        )
        travel_direction = 1.0 if delta_x > 0.0 else -1.0
        before_x = center[0] - travel_direction * (half[0] + 0.08)
        after_x = center[0] + travel_direction * (half[0] + 0.08)
        t_before = float(np.clip((before_x - start[0]) / delta_x, 0.04, 0.90))
        t_after = float(np.clip((after_x - start[0]) / delta_x, t_before + 0.02, 0.96))
        before = start + t_before * (end - start)
        after = start + t_after * (end - start)
        waypoint_y = float(
            np.clip(
                cross[1] + waypoint_side * required_bow,
                -lateral_limit,
                lateral_limit,
            )
        )
        before[1] = waypoint_y
        after[1] = waypoint_y
        rotation = segment.end_pose[:3, :3]
        pose_before = self._place_pose(before, rotation)
        pose_after = self._place_pose(after, rotation)
        actual_bow = float(waypoint_side * (waypoint_y - cross[1]))
        self._record_bow(
            prefix,
            planned_bow_m=required_bow,
            accepted_bow_m=actual_bow,
            bow_fallback_taken=False,
            straight_clearance_m=straight_clearance,
            required_clearance_m=safe_gap,
            response_source="actual_episode_clearance_geometry",
            waypoint_y_m=waypoint_y,
            waypoint_side=waypoint_side,
        )
        log.info(
            f"[PactPlace] {prefix} DEFLECT: straight clearance "
            f"{straight_clearance * 100:.1f}cm -> y={waypoint_y:+.3f}, "
            f"required gap={safe_gap * 100:.1f}cm, "
            f"accepted bow={actual_bow * 100:.1f}cm, fallback=False"
        )
        approach_speed = (
            self.OUTBOUND_PASS_SPEED
            if prefix == "outbound"
            else self.policy_config.speed_fast
        )
        pass_speed = (
            self.OUTBOUND_PASS_SPEED if prefix == "outbound" else self.PASS_SPEED
        )
        exit_speed = (
            self.OUTBOUND_PASS_SPEED
            if prefix == "outbound"
            else self.policy_config.speed_slow
        )
        return (
            [
                TCPMoveSegment(
                    name=f"{prefix}_approach",
                    start_pose=segment.start_pose,
                    end_pose=pose_before,
                    speed=approach_speed,
                ),
                TCPMoveSegment(
                    name=f"{prefix}_pass",
                    start_pose=pose_before,
                    end_pose=pose_after,
                    speed=pass_speed,
                ),
                TCPMoveSegment(
                    name=f"{prefix}_exit",
                    start_pose=pose_after,
                    end_pose=segment.end_pose,
                    speed=exit_speed,
                ),
            ],
            True,
        )

    def _v95_exact_fixture_clearance(self, pose: np.ndarray, fixture: dict[str, Any]) -> float | None:
        """Solve one IK pose and measure exact distal-link clearance to the fixture."""
        try:
            from pact_geom_distance import true_distance
        except ImportError:
            return None
        robot_view = self.task.env.current_robot.robot_view
        kinematics = self.task.env.current_robot.kinematics
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        saved = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in robot_view.get_qpos_dict().items()
        }
        solution = kinematics.ik(
            gripper_mg_id,
            pose,
            robot_view.move_group_ids(),
            saved,
            base_pose=robot_view.base.pose,
        )
        if solution is None:
            return None
        model, data = self.task.env.current_model, self.task.env.current_data
        fixture_gid = int(model.geom(f"{fixture['body']}_g").id)
        robot_gids = []
        for gid in range(int(model.ngeom)):
            body = model.body(int(model.geom_bodyid[gid])).name or ""
            if (
                "gripper/left_" in body
                or "gripper/right_" in body
                or "gripper/base" in body
                or body.endswith("wrist_cam_body")
                or any(
                    token in body
                    for token in (
                        "fr3_link5", "fr3_link6", "fr3_link7",
                        "link5_skin", "link5_front_skin", "link5_back_skin",
                        "link6_skin", "link7_skin",
                    )
                )
            ):
                if int(model.geom_contype[gid]) != 0 or int(model.geom_conaffinity[gid]) != 0:
                    robot_gids.append(gid)
        try:
            robot_view.set_qpos_dict(solution)
            mujoco.mj_forward(model, data)
            return float(true_distance(model, data, robot_gids, [fixture_gid]))
        finally:
            robot_view.set_qpos_dict(saved)
            mujoco.mj_forward(model, data)

    def _v95_link_aware_fixture_segment(
        self, segment: TCPMoveSegment, fixture: dict[str, Any], *, prefix: str
    ) -> tuple[list[TCPMoveSegment], bool]:
        """Search one lateral degree of freedom using exact IK/link clearance."""
        center = np.asarray(fixture["center"], dtype=float)
        half = np.asarray(fixture["half"], dtype=float)
        start = segment.start_pose[:3, 3].copy()
        end = segment.end_pose[:3, 3].copy()
        dx = float(end[0] - start[0])
        if abs(dx) < 1e-9:
            return [segment], False
        t_cross = float((center[0] - start[0]) / dx)
        if not 0.02 < t_cross < 0.98:
            return [segment], False
        cross = start + t_cross * (end - start)
        support_side = 1.0 if str(fixture.get("support")) == "wall_left" else -1.0
        waypoint_side = -support_side
        aperture_width = float((getattr(self.task, "scene_params", {}) or {}).get("ap_w", 0.85))
        lateral_limit = aperture_width / 2.0 - self.MOUNTED_FIXTURE_ENVELOPE_HALF_Y - self.APERTURE_EDGE_RESERVE
        max_bow = max(0.0, waypoint_side * (waypoint_side * lateral_limit - cross[1]))
        candidate_bows = np.arange(
            self.V95_MIN_FIXTURE_BOW_M,
            max_bow + self.V95_BOW_SEARCH_STEP_M * 0.5,
            self.V95_BOW_SEARCH_STEP_M,
        )
        travel_side = 1.0 if dx > 0.0 else -1.0
        before_x = center[0] - travel_side * (half[0] + 0.08)
        after_x = center[0] + travel_side * (half[0] + 0.08)
        t_before = float(np.clip((before_x - start[0]) / dx, 0.04, 0.90))
        t_after = float(np.clip((after_x - start[0]) / dx, t_before + 0.02, 0.96))
        rotation = segment.end_pose[:3, :3]
        evaluated = 0
        for bow in candidate_bows:
            evaluated += 1
            waypoint_y = float(cross[1] + waypoint_side * bow)
            before = start + t_before * (end - start)
            after = start + t_after * (end - start)
            before[1] = waypoint_y
            after[1] = waypoint_y
            poses = (
                self._place_pose(before, rotation),
                self._place_pose(np.asarray([center[0], waypoint_y, cross[2]]), rotation),
                self._place_pose(after, rotation),
            )
            clearances = [self._v95_exact_fixture_clearance(pose, fixture) for pose in poses]
            if any(value is None for value in clearances):
                continue
            minimum = min(float(value) for value in clearances if value is not None)
            if minimum < self.V95_MIN_PLANNED_LINK_CLEARANCE_M:
                continue
            self._record_bow(
                prefix,
                planned_bow_m=float(bow),
                accepted_bow_m=float(bow),
                bow_fallback_taken=False,
                straight_clearance_m=None,
                required_clearance_m=self.V95_MIN_PLANNED_LINK_CLEARANCE_M,
                response_source="exact_ik_link5_link6_fixture_clearance_search",
            )
            self._pact_place_bow_diagnostics[prefix].update(
                {
                    "planned_min_link_clearance_m": minimum,
                    "link_clearance_candidate_count": evaluated,
                    "planning_basis": "exact_mujoco_geom_distance_after_ik",
                }
            )
            return (
                [
                    TCPMoveSegment(name=f"{prefix}_approach", start_pose=segment.start_pose, end_pose=poses[0], speed=self.policy_config.speed_fast),
                    TCPMoveSegment(name=f"{prefix}_pass", start_pose=poses[0], end_pose=poses[2], speed=self.PASS_SPEED),
                    TCPMoveSegment(name=f"{prefix}_exit", start_pose=poses[2], end_pose=segment.end_pose, speed=self.policy_config.speed_slow),
                ],
                True,
            )
        raise ValueError(
            f"V9.5 has no exact link-clear fixture bow for {prefix}; candidates={evaluated}"
        )

    def _sequence(
        self, segments: list[TCPMoveSegment], *, holding: bool
    ) -> TCPMoveSequence:
        robot_view = self.task.env.current_robot.robot_view
        return PactPlaceTCPMoveSequence(
            robot_view,
            self._tcp_to_jp_fn,
            self.policy_config.move_settle_time,
            is_holding_object=holding,
            gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
            tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
            tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
            move_segments=segments,
        )

    @staticmethod
    def _renamed(segment: TCPMoveSegment, name: str) -> TCPMoveSegment:
        return TCPMoveSegment(
            name=name,
            start_pose=segment.start_pose,
            end_pose=segment.end_pose,
            speed=segment.speed,
        )

    def _v99_empty_route_record(self) -> dict[str, Any]:
        return {
            "lane_y_m": None,
            "padding_m": None,
            "entry_x_m": None,
            "exit_x_m": None,
            "min_abs_detour_m": 0.0,
            "detour_meets_minimum": False,
            "nominal_clearance_m": None,
            "robust_clearance_m": None,
            "ik_ok": False,
            "ik_failures": 0,
            "fallback_taken": False,
            "clipped": False,
            "wrong_way": False,
            "parked": False,
        }

    def _v99_record_route(self, prefix: str, **fields: Any) -> None:
        record = self._v99_empty_route_record()
        record.update(fields)
        self._pact_place_v99_route[prefix] = record

    def _v99_collision_robot_geom_ids(self) -> list[int]:
        model = self.task.env.current_model
        ids: list[int] = []
        for geom_id in range(int(model.ngeom)):
            body = model.body(int(model.geom_bodyid[geom_id])).name or ""
            if not str(body).startswith("robot_0/"):
                continue
            if int(model.geom_contype[geom_id]) == 0 and int(
                model.geom_conaffinity[geom_id]
            ) == 0:
                continue
            ids.append(int(geom_id))
        return ids

    def _v99_target_geom_ids(self) -> list[int]:
        env = self.task.env
        model = env.current_model
        manager = env.object_managers[env.current_batch_index]
        pickup = manager.get_object_by_name(self.config.task_config.pickup_obj_name)
        if pickup is None:
            return []
        body_name = str(getattr(pickup, "name", "") or "")
        ids: list[int] = []
        for geom_id in range(int(model.ngeom)):
            body = model.body(int(model.geom_bodyid[geom_id])).name or ""
            if body_name and body_name not in str(body):
                continue
            if int(model.geom_contype[geom_id]) == 0 and int(
                model.geom_conaffinity[geom_id]
            ) == 0:
                continue
            if "Cup" not in str(body) and body_name not in str(body):
                continue
            ids.append(int(geom_id))
        return ids

    def _v99_sequential_ik_clearance(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        *,
        include_target: bool,
        fixture: dict[str, Any],
    ) -> dict[str, Any]:
        from pact_geom_distance import true_distance
        from pact_place_v99_pendant_contract import (
            MIN_NOMINAL_CLEARANCE_M,
            PENDANT_GEOM,
        )

        robot_view = self.task.env.current_robot.robot_view
        kinematics = self.task.env.current_robot.kinematics
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        saved = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in robot_view.get_qpos_dict().items()
        }
        model, data = self.task.env.current_model, self.task.env.current_data
        robot_ids = self._v99_collision_robot_geom_ids()
        target_ids = self._v99_target_geom_ids() if include_target else []
        probe_ids = robot_ids + target_ids
        try:
            pendant_gid = int(model.geom(PENDANT_GEOM).id)
        except (KeyError, ValueError, TypeError):
            return {"ik_ok": False, "ik_failures": len(positions), "nominal_clearance_m": None}
        seed = dict(saved)
        failures = 0
        clearances: list[float] = []
        try:
            for index in range(len(positions)):
                pose = self._place_pose(positions[index], rotations[index])
                solution = kinematics.ik(
                    gripper_mg_id,
                    pose,
                    robot_view.move_group_ids(),
                    seed,
                    base_pose=robot_view.base.pose,
                )
                if solution is None:
                    failures += 1
                    continue
                robot_view.set_qpos_dict(solution)
                mujoco.mj_forward(model, data)
                seed = {
                    key: np.asarray(value, dtype=float).copy()
                    for key, value in robot_view.get_qpos_dict().items()
                }
                if probe_ids:
                    clearances.append(
                        float(true_distance(model, data, probe_ids, [pendant_gid]))
                    )
        finally:
            robot_view.set_qpos_dict(saved)
            mujoco.mj_forward(model, data)
        nominal = min(clearances) if clearances else None
        return {
            "ik_ok": bool(failures == 0 and clearances),
            "ik_failures": int(failures),
            "nominal_clearance_m": None if nominal is None else float(nominal),
            "meets_nominal": bool(
                nominal is not None and float(nominal) + 1e-12 >= MIN_NOMINAL_CLEARANCE_M
            ),
        }

    def _v99_apply_lane(
        self,
        segments: list[TCPMoveSegment],
        *,
        prefix: str,
        include_target: bool,
    ) -> list[TCPMoveSegment]:
        if not self._v99_enabled() or not segments:
            return segments
        scene = getattr(self.task, "scene_params", {}) or {}
        if scene.get("pact_v99_pendant_parked"):
            self._v99_record_route(prefix, parked=True, ik_ok=True)
            return segments
        from pact_place_v99_pendant_contract import MIN_DETOUR_M
        from pact_place_v99_route import named_lane_segments, plan_lane

        fixture = dict(scene.get("pact_v99_pendant_fixture") or {})
        route = dict(scene.get("pact_v99_route") or {})
        layout = scene.get("pact_clutter_layout") or {}
        panel_side = str(
            scene.get("intrusion_side") or layout.get("intrusion_side") or ""
        )
        if prefix.startswith("inbound"):
            lane_y = route.get("inbound_lane_y_m")
            padding = route.get("inbound_padding_m", route.get("slab_padding_m"))
            freeze_start, freeze_final = False, True
        else:
            lane_y = route.get("outbound_lane_y_m")
            padding = route.get("outbound_padding_m", route.get("slab_padding_m"))
            freeze_start, freeze_final = True, False
        if lane_y is None or padding is None or not fixture:
            self._v99_record_route(prefix, fallback_taken=True)
            raise ValueError(f"V9.9 {prefix} route parameters are missing")
        positions = [segments[0].start_pose[:3, 3].copy()]
        rotations = [segments[0].start_pose[:3, :3].copy()]
        for segment in segments:
            positions.append(segment.end_pose[:3, 3].copy())
            rotations.append(segment.end_pose[:3, :3].copy())
        planned = plan_lane(
            np.asarray(positions, dtype=float),
            np.asarray(rotations, dtype=float),
            fixture=fixture,
            panel_side=panel_side,
            lane_y_m=float(lane_y),
            padding_m=float(padding),
            aperture_width_m=float(scene.get("ap_w", 0.85)),
            freeze_start=freeze_start,
            freeze_final=freeze_final,
        )
        detour = planned["detour"]
        if planned["clipped"] or planned["wrong_way"]:
            self._v99_record_route(
                prefix,
                lane_y_m=float(lane_y),
                padding_m=float(padding),
                clipped=bool(planned["clipped"]),
                wrong_way=bool(planned["wrong_way"]),
                fallback_taken=True,
                min_abs_detour_m=float(detour["min_abs_detour_m"]),
            )
            raise ValueError(f"V9.9 {prefix} lane is clipped or wrong-way")
        if not detour["meets_minimum"]:
            self._v99_record_route(
                prefix,
                lane_y_m=float(lane_y),
                padding_m=float(padding),
                entry_x_m=float(planned["entry_x_m"]),
                exit_x_m=float(planned["exit_x_m"]),
                min_abs_detour_m=float(detour["min_abs_detour_m"]),
                detour_meets_minimum=False,
                fallback_taken=True,
            )
            raise ValueError(
                f"V9.9 {prefix} detour {detour['min_abs_detour_m']:.4f} m "
                f"< {MIN_DETOUR_M} m"
            )
        clearance = self._v99_sequential_ik_clearance(
            planned["planned_positions_m"],
            planned["planned_rotations"],
            include_target=include_target,
            fixture=fixture,
        )
        if not clearance["ik_ok"] or not clearance["meets_nominal"]:
            self._v99_record_route(
                prefix,
                lane_y_m=float(lane_y),
                padding_m=float(padding),
                entry_x_m=float(planned["entry_x_m"]),
                exit_x_m=float(planned["exit_x_m"]),
                min_abs_detour_m=float(detour["min_abs_detour_m"]),
                detour_meets_minimum=True,
                nominal_clearance_m=clearance.get("nominal_clearance_m"),
                ik_ok=bool(clearance["ik_ok"]),
                ik_failures=int(clearance["ik_failures"]),
                fallback_taken=True,
            )
            raise ValueError(f"V9.9 {prefix} sequential IK or clearance failed")
        pieces = named_lane_segments(
            planned["planned_positions_m"],
            planned["planned_rotations"],
            prefix=prefix,
            entry_x=float(planned["entry_x_m"]),
            exit_x=float(planned["exit_x_m"]),
            stock_end=segments[-1].end_pose[:3, 3],
        )
        speed = float(segments[0].speed)
        rebuilt: list[TCPMoveSegment] = []
        for piece in pieces:
            poses = np.asarray(piece["positions_m"], dtype=float)
            rots = np.asarray(piece["rotations"], dtype=float)
            if len(poses) < 2:
                continue
            for index in range(1, len(poses)):
                rebuilt.append(
                    TCPMoveSegment(
                        name=str(piece["name"]),
                        start_pose=self._place_pose(poses[index - 1], rots[index - 1]),
                        end_pose=self._place_pose(poses[index], rots[index]),
                        speed=speed,
                    )
                )
        if not rebuilt:
            self._v99_record_route(prefix, fallback_taken=True)
            raise ValueError(f"V9.9 {prefix} produced no lane segments")
        rebuilt[-1].end_pose = segments[-1].end_pose.copy()
        rebuilt[0].start_pose = segments[0].start_pose.copy()
        self._v99_record_route(
            prefix,
            lane_y_m=float(lane_y),
            padding_m=float(padding),
            entry_x_m=float(planned["entry_x_m"]),
            exit_x_m=float(planned["exit_x_m"]),
            min_abs_detour_m=float(detour["min_abs_detour_m"]),
            detour_meets_minimum=True,
            nominal_clearance_m=clearance.get("nominal_clearance_m"),
            ik_ok=True,
            ik_failures=0,
            fallback_taken=False,
            clipped=False,
            wrong_way=False,
        )
        if prefix.startswith("inbound"):
            self.inbound_deflected = True
        else:
            self.outbound_deflected = True
        return rebuilt

    def _v10_record_route(self, prefix: str, **fields: Any) -> None:
        record = self._v99_empty_route_record()
        record.update(fields)
        self._pact_place_v10_route[prefix] = record

    def _v10_active_pendant_geom_ids(self) -> list[int]:
        from pact_place_v10_compound_pendant_contract import ALL_GEOMS

        model = self.task.env.current_model
        ids: list[int] = []
        for name in ALL_GEOMS:
            geom_id = int(model.geom(name).id)
            if int(model.geom_contype[geom_id]) == 0 and int(
                model.geom_conaffinity[geom_id]
            ) == 0:
                continue
            ids.append(geom_id)
        return ids

    def _v10_strict_environment_geom_ids(self) -> list[int]:
        model = self.task.env.current_model
        target_ids = set(self._v99_target_geom_ids())
        pendant_ids = set(self._v10_active_pendant_geom_ids())
        ids: list[int] = []
        for geom_id in range(int(model.ngeom)):
            if geom_id in target_ids or geom_id in pendant_ids:
                continue
            body = str(model.body(int(model.geom_bodyid[geom_id])).name or "")
            if body.startswith("robot_0/"):
                continue
            if int(model.geom_contype[geom_id]) == 0 and int(
                model.geom_conaffinity[geom_id]
            ) == 0:
                continue
            ids.append(int(geom_id))
        return ids

    def _v10_sequential_ik_clearance(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        *,
        include_target: bool,
        min_pendant_m: float | None = None,
        pendant_geom_ids: list[int] | None = None,
        environment_geom_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        from pact_geom_distance import true_distance
        from pact_place_v10_compound_pendant_contract import MIN_NOMINAL_CLEARANCE_M
        from pact_place_v10_route import sequential_ik_split_clearance

        robot_view = self.task.env.current_robot.robot_view
        kinematics = self.task.env.current_robot.kinematics
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        saved = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in robot_view.get_qpos_dict().items()
        }
        model, data = self.task.env.current_model, self.task.env.current_data
        robot_ids = list(self._v99_collision_robot_geom_ids())
        target_ids = list(self._v99_target_geom_ids()) if include_target else []
        probe_ids = robot_ids + target_ids
        pendant_ids = (
            list(pendant_geom_ids)
            if pendant_geom_ids is not None
            else list(self._v10_active_pendant_geom_ids())
        )
        env_ids = (
            list(environment_geom_ids)
            if environment_geom_ids is not None
            else list(self._v10_strict_environment_geom_ids())
        )
        threshold = (
            float(MIN_NOMINAL_CLEARANCE_M)
            if min_pendant_m is None
            else float(min_pendant_m)
        )

        def solve_ik(pose, seed):
            return kinematics.ik(
                gripper_mg_id,
                pose,
                robot_view.move_group_ids(),
                seed,
                base_pose=robot_view.base.pose,
            )

        def measure_pendant():
            if not probe_ids or not pendant_ids:
                return None
            return true_distance(model, data, probe_ids, pendant_ids)

        def measure_environment():
            if not probe_ids or not env_ids:
                return None
            return true_distance(model, data, probe_ids, env_ids)

        report = sequential_ik_split_clearance(
            positions,
            rotations,
            saved_qpos=saved,
            set_qpos=robot_view.set_qpos_dict,
            get_qpos=robot_view.get_qpos_dict,
            solve_ik=solve_ik,
            forward=lambda: mujoco.mj_forward(model, data),
            place_pose=self._place_pose,
            measure_pendant=measure_pendant,
            measure_environment=measure_environment,
            min_pendant_m=threshold,
        )
        report["probe_ids"] = list(probe_ids)
        report["pendant_geom_ids"] = list(pendant_ids)
        report["environment_geom_ids"] = list(env_ids)
        report["include_target"] = bool(include_target)
        report["nominal_clearance_m"] = report.get("pendant_clearance_m")
        report["meets_nominal"] = bool(report.get("meets_pendant"))
        return report

    def _v10_evaluate_nominal_and_robust(
        self,
        planned: dict[str, Any],
        *,
        include_target: bool,
        fixture: dict[str, Any],
        freeze_start: bool,
        freeze_final: bool,
        aperture_width_m: float,
        panel_side: str,
        padding_m: float,
        endpoint_only: bool = False,
    ) -> dict[str, Any]:
        from pact_place_v10_compound_pendant_contract import (
            MIN_NOMINAL_CLEARANCE_M,
            MIN_ROBUST_CLEARANCE_M,
        )
        from pact_place_v10_route import (
            evaluate_all_perturbation_corners,
            evaluate_environment_no_intersection,
            evaluate_pendant_nominal_and_robust,
            plan_lane_at_parameters,
            plan_lane_at_parameters_endpoint_only,
        )

        stock_p = planned.get("stock_positions_m")
        stock_r = planned.get("stock_rotations")
        if stock_p is None or stock_r is None:
            raise ValueError("planned lane is missing densified stock TCP")
        planner = (
            plan_lane_at_parameters_endpoint_only
            if endpoint_only
            else plan_lane_at_parameters
        )
        nominal = self._v10_sequential_ik_clearance(
            planned["planned_positions_m"],
            planned["planned_rotations"],
            include_target=include_target,
            min_pendant_m=MIN_NOMINAL_CLEARANCE_M,
        )

        def evaluate_corner(corner: dict[str, Any]) -> dict[str, Any]:
            corner_plan = planner(
                stock_p,
                stock_r,
                fixture=fixture,
                panel_side=panel_side,
                lane_y_m=float(corner["lane_y_m"]),
                padding_m=float(padding_m),
                entry_x_m=float(corner["entry_x_m"]),
                exit_x_m=float(corner["exit_x_m"]),
                aperture_width_m=float(aperture_width_m),
                freeze_start=freeze_start,
                freeze_final=freeze_final,
            )
            report = self._v10_sequential_ik_clearance(
                corner_plan["planned_positions_m"],
                corner_plan["planned_rotations"],
                include_target=include_target,
                min_pendant_m=MIN_ROBUST_CLEARANCE_M,
            )
            report["planned_positions_m"] = corner_plan["planned_positions_m"]
            report["lane_y_m"] = float(corner_plan["lane_y_m"])
            report["entry_x_m"] = float(corner_plan["entry_x_m"])
            report["exit_x_m"] = float(corner_plan["exit_x_m"])
            report["clipped"] = bool(corner_plan["clipped"])
            report["wrong_way"] = bool(corner_plan["wrong_way"])
            return report

        corners = evaluate_all_perturbation_corners(planned, evaluate_corner)
        pendant = evaluate_pendant_nominal_and_robust(
            min_nominal_m=MIN_NOMINAL_CLEARANCE_M,
            min_robust_m=MIN_ROBUST_CLEARANCE_M,
            nominal_clearance_m=nominal.get("pendant_clearance_m"),
            corner_clearances_m=[item.get("pendant_clearance_m") for item in corners],
        )
        env_distances = [nominal.get("environment_clearance_m")] + [
            item.get("environment_clearance_m") for item in corners
        ]
        environment = evaluate_environment_no_intersection(env_distances)
        robust_ik_ok = all(bool(item.get("ik_ok")) for item in corners)
        accepted = bool(
            nominal.get("ik_ok")
            and robust_ik_ok
            and pendant["meets_nominal"]
            and pendant["meets_robust"]
            and environment["environment_clear"]
            and not any(item.get("clipped") or item.get("wrong_way") for item in corners)
        )
        return {
            "nominal": nominal,
            "corners": corners,
            "pendant": pendant,
            "environment": environment,
            "n_corners_evaluated": 8,
            "all_corners_evaluated": True,
            "robust_ik_ok": bool(robust_ik_ok),
            "accepted": accepted,
            "min_robust_clearance_m": (
                None
                if any(item.get("pendant_clearance_m") is None for item in corners)
                else float(min(float(item["pendant_clearance_m"]) for item in corners))
            ),
        }

    def _v102_active_component_geoms(self) -> list[tuple[str, int]]:
        """(component name, geom id) for every collision-enabled V10.2 geom."""
        scene = getattr(self.task, "scene_params", {}) or {}
        assembly = dict(scene.get("pact_v10_pendant_assembly") or {})
        model = self.task.env.current_model
        pairs: list[tuple[str, int]] = []
        for item in assembly.get("components") or []:
            if not item.get("active"):
                continue
            geom_id = int(model.geom(str(item["geom"])).id)
            if int(model.geom_contype[geom_id]) == 0 and int(
                model.geom_conaffinity[geom_id]
            ) == 0:
                continue
            pairs.append((str(item["name"]), geom_id))
        return pairs

    def _v102_probe_geom_ids(self, *, include_target: bool) -> list[int]:
        """Cached robot (and optionally target) collision geoms.

        Geom membership is fixed for an episode; rescanning ``model.ngeom`` on
        every policy frame would dominate the telemetry cost.
        """
        cache = self._pact_v102_probe_ids
        key = bool(include_target)
        if key not in cache:
            ids = list(self._v99_collision_robot_geom_ids())
            if key:
                ids = ids + list(self._v99_target_geom_ids())
            cache[key] = ids
        return list(cache[key])

    def _v102_component_clearances(
        self, *, include_target: bool
    ) -> dict[str, float | None]:
        from pact_geom_distance import true_distance

        model, data = self.task.env.current_model, self.task.env.current_data
        probe_ids = self._v102_probe_geom_ids(include_target=include_target)
        out: dict[str, float | None] = {}
        for name, geom_id in self._v102_cached_component_geoms():
            value = float(true_distance(model, data, probe_ids, [geom_id]))
            out[name] = None if not np.isfinite(value) else value
        return out

    def _v102_route_sequential_ik(
        self, planned: dict[str, Any], *, include_target: bool
    ) -> dict[str, Any]:
        """Full-waypoint sequential IK with per-component pendant clearance.

        Deliberately does not call the flawed scalar robot-versus-all-environment
        preclearance. It never aborts early, so an environment abort after one
        waypoint can never be reported as an IK pass.
        """
        import mujoco

        from pact_place_v102_route import sequential_ik_component_clearance

        robot_view = self.task.env.current_robot.robot_view
        kinematics = self.task.env.current_robot.kinematics
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        model, data = self.task.env.current_model, self.task.env.current_data
        saved = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in robot_view.get_qpos_dict().items()
        }
        component_names = [name for name, _gid in self._v102_active_component_geoms()]

        def solve_ik(pose, seed):
            return kinematics.ik(
                gripper_mg_id,
                pose,
                robot_view.move_group_ids(),
                seed,
                base_pose=robot_view.base.pose,
            )

        return sequential_ik_component_clearance(
            planned["planned_positions_m"],
            planned["planned_rotations"],
            saved_qpos=saved,
            set_qpos=robot_view.set_qpos_dict,
            get_qpos=robot_view.get_qpos_dict,
            solve_ik=solve_ik,
            forward=lambda: mujoco.mj_forward(model, data),
            place_pose=self._place_pose,
            component_names=component_names,
            measure_components=lambda: self._v102_component_clearances(
                include_target=include_target
            ),
        )

    def _v10_apply_lane(
        self,
        segments: list[TCPMoveSegment],
        *,
        prefix: str,
        include_target: bool,
    ) -> list[TCPMoveSegment]:
        if not self._v10_enabled() or not segments:
            return segments
        scene = getattr(self.task, "scene_params", {}) or {}
        if scene.get("pact_v10_pendant_parked"):
            self._v10_record_route(prefix, parked=True, ik_ok=True)
            return segments
        from pact_place_v10_compound_pendant_contract import MIN_DETOUR_M
        from pact_place_v10_route import (
            frozen_endpoint_preserved,
            named_lane_segments,
            plan_lane,
            plan_lane_endpoint_only,
            resolve_v10_runtime_route,
        )
        from pact_place_v102_route import resolve_v102_runtime_route

        assembly = dict(scene.get("pact_v10_pendant_assembly") or {})
        route = dict(scene.get("pact_v10_route") or {})
        # V10.2 dispatch needs the exact contract marker and speed-schedule
        # hash. Without them this returns None and V10/V10.1 rows keep the
        # frozen historical dispatch unchanged.
        v102 = resolve_v102_runtime_route(scene, route)
        dispatch = v102 if v102 is not None else resolve_v10_runtime_route(route)
        layout = scene.get("pact_clutter_layout") or {}
        panel_side = str(
            scene.get("intrusion_side") or layout.get("intrusion_side") or ""
        )
        if prefix.startswith("inbound"):
            lane_y = route.get("inbound_lane_y_m")
            padding = route.get("inbound_padding_m", route.get("slab_padding_m"))
            freeze_start, freeze_final = False, True
        else:
            lane_y = route.get("outbound_lane_y_m")
            padding = route.get("outbound_padding_m", route.get("slab_padding_m"))
            freeze_start, freeze_final = True, False
        telemetry = {
            "rewrite_primitive": dispatch["rewrite_primitive"],
            "qualification_mode": dispatch["qualification_mode"],
            "offline_strict_environment_preclearance_used": (
                not dispatch["skip_offline_strict_environment"]
            ),
            "strict_environment_preclearance_intentionally_not_used": bool(
                dispatch["skip_offline_strict_environment"]
            ),
        }
        if v102 is not None:
            telemetry["speed_schedule_sha256"] = str(v102["speed_schedule_sha256"])
            telemetry["speed_schedule"] = dict(v102["speed_schedule"])
        if lane_y is None or padding is None or not assembly:
            self._v10_record_route(prefix, fallback_taken=True, **telemetry)
            raise ValueError(f"V10 {prefix} route parameters are missing")
        telemetry.update(lane_y_m=float(lane_y), padding_m=float(padding))
        positions = [segments[0].start_pose[:3, 3].copy()]
        rotations = [segments[0].start_pose[:3, :3].copy()]
        for segment in segments:
            positions.append(segment.end_pose[:3, 3].copy())
            rotations.append(segment.end_pose[:3, :3].copy())
        planner = (
            plan_lane_endpoint_only if dispatch["use_endpoint_only"] else plan_lane
        )
        planned = planner(
            np.asarray(positions, dtype=float),
            np.asarray(rotations, dtype=float),
            assembly=assembly,
            panel_side=panel_side,
            lane_y_m=float(lane_y),
            padding_m=float(padding),
            aperture_width_m=float(scene.get("ap_w", 0.85)),
            freeze_start=freeze_start,
            freeze_final=freeze_final,
        )
        detour = planned["detour"]
        endpoints = dict(planned.get("frozen_endpoints") or {})
        if not endpoints:
            endpoints = frozen_endpoint_preserved(
                planned["planned_positions_m"],
                planned["planned_rotations"],
                planned.get("stock_positions_m", np.asarray(positions, dtype=float)),
                planned.get("stock_rotations", np.asarray(rotations, dtype=float)),
                freeze_start=freeze_start,
                freeze_final=freeze_final,
            )
        telemetry.update(
            entry_x_m=float(planned["entry_x_m"]),
            exit_x_m=float(planned["exit_x_m"]),
            min_abs_detour_m=float(detour["min_abs_detour_m"]),
            detour_meets_minimum=bool(detour["meets_minimum"]),
            clipped=bool(planned["clipped"]),
            wrong_way=bool(planned["wrong_way"]),
            frozen_endpoint_preserved=bool(endpoints.get("preserved")),
            start_preserved=bool(endpoints.get("start_preserved")),
            final_preserved=bool(endpoints.get("final_preserved")),
            densify_max_translation_m=float(
                (planned.get("path_steps") or {}).get("max_translation_m") or 0.0
            ),
            densify_max_rotation_deg=float(
                (planned.get("path_steps") or {}).get("max_rotation_deg") or 0.0
            ),
        )
        if planned["clipped"] or planned["wrong_way"]:
            self._v10_record_route(prefix, fallback_taken=True, **telemetry)
            raise ValueError(f"V10 {prefix} lane is clipped or wrong-way")
        if not detour["meets_minimum"]:
            self._v10_record_route(prefix, fallback_taken=True, **telemetry)
            raise ValueError(
                f"V10 {prefix} detour {detour['min_abs_detour_m']:.4f} m "
                f"< {MIN_DETOUR_M} m"
            )
        if dispatch["use_endpoint_only"] and not endpoints.get("preserved"):
            self._v10_record_route(prefix, fallback_taken=True, **telemetry)
            raise ValueError(f"V10 {prefix} frozen endpoints were mutated")
        if dispatch["use_endpoint_only"] and not planned.get("continuous_after_densify"):
            self._v10_record_route(prefix, fallback_taken=True, **telemetry)
            raise ValueError(f"V10 {prefix} densified path exceeds step limits")
        clearance = None
        nominal: dict[str, Any] = {}
        if not dispatch["skip_offline_strict_environment"]:
            clearance = self._v10_evaluate_nominal_and_robust(
                planned,
                include_target=include_target,
                fixture=planned["union_fixture"],
                freeze_start=freeze_start,
                freeze_final=freeze_final,
                aperture_width_m=float(scene.get("ap_w", 0.85)),
                panel_side=panel_side,
                padding_m=float(padding),
                endpoint_only=bool(dispatch["use_endpoint_only"]),
            )
            nominal = clearance["nominal"]
            telemetry.update(
                nominal_clearance_m=nominal.get("pendant_clearance_m"),
                robust_clearance_m=clearance.get("min_robust_clearance_m"),
                environment_clear=bool(clearance["environment"]["environment_clear"]),
                n_corners_evaluated=int(clearance["n_corners_evaluated"]),
                all_corners_evaluated=True,
                ik_ok=bool(nominal.get("ik_ok") and clearance.get("robust_ik_ok")),
                ik_failures=int(nominal.get("ik_failures") or 0),
            )
            if not clearance["accepted"]:
                self._v10_record_route(prefix, fallback_taken=True, **telemetry)
                raise ValueError(
                    f"V10 {prefix} sequential IK or split clearance failed"
                )
        else:
            telemetry.update(
                n_corners_evaluated=0,
                all_corners_evaluated=False,
                environment_clear=None,
                nominal_clearance_m=None,
                robust_clearance_m=None,
            )
            if v102 is not None:
                ik_report = self._v102_route_sequential_ik(
                    planned, include_target=include_target
                )
                telemetry.update(
                    waypoints_attempted=int(ik_report["waypoints_attempted"]),
                    waypoints_solved=int(ik_report["waypoints_solved"]),
                    complete_sequential_ik=bool(ik_report["complete_sequential_ik"]),
                    ik_failures=len(ik_report["ik_failure_indices"]),
                    route_qpos_restored=bool(ik_report["qpos_restored"]),
                    nominal_clearance_m=ik_report["min_clearance_m"],
                    per_component_min_clearance_m=dict(
                        ik_report["per_component_min_clearance_m"]
                    ),
                )
        pieces = named_lane_segments(
            planned["planned_positions_m"],
            planned["planned_rotations"],
            prefix=prefix,
            entry_x=float(planned["entry_x_m"]),
            exit_x=float(planned["exit_x_m"]),
            stock_end=segments[-1].end_pose[:3, 3],
        )
        # Historical rows keep the inherited behaviour: every rebuilt piece
        # carries segments[0].speed. V10.2 assigns a speed per named piece so
        # the fast empty-arm approach is not copied onto the pendant pass.
        inherited_speed = float(segments[0].speed)
        piece_speeds: list[dict[str, Any]] = []
        rebuilt: list[TCPMoveSegment] = []
        for piece in pieces:
            poses = np.asarray(piece["positions_m"], dtype=float)
            rots = np.asarray(piece["rotations"], dtype=float)
            if len(poses) < 2:
                continue
            piece_name = str(piece["name"])
            if v102 is not None:
                from pact_place_v102_route import (
                    classify_route_piece,
                    route_piece_speed,
                    speed_cap_violation,
                )

                speed = float(
                    route_piece_speed(piece_name, inherited_speed_m_s=inherited_speed)
                )
                violation = speed_cap_violation(piece_name, speed)
                if violation is not None:
                    telemetry["piece_speeds"] = piece_speeds
                    self._v10_record_route(prefix, fallback_taken=True, **telemetry)
                    raise ValueError(f"V10.2 {piece_name} {violation}: {speed:.4f} m/s")
                speed_class = classify_route_piece(piece_name)
            else:
                speed = inherited_speed
                speed_class = "inherited"
            piece_speeds.append(
                {
                    "name": piece_name,
                    "speed_class": speed_class,
                    "requested_speed_m_s": float(speed),
                    "inherited_speed_m_s": inherited_speed,
                    "n_segments": int(len(poses) - 1),
                }
            )
            for index in range(1, len(poses)):
                rebuilt.append(
                    TCPMoveSegment(
                        name=piece_name,
                        start_pose=self._place_pose(poses[index - 1], rots[index - 1]),
                        end_pose=self._place_pose(poses[index], rots[index]),
                        speed=float(speed),
                    )
                )
        telemetry["piece_speeds"] = piece_speeds
        if not rebuilt:
            self._v10_record_route(prefix, fallback_taken=True, **telemetry)
            raise ValueError(f"V10 {prefix} produced no lane segments")
        rebuilt[-1].end_pose = segments[-1].end_pose.copy()
        rebuilt[0].start_pose = segments[0].start_pose.copy()
        start_ok = bool(
            np.allclose(
                rebuilt[0].start_pose[:3, 3],
                segments[0].start_pose[:3, 3],
                atol=1e-9,
            )
        )
        final_ok = bool(
            np.allclose(
                rebuilt[-1].end_pose[:3, 3],
                segments[-1].end_pose[:3, 3],
                atol=1e-9,
            )
        )
        if dispatch["use_endpoint_only"] and not (start_ok and final_ok):
            telemetry["frozen_endpoint_preserved"] = False
            self._v10_record_route(prefix, fallback_taken=True, **telemetry)
            raise ValueError(f"V10 {prefix} rebuilt path changed endpoints")
        telemetry.update(
            fallback_taken=False,
            clipped=False,
            wrong_way=False,
        )
        if v102 is None:
            telemetry.update(ik_ok=True, ik_failures=0)
        else:
            telemetry["ik_ok"] = bool(telemetry.get("complete_sequential_ik"))
        self._v10_record_route(prefix, **telemetry)
        if prefix.startswith("inbound"):
            self.inbound_deflected = True
        else:
            self.outbound_deflected = True
        return rebuilt

    @staticmethod
    def _interpolate_pose(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        lin_vel, ang_vel = transform_to_twist(np.linalg.inv(start) @ end)
        return start @ twist_to_transform(lin_vel * float(t), ang_vel * float(t))

    @classmethod
    def _subdivide_tcp_segment(
        cls, segment: TCPMoveSegment, max_step_m: float
    ) -> list[TCPMoveSegment]:
        dist = float(
            np.linalg.norm(segment.end_pose[:3, 3] - segment.start_pose[:3, 3])
        )
        n_pieces = max(1, int(np.ceil(dist / max_step_m - 1e-12)))
        if n_pieces == 1:
            return [segment]
        pieces: list[TCPMoveSegment] = []
        previous = segment.start_pose.copy()
        for index in range(1, n_pieces + 1):
            pose = cls._interpolate_pose(
                segment.start_pose, segment.end_pose, index / n_pieces
            )
            pieces.append(
                TCPMoveSegment(
                    name=segment.name,
                    start_pose=previous,
                    end_pose=pose,
                    speed=segment.speed,
                )
            )
            previous = pose
        return pieces

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        # Scripted two-phase fallback: keep the validated corridor pick path at
        # the primitive level, then plan placement from its lift pose. The
        # generic pick-and-place approach lost Cup_10 before a 1 cm lift in the
        # development screen.
        pick_helper = PactCollisionCorridorPolicy(self.config, self.task)
        pick_primitives = pick_helper._compute_trajectory()
        inbound = pick_primitives[1]
        inbound_pre = inbound._move_segments[-2]
        stock_inbound_grasp = inbound._move_segments[-1]
        inbound_prefix = list(inbound._move_segments[:-2])
        if self._v9_enabled():
            inbound_hazard_role = (
                "inbound_vessel"
                if str(
                    (getattr(self.task, "scene_params", {}) or {}).get(
                        "pact_place_environment_version", ""
                    )
                )
                in {
                    "pact_place_corridor_v9_3",
                    "pact_place_corridor_v9_4_mounted_preview",
                    "pact_place_corridor_v9_5_low_wall",
                    "pact_place_corridor_v9_8_pendant",
                    "pact_place_corridor_v9_9_pendant",
                    PACT_PLACE_V10_ENVIRONMENT_VERSION,
                    PACT_PLACE_V102_ENVIRONMENT_VERSION,
                    PACT_PLACE_V105_ENVIRONMENT_VERSION,
                    *PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS,
                }
                else "outbound_vessel"
            )
            route_blocker = next(
                (
                    item
                    for item in self._hazard_list()
                    if item.get("role") == inbound_hazard_role
                ),
                None,
            )
            if route_blocker is not None:
                candidates = (
                    list(inbound._move_segments[:-1])
                    if inbound_hazard_role == "inbound_vessel"
                    else [inbound_pre]
                )
                inbound_bow = []
                inbound_vessel_bowed = False
                for candidate in candidates:
                    pieces, bowed = self._bow_segment(
                        candidate,
                        prefix="inbound_vessel",
                        envelope_half_y=self.INBOUND_ENVELOPE_HALF_Y,
                        safe_gap=self.INBOUND_SAFE_GAP,
                        center=np.asarray(route_blocker["center"], dtype=float),
                        half=np.asarray(route_blocker["half"], dtype=float),
                        preferred_waypoint_side=(
                            self._preferred_v9_waypoint_side()
                        ),
                    )
                    inbound_bow.extend(pieces)
                    inbound_vessel_bowed |= bowed
                if inbound_hazard_role == "inbound_vessel":
                    inbound_prefix = inbound_bow
                    cross_hazard = next(
                        (
                            item
                            for item in self._hazard_list()
                            if item.get("role") == "outbound_vessel"
                        ),
                        None,
                    )
                    if cross_hazard is not None:
                        cross_segments: list[TCPMoveSegment] = []
                        for candidate in inbound_prefix:
                            pieces, bowed = self._bow_segment(
                                candidate,
                                prefix="inbound_cross_vessel",
                                envelope_half_y=self.INBOUND_ENVELOPE_HALF_Y,
                                safe_gap=self.INBOUND_SAFE_GAP,
                                center=np.asarray(cross_hazard["center"], dtype=float),
                                half=np.asarray(cross_hazard["half"], dtype=float),
                                preferred_waypoint_side=self._preferred_v9_waypoint_side(),
                            )
                            cross_segments.extend(pieces)
                            inbound_vessel_bowed |= bowed
                        inbound_prefix = cross_segments
                else:
                    inbound_prefix.extend(inbound_bow)
                self.inbound_deflected = bool(inbound_vessel_bowed)
            else:
                inbound_prefix.append(inbound_pre)
        else:
            inbound_prefix.append(inbound_pre)
        scene = getattr(self.task, "scene_params", {}) or {}
        environment_version = str(scene.get("pact_place_environment_version", ""))
        fixture_roles = self._mounted_fixture_roles(
            environment_version,
            pendant_lateral_bow=bool(scene.get("pact_v98_pendant_lateral_bow")),
        )
        if fixture_roles:
            for fixture_role in fixture_roles:
                fixture = next(
                    (
                        item
                        for item in self._hazard_list()
                        if item.get("role") == fixture_role
                    ),
                    None,
                )
                if fixture is None:
                    continue
                fixture_segments: list[TCPMoveSegment] = []
                fixture_bowed = False
                prefix = f"inbound_{fixture_role}"
                for candidate_segment in inbound_prefix:
                    if environment_version == "pact_place_corridor_v9_5_low_wall":
                        pieces, bowed = self._v95_link_aware_fixture_segment(
                            candidate_segment, fixture, prefix=prefix
                        )
                    else:
                        route_side = self._route_side_at_x(
                            candidate_segment, float(fixture["center"][0])
                        )
                        pieces, bowed = self._bow_segment(
                            candidate_segment,
                            prefix=prefix,
                            envelope_half_y=self._mounted_fixture_bow_envelope_half_y(
                                fixture_role, route_side
                            ),
                            safe_gap=self.MOUNTED_FIXTURE_SAFE_GAP,
                            center=np.asarray(fixture["center"], dtype=float),
                            half=np.asarray(fixture["half"], dtype=float),
                            preferred_waypoint_side=(
                                route_side if fixture_role == "ceiling_fixture" else None
                            ),
                        )
                    fixture_segments.extend(pieces)
                    fixture_bowed |= bowed
                inbound_prefix = fixture_segments
                self.inbound_deflected |= fixture_bowed
        inbound_prefix = self._v99_apply_lane(
            inbound_prefix, prefix="inbound_pendant", include_target=False
        )
        inbound_prefix = self._v10_apply_lane(
            inbound_prefix, prefix="inbound_pendant", include_target=False
        )
        adjusted_grasp_pose = stock_inbound_grasp.end_pose.copy()
        adjusted_grasp_pose[2, 3] += self.GRASP_WORLD_Z_OFFSET_M
        if not self.check_feasible_ik(adjusted_grasp_pose):
            raise ValueError("IK failed for vertically adjusted grasp pose")
        inbound_grasp = TCPMoveSegment(
            name=stock_inbound_grasp.name,
            start_pose=stock_inbound_grasp.start_pose,
            end_pose=adjusted_grasp_pose,
            speed=stock_inbound_grasp.speed,
        )
        robot_view = self.task.env.current_robot.robot_view
        pick_primitives[1] = TCPMoveSequence(
            robot_view,
            self._tcp_to_jp_fn,
            self.policy_config.move_settle_time,
            gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
            tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
            tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
            move_segments=inbound_prefix + [inbound_grasp],
        )
        stock_lift = pick_primitives[3]._move_segments[-1]
        adjusted_lift_pose = stock_lift.end_pose.copy()
        adjusted_lift_pose[2, 3] += self.GRASP_WORLD_Z_OFFSET_M
        if not self.check_feasible_ik(adjusted_lift_pose):
            raise ValueError("IK failed for vertically adjusted lift pose")
        lift = TCPMoveSegment(
            name=stock_lift.name,
            start_pose=adjusted_grasp_pose,
            end_pose=adjusted_lift_pose,
            speed=stock_lift.speed,
        )
        pick_primitives[3] = self._sequence([lift], holding=True)
        self.inbound_deflected = bool(
            self.inbound_deflected or pick_helper.behavior_class == "deflect"
        )

        manager = self.task.env.object_managers[self.task.env.current_batch_index]
        task_config = self.config.task_config
        pickup = manager.get_object_by_name(task_config.pickup_obj_name)
        pickup_pose = pos_quat_to_pose_mat(pickup.position, pickup.quat)
        self._pact_place_grasp_diagnostics = {
            "stock_grasp_world_position_m": list(
                map(float, stock_inbound_grasp.end_pose[:3, 3])
            ),
            "adjusted_grasp_world_position_m": list(
                map(float, adjusted_grasp_pose[:3, 3])
            ),
            "stock_grasp_object_local_position_m": list(
                map(
                    float,
                    (np.linalg.inv(pickup_pose) @ stock_inbound_grasp.end_pose)[:3, 3],
                )
            ),
            "adjusted_grasp_object_local_position_m": list(
                map(float, (np.linalg.inv(pickup_pose) @ adjusted_grasp_pose)[:3, 3])
            ),
            "world_z_offset_m": float(self.GRASP_WORLD_Z_OFFSET_M),
        }
        receptacle = manager.get_object_by_name(task_config.place_receptacle_name)
        preplace_pose, place_pose, postplace_pose = self._get_placement_poses(
            inbound_grasp.end_pose,
            pickup,
            receptacle,
        )
        carry_pose = lift.end_pose.copy()
        carry_pose[2, 3] += self.OUTBOUND_CARRY_RAISE_M
        if not self.check_feasible_ik(carry_pose):
            raise ValueError("IK failed for outbound carry-raise pose")
        carry_raise = []
        if self.OUTBOUND_CARRY_RAISE_M > 0.0:
            carry_raise.append(
                TCPMoveSegment(
                    name="outbound_lift",
                    start_pose=lift.end_pose,
                    end_pose=carry_pose,
                    speed=self.policy_config.speed_slow,
                )
            )
        outside_staging_pose = carry_pose.copy()
        environment_version = str(
            (getattr(self.task, "scene_params", {}) or {}).get(
                "pact_place_environment_version", ""
            )
        )
        outside_staging_pose[0, 3] = (
            self.V93_OUTSIDE_STAGING_X_M
            if environment_version
            in {
                "pact_place_corridor_v9_3",
                "pact_place_corridor_v9_4_mounted_preview",
                "pact_place_corridor_v9_5_low_wall",
                "pact_place_corridor_v9_8_pendant",
                "pact_place_corridor_v9_9_pendant",
                PACT_PLACE_V10_ENVIRONMENT_VERSION,
                PACT_PLACE_V102_ENVIRONMENT_VERSION,
            }
            else self.V9_OUTSIDE_STAGING_X_M
            if self._v9_enabled()
            else self.OUTSIDE_STAGING_X_M
        )
        outside_staging_pose[1, 3] = preplace_pose[1, 3]
        if not self.check_feasible_ik(outside_staging_pose):
            raise ValueError("IK failed for outside staging pose")
        corridor_transfer = TCPMoveSegment(
            name="outbound_staging",
            start_pose=carry_pose,
            end_pose=outside_staging_pose,
            speed=self.policy_config.speed_fast,
        )
        preplace_transition = TCPMoveSegment(
            name="preplace",
            start_pose=outside_staging_pose,
            end_pose=preplace_pose,
            speed=self.policy_config.speed_fast,
        )
        place = TCPMoveSegment(
            name="placement_descent",
            start_pose=preplace_pose,
            end_pose=place_pose,
            speed=self.policy_config.speed_slow,
        )
        self._pact_place_canonical_target_poses = {
            "pregrasp": inbound_pre.end_pose,
            "grasp": inbound_grasp.end_pose,
            "lift": lift.end_pose,
            "carry": carry_pose,
            "outside_staging": outside_staging_pose,
            "preplace": preplace_pose,
            "place": place_pose,
            "postplace": postplace_pose,
        }
        outbound_segments, panel_bowed = self._bow_segment(
            corridor_transfer,
            prefix="outbound",
            envelope_half_y=self.OUTBOUND_ENVELOPE_HALF_Y,
            safe_gap=self.OUTBOUND_SAFE_GAP,
        )
        vessel_bowed = False
        if self._v9_enabled():
            outbound_hazard = next(
                (
                    item
                    for item in self._hazard_list()
                    if item.get("role") == "outbound_vessel"
                ),
                None,
            )
            if outbound_hazard is not None:
                vessel_segments: list[TCPMoveSegment] = []
                for candidate_segment in outbound_segments:
                    pieces, bowed = self._bow_segment(
                        candidate_segment,
                        prefix="outbound_vessel",
                        envelope_half_y=self.OUTBOUND_ENVELOPE_HALF_Y,
                        safe_gap=self.V9_VESSEL_SAFE_GAP,
                        center=np.asarray(outbound_hazard["center"], dtype=float),
                        half=np.asarray(outbound_hazard["half"], dtype=float),
                        preferred_waypoint_side=self._preferred_v9_waypoint_side(),
                    )
                    vessel_segments.extend(pieces)
                    vessel_bowed |= bowed
                outbound_segments = vessel_segments
            if str(
                (getattr(self.task, "scene_params", {}) or {}).get(
                    "pact_place_environment_version", ""
                )
            ) in {
                "pact_place_corridor_v9_3",
                "pact_place_corridor_v9_4_mounted_preview",
                "pact_place_corridor_v9_5_low_wall",
                "pact_place_corridor_v9_8_pendant",
                "pact_place_corridor_v9_9_pendant",
                PACT_PLACE_V10_ENVIRONMENT_VERSION,
                PACT_PLACE_V102_ENVIRONMENT_VERSION,
            }:
                inbound_hazard = next(
                    (
                        item
                        for item in self._hazard_list()
                        if item.get("role") == "inbound_vessel"
                    ),
                    None,
                )
                if inbound_hazard is not None:
                    cross_vessel_segments: list[TCPMoveSegment] = []
                    for candidate_segment in outbound_segments:
                        pieces, bowed = self._bow_segment(
                            candidate_segment,
                            prefix="outbound_cross_vessel",
                            envelope_half_y=self.OUTBOUND_ENVELOPE_HALF_Y,
                            safe_gap=self.V9_VESSEL_SAFE_GAP,
                            center=np.asarray(inbound_hazard["center"], dtype=float),
                            half=np.asarray(inbound_hazard["half"], dtype=float),
                            preferred_waypoint_side=self._preferred_v9_waypoint_side(),
                        )
                        cross_vessel_segments.extend(pieces)
                        vessel_bowed |= bowed
                    outbound_segments = cross_vessel_segments
            fixture_roles = self._mounted_fixture_roles(
                environment_version,
                pendant_lateral_bow=bool(
                    (getattr(self.task, "scene_params", {}) or {}).get(
                        "pact_v98_pendant_lateral_bow"
                    )
                ),
            )
            if fixture_roles:
                for fixture_role in fixture_roles:
                    fixture = next(
                        (
                            item
                            for item in self._hazard_list()
                            if item.get("role") == fixture_role
                        ),
                        None,
                    )
                    if fixture is None:
                        continue
                    fixture_segments: list[TCPMoveSegment] = []
                    fixture_bowed = False
                    prefix = f"outbound_{fixture_role}"
                    for candidate_segment in outbound_segments:
                        if environment_version == "pact_place_corridor_v9_5_low_wall":
                            pieces, bowed = self._v95_link_aware_fixture_segment(
                                candidate_segment, fixture, prefix=prefix
                            )
                        else:
                            route_side = self._route_side_at_x(
                                candidate_segment, float(fixture["center"][0])
                            )
                            pieces, bowed = self._bow_segment(
                                candidate_segment,
                                prefix=prefix,
                                envelope_half_y=self._mounted_fixture_bow_envelope_half_y(
                                    fixture_role, route_side
                                ),
                                safe_gap=self.MOUNTED_FIXTURE_SAFE_GAP,
                                center=np.asarray(fixture["center"], dtype=float),
                                half=np.asarray(fixture["half"], dtype=float),
                                preferred_waypoint_side=(
                                    route_side
                                    if fixture_role == "ceiling_fixture"
                                    else None
                                ),
                            )
                        fixture_segments.extend(pieces)
                        fixture_bowed |= bowed
                    outbound_segments = fixture_segments
                    vessel_bowed |= fixture_bowed
            self.outbound_deflected = bool(panel_bowed or vessel_bowed)
        else:
            self.outbound_deflected = panel_bowed
        outbound_segments = self._v99_apply_lane(
            outbound_segments, prefix="outbound_pendant", include_target=True
        )
        outbound_segments = self._v10_apply_lane(
            outbound_segments, prefix="outbound_pendant", include_target=True
        )
        if outbound_segments:
            outbound_segments = [
                *self._subdivide_tcp_segment(
                    outbound_segments[0], self.OUTBOUND_APPROACH_MAX_STEP_M
                ),
                *outbound_segments[1:],
            ]
        self._pact_place_grasp_diagnostics.update(
            {
                "carry_position_m": list(map(float, carry_pose[:3, 3])),
                "outside_staging_position_m": list(
                    map(float, outside_staging_pose[:3, 3])
                ),
                "preplace_position_m": list(map(float, preplace_pose[:3, 3])),
                "place_position_m": list(map(float, place_pose[:3, 3])),
                "release_clearance_m": float(self.RELEASE_CLEARANCE_M),
                "outbound_waypoint_positions_m": [
                    list(map(float, segment.end_pose[:3, 3]))
                    for segment in outbound_segments
                ],
                "bow_diagnostics": self._pact_place_bow_diagnostics,
            }
        )
        placement_sequence = self._sequence(
            carry_raise + outbound_segments + [preplace_transition, place],
            holding=True,
        )
        release = GripperAction(
            robot_view, True, self.policy_config.gripper_open_duration
        )
        retreat = self._sequence(
            [
                TCPMoveSegment(
                    name="retreat",
                    start_pose=place_pose,
                    end_pose=postplace_pose,
                    speed=self.policy_config.speed_fast,
                )
            ],
            holding=False,
        )
        primitives = pick_primitives + [
            placement_sequence,
            release,
            retreat,
            NoopAction(robot_view, 2.0),
        ]
        if self.inbound_deflected or self.outbound_deflected:
            self.behavior_class = "scripted_two_phase_bidirectional_deflect"
        else:
            self.behavior_class = "scripted_two_phase_straight"
        primitives = self._v104_apply_speed_amendment(primitives)
        primitives = self._v105_apply_speed_amendment(primitives)
        primitives = self._v106_apply_speed_amendment(primitives)
        return primitives

    @staticmethod
    def _traversal_phase(policy_phase: str) -> str:
        if policy_phase.startswith("inbound") or policy_phase in {
            "gripper-open",
            "pregrasp",
            "grasp",
            "grasp_settle",
            "gripper-close",
        }:
            return "inbound"
        if policy_phase.startswith("outbound") or policy_phase in {
            "lift",
        }:
            return "outbound"
        # preplace is the deliberate approach to the tray. Exempt receptacle
        # contact only during placement; treat preplace as placement so the
        # existing phase_frames_with_contact buckets can enforce that rule.
        if policy_phase.startswith("placement") or policy_phase in {
            "preplace",
            "place",
            "retreat",
            "go_home",
        }:
            return "placement"
        return "other"

    def get_all_phases(self):
        phases = super().get_all_phases()
        phase_names = (
            "inbound_approach",
            "inbound_pass",
            "inbound_exit",
            "inbound_vessel_approach",
            "inbound_vessel_pass",
            "inbound_vessel_exit",
            "inbound_wall_fixture_approach",
            "inbound_wall_fixture_pass",
            "inbound_wall_fixture_exit",
            "inbound_ceiling_fixture_approach",
            "inbound_ceiling_fixture_pass",
            "inbound_ceiling_fixture_exit",
            "inbound_pendant_approach",
            "inbound_pendant_pass",
            "inbound_pendant_exit",
            "deflect",
            "pass_protrusion",
            "inbound_grasp",
            "grasp_settle",
            "outbound_lift",
            "outbound_approach",
            "outbound_pass",
            "outbound_exit",
            "outbound_vessel_approach",
            "outbound_vessel_pass",
            "outbound_vessel_exit",
            "outbound_wall_fixture_approach",
            "outbound_wall_fixture_pass",
            "outbound_wall_fixture_exit",
            "outbound_ceiling_fixture_approach",
            "outbound_ceiling_fixture_pass",
            "outbound_ceiling_fixture_exit",
            "outbound_pendant_approach",
            "outbound_pendant_pass",
            "outbound_pendant_exit",
            "placement_descent",
        )
        next_id = max(phases.values()) + 1
        phases.update({name: next_id + index for index, name in enumerate(phase_names)})
        return phases

    def reset(self, reset_retries: bool = True):
        from molmo_spaces.tasks.pact_place_contact_audit import (
            PactPlaceContactAudit,
        )

        if reset_retries or not hasattr(self, "_pact_place_contact_audit"):
            self._pact_place_contact_audit = PactPlaceContactAudit()
            self._pact_place_control_step = 0
            self._cup_retrieved_outside_aperture = False
            self._cup_lifted = False
            self._pickup_start_z = None
            self._pickup_max_z = None
            self._pickup_final_position = None
            self._pickup_start_position = None
            self._pickup_start_quat = None
            self._pickup_final_quat = None
            self._object_position_window = deque(maxlen=self.SETTLE_WINDOW_STEPS)
            self._pact_place_trajectory = []
            self._pact_abort_branch_steps = []
            self._pact_abort_branch_terminal = None
            self._gripper_width_min = None
            self._gripper_width_max = None
            self._pact_clutter_stability_events = []
            self._pact_clutter_stability_bodies = set()
        self.task._contact_audit_hook = self._pact_place_contact_audit
        self.behavior_class = "straight"
        self.inbound_deflected = False
        self.outbound_deflected = False
        self._pact_place_bow_diagnostics = self._empty_bow_diagnostics()
        self._pact_place_v99_route = {
            "inbound_pendant": self._v99_empty_route_record(),
            "outbound_pendant": self._v99_empty_route_record(),
        }
        self._pact_place_v10_route = {
            "inbound_pendant": self._v99_empty_route_record(),
            "outbound_pendant": self._v99_empty_route_record(),
        }
        self._sensor_cam_ids = None
        self._pact_detected_hazard_names = set()
        self._pact_detected_hazards = []
        self._pact_maneuver_interactions = []
        self._pact_active_maneuver = None
        result = super().reset(reset_retries)
        self.target_poses.update(self._pact_place_canonical_target_poses)
        return result

    def _update_manipulation_progress(self) -> None:
        try:
            task_config = self.task.config.task_config
            manager = self.task.env.object_managers[
                self.task.env.current_batch_index
            ]
            pickup = manager.get_object_by_name(task_config.pickup_obj_name)
            if self._pickup_start_z is None:
                self._pickup_start_z = float(pickup.position[2])
            pickup_z = float(pickup.position[2])
            self._pickup_max_z = (
                pickup_z
                if self._pickup_max_z is None
                else max(self._pickup_max_z, pickup_z)
            )
            self._pickup_final_position = list(map(float, pickup.position))
            self._pickup_final_quat = list(map(float, pickup.quat))
            if self._pickup_start_position is None:
                self._pickup_start_position = list(self._pickup_final_position)
                self._pickup_start_quat = list(self._pickup_final_quat)
            self._object_position_window.append(list(self._pickup_final_position))
            self._cup_lifted |= bool(
                pickup_z >= self._pickup_start_z + 0.01
            )
            self._cup_retrieved_outside_aperture |= bool(
                self._cup_lifted and float(pickup.position[0]) < TUBE_X0 - 0.03
            )
            robot_view = self.task.env.current_robot.robot_view
            gripper_id = robot_view.get_gripper_movegroup_ids()[0]
            gripper = robot_view.get_gripper(gripper_id)
            width = float(gripper.inter_finger_dist)
            self._gripper_width_min = (
                width
                if self._gripper_width_min is None
                else min(self._gripper_width_min, width)
            )
            self._gripper_width_max = (
                width
                if self._gripper_width_max is None
                else max(self._gripper_width_max, width)
            )
        except Exception:
            return

    def _update_clutter_stability(self) -> None:
        """Turn a displaced/toppled v5 free body into an outcome-bearing event."""
        scene = getattr(self.task, "scene_params", {}) or {}
        settle = scene.get("pact_clutter_settle") or {}
        for baseline in settle.get("objects") or []:
            body_name = str(baseline["body"])
            if body_name in self._pact_clutter_stability_bodies:
                continue
            try:
                model = self.task.env.current_model
                data = self.task.env.current_data
                body_id = int(model.body(body_name).id)
                position = np.asarray(data.xpos[body_id], dtype=float)
                reference_position = np.asarray(baseline["position_m"], dtype=float)
                rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
                reference_rotation = np.asarray(
                    baseline["xmat"], dtype=float
                ).reshape(3, 3)
                cosine = float(
                    np.clip(
                        (np.trace(reference_rotation.T @ rotation) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                )
                displacement = float(np.linalg.norm(position - reference_position))
                rotation_angle = float(np.arccos(cosine))
                if displacement <= 0.02 and rotation_angle <= np.deg2rad(25.0):
                    continue
                self._pact_clutter_stability_bodies.add(body_name)
                self._pact_clutter_stability_events.append(
                    {
                        "step": int(self._pact_place_control_step),
                        "policy_phase": str(self.get_phase()),
                        "body": body_name,
                        "classification": "other_environment",
                        "reason": "movable_clutter_toppled_or_displaced",
                        "displacement_m": displacement,
                        "rotation_angle_rad": rotation_angle,
                    }
                )
            except Exception:
                continue

    def _pact_abort_recorder(self):
        scripts = Path(__file__).resolve().parents[4] / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from pact_place_abort_branch_telemetry import (  # noqa: E402
            record_on_policy,
            terminal_tracking_fields,
            write_sidecar,
        )

        return record_on_policy, terminal_tracking_fields, write_sidecar

    def _check_for_failures(self) -> bool:
        failed = super()._check_for_failures()
        try:
            record_on_policy, _, _ = self._pact_abort_recorder()
            record_on_policy(self, failed)
        except Exception:
            pass
        return failed

    def _handle_failure(self) -> dict[str, Any]:
        try:
            record_on_policy, _, _ = self._pact_abort_recorder()
            if getattr(self, "_pact_abort_branch_terminal", None) is None:
                snapshot = record_on_policy(self, True)
                if (
                    int(self.sequential_ik_failures)
                    >= int(self.policy_config.max_sequential_ik_failures)
                ):
                    snapshot["branch"] = "ik_cascade"
                    self._pact_abort_branch_terminal = snapshot
        except Exception:
            pass
        return super()._handle_failure()

    def _v102_cached_component_geoms(self) -> list[tuple[str, int]]:
        if self._pact_v102_component_geoms is None:
            self._pact_v102_component_geoms = self._v102_active_component_geoms()
        return list(self._pact_v102_component_geoms)

    def _v102_live_pendant_contacts(self) -> dict[str, Any]:
        """Raw ``data.contact`` and classifier views of pendant contact."""
        from molmo_spaces.tasks.pact_place_contact_audit import (
            classify_contact,
            place_environment_contact_pairs,
        )

        model, data = self.task.env.current_model, self.task.env.current_data
        pendant_ids = {gid for _name, gid in self._v102_cached_component_geoms()}
        raw: list[dict[str, Any]] = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if float(contact.dist) > 0.0:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 not in pendant_ids and geom2 not in pendant_ids:
                continue
            raw.append(
                {
                    "geom1": model.geom(geom1).name or f"geom_{geom1}",
                    "geom2": model.geom(geom2).name or f"geom_{geom2}",
                    "distance_m": float(contact.dist),
                }
            )
        classified = [
            pair
            for pair in place_environment_contact_pairs(self.task.env)
            if classify_contact(pair) == "mounted_fixture"
        ]
        return {
            "raw_contact_pairs": raw[:8],
            "n_raw_contact_pairs": len(raw),
            "n_classified_mounted_fixture_pairs": len(classified),
            "classified_pairs": [
                {
                    "geom1": pair.get("geom1"),
                    "geom2": pair.get("geom2"),
                    "distance_m": pair.get("distance_m"),
                }
                for pair in classified[:8]
            ],
            "contact": bool(raw or classified),
        }

    def _v102_frame_telemetry(self) -> dict[str, Any]:
        holding = str(self.get_phase() or "")
        include_target = not (
            holding.startswith("inbound")
            or holding in {"gripper-open", "pregrasp", "grasp", "grasp_settle", "gripper-close"}
        )
        clearances = self._v102_component_clearances(include_target=include_target)
        finite = [value for value in clearances.values() if value is not None]
        contacts = self._v102_live_pendant_contacts()
        return {
            "component_clearance_m": clearances,
            "min_clearance_m": float(min(finite)) if finite else None,
            "clearance_includes_target": bool(include_target),
            "pendant_contact": bool(contacts["contact"]),
            "n_raw_pendant_contact_pairs": int(contacts["n_raw_contact_pairs"]),
            "n_classified_mounted_fixture_pairs": int(
                contacts["n_classified_mounted_fixture_pairs"]
            ),
            "contact_pairs": contacts["raw_contact_pairs"],
        }

    def _current_move_segment(self):
        if self.action_idx >= len(self.action_primitives):
            return None
        primitive = self.action_primitives[self.action_idx]
        if not isinstance(primitive, TCPMoveSequence):
            return None
        index = primitive.move_seg_idx
        if index is None:
            return None
        segments = getattr(primitive, "_move_segments", None) or []
        if not 0 <= int(index) < len(segments):
            return None
        return segments[int(index)]

    def _v104_apply_speed_amendment(self, primitives):
        """The single registered V10.4 initial free-space speed cap.

        Gated on the exact V10.4 marker; V6c and every historical environment
        return unchanged. Records the untouched baseline plan alongside the
        amended one so the caller can prove exactly one segment differs.
        """
        from pact_place_v104_runtime import (
            apply_initial_free_space_speed_cap,
            plan_signature,
            verify_plan_matches_baseline,
        )

        if not self._v104_enabled():
            self._pact_place_v104_speed_amendment = {
                "applied": False,
                "reason": "environment marker is not V10.4",
            }
            return primitives
        baseline = plan_signature(primitives)
        record = apply_initial_free_space_speed_cap(primitives)
        amended = plan_signature(primitives)
        record["baseline_vs_amended"] = verify_plan_matches_baseline(baseline, amended)
        record["baseline_plan"] = baseline
        self._pact_place_v104_speed_amendment = record
        return primitives

    def _v104_pendant_geom_ids(self) -> list[int]:
        if self._pact_v104_pendant_geom_ids is None:
            from pact_place_v104_runtime import PENDANT_GEOM_PREFIX

            model = self.task.env.current_model
            ids = []
            for geom_id in range(int(model.ngeom)):
                name = str(model.geom(geom_id).name or "")
                if name.startswith(PENDANT_GEOM_PREFIX):
                    ids.append(int(geom_id))
            self._pact_v104_pendant_geom_ids = ids
        return list(self._pact_v104_pendant_geom_ids)

    def _v104_probe_ids(self) -> list[int]:
        if self._pact_v104_probe_ids is None:
            from pact_place_v104_clearance import (
                robot_collision_geom_ids,
                target_collision_geom_ids,
            )

            model = self.task.env.current_model
            ids = list(robot_collision_geom_ids(model)) + list(
                target_collision_geom_ids(self.task)
            )
            self._pact_v104_probe_ids = ids
        return list(self._pact_v104_probe_ids)

    def _v104_current_segment(self):
        """(primitive index, segment index, segment, holding) or Nones."""
        if self.action_idx >= len(self.action_primitives):
            return None, None, None, None
        primitive = self.action_primitives[self.action_idx]
        holding = bool(getattr(primitive, "is_holding_object", False))
        if not isinstance(primitive, TCPMoveSequence):
            return int(self.action_idx), None, None, holding
        index = primitive.move_seg_idx
        segments = getattr(primitive, "_move_segments", None) or []
        if index is None or not 0 <= int(index) < len(segments):
            return int(self.action_idx), None, None, holding
        return int(self.action_idx), int(index), segments[int(index)], holding

    def _v106_apply_speed_amendment(self, primitives):
        """The single registered V10.6 initial free-space speed cap.

        Gated on the exact V10.6 marker and on the baseline schedule hash, the
        same double gate V10.5 uses. Every other inherited speed is untouched.
        """
        from pact_place_v105_runtime import (
            apply_initial_free_space_speed_cap,
            plan_signature,
            schedule_sha256,
            verify_plan_matches_baseline,
        )

        if not self._v106_enabled():
            self._pact_place_v106_speed_amendment = {
                "applied": False,
                "reason": "environment marker is not V10.6",
            }
            return primitives
        baseline = plan_signature(primitives)
        baseline_schedule = schedule_sha256(primitives)
        record = apply_initial_free_space_speed_cap(primitives)
        amended = plan_signature(primitives)
        record["baseline_vs_amended"] = verify_plan_matches_baseline(baseline, amended)
        record["baseline_plan"] = baseline
        record["baseline_schedule_sha256"] = baseline_schedule
        record["marker_gated"] = True
        record["schedule_gated"] = True
        self._pact_place_v106_speed_amendment = record
        return primitives

    def _v106_assembly(self):
        if self._pact_v106_assembly is None:
            from pact_place_v106_geometry import POSE_OFFSETS_M, build_assembly

            scene = getattr(self.task, "scene_params", {}) or {}
            pose_id = scene.get("pact_v106_pose_id")
            x = scene.get("pact_v106_x_m")
            rn = scene.get("pact_v106_r_neg_m")
            rp = scene.get("pact_v106_r_pos_m")
            if pose_id is None or x is None or rn is None or rp is None:
                return None
            self._pact_v106_assembly = build_assembly(
                float(x), float(rn), float(rp), POSE_OFFSETS_M[str(pose_id)],
                pose_id=str(pose_id),
            )
        return self._pact_v106_assembly

    def _v106_pendant_geom_ids(self) -> list[int]:
        if self._pact_v106_pendant_geom_ids is None:
            model = self.task.env.current_model
            ids = []
            for geom_id in range(int(model.ngeom)):
                name = str(model.geom(geom_id).name or "")
                if name.startswith("pact_clutter_mount_v106_"):
                    ids.append(int(geom_id))
            self._pact_v106_pendant_geom_ids = ids
        return list(self._pact_v106_pendant_geom_ids)

    def _v106_probe_ids(self) -> list[int]:
        if self._pact_v106_probe_ids is None:
            from pact_place_v105_clearance import (
                robot_collision_geom_ids,
                target_collision_geom_ids,
            )

            model = self.task.env.current_model
            self._pact_v106_probe_ids = list(
                robot_collision_geom_ids(model)
            ) + list(target_collision_geom_ids(self.task))
        return list(self._pact_v106_probe_ids)

    def _v106_frame_telemetry(self) -> dict[str, Any]:
        from pact_place_v105_clearance import (
            assembly_boxes,
            frame_clearances,
            geom_shape_cache,
            pendant_contact_state,
            risk_boxes,
        )

        model, data = self.task.env.current_model, self.task.env.current_data
        assembly = self._v106_assembly()
        if assembly is None:
            return {"telemetry_error": "V10.6 row lacks (x, r_neg, r_pos, pose_id)"}
        if self._pact_v106_boxes is None:
            self._pact_v106_boxes = assembly_boxes(assembly)
            self._pact_v106_risk_boxes = risk_boxes(assembly)
        probe = self._v106_probe_ids()
        if self._pact_v106_shape_cache is None:
            self._pact_v106_shape_cache = geom_shape_cache(model, probe)
        report = frame_clearances(
            model, data, self._pact_v106_boxes, probe, self._pact_v106_shape_cache
        )
        risk = frame_clearances(
            model, data, self._pact_v106_risk_boxes, probe,
            self._pact_v106_shape_cache,
        )
        contact = pendant_contact_state(model, data, self._v106_pendant_geom_ids())
        return {
            "pendant_min_clearance_m": report["min_m"],
            "pendant_per_component_m": report["per_component_m"],
            "pendant_limiting": report["limiting"],
            "lobe_stem_min_clearance_m": risk["min_m"],
            "lobe_stem_limiting": risk["limiting"],
            "pendant_contact": bool(contact["contact"]),
            "pendant_robot_or_target_contact": bool(
                contact["robot_or_target_contact"]
            ),
            "pendant_contact_pairs": contact["pairs"],
            "pendant_contact_classes": contact["contact_classes"],
            "n_pendant_contact_pairs": int(contact["n_pairs"]),
            "pose_id": (
                getattr(self.task, "scene_params", {}) or {}
            ).get("pact_v106_pose_id"),
            "scene_sha256": (
                getattr(self.task, "scene_params", {}) or {}
            ).get("pact_v106_scene_sha256"),
        }

    def _v106_frame_summary(self) -> dict[str, Any]:
        frames = list(self._pact_place_v106_frames)
        if not frames:
            return {}
        names: list[str] = []
        for frame in frames:
            for name in (frame.get("pendant_per_component_m") or {}):
                if name not in names:
                    names.append(name)
        per_component = {}
        for name in names:
            values = [
                float(frame["pendant_per_component_m"][name])
                for frame in frames
                if (frame.get("pendant_per_component_m") or {}).get(name) is not None
            ]
            per_component[name] = float(min(values)) if values else None
        measured = [f for f in frames if f.get("pendant_min_clearance_m") is not None]
        risk_measured = [
            f for f in frames if f.get("lobe_stem_min_clearance_m") is not None
        ]
        contact_frames = [f for f in frames if f.get("pendant_contact")]
        robot_contact = [
            f for f in frames if f.get("pendant_robot_or_target_contact")
        ]
        worst = (
            min(measured, key=lambda f: float(f["pendant_min_clearance_m"]))
            if measured else None
        )
        worst_risk = (
            min(risk_measured, key=lambda f: float(f["lobe_stem_min_clearance_m"]))
            if risk_measured else None
        )
        speeds: list[dict[str, Any]] = []
        seen = set()
        for frame in frames:
            key = (
                frame.get("primitive_index"), frame.get("segment_index"),
                frame.get("segment_name"),
                None if frame.get("commanded_speed_m_s") is None
                else round(float(frame["commanded_speed_m_s"]), 9),
            )
            if key in seen or key[3] is None:
                continue
            seen.add(key)
            speeds.append({
                "primitive_index": key[0], "segment_index": key[1],
                "segment_name": key[2], "commanded_speed_m_s": key[3],
                "first_step": int(frame["step"]),
            })
        realized = [
            float(f["realized_tcp_speed_m_s"]) for f in frames
            if f.get("realized_tcp_speed_m_s") is not None
        ]
        return {
            "schema_version": "pact_place_v106_frame_telemetry_v1",
            "n_frames": len(frames), "n_frames_measured": len(measured),
            "min_clearance_m": (
                float(worst["pendant_min_clearance_m"]) if worst else None
            ),
            "min_lobe_stem_clearance_m": (
                float(worst_risk["lobe_stem_min_clearance_m"])
                if worst_risk else None
            ),
            "min_clearance_witness": ({
                "step": int(worst["step"]),
                "policy_phase": worst.get("policy_phase"),
                "segment_name": worst.get("segment_name"),
                "limiting": worst.get("pendant_limiting"),
                "target_held": worst.get("target_held"),
            } if worst else None),
            "min_lobe_stem_witness": ({
                "step": int(worst_risk["step"]),
                "policy_phase": worst_risk.get("policy_phase"),
                "segment_name": worst_risk.get("segment_name"),
                "limiting": worst_risk.get("lobe_stem_limiting"),
                "target_held": worst_risk.get("target_held"),
            } if worst_risk else None),
            "per_component_min_clearance_m": per_component,
            "pendant_contact_frames": len(contact_frames),
            "pendant_robot_or_target_contact_frames": len(robot_contact),
            "first_pendant_contact_step": (
                int(contact_frames[0]["step"]) if contact_frames else None
            ),
            "first_pendant_contact_pairs": (
                contact_frames[0].get("pendant_contact_pairs")
                if contact_frames else None
            ),
            "segment_speeds": speeds,
            "max_realized_tcp_speed_m_s": max(realized) if realized else None,
            "pose_id": frames[0].get("pose_id"),
            "scene_sha256": frames[0].get("scene_sha256"),
        }

    def _v105_apply_speed_amendment(self, primitives):
        """The single registered V10.5 initial free-space speed cap.

        Gated twice: on the exact V10.5 marker, and on the hash of the
        inherited baseline speed schedule. If the V9.3 plan ever changes shape
        the schedule hash stops matching and this refuses rather than silently
        capping a different segment. Every other inherited speed is untouched.
        """
        from pact_place_v105_runtime import (
            apply_initial_free_space_speed_cap,
            plan_signature,
            schedule_sha256,
            verify_plan_matches_baseline,
        )

        if not self._v105_enabled():
            self._pact_place_v105_speed_amendment = {
                "applied": False,
                "reason": "environment marker is not V10.5",
            }
            return primitives
        baseline = plan_signature(primitives)
        baseline_schedule = schedule_sha256(primitives)
        record = apply_initial_free_space_speed_cap(primitives)
        amended = plan_signature(primitives)
        record["baseline_vs_amended"] = verify_plan_matches_baseline(baseline, amended)
        record["baseline_plan"] = baseline
        record["baseline_schedule_sha256"] = baseline_schedule
        record["marker_gated"] = True
        record["schedule_gated"] = True
        self._pact_place_v105_speed_amendment = record
        return primitives

    def _v105_assembly(self):
        if self._pact_v105_assembly is None:
            from pact_place_v105_geometry import POSE_OFFSETS_M, build_assembly

            row = self._pact_manifest_row or {}
            scene = getattr(self.task, "scene_params", {}) or {}
            pose_id = row.get("pose_id") or scene.get("pact_v105_pose_id")
            x = row.get("pact_v105_x_m")
            r = row.get("pact_v105_r_m")
            if pose_id is None or x is None or r is None:
                return None
            self._pact_v105_assembly = build_assembly(
                float(x), float(r), POSE_OFFSETS_M[str(pose_id)], pose_id=str(pose_id)
            )
        return self._pact_v105_assembly

    def _v105_pendant_geom_ids(self) -> list[int]:
        if self._pact_v105_pendant_geom_ids is None:
            from pact_place_v105_runtime import PENDANT_GEOM_PREFIX

            model = self.task.env.current_model
            ids = []
            for geom_id in range(int(model.ngeom)):
                name = str(model.geom(geom_id).name or "")
                if name.startswith(PENDANT_GEOM_PREFIX):
                    ids.append(int(geom_id))
            self._pact_v105_pendant_geom_ids = ids
        return list(self._pact_v105_pendant_geom_ids)

    def _v105_probe_ids(self) -> list[int]:
        if self._pact_v105_probe_ids is None:
            from pact_place_v105_clearance import (
                robot_collision_geom_ids,
                target_collision_geom_ids,
            )

            model = self.task.env.current_model
            self._pact_v105_probe_ids = list(
                robot_collision_geom_ids(model)
            ) + list(target_collision_geom_ids(self.task))
        return list(self._pact_v105_probe_ids)

    def _v105_frame_telemetry(self) -> dict[str, Any]:
        from pact_place_v105_clearance import (
            assembly_boxes,
            frame_clearances,
            geom_shape_cache,
            pendant_contact_state,
            risk_boxes,
        )

        model, data = self.task.env.current_model, self.task.env.current_data
        assembly = self._v105_assembly()
        if assembly is None:
            return {"telemetry_error": "V10.5 row does not carry (x, r, pose_id)"}
        if self._pact_v105_boxes is None:
            self._pact_v105_boxes = assembly_boxes(assembly)
            self._pact_v105_risk_boxes = risk_boxes(assembly)
        probe = self._v105_probe_ids()
        if self._pact_v105_shape_cache is None:
            self._pact_v105_shape_cache = geom_shape_cache(model, probe)
        report = frame_clearances(
            model, data, self._pact_v105_boxes, probe, self._pact_v105_shape_cache
        )
        risk = frame_clearances(
            model, data, self._pact_v105_risk_boxes, probe,
            self._pact_v105_shape_cache,
        )
        contact = pendant_contact_state(
            model, data, self._v105_pendant_geom_ids()
        )
        return {
            "pendant_min_clearance_m": report["min_m"],
            "pendant_per_component_m": report["per_component_m"],
            "pendant_limiting": report["limiting"],
            "lobe_stem_min_clearance_m": risk["min_m"],
            "lobe_stem_limiting": risk["limiting"],
            "pendant_contact": bool(contact["contact"]),
            "pendant_robot_or_target_contact": bool(
                contact["robot_or_target_contact"]
            ),
            "pendant_contact_pairs": contact["pairs"],
            "pendant_contact_classes": contact["contact_classes"],
            "n_pendant_contact_pairs": int(contact["n_pairs"]),
            "pose_id": (self._pact_manifest_row or {}).get("pose_id"),
            "scene_sha256": (self._pact_manifest_row or {}).get(
                "pact_v105_scene_sha256"
            ),
        }

    def _v105_frame_summary(self) -> dict[str, Any]:
        frames = list(self._pact_place_v105_frames)
        if not frames:
            return {}
        names: list[str] = []
        for frame in frames:
            for name in (frame.get("pendant_per_component_m") or {}):
                if name not in names:
                    names.append(name)
        per_component = {}
        for name in names:
            values = [
                float(frame["pendant_per_component_m"][name])
                for frame in frames
                if (frame.get("pendant_per_component_m") or {}).get(name) is not None
            ]
            per_component[name] = float(min(values)) if values else None
        measured = [
            f for f in frames if f.get("pendant_min_clearance_m") is not None
        ]
        risk_measured = [
            f for f in frames if f.get("lobe_stem_min_clearance_m") is not None
        ]
        contact_frames = [f for f in frames if f.get("pendant_contact")]
        robot_contact_frames = [
            f for f in frames if f.get("pendant_robot_or_target_contact")
        ]
        worst = (
            min(measured, key=lambda f: float(f["pendant_min_clearance_m"]))
            if measured
            else None
        )
        worst_risk = (
            min(risk_measured, key=lambda f: float(f["lobe_stem_min_clearance_m"]))
            if risk_measured
            else None
        )
        speeds: list[dict[str, Any]] = []
        seen = set()
        for frame in frames:
            key = (
                frame.get("primitive_index"),
                frame.get("segment_index"),
                frame.get("segment_name"),
                None
                if frame.get("commanded_speed_m_s") is None
                else round(float(frame["commanded_speed_m_s"]), 9),
            )
            if key in seen or key[3] is None:
                continue
            seen.add(key)
            speeds.append(
                {
                    "primitive_index": key[0],
                    "segment_index": key[1],
                    "segment_name": key[2],
                    "commanded_speed_m_s": key[3],
                    "first_step": int(frame["step"]),
                }
            )
        realized = [
            float(f["realized_tcp_speed_m_s"])
            for f in frames
            if f.get("realized_tcp_speed_m_s") is not None
        ]
        return {
            "schema_version": "pact_place_v105_frame_telemetry_v1",
            "n_frames": len(frames),
            "n_frames_measured": len(measured),
            "min_clearance_m": (
                float(worst["pendant_min_clearance_m"]) if worst else None
            ),
            "min_lobe_stem_clearance_m": (
                float(worst_risk["lobe_stem_min_clearance_m"]) if worst_risk else None
            ),
            "min_clearance_witness": (
                {
                    "step": int(worst["step"]),
                    "policy_phase": worst.get("policy_phase"),
                    "segment_name": worst.get("segment_name"),
                    "limiting": worst.get("pendant_limiting"),
                    "target_held": worst.get("target_held"),
                }
                if worst
                else None
            ),
            "min_lobe_stem_witness": (
                {
                    "step": int(worst_risk["step"]),
                    "policy_phase": worst_risk.get("policy_phase"),
                    "segment_name": worst_risk.get("segment_name"),
                    "limiting": worst_risk.get("lobe_stem_limiting"),
                    "target_held": worst_risk.get("target_held"),
                }
                if worst_risk
                else None
            ),
            "per_component_min_clearance_m": per_component,
            "pendant_contact_frames": len(contact_frames),
            "pendant_robot_or_target_contact_frames": len(robot_contact_frames),
            "first_pendant_contact_step": (
                int(contact_frames[0]["step"]) if contact_frames else None
            ),
            "first_pendant_contact_pairs": (
                contact_frames[0].get("pendant_contact_pairs")
                if contact_frames
                else None
            ),
            "segment_speeds": speeds,
            "max_realized_tcp_speed_m_s": max(realized) if realized else None,
            "pose_id": frames[0].get("pose_id"),
            "scene_sha256": frames[0].get("scene_sha256"),
        }

    def _v104_frame_telemetry(self) -> dict[str, Any]:
        from pact_place_v104_clearance import assembly_boxes, frame_clearances, geom_shape_cache
        from pact_place_v104_geometry import production_assembly
        from pact_place_v104_runtime import pendant_contact_state

        model, data = self.task.env.current_model, self.task.env.current_data
        if self._pact_v104_boxes is None:
            self._pact_v104_boxes = assembly_boxes(production_assembly())
        probe = self._v104_probe_ids()
        if self._pact_v104_shape_cache is None:
            self._pact_v104_shape_cache = geom_shape_cache(model, probe)
        report = frame_clearances(
            model, data, self._pact_v104_boxes, probe, self._pact_v104_shape_cache
        )
        contact = pendant_contact_state(model, data, self._v104_pendant_geom_ids())
        return {
            "pendant_min_clearance_m": report["min_m"],
            "pendant_per_component_m": report["per_component_m"],
            "pendant_limiting": report["limiting"],
            "pendant_contact": bool(contact["contact"]),
            "pendant_contact_pairs": contact["pairs"],
            "n_pendant_contact_pairs": int(contact["n_pairs"]),
        }

    def _v104_frame_summary(self) -> dict[str, Any]:
        frames = list(self._pact_place_v104_frames)
        if not frames:
            return {}
        names: list[str] = []
        for frame in frames:
            for name in (frame.get("pendant_per_component_m") or {}):
                if name not in names:
                    names.append(name)
        per_component = {}
        for name in names:
            values = [
                float(frame["pendant_per_component_m"][name])
                for frame in frames
                if (frame.get("pendant_per_component_m") or {}).get(name) is not None
            ]
            per_component[name] = float(min(values)) if values else None
        measured = [f for f in frames if f.get("pendant_min_clearance_m") is not None]
        contact_frames = [f for f in frames if f.get("pendant_contact")]
        worst = min(measured, key=lambda f: float(f["pendant_min_clearance_m"])) if measured else None
        speeds: list[dict[str, Any]] = []
        seen = set()
        for frame in frames:
            key = (
                frame.get("primitive_index"),
                frame.get("segment_index"),
                frame.get("segment_name"),
                None if frame.get("commanded_speed_m_s") is None
                else round(float(frame["commanded_speed_m_s"]), 9),
            )
            if key in seen or key[3] is None:
                continue
            seen.add(key)
            speeds.append(
                {
                    "primitive_index": key[0],
                    "segment_index": key[1],
                    "segment_name": key[2],
                    "commanded_speed_m_s": key[3],
                    "first_step": int(frame["step"]),
                }
            )
        return {
            "schema_version": "pact_place_v104_frame_telemetry_v1",
            "n_frames": len(frames),
            "n_frames_measured": len(measured),
            "min_clearance_m": (
                float(worst["pendant_min_clearance_m"]) if worst else None
            ),
            "min_clearance_witness": (
                {
                    "step": int(worst["step"]),
                    "policy_phase": worst.get("policy_phase"),
                    "segment_name": worst.get("segment_name"),
                    "limiting": worst.get("pendant_limiting"),
                    "target_held": worst.get("target_held"),
                }
                if worst
                else None
            ),
            "per_component_min_clearance_m": per_component,
            "pendant_contact_frames": len(contact_frames),
            "first_pendant_contact_step": (
                int(contact_frames[0]["step"]) if contact_frames else None
            ),
            "first_pendant_contact_pairs": (
                contact_frames[0].get("pendant_contact_pairs") if contact_frames else None
            ),
            "segment_speeds": speeds,
            "max_realized_tcp_speed_m_s": (
                float(max(
                    float(f["realized_tcp_speed_m_s"]) for f in frames
                    if f.get("realized_tcp_speed_m_s") is not None
                ))
                if any(f.get("realized_tcp_speed_m_s") is not None for f in frames)
                else None
            ),
            "telemetry_errors": [
                {"step": int(f["step"]), "error": f["telemetry_error"]}
                for f in frames if f.get("telemetry_error")
            ],
        }

    def _record_place_trajectory_step(self) -> None:
        tcp_pos = None
        try:
            gripper_id = self.robot_view.get_gripper_movegroup_ids()[0]
            tcp = self.robot_view.get_gripper(gripper_id).leaf_frame_to_world
            tcp_pos = list(map(float, tcp[:3, 3]))
        except Exception:
            tcp_pos = None
        sim_time_s = float(self.task.env.current_data.time)
        segment = None
        try:
            segment = self._current_move_segment()
        except Exception:
            segment = None
        realized_step_m = None
        realized_speed_m_s = None
        if tcp_pos is not None:
            current = np.asarray(tcp_pos, dtype=float)
            if self._pact_place_last_tcp_m is not None:
                realized_step_m = float(
                    np.linalg.norm(current - self._pact_place_last_tcp_m)
                )
                if self._pact_place_last_sim_time_s is not None:
                    dt = sim_time_s - float(self._pact_place_last_sim_time_s)
                    if dt > 1e-9:
                        realized_speed_m_s = float(realized_step_m / dt)
            self._pact_place_last_tcp_m = current
        self._pact_place_last_sim_time_s = sim_time_s
        frame_extra: dict[str, Any] = {
            "segment_name": None if segment is None else str(segment.name),
            "commanded_speed_m_s": None if segment is None else float(segment.speed),
            "realized_tcp_displacement_m": realized_step_m,
            "realized_tcp_speed_m_s": realized_speed_m_s,
        }
        if self._v104_enabled():
            primitive_index, segment_index, segment, holding = (
                self._v104_current_segment()
            )
            frame_extra["primitive_index"] = primitive_index
            frame_extra["segment_index"] = segment_index
            frame_extra["target_held"] = holding
            if segment is not None:
                frame_extra["segment_name"] = str(segment.name)
                frame_extra["commanded_speed_m_s"] = float(segment.speed)
            frame_extra["traversal_phase"] = self._traversal_phase(str(self.get_phase()))
            try:
                frame_extra.update(self._v104_frame_telemetry())
            except Exception as error:  # noqa: BLE001 - telemetry must not be silent
                frame_extra["telemetry_error"] = f"{type(error).__name__}: {error}"
            self._pact_place_v104_frames.append(
                {
                    "step": int(self._pact_place_control_step),
                    "sim_time_s": sim_time_s,
                    "policy_phase": str(self.get_phase()),
                    **frame_extra,
                }
            )
        if self._v106_enabled():
            primitive_index, segment_index, segment, holding = (
                self._v104_current_segment()
            )
            frame_extra["primitive_index"] = primitive_index
            frame_extra["segment_index"] = segment_index
            frame_extra["target_held"] = holding
            if segment is not None:
                frame_extra["segment_name"] = str(segment.name)
                frame_extra["commanded_speed_m_s"] = float(segment.speed)
            frame_extra["traversal_phase"] = self._traversal_phase(
                str(self.get_phase())
            )
            try:
                frame_extra.update(self._v106_frame_telemetry())
            except Exception as error:  # noqa: BLE001 - telemetry must not be silent
                frame_extra["telemetry_error"] = f"{type(error).__name__}: {error}"
            self._pact_place_v106_frames.append({
                "step": int(self._pact_place_control_step),
                "sim_time_s": sim_time_s,
                "policy_phase": str(self.get_phase()),
                **frame_extra,
            })
        if self._v105_enabled():
            primitive_index, segment_index, segment, holding = (
                self._v104_current_segment()
            )
            frame_extra["primitive_index"] = primitive_index
            frame_extra["segment_index"] = segment_index
            frame_extra["target_held"] = holding
            if segment is not None:
                frame_extra["segment_name"] = str(segment.name)
                frame_extra["commanded_speed_m_s"] = float(segment.speed)
            frame_extra["traversal_phase"] = self._traversal_phase(
                str(self.get_phase())
            )
            try:
                frame_extra.update(self._v105_frame_telemetry())
            except Exception as error:  # noqa: BLE001 - telemetry must not be silent
                frame_extra["telemetry_error"] = f"{type(error).__name__}: {error}"
            self._pact_place_v105_frames.append(
                {
                    "step": int(self._pact_place_control_step),
                    "sim_time_s": sim_time_s,
                    "policy_phase": str(self.get_phase()),
                    **frame_extra,
                }
            )
        if self._v102_enabled():
            try:
                frame_extra.update(self._v102_frame_telemetry())
            except Exception as error:  # noqa: BLE001 - telemetry must not be silent
                frame_extra["telemetry_error"] = f"{type(error).__name__}: {error}"
            self._pact_place_v102_frames.append(
                {
                    "step": int(self._pact_place_control_step),
                    "sim_time_s": sim_time_s,
                    "policy_phase": str(self.get_phase()),
                    **frame_extra,
                }
            )
        self._pact_place_trajectory.append(
            {
                "step": int(self._pact_place_control_step),
                "sim_time_s": sim_time_s,
                "policy_phase": str(self.get_phase()),
                **frame_extra,
                "tcp_position_m": tcp_pos,
                "object_position_m": (
                    None
                    if self._pickup_final_position is None
                    else list(self._pickup_final_position)
                ),
                "object_quat_xyzw": (
                    None
                    if self._pickup_final_quat is None
                    else list(self._pickup_final_quat)
                ),
                "qpos": [
                    float(value)
                    for value in np.asarray(self.task.env.current_data.qpos).tolist()
                ],
            }
        )

    def _v102_frame_summary(self) -> dict[str, Any]:
        """Aggregate the per-policy-frame V10.2 pendant telemetry."""
        from pact_place_v102_raised_pendant_contract import MIN_PENDANT_CLEARANCE_M

        frames = list(self._pact_place_v102_frames)
        if not frames:
            return {}
        component_names: list[str] = []
        for frame in frames:
            for name in (frame.get("component_clearance_m") or {}):
                if name not in component_names:
                    component_names.append(name)
        per_component: dict[str, float | None] = {}
        for name in component_names:
            values = [
                float(frame["component_clearance_m"][name])
                for frame in frames
                if (frame.get("component_clearance_m") or {}).get(name) is not None
            ]
            per_component[name] = float(min(values)) if values else None
        measured = [
            frame for frame in frames if frame.get("min_clearance_m") is not None
        ]
        contact_frames = [frame for frame in frames if frame.get("pendant_contact")]
        below = [
            frame
            for frame in measured
            if float(frame["min_clearance_m"]) < MIN_PENDANT_CLEARANCE_M - 1e-12
        ]
        speeds: list[dict[str, Any]] = []
        seen: set[tuple[str, float]] = set()
        for frame in frames:
            name = frame.get("segment_name")
            speed = frame.get("commanded_speed_m_s")
            if name is None or speed is None:
                continue
            key = (str(name), round(float(speed), 9))
            if key in seen:
                continue
            seen.add(key)
            speeds.append(
                {
                    "name": str(name),
                    "commanded_speed_m_s": float(speed),
                    "first_step": int(frame["step"]),
                }
            )
        realized = [
            float(frame["realized_tcp_speed_m_s"])
            for frame in frames
            if frame.get("realized_tcp_speed_m_s") is not None
        ]
        return {
            "schema_version": "pact_place_v102_frame_telemetry_v1",
            "n_frames": len(frames),
            "n_frames_measured": len(measured),
            "min_clearance_m": (
                float(min(float(frame["min_clearance_m"]) for frame in measured))
                if measured
                else None
            ),
            "min_clearance_floor_m": float(MIN_PENDANT_CLEARANCE_M),
            "per_component_min_clearance_m": per_component,
            "live_pendant_contact_frames": len(contact_frames),
            "first_pendant_contact_step": (
                int(contact_frames[0]["step"]) if contact_frames else None
            ),
            "first_pendant_contact_witness": (
                contact_frames[0].get("contact_pairs") if contact_frames else None
            ),
            "frames_below_clearance_floor": len(below),
            "first_below_floor_step": int(below[0]["step"]) if below else None,
            "first_below_floor_witness": (
                {
                    "step": int(below[0]["step"]),
                    "policy_phase": below[0].get("policy_phase"),
                    "segment_name": below[0].get("segment_name"),
                    "component_clearance_m": below[0].get("component_clearance_m"),
                }
                if below
                else None
            ),
            "segment_speeds": speeds,
            "max_realized_tcp_speed_m_s": float(max(realized)) if realized else None,
            "telemetry_errors": [
                {"step": int(frame["step"]), "error": frame["telemetry_error"]}
                for frame in frames
                if frame.get("telemetry_error")
            ],
        }

    def _endpoint_scalars(self) -> dict[str, Any]:
        end = self._pickup_final_position
        start_z = self._pickup_start_z
        end_z = None if end is None else float(end[2])
        window = list(self._object_position_window)
        settle = None
        if len(window) >= 2:
            settle = float(
                np.linalg.norm(np.asarray(window[-1], dtype=float) - np.asarray(window[0], dtype=float))
            )
        receptacle_distance = None
        try:
            if end is not None:
                manager = self.task.env.object_managers[
                    self.task.env.current_batch_index
                ]
                receptacle = manager.get_object_by_name(
                    self.task.config.task_config.place_receptacle_name
                )
                receptacle_distance = float(
                    np.linalg.norm(
                        np.asarray(end, dtype=float)
                        - np.asarray(receptacle.position, dtype=float)
                    )
                )
        except Exception:
            receptacle_distance = None
        return {
            "object_start_position_m": self._pickup_start_position,
            "object_start_quat_xyzw": self._pickup_start_quat,
            "object_end_position_m": end,
            "object_end_quat_xyzw": self._pickup_final_quat,
            "object_max_z_m": self._pickup_max_z,
            "object_height_above_start_at_terminal_m": (
                None
                if start_z is None or end_z is None
                else float(end_z - start_z)
            ),
            "object_to_receptacle_distance_m": receptacle_distance,
            "settle_window_steps": int(self.SETTLE_WINDOW_STEPS),
            "settle_displacement_m": settle,
            "endpoint_values_emitted_during_compaction": True,
        }

    def get_action(self, observation):
        if self._v9_enabled():
            allowed = {"inbound_vessel", "panel", "outbound_vessel"}
            detected = self._protrusion_detected(allowed)
            if detected is not None and self._phase_for_hazard(detected):
                self._handle_detected_hazard(detected)
        policy_phase = self.get_phase()
        self._pact_place_contact_audit.set_phase(
            self._traversal_phase(policy_phase), policy_phase
        )
        self._pact_place_contact_audit.observe(
            self.task.env, self._pact_place_control_step
        )
        action = super().get_action(observation)
        self._update_manipulation_progress()
        self._update_clutter_stability()
        self._record_place_trajectory_step()
        self._pact_place_control_step += 1
        return action

    def get_info(self):
        from molmo_spaces.tasks.pact_place_contact_audit import (
            place_environment_contact_pairs,
        )

        policy_phase = self.get_phase()
        self._pact_place_contact_audit.set_phase(
            self._traversal_phase(policy_phase), policy_phase
        )
        self._pact_place_contact_audit.observe(
            self.task.env, self._pact_place_control_step
        )
        self._update_manipulation_progress()
        self._update_clutter_stability()
        self._record_place_trajectory_step()
        info = super().get_info()
        terminal_tracking: dict[str, Any] = {
            "sequential_ik_failures": int(self.sequential_ik_failures),
            "action_index": int(self.action_idx),
        }
        if self.action_idx < len(self.action_primitives):
            primitive = self.action_primitives[self.action_idx]
            terminal_tracking["action_primitive"] = type(primitive).__name__
            if isinstance(primitive, TCPMoveSequence) and primitive.move_seg_idx is not None:
                target_pose = primitive.get_current_target_pose()
                gripper_id = self.robot_view.get_gripper_movegroup_ids()[0]
                actual_pose = self.robot_view.get_gripper(gripper_id).leaf_frame_to_world
                terminal_tracking.update(
                    {
                        "move_segment_index": int(primitive.move_seg_idx),
                        "target_position_m": list(map(float, target_pose[:3, 3])),
                        "actual_position_m": list(map(float, actual_pose[:3, 3])),
                        "position_error_m": float(
                            np.linalg.norm(target_pose[:3, 3] - actual_pose[:3, 3])
                        ),
                    }
                )
        try:
            _, terminal_tracking_fields, write_sidecar = self._pact_abort_recorder()
            terminal_tracking.update(terminal_tracking_fields(self))
            write_sidecar(self)
        except Exception:
            pass
        place_metrics = self.task.get_info()[0]
        endpoint_scalars = self._endpoint_scalars()
        info.update(
            {
                "pact_contact_audit": self._pact_place_contact_audit.summary(),
                "grasp_phase_success": bool(
                    self._cup_retrieved_outside_aperture
                ),
                "cup_lifted_one_cm": bool(self._cup_lifted),
                "pickup_start_z_m": self._pickup_start_z,
                "pickup_max_z_m": self._pickup_max_z,
                "pickup_final_position_m": self._pickup_final_position,
                "gripper_width_min_m": self._gripper_width_min,
                "gripper_width_max_m": self._gripper_width_max,
                "place_phase_success": bool(place_metrics["success"]),
                "place_metrics": place_metrics,
                "inbound_deflected": bool(self.inbound_deflected),
                "outbound_deflected": bool(self.outbound_deflected),
                "detected_hazards": list(self._pact_detected_hazards),
                "maneuver_interactions": list(self._pact_maneuver_interactions),
                "active_maneuver": self._active_maneuver_for_phase(),
                "behavior_class": self.behavior_class,
                "grasp_diagnostics": self._pact_place_grasp_diagnostics,
                "bow_diagnostics": self._pact_place_bow_diagnostics,
                "accepted_bow_m": float(
                    self._pact_place_bow_diagnostics["outbound"]["accepted_bow_m"]
                ),
                "planned_bow_m": float(
                    self._pact_place_bow_diagnostics["outbound"]["planned_bow_m"]
                ),
                "bow_fallback_taken": bool(
                    self._pact_place_bow_diagnostics["outbound"]["bow_fallback_taken"]
                ),
                "pendant_bow": {
                    "inbound": dict(
                        self._pact_place_bow_diagnostics.get(
                            "inbound_ceiling_fixture", self._empty_bow_record()
                        )
                    ),
                    "outbound": dict(
                        self._pact_place_bow_diagnostics.get(
                            "outbound_ceiling_fixture", self._empty_bow_record()
                        )
                    ),
                },
                "pendant_v99": {
                    "inbound": dict(
                        (self._pact_place_v99_route or {}).get(
                            "inbound_pendant", self._v99_empty_route_record()
                        )
                    ),
                    "outbound": dict(
                        (self._pact_place_v99_route or {}).get(
                            "outbound_pendant", self._v99_empty_route_record()
                        )
                    ),
                },
                "pendant_v10": {
                    "inbound": dict(
                        (self._pact_place_v10_route or {}).get(
                            "inbound_pendant", self._v99_empty_route_record()
                        )
                    ),
                    "outbound": dict(
                        (self._pact_place_v10_route or {}).get(
                            "outbound_pendant", self._v99_empty_route_record()
                        )
                    ),
                },
                "terminal_tracking": terminal_tracking,
                "terminal_robot_environment_contacts": place_environment_contact_pairs(
                    self.task.env
                ),
                "endpoint_scalars": endpoint_scalars,
                "pendant_frame_telemetry": self._v102_frame_summary(),
                "pact_v104_frame_telemetry": self._v104_frame_summary(),
                "pact_v104_speed_amendment": dict(
                    self._pact_place_v104_speed_amendment or {}
                ),
                "pact_v106_frame_telemetry": self._v106_frame_summary(),
                "pact_v106_speed_amendment": dict(
                    self._pact_place_v106_speed_amendment or {}
                ),
                "pact_v105_frame_telemetry": self._v105_frame_summary(),
                "pact_v105_speed_amendment": dict(
                    self._pact_place_v105_speed_amendment or {}
                ),
                "trajectory": list(self._pact_place_trajectory),
                "clutter_stability_events": list(
                    self._pact_clutter_stability_events
                ),
                "clutter_stability_ok": not bool(
                    self._pact_clutter_stability_events
                ),
            }
        )
        return info


class PactPlaceCorridorPolicyConfig(PickAndPlacePlannerPolicyConfig):
    """Wire the composed expert; rollout-start failures remain terminal."""

    max_retries: int = 0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PactPlaceCorridorPolicy
