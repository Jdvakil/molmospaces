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
from molmo_spaces.configs.policy_configs import ObjectManipulationPlannerPolicyConfig  # noqa: E402


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
