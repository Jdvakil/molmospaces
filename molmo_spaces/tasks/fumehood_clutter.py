"""Cluttered fume-hood PICK-AND-PLACE.

The obstacle line (``enclosure_reach.py``) put a single hazard bar next to the approach corridor.
That loads a handful of wrist and hand sensors for a couple of seconds near the grasp. This module
loads the *whole arm* for the *whole episode*, and then makes the arm travel: the robot retrieves an
object from inside the fume hood and sets it down on a rolling cart on its other side, sweeping the
full width of a cluttered lab bay on the way.

Three things are new relative to ``ObstacleFumehoodPickSampler``:

1. **A cluttered bay.** ``fumehood_clutter.xml`` adds a three-tier shelving unit on the robot's
   left, a floor cabinet behind it, and the cart on its right. Those are static, so the skin has a
   structured return field on three sides at every arm height. On top of them sit 16 mocap clutter
   items that this sampler re-poses every episode.

2. **No floating geometry.** Every clutter item's z is pinned to ``surface height + its own
   half-height``, and candidate poses are rejected until they clear both the static furniture and
   the arm's keep-out volume. A SPAD reads real surfaces in a real lab, so the sim must not invent
   depth that could not exist.

3. **A place leg.** ``ClutteredPickPlacePolicy`` appends retract / swing / descend / release /
   retreat to the inherited pick, and ``ClutteredPickPlaceTask`` scores the episode on where the
   object ends up rather than on lift height alone.

The clutter is deliberately placed **outside** the demonstrated path, never on it. That is the
point of the experiment: the expert never touches it, so every demonstration is clean, but the
skin is loaded the entire time. A policy that ignores the skin drifts off the demonstrated
corridor and hits something -- and ``PickTask._accumulate_obstacle_diag`` already counts exactly
that, because each clutter item is its own body.
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
from molmo_spaces.tasks.enclosure_reach import (
    SHELF_TOP_Z,
    TUBE_X0,
    EnclosureReachTask,
    InvisibleObstacleFumehoodPickSampler,
    ObstacleAwarePickPlannerPolicy,
)
from molmo_spaces.env.data_views import MlSpacesObject

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# Scene constants. These MUST agree with fumehood_clutter.xml -- the XML owns the geometry, this
# table is how Python knows where the surfaces are and how big each item is.
# --------------------------------------------------------------------------------------------

BENCH_TOP_Z = SHELF_TOP_Z          # 0.72, the fume hood's work surface
CART_TOP_Z = 0.62                  # top face of cart_top
# Centre of the cart's top face -- the place destination. Reach-constrained, see the comment on
# the cart in fumehood_clutter.xml: from the shoulder at (0.08, 0, 0.35) against a 0.855 m reach,
# this keeps the whole place sequence inside 0.75 m. The first draft at (0.30, -0.62) put the
# over-cart waypoint 0.925 m out and every episode failed IK on the transport leg.
CART_XY = (0.32, -0.56)
CABINET_TOP_Z = 0.86               # top face of cabinet_top
SHELF_BOARD_TOPS = (0.40, 0.80, 1.20)

# name -> (half_x, half_y, half_z, surface family). Half-extents mirror the geom sizes in the XML;
# for cylinders the radius is used for both half_x and half_y.
CLUTTER_ITEMS: dict[str, tuple[float, float, float, str]] = {
    "clut_sh0": (0.045, 0.045, 0.10, "shelf"),
    "clut_sh1": (0.045, 0.045, 0.10, "shelf"),
    "clut_sh2": (0.045, 0.045, 0.10, "shelf"),
    "clut_sh3": (0.045, 0.045, 0.10, "shelf"),
    "clut_sh4": (0.045, 0.045, 0.10, "shelf"),
    "clut_hd0": (0.040, 0.040, 0.085, "hood"),
    "clut_hd1": (0.040, 0.040, 0.085, "hood"),
    "clut_hd2": (0.040, 0.040, 0.085, "hood"),
    "clut_hd3": (0.040, 0.040, 0.085, "hood"),
    "clut_cb0": (0.080, 0.060, 0.10, "cabinet"),
    "clut_cb1": (0.080, 0.060, 0.10, "cabinet"),
    "clut_cb2": (0.080, 0.060, 0.10, "cabinet"),
    "clut_fl0": (0.110, 0.110, 0.28, "floor"),
    "clut_fl1": (0.110, 0.110, 0.28, "floor"),
    "clut_fl2": (0.090, 0.090, 0.65, "floor"),   # gas cylinder, top at 1.30
    "clut_fl3": (0.090, 0.090, 0.65, "floor"),
}

# Static furniture as (centre, half) AABBs. Used only to keep clutter from being posed inside a
# shelf upright or through the bench; the skin sees these geoms whether or not they are listed.
_FURNITURE_AABBS: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
    ((0.95, 0.00, 0.70), (0.45, 0.60, 0.02)),    # bench_top
    ((0.95, 0.00, 0.35), (0.42, 0.57, 0.33)),    # bench_body
    ((0.02, 0.48, 0.68), (0.02, 0.02, 0.68)),    # shelf uprights
    ((0.02, 0.76, 0.68), (0.02, 0.02, 0.68)),
    ((0.43, 0.48, 0.68), (0.02, 0.02, 0.68)),
    ((0.43, 0.76, 0.68), (0.02, 0.02, 0.68)),
    ((-0.55, 0.00, 0.42), (0.20, 0.35, 0.42)),   # cabinet_body
    ((0.32, -0.56, 0.44), (0.18, 0.16, 0.44)),   # cart envelope (top, shelf, legs)
]

# --------------------------------------------------------------------------------------------
# The arm's keep-out volume: where clutter may NEVER go, so every demonstration stays clean.
# --------------------------------------------------------------------------------------------

# Vertical cylinder around the pedestal axis covering the arm's rotational sweep. The pedestal is
# 0.4 m square, so its own corner is already at 0.28; 0.26 plus the gap below keeps clutter just
# outside the swept envelope of link1/link2 without pushing it out of sensor range.
KEEPOUT_BASE_R = 0.26
KEEPOUT_BASE_Z = (0.30, 1.10)
# Reach corridor from the base into the fume hood. Half-width 0.22 is the hand's insertion
# envelope (DIST_W = 0.18) plus a margin -- the first draft used 0.34, which was wide enough to
# reject every in-hood clutter pose and left that whole family permanently parked.
KEEPOUT_REACH = ((-0.05, 0.95), (-0.22, 0.22), (0.60, 1.15))     # (x range, y range, z range)
# Transport corridor: the lateral sweep from the hood mouth across to the cart. It stops at
# x = 0.52 because ClutteredPickPlacePolicy retracts to x = 0.44 *before* it swings -- extending
# this to the back of the hood would forbid clutter on the whole right-hand side of the bench for
# a sweep that never happens there.
KEEPOUT_TRANSPORT = ((0.05, 0.52), (-0.80, 0.10), (0.50, 1.15))

# Surface-to-surface gap enforced between clutter and every keep-out volume.
CLUT_MIN_GAP = 0.06

# Clearance enforced against the NOMINAL TCP PATH itself, on top of the box keep-outs above.
# The boxes are a coarse over-approximation of where the arm goes; they are cheap but they leave a
# thin tail of episodes where clutter lands millimetres off the real path (measured: 0 outright
# collisions in 1600 sampled episodes, but a worst case of 0.001 m). Inflating CLUT_MIN_GAP to fix
# that pushes *all* clutter further away and costs proximity signal everywhere. Checking the actual
# polyline instead is targeted: it only moves the items that are genuinely in the way.
# 0.10 is the open gripper's lateral half-extent (ObstacleAwarePickPlannerPolicy.GRIP_HALF); the
# rest absorbs servo error and the lateral bow of a hazard-bar deflection.
PATH_CLEAR = 0.10 + 0.07


def _aabb_overlap(c0, h0, c1, h1, pad: float = 0.0) -> bool:
    """True when two axis-aligned boxes overlap, after growing the first by ``pad``."""
    return all(abs(c0[i] - c1[i]) < (h0[i] + pad) + h1[i] for i in range(3))


def _box_to_points_dist(centre, half, pts: np.ndarray) -> float:
    """Smallest surface distance from an axis-aligned box to any of ``pts``."""
    d = np.abs(pts - np.asarray(centre, dtype=float)) - np.asarray(half, dtype=float)
    return float(np.linalg.norm(np.maximum(d, 0.0), axis=1).min())


def nominal_tcp_path(obj_xyz, cart_xy, cart_top_z, retract_x, transport_z, n: int = 12) -> np.ndarray:
    """The waypoints the expert's TCP visits, densified along each straight segment.

    pregrasp -> grasp -> lift -> retract out of the hood -> swing over the cart -> release.
    Built from the same numbers ClutteredPickPlacePolicy uses, so the sampler can keep clutter off
    the path it is about to demonstrate. Approximate by construction: the real grasp height comes
    from the object's grasp file and a hazard-bar deflection bows the approach sideways, which is
    what PATH_CLEAR's margin is for.
    """
    ox, oy, oz = (float(v) for v in obj_xyz)
    grasp_z = oz + 0.09          # the TCP rides ~9 cm above the support surface at the grasp
    lift_z = grasp_z + 0.09
    wps = [
        np.array([TUBE_X0 - 0.10, oy, grasp_z]),
        np.array([ox, oy, grasp_z]),
        np.array([ox, oy, lift_z]),
        np.array([retract_x, oy, lift_z]),
        np.array([cart_xy[0], cart_xy[1], transport_z]),
        np.array([cart_xy[0], cart_xy[1], cart_top_z + 0.09]),
    ]
    segs = [a + t * (b - a) for a, b in zip(wps[:-1], wps[1:]) for t in np.linspace(0, 1, n)]
    return np.asarray(segs)


class ClutteredFumehoodPickPlaceSampler(InvisibleObstacleFumehoodPickSampler):
    """Cluttered-bay pick-and-place.

    Inherits the whole obstacle line, so ``OBSTACLE_P`` (hazard bar present) and ``INVIS_P`` (bar
    hidden from RGB but not from the skin) still work exactly as before and can be turned off by
    subclassing. Clutter is additive: the bar is a close-range hazard by the gripper, the clutter
    is a mid-range hazard around the whole arm.
    """

    # A bar on 60% of episodes rather than the parent's 75%: with clutter in the scene the skin is
    # never idle, so the bar no longer has to carry the proximity signal on its own.
    OBSTACLE_P = 0.60
    # How many of the 16 clutter items are placed each episode. The rest are parked off-scene.
    N_CLUTTER = (9, 15)
    # Attempts per item before giving up and parking it -- rejection sampling against the keep-out
    # volumes can legitimately fail when a narrow episode leaves no room.
    PLACE_ATTEMPTS = 40
    # Object must end up within this of the cart-top centre to count as placed.
    PLACE_TOL = 0.12

    # ---------------------------------------------------------------- clutter placement

    def _draw_clutter_pose(self, name: str, th: dict[str, Any]) -> list[float] | None:
        """One candidate world pose for ``name``, or None if no legal pose was found.

        z is never drawn: it is fixed by the surface the item stands on plus the item's own
        half-height, which is what makes the placement physical rather than decorative.
        """
        hx, hy, hz, family = CLUTTER_ITEMS[name]
        bx, by, _ = self._cur_base_xyz

        for _ in range(self.PLACE_ATTEMPTS):
            if family == "shelf":
                z_top = float(np.random.choice(SHELF_BOARD_TOPS))
                x = float(np.random.uniform(0.06, 0.39))
                y = float(np.random.uniform(0.51, 0.73))
            elif family == "cabinet":
                z_top = CABINET_TOP_Z
                x = float(np.random.uniform(-0.70, -0.40))
                y = float(np.random.uniform(-0.27, 0.27))
            elif family == "hood":
                z_top = BENCH_TOP_Z
                # Half flank the approach corridor, which is where they do the most good: the hand
                # and wrist sensors pass within ~0.1 m of them on the way to the grasp. That band
                # only exists when the jambs are narrow enough to leave room, and ap_w is drawn
                # per episode, so the rest go deep behind the pick target instead.
                # 0.33 is the floor: the reach corridor is +-0.22 and the gap plus the item radius
                # add another 0.09, so anything nearer the centreline is rejected anyway.
                y_lo = max(float(th.get("ap_w", 0.6)) / 2 + 0.05, 0.33)
                if np.random.random() < 0.5 and y_lo < 0.40:
                    x = float(np.random.uniform(0.64, 0.92))
                    y = float(np.random.choice([-1.0, 1.0]) * np.random.uniform(y_lo, 0.42))
                else:
                    x = float(np.random.uniform(1.05, 1.26))
                    y = float(np.random.uniform(-0.34, 0.34))
            else:  # floor -- solvent drums standing around the pedestal
                z_top = 0.0
                r = float(np.random.uniform(0.36, 0.62))
                # Sectors that avoid the forward reach and the cart approach.
                lo, hi = ((50.0, 165.0) if np.random.random() < 0.55 else (195.0, 290.0))
                a = np.deg2rad(float(np.random.uniform(lo, hi)))
                x = float(bx + r * np.cos(a))
                y = float(by + r * np.sin(a))

            pos = [x, y, z_top + hz]
            if self._clutter_pose_ok(pos, (hx, hy, hz), th):
                return pos
        return None

    def _clutter_pose_ok(self, pos, half, th: dict[str, Any]) -> bool:
        """Reject anything that intersects the furniture, the arm's keep-out volume, the hazard
        bar, the pick target, or another clutter item already placed this episode."""
        bx, by, _ = self._cur_base_xyz
        x, y, z = pos
        hx, hy, hz = half

        # 1. arm sweep cylinder around the pedestal
        if (z + hz) > KEEPOUT_BASE_Z[0] and (z - hz) < KEEPOUT_BASE_Z[1]:
            r_xy = float(np.hypot(x - bx, y - by)) - max(hx, hy)
            if r_xy < KEEPOUT_BASE_R + CLUT_MIN_GAP:
                return False

        # 2. reach + transport corridors
        for (xr, yr, zr) in (KEEPOUT_REACH, KEEPOUT_TRANSPORT):
            c = ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2)
            h = ((xr[1] - xr[0]) / 2, (yr[1] - yr[0]) / 2, (zr[1] - zr[0]) / 2)
            if _aabb_overlap(c, h, pos, half, pad=CLUT_MIN_GAP):
                return False

        # 3. static furniture (bench, shelf uprights, cabinet, cart). Every item rests exactly on
        # one of these surfaces, so its box and the surface's box share a face and the overlap
        # test lands on a floating-point tie -- which silently rejected the entire in-hood family
        # on the first run. Lift the item's underside by 4 mm for this test only: real
        # intersections are still caught, contact with the surface it stands on is not.
        rest_c = (x, y, z + 0.004)
        rest_h = (hx, hy, max(hz - 0.004, 0.001))
        for c, h in _FURNITURE_AABBS:
            if _aabb_overlap(c, h, rest_c, rest_h):
                return False

        # 4. this episode's hazard bar, and the object we are about to pick
        if th.get("protrusion_present") and "protr_center" in th:
            if _aabb_overlap(th["protr_center"], th["protr_half"], pos, half, pad=0.04):
                return False
        obj = th.get("_obj_rest_xyz")
        if obj is not None and _aabb_overlap(obj, (0.09, 0.09, 0.12), pos, half, pad=0.03):
            return False

        # 5. clutter already placed this episode
        for c, h in th.get("_clutter_placed", []):
            if _aabb_overlap(c, h, pos, half, pad=0.02):
                return False

        # 6. the nominal TCP path itself. The box keep-outs above are a coarse envelope; this is
        # the actual polyline the expert will fly, and it is what closes the thin tail of episodes
        # where clutter lands a millimetre off the demonstrated trajectory.
        path = th.get("_tcp_path")
        if path is not None and _box_to_points_dist(pos, half, path) < PATH_CLEAR:
            return False
        return True

    def _place_clutter(self, env, th: dict[str, Any]) -> None:
        """Park every clutter item, then pose the subset drawn for this episode."""
        for i, name in enumerate(CLUTTER_ITEMS):
            self._mocap_set(env, name, [0.0, 3.0 + 0.2 * i, -2.0])

        # Build this episode's nominal TCP path once, so _clutter_pose_ok can keep clutter off it.
        obj = th.get("_obj_rest_xyz")
        if obj is not None:
            from molmo_spaces.tasks.fumehood_clutter import ClutteredPickPlacePolicy as _P

            th["_tcp_path"] = nominal_tcp_path(
                obj, CART_XY, CART_TOP_Z, _P.RETRACT_X, _P.TRANSPORT_Z
            )
        else:
            th["_tcp_path"] = None

        n = int(np.random.randint(self.N_CLUTTER[0], self.N_CLUTTER[1] + 1))
        chosen = list(np.random.permutation(list(CLUTTER_ITEMS))[:n])
        th["_clutter_placed"] = []
        boxes: list[tuple[list[float], list[float]]] = []

        for name in chosen:
            hx, hy, hz, _ = CLUTTER_ITEMS[name]
            pos = self._draw_clutter_pose(name, th)
            if pos is None:
                continue
            self._mocap_set(env, name, pos)
            half = [hx, hy, hz]
            th["_clutter_placed"].append((pos, half))
            boxes.append((pos, half))

        th["clutter_boxes"] = [[list(map(float, c)), list(map(float, h))] for c, h in boxes]
        th["n_clutter"] = len(boxes)

    # ---------------------------------------------------------------- theta / apply

    def _draw_theta(self):
        th = super()._draw_theta()
        # The place destination is fixed furniture, but log it in scene_params so the expert and
        # the success test read one number rather than each hardcoding the cart.
        th["place_target"] = [CART_XY[0], CART_XY[1], CART_TOP_Z]
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)   # hood, jambs, bar, invisibility, lighting
        # Where the object will come to rest, so clutter rejection can avoid landing on top of it.
        # _obj_rest reads self._theta, but EnclosureReachSampler._sample_task only assigns that
        # AFTER its _apply_theta loop finishes -- so calling it here plainly would silently use the
        # PREVIOUS episode's theta. Bind this episode's theta for the call and put it back.
        prev_theta = getattr(self, "_theta", None)
        self._theta = th
        try:
            th["_obj_rest_xyz"] = list(map(float, self._obj_rest()))
        except Exception:
            th["_obj_rest_xyz"] = None
        finally:
            self._theta = prev_theta
        self._place_clutter(env, th)
        # Extend the live obstacle list the expert's speed law reads, so clutter counts as a
        # skin-sensable surface exactly like the hood shell and the hazard bar do.
        # The parent's _stash_aabbs OVERWRITES obstacle_aabbs on every call, so reading it back
        # here always yields the parent's list alone -- clutter cannot accumulate across the
        # retry loop in _sample_task.
        existing = th.get("obstacle_aabbs", [])
        th["obstacle_aabbs"] = existing + th.get("clutter_boxes", [])
        # The parent already forwarded, but clutter was posed after that, so pose again.
        mujoco.mj_forward(env.current_model, env.current_data)

    def _sample_task(self, env):
        task = super()._sample_task(env)
        task.__class__ = ClutteredPickPlaceTask
        th = self._theta
        # Score the episode against the cart, not against lift height.
        task.config.task_config.pickup_obj_goal_pose = [
            CART_XY[0], CART_XY[1], CART_TOP_Z + 0.05, 1.0, 0.0, 0.0, 0.0,
        ]
        task.place_tol = self.PLACE_TOL
        # scene_params is written into the h5 as a zero-padded JSON field, so drop the
        # underscore-prefixed scratch keys used only during placement. clutter_boxes keeps the
        # same information in the form downstream analysis wants.
        task.scene_params = {k: v for k, v in th.items() if not k.startswith("_")}
        log.info(
            f"[ClutterPnP] n_clutter={th.get('n_clutter')} bar={th.get('protrusion_present')} "
            f"invis={th.get('bar_invisible')} place=({CART_XY[0]:.2f},{CART_XY[1]:.2f},"
            f"{CART_TOP_Z:.2f})"
        )
        return task


class ClutteredFumehoodPickPlaceCheckSampler(ClutteredFumehoodPickPlaceSampler):
    """Preflight: maximum clutter on every episode, bar always present and always invisible.

    Use this to eyeball one or two episodes before committing to a collection run. What to look
    for: the exo video shows a shelving unit, a cabinet and a cart with items on them; no hazard
    bar is visible anywhere; the log carries ``[ClutterPnP]``, ``[InvisBar]`` and ``[ClutterPnP]
    PLACE``; and the episode ends with the object sitting on the cart.
    """

    OBSTACLE_P = 1.0
    INVIS_P = 1.0
    N_CLUTTER = (14, 16)


class ClutteredPickPlaceTask(EnclosureReachTask):
    """Success = the object came to rest on the cart, out of the gripper.

    The inherited criterion (lift height above the start pose, with the object touching only
    robot geoms) scores a pick. Here the episode is not over until the object has been carried
    across the bay and released, so success is measured at the destination instead.
    """

    def judge_success(self) -> bool:
        try:
            goal = np.asarray(self.config.task_config.pickup_obj_goal_pose[:3], dtype=float)
            tol = float(getattr(self, "place_tol", 0.12))
            data = self._env.mj_datas[0]
            obj = MlSpacesObject(data=data, object_name=self.config.task_config.pickup_obj_name)
            p = np.asarray(obj.position, dtype=float)
            # Horizontal tolerance is the real test; vertically the object only has to be resting
            # near the cart top rather than still hanging from the gripper.
            dxy = float(np.linalg.norm(p[:2] - goal[:2]))
            dz = abs(float(p[2]) - CART_TOP_Z)
            ok = dxy <= tol and dz < 0.18
            # The rollout loop polls this EVERY step, so logging unconditionally buries the log
            # under hundreds of identical lines. Report only when the verdict flips.
            if ok != getattr(self, "_last_place_verdict", None):
                self._last_place_verdict = ok
                log.info(
                    f"[ClutterPnP] placed={ok}: object at "
                    f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), {dxy:.3f} m from the cart centre "
                    f"(tol {tol:.2f}), {dz:.3f} m off the cart top"
                )
            return ok
        except Exception as e:  # pragma: no cover - never break a rollout on a metric
            # WARNING, not debug: this except used to swallow real errors into a bare False, which
            # is indistinguishable from an honest miss and hid the cause of a failing preflight.
            log.warning(f"[ClutterPnP] judge_success raised, scoring as failure: {e!r}")
            return False


class ClutteredPickPlacePolicy(ObstacleAwarePickPlannerPolicy):
    """The inherited pick, plus a place.

    The parent produces ``[open, approach(+deflect), close, lift]``. This appends, in order:

      retract   straight back out of the hood before any lateral motion, so the object and the
                gripper clear the jambs instead of being swung into them   [phase: lift]
      swing     across the bay to above the cart at TRANSPORT_Z -- the long leg, with the skin
                loaded the whole way                                       [phase: preplace]
      descend   onto the cart top                                          [phase: place]
      release   open the gripper
      retreat   lift clear so the object is unambiguously free of the hand [phase: retreat]

    The bracketed names are the only ones ``PolicyPhaseSensor`` recognises; see the comment on the
    segment list below.
    """

    RETRACT_X = TUBE_X0 - 0.14   # clear of the aperture plane before swinging
    # Height of the lateral sweep. This is a REACH limit, not a clearance limit: the transport
    # corridor is a clutter keep-out volume, so there is nothing up there to clear, but carrying
    # the object high and wide at the same time puts the wrist outside the FR3's 0.855 m envelope.
    # The first draft used 1.00, which is 0.925 m from the shoulder over the cart, and every
    # episode died with "IK failed, holding current position" partway through the swing. 0.78
    # keeps the over-cart pose at 0.746 m while still clearing the cart top by 0.16 m.
    TRANSPORT_Z = 0.78
    # Kept short for the same reason: retreating 0.16 m above the release pose reaches 0.800 m,
    # close enough to the 0.855 m limit that the IK solver starts to struggle. 0.12 sits at 0.775
    # and is still plainly clear of the object.
    RETREAT_UP = 0.12

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        prims = super()._compute_trajectory()
        th = getattr(self.task, "scene_params", {}) or {}
        target = th.get("place_target")
        if target is None or len(prims) < 4:
            return prims

        lift_seq = prims[3]
        grasp_pose = prims[1]._move_segments[-1].end_pose
        lift_pose = lift_seq._move_segments[-1].end_pose
        R = grasp_pose[:3, :3]
        lift_p = lift_pose[:3, 3].copy()

        # Hold the gripper at the same height above the destination surface as it held above the
        # bench when it grasped: the grasp geometry is known to work, so reuse it rather than
        # inventing a new approach height for the cart.
        #
        # CLAMPED, because the grasp pose cannot be trusted unconditionally. _compute_trajectory is
        # re-run on every grasp retry, and a degenerate replan can hand back a grasp at z = 0.148 --
        # observed in collection, where it produced "lift z=0.228 -> cart (0.32, -0.56, 0.048)",
        # i.e. a release point 0.57 m BELOW the cart top. The clamp keeps the release in a band
        # just above the cart whatever the grasp pose says.
        raw_place_z = float(target[2]) + (float(grasp_pose[2, 3]) - BENCH_TOP_Z)
        place_z = float(np.clip(raw_place_z, float(target[2]) + 0.04, float(target[2]) + 0.16))
        if abs(place_z - raw_place_z) > 1e-6:
            log.warning(
                f"[ClutterPnP] implausible grasp height z={float(grasp_pose[2, 3]):.3f} would put "
                f"the release at z={raw_place_z:.3f}; clamped to {place_z:.3f}"
            )

        p_retract = np.array([self.RETRACT_X, lift_p[1], lift_p[2]])
        p_over = np.array([float(target[0]), float(target[1]), self.TRANSPORT_Z])
        p_place = np.array([float(target[0]), float(target[1]), place_z])
        p_up = p_place + np.array([0.0, 0.0, self.RETREAT_UP])

        robot_view = self.task.env.current_robot.robot_view
        fast = self.policy_config.speed_fast
        slow = self.policy_config.speed_slow

        def seq(segs, holding):
            return TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                is_holding_object=holding,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                move_segments=segs,
            )

        # Segment names must come from BaseObjectManipulationPlannerPolicy.get_all_phases():
        # gripper-open / pregrasp / grasp / gripper-close / lift / preplace / place / retreat /
        # go_home. PolicyPhaseSensor looks the name up in that dict and writes -1 into
        # obs/policy_phase for anything it does not recognise, so inventing names like "retract"
        # or "transport" silently destroys the phase channel that every downstream analysis reads.
        # Extraction from the hood is still "lift"; everything up to the descent is "preplace".
        transport = seq(
            [
                TCPMoveSegment(name="lift", start_pose=lift_pose,
                               end_pose=self._pose(p_retract, R), speed=slow),
                TCPMoveSegment(name="preplace", start_pose=self._pose(p_retract, R),
                               end_pose=self._pose(p_over, R), speed=fast),
                TCPMoveSegment(name="place", start_pose=self._pose(p_over, R),
                               end_pose=self._pose(p_place, R), speed=slow),
            ],
            holding=True,
        )
        release = GripperAction(robot_view, True, self.policy_config.gripper_close_duration)
        retreat = seq(
            [
                TCPMoveSegment(name="retreat", start_pose=self._pose(p_place, R),
                               end_pose=self._pose(p_up, R), speed=slow),
            ],
            holding=False,
        )

        log.info(
            f"[ClutterPnP] PLACE: lift z={lift_p[2]:.3f} -> transport z={self.TRANSPORT_Z:.2f} "
            f"-> cart ({p_place[0]:.2f}, {p_place[1]:.2f}, {p_place[2]:.3f})"
        )
        return [*prims, transport, release, retreat]


class ClutteredPickPlacePolicyConfig(PickPlannerPolicyConfig):
    """Wires ClutteredPickPlacePolicy as the rollout policy."""

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = ClutteredPickPlacePolicy
