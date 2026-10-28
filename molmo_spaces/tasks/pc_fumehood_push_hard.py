"""pc_fumehood_push_hard: the fume-hood push with the hazards moved into the arm's path.

Same task, scene file, cameras, skin, timing and expert strategy as ``pc_fumehood_push`` (read
that module first); this module only ADDS subclasses, the base classes keep their behaviour.

What is harder (all real, camera-visible, resting geometry; nothing floating):
  * Sash lowered: the sash rail's bottom edge sits a few cm above the highest point the arm
    reaches where it crosses the sash plane (``SASH_OPENING``), instead of ~8-10 cm. The
    horizontal panes (the paper's jamb slabs) are cut to the opening height, so a pane never
    pokes through the lowered rail.
  * Panes closed in on the forearm: inner edges ``JAMB_GAP`` beyond the measured arm envelope
    with no extra margin (base: 1.5 cm margin + 0.5-4.5 cm).
  * Lab items hug the arm: the flank items beside the wrist/forearm at the hood mouth and beside
    the push stroke sit ``FLANK_EXTRA_GAP`` beyond the measured envelope with no margin and a
    narrower jar corridor (base kept >= 3.6 cm), and one more extra item is placed.
  * The expert pushes with the TCP 13 mm higher (fingertips further off the work surface), so
    the DART jitter below does not turn every demo into a bench scrape.

Training-mix knobs (collection configs only; the eval/definition config has them off):
  * DART-style action noise (``robot_config.action_noise_config``): the EXECUTED arm command is
    perturbed (TCP-bounded, through the Jacobian), the RECORDED action label stays the clean
    expert command. The expert is closed loop (absolute joint targets; task-space lines servo the
    measured TCP back onto the line), so it corrects every perturbation on the next step.
  * Contact-tolerant save rule (``ALLOW_LIGHT_CONTACT``): a successful expert episode is saved if
    the arm touched nothing but the jar, OR if its contact with hazards was LIGHT: at most
    ``LIGHT_CONTACT_MAX_FRAMES`` 2-ms control frames with any non-target contact, max penetration
    below ``LIGHT_CONTACT_MAX_PEN_M``, jar upright (the lab items are fixed props and cannot be
    knocked over; the jar is the only free body). Contact frames, bodies and the verdict are
    logged in ``obs_scene['contact_summary']`` and ``scene_params['contact_summary']``.

A learned policy under evaluation is always scored on the task criterion alone (as in the base
suite); its collisions are reported separately by the evaluator.
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.pc_fumehood_push import (
    BENCH_Z,
    CLUTTER_ITEMS,
    FRAME_POST_Y,
    HOOD_BACK_X,
    HOOD_FRONT_X,
    HOOD_INNER_Y,
    JAMB_HALF,
    JAR_R,
    OCC_DX,
    OCC_X0,
    OCC_X1,
    SASH_X,
    _occ_interval,
    PcFumehoodPushPolicy,
    PcFumehoodPushPolicyConfig,
    PcFumehoodPushSampler,
    PcFumehoodPushTask,
)

log = logging.getLogger(__name__)

PC_FUMEHOOD_PUSH_HARD_ENVIRONMENT_VERSION = "pc_fumehood_push_hard_v1"


# ============================================================================================
# Task
# ============================================================================================
class PcFumehoodPushHardTask(PcFumehoodPushTask):
    """Base push task + true-distance clearance diagnostics + the light-contact save rule."""

    allow_light_contact: bool = False
    true_clearance_diag: bool = False
    light_contact_max_frames: int = 150
    light_contact_max_pen_m: float = 0.010
    TRUE_DIST_MAX = 0.10

    # ---------------------------------------------------------------- diagnostics
    def _true_clearance_step(self, data) -> None:
        """Exact min distance (mj_geomDistance, convex geometry) from the robot's collision geoms
        to every hazard group; bounding-sphere prefilter keeps it to a few dozen pairs."""
        model = self.env.current_model
        robot_geoms, groups = self._robot_and_hazard_geoms()
        tc = self._pd.setdefault("true_clear", {})
        tb = self._pd.setdefault("true_clear_body", {})
        rg = np.asarray(robot_geoms, dtype=int)
        rpos = data.geom_xpos[rg]
        rrad = model.geom_rbound[rg]
        fromto = np.zeros(6)
        for key, hz in groups.items():
            best = tc.get(key, 9.0)
            best_body = tb.get(key)
            for hg in hz:
                gap = (np.linalg.norm(rpos - data.geom_xpos[hg], axis=1) - rrad
                       - model.geom_rbound[hg])
                lim = min(best, self.TRUE_DIST_MAX)
                for j in np.nonzero(gap < lim)[0]:
                    g = int(rg[j])
                    d = float(mujoco.mj_geomDistance(model, data, g, int(hg), lim, fromto))
                    # mj_geomDistance (MuJoCo 3.6) sometimes returns exactly 0.0 for separated
                    # box/mesh pairs; real penetration is negative and is caught by the contact
                    # audit anyway, so an exact zero is treated as "no reading".
                    if d == 0.0:
                        continue
                    if d < best:
                        best = d
                        lim = min(best, self.TRUE_DIST_MAX)
                        best_body = (model.geom(g).name
                                     or model.body(int(model.geom_bodyid[g])).name)
            tc[key] = float(best)
            if best_body is not None:
                tb[key] = best_body

    def get_info(self) -> list[dict[str, Any]]:
        out = super().get_info()
        if self.true_clearance_diag and self.episode_step_count % 2 == 0:
            try:
                self._true_clearance_step(self._env.mj_datas[0])
            except Exception as e:  # pragma: no cover - diagnostics never break a rollout
                log.debug(f"[PcPushHard] true clearance diag failed: {e!r}")
        return out

    # ---------------------------------------------------------------- contact summary
    def _expert_audit(self):
        pol = getattr(self, "_registered_policy", None)
        if not isinstance(pol, PcFumehoodPushPolicy):
            return None
        return getattr(pol, "_pact_contact_audit", None)

    def _contact_summary(self) -> dict[str, Any]:
        pd = getattr(self, "_pd", None) or {}
        out: dict[str, Any] = {
            "rule": (f"saved if success and (no hazard contact or <= {self.light_contact_max_frames}"
                     f" 2-ms frames with hazard contact, penetration <= "
                     f"{self.light_contact_max_pen_m:.3f} m, jar upright)"),
            "policy_step_contact_steps": int(pd.get("contact_steps", 0)),
            "policy_step_contacts_by_body": dict(pd.get("contacts_by_body", {})),
            "max_jar_tilt_deg": float(pd.get("max_tilt_deg", 0.0)),
            "true_clearance_m": {k: float(v) for k, v in (pd.get("true_clear") or {}).items()},
            "true_clearance_body": dict(pd.get("true_clear_body") or {}),
        }
        audit = self._expert_audit()
        frames = None
        bodies: dict[str, int] = {}
        pen = 0.0
        if audit is not None:
            try:
                s = audit.summary()
                fw = s.get("frames_with_contact", {})
                pen = max(float(s.get("maximum_penetration_depth_m", {}).get(k, 0.0))
                          for k in ("hazard_bar", "other_environment"))
                cf = s.get("contact_frames") or []
                if cf or not s.get("contact_frame_payload_retained", True):
                    times = set()
                    for fr in cf:
                        cc = fr.get("contact_classes", {})
                        if int(cc.get("hazard_bar", 0)) + int(cc.get("other_environment", 0)) <= 0:
                            continue
                        times.add(fr.get("sim_time_s"))
                        seen = set()
                        for pair in fr.get("pairs", []):
                            r1, r2 = str(pair.get("root1", "")), str(pair.get("root2", ""))
                            env_body = (pair.get("body2") if r1.startswith("robot_0/")
                                        else pair.get("body1")) or ""
                            env_geom = (pair.get("geom2") if r1.startswith("robot_0/")
                                        else pair.get("geom1")) or ""
                            name = env_body if env_body and env_body != "world" else env_geom
                            if name.startswith("cavity_obj_") or name in seen:
                                continue
                            seen.add(name)
                            bodies[name] = bodies.get(name, 0) + 1
                    frames = len(times) if cf else int(fw.get("hazard_bar", 0)) + int(
                        fw.get("other_environment", 0))
                else:
                    frames = 0
            except Exception as e:  # pragma: no cover
                log.debug(f"[PcPushHard] audit summary failed: {e!r}")
                frames = None
        out["hazard_contact_frames_2ms"] = frames
        out["hazard_contact_time_s"] = (None if frames is None else round(0.002 * frames, 4))
        out["hazard_contact_frames_by_body"] = bodies
        out["max_penetration_m"] = float(pen)
        clean = (frames == 0) if frames is not None else (int(pd.get("contact_steps", 0)) == 0)
        light = bool(frames is not None and 0 < frames <= self.light_contact_max_frames
                     and pen <= self.light_contact_max_pen_m)
        upright = float(pd.get("max_tilt_deg", 0.0)) < self.MAX_TILT_DEG
        out["clean"] = bool(clean and int(pd.get("contact_steps", 0)) == 0)
        out["light_contact"] = bool(light and not out["clean"])
        out["save_ok"] = bool(upright and (out["clean"] or out["light_contact"]))
        return out

    def get_obs_scene(self) -> dict[str, Any]:
        scene = super().get_obs_scene()
        cs = self._contact_summary()
        scene["contact_summary"] = cs
        params = scene.get("scene_params")
        if isinstance(params, dict):
            params["contact_summary"] = cs
        return scene

    def judge_success(self) -> bool:
        if not self.allow_light_contact:
            return super().judge_success()
        data = self._env.mj_datas[0]
        _obj, p, tilt = self._jar_state(data)
        ok, dxy = self._verdict(p, tilt)
        expert = isinstance(getattr(self, "_registered_policy", None), PcFumehoodPushPolicy)
        if ok and self.require_clean and expert:
            cs = self._contact_summary()
            if not cs["save_ok"]:
                ok = False
                log.info(f"[PcPushHard] judge: task success but contact too heavy "
                         f"(frames={cs['hazard_contact_frames_2ms']}, pen="
                         f"{cs['max_penetration_m']:.4f}, bodies="
                         f"{cs['hazard_contact_frames_by_body']}, tilt={cs['max_jar_tilt_deg']:.1f})")
        if ok != getattr(self, "_last_judge", None):
            self._last_judge = ok
            log.info(f"[PcPushHard] judge: success={ok} d_goal={dxy:.3f} tilt={tilt:.1f} "
                     f"z={p[2]:.3f}")
        return ok


# ============================================================================================
# Sampler
# ============================================================================================
class PcFumehoodPushHardSampler(PcFumehoodPushSampler):
    """Base layout distribution with the hazards moved onto the arm (see module docstring)."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_FUMEHOOD_PUSH_HARD_ENVIRONMENT_VERSION

    # sash bottom edge above the work surface: the arm's highest crossing of the sash plane is
    # 0.35-0.39 m (conservative AABB measure, base expert), so the rail ends up a few cm above it
    # Measured with exact distances (probe, 16 expert episodes): at 0.30-0.33 m the rail passes
    # 1.7-5 cm from the wrist (fr3_link6/7) where it crosses the sash plane (base: 0.43-0.48 m,
    # 8-10 cm). 30-33 cm is a normal "sash as low as practical" working height.
    SASH_OPENING = (0.300, 0.330)
    # clutter / panes against the measured envelope (base: CLEAR_MARGIN 0.015 on top of the
    # envelope's 1 cm inflation, FLANK_EXTRA_GAP (0, 0.025), JAR_CORRIDOR 0.075, JAMB_GAP
    # (0.005, 0.045), N_EXTRA (2, 4)). Hard: lab items are placed against the RAW (uninflated)
    # measured envelope, which is itself a conservative AABB bound (1-2 cm larger than the
    # meshes), pulled in by 1.5 cm. Measured exact clearances of the expert: nearest lab item
    # 1.6-5 cm, panes 0.9-2.1 cm, fingertips 0.8-1.7 cm off the work surface, zero contacts.
    CLEAR_MARGIN = -0.015
    FLANK_EXTRA_GAP = (0.0, 0.012)
    JAR_CORRIDOR = 0.058
    JAMB_GAP = (-0.005, 0.010)
    N_EXTRA = (3, 5)
    OCC_INFLATE = 0.01                 # what _occ_interval adds; removed again for the hard items
    # items beside the push stroke: the base keeps every item r + 10 cm from the goal centre,
    # which (with the closer flank offset) left the stroke-side slots empty; hard keeps r + 7.5
    # cm (ring radius 5 cm + 2.5 cm) and searches a wider station range with more tries
    GOAL_KEEP = 0.075
    MID_X = (0.74, 0.95)
    FLANK_TRIES = 32

    ALLOW_LIGHT_CONTACT = False
    TRUE_CLEARANCE_DIAG = False
    LIGHT_CONTACT_MAX_FRAMES = 150
    LIGHT_CONTACT_MAX_PEN_M = 0.010

    def _draw_theta(self) -> dict[str, Any]:
        th = super()._draw_theta()
        th["pc_suite"] = "pc_fumehood_push_hard"
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["hard_variant"] = {
            "sash_opening_range_m": list(self.SASH_OPENING),
            "clear_margin_m": float(self.CLEAR_MARGIN),
            "flank_extra_gap_m": list(self.FLANK_EXTRA_GAP),
            "jar_corridor_m": float(self.JAR_CORRIDOR),
            "jamb_gap_m": list(self.JAMB_GAP),
            "n_extra": list(self.N_EXTRA),
            "pane_height_capped_to_opening": True,
        }
        return th

    # ---------------------------------------------------------------- hard clutter layout
    def _occ_raw(self, x: float, z_top: float):
        lo, hi = _occ_interval(x, z_top)
        if lo is None:
            return None, None
        return lo + self.OCC_INFLATE, hi - self.OCC_INFLATE

    def _clear_of_task(self, x, y, r, h, th) -> bool:
        """Base rule against the raw (uninflated) envelope with the hard margin and corridor."""
        lane = th["lane_y"]
        m = self.CLEAR_MARGIN
        yr = y - lane
        for xb0 in np.arange(OCC_X0, OCC_X1 - 1e-9, OCC_DX):
            if xb0 + OCC_DX < x - r - m or xb0 > x + r + m:
                continue
            lo, hi = self._occ_raw(xb0 + 0.5 * OCC_DX, BENCH_Z + h + m)
            if lo is not None and (yr + r + m > lo) and (yr - r - m < hi):
                return False
        x0 = th["jar_start_xy"][0] - JAR_R - 0.03
        x1 = th["goal_xy"][0] + JAR_R + 0.06
        if x0 - r <= x <= x1 + r and abs(yr) - r < self.JAR_CORRIDOR:
            return False
        gx, gy = th["goal_xy"]
        if np.hypot(x - gx, y - gy) < r + self.GOAL_KEEP:
            return False
        return True

    def _draw_clutter(self, th: dict[str, Any]) -> list[dict[str, Any]]:
        """Base layout (same slots, pools, back item, extras); flank items hug the raw envelope."""
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
            # closest legal offset: beyond the (pulled-in) arm envelope AND beyond the jar
            # corridor (the corridor never takes the arm-envelope margin)
            m = self.CLEAR_MARGIN
            need = self.JAR_CORRIDOR
            for xb0 in np.arange(OCC_X0, OCC_X1 - 1e-9, OCC_DX):
                if xb0 + OCC_DX < x - r - m or xb0 > x + r + m:
                    continue
                lo, hi = self._occ_raw(xb0 + 0.5 * OCC_DX, BENCH_Z + h + m)
                if lo is None:
                    continue
                need = max(need, (hi if sgn > 0 else -lo) + m)
            return lane + sgn * (need + r + rng.uniform(*self.FLANK_EXTRA_GAP) + 0.002)

        tall_pool = ["pact_clutter_solvent_bottle", "pact_clutter_media_bottle",
                     "pact_clutter_grad_cylinder", "pact_clutter_wash_bottle"]
        mid_pool = ["pact_clutter_amber_bottle", "pact_clutter_flask", "pact_clutter_beaker"]
        wide_pool = ["pact_clutter_tube_rack", "pact_clutter_hotplate", "pact_clutter_sharps_bin",
                     "pact_clutter_glove_box", "pact_clutter_tip_box"]
        slots = [
            ("front_a", (0.63, 0.74), side, tall_pool),
            ("front_b", (0.63, 0.74), -side, mid_pool + tall_pool),
            ("mid_a", self.MID_X, side, wide_pool + mid_pool),
            ("mid_b", self.MID_X, -side, wide_pool + mid_pool),
        ]
        for slot, xr, sgn, pool in slots:
            cands = [n for n in pool if n not in used]
            rng.shuffle(cands)
            done = False
            for name in cands:
                r, h, _ = CLUTTER_ITEMS[name]
                for _ in range(self.FLANK_TRIES):
                    x = float(rng.uniform(*xr))
                    y = float(flank_y(x, r, h, sgn))
                    if (self._inside_hood(x, y, r) and self._clear_of_task(x, y, r, h, th)
                            and self._clear_of_placed(x, y, r, placed)):
                        put(name, x, y, slot)
                        done = True
                        break
                if done:
                    break
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

    def _apply_theta(self, env, th: dict[str, Any]) -> None:
        super()._apply_theta(env, th)
        # horizontal panes end flush under the lowered sash rail (never through it)
        m = env.current_model
        hz = float(min(JAMB_HALF[2], 0.5 * th["sash_opening_m"]))
        yl, yr = th["jamb_inner_y"]
        hl = 0.5 * (FRAME_POST_Y - yl)
        hr = 0.5 * (FRAME_POST_Y + yr)
        for gname, half in (("jamb_l_g", hl), ("jamb_r_g", hr)):
            g = m.geom(gname).id
            size = np.array([JAMB_HALF[0], half, hz], dtype=float)
            m.geom_size[g] = size
            m.geom_rbound[g] = float(np.linalg.norm(size))
            m.geom_aabb[g, 3:] = size
        jl = [SASH_X, yl + hl, BENCH_Z + hz]
        jr = [SASH_X, yr - hr, BENCH_Z + hz]
        self._mocap_set(env, "jamb_l", jl)
        self._mocap_set(env, "jamb_r", jr)
        th["jamb_centers"] = [list(map(float, jl)), list(map(float, jr))]
        th["pane_height_m"] = float(2 * hz)
        mujoco.mj_forward(m, env.current_data)

    def _sample_task(self, env: CPUMujocoEnv):
        th = self._draw_theta()
        self._theta = th
        self._apply_theta(env, th)
        self._settle_jar(env, th)
        self.candidate_objects = self._get_scene_objects(env)
        self._place_robot(env, th)
        self.setup_cameras(env)
        task = PcFumehoodPushHardTask(env, self.config)
        task.scene_params = {k: v for k, v in th.items() if not k.startswith("_")}
        task.goal_tol = float(self.GOAL_TOL)
        task.clearance_diag = bool(self.CLEARANCE_DIAG)
        task.require_clean = bool(self.REQUIRE_CLEAN)
        task.allow_light_contact = bool(self.ALLOW_LIGHT_CONTACT)
        task.true_clearance_diag = bool(self.TRUE_CLEARANCE_DIAG)
        task.light_contact_max_frames = int(self.LIGHT_CONTACT_MAX_FRAMES)
        task.light_contact_max_pen_m = float(self.LIGHT_CONTACT_MAX_PEN_M)
        log.info(
            f"[PcPushHard] cell={th['cell']} lane_y={th['lane_y']:+.3f} "
            f"jar=({th['jar_start_xy'][0]:.3f},{th['jar_start_xy'][1]:+.3f}) "
            f"goal=({th['goal_xy'][0]:.3f},{th['goal_xy'][1]:+.3f}) "
            f"push={th['push_distance_m']:.3f} sash_open={th['sash_opening_m']:.3f} "
            f"jambs=({th['jamb_inner_y'][0]:+.3f},{th['jamb_inner_y'][1]:+.3f}) "
            f"clutter={[c['body'].replace('pact_clutter_', '') + '@' + c['slot'] for c in th['clutter']]}"
        )
        return task


class PcFumehoodPushHardCheckSampler(PcFumehoodPushHardSampler):
    """Preflight: the collection rule plus per-step clearance diagnostics (AABB + exact)."""

    CLEARANCE_DIAG = True
    TRUE_CLEARANCE_DIAG = True
    REQUIRE_CLEAN = True
    ALLOW_LIGHT_CONTACT = True


class PcFumehoodPushHardCollectSampler(PcFumehoodPushHardSampler):
    """Collection + eval/definition sampler. With the SCRIPTED EXPERT registered, an episode
    counts as a success (and is saved) only if the task succeeded and hazard contact was none or
    light (see module docstring). With a learned policy registered, success is the task
    criterion alone."""

    REQUIRE_CLEAN = True
    ALLOW_LIGHT_CONTACT = True
    TRUE_CLEARANCE_DIAG = True


# ============================================================================================
# Expert
# ============================================================================================
class PcFumehoodPushHardPolicy(PcFumehoodPushPolicy):
    """Same scripted pusher; TCP 13 mm higher during the stroke (base 0.034 m). With the DART
    jitter at 0.040 m the fingertips grazed the work surface in 4 of the first 8 collected demos;
    at 0.047 m: 0 bench contacts in 32 probe demos, all 32 on goal. The finger face still meets the
    jar ~3.5 cm above the surface, far below its tipping height."""

    PUSH_TCP_H = 0.047


class PcFumehoodPushHardPolicyConfig(PcFumehoodPushPolicyConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PcFumehoodPushHardPolicy


__all__ = [
    "PC_FUMEHOOD_PUSH_HARD_ENVIRONMENT_VERSION",
    "PcFumehoodPushHardTask",
    "PcFumehoodPushHardSampler",
    "PcFumehoodPushHardCheckSampler",
    "PcFumehoodPushHardCollectSampler",
    "PcFumehoodPushHardPolicy",
    "PcFumehoodPushHardPolicyConfig",
]
