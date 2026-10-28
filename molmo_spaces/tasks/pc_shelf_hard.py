"""``pc_shelf_pnp_hard``: the shelf pick-and-place suite with the hand working in tight space.

Built on :mod:`molmo_spaces.tasks.pc_shelf` (same room, bookcase family, cameras, skin, timing,
expert strategy and success rule). Nothing in ``pc_shelf`` changes; this module only adds
subclasses, eight new scene XMLs (``custom_scenes/pc_shelf_hard_c<k>.xml``) and a contact-
tolerant save gate that only collection configs switch on.

What is harder than ``pc_shelf_pnp``
------------------------------------
* Bookcase: the two work bays are 2-5 cm lower (clear height 0.27-0.29 m instead of
  0.29-0.33 m) and the centre bay is narrower (0.42-0.46 m instead of 0.46-0.54 m), so the upper
  board and the inner uprights sit closer to the hand, wrist and forearm.
* Neighbours: both sides of the hand's lane are ALWAYS flanked by standing books whose faces are
  1-3 cm outside the open hand's swept envelope (pc_shelf: 5-8 cm), starting at the board's front
  edge, so the hand and wrist travel between them for the whole insertion. In ~15 % of the bays
  one flank stands tighter (0-0.8 cm, a book left slightly proud): the arm may brush it.
  Another item stands 1.2-2.5 cm behind the can (pc_shelf: >= 4 cm).
* Can: drawn 2 cm deeper in the bay (12-17 cm behind the bookcase face).
* Expert: the pc_shelf_pnp strategy, plus a slow 6 cm straight pull-back right after the release
  (the noise-tolerant release; see RELEASE_PULLBACK).
All neighbours are mocap primitives resting on a board, exactly as in ``pc_shelf``; every
waypoint chain is IK-checked before a draw is accepted.

Collection-only additions (switched on by the collection configs, never by the eval config)
-------------------------------------------------------------------------------------------
* DART-style action noise: ``robot_config.action_noise_config`` perturbs the EXECUTED arm command
  (TCP-bounded, through the Jacobian); the recorded action stays the clean expert command. The
  expert is time-indexed toward absolute planned TCP poses (IK seeded from the current state),
  so every step pulls the arm back onto the plan.
* Contact-tolerant save gate (:class:`PCShelfPnPHardExpertConfig.contact_gate_max_frames`): a
  successful episode is saved when the arm's contact with hazards (anything but the can and the
  mat, scored like the evaluator: pact_clutter_* = clutter, everything else = other_environment)
  lasts at most that many 2 ms control frames. Longer contact = not saved. Neighbours are mocap
  (cannot be knocked over); the can must end upright on the mat (the success rule). Contact
  frames, touched bodies and penetration depth are logged per episode in
  ``scene_params['pc_contact']`` (saved in obs_scene).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.tasks import pc_shelf as _base
from molmo_spaces.tasks.pact_place_contact_audit import (
    PactPlaceContactAudit,
    classify_contact,
    place_environment_contact_pairs,
)
from molmo_spaces.tasks.pc_shelf import (
    BOOK_COLORS,
    BOX_COLORS,
    CARRY_UP,
    D_INT,
    GRASP_H,
    JAR_COLORS,
    MAT_HALF,
    MAT_HALF_T,
    N_CELLS,
    READY_TCP,
    SCENE_DIR,
    X_OUT,
    PCShelfPnPExpert,
    PCShelfPnPExpertConfig,
    PCShelfPnPSampler,
    PCShelfPnPTask,
    ShelfCell,
    _T,
    tool_rotation,
)

log = logging.getLogger(__name__)

PC_SHELF_PNP_HARD_ENV_VERSION = "pc_shelf_pnp_hard_v1"

# Same bookcase fronts / lower work boards as pc_shelf (reach-checked there); lower and narrower
# work bays. Fields: x0, b_lo, bay_lo, bay_hi, bay_w, pnp_pick_upper, push_upper, wood.
PC_SHELF_HARD_CELLS: tuple[ShelfCell, ...] = (
    ShelfCell(0.58, 0.70, 0.28, 0.28, 0.44, True, False, (0.72, 0.58, 0.40)),   # light oak
    ShelfCell(0.60, 0.69, 0.29, 0.27, 0.42, False, True, (0.42, 0.28, 0.18)),   # walnut
    ShelfCell(0.57, 0.72, 0.27, 0.29, 0.46, True, False, (0.90, 0.89, 0.86)),   # white laminate
    ShelfCell(0.61, 0.69, 0.29, 0.27, 0.43, False, True, (0.80, 0.70, 0.52)),   # birch
    ShelfCell(0.59, 0.70, 0.27, 0.28, 0.45, True, True, (0.62, 0.62, 0.60)),    # grey laminate
    ShelfCell(0.56, 0.69, 0.28, 0.27, 0.42, False, False, (0.35, 0.24, 0.15)),  # dark oak
    ShelfCell(0.62, 0.71, 0.27, 0.28, 0.44, True, True, (0.78, 0.62, 0.42)),    # maple
    ShelfCell(0.58, 0.70, 0.28, 0.27, 0.43, False, False, (0.24, 0.22, 0.21)),  # black ash
)

PC_SHELF_HARD_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_shelf_hard_c{k}.xml") for k in range(N_CELLS)
)

# Open hand's half-extent across the lane (fingers spread, rolled 25 deg): measured min clearance
# of pc_shelf demos = flank offset - 0.054 m with the lane half-width 0.125 -> 0.075 m.
HAND_HALF_OPEN = 0.075
# First retreat after the release: TCP straight back this far at the slow 'lift' speed.
RELEASE_PULLBACK = 0.06


def build_hard_scene_xml(cell_index: int) -> str:
    """Static MJCF of one hard cell: pc_shelf's generator applied to PC_SHELF_HARD_CELLS."""
    saved = _base.PC_SHELF_CELLS
    try:
        _base.PC_SHELF_CELLS = PC_SHELF_HARD_CELLS
        xml = _base.build_scene_xml(cell_index)
    finally:
        _base.PC_SHELF_CELLS = saved
    xml = xml.replace(f'<mujoco model="pc_shelf_c{cell_index}">',
                      f'<mujoco model="pc_shelf_hard_c{cell_index}">', 1)
    return xml.replace("Generated by molmo_spaces.tasks.pc_shelf.write_pc_shelf_scenes",
                       "Generated by molmo_spaces.tasks.pc_shelf_hard.write_pc_shelf_hard_scenes", 1)


def write_pc_shelf_hard_scenes() -> list[str]:
    """(Re)generate the eight hard cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_SHELF_HARD_SCENES[k])
        p.write_text(build_hard_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
class PCShelfPnPHardSampler(PCShelfPnPSampler):
    """pc_shelf_pnp with tight flanks, lower / narrower bays and a deeper can."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_SHELF_PNP_HARD_ENV_VERSION
    SUITE = "pnp_hard"
    N_DYN = 30                      # 24 books, 3 boxes, 3 jars per house
    CAN_X = (0.12, 0.17)
    CAN_Y = (-0.05, 0.09)
    MAT_X = (0.12, 0.17)
    MAT_Y = (-0.05, 0.09)
    FLANK_GAP = (0.010, 0.030)      # book face -> open-hand envelope
    TIGHT_P = 0.15                  # chance (per bay) that one flank stands tighter
    TIGHT_GAP = (0.000, 0.008)
    FLANK_FRONT = (0.008, 0.03)     # flank books' front faces behind the board's front edge
    BEHIND_GAP = (0.012, 0.025)     # item behind the can's back face / the mat's back edge
    MAX_DRAWS = 40

    @property
    def cell(self) -> ShelfCell:
        return PC_SHELF_HARD_CELLS[self.current_house_index % N_CELLS]

    def _dyn_pool(self, cell_index: int) -> list[dict[str, Any]]:
        c = PC_SHELF_HARD_CELLS[cell_index]
        rng = np.random.default_rng(9300 + cell_index)
        max_h = min(c.bay_lo, c.bay_hi) - 0.035
        pool: list[dict[str, Any]] = []
        for i in range(self.N_DYN):
            if i < 24:
                t = float(rng.uniform(0.022, 0.045))
                d = float(rng.uniform(0.15, 0.23))
                h = float(rng.uniform(0.15, min(0.26, max_h)))
                pool.append(dict(kind="book", half=(d / 2, t / 2, h / 2),
                                 rgba=BOOK_COLORS[int(rng.integers(len(BOOK_COLORS)))]))
            elif i < 27:
                w = float(rng.uniform(0.09, 0.15))
                d = float(rng.uniform(0.10, 0.16))
                h = float(rng.uniform(0.08, min(0.16, max_h)))
                pool.append(dict(kind="box", half=(d / 2, w / 2, h / 2),
                                 rgba=BOX_COLORS[int(rng.integers(len(BOX_COLORS)))]))
            else:
                r = float(rng.uniform(0.035, 0.05))
                h = float(rng.uniform(0.09, min(0.17, max_h)))
                pool.append(dict(kind="jar", half=(r, r, h / 2),
                                 rgba=JAR_COLORS[int(rng.integers(len(JAR_COLORS)))]))
        for i, it in enumerate(pool):
            it["name"] = f"pact_clutter_sh{i}"
        return pool

    def _task_class(self):
        return PCShelfPnPHardTask

    def _place_group(self, avail: list[int], placed: list, keepouts: list, bay: dict,
                     y_edge: float, side: float, x_front: float, n_books: int,
                     out: list) -> None:
        """pc_shelf's book group with a 4 mm spacing between spines (pc_shelf steps 2 mm, which
        its own 3 mm overlap margin rejects, so its groups stop after the first book)."""
        y = y_edge
        n = 0
        for i in [j for j in list(avail) if self._dyn[j]["kind"] == "book"]:
            if n >= n_books:
                return
            hx, hy, hz = self._dyn[i]["half"]
            cy = y + side * hy
            cx = x_front + hx
            if not self._fits(bay, (cx, cy, hx, hy), hz, placed, keepouts):
                continue
            avail.remove(i)
            placed.append((cx, cy, hx, hy))
            out.append((i, [cx, cy, bay["z"] + hz + 0.0005]))
            y = cy + side * (hy + 0.004)
            n += 1

    def _flank_lane(self, avail: list[int], bay: dict, y_lane: float, x_lane_end: float,
                    extra_keep: list, out: list, gaps: tuple[float, float]) -> list:
        """Book groups on both sides of the hand's lane at the given face gaps, an item right
        behind the lane end, then 0-2 items anywhere else on the board."""
        c = self.cell
        lane = ((c.x0 - 0.05 + x_lane_end) / 2, y_lane, (x_lane_end - c.x0 + 0.05) / 2,
                HAND_HALF_OPEN)
        keep = [lane] + list(extra_keep)
        placed: list = []
        for side, gap in zip((1.0, -1.0), gaps):
            y_edge = y_lane + side * (HAND_HALF_OPEN + gap)
            x_front = c.x0 + float(np.random.uniform(*self.FLANK_FRONT))
            self._place_group(avail, placed, keep, bay, y_edge, side, x_front,
                              int(np.random.randint(3, 6)), out)
        if np.random.random() < 0.85:
            self._random_item(avail, placed, keep, bay, out, kinds=("box", "jar", "book"),
                              x_min=x_lane_end)
        for _ in range(2):
            if np.random.random() < 0.5:
                self._random_item(avail, placed, keep, bay, out)
        return placed

    def _draw_gaps(self) -> tuple[float, float]:
        g = [float(np.random.uniform(*self.FLANK_GAP)) for _ in range(2)]
        if np.random.random() < self.TIGHT_P:
            g[int(np.random.randint(2))] = float(np.random.uniform(*self.TIGHT_GAP))
        return g[0], g[1]

    def _draw_episode(self, env) -> dict[str, Any] | None:
        c = self.cell
        td = self._target_spec
        r, h = td["radius"], td["height"]
        pick_z, place_z = (c.b_hi, c.b_lo) if c.pnp_pick_upper else (c.b_lo, c.b_hi)
        pick_h, place_h = (c.bay_hi, c.bay_lo) if c.pnp_pick_upper else (c.bay_lo, c.bay_hi)
        xa = c.x0 + float(np.random.uniform(*self.CAN_X))
        ya = float(np.random.uniform(*self.CAN_Y))
        xb = c.x0 + float(np.random.uniform(*self.MAT_X))
        yb = float(np.random.uniform(*self.MAT_Y))
        yaw = float(np.random.uniform(0, 2 * np.pi))
        mat_top = place_z + 2 * MAT_HALF_T
        # expert waypoints (world): identical strategy and speeds to pc_shelf_pnp
        Rt = tool_rotation()
        pa = np.array([xa, ya, pick_z + GRASP_H])
        pb = np.array([xb, yb, mat_top + GRASP_H + 0.003])
        x_out = c.x0 - X_OUT
        wps = [
            ("pregrasp", [c.x0 - 0.055, ya, pa[2] + 0.005], "fast"),
            ("grasp", pa.tolist(), "insert"),
            ("CLOSE", None, None),
            ("lift", (pa + [0, 0, CARRY_UP]).tolist(), "lift"),
            ("lift", [x_out, ya, pa[2] + CARRY_UP], "mid"),
            ("preplace", [x_out, yb, pb[2] + CARRY_UP], "fast"),
            ("preplace", [xb, yb, pb[2] + CARRY_UP], "insert"),
            ("place", pb.tolist(), "lower"),
            ("OPEN", None, None),
            # slow straight pull-back until the open fingers clear the can (noise std scales
            # with the commanded step, so a slow first retreat keeps the jitter from tipping it)
            ("retreat", [xb - RELEASE_PULLBACK, yb, pb[2] + 0.004], "lift"),
            ("retreat", [x_out - 0.02, yb, pb[2] + 0.01], "mid"),
            ("retreat", list(READY_TCP), "fast"),
        ]
        poses = [_T(p, Rt) for (_n, p, _s) in wps if p is not None]
        if not self._ik_chain_ok(env, poses):
            return None
        avail = list(range(len(self._dyn)))
        out: list = []
        bay_a = self._bay(pick_z, pick_h)
        bay_b = self._bay(place_z, place_h)
        can_fp = (xa, ya, r + 0.008, r + 0.008)
        mat_fp = (xb, yb, MAT_HALF + 0.008, MAT_HALF + 0.008)
        gaps_a = self._draw_gaps()
        gaps_b = self._draw_gaps()
        end_a = xa + r + float(np.random.uniform(*self.BEHIND_GAP))
        end_b = xb + MAT_HALF + float(np.random.uniform(0.004, 0.012))
        self._flank_lane(avail, bay_a, ya, end_a, [can_fp], out, gaps_a)
        self._flank_lane(avail, bay_b, yb, end_b, [mat_fp], out, gaps_b)
        self._apply_layout(env, out)
        self._mocap(env, "place_receptacle_mat", [xb, yb, place_z + MAT_HALF_T + 0.0003])
        quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
        self._settle_target(env, [xa, ya, pick_z + h / 2 + 0.002], quat)
        # Every mocap pose above is final: refresh kinematics so the t=0 observation (cameras and
        # skin) already shows the neighbours on their boards, not at the parking spot.
        mujoco.mj_forward(env.current_model, env.current_data)
        # realised flank clearances (nearest neighbour face to the open-hand envelope, per side)
        realised = {}
        for tag, y_lane, z_b, x_end in (("pick", ya, pick_z, end_a), ("place", yb, place_z, end_b)):
            for side, sname in ((1.0, "left"), (-1.0, "right")):
                best = None
                for i, pos in out:
                    hx, hy, hz = self._dyn[i]["half"]
                    if abs(pos[2] - hz - 0.0005 - z_b) > 1e-3:
                        continue
                    if pos[0] - hx > x_end or pos[0] + hx < c.x0 - 0.05:
                        continue
                    face = (pos[1] - hy - y_lane) if side > 0 else (y_lane - (pos[1] + hy))
                    if face <= 0:
                        continue
                    gap = face - HAND_HALF_OPEN
                    best = gap if best is None or gap < best else best
                realised[f"{tag}_{sname}"] = None if best is None else round(float(best), 4)
        return dict(
            **self._cell_common(),
            pick_board_z=pick_z, place_board_z=place_z, pick_upper=bool(c.pnp_pick_upper),
            can_radius=r, can_height=h, can_label=td["label"],
            start_xyz=[xa, ya, pick_z + h / 2], goal_xyz=[xb, yb, mat_top + h / 2],
            mat_xyz=[xb, yb, place_z + MAT_HALF_T], mat_top=mat_top, place_tol=self.PLACE_TOL,
            expert=[[n, p, s] for (n, p, s) in wps],
            neighbours=[[self._dyn[i]["name"], self._dyn[i]["kind"], pos,
                         list(self._dyn[i]["half"])] for i, pos in out],
            n_neighbours=len(out),
            hard=dict(flank_gaps_pick=list(gaps_a), flank_gaps_place=list(gaps_b),
                      flank_gap_realised=realised, behind_x_pick=end_a, behind_x_place=end_b,
                      hand_half_open=HAND_HALF_OPEN),
        )


# --------------------------------------------------------------------------------------------
# Contact accounting + task
# --------------------------------------------------------------------------------------------
HAZARD_CLASSES = ("hazard_bar", "other_environment", "clutter", "mounted_fixture")


class _HazardContactAudit(PactPlaceContactAudit):
    """The evaluator's contact audit plus a union count of 2 ms frames with ANY hazard contact
    and the touched hazard bodies (frames per name)."""

    def reset(self) -> None:
        super().reset()
        self.hazard_frames = 0
        self.hazard_bodies: dict[str, int] = {}
        self.hazard_max_pen = 0.0
        self.hazard_first_step: int | None = None

    def observe(self, env, step: int) -> None:
        sim_time = round(float(env.current_data.time), 12)
        if sim_time in self._seen_times:
            return
        super().observe(env, step)
        try:
            names = set()
            pen = 0.0
            for pair in place_environment_contact_pairs(env):
                if classify_contact(pair) not in HAZARD_CLASSES:
                    continue
                robot1 = str(pair["root1"]).startswith("robot_0/")
                robot2 = str(pair["root2"]).startswith("robot_0/")
                if robot1 or robot2:
                    second, prefix = (not robot2), ""
                else:   # the can (target) against clutter: name the clutter side
                    second = "cavity_obj_" in str(pair["root1"]) or "cavity_obj_" in str(pair["body1"])
                    prefix = "can-vs:"
                other_body = pair["body2"] if second else pair["body1"]
                other_geom = pair["geom2"] if second else pair["geom1"]
                nm = other_body if other_body not in ("", "world") else other_geom
                names.add(prefix + str(nm))
                pen = max(pen, -float(pair["distance_m"]))
            if names:
                self.hazard_frames += 1
                self.hazard_max_pen = max(self.hazard_max_pen, pen)
                if self.hazard_first_step is None:
                    self.hazard_first_step = int(step)
                for nm in names:
                    self.hazard_bodies[nm] = self.hazard_bodies.get(nm, 0) + 1
        except Exception as e:  # pragma: no cover - a diagnostic must never break a rollout
            log.debug(f"[PCShelfHard] hazard audit failed: {e}")


class PCShelfPnPHardTask(PCShelfPnPTask):
    """pc_shelf_pnp success rule; with a collection gate set by the expert, a success also needs
    hazard contact <= gate frames (light contact). The evaluator never sets the gate."""

    def pc_contact_stats(self) -> dict[str, Any] | None:
        audit = getattr(self, "_pc_hard_audit", None)
        if audit is None:
            return None
        frames = dict(audit._frames_with)
        gate = getattr(self, "_pc_contact_gate", None)
        hf = int(audit.hazard_frames)
        return dict(
            hazard_frames=hf,
            hazard_contact_s=round(hf * 0.002, 4),
            frames_by_class={k: int(frames.get(k, 0)) for k in HAZARD_CLASSES + ("place_receptacle",)},
            bodies=dict(sorted(audit.hazard_bodies.items(), key=lambda kv: -kv[1])),
            max_penetration_m=round(float(audit.hazard_max_pen), 5),
            first_contact_step=audit.hazard_first_step,
            samples=len(audit._seen_times),
            gate_max_frames=None if gate is None else int(gate),
            contact_level="none" if hf == 0 else ("light" if gate is None or hf <= gate else "heavy"),
        )

    def _success(self) -> tuple[bool, str]:
        ok, msg = super()._success()
        gate = getattr(self, "_pc_contact_gate", None)
        if gate is None:
            return ok, msg
        st = self.pc_contact_stats() or {}
        hf = int(st.get("hazard_frames", 0))
        light = hf <= int(gate)
        return ok and light, f"{msg} hazard_frames={hf} gate<={int(gate)}"

    def get_obs_scene(self) -> dict[str, Any]:
        d = super().get_obs_scene()
        st = self.pc_contact_stats()
        if st is not None:
            sp = dict(d.get("scene_params") or {})
            sp["pc_contact"] = _base._jsonable(st)
            d["scene_params"] = sp
            d["pc_contact"] = _base._jsonable(st)
        return d

    def _maybe_log_obstacle_diag(self, success: bool) -> None:
        already = getattr(self, "_obstacle_diag_logged", True)
        super()._maybe_log_obstacle_diag(success)
        if not already and getattr(self, "_obstacle_diag_logged", False):
            st = self.pc_contact_stats()
            if st is not None:
                log.info(f"[PCShelfHard] contact level={st['contact_level']} "
                         f"hazard_frames={st['hazard_frames']} max_pen={st['max_penetration_m']:.4f} "
                         f"bodies={st['bodies']} gate={st['gate_max_frames']}")


# --------------------------------------------------------------------------------------------
# Expert
# --------------------------------------------------------------------------------------------
class PCShelfPnPHardExpert(PCShelfPnPExpert):
    """pc_shelf_pnp expert + the per-2-ms hazard audit (and the collection save gate)."""

    def reset(self, reset_retries: bool = True):
        if reset_retries or getattr(self.task, "_pc_hard_audit", None) is None:
            audit = _HazardContactAudit()
            self.task._pc_hard_audit = audit
            self.task._contact_audit_hook = audit
        gate = getattr(self.policy_config, "contact_gate_max_frames", None)
        self.task._pc_contact_gate = None if gate is None else int(gate)
        return super().reset(reset_retries)


class PCShelfPnPHardExpertConfig(PCShelfPnPExpertConfig):
    # None = plain pc_shelf success (definition / eval config). Collection: max 2 ms frames of
    # arm-vs-hazard contact for a successful episode to be saved.
    contact_gate_max_frames: int | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCShelfPnPHardExpert


__all__ = [
    "PC_SHELF_PNP_HARD_ENV_VERSION", "PC_SHELF_HARD_CELLS", "PC_SHELF_HARD_SCENES",
    "PCShelfPnPHardSampler", "PCShelfPnPHardTask", "PCShelfPnPHardExpert",
    "PCShelfPnPHardExpertConfig", "write_pc_shelf_hard_scenes", "build_hard_scene_xml",
    "HAND_HALF_OPEN",
]
