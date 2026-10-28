"""``pc_kitchen_blind``: the kitchen jar transfer of ``pc_kitchen_reach`` with a HIDDEN under-cabinet
fixture that only the arm skin can see. Built for the PC-ACT rebuttal claim "the skin tells the
policy where a hazard is when the cameras cannot".

What is hidden, and why only the skin can find it
--------------------------------------------------
An under-cabinet radio (``pact_intrusion_undercab_radio``, a 12 x 23 x 6 cm box screwed to the
cabinet underside, its front face 12 cm behind the cabinet front) hangs in the path of the
skin-covered wrist knuckle / forearm (fr3_link5 / fr3_link6) during the carry, in 50 % of the
episodes, at a random place along the backsplash (centre y ~ U(-0.22, 0.24), independent of the
jar, the coaster, the clutter, the cell and the transfer direction). A 2.5 cm light-rail molding
along the bottom front edge of the wall cabinets (a standard trim that hides under-cabinet
lights) keeps it out of the exo camera's view, and the wrist camera looks down along the fingers,
so neither RGB camera ever sees it. The skin sensors on top of the wrist / forearm
(link6_sensor_1, link5_back_sensor_2 / _5) read it 6-18 cm before the arm reaches it.

Body names: the radio's ``pact_intrusion_`` prefix makes the PACT contact audit classify its
contacts as ``hazard_bar`` (evaluator: ``hit_bar`` / ``bar_contact_frames``), separate from the
cabinet / counter / clutter contacts.

Expert (privileged): the pc_kitchen_reach expert (tilted hand, slide-in grasp, slow lateral carry
under the cabinets), but the carry height now depends on the hidden radio. Without it (or away
from it) the jar is carried HIGH (skinned wrist / forearm top 2 cm under the light rail / LED
strip / cabinet bottom, jar bottom LIFT = DIP + 1.0 cm above the counter). Where the wrist /
forearm would pass within FIX_CLEAR = 2 cm of the radio, the expert DIPS by DIP (radio overlap with
the high carry + 1.7 cm; the jar then rides 1.0 cm above the counter, 7 mm over the 3 mm
coaster), with 6 cm ramps in and out. Measured on real expert rollouts with densified collision
hulls: the skinned forearm passes ~0-1.5 cm under the light rail on the high carry (the hull-vertex
design estimate said 2 cm), so DART-noise demos can graze the rail lightly (within the 150-frame
save gate); the gripper / link7 stay >= ~5 cm from everything. Why the high carry is not free: the radio overlaps it by
~3.2 cm. Why a blind policy cannot just
always carry low: an imitator converges to the demos' mean given what it sees; the mean
trajectory over hidden radio positions still hits the radio (Monte-Carlo below). The always-low
plan is a loophole a blind policy could in principle find; it is not in the demos.

Cells (house_index % 8): upper-cabinet bottom 0.51 / 0.53 / 0.55 / 0.57 m above the counter x
cabinet depth 0.33 / 0.35 m (front face x = 0.53 / 0.51); carry lane 13 cm behind the cabinet
front; fridge side y = 0.36 / 0.40; the jar always starts on the cooktop side and goes to the
fridge side (see the cell table). Scenes:
``custom_scenes/pc_kitchen_blind_c<k>.xml`` (:func:`write_pc_kitchen_blind_scenes`).

R4 Monte-Carlo (design, scratch IK + collision-hull vertex clearances, P(radio) = 0.5, 400 draws
over the 8 cells, mean trajectory = average expert carry over 200 radio draws for the same
visible scene): hit rate over all draws -- always-high 48 %, always-low 0 %, mean expert
trajectory 47 % (median overlap 0.8 cm), expert 0 % (min skinned clearance 1.7 cm); gripper /
link7 >= 13 cm from the radio.
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
from molmo_spaces.tasks.pc_kitchen_reach import (
    BACKSPLASH_X,
    CLR,
    COASTER_HALF_T,
    COASTER_R,
    COUNTER_Z,
    FRONT_DX,
    GRIP_DX,
    JAR_TOP_ABOVE_TCP,
    JARS,
    LED_H,
    LINK5_DY_NEG,
    PARK_Z,
    PLACE_TOL,
    READY_TCP,
    SLIDE,
    TOOL_AZ_DEG,
    TOOL_PITCH_DEG,
    WALL_X,
    PCKitchenReachExpertConfig,
    PCKitchenReachSampler,
    PCKitchenReachTask,
    _T,
    reach_tool_rotation,
    slide_dir,
)
from molmo_spaces.tasks.pc_shelf import SCENE_DIR

log = logging.getLogger(__name__)

PC_KITCHEN_BLIND_ENV_VERSION = "pc_kitchen_blind_v1"
N_CELLS = 8
RAIL_H = 0.025              # light-rail molding under the cabinet front edge (hides the radio
                            # from the exo camera in every cell: analytic margin >= 7 mm)
HIGH_OFF = 0.3583           # high-carry TCP = cabinet bottom - this (skinned top 2.0 cm under the
                            # rail / LED / cabinet bottom, vertex clearance; measured over the lane,
                            # both depths. 1.5 cm let DART noise graze the light rail.)
HUMP_TOP = 0.3298           # skinned wrist/forearm top above the TCP inside the radio's x band
HUMP_DY = (-0.150, 0.056)   # y extent (TCP-relative) of the skinned top under the radio's x band
LANE_SETBACK = 0.13         # carry lane TCP x = cabinet front + this
LANE_JIT = 0.008

FIX_NAME = "pact_intrusion_undercab_radio"
FIX_P = 0.5
FIX_Y = (-0.22, 0.24)
FIX_SETBACK = 0.12          # radio front face behind the cabinet front face
FIX_DEPTH = 0.12
FIX_W = 0.23
FIX_T = 0.060               # hangs this far below the cabinet bottom
FIX_CLEAR = 0.02            # expert: dip where the high carry would pass within this of the radio
C_DIP = 0.017               # skinned clearance to the radio on the dipped carry
RAMP = 0.06                 # y travel of the ramps into / out of the dip
JAR_LOW_GAP = 0.010         # jar bottom above the counter on the dipped carry (7 mm over the coaster)
COASTER_HALF_T_BLIND = 0.0015   # 3 mm cork coaster


@dataclass(frozen=True)
class BlindCell:
    gap: float
    depth: float
    fridge_y: float
    jar_fridge_side: bool

    @property
    def cab_bottom_z(self) -> float:
        return COUNTER_Z + self.gap

    @property
    def cab_front_x(self) -> float:
        return WALL_X - self.depth

    @property
    def lane_x(self) -> float:
        return self.cab_front_x + LANE_SETBACK

    @property
    def z_high(self) -> float:
        return self.cab_bottom_z - HIGH_OFF

    @property
    def dip(self) -> float:
        overlap = FIX_T - (HIGH_OFF - HUMP_TOP)
        return overlap + C_DIP

    @property
    def z_low(self) -> float:
        return self.z_high - self.dip

    @property
    def lift(self) -> float:
        return self.dip + JAR_LOW_GAP

    @property
    def z_grasp(self) -> float:
        return self.z_high - self.lift

    @property
    def jar_h(self) -> float:
        return round(self.z_grasp - COUNTER_Z + JAR_TOP_ABOVE_TCP, 4)

    def params(self, k: int) -> dict[str, float]:
        return {
            "cell": k, "fridge_y": self.fridge_y, "cab_gap": self.gap, "cab_depth": self.depth,
            "cab_bottom_z": self.cab_bottom_z, "cab_front_x": self.cab_front_x,
            "left_wall_y": self.fridge_y + _kit.FRIDGE_W + 0.04,
        }


# The jar always starts on the cooktop side (-y) and goes to the fridge side: the wrist leans
# back toward -y (20 deg hand tilt), so moving +y the long skinned hump trails and the up-facing
# skin sensors (link5_back_sensor_5 / _2) reach the radio before the expert commits to the dip in
# every configuration (R5 Monte-Carlo 47/47). With the jar on the fridge side 7/40 dips started
# before any sensor could see the radio.
PC_KITCHEN_BLIND_CELLS: tuple[BlindCell, ...] = (
    BlindCell(0.51, 0.33, 0.36, False),
    BlindCell(0.53, 0.35, 0.40, False),
    BlindCell(0.55, 0.33, 0.40, False),
    BlindCell(0.57, 0.35, 0.36, False),
    BlindCell(0.51, 0.35, 0.40, False),
    BlindCell(0.53, 0.33, 0.36, False),
    BlindCell(0.55, 0.35, 0.36, False),
    BlindCell(0.57, 0.33, 0.40, False),
)

PC_KITCHEN_BLIND_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_kitchen_blind_c{k}.xml") for k in range(N_CELLS)
)


def write_pc_kitchen_blind_scenes() -> list[str]:
    """(Re)generate the eight cell XMLs (+ metadata). Never while a run uses them."""
    paths = []
    for k, c in enumerate(PC_KITCHEN_BLIND_CELLS):
        p = Path(PC_KITCHEN_BLIND_SCENES[k])
        xml = _kit.build_scene_xml(k, params=c.params(k), model="pc_kitchen_blind")
        xml = xml.replace("GENERATED by molmo_spaces/tasks/pc_kitchen.py",
                          "GENERATED by molmo_spaces/tasks/pc_kitchen_blind.py "
                          "(write_pc_kitchen_blind_scenes, pc_kitchen room generator)", 1)
        xf, zb, fy = c.cab_front_x, c.cab_bottom_z, c.fridge_y
        yr = _kit.ROOM_RIGHT_Y
        rail = (f'    <geom name="upper_cab_light_rail" class="furn" type="box" '
                f'pos="{xf - 0.009:.4f} {(yr + fy) / 2:.4f} {zb - RAIL_H / 2:.4f}" '
                f'size="0.0090 {(fy - yr) / 2:.4f} {RAIL_H / 2:.4f}" material="mat_cab_door"/>\n')
        anchor = "    <!-- ============ movable"
        assert anchor in xml
        xml = xml.replace(anchor, rail + anchor, 1)
        p.write_text(xml)
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


def radio_box(c: BlindCell, y: float) -> tuple[np.ndarray, np.ndarray]:
    """(centre, half sizes) of the radio at centre y, screwed to the cabinet underside."""
    centre = np.array([c.cab_front_x + FIX_SETBACK + FIX_DEPTH / 2, y, c.cab_bottom_z - FIX_T / 2])
    return centre, np.array([FIX_DEPTH / 2, FIX_W / 2, FIX_T / 2])


def carry_profile(c: BlindCell, radio_y: float | None):
    """z(y) of the carry: high, or dipped by c.dip where the skinned top would pass within
    FIX_CLEAR of the radio (TCP y in [y1, y2]), with RAMP-long linear ramps outside."""
    if radio_y is None:
        return None
    y1 = radio_y - FIX_W / 2 - FIX_CLEAR - HUMP_DY[1]
    y2 = radio_y + FIX_W / 2 + FIX_CLEAR - HUMP_DY[0]
    return (y1, y2)


def carry_z(c: BlindCell, zone, y: float) -> float:
    if zone is None:
        return c.z_high
    y1, y2 = zone
    out = max(y1 - y, y - y2, 0.0)
    if out <= 0.0:
        return c.z_low
    if out < RAMP:
        return c.z_low + (c.z_high - c.z_low) * out / RAMP
    return c.z_high


def blind_waypoints(c: BlindCell, xa: float, ya: float, xb: float, yb: float, radio_y: float | None):
    """Expert TCP waypoints [(phase, xyz | None, speed)]."""
    u = slide_dir()
    zg = c.z_grasp
    g = np.array([xa, ya, zg])
    pre = g + SLIDE * u
    front = pre - np.array([FRONT_DX, 0.0, 0.0])
    z_place = zg + 2 * COASTER_HALF_T_BLIND + 0.006
    pl = np.array([xb, yb, z_place])
    out = pl + SLIDE * u
    back = out + np.array([-FRONT_DX, 0.0, 0.01])
    zone = carry_profile(c, radio_y)
    # carry breakpoints in travel order (ramp corners inside (ya, yb))
    s = 1.0 if yb >= ya else -1.0
    ys = [ya]
    if zone is not None:
        y1, y2 = zone
        corners = sorted({y1 - RAMP, y1, y2, y2 + RAMP})
        corners = [y for y in corners if (y - ya) * s > 1e-4 and (yb - y) * s > 1e-4]
        ys += sorted(corners, key=lambda y: (y - ya) * s)
    ys.append(yb)
    span = (yb - ya) if abs(yb - ya) > 1e-6 else 1.0
    carry = []
    for y in ys:
        x = xa + (xb - xa) * (y - ya) / span
        carry.append([float(x), float(y), float(carry_z(c, zone, y))])
    wps = [
        ("pregrasp", front.tolist(), "fast"),
        ("pregrasp", pre.tolist(), "insert"),
        ("grasp", g.tolist(), "insert"),
        ("CLOSE", None, None),
        ("lift", carry[0], "lift"),
    ]
    for p in carry[1:]:
        wps.append(("preplace", p, "carry"))
    wps += [
        ("place", pl.tolist(), "lower"),
        ("OPEN", None, None),
        ("retreat", out.tolist(), "lift"),
        ("retreat", back.tolist(), "mid"),
        ("retreat", list(READY_TCP), "fast"),
    ]
    return wps


class PCKitchenBlindSampler(PCKitchenReachSampler):
    """pc_kitchen_reach sampler + the hidden under-cabinet radio and the dip expert."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_KITCHEN_BLIND_ENV_VERSION
    SUITE = "kitchen_blind"
    COOK_Y_BY_DEPTH = {0.33: (-0.16, -0.08), 0.35: (-0.20, -0.10)}   # reach at the deeper lane
    LANE_MIN_SPAN = 0.20

    @property
    def cell(self) -> BlindCell:  # type: ignore[override]
        return PC_KITCHEN_BLIND_CELLS[self.current_house_index % N_CELLS]

    def _target_dims(self, cell_index: int) -> dict[str, Any]:
        r, rgba, lid, label = JARS[cell_index % len(JARS)]
        c = PC_KITCHEN_BLIND_CELLS[cell_index]
        return dict(radius=r, height=c.jar_h, rgba=rgba, cap=lid, label=label)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        super()._add_task_bodies(spec, cell_index)
        # 3 mm cork coaster instead of pc_kitchen_reach's 6 mm one (the dipped jar rides 1 cm
        # above the counter)
        g = spec.geom("place_receptacle_coaster_g")
        g.size = [COASTER_R, COASTER_HALF_T_BLIND, 0.0]

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        super().add_auxiliary_objects(spec)
        b = spec.worldbody.add_body(name=FIX_NAME, mocap=True, pos=[-0.9, -0.6, PARK_Z])
        kw = dict(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
        b.add_geom(name=FIX_NAME + "_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[FIX_DEPTH / 2, FIX_W / 2, FIX_T / 2], rgba=[0.13, 0.13, 0.14, 1.0], **kw)
        # front panel: display + speaker grille (visual only)
        b.add_geom(name=FIX_NAME + "_panel", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.001, FIX_W / 2 - 0.01, FIX_T / 2 - 0.008],
                   pos=[-FIX_DEPTH / 2 - 0.001, 0.0, 0.0], rgba=[0.30, 0.31, 0.33, 1.0],
                   contype=0, conaffinity=0)

    def _park_all(self, env) -> None:
        super()._park_all(env)
        self._mocap(env, FIX_NAME, [-0.9, -0.6, PARK_Z])

    def _cell_common(self) -> dict[str, Any]:
        c = self.cell
        return dict(cab_gap=c.gap, cab_depth=c.depth, cab_bottom_z=c.cab_bottom_z,
                    cab_front_x=c.cab_front_x, rail_bottom_z=c.cab_bottom_z - RAIL_H,
                    led_bottom_z=c.cab_bottom_z - LED_H, fridge_y=c.fridge_y,
                    jar_fridge_side=bool(c.jar_fridge_side), lane_x=c.lane_x,
                    z_grasp=c.z_grasp, z_high=c.z_high, z_low=c.z_low, dip=c.dip, lift=c.lift,
                    tool_pitch_deg=TOOL_PITCH_DEG, tool_az_deg=TOOL_AZ_DEG, slide=SLIDE)

    def _draw_episode(self, env) -> dict[str, Any] | None:
        c = self.cell
        td = self._target_spec
        r, h = td["radius"], td["height"]
        y_fr = float(np.random.uniform(self.FRIDGE_Y_MIN,
                                       min(self.FRIDGE_Y_MAX, c.fridge_y - self.FRIDGE_MARGIN)))
        y_ck = float(np.random.uniform(*self.COOK_Y_BY_DEPTH[round(c.depth, 2)]))
        if y_fr - y_ck < self.LANE_MIN_SPAN:
            return None
        ya, yb = (y_fr, y_ck) if c.jar_fridge_side else (y_ck, y_fr)
        xa = c.lane_x + float(np.random.uniform(-LANE_JIT, LANE_JIT))
        xb = c.lane_x + float(np.random.uniform(-LANE_JIT, LANE_JIT))
        # hidden radio: drawn independently of everything above
        present = bool(np.random.random() < FIX_P)
        radio_y = float(np.random.uniform(*FIX_Y)) if present else None
        mat_top = COUNTER_Z + 2 * COASTER_HALF_T_BLIND
        Rt = reach_tool_rotation()
        wps = blind_waypoints(c, xa, ya, xb, yb, radio_y)
        poses = [_T(p, Rt) for (_n, p, _s) in wps if p is not None]
        if not self._ik_chain_ok(env, poses):
            return None
        if radio_y is not None:
            # Accept a visible layout only if the radio-free (high-carry) plan is feasible too, so
            # IK rejections never depend on the hidden radio (2026-10-14 03:30 fix: the collected
            # demos were drawn without this check and show P(radio | cell 6 / 7) = 0.73 / 0.87,
            # the two cells where the high carry is hardest to reach; 0.37-0.53 elsewhere).
            wps0 = blind_waypoints(c, xa, ya, xb, yb, None)
            if not self._ik_chain_ok(env, [_T(p, Rt) for (_n, p, _s) in wps0 if p is not None]):
                return None
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
            rr = self._dyn[i]["r"]
            x_min = x_lane + GRIP_DX + CLR + rr
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
            rr = self._dyn[i]["r"]
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
        self._mocap(env, "place_receptacle_coaster", [xb, yb, COUNTER_Z + COASTER_HALF_T_BLIND + 0.0003])
        radio = None
        if radio_y is not None:
            centre, half = radio_box(c, radio_y)
            self._mocap(env, FIX_NAME, centre.tolist())
            zone = carry_profile(c, radio_y)
            radio = dict(present=True, centre=centre.tolist(), half=half.tolist(), y=radio_y,
                         dip_zone=list(zone))
        quat = R.from_euler("z", float(np.random.uniform(0, 2 * np.pi))).as_quat(scalar_first=True)
        self._settle_target(env, [xa, ya, COUNTER_Z + h / 2 + 0.002], quat)
        mujoco.mj_forward(env.current_model, env.current_data)
        return dict(
            **self._cell_common(),
            can_radius=r, can_height=h, can_label=td["label"], jar_label=td["label"],
            start_xyz=[xa, ya, COUNTER_Z + h / 2], goal_xyz=[xb, yb, mat_top + h / 2],
            mat_xyz=[xb, yb, COUNTER_Z + COASTER_HALF_T_BLIND], mat_top=mat_top, place_tol=PLACE_TOL,
            expert=[[n, p, s] for (n, p, s) in wps],
            neighbours=[[self._dyn[i]["name"], self._dyn[i]["kind"], pos,
                         list(self._dyn[i]["half"])] for i, pos in out],
            n_neighbours=len(out),
            hidden_radio=radio if radio is not None else dict(present=False),
            expert_dips=bool(any(n == "preplace" and abs(p[2] - c.z_high) > 1e-4
                                 for (n, p, _s) in wps if p is not None)),
        )

    def _task_class(self):
        return PCKitchenBlindTask


class PCKitchenBlindTask(PCKitchenReachTask):
    """pc_kitchen_reach task (success rule, hazard audit, per-link clearance audit). The radio is
    a hazard like any other scene geom in the clearance audit; the contact audit scores it as
    'hazard_bar' through its pact_intrusion_ body name.

    The arm links' clearances use DENSIFIED collision hulls: hull vertices plus every hull face's
    centroid and a point every 1 cm along every hull edge (the plain vertex estimate of pc_kitchen_reach overestimates the
    gap to thin edges such as the light rail by a few mm; MuJoCo's mj_geomDistance returns
    spurious zeros on these meshes in this scene). Gripper meshes keep the subsampled vertices."""

    def _kr_setup(self, model) -> None:
        super()._kr_setup(model)
        kr = self._kr
        rv, owners = [], []
        for g, v in zip(kr["rg"], kr["rv"]):
            b = model.body(int(model.geom_bodyid[g])).name or ""
            short = b.split("/", 1)[-1] if b.startswith("robot_0/") else b
            key = "gripper" if short.startswith("gripper/") else short
            if short.startswith("fr3_link"):
                mid = int(model.geom_dataid[g])
                ga = int(model.mesh_graphadr[mid])
                if ga >= 0:
                    nv = int(model.mesh_graph[ga]); nf = int(model.mesh_graph[ga + 1])
                    off = ga + 2 + 2 * nv + (nv + 3 * nf)
                    faces = np.asarray(model.mesh_graph[off: off + 3 * nf], dtype=int).reshape(-1, 3)
                    adr = int(model.mesh_vertadr[mid])
                    allv = np.asarray(model.mesh_vert[adr: adr + int(model.mesh_vertnum[mid])], dtype=float)
                    tri = allv[faces]                                   # (nf, 3, 3)
                    edges = np.unique(np.sort(np.concatenate(
                        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0), axis=1), axis=0)
                    pts = [tri.mean(axis=1)]
                    for a_, b_ in edges:               # every 1 cm along each hull edge
                        pa, pb = allv[a_], allv[b_]
                        n_ = int(np.ceil(np.linalg.norm(pb - pa) / 0.010))
                        if n_ > 1:
                            t_ = np.linspace(0.0, 1.0, n_ + 1)[1:-1, None]
                            pts.append(pa + t_ * (pb - pa))
                    extra = np.concatenate(pts, axis=0)
                    v = np.concatenate([v, extra], axis=0)
            rv.append(v)
            owners += [key] * len(v)
        kr["rv"] = rv
        kr["owners"] = np.asarray(owners)
        # per-geom blocks for a fast, prefiltered per-step evaluation
        keys = sorted(set(owners))
        blocks = []
        for g, v in zip(kr["rg"], rv):
            b = model.body(int(model.geom_bodyid[g])).name or ""
            short = b.split("/", 1)[-1] if b.startswith("robot_0/") else b
            key = "gripper" if short.startswith("gripper/") else short
            blocks.append((g, v, float(np.linalg.norm(v, axis=1).max()), key))
        kr["blocks"] = blocks
        kr["keys"] = keys

    EVERY = 2   # policy steps between clearance evaluations

    def _kr_step(self, data) -> None:
        model = data.model
        if getattr(self, "_kr", None) is None or self.episode_step_count == 0:
            self._kr_setup(model)
        kr = self._kr
        step = int(self.episode_step_count)
        if step % self.EVERY:
            return
        cache: dict[int, np.ndarray] = {}
        for gh, name, _bn in kr["hz"]:
            ch = data.geom_xpos[gh]
            rbh = float(model.geom_rbound[gh])
            Rm = data.geom_xmat[gh].reshape(3, 3)
            sz = model.geom_size[gh]
            t = int(model.geom_type[gh])
            for g, v, rbv, key in kr["blocks"]:
                if float(np.linalg.norm(data.geom_xpos[g] - ch)) - rbv - rbh > 0.30:
                    continue
                P = cache.get(g)
                if P is None:
                    P = data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T
                    cache[g] = P
                loc = (P - ch) @ Rm
                if t == int(mujoco.mjtGeom.mjGEOM_BOX):
                    q = np.abs(loc) - sz
                    dist = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
                elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    q = np.stack([np.linalg.norm(loc[:, :2], axis=1) - sz[0], np.abs(loc[:, 2]) - sz[1]], 1)
                    dist = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
                else:
                    dist = np.linalg.norm(loc, axis=1) - sz[0]
                dm = float(dist.min())
                if dm > 0.30:
                    continue
                prev = kr["best"].get(key)
                if prev is None or dm < prev[0]:
                    kr["best"][key] = [round(dm, 4), name, step]


class PCKitchenBlindExpertConfig(PCKitchenReachExpertConfig):
    pass


__all__ = [
    "PC_KITCHEN_BLIND_ENV_VERSION", "PC_KITCHEN_BLIND_CELLS", "PC_KITCHEN_BLIND_SCENES",
    "PCKitchenBlindSampler", "PCKitchenBlindTask", "PCKitchenBlindExpertConfig",
    "write_pc_kitchen_blind_scenes", "blind_waypoints", "radio_box", "carry_profile", "N_CELLS",
    "FIX_NAME",
]


if __name__ == "__main__":
    import sys

    if "--write-scenes" in sys.argv:
        for p in write_pc_kitchen_blind_scenes():
            print(p)
