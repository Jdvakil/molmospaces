"""``pc_occ_pantry``: take a jar / tin from the BACK of a pantry shelf bay and set it on a mat at the
front, while two china espresso cups hanging on cup hooks under the shelf above (left OR right of
the reach lane, coin flip) threaten the SKIN-COVERED wrist knuckle / forearm, never the gripper.

Why this suite exists (2026-10-15, OCCLUDED NATURAL ITEMS)
-----------------------------------------------------------
The earlier hidden-hazard suites hid artificial-looking objects (a wire basket, a valve box, a
drain pipe, a radio) behind trim. Here the hidden thing is an ordinary household object with a real
mesh (the THOR ``Mug_1`` china mug, scaled to espresso-cup size), hanging where cups really hang (on
brass cup hooks screwed into the underside of a pantry shelf), and what hides it is the pantry
itself: the shelf board above the bay (its front edge throws the exo camera's "lintel shadow" over
the back of the bay ceiling) and the pantry goods standing in the bay. The cups are present in
every scene; only their side changes, drawn independently of everything visible.

Why hanging cups and not a standing bottle (design attempt A, measured 2026-10-15)
-----------------------------------------------------------------------------------
The brief asked for a tall glass bottle STANDING in the back row behind a front row of cereal
boxes. Two camera facts rule that out for a top-down reach:
* the wrist camera sits 13 cm above the TCP and looks DOWN (its 87 deg wide axis along the
  reach); every standing object beside the lane that the skinned wrist / forearm can touch is
  below it and inside its view for most of the reach. Measured on the pc_shelf2_blind expert
  rollouts (houses 0 / 3, a 7 x 7 x 30 cm bottle-sized box standing 13.5-15 cm left / right of the
  lane, beside the can or 10 cm in front of it): the wrist image changes by up to 107-172 of 255 in
  10,700-81,800 pixels, the exo image (no front row there) in 400-960 pixels;
* exo_camera_1 looks down at ~17-25 deg from behind-right. A front row only hides what is lower
  than itself, but the forearm must pass OVER the front row and still reach the hidden item, which
  then has to be taller than the front row (visible); for the LEFT position the line of sight
  runs back-right through the reach lane, where no occluder can stand.
Only things ABOVE the hand are invisible to the wrist camera, so the natural item hangs under the
shelf above; the shelf board hides it from the exo camera.

Scene (``custom_scenes/pc_occ_pantry_c<k>.xml``, :func:`write_pc_occ_pantry_scenes`)
-----------------------------------------------------------------------------------
The ``pc_shelf_blind`` walk-in pantry (closed room, three-bay pantry bookcase 0.40 m deep against
the far wall, side bookcase left, sideboard right, low cabinet behind; same eight cells: tall lower
work bay sized per cell so the skinned wrist top passes ~2.5 cm under the board above). Per episode
the bay holds pantry goods (mocap, re-posed): one tall item at the FRONT on each side of the lane
(cereal box / oat canister / pasta box), one item at the back on each side (jar / tin / canister)
and three items in the bay above.

Hidden hazard (``pact_intrusion_cups_left`` / ``_right``, scored by the evaluator as ``hazard_bar``)
------------------------------------------------------------------------------------------------
Two cup hooks 1 cm long, 28-30 cm behind the board's front edge, each holding an espresso cup
(``Mug_1`` mesh x 0.55: 5.5 cm diameter, 5.7 cm long) hung by its handle (handle up, cup axis
across the bay, opening facing outward); the cups' bottoms are 7.9 cm under the board. The pair
hangs LEFT or RIGHT of the lane (two mocap bodies, the unused one parked outside the room), inner
edge 7.8-8.6 cm left of the lane centre / 1.6-2.4 cm right of it reaching over the lane (the
pc_shelf2_blind basket offsets: the FR3 forearm leans to +y). Collision = primitives (cylinder
body, box handle, capsule hook) matching the mesh; the meshes are visual only.

Task + expert: exactly ``pc_shelf2_blind`` (scripted TCP segments, hand pointing down): pick the
item 26.5-28.5 cm deep, set it on the coaster mat 13-15 cm deep (pc_shelf_pnp success rule).
Pre-insert -> slide in untilted (0.11 m/s) until the TCP is :data:`D_DIV` (16 cm) inside -> pause
``dwell_s`` (1.0 s, spans the policy's step-50 chunk query) ->
tilt the hand 20 deg away from the cups (wrist knuckle swings ~9 cm, the only plan difference) ->
pick -> carry -> place -> untilt -> back out.

Acceptance checks (2026-10-15, scripts/audit_pc_occ_pantry.py: 8 expert rollouts, 78 kinematic
draws x every fixed tilt -20..+20 deg in 2.5 deg steps -> reports/2026-10-15/pc_occ_pantry_audit.json;
360 sampler draws for R3 -> reports/2026-10-15/pc_occ_pantry_r3_draws.json)
* R1: fr3_link7 + gripper >= 6.8 cm from the cups in the rollouts and >= 6.4 cm in EVERY fixed-tilt
  plan; skinned links 3.4-7.6 cm in the rollouts (expert plans 2.9-7.8 cm).
* R2: exo / wrist max pixel change 0 / <= 1 (of 255) at t=0, at the pause, at the closest pass
  (cups left / right / removed) and at every 2nd rollout step.
* R3: side vs every visible draw variable |r| <= 0.075 (360 draws, left 46.9 %).
* R4: every fixed tilt hits the cups in >= 45 % of draws (45 % = share of left draws for tilts
  <= -12.5; 55 % for >= -5; -10 / -7.5: 100 / 76 %); expert 0 %, opposite tilt 100 %.
* R5: at the pause (hand arrives step 43-48, tilts at 61-65) link6_sensor_3 reads LEFT cups at
  8.2-10.2 cm, link6_sensor_4 RIGHT cups at 5.0-6.7 cm, vs 16.7-17.8 cm without them.
* R7: expert 8/8 (noise off).

Collection-only additions (switched on by the collection configs, never by the eval config):
DART-style executed-action noise, the contact-tolerant save gate of ``pc_shelf_hard`` and one skin
reading per policy step. Per-link clearance to the cups is recorded in ``obs_scene['pc_clearance']``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.tasks.pc_shelf import (
    D_INT,
    MAT_HALF,
    MAT_HALF_T,
    N_CELLS,
    SCENE_DIR,
    PCShelfPnPSampler,
    _T,
)
from molmo_spaces.tasks.pc_shelf_blind import (
    CLEAR_BODIES,
    INS_CLEAR,
    GRASP_BELOW_TOP,
    PC_SHELF_BLIND_CELLS,
    READY_TCP,
    X_PRE,
    PCShelfBlindSampler,
    PCShelfBlindTask,
    blind_tool_rotation,
    build_blind_scene_xml,
    robot_vertex_sets,
)
from molmo_spaces.tasks.pc_sink_blind import (
    PCSinkBlindExpert,
    PCSinkBlindExpertConfig,
    primitive_distance,
)

log = logging.getLogger(__name__)

PC_OCC_PANTRY_ENV_VERSION = "pc_occ_pantry_v1"

PC_OCC_PANTRY_SCENES: tuple[str, ...] = tuple(
    str(SCENE_DIR / f"pc_occ_pantry_c{k}.xml") for k in range(N_CELLS)
)

TILT_DEG = 20.0
D_DIV = 0.16            # TCP depth of the common (untilted) insertion where the pause / tilt happens
DWELL_S = 1.0           # the common insertion runs at 0.11 m/s ('approach'): the hand arrives ~step 45 and
                        # tilts ~step 60, so the policy's step-50 chunk query sees the skin at the pause

# ---- the cups (THOR Mug_1, Y-up mesh: origin at the bottom centre, rim at y = 0.103, handle on +x)
CUP_UID = "Mug_1"
CUP_S = 0.55
CUP_R = 0.050 * CUP_S          # body radius
CUP_L = 0.103 * CUP_S          # bottom -> rim
CUP_HANDLE = 0.0756 * CUP_S    # handle top, radially from the axis
CUP_HANDLE_Y = 0.0645 * CUP_S  # handle centre along the axis (from the bottom)
HOOK_L = 0.010                 # hook: screw eye + curl under the board
CUP_GAP = 0.012                # between the two cups of a pair
PAIR_W = 2 * CUP_L + CUP_GAP   # lateral extent of a pair
CUP_DROP = HOOK_L + CUP_HANDLE + CUP_R   # board underside -> cup bottom
CUP_COLORS = (  # glaze rgba per cell
    (0.93, 0.92, 0.88), (0.20, 0.33, 0.55), (0.72, 0.18, 0.15), (0.25, 0.45, 0.35),
    (0.95, 0.80, 0.30), (0.12, 0.12, 0.13), (0.55, 0.70, 0.80), (0.85, 0.55, 0.40),
)
HAZ_BODY = "pact_intrusion_cups"
HAZ_BODIES = {1: HAZ_BODY + "_left", -1: HAZ_BODY + "_right"}
PARK_HAZ = {1: (-3.0, -1.6, 0.3), -1: (-3.0, -1.0, 0.3)}
Q_ALONG_Y = (0.70710678, 0.70710678, 0.0, 0.0)   # cylinder axis (local z) -> world y

# ---- pantry goods (visible, mocap): kind, half extents (x depth, y width, z), rgba
PANTRY_FRONT = (
    # <= 21.5 cm tall: with the hand tilted 20 deg the forearm passes ~25 cm above the board over
    # the front row while the hand grasps at the back (kinematic sweep 2026-10-15: 27 cm boxes came
    # within 3 mm of fr3_link5)
    ("box", (0.090, 0.031, 0.1075), (0.85, 0.65, 0.15)),  # cereal box (narrow side out)
    ("box", (0.090, 0.031, 0.1050), (0.70, 0.15, 0.12)),  # cereal box
    ("cyl", (0.048, 0.048, 0.1000), (0.80, 0.70, 0.45)),  # oat canister
    ("box", (0.085, 0.028, 0.1000), (0.20, 0.35, 0.65)),  # pasta box
)
PANTRY_BACK = (
    ("jar", (0.045, 0.045, 0.075), (0.62, 0.38, 0.14)),   # honey jar
    ("jar", (0.040, 0.040, 0.060), (0.85, 0.20, 0.15)),   # jam jar
    ("cyl", (0.034, 0.034, 0.058), (0.75, 0.76, 0.78)),   # tin
    ("cyl", (0.050, 0.050, 0.090), (0.30, 0.50, 0.32)),   # tea canister
)
PANTRY_UPPER = (
    ("box", (0.095, 0.034, 0.125), (0.92, 0.85, 0.65)),
    ("jar", (0.045, 0.045, 0.085), (0.60, 0.40, 0.20)),
    ("cyl", (0.040, 0.040, 0.065), (0.80, 0.10, 0.10)),
    ("box", (0.080, 0.050, 0.060), (0.35, 0.55, 0.30)),
    ("jar", (0.038, 0.038, 0.070), (0.90, 0.90, 0.88)),
)


def build_occ_pantry_scene_xml(cell_index: int) -> str:
    """pc_shelf_blind's cell XML, renamed (geometry identical)."""
    xml = build_blind_scene_xml(cell_index)
    xml = xml.replace(f'<mujoco model="pc_shelf_blind_c{cell_index}">',
                      f'<mujoco model="pc_occ_pantry_c{cell_index}">', 1)
    return xml.replace(
        "Generated by molmo_spaces.tasks.pc_shelf_blind.write_pc_shelf_blind_scenes",
        "Generated by molmo_spaces.tasks.pc_occ_pantry.write_pc_occ_pantry_scenes", 1)


def write_pc_occ_pantry_scenes() -> list[str]:
    """(Re)generate the eight cell XMLs + metadata sidecars. Never while a run uses them."""
    paths = []
    for k in range(N_CELLS):
        p = Path(PC_OCC_PANTRY_SCENES[k])
        p.write_text(build_occ_pantry_scene_xml(k))
        p.with_name(p.stem + "_metadata.json").write_text('{ "objects": {} }\n')
        paths.append(str(p))
    return paths


def _cup_mesh_files() -> list[str]:
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    d = Path(install_uid(CUP_UID)).parent / f"{CUP_UID}_mug_1"
    return [str(d / f"{CUP_UID}_mug_1_1.obj")]


def cup_pair_parts(side: int) -> list[dict[str, Any]]:
    """Primitives + mesh placements of one hook pair in the body frame (origin = inner hook's
    screw point on the board underside). side +1: the pair extends to +y (left of the lane)."""
    s = float(side)
    z_ax = -HOOK_L - CUP_HANDLE
    parts: list[dict[str, Any]] = []
    for k in range(2):
        y_bot = s * k * (CUP_L + CUP_GAP)           # cup bottom (inner end) along y
        y_hook = y_bot + s * CUP_HANDLE_Y
        y_mid = y_bot + s * CUP_L / 2
        parts.append(dict(name=f"hook{k}", kind="capsule", a=[0, y_hook, 0.0], b=[0, y_hook, -HOOK_L], r=0.0025))
        parts.append(dict(name=f"hookcurl{k}", kind="capsule", a=[-0.006, y_hook, -HOOK_L], b=[0.006, y_hook, -HOOK_L],
                          r=0.0022))
        parts.append(dict(name=f"cup{k}", kind="cylinder", c=[0, y_mid, z_ax], r=CUP_R, h=CUP_L / 2))
        parts.append(dict(name=f"handle{k}", kind="box", c=[0, y_hook, z_ax + 0.0632 * CUP_S],
                          half=[0.0078 * CUP_S, 0.0352 * CUP_S, 0.0128 * CUP_S]))
        # mesh: local x (handle) -> +z, local y (axis) -> s*y, local z -> x cross y
        Rm = np.stack([[0, 0, 1.0], [0, s, 0], np.cross([0, 0, 1.0], [0, s, 0])], 1)
        parts.append(dict(name=f"mesh{k}", kind="mesh", pos=[0, y_bot, z_ax],
                          quat=R.from_matrix(Rm).as_quat(scalar_first=True).tolist()))
    return parts


def _quat_z_to(v) -> list[float]:
    z = np.array([0.0, 0.0, 1.0])
    v = np.asarray(v, float) / np.linalg.norm(v)
    ax = np.cross(z, v)
    sn = float(np.linalg.norm(ax))
    if sn < 1e-9:
        return [1.0, 0.0, 0.0, 0.0] if v[2] > 0 else [0.0, 1.0, 0.0, 0.0]
    return R.from_rotvec(ax / sn * float(np.arctan2(sn, float(z @ v)))).as_quat(scalar_first=True).tolist()


def add_cup_pair_body(spec: MjSpec, side: int, rgba, mesh_name: str):
    name = HAZ_BODIES[side]
    b = spec.worldbody.add_body(name=name, mocap=True, pos=list(PARK_HAZ[side]))
    col = dict(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001], group=3)
    brass = [0.72, 0.58, 0.30, 1.0]
    for p in cup_pair_parts(side):
        nm = f"{name}_{p['name']}"
        if p["kind"] == "capsule":
            a, bb = np.asarray(p["a"], float), np.asarray(p["b"], float)
            L = float(np.linalg.norm(bb - a))
            # hooks: visible brass, also collision
            b.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[p["r"], L / 2, 0.0],
                       pos=((a + bb) / 2).tolist(), quat=_quat_z_to(bb - a), rgba=brass,
                       contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
        elif p["kind"] == "cylinder":
            b.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[p["r"], p["h"], 0.0],
                       pos=list(p["c"]), quat=list(Q_ALONG_Y), rgba=list(rgba) + [1.0], **col)
        elif p["kind"] == "box":
            b.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(p["half"]), pos=list(p["c"]),
                       rgba=list(rgba) + [1.0], **col)
        else:
            b.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name, pos=list(p["pos"]),
                       quat=list(p["quat"]), rgba=list(rgba) + [1.0], contype=0, conaffinity=0, group=0)
    return b


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
class PCOccPantrySampler(PCShelfBlindSampler):
    """pc_shelf2_blind's pick-and-place; pantry goods; hidden cup pair left or right."""

    PACT_PLACE_ENVIRONMENT_VERSION = PC_OCC_PANTRY_ENV_VERSION
    SUITE = "occ_pantry"
    # inner edge of the pair beside the lane centre (+1 = left, -1 = right; negative = reaches over
    # the lane centre), = pc_shelf2_blind's basket offsets
    CUP_GAP_LANE = {1: (0.078, 0.086), -1: (-0.024, -0.016)}
    CUP_D = (0.280, 0.300)        # hook line behind the board's front edge
    # visible goods keep this far (edge) off the lane centre: the tilted hand's link 7 reaches 13.8 cm
    # sideways while it places on the mat (+-2 cm off the lane), so the tall front row keeps 19 cm
    ITEM_LANE = {"f": 0.190, "b": 0.165}

    def _dyn_pool(self, cell_index: int) -> list[dict[str, Any]]:
        pool = []
        for tag, src in (("f", PANTRY_FRONT), ("b", PANTRY_BACK), ("u", PANTRY_UPPER)):
            for i, (kind, half, rgba) in enumerate(src):
                pool.append(dict(name=f"pact_clutter_pp{tag}{i}", kind="jar" if kind == "jar" else
                                 ("cyl" if kind == "cyl" else "box"), half=tuple(half), rgba=rgba, row=tag))
        return pool

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

        PickTaskSampler.add_auxiliary_objects(self, spec)
        k = self.current_house_index % N_CELLS
        self._house_cell = k
        self._dyn = self._dyn_pool(k)
        px, py, pz = self.PARK_XYZ
        for i, it in enumerate(self._dyn):
            b = spec.worldbody.add_body(name=it["name"], mocap=True, pos=[px, py + 0.3 * i, pz + it["half"][2] + 0.001])
            rgba = list(it["rgba"]) + [1.0]
            hx, hy, hz = it["half"]
            kw = dict(contype=8, conaffinity=15, friction=[0.6, 0.6, 0.001])
            if it["kind"] == "box":
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], rgba=rgba, **kw)
                # printed front panel (flat, contactless) on the face toward the robot
                b.add_geom(name=it["name"] + "_label", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[0.0008, hy * 0.8, hz * 0.45], pos=[-hx - 0.0008, 0.0, hz * 0.1],
                           rgba=[0.96, 0.95, 0.90, 1.0], contype=0, conaffinity=0)
            else:
                b.add_geom(name=it["name"] + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx, hz, 0.0], rgba=rgba, **kw)
                lid = [0.15, 0.15, 0.16, 1.0] if it["kind"] == "jar" else [0.78, 0.79, 0.80, 1.0]
                b.add_geom(name=it["name"] + "_lid", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[hx * 1.03, 0.007, 0.0], pos=[0.0, 0.0, hz - 0.006], rgba=lid, **kw)
                b.add_geom(name=it["name"] + "_label", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[hx * 1.004, hz * 0.35, 0.0], pos=[0.0, 0.0, -hz * 0.15],
                           rgba=[0.95, 0.93, 0.86, 1.0], contype=0, conaffinity=0)
        self._add_task_bodies(spec, k)

    def _add_task_bodies(self, spec: MjSpec, cell_index: int) -> None:
        PCShelfPnPSampler._add_task_bodies(self, spec, cell_index)
        mesh_name = "pc_occ_cup_mesh"
        files = _cup_mesh_files()
        spec.add_mesh(name=mesh_name, file=files[0], scale=[CUP_S, CUP_S, CUP_S])
        rgba = CUP_COLORS[cell_index % len(CUP_COLORS)]
        for sd in (1, -1):
            add_cup_pair_body(spec, sd, rgba, mesh_name)

    def _place_row(self, avail, placed, keep, bay, x_lo, x_hi, y_side, yc, row, out):
        """One item of ``row`` on side y_side (+1 / -1) between depths x_lo..x_hi, off the lane."""
        cands = [i for i in avail if self._dyn[i]["row"] == row]
        np.random.shuffle(cands)
        for i in cands:
            hx, hy, hz = self._dyn[i]["half"]
            for _ in range(12):
                cx = float(np.random.uniform(x_lo + hx, max(x_lo + hx, x_hi - hx)))
                if y_side > 0:
                    ylo, yhi = yc + self.ITEM_LANE[row] + hy, bay["y_hi"] - hy
                else:
                    ylo, yhi = bay["y_lo"] + hy, yc - self.ITEM_LANE[row] - hy
                if yhi < ylo:
                    break
                cy = float(np.random.uniform(ylo, yhi))
                if self._fits(bay, (cx, cy, hx, hy), hz, placed, keep):
                    avail.remove(i)
                    placed.append((cx, cy, hx, hy))
                    out.append((i, [cx, cy, bay["z"] + hz + 0.0005]))
                    return True
        return False

    def cup_origin(self, c, yc: float, side: int, gap: float, d: float) -> list[float]:
        """World position of the pair's inner hook screw point (board underside)."""
        ceil = c.b_lo + c.bay_lo
        return [c.x0 + d, yc + side * gap, ceil]

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
        # hidden cup pair: side drawn independently of everything visible
        side = 1 if np.random.random() < 0.5 else -1          # +1 = left (+y), -1 = right (-y)
        gap = float(np.random.uniform(*self.CUP_GAP_LANE[side]))
        dcup = float(np.random.uniform(*self.CUP_D))
        origin = self.cup_origin(c, yc, side, gap, dcup)
        y_out = origin[1] + side * PAIR_W
        lim = c.bay_w / 2 - 0.008
        if abs(y_out) > lim:
            return None
        tilt = TILT_DEG * side       # approach leans toward the cups' side -> wrist swings away
        z_hi = z_b + h + INS_CLEAR
        z_g = z_b + h - GRASP_BELOW_TOP
        z_p = mat_top + h - GRASP_BELOW_TOP + 0.004
        x_pre = c.x0 - X_PRE
        x_div = c.x0 + D_DIV
        wps = [
            ("pregrasp", [x_pre, yc, z_hi], "fast", 0.0),
            ("pregrasp", [x_div, yc, z_hi], "approach", 0.0),
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
        # pantry goods: a tall item at the front and a short one at the back on each side of the
        # lane, three in the bay above (positions never depend on the cup side)
        avail = list(range(len(self._dyn)))
        out: list = []
        bay = self._bay(z_b, c.bay_lo)
        can_fp = (xc, yc, r + 0.03, r + 0.03)
        mat_fp = (xm, ym, MAT_HALF + 0.02, MAT_HALF + 0.02)
        placed: list = []
        for sd in (1, -1):
            self._place_row(avail, placed, [can_fp, mat_fp], bay, c.x0 + 0.008, c.x0 + 0.13, sd, yc, "f", out)
            self._place_row(avail, placed, [can_fp, mat_fp], bay, c.x0 + 0.20, c.x0 + D_INT - 0.006, sd, yc, "b", out)
        bay_hi = self._bay(c.b_hi, c.bay_hi)
        placed_hi: list = []
        for _ in range(3):
            self._random_item(avail, placed_hi, [], bay_hi, out, kinds=("box", "jar", "cyl"))
        self._apply_layout(env, out)
        self._mocap(env, "place_receptacle_mat", [xm, ym, z_b + MAT_HALF_T + 0.0003])
        for sd in (1, -1):
            self._mocap(env, HAZ_BODIES[sd], PARK_HAZ[sd])
        self._mocap(env, HAZ_BODIES[side], origin)
        quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
        self._settle_target(env, [xc, yc, z_b + h / 2 + 0.002], quat)
        mujoco.mj_forward(env.current_model, env.current_data)
        side_name = "left" if side > 0 else "right"
        y_mid = origin[1] + side * PAIR_W / 2
        return dict(
            **self._cell_common(),
            pick_board_z=z_b, place_board_z=z_b, pick_upper=False,
            can_radius=r, can_height=h, can_label=td["label"],
            start_xyz=[xc, yc, z_b + h / 2], goal_xyz=[xm, ym, mat_top + h / 2],
            mat_xyz=[xm, ym, z_b + MAT_HALF_T], mat_top=mat_top, place_tol=self.PLACE_TOL,
            expert=[[n, p, s, t] for (n, p, s, t) in wps], dwell_s=DWELL_S,
            neighbours=[[self._dyn[i]["name"], self._dyn[i]["kind"], pos,
                         list(self._dyn[i]["half"])] for i, pos in out],
            n_neighbours=len(out),
            # evaluator contract (eval_act_place.py writes hidden_hazard_present / _y / _side)
            hidden_hazard=dict(present=True, side=side_name, y=float(y_mid), x=float(origin[0]),
                               z_bottom=float(ceil - CUP_DROP), body=HAZ_BODIES[side]),
            # diagnostic free camera (reports/2026-10-13/video_blind.py): low in the bay's far
            # corner on the other side, looking up at the cups and the wrist
            reveal_view=dict(pos=[c.x0 + 0.36, yc - side * 0.20, z_b + 0.12],
                             lookat=[c.x0 + 0.24, yc + side * 0.06, ceil - 0.06], fovy=80.0,
                             title="Inside the bay (not a robot camera): cups in red"),
            blind=dict(side=int(side), side_name=side_name, tilt_deg=float(tilt), cup_origin=origin,
                       cup_gap=gap, cup_depth=dcup, ceiling_z=ceil, z_carry=z_hi, cell=int(k)),
        )

    def _referral(self) -> str:
        return self._target_spec.get("label", "can")

    def _task_class(self):
        return PCOccPantryTask


# --------------------------------------------------------------------------------------------
# Task: pc_shelf_blind (pnp success, hazard audit, gate) with clearance to the cup primitives
# --------------------------------------------------------------------------------------------
class PCOccPantryTask(PCShelfBlindTask):
    def _blind_setup(self, model) -> None:
        robot_root = self.env.current_robot.robot_view.base.root_body_id
        hg = [g for g in range(model.ngeom)
              if (model.body(int(model.geom_bodyid[g])).name or "").startswith(HAZ_BODY)
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
        sp = getattr(self, "scene_params", {}) or {}
        body = (sp.get("hidden_hazard") or {}).get("body", HAZ_BODY)
        self._bclr = dict(hg=hg, sets=robot_vertex_sets(model, robot_root, keys=CLEAR_BODIES), best={},
                          body=body)

    def _blind_step(self, data) -> None:
        model = data.model
        if getattr(self, "_bclr", None) is None or self.episode_step_count == 0:
            self._blind_setup(model)
        if self.episode_step_count % self.CLEAR_EVERY_BLIND:
            return
        bc = self._bclr
        for key, dist in primitive_distance(data, bc["sets"], bc["hg"]).items():
            prev = bc["best"].get(key)
            if dist < 0.4 and (prev is None or dist < prev[0]):
                bc["best"][key] = (dist, bc["body"], int(self.episode_step_count))


class PCOccPantryExpert(PCSinkBlindExpert):
    """pc_shelf_blind expert (per-waypoint hand tilt), dwell from scene_params['dwell_s'], plus the
    'approach' speed of the common untilted insertion."""

    SPEEDS = dict(PCSinkBlindExpert.SPEEDS, approach=0.11)


class PCOccPantryExpertConfig(PCSinkBlindExpertConfig):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PCOccPantryExpert


__all__ = [
    "PC_OCC_PANTRY_ENV_VERSION", "PC_OCC_PANTRY_SCENES", "PCOccPantrySampler", "PCOccPantryTask",
    "PCOccPantryExpert", "PCOccPantryExpertConfig", "write_pc_occ_pantry_scenes", "cup_pair_parts",
    "HAZ_BODY", "HAZ_BODIES", "PARK_HAZ", "TILT_DEG", "D_DIV", "CUP_DROP", "PAIR_W",
    "PC_SHELF_BLIND_CELLS",
]


if __name__ == "__main__":
    import sys

    if "--write-scenes" in sys.argv:
        for p in write_pc_occ_pantry_scenes():
            print(p)
