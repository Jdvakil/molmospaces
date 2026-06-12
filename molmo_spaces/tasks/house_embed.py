"""Embed the parameterized reach tasks (enclosure / fumehood / panel / cubby) inside REAL
ProcTHOR / iThor houses for naturalism (advisor: "robot in a house"), WITHOUT changing the
task geometry the user likes and WITHOUT camera occlusion.

Mechanism
---------
The loaded *scene* is a real house. In `add_auxiliary_objects(house_spec)` we:
  1. discover an open floor spot + facing yaw whose robot-mounted EXO camera looks INTO the
     room (never into a wall) and whose task footprint is clear of house furniture
     (`_discover_floor`, cached per house);
  2. replay the standalone task scene's worldbody geoms (static furniture) and mocap bodies
     (the parameterized aperture / obstacle slabs) straight into the house worldbody, rigidly
     transformed by the discovered pose `(O, yaw)` — no mesh/texture copy needed (task furniture
     is simple rgba boxes; the HOUSE supplies the realism);
  3. let the base sampler inject the graspable target at the (world) rest spot.

The robot is placed at `(O, yaw)`; per-episode mocap posing transforms task-local targets to
world (`_mocap_set` override). The expert is made embed-aware separately (see EnclosureExpert):
every commanded pose is mapped local->world by the same transform, while proximity margins stay
in the task-local frame. Standalone runs use the identity transform, so nothing changes there.
"""
from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

log = logging.getLogger(__name__)


def yaw_quat(yaw: float) -> np.ndarray:
    return R.from_euler("z", yaw).as_quat(scalar_first=True)


def embed_T(bx: float, by: float, yaw: float) -> np.ndarray:
    """4x4 world<-local rigid transform (rotation about z by yaw, then translate to (bx,by,0))."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = np.array([bx, by, 0.0])
    return T


class HouseEmbeddedMixin:
    """Mix INTO a reach-task sampler (after it, so the task's _draw_theta/_apply_theta win).

    Subclass must set FURNITURE_XML (path to the standalone task scene whose worldbody is
    replayed) and may tune FOOTPRINT / EXO_OFFSET / WALL_BEHIND_LOCAL.
    """

    FURNITURE_XML: str = ""
    # task footprint in the LOCAL frame (robot at origin facing +x): keep clear of house clutter
    FOOTPRINT = dict(x0=-0.45, x1=1.55, y0=-0.70, y1=0.70, step=0.22, clear=0.16)
    EXO_OFFSET = (0.10, 0.57)          # robot-mounted exo cam xy offset in base frame
    WALL_BEHIND_LOCAL = (1.75, 0.0)    # a wall here (behind the furniture) looks natural
    SKIP_REPLAY_NAMES = ("floor",)     # house provides the floor; never replay our own plane

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._floor_cache: dict[str, dict] = {}
        self._embed: tuple[float, float, float] | None = None

    # ---------------- floor / orientation discovery ----------------
    @staticmethod
    def _world_box(m, d, g):
        """World axis-aligned (xmin,xmax,ymin,ymax,zmin,zmax) of geom g from its AABB corners."""
        aabb = m.geom_aabb[g]
        c, h = aabb[:3], aabb[3:]
        xp = d.geom_xpos[g]
        xm = d.geom_xmat[g].reshape(3, 3)
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(xp + xm @ (c + np.array([sx, sy, sz]) * h))
        corners = np.array(corners)
        return (corners[:, 0].min(), corners[:, 0].max(),
                corners[:, 1].min(), corners[:, 1].max(),
                corners[:, 2].min(), corners[:, 2].max())

    @staticmethod
    def _rect_dist(px, py, rect) -> float:
        """xy distance from a point to a (xmin,xmax,ymin,ymax) rectangle (0 if inside)."""
        xmin, xmax, ymin, ymax = rect[0], rect[1], rect[2], rect[3]
        dx = max(xmin - px, 0.0, px - xmax)
        dy = max(ymin - py, 0.0, py - ymax)
        return float(np.hypot(dx, dy))

    def _discover_floor(self, house_xml: str) -> dict:
        if house_xml in self._floor_cache:
            return self._floor_cache[house_xml]
        m = mujoco.MjModel.from_xml_path(house_xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        room_rects, obst_rects = [], []
        for g in range(m.ngeom):
            bname = m.body(int(m.geom_bodyid[g])).name.lower()
            if bname == "world" or m.geom(g).type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            box = self._world_box(m, d, g)
            if bname.startswith("room"):                # floor tile = interior region
                room_rects.append(box)
            elif box[5] > 0.05 and box[4] < 1.7:        # occupies the navigable band -> obstacle
                obst_rects.append(box)
        if not room_rects:
            self._floor_cache[house_xml] = {}
            return {}

        def on_floor(px, py, margin=0.10):
            return any(self._rect_dist(px, py, r) <= -1e-9 or
                       (r[0] + margin <= px <= r[1] - margin and r[2] + margin <= py <= r[3] - margin)
                       for r in room_rects)

        def min_obst(px, py):
            return min((self._rect_dist(px, py, r) for r in obst_rects), default=9.9)

        fp = self.FOOTPRINT
        lx = np.arange(fp["x0"], fp["x1"] + 1e-6, fp["step"])
        ly = np.arange(fp["y0"], fp["y1"] + 1e-6, fp["step"])
        local_pts = np.array([[a, b] for a in lx for b in ly])

        def evaluate(bx, by, yaw):
            c, s = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[c, -s], [s, c]])
            world = (Rz @ local_pts.T).T + np.array([bx, by])
            for w in world:
                if not on_floor(w[0], w[1]) or min_obst(w[0], w[1]) < fp["clear"]:
                    return None                          # footprint must be inside a room AND clear
            exo = Rz @ np.array(self.EXO_OFFSET) + np.array([bx, by])
            if not on_floor(exo[0], exo[1]) or min_obst(exo[0], exo[1]) < 0.25:
                return None                              # exo cam must look into open room
            wb = Rz @ np.array(self.WALL_BEHIND_LOCAL) + np.array([bx, by])
            wall_bonus = 1.0 if min_obst(wb[0], wb[1]) < 0.4 else 0.0
            # prefer interior spots: maximize clearance to nearest obstacle, plus wall-behind bonus
            return -min(min_obst(bx, by), 1.0) - wall_bonus

        xs = [r[0] for r in room_rects] + [r[1] for r in room_rects]
        ys = [r[2] for r in room_rects] + [r[3] for r in room_rects]
        gx = np.arange(min(xs) + 0.4, max(xs) - 0.4, 0.30)
        gy = np.arange(min(ys) + 0.4, max(ys) - 0.4, 0.30)
        best = None
        for bx in gx:
            for by in gy:
                for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2,
                            np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4):
                    score = evaluate(bx, by, yaw)
                    if score is None:
                        continue
                    if best is None or score < best[0]:
                        best = (score, float(bx), float(by), float(yaw))
        if best is None:
            self._floor_cache[house_xml] = {}
            return {}
        _, bx, by, yaw = best
        info = dict(base=(bx, by, yaw))
        log.info(f"[HouseEmbed] {house_xml.split('/')[-1]}: base=({bx:.2f},{by:.2f},"
                 f"yaw={np.degrees(yaw):.0f}) score={best[0]:.2f} rooms={len(room_rects)}")
        self._floor_cache[house_xml] = info
        return info

    def _cur_embed(self) -> tuple[float, float, float]:
        if self._embed is not None:
            return self._embed
        path = str(self._current_house_scene_path(variant="base"))
        info = self._discover_floor(path)
        if not info:
            raise HouseInvalidForTask("no clear task footprint in this house")
        self._embed = info["base"]
        return self._embed

    # ---------------- local<->world helpers ----------------
    def _T(self) -> np.ndarray:
        return embed_T(*self._cur_embed())

    def _to_world(self, local_xyz) -> list[float]:
        w = self._T() @ np.array([local_xyz[0], local_xyz[1], local_xyz[2], 1.0])
        return [float(w[0]), float(w[1]), float(w[2])]

    # mocap posing in WORLD: _apply_theta passes LOCAL targets; transform them here
    def _mocap_set(self, env, body, pos):
        m, d = env.current_model, env.current_data
        mid = int(m.body_mocapid[m.body(body).id])
        d.mocap_pos[mid] = np.asarray(self._to_world(pos), dtype=float)

    # ---------------- replay task furniture into the house spec ----------------
    def _replay_furniture(self, spec: MjSpec) -> None:
        bx, by, yaw = self._cur_embed()
        T = embed_T(bx, by, yaw)
        q_yaw = yaw_quat(yaw)
        child = MjSpec.from_file(self.FURNITURE_XML)
        wb = spec.worldbody

        def world_pos(p):
            w = T @ np.array([p[0], p[1], p[2], 1.0])
            return [float(w[0]), float(w[1]), float(w[2])]

        def compose_quat(local_q):
            lq = np.array(local_q, dtype=float)
            if np.linalg.norm(lq) < 1e-9:
                lq = np.array([1.0, 0.0, 0.0, 0.0])
            comp = R.from_quat(q_yaw, scalar_first=True) * R.from_quat(lq, scalar_first=True)
            return comp.as_quat(scalar_first=True).tolist()

        # static worldbody geoms (skip the floor plane; house has its own)
        for g in child.worldbody.geoms:
            if any(s in (g.name or "").lower() for s in self.SKIP_REPLAY_NAMES):
                continue
            if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            ng = wb.add_geom()
            ng.name = f"task_{g.name}" if g.name else ""
            ng.type = g.type
            ng.size = np.array(g.size)
            ng.pos = world_pos(np.array(g.pos))
            ng.quat = compose_quat(g.quat)
            ng.rgba = np.array(g.rgba)
            ng.contype, ng.conaffinity = 8, 15
            ng.group = 0

        # mocap bodies (aperture slabs / obstacles): recreate as worldbody mocap bodies so
        # per-episode _mocap_set keeps working; geoms inside stay at body-local offset.
        for b in child.worldbody.bodies:
            if not b.mocap:
                continue
            nb = wb.add_body()
            nb.name = b.name
            nb.mocap = True
            nb.pos = world_pos(np.array(b.pos))
            nb.quat = q_yaw.tolist()
            for g in b.geoms:
                ng = nb.add_geom()
                ng.name = f"{b.name}_g" if not g.name else g.name
                ng.type = g.type
                ng.size = np.array(g.size)
                ng.pos = np.array(g.pos)
                ng.quat = (np.array(g.quat) if np.linalg.norm(g.quat) > 1e-9
                           else np.array([1.0, 0.0, 0.0, 0.0]))
                ng.rgba = np.array(g.rgba)
                ng.contype, ng.conaffinity = 8, 15
                ng.group = 0
        log.info(f"[HouseEmbed] replayed furniture from {self.FURNITURE_XML.split('/')[-1]}")

    # ---------------- dataset index map (houses use the stock mapping) ----------------
    def _get_dataset_index_map(self) -> dict:
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
        if self.config.scene_dataset != "user":
            return PickTaskSampler._get_dataset_index_map(self)
        return super()._get_dataset_index_map()

    # ---------------- hooks ----------------
    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        self._embed = None                         # re-discover per (possibly new) house
        self._cur_embed()                          # raises HouseInvalidForTask to skip a bad house
        self._replay_furniture(spec)
        super().add_auxiliary_objects(spec)        # injects the graspable target at _obj_rest()

    def _obj_rest(self):
        local = super()._obj_rest()
        return tuple(self._to_world(local))

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        bx, by, yaw = self._cur_embed()
        self._cur_base_xyz = (bx + float(np.random.uniform(-0.02, 0.02)),
                              by + float(np.random.uniform(-0.02, 0.02)), 0.0)
        self._cur_base_yaw = yaw + float(np.random.uniform(-0.04, 0.04))
        super()._sample_and_place_robot(env)

    def _sample_task(self, env: CPUMujocoEnv):
        task = super()._sample_task(env)
        # stamp the embed transform so the expert can map task-local poses -> world
        if getattr(task, "scene_params", None) is not None and self._embed is not None:
            task.scene_params["embed"] = list(self._cur_embed())
        return task


# ---------------------------------------------------------------------------------------------
# Concrete house-embedded task samplers. MRO: HouseEmbeddedMixin FIRST so its hooks win, then
# the standalone task sampler supplies _draw_theta / _apply_theta / expert pairing unchanged.
# ---------------------------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

from molmo_spaces.tasks.enclosure_reach import (  # noqa: E402
    FumehoodSampler,
    PanelSlalomSampler,
    CubbyOverreachSampler,
)

_SCENES = Path(__file__).resolve().parent.parent / "data_generation" / "custom_scenes"


class HouseFumehoodSampler(HouseEmbeddedMixin, FumehoodSampler):
    FURNITURE_XML = str(_SCENES / "fumehood.xml")


class HousePanelSlalomSampler(HouseEmbeddedMixin, PanelSlalomSampler):
    FURNITURE_XML = str(_SCENES / "panel_slalom.xml")
    # the slalom table is wider but shallow; keep the lateral band tighter so it still fits
    # ordinary rooms, with a bit more clearance margin than the default
    FOOTPRINT = dict(x0=-0.45, x1=1.55, y0=-0.62, y1=0.62, step=0.22, clear=0.14)


class HouseCubbyOverreachSampler(HouseEmbeddedMixin, CubbyOverreachSampler):
    FURNITURE_XML = str(_SCENES / "cubby_overreach.xml")
    FOOTPRINT = dict(x0=-0.45, x1=1.3, y0=-0.65, y1=0.65, step=0.20, clear=0.16)
