"""Pick-an-object-out-of-a-cabinet/drawer-cavity task sampler.

A purpose-built proximity-NECESSITY environment that does NOT use predefined houses.
The scene is a hand-authored cabinet cavity (see data_generation/custom_scenes/); a
graspable objaverse object is injected resting inside it, and the Franka arm must reach
INTO the cavity to grasp it — so the 29 SPAD proximity sensors are encapsulated by the
cavity walls throughout the approach/grasp (validated in scripts/cavity_scene.py:
~25/29 sensors active, median proximity depth ~0.19 m vs ~0.53 m in houses).

It reuses the stock privileged planner + grasp DB + rollout + h5 saving unchanged.
Three things differ from a house-based pick task, each isolated in an override:

  1. add_auxiliary_objects   inject one graspable objaverse object into the cavity
                             (+ _get_scene_objects returns exactly that object).
  2. _get_dataset_index_map  custom "user" scenes only have a "base" variant, but the
                             pipeline requests "ceiling" — map every variant to the file.
  3. _sample_and_place_robot custom scenes have no occupancy map (env.get_thormap raises),
                             so set the robot base pose directly instead of place_robot_near.
"""
from __future__ import annotations

import numpy as np
from mujoco import MjSpec, mjtJoint
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.object_metadata import ObjectMeta
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_7d
from molmo_spaces.utils.synset_utils import get_valid_pickupable_obja_uids

import logging

log = logging.getLogger(__name__)


class CavityPickTaskSampler(PickTaskSampler):
    """Pick a graspable object out of a hand-authored cabinet cavity."""

    # Cavity interior CENTER in world meters (must match the cavity body pos in the scene XML).
    CAVITY_CENTER = (0.5, 0.0, 0.8)
    CAVITY_INTERIOR = (0.26, 0.24, 0.22)  # interior HALF-extents (match the XML)
    # Robot base pose (xyz + wxyz quat). Identity quat => robot +x faces the cavity opening (-x).
    BASE_XYZ = (0.0, 0.0, 0.0)
    BASE_QUAT = (1.0, 0.0, 0.0, 0.0)
    POOL_SIZE = 12  # number of distinct objaverse objects to cycle across houses
    # Object rest placement. If OBJ_XY/OBJ_FLOOR_Z are None they default to the cavity center/floor;
    # shelf-style subclasses set them to drop the object deep at the back so the arm reaches all
    # the way in. OBJ_JIT_XY = per-episode (x,y) randomization half-ranges.
    OBJ_XY = None
    OBJ_FLOOR_Z = None
    OBJ_JIT_XY = (0.06, 0.08)
    # Shape filter: keep objects that actually have a collision-free grasp inside the cavity.
    MAX_OBJ_DIM = 0.16   # too big to fit + be grasped in the cavity
    MIN_OBJ_DIM = 0.03   # too thin to grasp
    MIN_FLATNESS = 0.30  # min/max bbox ratio; flat plates/sheets (~<0.3) have 0 grasps in-cavity

    def __init__(self, config) -> None:
        super().__init__(config)
        self._injected_obj_name: str | None = None
        self._cur_base_xyz = self.BASE_XYZ
        self._cur_base_yaw = 0.0
        # Build a pool of objaverse UIDs that (a) have grasps and (b) are chunky enough to be
        # grasped inside the cavity. One UID per house index gives object variety across the run.
        self._uid_pool = self._build_grasp_uid_pool(self.POOL_SIZE)
        log.info(f"[CavityPick] grasp-validated UID pool ({len(self._uid_pool)}): {self._uid_pool}")

    def _obj_rest(self):
        """(x, y, floor_z) for the object's resting spot. Defaults to cavity center/floor."""
        x, y = self.OBJ_XY if self.OBJ_XY is not None else (self.CAVITY_CENTER[0], self.CAVITY_CENTER[1])
        fz = self.OBJ_FLOOR_Z if self.OBJ_FLOOR_Z is not None else (self.CAVITY_CENTER[2] - self.CAVITY_INTERIOR[2])
        return x, y, fz

    def _passes_shape_filter(self, uid: str) -> bool:
        anno = ObjectMeta.annotation(uid) or {}
        bb = anno.get("boundingBox", {})
        dims = sorted(float(bb.get(k, 0.0)) for k in "xyz")
        if dims[2] <= 0:
            return True  # unknown bbox -> don't exclude (grasp check is the backstop)
        if dims[2] > self.MAX_OBJ_DIM or dims[0] < self.MIN_OBJ_DIM:
            return False
        return (dims[0] / dims[2]) >= self.MIN_FLATNESS

    # target categories to skip entirely (e.g. eggs slip out of the gripper at lift and
    # burn the whole per-house attempt budget — observed repeatedly with Egg_4)
    EXCLUDED_TARGET_CATEGORIES: tuple[str, ...] = ()

    def _build_grasp_uid_pool(self, n: int) -> list[str]:
        from molmo_spaces.utils.grasp_sample import has_grasp_folder, has_valid_grasp_file
        pool: list[str] = []
        candidates = list(get_valid_pickupable_obja_uids())
        np.random.shuffle(candidates)
        for uid in candidates:
            try:
                if not self._passes_shape_filter(uid):
                    continue
                cat = str((ObjectMeta.annotation(uid) or {}).get("category", "")).lower()
                if any(x in cat or x in uid.lower() for x in self.EXCLUDED_TARGET_CATEGORIES):
                    continue
                if has_grasp_folder(uid) and has_valid_grasp_file(uid):
                    pool.append(uid)
            except Exception:
                continue
            if len(pool) >= n:
                break
        if not pool:
            pool = candidates[:n]
        return pool

    # --- 1. inject the graspable object into the cavity --------------------------------- #
    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        super().add_auxiliary_objects(spec)  # policy aux + (no-op: added_pickup_objects is None)
        uid = self._uid_pool[self.current_house_index % len(self._uid_pool)]
        pickup_xml = install_uid(uid)
        pspec = MjSpec.from_file(str(pickup_xml))
        pbody = pspec.worldbody.bodies[0]
        if not pbody.first_joint():
            pbody.add_joint(name=f"{uid}_jntfree", type=mjtJoint.mjJNT_FREE,
                            damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING)
        anno = ObjectMeta.annotation(uid) or {}
        bbox = anno.get("boundingBox", {})
        z_half = float(bbox.get("z", 0.06)) / 2 + 0.01
        ox, oy, floor_z = self._obj_rest()
        pos = [ox, oy, floor_z + z_half]
        quat = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
        ns = "cavity_obj_0/"
        orig = pbody.name
        spec.worldbody.add_frame(pos=pos, quat=quat).attach_body(pbody, ns, "")
        self._injected_obj_name = ns + orig
        self._metadata_adder.update({
            self._injected_obj_name: {
                "asset_id": uid,
                "category": anno.get("category", "object"),
                "object_enum": "temp_object",
                "is_static": False,
                "boundingBox": bbox,
            }
        })
        log.info(f"[CavityPick] injected '{self._injected_obj_name}' (uid={uid}) at {np.round(pos,3)}")

    # --- settle the injected object onto the cavity floor before sampling -------------- #
    def _settle_injected_object(self, env: CPUMujocoEnv) -> None:
        """Randomize the object's start (XY within the cavity + yaw) and step the sim so it
        rests IN CONTACT on the floor (else get_supporting_geom returns nothing and it's
        dropped). Isolate the object: step everything, then restore all DOFs EXCEPT the
        object's free joint, so the uncontrolled arm doesn't droop into a bad pose."""
        import mujoco
        m, d = env.current_model, env.current_data
        bid = m.body(self._injected_obj_name).id
        jadr = int(m.body_jntadr[bid])
        if jadr < 0:
            mujoco.mj_forward(m, d)
            return
        qadr = int(m.jnt_qposadr[jadr])
        qpos0 = d.qpos.copy()
        bx, by, floor_z = self._obj_rest()
        jx, jy = self.OBJ_JIT_XY
        ox = bx + float(np.random.uniform(-jx, jx))
        oy = by + float(np.random.uniform(-jy, jy))
        yaw = float(np.random.uniform(0, 2 * np.pi))
        q = (R.from_euler("z", yaw) * R.from_euler("x", 90, degrees=True)).as_quat(scalar_first=True)
        d.qpos[qadr:qadr + 3] = [ox, oy, floor_z + 0.12]  # drop from ~12 cm up
        d.qpos[qadr + 3:qadr + 7] = q
        d.qvel[:] = 0.0
        for _ in range(180):
            mujoco.mj_step(m, d)
        obj_qpos = d.qpos[qadr:qadr + 7].copy()
        d.qpos[:], d.qvel[:] = qpos0, 0.0
        d.qpos[qadr:qadr + 7] = obj_qpos
        mujoco.mj_forward(m, d)

    def _sample_task(self, env: CPUMujocoEnv):
        # per-episode robot base jitter -> varied approach angle -> richer sensor data
        self._cur_base_xyz = (
            self.BASE_XYZ[0] + float(np.random.uniform(-0.02, 0.05)),
            self.BASE_XYZ[1] + float(np.random.uniform(-0.05, 0.05)),
            self.BASE_XYZ[2],
        )
        self._cur_base_yaw = float(np.random.uniform(-0.20, 0.20))
        self._settle_injected_object(env)
        # (re)populate candidates each attempt so scene reuse can't strand us empty
        self.candidate_objects = self._get_scene_objects(env)
        return super()._sample_task(env)

    # --- return exactly the injected object as the only pick candidate ------------------ #
    def _get_scene_objects(self, env: CPUMujocoEnv, mass_limit=100):
        om = env.object_managers[env.current_batch_index]
        obj = om.get_object_by_name(self._injected_obj_name)
        if obj is None:
            from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
            raise HouseInvalidForTask(f"injected cavity object '{self._injected_obj_name}' not found")
        return [obj]

    # --- 2. custom user scenes: every variant maps to the one base file ----------------- #
    def _get_dataset_index_map(self) -> dict:
        if self._dataset_index_map is not None:
            return self._dataset_index_map
        paths = self.config.task_sampler_config.scene_xml_paths
        split = self.config.data_split
        mapping = {split: {i: {"base": p, "ceiling": p, "map": p} for i, p in enumerate(paths)}}
        self._dataset_index_map = mapping
        return mapping

    # --- 3. set the base pose directly (no occupancy map for custom scenes) ------------- #
    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        import mujoco
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()

        robot_view = env.current_robot.robot_view
        base_quat = R.from_euler("z", self._cur_base_yaw).as_quat(scalar_first=True)
        robot_view.base.pose = pos_quat_to_pose_mat(
            np.array(self._cur_base_xyz, dtype=float), np.array(base_quat, dtype=float)
        )
        mujoco.mj_forward(env.current_model, env.current_data)

        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
        goal = pose_mat_to_7d(pickup_obj.pose)
        goal[2] += 0.10  # lift target 10 cm out of the cavity
        task_cfg.pickup_obj_goal_pose = goal.tolist()
        log.info(f"[CavityPick] base set to {self.BASE_XYZ}; pick '{task_cfg.pickup_obj_name}'")


class CavityPickTaskSamplerV2(CavityPickTaskSampler):
    """v2: slightly wider cavity (easier grasps, still encapsulating) + bigger object pool.
    Pair with cabinet_cavity_v2.xml whose interior matches CAVITY_INTERIOR below."""

    CAVITY_INTERIOR = (0.29, 0.27, 0.24)  # wider than v1 (0.26,0.24,0.22)
    POOL_SIZE = 24


class ShelfReachPickTaskSampler(CavityPickTaskSampler):
    """Reach-into-a-shelf/cupboard task: the arm goes deep into an enclosed cupboard (closed
    back/top/bottom/sides + a front wall with an entry hole + a rear panel behind the shoulder)
    to grasp an object at the back. Tuned so ALL 29 SPAD sensors are active >50% of the time
    (validated in scripts/shelf_design.py: 29/29, min ~0.80, mean ~0.97). Pair with shelf_reach.xml.
    """

    # Object rests at a comfortably-reachable depth in the shelf (deep enough that the whole arm
    # is inside -> all 29 sensors active, but within the arm's dexterous range so IK succeeds).
    OBJ_XY = (0.60, 0.0)
    OBJ_FLOOR_Z = 0.44
    OBJ_JIT_XY = (0.06, 0.08)
    POOL_SIZE = 24


class RealTablePickTaskSampler(CavityPickTaskSampler):
    """Realistic tabletop: pick the target off a real wooden table amid PHYSICAL real-object
    clutter (objaverse meshes, gravity-settled on the table), with a shelf unit and walls behind.
    Collision-aware planning stays ON, so demos genuinely avoid the clutter — the proximity skin
    records the obstacles being avoided. Pair with table_shelf_real.xml (table top z=0.46).

    Built after advisor feedback: no floating ghost geometry, no camera-occlusion gymnastics —
    clutter rests on surfaces like a real lab tabletop, which also keeps the exo view clear."""

    OBJ_XY = (0.50, 0.0)        # fallback target spot (overridden by the scene's target_spot site)
    OBJ_FLOOR_Z = 0.46
    OBJ_JIT_XY = (0.05, 0.10)
    POOL_SIZE = 24
    N_CLUTTER = 18              # DENSE physical clutter (advisor: skin must matter for the task)
    N_CORRIDOR = 6              # of which: ARM-HEIGHT obstacles crowding the robot->target corridor
    CLUTTER_RING = (0.10, 0.42) # radial band around the target (m)
    TALL_Z = 0.13               # min bbox height for the medium "tall" pickup-pool subset
    # arm-height obstacle pool (vases, lamps, lanterns, bottles, sculptures...): tall enough that
    # the arm must navigate BETWEEN them (not over), so the skin informs avoidance
    ARM_TALL_Z = (0.26, 0.60)
    ARM_TALL_CATEGORIES = ("vase", "table lamp", "lantern", "bottle", "jar", "pitcher",
                           "thermos", "candlestick", "candle holder", "potted plant", "plant",
                           "globe", "trophy", "sculpture", "bust", "statue")
    EXCLUDED_TARGET_CATEGORIES = ("egg",)

    def __init__(self, config) -> None:
        super().__init__(config)
        self._clutter_names: list[str] = []
        self._spot_cache: dict[str, tuple] = {}
        self._clutter_pool_tall, self._clutter_pool_any = self._build_clutter_pools()
        log.info(f"[RealTable] clutter pools: tall={len(self._clutter_pool_tall)} "
                 f"any={len(self._clutter_pool_any)}")

    def _clutter_clamp(self, px: float, py: float):
        """Keep clutter on the custom-scene work surface (subclasses override per scene)."""
        return float(np.clip(px, 0.30, 0.95)), float(np.clip(py, -0.42, 0.42))

    def _obj_rest(self):
        """Per-scene target spot: read the 'target_spot' site from the current scene XML,
        so one run can mix several realistic environments (table, kitchen, shelf bay, sink...)."""
        import re
        try:
            path = self._current_house_scene_path(variant="base")
        except Exception:
            path = None
        if path:
            if path not in self._spot_cache:
                spot = None
                try:
                    m = re.search(r'<site name="target_spot" pos="([\-\d.]+) ([\-\d.]+) ([\-\d.]+)"',
                                  open(path).read())
                    if m:
                        x, y, z = (float(m.group(i)) for i in (1, 2, 3))
                        spot = (x, y, z)
                except Exception:
                    spot = None
                self._spot_cache[path] = spot
            if self._spot_cache[path] is not None:
                return self._spot_cache[path]
        return super()._obj_rest()

    def _build_clutter_pools(self):
        # arm-height obstacles from the FULL annotation DB (clutter needn't be pickupable)
        zlo, zhi = self.ARM_TALL_Z
        tall = []
        for uid, anno in ObjectMeta.annotation().items():
            bb = anno.get("boundingBox") or {}
            z = float(bb.get("z", 0)); xy = max(float(bb.get("x", 0)), float(bb.get("y", 0)))
            cat = str(anno.get("category", "")).lower()
            # stable-base filter: wide enough footprint + not too top-heavy, so items SETTLE
            # STANDING (measured: narrow/top-heavy assets tip over and read as surface clutter)
            if (zlo <= z <= zhi and 0.10 <= xy <= 0.35 and z / max(xy, 1e-6) <= 3.0
                    and cat in self.ARM_TALL_CATEGORIES):
                tall.append(uid)
                if len(tall) >= 300:
                    break
        # small/medium filler from the pickupable pool
        anyp = []
        for uid in get_valid_pickupable_obja_uids():
            anno = ObjectMeta.annotation(uid) or {}
            bb = anno.get("boundingBox", {})
            if 0.03 <= max(float(bb.get(k, 0)) for k in "xyz") <= 0.35:
                anyp.append(uid)
            if len(anyp) >= 80:
                break
        return tall, anyp

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        super().add_auxiliary_objects(spec)  # injects the graspable target (cavity_obj_0/...)
        rng = np.random.default_rng(1000 + self.current_house_index)
        self._clutter_names = []
        n_tall = max(self.N_CORRIDOR, self.N_CLUTTER // 2)
        picks = (list(rng.choice(self._clutter_pool_tall, size=min(n_tall, len(self._clutter_pool_tall)), replace=False)) +
                 list(rng.choice(self._clutter_pool_any, size=self.N_CLUTTER - n_tall, replace=False)))
        tx, ty, fz = self._obj_rest()
        name_to_meta = {}
        for i, uid in enumerate(picks):
            try:
                cxml = install_uid(uid)
                cspec = MjSpec.from_file(str(cxml))
            except Exception as e:
                log.warning(f"[RealTable] skip clutter uid {uid}: {e}")
                continue
            body = cspec.worldbody.bodies[0]
            if not body.first_joint():
                body.add_joint(name=f"{uid}_cl{i}_jnt", type=mjtJoint.mjJNT_FREE,
                               damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING)
            anno = ObjectMeta.annotation(uid) or {}
            z_half = float(anno.get("boundingBox", {}).get("z", 0.08)) / 2 + 0.01
            # first N_CORRIDOR tall items crowd the approach corridor (robot -> target),
            # alternating sides at staggered depths; the rest ring densely around the target.
            # _corridor_dir = unit vector base->target (subclasses set it; default +x).
            cdir = np.asarray(getattr(self, "_corridor_dir", (1.0, 0.0)), dtype=float)
            perp = np.array([-cdir[1], cdir[0]])
            if i < self.N_CORRIDOR:
                u = rng.uniform(0.02, 0.28)
                v = (1 if i % 2 == 0 else -1) * rng.uniform(0.10, 0.24)
                px, py = np.array([tx, ty]) - cdir * u + perp * v
            else:
                ang = rng.uniform(0, 2 * np.pi)
                r = rng.uniform(*self.CLUTTER_RING)
                px, py = tx + r * np.cos(ang), ty + r * np.sin(ang)
            px, py = self._clutter_clamp(float(px), float(py))
            quat = R.from_euler("zx", [rng.uniform(0, 2 * np.pi), np.pi / 2]).as_quat(scalar_first=True)
            ns = f"clutter_obj_{i}/"
            orig = body.name
            spec.worldbody.add_frame(pos=[px, py, fz + z_half], quat=quat).attach_body(body, ns, "")
            full = ns + orig
            self._clutter_names.append(full)
            name_to_meta[full] = {"asset_id": uid, "category": anno.get("category", "object"),
                                  "object_enum": "temp_object", "is_static": False,
                                  "boundingBox": anno.get("boundingBox", {})}
        self._metadata_adder.update(name_to_meta)
        log.info(f"[RealTable] injected {len(self._clutter_names)} physical clutter objects on the table")

    def _settle_injected_object(self, env: CPUMujocoEnv) -> None:
        """Settle the target AND all clutter onto the table together (isolating the arm)."""
        import mujoco
        m, d = env.current_model, env.current_data
        names = [self._injected_obj_name] + list(self._clutter_names)
        qadrs = []
        for nm in names:
            try:
                bid = m.body(nm).id
            except Exception:
                continue
            jadr = int(m.body_jntadr[bid])
            if jadr >= 0:
                qadrs.append(int(m.jnt_qposadr[jadr]))
        qpos0 = d.qpos.copy()
        # randomize the target's start (xy + yaw) like the base class
        bx, by, fz = self._obj_rest()
        jx, jy = self.OBJ_JIT_XY
        tq = qadrs[0]
        d.qpos[tq:tq + 3] = [bx + float(np.random.uniform(-jx, jx)),
                             by + float(np.random.uniform(-jy, jy)), fz + 0.10]
        yaw = float(np.random.uniform(0, 2 * np.pi))
        d.qpos[tq + 3:tq + 7] = (R.from_euler("z", yaw) * R.from_euler("x", 90, degrees=True)
                                 ).as_quat(scalar_first=True)
        d.qvel[:] = 0.0
        for _ in range(250):
            mujoco.mj_step(m, d)
        settled = {q: d.qpos[q:q + 7].copy() for q in qadrs}
        d.qpos[:], d.qvel[:] = qpos0, 0.0
        for q, v in settled.items():
            d.qpos[q:q + 7] = v
        mujoco.mj_forward(m, d)


class RealHousePickTaskSampler(RealTablePickTaskSampler):
    """Controlled pick task inside REAL ProcTHOR houses (maximal realism — advisor feedback).

    Per house: discover a real work surface (dining table / countertop / desk...), choose the
    side whose floor is free AND whose robot-mounted exo camera faces into the room (not a
    wall), place the LOW robot there directly, inject the graspable target + tall real clutter
    ON the surface (gravity-settled), and run the stock collision-aware planner. Uses the stock
    procthor scene mapping (scene_dataset='procthor-objaverse'), occupancy map still bypassed.
    """

    SURFACE_PREFIXES = ("diningtable", "coffeetable", "sidetable", "kitchencounter",
                        "countertop", "desk", "table", "tvstand", "dresser")
    EXO_OFFSET = (0.10, 0.57, 0.66)   # robot-mounted exo cam offset from link0 (base frame)
    MOUNT_H = 0.35

    def __init__(self, config) -> None:
        super().__init__(config)
        self._house_cache: dict[str, dict] = {}

    # houses use the stock dataset mapping (the user-mode override only applies to custom XMLs)
    def _get_dataset_index_map(self) -> dict:
        if self.config.scene_dataset != "user":
            return PickTaskSampler._get_dataset_index_map(self)
        return super()._get_dataset_index_map()

    def _discover(self, house_xml: str) -> dict:
        """Find (surface, robot base pose, target spot) for a house; cached per xml path."""
        if house_xml in self._house_cache:
            return self._house_cache[house_xml]
        import mujoco
        m = mujoco.MjModel.from_xml_path(house_xml)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        cands = {}
        for g in range(m.ngeom):
            bname = m.body(int(m.geom_bodyid[g])).name.lower()
            if not any(p in bname for p in self.SURFACE_PREFIXES):
                continue
            aabb = m.geom_aabb[g]
            xpos = d.geom_xpos[g]; xmat = d.geom_xmat[g].reshape(3, 3)
            corners = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(xpos + xmat @ (aabb[:3] + np.array([sx, sy, sz]) * aabb[3:]))
            corners = np.array(corners)
            top = corners[:, 2].max()
            cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
            ex, ey = np.ptp(corners[:, 0]) / 2, np.ptp(corners[:, 1]) / 2
            if 0.35 <= top <= 0.80 and ex >= 0.30 and ey >= 0.30:
                key = bname
                if key not in cands or (ex * ey) > (cands[key][3] * cands[key][4]):
                    cands[key] = (cx, cy, top, ex, ey)
        if not cands:
            self._house_cache[house_xml] = {}
            return {}

        def occupied(px, py, rad=0.30, zmax=1.2):
            n = 0
            for g in range(m.ngeom):
                if m.body(int(m.geom_bodyid[g])).name == "world":
                    continue
                p = d.geom_xpos[g]
                if abs(p[0] - px) < rad and abs(p[1] - py) < rad and p[2] < zmax:
                    n += 1
            return n

        best = None
        for bname, (cx, cy, top, ex, ey) in cands.items():
            for ang in (0, np.pi / 2, np.pi, -np.pi / 2):
                off = (ex if abs(np.cos(ang)) > 0.5 else ey) + 0.42
                bx, by = cx + off * np.cos(ang), cy + off * np.sin(ang)
                yaw = ang + np.pi
                # exo camera world position for this base pose — must not sit in/behind a wall
                Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                exo_xy = np.array([bx, by]) + Rz @ np.array(self.EXO_OFFSET[:2])
                score = occupied(bx, by) * 3 + occupied(exo_xy[0], exo_xy[1], rad=0.25, zmax=2.0)
                if best is None or score < best[0]:
                    best = (score, bname, cx, cy, top, ex, ey, bx, by, yaw)
        score, bname, cx, cy, top, ex, ey, bx, by, yaw = best
        # target on the robot-near quarter of the surface (toward the base side)
        to_base = np.array([np.cos(yaw + np.pi), np.sin(yaw + np.pi)])
        t_xy = np.array([cx, cy]) + to_base * min(ex, ey) * 0.25
        info = dict(surface=bname, cx=cx, cy=cy, top=top, ex=ex, ey=ey,
                    base=(bx, by, yaw), target=(float(t_xy[0]), float(t_xy[1]), top), score=score)
        log.info(f"[RealHouse] {house_xml.split('/')[-1]}: surface '{bname}' top={top:.2f} "
                 f"base=({bx:.2f},{by:.2f},yaw={np.degrees(yaw):.0f}) score={score}")
        self._house_cache[house_xml] = info
        return info

    def _cur_info(self) -> dict:
        path = self._current_house_scene_path(variant="base")
        info = self._discover(str(path))
        if not info:
            from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
            raise HouseInvalidForTask("no suitable work surface in this house")
        return info

    def _obj_rest(self):
        info = self._cur_info()
        return info["target"]

    def _clutter_clamp(self, px: float, py: float):
        info = self._cur_info()
        return (float(np.clip(px, info["cx"] - info["ex"] + 0.06, info["cx"] + info["ex"] - 0.06)),
                float(np.clip(py, info["cy"] - info["ey"] + 0.06, info["cy"] + info["ey"] - 0.06)))

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        info = self._cur_info()
        # corridor direction = from base toward target (world frame)
        bx, by, yaw = info["base"]
        self._corridor_dir = np.array([np.cos(yaw), np.sin(yaw)])
        super().add_auxiliary_objects(spec)

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        info = self._cur_info()
        bx, by, yaw = info["base"]
        self._cur_base_xyz = (bx + float(np.random.uniform(-0.03, 0.03)),
                              by + float(np.random.uniform(-0.04, 0.04)), 0.0)
        self._cur_base_yaw = yaw + float(np.random.uniform(-0.12, 0.12))
        # reuse the direct-base-pose logic from CavityPickTaskSampler
        import mujoco
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        robot_view = env.current_robot.robot_view
        base_quat = R.from_euler("z", self._cur_base_yaw).as_quat(scalar_first=True)
        robot_view.base.pose = pos_quat_to_pose_mat(
            np.array(self._cur_base_xyz, dtype=float), np.array(base_quat, dtype=float))
        mujoco.mj_forward(env.current_model, env.current_data)
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
        goal = pose_mat_to_7d(pickup_obj.pose)
        goal[2] += 0.10
        task_cfg.pickup_obj_goal_pose = goal.tolist()
