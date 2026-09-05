"""Supported PACT pick-and-place environments.

This clean release contains only the three environment lineages used for
reported experiments:

* experiment V5 (the ``pact_place_corridor_v2`` sampler/scene),
* V9.5 fixture-free eight-object real clutter, and
* V10.10 four-object real clutter with a compiled-static pendant.

The public configs live in ``pact_place_datagen_configs.py``. Historical
failed geometry variants are intentionally not exposed here.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import MjSpec

from molmo_spaces.configs.policy_configs import (
    PickAndPlacePlannerPolicyConfig,
    PickPlannerPolicyConfig,
)
from molmo_spaces.data_generation.pact_place.contracts import (
    V107_SPACED_ENVIRONMENT_VERSION,
    V1010_ENVIRONMENT_VERSION,
    build_v95_manifest_row,
    build_v107_spaced_manifest_row,
    build_v1010_manifest_row,
    build_v1011c_manifest_row,
    build_v1011d_manifest_row,
    v95_cell,
    v107_spaced_cell,
    v1010_cell,
    v1011_cell,
)
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    GripperAction,
    NoopAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.tasks.enclosure_reach import (
    SENSOR_HALF_FOV_COS,
    SENSOR_RANGE,
    SENSOR_RANGE_DERATE,
    SHELF_TOP_Z,
    TUBE_X0,
    BigFumehoodPickSampler,
    FumehoodSampler,
    ObstacleAwarePickPlannerPolicy,
)
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.utils.linalg_utils import transform_to_twist, twist_to_transform
from molmo_spaces.utils.object_metadata import ObjectMeta
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)

# Legacy marker constants used only to preserve exact branch membership inside
# the evaluated trajectory builder. No sampler for those failed variants is
# present or registered in this release.
PACT_PLACE_V10_ENVIRONMENT_VERSION = "pact_place_corridor_v10_compound_pendant"
PACT_PLACE_V102_ENVIRONMENT_VERSION = "pact_place_corridor_v10_2_raised_pendant"
PACT_PLACE_V105_ENVIRONMENT_VERSION = "pact_place_corridor_v10_5_v95_clutter_static_pendant"
PACT_PLACE_V106_ENVIRONMENT_VERSION = "pact_place_corridor_v10_6_v95_clutter_asymmetric_pendant"
PACT_PLACE_V1011_ENVIRONMENT_VERSION = "pact_place_corridor_v10_11_mixed_clutter"
PACT_PLACE_V1011B_ENVIRONMENT_VERSION = (
    "pact_place_corridor_v10_11b_tall_primitives"
)
PACT_PLACE_V1011C_ENVIRONMENT_VERSION = (
    "pact_place_corridor_v10_11c_33pct_taller_primitives"
)
PACT_PLACE_V1011D_ENVIRONMENT_VERSION = (
    "pact_place_corridor_v10_11d_randomized_clutter"
)
PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS = (
    PACT_PLACE_V106_ENVIRONMENT_VERSION,
    V1010_ENVIRONMENT_VERSION,
    PACT_PLACE_V1011_ENVIRONMENT_VERSION,
    PACT_PLACE_V1011B_ENVIRONMENT_VERSION,
    PACT_PLACE_V1011C_ENVIRONMENT_VERSION,
    PACT_PLACE_V1011D_ENVIRONMENT_VERSION,
)


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
    _pact_manifest_row_is_explicit: bool = False
    _pact_auto_house_index: int | None = None

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
        self._pact_manifest_row_is_explicit = True
        self._pact_auto_house_index = None

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
        face_jitter = float(row.get("panel_face_jitter_m", 0.0)) if row is not None else 0.0
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
        task_config.place_receptacle_start_pose = list(self.PLACE_RECEPTACLE_START_POSE)
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


class PactPlaceV5Sampler(PactPlaceCorridorV2Sampler):
    """Public experiment-V5 sampler (runtime marker remains corridor v2)."""

    def _ensure_manifest_row(self) -> dict[str, Any]:
        if self._pact_manifest_row_is_explicit:
            return self._pact_manifest_row or {}
        try:
            house_index = int(self.current_house_index)
        except (AttributeError, TypeError, ValueError):
            house_index = 0
        if self._pact_manifest_row is None or self._pact_auto_house_index != house_index:
            self._pact_manifest_row = {
                "intrusion_side": "left" if house_index % 2 == 0 else "right",
                "panel_x_jitter_m": 0.0,
                "panel_face_jitter_m": 0.0,
            }
            self._pact_auto_house_index = house_index
        return self._pact_manifest_row

    def _draw_theta(self):
        self._ensure_manifest_row()
        return super()._draw_theta()


class _PactPlaceLegacyClutterShellSampler(PactPlaceCorridorV2Sampler):
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


class _PactPlaceRealClutterSampler(_PactPlaceLegacyClutterShellSampler):
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
    LEGACY_CLUTTER_BODY_NAMES = _PactPlaceLegacyClutterShellSampler.CLUTTER_BODY_NAMES

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
            namespace = f"pact_clutter_{slot}/"
            park = np.asarray(self.CLUTTER_PARK_XYZ_M, dtype=float) + np.array(
                [0.35 * index, 0.0, 0.0]
            )
            primitive = dict(item.get("primitive") or {})
            if primitive:
                if slot_class != "prop":
                    raise ValueError("primitive clutter must be a movable prop")
                shape = str(primitive.get("shape") or "").lower()
                dimensions = np.asarray(item.get("dimensions_m") or [], dtype=float)
                if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
                    raise ValueError(
                        f"primitive clutter {slot!r} has invalid dimensions: "
                        f"{item.get('dimensions_m')!r}"
                    )
                if shape == "cylinder":
                    radius = float(primitive.get("radius_m", -1.0))
                    height = float(primitive.get("height_m", -1.0))
                    if radius <= 0.0 or height <= 0.0:
                        raise ValueError(f"invalid cylinder primitive for slot {slot!r}")
                    expected = np.asarray([2.0 * radius, 2.0 * radius, height])
                    if not np.allclose(dimensions, expected, atol=1e-9, rtol=0.0):
                        raise ValueError(
                            f"cylinder dimensions disagree with primitive spec: "
                            f"{dimensions.tolist()} != {expected.tolist()}"
                        )
                    geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
                    geom_size = [radius, height / 2.0, 0.0]
                elif shape == "box":
                    size = np.asarray(primitive.get("size_m") or [], dtype=float)
                    if size.shape != (3,) or np.any(size <= 0.0):
                        raise ValueError(f"invalid box primitive for slot {slot!r}")
                    if not np.allclose(dimensions, size, atol=1e-9, rtol=0.0):
                        raise ValueError(
                            f"box dimensions disagree with primitive spec: "
                            f"{dimensions.tolist()} != {size.tolist()}"
                        )
                    geom_type = mujoco.mjtGeom.mjGEOM_BOX
                    geom_size = (size / 2.0).tolist()
                else:
                    raise ValueError(
                        f"unsupported primitive clutter shape {shape!r} in slot {slot!r}"
                    )
                full_name = f"{namespace}{uid}"
                body = spec.worldbody.add_body(name=full_name, pos=park)
                body.add_joint(
                    name=f"{uid}_pact_clutter_{slot}_free",
                    type=mujoco.mjtJoint.mjJNT_FREE,
                    damping=float(self.CLUTTER_FREE_JOINT_DAMPING),
                )
                rgba = list(primitive.get("rgba") or [0.55, 0.58, 0.62, 1.0])
                if len(rgba) != 4:
                    raise ValueError(f"primitive clutter {slot!r} needs four RGBA values")
                body.add_geom(
                    name=f"{namespace}{uid}_collision",
                    type=geom_type,
                    size=geom_size,
                    rgba=rgba,
                    contype=1,
                    conaffinity=1,
                    density=float(primitive.get("density_kg_m3", 1000.0)),
                )
                annotation = {
                    "category": str(item.get("category") or "object"),
                    "boundingBox": {
                        "x": float(dimensions[0]),
                        "y": float(dimensions[1]),
                        "z": float(dimensions[2]),
                    },
                }
            else:
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
                original_name = body.name
                frame = spec.worldbody.add_frame(pos=park)
                frame.attach_body(body, namespace, "")
                full_name = namespace + original_name
                annotation = ObjectMeta.annotation(uid) or {}
            if any(
                forbidden in full_name
                for forbidden in ("cavity_obj_", "pact_intrusion_", "place_receptacle")
            ):
                raise ValueError(f"illegal clutter body name {full_name!r}")
            if not full_name.startswith("pact_clutter_"):
                raise ValueError(f"clutter body lacks required prefix: {full_name!r}")
            record = {
                "slot": slot,
                "uid": uid,
                "body": full_name,
                "park_m": park.tolist(),
                "slot_class": slot_class,
                "primitive": primitive or None,
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
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
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
    def _body_collision_aabb(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a body's descendant collision-geom AABB in world axes."""
        root_id = int(model.body_rootid[int(model.body(body_name).id)])
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for geom_id in range(int(model.ngeom)):
            body_id = int(model.geom_bodyid[geom_id])
            if int(model.body_rootid[body_id]) != root_id:
                continue
            if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
                continue
            # ``geom_aabb`` is expressed in the geom-local frame.  Its world
            # transform must therefore use geom_xpos/geom_xmat (important for
            # the nested, rotated child bodies in THOR assets).
            local_center = np.asarray(model.geom_aabb[geom_id, :3], dtype=float)
            local_half = np.asarray(model.geom_aabb[geom_id, 3:], dtype=float)
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            world_center = (
                np.asarray(data.geom_xpos[geom_id], dtype=float) + rotation @ local_center
            )
            world_half = np.abs(rotation) @ local_half
            lows.append(world_center - world_half)
            highs.append(world_center + world_half)
        if not lows:
            raise ValueError(f"active clutter body has no collision geoms: {body_name}")
        return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)

    def _prepare_pact_clutter_layout(self, th: dict[str, Any]) -> None:
        """Episode hook before the manifest layout is materialized.

        Historical samplers intentionally do nothing. V10.11 uses this narrow
        hook to draw the target rest pose once, before its two target-relative
        clutter placements are generated, without changing older random
        streams or duplicating the V5/V9 theta implementation.
        """

    def _draw_theta(self):
        # Call the pre-clutter implementation directly: V3/V4 remain unchanged,
        # while V5 does not try to treat free mesh bodies as scalar mocap boxes.
        th = PactPlaceCorridorV2Sampler._draw_theta(self)
        self._prepare_pact_clutter_layout(th)
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
        layout_by_slot = {str(item["palette_slot"]): dict(item) for item in list(layout["objects"])}
        # Recheck the frozen metadata boxes against this episode's exact shell.
        # B2 admits against the minimum 0.20 m depth, while this assertion also
        # protects later manifest edits from silently intersecting a wall.
        tolerance = float(self.CLUTTER_CONTAINMENT_TOLERANCE_M)
        shell_lo = np.asarray([TUBE_X0, -float(th["ap_w"]) / 2.0, SHELF_TOP_Z], dtype=float)
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
            str(item["body"]): str(item["slot_class"]) for item in self._pact_clutter_objects
        }
        active_props = [name for name in active if slot_class_by_body[name] == "prop"]
        active_mounts = [name for name in active if slot_class_by_body[name] == "mount"]
        names = [target_body, *active_props]
        addresses = {name: self._free_joint_addresses(model, name) for name in names}

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
            name: data.qpos[qadr : qadr + 7].copy() for name, (qadr, _dadr) in addresses.items()
        }
        active_roots = {int(model.body_rootid[int(model.body(name).id)]): name for name in active}
        target_root = int(model.body_rootid[int(model.body(target_body).id)])
        initial_object_contacts = []
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            left_root = int(model.body_rootid[int(model.geom_bodyid[int(contact.geom1)])])
            right_root = int(model.body_rootid[int(model.geom_bodyid[int(contact.geom2)])])
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
            if left_active is not None and right_active is not None and left_root != right_root:
                raise ValueError(f"settled clutter objects overlap: {pair}")
            if (left_active is not None and right_root == target_root) or (
                right_active is not None and left_root == target_root
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
                raise ValueError(f"clutter drifted during settle: {name} {xy_drift:.6f} m")
            if linear > self.CLUTTER_MAX_SETTLED_LINEAR_SPEED_M_S:
                raise ValueError(f"clutter did not settle: {name} linear speed {linear:.6f} m/s")
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
            if np.any(low[:2] < shell_lo[:2] - tolerance) or np.any(high > shell_hi + tolerance):
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
                "linear_speed_threshold_m_s": float(self.CLUTTER_MAX_SETTLED_LINEAR_SPEED_M_S),
                "angular_speed_threshold_rad_s": float(
                    self.CLUTTER_MAX_SETTLED_ANGULAR_SPEED_RAD_S
                ),
                "xy_drift_threshold_m": float(self.CLUTTER_MAX_SETTLED_XY_DRIFT_M),
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
        task.scene_params["pact_clutter_active_body_names"] = list(self._pact_active_clutter_names)
        return task


class _PactPlaceV9ChicaneSampler(_PactPlaceRealClutterSampler):
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
    # V10.11c raises only its own ceiling; every older sampler keeps 0.25 m.
    VESSEL_HEIGHT_RANGE_M = (0.15, 0.25)
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
    DECOR_CATEGORIES = frozenset({"mug", "apple", "bowl", "plate", "potato", "can", "candle"})
    EXCLUDED_CATEGORIES = frozenset({"cup", "teacup", "plastic cup", "ceramic cup", "clay cup"})
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
            height_low, height_high = self.VESSEL_HEIGHT_RANGE_M
            if len(dimensions) != 3 or not height_low <= dimensions[2] <= height_high:
                raise ValueError(
                    f"v9 vessel height is outside {height_low}-{height_high} m: "
                    f"{dimensions}"
                )
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
            str(palette_by_slot[str(item["palette_slot"])]["role"]): item for item in objects
        }
        if set(by_role) != set(self.VESSEL_ROLES) | {"decor"}:
            raise ValueError("v9 layout roles do not match the frozen palette")
        if (
            len(
                [
                    item
                    for item in objects
                    if str(palette_by_slot[str(item["palette_slot"])]["role"]) in self.VESSEL_ROLES
                ]
            )
            != 2
        ):
            raise ValueError("v9 layout must activate both vessel slots")
        return layout

    def _draw_theta(self):
        th = _PactPlaceRealClutterSampler._draw_theta(self)
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
            str(item["palette_slot"]): item for item in list(th["pact_clutter_layout"]["objects"])
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
            str(item["palette_slot"]): item for item in list(th["pact_clutter_layout"]["objects"])
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
        body_by_slot = {str(item["slot"]): str(item["body"]) for item in self._pact_clutter_objects}
        palette_by_slot = {str(item["slot"]): item for item in self._palette()}
        exact_hazards = []
        for hazard in list(th.get("pact_v9_hazards") or []):
            body_name = body_by_slot[str(hazard["slot"])]
            low, high = self._body_collision_aabb(env.current_model, env.current_data, body_name)
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


class PactPlaceCorridorV93Sampler(_PactPlaceV9ChicaneSampler):
    """V9.3 two-bottle 2-D chicane with paired, side-independent jitter."""

    PACT_PLACE_ENVIRONMENT_VERSION = "pact_place_corridor_v9_3"
    VESSEL_JITTER_LIMIT_M = 0.020

    def _ensure_manifest_row(self) -> dict[str, Any]:
        if self._pact_manifest_row_is_explicit:
            return self._pact_manifest_row or {}
        try:
            house_index = int(self.current_house_index)
        except (AttributeError, TypeError, ValueError):
            house_index = 0
        if self._pact_manifest_row is None or self._pact_auto_house_index != house_index:
            family, side = v95_cell(house_index)
            self._pact_manifest_row = build_v95_manifest_row(family, side)
            self._pact_auto_house_index = house_index
        return self._pact_manifest_row

    def _palette(self) -> list[dict[str, Any]]:
        self._ensure_manifest_row()
        return super()._palette()

    def _layout(self) -> dict[str, Any]:
        self._ensure_manifest_row()
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


# Explicit public name for the successful fixture-free V9.5 lineage.
PactPlaceV95RealClutterSampler = PactPlaceCorridorV93Sampler


class _PactPlaceStaticPendantSampler(PactPlaceCorridorV93Sampler):
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

    @staticmethod
    def _auto_manifest_row_for_house(house_index: int) -> dict[str, Any]:
        """Row bound before scene validation when no explicit row was given.

        Successors that ship their own palette must override this as well as
        ``_ensure_manifest_row``; ``sample_task`` binds the row before
        ``current_house_index`` exists, so it cannot go through the latter.
        """
        family, side, pose = v1010_cell(house_index)
        return build_v1010_manifest_row(family, side, pose)

    def sample_task(self, *args, **kwargs):
        """Refuse a scene/pose/hash mismatch before the task is created."""
        requested_house_index = kwargs.get("house_index")
        if requested_house_index is None and args:
            requested_house_index = args[0]
        if requested_house_index is None:
            requested_house_index = getattr(self, "current_house_index", 0)
        try:
            requested_house_index = int(requested_house_index)
        except (TypeError, ValueError):
            requested_house_index = 0

        # BaseTaskSampler assigns current_house_index inside super().sample_task.
        # Bind the auto row to the requested house before validating its scene;
        # otherwise a sampler reused across houses would validate the previous
        # scene and only switch rows later in _draw_theta.
        if not self._pact_manifest_row_is_explicit:
            self._pact_manifest_row = self._auto_manifest_row_for_house(
                requested_house_index
            )
            self._pact_auto_house_index = requested_house_index
        row = self._pact_manifest_row or {}
        expected = row.get("pact_v106_scene_sha256")
        if expected:
            paths = list(self.config.task_sampler_config.scene_xml_paths or [])
            try:
                scene = paths[requested_house_index]
            except (IndexError, TypeError, ValueError):
                scene = paths[0] if paths else None
            if scene is None:
                raise ValueError("V10.10 row binds a scene hash but no scene is set")
            observed = hashlib.sha256(Path(str(scene)).read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError(
                    f"V10.10 scene hash mismatch for pose {row.get('pose_id')!r}: "
                    f"{observed} != {expected}"
                )
        return super().sample_task(*args, **kwargs)


class PactPlaceCorridorV1010FourObjectSampler(_PactPlaceStaticPendantSampler):
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
        "01": "Soap_Bottle_30",
        "03": "Plate_10",
        "04": "Plate_22",
        "06": "Soap_Bottle_11",
    }

    def _ensure_manifest_row(self) -> dict[str, Any]:
        if self._pact_manifest_row_is_explicit:
            row = self._pact_manifest_row or {}
            if "pose_id" not in row:
                raise ValueError("an explicit V10.10 manifest row must bind pose_id")
            return row
        try:
            house_index = int(self.current_house_index)
        except (AttributeError, TypeError, ValueError):
            house_index = 0
        if self._pact_manifest_row is None or self._pact_auto_house_index != house_index:
            family, side, pose = v1010_cell(house_index)
            self._pact_manifest_row = build_v1010_manifest_row(family, side, pose)
            self._pact_auto_house_index = house_index
        return self._pact_manifest_row

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
            {
                "palette_slot": str(o["palette_slot"]),
                "uid": str(o["uid"]),
                "role": str(o.get("role", "")),
            }
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
                "palette_slot": str(o["palette_slot"]),
                "uid": str(o["uid"]),
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
                    f"V10.10 slot {slot} carries {by_slot[slot]['uid']!r}, expected {uid!r}"
                )
        active = [by_slot[s] for s in self.ACTIVE_CLUTTER_SLOTS]
        if len(active) != self.ACTIVE_CLUTTER_COUNT:
            raise ValueError(f"V10.10 activated {len(active)} slots, expected 4")
        parked = sorted(set(by_slot) - set(self.ACTIVE_CLUTTER_SLOTS))
        if parked != sorted(self.INACTIVE_CLUTTER_SLOTS):
            raise ValueError(
                f"V10.10 parked slots {parked} != {sorted(self.INACTIVE_CLUTTER_SLOTS)}"
            )
        layout["objects"] = active
        layout["active_clutter_slots"] = list(self.ACTIVE_CLUTTER_SLOTS)
        layout["inactive_clutter_slots"] = list(self.INACTIVE_CLUTTER_SLOTS)
        layout["active_clutter_count"] = self.ACTIVE_CLUTTER_COUNT
        layout["active_clutter_uids"] = {str(o["palette_slot"]): str(o["uid"]) for o in active}
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
                f"got {len(active)}: {active}"
            )


class PactPlaceCorridorV107SpacedBenchSampler(_PactPlaceStaticPendantSampler):
    """V10.7 spaced bench: the V10.6 pendant over a fully populated bench.

    The pendant, panel, target and vessel jitter are inherited unchanged. Only
    the bench population differs: all eight palette slots are live, the glass
    moves back into the otherwise empty mid-bench, and one bottle stays forward
    as the route blocker. Every decor object is naturally tall and standing, so
    the link-5/link-6 skin has something to sense across the whole table.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = V107_SPACED_ENVIRONMENT_VERSION

    @staticmethod
    def _auto_manifest_row_for_house(house_index: int) -> dict[str, Any]:
        family, side, pose = v107_spaced_cell(house_index)
        return build_v107_spaced_manifest_row(family, side, pose)

    def _ensure_manifest_row(self) -> dict[str, Any]:
        if self._pact_manifest_row_is_explicit:
            return self._pact_manifest_row or {}
        try:
            house_index = int(self.current_house_index)
        except (AttributeError, TypeError, ValueError):
            house_index = 0
        if self._pact_manifest_row is None or self._pact_auto_house_index != house_index:
            self._pact_manifest_row = self._auto_manifest_row_for_house(house_index)
            self._pact_auto_house_index = house_index
        return self._pact_manifest_row


# The published manifests for data/v107_spaced record the sampler under its
# pre-rename name. Keep it resolvable so those rows stay reproducible.
PactPlaceCorridorV106Sampler = _PactPlaceStaticPendantSampler


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
        disarmed = self._current_move_segment_name() in self.EMPTY_GRIPPER_DISARMED_SEGMENTS
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
        return bool(pos_err > self.tcp_pos_err_threshold or rot_err > self.tcp_rot_err_threshold)

    def check_failure(self) -> bool:
        if self._persistent_empty_gripper_failure():
            return True
        return self._tcp_tracking_failure()


class PactPlaceCorridorPolicy(PickAndPlacePlannerPolicy):
    """Evaluated bidirectional place expert for V5, V9.5 and V10.10."""

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
    INBOUND_ENVELOPE_HALF_Y = 0.11
    INBOUND_SAFE_GAP = 0.04
    OUTBOUND_ENVELOPE_HALF_Y = 0.15
    OUTBOUND_SAFE_GAP = 0.14
    V9_VESSEL_SAFE_GAP = 0.04
    OUTBOUND_CARRY_RAISE_M = 0.0
    OUTBOUND_PASS_SPEED = 0.045
    OUTSIDE_STAGING_X_M = TUBE_X0 - 0.10
    V9_OUTSIDE_STAGING_X_M = TUBE_X0 - 0.14
    V93_OUTSIDE_STAGING_X_M = TUBE_X0 - 0.22
    GRASP_WORLD_Z_OFFSET_M = 0.0
    PASS_SPEED = 0.045
    APERTURE_EDGE_RESERVE = 0.02
    RELEASE_CLEARANCE_M = 0.005
    OUTBOUND_APPROACH_MAX_STEP_M = 0.04
    SETTLE_WINDOW_STEPS = 25

    def __init__(self, config, task) -> None:
        super().__init__(config, task)
        self.behavior_class = "straight"
        self.inbound_deflected = False
        self.outbound_deflected = False
        self._pact_place_bow_diagnostics = self._empty_bow_diagnostics()
        self._pact_place_v106_speed_amendment: dict[str, Any] = {}
        self._pact_place_last_tcp_m: np.ndarray | None = None
        self._pact_place_last_sim_time_s: float | None = None
        self._sensor_cam_ids: list[int] | None = None
        self._pact_detected_hazard_names: set[str] = set()
        self._pact_detected_hazards: list[dict[str, Any]] = []
        self._pact_maneuver_interactions: list[dict[str, Any]] = []
        self._pact_active_maneuver: str | None = None

    def _environment_version(self) -> str:
        return str(
            (getattr(self.task, "scene_params", {}) or {}).get("pact_place_environment_version", "")
        )

    def _v9_enabled(self) -> bool:
        return self._environment_version() in {
            "pact_place_corridor_v9_3",
            V1010_ENVIRONMENT_VERSION,
            *PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS,
        }

    def _v106_enabled(self) -> bool:
        # Gates the initial free-space speed cap and the per-frame telemetry.
        # A V10.11 version missing here would run different speeds and emit no
        # telemetry while otherwise looking healthy.
        return self._environment_version() in PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS

    @staticmethod
    def _mounted_fixture_roles(
        environment_version: str, *, pendant_lateral_bow: bool = False
    ) -> tuple[str, ...]:
        return ()

    def _v99_apply_lane(self, segments, **_kwargs):
        return segments

    def _v10_apply_lane(self, segments, **_kwargs):
        return segments

    def _v104_apply_speed_amendment(self, primitives):
        return primitives

    def _v105_apply_speed_amendment(self, primitives):
        return primitives

    def _v106_apply_speed_amendment(self, primitives):
        from molmo_spaces.tasks.pact_place_speed import (
            apply_initial_free_space_speed_cap,
            plan_signature,
            schedule_sha256,
            verify_plan_matches_baseline,
        )

        if not self._v106_enabled():
            self._pact_place_v106_speed_amendment = {
                "applied": False,
                "reason": "environment marker is not V10.10",
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

    def _protrusion_detected(self, allowed_roles: set[str] | None = None) -> dict[str, Any] | None:
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
                self._pact_place_bow_diagnostics.get(prefix, {}).get("accepted_bow_m", 0.0)
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
            float((getattr(self.task, "scene_params", {}) or {}).get("ap_w", 0.85)) / 2.0
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
            hazard = next(item for item in self._hazard_list() if item["name"] == hazard)
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
        waypoints = self._inbound_deflect_segments(current, target_pregrasp[:3, 3].copy(), hazard)
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
        previous = self._pact_place_bow_diagnostics.get(prefix, self._empty_bow_record())
        take_waypoint = float(planned_bow_m) + 1e-12 >= float(previous.get("planned_bow_m", 0.0))
        waypoint_y_out = previous.get("waypoint_y_m")
        waypoint_side_out = previous.get("waypoint_side")
        if take_waypoint and waypoint_y_m is not None:
            waypoint_y_out = float(waypoint_y_m)
            waypoint_side_out = None if waypoint_side is None else float(waypoint_side)
        self._pact_place_bow_diagnostics[prefix] = {
            # A compound hazard can be evaluated against several consecutive
            # segments.  Preserve the strongest admitted bow instead of letting
            # a later non-crossing segment erase it with zeros.
            "planned_bow_m": max(float(previous.get("planned_bow_m", 0.0)), float(planned_bow_m)),
            "accepted_bow_m": max(
                float(previous.get("accepted_bow_m", 0.0)), float(accepted_bow_m)
            ),
            "bow_fallback_taken": bool(previous.get("bow_fallback_taken") or bow_fallback_taken),
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
        straight_clearance = waypoint_side * (cross[1] - open_face_y) - envelope_half_y
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
            aperture_width / 2 - envelope_half_y - self.APERTURE_EDGE_RESERVE,
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
            self.OUTBOUND_PASS_SPEED if prefix == "outbound" else self.policy_config.speed_fast
        )
        pass_speed = self.OUTBOUND_PASS_SPEED if prefix == "outbound" else self.PASS_SPEED
        exit_speed = (
            self.OUTBOUND_PASS_SPEED if prefix == "outbound" else self.policy_config.speed_slow
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

    def _sequence(self, segments: list[TCPMoveSegment], *, holding: bool) -> TCPMoveSequence:
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
    def _interpolate_pose(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        lin_vel, ang_vel = transform_to_twist(np.linalg.inv(start) @ end)
        return start @ twist_to_transform(lin_vel * float(t), ang_vel * float(t))

    @classmethod
    def _subdivide_tcp_segment(
        cls, segment: TCPMoveSegment, max_step_m: float
    ) -> list[TCPMoveSegment]:
        dist = float(np.linalg.norm(segment.end_pose[:3, 3] - segment.start_pose[:3, 3]))
        n_pieces = max(1, int(np.ceil(dist / max_step_m - 1e-12)))
        if n_pieces == 1:
            return [segment]
        pieces: list[TCPMoveSegment] = []
        previous = segment.start_pose.copy()
        for index in range(1, n_pieces + 1):
            pose = cls._interpolate_pose(segment.start_pose, segment.end_pose, index / n_pieces)
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
                (item for item in self._hazard_list() if item.get("role") == inbound_hazard_role),
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
                        preferred_waypoint_side=(self._preferred_v9_waypoint_side()),
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
                    (item for item in self._hazard_list() if item.get("role") == fixture_role),
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
            "stock_grasp_world_position_m": list(map(float, stock_inbound_grasp.end_pose[:3, 3])),
            "adjusted_grasp_world_position_m": list(map(float, adjusted_grasp_pose[:3, 3])),
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
            (getattr(self.task, "scene_params", {}) or {}).get("pact_place_environment_version", "")
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
                (item for item in self._hazard_list() if item.get("role") == "outbound_vessel"),
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
                    (item for item in self._hazard_list() if item.get("role") == "inbound_vessel"),
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
                        (item for item in self._hazard_list() if item.get("role") == fixture_role),
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
                                    route_side if fixture_role == "ceiling_fixture" else None
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
                "outside_staging_position_m": list(map(float, outside_staging_pose[:3, 3])),
                "preplace_position_m": list(map(float, preplace_pose[:3, 3])),
                "place_position_m": list(map(float, place_pose[:3, 3])),
                "release_clearance_m": float(self.RELEASE_CLEARANCE_M),
                "outbound_waypoint_positions_m": [
                    list(map(float, segment.end_pose[:3, 3])) for segment in outbound_segments
                ],
                "bow_diagnostics": self._pact_place_bow_diagnostics,
            }
        )
        placement_sequence = self._sequence(
            carry_raise + outbound_segments + [preplace_transition, place],
            holding=True,
        )
        release = GripperAction(robot_view, True, self.policy_config.gripper_open_duration)
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

    def _update_manipulation_progress(self) -> None:
        try:
            task_config = self.task.config.task_config
            manager = self.task.env.object_managers[self.task.env.current_batch_index]
            pickup = manager.get_object_by_name(task_config.pickup_obj_name)
            if self._pickup_start_z is None:
                self._pickup_start_z = float(pickup.position[2])
            pickup_z = float(pickup.position[2])
            self._pickup_max_z = (
                pickup_z if self._pickup_max_z is None else max(self._pickup_max_z, pickup_z)
            )
            self._pickup_final_position = list(map(float, pickup.position))
            self._pickup_final_quat = list(map(float, pickup.quat))
            if self._pickup_start_position is None:
                self._pickup_start_position = list(self._pickup_final_position)
                self._pickup_start_quat = list(self._pickup_final_quat)
            self._object_position_window.append(list(self._pickup_final_position))
            self._cup_lifted |= bool(pickup_z >= self._pickup_start_z + 0.01)
            self._cup_retrieved_outside_aperture |= bool(
                self._cup_lifted and float(pickup.position[0]) < TUBE_X0 - 0.03
            )
            robot_view = self.task.env.current_robot.robot_view
            gripper_id = robot_view.get_gripper_movegroup_ids()[0]
            gripper = robot_view.get_gripper(gripper_id)
            width = float(gripper.inter_finger_dist)
            self._gripper_width_min = (
                width if self._gripper_width_min is None else min(self._gripper_width_min, width)
            )
            self._gripper_width_max = (
                width if self._gripper_width_max is None else max(self._gripper_width_max, width)
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
                reference_rotation = np.asarray(baseline["xmat"], dtype=float).reshape(3, 3)
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

    def _endpoint_scalars(self) -> dict[str, Any]:
        end = self._pickup_final_position
        start_z = self._pickup_start_z
        end_z = None if end is None else float(end[2])
        window = list(self._object_position_window)
        settle = None
        if len(window) >= 2:
            settle = float(
                np.linalg.norm(
                    np.asarray(window[-1], dtype=float) - np.asarray(window[0], dtype=float)
                )
            )
        receptacle_distance = None
        try:
            if end is not None:
                manager = self.task.env.object_managers[self.task.env.current_batch_index]
                receptacle = manager.get_object_by_name(
                    self.task.config.task_config.place_receptacle_name
                )
                receptacle_distance = float(
                    np.linalg.norm(
                        np.asarray(end, dtype=float) - np.asarray(receptacle.position, dtype=float)
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
                None if start_z is None or end_z is None else float(end_z - start_z)
            ),
            "object_to_receptacle_distance_m": receptacle_distance,
            "settle_window_steps": int(self.SETTLE_WINDOW_STEPS),
            "settle_displacement_m": settle,
            "endpoint_values_emitted_during_compaction": True,
        }

    def reset(self, reset_retries: bool = True):
        from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit

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
            self._gripper_width_min = None
            self._gripper_width_max = None
            self._pact_clutter_stability_events = []
            self._pact_clutter_stability_bodies = set()
        self.task._contact_audit_hook = self._pact_place_contact_audit
        self.behavior_class = "straight"
        self.inbound_deflected = False
        self.outbound_deflected = False
        self._pact_place_bow_diagnostics = self._empty_bow_diagnostics()
        self._sensor_cam_ids = None
        self._pact_detected_hazard_names = set()
        self._pact_detected_hazards = []
        self._pact_maneuver_interactions = []
        self._pact_active_maneuver = None
        result = super().reset(reset_retries)
        self.target_poses.update(self._pact_place_canonical_target_poses)
        return result

    def _record_place_trajectory_step(self) -> None:
        tcp_pos = None
        try:
            gripper_id = self.robot_view.get_gripper_movegroup_ids()[0]
            tcp = self.robot_view.get_gripper(gripper_id).leaf_frame_to_world
            tcp_pos = list(map(float, tcp[:3, 3]))
        except Exception:
            pass
        sim_time_s = float(self.task.env.current_data.time)
        segment = self._current_move_segment()
        realized_step_m = None
        realized_speed_m_s = None
        if tcp_pos is not None:
            current = np.asarray(tcp_pos, dtype=float)
            if self._pact_place_last_tcp_m is not None:
                realized_step_m = float(np.linalg.norm(current - self._pact_place_last_tcp_m))
                if self._pact_place_last_sim_time_s is not None:
                    dt = sim_time_s - self._pact_place_last_sim_time_s
                    if dt > 1e-9:
                        realized_speed_m_s = realized_step_m / dt
            self._pact_place_last_tcp_m = current
        self._pact_place_last_sim_time_s = sim_time_s
        self._pact_place_trajectory.append(
            {
                "step": int(self._pact_place_control_step),
                "sim_time_s": sim_time_s,
                "policy_phase": str(self.get_phase()),
                "traversal_phase": self._traversal_phase(str(self.get_phase())),
                "segment_name": None if segment is None else str(segment.name),
                "commanded_speed_m_s": None if segment is None else float(segment.speed),
                "realized_tcp_displacement_m": realized_step_m,
                "realized_tcp_speed_m_s": realized_speed_m_s,
                "tcp_position_m": tcp_pos,
                "object_position_m": self._pickup_final_position,
                "object_quat_xyzw": self._pickup_final_quat,
                "qpos": [
                    float(value) for value in np.asarray(self.task.env.current_data.qpos).tolist()
                ],
            }
        )

    def get_action(self, observation):
        if self._v9_enabled():
            detected = self._protrusion_detected({"inbound_vessel", "panel", "outbound_vessel"})
            if detected is not None and self._phase_for_hazard(detected):
                self._handle_detected_hazard(detected)
        policy_phase = self.get_phase()
        self._pact_place_contact_audit.set_phase(self._traversal_phase(policy_phase), policy_phase)
        self._pact_place_contact_audit.observe(self.task.env, self._pact_place_control_step)
        action = super().get_action(observation)
        self._update_manipulation_progress()
        self._update_clutter_stability()
        self._record_place_trajectory_step()
        self._pact_place_control_step += 1
        return action

    def get_info(self):
        from molmo_spaces.tasks.pact_place_contact_audit import place_environment_contact_pairs

        policy_phase = self.get_phase()
        self._pact_place_contact_audit.set_phase(self._traversal_phase(policy_phase), policy_phase)
        self._pact_place_contact_audit.observe(self.task.env, self._pact_place_control_step)
        self._update_manipulation_progress()
        self._update_clutter_stability()
        self._record_place_trajectory_step()
        info = super().get_info()
        place_metrics = self.task.get_info()[0]
        info.update(
            {
                "pact_contact_audit": self._pact_place_contact_audit.summary(),
                "grasp_phase_success": bool(self._cup_retrieved_outside_aperture),
                "cup_lifted_one_cm": bool(self._cup_lifted),
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
                "terminal_robot_environment_contacts": place_environment_contact_pairs(
                    self.task.env
                ),
                "endpoint_scalars": self._endpoint_scalars(),
                "pact_v106_speed_amendment": dict(self._pact_place_v106_speed_amendment),
                "trajectory": list(self._pact_place_trajectory),
                "clutter_stability_events": list(self._pact_clutter_stability_events),
                "clutter_stability_ok": not bool(self._pact_clutter_stability_events),
            }
        )
        return info


class PactPlaceCorridorPolicyConfig(PickAndPlacePlannerPolicyConfig):
    """Wire the supported place-corridor expert; rollout failures are terminal."""

    max_retries: int = 0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = PactPlaceCorridorPolicy


class PactPlaceCorridorV1011MixedClutterSampler(_PactPlaceStaticPendantSampler):
    """V10.10 lane with three mesh props and three runtime primitives.

    The certified pose-specific scene remains byte-identical. Primitive bodies
    are added to the episode MjSpec by the shared V5 injector, and all six live
    clutter bodies remain movable free bodies. Slots 08/09 are sampled in a
    bounded target-relative annular sector from a single target rest draw.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = PACT_PLACE_V1011_ENVIRONMENT_VERSION
    ACTIVE_CLUTTER_SLOTS = ("01", "03", "04", "06", "08", "09")
    INACTIVE_CLUTTER_SLOTS = ("00", "02", "05", "07")
    ACTIVE_CLUTTER_COUNT = 6
    PRIMITIVE_SLOTS = ("01", "08", "09")
    MESH_SLOTS = ("03", "04", "06")
    NEAR_TARGET_SLOTS = ("08", "09")
    EXPECTED_ACTIVE_UIDS = {
        "01": "pact_primitive_cylinder_01",
        "03": "Plate_10",
        "04": "Plate_22",
        "06": "Soap_Bottle_11",
        "08": "pact_primitive_cylinder_08",
        "09": "pact_primitive_box_09",
    }
    NEAR_ANGLE_LOW_RAD = float(np.deg2rad(-65.0))
    NEAR_ANGLE_HIGH_RAD = float(np.deg2rad(65.0))
    NEAR_RADIUS_MAX_M = 0.220
    NEAR_TARGET_GAP_M = 0.020
    NEAR_OBJECT_GAP_M = 0.010
    NEAR_MAX_CANDIDATES = 64
    # The target location is drawn once in _prepare_pact_clutter_layout.
    # A second independent jitter would invalidate target-relative placement.
    OBJ_JIT_XY = (0.0, 0.0)

    @classmethod
    def mixed_identity_sha256(cls, palette) -> str:
        import hashlib
        import json as _json

        payload = [
            {
                "slot": str(item["slot"]),
                "uid": str(item["uid"]),
                "role": str(item.get("role", "")),
                "primitive": item.get("primitive"),
            }
            for item in sorted(palette, key=lambda value: str(value["slot"]))
        ]
        return hashlib.sha256(
            _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def mixed_layout_sha256(cls, objects) -> str:
        import hashlib
        import json as _json

        payload = [
            {
                "palette_slot": str(item["palette_slot"]),
                "uid": str(item["uid"]),
                "center_m": [float(value) for value in item["center_m"]],
                "quat_wxyz": [float(value) for value in item["quat_wxyz"]],
            }
            for item in sorted(objects, key=lambda value: str(value["palette_slot"]))
        ]
        return hashlib.sha256(
            _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def target_planar_bounding_radius_m(self) -> tuple[float, str]:
        """Conservative target footprint about its free-joint origin.

        Cup_10 settles with the inherited 90-degree X rotation, so metadata
        X/Y alone is not its table footprint. Once the scene is compiled, use
        collision-geom AABBs and the current fixed tilt to bound every corner's
        radial XY distance. Yaw during settling preserves that radius.
        """
        env = getattr(self, "env", None)
        if env is not None and getattr(self, "_injected_obj_name", None):
            model, data = env.current_model, env.current_data
            mujoco.mj_forward(model, data)
            body_id = int(model.body(str(self._injected_obj_name)).id)
            root_id = int(model.body_rootid[body_id])
            origin = np.asarray(data.xpos[root_id], dtype=float)
            radii: list[float] = []
            for geom_id in range(int(model.ngeom)):
                geom_body = int(model.geom_bodyid[geom_id])
                if int(model.body_rootid[geom_body]) != root_id:
                    continue
                if int(model.geom_contype[geom_id]) == 0 and int(
                    model.geom_conaffinity[geom_id]
                ) == 0:
                    continue
                local_center = np.asarray(model.geom_aabb[geom_id, :3], dtype=float)
                local_half = np.asarray(model.geom_aabb[geom_id, 3:], dtype=float)
                rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                geom_origin = np.asarray(data.geom_xpos[geom_id], dtype=float)
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        for sz in (-1.0, 1.0):
                            corner = local_center + local_half * np.asarray([sx, sy, sz])
                            world = geom_origin + rotation @ corner
                            radii.append(float(np.linalg.norm((world - origin)[:2])))
            if radii:
                return max(radii), "compiled_collision_aabb_corner_radius"
        annotation = ObjectMeta.annotation(self.TARGET_UID) or {}
        bounds = annotation.get("boundingBox") or {}
        try:
            # The inherited target orientation maps source Z into the table
            # plane. Half the X/Z diagonal is invariant to the later yaw.
            return (
                float(np.hypot(float(bounds["x"]), float(bounds["z"]))) / 2.0,
                "metadata_xz_half_diagonal_fallback",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cup_10 target metadata lacks a valid X/Z bounding box") from exc

    def _prepare_pact_clutter_layout(self, th: dict[str, Any]) -> None:
        # This is the one and only target-XY random draw for the episode. The
        # inherited target settle later calls _obj_rest and receives this value.
        x = TUBE_X0 + max(0.12, float(th["target_frac"]) * float(th["depth"]) - 0.04)
        y = float(np.random.uniform(-1.0, 1.0) * (float(th["ap_w"]) / 2.0 - 0.05))
        self._v1011_target_rest = (float(x), y, float(SHELF_TOP_Z))
        self._v1011_target_draw_count = 1

    def _obj_rest(self):
        cached = getattr(self, "_v1011_target_rest", None)
        return tuple(cached) if cached is not None else super()._obj_rest()

    @staticmethod
    def _planar_boxes_separated(
        center: np.ndarray,
        half: np.ndarray,
        other_center: np.ndarray,
        other_half: np.ndarray,
        gap: float,
    ) -> bool:
        delta = np.abs(center[:2] - other_center[:2])
        required = half[:2] + other_half[:2] + float(gap)
        return bool(np.any(delta >= required - 1e-12))

    def _sample_near_target_center(
        self,
        *,
        half: np.ndarray,
        object_planar_radius_m: float,
        occupied: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[list[float], dict[str, Any]]:
        target = np.asarray(self._v1011_target_rest, dtype=float)
        target_radius, target_radius_source = self.target_planar_bounding_radius_m()
        object_radius = float(object_planar_radius_m)
        if not np.isfinite(object_radius) or object_radius <= 0.0:
            raise ValueError("V10.11 object planar radius must be finite and positive")
        radius_min = target_radius + object_radius + self.NEAR_TARGET_GAP_M
        if radius_min >= self.NEAR_RADIUS_MAX_M:
            raise ValueError("V10.11 target-relative annulus is empty")
        shell_low = np.asarray(self.CLUTTER_WORKSPACE_LOW, dtype=float)
        shell_high = np.asarray(self.CLUTTER_WORKSPACE_HIGH, dtype=float)
        for candidate_index in range(self.NEAR_MAX_CANDIDATES):
            angle = float(
                np.random.uniform(self.NEAR_ANGLE_LOW_RAD, self.NEAR_ANGLE_HIGH_RAD)
            )
            unit = float(np.random.uniform())
            radius = float(
                np.sqrt(
                    radius_min * radius_min
                    + unit
                    * (self.NEAR_RADIUS_MAX_M * self.NEAR_RADIUS_MAX_M
                       - radius_min * radius_min)
                )
            )
            center = np.asarray(
                [
                    target[0] + radius * np.cos(angle),
                    target[1] + radius * np.sin(angle),
                    SHELF_TOP_Z + float(half[2]),
                ],
                dtype=float,
            )
            if np.any(center - half < shell_low - 1e-12) or np.any(
                center + half > shell_high + 1e-12
            ):
                continue
            if any(
                not self._planar_boxes_separated(
                    center, half, other_center, other_half, self.NEAR_OBJECT_GAP_M
                )
                for other_center, other_half in occupied
            ):
                continue
            return center.tolist(), {
                "candidate_index": candidate_index,
                "angle_rad": angle,
                "area_uniform_u": unit,
                "radius_m": radius,
                "radius_min_m": radius_min,
                "radius_max_m": self.NEAR_RADIUS_MAX_M,
                "target_planar_bounding_radius_m": target_radius,
                "target_radius_source": target_radius_source,
                "object_planar_bounding_radius_m": object_radius,
            }
        raise ValueError(
            f"V10.11 could not place target-relative clutter in "
            f"{self.NEAR_MAX_CANDIDATES} deterministic candidates"
        )

    def _randomize_base_slot_centers(self, by_slot, layout) -> dict[str, Any]:
        """Redraw slots 01/03/04/06 before the near-target annulus is sampled.

        V10.11, V10.11b and V10.11c keep the inherited V9.5 layout, so this is
        a no-op for them and their recorded layouts are unaffected.
        """
        return {}

    def _layout(self):
        import copy

        from molmo_spaces.data_generation.pact_place.contracts import (
            panel_corridor_metrics,
            route_blocker_metrics,
        )

        layout = copy.deepcopy(super()._layout())
        by_slot = {str(item["palette_slot"]): item for item in layout["objects"]}
        missing = [slot for slot in self.ACTIVE_CLUTTER_SLOTS if slot not in by_slot]
        if missing:
            raise ValueError(f"V10.11 active slots absent from layout: {missing}")
        for slot, uid in self.EXPECTED_ACTIVE_UIDS.items():
            if str(by_slot[slot]["uid"]) != uid:
                raise ValueError(
                    f"V10.11 slot {slot} carries {by_slot[slot]['uid']!r}, "
                    f"expected {uid!r}"
                )
        if getattr(self, "_v1011_target_rest", None) is None:
            raise ValueError("V10.11 target rest must be drawn before clutter layout")
        # Successors may redraw the four non-target-relative slots before the
        # near-target annulus is sampled, so that slots 08/09 see the final
        # occupancy. The default is a no-op, which leaves V10.11/b/c layouts
        # bit-identical to what they produced before this hook existed.
        base_randomization = self._randomize_base_slot_centers(by_slot, layout)
        occupied: list[tuple[np.ndarray, np.ndarray]] = []
        for slot in ("01", "03", "04", "06"):
            item = by_slot[slot]
            occupied.append(
                (np.asarray(item["center_m"], dtype=float),
                 np.asarray(item["half_m"], dtype=float))
            )
        placements = {}
        for slot in self.NEAR_TARGET_SLOTS:
            item = by_slot[slot]
            half = np.asarray(item["half_m"], dtype=float)
            primitive_shape = str((item.get("primitive") or {}).get("shape") or "")
            if primitive_shape == "cylinder":
                object_planar_radius = float(max(half[0], half[1]))
            elif primitive_shape == "box":
                # A box can yaw while settling. Its circumscribed XY radius,
                # not its half-width, is the rotation-invariant footprint.
                object_planar_radius = float(np.linalg.norm(half[:2]))
            else:
                raise ValueError(
                    f"V10.11 near-target slot {slot} has unsupported shape "
                    f"{primitive_shape!r}"
                )
            center, detail = self._sample_near_target_center(
                half=half,
                object_planar_radius_m=object_planar_radius,
                occupied=occupied,
            )
            item["center_m"] = center
            item["near_target_placement"] = detail
            occupied.append((np.asarray(center, dtype=float), half))
            placements[slot] = detail
        active = [by_slot[slot] for slot in self.ACTIVE_CLUTTER_SLOTS]
        if len(active) != self.ACTIVE_CLUTTER_COUNT:
            raise ValueError(
                f"V10.11 activated {len(active)} slots, expected "
                f"{self.ACTIVE_CLUTTER_COUNT}"
            )
        layout["objects"] = active
        layout["active_clutter_slots"] = list(self.ACTIVE_CLUTTER_SLOTS)
        layout["inactive_clutter_slots"] = list(self.INACTIVE_CLUTTER_SLOTS)
        layout["active_clutter_count"] = self.ACTIVE_CLUTTER_COUNT
        layout["near_target_placements"] = placements
        # Only successors that actually randomize record this, so V10.11/b/c
        # layouts keep exactly the key set they had before the hook existed.
        if base_randomization:
            layout["base_slot_randomization"] = base_randomization
        layout["target_rest_m"] = list(self._v1011_target_rest)
        layout["target_draw_count"] = int(self._v1011_target_draw_count)
        layout["pact_v1011_identity_sha256"] = self.mixed_identity_sha256(
            self._palette()
        )
        layout["pact_v1011_layout_sha256"] = self.mixed_layout_sha256(active)
        layout["layout_id"] = f"{layout['layout_id']}_v1011_mixed"
        # The route-bearing slot changed shape. Recompute these predicates from
        # the actual primitive half extents instead of retaining V9.5 numbers.
        layout["route_blocker_center_xy_m"] = list(by_slot["01"]["center_m"][:2])
        layout["nominal_route_metrics"] = route_blocker_metrics(layout)
        layout["panel_corridor_metrics"] = panel_corridor_metrics(layout)
        if not layout["nominal_route_metrics"]["detour_admitted"]:
            raise ValueError("V10.11 primitive vessel closes the nominal detour")
        if not layout["panel_corridor_metrics"]["detour_admitted"]:
            raise ValueError("V10.11 primitive vessel closes the panel corridor")
        return layout

    def _draw_theta(self):
        th = super()._draw_theta()
        layout = th.get("pact_clutter_layout") or {}
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_v1011_active_clutter_slots"] = list(self.ACTIVE_CLUTTER_SLOTS)
        th["pact_v1011_inactive_clutter_slots"] = list(self.INACTIVE_CLUTTER_SLOTS)
        th["pact_v1011_active_clutter_count"] = self.ACTIVE_CLUTTER_COUNT
        th["pact_v1011_primitive_slots"] = list(self.PRIMITIVE_SLOTS)
        th["pact_v1011_mesh_slots"] = list(self.MESH_SLOTS)
        th["pact_v1011_target_rest_m"] = layout.get("target_rest_m")
        th["pact_v1011_target_draw_count"] = layout.get("target_draw_count")
        th["pact_v1011_near_target_placements"] = layout.get(
            "near_target_placements"
        )
        th["pact_v1011_identity_sha256"] = layout.get(
            "pact_v1011_identity_sha256"
        )
        th["pact_v1011_layout_sha256"] = layout.get("pact_v1011_layout_sha256")
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        active = list(getattr(self, "_pact_active_clutter_names", []))
        if len(active) != self.ACTIVE_CLUTTER_COUNT:
            raise ValueError(
                f"V10.11 expected {self.ACTIVE_CLUTTER_COUNT} active clutter "
                f"bodies, got {len(active)}: {active}"
            )
        if int(th.get("pact_v1011_target_draw_count") or 0) != 1:
            raise ValueError("V10.11 target was not drawn exactly once")


class PactPlaceCorridorV1011BTallPrimitiveSampler(
    PactPlaceCorridorV1011MixedClutterSampler
):
    """V10.11 successor with taller primitives and identical XY footprints."""

    PACT_PLACE_ENVIRONMENT_VERSION = PACT_PLACE_V1011B_ENVIRONMENT_VERSION
    EXPECTED_PRIMITIVE_HEIGHTS_M = {"01": 0.245, "08": 0.180, "09": 0.180}

    def _draw_theta(self):
        th = super()._draw_theta()
        palette = {
            str(item["slot"]): item
            for item in list(th.get("pact_clutter_palette") or [])
        }
        observed = {
            slot: float(palette[slot]["dimensions_m"][2])
            for slot in self.EXPECTED_PRIMITIVE_HEIGHTS_M
        }
        for slot, expected in self.EXPECTED_PRIMITIVE_HEIGHTS_M.items():
            if not np.isclose(observed[slot], expected, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"V10.11b slot {slot} height {observed[slot]} != {expected}"
                )
        th["pact_v1011b_tall_primitive_heights_m"] = observed
        th["pact_v1011b_footprints_unchanged"] = True
        return th


class PactPlaceCorridorV1011C33PctTallerPrimitiveSampler(
    PactPlaceCorridorV1011BTallPrimitiveSampler
):
    """V10.11b successor with all three primitive heights scaled by 1.33."""

    PACT_PLACE_ENVIRONMENT_VERSION = PACT_PLACE_V1011C_ENVIRONMENT_VERSION
    EXPECTED_PRIMITIVE_HEIGHTS_M = {
        "01": 0.32585,
        "08": 0.23940,
        "09": 0.23940,
    }
    # V10.11c explicitly amends only the inherited vessel-height ceiling. All
    # older samplers retain PactPlaceCorridorV9Sampler's 0.25 m maximum.
    VESSEL_HEIGHT_RANGE_M = (0.15, 0.32585000000000003)

    def _draw_theta(self):
        # Skip V10.11b's version-specific labelling while retaining V10.11's
        # exact sampling and layout implementation.
        th = PactPlaceCorridorV1011MixedClutterSampler._draw_theta(self)
        palette = {
            str(item["slot"]): item
            for item in list(th.get("pact_clutter_palette") or [])
        }
        observed = {
            slot: float(palette[slot]["dimensions_m"][2])
            for slot in self.EXPECTED_PRIMITIVE_HEIGHTS_M
        }
        for slot, expected in self.EXPECTED_PRIMITIVE_HEIGHTS_M.items():
            if not np.isclose(observed[slot], expected, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"V10.11c slot {slot} height {observed[slot]} != {expected}"
                )
        th["pact_v1011c_primitive_heights_m"] = observed
        th["pact_v1011c_height_multiplier_from_v1011b"] = 1.33
        th["pact_v1011c_footprints_unchanged"] = True
        return th

    def _ensure_manifest_row(self) -> dict[str, Any]:
        # Without this the inherited V9.3 implementation would hand back a plain
        # V9.5 row -- eight mesh objects and no primitives -- and the layout
        # check below would reject it.
        if self._pact_manifest_row_is_explicit:
            row = self._pact_manifest_row or {}
            if "pose_id" not in row:
                raise ValueError(
                    "an explicit V10.11c manifest row must bind pose_id"
                )
            return row
        try:
            house_index = int(self.current_house_index)
        except (AttributeError, TypeError, ValueError):
            house_index = 0
        if self._pact_manifest_row is None or self._pact_auto_house_index != house_index:
            self._pact_manifest_row = self._auto_manifest_row_for_house(house_index)
            self._pact_auto_house_index = house_index
        return self._pact_manifest_row

    @staticmethod
    def _auto_manifest_row_for_house(house_index: int) -> dict[str, Any]:
        family, side, pose = v1011_cell(house_index)
        return build_v1011c_manifest_row(family, side, pose)


class PactPlaceCorridorV1011DRandomizedLayoutSampler(
    PactPlaceCorridorV1011C33PctTallerPrimitiveSampler
):
    """V10.11c clutter, with every clutter item's position randomized.

    V10.11c inherits the frozen V9.5 layout, in which the two plates sit at
    exactly (0.980, -0.220) and (1.090, +0.300) in all eight family/side
    combinations and never move, while the two vessels receive only the
    inherited millimetre-scale jitter. V10.11d keeps the palette, the primitive
    shapes and every height byte-identical to V10.11c and redraws the centres of
    slots 01/03/04/06 per episode. Slots 08/09 keep their inherited
    target-relative annulus and are sampled after these four, so they see the
    final occupancy.

    Nothing here is free-for-all. Every candidate is rejected unless it stays
    inside the bench shell, clears every already-placed body and the target, and
    -- for the route-bearing slot 01 -- still satisfies both registered route
    predicates. Slot 01's admissible y window is only about 60 mm wide and its
    sign depends on the panel side, so the predicates are re-evaluated per
    candidate rather than trusted to a hardcoded box.
    """

    PACT_PLACE_ENVIRONMENT_VERSION = PACT_PLACE_V1011D_ENVIRONMENT_VERSION
    # Proposal boxes only. The registered predicates below decide admissibility;
    # these merely bound the proposal distribution.
    SLOT_RANDOMIZATION_BOXES_M = {
        # Slot 01's floor is 0.650 rather than the route predicates' wider
        # admissible span: the two vessels need 98 mm of x separation and slot
        # 06 cannot go below 0.545 without leaving the bench shell, so a lower
        # floor here starves slot 06 and the layout fails as a whole.
        "01": {"x": (0.650, 0.740), "y": (-0.055, 0.055)},
        "06": {"x": (0.545, 0.600), "y": (-0.050, 0.050)},
        "03": {"x": (0.920, 1.240), "y": (-0.320, 0.320)},
        "04": {"x": (0.920, 1.240), "y": (-0.320, 0.320)},
    }
    # Most constrained first: a rejected vessel is far more likely than a
    # rejected plate, and placing it last would waste the plates' draws.
    RANDOMIZED_SLOT_ORDER = ("01", "06", "03", "04")
    BASE_MAX_CANDIDATES = 96

    def _randomize_base_slot_centers(self, by_slot, layout) -> dict[str, Any]:
        from molmo_spaces.data_generation.pact_place.contracts import (
            panel_corridor_metrics,
            route_blocker_metrics,
        )

        # No target-clearance rule is applied here. In a measured V10.11c row
        # the cup AABB overlaps slot 01's in all three axes -- 55 mm in x and
        # 15 mm in y -- while the episode still records zero forbidden initial
        # contact, because the cup mesh and the cylinder do not actually touch.
        # Any conservative planar separation would therefore reject V10.11c's
        # own working layout. The runtime settle and initial-contact check stay
        # the authority for cup/clutter overlap, exactly as in V10.11c.
        shell_low = np.asarray(self.CLUTTER_WORKSPACE_LOW, dtype=float)
        shell_high = np.asarray(self.CLUTTER_WORKSPACE_HIGH, dtype=float)
        placed: list[tuple[np.ndarray, np.ndarray]] = []
        detail: dict[str, Any] = {
            "target_clearance_delegated_to_runtime_contact_check": True,
            "slots": {},
        }
        for slot in self.RANDOMIZED_SLOT_ORDER:
            item = by_slot[slot]
            half = np.asarray(item["half_m"], dtype=float)
            box = self.SLOT_RANDOMIZATION_BOXES_M[slot]
            object_radius = float(np.linalg.norm(half[:2]))
            rejections: dict[str, int] = {}

            def reject(reason: str) -> None:
                rejections[reason] = rejections.get(reason, 0) + 1

            for candidate_index in range(self.BASE_MAX_CANDIDATES):
                x = float(np.random.uniform(*box["x"]))
                y = float(np.random.uniform(*box["y"]))
                center = np.asarray(
                    [x, y, SHELF_TOP_Z + float(half[2])], dtype=float
                )
                if np.any(center - half < shell_low - 1e-12) or np.any(
                    center + half > shell_high + 1e-12
                ):
                    reject("escapes_bench_shell")
                    continue
                if any(
                    not self._planar_boxes_separated(
                        center, half, other_center, other_half,
                        self.NEAR_OBJECT_GAP_M,
                    )
                    for other_center, other_half in placed
                ):
                    reject("overlaps_placed_clutter")
                    continue
                if slot == "01":
                    trial = dict(layout)
                    trial["objects"] = [
                        {**item, "center_m": center.tolist(), "half_m": half.tolist()}
                    ]
                    trial["route_blocker_center_xy_m"] = center[:2].tolist()
                    try:
                        route = route_blocker_metrics(trial)
                        corridor = panel_corridor_metrics(trial)
                    except ValueError:
                        reject("route_predicate_raised")
                        continue
                    if not route["detour_admitted"]:
                        reject("closes_nominal_detour")
                        continue
                    if not corridor["detour_admitted"]:
                        reject("closes_panel_corridor")
                        continue
                item["center_m"] = center.tolist()
                placed.append((center, half))
                detail["slots"][slot] = {
                    "center_m": center.tolist(),
                    "candidate_index": candidate_index,
                    "proposal_box_m": {
                        "x": list(box["x"]), "y": list(box["y"]),
                    },
                    "object_planar_bounding_radius_m": object_radius,
                    "rejections": dict(rejections),
                }
                break
            else:
                raise ValueError(
                    f"V10.11d could not place slot {slot} in "
                    f"{self.BASE_MAX_CANDIDATES} candidates: {rejections}"
                )
        return detail

    def _draw_theta(self):
        th = PactPlaceCorridorV1011C33PctTallerPrimitiveSampler._draw_theta(self)
        layout = th.get("pact_clutter_layout") or {}
        th["pact_place_environment_version"] = self.PACT_PLACE_ENVIRONMENT_VERSION
        th["pact_v1011d_base_slot_randomization"] = layout.get(
            "base_slot_randomization"
        )
        th["pact_v1011d_randomized_slots"] = list(self.RANDOMIZED_SLOT_ORDER)
        th["pact_v1011d_all_clutter_randomized"] = True
        return th

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        randomization = th.get("pact_v1011d_base_slot_randomization") or {}
        placed = set((randomization.get("slots") or {}))
        expected = set(self.RANDOMIZED_SLOT_ORDER)
        if placed != expected:
            raise ValueError(
                f"V10.11d randomized {sorted(placed)}, expected {sorted(expected)}"
            )

    def _ensure_manifest_row(self) -> dict[str, Any]:
        # Without this the inherited V9.3 implementation would hand back a plain
        # V9.5 row -- eight mesh objects and no primitives -- and the layout
        # check below would reject it.
        if self._pact_manifest_row_is_explicit:
            row = self._pact_manifest_row or {}
            if "pose_id" not in row:
                raise ValueError(
                    "an explicit V10.11d manifest row must bind pose_id"
                )
            return row
        try:
            house_index = int(self.current_house_index)
        except (AttributeError, TypeError, ValueError):
            house_index = 0
        if self._pact_manifest_row is None or self._pact_auto_house_index != house_index:
            self._pact_manifest_row = self._auto_manifest_row_for_house(house_index)
            self._pact_auto_house_index = house_index
        return self._pact_manifest_row

    @staticmethod
    def _auto_manifest_row_for_house(house_index: int) -> dict[str, Any]:
        family, side, pose = v1011_cell(house_index)
        return build_v1011d_manifest_row(family, side, pose)


__all__ = [
    "PactPlaceCorridorPolicy",
    "PactPlaceCorridorPolicyConfig",
    "PactPlaceCorridorTask",
    "PactPlaceCorridorV2Sampler",
    "PactPlaceCorridorV93Sampler",
    "PactPlaceCorridorV1010FourObjectSampler",
    # V10.11a/b are intermediate bases for V10.11c and are deliberately not
    # exported; only the two qualified endpoints are public.
    "PactPlaceCorridorV1011C33PctTallerPrimitiveSampler",
    "PactPlaceCorridorV1011DRandomizedLayoutSampler",
    "PactPlaceV5Sampler",
    "PactPlaceV95RealClutterSampler",
]
