"""Realistic-kitchen suites for the 40-sensor hybrid-skin FR3 (see molmo_spaces/tasks/pc_kitchen.py).

* ``FrankaSkinPCKitchenPnPCheckConfig`` / ``FrankaSkinPCKitchenPnPConfig``   -> pc_kitchen_pnp
  (take a cup from under the upper cabinets, put it on a tray further along the counter)
* ``FrankaSkinPCKitchenPushCheckConfig`` / ``FrankaSkinPCKitchenPushConfig`` -> pc_kitchen_push
  (non-prehensile: push a biscuit tin with the closed gripper back under the cabinet edge onto a
  flat goal trivet)

Eight layout cells (fridge flank x upper-cabinet height x upper-cabinet depth), one XML each.
Collection configs use wrap-around house indices (index i -> cell i % 8) so the pipeline can hand
many short houses to a few workers: each house keeps only ``samples_per_house`` episodes in RAM
before it is written, which keeps worker memory flat.

Run from submodules/molmospaces:
    python -m molmo_spaces.data_generation.main FrankaSkinPCKitchenPnPCheckConfig
"""

from pathlib import Path

from molmo_spaces.configs import BasePolicyConfig
from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem
from molmo_spaces.configs.task_configs import PickTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinHybridObstacleCheckConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pc_kitchen import (
    HARD_N_CELLS,
    N_CELLS,
    PCKitchenPnPCheckSampler,
    PCKitchenPnPHardCollectCheckSampler,
    PCKitchenPnPHardCollectGrazeSampler,
    PCKitchenPnPHardCollectSampler,
    PCKitchenPnPHardCollectTask,
    PCKitchenPnPHardSampler,
    PCKitchenPnPHardTask,
    PCKitchenPnPPolicyConfig,
    PCKitchenPnPSampler,
    PCKitchenPnPTask,
    PCKitchenPushCheckSampler,
    PCKitchenPushPolicyConfig,
    PCKitchenPushSampler,
    PCKitchenPushTask,
    hard_scene_paths,
    scene_paths,
)

# Collection layout: 48 houses x 6 saved episodes = 288 successes per suite (>= 220 target),
# every cell visited 6 times. Small houses keep the in-RAM episode buffer short (RGB-only
# histories, ~0.25 GB per held episode) while amortising the ~30 s scene load.
_COLLECT_HOUSES = 48
_COLLECT_PER_HOUSE = 6


def _sampler_cfg(sampler, houses: list[int], per_house: int, paths=scene_paths) -> PickTaskSamplerConfig:
    return PickTaskSamplerConfig(
        task_sampler_class=sampler,
        # always at least one path per cell, so an evaluator can address cells 0..7 directly
        scene_xml_paths=paths(max(max(houses) + 1, N_CELLS)),
        house_inds=list(houses),
        samples_per_house=per_house,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=4,
        max_allowed_sequential_task_sampler_failures=40,
        max_allowed_sequential_rollout_failures=40,
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_object_z_offset_random_min=0.0,
        robot_object_z_offset_random_max=0.0,
        robot_placement_rotation_range_rad=0.0,
        randomize_textures=False,
        randomize_lighting=False,
    )


def _rgb_only(cfg) -> None:
    """Collection runs record exo/wrist RGB (what the converter and policies read) but not their
    float depth maps, which doubled the per-episode RAM held until a house is saved. Skin depth is
    unaffected."""
    cams = FrankaSkinHybridCameraSystem()
    for cam in cams.cameras:
        if cam.name in ("exo_camera_1", "wrist_camera"):
            cam.record_depth = False
    cfg.camera_config = cams


class _PCKitchenBaseConfig(FrankaSkinHybridObstacleCheckConfig):
    """Common observation / control contract (matches the v12 hybrid-skin config)."""

    scene_dataset: str = "user"
    camera_config: FrankaSkinHybridCameraSystem = FrankaSkinHybridCameraSystem()
    policy_dt_ms: float = 66.0
    proximity_sensor_period_ms: float = 16.6667
    task_horizon: int | None = 600
    end_on_success: bool = False
    viz_sensor_rgb: bool = False
    num_workers: int = 1

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
        _rgb_only(self)


# ------------------------------------------------------------------ pick-and-place
@register_config("FrankaSkinPCKitchenPnPCheckConfig")
class FrankaSkinPCKitchenPnPCheckConfig(_PCKitchenBaseConfig):
    """Preflight: cells 0, 3, 5, 6 x 1 episode, every clutter slot filled, failures saved too."""

    policy_config: BasePolicyConfig = PCKitchenPnPPolicyConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCKitchenPnPTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(PCKitchenPnPCheckSampler, [0, 3, 5, 6], 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_pnp_check"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_pnp_check"


@register_config("FrankaSkinPCKitchenPnPConfig")
class FrankaSkinPCKitchenPnPConfig(FrankaSkinPCKitchenPnPCheckConfig):
    """Collection: 48 wrap-around houses (6 per cell) x 6 successes = 288 episodes."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(
        PCKitchenPnPSampler, list(range(_COLLECT_HOUSES)), _COLLECT_PER_HOUSE
    )
    filter_for_successful_trajectories: bool = True
    seed: int | None = None          # per-worker urandom seed (see _PCKitchenSampler.__init__)
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_pnp"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_pnp"


# ------------------------------------------------------------------ push (non-prehensile)
@register_config("FrankaSkinPCKitchenPushCheckConfig")
class FrankaSkinPCKitchenPushCheckConfig(_PCKitchenBaseConfig):
    """Preflight: cells 0, 3, 5, 6 x 1 episode, every clutter slot filled, failures saved too."""

    policy_config: BasePolicyConfig = PCKitchenPushPolicyConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCKitchenPushTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(PCKitchenPushCheckSampler, [0, 3, 5, 6], 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_push_check"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_push_check"


@register_config("FrankaSkinPCKitchenPushConfig")
class FrankaSkinPCKitchenPushConfig(FrankaSkinPCKitchenPushCheckConfig):
    """Collection: 48 wrap-around houses (6 per cell) x 6 successes = 288 episodes."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(
        PCKitchenPushSampler, list(range(_COLLECT_HOUSES)), _COLLECT_PER_HOUSE
    )
    filter_for_successful_trajectories: bool = True
    seed: int | None = None          # per-worker urandom seed (see _PCKitchenSampler.__init__)
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_push"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_push"


# ------------------------------------------------------------------ pick-and-place, HARD
# pc_kitchen_pnp_hard (2026-10-13): lower / deeper upper cabinets, fridge closer to the can,
# canisters 1-3 cm from the swept hand (see the HARD section of molmo_spaces/tasks/pc_kitchen.py).
# FrankaSkinPCKitchenPnPHardConfig is the eval / definition config (noise OFF, plain success).
# The collection configs share its sampler scenes and house cells but turn DART-style action noise
# ON (executed command only; the recorded action is the clean expert command) and save successes
# with LIGHT hazard contact (<= 150 x 2 ms frames, <= 1 cm deep; PCKitchenPnPHardCollectTask).
_HARD_HOUSES = list(range(48))     # wrap-around: house h -> hard cell h % 8 (eval cells = these)
_HARD_PER_HOUSE = 4
# Collection: three processes on disjoint house ranges (24 houses each, 4 saved per house).
_HARD_PARTS = {1: list(range(0, 24)), 2: list(range(24, 48)), 3: list(range(48, 72)), 4: list(range(72, 96))}
HARD_ACTION_NOISE = {
    "action_scale_factor": 0.3,       # noise std = 0.3 x commanded TCP step (per 66 ms action)
    "rotation_noise_scale": 0.1,
    "max_tcp_position_noise": 0.025,  # truncated at 2.5 cm
    "max_tcp_rotation_noise": 0.1,
}


def _noise_on(cfg) -> None:
    noise = cfg.robot_config.action_noise_config
    noise.enabled = True
    for k, v in HARD_ACTION_NOISE.items():
        setattr(noise, k, v)


@register_config("FrankaSkinPCKitchenPnPHardConfig")
class FrankaSkinPCKitchenPnPHardConfig(_PCKitchenBaseConfig):
    """pc_kitchen_pnp_hard eval / definition config: 48 wrap-around houses (cell = house % 8),
    action noise OFF, plain pc_kitchen_pnp success test."""

    policy_config: BasePolicyConfig = PCKitchenPnPPolicyConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCKitchenPnPHardTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(
        PCKitchenPnPHardSampler, _HARD_HOUSES, _HARD_PER_HOUSE, hard_scene_paths
    )
    filter_for_successful_trajectories: bool = True
    seed: int | None = None
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_pnp_hard"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_pnp_hard"


@register_config("FrankaSkinPCKitchenPnPHardCheckConfig")
class FrankaSkinPCKitchenPnPHardCheckConfig(FrankaSkinPCKitchenPnPHardConfig):
    """Preflight of the collection: cells 0, 3, 5, 6 x 1 episode, every slot filled, noise ON,
    light-contact rule ON, failures saved too."""

    policy_config: BasePolicyConfig = PCKitchenPnPPolicyConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCKitchenPnPHardCollectTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(
        PCKitchenPnPHardCollectCheckSampler, [0, 3, 5, 6], 1, hard_scene_paths
    )
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_pnp_hard_check"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_pnp_hard_check"


class _PCKitchenPnPHardCollectBase(FrankaSkinPCKitchenPnPHardConfig):
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCKitchenPnPHardCollectTask)
    filter_for_successful_trajectories: bool = True
    seed: int | None = None          # per-worker urandom seed (see _PCKitchenSampler.__init__)
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_pnp_hard"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_pnp_hard"


@register_config("FrankaSkinPCKitchenPnPHardCollectConfig")
class FrankaSkinPCKitchenPnPHardCollectConfig(_PCKitchenPnPHardCollectBase):
    """Collection part 1: houses 0-23 (3 per hard cell) x 4 saved episodes."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(
        PCKitchenPnPHardCollectSampler, _HARD_PARTS[1], _HARD_PER_HOUSE, hard_scene_paths
    )


@register_config("FrankaSkinPCKitchenPnPHardCollectPart2Config")
class FrankaSkinPCKitchenPnPHardCollectPart2Config(_PCKitchenPnPHardCollectBase):
    """Collection part 2: houses 24-47."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(
        PCKitchenPnPHardCollectSampler, _HARD_PARTS[2], _HARD_PER_HOUSE, hard_scene_paths
    )


@register_config("FrankaSkinPCKitchenPnPHardCollectPart3Config")
class FrankaSkinPCKitchenPnPHardCollectPart3Config(_PCKitchenPnPHardCollectBase):
    """Collection part 3: houses 48-71."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(
        PCKitchenPnPHardCollectSampler, _HARD_PARTS[3], _HARD_PER_HOUSE, hard_scene_paths
    )


@register_config("FrankaSkinPCKitchenPnPHardCollectPart4Config")
class FrankaSkinPCKitchenPnPHardCollectPart4Config(_PCKitchenPnPHardCollectBase):
    """Collection part 4 (top-up): houses 72-95, grazes more often and touching
    (PCKitchenPnPHardCollectGrazeSampler) to bring the light-contact share toward ~15 %."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_cfg(
        PCKitchenPnPHardCollectGrazeSampler, _HARD_PARTS[4], _HARD_PER_HOUSE, hard_scene_paths
    )


assert N_CELLS == 8
assert HARD_N_CELLS == 8

__all__ = [
    "FrankaSkinPCKitchenPnPCheckConfig",
    "FrankaSkinPCKitchenPnPConfig",
    "FrankaSkinPCKitchenPushCheckConfig",
    "FrankaSkinPCKitchenPushConfig",
    "FrankaSkinPCKitchenPnPHardConfig",
    "FrankaSkinPCKitchenPnPHardCheckConfig",
    "FrankaSkinPCKitchenPnPHardCollectConfig",
    "FrankaSkinPCKitchenPnPHardCollectPart2Config",
    "FrankaSkinPCKitchenPnPHardCollectPart3Config",
    "FrankaSkinPCKitchenPnPHardCollectPart4Config",
]
