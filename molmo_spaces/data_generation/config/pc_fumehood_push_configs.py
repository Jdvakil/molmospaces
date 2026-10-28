"""Datagen configs for the pc_fumehood_push suite (non-prehensile push in the paper fume hood).

Check (preflight, 1 worker, a few episodes, failures saved, clearance diagnostics on):
    python -m molmo_spaces.data_generation.main FrankaSkinPCFumehoodPushCheckConfig
Collection (>= 220 saved successes, successes only):
    python -m molmo_spaces.data_generation.main FrankaSkinPCFumehoodPushConfig

Observation contract (same as the v12 paper config): hybrid-skin FR3 (40 sensors), exo_camera_1 +
wrist_camera RGB, policy_dt 66 ms, skin sampled every 16.67 ms, actuator noise off, no texture or
lighting randomization, viz_sensor_rgb off. Layout cells: house_index % 8 (see
molmo_spaces/tasks/pc_fumehood_push.py); the collection uses wrap-around house indices only to
parallelise and to bound per-house RAM (episodes are held in memory until a house is saved).
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
from molmo_spaces.tasks.pc_fumehood_push import (
    N_CELLS,
    PcFumehoodPushCheckSampler,
    PcFumehoodPushCollectSampler,
    PcFumehoodPushPolicyConfig,
    PcFumehoodPushTask,
)

_CUSTOM_SCENES = Path(__file__).resolve().parents[1] / "custom_scenes"
_SCENE = str(_CUSTOM_SCENES / "pc_fumehood_push.xml")


def _sampler_config(sampler, n_houses: int, house_inds: list[int], samples_per_house: int,
                    attempts_multiplier: int = 3) -> PickTaskSamplerConfig:
    return PickTaskSamplerConfig(
        task_sampler_class=sampler,
        scene_xml_paths=[_SCENE] * n_houses,
        house_inds=house_inds,
        samples_per_house=samples_per_house,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=attempts_multiplier,
        max_allowed_sequential_task_sampler_failures=50,
        max_allowed_sequential_rollout_failures=50,
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_object_z_offset_random_min=0.0,
        robot_object_z_offset_random_max=0.0,
        robot_placement_rotation_range_rad=0.0,
        randomize_textures=False,
        randomize_lighting=False,
    )


@register_config("FrankaSkinPCFumehoodPushCheckConfig")
class FrankaSkinPCFumehoodPushCheckConfig(FrankaSkinHybridObstacleCheckConfig):
    """Preflight: 4 episodes over 4 different layout cells, failures saved for inspection."""

    task_type: str = "pick"
    camera_config: FrankaSkinHybridCameraSystem = FrankaSkinHybridCameraSystem()
    policy_config: BasePolicyConfig = PcFumehoodPushPolicyConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PcFumehoodPushTask)
    task_horizon: int | None = 600
    policy_dt_ms: float = 66.0
    proximity_sensor_period_ms: float = 16.6667
    end_on_success: bool = False
    filter_for_successful_trajectories: bool = False
    viz_sensor_rgb: bool = False
    num_workers: int = 1
    seed: int | None = 2026
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushCheckSampler, N_CELLS, [0, 3, 5, 6], 1, attempts_multiplier=2
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_fumehood_push_check"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False

    @property
    def tag(self) -> str:
        return "franka_skin_pc_fumehood_push_check"


@register_config("FrankaSkinPCFumehoodPushConfig")
class FrankaSkinPCFumehoodPushConfig(FrankaSkinPCFumehoodPushCheckConfig):
    """Collection: 256 wrap-around houses (cell = house % 8, 32 houses per cell) x 1 saved
    success = 256 target episodes, ONE in-process worker (~5 GB RSS; the machine is shared and
    a single-worker run has no forkserver/main-process overhead). Throughput is added with
    FrankaSkinPCFumehoodPushPart2Config, a second single-worker run over DISJOINT houses.
    1 per house bounds RAM (the pipeline holds a house's episodes in memory until the house is
    saved, ~0.55 GB/episode with the 624x352 RGB+depth exo/wrist streams) and makes saves
    frequent. Saved = task success AND the scripted expert touched nothing but the jar
    (PcFumehoodPushCollectSampler; with a learned policy registered, e.g. closed-loop eval,
    success is the task criterion alone)."""

    filter_for_successful_trajectories: bool = True
    num_workers: int = 1
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushCollectSampler, 256, list(range(256)), 1, attempts_multiplier=4
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_fumehood_push"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_fumehood_push"


@register_config("FrankaSkinPCFumehoodPushPart2Config")
class FrankaSkinPCFumehoodPushPart2Config(FrankaSkinPCFumehoodPushConfig):
    """Extra capacity for the same collection, run side by side when RAM allows: DISJOINT house
    indices 256..511 (same cells, house % 8), so its episodes never duplicate the main run's.
    Its house_* dirs can be symlinked into the main run dir for conversion (names never
    collide)."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushCollectSampler, 512, list(range(256, 512)), 1, attempts_multiplier=4
    )


@register_config("FrankaSkinPCFumehoodPushPart3Config")
class FrankaSkinPCFumehoodPushPart3Config(FrankaSkinPCFumehoodPushConfig):
    """Extra capacity, DISJOINT houses 512..767 (see Part2)."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushCollectSampler, 768, list(range(512, 768)), 1, attempts_multiplier=4
    )


@register_config("FrankaSkinPCFumehoodPushPart4Config")
class FrankaSkinPCFumehoodPushPart4Config(FrankaSkinPCFumehoodPushConfig):
    """Extra capacity, DISJOINT houses 768..1023 (see Part2)."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushCollectSampler, 1024, list(range(768, 1024)), 1, attempts_multiplier=4
    )


import os as _os

_RANGE_START = int(_os.environ.get("PC_FH_HOUSE_START", "1024"))
_RANGE_END = int(_os.environ.get("PC_FH_HOUSE_END", str(_RANGE_START + 25)))


@register_config("FrankaSkinPCFumehoodPushRangeConfig")
class FrankaSkinPCFumehoodPushRangeConfig(FrankaSkinPCFumehoodPushConfig):
    """Top-up collection in short-lived processes (orchestrator, 2026-10-13 10:00): houses
    [PC_FH_HOUSE_START, PC_FH_HOUSE_END) read from the environment at import (default
    1024..1048), disjoint from the main / Part runs (0..1023). A single long-lived worker grew to
    16 GB RSS and was OOM-killed after ~60 houses, so a driver loop runs one process per 25 houses."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushCollectSampler, _RANGE_END, list(range(_RANGE_START, _RANGE_END)), 1,
        attempts_multiplier=4
    )


# ============================================================================================
# pc_fumehood_push_hard (2026-10-13 15:xx): hazards moved onto the arm (lower sash, panes and lab
# items 1-3 cm off the arm envelope), DART-style action noise and light-contact demos in the
# training mix. Strictly additive: the classes above keep their behaviour.
#   eval / definition : FrankaSkinPCFumehoodPushHardConfig      (noise OFF, houses 0..255)
#   preflight         : FrankaSkinPCFumehoodPushHardCheckConfig (noise ON, failures saved)
#   collection        : FrankaSkinPCFumehoodPushHardCollectConfig (noise ON, light-contact rule,
#                       houses [PC_FH_HARD_HOUSE_START, PC_FH_HARD_HOUSE_END) from the environment,
#                       one short-lived process per ~25 houses: a long-lived worker leaks RAM)
# ============================================================================================
from molmo_spaces.tasks.pc_fumehood_push_hard import (  # noqa: E402
    PcFumehoodPushHardCheckSampler,
    PcFumehoodPushHardCollectSampler,
    PcFumehoodPushHardPolicyConfig,
    PcFumehoodPushHardTask,
)

# DART-style noise for the hard collections: per policy step the executed arm command gets
# TCP noise with std = 0.7 x |commanded TCP step| per axis (truncated at 3 cm / 0.08 rad),
# mapped to joints through the Jacobian; the recorded action stays the clean expert command.
# Probe (32 expert episodes, hard layouts): scale 0.3 / 0.5 -> no hazard contact at all;
# 0.7 -> 3/16 light contacts (8-37 frames of 2 ms), 1/16 missed goal; mean executed-vs-label
# joint offset 0.045 rad.
PC_FH_HARD_NOISE = dict(
    action_scale_factor=0.7,
    rotation_noise_scale=0.1,
    max_tcp_position_noise=0.03,
    max_tcp_rotation_noise=0.08,
)


def _hard_noise(cfg, on: bool) -> None:
    nc = cfg.robot_config.action_noise_config
    nc.enabled = bool(on)
    if on:
        for k, v in PC_FH_HARD_NOISE.items():
            setattr(nc, k, v)


@register_config("FrankaSkinPCFumehoodPushHardCheckConfig")
class FrankaSkinPCFumehoodPushHardCheckConfig(FrankaSkinPCFumehoodPushCheckConfig):
    """Preflight of the hard COLLECTION condition: noise on, light-contact rule, AABB + exact
    clearance diagnostics, 4 cells, failures saved for inspection."""

    policy_config: BasePolicyConfig = PcFumehoodPushHardPolicyConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PcFumehoodPushHardTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushHardCheckSampler, N_CELLS, [0, 3, 5, 6], 1, attempts_multiplier=2
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_fumehood_push_hard_check"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _hard_noise(self, True)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_fumehood_push_hard_check"


@register_config("FrankaSkinPCFumehoodPushHardConfig")
class FrankaSkinPCFumehoodPushHardConfig(FrankaSkinPCFumehoodPushCheckConfig):
    """Eval / definition config of pc_fumehood_push_hard (eval_act_place.py --env
    pc_fumehood_push_hard): hard layouts, ACTION NOISE OFF, houses 0..255 (cell = house % 8),
    successes only. A learned policy is scored on the task criterion alone."""

    policy_config: BasePolicyConfig = PcFumehoodPushHardPolicyConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PcFumehoodPushHardTask)
    filter_for_successful_trajectories: bool = True
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushHardCollectSampler, 256, list(range(256)), 1, attempts_multiplier=4
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_fumehood_push_hard"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _hard_noise(self, False)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_fumehood_push_hard"


_HARD_START = int(_os.environ.get("PC_FH_HARD_HOUSE_START", "0"))
_HARD_END = int(_os.environ.get("PC_FH_HARD_HOUSE_END", str(_HARD_START + 25)))


@register_config("FrankaSkinPCFumehoodPushHardCollectConfig")
class FrankaSkinPCFumehoodPushHardCollectConfig(FrankaSkinPCFumehoodPushHardConfig):
    """Hard collection: same sampler / cells as the eval config, DART noise ON, light-contact
    save rule; houses [PC_FH_HARD_HOUSE_START, PC_FH_HARD_HOUSE_END) read from the environment
    at import (default 0..24). Run one process per ~25 houses (driver loop), disjoint ranges per
    worker."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PcFumehoodPushHardCollectSampler, _HARD_END, list(range(_HARD_START, _HARD_END)), 1,
        attempts_multiplier=4
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _hard_noise(self, True)


__all__ = [
    "FrankaSkinPCFumehoodPushCheckConfig",
    "FrankaSkinPCFumehoodPushConfig",
    "FrankaSkinPCFumehoodPushPart2Config",
    "FrankaSkinPCFumehoodPushPart3Config",
    "FrankaSkinPCFumehoodPushPart4Config",
    "FrankaSkinPCFumehoodPushHardCheckConfig",
    "FrankaSkinPCFumehoodPushHardConfig",
    "FrankaSkinPCFumehoodPushHardCollectConfig",
]
