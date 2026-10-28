"""PC-ACT rebuttal suites in a realistic storage-room BOOKCASE (see molmo_spaces/tasks/pc_shelf.py).

* ``pc_shelf_pnp``  -- pick a can off one work board of the bookcase and set it on the coaster mat
  on the other work board (prehensile).  Env version ``pc_shelf_pnp_v1``.
* ``pc_shelf_push`` -- with the gripper closed, slide a box from the front of a work board into
  the labelled gap near the back panel (non-prehensile).  Env version ``pc_shelf_push_v1``.

Eight layout cells (house_index % 8) = eight static bookcase XMLs (custom_scenes/pc_shelf_c*.xml).
Robot: FR3 + Robotiq with the 40-sensor hybrid skin, exo_camera_1 + wrist_camera RGB.
"""

from pathlib import Path

from molmo_spaces.configs import BasePolicyConfig, BaseRobotConfig
from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem
from molmo_spaces.configs.robot_configs import FrankaSkinHybridRobotConfig
from molmo_spaces.configs.task_configs import PickTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinHybridObstacleCheckConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pc_shelf import (
    N_CELLS,
    PC_SHELF_READY_QPOS,
    PC_SHELF_SCENES,
    PCShelfPnPExpertConfig,
    PCShelfPnPSampler,
    PCShelfPnPTask,
    PCShelfPushExpertConfig,
    PCShelfPushSampler,
    PCShelfPushTask,
)


def _sampler_config(sampler, house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    n = max(house_inds) + 1
    # one XML per house index; index k uses cell k % 8
    paths = [PC_SHELF_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return PickTaskSamplerConfig(
        task_sampler_class=sampler,
        scene_xml_paths=paths,
        house_inds=list(house_inds),
        samples_per_house=samples_per_house,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=3,
        max_allowed_sequential_task_sampler_failures=60,
        max_allowed_sequential_rollout_failures=60,
        max_allowed_sequential_irrecoverable_failures=1000,
        robot_placement_rotation_range_rad=0.0,
        randomize_textures=False,
        randomize_lighting=False,
    )


def _robot() -> FrankaSkinHybridRobotConfig:
    return FrankaSkinHybridRobotConfig(
        base_size=[0.4, 0.4, 0.35],
        init_qpos={"arm": list(PC_SHELF_READY_QPOS), "gripper": [0.00296, 0.00296]},
        # small start-pose jitter (TCP displacement a few cm) so the policy sees varied starts
        init_qpos_noise_range={"arm": [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.06]},
    )


class _PCShelfBaseConfig(FrankaSkinHybridObstacleCheckConfig):
    """Shared contract: hybrid skin robot, exo + wrist RGB, 66 ms policy step, no action noise."""

    task_type: str = "pick"
    robot_config: BaseRobotConfig = _robot()
    camera_config: FrankaSkinHybridCameraSystem = FrankaSkinHybridCameraSystem()
    policy_dt_ms: float = 66.0
    proximity_sensor_period_ms: float = 16.6667
    end_on_success: bool = False
    viz_sensor_rgb: bool = False
    task_horizon: int | None = 600
    num_workers: int = 1

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


def _rgb_only(cfg) -> None:
    """Collection runs: keep exo/wrist RGB (what the converter and the policies read) but do not
    record their float depth maps -- they doubled the per-episode RAM the pipeline holds until a
    house is saved and nothing downstream reads them. Skin depth is unaffected."""
    cams = FrankaSkinHybridCameraSystem()
    for cam in cams.cameras:
        if cam.name in ("exo_camera_1", "wrist_camera"):
            cam.record_depth = False
    cfg.camera_config = cams


# ------------------------------------------------------------------------------------- PnP
@register_config("FrankaSkinPCShelfPnPCheckConfig")
class FrankaSkinPCShelfPnPCheckConfig(_PCShelfBaseConfig):
    """Preflight: all 8 cells x 1 episode, failures saved too (videos for inspection)."""

    policy_config: BasePolicyConfig = PCShelfPnPExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCShelfPnPTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(PCShelfPnPSampler, list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_pnp_check"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_pnp_check"


@register_config("FrankaSkinPCShelfPnPConfig")
class FrankaSkinPCShelfPnPConfig(FrankaSkinPCShelfPnPCheckConfig):
    """Collection: 48 houses (house k = cell k % 8) x 5 saved successes = 240 target, 2 workers.
    Few episodes per house keep the per-worker RAM bounded (histories are held until the house
    is saved). For evaluation use the Check config's 8-path scene list (one path per cell)."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(PCShelfPnPSampler, list(range(48)), 5)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 2
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_pnp"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_pnp_v1"


@register_config("FrankaSkinPCShelfPnPPart2Config")
class FrankaSkinPCShelfPnPPart2Config(FrankaSkinPCShelfPnPConfig):
    """Extra capacity (orchestrator, 2026-10-13): DISJOINT houses 48..95 (same cells, k % 8),
    5 saved successes each, ONE worker. Its house_* dirs merge with the main run's for
    conversion (names never collide; seeds come from urandom per worker)."""

    task_sampler_config: PickTaskSamplerConfig = _sampler_config(PCShelfPnPSampler, list(range(48, 96)), 5)
    num_workers: int = 1


# ------------------------------------------------------------------------------------ Push
@register_config("FrankaSkinPCShelfPushCheckConfig")
class FrankaSkinPCShelfPushCheckConfig(_PCShelfBaseConfig):
    """Preflight: all 8 cells x 1 episode, failures saved too (videos for inspection)."""

    policy_config: BasePolicyConfig = PCShelfPushExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCShelfPushTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(PCShelfPushSampler, list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_push_check"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_push_check"


@register_config("FrankaSkinPCShelfPushConfig")
class FrankaSkinPCShelfPushConfig(FrankaSkinPCShelfPushCheckConfig):
    """Collection: 40 houses (house k = cell k % 8) x 6 saved successes = 240 target, 2 workers.
    For evaluation use the Check config's 8-path scene list (one path per cell)."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(PCShelfPushSampler, list(range(40)), 6)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 2
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_push"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_push_v1"


# ------------------------------------------------------------------------------- PnP HARD
# pc_shelf_pnp_hard (2026-10-13, ICLR rebuttal): tighter bookcase (lower / narrower work bays,
# books 1-3 cm off the open hand's lane, deeper can; molmo_spaces/tasks/pc_shelf_hard.py).
# Collections add DART-style action noise and keep successful episodes with LIGHT arm-vs-hazard
# contact (<= HARD_CONTACT_GATE_FRAMES frames x 2 ms). The definition / eval config below has
# neither (the evaluator also forces the noise off). Nothing above this line changes.
from molmo_spaces.tasks.pc_shelf_hard import (  # noqa: E402
    PC_SHELF_HARD_SCENES,
    PCShelfPnPHardExpertConfig,
    PCShelfPnPHardSampler,
    PCShelfPnPHardTask,
)

HARD_NOISE_SCALE = 0.3           # executed-command noise std = 0.3 x commanded TCP step
HARD_NOISE_MAX_POS = 0.025       # truncated at 2.5 cm (TCP)
HARD_CONTACT_GATE_FRAMES = 150   # light contact: <= 150 control frames (0.3 s) per episode


def _hard_sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    cfg = _sampler_config(PCShelfPnPHardSampler, house_inds, samples_per_house)
    n = max(house_inds) + 1
    cfg.scene_xml_paths = [PC_SHELF_HARD_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return cfg


def _noise_on(cfg) -> None:
    noise = cfg.robot_config.action_noise_config
    noise.enabled = True
    noise.action_scale_factor = HARD_NOISE_SCALE
    noise.max_tcp_position_noise = HARD_NOISE_MAX_POS


@register_config("FrankaSkinPCShelfPnPHardConfig")
class FrankaSkinPCShelfPnPHardConfig(_PCShelfBaseConfig):
    """pc_shelf_pnp_hard DEFINITION / EVAL world (eval_act_place.py --env pc_shelf_pnp_hard):
    the 8 hard cells x 1 episode, action noise OFF, no contact gate (plain pc_shelf success),
    failures saved too. Env version pc_shelf_pnp_hard_v1. Same RGB-only camera recording as
    the collection configs."""

    robot_config: BaseRobotConfig = _robot()
    policy_config: BasePolicyConfig = PCShelfPnPHardExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCShelfPnPHardTask)
    task_sampler_config: PickTaskSamplerConfig = _hard_sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_pnp_hard"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_pnp_hard_v1"


@register_config("FrankaSkinPCShelfPnPHardCheckConfig")
class FrankaSkinPCShelfPnPHardCheckConfig(FrankaSkinPCShelfPnPHardConfig):
    """Preflight of the collection recipe: 8 cells x 1 episode, action noise ON, contact gate on
    (a heavy-contact success is scored as a failure), everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _robot()
    policy_config: BasePolicyConfig = PCShelfPnPHardExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 2

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_pnp_hard_check"


@register_config("FrankaSkinPCShelfPnPHardCollectConfig")
class FrankaSkinPCShelfPnPHardCollectConfig(FrankaSkinPCShelfPnPHardCheckConfig):
    """Collection: houses 0..47 (house k = hard cell k % 8) x 5 saved episodes, action noise ON,
    saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 2 workers."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _hard_sampler_config(list(range(48)), 5)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 2

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_pnp_hard_v1"


@register_config("FrankaSkinPCShelfPnPHardCollectPart2Config")
class FrankaSkinPCShelfPnPHardCollectPart2Config(FrankaSkinPCShelfPnPHardCollectConfig):
    """Extra capacity: DISJOINT houses 48..95 (same cells, k % 8), same recipe. Its house_* dirs
    merge with the main run's for conversion."""

    task_sampler_config: PickTaskSamplerConfig = _hard_sampler_config(list(range(48, 96)), 5)


@register_config("FrankaSkinPCShelfPnPHardCollectPart3Config")
class FrankaSkinPCShelfPnPHardCollectPart3Config(FrankaSkinPCShelfPnPHardCollectConfig):
    """Extra capacity: DISJOINT houses 96..191 (same cells, k % 8), same recipe but 2 saved
    episodes per house: a worker holds at most 2 episodes of frames before saving (smaller RSS on
    a crowded machine; 5-episode workers were OOM-killed at ~4.6 GB)."""

    task_sampler_config: PickTaskSamplerConfig = _hard_sampler_config(list(range(96, 192)), 2)


@register_config("FrankaSkinPCShelfPnPHardCollectPart4Config")
class FrankaSkinPCShelfPnPHardCollectPart4Config(FrankaSkinPCShelfPnPHardCollectConfig):
    """Extra capacity: DISJOINT houses 192..287, 2 saved episodes per house (as Part3)."""

    task_sampler_config: PickTaskSamplerConfig = _hard_sampler_config(list(range(192, 288)), 2)


@register_config("FrankaSkinPCShelfPnPHardCollectPart5Config")
class FrankaSkinPCShelfPnPHardCollectPart5Config(FrankaSkinPCShelfPnPHardCollectConfig):
    """Extra capacity: DISJOINT houses 288..383, 2 saved episodes per house (as Part3)."""

    task_sampler_config: PickTaskSamplerConfig = _hard_sampler_config(list(range(288, 384)), 2)


__all__ = [
    "FrankaSkinPCShelfPnPCheckConfig",
    "FrankaSkinPCShelfPnPConfig",
    "FrankaSkinPCShelfPushCheckConfig",
    "FrankaSkinPCShelfPushConfig",
    "FrankaSkinPCShelfPnPHardConfig",
    "FrankaSkinPCShelfPnPHardCheckConfig",
    "FrankaSkinPCShelfPnPHardCollectConfig",
    "FrankaSkinPCShelfPnPHardCollectPart2Config",
    "FrankaSkinPCShelfPnPHardCollectPart3Config",
    "FrankaSkinPCShelfPnPHardCollectPart4Config",
    "FrankaSkinPCShelfPnPHardCollectPart5Config",
]
