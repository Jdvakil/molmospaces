"""``pc_shelf_blind`` (2026-10-14, ICLR rebuttal): retrieve a can from deep inside a tall pantry
bay of the pc_shelf room; a hidden under-shelf basket (left or right of the can's lane, invisible
to exo_camera_1 and wrist_camera) threatens the skinned wrist knuckle / forearm. Design notes,
camera analysis and the hazard layout: molmo_spaces/tasks/pc_shelf_blind.py.

* ``FrankaSkinPCShelfBlindConfig``          definition / eval world (8 cells x 1, noise off,
  no contact gate). Env version pc_shelf_blind_v1.
* ``FrankaSkinPCShelfBlindCheckConfig``     collection recipe preflight (noise on, gate on).
* ``FrankaSkinPCShelfBlindCollect*Config``  collection, DISJOINT house ranges of 25 houses,
  2 saved episodes per house, 1 worker per process.
"""

from pathlib import Path

from molmo_spaces.configs import BasePolicyConfig, BaseRobotConfig
from molmo_spaces.configs.robot_configs import FrankaSkinHybridRobotConfig
from molmo_spaces.configs.task_configs import PickTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config.pc_shelf_configs import (
    HARD_CONTACT_GATE_FRAMES,
    _noise_on,
    _PCShelfBaseConfig,
    _rgb_only,
    _sampler_config,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pc_shelf import N_CELLS
from molmo_spaces.tasks.pc_shelf_blind import (
    PC_SHELF_BLIND_READY_QPOS,
    PC_SHELF_BLIND_SCENES,
    PCShelfBlindExpertConfig,
    PCShelfBlindSampler,
    PCShelfBlindTask,
)


def _blind_sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    cfg = _sampler_config(PCShelfBlindSampler, house_inds, samples_per_house)
    n = max(house_inds) + 1
    cfg.scene_xml_paths = [PC_SHELF_BLIND_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return cfg


def _blind_robot() -> FrankaSkinHybridRobotConfig:
    return FrankaSkinHybridRobotConfig(
        base_size=[0.4, 0.4, 0.35],
        init_qpos={"arm": list(PC_SHELF_BLIND_READY_QPOS), "gripper": [0.00296, 0.00296]},
        init_qpos_noise_range={"arm": [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.06]},
    )


@register_config("FrankaSkinPCShelfBlindConfig")
class FrankaSkinPCShelfBlindConfig(_PCShelfBaseConfig):
    """pc_shelf_blind DEFINITION / EVAL world (eval_act_place.py --env pc_shelf_blind): the 8
    blind cells x 1 episode, action noise OFF, no contact gate, failures saved too."""

    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCShelfBlindExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCShelfBlindTask)
    task_sampler_config: PickTaskSamplerConfig = _blind_sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_blind_def"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_blind_v1"


@register_config("FrankaSkinPCShelfBlindCheckConfig")
class FrankaSkinPCShelfBlindCheckConfig(FrankaSkinPCShelfBlindConfig):
    """Preflight of the collection recipe: 8 cells x 1 episode, noise ON, contact gate ON,
    everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCShelfBlindExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_blind_check"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_blind_check"


@register_config("FrankaSkinPCShelfBlindCollectConfig")
class FrankaSkinPCShelfBlindCollectConfig(FrankaSkinPCShelfBlindCheckConfig):
    """Collection Part1: houses 0..24 (house k = blind cell k % 8) x 2 saved episodes, noise ON,
    saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 1 worker."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _blind_sampler_config(list(range(0, 25)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_blind"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_blind_v1"


@register_config("FrankaSkinPCShelfBlindCollectPart2Config")
class FrankaSkinPCShelfBlindCollectPart2Config(FrankaSkinPCShelfBlindCollectConfig):
    """Collection: DISJOINT houses 25..49, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _blind_sampler_config(list(range(25, 50)), 2)


@register_config("FrankaSkinPCShelfBlindCollectPart3Config")
class FrankaSkinPCShelfBlindCollectPart3Config(FrankaSkinPCShelfBlindCollectConfig):
    """Collection: DISJOINT houses 50..74, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _blind_sampler_config(list(range(50, 75)), 2)


@register_config("FrankaSkinPCShelfBlindCollectPart4Config")
class FrankaSkinPCShelfBlindCollectPart4Config(FrankaSkinPCShelfBlindCollectConfig):
    """Collection: DISJOINT houses 75..99, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _blind_sampler_config(list(range(75, 100)), 2)


@register_config("FrankaSkinPCShelfBlindCollectPart5Config")
class FrankaSkinPCShelfBlindCollectPart5Config(FrankaSkinPCShelfBlindCollectConfig):
    """Collection: DISJOINT houses 100..124, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _blind_sampler_config(list(range(100, 125)), 2)


@register_config("FrankaSkinPCShelfBlindCollectPart6Config")
class FrankaSkinPCShelfBlindCollectPart6Config(FrankaSkinPCShelfBlindCollectConfig):
    """Collection: DISJOINT houses 125..149, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _blind_sampler_config(list(range(125, 150)), 2)


__all__ = [
    "FrankaSkinPCShelfBlindConfig",
    "FrankaSkinPCShelfBlindCheckConfig",
    "FrankaSkinPCShelfBlindCollectConfig",
    "FrankaSkinPCShelfBlindCollectPart2Config",
    "FrankaSkinPCShelfBlindCollectPart3Config",
    "FrankaSkinPCShelfBlindCollectPart4Config",
    "FrankaSkinPCShelfBlindCollectPart5Config",
    "FrankaSkinPCShelfBlindCollectPart6Config",
]
