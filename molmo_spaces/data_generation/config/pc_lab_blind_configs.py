"""``pc_lab_blind`` (2026-10-14, ICLR rebuttal, BLINDLAB): pick a reagent bottle / beaker from the
back of a laboratory bench, under the reagent shelf, and set it on the spill mat at the bench
front; a service fixture screwed under the shelf (left or right of the lane, invisible to
exo_camera_1 and wrist_camera) threatens the skinned wrist knuckle / forearm. Design notes, camera
analysis and the hazard layout: molmo_spaces/tasks/pc_lab_blind.py.

* ``FrankaSkinPCLabBlindConfig``          definition / eval world (8 cells x 1, noise off,
  no contact gate). Env version pc_lab_blind_v1.
* ``FrankaSkinPCLabBlindCheckConfig``     collection recipe preflight (noise on, gate on, skin
  recorded once per policy step = the eval's snapshot; the older pc suites min-pool 4 substeps).
* ``FrankaSkinPCLabBlindCheck16Config``   larger preflight (16 draws, 2 per cell) for R7.
* ``FrankaSkinPCLabBlindCollect*Config``  collection, DISJOINT house ranges, 2 saved episodes per
  house, 1 worker per process (Collect..Part5: 25 houses = 50 demos each, at most 3 at a time;
  Part6: 13-house reserve). 25 houses per process: a worker holds every episode's frames until it
  exits (~0.11 GB per saved episode).
"""

from pathlib import Path

from molmo_spaces.configs import BasePolicyConfig, BaseRobotConfig
from molmo_spaces.configs.task_configs import PickTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config.pc_shelf_blind_configs import _blind_robot
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
from molmo_spaces.tasks.pc_lab_blind import (
    PC_LAB_BLIND_SCENES,
    PCLabBlindExpertConfig,
    PCLabBlindSampler,
    PCLabBlindTask,
)


def _lab_sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    cfg = _sampler_config(PCLabBlindSampler, house_inds, samples_per_house)
    n = max(house_inds) + 1
    cfg.scene_xml_paths = [PC_LAB_BLIND_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return cfg


@register_config("FrankaSkinPCLabBlindConfig")
class FrankaSkinPCLabBlindConfig(_PCShelfBaseConfig):
    """pc_lab_blind DEFINITION / EVAL world (eval_act_place.py --env pc_lab_blind): the 8 lab
    cells x 1 episode, action noise OFF, no contact gate, failures saved too."""

    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCLabBlindExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCLabBlindTask)
    task_sampler_config: PickTaskSamplerConfig = _lab_sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_lab_blind_def"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_lab_blind_v1"


@register_config("FrankaSkinPCLabBlindCheckConfig")
class FrankaSkinPCLabBlindCheckConfig(FrankaSkinPCLabBlindConfig):
    """Preflight of the collection recipe: 8 cells x 1 episode, noise ON, contact gate ON,
    everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCLabBlindExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_lab_blind_check"
    # Skin recorded ONCE per 66 ms policy step (the end-of-step snapshot = what the headline eval's
    # --skin_substeps snapshot feeds the policy), as pc_shelf_push_blind; the older pc suites record
    # 4 sub-steps that the converter min-pools. One render per step is ~2.5x faster on the shared
    # GPU. The definition / eval config is unchanged.
    proximity_sensor_period_ms: float = 66.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_lab_blind_check"


@register_config("FrankaSkinPCLabBlindCheck16Config")
class FrankaSkinPCLabBlindCheck16Config(FrankaSkinPCLabBlindCheckConfig):
    """Larger preflight of the collection recipe (R7): houses 200..215 (16 draws, 2 per cell)."""

    task_sampler_config: PickTaskSamplerConfig = _lab_sampler_config(list(range(200, 216)), 1)


@register_config("FrankaSkinPCLabBlindCollectConfig")
class FrankaSkinPCLabBlindCollectConfig(FrankaSkinPCLabBlindCheckConfig):
    """Collection Part1: houses 0..24 (house k = lab cell k % 8) x 2 saved episodes, noise ON,
    saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 1 worker."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _lab_sampler_config(list(range(0, 25)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_lab_blind"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_lab_blind_v1"


@register_config("FrankaSkinPCLabBlindCollectPart2Config")
class FrankaSkinPCLabBlindCollectPart2Config(FrankaSkinPCLabBlindCollectConfig):
    """Collection: DISJOINT houses 25..49, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _lab_sampler_config(list(range(25, 50)), 2)


@register_config("FrankaSkinPCLabBlindCollectPart3Config")
class FrankaSkinPCLabBlindCollectPart3Config(FrankaSkinPCLabBlindCollectConfig):
    """Collection: DISJOINT houses 50..74, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _lab_sampler_config(list(range(50, 75)), 2)


@register_config("FrankaSkinPCLabBlindCollectPart4Config")
class FrankaSkinPCLabBlindCollectPart4Config(FrankaSkinPCLabBlindCollectConfig):
    """Collection: DISJOINT houses 75..99, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _lab_sampler_config(list(range(75, 100)), 2)


@register_config("FrankaSkinPCLabBlindCollectPart5Config")
class FrankaSkinPCLabBlindCollectPart5Config(FrankaSkinPCLabBlindCollectConfig):
    """Collection: DISJOINT houses 100..124, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _lab_sampler_config(list(range(100, 125)), 2)


@register_config("FrankaSkinPCLabBlindCollectPart6Config")
class FrankaSkinPCLabBlindCollectPart6Config(FrankaSkinPCLabBlindCollectConfig):
    """Reserve: DISJOINT houses 125..137 (13 houses), launched only if the main parts fall short."""

    task_sampler_config: PickTaskSamplerConfig = _lab_sampler_config(list(range(125, 138)), 2)


__all__ = [
    "FrankaSkinPCLabBlindConfig",
    "FrankaSkinPCLabBlindCheckConfig",
    "FrankaSkinPCLabBlindCheck16Config",
    "FrankaSkinPCLabBlindCollectConfig",
    "FrankaSkinPCLabBlindCollectPart2Config",
    "FrankaSkinPCLabBlindCollectPart3Config",
    "FrankaSkinPCLabBlindCollectPart4Config",
    "FrankaSkinPCLabBlindCollectPart5Config",
    "FrankaSkinPCLabBlindCollectPart6Config",
]
