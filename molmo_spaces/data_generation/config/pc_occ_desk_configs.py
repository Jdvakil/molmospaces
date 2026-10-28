"""``pc_occ_desk`` (2026-10-15, ICLR rebuttal, occluded natural items): a home-office desk; the
robot stands at the desk's left end, reaches behind the computer monitor for a cup and sets it on a
coaster nearer the desk end, while an adjustable desk lamp standing behind the monitor (left or
right of the arm's lane, invisible to exo_camera_1 and wrist_camera) threatens the skinned wrist
knuckle / forearm. Design notes: molmo_spaces/tasks/pc_occ_desk.py.

* ``FrankaSkinPCOccDeskConfig``          definition / eval world (8 cells x 1, noise off,
  no contact gate). Env version pc_occ_desk_v1. Robot base link0 at 0.36 m.
* ``FrankaSkinPCOccDeskCheckConfig``     collection recipe preflight (noise on, gate on, skin
  recorded once per policy step).
* ``FrankaSkinPCOccDeskCollect*Config``  collection, DISJOINT house ranges, 2 saved episodes per
  house, 1 worker per process.
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
from molmo_spaces.tasks.pc_occ_desk import (
    H_BASE,
    N_CELLS,
    PC_OCC_DESK_READY_QPOS,
    PC_OCC_DESK_SCENES,
    PCOccDeskExpertConfig,
    PCOccDeskSampler,
    PCOccDeskTask,
)


def _desk_sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    cfg = _sampler_config(PCOccDeskSampler, house_inds, samples_per_house)
    n = max(house_inds) + 1
    cfg.scene_xml_paths = [PC_OCC_DESK_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return cfg


def _desk_robot() -> FrankaSkinHybridRobotConfig:
    return FrankaSkinHybridRobotConfig(
        base_size=[0.4, 0.4, H_BASE],
        init_qpos={"arm": list(PC_OCC_DESK_READY_QPOS), "gripper": [0.00296, 0.00296]},
        init_qpos_noise_range={"arm": [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.06]},
    )


@register_config("FrankaSkinPCOccDeskConfig")
class FrankaSkinPCOccDeskConfig(_PCShelfBaseConfig):
    """pc_occ_desk DEFINITION / EVAL world (eval_act_place.py --env pc_occ_desk): the 8 desk
    cells x 1 episode, action noise OFF, no contact gate, failures saved too."""

    robot_config: BaseRobotConfig = _desk_robot()
    policy_config: BasePolicyConfig = PCOccDeskExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCOccDeskTask)
    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_desk_def"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_desk_v1"


@register_config("FrankaSkinPCOccDeskCheckConfig")
class FrankaSkinPCOccDeskCheckConfig(FrankaSkinPCOccDeskConfig):
    """Preflight of the collection recipe: 8 cells x 1 episode, noise ON, contact gate ON,
    everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _desk_robot()
    policy_config: BasePolicyConfig = PCOccDeskExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_desk_check"
    # Skin recorded ONCE per 66 ms policy step (the end-of-step snapshot = what the headline eval's
    # --skin_substeps snapshot feeds the policy), as pc_shelf_push_blind; the older pc suites
    # record 4 sub-steps that the converter min-pools. ~2.5x faster on the shared GPU.
    proximity_sensor_period_ms: float = 66.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_desk_check"


@register_config("FrankaSkinPCOccDeskCheck16Config")
class FrankaSkinPCOccDeskCheck16Config(FrankaSkinPCOccDeskCheckConfig):
    """Larger preflight of the collection recipe (R7): houses 200..215 (16 draws, 2 per cell)."""

    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(200, 216)), 1)


@register_config("FrankaSkinPCOccDeskCollectConfig")
class FrankaSkinPCOccDeskCollectConfig(FrankaSkinPCOccDeskCheckConfig):
    """Collection Part1: houses 0..24 (house k = desk cell k % 8) x 2 saved episodes, noise ON,
    saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 1 worker."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(0, 25)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_desk"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_desk_v1"


@register_config("FrankaSkinPCOccDeskCollectPart2Config")
class FrankaSkinPCOccDeskCollectPart2Config(FrankaSkinPCOccDeskCollectConfig):
    """Collection: DISJOINT houses 25..49, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(25, 50)), 2)


@register_config("FrankaSkinPCOccDeskCollectPart3Config")
class FrankaSkinPCOccDeskCollectPart3Config(FrankaSkinPCOccDeskCollectConfig):
    """Collection: DISJOINT houses 50..74, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(50, 75)), 2)


@register_config("FrankaSkinPCOccDeskCollectPart4Config")
class FrankaSkinPCOccDeskCollectPart4Config(FrankaSkinPCOccDeskCollectConfig):
    """Collection: DISJOINT houses 75..91 (17 houses: the second wave runs three smaller parts so
    the three process slots stay busy), same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(75, 92)), 2)


@register_config("FrankaSkinPCOccDeskCollectPart5Config")
class FrankaSkinPCOccDeskCollectPart5Config(FrankaSkinPCOccDeskCollectConfig):
    """Collection: DISJOINT houses 92..108 (17 houses), same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(92, 109)), 2)


@register_config("FrankaSkinPCOccDeskCollectPart6Config")
class FrankaSkinPCOccDeskCollectPart6Config(FrankaSkinPCOccDeskCollectConfig):
    """Collection: DISJOINT houses 109..125 (17 houses), same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(109, 126)), 2)


@register_config("FrankaSkinPCOccDeskCollectPart7Config")
class FrankaSkinPCOccDeskCollectPart7Config(FrankaSkinPCOccDeskCollectConfig):
    """Reserve: DISJOINT houses 126..138 (13 houses, launched only if the target is not reached)."""

    task_sampler_config: PickTaskSamplerConfig = _desk_sampler_config(list(range(126, 139)), 2)


__all__ = [
    "FrankaSkinPCOccDeskConfig",
    "FrankaSkinPCOccDeskCheckConfig",
    "FrankaSkinPCOccDeskCheck16Config",
    "FrankaSkinPCOccDeskCollectConfig",
    "FrankaSkinPCOccDeskCollectPart2Config",
    "FrankaSkinPCOccDeskCollectPart3Config",
    "FrankaSkinPCOccDeskCollectPart4Config",
    "FrankaSkinPCOccDeskCollectPart5Config",
    "FrankaSkinPCOccDeskCollectPart6Config",
    "FrankaSkinPCOccDeskCollectPart7Config",
]
