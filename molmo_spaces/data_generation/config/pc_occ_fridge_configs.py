"""``pc_occ_fridge`` (2026-10-15, ICLR rebuttal, occluded natural items): reach into an open
refrigerator, pick a jar from the back of the lowest shelf and set it on a tray at the front; two
wine bottles lying in a wine rack clipped under the shelf above (left or right of the reach lane,
hidden from exo_camera_1 by the opaque shelf and its trim, above the wrist camera) threaten the
skinned wrist knuckle. Design notes: molmo_spaces/tasks/pc_occ_fridge.py.

* ``FrankaSkinPCOccFridgeConfig``          definition / eval world (8 cells x 1, noise off,
  no contact gate). Env version pc_occ_fridge_v1. Robot on the standard 0.35 m base.
* ``FrankaSkinPCOccFridgeCheckConfig``     collection recipe preflight (noise on, gate on, skin
  recorded once per policy step = the eval's snapshot).
* ``FrankaSkinPCOccFridgeCheck16Config``   16-draw preflight (houses 200..215).
* ``FrankaSkinPCOccFridgeCollect*Config``  collection, DISJOINT house ranges (Part1-3: 25 houses,
  Part4-6: 17 houses, Part7: 13-house reserve), 2 saved episodes per house, 1 worker per process.
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
from molmo_spaces.tasks.pc_occ_fridge import (
    H_BASE,
    N_CELLS,
    PC_OCC_FRIDGE_READY_QPOS,
    PC_OCC_FRIDGE_SCENES,
    PCOccFridgeExpertConfig,
    PCOccFridgeSampler,
    PCOccFridgeTask,
)


def _fridge_sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    cfg = _sampler_config(PCOccFridgeSampler, house_inds, samples_per_house)
    n = max(house_inds) + 1
    cfg.scene_xml_paths = [PC_OCC_FRIDGE_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return cfg


def _fridge_robot() -> FrankaSkinHybridRobotConfig:
    return FrankaSkinHybridRobotConfig(
        base_size=[0.4, 0.4, H_BASE],
        init_qpos={"arm": list(PC_OCC_FRIDGE_READY_QPOS), "gripper": [0.00296, 0.00296]},
        init_qpos_noise_range={"arm": [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.06]},
    )


@register_config("FrankaSkinPCOccFridgeConfig")
class FrankaSkinPCOccFridgeConfig(_PCShelfBaseConfig):
    """pc_occ_fridge DEFINITION / EVAL world (eval_act_place.py --env pc_occ_fridge): the 8 fridge
    cells x 1 episode, action noise OFF, no contact gate, failures saved too."""

    robot_config: BaseRobotConfig = _fridge_robot()
    policy_config: BasePolicyConfig = PCOccFridgeExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCOccFridgeTask)
    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_fridge_def"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_fridge_v1"


@register_config("FrankaSkinPCOccFridgeCheckConfig")
class FrankaSkinPCOccFridgeCheckConfig(FrankaSkinPCOccFridgeConfig):
    """Preflight of the collection recipe: 8 cells x 1 episode, noise ON, contact gate ON,
    everything saved for inspection. Skin recorded ONCE per 66 ms policy step (the eval's
    --skin_substeps snapshot), as pc_sink_blind."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _fridge_robot()
    policy_config: BasePolicyConfig = PCOccFridgeExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_fridge_check"
    proximity_sensor_period_ms: float = 66.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_fridge_check"


@register_config("FrankaSkinPCOccFridgeCheck16Config")
class FrankaSkinPCOccFridgeCheck16Config(FrankaSkinPCOccFridgeCheckConfig):
    """Larger preflight of the collection recipe: houses 200..215 (16 draws, 2 per cell)."""

    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(200, 216)), 1)


@register_config("FrankaSkinPCOccFridgeCollectConfig")
class FrankaSkinPCOccFridgeCollectConfig(FrankaSkinPCOccFridgeCheckConfig):
    """Collection Part1: houses 0..24 (house k = fridge cell k % 8) x 2 saved episodes, noise ON,
    saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 1 worker."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(0, 25)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_fridge"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_fridge_v1"


@register_config("FrankaSkinPCOccFridgeCollectPart2Config")
class FrankaSkinPCOccFridgeCollectPart2Config(FrankaSkinPCOccFridgeCollectConfig):
    """Collection: DISJOINT houses 25..49, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(25, 50)), 2)


@register_config("FrankaSkinPCOccFridgeCollectPart3Config")
class FrankaSkinPCOccFridgeCollectPart3Config(FrankaSkinPCOccFridgeCollectConfig):
    """Collection: DISJOINT houses 50..74, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(50, 75)), 2)


@register_config("FrankaSkinPCOccFridgeCollectPart4Config")
class FrankaSkinPCOccFridgeCollectPart4Config(FrankaSkinPCOccFridgeCollectConfig):
    """Collection: DISJOINT houses 75..91 (17 houses), same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(75, 92)), 2)


@register_config("FrankaSkinPCOccFridgeCollectPart5Config")
class FrankaSkinPCOccFridgeCollectPart5Config(FrankaSkinPCOccFridgeCollectConfig):
    """Collection: DISJOINT houses 92..108 (17 houses), same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(92, 109)), 2)


@register_config("FrankaSkinPCOccFridgeCollectPart6Config")
class FrankaSkinPCOccFridgeCollectPart6Config(FrankaSkinPCOccFridgeCollectConfig):
    """Collection: DISJOINT houses 109..125 (17 houses), same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(109, 126)), 2)


@register_config("FrankaSkinPCOccFridgeCollectPart7Config")
class FrankaSkinPCOccFridgeCollectPart7Config(FrankaSkinPCOccFridgeCollectConfig):
    """Reserve: DISJOINT houses 126..138 (13 houses, launched only if the target is not reached)."""

    task_sampler_config: PickTaskSamplerConfig = _fridge_sampler_config(list(range(126, 139)), 2)


__all__ = [
    "FrankaSkinPCOccFridgeConfig",
    "FrankaSkinPCOccFridgeCheckConfig",
    "FrankaSkinPCOccFridgeCheck16Config",
    "FrankaSkinPCOccFridgeCollectConfig",
    "FrankaSkinPCOccFridgeCollectPart2Config",
    "FrankaSkinPCOccFridgeCollectPart3Config",
    "FrankaSkinPCOccFridgeCollectPart4Config",
    "FrankaSkinPCOccFridgeCollectPart5Config",
    "FrankaSkinPCOccFridgeCollectPart6Config",
    "FrankaSkinPCOccFridgeCollectPart7Config",
]
