"""``pc_occ_pantry`` (2026-10-15, occluded natural items): pick a jar / tin from the back of a pantry
bay and set it on a mat at the front; two china espresso cups hanging on cup hooks under the shelf
above (left or right of the lane, hidden from exo_camera_1 by the shelf board and from the
downward-looking wrist camera because they hang above the hand) threaten the skinned wrist /
forearm. Design notes: molmo_spaces/tasks/pc_occ_pantry.py.

* ``FrankaSkinPCOccPantryConfig``          definition / eval world (8 cells x 1, noise off,
  no contact gate). Env version pc_occ_pantry_v1.
* ``FrankaSkinPCOccPantryCheckConfig``     collection recipe preflight (noise on, gate on, skin
  recorded once per policy step); ``Check16`` = houses 200..215.
* ``FrankaSkinPCOccPantryCollect*Config``  collection, DISJOINT house ranges, 2 saved episodes per
  house, 1 worker per process: Collect, Part2..Part10 = 13 houses each (houses 0..129), Part11 =
  reserve (houses 130..142).
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
from molmo_spaces.tasks.pc_occ_pantry import (
    PC_OCC_PANTRY_SCENES,
    PCOccPantrySampler,
    PCOccPantryExpertConfig,
    PCOccPantryTask,
)


def _occ_pantry_sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    cfg = _sampler_config(PCOccPantrySampler, house_inds, samples_per_house)
    n = max(house_inds) + 1
    cfg.scene_xml_paths = [PC_OCC_PANTRY_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return cfg


@register_config("FrankaSkinPCOccPantryConfig")
class FrankaSkinPCOccPantryConfig(_PCShelfBaseConfig):
    """pc_occ_pantry DEFINITION / EVAL world (eval_act_place.py --env pc_occ_pantry): the 8
    cells x 1 episode, action noise OFF, no contact gate, failures saved too."""

    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCOccPantryExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCOccPantryTask)
    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_pantry_def"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_pantry_v1"


@register_config("FrankaSkinPCOccPantryCheckConfig")
class FrankaSkinPCOccPantryCheckConfig(FrankaSkinPCOccPantryConfig):
    """Preflight of the collection recipe: 8 cells x 1 episode, noise ON, contact gate ON,
    everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCOccPantryExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_pantry_check"
    # Skin recorded ONCE per 66 ms policy step (the end-of-step snapshot = what the headline eval's
    # --skin_substeps snapshot feeds the policy) instead of pc_shelf_blind's 4 min-pooled
    # sub-steps: on the shared GPU the 160 skin renders per step ran ~4x slower than in the night's
    # collections (pc_shelf_push_blind, 2026-10-14 08:20). The definition / eval config is unchanged.
    proximity_sensor_period_ms: float = 66.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_pantry_check"


@register_config("FrankaSkinPCOccPantryCheck16Config")
class FrankaSkinPCOccPantryCheck16Config(FrankaSkinPCOccPantryCheckConfig):
    """Larger preflight of the collection recipe (R7): houses 200..215 (16 draws, 2 per cell)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(200, 216)), 1)


@register_config("FrankaSkinPCOccPantryCollectConfig")
class FrankaSkinPCOccPantryCollectConfig(FrankaSkinPCOccPantryCheckConfig):
    """Collection Part1: houses 0..12 (house k = cell k % 8) x 2 saved episodes, noise ON,
    saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 1 worker. 13-house
    parts (pc_shelf_blind: 25): the datagen worker's RSS grows ~0.1-0.2 GB per episode / house and
    the machine is shared by two more collections; small parts keep each process near 7.5 GB."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(0, 13)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_occ_pantry"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_occ_pantry_v1"


@register_config("FrankaSkinPCOccPantryCollectPart2Config")
class FrankaSkinPCOccPantryCollectPart2Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 13..25, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(13, 26)), 2)


@register_config("FrankaSkinPCOccPantryCollectPart3Config")
class FrankaSkinPCOccPantryCollectPart3Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 26..38, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(26, 39)), 2)


@register_config("FrankaSkinPCOccPantryCollectPart4Config")
class FrankaSkinPCOccPantryCollectPart4Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 39..51, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(39, 52)), 2)


@register_config("FrankaSkinPCOccPantryCollectPart5Config")
class FrankaSkinPCOccPantryCollectPart5Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 52..64, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(52, 65)), 2)


@register_config("FrankaSkinPCOccPantryCollectPart6Config")
class FrankaSkinPCOccPantryCollectPart6Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 65..77, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(65, 78)), 2)


@register_config("FrankaSkinPCOccPantryCollectPart7Config")
class FrankaSkinPCOccPantryCollectPart7Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 78..90, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(78, 91)), 2)


@register_config("FrankaSkinPCOccPantryCollectPart8Config")
class FrankaSkinPCOccPantryCollectPart8Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 91..103, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(91, 104)), 2)


@register_config("FrankaSkinPCOccPantryCollectPart9Config")
class FrankaSkinPCOccPantryCollectPart9Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 104..116, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(104, 117)), 2)


@register_config("FrankaSkinPCOccPantryCollectPart10Config")
class FrankaSkinPCOccPantryCollectPart10Config(FrankaSkinPCOccPantryCollectConfig):
    """Collection: DISJOINT houses 117..129, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(117, 130)), 2)



@register_config("FrankaSkinPCOccPantryCollectPart11Config")
class FrankaSkinPCOccPantryCollectPart11Config(FrankaSkinPCOccPantryCollectConfig):
    """Reserve: DISJOINT houses 130..142, same recipe (launched only if the target is not reached)."""

    task_sampler_config: PickTaskSamplerConfig = _occ_pantry_sampler_config(list(range(130, 143)), 2)


__all__ = [
    "FrankaSkinPCOccPantryConfig",
    "FrankaSkinPCOccPantryCheckConfig",
    "FrankaSkinPCOccPantryCheck16Config",
    "FrankaSkinPCOccPantryCollectConfig",
    "FrankaSkinPCOccPantryCollectPart2Config",
    "FrankaSkinPCOccPantryCollectPart3Config",
    "FrankaSkinPCOccPantryCollectPart4Config",
    "FrankaSkinPCOccPantryCollectPart5Config",
    "FrankaSkinPCOccPantryCollectPart6Config",
    "FrankaSkinPCOccPantryCollectPart7Config",
    "FrankaSkinPCOccPantryCollectPart8Config",
    "FrankaSkinPCOccPantryCollectPart9Config",
    "FrankaSkinPCOccPantryCollectPart10Config",
    "FrankaSkinPCOccPantryCollectPart11Config",
]
