"""``pc_shelf2_blind`` (2026-10-14, ICLR rebuttal): ``pc_shelf_blind`` with the fixed-tilt hole
closed (a fixed -5 / -7.5 deg hand tilt cleared both of pc_shelf_blind's basket positions). Same
room, cells, can, mat, expert (0.6 s pause), success rule and collection recipe as pc_shelf_blind;
only the hidden basket's two positions moved closer together. Design notes:
molmo_spaces/tasks/pc_shelf2_blind.py.

* ``FrankaSkinPCShelf2BlindConfig``          definition / eval world (8 cells x 1, noise off,
  no contact gate). Env version pc_shelf2_blind_v1.
* ``FrankaSkinPCShelf2BlindCheckConfig``     collection recipe preflight (noise on, gate on, skin
  recorded once per policy step).
* ``FrankaSkinPCShelf2BlindCollect*Config``  collection, DISJOINT house ranges, 2 saved episodes per
  house, 1 worker per process: Collect, Part2..Part10 = 13 houses each (houses 0..129).
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
from molmo_spaces.tasks.pc_shelf2_blind import (
    PC_SHELF2_BLIND_SCENES,
    PCShelf2BlindSampler,
    PCShelf2BlindTask,
)
from molmo_spaces.tasks.pc_shelf_blind import PCShelfBlindExpertConfig


def _shelf2_sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    cfg = _sampler_config(PCShelf2BlindSampler, house_inds, samples_per_house)
    n = max(house_inds) + 1
    cfg.scene_xml_paths = [PC_SHELF2_BLIND_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return cfg


@register_config("FrankaSkinPCShelf2BlindConfig")
class FrankaSkinPCShelf2BlindConfig(_PCShelfBaseConfig):
    """pc_shelf2_blind DEFINITION / EVAL world (eval_act_place.py --env pc_shelf2_blind): the 8
    cells x 1 episode, action noise OFF, no contact gate, failures saved too."""

    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCShelfBlindExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCShelf2BlindTask)
    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf2_blind_def"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf2_blind_v1"


@register_config("FrankaSkinPCShelf2BlindCheckConfig")
class FrankaSkinPCShelf2BlindCheckConfig(FrankaSkinPCShelf2BlindConfig):
    """Preflight of the collection recipe: 8 cells x 1 episode, noise ON, contact gate ON,
    everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCShelfBlindExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf2_blind_check"
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
        return "franka_skin_pc_shelf2_blind_check"


@register_config("FrankaSkinPCShelf2BlindCheckBConfig")
class FrankaSkinPCShelf2BlindCheckBConfig(FrankaSkinPCShelf2BlindCheckConfig):
    """Second preflight batch of the collection recipe (R7): houses 200..207 (8 more draws)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(200, 208)), 1)


@register_config("FrankaSkinPCShelf2BlindCheckCConfig")
class FrankaSkinPCShelf2BlindCheckCConfig(FrankaSkinPCShelf2BlindCheckConfig):
    """Third preflight batch of the collection recipe (R7): houses 208..215 (8 more draws)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(208, 216)), 1)


@register_config("FrankaSkinPCShelf2BlindCollectConfig")
class FrankaSkinPCShelf2BlindCollectConfig(FrankaSkinPCShelf2BlindCheckConfig):
    """Collection Part1: houses 0..12 (house k = cell k % 8) x 2 saved episodes, noise ON,
    saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 1 worker. 13-house
    parts (pc_shelf_blind: 25): the datagen worker's RSS grows ~0.1-0.2 GB per episode / house and
    the machine is shared by two more collections; small parts keep each process near 7.5 GB."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(0, 13)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf2_blind"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf2_blind_v1"


@register_config("FrankaSkinPCShelf2BlindCollectPart2Config")
class FrankaSkinPCShelf2BlindCollectPart2Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 13..25, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(13, 26)), 2)


@register_config("FrankaSkinPCShelf2BlindCollectPart3Config")
class FrankaSkinPCShelf2BlindCollectPart3Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 26..38, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(26, 39)), 2)


@register_config("FrankaSkinPCShelf2BlindCollectPart4Config")
class FrankaSkinPCShelf2BlindCollectPart4Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 39..51, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(39, 52)), 2)


@register_config("FrankaSkinPCShelf2BlindCollectPart5Config")
class FrankaSkinPCShelf2BlindCollectPart5Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 52..64, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(52, 65)), 2)


@register_config("FrankaSkinPCShelf2BlindCollectPart6Config")
class FrankaSkinPCShelf2BlindCollectPart6Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 65..77, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(65, 78)), 2)


@register_config("FrankaSkinPCShelf2BlindCollectPart7Config")
class FrankaSkinPCShelf2BlindCollectPart7Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 78..90, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(78, 91)), 2)


@register_config("FrankaSkinPCShelf2BlindCollectPart8Config")
class FrankaSkinPCShelf2BlindCollectPart8Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 91..103, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(91, 104)), 2)


@register_config("FrankaSkinPCShelf2BlindCollectPart9Config")
class FrankaSkinPCShelf2BlindCollectPart9Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 104..116, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(104, 117)), 2)


@register_config("FrankaSkinPCShelf2BlindCollectPart10Config")
class FrankaSkinPCShelf2BlindCollectPart10Config(FrankaSkinPCShelf2BlindCollectConfig):
    """Collection: DISJOINT houses 117..129, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _shelf2_sampler_config(list(range(117, 130)), 2)


__all__ = [
    "FrankaSkinPCShelf2BlindConfig",
    "FrankaSkinPCShelf2BlindCheckConfig",
    "FrankaSkinPCShelf2BlindCheckBConfig",
    "FrankaSkinPCShelf2BlindCheckCConfig",
    "FrankaSkinPCShelf2BlindCollectConfig",
    "FrankaSkinPCShelf2BlindCollectPart2Config",
    "FrankaSkinPCShelf2BlindCollectPart3Config",
    "FrankaSkinPCShelf2BlindCollectPart4Config",
    "FrankaSkinPCShelf2BlindCollectPart5Config",
    "FrankaSkinPCShelf2BlindCollectPart6Config",
    "FrankaSkinPCShelf2BlindCollectPart7Config",
    "FrankaSkinPCShelf2BlindCollectPart8Config",
    "FrankaSkinPCShelf2BlindCollectPart9Config",
    "FrankaSkinPCShelf2BlindCollectPart10Config",
]
