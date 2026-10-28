"""``pc_shelf_push_blind`` (2026-10-14, ICLR rebuttal): NON-PREHENSILE version of pc_shelf_blind.
With the fingers closed, push a box from the front of a pantry bay to the label by its back panel;
the hidden under-shelf basket of pc_shelf_blind (left or right of the push lane, invisible to
exo_camera_1 and wrist_camera) threatens the skinned wrist knuckle / forearm. Design notes:
molmo_spaces/tasks/pc_shelf_push_blind.py.

* ``FrankaSkinPCShelfPushBlindConfig``          definition / eval world (8 cells x 1, noise off,
  no contact gate). Env version pc_shelf_push_blind_v1.
* ``FrankaSkinPCShelfPushBlindCheckConfig``     collection recipe preflight (noise on, gate on,
  skin recorded once per policy step = the eval's snapshot; other pc suites: 4 min-pooled substeps).
* ``FrankaSkinPCShelfPushBlindCollect*Config``  collection, DISJOINT house ranges, 2 saved
  episodes per house, 1 worker per process (Collect..Part4: 32 houses = 64 demos each, one wave of
  four processes; Part5/Part6: 13-house reserves). 32 houses (not 25) per process: the skin
  renders ran ~4x slower than in the night's collections on the shared GPU (2026-10-14 07:50).
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
from molmo_spaces.tasks.pc_shelf_push_blind import (
    PC_SHELF_PUSH_BLIND_SCENES,
    PCShelfPushBlindExpertConfig,
    PCShelfPushBlindSampler,
    PCShelfPushBlindTask,
)


def _push_blind_sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    cfg = _sampler_config(PCShelfPushBlindSampler, house_inds, samples_per_house)
    n = max(house_inds) + 1
    cfg.scene_xml_paths = [PC_SHELF_PUSH_BLIND_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))]
    return cfg


@register_config("FrankaSkinPCShelfPushBlindConfig")
class FrankaSkinPCShelfPushBlindConfig(_PCShelfBaseConfig):
    """pc_shelf_push_blind DEFINITION / EVAL world (eval_act_place.py --env pc_shelf_push_blind):
    the 8 push-blind cells x 1 episode, action noise OFF, no contact gate, failures saved too."""

    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCShelfPushBlindExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCShelfPushBlindTask)
    task_sampler_config: PickTaskSamplerConfig = _push_blind_sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_push_blind_def"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_push_blind_v1"


@register_config("FrankaSkinPCShelfPushBlindCheckConfig")
class FrankaSkinPCShelfPushBlindCheckConfig(FrankaSkinPCShelfPushBlindConfig):
    """Preflight of the collection recipe: 8 cells x 1 episode, noise ON, contact gate ON,
    everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _blind_robot()
    policy_config: BasePolicyConfig = PCShelfPushBlindExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_push_blind_check"
    # Skin recorded ONCE per 66 ms policy step (the end-of-step snapshot = what the headline eval's
    # --skin_substeps snapshot feeds the policy) instead of 4 sub-steps min-pooled by the converter
    # (every other pc suite). 2026-10-14 08:20: on the shared GPU the 160 skin renders per step ran
    # ~4x slower than in the night's collections (~180 s per episode), too slow for 250 demos by
    # the deadline; one render per step is ~2.5x faster. The definition / eval config is unchanged.
    proximity_sensor_period_ms: float = 66.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_push_blind_check"


@register_config("FrankaSkinPCShelfPushBlindCheck16Config")
class FrankaSkinPCShelfPushBlindCheck16Config(FrankaSkinPCShelfPushBlindCheckConfig):
    """Larger preflight of the collection recipe (R7): houses 200..215 (16 draws, 2 per cell)."""

    task_sampler_config: PickTaskSamplerConfig = _push_blind_sampler_config(list(range(200, 216)), 1)
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_push_blind_check"


@register_config("FrankaSkinPCShelfPushBlindCollectConfig")
class FrankaSkinPCShelfPushBlindCollectConfig(FrankaSkinPCShelfPushBlindCheckConfig):
    """Collection Part1: houses 0..31 (house k = push-blind cell k % 8) x 2 saved episodes, noise
    ON, saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 1 worker."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _push_blind_sampler_config(list(range(0, 32)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_shelf_push_blind"

    @property
    def tag(self) -> str:
        return "franka_skin_pc_shelf_push_blind_v1"


@register_config("FrankaSkinPCShelfPushBlindCollectPart2Config")
class FrankaSkinPCShelfPushBlindCollectPart2Config(FrankaSkinPCShelfPushBlindCollectConfig):
    """Collection: DISJOINT houses 32..63, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _push_blind_sampler_config(list(range(32, 64)), 2)


@register_config("FrankaSkinPCShelfPushBlindCollectPart3Config")
class FrankaSkinPCShelfPushBlindCollectPart3Config(FrankaSkinPCShelfPushBlindCollectConfig):
    """Collection: DISJOINT houses 64..95, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _push_blind_sampler_config(list(range(64, 96)), 2)


@register_config("FrankaSkinPCShelfPushBlindCollectPart4Config")
class FrankaSkinPCShelfPushBlindCollectPart4Config(FrankaSkinPCShelfPushBlindCollectConfig):
    """Collection: DISJOINT houses 96..127, same recipe (merge with Part1)."""

    task_sampler_config: PickTaskSamplerConfig = _push_blind_sampler_config(list(range(96, 128)), 2)


@register_config("FrankaSkinPCShelfPushBlindCollectPart5Config")
class FrankaSkinPCShelfPushBlindCollectPart5Config(FrankaSkinPCShelfPushBlindCollectConfig):
    """Reserve: DISJOINT houses 128..140 (13 houses, launched only if a main part fails)."""

    task_sampler_config: PickTaskSamplerConfig = _push_blind_sampler_config(list(range(128, 141)), 2)


@register_config("FrankaSkinPCShelfPushBlindCollectPart6Config")
class FrankaSkinPCShelfPushBlindCollectPart6Config(FrankaSkinPCShelfPushBlindCollectConfig):
    """Reserve: DISJOINT houses 141..153 (13 houses, launched only if a main part fails)."""

    task_sampler_config: PickTaskSamplerConfig = _push_blind_sampler_config(list(range(141, 154)), 2)


__all__ = [
    "FrankaSkinPCShelfPushBlindConfig",
    "FrankaSkinPCShelfPushBlindCheckConfig",
    "FrankaSkinPCShelfPushBlindCheck16Config",
    "FrankaSkinPCShelfPushBlindCollectConfig",
    "FrankaSkinPCShelfPushBlindCollectPart2Config",
    "FrankaSkinPCShelfPushBlindCollectPart3Config",
    "FrankaSkinPCShelfPushBlindCollectPart4Config",
    "FrankaSkinPCShelfPushBlindCollectPart5Config",
    "FrankaSkinPCShelfPushBlindCollectPart6Config",
]
