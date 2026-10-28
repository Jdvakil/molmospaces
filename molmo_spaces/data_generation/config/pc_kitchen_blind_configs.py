"""``pc_kitchen_blind`` (2026-10-14, ICLR rebuttal): pc_kitchen_reach plus a HIDDEN under-cabinet
radio that only the arm skin can see (molmo_spaces/tasks/pc_kitchen_blind.py). Env version
pc_kitchen_blind_v1.

* ``FrankaSkinPCKitchenBlindConfig``        definition / EVAL world (eval_act_place.py --env
  pc_kitchen_blind): 8 cells x 1 episode, action noise OFF, no contact gate, failures saved.
* ``FrankaSkinPCKitchenBlindCheckConfig``   preflight of the collection recipe (noise + gate).
* ``FrankaSkinPCKitchenBlindCleanCheckConfig`` expert preflight without noise / gate.
* ``FrankaSkinPCKitchenBlindCollectConfig`` / ``...Part<N>Config``: disjoint 24-house ranges
  (house k = cell k % 8), 2 saved episodes per house, 3 workers (siblings share the machine:
  at most 3 datagen workers at a time), DART-style action noise, saved = success AND hazard
  contact <= HARD_CONTACT_GATE_FRAMES x 2 ms.

Run from submodules/molmospaces:
    python -m molmo_spaces.data_generation.main FrankaSkinPCKitchenBlindCheckConfig
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
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pc_kitchen_blind import (
    N_CELLS,
    PC_KITCHEN_BLIND_SCENES,
    PCKitchenBlindExpertConfig,
    PCKitchenBlindSampler,
    PCKitchenBlindTask,
)
from molmo_spaces.tasks.pc_kitchen_reach import PC_KITCHEN_REACH_READY_QPOS


def _sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    n = max(house_inds) + 1
    return PickTaskSamplerConfig(
        task_sampler_class=PCKitchenBlindSampler,
        # one XML per house index (index k -> cell k % 8); always >= 8 paths so an evaluator
        # can address the cells 0..7 directly
        scene_xml_paths=[PC_KITCHEN_BLIND_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))],
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
        init_qpos={"arm": list(PC_KITCHEN_REACH_READY_QPOS), "gripper": [0.00296, 0.00296]},
        # small start-pose jitter (TCP a few cm), as pc_shelf
        init_qpos_noise_range={"arm": [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.06]},
    )


@register_config("FrankaSkinPCKitchenBlindConfig")
class FrankaSkinPCKitchenBlindConfig(_PCShelfBaseConfig):
    """Definition / eval world: 8 cells x 1, noise off, no gate, failures saved, RGB-only."""

    scene_dataset: str = "user"
    robot_config: BaseRobotConfig = _robot()
    policy_config: BasePolicyConfig = PCKitchenBlindExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCKitchenBlindTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_blind"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_blind_v1"


@register_config("FrankaSkinPCKitchenBlindCheckConfig")
class FrankaSkinPCKitchenBlindCheckConfig(FrankaSkinPCKitchenBlindConfig):
    """Preflight of the collection recipe: 8 cells x 2, action noise ON, contact gate ON
    (a heavy-contact success counts as a failure), everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _robot()
    policy_config: BasePolicyConfig = PCKitchenBlindExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(list(range(8)), 2)
    num_workers: int = 2

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_blind_check"


@register_config("FrankaSkinPCKitchenBlindCleanCheckConfig")
class FrankaSkinPCKitchenBlindCleanCheckConfig(FrankaSkinPCKitchenBlindConfig):
    """Expert preflight WITHOUT noise or gate (acceptance check 1: the scripted expert alone
    must succeed in every cell); 8 cells x 1, everything saved, 1 worker."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(list(range(8)), 1)
    num_workers: int = 1

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_blind_clean_check"


@register_config("FrankaSkinPCKitchenBlindCollectConfig")
class FrankaSkinPCKitchenBlindCollectConfig(FrankaSkinPCKitchenBlindCheckConfig):
    """Collection: houses 0..23 (house k = cell k % 8) x 2 saved episodes, noise ON, saved =
    success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 3 workers."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(list(range(24)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 3

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_blind_v1"


def _part(k: int) -> PickTaskSamplerConfig:
    return _sampler_config(list(range(24 * k, 24 * (k + 1))), 2)


@register_config("FrankaSkinPCKitchenBlindCollectPart2Config")
class FrankaSkinPCKitchenBlindCollectPart2Config(FrankaSkinPCKitchenBlindCollectConfig):
    """DISJOINT houses 24..47, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(1)


@register_config("FrankaSkinPCKitchenBlindCollectPart3Config")
class FrankaSkinPCKitchenBlindCollectPart3Config(FrankaSkinPCKitchenBlindCollectConfig):
    """DISJOINT houses 48..71, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(2)


@register_config("FrankaSkinPCKitchenBlindCollectPart4Config")
class FrankaSkinPCKitchenBlindCollectPart4Config(FrankaSkinPCKitchenBlindCollectConfig):
    """DISJOINT houses 72..95, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(3)


@register_config("FrankaSkinPCKitchenBlindCollectPart5Config")
class FrankaSkinPCKitchenBlindCollectPart5Config(FrankaSkinPCKitchenBlindCollectConfig):
    """DISJOINT houses 96..119, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(4)


@register_config("FrankaSkinPCKitchenBlindCollectPart6Config")
class FrankaSkinPCKitchenBlindCollectPart6Config(FrankaSkinPCKitchenBlindCollectConfig):
    """DISJOINT houses 120..143, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(5)


@register_config("FrankaSkinPCKitchenBlindCollectPart7Config")
class FrankaSkinPCKitchenBlindCollectPart7Config(FrankaSkinPCKitchenBlindCollectConfig):
    """DISJOINT houses 144..167, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(6)


@register_config("FrankaSkinPCKitchenBlindCollectPart8Config")
class FrankaSkinPCKitchenBlindCollectPart8Config(FrankaSkinPCKitchenBlindCollectConfig):
    """DISJOINT houses 168..191, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(7)


__all__ = [
    "FrankaSkinPCKitchenBlindConfig",
    "FrankaSkinPCKitchenBlindCheckConfig",
    "FrankaSkinPCKitchenBlindCleanCheckConfig",
    "FrankaSkinPCKitchenBlindCollectConfig",
    "FrankaSkinPCKitchenBlindCollectPart2Config",
    "FrankaSkinPCKitchenBlindCollectPart3Config",
    "FrankaSkinPCKitchenBlindCollectPart4Config",
    "FrankaSkinPCKitchenBlindCollectPart5Config",
    "FrankaSkinPCKitchenBlindCollectPart6Config",
    "FrankaSkinPCKitchenBlindCollectPart7Config",
    "FrankaSkinPCKitchenBlindCollectPart8Config",
]
