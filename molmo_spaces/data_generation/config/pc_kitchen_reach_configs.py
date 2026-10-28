"""``pc_kitchen_reach`` (2026-10-13, ICLR rebuttal): a kitchen suite whose collision risk sits on the
SKIN-COVERED wrist knuckle / forearm (molmo_spaces/tasks/pc_kitchen_reach.py).

Move a lidded jar along the counter under the wall cabinets with a top-down hand: the wrist
knuckle (fr3_link6) rides ~1.5-3.5 cm under the cabinet bottom / LED strip, the forearm ~4 cm,
while the gripper and fr3_link7 keep >= 6.6 cm from everything. Env version pc_kitchen_reach_v1.

* ``FrankaSkinPCKitchenReachConfig``        definition / EVAL world (eval_act_place.py --env
  pc_kitchen_reach): 8 cells x 1 episode, action noise OFF, no contact gate, failures saved.
* ``FrankaSkinPCKitchenReachCheckConfig``   preflight of the collection recipe (noise + gate).
* ``FrankaSkinPCKitchenReachCollectConfig`` / ``...Part<N>Config``: disjoint 24-house ranges
  (house k = cell k % 8), 2 saved episodes per house, 2 workers, DART-style action noise,
  saved = success AND hazard contact <= HARD_CONTACT_GATE_FRAMES x 2 ms.

Run from submodules/molmospaces:
    python -m molmo_spaces.data_generation.main FrankaSkinPCKitchenReachCheckConfig
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
from molmo_spaces.tasks.pc_kitchen_reach import (
    N_CELLS,
    PC_KITCHEN_REACH_READY_QPOS,
    PC_KITCHEN_REACH_SCENES,
    PCKitchenReachExpertConfig,
    PCKitchenReachSampler,
    PCKitchenReachTask,
)


def _sampler_config(house_inds: list[int], samples_per_house: int) -> PickTaskSamplerConfig:
    n = max(house_inds) + 1
    return PickTaskSamplerConfig(
        task_sampler_class=PCKitchenReachSampler,
        # one XML per house index (index k -> cell k % 8); always >= 8 paths so an evaluator
        # can address the cells 0..7 directly
        scene_xml_paths=[PC_KITCHEN_REACH_SCENES[k % N_CELLS] for k in range(max(n, N_CELLS))],
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


@register_config("FrankaSkinPCKitchenReachConfig")
class FrankaSkinPCKitchenReachConfig(_PCShelfBaseConfig):
    """Definition / eval world: 8 cells x 1, noise off, no gate, failures saved, RGB-only."""

    scene_dataset: str = "user"
    robot_config: BaseRobotConfig = _robot()
    policy_config: BasePolicyConfig = PCKitchenReachExpertConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PCKitchenReachTask)
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(list(range(8)), 1)
    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "pc_kitchen_reach"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _rgb_only(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_reach_v1"


@register_config("FrankaSkinPCKitchenReachCheckConfig")
class FrankaSkinPCKitchenReachCheckConfig(FrankaSkinPCKitchenReachConfig):
    """Preflight of the collection recipe: 8 cells x 1, action noise ON, contact gate ON
    (a heavy-contact success counts as a failure), everything saved for inspection."""

    seed: int | None = None
    robot_config: BaseRobotConfig = _robot()
    policy_config: BasePolicyConfig = PCKitchenReachExpertConfig(
        contact_gate_max_frames=HARD_CONTACT_GATE_FRAMES
    )
    num_workers: int = 2

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _noise_on(self)

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_reach_check"


@register_config("FrankaSkinPCKitchenReachCleanCheckConfig")
class FrankaSkinPCKitchenReachCleanCheckConfig(FrankaSkinPCKitchenReachConfig):
    """Expert preflight WITHOUT noise or gate (acceptance check 1: the scripted expert alone
    must succeed in every cell); 8 cells x 2, everything saved, 2 workers."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(list(range(8)), 2)
    num_workers: int = 2

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_reach_clean_check"


@register_config("FrankaSkinPCKitchenReachCollectConfig")
class FrankaSkinPCKitchenReachCollectConfig(FrankaSkinPCKitchenReachCheckConfig):
    """Collection: houses 0..23 (house k = cell k % 8) x 2 saved episodes, noise ON, saved =
    success AND hazard contact <= HARD_CONTACT_GATE_FRAMES frames, 2 workers."""

    seed: int | None = None
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(list(range(24)), 2)
    filter_for_successful_trajectories: bool = True
    num_workers: int = 2

    @property
    def tag(self) -> str:
        return "franka_skin_pc_kitchen_reach_v1"


def _part(k: int) -> PickTaskSamplerConfig:
    return _sampler_config(list(range(24 * k, 24 * (k + 1))), 2)


@register_config("FrankaSkinPCKitchenReachCollectPart2Config")
class FrankaSkinPCKitchenReachCollectPart2Config(FrankaSkinPCKitchenReachCollectConfig):
    """DISJOINT houses 24..47, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(1)


@register_config("FrankaSkinPCKitchenReachCollectPart3Config")
class FrankaSkinPCKitchenReachCollectPart3Config(FrankaSkinPCKitchenReachCollectConfig):
    """DISJOINT houses 48..71, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(2)


@register_config("FrankaSkinPCKitchenReachCollectPart4Config")
class FrankaSkinPCKitchenReachCollectPart4Config(FrankaSkinPCKitchenReachCollectConfig):
    """DISJOINT houses 72..95, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(3)


@register_config("FrankaSkinPCKitchenReachCollectPart5Config")
class FrankaSkinPCKitchenReachCollectPart5Config(FrankaSkinPCKitchenReachCollectConfig):
    """DISJOINT houses 96..119, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(4)


@register_config("FrankaSkinPCKitchenReachCollectPart6Config")
class FrankaSkinPCKitchenReachCollectPart6Config(FrankaSkinPCKitchenReachCollectConfig):
    """DISJOINT houses 120..143, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(5)


@register_config("FrankaSkinPCKitchenReachCollectPart7Config")
class FrankaSkinPCKitchenReachCollectPart7Config(FrankaSkinPCKitchenReachCollectConfig):
    """DISJOINT houses 144..167, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(6)


@register_config("FrankaSkinPCKitchenReachCollectPart8Config")
class FrankaSkinPCKitchenReachCollectPart8Config(FrankaSkinPCKitchenReachCollectConfig):
    """DISJOINT houses 168..191, same recipe."""

    task_sampler_config: PickTaskSamplerConfig = _part(7)


__all__ = [
    "FrankaSkinPCKitchenReachConfig",
    "FrankaSkinPCKitchenReachCheckConfig",
    "FrankaSkinPCKitchenReachCleanCheckConfig",
    "FrankaSkinPCKitchenReachCollectConfig",
    "FrankaSkinPCKitchenReachCollectPart2Config",
    "FrankaSkinPCKitchenReachCollectPart3Config",
    "FrankaSkinPCKitchenReachCollectPart4Config",
    "FrankaSkinPCKitchenReachCollectPart5Config",
    "FrankaSkinPCKitchenReachCollectPart6Config",
    "FrankaSkinPCKitchenReachCollectPart7Config",
    "FrankaSkinPCKitchenReachCollectPart8Config",
]
