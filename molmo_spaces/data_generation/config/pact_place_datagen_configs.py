"""Ready-to-run configs for the supported PACT place environments."""

from pathlib import Path

from molmo_spaces.configs import BasePolicyConfig
from molmo_spaces.configs.camera_configs import (
    FrankaSkinHybridWristOnlyCameraSystem,
)
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinHybridObstacleCheckConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.data_generation.pact_place.contracts import (
    V1010_SCENE_BY_POSE,
    v1010_cell,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pact_place import (
    PactPlaceCorridorPolicyConfig,
    PactPlaceCorridorTask,
    PactPlaceCorridorV1010FourObjectSampler,
    PactPlaceV5Sampler,
    PactPlaceV95RealClutterSampler,
)

_CUSTOM_SCENES = Path(__file__).resolve().parents[1] / "custom_scenes"


def _sampler_config(sampler, scene_paths: list[str]) -> PickTaskSamplerConfig:
    return PickTaskSamplerConfig(
        task_sampler_class=sampler,
        scene_xml_paths=scene_paths,
        house_inds=list(range(len(scene_paths))),
        samples_per_house=1,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_object_z_offset_random_min=-0.5,
        robot_object_z_offset_random_max=0.5,
        robot_placement_rotation_range_rad=0.20,
        randomize_textures=False,
        randomize_lighting=False,
    )


class _PactPlaceBaseConfig(FrankaSkinHybridObstacleCheckConfig):
    """Shared observation, task and expert settings for the released lineages."""

    task_type: str = "pick_and_place"
    camera_config: FrankaSkinHybridWristOnlyCameraSystem = FrankaSkinHybridWristOnlyCameraSystem()
    policy_config: BasePolicyConfig = PactPlaceCorridorPolicyConfig()
    task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(task_cls=PactPlaceCorridorTask)
    policy_dt_ms: float = 66.0
    proximity_sensor_period_ms: float = 16.6667
    end_on_success: bool = False
    filter_for_successful_trajectories: bool = False
    viz_sensor_rgb: bool = False
    num_workers: int = 1

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # The reported expert collections explicitly disabled actuator noise.
        # Keep the public configs deterministic instead of relying on a runner
        # to remember this scientific-contract setting.
        self.robot_config.action_noise_config.enabled = False


@register_config("FrankaSkinPactPlaceV5Config")
class FrankaSkinPactPlaceV5Config(_PactPlaceBaseConfig):
    """Experiment V5: two deterministic panel sides, no household clutter."""

    task_horizon: int | None = 900
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PactPlaceV5Sampler,
        [str(_CUSTOM_SCENES / "pact_place_corridor_v2.xml")] * 2,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pact_place_v5"

    @property
    def tag(self) -> str:
        return "franka_skin_pact_place_v5"


@register_config("FrankaSkinPactPlaceV95RealClutterConfig")
class FrankaSkinPactPlaceV95RealClutterConfig(_PactPlaceBaseConfig):
    """V9.5: four layout families × two panel sides, eight movable objects."""

    task_horizon: int | None = 900
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PactPlaceV95RealClutterSampler,
        [str(_CUSTOM_SCENES / "pact_place_corridor_v5.xml")] * 8,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pact_place_v95_real_clutter"

    @property
    def tag(self) -> str:
        return "franka_skin_pact_place_v95_real_clutter"


def _v1010_scene_paths() -> list[str]:
    paths = []
    for index in range(24):
        _family, _side, pose = v1010_cell(index)
        paths.append(str(_CUSTOM_SCENES / V1010_SCENE_BY_POSE[pose]["filename"]))
    return paths


@register_config("FrankaSkinPactPlaceV1010FourObjectConfig")
class FrankaSkinPactPlaceV1010FourObjectConfig(_PactPlaceBaseConfig):
    """V10.10: V9.5 chicane, four live objects and one static pendant."""

    task_horizon: int | None = 1050
    task_sampler_config: PickTaskSamplerConfig = _sampler_config(
        PactPlaceCorridorV1010FourObjectSampler,
        _v1010_scene_paths(),
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pact_place_v1010_four_object"

    @property
    def tag(self) -> str:
        return "franka_skin_pact_place_v1010_four_object"


__all__ = [
    "FrankaSkinPactPlaceV5Config",
    "FrankaSkinPactPlaceV95RealClutterConfig",
    "FrankaSkinPactPlaceV1010FourObjectConfig",
]
