"""
Data generation configs for Franka move-to-pose tasks.

These configs subclass from the base_pick_config and are registered
for use in the data generation pipeline.
"""

import math
from pathlib import Path
import numpy as np

from molmo_spaces.configs import BasePolicyConfig, BaseRobotConfig
from molmo_spaces.configs.base_open_task_configs import ClosingBaseConfig, OpeningBaseConfig
from molmo_spaces.configs.base_pick_and_place_color_configs import PickAndPlaceColorDataGenConfig

# This is here so that un-pickling benchmarks works
from molmo_spaces.configs.base_pick_and_place_configs import (
    PickAndPlaceDataGenConfig,
)
from molmo_spaces.configs.base_pick_and_place_next_to_configs import PickAndPlaceNextToDataGenConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    FrankaEasyRandomizedDroidCameraSystem,
    FrankaGoProD405D455CameraSystem,
    FrankaOmniPurposeCameraSystem,
    FrankaRandomizedD405D455CameraSystem,
    FrankaRandomizedDroidCameraSystem,
    FrankaSkinCameraSystem,
    RBY1GoProD455CameraSystem,
)
from molmo_spaces.configs.policy_configs import (
    CuroboOpenClosePlannerPolicyConfig,
    CuroboPickAndPlacePlannerPolicyConfig,
    OpenClosePlannerPolicyConfig,
    PickPlannerPolicyConfig,
)
from molmo_spaces.configs.robot_configs import (
    FloatingRUMRobotConfig,
    FrankaRobotConfig,
    FrankaSkinRobotConfig,
    RBY1MConfig,
    RBY1MOpenCloseConfig,
)
from molmo_spaces.configs.task_sampler_configs import (
    OpenTaskSamplerConfig,
    PickAndPlaceColorTaskSamplerConfig,
    PickAndPlaceNextToTaskSamplerConfig,
    PickAndPlaceTaskSamplerConfig,
    PickTaskSamplerConfig,
    RUMPickTaskSamplerConfig,
)

# Oder of configs should be order the code is executed in
# scenes, robots, camera, task_sampler, policy, output
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_robot_paths
from molmo_spaces.tasks.opening_task_samplers import OpenTaskSampler
from molmo_spaces.tasks.pick_and_place_color_task_sampler import PickAndPlaceColorTaskSampler
from molmo_spaces.tasks.pick_and_place_next_to_task_sampler import PickAndPlaceNextToTaskSampler
from molmo_spaces.tasks.pick_and_place_task_sampler import (
    PickAndPlaceFixedTaskSampler,
    PickAndPlaceMultiTaskSampler,
    PickAndPlaceResampleCandidatesTaskSampler,
    PickAndPlaceTaskSampler,
)
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.constants.object_constants import PICK_AND_PLACE_OBJECTS
from molmo_spaces.utils.synset_utils import get_valid_pickupable_obja_uids


@register_config("FrankaPickDroidDataGenConfig")
class FrankaPickDroidDataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with DROID-style fixed cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_droid_datagen"


@register_config("FrankaPickGoProD405D455DataGenConfig")
class FrankaPickGoProD405D455DataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with GoPro D405 cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    num_workers: int = 4
    task_horizon: int = 150
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_go_pro_d405_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_go_pro_d405_datagen"


@register_config("FrankaPickRandomizedDataGenConfig")
class FrankaPickRandomizedDataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with randomized exocentric cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_randomized_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_randomized_datagen"


@register_config("RUMPickDataGenConfig")
class RUMPickDataGenConfig(PickBaseConfig):
    scene_dataset: str = "holodeck-objaverse"
    robot_config: FloatingRUMRobotConfig = FloatingRUMRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaRandomizedD405D455CameraSystem(
        img_resolution=(960, 720)
    )
    task_sampler_config: RUMPickTaskSamplerConfig = RUMPickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler, robot_object_z_offset=0
    )
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rum_pick_v1"

    @property
    def tag(self) -> str:
        return "rum_pick_datagen"


@register_config("FrankaPickAndPlaceDataGenConfig")
class FrankaPickAndPlaceDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedDroidCameraSystem = FrankaRandomizedDroidCameraSystem()
    policy_dt_ms: float = 66.0  # ~15hz
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_randomized_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_datagen"


@register_config("FrankaPickAndPlaceEasyDataGenConfig")
class FrankaPickAndPlaceEasyDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaEasyRandomizedDroidCameraSystem = FrankaEasyRandomizedDroidCameraSystem()
    policy_dt_ms: float = 66.0  # ~15hz
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_randomized_easy_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_easy_datagen"


@register_config("FrankaPickAndPlaceDroidDataGenConfig")
class FrankaPickAndPlaceDroidDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_droid_datagen"


@register_config("FrankaPickAndPlaceGoProD405D455DataGenConfig")
class FrankaPickAndPlaceGoProD405D455DataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_go_pro_d405_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_go_pro_d405_datagen"


@register_config("FrankaPickAndPlaceNextToDataGenConfig")
class FrankaPickAndPlaceNextToDataGenConfig(PickAndPlaceNextToDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_randomized_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_datagen"


@register_config("FrankaPickAndPlaceNextToDroidDataGenConfig")
class FrankaPickAndPlaceNextToDroidDataGenConfig(PickAndPlaceNextToDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_droid_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_droid_datagen"


@register_config("FrankaPickAndPlaceColorDataGenConfig")
class FrankaPickAndPlaceColorDataGenConfig(PickAndPlaceColorDataGenConfig):
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_colors_randomized_v1"
    )
    wandb_project: str = "molmo-spaces-data-generation"
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_datagen"


@register_config("FrankaPickAndPlaceColorDroidDataGenConfig")
class FrankaPickAndPlaceColorDroidDataGenConfig(PickAndPlaceColorDataGenConfig):
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_colors_droid_randomized_v1"
    )
    wandb_project: str = "molmo-spaces-data-generation"
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_droid_datagen"

@register_config("FrankaSkinPickAndPlaceDataGenConfig")
class FrankaSkinPickAndPlaceDataGenConfig(PickAndPlaceDataGenConfig):
    """Pick-and-place data generation with the franka_skin robot (29 SPAD proximity sensors)."""

    scene_dataset: str = "ithor"
    data_split: str = "train"
    robot_config: BaseRobotConfig = FrankaSkinRobotConfig()
    camera_config: FrankaSkinCameraSystem = FrankaSkinCameraSystem()
    # 60 Hz sub-stepping (matches abstract_exp_config default; period=0 was a footgun
    # that disabled proximity recording entirely — see prox_learning README §6.1 / PLA
    # memory `dataset_zero_proximity_bug.md`). Substep dim is mean-pooled downstream.
    proximity_sensor_period_ms: float = 16.6667
    task_horizon: int | None = 300
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_datagen"


@register_config("FrankaSkinPickAndPlacePilotConfig")
class FrankaSkinPickAndPlacePilotConfig(FrankaSkinPickAndPlaceDataGenConfig):
    """Mass iTHOR pick-and-place collection on the franka_skin pipeline. See README §6.1."""
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "train"
    num_workers: int = 4
    seed: int | None = np.random.randint(1, 1000000)
    filter_for_successful_trajectories: bool = True
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=5,
        house_inds= list(range(1999)),
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR /  "datagen" / "pick_and_place_skin_pilot_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_pilot"


@register_config("PACT")
class PACT(FrankaSkinPickAndPlacePilotConfig):
    seed: int | None = np.random.randint(1000)
    num_workers: int = 1
    # Collision probe: accept robot placements even in collision-prone spots and let
    # the arm operate freely; MuJoCo contact detection stays on so the per-step
    # collision metric (obs_scene["collision_metrics"]) is recorded. The 29 skin
    # proximity sensors record every policy step (inherited proximity_sensor_period_ms).
    disable_collision_checks: bool = True
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=1,
        house_inds=list(range(11, 21)),  #
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_object_z_offset_random_min=-np.random.uniform(0.0, 1.0),
        robot_object_z_offset_random_max=np.random.uniform(0.0, 1.0),
        robot_placement_rotation_range_rad=0.52,
        #randomize_textures=True,
        randomize_lighting=True,
        #randomize_textures_all = True,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "house_7_test" / str(seed)

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_pilot_medium"
@register_config("FrankaSkinPickAndPlacePilotEvalHoldoutConfig")
class FrankaSkinPickAndPlacePilotEvalHoldoutConfig(FrankaSkinPickAndPlacePilotConfig):
    """Held-out 10-house eval set for the franka_skin smoke validation round.
    Mirrors the smoke training config but on houses 11-20 (disjoint from the
    1-10 training houses) with a different seed (2027 vs 2026). Used to build
    a JsonBenchmark for evaluating PLA vs baseline ACT policies under the same
    robot + camera + proximity setup they were trained on. The cached
    FrankaPickandPlaceHardBench is built for franka_droid, which has neither
    our cameras (exo_camera_1, wrist_camera) nor our 29 SPAD sensors."""
    seed: int | None = 2027
    num_workers: int = 2
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=4,
        house_inds=list(range(11, 21)),
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_pilot_eval_holdout_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_pilot_eval_holdout"


@register_config("FrankaSkinPickAndPlacePilotSmokeConfig")
class FrankaSkinPickAndPlacePilotSmokeConfig(FrankaSkinPickAndPlacePilotConfig):
    """10-house, 4-samples-each smoke test for proximity recording. Run this first to
    confirm proximity_sensor_period_ms is wired correctly before launching the full
    pilot. Output is segregated under pick_and_place_skin_pilot_smoke_v1 so it
    doesn't collide with the full pilot."""
    seed: int | None = 2026
    num_workers: int = 2
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=4,
        house_inds=list(range(1, 11)),
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_pilot_smoke_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_pilot_smoke"


@register_config("FrankaSkinPickAndPlaceOneHouseMugConfig")
class FrankaSkinPickAndPlaceOneHouseMugConfig(FrankaSkinPickAndPlaceDataGenConfig):
    """Single-house, single-skill collection: pick THE mug and place it on THE
    same receptacle, repeated `samples_per_house` times in one house.

    Uses `PickAndPlaceFixedTaskSampler`, which locks the (pickup object, place
    receptacle) pair to the first valid task sampled in the house. Every
    episode reproduces that exact task identity while textures, lighting,
    robot base pose, and receptacle position are still randomized per attempt
    — so the dataset is visually diverse but the *skill* is fixed. This is the
    setup for the proximity-vs-RGB ablation, where a drifting pickup object or
    auto-advancing receptacle would contaminate the comparison.

    Throughput: single worker by design (num_workers>1 collides on a single
    house — every worker calls setup_house_dirs(house_id) and clobbers the
    others' outputs). ~190 s per successful 300-step rollout → ~13 h for 250
    episodes. To parallelize, launch N processes with disjoint output_dirs
    (scripts/run_v3_parallel.py) and merge (scripts/merge_v3_to_act_style.py).
    """
    seed: int | None = 2026
    num_workers: int = 1
    filter_for_successful_trajectories: bool = True
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceFixedTaskSampler,
        pickup_types=["mug"],
        samples_per_house=250,
        house_inds=[1],
        # episodes_per_receptacle is forced to 0 by the sampler, but set it here
        # too so the intent is explicit and the config reads correctly.
        episodes_per_receptacle=0,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_one_house_mug_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_one_house_mug"


# Body-name prefixes (case-insensitive) for surfaces where the proximity skin is
# expected to shine: low seats, beds/baths, sinks, bookshelves, dressers, etc.
# Excludes flat tables/counters which dominate the unfiltered distribution.
LOW_SURFACE_PREFIXES: tuple[str, ...] = (
    "sink",
    "shelf",
    "bookshelf",
    "chair",
    "armchair",
    "stool",
    "sofa",
    "bed",
    "bathtub",
    "toilet",
    "crapper",
    "dresser",
    "chestofdrawers",
)


@register_config("FrankaSkinLowSurfacePickAndPlaceDataGenConfig")
class FrankaSkinLowSurfacePickAndPlaceDataGenConfig(FrankaSkinPickAndPlaceDataGenConfig):
    """Pick-and-place biased to low/enclosed source surfaces (sinks, shelves, low seats,
    beds, bathtubs, toilets, dressers). Filters candidate pickups by the body name of
    the surface they rest on so the proximity skin gets exercised."""

    scene_dataset: str = "procthor-objaverse"
    data_split: str = "train"
    num_workers: int = 4
    seed: int | None = np.random.randint(1, 1000000)
    filter_for_successful_trajectories: bool = True
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=5,
        house_inds=list(range(1999)),
        max_allowed_sequential_irrecoverable_failures=10000,
        source_surface_types=LOW_SURFACE_PREFIXES,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_low_surface_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_low_surface"


@register_config("FrankaSkinLowSurfacePickAndPlacePilotConfig")
class FrankaSkinLowSurfacePickAndPlacePilotConfig(FrankaSkinLowSurfacePickAndPlaceDataGenConfig):
    """Smaller pilot of the low-surface pick-and-place dataset for quick verification."""

    seed: int | None = 7
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=3,
        house_inds=list(range(200)),
        max_allowed_sequential_irrecoverable_failures=10000,
        source_surface_types=LOW_SURFACE_PREFIXES,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_low_surface_pilot_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_low_surface_pilot"

@register_config("PACT_LowSurface")
class PACT_LowSurface(FrankaSkinLowSurfacePickAndPlaceDataGenConfig):
    """PACT experiment on low/enclosed source surfaces. The proximity skin
    is actually exercised here because the arm has to approach beds, shelves,
    sinks, etc. from the side or from within a confined volume."""
    seed: int | None = 2026
    num_workers: int = 1
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=2,
        house_inds=[0],   # 5 houses, not 1
        max_allowed_sequential_irrecoverable_failures=10000,
        source_surface_types=LOW_SURFACE_PREFIXES,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pact_low_surface" / str(seed)

    @property
    def tag(self) -> str:
        return "pact_low_surface"


@register_config("FrankaSkinProxNecessityPilotConfig")
class FrankaSkinProxNecessityPilotConfig(FrankaSkinLowSurfacePickAndPlaceDataGenConfig):
    """Pilot for the proximity-NECESSITY regime — the setting where vision fails and
    the skin must carry the trajectory (CoRL thesis).

    Three levers, all relative to PACT:
      1. ``check_robot_placement_visibility=False`` — drop the guarantee that the exo
         camera can see the target before placing the robot. PACT keeps this True, which
         is *why* its vision-blind fraction is 0% (measured: scripts/proximity_necessity.py).
      2. ``source_surface_types=LOW_SURFACE_PREFIXES`` — enclosed/recessed targets (sink
         basin, shelf interior, under furniture) where even the wrist cam loses the object.
      3. heavy clutter packed close + small robot stand-off so the arm cannot back off for
         a clear view and must operate among surfaces (skin active) most of the trajectory.

    This OVER-GENERATES candidates; it does not hand-design scenes. Curate with
    ``scripts/proximity_necessity.py`` and KEEP trajectories with
    ``prox_active_frac >= 0.8`` and high ``vision_blind_frac``. Collisions are left ON
    (privileged planner avoids them in the demo; a vision-only policy cannot, which is the
    intended P+ACT vs ACT contrast at rollout)."""

    num_workers: int = 2
    seed: int | None = 2026
    disable_collision_checks: bool = False  # real obstacles — vision-only policy must collide at rollout
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        source_surface_types=LOW_SURFACE_PREFIXES,   # enclosed / recessed targets
        check_robot_placement_visibility=False,      # KEY: no "camera must see target" guarantee
        num_added_pickups=60,                        # heavy occluding clutter (PACT: 30)
        min_reference_to_added_pickup_dist=0.05,     # pack clutter right around the target
        max_reference_to_added_pickup_dist=0.30,
        max_robot_to_added_pickup_dist=0.5,
        base_pose_sampling_radius_range=(0.0, 0.45),  # robot can't retreat for a clear view
        robot_placement_rotation_range_rad=0.52,
        samples_per_house=3,
        house_inds=list(range(1, 51)),               # small pilot — iterate, then scale
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_prox_necessity_pilot_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_prox_necessity_pilot"


@register_config("FrankaOpenDataGenConfig")
class FrankaOpenDataGenConfig(OpeningBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0.67 for close task, 0 for open task
    )
    policy_config: BasePolicyConfig = OpenClosePlannerPolicyConfig()
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "open_v1"

    @property
    def tag(self) -> str:
        return "franka_open_datagen"


@register_config("RBY1OpenDataGenConfig")
class RBY1OpenDataGenConfig(OpeningBaseConfig):
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_open_v1"
    wandb_project: str = "mujoco-thor-data-generation"
    robot_config: RBY1MOpenCloseConfig = RBY1MOpenCloseConfig()
    policy_config: BasePolicyConfig = CuroboOpenClosePlannerPolicyConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0.67 for close task, 0 for open task
        robot_safety_radius=0.2,
        base_pose_sampling_radius_range=(0.3, 1.0),
    )
    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    use_passive_viewer: bool = False
    seed: int = None
    filter_for_successful_trajectories: bool = True
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step

    @property
    def tag(self) -> str:
        return "rby1_open_datagen"

    def _init_policy_config(self) -> CuroboPickAndPlacePlannerPolicyConfig:
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
        from molmo_spaces.policy.solvers.object_manipulation.curobo_open_close_planner_policy import (
            CuroboOpenClosePlannerPolicy,
        )

        rby1_path = get_robot_paths().get("rby1m")
        assert rby1_path is not None, "RBY1 robot path not found"

        left_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1_path / "curobo_config" / "rby1m_left_arm_holobase.yml"
            ),
            collision_activation_distance=0.01,
            num_trajopt_seeds=12,
            max_attempts=15,
            num_ik_seeds=128,
            trajopt_tsteps=48,
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
            check_start_validity=False,
            enable_finetune_trajopt=True,
        )
        right_curobo_planner_config = left_curobo_planner_config.model_copy(deep=True)
        right_curobo_planner_config.curobo_robot_config_path = str(
            rby1_path / "curobo_config" / "rby1m_right_arm_holobase.yml"
        )
        return CuroboOpenClosePlannerPolicyConfig(
            policy_cls=CuroboOpenClosePlannerPolicy,
            left_curobo_planner_config=left_curobo_planner_config,
            right_curobo_planner_config=right_curobo_planner_config,
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_config = self._init_policy_config()
        self.task_config.task_success_threshold = 0.67
        self.task_sampler_config.randomize_textures = True


@register_config("RBY1PickAndPlaceDataGenConfig")
class RBY1PickAndPlaceDataGenConfig(PickAndPlaceDataGenConfig):
    seed: int | None = None  # 75133535  # 4821697
    viewer_cam_dict: dict = {"camera": "robot_0/camera_follower"}
    use_passive_viewer: bool = False
    task_horizon: int | None = 400  # Maximum number of steps per episode (if None, no time limit)
    filter_for_successful_trajectories: bool = True
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_pick_and_place_v1"
    wandb_project: str = "mujoco-thor-data-generation"
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step

    robot_config: RBY1MConfig = RBY1MConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    policy_config: CuroboPickAndPlacePlannerPolicyConfig | None = None

    def _init_policy_config(self) -> CuroboPickAndPlacePlannerPolicyConfig:
        from molmo_spaces.policy.solvers.object_manipulation.curobo_pick_and_place_planner_policy import (
            CuroboPickAndPlacePlannerPolicy,
        )

        rby1_path = get_robot_paths().get("rby1m")
        assert rby1_path is not None, "RBY1 robot path not found"
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig

        left_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1_path / "curobo_config" / "rby1m_left_arm_holobase.yml"
            ),
            collision_activation_distance=0.01,
            num_trajopt_seeds=12,
            max_attempts=15,
            num_ik_seeds=128,
            trajopt_tsteps=48,
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
            check_start_validity=False,
            enable_finetune_trajopt=True,
        )
        right_curobo_planner_config = left_curobo_planner_config.model_copy(deep=True)
        right_curobo_planner_config.curobo_robot_config_path = str(
            rby1_path / "curobo_config" / "rby1m_right_arm_holobase.yml"
        )
        return CuroboPickAndPlacePlannerPolicyConfig(
            policy_cls=CuroboPickAndPlacePlannerPolicy,
            left_curobo_planner_config=left_curobo_planner_config,
            right_curobo_planner_config=right_curobo_planner_config,
            enable_collision_avoidance=True,
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        try:
            self.policy_config = self._init_policy_config()
        except RuntimeError as e:
            # Check if this is a CUDA/GPU-related error
            error_msg = str(e)
            if "NVIDIA" in error_msg or "CUDA" in error_msg or "GPU" in error_msg:
                # No GPU available - this is expected on manager nodes that just coordinate jobs
                # Policy config will be initialized later on worker nodes that have GPUs
                print(
                    f"Warning: Skipping policy config initialization due to missing GPU: {error_msg}"
                )
                self.policy_config = None
            else:
                raise
        self.robot_config.init_qpos["head"][1] = 0.6
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.max_robot_to_obj_dist = 0.5
        self.task_sampler_config.object_placement_radius_range = (0.1, 0.5)
        self.task_sampler_config.min_object_to_receptacle_dist = 0.05
        self.task_sampler_config.max_robot_to_place_receptacle_dist = 0.5

    @property
    def tag(self) -> str:
        return "rby1_pick_and_place_datagen"


@register_config("RBY1PickDataGenConfig")
class RBY1PickDataGenConfig(PickBaseConfig):
    seed: int | None = None
    viewer_cam_dict: dict = {"camera": "robot_0/camera_follower"}
    use_passive_viewer: bool = False
    task_horizon: int | None = 400  # Maximum number of steps per episode (if None, no time limit)
    filter_for_successful_trajectories: bool = True
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_pick_v1"
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step

    robot_config: RBY1MConfig = RBY1MConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    policy_config: CuroboPickAndPlacePlannerPolicyConfig | None = None

    def _init_policy_config(self) -> CuroboPickAndPlacePlannerPolicyConfig:
        from molmo_spaces.policy.solvers.object_manipulation.curobo_pick_and_place_planner_policy import (
            CuroboPickAndPlacePlannerPolicy,
        )

        rby1_path = get_robot_paths().get("rby1m")
        assert rby1_path is not None, "RBY1 robot path not found"
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig

        left_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1_path / "curobo_config" / "rby1m_left_arm_holobase.yml"
            ),
            collision_activation_distance=0.01,
            num_trajopt_seeds=12,
            max_attempts=15,
            num_ik_seeds=128,
            trajopt_tsteps=48,
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
            check_start_validity=True,
            enable_finetune_trajopt=True,
        )
        right_curobo_planner_config = left_curobo_planner_config.model_copy(deep=True)
        right_curobo_planner_config.curobo_robot_config_path = str(
            rby1_path / "curobo_config" / "rby1m_right_arm_holobase.yml"
        )
        return CuroboPickAndPlacePlannerPolicyConfig(
            policy_cls=CuroboPickAndPlacePlannerPolicy,
            left_curobo_planner_config=left_curobo_planner_config,
            right_curobo_planner_config=right_curobo_planner_config,
            enable_collision_avoidance=True,
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        try:
            self.policy_config = self._init_policy_config()
        except RuntimeError as e:
            # Check if this is a CUDA/GPU-related error
            error_msg = str(e)
            if "NVIDIA" in error_msg or "CUDA" in error_msg or "GPU" in error_msg:
                # No GPU available - this is expected on manager nodes that just coordinate jobs
                # Policy config will be initialized later on worker nodes that have GPUs
                print(
                    f"Warning: Skipping policy config initialization due to missing GPU: {error_msg}"
                )
                self.policy_config = None
            else:
                raise
        self.robot_config.init_qpos["head"][1] = 0.6
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.max_robot_to_obj_dist = 0.5
        self.task_sampler_config.object_placement_radius_range = (0.1, 0.5)

    @property
    def tag(self) -> str:
        return "rby1_pick_datagen"


@register_config("FrankaCloseDataGenConfig")
class FrankaCloseDataGenConfig(ClosingBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0.5,  # 0.67 for close task, 0 for open task
    )
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "close_v1"

    @property
    def tag(self) -> str:
        return "franka_close_datagen"


@register_config("FrankaPickAndPlaceGoProD405D455DataGenConfigDebug")
class FrankaPickAndPlaceGoProD405D455DataGenConfigDebug(FrankaPickAndPlaceDroidDataGenConfig):
    """Data generation config for Franka pick and place task with GoPro D405 cameras - deterministic version."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        samples_per_house=10,
        max_tasks=100,
        # pickup_types=["Mug"],
        house_inds=[2],
    )
    num_workers: int = 1
    task_horizon: int = 100
    use_wandb: bool = False
    log_level: str = "debug"
    filter_for_successful_trajectories: bool = False
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_go_pro_d405_v1_debug"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_go_pro_d405_d455_datagen_debug"


@register_config("FrankaPickOmniCamConfig")
class FrankaPickOmniCamConfig(PickBaseConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_omni_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_omni_datagen"


@register_config("FrankaPickOmniCamAblationConfig")
class FrankaPickOmniCamAblationConfig(FrankaPickOmniCamConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_omni_v1_cam_ablation"

    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        added_pickup_objects=get_valid_pickupable_obja_uids(),
        # num_added_pickups=30, these are defaults
        # episodes_per_added_pickup=1,
    )

    @property
    def tag(self) -> str:
        return "franka_pick_omni_v1_cam_ablation"


@register_config("FrankaPickAndPlaceOmniCamConfig")
class FrankaPickAndPlaceOmniCamConfig(PickAndPlaceDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_omni_v1"
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_omnicam_datagen"


# PickAndPlaceNextToDataGenConfig
@register_config("FrankaPickAndPlaceNextToOmniCamConfig")
class FrankaPickAndPlaceNextToOmniCamConfig(PickAndPlaceNextToDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_omni_v1"
    )
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_omnicam_datagen"


# PickAndPlaceColorDataGenConfig
@register_config("FrankaPickAndPlaceColorOmniCamConfig")
class FrankaPickAndPlaceColorOmniCamConfig(PickAndPlaceColorDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_color_omni_v1"
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_omnicam_datagen"


################################################################################
# Benchmark configs
################################################################################


@register_config("FrankaPickDroidMiniBench")
class FrankaPickDroidMiniBench(PickBaseConfig):
    scene_dataset: str = "procthor-10k"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_minbench"


@register_config("FrankaPickandPlaceMiniBench")
class FrankaPickandPlaceDroidMiniBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-10k"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplace_minbench"


@register_config("FrankaPickDroidBench")
class FrankaPickDroidBench(PickBaseConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_bench"


@register_config("FrankaPickandPlaceDroidBench")
class FrankaPickandPlaceDroidBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplace_bench"


@register_config("FrankaPickandPlaceNextToDroidBench")
class FrankaPickandPlaceNextToDroidBench(PickAndPlaceNextToDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_next_to_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplacenextto_bench"


@register_config("FrankaPickandPlaceColorDroidBench")
class FrankaPickandPlaceColorDroidBench(PickAndPlaceColorDataGenConfig):
    """Data generation config for Franka pick task with DROID-style fixed cameras."""

    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_color_obja_v1"
    scene_dataset: str = "procthor-objaverse"

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pickandplacecolor_bench"


@register_config("FrankaOpenHardBench")
class FrankaOpenHardBench(OpeningBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "val"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0.67 for close task, 0 for open task
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    policy_config: BasePolicyConfig = OpenClosePlannerPolicyConfig()
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "open_bench"

    @property
    def tag(self) -> str:
        return "franka_open_hard_bench"


@register_config("FrankaCloseHardBench")
class FrankaCloseHardBench(ClosingBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "val"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0.5,  # 0.67 for close task, 0 for open task
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "close_bench"

    @property
    def tag(self) -> str:
        return "franka_close_hard_bench"


@register_config("FrankaPickHardBench")
class FrankaPickHardBench(PickBaseConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_hard_bench"


@register_config("FrankaPickandPlaceHardBench")
class FrankaPickandPlaceHardBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )

    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_hard_bench"


@register_config("FrankaPickandPlaceNextToHardBench")
class FrankaPickandPlaceNextToHardBench(PickAndPlaceNextToDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceNextToTaskSamplerConfig(
        task_sampler_class=PickAndPlaceNextToTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_next_to_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_hard_bench"


@register_config("FrankaPickandPlaceColorHardBench")
class FrankaPickandPlaceColorHardBench(PickAndPlaceColorDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )

    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceColorTaskSamplerConfig = PickAndPlaceColorTaskSamplerConfig(
        task_sampler_class=PickAndPlaceColorTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_color_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_hard_bench"


@register_config("MultiPnPTask")
class RUMPickAndPlaceMultiDataGenConfig(PickAndPlaceDataGenConfig):
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pnpmulti_V1"
    wandb_project: str = "mujoco-thor-data-generation"
    robot_config: FloatingRUMRobotConfig = FloatingRUMRobotConfig()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceMultiTaskSampler,
        pickup_types=None,
        samples_per_house=20,
        house_inds=[7],  # TODO(Omar): choose one you like.
        robot_object_z_offset=0.2,
        check_robot_placement_visibility=False,
    )

    camera_config: FrankaDroidCameraSystem = FrankaRandomizedD405D455CameraSystem(
        img_resolution=(960, 720),
        visibility_constraints=None,
        allow_relaxed_constraints=True,
    )

    @property
    def tag(self) -> str:
        return "pnpmulti_bench"
