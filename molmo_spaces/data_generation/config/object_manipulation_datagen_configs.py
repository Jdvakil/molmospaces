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
    FrankaSkinHybridCameraSystem,
    RBY1GoProD455CameraSystem,
)
from molmo_spaces.configs.policy_configs import (
    CuroboOpenClosePlannerPolicyConfig,
    CuroboPickAndPlacePlannerPolicyConfig,
    OpenClosePlannerPolicyConfig,
    PickPlannerPolicyConfig,
)
from molmo_spaces.configs.robot_configs import (
    ActionNoiseConfig,
    FloatingRUMRobotConfig,
    FrankaRobotConfig,
    FrankaSkinRobotConfig,
    FrankaSkinHybridRobotConfig,
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
    # Regular (collision-avoiding) data collection: keep the standard placement
    # collision rejection so demos spawn and move clean. The per-step collision
    # metric (obs_scene["collision_metrics"]) is still recorded — it should now read
    # ~0, confirming clean motions. (Set True to re-enable the collision probe.)
    disable_collision_checks: bool = False
    # Minimal Gaussian action noise on every executed action, for state-coverage /
    # robustness so the learned policy makes cleaner, collision-avoiding motions at
    # inference. Truncated-Gaussian end-effector noise (std = 10% of each commanded TCP
    # delta) hard-capped at 1 cm (0.01 m) and ~0.6 deg, mapped to joints via the Jacobian.
    robot_config: BaseRobotConfig = FrankaSkinRobotConfig(
        action_noise_config=ActionNoiseConfig(
            enabled=True,
            action_scale_factor=0.1,
            max_tcp_position_noise=0.01,   # "very minimal, ~0.01 m"
            rotation_noise_scale=0.1,
            max_tcp_rotation_noise=0.01,
        )
    )
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


@register_config("FrankaSkinPickAndPlacePilotMediumConfig")
class FrankaSkinPickAndPlacePilotMediumConfig(FrankaSkinPickAndPlacePilotConfig):
    """Medium-scale collection (~500 episodes) for first-pass PLA training before the
    full 1999-house pilot. Smoke (10 houses) already confirmed proximity recording
    works end-to-end; this is the dataset we actually train on while waiting for full.

    num_workers=2 (intentional): each worker spawns its own MuJoCo simulator + scene
    loader at ~6-7 GB RSS. On a 62 GB box, num_workers=8 → ~50 GB and OOMs adjacent
    workloads. Stay at 2 unless on a server with >128 GB RAM."""
    seed: int | None = 2026
    num_workers: int = 1
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=1,
        house_inds=[3],  #
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_object_z_offset_random_min=-np.random.uniform(0.0, 1.0),
        robot_object_z_offset_random_max=np.random.uniform(0.0, 1.0),
        robot_placement_rotation_range_rad=0.52,
        #randomize_textures=True,
        randomize_lighting=True,
        #randomize_textures_all = True,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "mug_house_3_random_everything"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_pilot_medium"


@register_config("FrankaSkinPickAndPlaceOneHouseMugConfig")
class FrankaSkinPickAndPlaceOneHouseMugConfig(FrankaSkinPickAndPlacePilotConfig):
    """Single-house single-task collection for the vanilla-ACT reproduction baseline.

    ONE house (house_1), ONE pickup type (mug), 250 episodes. With aggressive
    per-attempt randomization so the resulting dataset is not a degenerate
    near-identical replay (which previously produced 0% eval success — see
    PLA memory `dataset_dup250_duplicate_demos`).

    Randomization sources (all per-attempt unless noted):
      - textures (scene + robot)            -> randomize_textures / randomize_robot_textures
      - lighting                            -> randomize_lighting
      - object/robot dynamics               -> randomize_dynamics
      - object placement & robot base pose  -> sampled in PickAndPlaceTaskSampler
      - initial arm qpos noise              -> FrankaRobotConfig.init_qpos_noise_range
      - per-step TCP-bounded action noise   -> robot_config.action_noise_config (enabled by default)

    Throughput notes (RTX 4090 / 48 CPU / 62 GB RAM):
      - As shipped, this config is single-worker single-house: ~190 s per
        successful 300-step rollout → ~13 h for 250 episodes overnight.
      - For 4x parallelism, do NOT use `num_workers>1` with duplicate
        `house_inds=[1,1,1,1]`: every worker calls
        `setup_house_dirs(house_id=1)` and clobbers the others'
        `house_1/episode_*.mp4` and `trajectories_batch_1_of_1.h5`. Instead
        launch 4 separate processes with disjoint `output_dir`s, e.g. via
        `scripts/run_v3_parallel.py`, then merge with
        `scripts/merge_v3_to_act_style.py`. ~3.3 h wall-clock for 250 ep.
      - `task_sampler_class=PickAndPlaceResampleCandidatesTaskSampler` is
        required: the default sampler permanently blacklists pickup
        candidates after robot-placement failures, which strands single-house
        runs at 1/N successful saves once the (small) candidate pool drains.
      - max_total_attempts_multiplier=25 buys headroom for the higher
        rejection rate from per-attempt randomization without burning budget.
    """
    seed: int | None = 2026
    num_workers: int = 1
    # `collision_free_pose_limit` lives on the exp config (not task sampler
    # config). Default of 3 makes place_robot_near give up after finding
    # 3 collision-free poses, so visibility-check fails out without
    # exhausting `max_robot_placement_attempts`. Bumped to 30 so the
    # planner actually uses the placement-attempt budget for finding a
    # visible angle on house_1's tricky mug spawn (bench 20260516_2354).
    collision_free_pose_limit: int = 30
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceResampleCandidatesTaskSampler,
        pickup_types=["mug"],
        samples_per_house=250,
        house_inds=[1],
        max_allowed_sequential_task_sampler_failures=2000,
        max_allowed_sequential_rollout_failures=2000,
        max_allowed_sequential_irrecoverable_failures=10000,
        max_total_attempts_multiplier=25,
        # Effectively disable per-asset blacklisting: house_1's 2 mugs are
        # known-good and we want them reused indefinitely. Default of 10
        # bricks the run after ~5-8 successes (see parallel_20260516_182123).
        max_asset_failures=10_000,
        # The fixed exo_camera_1 offset means SOME house_1 mug spawn poses are
        # only visible from a narrow band of robot placements. Default 10
        # samples isn't enough; bump to 80 so the planner has a real chance
        # to find a placement that satisfies visibility. Also requires
        # bumping collision_free_pose_limit (above) so the placement loop
        # doesn't early-exit at 3 candidates.
        max_robot_placement_attempts=80,
        base_pose_sampling_radius_range=(0.0, 1.2),
        # Both mugs in house_1 have spawn positions where NO robot placement
        # within the search radius satisfies the visibility constraint
        # (verified bench_v3_v2_20260516_235649: 30/30 collision-free poses
        # had visibility=0.000 for mug_ac3a and mug_3eb). The scripted
        # teleop policy uses ground-truth pose and doesn't need visibility
        # to succeed; the ACT policy at eval time is the only consumer of
        # the visibility constraint, and "mug barely visible at t=0" is a
        # tolerable / arguably desirable diversity signal for training.
        check_robot_placement_visibility=False,
        randomize_textures=True,
        randomize_robot_textures=True,
        randomize_lighting=True,
        randomize_dynamics=True,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_one_house_mug_v3"

    @property
    def tag(self) -> str:
        return "franka_skin_one_house_mug"


@register_config("FrankaSkinPickAndPlaceOneHouseMugFixedConfig")
class FrankaSkinPickAndPlaceOneHouseMugFixedConfig(FrankaSkinPickAndPlaceDataGenConfig):
    """Single-house, single-skill mug collection using a fixed task identity."""

    seed: int | None = 2026
    num_workers: int = 1
    filter_for_successful_trajectories: bool = True
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceFixedTaskSampler,
        pickup_types=["mug"],
        samples_per_house=250,
        house_inds=[1],
        episodes_per_receptacle=0,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_one_house_mug_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_one_house_mug"


@register_config("FrankaSkinPickAndPlaceOneHouseMugFastConfig")
class FrankaSkinPickAndPlaceOneHouseMugFastConfig(FrankaSkinPickAndPlaceOneHouseMugConfig):
    """Faster variant of FrankaSkinPickAndPlaceOneHouseMugConfig with texture
    randomization disabled.

    Texture rebinds (scene + robot) are the single most expensive per-attempt
    randomization in the MuJoCo classic renderer and the least load-bearing
    for behavior cloning on a fixed camera setup on a fixed scene — the
    policy sees a static viewpoint of a static room, so re-skinning walls
    and the Franka doesn't change the task-relevant feature distribution.

    Diversity preserved (these are what prevented the dataset_dup250 ACT
    collapse, not textures):
      - lighting                            -> randomize_lighting=True
      - dynamics                            -> randomize_dynamics=True
      - object/robot placement & base pose  -> task sampler
      - initial arm qpos noise              -> FrankaRobotConfig defaults
      - per-step TCP-bounded action noise   -> robot_config.action_noise_config

    Validation protocol before committing to a 250-ep run:
      1. Collect 50 ep with FrankaSkinPickAndPlaceOneHouseMugFastPilotConfig
         (below).
      2. Train vanilla ACT on those 50.
      3. Compare eval success against a 50-ep ACT trained on v3
         (textures-on parent). If within noise -> textures aren't
         load-bearing, run the full 250 here.

    Expected speedup is empirical, not theoretical: profile a 5-episode
    sample of v3 vs this and measure. My prior is 1.5-2.5x per-rollout,
    but it depends on how much of v3's 190 s/rollout was actually
    texture-rebind vs other rendering / proximity-substepping / placement
    search. If the speedup is <1.5x, the bottleneck is elsewhere and you
    should profile before chasing it.

    Knob NOT changed but worth tuning next if you need more speed:
      - max_robot_placement_attempts=80 + collision_free_pose_limit=30
        were tuned for visibility-ON case. With check_robot_placement_visibility
        already False here, both are likely overprovisioned. Try halving
        them in a follow-up config once this one is validated.

    Output is segregated to pick_and_place_one_house_mug_v4 so it does not
    collide with the v3 textures-on dataset.
    """
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceResampleCandidatesTaskSampler,
        pickup_types=["mug"],
        samples_per_house=250,
        house_inds=[1],
        max_allowed_sequential_task_sampler_failures=2000,
        max_allowed_sequential_rollout_failures=2000,
        max_allowed_sequential_irrecoverable_failures=10000,
        max_total_attempts_multiplier=25,
        max_asset_failures=10_000,
        max_robot_placement_attempts=80,
        base_pose_sampling_radius_range=(0.0, 1.2),
        check_robot_placement_visibility=False,
        randomize_textures=False,         # <-- changed (was True)
        randomize_robot_textures=False,   # <-- changed (was True)
        randomize_lighting=True,
        randomize_dynamics=True,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_one_house_mug_v4"

    @property
    def tag(self) -> str:
        return "franka_skin_one_house_mug_fast"


@register_config("FrankaSkinPickAndPlaceOneHouseMugFastPilotConfig")
class FrankaSkinPickAndPlaceOneHouseMugFastPilotConfig(
    FrankaSkinPickAndPlaceOneHouseMugFastConfig
):
    """50-episode pilot of the textures-off variant for parity validation.

    Use this to verify that an ACT trained on textures-off data matches
    eval performance of an ACT trained on the textures-on v3 data, before
    burning compute on the full 250-ep collection. Different seed (2027)
    from the parent so the 50 episodes are not a prefix of any 250-ep run.

    Output is segregated to pick_and_place_one_house_mug_v4_pilot.
    """
    seed: int | None = 2027
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceResampleCandidatesTaskSampler,
        pickup_types=["mug"],
        samples_per_house=50,             # <-- changed (was 250)
        house_inds=[1],
        max_allowed_sequential_task_sampler_failures=2000,
        max_allowed_sequential_rollout_failures=2000,
        max_allowed_sequential_irrecoverable_failures=10000,
        max_total_attempts_multiplier=25,
        max_asset_failures=10_000,
        max_robot_placement_attempts=80,
        base_pose_sampling_radius_range=(0.0, 1.2),
        check_robot_placement_visibility=False,
        randomize_textures=False,
        randomize_robot_textures=False,
        randomize_lighting=True,
        randomize_dynamics=True,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_one_house_mug_v4_pilot"

    @property
    def tag(self) -> str:
        return "franka_skin_one_house_mug_fast_pilot"


@register_config("FrankaSkinPickAndPlaceHouse10CupConfig")
class FrankaSkinPickAndPlaceHouse10CupConfig(FrankaSkinPickAndPlaceOneHouseMugFastConfig):
    """Locked-down house_10 cup pick-and-place task for ACT/PACT datagen and eval."""

    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceResampleCandidatesTaskSampler,
        pickup_types=["cup"],
        samples_per_house=250,
        house_inds=[10],
        max_allowed_sequential_task_sampler_failures=2000,
        max_allowed_sequential_rollout_failures=2000,
        max_allowed_sequential_irrecoverable_failures=10000,
        max_total_attempts_multiplier=25,
        max_asset_failures=10_000,
        max_robot_placement_attempts=80,
        base_pose_sampling_radius_range=(0.0, 1.2),
        check_robot_placement_visibility=False,
        randomize_textures=False,
        randomize_robot_textures=False,
        randomize_lighting=True,
        randomize_dynamics=True,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_house10_cup_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_house10_cup"


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


@register_config("FrankaSkinProxNecessitySimplePilotConfig")
class FrankaSkinProxNecessitySimplePilotConfig(FrankaSkinPickAndPlacePilotConfig):
    """Legacy proximity-necessity pilot with the original pilot defaults."""

    filter_for_successful_trajectories: bool = True
    output_dir: Path = ASSETS_DIR / "datagen" / "franka_skin_prox_necessity_pilot_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_prox_necessity_pilot"


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

    num_workers: int = 1
    seed: int | None = 69
    disable_collision_checks: bool = False  # collision-avoiding demos
    # Minimal Gaussian action noise (truncated-Gaussian end-effector noise, std = 10%
    # of each commanded TCP delta, hard-capped at 1 cm / ~0.6 deg, mapped to joints via
    # the Jacobian) — state-coverage so the trained policy makes cleaner motions and so
    # the proximity skin earns its keep when states drift near surfaces.
    robot_config: BaseRobotConfig = FrankaSkinRobotConfig(
        action_noise_config=ActionNoiseConfig(
            enabled=True,
            action_scale_factor=0.1,
            max_tcp_position_noise=0.03,   # "very minimal, ~0.01 m"
            rotation_noise_scale=0.1,
            max_tcp_rotation_noise=0.01,
        )
    )
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
        samples_per_house=25,
        house_inds=[11],               # small pilot — iterate, then scale
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_prox_v1_samples"

    @property
    def tag(self) -> str:
        return "franka_skin_prox_necessity_pilot"


<<<<<<< HEAD
DEEP_CAVITY_PREFIXES: tuple[str, ...] = (
    "cabinet",
    "drawer",
    "fridge",
    "microwave",
    "safe",
    "oven",
    "dishwasher",
)


@register_config("FrankaSkinDeepCavityPickAndPlaceDataGenConfig")
class FrankaSkinDeepCavityPickAndPlaceDataGenConfig(FrankaSkinPickAndPlaceDataGenConfig):
    """Pick-and-place biased to deep/enclosed source surfaces (cabinets, drawers, microwaves).
    Forces the robot to reach horizontally into constrained spaces to maximize proximity utility
    and minimize self-sensing errors."""

    scene_dataset: str = "procthor-objaverse"
    data_split: str = "train"
    num_workers: int = 4
    seed: int | None = np.random.randint(1, 1000000)
    filter_for_successful_trajectories: bool = True
    collision_free_pose_limit: int = 30
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=5,
        house_inds=list(range(1999)),
        max_allowed_sequential_irrecoverable_failures=10000,
        max_total_attempts_multiplier=25,
        max_robot_placement_attempts=80,
        source_surface_types=DEEP_CAVITY_PREFIXES,
        base_pose_sampling_radius_range=(0.4, 0.65),
        robot_object_z_offset=-0.2,
        robot_object_z_offset_random_min=-0.1,
        robot_object_z_offset_random_max=0.1,
        robot_placement_rotation_range_rad=math.radians(15),
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_deep_cavity_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_deep_cavity"


@register_config("FrankaSkinDeepCavityPickAndPlacePilotConfig")
class FrankaSkinDeepCavityPickAndPlacePilotConfig(FrankaSkinDeepCavityPickAndPlaceDataGenConfig):
    """Smaller pilot of the deep-cavity pick-and-place dataset for quick verification."""

    seed: int | None = 7
    num_workers: int = 1
    collision_free_pose_limit: int = 30
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=3,
        house_inds=list(range(200)),
        max_allowed_sequential_irrecoverable_failures=10000,
        max_total_attempts_multiplier=25,
        max_robot_placement_attempts=80,
        source_surface_types=DEEP_CAVITY_PREFIXES,
        base_pose_sampling_radius_range=(0.4, 0.65),
        robot_object_z_offset=-0.2,
        robot_object_z_offset_random_min=-0.1,
        robot_object_z_offset_random_max=0.1,
        robot_placement_rotation_range_rad=math.radians(15),
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_deep_cavity_pilot_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_deep_cavity_pilot"


OPEN_FRONT_CLUTTER_PREFIXES: tuple[str, ...] = (
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
    "side_table",
    "sidetable",
    "tvstand",
    "ottoman",
)

OPEN_FRONT_COMPACT_PICKUP_TYPES: list[str] = [
    "apple",
    "cellphone",
    "creditcard",
    "cup",
    "egg",
    "egg_cracked",
    "fork",
    "keychain",
    "knife",
    "mug",
    "pen",
    "pencil",
    "pepper_shaker",
    "potato",
    "remote",
    "salt_shaker",
    "spoon",
    "tomato",
    "watch",
]


@register_config("FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig")
class FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig(
    FrankaSkinPickAndPlaceDataGenConfig
):
    """Open-front cluttered reach pilot.

    Biases pickup objects to open-front or low-side surfaces where link5/link6
    should get incidental proximity during approach/pregrasp, without forcing
    continuous deep-cavity wall sensing.
    """

    scene_dataset: str = "procthor-objaverse"
    data_split: str = "train"
    seed: int | None = 7
    num_workers: int = 1
    filter_for_successful_trajectories: bool = True
    collision_free_pose_limit: int = 30
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceResampleCandidatesTaskSampler,
        pickup_types=OPEN_FRONT_COMPACT_PICKUP_TYPES,
        samples_per_house=1,
        house_inds=list(range(300)),
        max_allowed_sequential_task_sampler_failures=2000,
        max_allowed_sequential_rollout_failures=2000,
        max_allowed_sequential_irrecoverable_failures=10000,
        max_total_attempts_multiplier=25,
        max_asset_failures=10_000,
        max_robot_placement_attempts=80,
        source_surface_types=OPEN_FRONT_CLUTTER_PREFIXES,
        num_place_receptacles=6,
        episodes_per_receptacle=1,
        check_robot_placement_visibility=False,
        base_pose_sampling_radius_range=(0.55, 0.90),
        robot_object_z_offset=-0.30,
        robot_object_z_offset_random_min=-0.05,
        robot_object_z_offset_random_max=0.05,
        robot_placement_rotation_range_rad=math.radians(20),
        min_object_to_receptacle_dist=0.08,
        max_object_to_receptacle_dist=0.40,
        max_place_receptacle_sampling_attempts=300,
    )
    output_dir: Path = (
        ASSETS_DIR / "datagen" / "pick_and_place_skin_open_front_cluttered_reach_pilot_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_skin_open_front_cluttered_reach_pilot"

#NEW ENVIRONMENTS :
PACT_MEANINGFUL_REACH_PREFIXES: tuple[str, ...] = (
    # Semi-constrained / open-front surfaces where wrist + forearm should pass
    # near useful scene geometry without being permanently inside a deep cavity.
    "shelf",
    "bookshelf",
    "cabinet",
    "drawer",
    "dresser",
    "chestofdrawers",
    "tvstand",
    "sidetable",
    "side_table",
    "sink",

    # Optional low-side clutter surfaces. Keep these if yield is too low,
    # remove them if audits show too much generic low-surface activation.
    "chair",
    "armchair",
    "stool",
    "sofa",
    "ottoman",
)


@register_config("FrankaSkinPACTMeaningfulReachPickAndPlacePilotConfig")
class FrankaSkinPACTMeaningfulReachPickAndPlacePilotConfig(
    FrankaSkinPickAndPlaceDataGenConfig
):
    """PACT meaningful-reach pilot.

    Candidate environment for finding high-value proximity data for PACT.

    The goal is not maximum raw proximity activation. The goal is meaningful,
    non-erroneous activation from distal sensors, especially link5/link6,
    during approach, pregrasp, and grasp-lift.

    This config biases toward semi-constrained open-front / low-side surfaces
    such as shelves, cabinets, dressers, side tables, sinks, and low cluttered
    furniture. These should force the wrist and forearm to travel near nearby
    geometry while avoiding the degenerate case where the robot is constantly
    sensing a wall, table, or its own body.

    Evaluation target:
      - viable trajectory yield
      - link5/link6 activation under 0.20 m
      - nonzero approach/pregrasp activation
      - low suspicious/self-sensing activation
      - phase-aligned signal, not constant background activation
    """

    scene_dataset: str = "procthor-objaverse"
    data_split: str = "train"
    seed: int | None = 7
    num_workers: int = 1
    filter_for_successful_trajectories: bool = True

    # Important: default of 3 can cause placement to give up too early.
    collision_free_pose_limit: int = 30

    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceResampleCandidatesTaskSampler,

        # Compact objects are better here because large objects can dominate
        # proximity readings and make activation less interpretable.
        pickup_types=OPEN_FRONT_COMPACT_PICKUP_TYPES,

        # Pilot scale: use this for environment search first, not final dataset collection.
        samples_per_house=1,
        house_inds=list(range(50)),

        # Keep failure budgets high because this is an environment-search config.
        max_allowed_sequential_task_sampler_failures=2000,
        max_allowed_sequential_rollout_failures=2000,
        max_allowed_sequential_irrecoverable_failures=10000,
        max_total_attempts_multiplier=25,
        max_asset_failures=10_000,
        max_robot_placement_attempts=80,

        # More targeted than OPEN_FRONT_CLUTTER_PREFIXES.
        # Avoids overly broad surfaces that may produce weak or generic activation.
        source_surface_types=PACT_MEANINGFUL_REACH_PREFIXES,

        # Lower difficulty than the previous open-front config.
        # Six receptacles + short object-to-receptacle distance was likely too restrictive.
        num_place_receptacles=1,
        episodes_per_receptacle=1,
        min_object_to_receptacle_dist=0.08,
        max_object_to_receptacle_dist=0.75,
        max_place_receptacle_sampling_attempts=500,

        # Disable visibility check for scripted datagen.
        # The planner uses ground-truth poses; requiring exo_camera_1 visibility
        # rejects many otherwise useful close-proximity placements.
        check_robot_placement_visibility=False,

        # Geometry chosen to encourage bent-elbow, lower-profile reaches.
        # Less aggressive than pure deep-cavity, but much better for link5/link6
        # than the old (0.65, 1.05), z=-0.65 setup.
        base_pose_sampling_radius_range=(0.45, 0.80),
        robot_object_z_offset=-0.25,
        robot_object_z_offset_random_min=-0.05,
        robot_object_z_offset_random_max=0.05,
        robot_placement_rotation_range_rad=math.radians(20),
    )

    output_dir: Path = (
        ASSETS_DIR / "datagen" / "pick_and_place_skin_pact_meaningful_reach_pilot_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_skin_pact_meaningful_reach_pilot"
=======
@register_config("FrankaSkinProxVizSampleConfig")
class FrankaSkinProxVizSampleConfig(FrankaSkinProxNecessityPilotConfig):
    """Small sample collection for VISUALIZING what the proximity skin sees.

    Identical regime to FrankaSkinProxNecessityPilotConfig (same robot, cameras,
    action noise, clutter), but with ``viz_sensor_rgb=True`` so every one of the 29
    SPAD sensors ALSO emits two high-res, directly-viewable videos per trajectory --
    a scene RGB (``episode_XXXX_link{L}_sensor_{S}_viz_rgb.mp4``) and a turbo-colormapped
    depth (``..._viz_depth_turbo.mp4``, near=warm/red, far=cool/blue) -- alongside the
    unchanged native 8x8 proximity depth. Kept tiny on purpose (one house, a few samples)
    because each trajectory writes 29x2 videos. Bump samples_per_house / house_inds once
    you've eyeballed the output.

    The viz uses the SAME camera/fovy as the 8x8 sensor (vertical FOV = 45 deg, focal
    length unchanged); only the pixel resolution differs. viz_sensor_resolution defaults
    to 640x480 -> arrays of shape (480, 640); set width==height for a FOV that exactly
    matches the square 8x8 SPAD. viz_depth_range sets the turbo near/far clip (meters).
    """
    seed: int | None = 2026
    num_workers: int = 1
    viz_sensor_rgb: bool = True
    viz_sensor_resolution: tuple[int, int] = (640, 480)  # (width, height)
    # turbo near/far clip, meters. Narrowed from the full SPAD range (0.05, 4.0) to (0.05, 1.0)
    # after a smoke run: the skin's depth median is ~0.53 m (p25=0.24, p75=1.70), so against a
    # 4 m span nearly everything collapsed into the warm end (uniform red, no contrast). Clipping
    # at 1.0 m spreads turbo across the manipulation-relevant near field (red<->near, blue=>=1 m).
    viz_depth_range: tuple[float, float] = (0.05, 1.0)  # turbo near/far clip, meters
    # SOFTENED relative to FrankaSkinProxNecessityPilotConfig so house 11 actually yields a
    # usable sample set (the strict necessity knobs -- 60 clutter objects packed at 5 cm and a
    # 0-0.45 m stand-off -- starved the place-target sampler and IK, giving ~1 trajectory before
    # the run aborted). Clutter cut 60 -> 20 and dist-to-clutter widened so a receptacle fits;
    # robot stand-off raised to 0.2-0.65 m so grasp/lift IK is reachable. Still low-surface /
    # skin-active (proximity sensors fire on approach), just no longer near-zero yield.
    # CRITICAL FIX: max_allowed_sequential_task_sampler_failures was inheriting its default of 10,
    # so 10 consecutive (expected) sampling misses aborted the whole house even though the sibling
    # irrecoverable cap was 10000. Raised to 300 so the over-generating sampler keeps going.
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        source_surface_types=LOW_SURFACE_PREFIXES,
        check_robot_placement_visibility=False,
        num_added_pickups=20,
        min_reference_to_added_pickup_dist=0.10,
        max_reference_to_added_pickup_dist=0.40,
        max_robot_to_added_pickup_dist=0.6,
        base_pose_sampling_radius_range=(0.2, 0.65),
        robot_placement_rotation_range_rad=0.52,
        samples_per_house=100,           # early-stop target; ~8% rollout yield on house 11
        house_inds=[11],
        # With ~8% rollout yield, runs of 10+ consecutive IK-rollout failures are near-certain;
        # the default rollout cap (10) would abort the house almost immediately. Raised to 300 so
        # the house keeps grinding. multiplier=15 -> up to 1500 attempts (early-stops at 100
        # collected); at ~8% that lands ~100 successes (mostly the few graspable objects in
        # house 11). Single-house was chosen knowingly: low object diversity, ~1-2 day run.
        max_total_attempts_multiplier=15,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "house_11_100_samples"

    @property
    def tag(self) -> str:
        return "franka_skin_prox_viz_sample"


@register_config("FrankaSkinProxOvernightConfig")
class FrankaSkinProxOvernightConfig(FrankaSkinProxNecessityPilotConfig):
    """Multi-house overnight collection in the proximity-NECESSITY regime.

    This is the "safety net" / scale run: same skin-active, low-surface, clutter-packed
    regime as FrankaSkinProxNecessityPilotConfig, but spread across MANY houses with 4
    parallel workers so trajectories accumulate over a long run. Single-house (house 11)
    was intractably slow (~8% yield, ~1-2 days for 100); spreading the work across houses
    gives both throughput AND object/scene diversity.

    Regime is MODERATELY softened from the strict necessity knobs (which starved yield):
    clutter 60 -> 25, packed at 6-35 cm, robot stand-off 0.15-0.55 m. Still low-surface and
    skin-active (the arm operates among surfaces), just no longer near-zero yield. The
    per-house budget is bounded by max_total_attempts_multiplier (samples_per_house * 12),
    so unproductive houses are abandoned and a worker moves on rather than grinding forever.
    Sequential-failure caps are raised above that budget so the multiplier is the binding
    limit, not a premature consecutive-failure abort.
    """

    seed: int | None = 2026
    num_workers: int = 4
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        source_surface_types=LOW_SURFACE_PREFIXES,   # enclosed / recessed targets
        check_robot_placement_visibility=False,
        num_added_pickups=25,                        # meaningful clutter, not yield-killing 60
        min_reference_to_added_pickup_dist=0.06,     # pack close for skin activation
        max_reference_to_added_pickup_dist=0.35,
        max_robot_to_added_pickup_dist=0.55,
        base_pose_sampling_radius_range=(0.15, 0.55),  # close stand-off -> arm among surfaces
        robot_placement_rotation_range_rad=0.52,
        samples_per_house=8,                         # move across houses for diversity
        house_inds=list(range(0, 100)),
        max_total_attempts_multiplier=12,            # <= 96 attempts/house, then move on
        max_allowed_sequential_task_sampler_failures=200,
        max_allowed_sequential_rollout_failures=200,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "overnight_prox_multi_house"

    @property
    def tag(self) -> str:
        return "franka_skin_prox_overnight_multi_house"

    # seed: int | None = 2026
    # num_workers: int = 2
    # task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
    #     task_sampler_class=PickAndPlaceTaskSampler,
    #     pickup_types=PICK_AND_PLACE_OBJECTS,
    #     samples_per_house=4,
    #     house_inds=list(range(1, 11)),
    #     max_allowed_sequential_irrecoverable_failures=10000,
    # )
    # output_dir: Path = ASSETS_DIR / "datagen" / "pick_and_place_skin_pilot_smoke_v1"

>>>>>>> 76f15d6 (fumehood stuff)

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


# --------------------------------------------------------------------------------------- #
# Custom "reach into a drawer/cabinet cavity" proximity-necessity environment.
# Does NOT use predefined houses: a hand-authored cavity scene + a graspable objaverse
# object injected inside it (see molmo_spaces/tasks/cavity_pick_task_sampler.py and
# molmo_spaces/data_generation/custom_scenes/cabinet_cavity.xml). Reuses the stock
# franka_skin robot + skin cameras + proximity recording + privileged pick planner.
# --------------------------------------------------------------------------------------- #
from molmo_spaces.molmo_spaces_constants import ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR
from molmo_spaces.tasks.cavity_pick_task_sampler import (
    CavityPickTaskSampler,
    CavityPickTaskSamplerV2,
    RealHousePickTaskSampler,
    RealTablePickTaskSampler,
    ShelfReachPickTaskSampler,
)

_CUSTOM_SCENES = (
    ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / "molmo_spaces" / "data_generation" / "custom_scenes"
)
_CAVITY_XML = str(_CUSTOM_SCENES / "cabinet_cavity.xml")
_CAVITY_XML_V2 = str(_CUSTOM_SCENES / "cabinet_cavity_v2.xml")
_SHELF_XML = str(_CUSTOM_SCENES / "shelf_reach.xml")
_CLUTTER_XML = str(_CUSTOM_SCENES / "clutter_reach.xml")


@register_config("FrankaSkinCabinetCavitySmokeConfig")
class FrankaSkinCabinetCavitySmokeConfig(PickBaseConfig):
    """Tiny smoke test: 1 cavity scene, pick once. Run first to confirm the pipeline."""

    scene_dataset: str = "user"
    data_split: str = "train"
    num_workers: int = 1
    seed: int | None = 2026
    filter_for_successful_trajectories: bool = True
    disable_collision_checks: bool = False
    robot_config: BaseRobotConfig = FrankaSkinRobotConfig()
    camera_config: FrankaSkinCameraSystem = FrankaSkinCameraSystem()
    task_type: str = "pick"
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=CavityPickTaskSampler,
        scene_xml_paths=[_CAVITY_XML],
        house_inds=[0],
        samples_per_house=1,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=10,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "cabinet_cavity_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_cabinet_cavity_smoke"


@register_config("FrankaSkinCabinetCavityConfig")
class FrankaSkinCabinetCavityConfig(FrankaSkinCabinetCavitySmokeConfig):
    """Full collection: 8 cavity instances (object variety) x N samples each."""

    num_workers: int = 2
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=CavityPickTaskSampler,
        scene_xml_paths=[_CAVITY_XML] * 8,   # 8 "houses" -> 8 different objaverse objects
        house_inds=list(range(8)),
        samples_per_house=15,                # ~120 target
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "cabinet_cavity_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_cabinet_cavity"


@register_config("FrankaSkinCabinetCavityV2Config")
class FrankaSkinCabinetCavityV2Config(FrankaSkinCabinetCavitySmokeConfig):
    """v2 collection: wider cavity + flat/oversized objects filtered out + more object variety.
    16 distinct objects (filtered for graspable-in-cavity shapes) x 10 samples -> ~160 target."""

    num_workers: int = 3
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=CavityPickTaskSamplerV2,
        scene_xml_paths=[_CAVITY_XML_V2] * 16,
        house_inds=list(range(16)),
        samples_per_house=10,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "cabinet_cavity_v2"

    @property
    def tag(self) -> str:
        return "franka_skin_cabinet_cavity_v2"


@register_config("FrankaSkinShelfReachSmokeConfig")
class FrankaSkinShelfReachSmokeConfig(FrankaSkinCabinetCavitySmokeConfig):
    """Smoke test for the reach-into-a-shelf cupboard (all 29 sensors active >50% of the time).

    disable_collision_checks=True: the cupboard is deliberately snug (so every sensor sees a wall
    >50% of the time), which is too tight for the collision-checked planner to assume the
    pregrasp/lift poses. Turning collision checks off lets the privileged demo reach in; the
    proximity readings are real (the point of the dataset) and the arm only lightly grazes walls
    at the extremes. Use the looser collision-free variant if physically-clean demos matter more.
    """

    disable_collision_checks: bool = False  # only the floor is physical (walls are ghost), planner handles it
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=ShelfReachPickTaskSampler,
        scene_xml_paths=[_SHELF_XML],
        house_inds=[0],
        samples_per_house=1,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=15,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "shelf_reach_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_shelf_reach_smoke"


@register_config("FrankaSkinShelfReachConfig")
class FrankaSkinShelfReachConfig(FrankaSkinShelfReachSmokeConfig):
    """Full reach-into-shelf collection: 16 distinct objects x 10 samples (~160 target)."""

    num_workers: int = 3
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=ShelfReachPickTaskSampler,
        scene_xml_paths=[_SHELF_XML] * 16,
        house_inds=list(range(16)),
        samples_per_house=10,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "shelf_reach_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_shelf_reach"


@register_config("FrankaSkinClutterReachSmokeConfig")
class FrankaSkinClutterReachSmokeConfig(FrankaSkinCabinetCavitySmokeConfig):
    """Smoke test for 'clutter everywhere around the robot': a ghost-clutter field surrounds the
    arm so ~all 29 sensors are active (28/29 measured; 1 self-occluded). Arm passes through the
    ghost clutter, so the planner picks the target off the physical pad unobstructed."""

    disable_collision_checks: bool = False  # only floor + obj_pad are physical
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=ShelfReachPickTaskSampler,
        scene_xml_paths=[_CLUTTER_XML],
        house_inds=[0],
        samples_per_house=1,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=15,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "clutter_reach_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_clutter_reach_smoke"


@register_config("FrankaSkinClutterReachConfig")
class FrankaSkinClutterReachConfig(FrankaSkinClutterReachSmokeConfig):
    """Full 'clutter everywhere' collection: 24 distinct objects x 25 samples (~600 target),
    8 parallel workers. Each object gets 25 trajectories with randomized object pose + robot base
    jitter (the ghost-clutter field is fixed; randomize it per-house in add_auxiliary_objects for
    more clutter diversity later)."""

    num_workers: int = 8
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=ShelfReachPickTaskSampler,
        scene_xml_paths=[_CLUTTER_XML] * 24,
        house_inds=list(range(24)),
        samples_per_house=25,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "clutter_reach_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_clutter_reach"


@register_config("FrankaSkinClutterReachSmallConfig")
class FrankaSkinClutterReachSmallConfig(FrankaSkinClutterReachSmokeConfig):
    """Smaller parallel clutter collection (quick set / can serve as val): 12 objects x 8 samples
    (~96 target), 3 workers, separate output dir. Distinct seed -> samples different objects/poses
    than the main 24x25 run. Low samples_per_house -> frequent saves, low crash-loss risk. Runs
    concurrently with FrankaSkinClutterReachConfig (total 11 workers on 48 cores)."""

    seed: int | None = 7
    num_workers: int = 3
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=ShelfReachPickTaskSampler,
        scene_xml_paths=[_CLUTTER_XML] * 12,
        house_inds=list(range(12)),
        samples_per_house=8,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "clutter_reach_small"

    @property
    def tag(self) -> str:
        return "franka_skin_clutter_reach_small"


_PILLAR_XMLS = [str(_CUSTOM_SCENES / f"pillar_avoid_{v}.xml") for v in range(12)]
_TABLE_REAL_XML = str(_CUSTOM_SCENES / "table_shelf_real.xml")
_REAL_ENV_XMLS = [
    str(_CUSTOM_SCENES / f"real_env_{v}_{n}.xml")
    for v, n in enumerate(["table_shelf", "kitchen", "shelfbay", "desk_hutch", "sink", "corner"])
]


@register_config("FrankaSkinPillarAvoidSmokeConfig")
class FrankaSkinPillarAvoidSmokeConfig(FrankaSkinCabinetCavitySmokeConfig):
    """Smoke test for the obstacle-AVOIDANCE environment: tall PHYSICAL pillars around the
    robot/corridor. Collision checks stay ON so the privileged planner detours around the
    pillars — that avoidance, with the skin recording the obstacles, is the training signal."""

    disable_collision_checks: bool = False
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=ShelfReachPickTaskSampler,
        scene_xml_paths=[_PILLAR_XMLS[0]],
        house_inds=[0],
        samples_per_house=1,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=15,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pillar_avoid_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_pillar_avoid_smoke"


@register_config("FrankaSkinPillarAvoidConfig")
class FrankaSkinPillarAvoidConfig(FrankaSkinPillarAvoidSmokeConfig):
    """Obstacle-avoidance collection: 12 pillar-layout variants x 8 samples (~96 target),
    one objaverse object per variant, frequent saves. Runs alongside the clutter collections."""

    num_workers: int = 2
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=ShelfReachPickTaskSampler,
        scene_xml_paths=_PILLAR_XMLS,
        house_inds=list(range(12)),
        samples_per_house=8,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "pillar_avoid_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pillar_avoid"


@register_config("FrankaSkinRealTableSmokeConfig")
class FrankaSkinRealTableSmokeConfig(FrankaSkinCabinetCavitySmokeConfig):
    """Smoke test for the REALISTIC tabletop env (advisor feedback: no floating geometry).
    Real wooden table + shelf + walls; PHYSICAL objaverse clutter gravity-settled on the table
    around the target; collision-aware planning ON -> demos genuinely avoid the clutter."""

    disable_collision_checks: bool = False
    # LOW robot mount (advisor/user feedback): 0.35 m platform instead of the default 0.58 m,
    # so the arm works AT table height and threads THROUGH the clutter rather than reaching
    # down from a tower — makes the skin informative and looks like a real lab mount.
    robot_config: BaseRobotConfig = FrankaSkinRobotConfig(base_size=[0.4, 0.4, 0.35])
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=RealTablePickTaskSampler,
        scene_xml_paths=[_TABLE_REAL_XML],
        house_inds=[0],
        samples_per_house=1,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=15,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "real_table_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_real_table_smoke"


@register_config("FrankaSkinRealTableConfig")
class FrankaSkinRealTableConfig(FrankaSkinRealTableSmokeConfig):
    """Realistic-environment collection across 6 scene variants (work table+shelf, kitchen
    counter+overhead cabinets, bookshelf bay, office desk+hutch, kitchen sink, corner tables) x 4
    target objects each = 24 houses x 10 samples (~240 target). Per-house clutter draws (different
    real objects each house, gravity-settled); per-scene target spot via the 'target_spot' site."""

    num_workers: int = 6
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=RealTablePickTaskSampler,
        scene_xml_paths=_REAL_ENV_XMLS * 4,
        house_inds=list(range(24)),
        samples_per_house=10,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "real_table_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_real_table"


@register_config("FrankaSkinRealHouseSmokeConfig")
class FrankaSkinRealHouseSmokeConfig(FrankaSkinRealTableSmokeConfig):
    """Smoke: controlled tall-clutter pick INSIDE a real ProcTHOR house (maximal realism)."""

    scene_dataset: str = "procthor-objaverse"
    data_split: str = "train"
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=RealHousePickTaskSampler,
        house_inds=[33],
        samples_per_house=1,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=15,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "real_house_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_real_house_smoke"


@register_config("FrankaSkinRealHouseConfig")
class FrankaSkinRealHouseConfig(FrankaSkinRealHouseSmokeConfig):
    """Realistic in-house collection: known-good ProcTHOR houses, one work surface each,
    LOW robot + tall real clutter, collision-aware avoidance demos."""

    num_workers: int = 6
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=RealHousePickTaskSampler,
        house_inds=[11, 12, 13, 14, 16, 17, 19, 20, 23, 25, 31, 33, 34, 36, 37, 46],
        samples_per_house=10,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "real_house_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_real_house"


# --------------------------------------------------------------------------------------- #
# Parameterized enclosure-reach generator (advisor spec; ENCLOSURE_DATAGEN_DESIGN.md):
# per-episode θ over aperture clearance / depth / target pose / protrusion / lighting via
# mocap slabs; observation-realizable scripted expert (speed∝clearance, detection-gated
# deflection, clean aborts); θ + cam-visibility label + behavior class logged to obs_scene.
# --------------------------------------------------------------------------------------- #
from molmo_spaces.tasks.enclosure_reach import (
    EnclosureExpertPolicyConfig,
    EnclosureReachSampler,
    EnclosureReachTask,
    FumehoodExpertPolicyConfig,
    FumehoodSampler,
    TourFumehoodSampler,
    BigFumehoodPickSampler,
    CubbyExpertPolicyConfig,
    CubbyOverreachSampler,
    PanelSlalomSampler,
)
from molmo_spaces.tasks.house_embed import (
    HouseFumehoodSampler,
    HousePanelSlalomSampler,
    HouseCubbyOverreachSampler,
)
from molmo_spaces.configs.task_configs import PickTaskConfig

_ENCLOSURE_XML = str(_CUSTOM_SCENES / "enclosure_param.xml")


@register_config("FrankaSkinEnclosureSmokeConfig")
class FrankaSkinEnclosureSmokeConfig(FrankaSkinCabinetCavitySmokeConfig):
    """Smoke: 1 object x 4 episodes of the parameterized enclosure-reach generator."""

    robot_config: BaseRobotConfig = FrankaSkinRobotConfig(base_size=[0.4, 0.4, 0.35])
    policy_config: BasePolicyConfig = EnclosureExpertPolicyConfig()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=EnclosureReachTask)
    filter_for_successful_trajectories: bool = False  # smoke: save failures too (videos for debugging)
    task_horizon: int | None = 900
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=EnclosureReachSampler,
        scene_xml_paths=[_ENCLOSURE_XML] * 8,
        house_inds=list(range(8)),
        samples_per_house=3,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=10,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    num_workers: int = 4
    output_dir: Path = ASSETS_DIR / "datagen" / "enclosure_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_enclosure_smoke"


@register_config("FrankaSkinEnclosureGenConfig")
class FrankaSkinEnclosureGenConfig(FrankaSkinEnclosureSmokeConfig):
    """Full enclosure-reach collection: 24 target objects x per-episode θ draws.
    samples_per_house x houses sets the episode budget (~2-5k for the paper dataset)."""

    filter_for_successful_trajectories: bool = True
    num_workers: int = 6
    policy_config: BasePolicyConfig = EnclosureExpertPolicyConfig(max_retries=2)
    task_horizon: int | None = 450
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=EnclosureReachSampler,
        scene_xml_paths=[_ENCLOSURE_XML] * 360,
        house_inds=list(range(360)),
        samples_per_house=6,            # 360 x 6 ≈ 2160; small houses -> frequent saves
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=6,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "enclosure_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_enclosure_gen"

@register_config("FrankaSkinFumehoodSmokeConfig")
class FrankaSkinFumehoodSmokeConfig(FrankaSkinEnclosureSmokeConfig):
    """Fumehood finetune batch: VISIBLE-obstacle whole-arm-clearance task (no camera occlusion).
    Glass sash overhead, jambs beside, upright obstacles inside. 8 objects x 3 episodes."""

    policy_config: BasePolicyConfig = FumehoodExpertPolicyConfig(max_retries=2)
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=FumehoodSampler,
        scene_xml_paths=[str(_CUSTOM_SCENES / "fumehood.xml")] * 8,
        house_inds=list(range(8)),
        samples_per_house=3,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=10,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "fumehood_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_fumehood_smoke"

@register_config("FrankaSkinPanelSlalomSmokeConfig")
class FrankaSkinPanelSlalomSmokeConfig(FrankaSkinFumehoodSmokeConfig):
    """Photo-1 recreation: panel slalom on a table (visible obstacles, whole-arm clearance)."""

    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PanelSlalomSampler,
        scene_xml_paths=[str(_CUSTOM_SCENES / "panel_slalom.xml")] * 8,
        house_inds=list(range(8)),
        samples_per_house=3,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=10,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "panel_slalom_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_panel_slalom_smoke"


@register_config("FrankaSkinCubbySmokeConfig")
class FrankaSkinCubbySmokeConfig(FrankaSkinFumehoodSmokeConfig):
    """Photo-2 recreation: over-the-wall reach into an open-top cubby."""

    policy_config: BasePolicyConfig = CubbyExpertPolicyConfig(max_retries=2)
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=CubbyOverreachSampler,
        scene_xml_paths=[str(_CUSTOM_SCENES / "cubby_overreach.xml")] * 8,
        house_inds=list(range(8)),
        samples_per_house=3,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=10,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "cubby_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_cubby_smoke"


# --------------------------------------------------------------------------------------- #
# HOUSE-EMBEDDED variants: same parameterized task geometry, dropped into REAL ProcTHOR
# rooms (advisor: "robot in a house"). The sampler discovers an open floor spot whose
# robot-mounted exo cam faces room interior (no occlusion) and replays the task furniture +
# mocap slabs there; the expert is embed-aware (task-local poses -> world). No scene_xml_paths
# (the procthor dataset mapping supplies house XMLs via house_inds).
# --------------------------------------------------------------------------------------- #
def _house_sampler_cfg(cls, n_houses: int, samples: int) -> PickTaskSamplerConfig:
    return PickTaskSamplerConfig(
        task_sampler_class=cls,
        house_inds=list(range(n_houses)),
        samples_per_house=samples,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=12,
        max_allowed_sequential_task_sampler_failures=400,
        max_allowed_sequential_rollout_failures=400,
        max_allowed_sequential_irrecoverable_failures=10000,
    )


@register_config("FrankaSkinHouseFumehoodSmokeConfig")
class FrankaSkinHouseFumehoodSmokeConfig(FrankaSkinFumehoodSmokeConfig):
    """Fumehood task embedded in real ProcTHOR rooms."""

    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    task_sampler_config: PickTaskSamplerConfig = _house_sampler_cfg(HouseFumehoodSampler, 12, 2)
    output_dir: Path = ASSETS_DIR / "datagen" / "house_fumehood_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_house_fumehood_smoke"


@register_config("FrankaSkinHousePanelSmokeConfig")
class FrankaSkinHousePanelSmokeConfig(FrankaSkinPanelSlalomSmokeConfig):
    """Panel-slalom task embedded in real ProcTHOR rooms."""

    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    task_sampler_config: PickTaskSamplerConfig = _house_sampler_cfg(HousePanelSlalomSampler, 12, 2)
    output_dir: Path = ASSETS_DIR / "datagen" / "house_panel_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_house_panel_smoke"


@register_config("FrankaSkinHouseCubbySmokeConfig")
class FrankaSkinHouseCubbySmokeConfig(FrankaSkinCubbySmokeConfig):
    """Over-the-wall cubby task embedded in real ProcTHOR rooms."""

    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    task_sampler_config: PickTaskSamplerConfig = _house_sampler_cfg(HouseCubbyOverreachSampler, 12, 2)
    output_dir: Path = ASSETS_DIR / "datagen" / "house_cubby_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_house_cubby_smoke"


# --------------------------------------------------------------------------------------- #
# HYBRID 40-SENSOR SKIN datagen: same fumehood task, but the gentact hybrid skin robot +
# 40-sensor camera system (links 1-6) instead of the 29-sensor skin. Proves the new skin
# flows through the full pipeline (40 obs/proximity keys, near-field captured by the
# model_hybrid znear fix). Pair-checks with scripts/verify_hybrid_skin_sensors.py.
# --------------------------------------------------------------------------------------- #
@register_config("FrankaSkinHybridFumehoodSmokeConfig")
class FrankaSkinHybridFumehoodSmokeConfig(FrankaSkinFumehoodSmokeConfig):
    """Fumehood task with the 40-sensor HYBRID skin (model_hybrid.xml).
    Saves each sensor's 256x256 RGB view alongside the 8x8 depth (viz_sensor_rgb)."""

    robot_config: BaseRobotConfig = FrankaSkinHybridRobotConfig(base_size=[0.4, 0.4, 0.35])
    camera_config: FrankaSkinHybridCameraSystem = FrankaSkinHybridCameraSystem()
    viz_sensor_rgb: bool = True
    viz_sensor_resolution: tuple[int, int] = (256, 256)
    output_dir: Path = ASSETS_DIR / "datagen" / "hybrid_fumehood_smoke"

    @property
    def tag(self) -> str:
        return "franka_skin_hybrid_fumehood_smoke"


@register_config("FrankaSkinHybridPnP5Config")
class FrankaSkinHybridPnP5Config(FrankaSkinHybridFumehoodSmokeConfig):
    """Quick 6-episode (2 houses x 3) hybrid-skin pick(+extract) collection in the fumehood.
    Proven insertion frame (pedestal 0.35, bench 0.72, mouth x=0.58 — the frame with verified
    physical grasps) + exactly 6 videos/episode (exo/wrist RGB+depth + 2 sensor mosaics) +
    40-sensor depth and 256x256 sensor RGB in the h5. NOTE: tour-geometry (floor robot) datagen
    needs elbow-aware paths — the straight-line expert stalls on the bench lip there."""

    # Bench-height mount (pedestal 0.35): represents the FR3 bolted at the lab-bench height next
    # to the 0.72 fume hood -- the realistic, physically-correct setup, and the only height that
    # has reached a grasp. A floor-mounted arm cannot work a waist-high hood.
    robot_config: BaseRobotConfig = FrankaSkinHybridRobotConfig(base_size=[0.4, 0.4, 0.35])
    # PROVEN GRASP MACHINERY: the scripted EnclosureExpertPolicy (inherited via the smoke chain)
    # plans a blind pinch and kept bulldozing the object (0% over many rounds). The cabinet-cavity
    # pipeline -- same custom-scene sampler lineage -- grasps reliably because it uses
    # PickPlannerPolicy, which executes annotated grasp poses from each object's grasp file.
    # BigFumehoodPickSampler's pool is restricted to grasp-file-validated mugs/cups/produce.
    policy_config: BasePolicyConfig = PickPlannerPolicyConfig()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=BigFumehoodPickSampler,   # big sweep-scale door + clean pick -> grasps
        scene_xml_paths=[str(_CUSTOM_SCENES / "fumehood.xml")] * 2,
        house_inds=list(range(1)),
        samples_per_house=100,
        # NO added pickupables: they spawn on the far-away staging floor (not in the hood) and
        # the sampler RETARGETS the pick to one of them (observed: robot reached for 'Pan_9'
        # parked at the staging area). Object variety comes from the cavity pool (24 cups/mugs).
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=10,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_object_z_offset_random_min=-np.random.uniform(0.0, 1.0),
        robot_object_z_offset_random_max=np.random.uniform(0.0, 1.0),
        robot_placement_rotation_range_rad=0.52,
        randomize_textures=True,
        randomize_lighting=True,
        #randomize_textures_all = True,
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "hybrid_pnp5"

    @property
    def tag(self) -> str:
        return "franka_skin_hybrid_pnp5"



@register_config("FrankaSkinHybridPnP5MassConfig")
class FrankaSkinHybridPnP5MassConfig(FrankaSkinHybridPnP5Config):
    """Mass collection: 24 houses (one grasp-validated object each — mugs/cups first, then
    produce; pool wraps via house_index % pool) x 10 samples = 240 episodes. Walled fumehood
    room keeps all 40 skin sensors returning. Launch only after the Check config passes."""

    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=BigFumehoodPickSampler,
        scene_xml_paths=[str(_CUSTOM_SCENES / "fumehood.xml")] * 24,
        house_inds=list(range(24)),
        samples_per_house=10,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=10,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_placement_rotation_range_rad=0.52,
        randomize_textures=True,
        randomize_lighting=True,
    )
    num_workers: int = 8
    output_dir: Path = ASSETS_DIR / "datagen" / "hybrid_pnp5_mass"

    @property
    def tag(self) -> str:
        return "franka_skin_hybrid_pnp5_mass"
