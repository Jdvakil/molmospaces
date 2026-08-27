import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom

log = logging.getLogger(__name__)


class PickTask(BaseMujocoTask):
    """Pick task implementation."""

    def get_task_description(self) -> str:
        pickup_obj_name = self.config.task_config.referral_expressions["pickup_obj_name"]
        return f"Pick up the {pickup_obj_name}"

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        """Return task objects for pick task."""
        task_objects = super().get_task_objects(batch_index)
        task_config = self.config.task_config

        self.deduplicate_task_objects_name(
            task_config, "pickup_obj_name", task_objects, "pickup_obj"
        )
        return task_objects

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        """Create a sensor suite from configuration using the centralized get_core_sensors function."""
        from molmo_spaces.env.sensors import get_core_sensors

        sensors = get_core_sensors(config)
        return SensorSuite(sensors)

    def judge_success(self) -> bool:
        """Judge if the task was successful (for data generation)."""

        if self.config.task_type == "pick":
            return self.get_info()[0]["success"]
        else:
            raise ValueError(f"Invalid action_type {self.config.task_type}")

    def get_reward(self) -> np.ndarray:
        """Calculate reward for each environment in the batch."""
        rewards = np.zeros(self._env.n_batch)

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]

            # Get pickup object using Object class for proper positioning
            pickup_obj = MlSpacesObject(
                data=data, object_name=self.config.task_config.pickup_obj_name
            )

            # reward is height above starting positions
            # consider judge_success threshold when chaninging this
            lift_height = pickup_obj.position[2] - self.config.task_config.pickup_obj_start_pose[2]
            pickup_obj_supporting_geom = get_supporting_geom(data, pickup_obj.body_id)
            robot_geoms = descendant_geoms(
                self.env._mj_model, self.env.current_robot.robot_view.base.root_body_id
            )
            object_lifted = (
                pickup_obj_supporting_geom is None or pickup_obj_supporting_geom in robot_geoms
            )
            reward = int(object_lifted) * lift_height
            reward = np.clip(reward, 0.0, 1000.0)
            rewards[i] = reward

        return rewards

    def is_terminal(self) -> np.ndarray:
        """Standard terminal logic, plus the opt-in strict-safety criterion: terminate the
        episode the moment the arm first penetrates an obstacle body, when
        ``config.end_on_collision`` is enabled. Combined with the success demotion in
        ``get_info``, a single collision becomes an immediate failure and the rollout loop
        (``while not task.is_done()``) moves on to the next episode. Off by default, so
        ordinary datagen / non-strict eval is unchanged."""
        terminal = super().is_terminal()
        if getattr(self.config, "end_on_collision", False) and getattr(
            self, "_obstacle_collision_occurred", False
        ):
            terminal[0] = True
        return terminal

    def get_info(self) -> list[dict[str, Any]]:
        """Get additional metrics for each environment."""
        metrics = []

        # Per-episode arm<->obstacle collision diagnostic (see _accumulate_obstacle_diag).
        self._reset_obstacle_diag_if_new_episode()

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]

            # Get pickup object using Object class for proper positioning
            pickup_obj = MlSpacesObject(
                data=data, object_name=self.config.task_config.pickup_obj_name
            )

            place_target_pos = self.config.task_config.pickup_obj_goal_pose[:3]
            place_target_quat = self.config.task_config.pickup_obj_goal_pose[3:7]

            # Calculate errors
            pos_error = np.linalg.norm(pickup_obj.position - place_target_pos)
            pickup_rot = R.from_quat(pickup_obj.quat, scalar_first=True)
            target_rot = R.from_quat(place_target_quat, scalar_first=True)
            rot_error = (pickup_rot.inv() * target_rot).magnitude()

            # Would like to cache this, but no easy way atm
            lift_height = pickup_obj.position[2] - self.config.task_config.pickup_obj_start_pose[2]

            # Option 1: go via descendant geoms and check if all contacts are with robot geoms (didn't work)
            # obj_geoms = get_supporting_geom(data, pickup_obj.body_id)
            # robot_geoms = descendant_geoms(self.env._mj_model, self.env.current_robot.robot_view.base.root_body_id)
            # gripper_root_body_id = self.env.current_robot.robot_view.get_gripper("gripper").root_body_id
            # gripper_geoms = descendant_geoms(self.env._mj_model, gripper_root_body_id)
            # for c in data.contact:
            #     if (c.geom[0] in obj_geoms) ^ (c.geom[1] in obj_geoms):
            #         other_geom_id = c.geom[1] if c.geom[0] in obj_geoms else c.geom[0]
            #         if other_geom_id in robot_geoms:
            #             only_robot_collision = True
            #         else:
            #             only_robot_collision = False
            #             break

            # Option 2: go via root body and check if all contacts are with robot geoms
            # Check if object collides only with robot geoms
            robot_collision = False
            non_robot_collision = False
            for c in data.contact:
                root_body1 = data.model.body_rootid[data.model.geom_bodyid[c.geom1]]
                root_body2 = data.model.body_rootid[data.model.geom_bodyid[c.geom2]]
                if (root_body1 == pickup_obj.body_id) ^ (root_body2 == pickup_obj.body_id):
                    other_root_body = root_body1 if root_body1 != pickup_obj.body_id else root_body2
                    if other_root_body == self.env.current_robot.robot_view.base.root_body_id:
                        robot_collision = True
                    else:
                        non_robot_collision = True
                        break  # no need to keep checking if we already know there's a non-robot collision

            only_robot_collision = robot_collision and not non_robot_collision

            # Diagnostic only: count penetrating arm<->obstacle contacts this step
            # (env index 0; single-env eval). Never affects success / reward.
            if i == 0:
                self._accumulate_obstacle_diag(data, pickup_obj)

            # Success check
            success = (
                only_robot_collision and lift_height >= self.config.task_config.succ_pos_threshold
                # and rot_error < self.config.task_config.succ_rot_threshold
            )
            # Strict-safety criterion (opt-in via config.end_on_collision): any arm<->obstacle
            # penetration this episode demotes it to a FAILURE; PickTask.is_terminal() also ends
            # the episode on the first contact. Single-env (i==0) only; default off leaves
            # success / reward byte-for-byte as in collection.
            if (
                i == 0
                and getattr(self.config, "end_on_collision", False)
                and getattr(self, "_obstacle_collision_occurred", False)
            ):
                success = False

            metrics.append(
                {
                    "position_error": pos_error,
                    "rotation_error": rot_error,
                    "success": success,
                    "episode_step": self.episode_step_count,
                }
            )

        if metrics:
            self._maybe_log_obstacle_diag(bool(metrics[0]["success"]))

        return metrics

    # ------------------------------------------------------------------
    # Arm <-> obstacle collision diagnostic (additive; does NOT change success,
    # reward, or saved trajectory data). Emits one INFO line per episode:
    #   [ObstacleDiag] success=False obstacle_contact_steps=37/200 peak=4 first_contact_step=61
    # This is the safety metric the proximity-sensing policy is meant to beat: a
    # vision-only policy wedges the arm into the cavity / hazard bar (high
    # contact_steps), while a proximity policy should keep the arm clear (low / zero).
    # Penetrating (dist<=0) contacts only, excluding robot self-collision, the grasped
    # pickup object, and the floor.
    # ------------------------------------------------------------------
    def _reset_obstacle_diag_if_new_episode(self) -> None:
        if self.episode_step_count == 0 or not hasattr(self, "_obstacle_diag"):
            self._obstacle_diag: dict[int, int] = {}
            # Per-step obstacle BODY NAMES (same steps as _obstacle_diag). Lets consumers
            # split "rammed the hazard bar" from "brushed the cavity wall" — the counter
            # alone cannot (STATUS.md §7 item 1).
            self._obstacle_diag_bodies: dict[int, list[str]] = {}
            self._obstacle_diag_logged = False
            # Sticky flag: flips True the first step the arm penetrates any obstacle body.
            # Read by is_terminal() + get_info() for the opt-in strict-safety criterion.
            self._obstacle_collision_occurred = False

    def _accumulate_obstacle_diag(self, data, pickup_obj) -> None:
        try:
            robot_root = self.env.current_robot.robot_view.base.root_body_id
            model = data.model
            others = set()
            names: set[str] = set()
            for c in data.contact:
                if c.dist > 0:  # only actual penetrating contacts
                    continue
                r1 = model.body_rootid[model.geom_bodyid[c.geom1]]
                r2 = model.body_rootid[model.geom_bodyid[c.geom2]]
                is_r1 = r1 == robot_root
                is_r2 = r2 == robot_root
                if is_r1 == is_r2:  # robot self-collision or env<->env
                    continue
                other = int(r2 if is_r1 else r1)
                other_geom = int(c.geom2 if is_r1 else c.geom1)
                if other == pickup_obj.body_id:  # the cup we are grasping
                    continue
                body_name = model.body(other).name
                if "floor" in body_name.lower():
                    continue
                # Count DISTINCT obstacle bodies (cavity wall / shelf / hazard bar /
                # fumehood), not raw geom-pairs, so `peak` reads as "# obstacle bodies
                # the arm is wedged against this step".
                others.add(other)
                # Name what was touched. Static scene geoms hang off the world body, so
                # name those by GEOM instead ("world" alone cannot separate the bench
                # from the hood shell).
                if body_name in ("", "world"):
                    body_name = model.geom(other_geom).name or "world"
                names.add(body_name)
            self._obstacle_diag[self.episode_step_count] = len(others)
            if names:
                self._obstacle_diag_bodies[self.episode_step_count] = sorted(names)
            if len(others) > 0:
                self._obstacle_collision_occurred = True
        except Exception as e:  # pragma: no cover - metric must never break a rollout
            log.debug(f"[ObstacleDiag] accumulate failed: {e}")

    def _maybe_log_obstacle_diag(self, success: bool) -> None:
        if getattr(self, "_obstacle_diag_logged", True):
            return
        try:
            ending = bool(self.is_timed_out().any() or self.is_terminal().any())
        except Exception:
            ending = False
        if not ending:
            return
        counts = getattr(self, "_obstacle_diag", {})
        T = max(len(counts), 1)
        contact_steps = sum(1 for v in counts.values() if v > 0)
        peak = max(counts.values()) if counts else 0
        first = next((s for s in sorted(counts) if counts[s] > 0), None)
        self._obstacle_diag_logged = True
        by_body: dict[str, int] = {}
        for step_names in getattr(self, "_obstacle_diag_bodies", {}).values():
            for n in step_names:
                by_body[n] = by_body.get(n, 0) + 1
        body_str = (
            " bodies=" + ",".join(f"{n}:{c}" for n, c in sorted(by_body.items()))
            if by_body else ""
        )
        log.info(
            "[ObstacleDiag] success=%s obstacle_contact_steps=%d/%d peak=%d first_contact_step=%s%s",
            success, contact_steps, T, peak, str(first), body_str,
        )

    def get_obs_scene(self):
        """
        This is for observations that are constant over all time steps of an env.
        """
        obs_scene = super().get_obs_scene()
        text = self.config.task_type + " " + self.config.task_config.pickup_obj_name
        obs_extra = dict(text=text, object_name=self.config.task_config.pickup_obj_name)
        obs_scene.update(obs_extra)

        return obs_scene
