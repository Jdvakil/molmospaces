"""Manifest-driven rollout runner for the hybrid-obstacle collection.

This is a *dedicated* runner, not a rewrite. ``ParallelRolloutRunner`` keeps its
existing behavior for every other config; nothing here changes the default path.
The differences that matter:

* the unit of work is a committed manifest row, not a house;
* the parent knows the complete candidate-ID set before any worker launches;
* each row is claimed atomically, exactly once, and finalised with an explicit
  terminal outcome that survives interruption;
* the row's RNG contract is installed before the worker takes any task-level
  random draw;
* hazard presence comes from the row, never from a runtime Bernoulli;
* episode identity is ``episode_id + manifest_row_sha256`` and is written into
  the H5, so identity no longer depends on file layout;
* worker ID and house alias are recorded as descriptive metadata only and can
  never influence a row's scientific content;
* row lifecycle is reported through the existing worker-completeness mechanism,
  so a lost worker is still fail-loud.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from molmo_spaces.data_generation.episode_manifest import (
    MANIFEST_VERSION,
    OUTCOME_INFRASTRUCTURE_FAILURE,
    OUTCOME_SAMPLING_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_TASK_FAILURE,
    ManifestError,
    canonical_json,
    install_row_seed_contract,
    load_manifest,
    reset_episode_scoped_sampler_state,
    sha256_payload,
    validate_smoke_subset,
)
from molmo_spaces.data_generation.pipeline import (
    ParallelRolloutRunner,
    cleanup_episode_resources,
    setup_policy,
    mp_context,
)
from molmo_spaces.data_generation.row_ledger import (
    DuplicateClaimError,
    RowLedger,
)
from molmo_spaces.data_generation.worker_completeness import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SHUTDOWN,
    WorkerCompletenessError,
    WorkerRegistry,
    WorkerReport,
    build_final_summary,
    write_summary_atomically,
)
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.mp_logging import get_worker_logger
from molmo_spaces.utils.save_utils import prepare_episode_for_saving, save_trajectories

# Bar bodies are parked at z = -2.0 when no hazard is present, so any bar above
# this threshold means a hazard was actually compiled into the scene.
_PARKED_Z_THRESHOLD = -1.0
_PROTR_BODIES = ("protr_s", "protr_m", "protr_l")


class ManifestRowFailure(RuntimeError):
    """A row could not be completed. Carries the terminal outcome to record."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


# --------------------------------------------------------------------------- #
# Scene / geometry verification
# --------------------------------------------------------------------------- #


def observed_hazard_present(env) -> bool:
    """Whether a hazard bar is actually posed inside the scene as compiled."""
    model, data = env.current_model, env.current_data
    for body in _PROTR_BODIES:
        try:
            mocap_id = int(model.body_mocapid[model.body(body).id])
        except (KeyError, ValueError, IndexError):
            continue
        if mocap_id < 0:
            continue
        if float(data.mocap_pos[mocap_id][2]) > _PARKED_Z_THRESHOLD:
            return True
    return False


def extract_row_observations(task, row: dict[str, Any]) -> dict[str, Any]:
    """Collect everything the invariance audit compares, by episode ID."""
    theta = dict(getattr(task, "scene_params", {}) or {})
    env = task.env
    data = env.current_data

    def _as_list(value):
        return [float(x) for x in np.asarray(value, dtype=float).ravel()]

    observations: dict[str, Any] = {
        "scene_template_id": row["scene_template_id"],
        "obstacle_theta": {
            key: (theta[key] if not isinstance(theta[key], np.ndarray) else _as_list(theta[key]))
            for key in sorted(theta)
            if key != "obstacle_aabbs"
        },
        "obstacle_aabbs": theta.get("obstacle_aabbs", []),
        "observed_hazard_present": observed_hazard_present(env),
        "target_uid": theta.get("target_uid"),
        "robot_initial_qpos": _as_list(data.qpos),
        "robot_initial_qvel": _as_list(data.qvel),
        "mocap_pos": _as_list(data.mocap_pos),
    }

    config = getattr(task, "config", None)
    task_config = getattr(config, "task_config", None)
    observations["selected_object"] = getattr(task_config, "pickup_obj_name", None)
    grasp = getattr(task_config, "grasp_pose", None)
    if grasp is None:
        grasp = getattr(task, "_selected_grasp", None)
    observations["selected_grasp"] = (
        _as_list(grasp) if grasp is not None and np.ndim(grasp) > 0 else grasp
    )

    scene_metadata = env.current_scene_metadata or {}
    observations["object_initial_pose"] = _as_list(data.mocap_pos) if not scene_metadata else None
    obj_name = observations["selected_object"]
    if obj_name:
        try:
            body = env.current_model.body(obj_name)
            start = int(body.jntadr[0])
            observations["object_initial_pose"] = _as_list(data.qpos[start : start + 7])
        except (KeyError, ValueError, IndexError):
            pass

    observations["episode_spec_sha256"] = sha256_payload(
        {
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "obstacle_theta": observations["obstacle_theta"],
            "selected_object": observations["selected_object"],
            "robot_initial_qpos": observations["robot_initial_qpos"],
            "object_initial_pose": observations["object_initial_pose"],
        }
    )
    return observations


# --------------------------------------------------------------------------- #
# H5 publication
# --------------------------------------------------------------------------- #


def _write_manifest_metadata(
    h5_path: Path,
    *,
    row: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Add the identity block to a written H5 without touching existing fields.

    Fields are ADDED. Existing RGB, qpos, action, task-state and proximity
    datasets are left exactly as the committed offline converter expects them.
    """
    with h5py.File(h5_path, "a") as handle:
        group = handle.require_group("manifest")
        for key, value in record.items():
            payload = value if isinstance(value, str) else canonical_json(value)
            if key in group:
                del group[key]
            group.create_dataset(key, data=np.bytes_(payload.encode("utf-8")))
        # Mirror the primary identity onto the file root and every trajectory
        # group, so a consumer that only opens traj_0 still sees the identity.
        for name, value in (
            ("episode_id", row["episode_id"]),
            ("manifest_row_sha256", row["row_sha256"]),
            ("manifest_version", row["manifest_version"]),
            ("candidate_index", str(row["candidate_index"])),
            ("hazard_present", str(bool(row["hazard_present"]))),
        ):
            handle.attrs[name] = value
            for key in handle:
                if key.startswith("traj_"):
                    handle[key].attrs[name] = value


def publish_row_trajectory(
    ledger: RowLedger,
    row: dict[str, Any],
    *,
    episode_info: dict[str, Any],
    record: dict[str, Any],
    fps: float,
    worker_logger,
) -> Path:
    """Write the row's payload to a staging area and publish it atomically."""
    row_dir = ledger.row_dir(row["episode_id"])
    row_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=row_dir, prefix=".staging."))

    prepared = prepare_episode_for_saving(
        episode_info["history"],
        episode_info["sensor_suite"],
        fps=fps,
        save_dir=staging,
        episode_idx=0,
        save_file_suffix="",
    )
    if prepared is None:
        raise ManifestRowFailure(
            OUTCOME_INFRASTRUCTURE_FAILURE, "episode produced no saveable observations"
        )

    save_trajectories(
        [prepared],
        save_dir=str(staging),
        fps=fps,
        save_file_suffix="",
        save_mp4s=True,
        logger=worker_logger,
    )
    staged_h5 = staging / "trajectories.h5"
    if not staged_h5.exists():
        raise ManifestRowFailure(OUTCOME_INFRASTRUCTURE_FAILURE, "trajectory H5 was not written")

    _write_manifest_metadata(staged_h5, row=row, record=record)

    # Videos live beside the payload under the row's own directory.
    for artifact in sorted(staging.iterdir()):
        if artifact.name == "trajectories.h5":
            continue
        os.replace(artifact, row_dir / artifact.name)

    published = ledger.publish_trajectory(row["episode_id"], staged_h5)
    with __import__("contextlib").suppress(OSError):
        staging.rmdir()
    return published


# --------------------------------------------------------------------------- #
# Row execution
# --------------------------------------------------------------------------- #


def process_single_row(
    *,
    worker_id: int,
    worker_logger,
    row: dict[str, Any],
    exp_config,
    task_sampler,
    ledger: RowLedger,
    preloaded_policy=None,
    shutdown_event=None,
) -> dict[str, Any]:
    """Execute one candidate row and return its terminal record.

    A row represents one candidate scientific episode, not one successful
    trajectory slot. It may be retried a bounded number of times at task
    sampling; each retry derives its own independent streams. One row yields at
    most one accepted trajectory, and a row that exhausts its retries is recorded
    as failed rather than replaced.
    """
    started = time.time()
    max_retries = int(row.get("max_retries", 0))
    retry_history: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "manifest_version": row["manifest_version"],
        "episode_id": row["episode_id"],
        "candidate_index": row["candidate_index"],
        "row_sha256": row["row_sha256"],
        "hazard_present": bool(row["hazard_present"]),
        "stratum_rank": row["stratum_rank"],
        "canonical_rank": row["canonical_rank"],
        "split": row["split"],
        "split_rank": row["split_rank"],
        "scene_template_id": row["scene_template_id"],
        "sensor_order_sha256": row["sensor_order_sha256"],
        "runtime_contract_sha256": row["runtime_contract_sha256"],
        "molmospaces_source_commit": row["molmospaces_source_commit"],
        # Descriptive only. Recorded for operational debugging; by construction
        # neither value reaches any random draw or any scientific decision.
        "worker_id_descriptive": int(worker_id),
        "house_alias_descriptive": int(row["scene_template_house_index"]),
    }

    for retry_index in range(max_retries + 1):
        if shutdown_event is not None and shutdown_event.is_set():
            raise ManifestRowFailure(
                OUTCOME_INFRASTRUCTURE_FAILURE, "shutdown requested before the row completed"
            )

        task = None
        policy = None
        try:
            # 1. Drop everything that could make this row depend on rows this
            #    worker executed earlier. Documented immutable caches (the loaded
            #    scene, the dataset index map) are preserved on purpose.
            cleared = reset_episode_scoped_sampler_state(task_sampler)

            # 2. Install the row's RNG contract. This must precede every
            #    task-level draw, so it happens before sample_task is called.
            contract = install_row_seed_contract(row, retry_index, task_sampler=task_sampler)

            # 3. Pin the hazard assignment. The runtime Bernoulli is bypassed.
            task_sampler.set_manifest_row(row, retry_index)

            record.setdefault("cleared_sampler_state", sorted(cleared))
            record["seed_contract"] = contract.to_dict()
            record["retry_index"] = retry_index

            task = task_sampler.sample_task(house_index=row["scene_template_house_index"])
            if task is None:
                retry_history.append({"retry_index": retry_index, "reason": "sample_task_none"})
                continue

            # 4. Verify the compiled scene actually matches the row.
            observations = extract_row_observations(task, row)
            if observations["observed_hazard_present"] != bool(row["hazard_present"]):
                raise ManifestRowFailure(
                    OUTCOME_INFRASTRUCTURE_FAILURE,
                    "compiled hazard geometry does not match the manifest row: "
                    f"observed={observations['observed_hazard_present']} "
                    f"manifest={bool(row['hazard_present'])}",
                )

            policy = setup_policy(exp_config, task, preloaded_policy, None)
            success = ParallelRolloutRunner.run_single_rollout(
                episode_seed=contract.seed_map["global_compat"]["seed_u64"],
                task=task,
                policy=policy,
                profiler=None,
                viewer=None,
                shutdown_event=shutdown_event,
                datagen_profiler=None,
                end_on_success=exp_config.end_on_success,
            )

            observations["planner_phase_path"] = list(
                getattr(policy, "phase_history", None) or [getattr(policy, "behavior_class", "")]
            )
            observations["behavior_class"] = getattr(policy, "behavior_class", None)

            record["observations"] = observations
            record["retry_count"] = retry_index
            record["retry_history"] = retry_history
            record["success"] = bool(success)
            record["duration_s"] = time.time() - started
            record["episode_info"] = {
                "history": task.get_history(),
                "sensor_suite": task.sensor_suite,
            }
            return record

        except ManifestRowFailure:
            raise
        except HouseInvalidForTask as exc:
            retry_history.append(
                {"retry_index": retry_index, "reason": f"house_invalid: {exc.reason}"}
            )
        except Exception as exc:  # noqa: BLE001 - recorded, then retried or failed
            worker_logger.error(
                f"worker {worker_id} row {row['episode_id']} retry {retry_index} raised: "
                f"{exc}\n{traceback.format_exc()}"
            )
            retry_history.append(
                {"retry_index": retry_index, "reason": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            task_sampler.clear_manifest_row()
            if task is not None or policy is not None:
                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=None,
                    preloaded_policy=preloaded_policy,
                    close_task_sampler=False,
                )

    record["retry_count"] = max_retries
    record["retry_history"] = retry_history
    record["duration_s"] = time.time() - started
    raise ManifestRowFailure(
        OUTCOME_SAMPLING_FAILURE,
        f"row exhausted its {max_retries} retries: "
        + "; ".join(entry["reason"] for entry in retry_history),
    )


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


def manifest_row_worker(
    worker_id: int,
    exp_config,
    rows: list[dict[str, Any]],
    run_id: str,
    shutdown_event,
    counter_lock,
    row_counter,
    worker_reports=None,
) -> None:
    """Process manifest rows from the shared queue until it is empty."""
    worker_logger = get_worker_logger(worker_id)
    ledger = RowLedger(exp_config.output_dir, run_id)

    worker_status = STATUS_FAILED
    worker_error: str | None = None
    claimed_rows: list[str] = []
    finalized_rows: list[str] = []
    attempted = 0
    written = 0
    successful = 0

    task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)

    try:
        while True:
            if shutdown_event is not None and shutdown_event.is_set():
                worker_logger.info(f"worker {worker_id} stopping on shutdown signal")
                break

            with counter_lock:
                if row_counter.value >= len(rows):
                    break
                row = rows[row_counter.value]
                row_counter.value += 1

            episode_id = row["episode_id"]
            if not ledger.claim(episode_id, row["row_sha256"], worker_id):
                worker_logger.info(f"worker {worker_id} skipping finalised row {episode_id}")
                continue

            claimed_rows.append(episode_id)
            attempted += 1
            worker_logger.info(
                f"worker {worker_id} claimed candidate {row['candidate_index']} "
                f"({episode_id[:12]}) hazard={row['hazard_present']}"
            )

            try:
                record = process_single_row(
                    worker_id=worker_id,
                    worker_logger=worker_logger,
                    row=row,
                    exp_config=exp_config,
                    task_sampler=task_sampler,
                    ledger=ledger,
                    shutdown_event=shutdown_event,
                )
            except ManifestRowFailure as failure:
                ledger.finalize(
                    episode_id,
                    status=failure.status,
                    row=row,
                    metadata={
                        "reason": failure.reason,
                        "worker_id_descriptive": int(worker_id),
                    },
                )
                finalized_rows.append(episode_id)
                worker_logger.error(f"row {episode_id} failed: {failure.status}: {failure.reason}")
                continue

            episode_info = record.pop("episode_info")
            success = record["success"]
            status = OUTCOME_SUCCESS if success else OUTCOME_TASK_FAILURE

            published: Path | None = None
            if success:
                # Storage policy: full payloads are retained for accepted
                # trajectories only. Failed rows keep a complete atomic outcome
                # record, so no candidate identity is ever erased.
                published = publish_row_trajectory(
                    ledger,
                    row,
                    episode_info=episode_info,
                    record=record,
                    fps=exp_config.fps,
                    worker_logger=worker_logger,
                )
                written += 1
                successful += 1

            ledger.finalize(
                episode_id,
                status=status,
                row=row,
                metadata={
                    "trajectory_path": str(published) if published else None,
                    "retry_count": record.get("retry_count", 0),
                    "retry_history": record.get("retry_history", []),
                    "observations": record.get("observations", {}),
                    "seed_contract": record.get("seed_contract", {}),
                    "cleared_sampler_state": record.get("cleared_sampler_state", []),
                    "worker_id_descriptive": int(worker_id),
                    "duration_s": record.get("duration_s"),
                },
            )
            finalized_rows.append(episode_id)

        worker_status = (
            STATUS_SHUTDOWN
            if shutdown_event is not None and shutdown_event.is_set()
            else STATUS_COMPLETED
        )
    except BaseException as exc:  # noqa: BLE001 - must still publish a record
        worker_status = STATUS_FAILED
        worker_error = f"{type(exc).__name__}: {exc}"
        worker_logger.error(f"worker {worker_id} terminating: {worker_error}\n{traceback.format_exc()}")
        raise
    finally:
        if worker_reports is not None:
            try:
                report = WorkerReport(
                    worker_id=worker_id,
                    status=worker_status,
                    houses_assigned=[r["candidate_index"] for r in rows[:0]] or [],
                    houses_written=[],
                    episodes_attempted=attempted,
                    episodes_written=written,
                    episodes_successful=successful,
                    error=worker_error,
                ).to_dict()
                report["claimed_rows"] = claimed_rows
                report["finalized_rows"] = finalized_rows
                worker_reports[worker_id] = report
            except Exception as report_exc:  # pragma: no cover - diagnostic only
                worker_logger.error(f"worker {worker_id} could not publish status: {report_exc}")
        if task_sampler is not None:
            task_sampler.close()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class ManifestRolloutRunner(ParallelRolloutRunner):
    """Runs a committed candidate manifest instead of a house list."""

    def load_rows(self) -> list[dict[str, Any]]:
        manifest_path = getattr(self.config, "manifest_path", None)
        if not manifest_path:
            raise ManifestError("config does not declare manifest_path")
        document = load_manifest(manifest_path)

        subset_path = getattr(self.config, "smoke_subset_path", None)
        if subset_path:
            with open(subset_path) as stream:
                subset = json.load(stream)
            validate_smoke_subset(subset, document)
            rows = subset["rows"]
            self.logger.info(
                f"loaded smoke subset {subset['subset_sha256']} "
                f"({len(rows)} rows) from manifest {document['manifest_sha256']}"
            )
        else:
            rows = document["rows"]
            self.logger.info(
                f"loaded full manifest {document['manifest_sha256']} ({len(rows)} rows)"
            )

        self.manifest_document = document
        # Deterministic execution order by candidate index. Order affects only
        # which worker happens to take which row, never a row's content.
        return sorted(rows, key=lambda r: r["candidate_index"])

    def verify_contract(self, rows: list[dict[str, Any]]) -> None:
        """Fail loudly on any config/sensor/runtime hash disagreement."""
        expected_sensor = getattr(self.config, "expected_sensor_order_sha256", None)
        expected_env = getattr(self.config, "expected_env_config_sha256", None)
        expected_runtime = getattr(self.config, "expected_runtime_contract_sha256", None)
        for row in rows:
            if row["manifest_version"] != MANIFEST_VERSION:
                raise ManifestError(f"row {row['candidate_index']}: wrong manifest version")
            if expected_sensor and row["sensor_order_sha256"] != expected_sensor:
                raise ManifestError(
                    f"row {row['candidate_index']}: sensor-order hash "
                    f"{row['sensor_order_sha256']} != configured {expected_sensor}"
                )
            if expected_env and row["env_config_sha256"] != expected_env:
                raise ManifestError(
                    f"row {row['candidate_index']}: env/config hash mismatch"
                )
            if expected_runtime and row["runtime_contract_sha256"] != expected_runtime:
                raise ManifestError(
                    f"row {row['candidate_index']}: runtime-contract hash mismatch"
                )

    def run(self, preloaded_policy=None) -> tuple[int, int]:  # noqa: ARG002
        rows = self.load_rows()
        self.verify_contract(rows)

        run_id = getattr(self.config, "run_id", None) or uuid.uuid4().hex
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_config(output_dir=output_dir)

        ledger = RowLedger(output_dir, run_id)
        expected_ids = [row["episode_id"] for row in rows]

        # Resume: reclaim only properly abandoned rows (a claim from a previous
        # run with no terminal outcome). Finalised rows are never re-executed.
        reclaimed = ledger.reclaim_abandoned(expected_ids)
        pending = [row for row in rows if not ledger.is_complete(row["episode_id"])]
        already_done = len(rows) - len(pending)
        self.logger.info(
            f"run {run_id}: {len(rows)} manifest rows, {already_done} already finalised, "
            f"{len(reclaimed)} abandoned claims reclaimed, {len(pending)} to execute"
        )

        registry = WorkerRegistry(range(self.config.num_workers), reports=self.worker_reports)
        row_counter = self.house_counter
        row_counter.value = 0

        if self.config.num_workers > 1:
            processes = []
            for worker_id in range(self.config.num_workers):
                process = mp_context.Process(
                    target=manifest_row_worker,
                    args=(
                        worker_id,
                        self.config,
                        pending,
                        run_id,
                        self.shutdown_event,
                        self.counter_lock,
                        row_counter,
                        self.worker_reports,
                    ),
                )
                process.start()
                processes.append(process)
            for worker_id, process in enumerate(processes):
                process.join()
                registry.record_exit_code(worker_id, process.exitcode)
                process.close()
        else:
            manifest_row_worker(
                worker_id=0,
                exp_config=self.config,
                rows=pending,
                run_id=run_id,
                shutdown_event=self.shutdown_event,
                counter_lock=self.counter_lock,
                row_counter=row_counter,
                worker_reports=self.worker_reports,
            )
            registry.record_exit_code(0, 0)

        reconciliation = ledger.reconcile(expected_ids)

        summary = build_final_summary(
            registry,
            expected_house_indices=[row["candidate_index"] for row in rows],
            expected_episodes=len(rows),
            counters={
                "rows_expected": len(rows),
                "rows_finalized": reconciliation["finalized_rows"],
                "rows_succeeded": len(reconciliation["succeeded"]),
                "rows_failed": len(reconciliation["failed"]),
            },
        )
        # build_final_summary compares written houses against expected houses,
        # which is meaningless here: rows, not houses, are the unit of work.
        # Row reconciliation is authoritative and replaces that verdict.
        summary["houses_missing"] = []
        summary["houses_unexpected"] = []
        summary["manifest"] = {
            "manifest_version": MANIFEST_VERSION,
            "manifest_sha256": self.manifest_document["manifest_sha256"],
            "run_id": run_id,
            "reclaimed_abandoned_claims": reclaimed,
            "rows_already_finalised_on_entry": already_done,
        }
        summary["row_reconciliation"] = reconciliation
        workers_ok = summary["workers"]["complete"]
        summary["complete"] = bool(reconciliation["ok"] and workers_ok)
        summary["status"] = "complete" if summary["complete"] else "COLLECTION_INCOMPLETE"
        write_summary_atomically(output_dir, summary)

        if not summary["complete"]:
            self.logger.error(
                "MANIFEST COLLECTION INCOMPLETE — partial output retained at "
                f"{output_dir}. missing outcomes: {reconciliation['missing_outcome']}; "
                f"published without outcome: {reconciliation['published_without_outcome']}; "
                f"unexpected row dirs: {reconciliation['unexpected_row_dirs']}"
            )
            if not workers_ok:
                registry.validate()
            raise WorkerCompletenessError(
                "manifest rows did not reconcile: "
                f"{len(reconciliation['missing_outcome'])} rows have no terminal outcome"
            )

        succeeded = len(reconciliation["succeeded"])
        self.logger.info(
            f"manifest run complete: {succeeded}/{len(rows)} rows succeeded, "
            f"{len(reconciliation['failed'])} recorded as failures"
        )
        return succeeded, len(rows)


__all__ = [
    "ManifestRolloutRunner",
    "ManifestRowFailure",
    "manifest_row_worker",
    "observed_hazard_present",
    "process_single_row",
    "publish_row_trajectory",
]
