"""Fail-loud completion contract for the data-generation worker pool.

Background
----------
``ParallelRolloutRunner`` previously joined its workers without inspecting
``Process.exitcode`` and built its final summary purely from shared counters. A
worker that hung or died therefore contributed nothing to ``completed_houses``
*and* nothing to ``skipped_houses``, so its houses vanished from the summary
while the run still exited 0 with ``"skipped 0 houses"``. Because a house is
buffered in memory and only written when it completes, every trajectory that
worker held was discarded silently.

This module makes that failure mode impossible to miss:

* the expected worker IDs are declared before launch;
* every worker publishes a final completion record, including on failure;
* the parent records each worker's exit code;
* missing, crashed and silently terminated workers are detected and named;
* per-worker attempted/written/successful counts are reported;
* the run is declared incomplete (nonzero exit) unless every expected worker
  finished with an approved terminal status;
* the final summary is published atomically, so a partially written summary can
  never be mistaken for a complete one;
* partial output is retained and explicitly marked incomplete.

The module is deliberately free of simulator, torch and MuJoCo imports so that
its behaviour can be tested directly.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Terminal statuses a worker may legitimately publish.
STATUS_COMPLETED = "completed"
STATUS_SHUTDOWN = "shutdown_requested"
STATUS_FAILED = "failed"
APPROVED_FINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_SHUTDOWN})
ALL_FINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_SHUTDOWN, STATUS_FAILED})

SUMMARY_FILENAME = "collection_summary.json"
INCOMPLETE_MARKER = "COLLECTION_INCOMPLETE"


class WorkerCompletenessError(RuntimeError):
    """Raised when the worker pool cannot be shown to have finished completely."""


@dataclass
class WorkerReport:
    """Terminal record published by one worker."""

    worker_id: int
    status: str
    houses_assigned: list[int] = field(default_factory=list)
    houses_written: list[int] = field(default_factory=list)
    episodes_attempted: int = 0
    episodes_written: int = 0
    episodes_successful: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ALL_FINAL_STATUSES:
            raise ValueError(
                f"worker {self.worker_id}: status {self.status!r} is not a final status "
                f"(expected one of {sorted(ALL_FINAL_STATUSES)})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkerRegistry:
    """Tracks the expected worker set and validates terminal state.

    ``reports`` may be a plain dict (single-process use and tests) or a
    ``multiprocessing.Manager().dict()`` shared with worker processes.
    """

    def __init__(self, expected_worker_ids: Iterable[int], reports: Any | None = None) -> None:
        ids = list(expected_worker_ids)
        if len(set(ids)) != len(ids):
            raise ValueError(f"expected worker IDs contain duplicates: {ids}")
        if not ids:
            raise ValueError("expected worker ID set must not be empty")
        self.expected: set[int] = set(ids)
        self.reports: Any = {} if reports is None else reports
        self.exit_codes: dict[int, int | None] = {}

    # -- worker side ---------------------------------------------------------
    def publish(self, report: WorkerReport) -> None:
        """Record a worker's terminal status. Called from the worker process."""
        if report.worker_id not in self.expected:
            raise WorkerCompletenessError(
                f"worker {report.worker_id} is not in the expected set "
                f"{sorted(self.expected)}"
            )
        if report.worker_id in self.reports:
            raise WorkerCompletenessError(
                f"duplicate final status published for worker {report.worker_id}"
            )
        self.reports[report.worker_id] = report.to_dict()

    # -- parent side ---------------------------------------------------------
    def record_exit_code(self, worker_id: int, exit_code: int | None) -> None:
        self.exit_codes[worker_id] = exit_code

    def missing_reports(self) -> list[int]:
        return sorted(self.expected - set(self.reports.keys()))

    def failed_workers(self) -> list[int]:
        return sorted(
            wid
            for wid, rep in self.reports.items()
            if rep.get("status") not in APPROVED_FINAL_STATUSES
        )

    def bad_exit_codes(self) -> dict[int, int | None]:
        return {
            wid: code
            for wid, code in sorted(self.exit_codes.items())
            if code is None or code != 0
        }

    def summary(self) -> dict[str, Any]:
        reports = {int(k): dict(v) for k, v in self.reports.items()}
        missing = self.missing_reports()
        failed = self.failed_workers()
        bad_exits = self.bad_exit_codes()
        # Silent loss is exactly "no final status published". A zero exit code
        # alongside a missing report is the worst case, not an excuse: that is
        # precisely how a hung worker's buffered houses used to disappear while
        # the run still reported success.
        silently_lost = missing
        complete = not missing and not failed and not bad_exits
        return {
            "expected_workers": sorted(self.expected),
            "reporting_workers": sorted(reports),
            "missing_final_status": missing,
            "workers_with_failed_status": failed,
            "worker_exit_codes": {str(k): v for k, v in sorted(self.exit_codes.items())},
            "nonzero_or_unknown_exit_codes": {str(k): v for k, v in bad_exits.items()},
            "silently_lost_workers": silently_lost,
            "per_worker": {
                str(wid): {
                    "status": rep.get("status"),
                    "houses_assigned": rep.get("houses_assigned", []),
                    "houses_written": rep.get("houses_written", []),
                    "episodes_attempted": rep.get("episodes_attempted", 0),
                    "episodes_written": rep.get("episodes_written", 0),
                    "episodes_successful": rep.get("episodes_successful", 0),
                    "exit_code": self.exit_codes.get(wid),
                    "error": rep.get("error"),
                }
                for wid, rep in sorted(reports.items())
            },
            "totals": {
                "episodes_attempted": sum(r.get("episodes_attempted", 0) for r in reports.values()),
                "episodes_written": sum(r.get("episodes_written", 0) for r in reports.values()),
                "episodes_successful": sum(
                    r.get("episodes_successful", 0) for r in reports.values()
                ),
                "houses_written": sorted(
                    h for r in reports.values() for h in r.get("houses_written", [])
                ),
            },
            "complete": complete,
        }

    def validate(self) -> dict[str, Any]:
        """Return the summary, raising if the pool cannot be shown complete."""
        summary = self.summary()
        if summary["complete"]:
            return summary
        problems = []
        if summary["missing_final_status"]:
            problems.append(
                "workers published no final status: "
                f"{summary['missing_final_status']}"
            )
        if summary["workers_with_failed_status"]:
            problems.append(
                "workers reported a failed status: "
                f"{summary['workers_with_failed_status']}"
            )
        if summary["nonzero_or_unknown_exit_codes"]:
            problems.append(
                "workers exited nonzero or with unknown status: "
                f"{summary['nonzero_or_unknown_exit_codes']}"
            )
        raise WorkerCompletenessError(
            "data generation did not complete: " + "; ".join(problems)
        )


def write_summary_atomically(output_dir: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Publish the final summary atomically within ``output_dir``.

    The file appears complete or not at all: it is written to a temporary file in
    the same directory and then moved into place with ``os.replace``.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / SUMMARY_FILENAME
    # Serialise first: a payload that cannot be encoded must not disturb an
    # already-published summary.
    blob = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=".collection_summary.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return target


def build_final_summary(
    registry: WorkerRegistry,
    *,
    expected_house_indices: Iterable[int],
    expected_episodes: int,
    counters: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Compose the run summary, marking it incomplete when anything is unaccounted for."""
    worker_summary = registry.summary()
    expected_houses = sorted(set(expected_house_indices))
    written = sorted(set(worker_summary["totals"]["houses_written"]))
    missing_houses = sorted(set(expected_houses) - set(written))
    unexpected_houses = sorted(set(written) - set(expected_houses))
    complete = worker_summary["complete"] and not missing_houses and not unexpected_houses
    summary: dict[str, Any] = {
        "schema_version": "molmospaces_collection_summary_v1",
        "status": "complete" if complete else INCOMPLETE_MARKER,
        "complete": complete,
        "expected_houses": expected_houses,
        "houses_written": written,
        "houses_missing": missing_houses,
        "houses_unexpected": unexpected_houses,
        "expected_episodes": expected_episodes,
        "workers": worker_summary,
        "shared_counters": dict(counters or {}),
    }
    if not complete:
        summary["warning"] = (
            "Partial output retained. Some workers did not publish an approved final "
            "status or did not exit cleanly, so trajectories they buffered may be "
            "missing. Do not treat this collection as canonical."
        )
    return summary
