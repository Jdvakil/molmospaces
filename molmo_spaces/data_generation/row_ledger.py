"""Atomic per-row lifecycle ledger for the manifest-driven collection.

Episode identity is ``episode_id + manifest_row_sha256``. It is never derived
from a worker-local episode counter, a batch index, a house directory, a
wraparound house alias, or file ordering. That is the whole point: the previous
collection's identity was its position in a buffer, so resume, deduplication and
provenance were all unsound.

Layout under ``<output_dir>/rows/<episode_id>/``::

    claim.json      created with O_CREAT|O_EXCL -- the atomic claim
    outcome.json    written via mkstemp + fsync + os.replace -- the finalisation
    trajectory.h5   the published payload, also via atomic replace

A row is complete iff ``outcome.json`` exists and carries a terminal status. A
row is reclaimable iff it holds a claim from a *previous* run with no outcome:
that is a properly abandoned row. A claim from the *current* run with no outcome
is a duplicate-claim bug and is reported as such.

This module imports only the standard library so its behaviour can be tested
without a simulator.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from molmo_spaces.data_generation.episode_manifest import TERMINAL_OUTCOMES

CLAIM_FILENAME = "claim.json"
OUTCOME_FILENAME = "outcome.json"
TRAJECTORY_FILENAME = "trajectory.h5"
ROWS_DIRNAME = "rows"


class RowLedgerError(RuntimeError):
    """Raised when the row lifecycle contract is violated."""


class DuplicateClaimError(RowLedgerError):
    """Raised when a row is claimed twice within the same run."""


class DuplicatePublicationError(RowLedgerError):
    """Raised when a row's payload is published twice."""


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> Path:
    """Write JSON so the file appears complete or not at all."""
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
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


class RowLedger:
    """Filesystem-backed, crash-safe ledger of manifest-row lifecycle."""

    def __init__(self, output_dir: str | os.PathLike[str], run_id: str) -> None:
        self.root = Path(output_dir) / ROWS_DIRNAME
        self.run_id = str(run_id)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------
    def row_dir(self, episode_id: str) -> Path:
        return self.root / episode_id

    def claim_path(self, episode_id: str) -> Path:
        return self.row_dir(episode_id) / CLAIM_FILENAME

    def outcome_path(self, episode_id: str) -> Path:
        return self.row_dir(episode_id) / OUTCOME_FILENAME

    def trajectory_path(self, episode_id: str) -> Path:
        return self.row_dir(episode_id) / TRAJECTORY_FILENAME

    # -- queries -------------------------------------------------------------
    def read_outcome(self, episode_id: str) -> dict[str, Any] | None:
        path = self.outcome_path(episode_id)
        if not path.exists():
            return None
        try:
            with open(path) as stream:
                return json.load(stream)
        except (OSError, json.JSONDecodeError):
            # A truncated outcome cannot happen through _atomic_write_json, so
            # treat it as absent rather than silently accepting broken work.
            return None

    def read_claim(self, episode_id: str) -> dict[str, Any] | None:
        path = self.claim_path(episode_id)
        if not path.exists():
            return None
        try:
            with open(path) as stream:
                return json.load(stream)
        except (OSError, json.JSONDecodeError):
            return None

    def is_complete(self, episode_id: str) -> bool:
        outcome = self.read_outcome(episode_id)
        return bool(outcome and outcome.get("status") in TERMINAL_OUTCOMES)

    # -- lifecycle -----------------------------------------------------------
    def reclaim_abandoned(self, episode_ids: list[str]) -> list[str]:
        """Drop stale claims from previous runs so resume can retry those rows.

        Only rows with a claim from a *different* run and no terminal outcome are
        reclaimed. A row finalised by any run stays finalised: resume never
        re-executes completed work, and never silently discards it.
        """
        reclaimed: list[str] = []
        for episode_id in episode_ids:
            if self.is_complete(episode_id):
                continue
            claim = self.read_claim(episode_id)
            if claim is None:
                continue
            if claim.get("run_id") == self.run_id:
                continue
            with contextlib.suppress(OSError):
                self.claim_path(episode_id).unlink()
                reclaimed.append(episode_id)
        return reclaimed

    def claim(self, episode_id: str, row_sha256: str, worker_id: int) -> bool:
        """Atomically claim a row. Returns False if it is already finalised.

        Raises ``DuplicateClaimError`` when the row already holds a claim from
        this run: within one run each row must be handed out exactly once.
        """
        if self.is_complete(episode_id):
            return False
        directory = self.row_dir(episode_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "episode_id": episode_id,
            "row_sha256": row_sha256,
            "run_id": self.run_id,
            # Worker ID and PID are descriptive only. They are recorded for
            # operational debugging and never influence scientific content.
            "worker_id": int(worker_id),
            "pid": os.getpid(),
        }
        blob = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        try:
            fd = os.open(self.claim_path(episode_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = self.read_claim(episode_id) or {}
            if existing.get("run_id") == self.run_id:
                raise DuplicateClaimError(
                    f"row {episode_id} was already claimed in run {self.run_id} by worker "
                    f"{existing.get('worker_id')} (pid {existing.get('pid')})"
                ) from None
            raise DuplicateClaimError(
                f"row {episode_id} holds an unreconciled claim from run "
                f"{existing.get('run_id')!r}; call reclaim_abandoned first"
            ) from None
        with os.fdopen(fd, "w") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        return True

    def publish_trajectory(self, episode_id: str, source: str | os.PathLike[str]) -> Path:
        """Move a staged payload into place atomically, exactly once."""
        target = self.trajectory_path(episode_id)
        if target.exists():
            raise DuplicatePublicationError(
                f"row {episode_id} already has a published trajectory at {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        return target

    def finalize(
        self,
        episode_id: str,
        *,
        status: str,
        row: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Record a row's terminal outcome. Exactly once per row.

        Failed rows are preserved as explicit outcomes: a failure is a recorded
        scientific result, not an invitation to substitute a different episode.
        """
        if status not in TERMINAL_OUTCOMES:
            raise RowLedgerError(
                f"row {episode_id}: {status!r} is not a terminal outcome "
                f"(expected one of {sorted(TERMINAL_OUTCOMES)})"
            )
        existing = self.read_outcome(episode_id)
        if existing is not None:
            raise RowLedgerError(
                f"row {episode_id} was already finalised as {existing.get('status')!r}"
            )
        payload = {
            "episode_id": episode_id,
            "candidate_index": row["candidate_index"],
            "row_sha256": row["row_sha256"],
            "manifest_version": row["manifest_version"],
            "hazard_present": row["hazard_present"],
            "stratum_rank": row["stratum_rank"],
            "split": row["split"],
            "status": status,
            "run_id": self.run_id,
        }
        payload.update(metadata or {})
        return _atomic_write_json(self.outcome_path(episode_id), payload)

    # -- reconciliation ------------------------------------------------------
    def reconcile(self, episode_ids: list[str]) -> dict[str, Any]:
        """Account for every expected row exactly once.

        ``ok`` is true only when every expected row carries exactly one terminal
        outcome, nothing was published without an outcome, and no unexpected row
        appeared. Anything else is unreconciled work and must exit nonzero.
        """
        expected = list(episode_ids)
        outcomes: dict[str, str] = {}
        missing: list[str] = []
        unclaimed: list[str] = []
        published_without_outcome: list[str] = []

        for episode_id in expected:
            outcome = self.read_outcome(episode_id)
            if outcome is None:
                missing.append(episode_id)
                if self.read_claim(episode_id) is None:
                    unclaimed.append(episode_id)
                if self.trajectory_path(episode_id).exists():
                    published_without_outcome.append(episode_id)
                continue
            outcomes[episode_id] = outcome["status"]

        seen = {p.name for p in self.root.iterdir() if p.is_dir()} if self.root.exists() else set()
        unexpected = sorted(seen - set(expected))

        succeeded = sorted(k for k, v in outcomes.items() if v == "success")
        failed = sorted(k for k, v in outcomes.items() if v != "success")

        return {
            "expected_rows": len(expected),
            "finalized_rows": len(outcomes),
            "succeeded": succeeded,
            "failed": failed,
            "outcomes": outcomes,
            "missing_outcome": sorted(missing),
            "never_claimed": sorted(unclaimed),
            "published_without_outcome": sorted(published_without_outcome),
            "unexpected_row_dirs": unexpected,
            "ok": not missing and not unexpected and not published_without_outcome,
        }
