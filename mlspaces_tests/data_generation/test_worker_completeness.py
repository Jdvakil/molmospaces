"""Tests for the fail-loud data-generation worker completion contract.

These cover the failure mode that silently lost a whole house from a completed
collection: a worker stopped producing output, published no terminal status, and
the parent joined it without inspecting its exit code, so the run still exited 0
reporting "skipped 0 houses".
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os

import pytest

from molmo_spaces.data_generation.worker_completeness import (
    INCOMPLETE_MARKER,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SHUTDOWN,
    WorkerCompletenessError,
    WorkerRegistry,
    WorkerReport,
    build_final_summary,
    write_summary_atomically,
)


def make_report(worker_id: int, **kwargs) -> WorkerReport:
    defaults = dict(
        status=STATUS_COMPLETED,
        houses_assigned=[worker_id],
        houses_written=[worker_id],
        episodes_attempted=10,
        episodes_written=8,
        episodes_successful=7,
    )
    defaults.update(kwargs)
    return WorkerReport(worker_id=worker_id, **defaults)


# --------------------------------------------------------------------------
# normal completion
# --------------------------------------------------------------------------
def test_normal_worker_completion_validates():
    reg = WorkerRegistry([0, 1, 2])
    for wid in (0, 1, 2):
        reg.publish(make_report(wid))
        reg.record_exit_code(wid, 0)
    summary = reg.validate()
    assert summary["complete"] is True
    assert summary["missing_final_status"] == []
    assert summary["silently_lost_workers"] == []
    assert summary["totals"]["episodes_successful"] == 21
    assert summary["per_worker"]["1"]["episodes_written"] == 8


def test_per_worker_counts_are_reported():
    reg = WorkerRegistry([0, 1])
    reg.publish(make_report(0, episodes_attempted=25, episodes_written=20, episodes_successful=19))
    reg.publish(make_report(1, episodes_attempted=30, episodes_written=25, episodes_successful=24))
    reg.record_exit_code(0, 0)
    reg.record_exit_code(1, 0)
    s = reg.validate()
    assert s["per_worker"]["0"]["episodes_attempted"] == 25
    assert s["per_worker"]["1"]["episodes_successful"] == 24
    assert s["totals"]["episodes_attempted"] == 55


# --------------------------------------------------------------------------
# worker exception
# --------------------------------------------------------------------------
def test_worker_exception_status_is_not_approved():
    reg = WorkerRegistry([0, 1])
    reg.publish(make_report(0))
    reg.publish(make_report(1, status=STATUS_FAILED, error="ValueError: boom"))
    reg.record_exit_code(0, 0)
    reg.record_exit_code(1, 1)
    with pytest.raises(WorkerCompletenessError) as exc:
        reg.validate()
    assert "failed status" in str(exc.value)
    assert reg.summary()["workers_with_failed_status"] == [1]


def test_shutdown_request_is_an_approved_terminal_status():
    reg = WorkerRegistry([0])
    reg.publish(make_report(0, status=STATUS_SHUTDOWN))
    reg.record_exit_code(0, 0)
    assert reg.validate()["complete"] is True


# --------------------------------------------------------------------------
# worker process death / missing final status
# --------------------------------------------------------------------------
def test_missing_final_status_is_detected_even_with_zero_exit_code():
    """The exact defect: worker vanished, exit code 0, no terminal record."""
    reg = WorkerRegistry([0, 1, 2, 3])
    for wid in (1, 2, 3):
        reg.publish(make_report(wid))
        reg.record_exit_code(wid, 0)
    reg.record_exit_code(0, 0)  # joined cleanly, but never reported
    with pytest.raises(WorkerCompletenessError) as exc:
        reg.validate()
    assert "no final status" in str(exc.value)
    s = reg.summary()
    assert s["missing_final_status"] == [0]
    assert s["silently_lost_workers"] == [0]
    assert s["complete"] is False


def test_nonzero_exit_code_is_detected():
    reg = WorkerRegistry([0, 1])
    reg.publish(make_report(0))
    reg.publish(make_report(1))
    reg.record_exit_code(0, 0)
    reg.record_exit_code(1, -9)  # SIGKILL
    with pytest.raises(WorkerCompletenessError) as exc:
        reg.validate()
    assert "exited nonzero" in str(exc.value)
    assert reg.summary()["nonzero_or_unknown_exit_codes"] == {"1": -9}


def test_unknown_exit_code_is_detected():
    reg = WorkerRegistry([0])
    reg.publish(make_report(0))
    reg.record_exit_code(0, None)  # still alive / never joined
    with pytest.raises(WorkerCompletenessError):
        reg.validate()


def _child_dies(reports, worker_id):  # pragma: no cover - runs in a subprocess
    os._exit(3)


def _child_reports(reports, worker_id):  # pragma: no cover - runs in a subprocess
    reports[worker_id] = WorkerReport(
        worker_id=worker_id, status=STATUS_COMPLETED, houses_written=[worker_id]
    ).to_dict()


def test_real_process_death_is_caught():
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        reports = manager.dict()
        reg = WorkerRegistry([0, 1], reports=reports)
        procs = {}
        for wid, target in ((0, _child_reports), (1, _child_dies)):
            p = ctx.Process(target=target, args=(reports, wid))
            p.start()
            procs[wid] = p
        for wid, p in procs.items():
            p.join(timeout=60)
            reg.record_exit_code(wid, p.exitcode)
        with pytest.raises(WorkerCompletenessError):
            reg.validate()
        s = reg.summary()
        assert 1 in s["missing_final_status"]
        assert s["nonzero_or_unknown_exit_codes"] == {"1": 3}


# --------------------------------------------------------------------------
# duplicate worker ID
# --------------------------------------------------------------------------
def test_duplicate_expected_worker_ids_rejected():
    with pytest.raises(ValueError):
        WorkerRegistry([0, 1, 1])


def test_duplicate_publication_rejected():
    reg = WorkerRegistry([0])
    reg.publish(make_report(0))
    with pytest.raises(WorkerCompletenessError) as exc:
        reg.publish(make_report(0))
    assert "duplicate final status" in str(exc.value)


def test_unexpected_worker_id_rejected():
    reg = WorkerRegistry([0, 1])
    with pytest.raises(WorkerCompletenessError):
        reg.publish(make_report(5))


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        WorkerReport(worker_id=0, status="probably_fine")


# --------------------------------------------------------------------------
# parent interruption / resume
# --------------------------------------------------------------------------
def test_parent_interruption_leaves_incomplete_summary(tmp_path):
    """A run interrupted before every worker reports is marked incomplete."""
    reg = WorkerRegistry([0, 1])
    reg.publish(make_report(0, houses_written=[0]))
    reg.record_exit_code(0, 0)
    # worker 1 never joined: parent interrupted
    summary = build_final_summary(
        reg, expected_house_indices=[0, 1], expected_episodes=50
    )
    assert summary["complete"] is False
    assert summary["status"] == INCOMPLETE_MARKER
    assert summary["houses_missing"] == [1]
    assert "Partial output retained" in summary["warning"]
    path = write_summary_atomically(tmp_path, summary)
    assert json.loads(path.read_text())["status"] == INCOMPLETE_MARKER


def test_resume_after_interruption_can_reach_complete(tmp_path):
    reg = WorkerRegistry([0, 1])
    reg.publish(make_report(0, houses_written=[0]))
    reg.record_exit_code(0, 0)
    first = build_final_summary(reg, expected_house_indices=[0, 1], expected_episodes=50)
    write_summary_atomically(tmp_path, first)
    assert not first["complete"]

    # resumed run completes the outstanding worker
    reg.publish(make_report(1, houses_written=[1]))
    reg.record_exit_code(1, 0)
    second = build_final_summary(reg, expected_house_indices=[0, 1], expected_episodes=50)
    path = write_summary_atomically(tmp_path, second)
    assert second["complete"] is True
    assert second["status"] == "complete"
    assert json.loads(path.read_text())["status"] == "complete"


def test_missing_house_marks_incomplete_even_when_workers_report(tmp_path):
    """Reproduces the observed defect end to end: 7 of 8 houses written."""
    reg = WorkerRegistry([0, 1, 2, 3])
    written = {0: [], 1: [25, 145, 169], 2: [49, 121], 3: [73, 97]}
    for wid in (1, 2, 3):
        reg.publish(make_report(wid, houses_written=written[wid]))
        reg.record_exit_code(wid, 0)
    reg.record_exit_code(0, 0)  # hung worker, joined, never reported
    summary = build_final_summary(
        reg,
        expected_house_indices=[1, 25, 49, 73, 97, 121, 145, 169],
        expected_episodes=200,
    )
    assert summary["complete"] is False
    assert summary["status"] == INCOMPLETE_MARKER
    assert summary["houses_missing"] == [1]
    assert summary["workers"]["silently_lost_workers"] == [0]


# --------------------------------------------------------------------------
# atomic final-summary publication
# --------------------------------------------------------------------------
def test_summary_publication_is_atomic(tmp_path):
    payload = {"status": "complete", "value": 1}
    path = write_summary_atomically(tmp_path, payload)
    assert path.name == "collection_summary.json"
    assert json.loads(path.read_text()) == payload
    # no temporary files left behind
    assert [p.name for p in tmp_path.iterdir()] == ["collection_summary.json"]


def test_summary_replacement_never_leaves_partial_file(tmp_path):
    write_summary_atomically(tmp_path, {"status": "complete", "n": 1})
    target = tmp_path / "collection_summary.json"
    original = target.read_text()

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        write_summary_atomically(tmp_path, {"bad": Unserialisable()})
    # the previously published summary is intact and no temp file survives
    assert target.read_text() == original
    assert [p.name for p in tmp_path.iterdir()] == ["collection_summary.json"]
