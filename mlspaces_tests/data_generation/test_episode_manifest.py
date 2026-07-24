"""Static proofs for the hybrid_obstacle_independent_v2 manifest and seed contract.

These tests run without a simulator: the contract modules deliberately import
only numpy and the standard library.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from molmo_spaces.data_generation.episode_manifest import (
    CANONICAL_HAZARD_ABSENT,
    CANONICAL_HAZARD_PRESENT,
    HAZARD_ABSENT_COUNT,
    HAZARD_PRESENT_COUNT,
    MANIFEST_VERSION,
    MASTER_SEED,
    STREAM_NAMES,
    TOTAL_CANDIDATES,
    TRAIN_HAZARD_ABSENT,
    TRAIN_HAZARD_PRESENT,
    VAL_HAZARD_ABSENT,
    VAL_HAZARD_PRESENT,
    ManifestError,
    build_manifest,
    build_smoke_subset,
    derive_seed_map,
    derive_stream_seed,
    episode_id_for,
    hazard_schedule,
    install_row_seed_contract,
    randomizer_base_seed,
    reset_episode_scoped_sampler_state,
    split_for_stratum_rank,
    validate_manifest,
    validate_smoke_subset,
)
from molmo_spaces.data_generation.row_ledger import (
    DuplicateClaimError,
    DuplicatePublicationError,
    RowLedger,
    RowLedgerError,
)

MOLMOSPACES_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MOLMOSPACES_ROOT.parents[1]
COMMITTED_MANIFEST = REPO_ROOT / "configs" / "hybrid_obstacle_candidate_manifest_v2.json"
COMMITTED_SMOKE = REPO_ROOT / "configs" / "hybrid_obstacle_manifest_v2_smoke8.json"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _manifest_kwargs() -> dict:
    return {
        "sensor_order_sha256": "s" * 64,
        "robot_model_sha256": "r" * 64,
        "env_config_sha256": "e" * 64,
        "safety_cvae_contract": {"safety_model_sha256": "c" * 64},
        "molmospaces_source_commit": "f" * 40,
        "runtime_contract_sha256": "t" * 64,
        "scene_sha256": "x" * 64,
    }


@pytest.fixture(scope="module")
def manifest() -> dict:
    return build_manifest(**_manifest_kwargs())


@pytest.fixture(scope="module")
def committed_manifest() -> dict:
    if not COMMITTED_MANIFEST.exists():
        pytest.skip("committed manifest is not present in this checkout")
    return json.loads(COMMITTED_MANIFEST.read_text())


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def test_manifest_has_160_unique_candidate_indices(manifest):
    indices = [row["candidate_index"] for row in manifest["rows"]]
    assert len(indices) == TOTAL_CANDIDATES == 160
    assert len(set(indices)) == 160
    assert sorted(indices) == list(range(160))


def test_manifest_has_160_unique_episode_ids(manifest):
    ids = [row["episode_id"] for row in manifest["rows"]]
    assert len(set(ids)) == 160
    assert all(len(episode_id) == 64 for episode_id in ids)


def test_manifest_has_160_unique_row_hashes(manifest):
    hashes = [row["row_sha256"] for row in manifest["rows"]]
    assert len(set(hashes)) == 160


def test_manifest_hazard_strata_are_exactly_120_and_40(manifest):
    present = sum(1 for row in manifest["rows"] if row["hazard_present"])
    absent = sum(1 for row in manifest["rows"] if not row["hazard_present"])
    assert (present, absent) == (HAZARD_PRESENT_COUNT, HAZARD_ABSENT_COUNT) == (120, 40)


def test_manifest_regeneration_is_deterministic():
    first = build_manifest(**_manifest_kwargs())
    second = build_manifest(**_manifest_kwargs())
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_hazard_schedule_is_deterministic_and_not_a_sorted_block():
    first = hazard_schedule()
    second = hazard_schedule()
    assert first == second
    assert sum(first) == 120
    # A permutation, not [True]*120 + [False]*40.
    assert first != [True] * 120 + [False] * 40


def test_manifest_records_the_fixed_master_seed(manifest):
    assert manifest["master_seed"] == MASTER_SEED == 20260725
    assert all(row["master_seed"] == 20260725 for row in manifest["rows"])


def test_canonical_and_split_ranks_are_fixed(manifest):
    present = sorted(
        (r for r in manifest["rows"] if r["hazard_present"]), key=lambda r: r["stratum_rank"]
    )
    absent = sorted(
        (r for r in manifest["rows"] if not r["hazard_present"]), key=lambda r: r["stratum_rank"]
    )
    assert [r["stratum_rank"] for r in present] == list(range(120))
    assert [r["stratum_rank"] for r in absent] == list(range(40))
    assert all(r["canonical_rank"] == r["stratum_rank"] for r in manifest["rows"])

    train_present = [r for r in present if r["split"] == "train"]
    val_present = [r for r in present if r["split"] == "val"]
    train_absent = [r for r in absent if r["split"] == "train"]
    val_absent = [r for r in absent if r["split"] == "val"]
    assert len(train_present) == TRAIN_HAZARD_PRESENT == 60
    assert len(val_present) == VAL_HAZARD_PRESENT == 15
    assert len(train_absent) == TRAIN_HAZARD_ABSENT == 20
    assert len(val_absent) == VAL_HAZARD_ABSENT == 5
    assert len(train_present) + len(val_present) == CANONICAL_HAZARD_PRESENT == 75
    assert len(train_absent) + len(val_absent) == CANONICAL_HAZARD_ABSENT == 25


def test_split_rule_never_depends_on_anything_but_stratum_rank():
    for rank in range(160):
        for hazard in (True, False):
            assert split_for_stratum_rank(hazard, rank) == split_for_stratum_rank(hazard, rank)
    assert split_for_stratum_rank(True, 0) == ("train", 0)
    assert split_for_stratum_rank(True, 59) == ("train", 59)
    assert split_for_stratum_rank(True, 60) == ("val", 0)
    assert split_for_stratum_rank(True, 74) == ("val", 14)
    assert split_for_stratum_rank(True, 75) == ("reserve", 0)
    assert split_for_stratum_rank(False, 19) == ("train", 19)
    assert split_for_stratum_rank(False, 20) == ("val", 0)
    assert split_for_stratum_rank(False, 25) == ("reserve", 0)


def test_validate_manifest_rejects_a_tampered_row(manifest):
    tampered = json.loads(json.dumps(manifest))
    tampered["rows"][7]["hazard_present"] = not tampered["rows"][7]["hazard_present"]
    with pytest.raises(ManifestError):
        validate_manifest(tampered)


def test_validate_manifest_rejects_a_tampered_manifest_hash(manifest):
    tampered = json.loads(json.dumps(manifest))
    tampered["manifest_sha256"] = "0" * 64
    with pytest.raises(ManifestError, match="manifest hash mismatch"):
        validate_manifest(tampered)


def test_committed_manifest_validates(committed_manifest):
    validate_manifest(committed_manifest)
    assert committed_manifest["manifest_version"] == MANIFEST_VERSION
    assert committed_manifest["master_seed"] == 20260725
    assert len(committed_manifest["rows"]) == 160


def test_committed_smoke_subset_is_the_lowest_ranked_four_of_each_stratum(committed_manifest):
    if not COMMITTED_SMOKE.exists():
        pytest.skip("committed smoke subset is not present in this checkout")
    subset = json.loads(COMMITTED_SMOKE.read_text())
    validate_smoke_subset(subset, committed_manifest)
    present = [r for r in subset["rows"] if r["hazard_present"]]
    absent = [r for r in subset["rows"] if not r["hazard_present"]]
    assert len(present) == len(absent) == 4
    assert sorted(r["stratum_rank"] for r in present) == [0, 1, 2, 3]
    assert sorted(r["stratum_rank"] for r in absent) == [0, 1, 2, 3]


def test_smoke_subset_hash_detects_row_substitution(manifest):
    subset = build_smoke_subset(manifest, per_stratum=4)
    validate_smoke_subset(subset, manifest)
    swapped = json.loads(json.dumps(subset))
    swapped["rows"][0] = json.loads(json.dumps(manifest["rows"][100]))
    with pytest.raises(ManifestError):
        validate_smoke_subset(swapped, manifest)


# --------------------------------------------------------------------------- #
# Seed contract
# --------------------------------------------------------------------------- #


def test_same_row_always_reconstructs_identical_stream_seeds():
    for candidate_index in (0, 7, 42, 159):
        assert derive_seed_map(MASTER_SEED, candidate_index, 0) == derive_seed_map(
            MASTER_SEED, candidate_index, 0
        )


def test_different_rows_have_distinct_stream_states():
    seen: dict[int, set[int]] = {stream: set() for stream in STREAM_NAMES.values()}
    for candidate_index in range(TOTAL_CANDIDATES):
        for name, stream_id in STREAM_NAMES.items():
            seed = derive_stream_seed(MASTER_SEED, candidate_index, stream_id, 0)["seed_u64"]
            assert seed not in seen[stream_id], f"{name} collides at candidate {candidate_index}"
            seen[stream_id].add(seed)


def test_streams_within_a_row_are_mutually_distinct():
    for candidate_index in (0, 55, 159):
        seeds = {
            name: derive_stream_seed(MASTER_SEED, candidate_index, sid, 0)["seed_u64"]
            for name, sid in STREAM_NAMES.items()
        }
        assert len(set(seeds.values())) == len(seeds)


def test_stream_derivation_does_not_depend_on_request_order():
    forward = {
        name: derive_stream_seed(MASTER_SEED, 11, sid, 0)
        for name, sid in sorted(STREAM_NAMES.items(), key=lambda kv: kv[1])
    }
    reverse = {
        name: derive_stream_seed(MASTER_SEED, 11, sid, 0)
        for name, sid in sorted(STREAM_NAMES.items(), key=lambda kv: -kv[1])
    }
    assert forward == reverse


def test_inserting_a_new_stream_id_does_not_perturb_existing_streams():
    before = derive_stream_seed(MASTER_SEED, 3, 5, 0)
    _ = derive_stream_seed(MASTER_SEED, 3, 99, 0)  # a hypothetical future stream
    after = derive_stream_seed(MASTER_SEED, 3, 5, 0)
    assert before == after


def test_worker_id_worker_count_and_house_alias_change_no_scientific_seed():
    """The derivation signature admits no worker or alias input at all.

    This is the structural guarantee: there is no argument through which a worker
    ID, a worker count or a wraparound house alias could reach a stream seed.
    """
    baseline = derive_seed_map(MASTER_SEED, 12, 0)
    for _worker_id in (0, 1, 2, 3, 7):
        for _worker_count in (1, 4, 8):
            for _house_alias in (1, 25, 49, 73, 97, 121, 145, 169):
                assert derive_seed_map(MASTER_SEED, 12, 0) == baseline

    import inspect

    signature = set(inspect.signature(derive_stream_seed).parameters)
    assert signature == {"master_seed", "candidate_index", "stream_id", "retry_index"}


def test_retry_index_changes_only_retry_derived_state():
    base = derive_seed_map(MASTER_SEED, 20, 0)
    retried = derive_seed_map(MASTER_SEED, 20, 1)
    # Every stream is re-derived at a new retry index, so no retry can inherit a
    # partially consumed state from the attempt before it.
    assert all(base[name] != retried[name] for name in STREAM_NAMES)
    # And the base state is unchanged by having asked for a retry.
    assert derive_seed_map(MASTER_SEED, 20, 0) == base


def test_episode_id_is_sha256_of_the_declared_fields():
    import hashlib

    expected = hashlib.sha256(
        "\x1f".join([MANIFEST_VERSION, str(MASTER_SEED), "5", "fumehood_red_cup_v1"]).encode()
    ).hexdigest()
    assert episode_id_for(5) == expected


def test_no_persistent_identity_uses_python_builtin_hash():
    """``hash()`` is salted per process, so it can never back a persistent ID.

    Checked by parsing the AST rather than grepping text, so a mention in a
    docstring is not a false positive and an aliased call is not a false
    negative.
    """
    import ast

    sources = [
        MOLMOSPACES_ROOT / "molmo_spaces" / "data_generation" / name
        for name in ("episode_manifest.py", "row_ledger.py", "manifest_runner.py")
    ]
    for source in sources:
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "hash", (
                    f"{source.name}:{node.lineno} calls the builtin hash()"
                )
            if isinstance(node, ast.Name) and node.id == "hash" and isinstance(
                getattr(node, "ctx", None), ast.Load
            ):
                # A bare reference could be aliased and called elsewhere.
                assert False, f"{source.name}:{node.lineno} references the builtin hash"


def _child_seed_map(queue, candidate_index):  # pragma: no cover - runs in a subprocess
    queue.put(derive_seed_map(MASTER_SEED, candidate_index, 0))


def test_seed_derivation_is_stable_across_processes():
    """PYTHONHASHSEED randomisation must not be able to reach a seed."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_child_seed_map, args=(queue, 33))
    process.start()
    child = queue.get(timeout=60)
    process.join(timeout=60)
    assert child == derive_seed_map(MASTER_SEED, 33, 0)


def test_installed_contract_reproduces_identical_draws():
    row = build_manifest(**_manifest_kwargs())["rows"][17]

    def draw():
        install_row_seed_contract(row, 0)
        import random as py_random

        return (
            np.random.random(),
            np.random.uniform(0.0, 1.0, size=3).tolist(),
            py_random.random(),
        )

    first = draw()
    # Deliberately disturb the global state between installs.
    np.random.random(1000)
    import random as py_random

    py_random.random()
    assert draw() == first


def test_installed_contract_differs_between_rows():
    rows = build_manifest(**_manifest_kwargs())["rows"]
    draws = []
    for row in (rows[0], rows[1], rows[2]):
        install_row_seed_contract(row, 0)
        draws.append(np.random.random())
    assert len(set(draws)) == 3


def test_randomizer_base_seed_stays_inside_the_uint32_domain():
    for value in (0, 1, 2**32 - 1, 2**32 - 3, 123456789):
        base = randomizer_base_seed(value)
        assert 0 <= base <= 2**32 - 9
        for offset in (0, 1, 2):
            np.random.RandomState(base + offset)  # must not raise


class _FakeRandomizer:
    def __init__(self) -> None:
        self.random_state = np.random.RandomState(0)


class _FakeSampler:
    def __init__(self) -> None:
        from collections import Counter, defaultdict

        self.current_seed = None
        self.object_synset_counter = Counter({"mug": 3})
        self.used_robot_positions = defaultdict(list, {"a": [np.zeros(3)]})
        self._asset_failure_counts = Counter({"uid": 2})
        self._dynamic_blacklist = {"uid"}
        self._samples_per_current_house = 7
        self.lighting_randomizer = _FakeRandomizer()
        self.texture_randomizer = _FakeRandomizer()
        self.dynamics_randomizer = None
        self._randomizer_base_seed_override = None
        self._env = "cached-scene"
        self._last_loaded_house_index = 1
        self._light_base = "stale"
        self._headlight_base = "stale"
        self._dataset_index_map = {"train": {}}
        self._theta = {"ap_w": 0.61, "target_frac": 0.5, "depth": 0.2}


def test_install_reseeds_owned_randomizers_and_publishes_the_override():
    row = build_manifest(**_manifest_kwargs())["rows"][4]
    sampler = _FakeSampler()
    contract = install_row_seed_contract(row, 0, task_sampler=sampler)

    base = randomizer_base_seed(contract.seed_map["camera_light"]["seed_u32"])
    assert sampler._randomizer_base_seed_override == base
    assert contract.randomizers_reseeded == ["lighting_randomizer", "texture_randomizer"]
    # A fresh scene load derives lighting=base, texture=base+1; the reseed path
    # must land on exactly the same states, or a worker's first row would differ
    # from its later rows.
    assert sampler.lighting_randomizer.random_state.randint(0, 2**31) == (
        np.random.RandomState(base).randint(0, 2**31)
    )
    assert sampler.texture_randomizer.random_state.randint(0, 2**31) == (
        np.random.RandomState(base + 1).randint(0, 2**31)
    )


def test_episode_scoped_state_reset_clears_cross_row_carryover():
    sampler = _FakeSampler()
    cleared = reset_episode_scoped_sampler_state(sampler)
    assert set(cleared) >= {
        "object_synset_counter",
        "used_robot_positions",
        "_asset_failure_counts",
        "_dynamic_blacklist",
        "_samples_per_current_house",
    }
    assert not sampler.object_synset_counter
    assert not sampler.used_robot_positions
    assert not sampler._asset_failure_counts
    assert not sampler._dynamic_blacklist
    assert sampler._samples_per_current_house == 0
    # Documented immutable caches survive.
    assert sampler._dataset_index_map == {"train": {}}


def test_episode_scoped_reset_forces_a_scene_reload_for_every_row():
    """Every row must take the scene-LOAD branch, never the scene-REUSE branch.

    The two branches are not equivalent -- the load branch resets task_config,
    resets the metadata adder and runs update_scene, which consumes global RNG
    draws before _draw_theta. Leaving the cache in place made a worker's first
    row differ from its later rows, and the worker count decides which rows land
    first. This is the exact defect the pre-freeze diagnostic exposed.
    """
    sampler = _FakeSampler()
    cleared = reset_episode_scoped_sampler_state(sampler)

    assert "_last_loaded_house_index" in cleared
    assert sampler._last_loaded_house_index is None, "scene reuse branch would be taken"
    # Lighting bases are recaptured from the freshly compiled model.
    assert not hasattr(sampler, "_light_base")
    assert not hasattr(sampler, "_headlight_base")


def test_episode_scoped_reset_drops_the_previous_rows_theta():
    """The stale-theta carry-over that broke the first A/B comparison.

    ``EnclosureSampler._obj_rest`` runs during scene setup, before the current
    row's ``_draw_theta``, and branches on whether ``self._theta`` is set:

        th = getattr(self, "_theta", None)
        if not th:
            return (TUBE_X0 + 0.25, 0.0, SHELF_TOP_Z)          # zero draws
        ...
        y = float(np.random.uniform(-1, 1) * (th["ap_w"] / 2 - 0.05))  # one draw

    So a worker's first row consumed one fewer global draw than its later rows,
    AND later rows read the previous row's aperture width. Both are cross-row
    dependencies, and the worker count decides which rows land first.
    """
    sampler = _FakeSampler()
    assert sampler._theta, "fixture must start with a stale theta"
    cleared = reset_episode_scoped_sampler_state(sampler)

    assert "_theta" in cleared
    assert not sampler._theta, "a truthy _theta takes the extra-draw branch"

    # The reset must be idempotent: a second row in a row must not re-report it.
    assert "_theta" not in reset_episode_scoped_sampler_state(sampler)


# --------------------------------------------------------------------------- #
# Row ledger / execution
# --------------------------------------------------------------------------- #


@pytest.fixture()
def rows() -> list[dict]:
    return build_manifest(**_manifest_kwargs())["rows"][:4]


def test_duplicate_row_claim_within_a_run_fails(tmp_path, rows):
    ledger = RowLedger(tmp_path, "run-1")
    row = rows[0]
    assert ledger.claim(row["episode_id"], row["row_sha256"], worker_id=0) is True
    with pytest.raises(DuplicateClaimError):
        ledger.claim(row["episode_id"], row["row_sha256"], worker_id=1)


def test_duplicate_output_publication_fails(tmp_path, rows):
    ledger = RowLedger(tmp_path, "run-1")
    row = rows[0]
    ledger.claim(row["episode_id"], row["row_sha256"], worker_id=0)
    for name in ("first.h5", "second.h5"):
        (tmp_path / name).write_bytes(b"payload")
    ledger.publish_trajectory(row["episode_id"], tmp_path / "first.h5")
    with pytest.raises(DuplicatePublicationError):
        ledger.publish_trajectory(row["episode_id"], tmp_path / "second.h5")


def test_failed_row_remains_present_in_the_outcome_ledger(tmp_path, rows):
    ledger = RowLedger(tmp_path, "run-1")
    row = rows[1]
    ledger.claim(row["episode_id"], row["row_sha256"], worker_id=0)
    ledger.finalize(row["episode_id"], status="task_failure", row=row, metadata={"reason": "no"})
    outcome = ledger.read_outcome(row["episode_id"])
    assert outcome["status"] == "task_failure"
    assert outcome["candidate_index"] == row["candidate_index"]
    assert outcome["row_sha256"] == row["row_sha256"]
    reconciliation = ledger.reconcile([row["episode_id"]])
    assert reconciliation["failed"] == [row["episode_id"]]
    assert reconciliation["ok"] is True


def test_a_row_cannot_be_finalised_twice(tmp_path, rows):
    ledger = RowLedger(tmp_path, "run-1")
    row = rows[2]
    ledger.claim(row["episode_id"], row["row_sha256"], worker_id=0)
    ledger.finalize(row["episode_id"], status="success", row=row)
    with pytest.raises(RowLedgerError, match="already finalised"):
        ledger.finalize(row["episode_id"], status="task_failure", row=row)


def test_finalize_rejects_a_non_terminal_status(tmp_path, rows):
    ledger = RowLedger(tmp_path, "run-1")
    with pytest.raises(RowLedgerError):
        ledger.finalize(rows[0]["episode_id"], status="in_progress", row=rows[0])


def test_resume_skips_completed_rows_and_reclaims_only_abandoned_ones(tmp_path, rows):
    first = RowLedger(tmp_path, "run-1")
    done, abandoned, untouched = rows[0], rows[1], rows[2]

    first.claim(done["episode_id"], done["row_sha256"], worker_id=0)
    first.finalize(done["episode_id"], status="success", row=done)
    first.claim(abandoned["episode_id"], abandoned["row_sha256"], worker_id=1)
    # run-1 dies here: `abandoned` holds a claim with no outcome.

    resumed = RowLedger(tmp_path, "run-2")
    ids = [row["episode_id"] for row in (done, abandoned, untouched)]
    reclaimed = resumed.reclaim_abandoned(ids)

    assert reclaimed == [abandoned["episode_id"]]
    assert resumed.is_complete(done["episode_id"]) is True
    assert resumed.claim(done["episode_id"], done["row_sha256"], worker_id=0) is False
    assert resumed.claim(abandoned["episode_id"], abandoned["row_sha256"], worker_id=0) is True
    assert resumed.claim(untouched["episode_id"], untouched["row_sha256"], worker_id=1) is True


def test_reconcile_reports_unfinalised_and_unexpected_rows(tmp_path, rows):
    ledger = RowLedger(tmp_path, "run-1")
    ids = [row["episode_id"] for row in rows]
    ledger.claim(rows[0]["episode_id"], rows[0]["row_sha256"], worker_id=0)
    ledger.finalize(rows[0]["episode_id"], status="success", row=rows[0])

    reconciliation = ledger.reconcile(ids)
    assert reconciliation["ok"] is False
    assert set(reconciliation["missing_outcome"]) == set(ids[1:])
    assert reconciliation["succeeded"] == [rows[0]["episode_id"]]

    (ledger.root / "an-unexpected-row").mkdir()
    assert ledger.reconcile(ids)["unexpected_row_dirs"] == ["an-unexpected-row"]


def test_published_payload_without_an_outcome_is_flagged(tmp_path, rows):
    ledger = RowLedger(tmp_path, "run-1")
    row = rows[0]
    ledger.claim(row["episode_id"], row["row_sha256"], worker_id=0)
    (tmp_path / "payload.h5").write_bytes(b"x")
    ledger.publish_trajectory(row["episode_id"], tmp_path / "payload.h5")
    reconciliation = ledger.reconcile([row["episode_id"]])
    assert reconciliation["published_without_outcome"] == [row["episode_id"]]
    assert reconciliation["ok"] is False


def test_missing_manifest_fails():
    from molmo_spaces.data_generation.episode_manifest import load_manifest

    with pytest.raises(ManifestError, match="manifest not found"):
        load_manifest("/nonexistent/manifest.json")


def test_incorrect_config_or_sensor_hash_fails(manifest):
    """The runner cross-checks every row against the configured hashes."""
    from molmo_spaces.data_generation.manifest_runner import ManifestRolloutRunner

    class _Config:
        expected_sensor_order_sha256 = "s" * 64
        expected_env_config_sha256 = "e" * 64
        expected_runtime_contract_sha256 = "t" * 64

    runner = ManifestRolloutRunner.__new__(ManifestRolloutRunner)
    runner.config = _Config()
    runner.verify_contract(manifest["rows"][:3])  # matching hashes pass

    _Config.expected_sensor_order_sha256 = "0" * 64
    with pytest.raises(ManifestError, match="sensor-order hash"):
        runner.verify_contract(manifest["rows"][:3])

    _Config.expected_sensor_order_sha256 = "s" * 64
    _Config.expected_env_config_sha256 = "0" * 64
    with pytest.raises(ManifestError, match="env/config hash"):
        runner.verify_contract(manifest["rows"][:3])

    _Config.expected_env_config_sha256 = "e" * 64
    _Config.expected_runtime_contract_sha256 = "0" * 64
    with pytest.raises(ManifestError, match="runtime-contract hash"):
        runner.verify_contract(manifest["rows"][:3])


def test_committed_manifest_regenerates_identically():
    """`build_hybrid_obstacle_manifest_v2.py --check` must pass in the checkout."""
    script = REPO_ROOT / "scripts" / "build_hybrid_obstacle_manifest_v2.py"
    if not script.exists():
        pytest.skip("root manifest builder is not present in this checkout")
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "submodules" / "molmospaces")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
