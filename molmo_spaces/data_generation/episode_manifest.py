"""Immutable episode manifest and episode-scoped RNG contract.

Why this module exists
----------------------
The previous hybrid-obstacle collection wrote 175 trajectories that represented
at most 75 defensibly distinct episodes: 50 replica classes of three members
each. The mechanism was structural, not incidental:

* ``BaseTaskSampler.seed_task_sampling`` seeds Python, NumPy and Torch from a
  single repeated ``config.seed``;
* exactly one task sampler is constructed per worker and it persists across every
  house that worker later claims;
* ``ParallelRolloutRunner.get_episode_seed`` returns that repeated sampler seed,
  so the "per-episode seed" recorded in the H5 was a worker constant;
* houses are claimed dynamically from a shared counter, and the hybrid-obstacle
  wraparound indices (1, 25, 49, ... 169) are aliases of one fumehood/red-cup
  task that exist only to parallelise execution;
* hazard presence was drawn at runtime with ``np.random.random() < OBSTACLE_P``
  off that same shared, order-dependent global stream.

The scientific episode was therefore *not* identified independently of worker
state, execution order or house alias.

The contract
------------
The content of candidate episode ``i`` is a deterministic function of exactly:

    manifest version, master seed, immutable candidate index,
    named random-stream identifier, and retry index where explicitly required.

It is a function of nothing else. In particular it does not depend on worker ID,
worker count, process scheduling, house-claim order, wraparound house alias,
output-file ordering, or any interruption/resume boundary.

Design notes
------------
* Stream seeds are derived from ``SeedSequence([master_seed, candidate_index,
  stream_id, retry_index])``. Every stream is derived independently from the full
  four-tuple. There is deliberately no chain of ``spawn()`` calls, because
  spawn-order semantics change the moment a new stream is inserted, and no
  reliance on the order in which streams are requested.
* Persistent identities use SHA-256. Python's built-in ``hash()`` is never used:
  it is salted per process (PYTHONHASHSEED) and is not stable across runs.
* This module imports only ``numpy`` and the standard library. It contains no
  MuJoCo, Warp or Torch import, so the contract can be tested without a
  simulator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Frozen contract constants. Changing any of these changes the manifest hash and
# invalidates every previously collected row.
# --------------------------------------------------------------------------- #

MANIFEST_VERSION = "hybrid_obstacle_independent_v2"

#: Fixed master seed. Declared before any simulation result was observed and
#: never changed afterwards.
MASTER_SEED = 20260725

#: Exactly one underlying fumehood/red-cup scene-template identity. The eight
#: wraparound house indices of the failed collection are NOT part of the v2
#: contract; the alias is a scheduling artefact and carries no scientific
#: meaning. Index 1 is the canonical (non-alias) index: the object pool is
#: indexed ``uid_pool[house_index % 24]`` and 1 == 1 (mod 24).
SCENE_TEMPLATE_ID = "fumehood_red_cup_v1"
SCENE_TEMPLATE_HOUSE_INDEX = 1

TOTAL_CANDIDATES = 160
HAZARD_PRESENT_COUNT = 120
HAZARD_ABSENT_COUNT = 40

#: Documented design probability. Retained for provenance only: the manifest
#: config never draws it. 120/160 == 0.75.
DESIGN_OBSTACLE_P = 0.75

#: Predeclared canonical dataset quotas (selection happens after collection, but
#: the rule is frozen now and never inspects rollout quality).
CANONICAL_HAZARD_PRESENT = 75
CANONICAL_HAZARD_ABSENT = 25

#: Predeclared trajectory-level split, locked now.
TRAIN_HAZARD_PRESENT = 60
TRAIN_HAZARD_ABSENT = 20
VAL_HAZARD_PRESENT = 15
VAL_HAZARD_ABSENT = 5

#: Bounded task-sampling retry budget per candidate row. A row that exhausts it
#: is recorded as failed; it is never silently replaced.
MAX_RETRIES_PER_ROW = 4

# --------------------------------------------------------------------------- #
# Named random streams. IDs are stable integers and must never be renumbered.
# New streams take the next free ID; inserting one does not perturb any existing
# stream, because derivation never depends on request order.
# --------------------------------------------------------------------------- #

STREAM_GLOBAL_COMPAT = 0
STREAM_TASK_SCENE = 1
STREAM_PLACEMENT = 2
STREAM_OBSTACLE = 3
STREAM_PLANNER_GRASP = 4
STREAM_ACTION_NOISE = 5
STREAM_CAMERA_LIGHT = 6
STREAM_PY_RANDOM = 7
STREAM_TORCH = 8
STREAM_RETRY = 9

STREAM_NAMES: dict[str, int] = {
    "global_compat": STREAM_GLOBAL_COMPAT,
    "task_scene": STREAM_TASK_SCENE,
    "placement": STREAM_PLACEMENT,
    "obstacle": STREAM_OBSTACLE,
    "planner_grasp": STREAM_PLANNER_GRASP,
    "action_noise": STREAM_ACTION_NOISE,
    "camera_light": STREAM_CAMERA_LIGHT,
    "py_random": STREAM_PY_RANDOM,
    "torch": STREAM_TORCH,
    "retry": STREAM_RETRY,
}

#: Reserved pseudo-stream used to derive the manifest-level hazard permutation.
#: It is outside the per-row stream namespace on purpose, so a future extra row
#: stream can never collide with the frozen hazard schedule.
STREAM_HAZARD_SCHEDULE = 1000

#: Terminal row outcomes. Every claimed row must reach exactly one of these.
OUTCOME_SUCCESS = "success"
OUTCOME_TASK_FAILURE = "task_failure"
OUTCOME_SAMPLING_FAILURE = "sampling_failure"
OUTCOME_INFRASTRUCTURE_FAILURE = "infrastructure_failure"
TERMINAL_OUTCOMES = frozenset(
    {
        OUTCOME_SUCCESS,
        OUTCOME_TASK_FAILURE,
        OUTCOME_SAMPLING_FAILURE,
        OUTCOME_INFRASTRUCTURE_FAILURE,
    }
)


class ManifestError(RuntimeError):
    """Raised when the manifest contract is violated."""


# --------------------------------------------------------------------------- #
# Canonical serialisation and hashing
# --------------------------------------------------------------------------- #


def canonical_json(payload: Any) -> str:
    """Byte-stable JSON: sorted keys, no incidental whitespace, ASCII-escaped."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_payload(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Seed derivation
# --------------------------------------------------------------------------- #


def stream_entropy(
    master_seed: int, candidate_index: int, stream_id: int, retry_index: int
) -> list[int]:
    """The root SeedSequence entropy tuple for one (row, stream, retry)."""
    for name, value in (
        ("master_seed", master_seed),
        ("candidate_index", candidate_index),
        ("stream_id", stream_id),
        ("retry_index", retry_index),
    ):
        if int(value) < 0:
            raise ManifestError(f"{name} must be non-negative, got {value}")
    return [int(master_seed), int(candidate_index), int(stream_id), int(retry_index)]


def derive_stream_seed(
    master_seed: int, candidate_index: int, stream_id: int, retry_index: int = 0
) -> dict[str, int]:
    """Derive one named stream's seed.

    Independent of the order in which streams are requested: the full entropy
    tuple is passed to ``SeedSequence`` directly, and no ``spawn()`` chain is
    used, so inserting a new stream ID later cannot perturb an existing one.

    Returns both a uint32 form (what ``np.random.seed`` and ``RandomState``
    accept) and a uint64 form (for ``Generator``/``PCG64`` construction).
    """
    entropy = stream_entropy(master_seed, candidate_index, stream_id, retry_index)
    state = np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    low, high = int(state[0]), int(state[1])
    return {
        "stream_id": int(stream_id),
        "seed_u32": low,
        "seed_u64": low | (high << 32),
    }


def derive_seed_map(
    master_seed: int, candidate_index: int, retry_index: int = 0
) -> dict[str, dict[str, int]]:
    """Derive every named stream for one candidate row at one retry index."""
    return {
        name: derive_stream_seed(master_seed, candidate_index, stream_id, retry_index)
        for name, stream_id in STREAM_NAMES.items()
    }


def episode_id_for(
    candidate_index: int,
    *,
    manifest_version: str = MANIFEST_VERSION,
    master_seed: int = MASTER_SEED,
    scene_template_id: str = SCENE_TEMPLATE_ID,
) -> str:
    """SHA-256 immutable episode ID.

    ``SHA256(manifest_version || master_seed || candidate_index || scene_template_id)``
    with an unambiguous field separator so no two distinct field tuples can
    produce the same preimage.
    """
    preimage = "\x1f".join(
        [manifest_version, str(int(master_seed)), str(int(candidate_index)), scene_template_id]
    )
    return sha256_text(preimage)


def hazard_schedule(
    total: int = TOTAL_CANDIDATES,
    present: int = HAZARD_PRESENT_COUNT,
    master_seed: int = MASTER_SEED,
) -> list[bool]:
    """The frozen 120-True / 40-False hazard assignment.

    Exactly ``present`` True values and ``total - present`` False values are
    created and then deterministically permuted from the fixed master seed. The
    result is committed to the manifest before any simulation, so hazard presence
    is never drawn at runtime for this config.
    """
    if present > total:
        raise ManifestError(f"present={present} exceeds total={total}")
    labels = np.array([True] * present + [False] * (total - present), dtype=bool)
    entropy = stream_entropy(master_seed, 0, STREAM_HAZARD_SCHEDULE, 0)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    return [bool(x) for x in labels[rng.permutation(total)]]


# --------------------------------------------------------------------------- #
# Predeclared selection and split
# --------------------------------------------------------------------------- #


def split_for_stratum_rank(hazard_present: bool, stratum_rank: int) -> tuple[str, int]:
    """Predeclared split for a row, from its stratum rank alone.

    Never inspects clearance, collision severity, trajectory length, action
    smoothness, proximity activation, image quality, planner retries, policy
    phase duration, or any model or audit score.
    """
    if hazard_present:
        train, val, canonical = TRAIN_HAZARD_PRESENT, VAL_HAZARD_PRESENT, CANONICAL_HAZARD_PRESENT
    else:
        train, val, canonical = TRAIN_HAZARD_ABSENT, VAL_HAZARD_ABSENT, CANONICAL_HAZARD_ABSENT
    if train + val != canonical:
        raise ManifestError("split quotas must sum to the canonical quota")
    if stratum_rank < train:
        return "train", stratum_rank
    if stratum_rank < canonical:
        return "val", stratum_rank - train
    return "reserve", stratum_rank - canonical


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ManifestRow:
    """One candidate scientific episode. Immutable once committed."""

    manifest_version: str
    master_seed: int
    candidate_index: int
    episode_id: str
    scene_template_id: str
    scene_template_house_index: int
    hazard_present: bool
    stratum_rank: int
    canonical_rank: int
    split: str
    split_rank: int
    max_retries: int
    root_seed_sequence_entropy: dict[str, list[int]]
    seed_map: dict[str, dict[str, int]]
    sensor_order_sha256: str
    robot_model_sha256: str
    env_config_sha256: str
    safety_cvae_contract: dict[str, Any]
    molmospaces_source_commit: str
    runtime_contract_sha256: str
    row_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "manifest_version": self.manifest_version,
            "master_seed": self.master_seed,
            "candidate_index": self.candidate_index,
            "episode_id": self.episode_id,
            "scene_template_id": self.scene_template_id,
            "scene_template_house_index": self.scene_template_house_index,
            "hazard_present": self.hazard_present,
            "stratum_rank": self.stratum_rank,
            "canonical_rank": self.canonical_rank,
            "split": self.split,
            "split_rank": self.split_rank,
            "max_retries": self.max_retries,
            "root_seed_sequence_entropy": self.root_seed_sequence_entropy,
            "seed_map": self.seed_map,
            "sensor_order_sha256": self.sensor_order_sha256,
            "robot_model_sha256": self.robot_model_sha256,
            "env_config_sha256": self.env_config_sha256,
            "safety_cvae_contract": self.safety_cvae_contract,
            "molmospaces_source_commit": self.molmospaces_source_commit,
            "runtime_contract_sha256": self.runtime_contract_sha256,
        }
        payload["row_sha256"] = self.row_sha256 or sha256_payload(payload)
        return payload


def _row_payload_hash(payload: dict[str, Any]) -> str:
    """Hash a row payload excluding its own hash field."""
    return sha256_payload({k: v for k, v in payload.items() if k != "row_sha256"})


def build_manifest(
    *,
    sensor_order_sha256: str,
    robot_model_sha256: str,
    env_config_sha256: str,
    safety_cvae_contract: dict[str, Any],
    molmospaces_source_commit: str,
    runtime_contract_sha256: str,
    scene_sha256: str,
    master_seed: int = MASTER_SEED,
    total: int = TOTAL_CANDIDATES,
    present: int = HAZARD_PRESENT_COUNT,
) -> dict[str, Any]:
    """Build the complete, frozen candidate manifest.

    Deterministic: given identical inputs this always produces the identical
    ``manifest_sha256``.
    """
    hazards = hazard_schedule(total=total, present=present, master_seed=master_seed)
    if sum(hazards) != present:
        raise ManifestError("hazard schedule lost its stratum counts under permutation")

    present_rank = 0
    absent_rank = 0
    rows: list[dict[str, Any]] = []

    for candidate_index in range(total):
        hazard_present = hazards[candidate_index]
        if hazard_present:
            stratum_rank = present_rank
            present_rank += 1
        else:
            stratum_rank = absent_rank
            absent_rank += 1

        split, split_rank = split_for_stratum_rank(hazard_present, stratum_rank)

        entropy = {
            name: stream_entropy(master_seed, candidate_index, stream_id, 0)
            for name, stream_id in STREAM_NAMES.items()
        }

        row = ManifestRow(
            manifest_version=MANIFEST_VERSION,
            master_seed=int(master_seed),
            candidate_index=candidate_index,
            episode_id=episode_id_for(candidate_index, master_seed=master_seed),
            scene_template_id=SCENE_TEMPLATE_ID,
            scene_template_house_index=SCENE_TEMPLATE_HOUSE_INDEX,
            hazard_present=hazard_present,
            stratum_rank=stratum_rank,
            # Canonical-selection rank IS the stratum rank: the canonical set is
            # the first 75 successful hazard-present rows and the first 25
            # successful hazard-absent rows by this predeclared rank.
            canonical_rank=stratum_rank,
            split=split,
            split_rank=split_rank,
            max_retries=MAX_RETRIES_PER_ROW,
            root_seed_sequence_entropy=entropy,
            seed_map=derive_seed_map(master_seed, candidate_index, 0),
            sensor_order_sha256=sensor_order_sha256,
            robot_model_sha256=robot_model_sha256,
            env_config_sha256=env_config_sha256,
            safety_cvae_contract=safety_cvae_contract,
            molmospaces_source_commit=molmospaces_source_commit,
            runtime_contract_sha256=runtime_contract_sha256,
        ).to_dict()
        rows.append(row)

    document: dict[str, Any] = {
        "schema": "hybrid_obstacle_candidate_manifest_v2",
        "manifest_version": MANIFEST_VERSION,
        "master_seed": int(master_seed),
        "total_candidates": total,
        "hazard_present_count": present,
        "hazard_absent_count": total - present,
        "design_obstacle_p": DESIGN_OBSTACLE_P,
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "scene_sha256": scene_sha256,
        "sensor_order_sha256": sensor_order_sha256,
        "robot_model_sha256": robot_model_sha256,
        "env_config_sha256": env_config_sha256,
        "runtime_contract_sha256": runtime_contract_sha256,
        "molmospaces_source_commit": molmospaces_source_commit,
        "safety_cvae_contract": safety_cvae_contract,
        "stream_ids": dict(STREAM_NAMES),
        "hazard_schedule_stream_id": STREAM_HAZARD_SCHEDULE,
        "max_retries_per_row": MAX_RETRIES_PER_ROW,
        "canonical_selection_rule": {
            "hazard_present": CANONICAL_HAZARD_PRESENT,
            "hazard_absent": CANONICAL_HAZARD_ABSENT,
            "basis": "first N successful rows by predeclared stratum rank",
            "forbidden_inputs": [
                "clearance",
                "collision_severity",
                "trajectory_length",
                "action_smoothness",
                "proximity_activation",
                "image_quality",
                "planner_retries",
                "policy_phase_duration",
                "any model or audit score",
            ],
        },
        "split_rule": {
            "train": {"hazard_present": TRAIN_HAZARD_PRESENT, "hazard_absent": TRAIN_HAZARD_ABSENT},
            "val": {"hazard_present": VAL_HAZARD_PRESENT, "hazard_absent": VAL_HAZARD_ABSENT},
        },
        "seed_derivation": (
            "SeedSequence([master_seed, candidate_index, stream_id, retry_index]).generate_state("
            "2, uint32); seed_u32 = state[0]; seed_u64 = state[0] | state[1] << 32"
        ),
        "rows": rows,
    }
    document["manifest_sha256"] = sha256_payload(
        {k: v for k, v in document.items() if k != "manifest_sha256"}
    )
    return document


# --------------------------------------------------------------------------- #
# Validation and loading
# --------------------------------------------------------------------------- #


def validate_manifest(document: dict[str, Any]) -> dict[str, Any]:
    """Validate a committed manifest end to end. Raises ``ManifestError``."""
    if document.get("manifest_version") != MANIFEST_VERSION:
        raise ManifestError(
            f"manifest_version {document.get('manifest_version')!r} != {MANIFEST_VERSION!r}"
        )
    recorded = document.get("manifest_sha256")
    recomputed = sha256_payload({k: v for k, v in document.items() if k != "manifest_sha256"})
    if recorded != recomputed:
        raise ManifestError(f"manifest hash mismatch: recorded {recorded}, recomputed {recomputed}")

    rows = document.get("rows") or []
    if len(rows) != document["total_candidates"]:
        raise ManifestError(f"expected {document['total_candidates']} rows, found {len(rows)}")

    indices, episode_ids, row_hashes = set(), set(), set()
    present = 0
    for row in rows:
        idx = row["candidate_index"]
        if idx in indices:
            raise ManifestError(f"duplicate candidate_index {idx}")
        indices.add(idx)

        expected_id = episode_id_for(
            idx,
            manifest_version=row["manifest_version"],
            master_seed=row["master_seed"],
            scene_template_id=row["scene_template_id"],
        )
        if row["episode_id"] != expected_id:
            raise ManifestError(f"row {idx}: episode_id does not match its derivation")
        if row["episode_id"] in episode_ids:
            raise ManifestError(f"duplicate episode_id at candidate_index {idx}")
        episode_ids.add(row["episode_id"])

        recomputed_row = _row_payload_hash(row)
        if row["row_sha256"] != recomputed_row:
            raise ManifestError(f"row {idx}: row_sha256 mismatch")
        if row["row_sha256"] in row_hashes:
            raise ManifestError(f"duplicate row_sha256 at candidate_index {idx}")
        row_hashes.add(row["row_sha256"])

        expected_seeds = derive_seed_map(row["master_seed"], idx, 0)
        if row["seed_map"] != expected_seeds:
            raise ManifestError(f"row {idx}: seed_map does not match its derivation")

        split, split_rank = split_for_stratum_rank(row["hazard_present"], row["stratum_rank"])
        if row["split"] != split or row["split_rank"] != split_rank:
            raise ManifestError(f"row {idx}: split assignment does not match the frozen rule")

        present += bool(row["hazard_present"])

    if present != document["hazard_present_count"]:
        raise ManifestError(
            f"hazard-present count {present} != {document['hazard_present_count']}"
        )
    if sorted(indices) != list(range(document["total_candidates"])):
        raise ManifestError("candidate indices are not exactly 0..N-1")
    return document


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and fully validate a committed manifest from disk."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    with open(path) as stream:
        document = json.load(stream)
    return validate_manifest(document)


def rows_by_episode_id(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in document["rows"]}


# --------------------------------------------------------------------------- #
# Bounded smoke subset
# --------------------------------------------------------------------------- #


def build_smoke_subset(document: dict[str, Any], per_stratum: int = 4) -> dict[str, Any]:
    """Deterministically select the lowest-ranked rows of each hazard stratum.

    No row may be replaced after its outcome is observed; the subset hash is
    recorded before simulation.
    """
    rows = document["rows"]
    present = sorted(
        (r for r in rows if r["hazard_present"]), key=lambda r: r["stratum_rank"]
    )[:per_stratum]
    absent = sorted(
        (r for r in rows if not r["hazard_present"]), key=lambda r: r["stratum_rank"]
    )[:per_stratum]
    selected = sorted(present + absent, key=lambda r: r["candidate_index"])

    subset: dict[str, Any] = {
        "schema": "hybrid_obstacle_manifest_v2_smoke_subset",
        "manifest_version": document["manifest_version"],
        "master_seed": document["master_seed"],
        "parent_manifest_sha256": document["manifest_sha256"],
        "selection_rule": (
            f"the {per_stratum} lowest-ranked hazard-present rows and the {per_stratum} "
            "lowest-ranked hazard-absent rows, by predeclared stratum rank"
        ),
        "per_stratum": per_stratum,
        "candidate_indices": [r["candidate_index"] for r in selected],
        "episode_ids": [r["episode_id"] for r in selected],
        "rows": selected,
    }
    subset["subset_sha256"] = sha256_payload(
        {k: v for k, v in subset.items() if k != "subset_sha256"}
    )
    return subset


def validate_smoke_subset(subset: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    recorded = subset.get("subset_sha256")
    recomputed = sha256_payload({k: v for k, v in subset.items() if k != "subset_sha256"})
    if recorded != recomputed:
        raise ManifestError(f"smoke subset hash mismatch: {recorded} != {recomputed}")
    if subset["parent_manifest_sha256"] != document["manifest_sha256"]:
        raise ManifestError("smoke subset was derived from a different manifest")
    by_id = rows_by_episode_id(document)
    for row in subset["rows"]:
        parent = by_id.get(row["episode_id"])
        if parent is None:
            raise ManifestError(f"smoke row {row['candidate_index']} is not in the manifest")
        if parent["row_sha256"] != row["row_sha256"]:
            raise ManifestError(
                f"smoke row {row['candidate_index']} does not match the committed manifest row"
            )
    return subset


# --------------------------------------------------------------------------- #
# Runtime seed installation
# --------------------------------------------------------------------------- #


def randomizer_base_seed(camera_light_seed_u32: int) -> int:
    """Base seed for the three owned randomizer ``RandomState`` objects.

    ``init_scene`` derives lighting/texture/dynamics seeds as base, base+1,
    base+2, so the base is kept clear of the uint32 ceiling that
    ``RandomState`` enforces.
    """
    return int(camera_light_seed_u32) % (2**32 - 8)


@dataclass
class InstalledSeedContract:
    """Record of what was actually installed for one row attempt."""

    episode_id: str
    candidate_index: int
    retry_index: int
    seed_map: dict[str, dict[str, int]]
    cuda_seeded: bool = False
    randomizers_reseeded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "candidate_index": self.candidate_index,
            "retry_index": self.retry_index,
            "seed_map": self.seed_map,
            "cuda_seeded": self.cuda_seeded,
            "randomizers_reseeded": sorted(self.randomizers_reseeded),
        }


def install_row_seed_contract(
    row: dict[str, Any],
    retry_index: int = 0,
    *,
    task_sampler: Any = None,
) -> InstalledSeedContract:
    """Install a row's RNG contract. Must run before any task-level random draw.

    Three things happen, in this order:

    1. The legacy global RNGs are seeded, because the overwhelming majority of
       draws on this path (theta, placement, planner offsets, per-step action
       noise via ``scipy.stats.truncnorm``) read the NumPy global state and are
       not worth a repository-wide Generator conversion.
    2. Torch, and CUDA when present, are seeded for the same reason.
    3. Components that already own an independent ``RandomState`` -- the
       lighting, texture and dynamics randomizers -- are reseeded explicitly.
       This is not optional: those objects are built once per scene load and
       would otherwise advance across rows, making a row's content a function of
       the rows previously executed by that worker.

    ``retry_index`` selects a fresh, independent state for every retry, so no
    retry inherits a partially consumed stream.
    """
    import random as _py_random

    master_seed = int(row["master_seed"])
    candidate_index = int(row["candidate_index"])
    seed_map = derive_seed_map(master_seed, candidate_index, retry_index)

    # 1. Legacy global RNGs.
    _py_random.seed(seed_map["py_random"]["seed_u32"])
    np.random.seed(seed_map["global_compat"]["seed_u32"])

    contract = InstalledSeedContract(
        episode_id=row["episode_id"],
        candidate_index=candidate_index,
        retry_index=retry_index,
        seed_map=seed_map,
    )

    # 2. Torch (imported lazily so this module stays simulator-free for tests).
    try:
        import torch

        torch.manual_seed(seed_map["torch"]["seed_u64"] % (2**63 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_map["torch"]["seed_u64"] % (2**63 - 1))
            contract.cuda_seeded = True
    except ImportError:  # pragma: no cover - torch is present in the real runtime
        pass

    if task_sampler is None:
        return contract

    # The sampler's own ``current_seed`` is the label the legacy code reads and
    # is also the base for randomizer construction on a fresh scene load.
    task_sampler.current_seed = seed_map["global_compat"]["seed_u32"]

    # 3. Explicitly reseed owned RandomState objects. Offsets are fixed so the
    #    three randomizers stay mutually independent within a row.
    #
    #    The same base seed is ALSO published as the sampler's randomizer
    #    override, so a row whose scene is built fresh (init_scene constructs the
    #    RandomState objects) and a row whose scene is cached (this reseed) end
    #    up in identical states. Without that, the first row a worker executes
    #    would differ from its later rows, and worker-count invariance would fail
    #    for exactly the rows that happen to land first.
    base = randomizer_base_seed(seed_map["camera_light"]["seed_u32"])
    task_sampler._randomizer_base_seed_override = base
    for offset, attribute in enumerate(
        ("lighting_randomizer", "texture_randomizer", "dynamics_randomizer")
    ):
        randomizer = getattr(task_sampler, attribute, None)
        if randomizer is None:
            continue
        randomizer.random_state = np.random.RandomState(base + offset)
        contract.randomizers_reseeded.append(attribute)

    return contract


#: Sampler attributes that accumulate across episodes and would otherwise make a
#: row depend on the rows previously executed by the same worker.
EPISODE_SCOPED_SAMPLER_STATE = (
    "object_synset_counter",
    "used_robot_positions",
    "_asset_failure_counts",
    "_dynamic_blacklist",
)

#: Caches that are deliberately preserved across rows. These are pure functions
#: of the installed dataset, not of any previously executed row.
PRESERVED_IMMUTABLE_CACHES = (
    "_dataset_index_map",
    "config",
)

#: Sampler attributes that must be invalidated so that every row rebuilds its
#: scene. See ``reset_episode_scoped_sampler_state`` for why this is mandatory.
SCENE_RELOAD_INVALIDATED = (
    "_last_loaded_house_index",
    "_light_base",
    "_headlight_base",
)

#: The previous row's theta. ``EnclosureSampler._obj_rest`` is called during
#: scene setup, BEFORE ``_draw_theta`` runs for the current row, and it branches
#: on whether ``self._theta`` is set:
#:
#:     th = getattr(self, "_theta", None)
#:     if not th:
#:         return (TUBE_X0 + 0.25, 0.0, SHELF_TOP_Z)      # no RNG draw
#:     ...
#:     y = float(np.random.uniform(-1, 1) * (th["ap_w"] / 2 - 0.05))   # one draw
#:
#: So a worker's FIRST row consumed one fewer global draw before ``_draw_theta``
#: than every later row, and it read the previous row's aperture width to boot.
#: Which rows land first is exactly what the worker count decides, which is how
#: this survived as a worker-count invariance failure until the A/B comparison.
STALE_EPISODE_THETA = "_theta"


def reset_episode_scoped_sampler_state(task_sampler: Any) -> list[str]:
    """Clear mutable sampler state so a row cannot depend on earlier rows.

    Two kinds of carry-over are removed.

    **Accumulated counters and blacklists.** The dynamic asset blacklist, the
    object-synset counter and the used-robot-position map all persist for the
    lifetime of a worker's sampler and change what later rows sample.

    **The scene cache.** This one is subtle and it is the reason the first
    implementation of this contract still failed worker-count invariance.
    ``BaseTaskSampler.sample_task`` branches on whether the requested house is
    already loaded, and the two branches are not equivalent: the load branch
    resets ``task_config`` from the experiment preset, resets the metadata adder,
    and runs the whole ``update_scene`` path (asset loading, MuJoCo compilation,
    object placement) which itself consumes global RNG draws. The reuse branch
    does none of that. So a row executed first by a worker consumed a different
    number of draws before ``_draw_theta`` than a row executed second -- and
    which rows land first is precisely what the worker count decides.

    Forcing a reload per row makes every row take one identical code path. It
    costs a scene compile per row; that is the price of the invariant.

    Returns the names actually cleared, for the row's audit record.
    """
    cleared: list[str] = []
    for name in EPISODE_SCOPED_SAMPLER_STATE:
        value = getattr(task_sampler, name, None)
        if value is None:
            continue
        if hasattr(value, "clear"):
            value.clear()
            cleared.append(name)

    # Per-house sample accounting must not carry over either.
    for name in ("_samples_per_current_house",):
        if hasattr(task_sampler, name):
            setattr(task_sampler, name, 0)
            cleared.append(name)

    # Drop the previous row's theta before scene setup can read it.
    if getattr(task_sampler, STALE_EPISODE_THETA, None) is not None:
        setattr(task_sampler, STALE_EPISODE_THETA, None)
        cleared.append(STALE_EPISODE_THETA)

    # Force the scene-load branch for every row.
    for name in SCENE_RELOAD_INVALIDATED:
        if hasattr(task_sampler, name):
            if name == "_last_loaded_house_index":
                task_sampler._last_loaded_house_index = None
            else:
                delattr(task_sampler, name)
            cleared.append(name)
    return cleared
