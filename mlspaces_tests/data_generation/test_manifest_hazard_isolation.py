"""Proof that the manifest hazard path and the legacy Bernoulli path are isolated.

The failed collection drew hazard presence at runtime with
``np.random.random() < OBSTACLE_P`` off a shared, order-dependent global stream.
The manifest config must bypass that draw entirely, and every legacy config must
keep it byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest

from molmo_spaces.data_generation.episode_manifest import (
    DESIGN_OBSTACLE_P,
    HAZARD_PRESENT_COUNT,
    TOTAL_CANDIDATES,
)
from molmo_spaces.tasks.enclosure_reach import (
    ObstacleFumehoodPickCheckSampler,
    ObstacleFumehoodPickSampler,
)


def _bare(cls):
    """An instance with the class-level hazard state but no simulator setup."""
    return object.__new__(cls)


def test_legacy_obstacle_sampler_still_draws_the_bernoulli():
    sampler = _bare(ObstacleFumehoodPickSampler)
    assert sampler._forced_hazard_present is None

    np.random.seed(20260725)
    before = np.random.get_state()[2]
    drawn = [sampler._hazard_present_for_episode() for _ in range(2000)]
    after = np.random.get_state()[2]

    # The draw was actually taken: the global stream advanced.
    assert after != before
    # And it is Bernoulli(OBSTACLE_P), not a constant.
    rate = sum(drawn) / len(drawn)
    assert 0.70 < rate < 0.80
    assert ObstacleFumehoodPickSampler.OBSTACLE_P == DESIGN_OBSTACLE_P == 0.75


def test_legacy_bernoulli_is_reproducible_from_the_global_seed():
    sampler = _bare(ObstacleFumehoodPickSampler)
    np.random.seed(4242)
    first = [sampler._hazard_present_for_episode() for _ in range(50)]
    np.random.seed(4242)
    second = [sampler._hazard_present_for_episode() for _ in range(50)]
    assert first == second


def test_legacy_check_sampler_keeps_forcing_the_bar_present():
    sampler = _bare(ObstacleFumehoodPickCheckSampler)
    assert ObstacleFumehoodPickCheckSampler.OBSTACLE_P == 1.0
    np.random.seed(1)
    assert all(sampler._hazard_present_for_episode() for _ in range(200))


@pytest.mark.parametrize("hazard_present", [True, False])
def test_manifest_row_overrides_the_runtime_bernoulli(hazard_present):
    sampler = _bare(ObstacleFumehoodPickSampler)
    sampler.set_manifest_row({"hazard_present": hazard_present}, retry_index=0)

    np.random.seed(20260725)
    position_before = np.random.get_state()[2]
    results = [sampler._hazard_present_for_episode() for _ in range(500)]
    position_after = np.random.get_state()[2]

    # The manifest answer is returned every time, regardless of OBSTACLE_P...
    assert results == [hazard_present] * 500
    # ...and, critically, no draw was consumed. Consuming one would make the
    # rest of the episode depend on the hazard label rather than the row.
    assert position_after == position_before


def test_clearing_the_manifest_row_restores_legacy_behavior():
    sampler = _bare(ObstacleFumehoodPickSampler)
    sampler.set_manifest_row({"hazard_present": False})
    np.random.seed(7)
    assert sampler._hazard_present_for_episode() is False

    sampler.clear_manifest_row()
    assert sampler._forced_hazard_present is None
    np.random.seed(7)
    expected = bool(np.random.random() < ObstacleFumehoodPickSampler.OBSTACLE_P)
    np.random.seed(7)
    assert sampler._hazard_present_for_episode() == expected


def test_manifest_row_on_one_instance_does_not_leak_to_another():
    pinned = _bare(ObstacleFumehoodPickSampler)
    legacy = _bare(ObstacleFumehoodPickSampler)
    pinned.set_manifest_row({"hazard_present": False})

    assert pinned._forced_hazard_present is False
    assert legacy._forced_hazard_present is None
    # The class attribute is the default, so setting it on one instance must not
    # rebind it on the class.
    assert ObstacleFumehoodPickSampler._forced_hazard_present is None


def test_set_manifest_row_rejects_a_row_without_a_hazard_label():
    sampler = _bare(ObstacleFumehoodPickSampler)
    with pytest.raises(ValueError, match="hazard_present"):
        sampler.set_manifest_row({"candidate_index": 0})


def test_manifest_hazard_rate_matches_the_documented_design_probability():
    """120/160 == OBSTACLE_P, so the design intent is preserved without the draw."""
    assert HAZARD_PRESENT_COUNT / TOTAL_CANDIDATES == DESIGN_OBSTACLE_P == 0.75
