"""Phase-aware contact audit for the forked PACT pick-and-place corridor."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from molmo_spaces.tasks.pact_contact_audit import (
    classify_contact as classify_legacy_contact,
    robot_environment_contact_pairs,
)


CONTACT_CLASSES = (
    "grasp_target",
    "hazard_bar",
    "other_environment",
    "place_receptacle",
)
TRAVERSAL_PHASES = ("inbound", "outbound", "placement", "other")
PLACE_ROOT_PREFIX = "place_receptacle"


def classify_contact(pair: dict[str, Any]) -> str:
    blob = " ".join(
        str(pair.get(key, ""))
        for key in ("geom1", "geom2", "body1", "body2", "root1", "root2")
    )
    if PLACE_ROOT_PREFIX in blob:
        return "place_receptacle"
    return classify_legacy_contact(pair)


class PactPlaceContactAudit:
    """Preserve the legacy classes and add an exempt receptacle plus phase split."""

    def __init__(self) -> None:
        self._retain_contact_frames = (
            os.environ.get("PACT_CONTACT_AUDIT_SUMMARY_ONLY") != "1"
        )
        self.reset()

    def reset(self) -> None:
        self._seen_times: set[float] = set()
        self._pair_totals = {key: 0 for key in CONTACT_CLASSES}
        self._frames_with = {key: 0 for key in CONTACT_CLASSES}
        self._maximum_penetration_depth_m = {key: 0.0 for key in CONTACT_CLASSES}
        self._first_step = {key: None for key in CONTACT_CLASSES}
        self._phase_pair_totals = {
            phase: {key: 0 for key in CONTACT_CLASSES} for phase in TRAVERSAL_PHASES
        }
        self._phase_frames_with = {
            phase: {key: 0 for key in CONTACT_CLASSES} for phase in TRAVERSAL_PHASES
        }
        self._phase = "other"
        self._policy_phase = "unknown"
        self._pairs_by_step: list[dict[str, Any]] = []

    def set_phase(self, traversal_phase: str, policy_phase: str) -> None:
        if traversal_phase not in TRAVERSAL_PHASES:
            raise ValueError(f"unknown traversal phase {traversal_phase!r}")
        self._phase = traversal_phase
        self._policy_phase = str(policy_phase)

    def observe(self, env, step: int) -> None:
        sim_time = round(float(env.current_data.time), 12)
        if sim_time in self._seen_times:
            return
        self._seen_times.add(sim_time)
        pairs = robot_environment_contact_pairs(env)
        frame_counts: defaultdict[str, int] = defaultdict(int)
        for pair in pairs:
            contact_class = classify_contact(pair)
            frame_counts[contact_class] += 1
            self._pair_totals[contact_class] += 1
            self._phase_pair_totals[self._phase][contact_class] += 1
            self._maximum_penetration_depth_m[contact_class] = max(
                self._maximum_penetration_depth_m[contact_class],
                max(0.0, -float(pair["distance_m"])),
            )
        for contact_class in CONTACT_CLASSES:
            if frame_counts[contact_class]:
                self._frames_with[contact_class] += 1
                self._phase_frames_with[self._phase][contact_class] += 1
                if self._first_step[contact_class] is None:
                    self._first_step[contact_class] = int(step)
        if pairs and self._retain_contact_frames:
            self._pairs_by_step.append(
                {
                    "step": int(step),
                    "sim_time_s": sim_time,
                    "traversal_phase": self._phase,
                    "policy_phase": self._policy_phase,
                    "pairs": pairs,
                    "contact_classes": {
                        key: int(frame_counts[key]) for key in CONTACT_CLASSES
                    },
                }
            )

    def summary(self) -> dict[str, Any]:
        non_target = (
            self._pair_totals["hazard_bar"]
            + self._pair_totals["other_environment"]
        )
        return {
            "contact_taxonomy_version": "pact_place_robot_environment_v1",
            "legacy_contact_classes_unchanged": [
                "grasp_target",
                "hazard_bar",
                "other_environment",
            ],
            "sampling_level": "every_2ms_control_physics_step_plus_episode_boundaries",
            "sample_count": len(self._seen_times),
            "contact_class_totals": dict(self._pair_totals),
            "frames_with_contact": dict(self._frames_with),
            "maximum_penetration_depth_m": dict(self._maximum_penetration_depth_m),
            "first_contact_step": dict(self._first_step),
            "phase_contact_class_totals": self._phase_pair_totals,
            "phase_frames_with_contact": self._phase_frames_with,
            "inbound_hazard_contact_frames": self._phase_frames_with["inbound"][
                "hazard_bar"
            ],
            "outbound_hazard_contact_frames": self._phase_frames_with["outbound"][
                "hazard_bar"
            ],
            "non_target_contact_entries": int(non_target),
            "collision_free": bool(non_target == 0),
            "place_receptacle_contact_exempt": True,
            "contact_frame_payload_retained": self._retain_contact_frames,
            "contact_frames": list(self._pairs_by_step),
        }
