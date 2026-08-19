"""Phase-aware contact audit for the forked PACT pick-and-place corridor."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from molmo_spaces.tasks.pact_contact_audit import (
    TARGET_ROOT_PREFIX,
    classify_contact as classify_legacy_contact,
    robot_environment_contact_pairs,
)


CONTACT_CLASSES = (
    "grasp_target",
    "hazard_bar",
    "other_environment",
    "place_receptacle",
    "clutter",
)
TRAVERSAL_PHASES = ("inbound", "outbound", "placement", "other")
PLACE_ROOT_PREFIX = "place_receptacle"
CLUTTER_BODY_PREFIX = "pact_clutter_"
# Receptacle contact is expected while putting the cup down. preplace is mapped
# into this bucket by PactPlaceCorridorPolicy._traversal_phase.
PLACEMENT_EXEMPT_TRAVERSAL_PHASES = frozenset({"placement"})


def disallowed_place_receptacle_contact_entries(summary: dict[str, Any]) -> int:
    phases = summary["phase_contact_class_totals"]
    return int(
        sum(
            int(phases[phase]["place_receptacle"])
            for phase in TRAVERSAL_PHASES
            if phase not in PLACEMENT_EXEMPT_TRAVERSAL_PHASES
        )
    )


def classify_contact(pair: dict[str, Any]) -> str:
    blob = " ".join(
        str(pair.get(key, ""))
        for key in ("geom1", "geom2", "body1", "body2", "root1", "root2")
    )
    # Clutter before cavity_obj_ / grasp_target. The shared classifier tests
    # cavity_obj_ first and would silently exempt any clutter spawned under
    # that namespace; these bodies are named pact_clutter_* so this branch
    # is reachable, including carried-cup vs clutter pairs.
    if CLUTTER_BODY_PREFIX in blob:
        return "clutter"
    if PLACE_ROOT_PREFIX in blob:
        return "place_receptacle"
    return classify_legacy_contact(pair)


def _contact_pair_record(model, contact, geom1: int, geom2: int) -> dict[str, Any]:
    body1_id = int(model.geom_bodyid[geom1])
    body2_id = int(model.geom_bodyid[geom2])
    root1 = model.body(int(model.body_rootid[body1_id])).name or ""
    root2 = model.body(int(model.body_rootid[body2_id])).name or ""
    return {
        "geom1": model.geom(geom1).name or f"geom_{geom1}",
        "geom2": model.geom(geom2).name or f"geom_{geom2}",
        "body1": model.body(body1_id).name or "",
        "body2": model.body(body2_id).name or "",
        "root1": root1,
        "root2": root2,
        "distance_m": float(contact.dist),
    }


def _is_clutter_name(name: str) -> bool:
    return CLUTTER_BODY_PREFIX in str(name)


def _is_target_name(name: str) -> bool:
    return TARGET_ROOT_PREFIX in str(name)


def target_clutter_contact_pairs(env) -> list[dict[str, Any]]:
    """Cup-vs-clutter contacts the shared robot-environment filter drops.

    ``robot_environment_contact_pairs`` keeps a pair only when exactly one
    root is ``robot_0/``. The carried cup striking immovable clutter has
    neither side on the robot, so it would otherwise go unscored.
    """
    model, data = env.current_model, env.current_data
    pairs: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        if float(contact.dist) > 0.0:
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        record = _contact_pair_record(model, contact, geom1, geom2)
        names = (
            record["geom1"],
            record["geom2"],
            record["body1"],
            record["body2"],
            record["root1"],
            record["root2"],
        )
        has_clutter = any(_is_clutter_name(name) for name in names)
        has_target = any(_is_target_name(name) for name in names)
        if has_clutter and has_target:
            pairs.append(record)
    return pairs


def place_environment_contact_pairs(env) -> list[dict[str, Any]]:
    return robot_environment_contact_pairs(env) + target_clutter_contact_pairs(env)


class PactPlaceContactAudit:
    """Preserve the legacy classes and add a phase-aware receptacle class."""

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
        pairs = place_environment_contact_pairs(env)
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
            + self._pair_totals["clutter"]
        )
        return {
            "contact_taxonomy_version": "pact_place_robot_environment_v2",
            "legacy_contact_classes_unchanged": [
                "grasp_target",
                "hazard_bar",
                "other_environment",
            ],
            "added_contact_classes": ["place_receptacle", "clutter"],
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
            "place_receptacle_contact_exempt": False,
            "place_receptacle_exempt_during_placement_including_preplace": True,
            "place_receptacle_outside_placement_entries": int(
                sum(
                    self._phase_pair_totals[phase]["place_receptacle"]
                    for phase in TRAVERSAL_PHASES
                    if phase not in PLACEMENT_EXEMPT_TRAVERSAL_PHASES
                )
            ),
            "contact_frame_payload_retained": self._retain_contact_frames,
            "contact_frames": list(self._pairs_by_step),
        }
