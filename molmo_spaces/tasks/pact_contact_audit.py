"""Contact taxonomy and episode audit for the PACT collision corridor.

Only robot-to-environment contacts at non-positive MuJoCo distance are counted.
Floor contacts and robot self-contacts are excluded. Contact with the selected
manipulation target is recorded but is not safety-relevant.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

CONTACT_CLASSES = ("grasp_target", "hazard_bar", "other_environment")
TARGET_ROOT_PREFIX = "cavity_obj_"
HAZARD_BODY_PREFIX = "pact_intrusion_"


def classify_contact(pair: dict[str, Any]) -> str:
    blob = " ".join(
        str(pair.get(key, ""))
        for key in ("geom1", "geom2", "body1", "body2", "root1", "root2")
    )
    if TARGET_ROOT_PREFIX in blob:
        return "grasp_target"
    if HAZARD_BODY_PREFIX in blob:
        return "hazard_bar"
    return "other_environment"


def robot_environment_contact_pairs(env) -> list[dict[str, Any]]:
    model, data = env.current_model, env.current_data
    pairs: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        if float(contact.dist) > 0.0:
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1_id = int(model.geom_bodyid[geom1])
        body2_id = int(model.geom_bodyid[geom2])
        root1 = model.body(int(model.body_rootid[body1_id])).name or ""
        root2 = model.body(int(model.body_rootid[body2_id])).name or ""
        robot1 = root1.startswith("robot_0/")
        robot2 = root2.startswith("robot_0/")
        if robot1 == robot2:
            continue
        environment_root = root2 if robot1 else root1
        if "floor" in environment_root.lower():
            continue
        pairs.append(
            {
                "geom1": model.geom(geom1).name or f"geom_{geom1}",
                "geom2": model.geom(geom2).name or f"geom_{geom2}",
                "body1": model.body(body1_id).name or "",
                "body2": model.body(body2_id).name or "",
                "root1": root1,
                "root2": root2,
                "distance_m": float(contact.dist),
            }
        )
    return pairs


class PactContactAudit:
    """Accumulate pair entries and frames-with-contact once per simulator time."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seen_times: set[float] = set()
        self._pair_totals = {key: 0 for key in CONTACT_CLASSES}
        self._frames_with = {key: 0 for key in CONTACT_CLASSES}
        self._first_step = {key: None for key in CONTACT_CLASSES}
        self._pairs_by_step: list[dict[str, Any]] = []

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
        for contact_class in CONTACT_CLASSES:
            if frame_counts[contact_class]:
                self._frames_with[contact_class] += 1
                if self._first_step[contact_class] is None:
                    self._first_step[contact_class] = int(step)
        if pairs:
            self._pairs_by_step.append(
                {
                    "step": int(step),
                    "sim_time_s": sim_time,
                    "pairs": pairs,
                    "contact_classes": {
                        key: int(frame_counts[key]) for key in CONTACT_CLASSES
                    },
                }
            )

    def summary(self) -> dict[str, Any]:
        non_target = self._pair_totals["hazard_bar"] + self._pair_totals["other_environment"]
        return {
            "contact_taxonomy_version": "pact_robot_environment_v1",
            "sampling_level": "every_2ms_control_physics_step_plus_episode_boundaries",
            "sample_count": len(self._seen_times),
            "contact_class_totals": dict(self._pair_totals),
            "frames_with_contact": dict(self._frames_with),
            "first_contact_step": dict(self._first_step),
            "non_target_contact_entries": int(non_target),
            "collision_free": bool(non_target == 0),
            "contact_frames": list(self._pairs_by_step),
        }
