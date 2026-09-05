"""Frozen contracts for the supported PACT place environments.

The palette and layout logic are packaged here so MolmoSpaces users do not need
the experiment-only ``prox_learning`` repository to construct V9.5 or V10.10
rows.  Values are copied from the sealed V9.5/V10.10 contracts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PALETTE_SOURCE_PATH = PACKAGE_DIR / "v95_palette_source.json"
SHELF_TOP_Z = 0.72
APERTURE_WIDTH = 0.85
TUBE_X0 = 0.58
MIN_DEPTH = 0.20
APERTURE_EDGE_RESERVE_M = 0.02
OUTBOUND_ENVELOPE_HALF_Y_M = 0.15
ROUTE_BLOCKER_SAFE_GAP_M = 0.04
MIN_FORCED_BOW_M = 0.04
NOMINAL_OUTBOUND_START_XY_M = (0.75, 0.02)
NOMINAL_OUTBOUND_END_XY_M = (0.44, 0.30)
PANEL_X_M = 0.615
PANEL_HALF_X_M = 0.055
PANEL_HALF_Y_M = 0.240
PANEL_INNER_FACE_Y_M = 0.100
PANEL_X_JITTER_MAX_M = 0.015
PANEL_FACE_JITTER_MAX_M = 0.005
PANEL_SAFE_GAP_M = 0.14
WORKSPACE_LOW_XYZ = (0.50, -0.43, SHELF_TOP_Z)
WORKSPACE_HIGH_XYZ = (1.34, 0.43, 1.50)
MIN_OBJECT_GAP_M = 0.010

V95_LAYOUT_FAMILIES = {
    "F0_target_side_stagger": ((0.570, -0.005), (0.680, 0.010)),
    "F1_inner_panel_stagger": ((0.575, 0.005), (0.690, -0.005)),
    "F2_outer_panel_stagger": ((0.565, -0.010), (0.680, 0.005)),
    "F3_aperture_side_stagger": ((0.580, 0.010), (0.690, -0.010)),
}
LAYOUT_FAMILIES: dict[str, dict[str, Any]] = {
    family: {
        "inbound_vessel_xy_m": inbound,
        "outbound_vessel_xy_m": outbound,
    }
    for family, (inbound, outbound) in V95_LAYOUT_FAMILIES.items()
}
DEFAULT_LAYOUT_FAMILY = next(iter(LAYOUT_FAMILIES))
V95_LAYOUT_FAMILY_IDS = tuple(V95_LAYOUT_FAMILIES)
INTRUSION_SIDES = ("left", "right")
POSE_IDS = ("neg5", "center", "pos5")
POSE_OFFSETS_M = {"neg5": -0.005, "center": 0.0, "pos5": 0.005}
V95_VESSEL_JITTER = (
    ({"01": -0.015, "06": -0.004}, {"01": -0.004, "06": 0.009}),
    ({"01": -0.005, "06": 0.003}, {"01": 0.003, "06": -0.006}),
    ({"01": 0.006, "06": -0.002}, {"01": -0.002, "06": 0.0045}),
    ({"01": 0.015, "06": 0.004}, {"01": 0.004, "06": -0.009}),
)

V1010_ENVIRONMENT_VERSION = "pact_place_corridor_v10_10_four_object"
V1010_ACTIVE_SLOTS = ("01", "03", "04", "06")
V1010_INACTIVE_SLOTS = ("00", "02", "05", "07")
V1010_ACTIVE_UIDS = {
    "01": "Soap_Bottle_30",
    "03": "Plate_10",
    "04": "Plate_22",
    "06": "Soap_Bottle_11",
}
V1010_ASSEMBLY = {"x_m": 0.800, "r_neg_m": 0.330, "r_pos_m": 0.300}
V1010_SCENE_BY_POSE = {
    "neg5": {
        "filename": "pact_place_corridor_v10_7_neg5.xml",
        "sha256": "df50679c749c6ad771d00023e73a08e0bfaf59d5391df9b42cf05de4ed7893a7",
    },
    "center": {
        "filename": "pact_place_corridor_v10_7_center.xml",
        "sha256": "b5a41d0d8934240b078f1cdbf3a6991b2e94a46558ddf1c9eae0119c8b8e138a",
    },
    "pos5": {
        "filename": "pact_place_corridor_v10_7_pos5.xml",
        "sha256": "762a5a4662a8fc0d31a3a0ee1135b347d6dd2c882daf4e65c2f706ab2d6fe565",
    },
}


V1011C_ENVIRONMENT_VERSION = "pact_place_corridor_v10_11c_33pct_taller_primitives"
V1011D_ENVIRONMENT_VERSION = "pact_place_corridor_v10_11d_randomized_clutter"
V1011_ACTIVE_SLOTS = ("01", "03", "04", "06", "08", "09")
V1011_INACTIVE_SLOTS = ("00", "02", "05", "07")
V1011_PRIMITIVE_SLOTS = ("01", "08", "09")
V1011_MESH_SLOTS = ("03", "04", "06")
V1011_NEAR_TARGET_SLOTS = ("08", "09")
# Slots 08/09 hold a valid metadata box before the sampler's single target draw
# replaces them with a target-relative placement.
V1011_NEAR_TARGET_PLACEHOLDER_XY_M = {"08": (0.90, -0.12), "09": (1.08, 0.12)}
# V10.11c heights: V10.11b's three primitive heights scaled by 1.33. XY
# footprints are identical to V10.11a/b.
V1011_PRIMITIVES = {
    "01": {
        "uid": "pact_primitive_cylinder_01",
        "role": "outbound_vessel",
        "category": "vase",
        "dimensions_m": [0.090, 0.090, 0.32585],
        "primitive": {
            "shape": "cylinder",
            "radius_m": 0.045,
            "height_m": 0.32585,
            "density_kg_m3": 1000.0,
            "rgba": [0.78, 0.56, 0.28, 1.0],
        },
    },
    "08": {
        "uid": "pact_primitive_cylinder_08",
        "role": "decor",
        "category": "primitive_cylinder",
        "dimensions_m": [0.070, 0.070, 0.23940],
        "primitive": {
            "shape": "cylinder",
            "radius_m": 0.035,
            "height_m": 0.23940,
            "density_kg_m3": 1000.0,
            "rgba": [0.26, 0.57, 0.82, 1.0],
        },
    },
    "09": {
        "uid": "pact_primitive_box_09",
        "role": "decor",
        "category": "primitive_box",
        "dimensions_m": [0.070, 0.070, 0.23940],
        "primitive": {
            "shape": "box",
            "size_m": [0.070, 0.070, 0.23940],
            "density_kg_m3": 1000.0,
            "rgba": [0.62, 0.36, 0.72, 1.0],
        },
    },
}

def load_v95_palette(path: Path = PALETTE_SOURCE_PATH) -> dict[str, Any]:
    """Return the sealed eight-object V9.5 palette."""
    document = json.loads(path.read_text())
    if document.get("authorizes_gate") is not False:
        raise ValueError("V9 palette unexpectedly authorizes the gate")
    palette = [dict(item) for item in list(document.get("palette") or [])]
    if len(palette) != 8:
        raise ValueError(f"V9.5 requires exactly eight palette slots, got {len(palette)}")
    by_slot = {str(item["slot"]): item for item in palette}
    records = {str(item.get("uid")): item for item in document.get("records") or []}
    source = records.get("Soap_Bottle_11")
    if not source or not source.get("accepted"):
        raise ValueError("V9.5 requires accepted Soap_Bottle_11 siting evidence")
    dimensions = [float(value) for value in source["collision_dimensions_m"]]
    by_slot["00"]["role"] = "decor"
    by_slot["06"] = {
        "slot": "06",
        "slot_class": "prop",
        "role": "inbound_vessel",
        "uid": "Soap_Bottle_11",
        "category": str(source["category"]),
        "dimensions_m": dimensions,
        "annotation_dimensions_m": [float(value) for value in source["dimensions_m"]],
        "half_m": [value / 2.0 for value in dimensions],
        "max_dimension_m": max(dimensions),
        "support": "shelf_standing",
        "quat_wxyz": [2**-0.5, 2**-0.5, 0.0, 0.0],
        "body_prefix": "pact_clutter_06/",
    }
    by_slot["01"]["role"] = "outbound_vessel"
    derived = dict(document)
    derived["palette"] = [by_slot[str(item["slot"])] for item in palette]
    derived["role_changes_from_source"] = {
        "00": "inbound_vessel_to_decor",
        "06": "decor_can_replaced_by_settled_Soap_Bottle_1_inbound_vessel",
    }
    derived["derived_for_environment_version"] = "pact_place_corridor_v9_5_raw_remediation"
    derived["v95_inbound_vessel_change"] = {
        "from_uid": "Soap_Bottle_1",
        "to_uid": "Soap_Bottle_11",
        "reason": "paired_side_raw_proximity_remediation",
    }
    return derived


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _vessel_for_role(palette: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [item for item in palette if str(item.get("role")) == role]
    if len(matches) != 1:
        raise ValueError(f"palette must contain exactly one {role}")
    return matches[0]


def route_blocker_metrics(layout: dict[str, Any]) -> dict[str, Any]:
    """Measure whether the nominal loaded route is blocked but laterally clearable."""
    blockers = [
        item
        for item in list(layout.get("objects") or [])
        if str(item.get("palette_slot")) == str(layout.get("route_blocker_slot"))
    ]
    if len(blockers) != 1:
        raise ValueError("layout must contain exactly one route blocker")
    blocker = blockers[0]
    center = tuple(map(float, blocker["center_m"][:2]))
    half = tuple(map(float, blocker["half_m"][:2]))
    start = NOMINAL_OUTBOUND_START_XY_M
    end = NOMINAL_OUTBOUND_END_XY_M
    delta_x = end[0] - start[0]
    t_cross = (center[0] - start[0]) / delta_x
    if not 0.02 < t_cross < 0.98:
        raise ValueError("route blocker is not crossed on the nominal loaded leg")
    crossing_y = start[1] + t_cross * (end[1] - start[1])
    bow_direction = str(layout.get("expected_bow_direction") or "")
    if bow_direction not in {"+y", "-y"}:
        raise ValueError("layout must declare an expected panel-selected bow direction")
    desired_side = 1.0 if bow_direction == "+y" else -1.0
    open_face_y = center[1] + desired_side * half[1]
    straight_clearance = desired_side * (crossing_y - open_face_y) - OUTBOUND_ENVELOPE_HALF_Y_M
    required_bow = ROUTE_BLOCKER_SAFE_GAP_M - straight_clearance
    waypoint_y = crossing_y + desired_side * required_bow
    lateral_limit = APERTURE_WIDTH / 2.0 - OUTBOUND_ENVELOPE_HALF_Y_M - APERTURE_EDGE_RESERVE_M
    return {
        "nominal_start_xy_m": list(start),
        "nominal_end_xy_m": list(end),
        "crossing_fraction": float(t_cross),
        "crossing_y_m": float(crossing_y),
        "straight_envelope_clearance_m": float(straight_clearance),
        "required_bow_m": float(required_bow),
        "planned_waypoint_y_m": float(waypoint_y),
        "lateral_limit_m": float(lateral_limit),
        "bow_direction": bow_direction,
        "direct_route_blocked": bool(straight_clearance < 0.0),
        "detour_admitted": bool(
            required_bow >= MIN_FORCED_BOW_M and abs(waypoint_y) <= lateral_limit
        ),
    }


def panel_corridor_metrics(layout: dict[str, Any]) -> dict[str, Any]:
    """Check that the active panel and centred bottle leave one safe lane."""
    side_name = str(layout.get("intrusion_side") or "")
    if side_name not in {"left", "right"}:
        raise ValueError("layout must bind exactly one left/right intrusion panel")
    side = 1.0 if side_name == "left" else -1.0
    expected_direction = "-y" if side_name == "left" else "+y"
    blockers = [
        item
        for item in list(layout.get("objects") or [])
        if str(item.get("palette_slot")) == str(layout.get("route_blocker_slot"))
    ]
    if len(blockers) != 1:
        raise ValueError("layout must contain exactly one route blocker")
    blocker = blockers[0]
    blocker_center = tuple(map(float, blocker["center_m"][:2]))
    blocker_half = tuple(map(float, blocker["half_m"][:2]))
    blocker_offset_toward_panel = side * blocker_center[1]
    worst_panel_face = PANEL_INNER_FACE_Y_M - PANEL_FACE_JITTER_MAX_M
    panel_lane_center = OUTBOUND_ENVELOPE_HALF_Y_M + PANEL_SAFE_GAP_M - worst_panel_face
    blocker_lane_center = (
        OUTBOUND_ENVELOPE_HALF_Y_M
        + ROUTE_BLOCKER_SAFE_GAP_M
        + blocker_half[1]
        - blocker_offset_toward_panel
    )
    required_lane_center = max(panel_lane_center, blocker_lane_center)
    lateral_limit = APERTURE_WIDTH / 2.0 - OUTBOUND_ENVELOPE_HALF_Y_M - APERTURE_EDGE_RESERVE_M

    panel_x_low = PANEL_X_M - PANEL_X_JITTER_MAX_M - PANEL_HALF_X_M
    panel_x_high = PANEL_X_M + PANEL_X_JITTER_MAX_M + PANEL_HALF_X_M
    blocker_x_low = blocker_center[0] - blocker_half[0]
    blocker_x_high = blocker_center[0] + blocker_half[0]
    x_overlap = min(panel_x_high, blocker_x_high) - max(panel_x_low, blocker_x_low)
    panel_blocker_surface_gap = worst_panel_face - (blocker_offset_toward_panel + blocker_half[1])
    return {
        "intrusion_side": side_name,
        "expected_bow_direction": expected_direction,
        "panel_active": bool(layout.get("legacy_panel_active")),
        "panel_x_range_with_jitter_m": [float(panel_x_low), float(panel_x_high)],
        "panel_inner_face_worst_case_m": float(worst_panel_face),
        "blocker_center_xy_m": list(blocker_center),
        "blocker_x_range_m": [float(blocker_x_low), float(blocker_x_high)],
        "panel_blocker_x_overlap_m": float(max(0.0, x_overlap)),
        "panel_blocker_surface_gap_m": float(panel_blocker_surface_gap),
        "required_lane_center_offset_m": float(required_lane_center),
        "lateral_limit_m": float(lateral_limit),
        "corridor_margin_m": float(lateral_limit - required_lane_center),
        "detour_admitted": bool(
            required_lane_center <= lateral_limit
            and (x_overlap <= 0.0 or panel_blocker_surface_gap >= MIN_OBJECT_GAP_M)
        ),
    }


def build_layout(
    palette_document: dict[str, Any],
    *,
    family_id: str = DEFAULT_LAYOUT_FAMILY,
    intrusion_side: str = "left",
    inbound_center_xy: tuple[float, float] | None = None,
    outbound_center_xy: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build a staggered bench layout from measured collision dimensions.

    ``inbound_center_xy`` and ``outbound_center_xy`` remain available to the
    measurement scripts, but production rows select one of the frozen layout
    families.  Slot 01 is the route blocker; slot 00 is a tall visual/sensing
    control parked behind and lateral to the target.
    """
    if family_id not in LAYOUT_FAMILIES:
        raise ValueError(f"unknown V9 layout family: {family_id}")
    palette = list(palette_document["palette"])
    if intrusion_side not in {"left", "right"}:
        raise ValueError(f"intrusion_side must be left or right, got {intrusion_side!r}")
    family = LAYOUT_FAMILIES[family_id]
    inbound_xy = tuple(inbound_center_xy or tuple(map(float, family["inbound_vessel_xy_m"])))
    blocker_xy = tuple(outbound_center_xy or tuple(map(float, family["outbound_vessel_xy_m"])))
    expected_bow_direction = "-y" if intrusion_side == "left" else "+y"
    positions: dict[str, tuple[float, float]] = {
        "00": (0.820, -0.350),
        "01": blocker_xy,
        "02": (0.840, 0.310),
        "03": (0.980, -0.220),
        "04": (1.090, 0.300),
        "05": (1.210, -0.280),
        "06": inbound_xy,
        "07": (1.060, 0.020),
        "08": (1.180, 0.090),
        "09": (1.170, -0.100),
        "10": (1.300, 0.310),
        "11": (1.300, -0.310),
    }
    objects: list[dict[str, Any]] = []
    for item in palette:
        slot = str(item["slot"])
        dimensions = [float(value) for value in item["dimensions_m"]]
        half = [value / 2.0 for value in dimensions]
        x, y = positions.get(slot, (0.68, 0.0))
        objects.append(
            {
                "palette_slot": slot,
                "uid": str(item["uid"]),
                "role": str(item["role"]),
                "category": str(item["category"]),
                "support": "bench_standing",
                "center_m": [float(x), float(y), float(SHELF_TOP_Z + half[2])],
                "half_m": half,
                "quat_wxyz": [float(value) for value in item["quat_wxyz"]],
                "size_class": (
                    "small"
                    if max(dimensions) <= 0.10
                    else "medium"
                    if max(dimensions) <= 0.18
                    else "large"
                ),
            }
        )
    layout = {
        "layout_id": f"v9_3_panel_{intrusion_side}_{family_id}",
        "layout_family_id": family_id,
        "intrusion_side": intrusion_side,
        "objects": objects,
        "inbound_vessel_slot": "06",
        "outbound_vessel_slot": "01",
        "route_blocker_slot": "01",
        "route_blocker_center_xy_m": list(map(float, blocker_xy)),
        "inbound_vessel_center_xy_m": list(map(float, inbound_xy)),
        "expected_bow_direction": expected_bow_direction,
        "shelf_top_z_m": SHELF_TOP_Z,
        "support": "bench_standing",
        "workspace_bounds_m": [list(WORKSPACE_LOW_XYZ), list(WORKSPACE_HIGH_XYZ)],
        "legacy_panel_active": True,
    }
    layout["nominal_route_metrics"] = route_blocker_metrics(layout)
    layout["panel_corridor_metrics"] = panel_corridor_metrics(layout)
    validate_layout(layout)
    return layout


def validate_layout(layout: dict[str, Any]) -> None:
    """Reject line layouts, overlaps, and objects outside the real bench."""
    objects = list(layout.get("objects") or [])
    if len(objects) < 8:
        raise ValueError("V9.2 layout must activate both vessels and 6-10 decor objects")
    centers = [tuple(map(float, item["center_m"])) for item in objects]
    if max(center[0] for center in centers) - min(center[0] for center in centers) < 0.40:
        raise ValueError("V9.2 objects collapsed into a transverse line")
    if len({round(center[0], 3) for center in centers}) < 6:
        raise ValueError("V9.2 layout lacks depth diversity")
    workspace_low = WORKSPACE_LOW_XYZ
    workspace_high = WORKSPACE_HIGH_XYZ
    for item, center in zip(objects, centers):
        half = tuple(map(float, item["half_m"]))
        if any(center[k] - half[k] < workspace_low[k] - 1e-6 for k in range(3)):
            raise ValueError(f"object {item['palette_slot']} escapes the low workspace bound")
        if any(center[k] + half[k] > workspace_high[k] + 1e-6 for k in range(3)):
            raise ValueError(f"object {item['palette_slot']} escapes the high workspace bound")
    for left_index, left in enumerate(objects):
        lc = tuple(map(float, left["center_m"]))
        lh = tuple(map(float, left["half_m"]))
        for right in objects[left_index + 1 :]:
            rc = tuple(map(float, right["center_m"]))
            rh = tuple(map(float, right["half_m"]))
            separated = any(abs(lc[k] - rc[k]) >= lh[k] + rh[k] + MIN_OBJECT_GAP_M for k in (0, 1))
            if not separated:
                raise ValueError(
                    f"layout objects overlap: {left['palette_slot']} and {right['palette_slot']}"
                )
    metrics = route_blocker_metrics(layout)
    if not metrics["direct_route_blocked"]:
        raise ValueError("route blocker does not obstruct the nominal loaded envelope")
    if not metrics["detour_admitted"]:
        raise ValueError("route blocker has no admitted lateral detour")
    if metrics["bow_direction"] != layout.get("expected_bow_direction"):
        raise ValueError("route blocker bow direction disagrees with its family contract")
    corridor = panel_corridor_metrics(layout)
    if layout.get("legacy_panel_active") is not True:
        raise ValueError("V9.2 requires one active legacy side panel")
    if corridor["expected_bow_direction"] != layout.get("expected_bow_direction"):
        raise ValueError("panel side and expected bow direction disagree")
    if not corridor["detour_admitted"]:
        raise ValueError("active panel and blocker close the only safe corridor")


def build_v95_layout(
    palette_document: dict[str, Any], *, family_id: str, intrusion_side: str
) -> dict[str, Any]:
    inbound_xy, outbound_xy = V95_LAYOUT_FAMILIES[family_id]
    layout = build_layout(
        palette_document,
        family_id=family_id,
        intrusion_side=intrusion_side,
        inbound_center_xy=inbound_xy,
        outbound_center_xy=outbound_xy,
    )
    layout["layout_id"] = f"v9_5_raw_{intrusion_side}_{family_id}"
    layout["layout_contract_version"] = "pact_place_v9_5_raw_remediation_v1"
    return layout


def build_v95_manifest_row(family_id: str, intrusion_side: str) -> dict[str, Any]:
    """Build the exact V9.5 row payload used by V10.7/V10.10."""
    if family_id not in V95_LAYOUT_FAMILY_IDS:
        raise ValueError(f"unknown V9.5 family {family_id!r}")
    if intrusion_side not in INTRUSION_SIDES:
        raise ValueError(f"unknown intrusion side {intrusion_side!r}")
    palette = load_v95_palette()
    layout = build_v95_layout(palette, family_id=family_id, intrusion_side=intrusion_side)
    jitter = V95_VESSEL_JITTER[V95_LAYOUT_FAMILY_IDS.index(family_id)]
    return {
        "family": family_id,
        "family_id": family_id,
        "layout_family_id": family_id,
        "layout_id": layout["layout_id"],
        "family_attempt": 0,
        "scene_template_house_index": 1,
        "max_sampling_retries": 12,
        "intrusion_side": intrusion_side,
        "clutter_x_jitter_m": dict(jitter[0]),
        "clutter_y_jitter_m": dict(jitter[1]),
        "panel_face_jitter_m": 0.0,
        "panel_x_jitter_m": 0.0,
        "target_x_jitter_m": 0.0,
        "target_y_jitter_m": 0.0,
        "pact_clutter_palette": copy.deepcopy(palette["palette"]),
        "pact_clutter_layout": layout,
    }


def v95_cell(index: int) -> tuple[str, str]:
    cells = [(family, side) for family in V95_LAYOUT_FAMILY_IDS for side in INTRUSION_SIDES]
    return cells[int(index) % len(cells)]


def v1010_cell(index: int) -> tuple[str, str, str]:
    cells = [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]
    return cells[int(index) % len(cells)]


def build_v1010_manifest_row(family_id: str, intrusion_side: str, pose_id: str) -> dict[str, Any]:
    if pose_id not in POSE_IDS:
        raise ValueError(f"unknown pendant pose {pose_id!r}")
    row = build_v95_manifest_row(family_id, intrusion_side)
    scene = V1010_SCENE_BY_POSE[pose_id]
    row.update(
        {
            "pose_id": pose_id,
            "pose_offset_m": POSE_OFFSETS_M[pose_id],
            "environment_version": V1010_ENVIRONMENT_VERSION,
            "sampler_class": "PactPlaceCorridorV1010FourObjectSampler",
            "pact_v106_x_m": V1010_ASSEMBLY["x_m"],
            "pact_v106_r_neg_m": V1010_ASSEMBLY["r_neg_m"],
            "pact_v106_r_pos_m": V1010_ASSEMBLY["r_pos_m"],
            "pact_v106_scene_sha256": scene["sha256"],
            "pact_v1010_scene_filename": scene["filename"],
            "pact_v1010_active_clutter_slots": list(V1010_ACTIVE_SLOTS),
            "pact_v1010_inactive_clutter_slots": list(V1010_INACTIVE_SLOTS),
            "pact_v1010_active_clutter_count": len(V1010_ACTIVE_SLOTS),
            "pact_v1010_active_clutter_uids": dict(V1010_ACTIVE_UIDS),
        }
    )
    active = [
        item
        for item in row["pact_clutter_layout"]["objects"]
        if str(item["palette_slot"]) in V1010_ACTIVE_SLOTS
    ]
    row["pact_v1010_identity_sha256"] = hashlib.sha256(
        canonical_json(
            [
                {
                    "palette_slot": str(item["palette_slot"]),
                    "uid": str(item["uid"]),
                    "role": str(item.get("role", "")),
                }
                for item in sorted(active, key=lambda value: str(value["palette_slot"]))
            ]
        ).encode()
    ).hexdigest()
    return row


def v1011_cell(index: int) -> tuple[str, str, str]:
    return v1010_cell(index)


def _v1011_primitive_palette_item(slot: str) -> dict[str, Any]:
    source = copy.deepcopy(V1011_PRIMITIVES[slot])
    dimensions = [float(value) for value in source["dimensions_m"]]
    return {
        "slot": slot,
        "slot_class": "prop",
        "role": source["role"],
        "uid": source["uid"],
        "category": source["category"],
        "dimensions_m": dimensions,
        "annotation_dimensions_m": list(dimensions),
        "half_m": [value / 2.0 for value in dimensions],
        "max_dimension_m": max(dimensions),
        "support": "shelf_standing",
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "body_prefix": f"pact_clutter_{slot}/",
        "primitive": source["primitive"],
    }


def _build_v1011_manifest_row(
    family_id: str,
    intrusion_side: str,
    pose_id: str,
    *,
    environment_version: str,
    sampler_class: str,
) -> dict[str, Any]:
    """The shared V10.11 row: six live bodies, three of them primitives.

    Slot 01 keeps the V9.5 route-blocker XY and becomes a primitive cylinder;
    slots 08/09 are new near-target primitives. The route predicates are
    recomputed from the primitive's own half extents rather than inheriting
    V9.5's numbers, because the route-bearing body changed shape.
    """
    if pose_id not in POSE_IDS:
        raise ValueError(f"unknown pendant pose {pose_id!r}")
    row = build_v1010_manifest_row(family_id, intrusion_side, pose_id)
    for key in (
        "pact_v1010_active_clutter_slots",
        "pact_v1010_inactive_clutter_slots",
        "pact_v1010_active_clutter_count",
        "pact_v1010_active_clutter_uids",
        "pact_v1010_identity_sha256",
    ):
        row.pop(key, None)

    palette = {str(item["slot"]): item for item in row["pact_clutter_palette"]}
    for slot in V1011_PRIMITIVE_SLOTS:
        palette[slot] = _v1011_primitive_palette_item(slot)
    row["pact_clutter_palette"] = [palette[key] for key in sorted(palette)]

    layout = row["pact_clutter_layout"]
    objects = {str(item["palette_slot"]): item for item in layout["objects"]}
    for slot in V1011_PRIMITIVE_SLOTS:
        source = palette[slot]
        half = list(source["half_m"])
        if slot == "01":
            centre_xy = list(objects["01"]["center_m"][:2])
        else:
            centre_xy = list(V1011_NEAR_TARGET_PLACEHOLDER_XY_M[slot])
        objects[slot] = {
            "palette_slot": slot,
            "uid": source["uid"],
            "role": source["role"],
            "category": source["category"],
            "support": "bench_standing",
            "center_m": [centre_xy[0], centre_xy[1], 0.72 + half[2]],
            "half_m": half,
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "size_class": "large" if slot == "01" else "medium",
            "primitive": copy.deepcopy(source["primitive"]),
            "target_relative_placeholder": slot in V1011_NEAR_TARGET_SLOTS,
        }
    layout["objects"] = [objects[key] for key in sorted(objects)]
    layout["route_blocker_center_xy_m"] = list(objects["01"]["center_m"][:2])
    layout["nominal_route_metrics"] = route_blocker_metrics(layout)
    layout["panel_corridor_metrics"] = panel_corridor_metrics(layout)
    if not layout["nominal_route_metrics"]["detour_admitted"]:
        raise ValueError("V10.11 primitive vessel does not admit the nominal detour")
    if not layout["panel_corridor_metrics"]["detour_admitted"]:
        raise ValueError("V10.11 primitive vessel closes the panel corridor")

    row.update(
        {
            "environment_version": environment_version,
            "sampler_class": sampler_class,
            "pact_v1011_active_clutter_slots": list(V1011_ACTIVE_SLOTS),
            "pact_v1011_inactive_clutter_slots": list(V1011_INACTIVE_SLOTS),
            "pact_v1011_active_clutter_count": len(V1011_ACTIVE_SLOTS),
            "pact_v1011_primitive_slots": list(V1011_PRIMITIVE_SLOTS),
            "pact_v1011_mesh_slots": list(V1011_MESH_SLOTS),
        }
    )
    active = [
        item
        for item in layout["objects"]
        if str(item["palette_slot"]) in V1011_ACTIVE_SLOTS
    ]
    row["pact_v1011_identity_sha256"] = hashlib.sha256(
        canonical_json(
            [
                {
                    "palette_slot": str(item["palette_slot"]),
                    "uid": str(item["uid"]),
                    "role": str(item.get("role", "")),
                    "primitive": item.get("primitive"),
                }
                for item in sorted(active, key=lambda value: str(value["palette_slot"]))
            ]
        ).encode()
    ).hexdigest()
    return row


def build_v1011c_manifest_row(
    family_id: str, intrusion_side: str, pose_id: str
) -> dict[str, Any]:
    return _build_v1011_manifest_row(
        family_id,
        intrusion_side,
        pose_id,
        environment_version=V1011C_ENVIRONMENT_VERSION,
        sampler_class="PactPlaceCorridorV1011C33PctTallerPrimitiveSampler",
    )


def build_v1011d_manifest_row(
    family_id: str, intrusion_side: str, pose_id: str
) -> dict[str, Any]:
    """V10.11d shares V10.11c's clutter exactly; only the sampler differs.

    The per-episode re-draw of slots 01/03/04/06 happens in the sampler, so the
    row is identical apart from its identity fields.
    """
    return _build_v1011_manifest_row(
        family_id,
        intrusion_side,
        pose_id,
        environment_version=V1011D_ENVIRONMENT_VERSION,
        sampler_class="PactPlaceCorridorV1011DRandomizedLayoutSampler",
    )


V107_SPACED_ENVIRONMENT_VERSION = "pact_place_corridor_v10_7_spaced_bench"
V107_SPACED_LAYOUT_CONTRACT_VERSION = "pact_place_v107_spaced_bench_v4"
# Decor keeps a wide gap from anything; the two vessels keep V9's 10 mm.
V107_SPACED_MIN_DECOR_GAP_M = 0.040
V107_SPACED_MIN_VESSEL_GAP_M = 0.010
V107_SPACED_MIN_DECOR_HEIGHT_M = 0.11
V107_SPACED_DECOR_SLOTS = ("02", "03", "04", "05", "07", "08")
# Naturally tall accepted UIDs; nothing here is stretched. The two soap-bottle
# meshes are labelled vase/pot because the vessels already consume both
# ``soapbottle`` entries under the V9 per-category cap of two.
V107_SPACED_TALL_DECOR: tuple[tuple[str, str, str], ...] = (
    ("02", "Soap_Bottle_3", "vase"),
    ("03", "Soap_Bottle_1", "pot"),
    ("04", "e3227ecd37d44cd6be1331941d9cfa2f", "spray can"),
    ("05", "663b5edc92a543668c1b602981e724a4", "can"),
    ("07", "5d13903e21044558bfb2bb7b72e76b4d", "can"),
    ("08", "Candle_4", "candle"),
)
# The glass sits in the empty mid-bench; one bottle stays forward as the route
# blocker. Y stagger follows the panel so the open lane stays on the panel side.
V107_SPACED_VESSEL_XY_M: dict[str, dict[str, tuple[float, float]]] = {
    "left": {"inbound": (1.02, -0.08), "outbound": (0.68, 0.02)},
    "right": {"inbound": (1.02, 0.08), "outbound": (0.68, -0.02)},
}
# Side rails around the glass and blocker so the bench is used, not empty.
V107_SPACED_DECOR_XY_M: dict[str, tuple[float, float]] = {
    "02": (0.72, 0.34),
    "03": (0.72, -0.34),
    "04": (0.95, 0.36),
    "05": (0.95, -0.36),
    "07": (1.22, 0.28),
    "08": (1.22, -0.28),
}


def _v107_spaced_slot(
    *, slot: str, role: str, uid: str, category: str, record: dict[str, Any]
) -> dict[str, Any]:
    dimensions = [float(value) for value in record["collision_dimensions_m"]]
    return {
        "slot": slot,
        "slot_class": "prop",
        "role": role,
        "uid": uid,
        "category": category,
        "dimensions_m": dimensions,
        "annotation_dimensions_m": [float(value) for value in record["dimensions_m"]],
        "half_m": [value / 2.0 for value in dimensions],
        "max_dimension_m": max(dimensions),
        "support": "shelf_standing",
        "quat_wxyz": [2**-0.5, 2**-0.5, 0.0, 0.0],
        "body_prefix": f"pact_clutter_{slot}/",
    }


def load_v107_spaced_palette() -> dict[str, Any]:
    """Two V9.5 vessels plus six tall standing decor objects at natural size."""
    base = load_v95_palette()
    records = {
        str(item.get("uid")): item
        for item in (base.get("records") or [])
        if item.get("accepted")
    }
    by_slot = {str(item["slot"]): dict(item) for item in base["palette"]}

    inbound = dict(by_slot["06"])
    inbound["role"] = "inbound_vessel"
    outbound = dict(by_slot["01"])
    outbound["role"] = "outbound_vessel"

    decor: list[dict[str, Any]] = []
    for slot, uid, category in V107_SPACED_TALL_DECOR:
        record = records.get(uid)
        if record is None:
            raise ValueError(f"spaced palette missing accepted UID {uid}")
        height = float(record["collision_dimensions_m"][2])
        if height < V107_SPACED_MIN_DECOR_HEIGHT_M:
            raise ValueError(f"decor {uid} is not standing-tall enough (h={height:.3f})")
        decor.append(
            _v107_spaced_slot(
                slot=slot, role="decor", uid=uid, category=category, record=record
            )
        )

    palette = sorted([inbound, outbound, *decor], key=lambda item: str(item["slot"]))
    if len(palette) != 8:
        raise ValueError(f"expected 8 palette entries, got {len(palette)}")
    return {
        "palette": palette,
        "derived_for_environment_version": V107_SPACED_ENVIRONMENT_VERSION,
        "layout_contract_version": V107_SPACED_LAYOUT_CONTRACT_VERSION,
        "selection_policy": {
            "stretch_meshes": False,
            "tall_standing_only": True,
            "active_decor_slots": list(V107_SPACED_DECOR_SLOTS),
            "parked_decor_slots": [],
            "min_decor_gap_m": V107_SPACED_MIN_DECOR_GAP_M,
            "min_vessel_gap_m": V107_SPACED_MIN_VESSEL_GAP_M,
        },
    }


def _v107_spaced_object(item: dict[str, Any], xy: tuple[float, float]) -> dict[str, Any]:
    dimensions = [float(value) for value in item["dimensions_m"]]
    half = [value / 2.0 for value in dimensions]
    x, y = xy
    return {
        "palette_slot": str(item["slot"]),
        "uid": str(item["uid"]),
        "role": str(item["role"]),
        "category": str(item["category"]),
        "support": "bench_standing",
        "center_m": [float(x), float(y), float(SHELF_TOP_Z + half[2])],
        "half_m": half,
        "quat_wxyz": [float(value) for value in item["quat_wxyz"]],
        "size_class": (
            "small"
            if max(dimensions) <= 0.10
            else "medium"
            if max(dimensions) <= 0.18
            else "large"
        ),
    }


def validate_v107_spaced_layout(layout: dict[str, Any]) -> None:
    """All eight objects stay on the bench and keep their role-dependent gap."""
    objects = list(layout.get("objects") or [])
    if len(objects) != 8:
        raise ValueError(f"spaced layout expects 8 active objects, got {len(objects)}")
    for item in objects:
        center = tuple(map(float, item["center_m"]))
        half = tuple(map(float, item["half_m"]))
        for axis in range(3):
            if center[axis] - half[axis] < WORKSPACE_LOW_XYZ[axis] - 1e-6:
                raise ValueError(f"slot {item['palette_slot']} escapes low workspace")
            if center[axis] + half[axis] > WORKSPACE_HIGH_XYZ[axis] + 1e-6:
                raise ValueError(f"slot {item['palette_slot']} escapes high workspace")
    vessel_roles = {"inbound_vessel", "outbound_vessel"}
    for index, left in enumerate(objects):
        lc = tuple(map(float, left["center_m"]))
        lh = tuple(map(float, left["half_m"]))
        for right in objects[index + 1 :]:
            rc = tuple(map(float, right["center_m"]))
            rh = tuple(map(float, right["half_m"]))
            both_vessels = left["role"] in vessel_roles and right["role"] in vessel_roles
            gap = V107_SPACED_MIN_VESSEL_GAP_M if both_vessels else V107_SPACED_MIN_DECOR_GAP_M
            if not any(abs(lc[k] - rc[k]) >= lh[k] + rh[k] + gap for k in (0, 1)):
                raise ValueError(
                    f"spaced overlap: {left['palette_slot']} vs {right['palette_slot']}"
                )


def build_v107_spaced_layout(
    palette_document: dict[str, Any], *, family_id: str, intrusion_side: str
) -> dict[str, Any]:
    if family_id not in LAYOUT_FAMILIES:
        raise ValueError(f"unknown family {family_id}")
    if intrusion_side not in INTRUSION_SIDES:
        raise ValueError(f"bad intrusion_side {intrusion_side}")
    # The family id still drives row metadata and vessel jitter, but the vessel
    # XY is spaced-bench: blocker forward, glass in the otherwise empty middle.
    vessel_xy = V107_SPACED_VESSEL_XY_M[intrusion_side]
    by_slot = {str(item["slot"]): item for item in palette_document["palette"]}
    objects = [
        _v107_spaced_object(by_slot["06"], tuple(map(float, vessel_xy["inbound"]))),
        _v107_spaced_object(by_slot["01"], tuple(map(float, vessel_xy["outbound"]))),
    ]
    for slot in V107_SPACED_DECOR_SLOTS:
        objects.append(_v107_spaced_object(by_slot[slot], V107_SPACED_DECOR_XY_M[slot]))

    layout = {
        "layout_id": f"v107_spaced_{intrusion_side}_{family_id}",
        "layout_family_id": family_id,
        "layout_contract_version": V107_SPACED_LAYOUT_CONTRACT_VERSION,
        "intrusion_side": intrusion_side,
        "objects": objects,
        "inbound_vessel_slot": "06",
        "outbound_vessel_slot": "01",
        "route_blocker_slot": "01",
        "route_blocker_center_xy_m": list(map(float, vessel_xy["outbound"])),
        "inbound_vessel_center_xy_m": list(map(float, vessel_xy["inbound"])),
        "expected_bow_direction": "-y" if intrusion_side == "left" else "+y",
        "shelf_top_z_m": SHELF_TOP_Z,
        "support": "bench_standing",
        "workspace_bounds_m": [list(WORKSPACE_LOW_XYZ), list(WORKSPACE_HIGH_XYZ)],
        "legacy_panel_active": True,
        "spaced_bench": True,
        "n_active_objects": len(objects),
        "n_parked_decor": 0,
    }
    validate_v107_spaced_layout(layout)
    return layout


def v107_spaced_cell(index: int) -> tuple[str, str, str]:
    return v1010_cell(index)


def build_v107_spaced_manifest_row(
    family_id: str, intrusion_side: str, pose_id: str
) -> dict[str, Any]:
    """Row for the published ``data/v107_spaced`` bench.

    The pendant assembly and scene hashes are inherited from V10.10 unchanged;
    only the bench population differs.
    """
    if family_id not in V95_LAYOUT_FAMILY_IDS:
        raise ValueError(f"unknown V9.5 family {family_id!r}")
    if intrusion_side not in INTRUSION_SIDES:
        raise ValueError(f"unknown intrusion side {intrusion_side!r}")
    if pose_id not in POSE_IDS:
        raise ValueError(f"unknown pendant pose {pose_id!r}")
    palette = load_v107_spaced_palette()
    layout = build_v107_spaced_layout(
        palette, family_id=family_id, intrusion_side=intrusion_side
    )
    jitter = V95_VESSEL_JITTER[V95_LAYOUT_FAMILY_IDS.index(family_id)]
    scene = V1010_SCENE_BY_POSE[pose_id]
    return {
        "family": family_id,
        "family_id": family_id,
        "layout_family_id": family_id,
        "layout_id": layout["layout_id"],
        "family_attempt": 0,
        "scene_template_house_index": 1,
        "max_sampling_retries": 12,
        "intrusion_side": intrusion_side,
        "pose_id": pose_id,
        "pose_offset_m": POSE_OFFSETS_M[pose_id],
        "clutter_x_jitter_m": dict(jitter[0]),
        "clutter_y_jitter_m": dict(jitter[1]),
        "panel_face_jitter_m": 0.0,
        "panel_x_jitter_m": 0.0,
        "target_x_jitter_m": 0.0,
        "target_y_jitter_m": 0.0,
        "environment_version": V107_SPACED_ENVIRONMENT_VERSION,
        "layout_contract_version": V107_SPACED_LAYOUT_CONTRACT_VERSION,
        "sampler_class": "PactPlaceCorridorV107SpacedBenchSampler",
        "pact_v106_x_m": V1010_ASSEMBLY["x_m"],
        "pact_v106_r_neg_m": V1010_ASSEMBLY["r_neg_m"],
        "pact_v106_r_pos_m": V1010_ASSEMBLY["r_pos_m"],
        "pact_v106_scene_sha256": scene["sha256"],
        "pact_v107_scene_filename": scene["filename"],
        "pact_v107_spaced_bench": True,
        "pact_clutter_palette": copy.deepcopy(palette["palette"]),
        "pact_clutter_layout": layout,
    }


__all__ = [
    "INTRUSION_SIDES",
    "POSE_IDS",
    "V95_LAYOUT_FAMILIES",
    "V95_LAYOUT_FAMILY_IDS",
    "V95_VESSEL_JITTER",
    "V1010_ACTIVE_SLOTS",
    "V1010_ACTIVE_UIDS",
    "V1010_ASSEMBLY",
    "V1010_ENVIRONMENT_VERSION",
    "V1010_INACTIVE_SLOTS",
    "V1010_SCENE_BY_POSE",
    "V1011C_ENVIRONMENT_VERSION",
    "V1011D_ENVIRONMENT_VERSION",
    "V1011_ACTIVE_SLOTS",
    "V1011_INACTIVE_SLOTS",
    "V1011_MESH_SLOTS",
    "V1011_NEAR_TARGET_SLOTS",
    "V1011_PRIMITIVES",
    "V1011_PRIMITIVE_SLOTS",
    "V107_SPACED_DECOR_SLOTS",
    "V107_SPACED_ENVIRONMENT_VERSION",
    "V107_SPACED_LAYOUT_CONTRACT_VERSION",
    "V107_SPACED_TALL_DECOR",
    "build_v95_layout",
    "build_v95_manifest_row",
    "build_v107_spaced_layout",
    "build_v107_spaced_manifest_row",
    "build_v1010_manifest_row",
    "build_v1011c_manifest_row",
    "build_v1011d_manifest_row",
    "load_v107_spaced_palette",
    "load_v95_palette",
    "sha256_payload",
    "v107_spaced_cell",
    "v95_cell",
    "v1010_cell",
    "v1011_cell",
]
