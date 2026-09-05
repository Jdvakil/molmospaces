from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from molmo_spaces.data_generation.pact_place.contracts import (
    INTRUSION_SIDES,
    POSE_IDS,
    V95_LAYOUT_FAMILY_IDS,
    V107_SPACED_DECOR_SLOTS,
    V107_SPACED_ENVIRONMENT_VERSION,
    V1010_ACTIVE_SLOTS,
    V1010_ACTIVE_UIDS,
    V1010_ENVIRONMENT_VERSION,
    V1010_INACTIVE_SLOTS,
    V1010_SCENE_BY_POSE,
    build_v95_manifest_row,
    build_v107_spaced_manifest_row,
    build_v1010_manifest_row,
    load_v95_palette,
    load_v107_spaced_palette,
    sha256_payload,
    v95_cell,
    v107_spaced_cell,
    v1010_cell,
)
from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
from molmo_spaces.tasks.pact_place_speed import (
    apply_initial_free_space_speed_cap,
    plan_signature,
    verify_plan_matches_baseline,
)

SCENE_DIR = (
    Path(__file__).resolve().parents[2] / "molmo_spaces" / "data_generation" / "custom_scenes"
)

EXPECTED_V95_PAYLOAD_HASHES = {
    ("F0_target_side_stagger", "left"): (
        "02d54f43921610303eb09441883183a0a0c27543415f4f775c32b2d93f9aad9a"
    ),
    ("F0_target_side_stagger", "right"): (
        "70886e52b1e41e0c43e424b8655cc65c618e0f765d926bb042ea8d3e93d4b806"
    ),
    ("F1_inner_panel_stagger", "left"): (
        "c9231aefca71b5f21a6ea3d4752cd36835fe4da6cee9da93cb5e7c696f1afea0"
    ),
    ("F1_inner_panel_stagger", "right"): (
        "54f0e5d62c21677f8a3ecd7bdbd27e143159d1ebd04b4dc80e7e89051dc5e7f7"
    ),
    ("F2_outer_panel_stagger", "left"): (
        "b445f9ab4c15b9201d11c03a087561c06f61249a846cec641edab7d065d2f80d"
    ),
    ("F2_outer_panel_stagger", "right"): (
        "8825a919627c3deb27a439c78be50f5d34156401d4217e861cf4b06612b488ac"
    ),
    ("F3_aperture_side_stagger", "left"): (
        "6a5bc968517cd99d5126fcc36661d35233775e976fe74311c08c83eea9e42e61"
    ),
    ("F3_aperture_side_stagger", "right"): (
        "5a15b28db0e51fd96a814040f57a578e6f1f7a6752228002414e89528ec1aa23"
    ),
}

EXPECTED_V107_SPACED_PAYLOAD_HASHES = {
    ("F0_target_side_stagger", "left", "neg5"): (
        "d1ce4cb0ab0db8a9f7dbec387ba6d4068272c6dc5bae373a972887d10052cb07"
    ),
    ("F0_target_side_stagger", "left", "center"): (
        "7c7d21f87ee351ed840832f3dc5a91d265c2c81eca24591f932dd3b5e4666950"
    ),
    ("F0_target_side_stagger", "left", "pos5"): (
        "0b9dc0d3786ca3ddd97ff5b0cbb5e1543add81c17f7bb35a318a71cec1fab3e8"
    ),
    ("F0_target_side_stagger", "right", "neg5"): (
        "47198b1aae37446ad1d9a9b5c3ed231931356f623c3c5a28d0c610b5ad2e6962"
    ),
    ("F0_target_side_stagger", "right", "center"): (
        "9ad111b423e528b918328ef041cf05f6c827fcd339f809bfc9bf0f250ae5849d"
    ),
    ("F0_target_side_stagger", "right", "pos5"): (
        "c58eff9d6b632744167dccb913aab6cb3ffb232a8f6338dd986a45f8293585ef"
    ),
    ("F1_inner_panel_stagger", "left", "neg5"): (
        "7e1ba00ff25ed8c227ac773060687015b820556035fdbf7e9dffbb7e807c8b15"
    ),
    ("F1_inner_panel_stagger", "left", "center"): (
        "9e2ec3bfc0fc08b7db57c0c1a8013591c41b905008b57229805051a6bf806a7f"
    ),
    ("F1_inner_panel_stagger", "left", "pos5"): (
        "78eb9dff4ade33a712b8f741a15798d0c88bfb9b57d2849dc37e8a1737a038a1"
    ),
    ("F1_inner_panel_stagger", "right", "neg5"): (
        "691e13cc876e69c9f825e77e337f06240a7bc11136309f373e56660c3664b241"
    ),
    ("F1_inner_panel_stagger", "right", "center"): (
        "765e4ae0e41e6cc5d177bf5fcb6cac3cb53f7f465fa37eebd28efdbcfd22e5df"
    ),
    ("F1_inner_panel_stagger", "right", "pos5"): (
        "3b5ad2eca44db869581d2a6c6174098004fd8ab20fa6e0e986a540ec98d12024"
    ),
    ("F2_outer_panel_stagger", "left", "neg5"): (
        "9f802217ea162104cfeffc476d727c00311187bedea2ede95c66c33cf4d06486"
    ),
    ("F2_outer_panel_stagger", "left", "center"): (
        "07ef8fe73deb60494e54ddb33b65a2e3e9a5221557da6b6c296b78ed937368ef"
    ),
    ("F2_outer_panel_stagger", "left", "pos5"): (
        "7ad0a50f9bbe36fb784e3d803321e0a7ff98f1d14bc3a03d73b26350f407ad83"
    ),
    ("F2_outer_panel_stagger", "right", "neg5"): (
        "0fbf4797e614b39b5ab71c42a3959f773a80c68a38d182aecce4d013ee3a9716"
    ),
    ("F2_outer_panel_stagger", "right", "center"): (
        "da7130c56e1425a9d14755bfcdaab9d8827c0d29ff86fd30d141758a6088bf94"
    ),
    ("F2_outer_panel_stagger", "right", "pos5"): (
        "0b6d2bd6591ea36ba50da2a53a1be4aff92fd77e119c56d80d691c63e29af9d8"
    ),
    ("F3_aperture_side_stagger", "left", "neg5"): (
        "eeeb213fc524b10da31cb264ff5dd45ceb06b2a5bd54bc168e74c15af294bd4d"
    ),
    ("F3_aperture_side_stagger", "left", "center"): (
        "5b3fc9702cd151926eb0a3a86276de41c5d7684defa848cfe46e100f7a81e02a"
    ),
    ("F3_aperture_side_stagger", "left", "pos5"): (
        "63d8ab77741f83f28287edc6cbdf6d69174c112d4211e597fcc01197ff628815"
    ),
    ("F3_aperture_side_stagger", "right", "neg5"): (
        "b71763bd07bf68880b956edb07d4ad360ef654da18eb8d5ad2abd0c81de5325b"
    ),
    ("F3_aperture_side_stagger", "right", "center"): (
        "9c1bec85f2776c64fd4b63fd53c5dd43ce58107b5b8bf1c54cbb947a59d300cf"
    ),
    ("F3_aperture_side_stagger", "right", "pos5"): (
        "9415f16c67b5b71241ff4c6a7413b4edd3126bae6365bab4e172d4d049e9388d"
    ),
}

EXPECTED_SCENE_HASHES = {
    "pact_place_corridor_v2.xml": (
        "920860de9426fe15d607a6318fc81fb51012f4b82aa3d0e437a76f648e38be5d"
    ),
    "pact_place_corridor_v3.xml": (
        "f094d98b660630151394d6bc1e56700c8b66a17eae3d2ca841e8ab516c191296"
    ),
    "pact_place_corridor_v5.xml": (
        "5ac1ebd3e04f0bf509f6b8e11f0d086ac8c43bd550349762aba6c4129aebd61c"
    ),
    "pact_place_corridor_v10_7_neg5.xml": V1010_SCENE_BY_POSE["neg5"]["sha256"],
    "pact_place_corridor_v10_7_center.xml": V1010_SCENE_BY_POSE["center"]["sha256"],
    "pact_place_corridor_v10_7_pos5.xml": V1010_SCENE_BY_POSE["pos5"]["sha256"],
    "pact_place_corridor_v10_metadata.json": (
        "7df36c5e26364f9b5bd6da98e59108d7745c2dbd1270cc3ca73d307a656b809c"
    ),
}


def test_v95_palette_and_all_eight_cells_match_the_sealed_contract() -> None:
    palette = load_v95_palette()
    assert sha256_payload(palette) == (
        "97c61928bce3a1cad86e559408bfd0b6fc241e90e377414b9fb2bf6585f434a0"
    )
    assert len(palette["palette"]) == 8
    assert [v95_cell(index) for index in range(8)] == [
        (family, side) for family in V95_LAYOUT_FAMILY_IDS for side in INTRUSION_SIDES
    ]

    for cell, expected_hash in EXPECTED_V95_PAYLOAD_HASHES.items():
        row = build_v95_manifest_row(*cell)
        # These two fields are the public sampler's enclosing row. The sealed
        # v95_row_payload source is exactly the remaining payload.
        assert row.pop("family_id") == cell[0]
        assert row.pop("intrusion_side") == cell[1]
        assert sha256_payload(row) == expected_hash


def test_v1010_has_24_balanced_cells_and_exactly_four_live_objects() -> None:
    cells = [v1010_cell(index) for index in range(24)]
    assert cells == [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]
    assert len(set(cells)) == 24

    identity_hashes = set()
    for family, side, pose in cells:
        row = build_v1010_manifest_row(family, side, pose)
        assert tuple(row["pact_v1010_active_clutter_slots"]) == V1010_ACTIVE_SLOTS
        assert tuple(row["pact_v1010_inactive_clutter_slots"]) == V1010_INACTIVE_SLOTS
        assert row["pact_v1010_active_clutter_uids"] == V1010_ACTIVE_UIDS
        active = {
            item["palette_slot"]: item["uid"]
            for item in row["pact_clutter_layout"]["objects"]
            if item["palette_slot"] in V1010_ACTIVE_SLOTS
        }
        assert active == V1010_ACTIVE_UIDS
        identity_hashes.add(row["pact_v1010_identity_sha256"])
    assert identity_hashes == {"70f5cab5f76a58b82a616ba5e34251a3db950e18497b92e61539d3e18c5505a6"}


def test_v107_spaced_activates_all_eight_slots_and_matches_the_sealed_contract() -> None:
    palette = load_v107_spaced_palette()
    assert sha256_payload(palette) == (
        "c07077777595c1f1754125361bf0271a34d717b29afdf8ef18e4c4739d89754a"
    )
    # Two V9.5 vessels plus six naturally tall standing decor objects, none of
    # which is stretched to reach its height.
    assert len(palette["palette"]) == 8
    assert palette["selection_policy"]["stretch_meshes"] is False
    decor = [item for item in palette["palette"] if item["role"] == "decor"]
    assert tuple(sorted(item["slot"] for item in decor)) == V107_SPACED_DECOR_SLOTS
    assert all(item["dimensions_m"][2] >= 0.11 for item in decor)

    cells = [v107_spaced_cell(index) for index in range(24)]
    assert cells == [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]
    assert len(set(cells)) == 24

    for cell in cells:
        row = build_v107_spaced_manifest_row(*cell)
        # The whole bench is live here; nothing is parked outside the workspace.
        assert len(row["pact_clutter_layout"]["objects"]) == 8
        assert row["environment_version"] == V107_SPACED_ENVIRONMENT_VERSION
        # The pendant scene is inherited from V10.10 byte-for-byte.
        assert row["pact_v106_scene_sha256"] == V1010_SCENE_BY_POSE[cell[2]]["sha256"]
        assert sha256_payload(row) == EXPECTED_V107_SPACED_PAYLOAD_HASHES[cell]


def test_scene_files_are_exact_and_pendant_is_compiled_static() -> None:
    for filename, expected in EXPECTED_SCENE_HASHES.items():
        assert hashlib.sha256((SCENE_DIR / filename).read_bytes()).hexdigest() == expected

    expected_geoms = {
        "pact_clutter_mount_v106_lobe_0_g",
        "pact_clutter_mount_v106_lobe_1_g",
        "pact_clutter_mount_v106_stem_0_g",
        "pact_clutter_mount_v106_stem_1_g",
        "pact_clutter_mount_v106_crossbar_g",
    }
    for pose in POSE_IDS:
        root = ET.parse(SCENE_DIR / V1010_SCENE_BY_POSE[pose]["filename"]).getroot()
        body = root.find(".//body[@name='pact_clutter_mount_v106']")
        assert body is not None
        assert body.get("mocap") not in {"true", "1"}
        assert not body.findall(".//joint")
        assert not body.findall(".//freejoint")
        assert {geom.get("name") for geom in body.findall(".//geom")} == expected_geoms


def test_contact_taxonomy_separates_target_panel_clutter_and_pendant() -> None:
    def pair(name: str) -> dict[str, str]:
        return {
            "geom1": "robot_0/fr3_link6_collision",
            "geom2": f"{name}_g",
            "body1": "robot_0/fr3_link6",
            "body2": name,
            "root1": "robot_0/",
            "root2": name,
        }

    assert classify_contact(pair("cavity_obj_Cup_10")) == "grasp_target"
    assert classify_contact(pair("pact_intrusion_left")) == "hazard_bar"
    assert classify_contact(pair("pact_clutter_03/Plate_10")) == "clutter"
    assert classify_contact(pair("pact_clutter_mount_v106")) == "mounted_fixture"
    assert classify_contact(pair("place_receptacle")) == "place_receptacle"
    assert classify_contact(pair("hood_top")) == "other_environment"


def test_v1010_speed_amendment_changes_only_the_first_free_space_segment() -> None:
    import numpy as np

    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
        GripperAction,
        TCPMoveSegment,
        TCPMoveSequence,
    )

    sequence = object.__new__(TCPMoveSequence)
    sequence._move_segments = [
        TCPMoveSegment("pregrasp", np.eye(4), np.eye(4), 0.20),
        TCPMoveSegment("inbound_vessel_pass", np.eye(4), np.eye(4), 0.045),
    ]
    close = object.__new__(GripperAction)
    close.target_open = False
    primitives = [object(), sequence, close]
    before = plan_signature(primitives)
    record = apply_initial_free_space_speed_cap(primitives)
    after = plan_signature(primitives)

    assert record["applied"] is True
    assert record["n_segments_altered"] == 1
    assert record["speed_before_m_s"] == 0.20
    assert record["speed_after_m_s"] == 0.12
    assert sequence._move_segments[1].speed == 0.045
    comparison = verify_plan_matches_baseline(before, after)
    assert comparison["passed"] is True
    assert comparison["poses_identical"] is True
    assert comparison["n_speed_changes"] == 1


def test_public_configs_expose_only_the_supported_lineages(monkeypatch) -> None:
    monkeypatch.setenv("MLSPACES_ASSETS_DIR", str(Path.home() / ".cache" / "molmo-spaces"))
    from molmo_spaces.data_generation.config import pact_place_datagen_configs
    from molmo_spaces.data_generation.config.pact_place_datagen_configs import (
        FrankaSkinPactPlaceV5Config,
        FrankaSkinPactPlaceV95RealClutterConfig,
        FrankaSkinPactPlaceV107SpacedBenchConfig,
        FrankaSkinPactPlaceV1010FourObjectConfig,
        FrankaSkinPactPlaceV1011CMixedClutterConfig,
        FrankaSkinPactPlaceV1011DRandomizedClutterConfig,
    )

    cases = (
        (FrankaSkinPactPlaceV5Config, 2, 900, "PactPlaceV5Sampler"),
        (
            FrankaSkinPactPlaceV95RealClutterConfig,
            8,
            900,
            "PactPlaceCorridorV93Sampler",
        ),
        (
            FrankaSkinPactPlaceV107SpacedBenchConfig,
            24,
            1050,
            "PactPlaceCorridorV107SpacedBenchSampler",
        ),
        (
            FrankaSkinPactPlaceV1010FourObjectConfig,
            24,
            1050,
            "PactPlaceCorridorV1010FourObjectSampler",
        ),
        (
            FrankaSkinPactPlaceV1011CMixedClutterConfig,
            24,
            1050,
            "PactPlaceCorridorV1011C33PctTallerPrimitiveSampler",
        ),
        (
            FrankaSkinPactPlaceV1011DRandomizedClutterConfig,
            24,
            1050,
            "PactPlaceCorridorV1011DRandomizedLayoutSampler",
        ),
    )
    # The public surface is exactly these lineages, so an environment added
    # without a config, or a config added without a case here, fails loudly.
    assert set(pact_place_datagen_configs.__all__) == {
        case[0].__name__ for case in cases
    }
    for config_class, n_scenes, horizon, sampler_name in cases:
        config = config_class()
        assert config.task_horizon == horizon
        assert len(config.task_sampler_config.scene_xml_paths) == n_scenes
        assert config.task_sampler_config.task_sampler_class.__name__ == sampler_name
        assert config.robot_config.action_noise_config.enabled is False
        assert len(config.camera_config.cameras) == 41
        assert sum(camera.is_proximity_sensor for camera in config.camera_config.cameras) == 40


def test_public_task_module_does_not_export_failed_variants() -> None:
    from molmo_spaces.tasks import pact_place

    assert set(pact_place.__all__) == {
        "PactPlaceCorridorPolicy",
        "PactPlaceCorridorPolicyConfig",
        "PactPlaceCorridorTask",
        "PactPlaceCorridorV2Sampler",
        "PactPlaceCorridorV93Sampler",
        "PactPlaceCorridorV107SpacedBenchSampler",
        "PactPlaceCorridorV1010FourObjectSampler",
        "PactPlaceCorridorV1011C33PctTallerPrimitiveSampler",
        "PactPlaceCorridorV1011DRandomizedLayoutSampler",
        "PactPlaceV5Sampler",
        "PactPlaceV95RealClutterSampler",
    }
    assert not hasattr(pact_place, "PactPlaceCorridorV95LowWallSampler")
    assert not hasattr(pact_place, "PactPlaceCorridorV98PendantSampler")
    assert not hasattr(pact_place, "PactPlaceCorridorV99PendantSampler")
    assert not hasattr(pact_place, "PactPlaceCorridorV10CompoundPendantSampler")
    # V10.11a/b are required bases for V10.11c, so they exist on the module but
    # are not part of the supported surface.
    for intermediate in (
        "PactPlaceCorridorV1011MixedClutterSampler",
        "PactPlaceCorridorV1011BTallPrimitiveSampler",
    ):
        assert hasattr(pact_place, intermediate)
        assert intermediate not in pact_place.__all__


def test_reused_sampler_advances_auto_rows_but_never_rewrites_explicit_rows() -> None:
    from molmo_spaces.tasks.pact_place import (
        PactPlaceCorridorV107SpacedBenchSampler,
        PactPlaceCorridorV1010FourObjectSampler,
        PactPlaceV5Sampler,
        PactPlaceV95RealClutterSampler,
    )

    def bare(sampler_class):
        sampler = object.__new__(sampler_class)
        sampler._pact_manifest_row = None
        sampler._pact_manifest_row_is_explicit = False
        sampler._pact_auto_house_index = None
        sampler._house_inds = [0]
        sampler._house_iterator_index = 0
        return sampler

    def select_house(sampler, index: int) -> None:
        sampler._house_inds = [index]
        sampler._house_iterator_index = 0

    v5 = bare(PactPlaceV5Sampler)
    sides = []
    for index in range(2):
        select_house(v5, index)
        sides.append(v5._ensure_manifest_row()["intrusion_side"])
    assert sides == ["left", "right"]

    v95 = bare(PactPlaceV95RealClutterSampler)
    cells = []
    for index in range(8):
        select_house(v95, index)
        row = v95._ensure_manifest_row()
        cells.append((row["family_id"], row["intrusion_side"]))
    assert cells == [v95_cell(index) for index in range(8)]

    v1010 = bare(PactPlaceCorridorV1010FourObjectSampler)
    cells_with_pose = []
    for index in range(24):
        select_house(v1010, index)
        row = v1010._ensure_manifest_row()
        cells_with_pose.append((row["family_id"], row["intrusion_side"], row["pose_id"]))
    assert cells_with_pose == [v1010_cell(index) for index in range(24)]

    # A successor that ships its own palette must override both row hooks, or it
    # silently inherits a V9.5/V10.10 row and collects the wrong environment.
    spaced = bare(PactPlaceCorridorV107SpacedBenchSampler)
    spaced_cells = []
    for index in range(24):
        select_house(spaced, index)
        row = spaced._ensure_manifest_row()
        assert row["environment_version"] == V107_SPACED_ENVIRONMENT_VERSION
        assert row["environment_version"] != V1010_ENVIRONMENT_VERSION
        assert len(row["pact_clutter_layout"]["objects"]) == 8
        spaced_cells.append((row["family_id"], row["intrusion_side"], row["pose_id"]))
    assert spaced_cells == [v107_spaced_cell(index) for index in range(24)]
    # sample_task binds the auto row before current_house_index exists, so the
    # static-pendant hook has to agree with _ensure_manifest_row.
    for index in range(24):
        auto = PactPlaceCorridorV107SpacedBenchSampler._auto_manifest_row_for_house(index)
        assert auto["environment_version"] == V107_SPACED_ENVIRONMENT_VERSION
        assert (auto["family_id"], auto["intrusion_side"], auto["pose_id"]) == v107_spaced_cell(
            index
        )

    explicit = build_v1010_manifest_row(V95_LAYOUT_FAMILY_IDS[-1], "right", "pos5")
    v1010.set_pact_manifest_row(explicit)
    select_house(v1010, 0)
    assert v1010._ensure_manifest_row() == explicit
