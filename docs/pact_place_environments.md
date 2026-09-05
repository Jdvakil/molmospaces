# PACT pick-and-place environments

This release packages the three PACT place lineages used by the project as
self-contained MolmoSpaces environments. It deliberately does not register the
intermediate low-wall, offset-pendant, rectangular-box, or route-search
variants.

## Included environments

| Public config | Experiment name | Contents | Deterministic cells |
|---|---|---|---:|
| `FrankaSkinPactPlaceV5Config` | V5 | Hidden left/right intrusion panel, target cup, outside placement tray; no household clutter | 2 |
| `FrankaSkinPactPlaceV95RealClutterConfig` | V9.5 real-clutter lineage | V5 shell, active left/right panel, eight movable Objaverse household objects, including two route-bearing bottles | 8 |
| `FrankaSkinPactPlaceV107SpacedBenchConfig` | V10.7 spaced bench | V10.10 pendant scenes with all eight palette slots live, spread across the bench as naturally tall standing objects | 24 |
| `FrankaSkinPactPlaceV1010FourObjectConfig` | V10.10 | V9.5 route layout with four live household objects and a two-lobe static pendant | 24 |
| `FrankaSkinPactPlaceV1011PreviewOneBottleConfig` | V10.11 preview (one bottle) | V10.10 route with the household cut to a single inbound bottle pulled toward the robot, plus ten kitchen objects standing on the bench | 8 |
| `FrankaSkinPactPlaceV1011CMixedClutterConfig` | V10.11c | Six live bodies: three mesh props and three runtime MuJoCo primitives, two of them sampled near the target | 24 |
| `FrankaSkinPactPlaceV1011DRandomizedClutterConfig` | V10.11d | V10.11c clutter with every clutter position redrawn per episode | 24 |

The runtime marker for experiment V5 remains
`pact_place_corridor_v2`, because V5 was the experiment name rather than a
sampler-version rename. The V9.5 entry is the real-clutter V9.3 runtime lineage;
it is not `PactPlaceCorridorV95LowWallSampler`.

V10.10 crosses four layout families, two intrusion sides, and three pendant
poses (`neg5`, `center`, `pos5`). A pose selects one of three separately
compiled scenes before sampling. The pendant body has no joint, free joint,
mocap flag, or actuator, and therefore remains static for the entire episode.
Those scene filenames retain their historical `v10_7_*` names because V10.10
changes the active clutter set in the sampler while deliberately reusing the
frozen pendant geometry byte-for-byte.
The four live household objects are:

- slot 01: `Soap_Bottle_30` (outbound route vessel)
- slot 03: `Plate_10`
- slot 04: `Plate_22`
- slot 06: `Soap_Bottle_11` (inbound route vessel)

The remaining four V9.5 palette assets stay compiled but are parked outside the
workspace. This keeps the observation and asset-installation contract aligned
with the eight-object lineage.

V10.7 spaced bench reuses the same three pendant scenes and the same V9.5
palette assets as V10.10, but parks nothing: all eight slots are live. The two
route vessels move to a spaced arrangement, with the outbound bottle forward at
`x=0.68` as the route blocker and the inbound glass back in the otherwise empty
mid-bench at `x=1.02`, staggered in `y` toward the panel side. The remaining six
slots become standing decor on side rails at `y = +/-0.28` to `+/-0.36`.

Every decor object is a naturally tall accepted asset used at its measured size;
none is stretched to reach its height. Two soap-bottle meshes are declared as
`vase` and `pot` because the two route vessels already consume both `soapbottle`
entries under the V9 per-category cap of two. Decor keeps a 40 mm clearance from
everything, while the two vessels keep V9's 10 mm, so the bench is densely
populated without closing either route detour. The point of the lineage is
sensing coverage: with objects across the full table, the link-5/link-6
proximity skin has something to register throughout the episode rather than only
near the route.

V10.11c reuses the V10.10 pendant scenes and route unchanged and activates six
clutter bodies instead of four. Three are the existing mesh assets and three are
MuJoCo primitives built on the episode `MjSpec`, so no scene file changes and
the certified pendant geometry stays byte-for-byte identical:

- slot 01: cylinder, radius 0.045 m, height 0.32585 m (outbound route vessel)
- slot 03: `Plate_10`
- slot 04: `Plate_22`
- slot 06: `Soap_Bottle_11` (inbound route vessel)
- slot 08: cylinder, radius 0.035 m, height 0.23940 m (near-target)
- slot 09: box, 0.070 x 0.070 x 0.23940 m (near-target)

Slots 08 and 09 are drawn per episode in a bounded annular sector around the
target cup, sampled uniformly by area, so near-target clutter varies between
episodes. Because the route-bearing slot changed shape, V10.11c recomputes the
route and corridor predicates from the primitive's own half extents rather than
inheriting V9.5's numbers, and refuses any layout that closes either detour.
Its vessel-height ceiling is raised to 0.32585 m; every other lineage, V10.10
included, keeps the original 0.25 m limit.

V10.11d changes only *where* the clutter stands. V10.11c inherits the frozen
V9.5 layout, in which `Plate_10` sits at exactly `(0.980, -0.220)` and
`Plate_22` at `(1.090, +0.300)` in all eight family/side combinations, while
the two vessels move only by the inherited millimetre-scale jitter. V10.11d
redraws slots 01, 03, 04 and 06 per episode inside registered proposal boxes,
rejecting any candidate that leaves the bench shell or touches an
already-placed body, and additionally re-checking slot 01 against both route
predicates on every candidate because its admissible lateral window is roughly
60 mm wide and its sign depends on the panel side. The clutter identity is
unchanged from V10.11c.

The V10.11 preview is the one environment here whose bench is not fully
described by its manifest row. Its row is the V10.10 row over a scene that only
wraps `pact_place_corridor_v10_7_center.xml`, so families, sides, jitter and the
route layout are inherited unchanged, and it is published at the centre pose
only, giving eight cells. What it adds happens at sample time: three of the four
live household objects are parked off the bench, the surviving inbound bottle
(`Soap_Bottle_11`) is pulled 0.15 m toward the robot, and ten kitchen meshes are
stood on the bench as mocap bodies.

Those ten are positioned by a greedy first-fit against live geometry rather than
from frozen coordinates, so the ordering of `V1011_PREVIEW_STANDING_KITCHEN` and
of the candidate slots in `_v1011_preview_candidate_xy` is part of the
environment definition, not an implementation detail. A candidate is rejected if
it leaves the safe bench box, overlaps an already-placed body, or intrudes on the
arm's motion lane. Objects that find no slot are parked off the bench instead of
being dropped, so a run that stands seven of the ten is expected rather than a
failure. `Soap_Bottle_1` is pinned just behind the grasp target so the arm always
has something to sense on approach.

This lineage also differs in what it records. It is the only one that keeps the
table camera (`exo_camera_1`) alongside the wrist, because the published
episodes carry those streams. On the hub it appears as `data/v12`; that tag is a
release label, and the environment's own marker is
`pact_place_corridor_v10_11_preview_onebottle`. Because the hub numbers releases
while this repo names benches, the two cannot be derived from each other, so
`HUB_DATASET_TAGS` and `environment_version_for_hub_tag()` in `contracts.py` hold
the mapping, and the config is additionally registered as
`FrankaSkinPactPlaceV12Config`. Resolve a dataset through those rather than by
matching version numbers by eye: `v12` sits closest in name to V10.10 and
V10.11c/d, which are different benches. Its published manifests record
`sampler_class` as `PactPlaceCorridorV1010FourObjectSampler`, since the released
episodes were collected by overlaying that sampler rather than by subclassing
it, and that value is preserved so those manifests still resolve.

## Setup

Follow the repository's normal MuJoCo installation instructions, then point
MolmoSpaces at an asset directory. The resource manager installs missing scene,
robot, THOR, and Objaverse assets on first use.

```bash
git clone https://github.com/Jdvakil/molmospaces.git
cd molmospaces
python -m venv .venv
source .venv/bin/activate
pip install -e '.[mujoco]'

export MLSPACES_ASSETS_DIR="$HOME/.cache/molmospaces/assets"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Generate expert episodes

Pass the fully qualified config so only this config module needs to be imported:

```bash
# Experiment V5: one episode for each panel side.
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.pact_place_datagen_configs:FrankaSkinPactPlaceV5Config

# V9.5: one episode for each family x side cell.
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.pact_place_datagen_configs:FrankaSkinPactPlaceV95RealClutterConfig

# V10.7 spaced bench: all eight slots live, one episode per cell.
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.pact_place_datagen_configs:FrankaSkinPactPlaceV107SpacedBenchConfig

# V10.10: one episode for each family x side x pendant-pose cell.
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.pact_place_datagen_configs:FrankaSkinPactPlaceV1010FourObjectConfig

# V10.11 preview: one inbound bottle plus a standing kitchen, table camera on.
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.pact_place_datagen_configs:FrankaSkinPactPlaceV1011PreviewOneBottleConfig

# V10.11c: six live bodies, three of them primitives.
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.pact_place_datagen_configs:FrankaSkinPactPlaceV1011CMixedClutterConfig

# V10.11d: V10.11c clutter with every clutter position randomized.
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.pact_place_datagen_configs:FrankaSkinPactPlaceV1011DRandomizedClutterConfig
```

Each command creates a timestamped directory under the config's `output_dir`.
The public configs disable action noise, texture randomization, and lighting
randomization, and expose the wrist RGB/depth camera plus all 40 proximity
sensors; the V10.11 preview additionally records the table camera. Their default `samples_per_house=1` makes the commands above small
smoke/inspection runs. Increase `samples_per_house` in a derived config for a
larger collection rather than duplicating scene paths.

The expert is intentionally allowed to produce unsuccessful episodes unless a
derived config opts into success filtering. Real-clutter yield is part of the
environment, so silently retrying until every row succeeds changes the sampled
distribution.

## Programmatic selection

The cell builders are public and deterministic:

```python
from molmo_spaces.data_generation.pact_place.contracts import (
    build_v95_manifest_row,
    build_v107_spaced_manifest_row,
    build_v1010_manifest_row,
    build_v1011c_manifest_row,
    build_v1011d_manifest_row,
)

v95_row = build_v95_manifest_row("F0_target_side_stagger", "left")
v107_spaced_row = build_v107_spaced_manifest_row(
    "F0_target_side_stagger", "left", "center"
)
v1010_row = build_v1010_manifest_row(
    "F0_target_side_stagger", "left", "center"
)
v1011c_row = build_v1011c_manifest_row(
    "F0_target_side_stagger", "left", "center"
)
v1011d_row = build_v1011d_manifest_row(
    "F0_target_side_stagger", "left", "center"
)
```

`set_pact_manifest_row(row)` can be called on the corresponding task sampler
before `sample_task(...)` when an external manifest owns seeds and episode IDs.
Without an external row, each public sampler derives the correct frozen row from
its `house_index`.

## Scope

This repository owns the MuJoCo scenes, task samplers, scripted expert,
40-sensor observation surface, and contact taxonomy. ACT/PACT dataset
conversion, model training, checkpoint loading, and learned-policy evaluation
remain in the ACT repository; they are not silently duplicated here.
