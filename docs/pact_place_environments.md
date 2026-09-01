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
| `FrankaSkinPactPlaceV1010FourObjectConfig` | V10.10 | V9.5 route layout with four live household objects and a two-lobe static pendant | 24 |

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

# V10.10: one episode for each family x side x pendant-pose cell.
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.pact_place_datagen_configs:FrankaSkinPactPlaceV1010FourObjectConfig
```

Each command creates a timestamped directory under the config's `output_dir`.
The public configs disable action noise, texture randomization, and lighting
randomization, and expose the wrist RGB/depth camera plus all 40 proximity
sensors. Their default `samples_per_house=1` makes the commands above small
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
    build_v1010_manifest_row,
)

v95_row = build_v95_manifest_row("F0_target_side_stagger", "left")
v1010_row = build_v1010_manifest_row(
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
