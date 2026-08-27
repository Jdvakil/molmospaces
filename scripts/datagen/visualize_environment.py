#!/usr/bin/env python3
"""Sample configured MolmoSpaces environments and render inspection images or MP4s.

This uses the same config and task sampler as data generation, then (by default) swaps in the
canonical 40-sensor hybrid full-body skin (`model_hybrid.xml` + ``FrankaSkinHybridCameraSystem``).
It does not execute a policy or save a trajectory.

Each sample writes four clean, text-free visual products:
  01_robot_scene.png          — scored hero + supporting scene views
  02_sensor_cones.png         — same views with every SPAD FOV cone overlaid
  03_cameras_and_sensors.png  — exo/wrist RGB + compact depth/RGB sensor atlases
  04_sensor_pointcloud.*      — skin-only 3D reconstruction (ply/npz/png/in-scene)

Examples:
    python scripts/datagen/visualize_environment.py --list
    python scripts/datagen/visualize_environment.py --all --show-hidden
    python scripts/datagen/visualize_environment.py \
        FrankaSkinHybridClutterPnPCheckConfig --format both --show-hidden
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import pkgutil
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make direct invocation from the MolmoSpaces checkout work without an editable install.
MOLMOSPACES_ROOT = Path(__file__).resolve().parents[2]
if str(MOLMOSPACES_ROOT) not in sys.path:
    sys.path.insert(0, str(MOLMOSPACES_ROOT))

# Rendering in this project is headless. Explicit shell values still take precedence.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem
from molmo_spaces.configs.robot_configs import FrankaSkinHybridRobotConfig
from molmo_spaces.data_generation.config_registry import (
    get_config_class,
    list_available_configs,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

CONFIG_PACKAGE = "molmo_spaces.data_generation.config"
DEFAULT_OUT = Path(ASSETS_DIR).resolve().parent / "experiments_output/default/environment_viz"
PLATE_WIDTH = 1920
PLATE_HEIGHT = 1080
PRESENTATION_VIEW_COUNT = 3
VIEW_PROBE_STEP_DEG = 15.0
SENSOR_FOVY_DEG = 45.0
CONE_LENGTH_M = 0.22
DEPTH_NEAR_M = 0.015
DEPTH_FAR_M = 4.0
PROX_TILE_PX = 92
PANEL_BG = (14, 16, 20)
PANEL_CARD = (22, 26, 32)
PANEL_STROKE = (42, 48, 58)
PANEL_ACCENT = (76, 201, 240)
PANEL_WARM = (255, 184, 108)
PRESENTATION_HEADLIGHT_AMBIENT = (0.45, 0.45, 0.45)
PRESENTATION_HEADLIGHT_DIFFUSE = (0.75, 0.75, 0.75)
PRESENTATION_HEADLIGHT_SPECULAR = (0.18, 0.18, 0.18)
# Hybrid skin order: group tiles by link for a readable anatomy map.
SENSOR_LINK_ORDER = (
    "link1",
    "link2",
    "link3",
    "link4",
    "link5_front",
    "link5_back",
    "link6",
)
SENSOR_LINK_COLORS = {
    "link1": (67, 97, 238),
    "link2": (76, 201, 240),
    "link3": (72, 219, 164),
    "link4": (170, 226, 80),
    "link5_front": (255, 209, 102),
    "link5_back": (255, 139, 89),
    "link6": (239, 71, 111),
    "other": (168, 178, 190),
}
RENDER_MAX_GEOM = 20000
# Datagen configs this project actually collects from (see README §7 / §12).
PROJECT_CONFIG_RE = re.compile(
    r"^FrankaSkin("
    r"Cabinet|Shelf|Clutter|Pillar|RealTable|RealHouse|Enclosure|"
    r"Fumehood|Panel|Cubby|House|Hybrid|ProxNecessity"
    r")"
)
FRANKA_SKIN_RE = re.compile(r"^FrankaSkin")


def _auto_import_configs() -> None:
    """Import every datagen config module, matching ``data_generation.main`` behavior."""
    package = importlib.import_module(CONFIG_PACKAGE)
    for module in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:
            logging.getLogger(__name__).warning("Could not import config module %s: %s", module.name, exc)


def _load_config_class(config_ref: str):
    """Resolve a registry name or ``module:ConfigName`` reference."""
    if ":" in config_ref:
        module_name, config_name = config_ref.split(":", 1)
        importlib.import_module(module_name)
    else:
        config_name = config_ref
        _auto_import_configs()
    return get_config_class(config_name), config_name


def _proximity_sensor_names(config) -> list[str]:
    camera_config = getattr(config, "camera_config", None)
    cameras = getattr(camera_config, "cameras", ()) if camera_config is not None else ()
    return [camera.name for camera in cameras if getattr(camera, "is_proximity_sensor", False)]


def _configured_houses(config) -> list[int]:
    sampler_config = config.task_sampler_config
    houses = getattr(sampler_config, "house_inds", None)
    if houses:
        return [int(house) for house in houses]
    paths = getattr(sampler_config, "scene_xml_paths", None)
    if paths:
        return list(range(len(paths)))
    return [0]


def _scene_path_for_house(config, house: int) -> Path | None:
    paths = getattr(config.task_sampler_config, "scene_xml_paths", None)
    if not paths:
        return None
    if 0 <= house < len(paths):
        return Path(paths[house]).resolve()
    if len(paths) == 1:
        return Path(paths[0]).resolve()
    return None


def _selected_houses(config, requested: list[int] | None, all_houses: bool) -> list[int]:
    if requested:
        return requested

    houses = _configured_houses(config)
    if all_houses:
        return houses

    # Custom configs often repeat one XML hundreds of times only to parallelize collection.
    # Render one representative house for each distinct XML unless explicitly asked for all.
    if getattr(config.task_sampler_config, "scene_xml_paths", None):
        selected: list[int] = []
        seen: set[str] = set()
        for house in houses:
            path = _scene_path_for_house(config, house)
            key = str(path) if path is not None else f"house:{house}"
            if key not in seen:
                seen.add(key)
                selected.append(house)
        return selected

    # Dataset-backed houses are all distinct. Default to one to avoid an accidental huge render.
    return houses[:1]


def _robot_model_label(config) -> str:
    robot = config.robot_config
    return f"{robot.name}/{Path(robot.robot_xml_path).name}"


def _format_house_ids(houses: set[int]) -> str:
    ordered = sorted(houses)
    if not ordered:
        return "none"
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:] + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return ",".join(ranges)


def _is_project_config(name: str) -> bool:
    """True for proximity datagen configs this repo actually uses."""
    return PROJECT_CONFIG_RE.match(name) is not None


def _config_priority(name: str) -> tuple:
    """Prefer cheap Check/Smoke configs when several share one scene."""
    is_check = 0 if name.endswith("CheckConfig") else 1
    is_smoke = 0 if "Smoke" in name else 1
    is_invis = 0 if "Invis" not in name else 1
    is_hybrid = 0 if "Hybrid" in name else 1
    return (is_check, is_smoke, is_invis, is_hybrid, name)


def _source_keys(config, houses: list[int]) -> dict[str, set[int]]:
    """Map a stable environment label to the house indices that use it."""
    paths = getattr(config.task_sampler_config, "scene_xml_paths", None)
    keys: dict[str, set[int]] = defaultdict(set)
    if paths:
        for house in houses:
            path = _scene_path_for_house(config, house)
            label = str(path) if path is not None else f"<unresolved house {house}>"
            keys[label].add(house)
        return keys

    sampler = getattr(config.task_sampler_config, "task_sampler_class", None)
    sampler_name = getattr(sampler, "__name__", str(sampler))
    label = f"{config.scene_dataset}:{config.data_split}:{sampler_name}"
    keys[label].update(houses)
    return keys


def _collect_environment_groups(
    scope: str,
    force_hybrid: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Group registered proximity configs by unique (scene source, robot).

    With ``force_hybrid``, FrankaSkin configs collapse onto the hybrid robot label so
    29-sensor and 40-sensor configs that share one XML are not double-rendered.
    """
    _auto_import_configs()
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: list[str] = []
    hybrid_label = "franka_skin/model_hybrid.xml"

    for name in sorted(list_available_configs()):
        if scope == "project" and not _is_project_config(name):
            continue
        try:
            config = get_config_class(name)()
        except Exception:
            skipped.append(name)
            continue

        sensors = _proximity_sensor_names(config)
        if not sensors and not (force_hybrid and FRANKA_SKIN_RE.match(name)):
            continue

        houses = _configured_houses(config)
        if force_hybrid and FRANKA_SKIN_RE.match(name):
            robot = hybrid_label
            sensor_count = 40
        else:
            robot = _robot_model_label(config)
            sensor_count = len(sensors)
            if not sensors:
                continue

        for label, source_houses in _source_keys(config, houses).items():
            key = (label, robot)
            group = groups.get(key)
            if group is None:
                group = {
                    "source": label,
                    "robot": robot,
                    "sensor_count": sensor_count,
                    "config_houses": {},
                    "houses": set(),
                }
                groups[key] = group
            group["config_houses"].setdefault(name, set()).update(source_houses)
            group["houses"].update(source_houses)
            group["sensor_count"] = max(int(group["sensor_count"]), sensor_count)

    ordered = [groups[key] for key in sorted(groups)]
    return ordered, skipped


def _representative_job(group: dict[str, Any]) -> tuple[str, int]:
    name = min(group["config_houses"], key=_config_priority)
    house = min(group["config_houses"][name])
    return name, house


def _jobs_by_config(groups: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Collapse unique-environment jobs into config -> houses to render."""
    wanted: dict[str, set[int]] = defaultdict(set)
    for group in groups:
        name, house = _representative_job(group)
        wanted[name].add(house)
    return {name: sorted(houses) for name, houses in wanted.items()}


def list_proximity_environments(scope: str, force_hybrid: bool = True) -> None:
    """Print unique scene sources used by registered proximity-sensor configs."""
    groups, skipped = _collect_environment_groups(scope, force_hybrid=force_hybrid)
    print(
        f"Found {len(groups)} unique environment sources "
        f"(scope={scope}, force_hybrid={force_hybrid})."
    )
    for group in groups:
        config_name, house = _representative_job(group)
        print(f"\n{group['source']}")
        print(f"  robot: {group['robot']}")
        print(f"  houses: {len(group['houses'])}")
        print(f"  proximity sensors: {group['sensor_count']}")
        print(f"  --all will render: {config_name} house {house}")
        print("  configs:")
        for name, houses in sorted(group["config_houses"].items()):
            print(f"    {name}: houses {_format_house_ids(houses)}")
    if skipped:
        print(f"\nSkipped {len(skipped)} configs that could not be instantiated.")


def _write_gallery(records: list[dict[str, Any]], output_dir: Path) -> Path | None:
    """Write a browsable HTML contact sheet preferring the robot-scene panel."""
    tiles: list[tuple[Path, str]] = []
    for record in records:
        outputs = [Path(path) for path in record.get("outputs", [])]
        preferred = None
        for path in outputs:
            if path.name.startswith("01_robot_scene") and path.suffix == ".png":
                preferred = path
                break
        if preferred is None:
            pngs = [path for path in outputs if path.suffix == ".png"]
            preferred = pngs[0] if pngs else None
        if preferred is None:
            continue
        caption = (
            f"{record.get('config')} · {Path(str(record.get('scene', ''))).name} · "
            f"house {record.get('house')} · {record.get('robot_model')}"
        )
        tiles.append((preferred, caption))
    if not tiles:
        return None

    gallery_path = output_dir / "gallery.html"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>datagen environment gallery</title>",
        "<style>",
        ":root{color-scheme:dark}",
        "*{box-sizing:border-box}",
        "body{font-family:Inter,ui-sans-serif,system-ui,sans-serif;"
        "background:radial-gradient(circle at 15% 0%,#1c2733 0,#0e1014 38%);"
        "color:#eef2f6;margin:0;padding:40px}",
        "main{max-width:1720px;margin:auto}",
        "h1{font-size:clamp(22px,3vw,38px);font-weight:560;letter-spacing:-.03em;"
        "margin:0 0 26px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));"
        "gap:22px}",
        "figure{margin:0;background:#171b21;border:1px solid #303843;padding:9px;"
        "border-radius:18px;box-shadow:0 18px 50px #0007;overflow:hidden}",
        "a{display:block;border-radius:12px;overflow:hidden}",
        "img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;"
        "transition:transform .25s ease}",
        "a:hover img{transform:scale(1.012)}",
        "figcaption{font-size:12px;line-height:1.4;padding:11px 8px 6px;color:#aeb9c5}",
        "</style></head><body>",
        "<main>",
        f"<h1>{len(tiles)} datagen environments (hybrid skin)</h1>",
        "<div class='grid'>",
    ]
    for path, caption in tiles:
        rel = os.path.relpath(path, output_dir)
        parts.append(
            f"<figure><a href='{rel}'><img src='{rel}' alt='{caption}'></a>"
            f"<figcaption>{caption}</figcaption></figure>"
        )
    parts.append("</div></main></body></html>")
    gallery_path.write_text("\n".join(parts) + "\n")
    return gallery_path


def _force_hybrid_skin(config) -> None:
    """Replace the robot/camera stack with the canonical 40-sensor hybrid skin."""
    base_size = getattr(getattr(config, "robot_config", None), "base_size", None)
    kwargs: dict[str, Any] = {}
    if base_size is not None:
        kwargs["base_size"] = list(base_size)
    config.robot_config = FrankaSkinHybridRobotConfig(**kwargs)
    config.camera_config = FrankaSkinHybridCameraSystem()


def _prepare_config(config, args: argparse.Namespace) -> None:
    """Trim data-generation-only work while preserving sampled scene content."""
    config.seed = args.seed
    config.num_workers = 1
    config.use_passive_viewer = False
    config.viz_sensor_rgb = False
    config.profile = False
    config.datagen_profiler = False
    config.task_sampler_config.task_batch_size = 1

    if not args.keep_config_robot and FRANKA_SKIN_RE.match(type(config).__name__):
        _force_hybrid_skin(config)

    if args.no_randomization:
        sampler = config.task_sampler_config
        sampler.randomize_lighting = False
        sampler.randomize_textures = False
        sampler.randomize_textures_all = False
        sampler.randomize_robot_textures = False
        sampler.randomize_dynamics = False


def _sample_task(task_sampler, house: int, variant: str, attempts: int):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            task = task_sampler.sample_task(house_index=house, variant=variant)
            if task is None:
                raise RuntimeError("task sampler returned no task")
            return task
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            last_error = exc
            print(
                f"  sample attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    raise RuntimeError(f"could not sample house {house} after {attempts} attempts") from last_error


def _focus_seed(task) -> tuple[np.ndarray, np.ndarray]:
    """Return robot-base and task-target points used to choose nearby scene geometry."""
    model, data = task.env.current_model, task.env.current_data
    robot_points = []
    base = np.zeros(3, dtype=float)
    for body_id in range(model.nbody):
        name = model.body(body_id).name or ""
        if name.startswith("robot_0/"):
            point = np.asarray(data.xpos[body_id], dtype=float)
            robot_points.append(point)
            if name in {"robot_0/base", "robot_0/fr3_link0"}:
                base = point.copy()

    task_point = None
    task_config = getattr(task.config, "task_config", None)
    for attr in ("pickup_obj_start_pose", "pickup_obj_goal_pose"):
        value = getattr(task_config, attr, None)
        if value is not None and len(value) >= 3:
            candidate = np.asarray(value[:3], dtype=float)
            if np.isfinite(candidate).all():
                task_point = candidate
                break

    if task_point is None:
        task_point = np.mean(robot_points, axis=0) if robot_points else np.array([0.5, 0.0, 0.8])
    return base, task_point


def _automatic_camera(task, show_hidden: bool) -> tuple[np.ndarray, float]:
    """Frame robot plus nearby work cell while ignoring outer room shells and parked objects."""
    model, data = task.env.current_model, task.env.current_data
    base, task_point = _focus_seed(task)
    seed = 0.5 * (base + task_point)

    points: list[np.ndarray] = []
    for body_id in range(model.nbody):
        name = model.body(body_id).name or ""
        if name.startswith("robot_0/"):
            points.append(np.asarray(data.xpos[body_id], dtype=float))

    visible_groups = {0, 1, 2}
    if show_hidden:
        visible_groups.add(4)

    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) not in visible_groups:
            continue
        if float(model.geom_rgba[geom_id, 3]) <= 0.01:
            continue
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue

        center = np.asarray(data.geom_xpos[geom_id], dtype=float)
        radius = float(model.geom_rbound[geom_id])
        if not np.isfinite(center).all() or not math.isfinite(radius):
            continue
        if center[2] < -0.25 or center[2] > 2.5:
            continue
        if np.linalg.norm(center[:2] - seed[:2]) > 1.75:
            continue
        # Room walls and ceilings have very large bounding spheres. They make the work cell tiny.
        if radius > 1.2:
            continue

        radius = min(radius, 0.55)
        points.extend(
            [
                center,
                center + np.array([radius, 0.0, 0.0]),
                center - np.array([radius, 0.0, 0.0]),
                center + np.array([0.0, radius, 0.0]),
                center - np.array([0.0, radius, 0.0]),
                center + np.array([0.0, 0.0, radius]),
                center - np.array([0.0, 0.0, radius]),
            ]
        )

    if not points:
        return np.array([0.45, 0.0, 0.8]), 2.5

    cloud = np.asarray(points)
    low = np.percentile(cloud, 1.0, axis=0)
    high = np.percentile(cloud, 99.0, axis=0)
    lookat = 0.5 * (low + high)
    lookat[2] = float(np.clip(lookat[2], 0.45, 1.25))
    radius = float(np.linalg.norm(0.5 * (high - low)))
    # Presentation framing: retain nearby context without shrinking the robot into the room.
    distance = float(np.clip(1.95 * radius, 1.55, 6.0))
    return lookat, distance


def _scene_option(show_hidden: bool, show_sensors: bool) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[:] = 0
    option.geomgroup[0:3] = 1  # normal scene, robot, and cosmetic skin
    option.geomgroup[4] = int(show_hidden)  # sensor-only hazard/staging geometry
    option.sitegroup[:] = 0
    if show_sensors:
        option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1
    return option


def _proximity_scene_option() -> mujoco.MjvOption:
    """Match datagen proximity renders: hide cosmetic skin, show sensor-only geoms."""
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[2] = 0
    option.geomgroup[4] = 1
    return option


def _free_camera(
    model: mujoco.MjModel,
    lookat: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _apply_presentation_lighting(model: mujoco.MjModel) -> None:
    """Add camera-mounted fill light after task lighting randomization."""
    headlight = model.vis.headlight
    headlight.active = 1
    headlight.ambient[:] = np.maximum(
        headlight.ambient,
        PRESENTATION_HEADLIGHT_AMBIENT,
    )
    headlight.diffuse[:] = np.maximum(
        headlight.diffuse,
        PRESENTATION_HEADLIGHT_DIFFUSE,
    )
    headlight.specular[:] = np.maximum(
        headlight.specular,
        PRESENTATION_HEADLIGHT_SPECULAR,
    )


def _grade_frame(frame: np.ndarray) -> np.ndarray:
    """Lift dark sampled lighting while preserving scene colors and geometry."""
    rgb = np.clip(frame.astype(np.float32) / 255.0, 0.0, 1.0)
    rgb = np.clip(np.power(rgb, 0.78) * 1.04, 0.0, 1.0)
    image = Image.fromarray((rgb * 255.0).astype(np.uint8))
    image = ImageEnhance.Contrast(image).enhance(1.06)
    image = ImageEnhance.Color(image).enhance(1.08)
    image = ImageEnhance.Sharpness(image).enhance(1.05)
    return np.asarray(image)


def _paste_rounded_cover(
    canvas: Image.Image,
    frame: np.ndarray | Image.Image,
    box: tuple[int, int, int, int],
    *,
    radius: int = 18,
    outline: tuple[int, int, int] = PANEL_STROKE,
    interpolation: Image.Resampling = Image.Resampling.LANCZOS,
) -> None:
    """Crop one frame into a rounded presentation cell."""
    x, y, width, height = box
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    fitted = ImageOps.fit(image.convert("RGB"), (width, height), method=interpolation)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=radius,
        fill=255,
    )
    canvas.paste(fitted, (x, y), mask)
    ImageDraw.Draw(canvas).rounded_rectangle(
        (x, y, x + width - 1, y + height - 1),
        radius=radius,
        outline=outline,
        width=2,
    )


def _write_visual_plate(
    frames: list[np.ndarray],
    output_path: Path,
    *,
    outline: tuple[int, int, int] = PANEL_STROKE,
) -> None:
    """Build a text-free editorial triptych: hero frame plus two supporting views."""
    if not frames:
        raise ValueError("visual plate requires at least one frame")
    frames = list(frames)
    while len(frames) < PRESENTATION_VIEW_COUNT:
        frames.append(frames[-1])

    margin = 18
    gap = 12
    content_width = PLATE_WIDTH - 2 * margin
    content_height = PLATE_HEIGHT - 2 * margin
    hero_width = int(round((content_width - gap) * 0.66))
    side_width = content_width - gap - hero_width
    side_height = (content_height - gap) // 2

    canvas = Image.new("RGB", (PLATE_WIDTH, PLATE_HEIGHT), PANEL_BG)
    _paste_rounded_cover(
        canvas,
        frames[0],
        (margin, margin, hero_width, content_height),
        radius=22,
        outline=outline,
    )
    side_x = margin + hero_width + gap
    _paste_rounded_cover(
        canvas,
        frames[1],
        (side_x, margin, side_width, side_height),
        outline=outline,
    )
    _paste_rounded_cover(
        canvas,
        frames[2],
        (
            side_x,
            margin + side_height + gap,
            side_width,
            content_height - side_height - gap,
        ),
        outline=outline,
    )
    canvas.save(output_path, quality=95)


def _angular_distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _robot_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for geom_id in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_id])).name or ""
        if body_name == "robot_0" or body_name.startswith("robot_0/"):
            ids.append(geom_id)
    return np.asarray(ids, dtype=np.int32)


def _view_mask_score(mask: np.ndarray) -> float:
    """Reward a visible, centered robot and reject wall-obscured or clipped views."""
    y, x = np.nonzero(mask)
    if len(x) < 12:
        return float("-inf")

    height, width = mask.shape
    pixel_fraction = len(x) / float(height * width)
    x0, x1 = int(x.min()), int(x.max())
    y0, y1 = int(y.min()), int(y.max())
    bbox_fraction = ((x1 - x0 + 1) * (y1 - y0 + 1)) / float(height * width)
    center_x = (x0 + x1) / (2.0 * width)
    center_y = (y0 + y1) / (2.0 * height)
    center_error = math.hypot(center_x - 0.5, center_y - 0.52)

    border_x = max(2, int(round(width * 0.04)))
    border_y = max(2, int(round(height * 0.04)))
    edge_pixels = (
        mask[:border_y].sum()
        + mask[-border_y:].sum()
        + mask[:, :border_x].sum()
        + mask[:, -border_x:].sum()
    )
    edge_fraction = float(edge_pixels) / max(float(mask.sum()), 1.0)
    return (
        6.0 * math.sqrt(pixel_fraction)
        + 2.0 * math.sqrt(bbox_fraction)
        - 1.25 * center_error
        - 2.5 * edge_fraction
    )


def _select_presentation_views(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lookat: np.ndarray,
    distance: float,
    option: mujoco.MjvOption,
    args: argparse.Namespace,
) -> list[tuple[float, float]]:
    """Score a full orbit with segmentation, then keep three clear, distinct views."""
    fallback = [
        ((args.azimuth + offset) % 360.0, args.elevation)
        for offset in (0.0, 55.0, -55.0)
    ]
    robot_ids = _robot_geom_ids(model)
    if len(robot_ids) == 0:
        return fallback

    probe_height = 180
    probe_width = max(240, int(round(probe_height * args.width / args.height)))
    candidates: list[tuple[float, float, float]] = []
    probe = None
    try:
        probe = mujoco.Renderer(
            model,
            height=probe_height,
            width=probe_width,
            max_geom=RENDER_MAX_GEOM,
        )
        probe.enable_segmentation_rendering()
        for offset in np.arange(0.0, 360.0, VIEW_PROBE_STEP_DEG):
            azimuth = float((args.azimuth + offset) % 360.0)
            camera = _free_camera(
                model,
                lookat,
                distance,
                azimuth,
                args.elevation,
            )
            probe.update_scene(data, camera=camera, scene_option=option)
            segmentation = probe.render()
            is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            mask = is_geom & np.isin(segmentation[..., 0], robot_ids)
            score = _view_mask_score(mask)
            if math.isfinite(score):
                candidates.append((score, azimuth, args.elevation))
    except Exception as exc:
        logging.getLogger(__name__).debug("View scoring failed: %s", exc)
        return fallback
    finally:
        if probe is not None:
            probe.close()

    if not candidates:
        return fallback

    best = max(candidates)
    quality_floor = best[0] - 0.65
    candidates = [candidate for candidate in candidates if candidate[0] >= quality_floor]
    selected: list[tuple[float, float, float]] = [best]
    remaining = [candidate for candidate in candidates if candidate != best]
    while remaining and len(selected) < PRESENTATION_VIEW_COUNT:
        eligible = [
            candidate
            for candidate in remaining
            if min(
                _angular_distance(candidate[1], chosen[1])
                for chosen in selected
            )
            >= 30.0
        ]
        pool = eligible or remaining
        chosen = max(
            pool,
            key=lambda candidate: candidate[0]
            + 0.30
            * min(
                _angular_distance(candidate[1], previous[1])
                for previous in selected
            )
            / 180.0,
        )
        selected.append(chosen)
        remaining.remove(chosen)

    views = [(azimuth, elevation) for _score, azimuth, elevation in selected]
    while len(views) < PRESENTATION_VIEW_COUNT:
        views.append(fallback[len(views)])
    return views


def _mjcf_sensor_cameras(model: mujoco.MjModel) -> list[str]:
    names = []
    for cam_id in range(model.ncam):
        name = model.camera(cam_id).name or ""
        if "_sensor_" in name:
            names.append(name)
    return sorted(names)


def _cam_pose(model: mujoco.MjModel, data: mujoco.MjData, full_name: str):
    cam_id = model.camera(full_name).id
    pos = np.asarray(data.cam_xpos[cam_id], dtype=float).copy()
    rot = np.asarray(data.cam_xmat[cam_id], dtype=float).reshape(3, 3).copy()
    return pos, rot


def _add_scene_line(scene: mujoco.MjvScene, p0, p1, rgba, width: float = 0.0016) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(rgba, np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p0, p1)
    scene.ngeom += 1


def _add_scene_sphere(scene: mujoco.MjvScene, point, rgba, radius: float = 0.006) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(point, np.float64),
        np.eye(3).ravel(),
        np.asarray(rgba, np.float32),
    )
    scene.ngeom += 1


def _draw_sensor_cones(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    length: float = CONE_LENGTH_M,
    fovy_deg: float = SENSOR_FOVY_DEG,
) -> int:
    """Overlay FOV pyramids for every MJCF proximity sensor camera. Returns cone count."""
    half = math.tan(math.radians(fovy_deg / 2.0)) * length
    corners_cam = np.array(
        [
            [-half, -half, -length],
            [half, -half, -length],
            [half, half, -length],
            [-half, half, -length],
        ],
        dtype=float,
    )
    drawn = 0
    for full_name in _mjcf_sensor_cameras(model):
        short_name = full_name.rsplit("/", 1)[-1]
        link_color = SENSOR_LINK_COLORS.get(
            _sensor_link_key(short_name),
            SENSOR_LINK_COLORS["other"],
        )
        cone_rgba = tuple(channel / 255.0 for channel in link_color) + (0.74,)
        axis_rgba = tuple(
            min(1.0, channel / 255.0 + 0.28) for channel in link_color
        ) + (0.82,)
        pos, rot = _cam_pose(model, data, full_name)
        fwd = -rot[:, 2]
        corners_world = (rot @ corners_cam.T).T + pos
        for corner in corners_world:
            _add_scene_line(scene, pos, corner, cone_rgba, width=0.0017)
        for index in range(4):
            _add_scene_line(
                scene,
                corners_world[index],
                corners_world[(index + 1) % 4],
                cone_rgba,
                width=0.0011,
            )
        _add_scene_line(scene, pos, pos + fwd * length, axis_rgba, width=0.0013)
        _add_scene_sphere(scene, pos, (1.0, 0.86, 0.35, 1.0), radius=0.006)
        drawn += 1
    return drawn


def _depth_to_rgb(depth: np.ndarray, near: float = DEPTH_NEAR_M, far: float = DEPTH_FAR_M) -> np.ndarray:
    """Map metric depth to a turbo-like RGB tile (near=warm, far=cool, invalid=dark)."""
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    out = np.full((*depth.shape, 3), 18, dtype=np.uint8)
    if not valid.any():
        return out
    t = np.clip((far - depth) / max(far - near, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    out[valid] = (rgb[valid] * 255.0).astype(np.uint8)
    return out


def _sensor_link_key(name: str) -> str:
    if name.startswith("link5_front"):
        return "link5_front"
    if name.startswith("link5_back"):
        return "link5_back"
    match = re.match(r"(link\d+)_sensor_", name)
    return match.group(1) if match else "other"


def _group_depth_tiles(
    depth_tiles: list[tuple[str, np.ndarray, bool]],
) -> list[tuple[str, list[tuple[str, np.ndarray, bool]]]]:
    buckets: dict[str, list[tuple[str, np.ndarray, bool]]] = {key: [] for key in SENSOR_LINK_ORDER}
    buckets["other"] = []
    for item in depth_tiles:
        buckets[_sensor_link_key(item[0])].append(item)
    groups: list[tuple[str, list[tuple[str, np.ndarray, bool]]]] = []
    for key in SENSOR_LINK_ORDER:
        if buckets[key]:
            groups.append((key, buckets[key]))
    if buckets["other"]:
        groups.append(("other", buckets["other"]))
    return groups


def _render_external_frame(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lookat: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
    option: mujoco.MjvOption,
    draw_cones: bool,
) -> np.ndarray:
    camera = _free_camera(model, lookat, distance, azimuth, elevation)
    renderer.update_scene(data, camera=camera, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    if draw_cones:
        _draw_sensor_cones(renderer.scene, model, data)
    return _grade_frame(renderer.render().copy())


def _write_turntable(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    path: Path,
    lookat: np.ndarray,
    distance: float,
    option: mujoco.MjvOption,
    args: argparse.Namespace,
    draw_cones: bool,
) -> Path:
    frame_count = max(1, int(round(args.seconds * args.fps)))
    with imageio.get_writer(
        path,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    ) as writer:
        for azimuth in np.linspace(
            args.azimuth,
            args.azimuth + 360.0,
            frame_count,
            endpoint=False,
        ):
            frame = _render_external_frame(
                renderer,
                model,
                data,
                lookat,
                distance,
                float(azimuth),
                args.elevation,
                option,
                draw_cones,
            )
            writer.append_data(frame)
    return path


def _compose_sensor_atlas(
    groups: list[tuple[str, list[tuple[str, np.ndarray, bool]]]],
    *,
    tile: int = PROX_TILE_PX,
    columns: int = 8,
    gap: int = 6,
    pixelated: bool = False,
    outline: tuple[int, int, int] = PANEL_STROKE,
) -> Image.Image:
    """Pack all sensor views into one compact, color-linked, text-free atlas."""
    items = [
        (link, rgb, active)
        for link, link_items in groups
        for _name, rgb, active in link_items
    ]
    if not items:
        return Image.new("RGB", (tile, tile), PANEL_BG)

    rows = int(math.ceil(len(items) / columns))
    padding = 10
    width = 2 * padding + columns * tile + max(0, columns - 1) * gap
    height = 2 * padding + rows * tile + max(0, rows - 1) * gap
    atlas = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(atlas)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=16,
        fill=PANEL_CARD,
        outline=outline,
        width=2,
    )

    resampling = Image.Resampling.NEAREST if pixelated else Image.Resampling.LANCZOS
    inset = 4
    inner_size = tile - 2 * inset
    for index, (link, rgb, active) in enumerate(items):
        row, column = divmod(index, columns)
        x = padding + column * (tile + gap)
        y = padding + row * (tile + gap)
        color = SENSOR_LINK_COLORS.get(link, SENSOR_LINK_COLORS["other"])
        if not active:
            color = tuple(
                int(round(0.42 * channel + 0.58 * stroke))
                for channel, stroke in zip(color, PANEL_STROKE)
            )

        draw.rounded_rectangle(
            (x, y, x + tile - 1, y + tile - 1),
            radius=9,
            fill=PANEL_BG,
            outline=color,
            width=3 if active else 2,
        )
        image = Image.fromarray(rgb).convert("RGB")
        image = ImageOps.fit(
            image,
            (inner_size, inner_size),
            method=resampling,
        )
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, inner_size - 1, inner_size - 1),
            radius=6,
            fill=255,
        )
        atlas.paste(image, (x + inset, y + inset), mask)
    return atlas


def _render_camera_sensor_panel(
    task,
    config,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Top: exo/wrist RGB. Bottom: paired compact sensor atlases. No labels."""
    env = task.env
    model, data = env.current_model, env.current_data
    sensor_names = _proximity_sensor_names(config)
    stats: dict[str, Any] = {
        "rgb_cameras": [],
        "proximity_sensors": len(sensor_names),
        "proximity_with_return": 0,
    }

    camera_tiles: list[np.ndarray] = []
    for cam_name in ("exo_camera_1", "wrist_camera"):
        try:
            if cam_name not in env.camera_manager.registry:
                continue
            rgb = env.render_rgb_frame(cam_name)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            camera_tiles.append(rgb)
            stats["rgb_cameras"].append(cam_name)
        except Exception as exc:
            print(f"  camera render failed ({cam_name}): {exc}", file=sys.stderr)

    viz_res = 256
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), viz_res)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), viz_res)

    prox_opt = _proximity_scene_option()
    depth_renderer = mujoco.Renderer(model, height=8, width=8)
    depth_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    rgb_renderer = mujoco.Renderer(model, height=viz_res, width=viz_res)
    rgb_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0

    depth_tiles: list[tuple[str, np.ndarray, bool]] = []
    rgb_tiles: list[tuple[str, np.ndarray, bool]] = []
    try:
        depth_renderer.enable_depth_rendering()
        for short_name in sensor_names:
            full = env._proximity_cam_full_name(short_name)
            if full is None:
                continue
            try:
                model.camera(full)
            except KeyError:
                continue

            depth_renderer.update_scene(data, full, scene_option=prox_opt)
            depth = depth_renderer.render().copy()
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth = depth.astype(np.float32)
            in_range = bool(np.any((depth >= DEPTH_NEAR_M) & (depth <= DEPTH_FAR_M)))
            if in_range:
                stats["proximity_with_return"] += 1
            depth_rgb = _depth_to_rgb(depth)
            depth_tiles.append((short_name, depth_rgb, in_range))

            rgb_renderer.update_scene(data, full, scene_option=prox_opt)
            sensor_rgb = rgb_renderer.render().copy()
            if sensor_rgb.dtype != np.uint8:
                sensor_rgb = np.clip(sensor_rgb, 0, 255).astype(np.uint8)
            rgb_tiles.append((short_name, sensor_rgb, in_range))
    finally:
        depth_renderer.close()
        rgb_renderer.close()

    if not camera_tiles and not depth_tiles:
        raise RuntimeError("no camera or proximity images could be rendered")

    # One consistent 16:9 plate: camera pair above, depth/RGB atlases below.
    margin = 18
    gap = 12
    depth_groups = _group_depth_tiles(depth_tiles)
    rgb_groups = _group_depth_tiles(rgb_tiles)
    depth_atlas = _compose_sensor_atlas(
        depth_groups,
        pixelated=True,
        outline=PANEL_ACCENT,
    )
    rgb_atlas = _compose_sensor_atlas(
        rgb_groups,
        pixelated=False,
        outline=PANEL_WARM,
    )

    canvas = Image.new("RGB", (PLATE_WIDTH, PLATE_HEIGHT), PANEL_BG)
    content_width = PLATE_WIDTH - 2 * margin
    bottom_width = depth_atlas.width + gap + rgb_atlas.width
    bottom_height = max(depth_atlas.height, rgb_atlas.height)
    bottom_x = (PLATE_WIDTH - bottom_width) // 2
    bottom_y = PLATE_HEIGHT - margin - bottom_height

    if camera_tiles:
        camera_height = max(1, bottom_y - margin - gap)
        camera_width = (
            content_width - gap * (len(camera_tiles) - 1)
        ) // len(camera_tiles)
        x = margin
        for rgb in camera_tiles:
            _paste_rounded_cover(
                canvas,
                rgb,
                (x, margin, camera_width, camera_height),
                outline=PANEL_STROKE,
            )
            x += camera_width + gap
    else:
        bottom_y = (PLATE_HEIGHT - bottom_height) // 2

    canvas.paste(depth_atlas, (bottom_x, bottom_y))
    canvas.paste(rgb_atlas, (bottom_x + depth_atlas.width + gap, bottom_y))
    canvas.save(output_path, quality=95)
    return output_path, stats



def _backproject_depth8(
    depth: np.ndarray,
    pos: np.ndarray,
    rot: np.ndarray,
    near: float = DEPTH_NEAR_M,
    far: float = DEPTH_FAR_M,
    fovy_deg: float = SENSOR_FOVY_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an 8x8 planar-z depth patch to world points. Sensors only."""
    height, width = depth.shape[:2]
    focal = (height / 2.0) / math.tan(math.radians(fovy_deg / 2.0))
    cx = cy = (height - 1) / 2.0
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    d = depth[valid].astype(np.float64)
    x_c = (uu[valid] - cx) * d / focal
    y_c = -(vv[valid] - cy) * d / focal
    pts_cam = np.stack([x_c, y_c, -d], axis=0)
    pts = (rot @ pts_cam).T + pos
    return pts, d


def _depth_rgb_color(depth_m: float, near: float = DEPTH_NEAR_M, far: float = DEPTH_FAR_M) -> tuple[int, int, int]:
    rgb = _depth_to_rgb(np.array([[depth_m]], dtype=np.float32), near, far)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _write_ply_xyzrgb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    n = int(points.shape[0])
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for (x, y, z), (r, g, b) in zip(points, colors):
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _build_sensor_pointcloud(
    task,
    config,
) -> dict[str, Any]:
    """Fuse every proximity sensor's 8x8 depth into one world-frame point cloud."""
    env = task.env
    model, data = env.current_model, env.current_data
    sensor_names = _proximity_sensor_names(config)
    prox_opt = _proximity_scene_option()
    renderer = mujoco.Renderer(model, height=8, width=8)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0

    points: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    sensor_ids: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    origin_colors: list[np.ndarray] = []
    per_sensor: dict[str, int] = {}

    try:
        renderer.enable_depth_rendering()
        for sensor_index, short_name in enumerate(sensor_names):
            full = env._proximity_cam_full_name(short_name)
            if full is None:
                continue
            try:
                model.camera(full)
            except KeyError:
                continue
            renderer.update_scene(data, full, scene_option=prox_opt)
            depth = renderer.render().copy()
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth = depth.astype(np.float32)
            pos, rot = _cam_pose(model, data, full)
            pts, dvals = _backproject_depth8(depth, pos, rot)
            per_sensor[short_name] = int(len(pts))
            origins.append(pos.reshape(1, 3))
            origin_colors.append(np.array([[255, 220, 80]], dtype=np.uint8))
            if len(pts) == 0:
                continue
            rgb = np.asarray([_depth_rgb_color(float(d)) for d in dvals], dtype=np.uint8)
            points.append(pts)
            depths.append(dvals)
            colors.append(rgb)
            sensor_ids.append(np.full(len(pts), sensor_index, dtype=np.int32))
    finally:
        renderer.close()

    if points:
        cloud = np.concatenate(points, axis=0)
        depth_arr = np.concatenate(depths, axis=0)
        color_arr = np.concatenate(colors, axis=0)
        id_arr = np.concatenate(sensor_ids, axis=0)
    else:
        cloud = np.zeros((0, 3), dtype=np.float64)
        depth_arr = np.zeros((0,), dtype=np.float64)
        color_arr = np.zeros((0, 3), dtype=np.uint8)
        id_arr = np.zeros((0,), dtype=np.int32)

    origin_arr = np.concatenate(origins, axis=0) if origins else np.zeros((0, 3))
    origin_rgb = np.concatenate(origin_colors, axis=0) if origin_colors else np.zeros((0, 3), dtype=np.uint8)

    return {
        "points": cloud,
        "depths": depth_arr,
        "colors": color_arr,
        "sensor_ids": id_arr,
        "sensor_names": sensor_names,
        "per_sensor_counts": per_sensor,
        "sensor_origins": origin_arr,
        "sensor_origin_colors": origin_rgb,
        "n_points": int(cloud.shape[0]),
        "n_sensors_with_return": int(sum(1 for count in per_sensor.values() if count > 0)),
    }


def _write_pointcloud_preview(
    cloud: dict[str, Any],
    output_path: Path,
) -> Path:
    """Text-free 3D reconstruction plate with one hero and two support views."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = cloud["points"]
    colors = cloud["colors"].astype(np.float64) / 255.0
    origins = cloud["sensor_origins"]
    bounds_data = []
    if len(points):
        bounds_data.append(points)
    if len(origins):
        bounds_data.append(origins)
    if bounds_data:
        bounds = np.concatenate(bounds_data, axis=0)
        center = 0.5 * (bounds.min(axis=0) + bounds.max(axis=0))
        span = max(float(np.ptp(bounds, axis=0).max()) * 0.58, 0.25)
    else:
        center = np.zeros(3, dtype=float)
        span = 0.5

    fig = plt.figure(figsize=(12.0, 6.75), dpi=160)
    fig.patch.set_facecolor("#0e1014")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.15, 1.15, 0.90),
        wspace=-0.06,
        hspace=-0.08,
    )
    axes = (
        fig.add_subplot(grid[:, :2], projection="3d"),
        fig.add_subplot(grid[0, 2], projection="3d"),
        fig.add_subplot(grid[1, 2], projection="3d"),
    )
    views = ((26, -52), (78, -90), (8, -90))
    floor_z = center[2] - span
    for index, (ax, (elev, azim)) in enumerate(zip(axes, views)):
        ax.set_facecolor("#0e1014")
        for offset in np.linspace(-span, span, 7):
            ax.plot(
                [center[0] + offset] * 2,
                [center[1] - span, center[1] + span],
                [floor_z, floor_z],
                color="#242b35",
                linewidth=0.55,
                alpha=0.62,
            )
            ax.plot(
                [center[0] - span, center[0] + span],
                [center[1] + offset] * 2,
                [floor_z, floor_z],
                color="#242b35",
                linewidth=0.55,
                alpha=0.62,
            )
        if len(points):
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=12 if index == 0 else 8,
                linewidths=0,
                depthshade=False,
                alpha=0.95,
            )
        if len(origins):
            ax.scatter(
                origins[:, 0],
                origins[:, 1],
                origins[:, 2],
                c="#fff0a8",
                s=34 if index == 0 else 22,
                marker="o",
                linewidths=0,
                depthshade=False,
            )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(
        output_path,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)
    return output_path


def _render_pointcloud_in_scene(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lookat: np.ndarray,
    distance: float,
    option: mujoco.MjvOption,
    elevation: float,
    cloud: dict[str, Any],
    azimuth: float = 45.0,
) -> np.ndarray:
    """External RGB with reconstructed points drawn as scene overlays."""
    camera = _free_camera(model, lookat, distance, azimuth, elevation)
    renderer.update_scene(data, camera=camera, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    scene = renderer.scene
    points = cloud["points"]
    colors = cloud["colors"]
    # Subsample if dense so overlay stays readable.
    step = 1 if len(points) <= 1200 else max(1, len(points) // 1200)
    for point, color in zip(points[::step], colors[::step]):
        rgba = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 0.95)
        _add_scene_sphere(scene, point, rgba, radius=0.008)
    for origin in cloud["sensor_origins"]:
        _add_scene_sphere(scene, origin, (1.0, 0.92, 0.30, 1.0), radius=0.011)
    return _grade_frame(renderer.render().copy())


def _write_sensor_pointcloud_outputs(
    task,
    config,
    output_dir: Path,
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lookat: np.ndarray,
    distance: float,
    option: mujoco.MjvOption,
    views: list[tuple[float, float]],
) -> list[Path]:
    """Build and save the skin-only reconstruction cloud + previews."""
    cloud = _build_sensor_pointcloud(task, config)
    written: list[Path] = []

    ply_path = output_dir / "04_sensor_pointcloud.ply"
    _write_ply_xyzrgb(ply_path, cloud["points"], cloud["colors"])
    written.append(ply_path)

    npz_path = output_dir / "04_sensor_pointcloud.npz"
    np.savez_compressed(
        npz_path,
        points=cloud["points"],
        depths=cloud["depths"],
        colors=cloud["colors"],
        sensor_ids=cloud["sensor_ids"],
        sensor_names=np.asarray(cloud["sensor_names"]),
        sensor_origins=cloud["sensor_origins"],
        near_m=DEPTH_NEAR_M,
        far_m=DEPTH_FAR_M,
        fovy_deg=SENSOR_FOVY_DEG,
    )
    written.append(npz_path)

    preview = output_dir / "04_sensor_pointcloud.png"
    _write_pointcloud_preview(cloud, preview)
    written.append(preview)

    # Same scored visual language as scene/cone plates.
    overlay = output_dir / "04_sensor_pointcloud_in_scene.png"
    overlay_frames = [
        _render_pointcloud_in_scene(
            renderer,
            model,
            data,
            lookat,
            distance,
            option,
            elevation,
            cloud,
            azimuth=azimuth,
        )
        for azimuth, elevation in views
    ]
    _write_visual_plate(overlay_frames, overlay, outline=PANEL_ACCENT)
    written.append(overlay)

    meta = {
        "n_points": cloud["n_points"],
        "n_sensors": len(cloud["sensor_names"]),
        "n_sensors_with_return": cloud["n_sensors_with_return"],
        "per_sensor_counts": cloud["per_sensor_counts"],
        "near_m": DEPTH_NEAR_M,
        "far_m": DEPTH_FAR_M,
        "fovy_deg": SENSOR_FOVY_DEG,
        "note": "points are proximity-sensor returns only (cosmetic skin hidden during depth render)",
    }
    meta_path = output_dir / "04_sensor_pointcloud.json"
    meta_path.write_text(json.dumps(_jsonable(meta), indent=2) + "\n")
    written.append(meta_path)
    return written


def _render_outputs(
    task,
    config,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[Path], np.ndarray, float, list[tuple[float, float]]]:
    env = task.env
    model, data = env.current_model, env.current_data
    if not args.keep_scene_lighting:
        _apply_presentation_lighting(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)

    auto_lookat, auto_distance = _automatic_camera(task, args.show_hidden)
    lookat = np.asarray(args.lookat, dtype=float) if args.lookat is not None else auto_lookat
    distance = float(args.distance) if args.distance is not None else auto_distance
    option = _scene_option(args.show_hidden, args.show_sensors)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    views = _select_presentation_views(
        model,
        data,
        lookat,
        distance,
        option,
        args,
    )

    renderer = mujoco.Renderer(model, height=args.height, width=args.width, max_geom=RENDER_MAX_GEOM)
    try:
        if args.format in {"png", "both"}:
            robot_frames = []
            cone_frames = []
            for azimuth, elevation in views:
                robot_frames.append(
                    _render_external_frame(
                        renderer,
                        model,
                        data,
                        lookat,
                        distance,
                        azimuth,
                        elevation,
                        option,
                        False,
                    )
                )
                cone_frames.append(
                    _render_external_frame(
                        renderer,
                        model,
                        data,
                        lookat,
                        distance,
                        azimuth,
                        elevation,
                        option,
                        True,
                    )
                )

            robot_path = output_dir / "01_robot_scene.png"
            _write_visual_plate(robot_frames, robot_path)
            written.append(robot_path)

            cones_path = output_dir / "02_sensor_cones.png"
            _write_visual_plate(cone_frames, cones_path, outline=PANEL_ACCENT)
            written.append(cones_path)

            try:
                cams_path = output_dir / "03_cameras_and_sensors.png"
                path, _stats = _render_camera_sensor_panel(task, config, cams_path)
                written.append(path)
            except Exception as exc:
                print(f"  camera/sensor panel failed: {exc}", file=sys.stderr)

            try:
                written.extend(
                    _write_sensor_pointcloud_outputs(
                        task,
                        config,
                        output_dir,
                        renderer,
                        model,
                        data,
                        lookat,
                        distance,
                        option,
                        views,
                    )
                )
            except Exception as exc:
                print(f"  sensor pointcloud failed: {exc}", file=sys.stderr)

        if args.format in {"mp4", "both"}:
            robot_mp4 = output_dir / "01_robot_scene_turntable.mp4"
            written.append(
                _write_turntable(
                    renderer,
                    model,
                    data,
                    robot_mp4,
                    lookat,
                    distance,
                    option,
                    args,
                    False,
                )
            )
            cones_mp4 = output_dir / "02_sensor_cones_turntable.mp4"
            written.append(
                _write_turntable(
                    renderer,
                    model,
                    data,
                    cones_mp4,
                    lookat,
                    distance,
                    option,
                    args,
                    True,
                )
            )
    finally:
        renderer.close()

    # Keep a stable alias so older gallery / docs paths still resolve.
    alias = output_dir / "environment.png"
    primary = output_dir / "01_robot_scene.png"
    if primary.exists():
        alias.write_bytes(primary.read_bytes())
        written.append(alias)

    return written, lookat, distance, views


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_metadata(
    task,
    config,
    config_name: str,
    house: int,
    sample_index: int,
    output_dir: Path,
    outputs: list[Path],
    lookat: np.ndarray,
    distance: float,
    views: list[tuple[float, float]],
    args: argparse.Namespace,
    variant: str,
) -> dict[str, Any]:
    env = task.env
    model = env.current_model
    sensor_names = _proximity_sensor_names(config)
    def geom_label(index: int) -> str:
        name = model.geom(index).name
        if name:
            return name
        body_name = model.body(int(model.geom_bodyid[index])).name or "world"
        return f"{body_name}/geom_{index}"

    group2 = [geom_label(index) for index in range(model.ngeom) if int(model.geom_group[index]) == 2]
    skin_geoms = []
    for index in range(model.ngeom):
        if int(model.geom_group[index]) != 2:
            continue
        body_id = int(model.geom_bodyid[index])
        while body_id > 0:
            body_name = model.body(body_id).name or ""
            if "skin" in body_name.lower():
                skin_geoms.append(geom_label(index))
                break
            body_id = int(model.body_parentid[body_id])
    group4 = [
        geom_label(index)
        for index in range(model.ngeom)
        if int(model.geom_group[index]) == 4
    ]
    try:
        task_description = task.get_task_description()
    except Exception:
        task_description = None

    record = {
        "config": config_name,
        "config_class": f"{type(config).__module__}:{type(config).__name__}",
        "house": house,
        "sample": sample_index,
        "seed": args.seed,
        "scene": str(env.current_model_path),
        "scene_dataset": config.scene_dataset,
        "data_split": config.data_split,
        "scene_variant": variant,
        "task_description": task_description,
        "robot_model": _robot_model_label(config),
        "proximity_sensor_count": len(sensor_names),
        "proximity_sensors": sensor_names,
        "robot_visual_and_skin_geom_count": len(group2),
        "robot_visual_and_skin_geoms": group2,
        "skin_geom_count": len(skin_geoms),
        "skin_geoms": skin_geoms,
        "sensor_only_geom_count": len(group4),
        "sensor_only_geoms": group4,
        "sensor_only_geoms_rendered": args.show_hidden,
        "sensor_markers_rendered": args.show_sensors,
        "scene_params": getattr(task, "scene_params", None),
        "camera": {
            "lookat": lookat,
            "distance": distance,
            "elevation": args.elevation,
            "presentation_views": [
                {"azimuth": azimuth, "elevation": elevation}
                for azimuth, elevation in views
            ],
        },
        "image_text_overlays": False,
        "presentation_headlight": {
            "enabled": not args.keep_scene_lighting,
            "ambient": PRESENTATION_HEADLIGHT_AMBIENT,
            "diffuse": PRESENTATION_HEADLIGHT_DIFFUSE,
            "specular": PRESENTATION_HEADLIGHT_SPECULAR,
        },
        "outputs": [str(path) for path in outputs],
    }
    record = _jsonable(record)
    (output_dir / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="Registered config names or module.path:ConfigName references",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List unique environment sources used by proximity-sensor configs and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render one sample of every unique environment (deduped by scene + robot)",
    )
    parser.add_argument(
        "--scope",
        choices=("project", "all"),
        default=None,
        help="Which configs --list/--all consider. Default: all for --list, project for --all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the --all render plan and exit without sampling or writing images",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root")
    parser.add_argument("--format", choices=("png", "mp4", "both"), default="png")
    parser.add_argument(
        "--house",
        type=int,
        action="append",
        help="Render this house index; repeat for more than one",
    )
    parser.add_argument(
        "--all-houses",
        action="store_true",
        help="Render every configured house instead of one per unique XML",
    )
    parser.add_argument("--samples", type=int, default=1, help="Sampled episodes per selected house")
    parser.add_argument("--attempts", type=int, default=3, help="Task-sampling attempts per output")
    parser.add_argument(
        "--variant",
        default=None,
        help="Scene variant (automatic: base for user XMLs, ceiling for scene datasets)",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Task-sampling seed")
    parser.add_argument(
        "--keep-config-robot",
        action="store_true",
        help="Keep the config's own robot/camera stack (default: force model_hybrid.xml + 40 SPADs)",
    )
    parser.add_argument(
        "--no-randomization",
        action="store_true",
        help="Disable generic texture, lighting, robot-texture, and dynamics randomizers",
    )
    parser.add_argument(
        "--keep-scene-lighting",
        action="store_true",
        help="Preserve sampled lighting instead of adding presentation headlight fill",
    )
    parser.add_argument(
        "--show-hidden",
        action="store_true",
        help="Reveal sensor-only geom group 4, including invisible hazard bars",
    )
    parser.add_argument(
        "--show-sensors",
        action="store_true",
        help="Draw MuJoCo camera markers for MJCF wrist and skin sensors",
    )
    parser.add_argument("--width", type=int, default=960, help="Width of each rendered view")
    parser.add_argument("--height", type=int, default=540, help="Height of each rendered view")
    parser.add_argument("--lookat", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--distance", type=float, default=None, help="Free-camera distance")
    parser.add_argument("--azimuth", type=float, default=45.0, help="MP4 starting azimuth")
    parser.add_argument("--elevation", type=float, default=-18.0, help="Free-camera elevation")
    parser.add_argument("--seconds", type=float, default=6.0, help="MP4 duration")
    parser.add_argument("--fps", type=float, default=30.0, help="MP4 frame rate")
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="warning",
    )
    return parser


def _resolve_scope(args: argparse.Namespace) -> str:
    if args.scope is not None:
        return args.scope
    return "project" if args.all else "all"


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.list:
        return
    if args.dry_run and not args.all:
        parser.error("--dry-run only applies to --all")
    if args.all:
        if args.configs:
            parser.error("--all chooses configs itself; do not pass config names")
        if args.house:
            parser.error("--all picks one house per unique scene; do not pass --house")
        if args.all_houses:
            parser.error("--all already dedupes scenes; do not pass --all-houses")
    elif not args.configs:
        parser.error("provide at least one config, or use --list / --all")
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.attempts < 1:
        parser.error("--attempts must be >= 1")
    if args.width < 64 or args.height < 64:
        parser.error("--width and --height must be >= 64")
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("--seconds and --fps must be positive")


def _plan_all_jobs(
    scope: str, force_hybrid: bool
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    groups, skipped = _collect_environment_groups(scope, force_hybrid=force_hybrid)
    jobs = _jobs_by_config(groups)
    print(
        f"--all will render {len(groups)} unique environments "
        f"(scope={scope}, force_hybrid={force_hybrid}):"
    )
    for group in groups:
        config_name, house = _representative_job(group)
        source = Path(group["source"]).name if "/" in group["source"] else group["source"]
        missing = ""
        if group["source"].endswith(".xml") and not Path(group["source"]).exists():
            missing = "  ** MISSING XML **"
        print(
            f"  {config_name} house {house}  <- {source}  "
            f"({group['robot']}, {group['sensor_count']} sensors){missing}"
        )
    if skipped:
        print(f"skipped {len(skipped)} configs that could not be instantiated")
    return groups, jobs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    force_hybrid = not args.keep_config_robot

    if args.list:
        list_proximity_environments(_resolve_scope(args), force_hybrid=force_hybrid)
        return

    houses_by_config: dict[str, list[int] | None] = {}
    if args.all:
        _groups, jobs = _plan_all_jobs(_resolve_scope(args), force_hybrid=force_hybrid)
        if args.dry_run:
            return
        args.configs = list(jobs)
        houses_by_config = jobs
    else:
        houses_by_config = {config_ref: args.house for config_ref in args.configs}

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for config_ref in args.configs:
        task_sampler = None
        current_task = None
        try:
            config_class, config_name = _load_config_class(config_ref)
            config = config_class()
            _prepare_config(config, args)
            requested_houses = houses_by_config.get(config_ref)
            houses = _selected_houses(config, requested_houses, args.all_houses)
            variant = args.variant or ("base" if config.scene_dataset == "user" else "ceiling")
            sampler_class = config.task_sampler_config.task_sampler_class
            if sampler_class is None:
                raise RuntimeError(f"{config_name} has no task sampler class")

            print(
                f"{config_name}: houses={houses}, variant={variant}, robot={_robot_model_label(config)}, "
                f"proximity_sensors={len(_proximity_sensor_names(config))}"
            )
            task_sampler = sampler_class(config)

            for house in houses:
                for sample_index in range(args.samples):
                    if current_task is not None:
                        current_task.close()
                        current_task = None

                    print(f"  sampling house {house}, sample {sample_index}...")
                    try:
                        current_task = _sample_task(
                            task_sampler, house, variant, args.attempts
                        )
                        scene_path = Path(current_task.env.current_model_path)
                        output_dir = (
                            args.out
                            / _slug(config_name)
                            / f"{_slug(scene_path.stem)}_house_{house}"
                            / f"sample_{sample_index:02d}"
                        )
                        outputs, lookat, distance, views = _render_outputs(
                            current_task,
                            config,
                            output_dir,
                            args,
                        )
                        record = _write_metadata(
                            current_task,
                            config,
                            config_name,
                            house,
                            sample_index,
                            output_dir,
                            outputs,
                            lookat,
                            distance,
                            views,
                            args,
                            variant,
                        )
                        records.append(record)
                        for output in outputs:
                            print(f"  wrote {output}")
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        failure = {
                            "config": config_name,
                            "house": str(house),
                            "sample": str(sample_index),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        failures.append(failure)
                        print(f"  FAILED: {failure['error']}", file=sys.stderr)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            failures.append(
                {
                    "config": config_ref,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"{config_ref}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            if current_task is not None:
                current_task.close()
            if task_sampler is not None:
                task_sampler.close()

    index = {"renders": records, "failures": failures}
    index_path = args.out / "index.json"
    index_path.write_text(json.dumps(_jsonable(index), indent=2) + "\n")
    print(f"wrote {index_path}")
    gallery_path = _write_gallery(records, args.out)
    if gallery_path is not None:
        print(f"wrote {gallery_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
