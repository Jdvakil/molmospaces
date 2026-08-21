#!/usr/bin/env python3
"""Sample configured MolmoSpaces environments and render inspection images or MP4s.

This uses the same config and task sampler as data generation. The resulting scene therefore
contains the configured robot model, cosmetic skin, sampled pickup object, per-episode geometry,
clutter, and texture/lighting randomization. It does not execute a policy or save a trajectory.

Examples:
    python scripts/datagen/visualize_environment.py --list
    python scripts/datagen/visualize_environment.py --all --show-hidden --show-sensors
    python scripts/datagen/visualize_environment.py \
        FrankaSkinHybridClutterPnPCheckConfig --format both
    python scripts/datagen/visualize_environment.py \
        FrankaSkinHybridInvisObstacleCheckConfig --show-hidden --show-sensors
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
from PIL import Image, ImageDraw, ImageFont

from molmo_spaces.data_generation.config_registry import (
    get_config_class,
    list_available_configs,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

CONFIG_PACKAGE = "molmo_spaces.data_generation.config"
DEFAULT_OUT = Path(ASSETS_DIR).resolve().parent / "experiments_output/default/environment_viz"
VIEW_AZIMUTHS = (45.0, 135.0, 225.0, 315.0)
# Datagen configs this project actually collects from (see README §7 / §12).
PROJECT_CONFIG_RE = re.compile(
    r"^FrankaSkin("
    r"Cabinet|Shelf|Clutter|Pillar|RealTable|RealHouse|Enclosure|"
    r"Fumehood|Panel|Cubby|House|Hybrid|ProxNecessity"
    r")"
)


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
) -> tuple[list[dict[str, Any]], list[str]]:
    """Group registered proximity configs by unique (scene source, robot)."""
    _auto_import_configs()
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: list[str] = []

    for name in sorted(list_available_configs()):
        if scope == "project" and not _is_project_config(name):
            continue
        try:
            config = get_config_class(name)()
        except Exception:
            skipped.append(name)
            continue

        sensors = _proximity_sensor_names(config)
        if not sensors:
            continue

        houses = _configured_houses(config)
        robot = _robot_model_label(config)
        for label, source_houses in _source_keys(config, houses).items():
            key = (label, robot)
            group = groups.get(key)
            if group is None:
                group = {
                    "source": label,
                    "robot": robot,
                    "sensor_count": len(sensors),
                    "config_houses": {},
                    "houses": set(),
                }
                groups[key] = group
            group["config_houses"].setdefault(name, set()).update(source_houses)
            group["houses"].update(source_houses)

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


def list_proximity_environments(scope: str) -> None:
    """Print unique scene sources used by registered proximity-sensor configs."""
    groups, skipped = _collect_environment_groups(scope)
    print(
        f"Found {len(groups)} unique environment sources "
        f"(scope={scope}, proximity-sensor configs)."
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
    """Write a browsable HTML contact sheet of every environment.png."""
    tiles: list[tuple[Path, str]] = []
    for record in records:
        pngs = [Path(path) for path in record.get("outputs", []) if str(path).endswith(".png")]
        if not pngs:
            continue
        caption = (
            f"{record.get('config')} · {Path(str(record.get('scene', ''))).name} · "
            f"house {record.get('house')}"
        )
        tiles.append((pngs[0], caption))
    if not tiles:
        return None

    gallery_path = output_dir / "gallery.html"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>datagen environment gallery</title>",
        "<style>",
        "body{font-family:sans-serif;background:#111;color:#eee;margin:24px}",
        "h1{font-size:20px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}",
        "figure{margin:0;background:#1c1c1c;padding:8px;border-radius:8px}",
        "img{width:100%;height:auto;display:block;border-radius:4px}",
        "figcaption{font-size:13px;margin-top:8px;color:#c8d0d8}",
        "</style></head><body>",
        f"<h1>{len(tiles)} datagen environments</h1>",
        "<div class='grid'>",
    ]
    for path, caption in tiles:
        rel = os.path.relpath(path, output_dir)
        parts.append(
            f"<figure><img src='{rel}' alt='{caption}'>"
            f"<figcaption>{caption}</figcaption></figure>"
        )
    parts.append("</div></body></html>")
    gallery_path.write_text("\n".join(parts) + "\n")
    return gallery_path


def _prepare_config(config, args: argparse.Namespace) -> None:
    """Trim data-generation-only work while preserving sampled scene content."""
    config.seed = args.seed
    config.num_workers = 1
    config.use_passive_viewer = False
    config.viz_sensor_rgb = False
    config.profile = False
    config.datagen_profiler = False
    config.task_sampler_config.task_batch_size = 1

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
    distance = float(np.clip(2.35 * radius, 1.8, 7.5))
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


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label_frame(frame: np.ndarray, label: str, font_size: int = 18) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    box = draw.textbbox((0, 0), label, font=font)
    height = box[3] - box[1] + 12
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _write_montage(
    frames: list[np.ndarray],
    azimuths: tuple[float, ...],
    output_path: Path,
    title: str,
    subtitle: str,
) -> None:
    height, width = frames[0].shape[:2]
    header_height = 72
    canvas = Image.new("RGB", (2 * width, 2 * height + header_height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill=(255, 255, 255), font=_font(23))
    draw.text((14, 40), subtitle, fill=(195, 205, 215), font=_font(15))

    for index, (frame, azimuth) in enumerate(zip(frames, azimuths)):
        labeled = Image.fromarray(_label_frame(frame, f"external view · azimuth {azimuth:.0f}°"))
        x = (index % 2) * width
        y = header_height + (index // 2) * height
        canvas.paste(labeled, (x, y))
    canvas.save(output_path)


def _render_outputs(
    task,
    output_dir: Path,
    config_name: str,
    scene_label: str,
    args: argparse.Namespace,
) -> tuple[list[Path], np.ndarray, float]:
    env = task.env
    model, data = env.current_model, env.current_data
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)

    auto_lookat, auto_distance = _automatic_camera(task, args.show_hidden)
    lookat = np.asarray(args.lookat, dtype=float) if args.lookat is not None else auto_lookat
    distance = float(args.distance) if args.distance is not None else auto_distance
    option = _scene_option(args.show_hidden, args.show_sensors)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        if args.format in {"png", "both"}:
            frames = []
            for azimuth in VIEW_AZIMUTHS:
                camera = _free_camera(model, lookat, distance, azimuth, args.elevation)
                renderer.update_scene(data, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
                frames.append(renderer.render().copy())
            image_path = output_dir / "environment.png"
            visibility = "physical view (sensor-only group 4 shown)" if args.show_hidden else "policy-visible geometry"
            _write_montage(
                frames,
                VIEW_AZIMUTHS,
                image_path,
                config_name,
                f"{scene_label} · {visibility}",
            )
            written.append(image_path)

        if args.format in {"mp4", "both"}:
            video_path = output_dir / "turntable.mp4"
            frame_count = max(1, int(round(args.seconds * args.fps)))
            with imageio.get_writer(
                video_path,
                fps=args.fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=16,
            ) as writer:
                for index, azimuth in enumerate(
                    np.linspace(args.azimuth, args.azimuth + 360.0, frame_count, endpoint=False)
                ):
                    camera = _free_camera(model, lookat, distance, float(azimuth), args.elevation)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
                    frame = renderer.render().copy()
                    writer.append_data(
                        _label_frame(
                            frame,
                            f"{config_name} · frame {index + 1}/{frame_count}",
                            font_size=16,
                        )
                    )
            written.append(video_path)
    finally:
        renderer.close()

    return written, lookat, distance


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
        "--no-randomization",
        action="store_true",
        help="Disable generic texture, lighting, robot-texture, and dynamics randomizers",
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


def _plan_all_jobs(scope: str) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    groups, skipped = _collect_environment_groups(scope)
    jobs = _jobs_by_config(groups)
    print(f"--all will render {len(groups)} unique environments (scope={scope}):")
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

    if args.list:
        list_proximity_environments(_resolve_scope(args))
        return

    houses_by_config: dict[str, list[int] | None] = {}
    if args.all:
        _groups, jobs = _plan_all_jobs(_resolve_scope(args))
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
                        scene_label = scene_path.name
                        output_dir = (
                            args.out
                            / _slug(config_name)
                            / f"{_slug(scene_path.stem)}_house_{house}"
                            / f"sample_{sample_index:02d}"
                        )
                        outputs, lookat, distance = _render_outputs(
                            current_task,
                            output_dir,
                            config_name,
                            scene_label,
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
