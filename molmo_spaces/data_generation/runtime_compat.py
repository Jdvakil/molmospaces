"""Startup compatibility check for the data-generation runtime.

The hybrid-obstacle collection only runs correctly on a narrow band of the
MuJoCo/Warp stack:

* ``mujoco-warp`` 3.5.x reads ``mujoco.mjtEnableBit.mjENBL_MULTICCD``, which
  MuJoCo removed after 3.5, so a newer MuJoCo fails at import time.
* ``warp-lang`` tightened ``wp.copy`` dtype checking after 1.11. On newer Warp,
  ``SimpleWarpKinematics.ik`` raises ``RuntimeError: Incompatible array data
  types`` at ``task.reset()``, so every rollout is rejected and the run produces
  zero episodes while still exiting successfully.

The second failure is the dangerous one: it looks like a task-sampling problem,
not a dependency problem. This module reports it up front, by version, instead of
letting it surface as an empty dataset.

Planner mathematics is never adjusted to accommodate a different Warp release;
the supported versions are pinned instead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import metadata

# Versions the canonical hybrid-obstacle collection was validated on.
SUPPORTED = {
    "mujoco": {"pinned": "3.5.0", "supported": ("3.5.0",)},
    "warp-lang": {"pinned": "1.11.1", "supported": ("1.11.0", "1.11.1")},
}
MIN_PYTHON = (3, 11)


@dataclass
class CompatIssue:
    package: str
    found: str | None
    pinned: str
    detail: str

    def __str__(self) -> str:
        found = self.found or "not installed"
        return f"{self.package}=={found} (pinned {self.pinned}): {self.detail}"


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def check_runtime() -> list[CompatIssue]:
    """Return an issue per unsupported dependency; empty means compatible."""
    issues: list[CompatIssue] = []

    if sys.version_info < MIN_PYTHON:
        issues.append(
            CompatIssue(
                package="python",
                found=".".join(map(str, sys.version_info[:3])),
                pinned=".".join(map(str, MIN_PYTHON)),
                detail="molmo-spaces requires Python >= 3.11",
            )
        )

    mujoco_version = installed_version("mujoco")
    if mujoco_version not in SUPPORTED["mujoco"]["supported"]:
        issues.append(
            CompatIssue(
                package="mujoco",
                found=mujoco_version,
                pinned=SUPPORTED["mujoco"]["pinned"],
                detail=(
                    "mujoco-warp 3.5.x requires mjtEnableBit.mjENBL_MULTICCD, which "
                    "later MuJoCo releases removed; import of mujoco_warp will fail"
                ),
            )
        )

    warp_version = installed_version("warp-lang")
    if warp_version not in SUPPORTED["warp-lang"]["supported"]:
        issues.append(
            CompatIssue(
                package="warp-lang",
                found=warp_version,
                pinned=SUPPORTED["warp-lang"]["pinned"],
                detail=(
                    "newer warp-lang enforces strict dtype matching in wp.copy, so "
                    "SimpleWarpKinematics.ik raises 'Incompatible array data types' at "
                    "task.reset() and every rollout is rejected — the run then "
                    "completes with zero episodes instead of failing"
                ),
            )
        )
    return issues


def format_report(issues: list[CompatIssue]) -> str:
    if not issues:
        return "runtime compatibility: OK"
    lines = ["UNSUPPORTED DATA-GENERATION RUNTIME:"]
    lines += [f"  - {issue}" for issue in issues]
    lines.append("  pin with:")
    lines.append(
        "    pip install "
        + " ".join(f'"{name}=={spec["pinned"]}"' for name, spec in SUPPORTED.items())
    )
    return "\n".join(lines)


def assert_supported_runtime(strict: bool = True, logger=None) -> list[CompatIssue]:
    """Report unsupported versions; raise when ``strict``."""
    issues = check_runtime()
    report = format_report(issues)
    if issues:
        if logger is not None:
            logger.error(report)
        else:
            print(report, file=sys.stderr)
        if strict:
            raise RuntimeError(report)
    elif logger is not None:
        logger.info(report)
    return issues


if __name__ == "__main__":
    found = check_runtime()
    print(format_report(found))
    sys.exit(1 if found else 0)
