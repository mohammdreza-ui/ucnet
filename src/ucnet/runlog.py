"""
Run manifests: every experiment records its configuration, seeds, library versions and git commit
next to its results, so a result can be traced back to the exact code and settings that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import numpy as np
import scipy


def _git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_manifest(out_dir: Union[str, Path], name: str, config: dict[str, Any]) -> Path:
    """Write `<out_dir>/<name>_manifest.json` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "experiment": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git(["rev-parse", "HEAD"], repo),
        "git_dirty": bool(_git(["status", "--porcelain"], repo)),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "config": config,
    }
    path = out_dir / f"{name}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
