"""Checks on the repository's own tooling setup."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER_IMPORT = re.compile(r"^\s*(import|from)\s+(bpy|bmesh|mathutils)\b", re.MULTILINE)


def test_blender_scripts_are_excluded_from_pyright() -> None:
    # Scripts that import Blender-only modules cannot be type checked outside
    # Blender, so pyproject.toml must list each of them under [tool.pyright]
    # exclude, and must not list anything else there by mistake.
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    excluded = {p for p in config["tool"]["pyright"]["exclude"] if p.endswith(".py")}
    blender = {
        str(path.relative_to(ROOT))
        for path in ROOT.glob("scripts/**/*.py")
        if BLENDER_IMPORT.search(path.read_text())
    }
    assert blender == excluded
