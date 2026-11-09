"""
Data guard: fails when git tracks a file that belongs in the gitignored data/
folder instead. Runs in CI and needs only Python and git.

Run it from anywhere inside the repository:
  python3 scripts/data_guard.py

A tracked file is an offender when it is under data/, when it is a medical
image (NIfTI, DICOM, NRRD or MetaImage, by extension, in any case), when it is
a mesh outside assets/ and tests/fixtures/ (a body scan arrives as a mesh, and
a low-detail one can be under 5 MB), or when it is larger than 5 MB. Each
offender is printed with its reason and the script exits 1. It exits 0 when there are none.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
DATA_DIR = "data/"
# Compared against the lower-cased path, so .NII.GZ and .Dcm match too.
IMAGE_SUFFIXES = (".nii", ".nii.gz", ".dcm", ".nrrd", ".mha", ".mhd")
MESH_SUFFIXES = (".glb", ".gltf", ".ply", ".obj", ".stl", ".fbx", ".vtk", ".vtp", ".usdz", ".blend")
MESH_FOLDERS = ("assets/", "tests/fixtures/")  # released assets and synthetic test meshes


def reasons(path: str, size: int | None) -> list[str]:
    """Why one tracked file must not be in git; empty when it is fine.

    `path` is relative to the repository root with forward slashes, as git
    prints it. `size` is in bytes, or None when the file is not on disk.
    """
    found: list[str] = []
    if path.startswith(DATA_DIR):
        found.append("it is under data/, which must stay out of git")
    lower = path.lower()
    for suffix in IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            found.append(f"it is a medical image ({suffix})")
            break
    if lower.endswith(MESH_SUFFIXES) and not path.startswith(MESH_FOLDERS):
        found.append("it is a mesh outside assets/ and tests/fixtures/")
    if size is not None and size > MAX_BYTES:
        found.append(f"it is {size / 1024 / 1024:.1f} MB, over the 5 MB limit")
    return found


def offenders(root: Path, paths: Iterable[str]) -> list[tuple[str, str]]:
    """Every (path, reason) pair for the given paths, read relative to root."""
    found: list[tuple[str, str]] = []
    for path in paths:
        file = root / path
        # A file deleted in the work tree but still in the index has no size.
        size = file.lstat().st_size if file.is_file() or file.is_symlink() else None
        found.extend((path, reason) for reason in reasons(path, size))
    return found


def tracked_files(root: Path) -> list[str]:
    """The paths git tracks under root, relative to the repository root."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--full-name"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    ).stdout
    return Path(out.strip())


def main() -> int:
    root = repo_root()
    found = offenders(root, tracked_files(root))
    for path, reason in found:
        print(f"{path}: {reason}")
    if found:
        print(f"data guard failed: {len(found)} problem(s). Keep data in the gitignored data/.")
        return 1
    print("data guard passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
