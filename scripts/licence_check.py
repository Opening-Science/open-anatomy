"""
Licence check: every asset folder carries its licence, and the repository root
carries LICENSE and NOTICE. Runs in CI and needs only Python.

Run it from the repository root, or pass the root as the only argument:
  python3 scripts/licence_check.py [ROOT]

Each immediate subfolder of assets/ (for example assets/cc-by/) must contain a
file named LICENSE. A missing file is printed and the script exits 1. It exits
0 when nothing is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_FILES = ("LICENSE", "NOTICE")


def missing_licences(root: Path) -> list[str]:
    """Paths, relative to root, of the licence files that should exist but do not."""
    missing = [name for name in ROOT_FILES if not (root / name).is_file()]
    assets = root / "assets"
    if assets.is_dir():
        for folder in sorted(p for p in assets.iterdir() if p.is_dir()):
            if not (folder / "LICENSE").is_file():
                missing.append(f"assets/{folder.name}/LICENSE")
    return missing


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path.cwd()
    missing = missing_licences(root)
    for path in missing:
        print(f"missing: {path}")
    if missing:
        print(f"licence check failed: {len(missing)} file(s) missing")
        return 1
    print("licence check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
