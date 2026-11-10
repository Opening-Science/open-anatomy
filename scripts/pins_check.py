"""
Pins check: every file that sources.json pins by SHA-256 and that is built or
downloaded here, under the gitignored data/ folder, must match its pin.

  python3 scripts/pins_check.py [--skip-sources] [ROOT]

sources.json pins:
- each source archive, data/sources/<file>;
- each case's CT, data/cases/<case>/ct.nii.gz (ct_sha256);
- each baked body, data/bodies/<file>;
- each reference body's built GLB, data/reference/<file>, and the CT meshes it
  was assembled from, data/work/<case>/<case>.reference.glb (ctSha256).

A file that is on disk and differs from its pin is printed with both hashes
and the script exits 1. A file that is absent is printed as "not built here"
and does not fail, so the check runs on a fresh clone. Run it after a rebuild
and before pushing a change to sources.json: data/ is not on the CI runner, so
CI cannot. --skip-sources leaves the source archives out (26 GB to hash;
fetch.py checks them whenever it unpacks a case).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

NOT_BUILT = "not built here"
MATCHES = "matches sources.json"


def pins(sources: dict[str, Any], skip_sources: bool = False) -> list[tuple[str, str]]:
    """Every (path relative to the repository root, pinned SHA-256) in sources.json."""
    found: list[tuple[str, str]] = []
    if not skip_sources:
        for source in sources.get("sources", {}).values():
            found.append((f"data/sources/{source['file']}", source["sha256"]))
    for case, entry in sources.get("cases", {}).items():
        if "ct_sha256" in entry:
            found.append((f"data/cases/{case}/ct.nii.gz", entry["ct_sha256"]))
    for body in sources.get("bodies", {}).values():
        found.append((f"data/bodies/{body['file']}", body["sha256"]))
    for reference in sources.get("references", {}).values():
        built = reference.get("built", {})
        case = reference["case"]
        if "sha256" in built:
            found.append((f"data/reference/{built['file']}", built["sha256"]))
        if "ctSha256" in built:
            found.append((f"data/work/{case}/{case}.reference.glb", built["ctSha256"]))
    return found


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def status(root: Path, path: str, pinned: str) -> str:
    """What one pinned file is here: MATCHES, NOT_BUILT, or how it differs."""
    file = root / path
    if not file.is_file():
        return NOT_BUILT
    found = sha256(file)
    return MATCHES if found == pinned else f"sha256 {found}, sources.json pins {pinned}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--skip-sources", action="store_true", help="leave data/sources/ out")
    args = parser.parse_args(argv[1:])
    root: Path = args.root
    sources: dict[str, Any] = json.loads((root / "sources.json").read_text())
    counts = {"match": 0, "absent": 0, "differ": 0}
    for path, pinned in pins(sources, args.skip_sources):
        result = status(root, path, pinned)
        counts["match" if result == MATCHES else "absent" if result == NOT_BUILT else "differ"] += 1
        print(f"{path}: {result}", flush=True)
    summary = f"{counts['match']} match, {counts['absent']} not built here"
    if counts["differ"]:
        print(f"pins check failed: {counts['differ']} differ from sources.json ({summary})")
        return 1
    print(f"pins check passed: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
