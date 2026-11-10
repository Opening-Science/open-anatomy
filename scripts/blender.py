"""
Runs the headless Blender scripts in blender/ with a checked Blender.

  python scripts/blender.py self-check
  python scripts/blender.py cleanup --in <glb> --out-dir <dir> [cleanup_mesh.py options]
  python scripts/blender.py qa --in <glb> --manifest <json> --out-dir <dir>
  python scripts/blender.py assemble --body <glb> --ct <glb> --out <glb> --report <json>
  python scripts/blender.py fit --atlas <glb> --scheme hra --body <glb> \
      --out <glb> --report <json> [fit_atlas.py options]
      (--scheme bodyparts3d also needs --allow-share-alike: its licence may be CC BY-SA)

Blender comes from BLENDER_BIN, else `blender` on PATH, and must be 5.2.x LTS.
Without one, self-check says so and exits 0, so CI (which has none) skips;
the other tasks fail. self-check runs this repository's glTF round trip, both
ported scripts and qa.py on synthetic meshes. Any argument that points under a
data/vitronic/ folder is refused, input or output, given as `--flag path` or
`--flag=path`: the scripts are only ever run on synthetic bodies, open atlases
and open CT data, never on a scan of a person. self-check tests that guard
first, with or without Blender.

Ported from twin-llm's scripts/blender/run.ts and blender-bin.ts (a4058fa8,
Apache-2.0).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERIES = (5, 2)
PERSONAL = re.compile(r"(^|[\\/])data[\\/]vitronic([\\/]|$)")


def find_blender() -> str | None:
    from_env = os.environ.get("BLENDER_BIN", "").strip()
    if from_env:
        if not Path(from_env).is_file():
            sys.exit(f"BLENDER_BIN is set to {from_env}, but there is no file there.")
        return from_env
    return shutil.which("blender")


def version(blender: str) -> tuple[int, int, int] | None:
    out = subprocess.run([blender, "--version"], capture_output=True, text=True).stdout
    match = re.search(r"^Blender (\d+)\.(\d+)(?:\.(\d+))?", out, re.M)
    return tuple(int(g or 0) for g in match.groups()) if match else None


def run(blender: str, script: Path, args: list[str]) -> int:
    command = [blender, "--background", "--factory-startup", "--python-exit-code", "1"]
    return subprocess.run([*command, "--python", str(script), "--", *args], cwd=ROOT).returncode


def personal_paths(args: list[str]) -> list[str]:
    """Every argument that points under data/vitronic/, whichever flag it belongs to.

    Each argument is checked, and the value of each `--flag=value`, so a new
    input or output flag needs no entry here. A relative path is checked both
    from here and from the repository root, where Blender runs.
    """
    values = [a.split("=", 1)[1] if a.startswith("--") and "=" in a else a for a in args]
    found = []
    for value in filter(None, values):
        seen = (value, str(Path(value).resolve()), str((ROOT / value).resolve()))
        if any(PERSONAL.search(v) for v in seen):
            found.append(value)
    return found


def guard_check() -> list[str]:
    """Failures of personal_paths on cases it must refuse and cases it must let through."""
    refused = [
        ["--in", "data/vitronic/scan.glb"],
        ["--in=data/vitronic/scan.glb"],
        ["--body", "x.glb", "--out", "data/vitronic/out.glb"],
        ["--render-dir=data/vitronic/renders"],
        ["--outline", "/home/someone/twin-llm/data/vitronic/p1/outline.glb"],
        ["--report", str(ROOT / "data" / "vitronic" / "fit.json")],
    ]
    allowed = [
        ["--in", "data/work/s1086/s1086.organs.glb", "--out-dir", "data/work/clean"],
        ["--atlas=data/atlas/hra.glb", "--scheme", "hra", "--max-shift", "0.5"],
        ["--out", "data/vitronic-like/x.glb"],
    ]
    failures = [f"not refused: {a}" for a in refused if not personal_paths(a)]
    return failures + [f"refused: {a}" for a in allowed if personal_paths(a)]


def main() -> int:
    task, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if task not in ("self-check", "cleanup", "fit", "assemble", "qa", "retopo", "studio"):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    if task == "self-check":
        if failures := guard_check():
            print("personal-data guard failed: " + "; ".join(failures), file=sys.stderr)
            return 1
        print("personal-data guard passed", flush=True)
    blender = find_blender()
    if blender is None:
        # Only the self-check skips (for CI); a build without Blender must fail.
        print("Blender was not found: set BLENDER_BIN to a Blender 5.2.x binary.", file=sys.stderr)
        return 0 if task == "self-check" else 1
    found = version(blender)
    if found is None or found[:2] != SERIES:
        shown = ".".join(map(str, found)) if found else "not a Blender this script can read"
        print(f"{blender} is {shown}; the pipeline needs Blender 5.2.x LTS.", file=sys.stderr)
        return 1
    if refused := personal_paths(args):
        print(f"Refusing personal data: {', '.join(refused)}", file=sys.stderr)
        return 2

    if task == "self-check":
        checks = [
            (ROOT / "scripts" / "self_check.py", []),
            (ROOT / "blender" / "fit_atlas.py", ["--self-check"]),
            (ROOT / "blender" / "cleanup_mesh.py", ["--self-check"]),
            (ROOT / "blender" / "qa.py", ["--self-check"]),
            (ROOT / "blender" / "retopo.py", ["--self-check"]),
            (ROOT / "blender" / "studio.py", ["--self-check"]),
        ]
        failed = [s.name for s, a in checks if run(blender, s, a) != 0]
        print("all Blender self-checks passed" if not failed else f"failed: {', '.join(failed)}")
        return 1 if failed else 0
    script = {
        "cleanup": "cleanup_mesh.py", "fit": "fit_atlas.py", "assemble": "assemble.py",
        "qa": "qa.py", "retopo": "retopo.py", "studio": "studio.py",
    }[task]  # fmt: skip
    return run(blender, ROOT / "blender" / script, args)


if __name__ == "__main__":
    sys.exit(main())
