"""
Runs the headless Blender scripts in blender/ with a checked Blender.

  python scripts/blender.py self-check
  python scripts/blender.py cleanup --in <glb> --out-dir <dir> [cleanup_mesh.py options]
  python scripts/blender.py fit --atlas <glb> --scheme bodyparts3d|hra --body <glb> \
      --out <glb> --report <json> [fit_atlas.py options]

Blender comes from BLENDER_BIN, else `blender` on PATH, and must be 5.2.x LTS.
Without one it says so and exits 0, so CI (which has none) skips. self-check
runs this repository's glTF round trip and both ported scripts on synthetic
meshes. Inputs under a data/vitronic/ folder are refused: the scripts are only
ever run on synthetic bodies, open atlases and open CT data, never on a scan of
a person.

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


def personal_inputs(args: list[str]) -> list[str]:
    paths = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("--in", "--atlas", "--body")]
    return [p for p in paths if PERSONAL.search(str(Path(p).resolve()))]


def main() -> int:
    task, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if task not in ("self-check", "cleanup", "fit"):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    blender = find_blender()
    if blender is None:
        print("Blender was not found: set BLENDER_BIN to a Blender 5.2.x binary. Skipping.")
        return 0
    found = version(blender)
    if found is None or found[:2] != SERIES:
        shown = ".".join(map(str, found)) if found else "not a Blender this script can read"
        print(f"{blender} is {shown}; the pipeline needs Blender 5.2.x LTS.", file=sys.stderr)
        return 1
    if refused := personal_inputs(args):
        print(f"Refusing personal data: {', '.join(refused)}", file=sys.stderr)
        return 2

    if task == "self-check":
        checks = [
            (ROOT / "scripts" / "self_check.py", []),
            (ROOT / "blender" / "fit_atlas.py", ["--self-check"]),
            (ROOT / "blender" / "cleanup_mesh.py", ["--self-check"]),
        ]
        failed = [s.name for s, a in checks if run(blender, s, a) != 0]
        print("all Blender self-checks passed" if not failed else f"failed: {', '.join(failed)}")
        return 1 if failed else 0
    script = "cleanup_mesh.py" if task == "cleanup" else "fit_atlas.py"
    return run(blender, ROOT / "blender" / script, args)


if __name__ == "__main__":
    sys.exit(main())
