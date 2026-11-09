"""
Checks scripts/bake_anny.py's lock and, when asked, its output against sources.json.

  python scripts/bake_check.py          # the lock, and any baked body already in data/bodies/
  python scripts/bake_check.py --bake   # also bakes every body again and compares the bytes

The lock (scripts/bake_anny.py.lock) must have a torch wheel for every
platform the bake supports: Linux x86_64 and aarch64, Windows x86_64 and
macOS on Apple Silicon. A body in data/bodies/ must match the bytes and
SHA-256 that sources.json pins for it. With --bake each body is baked again
with `uv run --script` (its made_by line) into a temporary folder and compared
the same way; sources.json pins Linux x86_64 output, so other machines may
differ in the last bits (see bake_anny.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "scripts" / "bake_anny.py.lock"
PLATFORMS = {
    "Linux x86_64": "manylinux_2_28_x86_64",
    "Linux aarch64": "manylinux_2_28_aarch64",
    "Windows x86_64": "win_amd64",
    "macOS arm64": "macosx_14_0_arm64",
}


def lock_failures() -> list[str]:
    packages = tomllib.loads(LOCK.read_text())["package"]
    wheels = [w["url"] for p in packages if p["name"] == "torch" for w in p.get("wheels", [])]
    failures = [f"no torch wheel for {name}" for name, tag in PLATFORMS.items()
                if not any(w.endswith(f"{tag}.whl") for w in wheels)]  # fmt: skip
    for p in packages:
        if not p.get("wheels") and not p.get("sdist") and "virtual" not in p["source"]:
            failures.append(f"{p['name']} {p['version']} has neither a wheel nor an sdist")
    return failures


def body_failures(path: Path, body: dict) -> list[str]:
    data = path.read_bytes()
    found = hashlib.sha256(data).hexdigest()
    if len(data) != body["bytes"] or found != body["sha256"]:
        return [f"{path.name}: {len(data)} bytes, sha256 {found}; sources.json pins "
                f"{body['bytes']} bytes, {body['sha256']}"]  # fmt: skip
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bake", action="store_true", help="bake every body again and compare")
    args = parser.parse_args()
    failures = lock_failures()
    print("lock: " + ("ok" if not failures else "; ".join(failures)))
    bodies = json.loads((ROOT / "sources.json").read_text())["bodies"]
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in bodies.items():
            script, *bake_args = shlex.split(body["made_by"])
            if args.bake:
                path = Path(tmp) / body["file"]
                command = ["uv", "run", "--script", str(ROOT / script), *bake_args, str(path)]
                subprocess.run(command, check=True, cwd=ROOT)
            else:
                path = ROOT / "data" / "bodies" / body["file"]
                if not path.exists():
                    print(f"{name}: not baked here; skipped")
                    continue
            problems = body_failures(path, body)
            print(f"{name}: " + ("matches sources.json" if not problems else problems[0]))
            failures += problems
    print("bake check passed" if not failures else f"bake check failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
