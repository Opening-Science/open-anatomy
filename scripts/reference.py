"""
Builds the reference bodies end to end: for each ANNY body in sources.json's
"references", fetch and segment its CT case, mesh the organs and trunk
skeleton, check the baked body against its pin, and assemble them in Blender.

  python scripts/reference.py [<reference> ...] [--render] [--skip-segment]

A reference names its ANNY body and its CT case in sources.json; bodies of
other sizes are baked with the weight and height pinned there.

Writes data/reference/<body>.reference.glb and <body>.report.json, with
--render three PNGs per body in data/renders/, and with --qa the close-up QA
(blender/qa.py) and its contact sheets in data/qa/<reference>/. Each step is a
script on its own (fetch.py, segment.py, mesh.py, bake_anny.py,
blender/assemble.py); this only runs them in order and stops at the first that
fails. It copies the case's record (pathology, location, whether the body is
typical) from sources.json into the report. Needs Blender 5.2.x
(scripts/blender.py finds it) and uv for the ANNY bake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = json.loads((ROOT / "sources.json").read_text())


def step(*command: str | Path) -> None:
    print("+", " ".join(str(c) for c in command), flush=True)
    if subprocess.run([str(c) for c in command], cwd=ROOT).returncode != 0:
        sys.exit(f"stopped: {Path(str(command[1])).name} failed")


def body_file(name: str) -> Path:
    pin = SOURCES["bodies"][name]
    path = ROOT / "data" / "bodies" / pin["file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        size = [a for k in ("weight", "height") if k in pin for a in (f"--{k}", str(pin[k]))]
        step("uv", "run", "--script", ROOT / "scripts" / "bake_anny.py", pin["sex"], path, *size)
    found = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != pin["sha256"]:
        sys.exit(f"{path.name}: sha256 {found} does not match {pin['sha256']}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("bodies", nargs="*", default=list(SOURCES["references"]))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip-segment", action="store_true", help="reuse existing masks")
    parser.add_argument("--qa", action="store_true", help="close-up QA and contact sheets")
    args = parser.parse_args()
    python = sys.executable
    blender = [python, ROOT / "scripts" / "blender.py"]

    for name in args.bodies:
        case = SOURCES["references"][name]["case"]
        step(python, ROOT / "scripts" / "fetch.py", "--case", case)
        if not args.skip_segment:
            step(python, ROOT / "scripts" / "segment.py", "--case", case, "--set", "reference")
        step(python, ROOT / "scripts" / "mesh.py", "--case", case, "--set", "reference")
        if args.qa:
            work = ROOT / "data" / "work" / case
            qa_dir = ROOT / "data" / "qa" / name
            glb, manifest = work / f"{case}.reference.glb", work / f"{case}.reference.manifest.json"
            step(*blender, "qa", "--in", glb, "--manifest", manifest, "--out-dir", qa_dir)
            step(python, ROOT / "scripts" / "qa_sheet.py", "--qa-dir", qa_dir, "--title",
                 f"{name} ({case})")  # fmt: skip
        body = body_file(SOURCES["references"][name].get("body", name))
        out = ROOT / "data" / "reference"
        assemble = [
            "--body", body,
            "--ct", ROOT / "data" / "work" / case / f"{case}.reference.glb",
            "--outline", ROOT / "data" / "work" / case / f"{case}.outline.glb",
            "--out", out / f"{name}.reference.glb",
            "--report", out / f"{name}.report.json",
        ]  # fmt: skip
        marks = ROOT / "data" / "work" / case / f"{case}.landmarks.glb"
        if marks.exists():
            assemble += ["--landmarks", marks]
        if args.render:
            assemble += ["--render-dir", ROOT / "data" / "renders", "--render-name", name]
        step(*blender, "assemble", *assemble)
        # The case's record goes into the body's report, so a pathological
        # reference is never mistaken for a typical one.
        record = SOURCES["cases"][case].get("record", {})
        report_path = out / f"{name}.report.json"
        report = json.loads(report_path.read_text())
        report["case"] = {"id": case, **record}
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if record.get("typical") is False:
            print(f"{name}: PATHOLOGICAL reference ({record.get('pathology')}, "
                  f"{record.get('location')}), not typical anatomy")  # fmt: skip


if __name__ == "__main__":
    main()
