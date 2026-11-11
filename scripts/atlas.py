"""
Builds an atlas skeleton from one complete-thorax CT case, end to end:

  python scripts/atlas.py <case> [--skip-segment] [--skip-mesh] [--samples 256]

For the case (pinned in sources.json) it fetches and segments the CT, meshes
the trunk skeleton under the cut and symmetry rules (mesh.py --set
skeleton), retopologises every bone into quads (blender/retopo.py), runs the
close-up QA (blender/qa.py and its contact sheets), renders the studio views
(blender/studio.py) and lays each render beside Gray's plate 112 at the same
height, for the difference check. Everything goes to data/atlas/<case>/.

Stage 1 is the thorax only: ribs, costal cartilages and the costal arch,
sternum, and the spine as far as the scan has it. The shoulder girdle
(clavicles, scapulae, humeri) is not part of it: both Stage 1 scans end
through the shoulders, and a structure the scan cuts is never exported.
atlas.json says so and names the girdle bones the build left out; the girdle
is Stage 2's, issue 45.
Each step is a script on its own; this runs them in order and stops at the
first that fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = json.loads((ROOT / "sources.json").read_text())
PLATE = ROOT / "data" / "reference" / "gray112.png"
SCOPE = (
    "Stage 1, thorax only: ribs, costal cartilages and arch, sternum, and the spine as far "
    "as the scan has it. The shoulder girdle is not part of Stage 1 (issue 45)."
)
GIRDLE = ("clavicula", "scapula", "humerus")
HDRI = ROOT / "data" / "studio" / "studio_small_09_2k.hdr"


def step(*command: str | Path) -> None:
    print("+", " ".join(str(c) for c in command), flush=True)
    if subprocess.run([str(c) for c in command], cwd=ROOT).returncode != 0:
        sys.exit(f"stopped: {Path(str(command[1])).name} failed")


def side_by_side(render: Path, plate: Path, out: Path, height: int = 1400) -> None:
    """Our render on the left, the plate on the right, both at one height."""
    from PIL import Image

    a = Image.open(render).convert("RGB")
    b = Image.open(plate).convert("RGB")
    a = a.resize((round(a.width * height / a.height), height), Image.LANCZOS)
    b = b.resize((round(b.width * height / b.height), height), Image.LANCZOS)
    sheet = Image.new("RGB", (a.width + b.width + 24, height), (255, 255, 255))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (a.width + 24, 0))
    sheet.save(out)


def girdle_left_out(manifest: dict) -> list[dict]:
    """The manifest's entries for the girdle bones the build left out."""
    return [d for d in manifest.get("dropped", []) if d["label"].startswith(GIRDLE)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case")
    parser.add_argument("--skip-segment", action="store_true", help="masks are already there")
    parser.add_argument(
        "--skip-mesh", action="store_true", help="the skeleton GLB is already there"
    )
    parser.add_argument("--samples", type=int, default=256, help="Cycles samples per pixel")
    args = parser.parse_args()
    case = args.case
    record = SOURCES["cases"].get(case)
    if record is None:
        sys.exit(f"{case} is not pinned in sources.json")
    python = sys.executable
    blender = [python, ROOT / "scripts" / "blender.py"]
    work = ROOT / "data" / "work" / case
    out = ROOT / "data" / "atlas" / case
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_segment:
        step(python, ROOT / "scripts" / "fetch.py", "--source", record["source"], "--case", case)
        step(python, ROOT / "scripts" / "segment.py", "--case", case, "--set", "skeleton")
    if not args.skip_mesh:
        step(python, ROOT / "scripts" / "mesh.py", "--case", case, "--set", "skeleton")
    meshed = work / f"{case}.skeleton.glb"
    manifest = work / f"{case}.skeleton.manifest.json"
    retopo = out / f"{case}.skeleton.glb"
    step(*blender, "retopo", "--in", meshed, "--out", retopo, "--report", out / "retopo.json")
    qa_dir = out / "qa"
    step(*blender, "qa", "--in", retopo, "--manifest", manifest, "--out-dir", qa_dir)
    step(python, ROOT / "scripts" / "qa_sheet.py", "--qa-dir", qa_dir, "--rows", "10",
         "--title", f"{case} atlas skeleton")  # fmt: skip
    if not HDRI.exists():
        sys.exit(f"{HDRI.relative_to(ROOT)} is missing; see sources.json reference_data")
    step(*blender, "studio", "--in", retopo, "--hdri", HDRI, "--out-dir", out, "--name", case,
         "--samples", str(args.samples))  # fmt: skip
    if PLATE.exists():
        side_by_side(out / f"{case}.front.png", PLATE, out / f"{case}.front.vs-gray112.png")
    report = {
        "case": case,
        "record": record["record"],
        "scope": SCOPE,
        "girdleLeftOut": girdle_left_out(json.loads(manifest.read_text())),
        "manifest": str(manifest.relative_to(ROOT)),
        "skeleton": str(retopo.relative_to(ROOT)),
        "renders": [str(p.relative_to(ROOT)) for p in sorted(out.glob(f"{case}.*.png"))],
    }
    (out / "atlas.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"atlas: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
