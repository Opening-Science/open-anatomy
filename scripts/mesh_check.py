"""
Self-check for mesh.py on synthetic masks: no CT, no download, no GPU.

  python scripts/mesh_check.py

Writes one ellipsoid mask per v0 structure to a temporary folder, with
anisotropic voxels and a left-right flipped affine as real CTs have, runs
mesh.py on them twice, and checks that:
- every structure became a closed, outward-facing mesh named by its UBERON ID;
- each mesh's volume is within 3% of the ellipsoid's exact volume;
- the two runs gave the same GLB bytes.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import STRUCTURES, node_name  # noqa: E402

SPACING = np.array([0.8, 0.8, 1.5])  # mm
SHAPE = (96, 96, 64)
AFFINE = np.diag([-SPACING[0], SPACING[1], SPACING[2], 1.0])  # x flipped, det < 0


def ellipsoid(radii: np.ndarray, centre: np.ndarray) -> np.ndarray:
    grid = np.indices(SHAPE).transpose(1, 2, 3, 0) * SPACING
    return (((grid - centre) / radii) ** 2).sum(-1) <= 1


def run(seg: Path, out: Path) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "mesh.py"), "--case", "synthetic",
         "--seg-dir", str(seg), "--out-dir", str(out)],
        check=True, capture_output=True,
    )  # fmt: skip
    return json.loads((out / "manifest.json").read_text())


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        seg = Path(tmp) / "seg"
        seg.mkdir()
        expected = {}
        for i, s in enumerate(STRUCTURES):
            radii = np.array([10.0 + i, 8.0 + i / 2, 12.0 + i])
            centre = np.array(SHAPE) * SPACING / 2 + (i - 4.5) * 2
            mask = ellipsoid(radii, centre).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, AFFINE), seg / f"{s.label}.nii.gz")
            expected[node_name(s.uberon)] = 4 / 3 * math.pi * radii.prod() / 1000  # ml

        first = run(seg, Path(tmp) / "a")
        second = run(seg, Path(tmp) / "b")
        names = [n["name"] for n in first["nodes"]]
        if names != [node_name(s.uberon) for s in STRUCTURES]:
            failures.append(f"node names {names}")
        for node in first["nodes"]:
            error = abs(node["volume_ml"] - expected[node["name"]]) / expected[node["name"]]
            print(f"{node['name']}  {node['volume_ml']:7.1f} ml  ({error:.1%} off)")
            if node["volume_ml"] <= 0 or error > 0.03:
                failures.append(f"{node['name']} volume {node['volume_ml']} ml")
        if first["sha256"] != second["sha256"]:
            failures.append("two runs gave different GLB bytes")

    for f in failures:
        print("FAIL", f)
    print("mesh check passed" if not failures else f"mesh check failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
