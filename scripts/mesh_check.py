"""
Self-check for mesh.py on synthetic masks: no CT, no download, no GPU.

  python scripts/mesh_check.py

Writes one ellipsoid mask per v0 structure to a temporary folder, with
anisotropic voxels and a left-right flipped affine as real CTs have, runs
mesh.py on them twice, and checks that:
- every structure became a closed, outward-facing mesh named by its UBERON ID;
- each mesh's volume, measured on the GLB itself, is within 1.5% of the
  ellipsoid's exact volume;
- the spleen's ellipsoid, placed off-centre towards the body's left, lands on
  +X in the GLB, at the expected distance from the liver's;
- a small stray blob next to the stomach is dropped and its share recorded,
  and a gallbladder split into two halves stops the run;
- the GLB carries an asset.copyright;
- the two runs gave the same GLB bytes.
"""

from __future__ import annotations

import json
import math
import struct
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
VOLUME_TOLERANCE = 0.015  # the smallest ellipsoid (4 ml) comes out about 1.1% small
LEFT_SHIFT_MM = 20.0  # the spleen moves this far along +i, which is the body's left


def ellipsoid(radii: np.ndarray, centre: np.ndarray) -> np.ndarray:
    grid = np.indices(SHAPE).transpose(1, 2, 3, 0) * SPACING
    return (((grid - centre) / radii) ** 2).sum(-1) <= 1


def run(seg: Path, out: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "mesh.py"), "--case", "synthetic",
         "--synthetic", "--seg-dir", str(seg), "--out-dir", str(out)],
        check=check, capture_output=True, text=True,
    )  # fmt: skip


def read_glb(path: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, float]]:
    """Returns (glTF JSON, bounding-box centre per node in m, enclosed volume per node in ml)."""
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + length])
    blob = data[20 + length + 8 :]

    def array(index: int, dtype: type) -> np.ndarray:
        view = gltf["bufferViews"][gltf["accessors"][index]["bufferView"]]
        raw = blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        return np.frombuffer(raw, dtype).reshape(-1, 3)

    centres, volumes = {}, {}
    for node in gltf["nodes"]:
        primitive = gltf["meshes"][node["mesh"]]["primitives"][0]
        vertices = array(primitive["attributes"]["POSITION"], np.float32).astype(np.float64)
        triangles = array(primitive["indices"], np.uint32)
        a, b, c = (vertices[triangles[:, i]] for i in range(3))
        volumes[node["name"]] = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6) * 1e6
        centres[node["name"]] = (vertices.min(0) + vertices.max(0)) / 2
    return gltf, centres, volumes


def main() -> int:
    failures = []
    by_label = {s.label: node_name(s.uberon) for s in STRUCTURES}
    with tempfile.TemporaryDirectory() as tmp:
        seg = Path(tmp) / "seg"
        seg.mkdir()
        expected, centres_mm = {}, {}
        for i, s in enumerate(STRUCTURES):
            radii = np.array([10.0 + i, 8.0 + i / 2, 12.0 + i])
            centre = np.array(SHAPE) * SPACING / 2 + (i - 4.5) * 2
            if s.label == "spleen":
                centre[0] += LEFT_SHIFT_MM
            mask = ellipsoid(radii, centre)
            if s.label == "stomach":
                mask[4:7, 4:7, 4:7] = True  # a stray blob far from the stomach
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), AFFINE), seg / f"{s.label}.nii.gz")
            expected[node_name(s.uberon)] = 4 / 3 * math.pi * radii.prod() / 1000  # ml
            centres_mm[s.label] = centre

        run(seg, Path(tmp) / "a")
        run(seg, Path(tmp) / "b")
        first = json.loads((Path(tmp) / "a" / "manifest.json").read_text())
        second = json.loads((Path(tmp) / "b" / "manifest.json").read_text())
        gltf, centres, volumes = read_glb(Path(tmp) / "a" / first["file"])
        names = [n["name"] for n in first["nodes"]]
        if names != [node_name(s.uberon) for s in STRUCTURES]:
            failures.append(f"node names {names}")
        for name, volume in volumes.items():
            error = abs(volume - expected[name]) / expected[name]
            print(f"{name}  {volume:7.2f} ml  ({error:.2%} off)")
            if volume <= 0 or error > VOLUME_TOLERANCE:
                failures.append(f"{name} volume {volume:.2f} ml, {error:.2%} off")

        # Voxel i runs towards the body's left (the affine's x is flipped), and
        # glTF +X is the body's left, so the spleen must sit on +X of the liver.
        shift = (centres_mm["spleen"][0] - centres_mm["liver"][0]) / 1000
        found = centres[by_label["spleen"]][0] - centres[by_label["liver"]][0]
        print(f"spleen is {found * 1000:+.1f} mm along X from the liver "
              f"(expected {shift * 1000:+.1f})")  # fmt: skip
        if abs(found - shift) > 0.001:
            failures.append(f"spleen at {found * 1000:+.1f} mm along X, not {shift * 1000:+.1f}")

        stomach = next(n for n in first["nodes"] if n["name"] == by_label["stomach"])
        print(f"stomach's stray blob: {stomach['dropped_share']:.2%} dropped")
        if not 0 < stomach["dropped_share"] <= 0.02:
            failures.append(f"stomach's stray blob dropped share {stomach['dropped_share']}")
        if not gltf["asset"].get("copyright"):
            failures.append("the GLB has no asset.copyright")
        if first["sha256"] != second["sha256"]:
            failures.append("two runs gave different GLB bytes")

        # A gallbladder in two halves would lose half its volume: the run must stop.
        split = ellipsoid(np.array([12.0, 10.0, 20.0]), np.array(SHAPE) * SPACING / 2)
        split[:, :, SHAPE[2] // 2 - 1 : SHAPE[2] // 2 + 1] = False
        nib.save(nib.Nifti1Image(split.astype(np.uint8), AFFINE), seg / "gallbladder.nii.gz")
        result = run(seg, Path(tmp) / "c", check=False)
        if result.returncode == 0 or "outside its largest piece" not in result.stderr:
            failures.append(f"a split gallbladder was not refused: {result.stderr.strip()!r}")
        else:
            print(f"refused as expected: {result.stderr.strip()}")

    for f in failures:
        print("FAIL", f)
    print("mesh check passed" if not failures else f"mesh check failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
