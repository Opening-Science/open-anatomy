"""
Turns the masks from segment.py into one glTF binary (GLB) with a mesh per
structure, named by its UBERON ID, plus a manifest.

  python scripts/mesh.py --case s1086 [--seg-dir <dir>] [--out-dir <dir>]
      [--max-triangles 40000]

For each structure: crop the mask, extract a closed surface with discrete
flying edges and smooth it (windowed sinc, VTK), keep the largest connected
piece, decimate it to at most --max-triangles (organs) or --max-bone-triangles
(bones) while it stays closed, and orient the normals outward. The share of
the volume in the other pieces goes into the manifest, and above
MAX_DROPPED_SHARE the run stops for an organ: an organ in two pieces is a
segmentation problem, not noise. Coordinates go from the CT's RAS millimetres
to glTF metres (+Y up, +Z anterior, +X the body's left), and the whole set is
moved so its bounding box is centred on the origin.

--set organs (the default) needs every organ; --set reference adds the trunk
skeleton and lists, rather than meshes, a bone the scan misses or cuts (the
femurs and humeri may be cut). An organ that is empty, cut by the scan or
split always stops the run; a bone in pieces only warns. With body_trunc.nii.gz in the masks it also
writes <case>.outline.glb, the trunk outline in the same frame, which
assemble.py measures and never releases.

Writes data/work/<case>/<case>.<set>.glb and <case>.<set>.manifest.json with each node's
UBERON ID, system, licence, triangle count, volume and dropped share, and the
GLB's SHA-256. The GLB's asset.copyright and asset.extras credit the source
dataset's creator, DOI and case, so a GLB copied on its own keeps its
attribution.
It stops if any surface is open, non-manifold or inside out. VTK runs on
one thread so the same masks give the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("VTK_SMP_BACKEND_IN_USE", "Sequential")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import vtk  # noqa: E402
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from structures import SETS, Structure, node_name  # noqa: E402

# RAS millimetres to glTF metres: X = -R (the body's left), Y = S, Z = A.
# Limb bones run out of a trunk scan by design; their joint ends are what count.
MIN_ML = 0.3  # a mask smaller than this is a stray label, not a bone (volume, not voxels)
CUT_ALLOWED = {"femur_left", "femur_right", "humerus_left", "humerus_right"}
OUTLINE = Structure("body_trunc", "outline", "trunk outline", "outline", (0.8, 0.8, 0.8))
RAS_TO_GLTF = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64) * 0.001
SOURCE_LICENCE = "CC-BY-4.0"
SOURCES = json.loads((ROOT / "sources.json").read_text())
# Pieces other than the largest may hold at most this share of a structure's
# volume: stray voxels are dropped, a split organ stops the run.
MAX_DROPPED_SHARE = 0.02


def triangles_of(poly: vtk.vtkPolyData) -> np.ndarray:
    return vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)


def decimate(poly: vtk.vtkPolyData, max_triangles: int) -> vtk.vtkPolyData:
    """Reduces to about max_triangles, keeping the surface closed and manifold.

    Either decimator can open a thin or holed bone (a rib, the sacrum), so each
    result is checked: quadric first, then DecimatePro, each at the target and
    then at 1.5 and 2.25 times it. If none stays closed the surface is kept whole.
    """
    for target in (max_triangles, max_triangles * 1.5, max_triangles * 2.25):
        reduction = 1 - target / poly.GetNumberOfCells()
        if reduction <= 0:
            break
        quadric = vtk.vtkQuadricDecimation()
        quadric.SetTargetReduction(reduction)
        quadric.VolumePreservationOn()
        pro = vtk.vtkDecimatePro()
        pro.SetTargetReduction(reduction)
        pro.PreserveTopologyOn()
        pro.SplittingOff()
        pro.BoundaryVertexDeletionOff()
        for decimator in (quadric, pro):
            decimator.SetInputData(poly)
            decimator.Update()
            out = decimator.GetOutput()
            if not open_edges(triangles_of(out)):
                return out
    return poly


def cut_reason(path: Path) -> str | None:
    """Why a mask cannot be a whole structure, or None if it is whole.

    Smaller than MIN_ML (measured with the voxel size from the affine, so the
    limit is the same at 0.8 mm and 1.5 mm), or touching the scan's first or
    last axial slice.
    """
    image = nib.load(path)
    data = np.asanyarray(image.dataobj) > 0
    ml = data.sum() * abs(np.linalg.det(image.affine[:3, :3])) / 1000
    if ml < MIN_ML:
        return f"absent or a stray label ({ml:.2f} ml)"
    k = np.flatnonzero(data.any((0, 1)))
    if k[0] == 0 or k[-1] == data.shape[2] - 1:
        return "cut by the scan's first or last slice"
    return None


def enclosed_volume(poly: vtk.vtkPolyData) -> float:
    """Volume enclosed by a closed surface, in its own units cubed."""
    return abs(signed_volume(vtk_to_numpy(poly.GetPoints().GetData()), triangles_of(poly)))


def surface(path: Path, max_triangles: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (vertices in RAS mm, triangles, dropped share) for the mask at path.

    The dropped share is the part of the smoothed surface's volume that lies
    in pieces other than the largest one, which is the only one kept.
    """
    image = nib.load(path)
    data = np.asanyarray(image.dataobj) > 0
    if not data.any():
        raise ValueError(f"{path.name} is empty")
    lo = np.maximum(np.argwhere(data).min(0) - 2, 0)
    hi = np.argwhere(data).max(0) + 3
    crop = np.pad(data[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], 1).astype(np.uint8)

    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(*crop.shape)
    vtk_image.SetSpacing(*spacing)
    scalars = numpy_to_vtk(crop.ravel(order="F"), deep=True)
    vtk_image.GetPointData().SetScalars(scalars)

    # Surface nets leaves open edges where voxels touch only diagonally;
    # discrete flying edges gives a closed surface, then windowed sinc smooths it.
    edges = vtk.vtkDiscreteFlyingEdges3D()
    edges.SetInputData(vtk_image)
    edges.SetValue(0, 1)
    edges.ComputeNormalsOff()
    edges.ComputeGradientsOff()
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(edges.GetOutputPort())
    smooth.SetNumberOfIterations(30)
    smooth.SetPassBand(0.01)
    smooth.NormalizeCoordinatesOn()
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()

    largest = vtk.vtkPolyDataConnectivityFilter()
    largest.SetInputConnection(smooth.GetOutputPort())
    largest.SetExtractionModeToLargestRegion()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(largest.GetOutputPort())
    clean.Update()
    poly = clean.GetOutput()
    smooth.Update()
    whole = enclosed_volume(smooth.GetOutput())
    dropped = 1 - enclosed_volume(poly) / whole if whole else 0.0

    if poly.GetNumberOfCells() > max_triangles:
        poly = decimate(poly, max_triangles)

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    triangles = triangles_of(poly)
    # The padded crop's first voxel sits at the origin.
    ijk = points / spacing + lo - 1
    ras = ijk @ image.affine[:3, :3].T + image.affine[:3, 3]
    return ras, triangles.astype(np.uint32), max(dropped, 0.0)


def outward(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orients triangles outward and returns (triangles, vertex normals)."""
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices.astype(np.float32), deep=True))
    poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, 3 * len(triangles) + 1, 3, dtype=np.int64)
    cells.SetData(
        numpy_to_vtk(offsets, deep=True),
        numpy_to_vtk(triangles.ravel().astype(np.int64), deep=True),
    )
    poly.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.Update()
    out = normals.GetOutput()
    tris = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.uint32)
    return tris, vtk_to_numpy(out.GetPointData().GetNormals()).astype(np.float32)


def open_edges(triangles: np.ndarray) -> int:
    """Edges used by other than exactly two triangles: holes or non-manifold seams."""
    edges = np.sort(
        np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), 1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts != 2).sum())


def signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def credit(case: str) -> dict:
    """The attribution a GLB built from case carries: creator, DOI, licence, case."""
    if case not in SOURCES["cases"]:
        sys.exit(f"{case} is not in sources.json, so its source cannot be credited")
    source = SOURCES["sources"][SOURCES["cases"][case]["source"]]
    return {
        "copyright": (
            f"Derived from case {case} of the {source['title']} by {source['creator']}, "
            f"doi:{source['doi']}, licensed {source['licence']}"
        ),
        "extras": {
            "source": source["title"],
            "creator": source["creator"],
            "doi": source["doi"],
            "licence": source["licence"],
            "case": case,
        },
    }


def write_glb(path: Path, parts: list[dict], attribution: dict | None = None) -> None:
    """Writes one node and mesh per part, each with its own material.

    attribution (from credit()) goes into asset.copyright and asset.extras.
    """
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "open-anatomy mesh.py", **(attribution or {})},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(parts)))}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [],
    }
    blob = bytearray()

    def add(array: np.ndarray, target: int, component: int, kind: str) -> int:
        while len(blob) % 4:
            blob.append(0)
        gltf["bufferViews"].append(
            {"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes, "target": target}
        )
        blob.extend(array.tobytes())
        accessor = {
            "bufferView": len(gltf["bufferViews"]) - 1,
            "componentType": component,
            "count": len(array) if kind != "SCALAR" else array.size,
            "type": kind,
        }
        if kind == "VEC3" and component == 5126:
            accessor["min"] = array.min(0).tolist()
            accessor["max"] = array.max(0).tolist()
        gltf["accessors"].append(accessor)
        return len(gltf["accessors"]) - 1

    for i, part in enumerate(parts):
        s: Structure = part["structure"]
        position = add(part["vertices"], 34962, 5126, "VEC3")
        normal = add(part["normals"], 34962, 5126, "VEC3")
        indices = add(part["triangles"].ravel(), 34963, 5125, "SCALAR")
        gltf["materials"].append(
            {
                "name": s.name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*s.colour, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                },
            }
        )
        gltf["meshes"].append(
            {
                "name": node_name(s.term),
                "primitives": [
                    {
                        "attributes": {"POSITION": position, "NORMAL": normal},
                        "indices": indices,
                        "material": i,
                    }
                ],
            }
        )
        gltf["nodes"].append(
            {
                "name": node_name(s.term),
                "mesh": i,
                "extras": {
                    "term": s.term,
                    "name": s.name,
                    "system": s.system,
                    "licence": SOURCE_LICENCE,
                },
            }
        )

    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"].append({"byteLength": len(blob)})
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    with path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--seg-dir", type=Path, help="default: data/work/<case>/seg")
    parser.add_argument("--out-dir", type=Path, help="default: data/work/<case>")
    parser.add_argument("--set", choices=SETS, default="organs", help="structures to mesh")
    parser.add_argument("--max-triangles", type=int, default=40000, help="per organ")
    parser.add_argument("--max-bone-triangles", type=int, default=6000, help="per bone")
    parser.add_argument(
        "--synthetic", action="store_true", help="test masks with no source to credit"
    )
    args = parser.parse_args()
    attribution = (
        {"copyright": "Synthetic test data", "extras": {"case": args.case}}
        if args.synthetic
        else credit(args.case)
    )

    work = args.out_dir or ROOT / "data" / "work" / args.case
    work.mkdir(parents=True, exist_ok=True)
    seg_dir = args.seg_dir or work / "seg"
    parts, missing, reasons = [], [], {}
    for s in SETS[args.set]:
        bone = s.system == "skeletal"
        cap = args.max_bone_triangles if bone else args.max_triangles
        path = seg_dir / f"{s.label}.nii.gz"
        checked = args.set == "reference" and s.label not in CUT_ALLOWED
        reason = cut_reason(path) if checked else None
        if reason:
            # A bone the scan covers only partly (an upper cervical vertebra) would
            # be a fragment; it is listed instead. Every organ must be whole.
            if not bone:
                sys.exit(f"{s.label}: {reason} in {args.case}; organs must be whole")
            missing.append(s.label)
            reasons[s.label] = reason
            print(f"{s.label}: {reason}; listed as missing", file=sys.stderr)
            continue
        try:
            ras, triangles, dropped = surface(path, cap)
        except ValueError:
            if not bone:
                sys.exit(f"{s.label}: the mask is empty in {args.case}; organs must be whole")
            missing.append(s.label)
            reasons[s.label] = "empty mask"
            continue
        if dropped > MAX_DROPPED_SHARE:
            message = (
                f"{s.label}: {dropped:.1%} of its volume is outside its largest piece "
                f"(at most {MAX_DROPPED_SHARE:.0%} may be dropped from an organ)"
            )
            if not bone:
                sys.exit(message)
            # A segmented bone often comes in pieces (a vertebra's processes, a
            # humerus leaving and re-entering the scan); its share is recorded
            # in the manifest and said here, not silently dropped.
            print(f"warning: {message}", file=sys.stderr)
        parts.append({
            "structure": s, "vertices": ras @ RAS_TO_GLTF.T, "triangles": triangles,
            "dropped": dropped,
        })  # fmt: skip

    everything = np.vstack([p["vertices"] for p in parts])
    centre = (everything.min(0) + everything.max(0)) / 2
    for p in parts:
        p["vertices"] = (p["vertices"] - centre).astype(np.float32)
        p["triangles"], p["normals"] = outward(p["vertices"], p["triangles"])
        name = p["structure"].label
        if open_edges(p["triangles"]):
            sys.exit(f"{name}: the surface is not closed and manifold")
        if signed_volume(p["vertices"], p["triangles"]) < 0:
            # VTK's auto-orientation can miss on a ragged fragment; a closed
            # surface with negative volume is simply wound the other way round.
            p["triangles"], p["normals"] = p["triangles"][:, ::-1].copy(), -p["normals"]
        if signed_volume(p["vertices"], p["triangles"]) <= 0:
            sys.exit(f"{name}: the normals point inward")

    glb = work / f"{args.case}.{args.set}.glb"
    write_glb(glb, parts, attribution)
    outline_path = seg_dir / "body_trunc.nii.gz"
    if args.set == "reference" and outline_path.exists():
        # The trunk outline, in the same frame, for assemble.py to measure. It is
        # the outer surface of a person's trunk, so it is never released.
        ras, triangles, _ = surface(outline_path, args.max_triangles)
        vertices = (ras @ RAS_TO_GLTF.T - centre).astype(np.float32)
        triangles, normals = outward(vertices, triangles)
        outline = {"structure": OUTLINE, "vertices": vertices, "triangles": triangles}
        write_glb(work / f"{args.case}.outline.glb", [{**outline, "normals": normals}], attribution)
    manifest = {
        "case": args.case,
        "set": args.set,
        "file": glb.name,
        "sha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
        "attribution": attribution["copyright"],
        "units": "metres",
        "axes": "+Y superior, +Z anterior, +X the body's left",
        "centre_offset_m": [round(float(c), 6) for c in centre],
        "seg_dir": str(seg_dir.relative_to(ROOT)) if seg_dir.is_relative_to(ROOT) else str(seg_dir),
        "nodes": [
            {
                "name": node_name(p["structure"].term),
                "term": p["structure"].term,
                "label": p["structure"].name,
                "system": p["structure"].system,
                "licence": SOURCE_LICENCE,
                "triangles": int(len(p["triangles"])),
                "volume_ml": round(signed_volume(p["vertices"], p["triangles"]) * 1e6, 1),
                "dropped_share": round(p["dropped"], 4),
            }
            for p in parts
        ],
    }
    manifest["missing"] = missing
    manifest["missing_reasons"] = reasons
    (work / f"{args.case}.{args.set}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for node in manifest["nodes"]:
        print(f"{node['name']}  {node['label']:20} {node['triangles']:>6} tris  "
              f"{node['volume_ml']:>8} ml")  # fmt: skip
    print(f"{glb}  sha256 {manifest['sha256']}")


if __name__ == "__main__":
    main()
