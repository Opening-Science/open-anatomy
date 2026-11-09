"""
Prototype of the imaging layer: axial, coronal and sagittal slices through a
reference body's own CT, as textured planes placed inside that body.

  python scripts/imaging.py [anny-adult-m|anny-adult-f] [--window soft|bone]

Reads the body's CT case (data/cases/<case>/ct.nii.gz), its masks
(data/work/<case>/seg/), the reference set's manifest (for the frame mesh.py
used) and the body's report from blender/assemble.py (for the trunk fit). It
writes, in data/imaging/<body>/:
- axial.png, coronal.png, sagittal.png: the CT in a soft-tissue (W400/L40) or
  bone (W1800/L400) window, the segmented structures tinted in their display
  colours, and the air outside the body transparent;
- <body>.imaging.glb: three double-sided, unlit quads carrying those textures,
  each where its slice lies in the reference body (the trunk fit, not the
  girdle step), so the GLB can be loaded together with <body>.reference.glb;
- imaging.json: each slice's plane, voxel index, position and source.

The slices are cut at kidney level (axial), through the kidneys (coronal) and
on the spine's midline (sagittal), and cropped to the trunk, from 5 cm below
the femurs' top to the scan top. A head guard (face_free_top) refuses a case
whose skull and brain were not segmented, and stops every slice at the top of
C7 when the scan reaches the head, so no face is ever included. This is CT, not
MRI: "MRI layer" is the product name for an imaging-style view. Pure numpy;
the same inputs give the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mesh import RAS_TO_GLTF  # noqa: E402
from structures import HEAD_GUARD, REFERENCE  # noqa: E402

WINDOWS = {"soft": (400, 40), "bone": (1800, 400)}
TINT = 0.38  # share of a structure's colour mixed into its voxels
SOURCE_LICENCE = "CC-BY-4.0"
GUARD_MARGIN_SLICES = 14  # about 2 cm at 1.5 mm


NECK = [f"vertebrae_C{i}" for i in range(1, 7)]


def face_free_top(masks: dict[str, np.ndarray | None]) -> int | None:
    """The highest axial index (canonical, superior = higher) a slice may show.

    masks maps HEAD_GUARD labels and C1-C6 to their masks, plus vertebrae_C7.
    With none of the skull, brain or C1-C6 in the scan, the scan starts below
    the head and nothing is cropped (None). Otherwise the cut is the lowest of:
    2 cm below the lowest skull or brain voxel (a bowed head brings the jaw down
    beside the neck, level with C7), the top of C7, and, without C7, 2 cm below
    the lowest of C1-C6. A guard label that was never segmented (None) is an
    error: absence must be shown, not assumed.
    """
    for label in HEAD_GUARD:
        if masks.get(label) is None:
            raise ValueError(f"{label} was not segmented; run segment.py --set reference")

    def present(labels):
        return [m for k, m in masks.items() if k in labels and m is not None and m.any()]

    def lowest(ms):
        return min(int(np.flatnonzero(m.any((0, 1)))[0]) for m in ms)

    head, neck = present(HEAD_GUARD), present(NECK)
    if not head and not neck:
        return None
    cuts = []
    if head:
        cuts.append(lowest(head) - GUARD_MARGIN_SLICES)
    c7 = masks.get("vertebrae_C7")
    if c7 is not None and c7.any():
        cuts.append(int(np.flatnonzero(c7.any((0, 1)))[-1]))
    elif neck:
        cuts.append(lowest(neck) - GUARD_MARGIN_SLICES)
    return min(cuts)


def srgb(linear) -> np.ndarray:
    """Linear RGB (structures.py's colours) to sRGB, the space of a PNG and of a
    glTF baseColorTexture, so a tint matches the same structure's mesh."""
    c = np.clip(np.asarray(linear, dtype=np.float64), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def canonical(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(path))  # voxel axes along R, A, S


def png(rgba: np.ndarray) -> bytes:
    """A minimal RGBA PNG writer: no dependencies, same bytes every time."""
    height, width, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def to_reference(ras: np.ndarray, centre: np.ndarray, scale: np.ndarray, shift: np.ndarray):
    """RAS millimetres to the reference body's glTF frame, as mesh.py and assemble.py do."""
    gltf = ras @ RAS_TO_GLTF.T - centre  # mesh.py
    blender = np.c_[gltf[:, 0], -gltf[:, 2], gltf[:, 1]]  # Blender's glTF import
    fitted = blender * scale + shift  # assemble.py's trunk fit
    return np.c_[fitted[:, 0], fitted[:, 2], -fitted[:, 1]]  # back to glTF (+Y up)


def slice_image(ct, labels, body, axis: int, index: int, window) -> tuple[np.ndarray, list]:
    """RGBA image of one slice: rows superior first (axial: anterior first)."""
    take = [slice(None)] * 3
    take[axis] = index
    hu, lab, inside = (np.asarray(v[tuple(take)]) for v in (ct, labels, body))
    width, level = window
    grey = np.clip((hu - (level - width / 2)) / width, 0, 1)
    rgb = np.repeat(grey[..., None], 3, -1)
    for i, s in enumerate(REFERENCE, start=1):
        hit = lab == i
        rgb[hit] = (1 - TINT) * rgb[hit] + TINT * srgb(s.colour)
    alpha = inside.astype(np.float64)
    image = np.dstack([rgb, alpha])
    # The 2D array's axes are the two remaining voxel axes, in order. Rows run
    # along the second from its high end, columns along the first from its high
    # end; quad_corners places the texture's corners the same way.
    image = image.transpose(1, 0, 2)[::-1, ::-1]
    return (np.round(image * 255)).astype(np.uint8), [i for i in np.unique(lab) if i]


def quad_corners(affine, axis: int, index: int, lo, hi) -> np.ndarray:
    """RAS corners of a slice's crop, in the image's order: top-left, top-right,
    bottom-left, bottom-right (matching slice_image's flips)."""
    other = [a for a in range(3) if a != axis]
    first, second = other  # image columns run along `first`, rows along `second`
    corners = []
    for row in (hi[second] + 0.5, lo[second] - 0.5):  # top row is the high end
        for col in (hi[first] + 0.5, lo[first] - 0.5):  # left column is the high end
            ijk = np.zeros(3)
            ijk[axis], ijk[first], ijk[second] = index, col, row
            corners.append(ijk)
    corners = np.array(corners)
    return corners @ affine[:3, :3].T + affine[:3, 3]


def write_glb(path: Path, planes: list[dict]) -> None:
    gltf = {
        "asset": {"version": "2.0", "generator": "open-anatomy imaging.py"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"name": "imaging", "nodes": list(range(len(planes)))}],
        "nodes": [], "meshes": [], "materials": [], "textures": [], "images": [],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "accessors": [], "bufferViews": [], "buffers": [],
    }  # fmt: skip
    blob = bytearray()

    def view(data: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        entry = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            entry["target"] = target
        gltf["bufferViews"].append(entry)
        blob.extend(data)
        return len(gltf["bufferViews"]) - 1

    def accessor(array: np.ndarray, component: int, kind: str, target: int) -> int:
        item = {
            "bufferView": view(array.tobytes(), target),
            "componentType": component,
            "count": len(array),
            "type": kind,
        }
        if kind == "VEC3":
            item["min"], item["max"] = array.min(0).tolist(), array.max(0).tolist()
        gltf["accessors"].append(item)
        return len(gltf["accessors"]) - 1

    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    indices = np.array([0, 2, 1, 1, 2, 3], dtype=np.uint16)
    for i, plane in enumerate(planes):
        gltf["images"].append({"bufferView": view(plane["png"]), "mimeType": "image/png"})
        gltf["textures"].append({"sampler": 0, "source": i})
        gltf["materials"].append({
            "name": f"imaging-{plane['plane']}",
            "pbrMetallicRoughness": {"baseColorTexture": {"index": i}, "metallicFactor": 0.0},
            "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True,
            "extensions": {"KHR_materials_unlit": {}},
        })  # fmt: skip
        gltf["meshes"].append({
            "name": f"imaging-{plane['plane']}",
            "primitives": [{
                "attributes": {
                    "POSITION": accessor(plane["corners"].astype(np.float32), 5126, "VEC3", 34962),
                    "TEXCOORD_0": accessor(uv, 5126, "VEC2", 34962),
                },
                "indices": accessor(indices, 5123, "SCALAR", 34963),
                "material": i,
            }],
        })  # fmt: skip
        gltf["nodes"].append(
            {
                "name": f"imaging-{plane['plane']}",
                "mesh": i,
                "extras": {"layer": "imaging", "plane": plane["plane"], "licence": SOURCE_LICENCE},
            }
        )
    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"].append({"byteLength": len(blob)})
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    with path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(blob)))
        f.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("body", nargs="?", default="anny-adult-m")
    parser.add_argument("--window", choices=WINDOWS, default="soft")
    args = parser.parse_args()
    sources = json.loads((ROOT / "sources.json").read_text())
    case = sources["references"][args.body]["case"]
    work = ROOT / "data" / "work" / case
    manifest = json.loads((work / f"{case}.reference.manifest.json").read_text())
    report = json.loads((ROOT / "data" / "reference" / f"{args.body}.report.json").read_text())
    centre = np.array(manifest["centre_offset_m"])
    scale, shift = np.array(report["scale"]), np.array(report["shiftM"])

    ct_image = canonical(ROOT / "data" / "cases" / case / "ct.nii.gz")
    affine = ct_image.affine
    ct = np.asanyarray(ct_image.dataobj).astype(np.float32)
    labels = np.zeros(ct.shape, dtype=np.uint8)
    for i, s in enumerate(REFERENCE, start=1):
        path = work / "seg" / f"{s.label}.nii.gz"
        if path.exists():
            labels[np.asanyarray(canonical(path).dataobj) > 0] = i
    body = np.asanyarray(canonical(work / "seg" / "body_trunc.nii.gz").dataobj) > 0

    def voxels(label: str) -> np.ndarray:
        found = np.argwhere(labels == [s.label for s in REFERENCE].index(label) + 1)
        if not len(found):
            sys.exit(f"{case}: {label} is empty or missing in {work / 'seg'}; "
                     "imaging.py needs it to place the slices")  # fmt: skip
        return found

    kidneys = (voxels("kidney_left").mean(0) + voxels("kidney_right").mean(0)) / 2
    spine = np.argwhere(np.isin(labels, [
        i for i, s in enumerate(REFERENCE, 1) if s.label.startswith("vertebrae_")
    ])).mean(0)  # fmt: skip
    # The femurs' top: the head or the greater trochanter, whichever is higher.
    # Both femurs are needed, so the crop's bottom does not depend on which one
    # the scan happened to keep.
    femur_top = max(int(voxels(f)[:, 2].max()) for f in ("femur_left", "femur_right"))
    guard = {}
    for label in (*HEAD_GUARD, *NECK, "vertebrae_C7"):
        path = work / "seg" / f"{label}.nii.gz"
        guard[label] = np.asanyarray(canonical(path).dataobj) > 0 if path.exists() else None
    try:
        top = face_free_top(guard)
    except ValueError as error:
        sys.exit(f"head guard: {error}")
    if top is not None:
        body[:, :, top + 1 :] = False  # nothing above the guard reaches an image
        print(f"head guard: the scan reaches the head; slices stop at axial index {top}")
    if not body.any():
        sys.exit("head guard: nothing of the trunk is left below the head")
    inside = np.argwhere(body)
    lo = inside.min(0)
    hi = inside.max(0)
    lo[2] = max(int(lo[2]), femur_top - int(round(50 / np.linalg.norm(affine[:3, 2]))))
    crop = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi, strict=True))
    ct, labels, body = ct[crop], labels[crop], body[crop]
    offset = lo
    origin = affine.copy()
    origin[:3, 3] = affine[:3, :3] @ offset + affine[:3, 3]  # the crop's first voxel

    picks = {
        "axial": (2, int(round(kidneys[2])) - int(offset[2])),
        "coronal": (1, int(round(kidneys[1])) - int(offset[1])),
        "sagittal": (0, int(round(spine[0])) - int(offset[0])),
    }
    out = ROOT / "data" / "imaging" / args.body
    out.mkdir(parents=True, exist_ok=True)
    planes, record = [], []
    zero, top = np.zeros(3, dtype=int), np.array(ct.shape) - 1
    for plane, (axis, index) in picks.items():
        image, shown = slice_image(ct, labels, body, axis, index, WINDOWS[args.window])
        data = png(image)
        (out / f"{plane}.png").write_bytes(data)
        ras = quad_corners(origin, axis, index, zero, top)
        corners = to_reference(ras, centre, scale, shift)
        planes.append({"plane": plane, "png": data, "corners": corners})
        record.append({
            "plane": plane,
            "voxelIndex": int(index + offset[axis]),
            "image": f"{plane}.png",
            "pixels": [int(image.shape[1]), int(image.shape[0])],
            "sha256": hashlib.sha256(data).hexdigest(),
            "structures": [REFERENCE[i - 1].term for i in shown],
            "cornersM": np.round(corners, 4).tolist(),
        })  # fmt: skip
    glb = out / f"{args.body}.imaging.glb"
    write_glb(glb, planes)
    (out / "imaging.json").write_text(json.dumps({
        "body": args.body, "case": case, "window": args.window,
        "source": "TotalSegmentator dataset v2.0.1 (CC BY 4.0), see NOTICE",
        "licence": SOURCE_LICENCE, "modality": "CT",
        "glb": glb.name, "glbSha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
        "slices": record,
    }, indent=2) + "\n")  # fmt: skip
    for r in record:
        print(f"{r['plane']:9} voxel {r['voxelIndex']:4}  {r['pixels'][0]}x{r['pixels'][1]}  "
              f"{len(r['structures'])} structures")  # fmt: skip
    print(f"{glb}")


if __name__ == "__main__":
    main()
