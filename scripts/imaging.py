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
  colours, and the air outside the body transparent, one pixel per voxel, for
  measurement;
- the same at twice the size (axial@2x.png and so on), Lanczos-3 upsampled
  with an anti-aliased edge, for display only: the GLB's textures;
- <body>.imaging.glb: three double-sided, unlit quads carrying those textures,
  each where its slice lies in the reference body (the trunk fit, not the
  girdle step), so the GLB can be loaded together with <body>.reference.glb;
- imaging.json: each slice's plane, voxel index, position and source.

The slices are cut at kidney level (axial), through the kidneys (coronal) and
on the spine's midline (sagittal), and cropped to the trunk, from 5 cm below
the femurs' top to the scan top. A head guard (face_free_top) refuses a case
whose skull and brain were not segmented, and when the scan reaches the head
stops every slice at the lowest of the C7 and skull/brain cuts (the top of C7,
2 cm below the lowest skull or brain voxel), so no face is ever included; a
case whose head comes within 2 cm of the scan's bottom is refused. This is CT,
not MRI: "MRI layer" is the product name for an imaging-style view. Pure numpy;
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
GUARD_MARGIN_MM = 20.0  # how far below the head or neck the guard cuts
DISPLAY_FACTOR = 2  # the GLB textures are upsampled by this for display
LANCZOS_TAPS = 3
EDGE_SOFTNESS = 0.7  # display pixels of Gaussian on the body mask's edge (display only)
EDGE_BLEED = 5  # display pixels of colour bled across the edge band and the soft fringe
WEIGHT_ONE = 4096  # Lanczos weights are fixed point, in steps of 1/4096


NECK = [f"vertebrae_C{i}" for i in range(1, 7)]


def guard_margin(spacing_mm: float) -> int:
    """GUARD_MARGIN_MM in axial slices, rounded up so it is never less than 2 cm."""
    return int(np.ceil(GUARD_MARGIN_MM / spacing_mm - 1e-9))


def face_free_top(masks: dict[str, np.ndarray | None], margin: int) -> int | None:
    """The highest axial index (canonical, superior = higher) a slice may show.

    masks maps HEAD_GUARD labels and C1-C6 to their masks, plus vertebrae_C7;
    margin is 2 cm in slices (guard_margin).
    With none of the skull, brain or C1-C6 in the scan, the scan starts below
    the head and nothing is cropped (None). Otherwise the cut is the lowest of:
    2 cm below the lowest skull or brain voxel (a bowed head brings the jaw down
    beside the neck, level with C7), the top of C7, and, without C7, 2 cm below
    the lowest of C1-C6. A guard label that was never segmented (None) is an
    error: absence must be shown, not assumed. The result can be negative when
    the head reaches the scan's bottom; apply_head_guard refuses that case.
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
        cuts.append(lowest(head) - margin)
    c7 = masks.get("vertebrae_C7")
    if c7 is not None and c7.any():
        cuts.append(int(np.flatnonzero(c7.any((0, 1)))[-1]))
    elif neck:
        cuts.append(lowest(neck) - margin)
    return min(cuts)


def apply_head_guard(
    body: np.ndarray, masks: dict[str, np.ndarray | None], spacing_mm: float
) -> tuple[np.ndarray, int | None]:
    """The body mask with nothing above face_free_top, and that top (None: no cut).

    Raises ValueError when a guard mask is missing, when the head comes within
    the margin of the scan's bottom slice (a negative top, which as an index
    would wrap round and blank only the top slices), or when nothing of the
    body is left below the cut.
    """
    top = face_free_top(masks, guard_margin(spacing_mm))
    if top is None:
        return body, None
    if top < 0:
        raise ValueError(
            f"the head reaches within {GUARD_MARGIN_MM:.0f} mm of the scan's bottom slice "
            f"(cut at {top}); refusing the case"
        )
    body = body.copy()
    body[:, :, top + 1 :] = False  # top + 1 >= 1, so this never wraps
    if not body.any():
        raise ValueError("nothing of the trunk is left below the head")
    return body, top


def srgb(linear) -> np.ndarray:
    """Linear RGB (structures.py's colours) to sRGB, the space of a PNG and of a
    glTF baseColorTexture, so a tint matches the same structure's mesh."""
    c = np.clip(np.asarray(linear, dtype=np.float64), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def canonical(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(path))  # voxel axes along R, A, S


def lanczos_weights(size: int, factor: int, taps: int = LANCZOS_TAPS) -> np.ndarray:
    """(size * factor, size) resampling matrix for a Lanczos-`taps` upsample.

    Output pixel x samples the input at (x + 0.5) / factor - 0.5, so pixel
    centres line up and the image's edges stay where they were; taps that fall
    outside the image are clamped to its edge.
    """
    out = np.arange(size * factor)
    centre = (out + 0.5) / factor - 0.5
    base = np.floor(centre).astype(int)
    matrix = np.zeros((size * factor, size))
    for offset in range(-taps + 1, taps + 1):
        source = base + offset
        distance = centre - source
        weight = np.sinc(distance) * np.sinc(distance / taps)
        np.add.at(matrix, (out, np.clip(source, 0, size - 1)), weight)
    return matrix / matrix.sum(1, keepdims=True)


def fixed_point(matrix: np.ndarray) -> np.ndarray:
    """Resampling weights as integers in steps of 1/WEIGHT_ONE, each row summing
    to exactly WEIGHT_ONE (the rounding error goes onto the row's largest weight)."""
    weights = np.round(matrix * WEIGHT_ONE).astype(np.int64)
    weights[np.arange(len(weights)), np.abs(weights).argmax(1)] += WEIGHT_ONE - weights.sum(1)
    return weights


def upsample(rgba: np.ndarray, factor: int = 2) -> np.ndarray:
    """Lanczos upsample of the colour for display, with an anti-aliased edge.

    Colour is filtered premultiplied by the native mask, so clear pixels add no
    dark fringe: a flat colour stays exactly flat up to any edge. The edge
    starts from the native mask with each pixel repeated factor x factor times
    (in place at corners, diagonals and one-voxel features), is pulled in by
    one source voxel and blurred slightly; so the display drops any part of
    the body two voxels wide or less (see docs/imaging-layer.md). The filter
    runs in integers (fixed-point weights, integer products and sums) and the
    blur in a fixed order, so the result does not depend on the BLAS build or
    the thread count; the float steps before that, computing the weights,
    would have to land within a rounding error of a boundary to change a byte.
    """
    inside = rgba[..., 3] >= 128
    premultiplied = np.dstack([rgba[..., :3] * inside[..., None], inside * 255]).astype(np.int64)
    rows = fixed_point(lanczos_weights(rgba.shape[0], factor))
    cols = fixed_point(lanczos_weights(rgba.shape[1], factor))
    big = np.stack([rows @ premultiplied[..., c] @ cols.T for c in range(4)], -1)
    weight = big[..., 3:].astype(np.float64)  # Lanczos alpha, times 255 * WEIGHT_ONE**2
    nearest = np.repeat(np.repeat(rgba[..., :3], factor, 0), factor, 1) / 255
    # Where the filtered alpha is small (a one-voxel feature, a sharp corner), the
    # ratio is ill-conditioned: take the source voxel's own colour instead.
    solid = weight > 0.25 * 255 * WEIGHT_ONE**2
    rgb = np.where(solid, big[..., :3] / np.maximum(weight, 1), nearest)
    # Lanczos rings beside a sharp change in colour (a tinted structure at the
    # body's edge); keep each pixel within the range of the body voxels around
    # its source voxel, so no ring can overshoot or darken the edge.
    low, high = neighbourhood_range(rgba[..., :3] / 255, inside)
    low, high = (np.repeat(np.repeat(v, factor, 0), factor, 1) for v in (low, high))
    rgb = np.clip(rgb, np.minimum(low, nearest), np.maximum(high, nearest))
    mask = np.repeat(np.repeat(inside, factor, 0), factor, 1)
    # The edge is anti-aliased: the native mask is pulled in by one source voxel
    # (the trunk mask's outermost voxel is often air at the skin, a dark rim)
    # and blurred slightly, and colour from inside that is bled outwards under
    # the soft fringe, so the edge shows the slice's own edge colour, not black.
    core = shrink(mask, factor)
    soft = np.clip(blur(core.astype(np.float64), EDGE_SOFTNESS), 0, 1)
    rgb = bleed(np.clip(rgb, 0, 1), core, steps=EDGE_BLEED)
    return np.round(np.dstack([rgb, soft]) * 255).astype(np.uint8)


def neighbourhood_range(rgb: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel min and max over each pixel's 3 x 3 neighbours inside the body
    (padded, not wrapped, at the image's border; +inf/-inf where none is inside)."""
    height, width = inside.shape
    low = np.pad(np.where(inside[..., None], rgb, np.inf), ((1, 1), (1, 1), (0, 0)),
                 constant_values=np.inf)  # fmt: skip
    high = np.pad(np.where(inside[..., None], rgb, -np.inf), ((1, 1), (1, 1), (0, 0)),
                  constant_values=-np.inf)  # fmt: skip
    shifts = [(dy, dx) for dy in range(3) for dx in range(3)]
    return (
        np.min([low[dy : dy + height, dx : dx + width] for dy, dx in shifts], 0),
        np.max([high[dy : dy + height, dx : dx + width] for dy, dx in shifts], 0),
    )


def shrink(mask: np.ndarray, steps: int) -> np.ndarray:
    """Erodes a 2D mask by `steps` pixels (4-neighbourhood)."""
    for _ in range(steps):
        p = np.pad(mask, 1, mode="edge")  # the image border is not an edge of the body
        mask = mask & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a 2D array, edges clamped. Written as a sum of
    shifted copies in a fixed order, not a convolution, so no BLAS is involved
    and the result is the same on every machine."""
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    height, width = image.shape
    padded = np.pad(image, radius, mode="edge")
    rows = sum(k * padded[:, i : i + width] for i, k in enumerate(kernel))
    return sum(k * rows[i : i + height] for i, k in enumerate(kernel))


def shifted(array: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """array moved by (dy, dx) pixels (|dy|, |dx| <= 1), as np.roll would, but
    with the uncovered border filled with `fill` instead of wrapping round."""
    pad = [(1, 1), (1, 1)] + [(0, 0)] * (array.ndim - 2)
    padded = np.pad(array, pad, constant_values=fill)
    height, width = array.shape[:2]
    return padded[1 - dy : 1 - dy + height, 1 - dx : 1 - dx + width]


def bleed(rgb: np.ndarray, valid: np.ndarray, steps: int) -> np.ndarray:
    """Extends colour from valid pixels into their neighbours, `steps` pixels out."""
    rgb, valid = rgb.copy(), valid.copy()
    for _ in range(steps):
        total = np.zeros_like(rgb)
        count = np.zeros(valid.shape)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            # Padded, not rolled: a pixel on one border must not take colour
            # from the opposite border.
            neighbour = shifted(valid, dy, dx, False)
            total += shifted(rgb, dy, dx, 0.0) * neighbour[..., None]
            count += neighbour
        grow = ~valid & (count > 0)
        rgb[grow] = total[grow] / count[grow, None]
        valid |= grow
    return rgb


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
            # MASK, not BLEND: the three quads cut through each other, and a
            # BLEND material is sorted per object and drawn without depth
            # writes (three.js's GLTFLoader), so one plane would be drawn over
            # another wrongly. The soft alpha in the texture still gives a
            # smooth cutoff contour under bilinear filtering, and a viewer that
            # turns on alpha-to-coverage gets the soft edge too.
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
    spacing = float(np.linalg.norm(affine[:3, 2]))
    try:
        body, top = apply_head_guard(body, guard, spacing)
    except ValueError as error:
        sys.exit(f"head guard: {case}: {error}")
    if top is not None:
        print(f"head guard: the scan reaches the head; slices stop at axial index {top}")
    inside = np.argwhere(body)
    lo = inside.min(0)
    hi = inside.max(0)
    lo[2] = max(int(lo[2]), femur_top - int(round(50 / spacing)))
    if lo[2] > hi[2]:
        sys.exit(f"head guard: {case}: nothing is left between the femurs and the head cut")
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
        display = png(upsample(image, DISPLAY_FACTOR))
        (out / f"{plane}@{DISPLAY_FACTOR}x.png").write_bytes(display)
        ras = quad_corners(origin, axis, index, zero, top)
        corners = to_reference(ras, centre, scale, shift)
        planes.append({"plane": plane, "png": display, "corners": corners})
        record.append({
            "plane": plane,
            "voxelIndex": int(index + offset[axis]),
            "image": f"{plane}.png",
            "pixels": [int(image.shape[1]), int(image.shape[0])],
            "sha256": hashlib.sha256(data).hexdigest(),
            "display": {
                "image": f"{plane}@{DISPLAY_FACTOR}x.png",
                "pixels": [int(v) * DISPLAY_FACTOR for v in image.shape[1::-1]],
                "resampling": f"Lanczos-{LANCZOS_TAPS}; edge one voxel in, anti-aliased",
                "sha256": hashlib.sha256(display).hexdigest(),
            },
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
