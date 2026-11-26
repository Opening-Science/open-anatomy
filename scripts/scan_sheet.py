"""
A contact sheet of a synthetic body before and after scripts/scan_damage.py.
The top row is the whole body, clean and damaged, front and back; the second
row is each seen from under the floor (the soles and the crotch) and both
hands close up from outside the body. Where a hole lets the inside of the body
show, it is red, so a hole never reads as a shadow.

  python scripts/scan_sheet.py --clean <clean.glb> --damaged <damaged.glb> --out <png>
      [--body <rigged ANNY glb>]

No face is ever drawn, synthetic or not: every panel draws only faces wholly
below the base of the neck, the whole-body panels stop at the neck under a
flat band, and the sheet says so. The cut is the lowest vertex that follows a
head or neck joint of --body's rig; without --body it is NECK_AT of the
height. The hands come from the same rig, or without one from the lowest part
of each arm.

Pure Python and numpy: points spread over the surface, the nearest one on
each pixel kept. A path under data/vitronic/ is refused.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scan_damage import PERSONAL, Mesh, read_mesh  # noqa: E402
from size_body import accessor, read_glb  # noqa: E402

FloatArray = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]  # height x width x 3, RGB
GreyImage = npt.NDArray[np.uint8]  # height x width

NECK_AT = 0.82  # share of the height where a body without a rig is cut: under the chin
HEAD_JOINTS = ("head", "neck", "eye")  # ANNY's joints above the base of the neck
HAND_JOINTS = ("wrist", "metacarpal", "finger")  # ANNY's joints of a hand
HAND_M = 0.2  # without a rig, a hand is the lowest 20 cm of an arm
ARM_SHARE = 0.65  # without a rig, an arm is what lies beyond 65% of the half-width
PAD_M = 0.015  # room around a hand's box
MARGIN_M = 0.03  # room around the body in a panel
PANEL = (420, 620)  # width and height of one grey panel from panel(), pixels
RANGE_M = 2.1  # the scanner's measuring height fills a grey panel
BODY_HEIGHT = 640  # pixels from the floor to the neck in the sheet's top row
LOWER_HEIGHT = 300  # pixels of the sheet's second row
LABEL = 22  # pixels of the band that names each panel
POINTS_PER_PIXEL = 4.0  # surface points drawn per pixel of surface area
MAX_POINTS = 4_000_000
PAPER, BAND, INK, INSIDE = 255, 205, 0, 25  # greys: background, the band, text, a hole in panel()
HOLE = (205, 30, 30)  # the inside of the body, seen through a hole
NOTE = "HEAD BLANKED ABOVE THE NECK: NO FACE IS RENDERED"
LEGEND = "RED: THE INSIDE OF THE BODY, SEEN THROUGH A HOLE"
NAMES = ("CLEAN FRONT", "CLEAN BACK", "DAMAGED FRONT", "DAMAGED BACK",
         "CLEAN FROM BELOW", "DAMAGED FROM BELOW", "CLEAN HANDS", "DAMAGED HANDS")  # fmt: skip
# Capital letters five dots wide and seven high, a row of bits per line.
FONT = {
    "A": "01110 10001 10001 11111 10001 10001 10001",
    "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110",
    "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111",
    "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111",
    "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110",
    "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001",
    "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110",
    "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100",
    "V": "10001 10001 10001 10001 10001 01010 00100",
    ":": "00000 00100 00100 00000 00100 00100 00000",
    " ": "00000 00000 00000 00000 00000 00000 00000",
    "P": "11110 10001 10001 11110 10000 10000 10000",
    "U": "10001 10001 10001 10001 10001 10001 01110",
    "W": "10001 10001 10001 10101 10101 11011 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100",
    ",": "00000 00000 00000 00000 00100 00100 01000",
}


def sample_surface(mesh: Mesh, count: int, rng: np.random.Generator):
    """`count` points spread evenly over the surface by area: the points, the
    unit normals of their faces and which face each lies on."""
    corners = mesh.points[mesh.triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(cross, axis=1)
    normal = cross / np.where(length == 0, 1.0, length)[:, None]
    face = np.searchsorted(np.cumsum(length) / length.sum(), rng.random(count), side="right")
    face = np.minimum(face, len(length) - 1)
    u, v = rng.random(count), rng.random(count)
    over = u + v > 1
    u[over], v[over] = 1 - u[over], 1 - v[over]
    a, b, c = corners[face, 0], corners[face, 1], corners[face, 2]
    return a + u[:, None] * (b - a) + v[:, None] * (c - a), normal[face], face


def neck_height(body: Path | None, mesh: Mesh) -> float:
    """The height (metres above the floor) at which every panel is cut."""
    floor, top = float(mesh.points[:, 1].min()), float(mesh.points[:, 1].max())
    if body is not None:
        gltf, blob = read_glb(body)
        if gltf.get("skins"):
            names = [gltf["nodes"][j].get("name", "") for j in gltf["skins"][0]["joints"]]
            counts = [a["count"] for a in gltf["accessors"]]
            best = max(gltf["meshes"], key=lambda m: counts[m["primitives"][0]["indices"]])
            attributes = best["primitives"][0]["attributes"]
            points = accessor(gltf, blob, attributes["POSITION"])
            joints = accessor(gltf, blob, attributes["JOINTS_0"]).astype(np.int64)
            weights = accessor(gltf, blob, attributes["WEIGHTS_0"])
            strongest = joints[np.arange(len(joints)), weights.argmax(1)]
            head = np.array([n.startswith(HEAD_JOINTS) for n in names])[strongest]
            if head.any():
                return float(points[head, 1].min()) - floor
    return NECK_AT * (top - floor)


FRONT, BACK, BELOW = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)
UP, FORWARD = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
Window = tuple[
    float, float, float, float
]  # across and up of the centre, metres shown across and up


def project(points: FloatArray, toward, up):
    """Across, up and depth of each point, for a viewer standing at `toward`
    (a unit vector from the body to the viewer) with `up` up the picture.
    A smaller depth is nearer the viewer."""
    toward, up = np.asarray(toward, dtype=np.float64), np.asarray(up, dtype=np.float64)
    right = np.cross(up, toward)
    return points @ right, points @ up, -(points @ toward)


def render(mesh: Mesh, keep: npt.NDArray[np.bool_], toward, up, window: Window,
           size: tuple[int, int]) -> Image:  # fmt: skip
    """An orthographic RGB picture of the faces in `keep`, seen from `toward`,
    `window` scaled to fit `size` (width, height in pixels).

    Each pixel shows the nearest surface point that falls on it. The outside
    of the body is grey, shaded by how squarely it faces the viewer, and each
    of its points covers three pixels by three, so no gap between points lets the
    far side through. The inside of the body, which shows only through a hole,
    is red and covers its own pixel only, so it never spills past an outline."""
    width, height = size
    image = np.full((height, width, 3), PAPER, dtype=np.uint8)
    if not keep.any():
        return image
    across, upward, span_across, span_up = window
    scale = min(width / span_across, height / span_up)  # pixels per metre
    body = Mesh(mesh.points, mesh.triangles[keep])
    corners = body.points[body.triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = float(np.linalg.norm(cross, axis=1).sum()) / 2
    count = int(np.clip(POINTS_PER_PIXEL * area * scale**2, 20_000, MAX_POINTS))
    where, normal, _ = sample_surface(body, count, np.random.default_rng(0))
    u, v, depth = project(where, toward, up)
    facing = normal @ np.asarray(toward, dtype=np.float64)
    x = np.floor((u - across) * scale + width / 2).astype(np.int64)
    y = np.floor(height / 2 - (v - upward) * scale).astype(np.int64)
    outside = facing > 0
    grey = (70 + 170 * np.clip(facing, 0, 1)).astype(np.uint8)
    colour = np.where(outside[:, None], np.repeat(grey[:, None], 3, 1),
                      np.array(HOLE, dtype=np.uint8)[None])  # fmt: skip
    # Outside points cover the three pixels by three around them; inside points
    # their own pixel.
    lit = np.flatnonzero(outside)
    xs, ys, ks = [x], [y], [np.arange(len(x))]
    for dx, dy in ((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy):
        xs.append(x[lit] + dx)
        ys.append(y[lit] + dy)
        ks.append(lit)
    x, y, k = np.concatenate(xs), np.concatenate(ys), np.concatenate(ks)
    ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    pixel, k = (y * width + x)[ok], k[ok]
    if not len(pixel):
        return image
    order = np.lexsort((depth[k], pixel))  # by pixel, the nearest first
    pixel, k = pixel[order], k[order]
    first = np.r_[True, pixel[1:] != pixel[:-1]]
    image.reshape(-1, 3)[pixel[first]] = colour[k[first]]
    return fill_gaps(image)


def fill_gaps(image: Image) -> Image:
    """Fills each blank pixel that has at least five drawn neighbours of eight
    with the colour of one of them: a gap between points inside a hole, which
    the inside's single-pixel points can leave. A blank pixel along an outline
    has at most four drawn neighbours and stays blank."""
    blank = (image == PAPER).all(2)
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), constant_values=PAPER)
    drawn = ~np.pad(blank, 1, constant_values=True)
    height, width = blank.shape
    count = np.zeros(blank.shape, dtype=np.int64)
    fill = image.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                near = drawn[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
                count += near
                fill[near] = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width][near]
    out = image.copy()
    gap = blank & (count >= 5)
    out[gap] = fill[gap]
    return out


def below_neck(mesh: Mesh, neck: float) -> npt.NDArray[np.bool_]:
    """The faces wholly below `neck` metres above the floor: the only ones
    any panel draws."""
    floor = mesh.points[:, 1].min()
    return (mesh.points[mesh.triangles][:, :, 1] - floor <= neck).all(1)


def body_box(frame: Mesh, neck: float) -> tuple[FloatArray, FloatArray]:
    """The box around `frame` (the clean body) from the floor to the neck,
    with MARGIN_M of room: every panel of the body is cropped to it, so the
    clean and damaged panels line up."""
    p = frame.points[frame.points[:, 1] - frame.points[:, 1].min() <= neck]
    low, high = p.min(0) - MARGIN_M, p.max(0) + MARGIN_M
    low[1], high[1] = p[:, 1].min() - MARGIN_M, p[:, 1].min() + neck
    return low, high


def whole_body(mesh: Mesh, box, toward, size: tuple[int, int]) -> Image:
    """The body from the front or from behind, from the floor to the neck and
    nothing above it."""
    low, high = box
    neck = float(high[1] - low[1] - MARGIN_M)
    centre = (low + high) / 2
    across, upward, _ = project(centre[None], toward, UP)
    window = (float(across[0]), float(upward[0]), float(high[0] - low[0]), float(high[1] - low[1]))
    return render(mesh, below_neck(mesh, neck), toward, UP, window, size)


def from_below(mesh: Mesh, box, size: tuple[int, int]) -> Image:
    """The body seen from under the floor, its front at the top: the soles
    and the crotch."""
    low, high = box
    neck = float(high[1] - low[1] - MARGIN_M)
    centre = (low + high) / 2
    across, upward, _ = project(centre[None], BELOW, FORWARD)
    window = (float(across[0]), float(upward[0]), float(high[0] - low[0]), float(high[2] - low[2]))
    return render(mesh, below_neck(mesh, neck), BELOW, FORWARD, window, size)


def hand_boxes(body: Path | None, frame: Mesh) -> list[tuple[FloatArray, FloatArray]]:
    """The box around each hand, from the wrist to the fingertips, ordered from
    -X to +X: from --body's rig (its wrist, metacarpal and finger joints), or
    without a rig, the lowest HAND_M of each arm, an arm being what lies beyond
    ARM_SHARE of the body's half-width."""
    if body is not None:
        gltf, blob = read_glb(body)
        if gltf.get("skins"):
            names = [gltf["nodes"][j].get("name", "") for j in gltf["skins"][0]["joints"]]
            counts = [a["count"] for a in gltf["accessors"]]
            best = max(gltf["meshes"], key=lambda m: counts[m["primitives"][0]["indices"]])
            attributes = best["primitives"][0]["attributes"]
            points = accessor(gltf, blob, attributes["POSITION"]).astype(np.float64)
            joints = accessor(gltf, blob, attributes["JOINTS_0"]).astype(np.int64)
            weights = accessor(gltf, blob, attributes["WEIGHTS_0"])
            strongest = joints[np.arange(len(joints)), weights.argmax(1)]
            boxes = []
            for side in (".L", ".R"):
                hand = np.array([n.startswith(HAND_JOINTS) and n.endswith(side) for n in names])
                if hand[strongest].any():
                    p = points[hand[strongest]]
                    boxes.append((p.min(0) - PAD_M, p.max(0) + PAD_M))
            if len(boxes) == 2:
                return sorted(boxes, key=lambda b: float(b[0][0]))
    p = frame.points
    middle = (p[:, 0].max() + p[:, 0].min()) / 2
    half = (p[:, 0].max() - p[:, 0].min()) / 2
    floor, top = p[:, 1].min(), p[:, 1].max()
    boxes = []
    for sign in (-1.0, 1.0):
        rise = (p[:, 1] - floor) / (top - floor)
        arm = (sign * (p[:, 0] - middle) > ARM_SHARE * half) & (rise > 0.3) & (rise < 0.75)
        if arm.any():
            hand = p[arm & (p[:, 1] < p[arm, 1].min() + HAND_M)]
            boxes.append((hand.min(0) - PAD_M, hand.max(0) + PAD_M))
    return boxes


def hands(mesh: Mesh, boxes, neck: float, size: tuple[int, int]) -> Image:
    """Each hand seen from outside the body, side by side. Only the faces in a
    hand's box are drawn, so the thigh does not stand in the way."""
    width, height = size
    image = np.full((height, width, 3), PAPER, dtype=np.uint8)
    middle = (mesh.points[:, 0].max() + mesh.points[:, 0].min()) / 2
    corners = mesh.points[mesh.triangles]
    for slot, (low, high) in enumerate(boxes[:2]):
        keep = ((corners >= low) & (corners <= high)).all((1, 2)) & below_neck(mesh, neck)
        toward = (1.0 if (low[0] + high[0]) / 2 > middle else -1.0, 0.0, 0.0)
        across, upward, _ = project(((low + high) / 2)[None], toward, UP)
        window = (
            float(across[0]),
            float(upward[0]),
            float(high[2] - low[2]),
            float(high[1] - low[1]),
        )
        half = render(mesh, keep, toward, UP, window, (width // 2, height))
        image[:, slot * (width - width // 2) : slot * (width - width // 2) + width // 2] = half
    return image


def panel(mesh: Mesh, neck: float, back: bool, size: tuple[int, int] = PANEL) -> GreyImage:
    """One whole-body view in grey, from the front or from behind, of the faces
    below `neck` (metres above the floor), RANGE_M of height filling the panel.
    Above the neck the panel holds a flat band and nothing else; the inside of
    the body, seen through a hole, is the dark grey INSIDE."""
    width, height = size
    scale = (height - 20) / RANGE_M
    floor = float(mesh.points[:, 1].min())
    middle = float(mesh.points[:, 0].max() + mesh.points[:, 0].min()) / 2
    toward = BACK if back else FRONT
    upward = floor + (height / 2 - 10) / scale
    window = (-middle if back else middle, upward, width / scale, height / scale)
    colour = render(mesh, below_neck(mesh, neck), toward, UP, window, size)
    image = colour[:, :, 1].copy()
    image[colour[:, :, 0] != colour[:, :, 1]] = INSIDE
    cut = int(round(height - 10 - neck * scale))
    image[:cut] = BAND
    return image


def write_text(image: Image, text: str, x: int, y: int, dot: int = 2) -> None:
    """Writes capital letters into the image, its top left corner at (x, y)."""
    for letter in text:
        rows = FONT[letter].split()
        for r, row in enumerate(rows):
            for c, bit in enumerate(row):
                if bit == "1":
                    image[y + r * dot : y + (r + 1) * dot, x + c * dot : x + (c + 1) * dot] = INK
        x += 6 * dot


def labelled(image: Image, name: str) -> Image:
    """The panel under a band that names it."""
    band = np.full((LABEL, image.shape[1], 3), BAND, dtype=np.uint8)
    write_text(band, name, 6, 4)
    return np.vstack([band, image])


def sheet(clean: Mesh, damaged: Mesh, neck: float, boxes=None,
          body_height: int = BODY_HEIGHT, lower_height: int = LOWER_HEIGHT) -> Image:  # fmt: skip
    """Eight named panels in two rows under two lines of text: the head is
    blanked, and red is the inside of the body. The top row is cropped to the
    clean body from the floor to the neck; the second row has the same panel
    width."""
    box = body_box(clean, neck)
    low, high = box
    width = int(np.ceil(body_height * (high[0] - low[0]) / (high[1] - low[1])))
    width += width % 2
    top_size, lower_size = (width, body_height), (width, lower_height)
    boxes = hand_boxes(None, clean) if boxes is None else boxes
    pictures = [
        whole_body(clean, box, FRONT, top_size),
        whole_body(clean, box, BACK, top_size),
        whole_body(damaged, box, FRONT, top_size),
        whole_body(damaged, box, BACK, top_size),
        from_below(clean, box, lower_size),
        from_below(damaged, box, lower_size),
        hands(clean, boxes, neck, lower_size),
        hands(damaged, boxes, neck, lower_size),
    ]
    panels = [labelled(p, name) for p, name in zip(pictures, NAMES, strict=True)]
    rows = []
    for row in (panels[:4], panels[4:]):
        gap = np.full((row[0].shape[0], 4, 3), PAPER, dtype=np.uint8)
        rows.append(np.hstack([part for p in row for part in (p, gap)][:-1]))
    head = np.full((34, rows[0].shape[1], 3), PAPER, dtype=np.uint8)
    write_text(head, NOTE, 6, 2)
    write_text(head, LEGEND, 6, 18)
    head[18:32, : 6 * 2 * 3] = np.where(
        head[18:32, : 6 * 2 * 3] == INK, HOLE, head[18:32, : 6 * 2 * 3]
    )
    between = np.full((6, rows[0].shape[1], 3), PAPER, dtype=np.uint8)
    return np.vstack([head, rows[0], between, rows[1]])


def write_png(path: Path, image: Image | GreyImage) -> None:
    """An 8-bit PNG, grey for a 2-D image and RGB for a 3-D one."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    kind = 2 if image.ndim == 3 else 0
    rows = b"".join(b"\0" + row.tobytes() for row in image)
    head = struct.pack(">IIBBBBB", image.shape[1], image.shape[0], 8, kind, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head)
                     + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--damaged", type=Path, required=True)
    parser.add_argument("--body", type=Path, help="the rigged body, to find the neck and hands")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.clean, args.damaged, args.body, args.out):
        if path and (PERSONAL.search(str(path)) or PERSONAL.search(str(path.resolve()))):
            sys.exit(f"{path}: refused: data/vitronic/ holds real people's scans, and no "
                     "picture is ever made of one")  # fmt: skip
    clean, damaged = read_mesh(args.clean), read_mesh(args.damaged)
    neck = neck_height(args.body, clean)
    boxes = hand_boxes(args.body, clean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_png(args.out, sheet(clean, damaged, neck, boxes))
    print(f"scan_sheet: wrote {args.out}, head blanked above {neck:.3f} m")


if __name__ == "__main__":
    main()
