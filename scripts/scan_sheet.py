"""
A contact sheet of a synthetic body before and after scripts/scan_damage.py:
clean and damaged, front and back, whole body, in grey.

  python scripts/scan_sheet.py --clean <clean.glb> --damaged <damaged.glb> --out <png>
      [--body <rigged ANNY glb>]

No face is ever drawn, synthetic or not: in every panel the body is cut off
at the base of the neck and the space above it is filled with a flat band,
and the sheet says so. The cut is the lowest vertex that follows a head or
neck joint of --body's rig; without --body it is NECK_AT of the height.

Pure Python and numpy: the picture is made of points spread over the surface,
each shaded by how squarely its face looks at the viewer; the inside of the
body, seen through a hole, is dark. A path under data/vitronic/ is refused.
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
Image = npt.NDArray[np.uint8]

NECK_AT = 0.82  # share of the height where a body without a rig is cut: under the chin
HEAD_JOINTS = ("head", "neck", "eye")  # ANNY's joints above the base of the neck
PANEL = (420, 620)  # width and height of one panel, pixels
RANGE_M = 2.1  # the scanner's measuring height fills a panel
POINTS = 400_000  # points drawn per panel
PAPER, BAND, INK, INSIDE = 255, 205, 0, 25  # greys: background, the band, text, inside a hole
NOTE = "HEAD BLANKED ABOVE THE NECK: NO FACE IS RENDERED"
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


def panel(mesh: Mesh, neck: float, back: bool, size: tuple[int, int] = PANEL) -> Image:
    """One whole-body view, from the front or from behind, of everything below
    `neck` (metres above the floor). Above it the panel holds a flat band and
    nothing else: faces that reach above the neck are not drawn at all."""
    width, height = size
    image = np.full((height, width), PAPER, dtype=np.uint8)
    floor = mesh.points[:, 1].min()
    scale = (height - 20) / RANGE_M
    cut = int(round(height - 10 - neck * scale))  # the first row below the band
    below = (mesh.points[mesh.triangles][:, :, 1] - floor <= neck).all(1)
    if below.any():
        body = Mesh(mesh.points, mesh.triangles[below])
        where, normal, _ = sample_surface(body, POINTS, np.random.default_rng(0))
        toward = -1.0 if back else 1.0  # the viewer stands at -Z or at +Z
        middle = (mesh.points[:, 0].min() + mesh.points[:, 0].max()) / 2
        x = (toward * (where[:, 0] - middle) * scale + width / 2).astype(np.int64)
        y = (height - 10 - (where[:, 1] - floor) * scale).astype(np.int64)
        facing = toward * normal[:, 2]
        shade = np.where(facing > 0, 70 + 170 * facing, INSIDE).astype(np.uint8)
        order = np.argsort(toward * where[:, 2])  # nearest drawn last
        ok = (x >= 0) & (x < width) & (y >= cut) & (y < height)
        order = order[ok[order]]
        image[y[order], x[order]] = shade[order]
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


def sheet(clean: Mesh, damaged: Mesh, neck: float, size: tuple[int, int] = PANEL) -> Image:
    """Four panels side by side, each with its name, under a line that says the
    head is blanked."""
    panels = []
    for name, mesh in (("CLEAN", clean), ("DAMAGED", damaged)):
        for back in (False, True):
            image = panel(mesh, neck, back, size)
            write_text(image, f"{name} {'BACK' if back else 'FRONT'}", 8, 8)
            panels.append(image)
    body = np.hstack(panels)
    top = np.full((30, body.shape[1]), PAPER, dtype=np.uint8)
    write_text(top, NOTE, 8, 8)
    return np.vstack([top, body])


def write_png(path: Path, image: Image) -> None:
    """An 8-bit grey PNG."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\0" + row.tobytes() for row in image)
    head = struct.pack(">IIBBBBB", image.shape[1], image.shape[0], 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head)
                     + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--damaged", type=Path, required=True)
    parser.add_argument("--body", type=Path, help="the rigged body, to find the neck")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.clean, args.damaged, args.body, args.out):
        if path and (PERSONAL.search(str(path)) or PERSONAL.search(str(path.resolve()))):
            sys.exit(f"{path}: refused: data/vitronic/ holds real people's scans, and no "
                     "picture is ever made of one")  # fmt: skip
    clean, damaged = read_mesh(args.clean), read_mesh(args.damaged)
    neck = neck_height(args.body, clean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_png(args.out, sheet(clean, damaged, neck))
    print(f"scan_sheet: wrote {args.out}, head blanked above {neck:.3f} m")


if __name__ == "__main__":
    main()
