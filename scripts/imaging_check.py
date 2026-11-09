"""
Self-check for imaging.py on synthetic volumes: no CT, no download.

  python scripts/imaging_check.py

Checks the slices' orientation on an asymmetric phantom stored in a
non-canonical voxel order: markers on the body's left, front and top must land
on the expected side of each image (radiology's layout: the body's right on the
image's left, superior or anterior at the top), each marker's pixel must map
through quad_corners back to the marker's own position, and the left marker
must end up on +X in glTF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imaging import WINDOWS, quad_corners, slice_image, to_reference  # noqa: E402

# Stored as (posterior, left, superior) at 0.9, 0.8 and 1.5 mm, so canonical()
# has to flip two axes and permute all three.
STORED_AFFINE = np.array([
    [0.0, -0.8, 0.0, 30.0],
    [-0.9, 0.0, 0.0, 40.0],
    [0.0, 0.0, 1.5, -20.0],
    [0.0, 0.0, 0.0, 1.0],
])  # fmt: skip
STORED_SHAPE = (60, 70, 40)
# Marker offsets from the phantom's centre in RAS millimetres (left is -x),
# and the HU each is drawn with: white, mid grey and dark grey in the soft window.
MARKERS = {
    "left": (np.array([-15.0, 0.0, 0.0]), 240.0),
    "anterior": (np.array([0.0, 15.0, 0.0]), 40.0),
    "superior": (np.array([0.0, 0.0, 15.0]), -60.0),
}


def phantom():
    """The canonical CT, labels and body mask, and their affine and centre voxel."""
    ct = np.full(STORED_SHAPE, -1000.0, dtype=np.float32)
    inverse = np.linalg.inv(STORED_AFFINE)
    centre_ras = STORED_AFFINE[:3, :3] @ (np.array(STORED_SHAPE) / 2) + STORED_AFFINE[:3, 3]
    for offset, hu in MARKERS.values():
        ijk = np.round(inverse[:3, :3] @ (centre_ras + offset) + inverse[:3, 3]).astype(int)
        ct[tuple(slice(i - 1, i + 2) for i in ijk)] = hu  # a 3-voxel cube
    image = nib.as_closest_canonical(nib.Nifti1Image(ct, STORED_AFFINE))
    ct = np.asanyarray(image.dataobj).astype(np.float32)
    affine = image.affine
    centre = np.round(np.linalg.inv(affine)[:3, :3] @ centre_ras + np.linalg.inv(affine)[:3, 3])
    labels = np.zeros(ct.shape, dtype=np.uint8)
    body = np.ones(ct.shape, dtype=bool)
    return ct, labels, body, affine, centre.astype(int), centre_ras


def grey(hu: float) -> int:
    width, level = WINDOWS["soft"]
    return int(np.round(np.clip((hu - (level - width / 2)) / width, 0, 1) * 255))


def main() -> int:
    failures = []

    def check(ok: bool, message: str) -> None:
        print(("ok   " if ok else "FAIL ") + message)
        if not ok:
            failures.append(message)

    ct, labels, body, affine, centre, centre_ras = phantom()
    check(bool(np.allclose(np.diag(affine)[:3], [0.8, 0.9, 1.5])), "the phantom is canonical")
    zero, top = np.zeros(3, dtype=int), np.array(ct.shape) - 1
    # plane: (axis, markers in that plane and where each must land: (row half, column half))
    expect = {
        "axial": (2, {"left": (None, "right"), "anterior": ("top", None)}),
        "coronal": (1, {"left": (None, "right"), "superior": ("top", None)}),
        "sagittal": (0, {"anterior": (None, "left"), "superior": ("top", None)}),
    }
    for plane, (axis, marks) in expect.items():
        image, _ = slice_image(ct, labels, body, axis, int(centre[axis]), WINDOWS["soft"])
        height, width = image.shape[:2]
        tl, tr, bl, br = quad_corners(affine, axis, int(centre[axis]), zero, top)
        for name, (row_half, col_half) in marks.items():
            offset, hu = MARKERS[name]
            pixels = np.argwhere(image[..., 0] == grey(hu))
            if not len(pixels):
                check(False, f"{plane}: the {name} marker is in the slice")
                continue
            r, c = pixels.mean(0)
            if row_half:
                ok = r < height / 2 if row_half == "top" else r > height / 2
                check(ok, f"{plane}: the {name} marker is in the image's {row_half} half")
            if col_half:
                ok = c < width / 2 if col_half == "left" else c > width / 2
                check(ok, f"{plane}: the {name} marker is in the image's {col_half} half")
            # The pixel's centre, through the quad (uv (0, 0) is the top-left corner).
            u, v = (c + 0.5) / width, (r + 0.5) / height
            ras = (1 - v) * ((1 - u) * tl + u * tr) + v * ((1 - u) * bl + u * br)
            error = np.linalg.norm(ras - (centre_ras + offset))
            check(error < 1.0, f"{plane}: the {name} marker's pixel maps back to it in 3D "
                  f"({error:.2f} mm off)")  # fmt: skip
        if plane == "axial":
            # The top-right corner is the body's left front.
            ok = tr[0] < tl[0] and tr[1] > br[1]
            check(bool(ok), "axial: the quad's top-right corner is the body's left front")

    left = to_reference((centre_ras + MARKERS["left"][0])[None], np.zeros(3), 1.0, np.zeros(3))
    middle = to_reference(centre_ras[None], np.zeros(3), 1.0, np.zeros(3))
    check(bool(left[0, 0] > middle[0, 0]), "the left marker lands on +X in glTF")

    print("imaging check passed" if not failures else f"imaging check failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
