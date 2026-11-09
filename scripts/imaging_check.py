"""
Self-check for imaging.py on synthetic volumes and masks: no CT, no download.

  python scripts/imaging_check.py

Checks the slices' orientation on an asymmetric phantom stored in a
non-canonical voxel order: markers on the body's left, front and top must land
on the expected side of each image (radiology's layout: the body's right on the
image's left, superior or anterior at the top), each marker's pixel must map
through quad_corners back to the marker's own position, and the left marker
must end up on +X in glTF.

Checks that the head guard (face_free_top and apply_head_guard)
- crops nothing when no skull, brain or C1-C6 is in the scan;
- stops at the top of C7 when the scan reaches the head and C7 is there;
- stops 2 cm below the skull when a bowed head brings the jaw level with C7;
- stops 2 cm below the lowest head or neck structure when C7 is not;
- turns 2 cm into slices from the scan's own slice spacing;
- refuses a case whose skull or brain was never segmented, and a case whose
  head comes within 2 cm of the scan's bottom slice;
and that a body mask cut by the guard keeps no voxel of a synthetic face, for a
head far above C7, a bowed head with the face level with C7, and a head just
above the margin at the scan's bottom; and that the 2x display upsample keeps
flat regions flat, the mask binary and in place, and adds no dark fringe at
the body's edge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imaging import (  # noqa: E402
    NECK,
    WINDOWS,
    apply_head_guard,
    face_free_top,
    guard_margin,
    quad_corners,
    slice_image,
    to_reference,
    upsample,
)
from structures import HEAD_GUARD  # noqa: E402

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


SHAPE = (32, 32, 120)  # canonical: the third axis runs inferior to superior


def block(k0: int, k1: int, j: tuple[int, int] = (10, 20)) -> np.ndarray:
    """A box over axial slices k0..k1-1; j picks its anterior-posterior range."""
    m = np.zeros(SHAPE, dtype=bool)
    m[10:20, j[0] : j[1], k0:k1] = True
    return m


def empty() -> np.ndarray:
    return np.zeros(SHAPE, dtype=bool)


def masks(**present: np.ndarray) -> dict:
    out = {label: empty() for label in (*HEAD_GUARD, *NECK, "vertebrae_C7")}
    out.update(present)
    return out


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

    margin = guard_margin(1.5)  # the masks below are at 1.5 mm slices
    check(
        (margin, guard_margin(0.8), guard_margin(3.0)) == (14, 25, 7),
        "2 cm is 14 slices at 1.5 mm, 25 at 0.8 mm and 7 at 3 mm",
    )
    check(face_free_top(masks(), margin) is None, "a trunk-only scan is not cropped")
    check(
        face_free_top(masks(vertebrae_C7=block(60, 70)), margin) is None,
        "C7 alone (a scan starting at the neck's base) is not cropped",
    )
    head = masks(skull=block(95, 120), vertebrae_C5=block(75, 80), vertebrae_C7=block(60, 70))
    check(face_free_top(head, margin) == 69, "with the head in view, slices stop at the top of C7")
    bowed = masks(skull=block(62, 72), vertebrae_C6=block(70, 75), vertebrae_C7=block(60, 70))
    check(
        face_free_top(bowed, margin) == 62 - margin,
        "a bowed head's jaw level with C7 is cut 2 cm below the jaw",
    )
    no_c7 = masks(brain=block(100, 120), vertebrae_C6=block(72, 78))
    check(
        face_free_top(no_c7, margin) == 72 - margin,
        "without C7, slices stop 2 cm below the lowest head or neck structure",
    )
    for label in HEAD_GUARD:
        missing = masks()
        missing[label] = None
        try:
            face_free_top(missing, margin)
            check(False, f"a case without a {label} mask is refused")
        except ValueError:
            check(True, f"a case without a {label} mask is refused")

    def survives(face: np.ndarray, guard: dict, spacing: float = 1.5) -> bool:
        body, _ = apply_head_guard(np.ones(SHAPE, dtype=bool), guard, spacing)
        return bool((body & face).any())

    # The face: everything above the neck, here in front of the spine.
    check(not survives(block(100, 120, (20, 30)), head), "no face voxel survives the guard")
    # A bowed head: the face hangs in front of the neck, level with C7.
    bowed_face = block(60, 75, (20, 30))
    bowed = masks(skull=bowed_face, vertebrae_C6=block(70, 75), vertebrae_C7=block(60, 70))
    check(not survives(bowed_face, bowed), "a bowed head's face level with C7 is cut away")
    # The same at 0.8 mm slices: the margin is 25 slices, not 14.
    check(
        apply_head_guard(np.ones(SHAPE, dtype=bool), bowed, 0.8)[1] == 60 - 25,
        "at 0.8 mm the bowed head is cut 25 slices (2 cm) below the jaw",
    )
    # A skull just above the margin at the scan's bottom: only slice 0 is left.
    low = masks(skull=block(margin, 60))
    body, top = apply_head_guard(np.ones(SHAPE, dtype=bool), low, 1.5)
    check(
        top == 0 and not body[:, :, 1:].any() and not survives(block(margin, 60), low),
        "a skull 2 cm above the scan's bottom leaves only the bottom slice",
    )
    # A skull nearer the bottom than the margin: a negative cut is refused, not
    # used as an index (body[..., top + 1 :] would wrap and keep the face).
    for k in (0, 3, margin - 1):
        try:
            apply_head_guard(np.ones(SHAPE, dtype=bool), masks(skull=block(k, 60)), 1.5)
            check(False, f"a skull starting at slice {k} is refused")
        except ValueError:
            check(True, f"a skull starting at slice {k} is refused")

    # The display upsample: twice the size, flat stays flat, no dark fringe at a
    # transparent edge, and the body's edge stays where it was.
    image = np.zeros((20, 30, 4), dtype=np.uint8)
    image[:, 10:, :3] = 180
    image[:, 10:, 3] = 255
    big = upsample(image, 2)
    check(big.shape == (40, 60, 4), "the display image is twice the size")
    check(set(np.unique(big[..., 3])) <= {0, 255}, "the display mask stays binary")
    # Lanczos-3 rings for up to 6 display pixels beside an edge; beyond that, flat.
    check(bool((big[:, 27:, :3] == 180).all()), "a flat region stays flat")
    check(int(big[:, 20:22, :3].min()) >= 170, "no dark fringe where the body meets the air")
    check(bool((big[:, :20, 3] == 0).all() and (big[:, 20:, 3] == 255).all()),
          "the body's edge stays at the same place")  # fmt: skip

    print("imaging check passed" if not failures else f"imaging check failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
