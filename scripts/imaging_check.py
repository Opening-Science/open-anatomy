"""
Self-check for imaging.py's head guard on synthetic masks: no CT, no download.

  python scripts/imaging_check.py

Checks that face_free_top
- crops nothing when no skull, brain or C1-C6 is in the scan;
- stops at the top of C7 when the scan reaches the head and C7 is there;
- stops 2 cm below the skull when a bowed head brings the jaw level with C7;
- stops 2 cm below the lowest head or neck structure when C7 is not;
- refuses a case whose skull or brain was never segmented;
and that a body mask cut at the guard keeps no voxel of a synthetic face.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imaging import GUARD_MARGIN_SLICES, NECK, face_free_top  # noqa: E402
from structures import HEAD_GUARD  # noqa: E402

SHAPE = (32, 32, 120)  # canonical: the third axis runs inferior to superior


def block(k0: int, k1: int) -> np.ndarray:
    m = np.zeros(SHAPE, dtype=bool)
    m[10:20, 10:20, k0:k1] = True
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

    check(face_free_top(masks()) is None, "a trunk-only scan is not cropped")
    check(
        face_free_top(masks(vertebrae_C7=block(60, 70))) is None,
        "C7 alone (a scan starting at the neck's base) is not cropped",
    )
    head = masks(skull=block(95, 120), vertebrae_C5=block(75, 80), vertebrae_C7=block(60, 70))
    check(face_free_top(head) == 69, "with the head in view, slices stop at the top of C7")
    bowed = masks(skull=block(62, 72), vertebrae_C6=block(70, 75), vertebrae_C7=block(60, 70))
    check(
        face_free_top(bowed) == 62 - GUARD_MARGIN_SLICES,
        "a bowed head's jaw level with C7 is cut 2 cm below the jaw",
    )
    no_c7 = masks(brain=block(100, 120), vertebrae_C6=block(72, 78))
    check(
        face_free_top(no_c7) == 72 - GUARD_MARGIN_SLICES,
        "without C7, slices stop 2 cm below the lowest head or neck structure",
    )
    for label in HEAD_GUARD:
        missing = masks()
        missing[label] = None
        try:
            face_free_top(missing)
            check(False, f"a case without a {label} mask is refused")
        except ValueError:
            check(True, f"a case without a {label} mask is refused")

    body = block(0, 120)
    face = block(100, 120)  # stands in for the face: everything above the neck
    top = face_free_top(head)
    body[:, :, top + 1 :] = False
    check(not (body & face).any(), "no face voxel survives the guard")

    print("imaging check passed" if not failures else f"imaging check failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
