"""Tests for select_cases.py's rules, on a synthetic segmentation: no archive download."""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

import select_cases
from select_cases import Case, check, cut
from structures import ORGANS

Mask = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
Box = tuple[tuple[int, int, int], tuple[int, int, int]]

SHAPE = (100, 60, 150)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])  # 2 mm voxels, already RAS


def body() -> dict[str, Box]:
    """Boxes (low and high voxel corners) for a body that qualifies: organs in the
    belly, short femurs at the bottom, hanging humeri with the scapulae at their
    tops, and ribs 5 to 8."""
    boxes: dict[str, Box] = {}
    for i, s in enumerate(ORGANS):
        boxes[s.label] = ((20 + 5 * i, 20, 40), (24 + 5 * i, 30, 50))
    boxes["femur_left"] = ((60, 25, 2), (66, 31, 40))  # 74 mm tall
    boxes["femur_right"] = ((30, 25, 2), (36, 31, 40))
    boxes["humerus_left"] = ((90, 28, 80), (94, 32, 130))  # 98 mm tall, hanging
    boxes["humerus_right"] = ((6, 28, 80), (10, 32, 130))
    boxes["scapula_left"] = ((82, 40, 125), (90, 50, 140))
    boxes["scapula_right"] = ((10, 40, 125), (18, 50, 140))
    for i in range(5, 9):
        boxes[f"rib_left_{i}"] = ((70, 15, 70 + 6 * i), (85, 45, 72 + 6 * i))
        boxes[f"rib_right_{i}"] = ((15, 15, 70 + 6 * i), (30, 45, 72 + 6 * i))
    return boxes


def to_mask(spec: Box | None) -> Mask:
    mask = np.zeros(SHAPE, bool)
    if spec is not None:
        (a, b, c), (d, e, f) = spec
        mask[a:d, b:e, c:f] = True
    return mask


class FakeCase(Case):
    """A Case whose masks are boxes, not files in an archive."""

    def __init__(self, boxes: dict[str, Box], extra: dict[str, Mask] | None = None) -> None:
        self.boxes, self.extra = boxes, extra or {}

    def mask(self, label: str) -> tuple[npt.NDArray[np.bool_], FloatArray]:
        if label in self.extra:
            return self.extra[label], AFFINE
        return to_mask(self.boxes.get(label)), AFFINE


def test_cut_is_about_the_first_and_last_slice_only() -> None:
    assert not cut(to_mask(((10, 10, 10), (20, 20, 20))))
    assert cut(to_mask(((10, 10, 0), (20, 20, 20))))
    assert cut(to_mask(((10, 10, 10), (20, 20, SHAPE[2]))))
    # Touching the in-plane border is not a cut by the scan's top or bottom.
    assert not cut(to_mask(((0, 0, 10), (SHAPE[0], SHAPE[1], 20))))


def test_a_qualifying_body_gives_its_measures() -> None:
    result = check(FakeCase(body()))
    assert isinstance(result, dict)
    # Ribs span x voxels 15..84: 69 steps of 2 mm.
    assert result["ribcage_width_mm"] == 138
    # Femur tops at k = 39, humerus tops at k = 129: 90 steps of 2 mm.
    assert result["hip_to_shoulder_mm"] == 180
    assert result["skull_voxels"] == 0


def test_a_missing_organ_fails() -> None:
    boxes = body()
    del boxes["spleen"]
    assert check(FakeCase(boxes)) == "no spleen"


def test_an_organ_cut_by_the_scan_fails() -> None:
    boxes = body()
    boxes["liver"] = ((20, 20, 0), (24, 30, 50))
    assert check(FakeCase(boxes)) == "liver cut"


def test_the_head_in_the_scan_fails_but_a_stray_skull_label_does_not() -> None:
    small = to_mask(((40, 20, 140), (49, 30, 150)))  # 9 x 10 x 10 = 900 voxels
    assert isinstance(check(FakeCase(body(), {"skull": small})), dict)
    large = to_mask(((40, 20, 140), (50, 30, 150)))  # 1000 voxels
    assert check(FakeCase(body(), {"skull": large})) == "head in scan"


def test_a_short_femur_fails() -> None:
    boxes = body()
    boxes["femur_right"] = ((30, 25, 10), (36, 31, 40))  # 58 mm
    assert check(FakeCase(boxes)) == "femoral head missing"


def test_a_missing_femur_fails() -> None:
    boxes = body()
    del boxes["femur_left"]
    assert check(FakeCase(boxes)) == "femoral head missing"


def test_a_short_humerus_fails() -> None:
    boxes = body()
    boxes["humerus_left"] = ((90, 28, 100), (94, 32, 130))  # 58 mm
    assert check(FakeCase(boxes)) == "humerus too short"


def test_a_humerus_through_the_top_slice_fails() -> None:
    boxes = body()
    boxes["humerus_right"] = ((6, 28, 100), (10, 32, SHAPE[2]))
    assert check(FakeCase(boxes)) == "humeral head cut"


def test_an_arm_held_out_sideways_fails() -> None:
    boxes = body()
    boxes["humerus_left"] = ((50, 28, 120), (99, 32, 124))  # along x
    assert check(FakeCase(boxes)) == "arm not hanging"


def test_an_arm_raised_above_the_shoulder_fails() -> None:
    # The scapula beside the bottom end makes that end the shoulder, so the elbow is above it.
    boxes = body()
    boxes["scapula_left"] = ((82, 40, 70), (90, 50, 85))
    assert check(FakeCase(boxes)) == "arm raised"


def test_too_few_ribs_fail() -> None:
    boxes = body()
    for i in (5, 6, 7):
        del boxes[f"rib_left_{i}"]
    assert check(FakeCase(boxes)) == "ribs missing"


def test_six_of_the_eight_ribs_are_enough() -> None:
    boxes = body()
    del boxes["rib_left_5"], boxes["rib_right_8"]
    assert isinstance(check(FakeCase(boxes)), dict)


def test_case_reads_masks_from_the_archive_in_canonical_orientation(tmp_path: Path) -> None:
    flipped = np.diag([-2.0, 2.0, 2.0, 1.0])
    mask = np.zeros((10, 8, 6), np.uint8)
    mask[0, 1, 2] = 1  # the first voxel along x is the most right one in RAS
    data = gzip.compress(nib.Nifti1Image(mask, flipped).to_bytes())
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as archive:
        archive.writestr("s0001/segmentations/liver.nii.gz", data)
    with zipfile.ZipFile(tmp_path / "a.zip") as archive:
        case = Case(archive, "s0001")
        canonical, affine = case.mask("liver")
        assert affine[0, 0] > 0
        assert canonical[9, 1, 2] and canonical.sum() == 1
        np.testing.assert_allclose(case.points("liver"), [[0.0, 2.0, 4.0]])


@pytest.mark.parametrize("study", sorted(select_cases.TRUNK_STUDIES))
def test_trunk_studies_are_lower_case_ct_names(study: str) -> None:
    assert study == study.lower()
    assert study.startswith("ct ")
