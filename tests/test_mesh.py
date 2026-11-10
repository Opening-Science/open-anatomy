"""Tests for mesh.py's cut and symmetry rules and its volume-keeping surface,
on small synthetic masks: no CT, no download, no GPU."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

import mesh
from structures import ORGANS, REFERENCE, node_name

Mask = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

SPACING = np.array([0.8, 0.8, 1.5])
FLIPPED = np.diag([-SPACING[0], SPACING[1], SPACING[2], 1.0])  # x flipped, as real CTs have
GIRDLE = [f"{b}_{side}" for b in ("clavicula", "scapula", "humerus") for side in ("left", "right")]


def save(path: Path, mask: Mask, affine: FloatArray = FLIPPED) -> Path:
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), path)
    return path


def box(shape: tuple[int, int, int], lo: tuple[int, int, int], hi: tuple[int, int, int]) -> Mask:
    mask = np.zeros(shape, bool)
    mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return mask


def ellipsoid(shape: tuple[int, int, int], radii_mm: FloatArray) -> Mask:
    grid = np.indices(shape).transpose(1, 2, 3, 0) * SPACING
    centre = np.array(shape) * SPACING / 2
    return (((grid - centre) / radii_mm) ** 2).sum(-1) <= 1


# cut_reason


def test_a_missing_mask_was_not_segmented(tmp_path: Path) -> None:
    assert mesh.cut_reason(tmp_path / "gone.nii.gz") == "not segmented"


def test_an_interior_mask_is_whole(tmp_path: Path) -> None:
    path = save(tmp_path / "m.nii.gz", box((20, 20, 20), (5, 5, 5), (15, 15, 15)))
    assert mesh.cut_reason(path) is None


def test_a_mask_one_voxel_from_every_face_is_whole(tmp_path: Path) -> None:
    path = save(tmp_path / "m.nii.gz", box((12, 12, 12), (1, 1, 1), (11, 11, 11)))
    assert mesh.cut_reason(path) is None


def padded_ct(tmp_path: Path, slanted: bool = False) -> Path:
    """A 40-cube CT: noisy air (about -1000 HU) round a body (40 HU), with the
    top 6 slices never scanned (-1024 fill), or a slanted unscanned edge as a
    tilted gantry leaves."""
    rng = np.random.default_rng(0)
    hu = rng.normal(-1000, 8, (40, 40, 40)).round()
    hu[8:32, 8:32, :] = 40
    z = np.arange(40)[None, None, :]
    edge = 34 - (np.arange(40)[None, :, None] // 8 if slanted else 0)
    hu = np.where(z >= edge, -1024, hu)
    return save_ct(tmp_path / "ct.nii.gz", hu)


def save_ct(path: Path, hu: FloatArray) -> Path:
    nib.save(nib.Nifti1Image(hu.astype(np.int16), FLIPPED), path)
    return path


@pytest.mark.parametrize("slanted", [False, True])
def test_a_mask_meeting_the_unscanned_padding_is_cut(tmp_path: Path, slanted: bool) -> None:
    outside = mesh.scan_outside(padded_ct(tmp_path, slanted))
    mask = box((40, 40, 40), (12, 12, 20), (20, 20, 34 - (2 if slanted else 0)))
    path = save(tmp_path / "m.nii.gz", mask)
    assert mesh.cut_reason(path) is None  # no array face: the old test called it whole
    assert mesh.cut_reason(path, outside) == "cut by the scan's edge (unscanned padding)"
    assert mesh.cut_faces(path, outside) == {"S"}


@pytest.mark.parametrize(("top", "cut"), [(32, True), (31, True), (30, False)])
def test_the_blend_at_the_fills_edge_counts_as_padding(tmp_path: Path, top: int, cut: bool) -> None:
    # The fill starts at slice 34. Resampling blends it into the slices below,
    # so a mask within PADDING_BLEND voxels of it, plus the one the cut test
    # reaches, is cut; one with a clear slice beyond that is whole.
    outside = mesh.scan_outside(padded_ct(tmp_path))
    mask = box((40, 40, 40), (12, 12, 20), (20, 20, top + 1))
    reason = mesh.cut_reason(save(tmp_path / "m.nii.gz", mask), outside)
    assert (reason == "cut by the scan's edge (unscanned padding)") is cut


def test_a_mask_ending_where_the_fill_blends_in_is_cut(tmp_path: Path) -> None:
    # As a resampled scan ends: the last scanned slice, two slices blending
    # the body into the fill (about -450 and -995 HU, neither at the fill
    # value), then the fill. A humeral head ending on the last scanned slice
    # was cut by the scan's end; one with a clear slice below it is whole.
    hu = np.full((40, 40, 40), 40.0)
    hu[:, :, 30], hu[:, :, 31], hu[:, :, 32:] = -450, -995, -1024
    outside = mesh.scan_outside(save_ct(tmp_path / "ct.nii.gz", hu))
    assert not outside[0][:, :, :30].any() and outside[0][:, :, 30:].all()
    head = save(tmp_path / "head.nii.gz", box((40, 40, 40), (12, 12, 10), (20, 20, 30)))
    assert mesh.cut_reason(head, outside) == "cut by the scan's edge (unscanned padding)"
    assert mesh.cut_faces(head, outside) == {"S"}
    whole = save(tmp_path / "whole.nii.gz", box((40, 40, 40), (12, 12, 10), (20, 20, 29)))
    assert mesh.cut_reason(whole, outside) is None


def test_air_round_the_body_is_not_padding(tmp_path: Path) -> None:
    outside, _ = mesh.scan_outside(padded_ct(tmp_path))
    assert not outside[:, :, :30].any()  # noisy air below the fill value's band stays scanned
    assert outside[:, :, 34:].all()


def test_the_thin_end_of_a_gantry_wedge_is_outside(tmp_path: Path) -> None:
    hu = np.full((40, 40, 40), 40.0)
    hu[:, :, 36:] = -1024  # a thick band of padding at the top
    for y in range(40):  # and a wedge from it, one or two voxels thin at y 30 and up
        hu[:, y, 36 - (1 if y >= 30 else 3) : 36] = -1024
    outside, _ = mesh.scan_outside(save_ct(tmp_path / "ct.nii.gz", hu))
    assert outside[20, 35, 35]  # a one-voxel-thin part of the wedge
    mask = box((40, 40, 40), (12, 30, 20), (20, 38, 35))  # reaches the wedge's thin end
    edge = (outside, FLIPPED)
    assert mesh.cut_reason(save(tmp_path / "m.nii.gz", mask), edge) is not None


def test_a_mask_on_another_grid_than_the_ct_raises(tmp_path: Path) -> None:
    edge = mesh.scan_outside(padded_ct(tmp_path))
    mask = box((40, 40, 40), (12, 12, 12), (20, 20, 20))
    shifted = FLIPPED.copy()
    shifted[2, 3] += 1.5
    with pytest.raises(ValueError, match="CT's grid"):
        mesh.cut_reason(save(tmp_path / "m.nii.gz", mask, shifted), edge)
    small = save(tmp_path / "s.nii.gz", mask[:30])
    with pytest.raises(ValueError, match="CT's grid"):
        mesh.cut_reason(small, edge)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("end", ["first", "last"])
def test_a_mask_touching_any_face_is_cut(tmp_path: Path, axis: int, end: str) -> None:
    lo, hi = [5, 5, 5], [15, 15, 15]
    if end == "first":
        lo[axis] = 0
    else:
        hi[axis] = 20
    path = save(
        tmp_path / "m.nii.gz", box((20, 20, 20), (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))
    )
    assert mesh.cut_reason(path) == f"cut by the scan's edge (axis {axis})"


VOXEL_ML = float(SPACING.prod()) / 1000  # 0.96 mm3


def test_under_min_ml_is_a_stray_label(tmp_path: Path) -> None:
    mask = box((40, 40, 40), (5, 5, 5), (17, 30, 6))  # 12 x 25 x 1 = 300 voxels, 0.288 ml
    assert mask.sum() * VOXEL_ML < mesh.MIN_ML
    path = save(tmp_path / "m.nii.gz", mask)
    assert mesh.cut_reason(path) == "absent or a stray label"


def test_at_least_min_ml_counts_as_a_structure(tmp_path: Path) -> None:
    mask = box((40, 40, 40), (5, 5, 5), (18, 30, 6))  # 13 x 25 x 1 = 325 voxels, 0.312 ml
    assert mask.sum() * VOXEL_ML >= mesh.MIN_ML
    assert mesh.cut_reason(save(tmp_path / "m.nii.gz", mask)) is None


def test_the_size_limit_is_a_volume_not_a_voxel_count(tmp_path: Path) -> None:
    # The same 200 voxels are 0.19 ml at 0.8 x 0.8 x 1.5 mm but 0.68 ml at 1.5 mm.
    mask = box((40, 40, 40), (5, 5, 5), (15, 25, 6))
    assert mesh.cut_reason(save(tmp_path / "fine.nii.gz", mask)) == "absent or a stray label"
    coarse = np.diag([-1.5, 1.5, 1.5, 1.0])
    assert mesh.cut_reason(save(tmp_path / "coarse.nii.gz", mask, coarse)) is None


def test_an_empty_mask_is_a_stray_label_not_a_crash(tmp_path: Path) -> None:
    path = save(tmp_path / "m.nii.gz", np.zeros((8, 8, 8), bool))
    assert mesh.cut_reason(path) == "absent or a stray label"


# The reference set's girdle and rib rules live inline in main(). They are run
# through main() with cut_reason and the surface builders replaced by stubs, so
# only the rules and the manifest they produce are exercised.

TETRA = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float64)
TETRA_TRIANGLES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
TETRA_ML = 1000 / 6 / 1000


def run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reasons: dict[str, str],
    which: str = "reference",
    faces: set[str] | None = None,
    shares: dict[str, float] | None = None,
) -> dict[str, Any]:
    surfaced: list[str] = []
    shares = shares or {}

    def fake_reason(path: Path, outside: object = None) -> str | None:
        return reasons.get(path.name.removesuffix(".nii.gz"))

    def fake_surface(path: Path, max_triangles: int, **_: object) -> tuple[Any, Any, float, float]:
        label = path.name.removesuffix(".nii.gz")
        surfaced.append(label)
        return TETRA + len(surfaced), TETRA_TRIANGLES.copy(), TETRA_ML, shares.get(label, 0.0)

    def fake_cartilage(seg_dir: Path, kept_ribs: list[str]) -> tuple[Any, Any, Any]:
        return TETRA.copy(), TETRA_TRIANGLES.copy(), []

    def fake_faces(path: Path, outside: object = None) -> set[str]:
        return {"I"} if faces is None else faces

    monkeypatch.setattr(mesh, "cut_reason", fake_reason)
    monkeypatch.setattr(mesh, "surface", fake_surface)
    monkeypatch.setattr(mesh, "modelled_cartilage", fake_cartilage)
    monkeypatch.setattr(mesh, "cut_faces", fake_faces)
    seg = tmp_path / "seg"
    seg.mkdir(exist_ok=True)
    argv = ["mesh.py", "--case", "c", "--synthetic", "--seg-dir", str(seg),
            "--out-dir", str(tmp_path)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", [*argv, "--set", which])
    mesh.main()
    manifest: dict[str, Any] = json.loads((tmp_path / f"c.{which}.manifest.json").read_text())
    manifest["surfaced"] = surfaced
    return manifest


def dropped(manifest: dict[str, Any]) -> dict[str, str]:
    return {d["label"]: d["reason"] for d in manifest["dropped"]}


def test_a_whole_reference_body_keeps_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = run_main(tmp_path, monkeypatch, {})
    assert manifest["dropped"] == []
    assert [n["name"] for n in manifest["nodes"]] == [node_name(s.term) for s in REFERENCE]
    assert not (tmp_path / "c.landmarks.glb").exists()
    cartilage = next(n for n in manifest["nodes"] if n["label"] == "set of costal cartilages")
    assert cartilage["modelled"] is True
    assert cartilage["volume_drift"] is None


@pytest.mark.parametrize("bone", GIRDLE)
def test_one_cut_girdle_bone_drops_both_girdles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bone: str
) -> None:
    manifest = run_main(tmp_path, monkeypatch, {bone: "cut by the scan's edge (axis 2)"})
    reasons = dropped(manifest)
    assert set(reasons) == set(GIRDLE)
    assert reasons[bone] == "cut by the scan's edge (axis 2)"
    for other in set(GIRDLE) - {bone}:
        assert reasons[other] == f"shoulder girdle dropped: {bone} cut"
    kept = {n["label"] for n in manifest["nodes"]}
    assert not kept & {s.name for s in REFERENCE if s.label in GIRDLE}


def test_the_girdle_reason_names_every_cut_bone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = {
        "clavicula_right": "cut by the scan's edge (axis 0)",
        "humerus_left": "absent or a stray label",
    }
    reasons = dropped(run_main(tmp_path, monkeypatch, cut))
    assert reasons["scapula_left"] == "shoulder girdle dropped: clavicula_right, humerus_left cut"
    assert reasons["clavicula_right"] == cut["clavicula_right"]
    assert reasons["humerus_left"] == cut["humerus_left"]


def test_cut_humeri_are_still_meshed_as_landmarks_but_absent_ones_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = {"scapula_left": "cut by the scan's edge (axis 1)", "humerus_right": "not segmented"}
    manifest = run_main(tmp_path, monkeypatch, cut)
    assert (tmp_path / "c.landmarks.glb").exists()
    # humerus_left was dropped only for symmetry, so its joint end is still measured.
    assert "humerus_left" in manifest["surfaced"]
    assert "humerus_right" not in manifest["surfaced"]


def test_a_cut_femur_is_dropped_without_touching_the_girdle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = run_main(tmp_path, monkeypatch, {"femur_left": "cut by the scan's edge (axis 2)"})
    assert dropped(manifest) == {"femur_left": "cut by the scan's edge (axis 2)"}
    assert (tmp_path / "c.landmarks.glb").exists()


def test_a_humerus_cut_at_its_head_is_not_a_landmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = {"humerus_left": "cut by the scan's edge (axis 2)"}
    manifest = run_main(tmp_path, monkeypatch, cut, faces={"S"})
    assert "humerus_left" not in manifest["surfaced"]
    note = next(d for d in manifest["dropped"] if d["label"] == "humerus_left")
    assert note["landmark"] == "not used: cut at S"
    # Its partner was dropped only for symmetry, so it is whole and still measured.
    assert "humerus_right" in manifest["surfaced"]


def test_a_cut_bone_says_which_side_the_scan_cut_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = {"clavicula_left": "cut by the scan's edge (unscanned padding)"}
    notes = {d["label"]: d for d in run_main(tmp_path, monkeypatch, cut, faces={"S"})["dropped"]}
    assert notes["clavicula_left"]["cut_at"] == "S"
    assert "cut_at" not in notes["scapula_right"]  # dropped by the girdle rule, not cut
    assert notes["scapula_right"]["reason"] == "shoulder girdle dropped: clavicula_left cut"


def test_bones_the_scan_cut_are_offered_as_neighbours_for_loose_pieces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, list[str]] = {}

    def fake_settle(
        seg_dir: Path,
        labels: list[str],
        left_out: list[str] | None = None,
        given: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Any]]:
        seen["labels"], seen["left_out"] = labels, left_out or []
        return {}, []

    monkeypatch.setattr(mesh, "settle_fragments", fake_settle)
    cut = {
        "vertebrae_T1": "cut by the scan's edge (unscanned padding)",
        "rib_left_4": "cut by the scan's edge (axis 0)",
        "vertebrae_C3": "absent or a stray label",
    }
    run_main(tmp_path, monkeypatch, cut)
    assert {"vertebrae_T1", "rib_left_4", "rib_right_4"} <= set(seen["left_out"])
    assert "vertebrae_C3" not in seen["left_out"]  # not in the scan: no neighbour
    assert not set(seen["left_out"]) & set(seen["labels"])
    assert "vertebrae_T2" in seen["labels"]


def test_a_bone_cut_only_by_a_neighbours_lump_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # T1 and C7 both meet the scan's end, T1 only with a lump of C7 on a thin
    # neck. T1 stays in the body, meshed from its mask without the lump; C7 is
    # left out; the manifest notes the lump.
    body = box((8, 8, 8), (2, 2, 2), (6, 6, 6))
    note = {"label": "vertebrae_T1", "voxels": 9, "ml": 0.03, "action": "moved",
            "to": "vertebrae_C7", "neck": True, "exported": False}  # fmt: skip
    seen: dict[str, Any] = {}

    def fake_lumps(
        seg_dir: Path, reasons: dict[str, str | None], outside: object = None
    ) -> tuple[dict[str, Any], list[Any]]:
        seen["cut"] = sorted(label for label, reason in reasons.items() if reason)
        return {"vertebrae_T1": body}, [note]

    def fake_settle(
        seg_dir: Path,
        labels: list[str],
        left_out: list[str] | None = None,
        given: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Any]]:
        seen["labels"], seen["left_out"], seen["given"] = labels, left_out, given
        return dict(given or {}), []

    monkeypatch.setattr(mesh, "whole_but_for_lumps", fake_lumps)
    monkeypatch.setattr(mesh, "settle_fragments", fake_settle)
    cut = dict.fromkeys(
        ("vertebrae_C7", "vertebrae_T1"), "cut by the scan's edge (unscanned padding)"
    )
    manifest = run_main(tmp_path, monkeypatch, cut)
    assert seen["cut"] == ["vertebrae_C7", "vertebrae_T1"]
    assert set(dropped(manifest)) == {"vertebrae_C7"}
    assert "vertebrae_T1" in manifest["surfaced"] and "vertebrae_C7" not in manifest["surfaced"]
    assert "vertebrae_T1" in seen["labels"] and "vertebrae_C7" in seen["left_out"]
    assert seen["given"]["vertebrae_T1"] is body
    assert manifest["fragments"] == [note]


@pytest.mark.parametrize(
    ("label", "reason", "faces", "ok"),
    [
        ("humerus_left", "cut by the scan's edge (axis 2)", {"I"}, True),
        ("humerus_left", "cut by the scan's edge (axis 2)", {"S"}, False),
        ("femur_right", "cut by the scan's edge (axis 0)", {"I", "R"}, False),
        ("humerus_right", "shoulder girdle dropped: scapula_left cut", set(), True),
        ("femur_left", "absent or a stray label", set(), False),
        ("rib_left_1", "cut by the scan's edge (axis 2)", {"I"}, False),
    ],
)
def test_landmark_ok(label: str, reason: str, faces: set[str], ok: bool) -> None:
    assert mesh.landmark_ok(label, reason, faces) is ok


def test_reference_rules_leave_the_input_alone() -> None:
    reasons: dict[str, str | None] = {"rib_left_4": "cut by the scan's edge (axis 0)"}
    ruled = mesh.reference_rules(reasons)
    assert ruled["rib_right_4"] == "its pair rib_left_4 is missing"
    assert reasons == {"rib_left_4": "cut by the scan's edge (axis 0)"}


@pytest.mark.parametrize(
    ("index", "expected"),
    [((0, 10), "I"), ((-1, 10), "S"), ((10, 0), "R"), ((10, -1), "L")],
)
def test_cut_faces_names_the_side_in_ras(
    tmp_path: Path, index: tuple[int, int], expected: str
) -> None:
    # FLIPPED runs x towards the right as the index falls, as real CTs do.
    mask = box((20, 20, 20), (5, 5, 5), (15, 15, 15))
    axis_k, z = index
    if index[0] in (0, -1):
        mask[10, 10, axis_k] = True  # reaches the bottom or top slice
    else:
        mask[z, 10, 10] = True  # reaches the low or high x index
    assert mesh.cut_faces(save(tmp_path / "m.nii.gz", mask)) == {expected}


def capped_ct(tmp_path: Path, shape: tuple[int, int, int] = (60, 40, 40)) -> Path:
    """A CT that was never scanned above slice 34 or below slice 4: with the
    blend, the padding starts at slice 32 and ends at slice 5."""
    hu = np.full(shape, 40.0)
    hu[:, :, 34:], hu[:, :, :4] = -1024, -1024
    return save_ct(tmp_path / "ct.nii.gz", hu)


def test_a_bone_lying_across_the_body_is_cut_where_the_padding_is(tmp_path: Path) -> None:
    # A clavicle runs across the body and its outer end rises to the scan's
    # last slice. The padding lies above it, so it is cut at S, although the
    # contact is far to the side of the bone's middle and only a little above.
    outside = mesh.scan_outside(capped_ct(tmp_path))
    clavicle = box((60, 40, 40), (5, 15, 24), (50, 20, 28))
    clavicle |= box((60, 40, 40), (44, 15, 24), (50, 20, 32))
    path = save(tmp_path / "m.nii.gz", clavicle)
    assert mesh.cut_reason(path, outside) == "cut by the scan's edge (unscanned padding)"
    assert mesh.cut_faces(path, outside) == {"S"}


def test_a_bone_cut_by_padding_at_both_ends_names_both(tmp_path: Path) -> None:
    # A humerus running out of the scan at both ends: its head is cut, so it
    # cannot give a joint centre, whichever end meets more padding.
    outside = mesh.scan_outside(capped_ct(tmp_path))
    shaft = box((60, 40, 40), (20, 16, 6), (24, 20, 32))
    shaft |= box((60, 40, 40), (16, 12, 26), (28, 24, 32))  # the head, wider than the shaft
    path = save(tmp_path / "m.nii.gz", shaft)
    assert mesh.cut_faces(path, outside) == {"S", "I"}
    assert not mesh.landmark_ok("humerus_left", "cut by the scan's edge (unscanned padding)",
                                mesh.cut_faces(path, outside))  # fmt: skip


@pytest.mark.parametrize(("top", "bottom", "expected"), [(34, 10, {"S"}), (20, 3, {"I"})])
def test_a_mask_reaching_into_the_padding_is_cut_on_that_side(
    tmp_path: Path, top: int, bottom: int, expected: set[str]
) -> None:
    # The segmentation may label a slice or two of the blend. The side is
    # still where the scanned part of the mask goes over into the padding.
    outside = mesh.scan_outside(capped_ct(tmp_path))
    mask = box((60, 40, 40), (20, 16, bottom), (23, 19, top))  # thin, like a rib's tip
    assert mesh.cut_faces(save(tmp_path / "m.nii.gz", mask), outside) == expected


def test_padding_at_the_side_is_named_in_ras(tmp_path: Path) -> None:
    # FLIPPED runs x towards the right as the index falls: padding at low x
    # indices (a field of view narrower than the grid) cuts a bone at R.
    hu = np.full((40, 40, 40), 40.0)
    hu[:6] = -1024
    outside = mesh.scan_outside(save_ct(tmp_path / "ct.nii.gz", hu))
    mask = box((40, 40, 40), (8, 12, 12), (20, 20, 20))
    assert mesh.cut_faces(save(tmp_path / "m.nii.gz", mask), outside) == {"R"}
    clear = box((40, 40, 40), (10, 12, 12), (20, 20, 20))
    assert mesh.cut_reason(save(tmp_path / "c.nii.gz", clear), outside) is None


def test_trim_knobs_takes_off_a_lump_on_a_thin_neck_and_keeps_thin_rib() -> None:
    mask = np.zeros((40, 24, 20), bool)
    mask[2:38, 8:12, 8:12] = True  # the rib, 4 voxels across
    mask[20, 12:15, 10] = True  # a neck one voxel thin, three long
    mask[19:22, 15:18, 9:12] = True  # a 27-voxel lump beyond it
    mask[26:38, 4:8, 9:11] = True  # a flange 2 voxels thin, 96 voxels: part of the rib
    trimmed, ml = mesh.trim_knobs(mask, voxel_ml=0.02)  # lump 0.54 ml, flange over 1 ml
    assert not trimmed[20, 16, 10]
    assert trimmed[25, 10, 10] and trimmed[34, 5, 10]
    assert ml == pytest.approx(0.02 * (mask.sum() - trimmed.sum()))
    assert mask.sum() - trimmed.sum() >= 27


@pytest.mark.parametrize(
    ("cut", "partner"), [("rib_left_3", "rib_right_3"), ("rib_right_11", "rib_left_11")]
)
def test_a_rib_whose_pair_is_cut_is_dropped_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cut: str, partner: str
) -> None:
    reasons = dropped(run_main(tmp_path, monkeypatch, {cut: "absent or a stray label"}))
    assert reasons == {cut: "absent or a stray label", partner: f"its pair {cut} is missing"}


def test_when_both_ribs_of_a_pair_are_cut_each_keeps_its_own_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = {"rib_left_12": "cut by the scan's edge (axis 2)", "rib_right_12": "not segmented"}
    assert dropped(run_main(tmp_path, monkeypatch, cut)) == cut


def test_the_organ_set_ignores_cuts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = run_main(
        tmp_path, monkeypatch, {"liver": "cut by the scan's edge (axis 2)"}, "organs"
    )
    assert manifest["dropped"] == []
    assert [n["name"] for n in manifest["nodes"]] == [node_name(s.term) for s in ORGANS]


def test_a_split_organ_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="aorta: 30.0% of its volume is outside"):
        run_main(tmp_path, monkeypatch, {}, shares={"aorta": 0.3})


def test_a_split_bone_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="vertebrae_L2: 60.0% .* a split structure stops"):
        run_main(tmp_path, monkeypatch, {}, shares={"vertebrae_L2": 0.6})


def two_piece_rib(gap_voxels: int, sideways: bool = False) -> Mask:
    """A straight rib 4 voxels across along x, in two pieces: the second in line
    beyond a gap, or beside the first."""
    mask = np.zeros((60, 30, 30), bool)
    mask[5:30, 13:17, 13:17] = True
    if sideways:
        mask[10:25, 13 + 4 + gap_voxels : 17 + 4 + gap_voxels, 13:17] = True
    else:
        mask[30 + gap_voxels : 55, 13:17, 13:17] = True
    return mask


def test_bridge_gap_joins_two_pieces_of_one_bone_in_line() -> None:
    mask = two_piece_rib(gap_voxels=5)  # 5 voxels at 0.8 mm: a 4 mm gap
    bridged, gap = mesh.bridge_gap(mask, SPACING)
    assert gap == pytest.approx(6 * SPACING[0], abs=0.1)  # voxel centres 6 apart
    _, count = mesh.ndimage.label(bridged, structure=np.ones((3, 3, 3)))
    assert count == 1
    assert bridged[32, 15, 15] and not bridged[32, 25, 15]  # filled along the bone, not elsewhere


@pytest.mark.parametrize("gap_voxels, sideways", [(30, False), (3, True)])
def test_bridge_gap_leaves_pieces_that_are_not_one_bone(gap_voxels: int, sideways: bool) -> None:
    mask = two_piece_rib(gap_voxels, sideways)  # too far, or beside the bone
    bridged, gap = mesh.bridge_gap(mask, SPACING)
    assert gap is None
    assert (bridged == mask).all()


def test_a_bridged_rib_is_whole_and_its_gap_is_recorded(tmp_path: Path) -> None:
    mask = two_piece_rib(gap_voxels=5)
    spacing = SPACING
    bridged, gap = mesh.bridge_gap(mask, spacing)
    path = save(tmp_path / "rib.nii.gz", mask)
    _, _, _, share_before = mesh.surface(path, 5000, centreline=False)
    _, _, _, share_after = mesh.surface(path, 5000, centreline=False, mask=bridged)
    assert share_before > mesh.MAX_DROPPED_SHARE
    assert share_after == 0.0
    assert gap is not None and gap > 0


def test_a_cut_organ_stops_the_reference_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="liver: cut by the scan"):
        run_main(tmp_path, monkeypatch, {"liver": "cut by the scan's edge (axis 2)"})


def test_a_split_organ_mask_has_its_dropped_share_measured(tmp_path: Path) -> None:
    # An ellipsoid cut in two by a 2-slice gap: the smaller half is dropped and counted.
    shape = (40, 40, 40)
    organ = ellipsoid(shape, np.array([10.0, 8.0, 20.0]))
    organ[:, :, 22:24] = False
    path = save(tmp_path / "split.nii.gz", organ)
    _, _, _, share = mesh.surface(path, 40000)
    labels, _ = mesh.ndimage.label(organ, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    assert share == pytest.approx(1 - sizes.max() / sizes.sum())
    assert share > mesh.MAX_DROPPED_SHARE


# The volume-keeping surface


def closed_volume_ml(vertices: FloatArray, triangles: npt.NDArray[np.uint32]) -> float:
    triangles, _ = mesh.outward(vertices, triangles)
    assert mesh.open_edges(triangles) == 0
    return abs(mesh.signed_volume(vertices, triangles)) / 1000


def test_an_ellipsoid_keeps_its_voxel_volume(tmp_path: Path) -> None:
    mask = ellipsoid((40, 40, 24), np.array([10.0, 7.0, 12.0]))
    path = save(tmp_path / "e.nii.gz", mask)
    vertices, triangles, voxel_ml, _ = mesh.surface(path, 40000)
    assert voxel_ml == pytest.approx(mask.sum() * SPACING.prod() / 1000)
    assert closed_volume_ml(vertices, triangles) == pytest.approx(
        voxel_ml, rel=mesh.MAX_VOLUME_DRIFT
    )
    # The exact ellipsoid is close too, though the voxels are what is kept.
    assert voxel_ml == pytest.approx(4 / 3 * math.pi * 10 * 7 * 12 / 1000, rel=0.05)


def test_a_thin_plate_keeps_its_voxel_volume(tmp_path: Path) -> None:
    # Two voxels thick: a fixed 0.5 contour of the blurred mask would shrink it a lot.
    mask = box((30, 30, 16), (5, 5, 7), (25, 25, 9))
    path = save(tmp_path / "plate.nii.gz", mask)
    vertices, triangles, voxel_ml, _ = mesh.surface(path, 40000, along_slices=True)
    assert closed_volume_ml(vertices, triangles) == pytest.approx(
        voxel_ml, rel=mesh.MAX_VOLUME_DRIFT
    )


def test_a_bodys_wall_fluted_the_same_in_every_slice_comes_out_round(tmp_path: Path) -> None:
    # A vertebral body as a 15 mm cylinder whose wall steps in and out by 0.8 mm
    # every 3 mm round it, the same in every slice, as voxel steps are.
    grid = np.indices((64, 64, 28)).transpose(1, 2, 3, 0) * SPACING
    x, y, z = grid[..., 0] - 25.3, grid[..., 1] - 25.0, grid[..., 2] - 20.0
    radius = 15 + 0.8 * np.sign(np.sin(30 * np.arctan2(y, x)))
    path = save(tmp_path / "vertebrae_T9.nii.gz", (np.hypot(x, y) <= radius) & (np.abs(z) <= 12))
    vertices, triangles, voxel_ml, _ = mesh.surface(path, 10**9, along_slices=True)
    assert closed_volume_ml(vertices, triangles) == pytest.approx(voxel_ml, rel=0.01)
    wall = np.abs(vertices[:, 2] - 20.0) < 2
    around = np.hypot(-vertices[wall, 0] - 25.3, vertices[wall, 1] - 25.0)
    assert around.mean() == pytest.approx(15.0, abs=0.3)
    assert around.std() < 0.3  # the steps themselves have a spread of 0.8 mm


def tube(line_mm: FloatArray, shape: tuple[int, int, int], radius: float, slab: int = 1) -> Mask:
    """The voxels within radius of a line; with slab > 1 every slab of that many
    slices repeats its middle slice, as a segmentation of thick slices does."""
    from scipy.spatial import KDTree

    grid = np.indices(shape).transpose(1, 2, 3, 0).astype(float)
    grid[..., 2] = (grid[..., 2] // slab) * slab + (slab - 1) / 2
    distance = KDTree(line_mm).query((grid * SPACING).reshape(-1, 3))[0]
    return distance.reshape(shape) <= radius


def climbing_arc() -> FloatArray:
    """A quarter circle of 70 mm radius that climbs 70 mm through the slices, in mm."""
    angle = np.linspace(0, np.pi / 2, 400)
    return np.stack(
        [8 + 70 * np.sin(angle), 78 - 70 * np.cos(angle), 10 + 70 * angle / (np.pi / 2)], 1
    )


def distance_to_line(vertices: FloatArray, line_mm: FloatArray) -> tuple[FloatArray, Any]:
    """Each RAS vertex's distance to a line given in voxel millimetres, and the
    nearest line point's index (FLIPPED turns x round)."""
    from scipy.spatial import KDTree

    return KDTree(line_mm * np.array([-1.0, 1.0, 1.0])).query(vertices)


def test_a_stepped_rib_comes_out_smooth_along_its_length(tmp_path: Path) -> None:
    # A 4 mm tube stepped in slabs of three slices: along its middle the surface
    # should be a 4 mm tube again. Blurring along the slices leaves the steps.
    line = climbing_arc()
    path = save(tmp_path / "rib_left_5.nii.gz", tube(line, (110, 110, 60), 4.0, slab=3))
    spread = {}
    for centreline in (False, True):
        vertices, triangles, voxel_ml, _ = mesh.surface(
            path, 40000, along_slices=True, centreline=centreline
        )
        assert closed_volume_ml(vertices, triangles) == pytest.approx(voxel_ml, rel=0.01)
        radius, nearest = distance_to_line(vertices, line)
        middle = (nearest > 60) & (nearest < 340)
        spread[centreline] = float(radius[middle].std())
        if centreline:
            assert radius[middle].mean() == pytest.approx(4.0, abs=0.2)
            assert radius.max() < 6.0  # nothing strays from the tube, ends included
    assert spread[True] < 0.2
    assert spread[True] < spread[False] / 3


def test_a_rib_cut_flat_ends_in_a_rounded_tip(tmp_path: Path) -> None:
    bar = box((90, 30, 20), (10, 10, 7), (80, 20, 13))  # 56 x 8 x 9 mm, flat ends
    path = save(tmp_path / "rib_left_5.nii.gz", bar)
    share = {}
    for centreline in (False, True):
        vertices, triangles, voxel_ml, _ = mesh.surface(
            path, 40000, along_slices=True, centreline=centreline
        )
        assert closed_volume_ml(vertices, triangles) == pytest.approx(voxel_ml, rel=0.01)
        along = -vertices[:, 0]  # mm along the bar
        tip = along > along.max() - 0.5
        middle = np.abs(along - along.mean()) < 5
        share[centreline] = float(np.ptp(vertices[tip, 1]) / np.ptp(vertices[middle, 1]))
        assert abs(along.max() - 63.6) < 1.5  # the end stays where the mask ends
    assert share[False] > 0.8  # a cut face, nearly as wide as the bar
    assert share[True] < 0.75


def test_a_bone_too_short_for_a_centreline_is_surfaced_as_before(tmp_path: Path) -> None:
    blob = box((20, 20, 12), (6, 6, 4), (14, 14, 8))
    assert mesh.unroll(np.pad(blob, mesh.PAD).astype(np.float32), SPACING) is None
    path = save(tmp_path / "rib_left_12.nii.gz", blob)
    plain = mesh.surface(path, 40000, along_slices=True)
    followed = mesh.surface(path, 40000, along_slices=True, centreline=True)
    np.testing.assert_array_equal(plain[0], followed[0])
    np.testing.assert_array_equal(plain[1], followed[1])


def test_a_bone_that_doubles_back_is_not_unrolled() -> None:
    # A hairpin with arms 9 mm apart: the smoothed centreline cuts the bend, so
    # the unrolled volume would miss part of the bone.
    arm, turn = np.linspace(0, 50, 100), np.linspace(0, np.pi, 60)
    line = np.vstack([
        np.stack([10 + arm, np.full(100, 20.0), np.full(100, 15.0)], 1),
        np.stack([60 + 4.5 * np.sin(turn), 24.5 - 4.5 * np.cos(turn), np.full(60, 15.0)], 1),
        np.stack([60 - arm, np.full(100, 29.0), np.full(100, 15.0)], 1),
    ])  # fmt: skip
    hairpin = np.pad(tube(line, (100, 65, 20), 3.0), mesh.PAD).astype(np.float32)
    assert mesh.unroll(hairpin, SPACING) is None


def test_unrolling_keeps_the_rib_and_maps_it_back(tmp_path: Path) -> None:
    line = climbing_arc()
    mask = tube(line, (110, 110, 60), 4.0)
    unrolled = mesh.unroll(np.pad(mask, mesh.PAD).astype(np.float32), SPACING)
    assert unrolled is not None
    straight, step, _ = unrolled
    assert step == pytest.approx(SPACING.min() / mesh.UPSAMPLE)
    assert straight.sum() * step**3 == pytest.approx(mask.sum() * SPACING.prod(), rel=0.03)
    # Along the rib the unrolled tube is the same everywhere: a straight bar.
    area = straight[100:-100].sum(axis=(1, 2)) * step**2
    assert area.std() / area.mean() < 0.05


def test_rib_axis_follows_the_bone_with_a_frame_that_does_not_twist() -> None:
    from scipy.spatial import KDTree

    line = climbing_arc()
    voxels_mm = np.argwhere(tube(line, (110, 110, 60), 4.0)) * SPACING
    axis = mesh.rib_axis(voxels_mm, SPACING, 0.4)
    assert axis is not None
    points, normals, binormals = axis
    inside = (points[:, 2] > line[40, 2]) & (points[:, 2] < line[-40, 2])
    assert KDTree(line).query(points[inside])[0].max() < 1.0
    tangents = np.cross(normals, binormals)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.einsum("ij,ij->i", normals, binormals), 0.0, atol=1e-6)
    assert np.einsum("ij,ij->i", normals[1:], normals[:-1]).min() > 0.999
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    np.testing.assert_allclose(steps, 0.4, atol=0.02)


def test_the_same_rib_gives_the_same_surface(tmp_path: Path) -> None:
    path = save(tmp_path / "rib_left_5.nii.gz", tube(climbing_arc(), (110, 110, 60), 4.0))
    a = mesh.surface(path, 5000, along_slices=True, centreline=True)
    b = mesh.surface(path, 5000, along_slices=True, centreline=True)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_the_surface_lies_where_the_affine_puts_the_mask(tmp_path: Path) -> None:
    mask = ellipsoid((40, 40, 24), np.array([10.0, 7.0, 12.0]))
    vertices, _, _, _ = mesh.surface(save(tmp_path / "e.nii.gz", mask), 40000)
    ijk_centre = np.argwhere(mask).mean(0)
    ras_centre = FLIPPED[:3, :3] @ ijk_centre + FLIPPED[:3, 3]
    np.testing.assert_allclose((vertices.min(0) + vertices.max(0)) / 2, ras_centre, atol=1.0)


def test_smoothest_bisects_to_the_target_volume() -> None:
    crop = np.pad(ellipsoid((30, 30, 20), np.array([8.0, 6.0, 10.0])), 2).astype(np.float32)
    zoomed = mesh.ndimage.zoom(crop, mesh.UPSAMPLE, order=1, grid_mode=True, mode="nearest")
    fine = np.asarray(zoomed, dtype=np.float32)
    voxel_mm3 = float(crop.sum() * SPACING.prod())
    for share in (0.9, 1.0, 1.1):
        poly = mesh.smoothest(fine, SPACING, 1.0, share * voxel_mm3, False)
        assert mesh.enclosed_volume(poly) == pytest.approx(share * voxel_mm3, rel=0.01)


def test_only_the_largest_piece_counts_for_a_single_structure(tmp_path: Path) -> None:
    big = box((40, 30, 20), (4, 5, 5), (20, 25, 15))
    stray = box((40, 30, 20), (30, 10, 8), (34, 14, 12))
    path = save(tmp_path / "m.nii.gz", big | stray)
    vertices, triangles, voxel_ml, share = mesh.surface(path, 40000)
    assert voxel_ml == pytest.approx(big.sum() * SPACING.prod() / 1000)
    assert share == pytest.approx(stray.sum() / (big.sum() + stray.sum()))
    assert closed_volume_ml(vertices, triangles) == pytest.approx(
        voxel_ml, rel=mesh.MAX_VOLUME_DRIFT
    )


def test_a_set_keeps_every_piece_but_loose_blobs(tmp_path: Path) -> None:
    shape = (60, 30, 20)
    one = box(shape, (4, 5, 5), (16, 25, 15))
    two = box(shape, (24, 5, 5), (36, 25, 15))
    blob = box(shape, (46, 12, 8), (49, 15, 11))  # 27 voxels, under MIN_PIECE_VOXELS
    path = save(tmp_path / "set.nii.gz", one | two | blob)
    vertices, triangles, voxel_ml, share = mesh.surface(path, 40000, pieces=True)
    assert share == 0.0  # a set keeps its pieces; nothing counts as dropped
    assert voxel_ml == pytest.approx((one.sum() + two.sum()) * SPACING.prod() / 1000)
    assert closed_volume_ml(vertices, triangles) == pytest.approx(
        voxel_ml, rel=mesh.MAX_VOLUME_DRIFT
    )
    # x is flipped to -R: the first box reaches R -3.2 mm, the second -28.8 mm, the blob -39.2 mm.
    r = vertices[:, 0]
    assert r.max() > -4
    assert -35 < r.min() < -28


def test_decimating_hard_keeps_the_surface_closed(tmp_path: Path) -> None:
    mask = ellipsoid((40, 40, 24), np.array([10.0, 7.0, 12.0]))
    path = save(tmp_path / "e.nii.gz", mask)
    _, full, _, _ = mesh.surface(path, 10**9)
    for budget in (200, 500, 2000):
        vertices, triangles, voxel_ml, _ = mesh.surface(path, budget)
        assert len(triangles) <= max(int(budget * 2.25), len(full))
        assert closed_volume_ml(vertices, triangles) == pytest.approx(voxel_ml, rel=0.08)


def test_the_same_mask_gives_the_same_surface(tmp_path: Path) -> None:
    mask = ellipsoid((40, 40, 24), np.array([10.0, 7.0, 12.0]))
    path = save(tmp_path / "e.nii.gz", mask)
    a = mesh.surface(path, 2000)
    b = mesh.surface(path, 2000)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_an_empty_mask_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        mesh.surface(save(tmp_path / "m.nii.gz", np.zeros((8, 8, 8), bool)), 1000)


# Mesh helpers and the GLB writer


def test_signed_volume_and_winding() -> None:
    assert mesh.signed_volume(TETRA, TETRA_TRIANGLES) == pytest.approx(1000 / 6)
    assert mesh.signed_volume(TETRA, TETRA_TRIANGLES[:, ::-1]) == pytest.approx(-1000 / 6)


def test_open_edges_counts_holes_and_seams() -> None:
    assert mesh.open_edges(TETRA_TRIANGLES) == 0
    assert mesh.open_edges(TETRA_TRIANGLES[:3]) == 3
    assert mesh.open_edges(np.vstack([TETRA_TRIANGLES, [[0, 1, 2]]])) == 3


def test_outward_turns_an_inside_out_surface_the_right_way() -> None:
    triangles, normals = mesh.outward(TETRA, np.ascontiguousarray(TETRA_TRIANGLES[:, ::-1]))
    assert mesh.signed_volume(TETRA, triangles) > 0
    assert normals.shape == (4, 3)


def test_write_glb_names_nodes_by_term_and_marks_modelled_ones(tmp_path: Path) -> None:
    triangles, normals = mesh.outward(TETRA, TETRA_TRIANGLES)
    vertices = (TETRA / 1000).astype(np.float32)
    parts: list[dict[str, Any]] = [
        {"structure": s, "vertices": vertices, "triangles": triangles, "normals": normals}
        for s in REFERENCE[:2]
    ]
    parts[1]["modelled"] = True
    mesh.write_glb(tmp_path / "a.glb", parts)
    mesh.write_glb(tmp_path / "b.glb", parts)
    data = (tmp_path / "a.glb").read_bytes()
    assert data == (tmp_path / "b.glb").read_bytes()
    assert len(data) % 4 == 0
    length = int.from_bytes(data[12:16], "little")
    gltf: dict[str, Any] = json.loads(data[20 : 20 + length])
    assert [n["name"] for n in gltf["nodes"]] == [node_name(s.term) for s in REFERENCE[:2]]
    assert gltf["nodes"][0]["extras"]["term"] == REFERENCE[0].term
    assert "modelled" not in gltf["nodes"][0]["extras"]
    assert gltf["nodes"][1]["extras"]["modelled"] is True


# settle_fragments: detached pieces of bones before meshing


def vertebra_pair(tmp_path: Path, chip: tuple[int, int, int] | None, moved: bool) -> Path:
    """T1 above T2, both 10-voxel cubes. T1's mask may carry a chip: either a
    piece lying on T2 (the segmentation's label error), or a small far one."""
    shape = (40, 40, 60)
    t1 = box(shape, (15, 15, 30), (25, 25, 40))
    t2 = box(shape, (15, 15, 18), (25, 25, 28))
    if chip is not None:
        lo = chip
        hi = (lo[0] + 5, lo[1] + 5, lo[2] + 4) if moved else (lo[0] + 2, lo[1] + 2, lo[2] + 2)
        t1 |= box(shape, lo, hi)
    save(tmp_path / "vertebrae_T1.nii.gz", t1)
    save(tmp_path / "vertebrae_T2.nii.gz", t2)
    return tmp_path


def test_a_piece_on_the_neighbour_is_moved_to_it(tmp_path: Path) -> None:
    seg = vertebra_pair(tmp_path, chip=(15, 25, 20), moved=True)  # 100 voxels against T2
    masks, notes = mesh.settle_fragments(seg, ["vertebrae_T1", "vertebrae_T2"])
    assert notes == [
        {"label": "vertebrae_T1", "voxels": 100, "ml": pytest.approx(0.1, abs=1e-6),
         "share": pytest.approx(100 / 1100, abs=1e-4), "action": "moved", "to": "vertebrae_T2"}
    ]  # fmt: skip
    assert masks["vertebrae_T1"].sum() == 1000
    assert masks["vertebrae_T2"].sum() == 1100
    assert mesh.ndimage.label(masks["vertebrae_T2"], structure=np.ones((3, 3, 3)))[1] == 1


def test_a_piece_on_a_neighbour_the_scan_cut_leaves_with_it(tmp_path: Path) -> None:
    # T2 is cut by the scan and left out of the body. T1's piece lying on it
    # is T2's bone under the wrong label: it goes too, and T1 is one piece.
    seg = vertebra_pair(tmp_path, chip=(15, 25, 20), moved=True)
    masks, notes = mesh.settle_fragments(seg, ["vertebrae_T1"], ["vertebrae_T2"])
    assert [(n["action"], n["to"], n["exported"]) for n in notes] == [
        ("moved", "vertebrae_T2", False)
    ]
    assert set(masks) == {"vertebrae_T1"} and masks["vertebrae_T1"].sum() == 1000
    # Without knowing of T2 the piece, 9% of T1, is no fragment: the build stops on it.
    masks, notes = mesh.settle_fragments(seg, ["vertebrae_T1"])
    assert notes == [] and masks["vertebrae_T1"].sum() == 1100


def test_chips_under_the_noise_limit_are_settled_when_together_they_are_over_it(
    tmp_path: Path,
) -> None:
    # Three chips of T1's label lie on T2: 49, 36 and 25 voxels, each under 2%
    # of T1 and 2.7% together. The largest goes to T2; the 1.5% left is noise.
    shape = (40, 40, 60)
    t1 = box(shape, (10, 10, 30), (30, 30, 40))
    t1 |= box(shape, (10, 10, 28), (17, 17, 29))
    t1 |= box(shape, (20, 10, 28), (26, 16, 29))
    t1 |= box(shape, (10, 20, 28), (15, 25, 29))
    save(tmp_path / "vertebrae_T1.nii.gz", t1)
    save(tmp_path / "vertebrae_T2.nii.gz", box(shape, (10, 10, 18), (30, 30, 28)))
    masks, notes = mesh.settle_fragments(tmp_path, ["vertebrae_T1", "vertebrae_T2"])
    assert [(n["voxels"], n["action"], n["to"]) for n in notes] == [(49, "moved", "vertebrae_T2")]
    assert masks["vertebrae_T1"].sum() == 4000 + 36 + 25
    assert masks["vertebrae_T2"].sum() == 4000 + 49
    # Two of them alone, 1.5%, are left as they are.
    t1[10:17, 10:17, 28] = False
    save(tmp_path / "vertebrae_T1.nii.gz", t1)
    assert mesh.settle_fragments(tmp_path, ["vertebrae_T1", "vertebrae_T2"])[1] == []


def test_a_small_far_chip_is_dropped(tmp_path: Path) -> None:
    seg = vertebra_pair(tmp_path, chip=(5, 5, 50), moved=False)  # 8 voxels, touching nothing
    # 8 of 1008 is under 2%, so it is only noise the manifest records; make it bigger.
    t1 = np.asanyarray(nib.load(seg / "vertebrae_T1.nii.gz").dataobj) > 0
    t1 |= box(t1.shape, (3, 3, 48), (7, 7, 51))  # 48 voxels in all, 4.6%: a fragment
    save(seg / "vertebrae_T1.nii.gz", t1)
    masks, notes = mesh.settle_fragments(seg, ["vertebrae_T1", "vertebrae_T2"])
    assert [n["action"] for n in notes] == ["dropped"]
    assert masks["vertebrae_T1"].sum() == 1000


def test_a_large_loose_piece_is_left_for_the_build_to_stop_on(tmp_path: Path) -> None:
    seg = vertebra_pair(tmp_path, chip=None, moved=False)
    t1 = np.asanyarray(nib.load(seg / "vertebrae_T1.nii.gz").dataobj) > 0
    t1 |= box(t1.shape, (2, 2, 46), (9, 9, 53))  # 343 voxels, 26%: touching nothing
    save(seg / "vertebrae_T1.nii.gz", t1)
    masks, notes = mesh.settle_fragments(seg, ["vertebrae_T1", "vertebrae_T2"])
    assert notes == []
    assert masks["vertebrae_T1"].sum() == 1343


def test_pieces_in_line_are_left_for_bridging(tmp_path: Path) -> None:
    rib = two_piece_rib(gap_voxels=5)
    save(tmp_path / "rib_left_5.nii.gz", rib)
    masks, notes = mesh.settle_fragments(tmp_path, ["rib_left_5"])
    assert notes == [] and masks["rib_left_5"].sum() == rib.sum()


def test_a_mask_already_corrected_stands_in_for_the_one_on_disk(tmp_path: Path) -> None:
    seg = vertebra_pair(tmp_path, chip=(15, 25, 20), moved=True)
    body = box((40, 40, 60), (15, 15, 30), (25, 25, 40))
    given = {"vertebrae_T1": body}
    masks, notes = mesh.settle_fragments(seg, ["vertebrae_T1", "vertebrae_T2"], None, given)
    assert notes == [] and masks["vertebrae_T1"].sum() == 1000
    assert masks["vertebrae_T1"] is not body  # a copy: the caller's mask is left alone


# whole_but_for_lumps: a cut neighbour's bone under a whole bone's label

SPINE = (40, 40, 60)
NECK = box(SPINE, (19, 19, 40), (21, 21, 44))  # two voxels across, from T1's top to the lump
LUMP = box(SPINE, (15, 15, 44), (25, 25, 52))  # 800 voxels, up to the padding
C7 = box(SPINE, (25, 15, 44), (31, 25, 52))  # beside the lump, and cut by the scan's end
PADDING_CUT = "cut by the scan's edge (unscanned padding)"


def spine_at_the_scans_end(
    tmp_path: Path, extra: Mask, others: dict[str, Mask]
) -> tuple[tuple[Any, Any], dict[str, str | None]]:
    """T1, a 16 x 16 x 20 block ending at slice 39, in a scan that was never
    scanned from slice 54 up: the padding, blend included, starts at slice 52.
    extra goes into T1's mask; others are its neighbours. Returns the padding
    and each label's cut reason."""
    hu = np.full(SPINE, 40.0)
    hu[:, :, 54:] = -1024
    outside = mesh.scan_outside(save_ct(tmp_path / "ct.nii.gz", hu))
    masks = {"vertebrae_T1": box(SPINE, (12, 12, 20), (28, 28, 40)) | extra, **others}
    for label, mask in masks.items():
        save(tmp_path / f"{label}.nii.gz", mask)
    labels = ["vertebrae_C6", "vertebrae_C7", "vertebrae_T1"]
    return outside, {lab: mesh.cut_reason(tmp_path / f"{lab}.nii.gz", outside) for lab in labels}


def test_a_cut_neighbours_lump_on_a_thin_neck_does_not_cut_the_bone(tmp_path: Path) -> None:
    outside, reasons = spine_at_the_scans_end(tmp_path, NECK | LUMP, {"vertebrae_C7": C7})
    assert reasons["vertebrae_T1"] == reasons["vertebrae_C7"] == PADDING_CUT
    whole, notes = mesh.whole_but_for_lumps(tmp_path, reasons, outside)
    assert set(whole) == {"vertebrae_T1"}
    t1 = whole["vertebrae_T1"]
    assert t1.shape == SPINE and t1.sum() == 16 * 16 * 20 + 8  # the block and the neck's foot
    assert np.flatnonzero(t1.any((0, 1))).max() == 41
    assert notes == [
        {"label": "vertebrae_T1", "voxels": 808, "ml": pytest.approx(0.78, abs=0.006),
         "share": pytest.approx(808 / 5936, abs=1e-4), "action": "moved", "to": "vertebrae_C7",
         "neck": True, "exported": False}
    ]  # fmt: skip


def test_a_thin_tip_the_scan_ends_through_is_the_bones_own(tmp_path: Path) -> None:
    # A plate two voxels thin rises from T1 to the padding and lies on C7: the
    # opening leaves no core of it, so it is T1's own process, and T1 is cut.
    # (The knob under T1 has a core, but the scan's end is nowhere near it.)
    plate = box(SPINE, (19, 12, 40), (21, 28, 52))
    knob = box(SPINE, (19, 19, 14), (21, 21, 20)) | box(SPINE, (16, 16, 6), (24, 24, 14))
    c7 = box(SPINE, (21, 12, 44), (27, 28, 52))
    outside, reasons = spine_at_the_scans_end(tmp_path, plate | knob, {"vertebrae_C7": c7})
    assert reasons["vertebrae_T1"] == PADDING_CUT
    assert mesh.whole_but_for_lumps(tmp_path, reasons, outside) == ({}, [])


@pytest.mark.parametrize("neighbour", [None, box(SPINE, (25, 15, 42), (31, 25, 48))])
def test_a_lump_with_no_cut_neighbour_under_it_leaves_the_bone_cut(
    tmp_path: Path, neighbour: Mask | None
) -> None:
    # No C7 at all, or a whole one that ends clear of the padding: the lump
    # cannot be shown to be another bone's, so T1 is cut as the test found it.
    others = {} if neighbour is None else {"vertebrae_C7": neighbour}
    outside, reasons = spine_at_the_scans_end(tmp_path, NECK | LUMP, others)
    assert reasons["vertebrae_T1"] == PADDING_CUT
    assert reasons["vertebrae_C7"] == ("not segmented" if neighbour is None else None)
    assert mesh.whole_but_for_lumps(tmp_path, reasons, outside) == ({}, [])


def test_a_lump_over_a_quarter_of_the_mask_leaves_the_bone_cut(tmp_path: Path) -> None:
    lump = box(SPINE, (11, 11, 44), (29, 29, 52))  # with its end of the neck, 2600 of 7728
    c7 = box(SPINE, (29, 15, 44), (35, 25, 52))
    outside, reasons = spine_at_the_scans_end(tmp_path, NECK | lump, {"vertebrae_C7": c7})
    assert mesh.whole_but_for_lumps(tmp_path, reasons, outside) == ({}, [])


def test_a_bone_that_meets_the_scans_end_itself_stays_cut(tmp_path: Path) -> None:
    # Besides the lump, a thick process of T1's own runs up into the padding
    # from the block's side, clear of the lump.
    process = box(SPINE, (6, 18, 30), (10, 22, 52)) | box(SPINE, (10, 18, 30), (12, 22, 34))
    outside, reasons = spine_at_the_scans_end(tmp_path, NECK | LUMP | process, {"vertebrae_C7": C7})
    assert mesh.whole_but_for_lumps(tmp_path, reasons, outside) == ({}, [])


def test_a_bone_given_a_lump_that_meets_the_scans_end_is_cut_after_all(tmp_path: Path) -> None:
    # T1 carries a lump lying on C7's mask; C7's own block is clear of the
    # scan's end, and the part of its mask that reaches it is a lump lying on
    # C6. T1 is whole. C7 would be, but T1's lump is C7's bone and meets the
    # scan's end: C7 is cut.
    t1_lump = box(SPINE, (15, 15, 44), (25, 21, 52))
    c7 = box(SPINE, (30, 10, 16), (39, 30, 40))  # its block, beside T1's
    c7 |= box(SPINE, (32, 19, 40), (34, 21, 44))  # a neck from it to its lump
    c7 |= box(SPINE, (25, 15, 44), (35, 25, 52))  # the lump, between T1's lump and C6
    c6 = box(SPINE, (25, 25, 44), (35, 31, 52))
    outside, reasons = spine_at_the_scans_end(
        tmp_path, NECK | t1_lump, {"vertebrae_C7": c7, "vertebrae_C6": c6}
    )
    assert set(reasons.values()) == {PADDING_CUT}
    whole, notes = mesh.whole_but_for_lumps(tmp_path, reasons, outside)
    assert set(whole) == {"vertebrae_T1"}
    assert [(n["label"], n["to"]) for n in notes] == [("vertebrae_T1", "vertebrae_C7")]


def test_a_lump_reaching_the_volumes_own_face_counts_as_meeting_the_scans_end(
    tmp_path: Path,
) -> None:
    # No CT to say where the scan ends: the lump and C7 run to the last slice.
    shape = (40, 40, 52)
    t1 = box(shape, (12, 12, 20), (28, 28, 40)) | NECK[:, :, :52] | LUMP[:, :, :52]
    save(tmp_path / "vertebrae_T1.nii.gz", t1)
    save(tmp_path / "vertebrae_C7.nii.gz", C7[:, :, :52])
    reasons = {lab: mesh.cut_reason(tmp_path / f"{lab}.nii.gz")
               for lab in ("vertebrae_C7", "vertebrae_T1")}  # fmt: skip
    assert set(reasons.values()) == {"cut by the scan's edge (axis 2)"}
    whole, notes = mesh.whole_but_for_lumps(tmp_path, reasons)
    assert set(whole) == {"vertebrae_T1"} and whole["vertebrae_T1"].sum() == 16 * 16 * 20 + 8
    assert [n["to"] for n in notes] == ["vertebrae_C7"]


def test_rounded_ends_keep_the_length_and_round_both_tips() -> None:
    curve = np.c_[np.linspace(0, 60, 31), np.zeros(31), np.zeros(31)]  # 60 mm straight band
    half_w, half_d = np.full(31, 6.0), np.full(31, 2.0)
    points, widths, depths = mesh.rounded_ends(curve, half_w, half_d)
    assert points[0, 0] == pytest.approx(0.0, abs=0.1)
    assert points[-1, 0] == pytest.approx(60.0, abs=0.1)
    assert np.all(np.diff(points[:, 0]) > 0)  # never folds back
    assert widths[0] < 1.0 and widths[-1] < 1.0 and widths.max() == 6.0
    assert depths[0] < 0.4 and depths[-1] < 0.4
    vertices, triangles = mesh.sweep(points, np.array([0.0, 1.0, 0.0]), widths, depths)
    assert mesh.open_edges(triangles) == 0
    assert abs(mesh.signed_volume(vertices, triangles)) > 0  # main() turns it outward
