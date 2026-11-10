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


def test_a_mask_one_slice_below_the_padding_is_whole(tmp_path: Path) -> None:
    outside = mesh.scan_outside(padded_ct(tmp_path))
    mask = box((40, 40, 40), (12, 12, 20), (20, 20, 33))  # top slice 32; padding from 34
    assert mesh.cut_reason(save(tmp_path / "m.nii.gz", mask), outside) is None


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

    given: dict[str, Any] = {}

    def fake_cartilage(sternum_path: Path, ribs: dict[str, Any]) -> tuple[Any, Any, Any]:
        given.update(sternum=sternum_path, ribs=ribs)
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
    manifest["cartilage_from"] = given
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
    # It is modelled from the sternum's mask and the ribs as they were meshed, in RAS mm.
    given = manifest["cartilage_from"]
    assert given["sternum"] == tmp_path / "seg" / "sternum.nii.gz"
    ribs = [label for label in manifest["surfaced"] if label.startswith("rib_")]
    assert len(ribs) == 24 and list(given["ribs"]) == ribs
    for label, (vertices, triangles) in given["ribs"].items():
        np.testing.assert_allclose(vertices, TETRA + manifest["surfaced"].index(label) + 1)
        np.testing.assert_array_equal(triangles, TETRA_TRIANGLES)
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


def test_rounded_ends_keep_the_length_and_round_both_tips() -> None:
    curve = np.c_[np.linspace(0, 60, 31), np.zeros(31), np.zeros(31)]  # 60 mm straight band
    half_w, half_d = np.full(31, 6.0), np.full(31, 2.0)
    points, widths, depths = mesh.rounded_ends(curve, half_w, half_d)
    assert points[0, 0] == pytest.approx(0.0, abs=0.1)
    assert points[-1, 0] == pytest.approx(60.0, abs=0.1)
    assert np.all(np.diff(points[:, 0]) > 0)  # never folds back
    assert widths[0] < 1.0 and widths[-1] < 1.0 and widths.max() == 6.0
    assert depths[0] < 0.4 and depths[-1] < 0.4


# The modelled costal cartilage: rib ends, frames, band layout, the bands' field and its surface


def bar(
    tip: tuple[float, float, float], direction: tuple[float, float, float], half_w: float,
    half_d: float, length: float = 90.0,
) -> tuple[FloatArray, npt.NDArray[np.uint32]]:  # fmt: skip
    """A closed, flat-ended bar of elliptical section, as a rib's meshed
    surface (RAS mm vertices, triangles): it ends at tip, runs there along
    direction, and is half_w wide upward and half_d deep."""
    unit = np.asarray(direction, float) / np.linalg.norm(direction)
    across = mesh.UP - unit * (mesh.UP @ unit)
    across /= np.linalg.norm(across)
    deep = np.cross(unit, across)
    sides, rings = 24, int(length / 1.5) + 1
    angle = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    section = half_w * np.outer(np.cos(angle), across) + half_d * np.outer(np.sin(angle), deep)
    back = np.linspace(length, 0, rings)
    centres = np.asarray(tip, float) - np.outer(back, unit)
    vertices = np.vstack([(centres[:, None, :] + section[None]).reshape(-1, 3), centres[[0, -1]]])
    triangles = []
    for i in range(rings - 1):
        for k in range(sides):
            a, b = i * sides + k, i * sides + (k + 1) % sides
            triangles += [(a, a + sides, b), (b, a + sides, b + sides)]
    for k in range(sides):
        triangles.append((rings * sides, k, (k + 1) % sides))
        last = (rings - 1) * sides
        triangles.append((rings * sides + 1, last + (k + 1) % sides, last + k))
    return vertices, np.array(triangles, dtype=np.uint32)


def test_a_rib_end_is_measured_on_its_meshed_surface() -> None:
    vertices, triangles = bar((40, 30, 100), (0.3, 0.9, -0.3), 7.0, 3.0)
    assert mesh.open_edges(triangles) == 0
    end = mesh.rib_end_of(vertices, triangles)
    assert end is not None
    unit = np.array([0.3, 0.9, -0.3]) / np.linalg.norm([0.3, 0.9, -0.3])
    assert end["direction"] @ unit > 0.999  # the front (+A) end, along the rib's own line
    np.testing.assert_allclose(end["tip"], [40, 30, 100], atol=0.5)
    assert end["halfWidth"] == pytest.approx(7.0, abs=0.3)
    assert end["halfDepth"] == pytest.approx(3.0, abs=0.3)
    assert abs(end["across"] @ end["direction"]) < 1e-9
    assert abs(end["across"][2]) > 0.9  # its width runs upward
    np.testing.assert_allclose(end["section"].mean(0), 0, atol=1e-9)
    np.testing.assert_allclose(end["section"] @ end["direction"], 0, atol=1e-9)  # flattened


def test_a_rib_too_short_to_walk_has_no_end() -> None:
    vertices, triangles = bar((0, 0, 0), (0, 1, 0), 1.5, 1.5, length=3.0)
    assert mesh.rib_end_of(vertices, triangles) is None


def test_extents_are_half_the_spread_and_never_under_the_floors() -> None:
    grid = np.mgrid[-5:5.01:0.25, -3:3.01:0.25].reshape(2, -1).T
    section = np.c_[grid[:, 0], np.zeros(len(grid)), grid[:, 1]]
    x, z = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    width, depth = mesh.extents(section, x, z)
    assert width == pytest.approx(5.0, abs=0.3) and depth == pytest.approx(3.0, abs=0.3)
    assert mesh.extents(section * 0.1, x, z) == (3.0, mesh.MIN_HALF_DEPTH_MM)


def test_wall_across_lies_in_the_wall_at_right_angles_to_the_band_and_points_up() -> None:
    forward = np.array([0.0, 1.0, 0.0])
    # A band running across the front of the chest: its width is straight up.
    np.testing.assert_allclose(mesh.wall_across(np.array([1.0, 0, 0]), forward), [0, 0, 1])
    np.testing.assert_allclose(mesh.wall_across(np.array([-1.0, 0, 0]), forward), [0, 0, 1])
    tangent = np.array([0.6, 0.0, 0.8])  # rising toward the sternum
    across = mesh.wall_across(tangent, forward)
    assert across @ tangent == pytest.approx(0.0, abs=1e-12) and across @ forward == 0
    assert np.linalg.norm(across) == pytest.approx(1.0) and across[2] > 0
    # Running straight out of the chest there is no such direction: the width goes up.
    np.testing.assert_allclose(mesh.wall_across(forward, forward), [0, 0, 1], atol=1e-12)


def test_frames_are_orthonormal_and_spread_the_end_twist_along_the_curve() -> None:
    theta = np.linspace(0, np.pi / 2, 41)
    curve = np.c_[50 * np.cos(theta), 50 * np.sin(theta), np.zeros_like(theta)]
    tangents, across, deep = mesh.frames(curve, mesh.UP)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    np.testing.assert_allclose(np.einsum("ij,ij->i", tangents, across), 0.0, atol=1e-9)
    np.testing.assert_allclose(deep, np.cross(tangents, across), atol=1e-12)
    np.testing.assert_allclose(across, np.tile(mesh.UP, (41, 1)), atol=1e-9)  # a flat curve
    turned = np.array([0.0, np.sin(np.radians(40)), np.cos(np.radians(40))])  # 40 deg about -x
    _, twisted, _ = mesh.frames(curve, mesh.UP, turned)
    angles = np.degrees(np.arccos(np.clip(twisted @ mesh.UP, -1, 1)))
    np.testing.assert_allclose(angles, np.linspace(0, 40, 41), atol=0.5)
    assert twisted[-1] @ tangents[-1] == pytest.approx(0.0, abs=1e-9)
    assert twisted[-1] @ turned > np.cos(np.radians(1.0))  # the target, squared to the end tangent
    # The opposite width direction is the same band: no half turn is added to reach it.
    _, same, _ = mesh.frames(curve, mesh.UP, -mesh.UP)
    np.testing.assert_allclose(same, across, atol=1e-9)


def test_densify_resamples_by_arc_length_and_carries_the_profiles() -> None:
    curve = np.c_[[0, 3, 10], [0, 0, 0], [0, 0, 0]].astype(float)
    np.testing.assert_allclose(mesh.arc_length(curve), [0, 3, 10])
    points, width = mesh.densify(curve, np.array([1.0, 4.0, 11.0]), step=0.5)
    assert len(points) == 21 and points[-1, 0] == pytest.approx(10.0)
    np.testing.assert_allclose(np.diff(points[:, 0]), 0.5)
    np.testing.assert_allclose(width, points[:, 0] + 1)


def straight_band(z: float = 0.0, half_w: float = 6.0, half_d: float = 2.4) -> dict[str, Any]:
    """A band 60 mm along x at height z: its width runs up, its depth along y."""
    curve = np.c_[np.linspace(0, 60, 121), np.zeros(121), np.full(121, z)]
    return {"curve": curve, "halfW": np.full(121, half_w), "halfD": np.full(121, half_d),
            "across0": mesh.UP}  # fmt: skip


def test_a_bands_field_is_its_depth_inside_a_flat_section_with_round_tips() -> None:
    band = mesh.laid_out(straight_band())
    assert np.all(np.diff(band["curve"][:, 0]) > 0) and band["halfW"].max() == 6.0
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(band["curve"], axis=0), axis=1), 0.5, atol=0.01
    )
    points = np.array([
        [30, 0, 0], [30, 1.4, 0], [30, 2.4, 0], [30, 3.4, 0],  # through the broad face
        [30, 0, 5.0], [30, 0, 6.0], [30, 0, 7.5],  # up through the border
        [30, 2.3, 5.8],  # the corner an ellipse would hold and a rectangle would too
        [-1.0, 0, 0], [61.0, 0, 0],  # past the two tips
        [30, 0, 40.0],  # out of reach
    ])  # fmt: skip
    depth = mesh.band_field(band, points, 12.0)
    np.testing.assert_allclose(depth[:4], [2.4, 1.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(depth[4:7], [1.0, 0.0, -1.5], atol=1e-6)
    assert depth[7] < 0 and depth[8] < -0.9 and depth[9] < -0.9
    assert depth[10] == -1e3
    # Flatter than an ellipse: a point inside the superellipse and outside the ellipse.
    inside = mesh.band_field(band, np.array([[30, 1.9, 4.2]]), 12.0)[0]
    assert (4.2 / 6.0) ** 2 + (1.9 / 2.4) ** 2 > 1 and inside > 0


def test_a_band_that_twists_has_no_step_in_its_field() -> None:
    band = straight_band()
    band["across1"] = np.array([0.0, 1.0, 1.0])  # the face turns 45 degrees on the way
    laid = mesh.laid_out(band)
    x = np.linspace(20, 40, 2001)
    depth = mesh.band_field(laid, np.c_[x, np.full_like(x, 1.5), np.full_like(x, 3.0)], 12.0)
    assert np.abs(np.diff(depth)).max() < 0.01  # 0.01 mm apart along the band


def test_rasterised_bands_run_together_where_they_meet_and_stay_apart_otherwise() -> None:
    field, origin = mesh.rasterise_bands([straight_band()], 1.0)

    def at(x: float, y: float, z: float) -> float:
        i, j, k = np.round(np.array([x, y, z]) - origin).astype(int)
        return float(field[i, j, k])

    assert at(30, 0, 0) == pytest.approx(2.4, abs=0.6) and at(30, 0, 5) > 0  # width up
    assert at(30, 4, 0) < 0 and at(30, 0, 8) < 0 and at(-3, 0, 0) < 0 and at(63, 0, 0) < 0
    assert field.min() == -mesh.FIELD_REACH_MM  # cut off, and outside all round
    for face in (field[0], field[-1], field[:, 0], field[:, -1], field[..., 0], field[..., -1]):
        assert face.max() < 0
    ones = np.ones((3, 3, 3))
    # 13 mm apart the borders are 1 mm apart: they run together in a fillet.
    merged, _ = mesh.rasterise_bands([straight_band(), straight_band(z=13.0)], 1.0)
    assert mesh.ndimage.label(merged > 0, structure=ones)[1] == 1
    apart, _ = mesh.rasterise_bands([straight_band(), straight_band(z=20.0)], 1.0)
    assert mesh.ndimage.label(apart > 0, structure=ones)[1] == 2


def edge_lengths(vertices: FloatArray, triangles: npt.NDArray[np.uint32]) -> FloatArray:
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return np.asarray(lengths, dtype=np.float64)


def test_evened_puts_the_vertices_on_the_fields_zero_level_with_even_triangles() -> None:
    centre, radius = np.array([15.0, 15.0, 15.0]), 10.2
    grid = np.indices((31, 31, 31)).transpose(1, 2, 3, 0).astype(float)
    field = (radius - np.linalg.norm(grid - centre, axis=-1)).astype(np.float32)
    image = mesh.vtk.vtkImageData()
    image.SetDimensions(*field.shape)
    image.GetPointData().SetScalars(mesh.numpy_to_vtk(field.ravel(order="F"), deep=True))
    poly = mesh.contour(image, 0.0).GetOutput()
    before = mesh.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    triangles = mesh.triangles_of(poly).astype(np.uint32)
    assert mesh.open_edges(triangles) == 0
    after = mesh.evened(before, triangles, field, np.zeros(3), 1.0)
    assert after.shape == before.shape
    off = np.abs(np.linalg.norm(after - centre, axis=1) - radius)
    assert off.max() < 0.002  # on the sphere, nearer than the contour's own vertices were
    assert np.abs(np.linalg.norm(before - centre, axis=1) - radius).max() > 0.005
    was, now = edge_lengths(before, triangles), edge_lengths(after, triangles)
    assert was.min() < 0.05 and now.min() > 0.25  # the contour's slivers are gone
    assert now.std() / now.mean() < 0.6 * was.std() / was.mean()
    # What a quality check takes for spikes on the raw contour is gone too.
    assert umbrella_scores(before, triangles).max() > 0.8
    assert umbrella_scores(after, triangles).max() < 0.2
    assert mesh.signed_volume(after, mesh.outward(after, triangles)[0]) == pytest.approx(
        4 / 3 * np.pi * radius**3, rel=0.01
    )
    # Where the field has no slope to follow, a vertex stays where it is.
    np.testing.assert_array_equal(
        mesh.evened(before, triangles, np.zeros_like(field), np.zeros(3), 1.0), before
    )


def umbrella_scores(vertices: FloatArray, triangles: npt.NDArray[np.uint32]) -> FloatArray:
    """blender/qa.py's spike score: each vertex's distance from its
    neighbours' mean, in local edge lengths."""
    edges = np.unique(
        np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), 1),
        axis=0,
    ).astype(np.int64)
    a, b = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    count = np.bincount(a, minlength=len(vertices))
    mean = np.stack([np.bincount(a, vertices[b, d], len(vertices)) for d in range(3)], 1)
    mean /= np.maximum(count, 1)[:, None]
    edge = np.bincount(a, np.linalg.norm(vertices[a] - vertices[b], axis=1), len(vertices))
    scores = np.linalg.norm(vertices - mean, axis=1) / np.maximum(edge / np.maximum(count, 1), 1e-9)
    return np.asarray(scores, dtype=np.float64)


def mesh_pieces(vertices: FloatArray, triangles: npt.NDArray[np.uint32]) -> int:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    used = np.unique(triangles)
    index = np.full(len(vertices), -1)
    index[used] = np.arange(len(used))
    edges = index[np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]]])]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(used),) * 2)
    return int(connected_components(graph, directed=False)[0])


def test_a_bands_surface_is_closed_flat_smooth_and_as_big_as_the_band() -> None:
    vertices, triangles = mesh.cartilage_surface([straight_band()])
    triangles, _ = mesh.outward(vertices, triangles)
    assert mesh.open_edges(triangles) == 0 and mesh_pieces(vertices, triangles) == 1
    assert vertices[:, 0].min() > -0.5 and vertices[:, 0].max() < 60.5  # round tips, nothing beyond
    middle = vertices[np.abs(vertices[:, 0] - 30) < 20]
    assert np.abs(middle[:, 2]).max() == pytest.approx(6.0, abs=0.15)  # 12 mm wide, upward
    assert np.abs(middle[:, 1]).max() == pytest.approx(2.4, abs=0.15)  # 4.8 mm deep: 2.5 to 1
    # A band of exponent CARTILAGE_SQUARE, 60 mm long, less what its two domed ends lack.
    n = mesh.CARTILAGE_SQUARE
    area = 4 * 6.0 * 2.4 * math.gamma(1 + 1 / n) ** 2 / math.gamma(1 + 2 / n)
    assert mesh.signed_volume(vertices, triangles) == pytest.approx(area * (60 - 4), rel=0.05)
    assert umbrella_scores(vertices, triangles).max() < 0.6  # the quality check flags 0.85
    lengths = edge_lengths(vertices, triangles)
    assert lengths.max() < 3 * mesh.CARTILAGE_VOXEL_MM  # not decimated: no long slivers


def test_two_bands_that_meet_are_one_surface_without_crossing_faces() -> None:
    lower = straight_band()
    upper = straight_band(z=9.0)
    upper["curve"] = upper["curve"] + np.c_[np.zeros(121), np.zeros(121), np.linspace(0, 12, 121)]
    vertices, triangles = mesh.cartilage_surface([lower, upper])  # a fork: together, then apart
    triangles, _ = mesh.outward(vertices, triangles)
    assert mesh.open_edges(triangles) == 0 and mesh_pieces(vertices, triangles) == 1
    assert mesh.signed_volume(vertices, triangles) > 0
    # One surface round the union: away from the rounded tips, where the light
    # blur shortens a band a little, no vertex of it lies inside either band.
    body = vertices[np.abs(vertices[:, 0] - 30) < 22]
    for band in (lower, upper):
        assert mesh.band_field(mesh.laid_out(band), body, 30.0).max() < 0.3


def rib_end_at(
    tip: tuple[float, float, float], direction: tuple[float, float, float], half_w: float,
    half_d: float,
) -> dict[str, Any]:  # fmt: skip
    """A measured rib end, as rib_end_of gives it, whose section is an ellipse
    half_w up the rib's border and half_d through its face."""
    unit = np.asarray(direction, float) / np.linalg.norm(direction)
    across = mesh.UP - unit * (mesh.UP @ unit)
    across /= np.linalg.norm(across)
    angle = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    section = half_w * np.outer(np.cos(angle), across)
    section = section + half_d * np.outer(np.sin(angle), np.cross(unit, across))
    return {"tip": np.asarray(tip, float), "direction": unit, "across": across, "section": section,
            "halfWidth": half_w, "halfDepth": half_d}  # fmt: skip


def plate_sternum(xiphoid: bool = False) -> FloatArray:
    """RAS points of a sternum-like plate, 24 mm wide about x = 0 at the front,
    from z = 60 to 160; with a xiphoid, a 9 mm wide tongue hangs 30 mm below."""
    grid = np.mgrid[-12:12.1:1.5, 60:66.1:1.5, 60:160.1:1.5].reshape(3, -1).T
    if xiphoid:
        tongue = np.mgrid[-4.5:4.6:1.5, 60:66.1:1.5, 30:59.9:1.5].reshape(3, -1).T
        grid = np.vstack([grid, tongue])
    return np.asarray(grid, dtype=np.float64)


def seventh_pair(half_w: tuple[float, float] = (7.0, 5.0)) -> dict[str, dict[str, Any]]:
    return {
        "rib_left_7": rib_end_at((-60, 40, 100), (0.3, 0.9, -0.3), half_w[0], 3.0),
        "rib_right_7": rib_end_at((60, 40, 100), (-0.3, 0.9, -0.3), half_w[1], 3.0),
    }


def shown(band: dict[str, Any]) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """A band from its rib's end to its target: the share of the way along
    (0 to 1), the centreline and the half widths and depths."""
    part = slice(band["tipAt"], band["endAt"] + 1)
    length = mesh.arc_length(band["curve"][part])
    return length / length[-1], band["curve"][part], band["halfW"][part], band["halfD"][part]


def test_the_sternums_body_ends_above_its_xiphoid_and_its_border_is_found_per_side() -> None:
    assert mesh.sternum_floor(plate_sternum()) == pytest.approx(60.0, abs=1.5)
    sternum = plate_sternum(xiphoid=True)
    assert sternum[:, 2].min() == 30 and mesh.sternum_floor(sternum) == pytest.approx(60.0, abs=1.5)
    edge, centre = mesh.sternum_edge(sternum, -1.0, 100.0)
    np.testing.assert_allclose(edge, [-12, 63, 100])
    assert centre == 0
    np.testing.assert_allclose(mesh.sternum_edge(sternum, 1.0, 100.0)[0], [12, 63, 100])
    # Beyond its top the nearest level stands in.
    np.testing.assert_allclose(mesh.sternum_edge(sternum, 1.0, 190.0)[0], [12, 63, 190])


def test_a_band_leaves_its_rib_in_line_with_the_ribs_own_section_and_takes_the_pairs() -> None:
    ends = seventh_pair()
    bands, skipped = mesh.band_curves(ends, plate_sternum())
    assert skipped == [] and set(bands) == set(ends)
    for rib, band in bands.items():
        e, at = ends[rib], band["tipAt"]
        own_w, own_d = mesh.extents(
            e["section"], band["across0"], np.cross(e["direction"], band["across0"])
        )
        assert own_w == pytest.approx(e["halfWidth"], abs=0.3)
        assert own_d == pytest.approx(e["halfDepth"], abs=0.4)
        # At the rib's end the band has the rib's own section: no step at the join.
        np.testing.assert_allclose(band["curve"][at], e["tip"])
        assert band["halfW"][at] == pytest.approx(own_w) and band["halfD"][at] == pytest.approx(
            own_d
        )
        assert band["across0"] @ e["direction"] == pytest.approx(0.0, abs=1e-9)
        assert band["across0"][2] > 0.9  # width up the rib's border: the broad face looks out
        # It begins out of sight inside the rib, thinner, straight in the rib's own line.
        lead = mesh.JOIN_OVERLAP_MM + mesh.INSIDE_SHARE * own_w
        np.testing.assert_allclose(band["curve"][0], e["tip"] - e["direction"] * lead)
        inside = band["curve"][: at + 1] - e["tip"]
        np.testing.assert_allclose(inside, np.outer(inside @ e["direction"], e["direction"]),
                                   atol=1e-9)  # fmt: skip
        assert band["halfW"][0] == pytest.approx(mesh.INSIDE_SHARE * own_w)
        assert np.all(np.diff(band["halfW"][: at + 1]) > 0)
        # No kink where it leaves the rib.
        assert band["tangent"][at] @ e["direction"] > 0.999
        np.testing.assert_allclose(np.linalg.norm(band["across"], axis=1), 1.0)
        np.testing.assert_allclose(np.einsum("ij,ij->i", band["across"], band["tangent"]), 0,
                                   atol=1e-9)  # fmt: skip
    f, curve, half_w, half_d = shown(bands["rib_left_7"])
    g, mirrored, other_w, other_d = shown(bands["rib_right_7"])
    w_pair = (half_w[0] + other_w[0]) / 2  # the mean of the two ribs' own widths
    assert w_pair == pytest.approx(6.0, abs=0.3)
    third = f >= 0.35
    # Beyond the first third both have the pair's section: flat bands, 2.5 to 1,
    # widening 15% toward the sternum.
    np.testing.assert_allclose(
        half_w[third], w_pair * (1 + mesh.CARTILAGE_WIDEN * f[third]), rtol=0.01
    )
    np.testing.assert_allclose(half_w[third] / half_d[third], mesh.CARTILAGE_FLAT)
    assert half_w[-1] == pytest.approx(1.15 * w_pair) and other_w[-1] == pytest.approx(
        1.15 * w_pair
    )
    assert np.all(np.abs(np.diff(half_w)) < 0.05)  # the width never steps
    np.testing.assert_allclose(np.interp(0.6, f, half_w), np.interp(0.6, g, other_w), rtol=0.01)
    # Into the sternum's border on its own side, NOTCH_MM deep at full width, at the
    # height of the pair's rib ends; the rounded end stays short of the sternum's middle.
    np.testing.assert_allclose(curve[-1], [-12 + mesh.NOTCH_MM, 63, 100])
    np.testing.assert_allclose(mirrored[-1], [12 - mesh.NOTCH_MM, 63, 100])
    assert -12 < bands["rib_left_7"]["curve"][-1][0] <= -1
    assert 12 > bands["rib_right_7"]["curve"][-1][0] >= 1
    # Left and right are mirror images where the ribs are.
    even = mesh.band_curves(seventh_pair((6.0, 6.0)), plate_sternum())[0]
    np.testing.assert_allclose(
        even["rib_left_7"]["curve"] * [-1, 1, 1], even["rib_right_7"]["curve"], atol=1e-9
    )
    np.testing.assert_allclose(even["rib_left_7"]["halfW"], even["rib_right_7"]["halfW"])


def test_a_band_is_never_wider_than_the_cap_and_a_steep_rib_end_does_not_hook() -> None:
    bands, _ = mesh.band_curves(seventh_pair((11.0, 11.0)), plate_sternum())
    f, _, half_w, _ = shown(bands["rib_left_7"])
    capped = half_w[f >= 0.35]  # the rib's own wider section is kept at the join
    assert half_w[0] == pytest.approx(11.0, abs=0.3)
    assert capped.max() == pytest.approx(1.15 * mesh.CARTILAGE_HALF_W_MM)
    assert capped.min() > 0.99 * mesh.CARTILAGE_HALF_W_MM
    steep = {
        "rib_left_7": rib_end_at((-60, 40, 100), (0.2, 0.5, -0.84), 7.0, 3.0),
        "rib_right_7": rib_end_at((60, 40, 100), (-0.2, 0.5, -0.84), 7.0, 3.0),
    }
    bands, _ = mesh.band_curves(steep, plate_sternum())
    _, curve, _, _ = shown(bands["rib_left_7"])
    assert curve[:, 2].min() > curve[0, 2] - 12.0  # turns up soon after the rib
    assert curve[-1, 2] == pytest.approx(100.0)


def test_pairs_meet_the_sternum_one_above_the_other_and_never_on_the_xiphoid() -> None:
    ends = {
        "rib_left_6": rib_end_at((-70, 40, 62), (0.3, 0.9, -0.3), 6.0, 3.0),
        "rib_right_6": rib_end_at((70, 40, 62), (-0.3, 0.9, -0.3), 6.0, 3.0),
        "rib_left_7": rib_end_at((-80, 35, 40), (0.3, 0.9, -0.3), 6.0, 3.0),
        "rib_right_7": rib_end_at((80, 35, 40), (-0.3, 0.9, -0.3), 6.0, 3.0),
    }
    bands, skipped = mesh.band_curves(ends, plate_sternum(xiphoid=True))
    assert skipped == []
    _, lower, lower_w, _ = shown(bands["rib_left_7"])
    _, upper, upper_w, _ = shown(bands["rib_left_6"])
    assert lower[-1, 2] == pytest.approx(60.0, abs=1.5)  # the foot of the body, not the xiphoid
    assert abs(lower[-1, 0]) == pytest.approx(12 - mesh.NOTCH_MM)
    # A full band's width higher: the two borders meet and do not overlap.
    assert upper[-1, 2] - lower[-1, 2] == pytest.approx(lower_w[-1] + upper_w[-1])
    assert lower_w[-1] == pytest.approx(1.15 * 6.0, abs=0.3)


def test_ribs_8_to_10_merge_into_the_band_above_and_an_orphan_is_skipped() -> None:
    ends = seventh_pair()
    ends["rib_left_8"] = rib_end_at((-70, 30, 70), (0.3, 0.9, -0.3), 6.0, 3.0)
    ends["rib_right_8"] = rib_end_at((70, 30, 70), (-0.3, 0.9, -0.3), 6.0, 3.0)
    ends["rib_left_10"] = rib_end_at((-75, 20, 40), (0.3, 0.9, -0.3), 5.0, 3.0)
    ends["rib_right_10"] = rib_end_at((75, 20, 40), (-0.3, 0.9, -0.3), 5.0, 3.0)
    ends["rib_left_11"] = rib_end_at((-80, 10, 20), (0.3, 0.9, -0.3), 5.0, 3.0)
    ends["rib_left_3"] = rib_end_at((-40, 40, 140), (0.3, 0.9, -0.3), 5.0, 3.0)
    bands, skipped = mesh.band_curves(ends, plate_sternum())
    assert sorted(skipped) == [
        ("rib_left_10", "no cartilage above it to join"),
        ("rib_left_3", "its pair has no cartilage"),
        ("rib_right_10", "no cartilage above it to join"),
    ]  # and rib 11 floats: it is not even listed
    assert set(bands) == {"rib_left_7", "rib_right_7", "rib_left_8", "rib_right_8"}
    for side in ("left", "right"):
        above, band = bands[f"rib_{side}_7"], bands[f"rib_{side}_8"]
        along = shown(above)[0]
        k = above["tipAt"] + int(np.searchsorted(along, mesh.ARCH_JOIN[8]))
        assert along[k - above["tipAt"]] == pytest.approx(mesh.ARCH_JOIN[8], abs=0.01)
        assert band["endAt"] == len(band["curve"]) - 1
        np.testing.assert_allclose(band["curve"][-1], above["curve"][k])  # on its centreline
        assert band["halfW"][-1] == pytest.approx(above["halfW"][k])
        assert band["halfD"][-1] == pytest.approx(above["halfD"][k])
        assert band["tangent"][-1] @ above["tangent"][k] > 0.99  # running along it
        assert abs(band["across"][-1] @ above["across"][k]) > 0.99  # with the same face
        assert np.all(np.abs(np.diff(shown(band)[2])) < 0.05)  # the width never steps


def test_a_rib_end_too_far_from_the_sternum_or_on_it_gets_no_cartilage() -> None:
    far = {
        "rib_left_5": rib_end_at((-200, 40, 100), (0.3, 0.9, -0.3), 6.0, 3.0),
        "rib_right_5": rib_end_at((12, 63, 100), (-0.3, 0.9, -0.3), 6.0, 3.0),
    }
    bands, skipped = mesh.band_curves(far, plate_sternum())
    assert bands == {}
    assert skipped == [
        ("rib_left_5", f"its end is over {mesh.CARTILAGE_MAX_MM} mm from the sternum"),
        ("rib_right_5", "its end already touches the sternum"),
    ]


def synthetic_cage(seg: Path) -> dict[str, tuple[FloatArray, npt.NDArray[np.uint32]]]:
    """A sternum plate's mask on the test grid, and ribs 7 and 8 on both sides
    plus a floating rib 11 as bars, in the same RAS millimetres."""
    save(seg / "sternum.nii.gz", box((120, 80, 80), (54, 66, 20), (66, 72, 74)))
    middle = -48.0  # the plate's middle in RAS x (the test grid's x is flipped)
    ribs = {}
    for side, sign in (("right", 1.0), ("left", -1.0)):
        ribs[f"rib_{side}_7"] = bar(
            (middle + sign * 45, 40, 70), (-sign * 0.3, 0.9, -0.3), 6.0, 3.0
        )
        ribs[f"rib_{side}_8"] = bar(
            (middle + sign * 60, 30, 45), (-sign * 0.3, 0.9, -0.3), 5.0, 3.0
        )
    ribs["rib_left_11"] = bar((middle - 70, 0, 10), (0.3, 0.9, -0.3), 4.0, 3.0)
    return ribs


def test_modelled_cartilage_is_one_closed_arch_per_side(tmp_path: Path) -> None:
    ribs = synthetic_cage(tmp_path)
    ras, triangles, skipped = mesh.modelled_cartilage(tmp_path / "sternum.nii.gz", ribs)
    assert skipped == []
    triangles, _ = mesh.outward(ras, triangles)
    assert mesh.open_edges(triangles) == 0
    assert mesh.signed_volume(ras, triangles) > 0
    assert mesh_pieces(ras, triangles) == 2  # ribs 7 and 8 merged: one arch on each side
    assert (ras[:, 0] < -48.0).any() and (ras[:, 0] > -48.0).any()
    assert umbrella_scores(ras, triangles).max() < 0.7
    # It reaches each rib's end and the sternum, and nothing reaches the floating rib.
    from scipy.spatial import cKDTree

    surface = cKDTree(ras)
    for rib in ("rib_left_7", "rib_right_7", "rib_left_8", "rib_right_8"):
        end = mesh.rib_end_of(*ribs[rib])
        assert end is not None and surface.query(end["tip"])[0] < 4.0
    floating = mesh.rib_end_of(*ribs["rib_left_11"])
    assert floating is not None and surface.query(floating["tip"])[0] > 30
    assert np.abs(ras[:, 0] + 48.0).min() < 6.0  # inside the sternum's 9.6 mm wide plate


def test_modelled_cartilage_without_usable_ribs_is_empty(tmp_path: Path) -> None:
    ribs = synthetic_cage(tmp_path)
    short = bar((0, 0, 0), (0, 1, 0), 1.5, 1.5, length=3.0)
    ras, triangles, skipped = mesh.modelled_cartilage(
        tmp_path / "sternum.nii.gz", {"rib_left_7": ribs["rib_left_7"], "rib_right_7": short}
    )
    assert len(ras) == 0 and len(triangles) == 0
    assert skipped == [
        ("rib_right_7", "its front end could not be measured"),
        ("rib_left_7", "its pair has no cartilage"),
    ]
