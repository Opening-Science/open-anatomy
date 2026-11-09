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
from structures import ORGANS, REFERENCE, SETS, node_name

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


def test_the_body_set_builds_the_skeleton_and_every_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = run_main(tmp_path, monkeypatch, {}, "body")
    assert manifest["set"] == "body"
    assert manifest["dropped"] == []
    assert [n["name"] for n in manifest["nodes"]] == [node_name(s.term) for s in SETS["body"]]
    assert {n["system"] for n in manifest["nodes"]} >= {"skeletal", "digestive", "cardiovascular"}


@pytest.mark.parametrize("which", ["digestive", "body"])
def test_an_atlas_set_lists_a_cut_organ_instead_of_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    cut = {"liver": "cut by the scan's edge (axis 2)", "gallbladder": "absent or a stray label"}
    manifest = run_main(tmp_path, monkeypatch, cut, which)
    assert dropped(manifest) == cut
    kept = {n["label"] for n in manifest["nodes"]}
    assert "stomach" in kept and not kept & {"liver", "gallbladder"}


def test_the_reference_set_needs_every_organ_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="liver: cut by the scan's edge .* organs must be whole"):
        run_main(tmp_path, monkeypatch, {"liver": "cut by the scan's edge (axis 2)"})


def test_an_empty_set_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(mesh.SETS, "nothing", ())
    with pytest.raises(SystemExit, match='the "nothing" set has no structures yet'):
        run_main(tmp_path, monkeypatch, {}, "nothing")


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
    vertices, triangles = mesh.sweep(points, np.array([0.0, 1.0, 0.0]), widths, depths)
    assert mesh.open_edges(triangles) == 0
    assert abs(mesh.signed_volume(vertices, triangles)) > 0  # main() turns it outward
