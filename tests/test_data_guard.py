from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import data_guard
from data_guard import MAX_BYTES, offenders, reasons


def test_an_ordinary_small_file_is_fine() -> None:
    assert reasons("scripts/self_check.py", 2_000) == []


def test_a_file_under_data_is_refused() -> None:
    assert reasons("data/notes.txt", 10) == ["it is under data/, which must stay out of git"]


def test_a_folder_merely_named_like_data_elsewhere_is_fine() -> None:
    assert reasons("docs/data/readme.md", 10) == []
    assert reasons("metadata/readme.md", 10) == []


@pytest.mark.parametrize(
    "path",
    [
        "scan.nii",
        "deep/scan.nii.gz",
        "SCAN.NII.GZ",
        "slice.dcm",
        "Slice.DCM",
        "volume.nrrd",
        "volume.mha",
        "volume.mhd",
    ],
)
def test_medical_images_are_refused_in_any_case(path: str) -> None:
    found = reasons(path, 10)
    assert len(found) == 1
    assert found[0].startswith("it is a medical image")


def test_size_limit_is_inclusive_of_five_megabytes() -> None:
    assert reasons("assets/cc-by/kidney.glb", MAX_BYTES) == []
    assert reasons("assets/cc-by/kidney.glb", MAX_BYTES + 1) == [
        "it is 5.0 MB, over the 5 MB limit"
    ]


def test_a_file_can_have_several_reasons() -> None:
    assert len(reasons("data/ct.nii.gz", MAX_BYTES + 1)) == 3


def test_offenders_reads_sizes_from_disk(tmp_path: Path) -> None:
    (tmp_path / "small.txt").write_bytes(b"x")
    (tmp_path / "big.bin").write_bytes(b"\0" * (MAX_BYTES + 1))
    found = offenders(tmp_path, ["small.txt", "big.bin", "gone.dcm"])
    assert [path for path, _ in found] == ["big.bin", "gone.dcm"]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_tracked_files_lists_only_what_git_tracks(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "kept.txt").write_text("x")
    (tmp_path / "untracked.dcm").write_text("x")
    subprocess.run(["git", "add", "kept.txt"], cwd=tmp_path, check=True)
    assert data_guard.tracked_files(tmp_path) == ["kept.txt"]


@pytest.mark.parametrize("path", ["scan.glb", "scripts/body.PLY", "docs/x.obj", "a/b.stl"])
def test_a_mesh_outside_assets_and_fixtures_is_refused(path: str) -> None:
    assert reasons(path, 1000) == ["it is a mesh outside assets/ and tests/fixtures/"]


@pytest.mark.parametrize("path", ["assets/cc-by/liver.glb", "tests/fixtures/cube.ply"])
def test_a_mesh_in_assets_or_fixtures_is_fine(path: str) -> None:
    assert reasons(path, 1000) == []
